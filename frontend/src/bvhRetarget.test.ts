// Integration-style coverage for loadBvhAnimation: constructs a minimal
// VRM-named BVH-shaped skeleton + clip and a minimal VRM-like target, mocks
// the dynamically-imported BVHLoader so no real file I/O happens, and
// asserts track renaming/binding plus the BVH-specific hooks (the 180° Y
// facing-flip normalization and the hips-height reference-height fallback)
// that retargetCore.ts's shared math doesn't own. Pure per-track math is
// covered in retargetCore.test.ts.
import { describe, expect, it, vi } from 'vitest'
import * as THREE from 'three'
import type { VRM } from '@pixiv/three-vrm'

vi.mock('three/examples/jsm/loaders/BVHLoader.js', () => {
  return {
    BVHLoader: class {
      async loadAsync() {
        return buildFakeBvhResult()
      }
    },
  }
})

/** A tiny two-bone VRM-named rig: hips (root) -> spine, with a clip whose
 * hips quaternion faces +Z (no flip needed) and a hips position track. */
function buildFakeBvhResult() {
  const hips = new THREE.Bone()
  hips.name = 'hips'
  hips.position.set(0, 1, 0) // rest hips height = 1
  const spine = new THREE.Bone()
  spine.name = 'spine'
  hips.add(spine)

  const skeleton = new THREE.Skeleton([hips, spine])
  hips.updateWorldMatrix(true, true)

  const identity = new THREE.Quaternion()
  const clip = new THREE.AnimationClip('bvh', 1, [
    new THREE.QuaternionKeyframeTrack('hips.quaternion', [0], [identity.x, identity.y, identity.z, identity.w]),
    new THREE.VectorKeyframeTrack('hips.position', [0], [1, 1, 3]),
    new THREE.QuaternionKeyframeTrack('spine.quaternion', [0], [0, 0, 0, 1]),
    // Eyes are stripped (SKIPPED_BONES) — the gaze system owns them.
    new THREE.QuaternionKeyframeTrack('leftEye.quaternion', [0], [0, 0, 0, 1]),
  ])

  return { skeleton, clip }
}

function buildFakeVrm(hipsWorldY: number): VRM {
  const vrmHips = new THREE.Object3D()
  vrmHips.name = 'VrmHips'
  vrmHips.position.set(0, hipsWorldY, 0)
  const vrmSpine = new THREE.Object3D()
  vrmSpine.name = 'VrmSpine'
  vrmHips.add(vrmSpine)

  const scene = new THREE.Group()
  scene.add(vrmHips)
  scene.updateWorldMatrix(true, true)

  const nodesByBone: Record<string, THREE.Object3D> = { hips: vrmHips, spine: vrmSpine }

  return {
    humanoid: {
      getNormalizedBoneNode: (name: string) => nodesByBone[name] ?? null,
    },
    scene,
  } as unknown as VRM
}

describe('loadBvhAnimation', () => {
  it('renames/binds tracks by VRM node name and strips skipped bones (eyes)', async () => {
    const { loadBvhAnimation } = await import('./bvhRetarget')
    const vrm = buildFakeVrm(1)
    const clip = await loadBvhAnimation('fake.bvh', vrm)

    const trackNames = clip.tracks.map((t) => t.name).sort()
    expect(trackNames).toEqual(['VrmHips.position', 'VrmHips.quaternion', 'VrmSpine.quaternion'])
  })

  it('scales the hips position track by the vrm/source hips-height ratio (no flip needed)', async () => {
    const { loadBvhAnimation } = await import('./bvhRetarget')
    const vrm = buildFakeVrm(2) // VRM hips twice the fake rig's rest height (1) -> scale 2
    const clip = await loadBvhAnimation('fake.bvh', vrm)

    const positionTrack = clip.tracks.find((t) => t.name === 'VrmHips.position') as THREE.VectorKeyframeTrack
    // No flip (hips face +Z already): X/Z keep their sign, all scaled by 2.
    expect(Array.from(positionTrack.values)).toEqual([2, 2, 6])
  })

  it('detects a -Z-facing clip and premultiplies the 180deg Y-flip + mirrors hips X/Z', async () => {
    vi.doMock('three/examples/jsm/loaders/BVHLoader.js', () => ({
      BVHLoader: class {
        async loadAsync() {
          const result = buildFakeBvhResult()
          // Face -Z: rotate the hips keyframe 180deg about Y so forward
          // (0,0,1) rotates to (0,0,-1), tripping needsFlip.
          const yFlip = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 1, 0), Math.PI)
          const hipsRotationTrack = result.clip.tracks.find(
            (t) => t.name === 'hips.quaternion'
          ) as THREE.QuaternionKeyframeTrack
          hipsRotationTrack.values.set([yFlip.x, yFlip.y, yFlip.z, yFlip.w])
          return result
        }
      },
    }))
    vi.resetModules()
    const { loadBvhAnimation } = await import('./bvhRetarget')
    const vrm = buildFakeVrm(1)
    const clip = await loadBvhAnimation('fake.bvh', vrm)

    const positionTrack = clip.tracks.find((t) => t.name === 'VrmHips.position') as THREE.VectorKeyframeTrack
    // X/Z mirrored (sign flipped), Y untouched, matching a needsFlip clip.
    expect(Array.from(positionTrack.values)).toEqual([-1, 1, -3])

    vi.doUnmock('three/examples/jsm/loaders/BVHLoader.js')
    vi.resetModules()
  })

  it('keys the height reference to the MEDIAN hips height, not the opening pose', async () => {
    vi.doMock('three/examples/jsm/loaders/BVHLoader.js', () => ({
      BVHLoader: class {
        async loadAsync() {
          const result = buildFakeBvhResult()
          // A gesture that opens crouched (y=0.4) but stands (y=1.0 = rest)
          // for most of the clip. A first-frame reference would rescale the
          // whole clip by 1/0.4 — the sinks-into-the-floor bug; the median
          // (1.0, within 25% of rest) must keep the rest-offset reference.
          const idx = result.clip.tracks.findIndex((t) => t.name === 'hips.position')
          result.clip.tracks[idx] = new THREE.VectorKeyframeTrack(
            'hips.position',
            [0, 0.25, 0.5, 0.75, 1],
            [0, 0.4, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0]
          )
          return result
        }
      },
    }))
    vi.resetModules()
    const { loadBvhAnimation } = await import('./bvhRetarget')
    const vrm = buildFakeVrm(1) // same rest height as the rig -> scale must be 1
    const clip = await loadBvhAnimation('fake.bvh', vrm)

    const positionTrack = clip.tracks.find((t) => t.name === 'VrmHips.position') as THREE.VectorKeyframeTrack
    const ys = Array.from(positionTrack.values).filter((_, i) => i % 3 === 1)
    const expected = [0.4, 1, 1, 1, 1]
    expect(ys.length).toBe(expected.length)
    ys.forEach((y, i) => expect(y).toBeCloseTo(expected[i], 5))

    vi.doUnmock('three/examples/jsm/loaders/BVHLoader.js')
    vi.resetModules()
  })
})
