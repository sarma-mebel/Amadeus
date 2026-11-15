// Integration-style coverage for loadMixamoAnimation: constructs a minimal
// Mixamo-shaped rig + AnimationClip and a minimal VRM-like target, mocks the
// dynamically-imported FBXLoader so no real file I/O happens, and asserts
// the retargeted clip's tracks are renamed/bound onto the VRM node names
// (not the humanoid bone names) with the documented rest-pose composition
// and hips-height scale applied. The pure per-track math itself is covered
// in retargetCore.test.ts; this file exercises the surrounding per-bone
// wiring (mixamoVRMRigMap lookup, rest rotation source, hips height calc).
import { describe, expect, it, vi } from 'vitest'
import * as THREE from 'three'
import type { VRM } from '@pixiv/three-vrm'

vi.mock('three/examples/jsm/loaders/FBXLoader.js', () => {
  return {
    FBXLoader: class {
      async loadAsync() {
        return buildFakeMixamoAsset()
      }
    },
  }
})

/** Non-identity rest rotation on the fake source rig's spine, so tests can
 * observe that the retarget compensates the SOURCE rest pose. */
const SPINE_REST_TILT = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 1, 0), Math.PI / 4)

/** A tiny two-bone Mixamo-shaped rig: hips (root) -> spine, with a single
 * "mixamo.com" clip carrying a hips quaternion + position track and a spine
 * quaternion track. */
function buildFakeMixamoAsset() {
  const hips = new THREE.Bone()
  hips.name = 'mixamorigHips'
  hips.position.set(0, 1, 0) // rest hips height = 1
  const spine = new THREE.Bone()
  spine.name = 'mixamorigSpine'
  spine.quaternion.copy(SPINE_REST_TILT) // observable source rest rotation
  hips.add(spine)

  const group = new THREE.Group() as THREE.Group & { animations: THREE.AnimationClip[] }
  group.add(hips)
  group.updateWorldMatrix(true, true)

  const quarterTurnZ = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 0, 1), Math.PI / 2)
  const clip = new THREE.AnimationClip('mixamo.com', 1, [
    new THREE.QuaternionKeyframeTrack(
      'mixamorigHips.quaternion',
      [0],
      [quarterTurnZ.x, quarterTurnZ.y, quarterTurnZ.z, quarterTurnZ.w]
    ),
    new THREE.VectorKeyframeTrack('mixamorigHips.position', [0], [1, 2, 3]),
    new THREE.QuaternionKeyframeTrack('mixamorigSpine.quaternion', [0], [0, 0, 0, 1]),
    // An unmapped bone name should be silently skipped (no VRM bone maps to it).
    new THREE.QuaternionKeyframeTrack('mixamorigUnmappedBone.quaternion', [0], [0, 0, 0, 1]),
  ])
  group.animations = [clip]
  return group
}

/** A minimal duck-typed VRM: only the two surfaces mixamoRetarget.ts reads
 * (humanoid.getNormalizedBoneNode, scene.getWorldPosition) need to exist. */
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

describe('loadMixamoAnimation', () => {
  it('renames/binds tracks by VRM node name and skips unmapped bones', async () => {
    const { loadMixamoAnimation } = await import('./mixamoRetarget')
    const vrm = buildFakeVrm(1) // same hips height as the fake asset -> scale 1
    const clip = await loadMixamoAnimation('fake.fbx', vrm)

    const trackNames = clip.tracks.map((t) => t.name).sort()
    expect(trackNames).toEqual(['VrmHips.position', 'VrmHips.quaternion', 'VrmSpine.quaternion'])
  })

  it('scales the hips position track by the vrm/mixamo hips-height ratio', async () => {
    const { loadMixamoAnimation } = await import('./mixamoRetarget')
    // VRM hips are twice as tall as the fake Mixamo rig's (rest height 1) ->
    // hipsPositionScale should be 2.
    const vrm = buildFakeVrm(2)
    const clip = await loadMixamoAnimation('fake.fbx', vrm)

    const positionTrack = clip.tracks.find((t) => t.name === 'VrmHips.position') as THREE.VectorKeyframeTrack
    expect(Array.from(positionTrack.values)).toEqual([2, 4, 6])
  })

  it('composes rotations against the SOURCE rig rest-world rotation (pixiv example)', async () => {
    const { loadMixamoAnimation } = await import('./mixamoRetarget')
    const vrm = buildFakeVrm(1)
    // The VRM node's own rest rotation must NOT enter the composition (a
    // previous port used it — on a normalized rig it's ~identity, so the
    // source rig's joint orientations went uncompensated and limbs
    // splayed). Give the VRM node a garbage rest rotation to prove it is
    // ignored, and rely on the fake SOURCE spine's rest rotation instead.
    const vrmSpine = vrm.humanoid!.getNormalizedBoneNode('spine')!
    vrmSpine.quaternion.setFromAxisAngle(new THREE.Vector3(1, 0, 0), Math.PI / 3)

    const clip = await loadMixamoAnimation('fake.fbx', vrm)
    const rotationTrack = clip.tracks.find((t) => t.name === 'VrmSpine.quaternion') as THREE.QuaternionKeyframeTrack

    // Fake source rig: hips (root, identity) -> spine with rest rotation
    // SPINE_REST_TILT; the spine track's keyframe value is identity, so the
    // retargeted value must be parentRestWorld(identity) * identity *
    // inverse(spineRestWorld) = inverse(SPINE_REST_TILT).
    const expected = SPINE_REST_TILT.clone().invert()
    expect(rotationTrack.values[0]).toBeCloseTo(expected.x)
    expect(rotationTrack.values[1]).toBeCloseTo(expected.y)
    expect(rotationTrack.values[2]).toBeCloseTo(expected.z)
    expect(rotationTrack.values[3]).toBeCloseTo(expected.w)
  })
})
