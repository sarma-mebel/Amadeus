// BVH -> VRM humanoid retargeting, for BVH packs whose rigs already use VRM
// humanoid bone names (e.g. SystemAnimatorOnline "VRM rig" Mixamo
// conversions — the SillyTavern community animation pack). Same rest-pose
// math as the pixiv loadMixamoAnimation example (and mixamoRetarget.ts):
// BVH rest poses are all-identity rotations by construction and these rigs
// are authored as VRM-matching T-poses, so the parent-rest/rest-inverse
// composition mostly passes rotations through — but we keep the full math
// so a BVH with baked rest rotations still retargets correctly.
//
// Like FBXLoader, BVHLoader is only imported when a manifest actually
// references a .bvh clip.
import * as THREE from 'three'
import type { VRM, VRMHumanBoneName } from '@pixiv/three-vrm'
import {
  buildRetargetedPositionTrack,
  buildRetargetedQuaternionTrack,
  hipsHeightScale,
  retargetPositionValues,
  retargetQuaternionValues,
} from './vrm/retargetCore'

// Track names from BVHLoader: "hips.quaternion" in three r180+, or
// ".bones[hips].quaternion" in older releases — handle both.
const parseTrackName = (name: string): { boneName: string; propertyName: string } | null => {
  const lastDot = name.lastIndexOf('.')
  if (lastDot <= 0) return null
  const target = name.slice(0, lastDot)
  const propertyName = name.slice(lastDot + 1)
  if (propertyName !== 'quaternion' && propertyName !== 'position') return null
  const bracket = target.match(/^\.bones\[(.+)\]$/)
  return { boneName: bracket ? bracket[1] : target, propertyName }
}

// The gaze/lookAt system owns the eyes; head/neck get stripped later by
// withoutBoneMotionTracks, but eye tracks would fight lookAt directly.
const SKIPPED_BONES = new Set(['leftEye', 'rightEye'])

export async function loadBvhAnimation(url: string, vrm: VRM): Promise<THREE.AnimationClip> {
  const { BVHLoader } = await import('three/examples/jsm/loaders/BVHLoader.js')

  const loader = new BVHLoader()
  const result = await loader.loadAsync(url)
  const skeleton = result.skeleton
  const clip = result.clip

  const rootBone = skeleton.bones[0]
  rootBone.updateWorldMatrix(true, true)

  const restRotationInverse = new THREE.Quaternion()
  const parentRestWorldRotation = new THREE.Quaternion()
  const _quatA = new THREE.Quaternion()
  const _vec3 = new THREE.Vector3()

  // Hips-height ratio rescales the hips translation track (BVH units are
  // arbitrary — this pack is in decimeter-ish units).
  const srcHipsY = Math.abs(
    (skeleton.getBoneByName('hips') ?? rootBone).getWorldPosition(_vec3).y
  ) || 1
  const vrmHipsY = vrm.humanoid?.getNormalizedBoneNode('hips')?.getWorldPosition(_vec3).y ?? 1
  const vrmRootY = vrm.scene.getWorldPosition(new THREE.Vector3()).y

  const tracks: THREE.KeyframeTrack[] = []

  // The community BVH pack is a multi-contributor collection with two
  // inconsistencies this normalizes per clip:
  //  - some clips face -Z (her back to the camera, keyed off the first hips
  //    keyframe): premultiply a 180° Y flip into the hips rotation track and
  //    mirror hips translation X/Z;
  //  - some clips' hips translation is in a different reference than the
  //    skeleton's rest offset (she sinks into the floor): when the MEDIAN
  //    hips height across the clip disagrees with the rest offset by >25%,
  //    trust the median as the standing reference instead. The median, not
  //    the first frame: a gesture that opens crouched/bowed/leaning would
  //    make a first-frame reference mis-scale the whole clip, while most
  //    clips spend most of their time near standing whatever pose they open
  //    with. (Clips that are genuinely seated/lying THROUGHOUT still
  //    mis-scale — none of those are in any automatic pool.)
  const hipsRotationTrack = clip.tracks.find((t) => {
    const parsed = parseTrackName(t.name)
    return parsed?.boneName === 'hips' && parsed.propertyName === 'quaternion'
  })
  const hipsPositionTrack = clip.tracks.find((t) => {
    const parsed = parseTrackName(t.name)
    return parsed?.boneName === 'hips' && parsed.propertyName === 'position'
  })
  const yFlip = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 1, 0), Math.PI)
  let needsFlip = false
  if (hipsRotationTrack) {
    const forward = _vec3
      .set(0, 0, 1)
      .applyQuaternion(_quatA.fromArray(hipsRotationTrack.values, 0))
    needsFlip = forward.z < 0
  }
  let translationReferenceY = srcHipsY
  if (hipsPositionTrack) {
    const heights: number[] = []
    for (let i = 1; i < hipsPositionTrack.values.length; i += 3) {
      heights.push(Math.abs(hipsPositionTrack.values[i]))
    }
    heights.sort((a, b) => a - b)
    const medianY = heights[Math.floor(heights.length / 2)] ?? 0
    if (medianY > 1e-3 && Math.abs(medianY - srcHipsY) / srcHipsY > 0.25) {
      translationReferenceY = medianY
    }
  }
  const translationScale = hipsHeightScale(Math.abs(vrmHipsY - vrmRootY), translationReferenceY)

  clip.tracks.forEach((track) => {
    const parsed = parseTrackName(track.name)
    if (!parsed) return
    const { boneName, propertyName } = parsed
    if (SKIPPED_BONES.has(boneName)) return

    const vrmNode = vrm.humanoid?.getNormalizedBoneNode(boneName as VRMHumanBoneName)
    if (!vrmNode) return
    const srcBone = skeleton.getBoneByName(boneName)
    if (!srcBone) return

    if (propertyName === 'quaternion' && track instanceof THREE.QuaternionKeyframeTrack) {
      srcBone.getWorldQuaternion(restRotationInverse).invert()
      if (srcBone.parent) {
        srcBone.parent.getWorldQuaternion(parentRestWorldRotation)
      } else {
        parentRestWorldRotation.identity()
      }

      const flipThisTrack = needsFlip && boneName === 'hips'
      const values = retargetQuaternionValues(track, {
        restRotationInverse,
        parentRestWorldRotation,
        preFlip: flipThisTrack ? yFlip : null,
      })
      tracks.push(buildRetargetedQuaternionTrack(vrmNode.name, track.times, values))
    } else if (propertyName === 'position' && boneName === 'hips') {
      const sign = needsFlip ? -1 : 1
      const values = retargetPositionValues(track, translationScale, sign, sign)
      tracks.push(buildRetargetedPositionTrack(vrmNode.name, track.times, values))
    }
  })

  return new THREE.AnimationClip(clip.name || 'bvh', clip.duration, tracks)
}
