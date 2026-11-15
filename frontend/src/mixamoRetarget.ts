// Mixamo FBX -> VRMA-equivalent retargeting.
//
// Adapted from the official @pixiv/three-vrm examples
// (examples/loadMixamoAnimation.js in pixiv/three-vrm), which maps Mixamo's
// fixed rig bone names onto VRM humanoid bone names, normalizes hips height
// for the target avatar, and fixes the coordinate/quaternion convention
// difference between Mixamo FBX exports and VRM.
//
// This module is intentionally the ONLY place that imports FBXLoader, and
// callers must import it with a dynamic `import()` so the loader (and its
// transitive fflate/binary-parsing weight) never lands in the main bundle —
// only projects that actually reference a manifest .fbx entry pay for it.
import * as THREE from 'three'
import type { VRM, VRMHumanBoneName } from '@pixiv/three-vrm'
import {
  buildRetargetedPositionTrack,
  buildRetargetedQuaternionTrack,
  hipsHeightScale,
  retargetPositionValues,
  retargetQuaternionValues,
} from './vrm/retargetCore'

// Mixamo rig bone name -> VRM humanoid bone name. Mixamo FBX rigs exported
// from mixamo.com consistently use this "mixamorigXxx" naming scheme
// regardless of which character model the animation was authored on.
export const mixamoVRMRigMap: Record<string, VRMHumanBoneName> = {
  mixamorigHips: 'hips',
  mixamorigSpine: 'spine',
  mixamorigSpine1: 'chest',
  mixamorigSpine2: 'upperChest',
  mixamorigNeck: 'neck',
  mixamorigHead: 'head',
  mixamorigLeftShoulder: 'leftShoulder',
  mixamorigLeftArm: 'leftUpperArm',
  mixamorigLeftForeArm: 'leftLowerArm',
  mixamorigLeftHand: 'leftHand',
  mixamorigLeftHandThumb1: 'leftThumbMetacarpal',
  mixamorigLeftHandThumb2: 'leftThumbProximal',
  mixamorigLeftHandThumb3: 'leftThumbDistal',
  mixamorigLeftHandIndex1: 'leftIndexProximal',
  mixamorigLeftHandIndex2: 'leftIndexIntermediate',
  mixamorigLeftHandIndex3: 'leftIndexDistal',
  mixamorigLeftHandMiddle1: 'leftMiddleProximal',
  mixamorigLeftHandMiddle2: 'leftMiddleIntermediate',
  mixamorigLeftHandMiddle3: 'leftMiddleDistal',
  mixamorigLeftHandRing1: 'leftRingProximal',
  mixamorigLeftHandRing2: 'leftRingIntermediate',
  mixamorigLeftHandRing3: 'leftRingDistal',
  mixamorigLeftHandPinky1: 'leftLittleProximal',
  mixamorigLeftHandPinky2: 'leftLittleIntermediate',
  mixamorigLeftHandPinky3: 'leftLittleDistal',
  mixamorigRightShoulder: 'rightShoulder',
  mixamorigRightArm: 'rightUpperArm',
  mixamorigRightForeArm: 'rightLowerArm',
  mixamorigRightHand: 'rightHand',
  mixamorigRightHandThumb1: 'rightThumbMetacarpal',
  mixamorigRightHandThumb2: 'rightThumbProximal',
  mixamorigRightHandThumb3: 'rightThumbDistal',
  mixamorigRightHandIndex1: 'rightIndexProximal',
  mixamorigRightHandIndex2: 'rightIndexIntermediate',
  mixamorigRightHandIndex3: 'rightIndexDistal',
  mixamorigRightHandMiddle1: 'rightMiddleProximal',
  mixamorigRightHandMiddle2: 'rightMiddleIntermediate',
  mixamorigRightHandMiddle3: 'rightMiddleDistal',
  mixamorigRightHandRing1: 'rightRingProximal',
  mixamorigRightHandRing2: 'rightRingIntermediate',
  mixamorigRightHandRing3: 'rightRingDistal',
  mixamorigRightHandPinky1: 'rightLittleProximal',
  mixamorigRightHandPinky2: 'rightLittleIntermediate',
  mixamorigRightHandPinky3: 'rightLittleDistal',
  mixamorigLeftUpLeg: 'leftUpperLeg',
  mixamorigLeftLeg: 'leftLowerLeg',
  mixamorigLeftFoot: 'leftFoot',
  mixamorigLeftToeBase: 'leftToes',
  mixamorigRightUpLeg: 'rightUpperLeg',
  mixamorigRightLeg: 'rightLowerLeg',
  mixamorigRightFoot: 'rightFoot',
  mixamorigRightToeBase: 'rightToes',
}

type FBXWithAnimations = THREE.Group & { animations: THREE.AnimationClip[] }

/**
 * Loads a Mixamo FBX animation and retargets it onto the given VRM's
 * humanoid rig, returning a THREE.AnimationClip ready for
 * mixer.clipAction(clip). Ported from the pixiv/three-vrm example script
 * of the same name (loadMixamoAnimation), trimmed to what this app needs
 * (single clip, no external "currentVrm" global).
 */
export async function loadMixamoAnimation(url: string, vrm: VRM): Promise<THREE.AnimationClip> {
  const { FBXLoader } = await import('three/examples/jsm/loaders/FBXLoader.js')

  const loader = new FBXLoader()
  const asset = (await loader.loadAsync(url)) as FBXWithAnimations
  const clip = THREE.AnimationClip.findByName(asset.animations, 'mixamo.com')
  if (!clip) {
    throw new Error(`No animation named "mixamo.com" found in FBX asset: ${url}`)
  }

  const tracks: THREE.KeyframeTrack[] = []

  const restRotationInverse = new THREE.Quaternion()
  const parentRestWorldRotation = new THREE.Quaternion()
  const _vec3 = new THREE.Vector3()
  const isVrm0 = (vrm.meta as { metaVersion?: string } | undefined)?.metaVersion === '0'

  // Adjust with hips height: Mixamo rigs are authored at a fixed real-world
  // scale, so a hip-position track has to be rescaled by the ratio between
  // the target VRM's actual hips height and Mixamo's reference hips height,
  // or a hip translation track will look wrong (character floats/sinks).
  const mixamoHipsHeight = getMixamoHipsHeight(asset)
  const vrmHipsY = vrm.humanoid
    ?.getNormalizedBoneNode('hips')
    ?.getWorldPosition(_vec3).y ?? 1
  const vrmRootY = vrm.scene.getWorldPosition(new THREE.Vector3()).y
  const vrmHipsHeight = Math.abs(vrmHipsY - vrmRootY) || 1
  const hipsPositionScale = hipsHeightScale(vrmHipsHeight, mixamoHipsHeight)

  clip.tracks.forEach((track) => {
    // Track names look like "mixamorigHips.quaternion" or
    // "mixamorigHips.position".
    const [mixamoRigName, propertyName] = track.name.split('.')
    const vrmBoneName = mixamoVRMRigMap[mixamoRigName]
    const vrmNode = vrmBoneName ? vrm.humanoid?.getNormalizedBoneNode(vrmBoneName) : null
    if (!vrmBoneName || !vrmNode) return

    const mixamoRigNode = asset.getObjectByName(mixamoRigName)
    if (!mixamoRigNode?.parent) return

    mixamoRigNode.parent.getWorldQuaternion(parentRestWorldRotation)

    if (track instanceof THREE.QuaternionKeyframeTrack) {
      // Retarget rotation: undo the SOURCE Mixamo node's rest-world
      // rotation and compose in its parent's rest-world rotation — exactly
      // the pixiv example. (A previous port used the VRM target node's
      // local rest here; on a normalized VRM rig that's ~identity, so
      // Mixamo's non-identity joint orientations were never compensated
      // and every limb splayed.)
      mixamoRigNode.getWorldQuaternion(restRotationInverse).invert()

      const values = retargetQuaternionValues(track, {
        restRotationInverse,
        parentRestWorldRotation,
      })
      // VRM0 rigs are rotated 180° vs VRM1: flip quaternion x/z, as in the
      // pixiv example.
      if (isVrm0) {
        for (let i = 0; i < values.length; i += 4) {
          values[i] *= -1
          values[i + 2] *= -1
        }
      }
      // Bind by the normalized bone node's NAME (matching the pixiv
      // example) — humanoid bone names aren't node names, so tracks named
      // after them never bound to anything.
      tracks.push(buildRetargetedQuaternionTrack(vrmNode.name, track.times, values))
    } else if (track instanceof THREE.VectorKeyframeTrack && propertyName === 'position') {
      const values = retargetPositionValues(track, hipsPositionScale)
      if (isVrm0) {
        for (let i = 0; i < values.length; i += 3) {
          values[i] *= -1
          values[i + 2] *= -1
        }
      }
      tracks.push(buildRetargetedPositionTrack(vrmNode.name, track.times, values))
    }
  })

  return new THREE.AnimationClip(clip.name, clip.duration, tracks)
}

function getMixamoHipsHeight(asset: FBXWithAnimations): number {
  const hips = asset.getObjectByName('mixamorigHips')
  if (!hips) return 1
  const worldPos = new THREE.Vector3()
  hips.getWorldPosition(worldPos)
  return Math.abs(worldPos.y) || 1
}
