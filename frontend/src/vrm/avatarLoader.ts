/**
 * VRM (avatar) loading helpers, moved verbatim out of VRMViewer's scene
 * effect: loader construction, the VRM0 yaw-flip rule, environment-anchor
 * placement, and post-parse mesh preparation. The load *orchestration*
 * (progress/status UI, fallback-to-default on a broken outfit, animation +
 * bridge wiring) stays in the component — these are the pieces with no
 * React state attached, shared with environmentManager.ts so an
 * environment swap repositions her with the exact same math the load
 * callback uses.
 */

import type * as THREE from 'three'
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js'
import { VRM, VRMLoaderPlugin, VRMUtils } from '@pixiv/three-vrm'
import type { EnvironmentDefinition } from '../environments'

/** VRM0 models are authored facing the opposite yaw from VRM1 (empirically
 * protractor-verified on this loader: at the same anchor yaw a VRM1 model
 * faces the camera and a VRM0 model shows her back). Every site that
 * anchors the scene root's yaw must add this 180° flip. Do NOT "fix" this
 * with VRMUtils.rotateVRM0 — it flips at the bone level invisibly to
 * scene.rotation and the two fight (that cost four debugging rounds). */
export const vrm0YawFlip = (vrm: VRM): number =>
  vrm.meta?.metaVersion === '0' ? Math.PI : 0

/** The GLTFLoader VRMViewer fetches avatars (not environments) with. */
export const createVrmLoader = (): GLTFLoader => {
  const loader = new GLTFLoader()
  loader.register((parser) => new VRMLoaderPlugin(parser))
  return loader
}

/** Set Kurisu's position and rotation for the given environment's anchor
 * (found empirically per-set — see environments.ts). */
export const placeVrmAtAnchor = (vrm: VRM, environment: EnvironmentDefinition): void => {
  vrm.scene.position.set(...environment.anchor.position)
  vrm.scene.rotation.y = environment.anchor.rotationY + vrm0YawFlip(vrm)
}

/** Post-parse mesh prep: strip unused vertices/joints, then mark her
 * skinned meshes as shadow casters.
 *
 * Real shadows (item 2), cast side: only skinned meshes (her actual
 * body/hair/clothing geometry) cast — this is the only castShadow
 * source in the scene now (the env is receive-only), so it's what
 * produces the contact shadow at her feet on the catcher/floor. */
export const prepareLoadedVrm = (vrm: VRM, gltfScene: THREE.Group): void => {
  VRMUtils.removeUnnecessaryVertices(gltfScene)
  VRMUtils.removeUnnecessaryJoints(gltfScene)
  vrm.scene.traverse((child) => {
    if ((child as THREE.SkinnedMesh).isSkinnedMesh) {
      child.castShadow = true
    }
  })
}
