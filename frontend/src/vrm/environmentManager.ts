/**
 * Environment (backdrop set) swapping, moved verbatim out of VRMViewer's
 * environment effect. This is the heart of the "environment swap must not
 * rebuild the scene" behavior: everything here mutates the LIVE scene/
 * light/catcher/VRM the scene effect owns — no teardown of the WebGL
 * scene, no VRM re-fetch, no AudioContext/lip-sync reset.
 *
 * Split into three testable pieces the component's environment effect
 * composes:
 *   - repositionForEnvironment: refit shadow rig + VRM anchor in place
 *   - disposeEnvironmentGroup: remove + free the outgoing set's GPU
 *     resources (geometries/materials) so repeated swaps don't leak
 *   - loadEnvironmentGroup: fetch the incoming GLB and add it to the scene,
 *     with a cancel guard for unmount-while-in-flight
 */

import * as THREE from 'three'
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js'
import { MeshoptDecoder } from 'three/examples/jsm/libs/meshopt_decoder.module.js'
import type { VRM } from '@pixiv/three-vrm'
import { resolveAssetPath } from '../config'
import type { EnvironmentDefinition } from '../environments'
import { applyShadowCenter, shadowCenterForEnvironment } from './sceneSetup'
import { placeVrmAtAnchor } from './avatarLoader'

/** Reposition shadow-center-derived objects on the existing light/
 * catcher instead of recreating them — the same applyShadowCenter math the
 * scene rig runs at construction time (see sceneSetup.ts for the "why" of
 * each offset), just applied in place rather than at construction time.
 *
 * Also repositions the VRM (if already loaded) at the new environment's
 * anchor — a swap while she's already on screen must move her, not
 * just affect the next load. */
export const repositionForEnvironment = (
  {
    mainLight,
    shadowCatcher,
    vrm,
  }: {
    mainLight: THREE.DirectionalLight | null
    shadowCatcher: THREE.Mesh | null
    vrm: VRM | null
  },
  environment: EnvironmentDefinition
): void => {
  applyShadowCenter(mainLight, shadowCatcher, shadowCenterForEnvironment(environment))
  if (vrm) {
    placeVrmAtAnchor(vrm, environment)
  }
}

/** Remove the outgoing environment GLB and dispose its geometries/
 * materials (it's replaced, not reused) before loading the next one. */
export const disposeEnvironmentGroup = (scene: THREE.Scene, group: THREE.Group): void => {
  scene.remove(group)
  group.traverse((child) => {
    const mesh = child as THREE.Mesh
    if (mesh.isMesh) {
      mesh.geometry?.dispose()
      const material = mesh.material
      if (Array.isArray(material)) {
        material.forEach((mat) => mat.dispose())
      } else {
        material?.dispose()
      }
    }
  })
}

/** Minimal loader surface so tests can inject a fake instead of a real
 * network-backed GLTFLoader. */
export interface EnvironmentGlbLoader {
  load: (
    url: string,
    onLoad: (gltf: { scene: THREE.Group }) => void,
    onProgress?: (event: ProgressEvent) => void,
    onError?: (error: unknown) => void
  ) => void
}

const createDefaultEnvironmentLoader = (): EnvironmentGlbLoader => {
  const envLoader = new GLTFLoader()
  envLoader.setMeshoptDecoder(MeshoptDecoder)
  return envLoader
}

/** Fetch the environment GLB and add it to the live scene. Returns a
 * cancel handle: after cancel(), a late-resolving load never touches the
 * scene (the caller's scene may already be torn down). */
export const loadEnvironmentGroup = ({
  scene,
  environment,
  debugLog,
  onLoaded,
  createLoader = createDefaultEnvironmentLoader,
}: {
  scene: THREE.Scene
  environment: EnvironmentDefinition
  debugLog: (...args: unknown[]) => void
  /** Called with the added group so the caller can track it for the next
   * swap's disposal. */
  onLoaded: (group: THREE.Group) => void
  createLoader?: () => EnvironmentGlbLoader
}): { cancel: () => void } => {
  let disposed = false
  const envLoader = createLoader()
  debugLog(`\n=== LOADING ${environment.label.toUpperCase()} ENVIRONMENT ===`)
  envLoader.load(
    resolveAssetPath(environment.file),
    (gltf) => {
      if (disposed) return
      const environmentGroup = gltf.scene
      // Real shadows (item 2), receive-only side: the set's own meshes
      // (floor, walls, the machine housing) should show the contact
      // shadow Kurisu casts, but the set itself doesn't need to cast
      // shadows onto her/itself — that would double the draw calls in
      // the shadow pass for geometry that's mostly static background and
      // rarely self-occludes in a way worth the cost.
      environmentGroup.traverse((child) => {
        if ((child as THREE.Mesh).isMesh) {
          child.receiveShadow = true
        }
      })
      scene.add(environmentGroup)
      onLoaded(environmentGroup)
      debugLog(`✅ ${environment.label} environment loaded`)
      const box = new THREE.Box3().setFromObject(environmentGroup)
      debugLog('Environment bounding box:', box)
    },
    (progress) => {
      const percent = (progress.loaded / progress.total) * 100
      debugLog(`Loading environment: ${percent.toFixed(1)}%`)
    },
    (error) => {
      console.error('❌ Failed to load environment:', error)
    }
  )
  return {
    cancel: () => {
      disposed = true
    },
  }
}
