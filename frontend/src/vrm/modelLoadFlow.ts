/**
 * Model (outfit) load bootstrap decision logic, moved verbatim out of
 * VRMViewer's scene effect.
 *
 *   - resolveRequestedModel: given the id the user last picked and the
 *     fetched registry, which file to actually load, and whether that id
 *     was a registry-miss that needs persisting back to localStorage
 *     (so SettingsPanel's picker state agrees with what's actually on
 *     screen).
 *   - shouldRetryWithDefaultModel: the load-failure fallback gate — a
 *     broken/removed non-default outfit file shouldn't strand the user on
 *     the permanent error overlay, so retry once against the default
 *     model; a busted *default* must not loop forever, so only retry when
 *     the failing id isn't already the default AND a retry hasn't already
 *     been attempted this mount.
 *   - placeAndFrameLoadedVrm: the imperative sequence that runs the instant
 *     a VRM finishes loading — anchor placement, rim lighting, adding it to
 *     the scene, the debug-harness bridge, fire-and-forget shader
 *     precompile, the camera-follow gaze proxy, and the initial bone-
 *     derived camera framing. Every scene/camera/ref dependency is passed
 *     in explicitly; the caller still owns assigning the returned gaze
 *     proxy to its own cleanup-tracking local exactly as the inline version
 *     did.
 *
 * The two resolve/retry functions don't touch refs/state directly —
 * VRMViewer's callbacks still own applying the result (assigning
 * modelIdRef, calling saveModelId/setModelReloadToken/setLoadStatus)
 * exactly as before; those two only isolate the branching that decides
 * what should happen.
 */

import * as THREE from 'three'
import type { VRM } from '@pixiv/three-vrm'
import { DEFAULT_MODEL, type ModelDefinition } from '../models'
import type { EnvironmentDefinition } from '../environments'
import { placeVrmAtAnchor } from './avatarLoader'
import { applyRimLighting } from './rimLight'
import { computeDefaultFraming } from './cameraFraming'
import { loadPetPreset, petPresetToFramingMode } from './petPreset'
import { diagnostics } from '../diagnostics'
import type { CameraDirector } from './cameraDirector'
import type { GazeRuntimeState } from './gazeDirector'

export interface ResolvedModel {
  /** The VRM file path to load. */
  file: string
  /** True when requestedId isn't in the registry (and isn't already the
   * default) — the caller should persist DEFAULT_MODEL.id back to
   * localStorage and update its modelId ref so picker state matches. */
  isRegistryMiss: boolean
}

/** Registry-miss fallback: resolveModelFile itself already falls back to
 * DEFAULT_MODEL.file for an unknown id, so this only adds the "should we
 * persist that fallback" signal on top. */
export const resolveRequestedModel = (
  requestedId: string,
  registry: ModelDefinition[],
  resolveModelFile: (id: string, registry: ModelDefinition[]) => string
): ResolvedModel => {
  const file = resolveModelFile(requestedId, registry)
  const isRegistryMiss =
    requestedId !== DEFAULT_MODEL.id && !registry.some((model) => model.id === requestedId)
  return { file, isRegistryMiss }
}

/** Progress callback total is 0 when the server doesn't send
 * Content-Length; a percent of 0 (not NaN/Infinity) in that case matches
 * the original inline ternary exactly. */
export const progressPercent = (loaded: number, total: number): number =>
  total > 0 ? (loaded / total) * 100 : 0

/** Load-failure fallback gate: true exactly when the failing model isn't
 * already the default AND a retry hasn't already been attempted this
 * mount — the caller is responsible for setting the retried flag before
 * triggering the reload, same as the original inline branch. */
export const shouldRetryWithDefaultModel = (
  currentModelId: string,
  alreadyRetried: boolean
): boolean => currentModelId !== DEFAULT_MODEL.id && !alreadyRetried

export interface PlaceAndFrameLoadedVrmDeps {
  scene: THREE.Scene
  camera: THREE.PerspectiveCamera
  renderer: THREE.WebGLRenderer
  cameraDirector: Pick<CameraDirector, 'adoptDefaultFraming'>
  gazeStateRef: { current: GazeRuntimeState }
  getActiveEnvironment: () => EnvironmentDefinition
  getCanvas: () => HTMLCanvasElement | null
  mainLightColor: THREE.Color
  fillLightColor: THREE.Color
  variant: 'full' | 'pet'
  fileToLoad: string
  debugLog: (...args: unknown[]) => void
}

/** Runs once, synchronously, the instant a VRM finishes loading — anchor
 * placement, rim lighting, adding it to the scene, the debug-harness
 * bridge, a fire-and-forget shader precompile, the camera-follow gaze
 * proxy, and the initial bone-derived camera framing. Returns the gaze
 * proxy Object3D so the caller can track it for its own teardown (same
 * object gazeStateRef.current.proxy is set to here). */
export const placeAndFrameLoadedVrm = (vrm: VRM, deps: PlaceAndFrameLoadedVrmDeps): THREE.Object3D => {
  // Set Kurisu's position and rotation for the active environment's anchor
  // (found empirically per-set — see environments.ts). Read live off
  // getActiveEnvironment (not a value captured at scene-mount time) so
  // that if the environment changed while this async VRM load was in
  // flight, she lands at the CURRENT environment's spot rather than the
  // one active when the load started.
  placeVrmAtAnchor(vrm, deps.getActiveEnvironment())
  applyRimLighting(vrm, { main: deps.mainLightColor, fill: deps.fillLightColor })

  deps.scene.add(vrm.scene)

  // Debug/harness bridge: lets the headless test harness interrogate the
  // loaded model (meta version, root yaw) without guessing from
  // screenshots. Read-only by convention.
  ;(window as unknown as { kurisuDebugVrm?: unknown }).kurisuDebugVrm = vrm

  // PSO-precache equivalent: compile every material's shader program now,
  // off the critical path (parallel via KHR_parallel_shader_compile where
  // available), while the connection screen still covers the canvas —
  // instead of paying a multi-hundred-ms compile hitch on the first frame
  // she's actually visible. Fire-and-forget: a compile failure just means
  // the old lazy path.
  void deps.renderer
    .compileAsync(deps.scene, deps.camera)
    .then(() => deps.debugLog('✅ Shaders precompiled'))
    .catch((err) => deps.debugLog('Shader precompile skipped:', err))

  // Gaze micro-behavior (item 6): lookAt.target is set ONCE to a proxy
  // Object3D that tracks the camera, instead of re-assigning
  // lookAt.target = camera every frame. The render loop nudges this proxy
  // to a small random offset every 2-6s for 0.4-1.2s before returning it
  // to dead-on-camera — a permanent, unblinking stare at the camera reads
  // as mannequin-like, so an occasional glance-away sells "there's someone
  // thinking in there".
  const gazeProxy = new THREE.Object3D()
  deps.scene.add(gazeProxy)
  gazeProxy.position.copy(deps.camera.position)
  deps.gazeStateRef.current.proxy = gazeProxy
  if (vrm.lookAt) {
    vrm.lookAt.target = gazeProxy
  } else {
    // FBX-converted packs ship without eye bones or a lookAt config — eye
    // gaze is structurally unavailable there (head-aim still tracks, which
    // visually masks it). Count it per the degrade-loudly convention
    // instead of pretending gaze works.
    diagnostics.increment('avatar.eye_gaze_unavailable', deps.fileToLoad)
  }

  // Compute the procedural default framing from the model's own bones
  // (item 1) and snap to it — replaces the old hardcoded position/target
  // pair, which was tuned for one specific window aspect and one specific
  // spot in the Time Machine set.
  const canvas = deps.getCanvas()
  const initialAspect = canvas ? canvas.clientWidth / canvas.clientHeight : deps.camera.aspect
  // Pet preset (localStorage kurisu_pet_preset, see ../vrm/petPreset):
  // 'compact' picks the tighter bust/head-and-shoulders framing mode in
  // cameraFraming.ts instead of the default medium shot. Pet-variant only
  // — the main window has no preset UI and always gets 'full'/'medium',
  // matching its unchanged prior behavior exactly.
  const initialFramingMode = deps.variant === 'pet' ? petPresetToFramingMode(loadPetPreset()) : 'medium'
  const framing = computeDefaultFraming(vrm, initialAspect, deps.camera.fov, initialFramingMode)
  if (framing) {
    deps.cameraDirector.adoptDefaultFraming(framing)
    deps.debugLog(
      '🎥 Default framing: aspect',
      initialAspect,
      'distance',
      framing.offset.length(),
      'target',
      framing.target.toArray()
    )
  }

  return gazeProxy
}

export interface ReframeForPetPresetChangeDeps {
  vrm: VRM | null
  canvas: HTMLCanvasElement | null
  cameraFov: number
  cameraDirector: Pick<CameraDirector, 'adoptDefaultFraming' | 'startReframeTween'>
}

/** Pet preset live reframe (small item, pet variant only): recompute the
 * bone-derived framing for the new preset and smoothly tween the camera to
 * it — reuses the exact same computeDefaultFraming + adoptDefaultFraming +
 * startReframeTween path the initial load and the double-click-to-reframe
 * escape hatch already use, so this is never a second, divergent framing
 * implementation. Caller (VRMViewer) is responsible for the variant==='pet'
 * gate before calling this — mirrors the inline version's own guard order. */
export const reframeForPetPresetChange = (deps: ReframeForPetPresetChangeDeps): void => {
  if (!deps.vrm || !deps.canvas) return
  const aspect = deps.canvas.clientWidth / deps.canvas.clientHeight
  const framingMode = petPresetToFramingMode(loadPetPreset())
  const framing = computeDefaultFraming(deps.vrm, aspect, deps.cameraFov, framingMode)
  if (!framing) return
  deps.cameraDirector.adoptDefaultFraming(framing)
  deps.cameraDirector.startReframeTween()
}
