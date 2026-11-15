/**
 * Scene/renderer/camera/lighting construction for VRMViewer, moved verbatim
 * out of the component's scene effect. One `createSceneRig` call per scene
 * mount owns everything that exists exactly once per WebGL context: the
 * THREE.Scene, camera, renderer, the three-light rig + fog + shadow
 * catcher, OrbitControls, and the time-of-day lighting lerp machinery.
 *
 * The shadow-center math (`applyShadowCenter`) is shared with
 * environmentManager.ts: the same fitting runs at construction time here
 * and again in place on every environment swap, so the two can never
 * drift apart.
 */

import * as THREE from 'three'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'
import type { EnvironmentDefinition } from '../environments'
import { PRE_LOAD_CAMERA_POSITION, PRE_LOAD_CAMERA_TARGET } from './cameraFraming'
import { getTimeOfDayLighting } from './timeOfDayLighting'

/** Shadow center: the world point mainLight.target, the shadow frustum,
 * and the shadow-catcher plane are all fitted around. Defaults to the
 * active environment's Kurisu anchor (shadowCenter is only for the rare
 * case a set wants the catcher/frustum centered somewhere else). */
export const shadowCenterForEnvironment = (
  environment: EnvironmentDefinition
): [number, number, number] => environment.shadowCenter ?? environment.anchor.position

/** Fit the shadow-center-derived pieces (mainLight position/target, shadow
 * frustum, shadow-catcher plane) around a world point. Called once at
 * construction below and again — in place, on the same objects — by the
 * environment effect on every swap.
 *
 * Kurisu is placed at the active environment's anchor (see the VRM load
 * callback) — the light itself stays at a fixed offset from the
 * camera-relative rig, but its shadow camera frustum is targeted at her
 * position, not the world origin.
 *
 * mainLight.target defaults to a THREE.Object3D at the origin; a
 * DirectionalLight's shadow frustum is built around target - position,
 * so without repointing target at Kurisu's actual spot the frustum
 * would be fitted to (0,0,0) and she'd sit right at its edge
 * (or outside it, depending on facing), giving a clipped/absent shadow.
 *
 * The frustum is fitted tightly to an asymmetric box around her position
 * (not the whole environment) so the limited shadow map resolution isn't
 * wasted on empty set geometry — the original Time Machine tuning (left
 * -8/right 2/top 6/bottom -3 around center (-5,0,0)) is preserved here as
 * offsets from the shadow center so it re-centers on whichever environment
 * is active.
 */
export const applyShadowCenter = (
  mainLight: THREE.DirectionalLight | null,
  shadowCatcher: THREE.Mesh | null,
  [shadowCenterX, shadowCenterY, shadowCenterZ]: [number, number, number]
): void => {
  if (mainLight) {
    mainLight.position.set(shadowCenterX + 2, shadowCenterY + 3, shadowCenterZ + 2)
    mainLight.target.position.set(shadowCenterX, shadowCenterY, shadowCenterZ)
    mainLight.shadow.camera.left = shadowCenterX - 3
    mainLight.shadow.camera.right = shadowCenterX + 7
    mainLight.shadow.camera.top = shadowCenterY + 6
    mainLight.shadow.camera.bottom = shadowCenterY - 3
    mainLight.shadow.camera.updateProjectionMatrix()
  }
  if (shadowCatcher) {
    // tiny lift above the anchor's y to avoid z-fighting with the set floor
    shadowCatcher.position.set(shadowCenterX, shadowCenterY + 0.001, shadowCenterZ)
  }
}

/** Time-of-day lighting is re-evaluated on this interval (see
 * refreshTimeOfDayLighting) so a session left open across a
 * dawn/day/evening/night boundary drifts into the new palette instead of
 * being stuck with whatever hour it launched in. */
export const LIGHTING_REFRESH_MS = 10 * 60 * 1000 // 10 minutes

export interface SceneRig {
  scene: THREE.Scene
  camera: THREE.PerspectiveCamera
  renderer: THREE.WebGLRenderer
  controls: OrbitControls
  ambientLight: THREE.AmbientLight
  mainLight: THREE.DirectionalLight
  fillLight: THREE.DirectionalLight
  shadowCatcher: THREE.Mesh
  /** Re-evaluate getTimeOfDayLighting() and start easing toward it. */
  refreshTimeOfDayLighting: () => void
  /** Per-frame lighting lerp step — eases toward the target set by
   * refreshTimeOfDayLighting() over ~2s instead of snapping. */
  advanceLightingLerp: (deltaTime: number) => void
  /** True while a time-of-day lighting lerp is easing — an in-flight
   * lighting transition disqualifies the render loop's idle throttle. */
  isLightingLerpActive: () => boolean
  dispose: () => void
}

/** Build the once-per-mount WebGL scene: renderer, camera, lights, fog,
 * shadow catcher, controls. Throws if the WebGL renderer cannot be created
 * (the component catches and shows its "3D unavailable" status — same
 * all-or-nothing behavior as the original inline try/catch). */
export const createSceneRig = ({
  canvas,
  variant,
  shadowCenter,
  debugLog,
}: {
  canvas: HTMLCanvasElement
  variant: 'full' | 'pet'
  shadowCenter: [number, number, number]
  debugLog: (...args: unknown[]) => void
}): SceneRig => {
  // Setup Three.js scene. No opaque background: the renderer clears to
  // transparent so the styled .vrm-panel gradients show behind the set.
  const scene = new THREE.Scene()
  debugLog('✅ Scene created')

  const camera = new THREE.PerspectiveCamera(
    45,
    canvas.clientWidth / canvas.clientHeight,
    0.1,
    100
  )
  // Placeholder pose until the VRM's bones are available to compute the
  // real framing (see computeDefaultFraming, applied on load-complete).
  camera.position.copy(PRE_LOAD_CAMERA_POSITION)
  debugLog('✅ Camera configured')

  const renderer = new THREE.WebGLRenderer({
    canvas,
    antialias: true,
    alpha: true,
  })
  // updateStyle=false: CSS owns the canvas layout size (100% of the pane).
  // The default writes inline px styles, which pins the canvas at whatever
  // size it had mid-layout (e.g. while the desktop shell is still settling)
  // and it never recovers without a window resize.
  renderer.setSize(canvas.clientWidth, canvas.clientHeight, false)
  // Cap at 2x: with MSAA + soft shadows, native 3x DPR renders up to 9x the
  // CSS pixel count for no visible gain — a large GPU/battery cost on HiDPI
  // laptops and the Capacitor mobile targets.
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
  renderer.shadowMap.enabled = true
  renderer.shadowMap.type = THREE.PCFSoftShadowMap
  // Static shadows (idle-perf): the only shadow caster is the VRM itself,
  // so once she's effectively motionless (the idle throttle in the render
  // loop) re-rendering the 2048² PCFSoft shadow map every frame buys
  // nothing. autoUpdate off; every frame that renders while NOT idle sets
  // shadowMap.needsUpdate = true instead (plus the first frame and the
  // VRM-load frame via the component's shadowMapNeedsRefresh), so shadows
  // only freeze while the pose is static — idle breathing/springbone
  // micro-motion under a soft contact shadow is imperceptible.
  renderer.shadowMap.autoUpdate = false
  // Env/lighting pass: three r0.180 defaults to NoToneMapping, which left
  // the mainLight's brighter highlights (and any HDR-ish emissive bits in
  // the environment GLBs) clipping straight to flat white instead of
  // rolling off. ACES gives a filmic shoulder so highlights compress
  // instead of clipping; exposure 1.0 is the starting point (the
  // time-of-day ambient table below got a small intensity bump to hold
  // parity with the pre-ACES midtones, since ACES darkens them a touch).
  renderer.toneMapping = THREE.ACESFilmicToneMapping
  renderer.toneMappingExposure = 1.0
  debugLog('✅ Renderer configured')

  // Lighting — color temperature nods to the local time of day. Initial
  // values are set immediately; getTimeOfDayLighting() is then re-evaluated
  // every 10 minutes (see the component's lightingRefreshInterval) so a
  // session left open across a dawn/day/evening/night boundary drifts into
  // the new palette instead of being stuck with whatever hour it launched in.
  const timeOfDay = getTimeOfDayLighting()

  const ambientLight = new THREE.AmbientLight(timeOfDay.ambientColor, timeOfDay.ambientIntensity)
  scene.add(ambientLight)

  const mainLight = new THREE.DirectionalLight(timeOfDay.mainColor, 0.8)
  mainLight.castShadow = true
  scene.add(mainLight)
  scene.add(mainLight.target)

  // Real shadows (item 2): mapSize 2048 for a crisp contact shadow under
  // her rather than a blurry smear; bias nudged negative to fight acne
  // without introducing visible peter-panning at this scale. The
  // shadow-center-dependent frustum fitting lives in applyShadowCenter
  // (shared with the environment-swap path).
  mainLight.shadow.mapSize.set(2048, 2048)
  mainLight.shadow.camera.near = 0.5
  mainLight.shadow.camera.far = 20
  mainLight.shadow.bias = -0.0005

  const fillLight = new THREE.DirectionalLight(timeOfDay.fillColor, 0.3)
  fillLight.position.set(-2, 2, -1)
  scene.add(fillLight)
  debugLog('✅ Lighting configured', timeOfDay.label)

  // Fog / depth cueing (item 3): FogExp2 (exponential-squared) reads more
  // natural than linear Fog for an interior/exterior mixed set and avoids
  // a visible start-distance seam. Density + color are re-evaluated in
  // the same time-of-day refresh/lerp path below so night doesn't read as
  // a hard black void past the environment GLB's modeled extent, while
  // day stays subtle enough not to wash out the set.
  scene.fog = new THREE.FogExp2(timeOfDay.fogColor, timeOfDay.fogDensity)

  // Shadow catcher (item 2 cont.): the environment GLB's own floor mesh
  // gets receiveShadow=true once it loads, but it's asynchronous
  // (and its authored materials may not respond well to shadows depending
  // on how they're set up), so a dedicated invisible-except-for-shadow
  // catcher plane is added directly under Kurisu's anchor spot as a
  // guaranteed fallback. ShadowMaterial only ever darkens what's behind it
  // by the shadow factor, so if the GLB floor is also a good catcher this
  // just adds a second, coincident soft shadow rather than a visible
  // double plane.
  const shadowCatcherGeometry = new THREE.PlaneGeometry(6, 6)
  const shadowCatcherMaterial = new THREE.ShadowMaterial({ opacity: 0.35 })
  const shadowCatcher = new THREE.Mesh(shadowCatcherGeometry, shadowCatcherMaterial)
  shadowCatcher.rotation.x = -Math.PI / 2
  shadowCatcher.receiveShadow = true
  scene.add(shadowCatcher)

  // Fit light/frustum/catcher around the active environment's shadow
  // center — the exact same call the environment effect makes on a swap.
  applyShadowCenter(mainLight, shadowCatcher, shadowCenter)

  // Lighting lerp state: rather than snapping colors/intensities the
  // instant the 10-minute re-check fires, the render loop eases
  // toward the new target over ~2s so a boundary crossing (e.g. day ->
  // evening) reads as a slow dusk, not a hard cut.
  const lightingLerpState = {
    active: false,
    elapsed: 0,
    duration: 2,
    fromAmbientColor: new THREE.Color(timeOfDay.ambientColor),
    toAmbientColor: new THREE.Color(timeOfDay.ambientColor),
    fromAmbientIntensity: timeOfDay.ambientIntensity,
    toAmbientIntensity: timeOfDay.ambientIntensity,
    fromMainColor: new THREE.Color(timeOfDay.mainColor),
    toMainColor: new THREE.Color(timeOfDay.mainColor),
    fromFillColor: new THREE.Color(timeOfDay.fillColor),
    toFillColor: new THREE.Color(timeOfDay.fillColor),
    fromFogColor: new THREE.Color(timeOfDay.fogColor),
    toFogColor: new THREE.Color(timeOfDay.fogColor),
    fromFogDensity: timeOfDay.fogDensity,
    toFogDensity: timeOfDay.fogDensity,
  }

  const refreshTimeOfDayLighting = () => {
    const next = getTimeOfDayLighting()
    lightingLerpState.fromAmbientColor.copy(ambientLight.color)
    lightingLerpState.toAmbientColor.set(next.ambientColor)
    lightingLerpState.fromAmbientIntensity = ambientLight.intensity
    lightingLerpState.toAmbientIntensity = next.ambientIntensity
    lightingLerpState.fromMainColor.copy(mainLight.color)
    lightingLerpState.toMainColor.set(next.mainColor)
    lightingLerpState.fromFillColor.copy(fillLight.color)
    lightingLerpState.toFillColor.set(next.fillColor)
    lightingLerpState.fromFogColor.copy(
      scene.fog instanceof THREE.FogExp2 ? scene.fog.color : lightingLerpState.toFogColor
    )
    lightingLerpState.toFogColor.set(next.fogColor)
    lightingLerpState.fromFogDensity =
      scene.fog instanceof THREE.FogExp2 ? scene.fog.density : lightingLerpState.toFogDensity
    lightingLerpState.toFogDensity = next.fogDensity
    lightingLerpState.elapsed = 0
    lightingLerpState.active = true
    debugLog('🌗 Lighting refresh ->', next.label)
  }

  // Lighting refresh lerp (item 7): eases toward the target set by
  // refreshTimeOfDayLighting() over ~2s instead of snapping.
  const advanceLightingLerp = (deltaTime: number) => {
    if (!lightingLerpState.active) return
    lightingLerpState.elapsed = Math.min(lightingLerpState.elapsed + deltaTime, lightingLerpState.duration)
    const t = lightingLerpState.duration > 0 ? lightingLerpState.elapsed / lightingLerpState.duration : 1
    ambientLight.color.copy(lightingLerpState.fromAmbientColor).lerp(lightingLerpState.toAmbientColor, t)
    ambientLight.intensity =
      lightingLerpState.fromAmbientIntensity +
      (lightingLerpState.toAmbientIntensity - lightingLerpState.fromAmbientIntensity) * t
    mainLight.color.copy(lightingLerpState.fromMainColor).lerp(lightingLerpState.toMainColor, t)
    fillLight.color.copy(lightingLerpState.fromFillColor).lerp(lightingLerpState.toFillColor, t)
    if (scene.fog instanceof THREE.FogExp2) {
      scene.fog.color.copy(lightingLerpState.fromFogColor).lerp(lightingLerpState.toFogColor, t)
      scene.fog.density =
        lightingLerpState.fromFogDensity +
        (lightingLerpState.toFogDensity - lightingLerpState.fromFogDensity) * t
    }
    if (t >= 1) {
      lightingLerpState.active = false
    }
  }

  // Controls
  const controls = new OrbitControls(camera, canvas)
  controls.target.copy(PRE_LOAD_CAMERA_TARGET)
  controls.enableDamping = true
  controls.dampingFactor = 0.05
  // Pet overlay: fixed framing — orbiting a frameless desktop window with
  // no visible chrome reads as broken, and the drag-bar owns the pointer.
  if (variant === 'pet') controls.enabled = false
  controls.update()

  return {
    scene,
    camera,
    renderer,
    controls,
    ambientLight,
    mainLight,
    fillLight,
    shadowCatcher,
    refreshTimeOfDayLighting,
    advanceLightingLerp,
    isLightingLerpActive: () => lightingLerpState.active,
    dispose: () => {
      controls.dispose()
      renderer.dispose()
    },
  }
}
