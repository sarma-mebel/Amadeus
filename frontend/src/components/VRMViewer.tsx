import { useRef, useEffect, useState } from 'react'
import * as THREE from 'three'
import { VRM } from '@pixiv/three-vrm'
import { resolveAssetPath } from '../config'
import { loadVolumePrefs, soundscape } from '../soundscape'
import { APP_VOLUME_CHANGE_EVENT, createLipSyncController } from '../vrm/lipSyncController'
import { createAnimationScheduler, warnUnresolvedGestureClipNames } from '../vrm/animationScheduler'
import { startPetIdleBehaviors, type PetIdleBehaviors } from '../vrm/petIdleBehaviors'
import {
  advanceDoze,
  createDozeMachineState,
  isDozing,
  wakeDoze,
  type DozeMachineState,
} from '../vrm/petDoze'
import { DEFAULT_WARMTH, startArcWarmth, type ArcWarmth } from '../vrm/arcWarmth'
import { readWindStrengthPref } from '../vrm/springboneWind'
import type { LipSyncState } from '../vrm/lipSyncController'
import {
  loadAnimationClips,
  normalizeStaticPath,
  resolveAnimationFiles,
} from '../vrm/clipLoader'
import { kurisuBridge } from '../vrm/kurisuBridge'
import {
  ENVIRONMENTS,
  ENVIRONMENT_CHANGE_EVENT,
  loadEnvironmentId,
  type EnvironmentDefinition,
  type EnvironmentId,
} from '../environments'
import {
  DEFAULT_MODEL,
  fetchModelRegistry,
  loadModelId,
  MODEL_CHANGE_EVENT,
  resolveModelFile,
  saveModelId,
} from '../models'
import { computeDefaultFraming } from '../vrm/cameraFraming'
import { loadPetPreset, PET_PRESET_CHANGE_EVENT, petPresetToFramingMode } from '../vrm/petPreset'
import {
  createSceneRig,
  LIGHTING_REFRESH_MS,
  shadowCenterForEnvironment,
  type SceneRig,
} from '../vrm/sceneSetup'
import { createVrmLoader, prepareLoadedVrm } from '../vrm/avatarLoader'
import {
  disposeEnvironmentGroup,
  loadEnvironmentGroup,
  repositionForEnvironment,
} from '../vrm/environmentManager'
import { createAdaptiveQuality } from '../vrm/adaptiveQuality'
import { createPostFxForViewer } from '../vrm/postfx'
import { applyRimLighting } from '../vrm/rimLight'
import { createProceduralBody } from '../vrm/proceduralBody'
import {
  createBlinkState,
  createExpressionEasingState,
  createMoodBaselineState,
  createExpressionDriver,
  type BlinkState,
  type ExpressionEasingState,
  type MoodBaselineState,
} from '../vrm/expressionDriver'
import {
  BLUSH_PATTERNS,
  BROW_RAISE_PATTERNS,
  findFaceChannel,
  type FaceChannel,
} from '../vrm/faceMorphs'
import { diagnostics } from '../diagnostics'
import {
  createGazeDirector,
  createGazeRuntimeState,
  type GazeRuntimeState,
} from '../vrm/gazeDirector'
import { createCursorGazeState, startCursorGazePolling } from '../vrm/cursorGaze'
import { isTauri } from '../platform'
import { createCameraDirector } from '../vrm/cameraDirector'
import { createPointerInteractions } from '../vrm/pointerInteractions'
import { createRenderLoop } from '../vrm/renderLoop'
import { applySpringboneWind, createSpringboneWindState } from '../vrm/springboneWind'
import { POKE_ACTION_NAMES, POKE_REGION_CLIPS } from '../vrm/pokeRegions'
import { applyRelaxedFingerPose } from '../vrm/fingerPose'
import { createInertializer, loadInertializationEnabled } from '../vrm/inertialization'
import { createIdleFrameIntervalS, createIsIdleNow, createCursorGazeWakeTracker } from '../vrm/idleThrottle'
import { createPetPerchHandler } from '../vrm/petPerch'
import { createAttentionEvents } from '../vrm/attentionEvents'
import { createReflexEngine } from '../vrm/reflexes'
import { isPetTucked } from '../vrm/petTuck'
import {
  buildVrmAnimationsBridge,
  buildLipSyncBridge,
  buildExpressionsBridge,
  deriveAutomaticPoolNames,
} from '../vrm/avatarBridgeWiring'
import {
  placeAndFrameLoadedVrm,
  progressPercent,
  reframeForPetPresetChange,
  resolveRequestedModel,
  shouldRetryWithDefaultModel,
} from '../vrm/modelLoadFlow'
import './VRMViewer.css'

// Verbose scene/asset logging is a debugging aid, not product output; opt in
// with localStorage.kurisu_debug = '1' (same flag as the chat latency readout).
const VRM_DEBUG =
  typeof window !== 'undefined' && window.localStorage.getItem('kurisu_debug') === '1'
const debugLog = (...args: unknown[]) => {
  if (VRM_DEBUG) console.log(...args)
}

type VrmLoadStatus =
  | { phase: 'loading'; progress: number }
  | { phase: 'ready' }
  | { phase: 'error'; message: string }

interface VRMViewerProps {
  /** 'pet' = desktop-pet overlay: transparent background (no environment
   * GLB), fixed camera (OrbitControls disabled). Default 'full'. */
  variant?: 'full' | 'pet'
}

const VRMViewer = ({ variant = 'full' }: VRMViewerProps = {}) => {
  const [loadStatus, setLoadStatus] = useState<VrmLoadStatus>({ phase: 'loading', progress: 0 })
  // Model (outfit) reload signal: bumping this tears down and remounts the
  // whole scene effect below (VRM re-fetch, fresh renderer/camera/animation
  // loop) — an outfit swap genuinely is a different VRM asset. Environment
  // changes are handled separately (environmentReloadToken below) and do
  // NOT touch this token, so switching backgrounds no longer re-downloads
  // the VRM or tears down the WebGL scene/AudioContext.
  const [modelReloadToken, setModelReloadToken] = useState(0)
  // Environment reload signal: bumping this only re-runs the environment
  // effect (GLB swap, shadow-center reposition, VRM anchor reposition,
  // lip-sync reverb tuning) — see the environment effect below. The scene/
  // renderer/camera/VRM/animation-loop effect does not depend on this.
  const [environmentReloadToken, setEnvironmentReloadToken] = useState(0)
  // Which environment GLB/anchor is active. Mirrors localStorage — kept as a
  // ref (not state) because the environment effect reads it once per run
  // and the scene effect reads it live every frame (shadow/anchor placement
  // must reflect whichever environment is currently loaded, without forcing
  // the scene effect itself to depend on environment changes). SettingsPanel
  // changes this via localStorage + a window event; the listener effect
  // further down updates the ref and bumps environmentReloadToken.
  const environmentIdRef = useRef<EnvironmentId>(loadEnvironmentId())
  // The full definition for whichever environment is currently loaded
  // (anchor, shadowCenter, audio/reverb) — kept live by the environment
  // effect so other effects (lip-sync reverb, VRM anchor placement on
  // outfit swap) always read the CURRENT environment instead of whatever
  // was active when they last ran. This is the key piece that decouples
  // "environment changed" from "scene must rebuild": nothing that reads
  // this ref needs to re-run just because the environment changed — only
  // the environment effect itself needs to update it and re-derive the
  // scene-graph-visible consequences (GLB, shadow center, VRM position).
  const activeEnvironmentRef = useRef<EnvironmentDefinition>(
    ENVIRONMENTS[environmentIdRef.current] ?? ENVIRONMENTS.time_machine
  )
  // Which VRM (outfit) to load this mount — same ref-mirror pattern as
  // environmentIdRef: SettingsPanel changes this via localStorage + a
  // window event, and the listener effect further down just updates the
  // ref and bumps modelReloadToken to force a full scene remount.
  const modelIdRef = useRef<string>(loadModelId())
  // Guards the load-failure fallback below so a broken *fallback* model
  // can't loop forever retrying the default.
  const modelRetriedRef = useRef<boolean>(false)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  // Scene-graph handles published by the scene effect so the environment
  // effect (below) can swap the environment GLB and reposition
  // shadow-center-derived objects on the SAME live scene/renderer, instead
  // of needing its own copies or forcing the scene effect to depend on
  // environment changes. Null before the scene effect has run its first
  // synchronous body / after it tears down.
  const sceneRef = useRef<THREE.Scene | null>(null)
  const cameraRef = useRef<THREE.PerspectiveCamera | null>(null)
  const rendererRef = useRef<THREE.WebGLRenderer | null>(null)
  const mainLightRef = useRef<THREE.DirectionalLight | null>(null)
  const shadowCatcherRef = useRef<THREE.Mesh | null>(null)
  // The currently-loaded environment GLB's root group, owned by the
  // environment effect (below) — tracked so a later swap can remove/dispose
  // it before loading the next one. Reset to null whenever the scene effect
  // tears down (a fresh scene mount has no environment loaded yet).
  const currentEnvironmentGroupRef = useRef<THREE.Group | null>(null)
  // Deferred-assignment hook (same pattern as talkAnimationHooks/perchHooks):
  // an environment swap teleports the VRM to the new set's anchor — possibly
  // rotated 90° — so the camera must re-derive its shot from her new bone
  // positions, but the camera/controls/cameraDirector that requires are
  // locals of the scene effect. The scene effect assigns this once the
  // director exists; the environment effect calls it after repositioning.
  const reframeForEnvironmentRef = useRef<(() => void) | null>(null)
  const vrmRef = useRef<VRM | null>(null)
  const animationMixerRef = useRef<THREE.AnimationMixer | null>(null)
  const currentActionRef = useRef<THREE.AnimationAction | null>(null)
  const currentClipNameRef = useRef<string | null>(null)
  const currentModeRef = useRef<'idle' | 'actual'>('idle')
  const scheduledActualTimeoutRef = useRef<number | null>(null)
  const mixerFinishedListenerRef = useRef<((event: THREE.Event & { action: THREE.AnimationAction }) => void) | null>(null)
  // Wall-clock (performance.now()) of the last user-initiated interaction
  // (poke, chat send) — gates the rare unattended gesture beat so it only
  // fires when nobody's actively engaging with her.
  const lastInteractionAtRef = useRef<number>(performance.now())
  // Target-based expression easing state, driven every frame from the
  // render loop (see expressionDriver.ts). Held on refs — not inside the
  // driver factory — so the eased face and mood EMA survive scene-effect
  // re-runs (an outfit swap must not reset her mood).
  const expressionStateRef = useRef<ExpressionEasingState>(createExpressionEasingState())
  // Item 5b: mood-biased resting face (see MoodBaselineState in
  // expressionDriver.ts for the full rationale).
  const moodBaselineStateRef = useRef<MoodBaselineState>(createMoodBaselineState())
  // Eased blink state machine, advanced by deltaTime in the render loop.
  const blinkStateRef = useRef<BlinkState>(createBlinkState())
  // Gaze proxy/pointer state (see GazeRuntimeState in gazeDirector.ts) —
  // written by the pointer-move handler, steered by advanceGaze.
  const gazeStateRef = useRef<GazeRuntimeState>(createGazeRuntimeState())
  // Cursor gaze-follow (pet + Tauri only, see ../vrm/cursorGaze): written
  // by the poller started below, read by gazeDirector as the lowest-
  // priority gaze input. Created unconditionally (cheap plain object) so
  // the gaze director always has a stable reference to pass, even on
  // variants/platforms where polling never starts and it just sits inert.
  const cursorGazeStateRef = useRef(createCursorGazeState())
  // Doze state machine (pet variant, see ../vrm/petDoze) — a ref (not
  // effect-local) so it survives an outfit-swap remount exactly like the
  // expression/mood refs above: her sleep progress shouldn't reset just
  // because the user changed clothes. Created unconditionally (cheap plain
  // object, inert unless advanced) so non-pet variants/effects that read it
  // defensively never need a null check beyond "was it ever advanced".
  const dozeStateRef = useRef<DozeMachineState>(createDozeMachineState())
  // Perch state (pet variant, see the 'kurisu-pet-perch-changed' listener
  // below): true while the sibling-owned pet UI reports she's docked to a
  // screen edge — suppresses the ambient idle beat and, when a sit clip
  // exists, drives the looping sit pose. Ref (not state) since only
  // imperative effects (the ambient-idle busy gate, the perch listener
  // itself) read/write it; no render depends on it.
  const perchedRef = useRef<boolean>(false)
  const lipSyncStateRef = useRef<LipSyncState>({
    audioContext: null,
    analyser: null,
    dryGain: null,
    wetGain: null,
    convolver: null,
    gainNode: null,
    appVolumeGain: null,
    sourceNode: null,
    freqData: null,
    timeData: null,
    expressionValues: { aa: 0, ih: 0, ee: 0, ou: 0, oh: 0, relaxed: 0.4 },
    previousViseme: null,
    scheduledSources: [],
    nextChunkStartTime: 0,
    streaming: false,
    streamDraining: false,
    speechEnvelope: 0,
    sentenceStartTimeoutIds: [],
  })

  // Environment switching (SettingsPanel's Environment select): the pref is
  // already persisted to localStorage by the time this fires, so this
  // listener just needs to pick it up and bump environmentReloadToken —
  // which re-runs ONLY the environment effect below (GLB swap, shadow
  // reposition, VRM reposition, reverb tuning). The scene/renderer/VRM/
  // animation-loop effect does not depend on this token, so switching
  // backgrounds no longer tears down the WebGL scene, re-fetches the VRM,
  // or resets the AudioContext/lip-sync graph.
  useEffect(() => {
    const handleEnvironmentChange = () => {
      environmentIdRef.current = loadEnvironmentId()
      // B2: swap the ambience preset (machine hum vs. street) to match the
      // newly selected set. Live no-op if ambience isn't currently running —
      // the next startAmbience() call (ChatInterface, on send) will just
      // pick up the new preset directly via soundscape's activePreset.
      const nextAudio = ENVIRONMENTS[environmentIdRef.current]?.audio
      soundscape.setAmbiencePreset(nextAudio?.ambience ?? 'machine')
      setEnvironmentReloadToken((token) => token + 1)
    }
    window.addEventListener(ENVIRONMENT_CHANGE_EVENT, handleEnvironmentChange)
    return () => window.removeEventListener(ENVIRONMENT_CHANGE_EVENT, handleEnvironmentChange)
  }, [])

  // Outfit switching (SettingsPanel's Outfit select): the pref is already
  // persisted to localStorage by the time this fires, so this just updates
  // the ref and forces the big scene effect to tear down/reload against the
  // new modelIdRef value — an outfit swap is a genuinely different VRM
  // asset, so (unlike environment changes) this still fully remounts the
  // scene.
  useEffect(() => {
    const handleModelChange = () => {
      modelIdRef.current = loadModelId()
      modelRetriedRef.current = false
      setLoadStatus({ phase: 'loading', progress: 0 })
      setModelReloadToken((token) => token + 1)
    }
    window.addEventListener(MODEL_CHANGE_EVENT, handleModelChange)
    return () => window.removeEventListener(MODEL_CHANGE_EVENT, handleModelChange)
  }, [])

  useEffect(() => {
    if (!canvasRef.current) return
    // Captured once so the cleanup below always removes listeners from the
    // same element it added them to, even if the ref is repointed later.
    const canvasEl = canvasRef.current

    // StrictMode double-mounts: the first mount's in-flight loader callbacks
    // must not touch state or start loops owned by the second mount.
    let disposed = false
    // Item 1/2: detected once the VRM loads (see the expression-logging
    // block inside the load callback below); declared here at effect scope
    // so the procedural body / the flustered beat (defined later in this
    // same effect, but outside the load callback's closure) can read them.
    // Channels, not expression names: discovery falls back from the VRM
    // expression map to raw mesh morphs (see ../vrm/faceMorphs — an audit
    // found no bundled model names a blush/brow *expression*, so the old
    // name-based wiring had been silently dead on the whole fleet).
    let blushChannel: FaceChannel | null = null
    let browChannel: FaceChannel | null = null
    // Flustered blush beat: eased per-frame instead of the old raw
    // setValue + setTimeout hard-reset (which bypassed every easing path
    // and could strand a lit blush if the timeout lost a race). target
    // holds until `until`, then decays to 0; the render loop advances it.
    const blushBeat = { level: 0, target: 0, until: 0 }
    // Local (non-ref) handle to the gaze proxy Object3D so cleanup below
    // doesn't read gazeStateRef.current (an effect-cleanup/ref-timing lint
    // warning) — this effect owns exactly one gaze proxy per mount, created
    // once the VRM loads.
    let gazeProxyForCleanup: THREE.Object3D | null = null
    // Same pattern for the mixer 'finished' listener: the scheduler assigns
    // mixerFinishedListenerRef.current exactly once, synchronously inside
    // createAnimationScheduler (see the load callback below), and reading
    // the ref directly in the effect cleanup is the same lint warning.
    let mixerFinishedListenerForCleanup:
      | ((event: THREE.Event & { action: THREE.AnimationAction }) => void)
      | null = null
    // Same deferred-assignment pattern: startPetIdleBehaviors is called
    // inside the VRM-load callback (it needs the scheduler instance), so
    // the timer handle it returns is only available for cleanup via this
    // effect-scoped local rather than a ref.
    let petIdleBehaviorsForCleanup: PetIdleBehaviors | null = null
    // Pet-variant pointer-over-window listeners for the idle throttle (item
    // 3) — declared here so effect cleanup can remove them regardless of
    // where in the effect body they were attached (see the isIdleNow block
    // below, which owns setting this).
    let petPointerHoverCleanup: (() => void) | null = null
    // Static-shadow refresh request (renderer.shadowMap.autoUpdate is off —
    // see createSceneRig): true forces the next rendered frame to re-render
    // the shadow map even if the avatar is idle — set at mount for the
    // first frame, and again when the VRM (the scene's only shadow caster)
    // loads in.
    let shadowMapNeedsRefresh = true
    setLoadStatus({ phase: 'loading', progress: 0 })

    debugLog('=== INITIALIZING VRM VIEWER ===')

    const enableDiscovery = window.__kurisuDiscoverAnimations ?? false

    // Always try the default manifest location — deployments that curate
    // clips (roles, sourced packs, talk gestures) publish one there, and the
    // loader already falls back to the built-in presets on 404.
    const manifestPath =
      normalizeStaticPath(window.__kurisuAnimationManifestPath) ??
      resolveAssetPath('/static/animations/manifest.json')

    // Scene rig: scene/camera/renderer/lights/fog/shadow catcher/controls +
    // the time-of-day lighting lerp (see ../vrm/sceneSetup). The shadow
    // center is read live off activeEnvironmentRef so an environment swap
    // (handled by the separate environment effect below) is immediately
    // visible here without this effect re-running; the environment effect
    // re-derives it in place when the environment actually changes.
    let rig: SceneRig
    try {
      rig = createSceneRig({
        canvas: canvasEl,
        variant,
        shadowCenter: shadowCenterForEnvironment(activeEnvironmentRef.current),
        debugLog,
      })
    } catch (error) {
      console.error('❌ Failed to create WebGL renderer:', error)
      setLoadStatus({
        phase: 'error',
        message: '3D rendering is unavailable on this device (WebGL failed to start). Chat still works.',
      })
      return () => {
        disposed = true
      }
    }
    const { scene, camera, renderer, controls } = rig
    const postfx = createPostFxForViewer(renderer, scene, camera, variant) // ../vrm/postfx: pet bypass + pref
    rendererRef.current = renderer
    // Published only now that the renderer is confirmed working — mirrors
    // the original code's all-or-nothing early return on WebGL failure
    // (above): the environment effect gates on sceneRef being set, so a
    // renderer failure must leave it null, or the environment effect would
    // still fetch a GLB into a scene that will never actually render.
    sceneRef.current = scene
    cameraRef.current = camera
    mainLightRef.current = rig.mainLight
    shadowCatcherRef.current = rig.shadowCatcher

    const lightingRefreshInterval = window.setInterval(
      rig.refreshTimeOfDayLighting,
      LIGHTING_REFRESH_MS
    )

    // Social gaze layer + per-frame lookAt-proxy steering (typing
    // attentiveness, emotion-driven gaze aversion, eye-contact lock-on,
    // micro-saccades) — see ../vrm/gazeDirector. `gazeSocial` is the
    // director's mutable state object; the window-event handlers below and
    // the scheduler's flustered beat write its fields directly.
    // Deferred hook (same pattern as talkAnimationHooks below): the gaze
    // director needs the blink machine for event-coupled blinks, but the
    // expression driver that owns it is constructed a few lines later
    // (and itself needs gazeDirector.applyEmotionGaze).
    const blinkHooks: { requestBlink: ((probability?: number) => void) | null } = {
      requestBlink: null,
    }
    const gazeDirector = createGazeDirector({
      camera,
      gazeStateRef,
      isSpeaking: () => lipSyncStateRef.current.streaming,
      cursorGazeState: cursorGazeStateRef.current,
      // Event-coupled blinks (lifelike pass): big deliberate gaze shifts
      // blink near-always, small ones sometimes.
      onGazeShift: (magnitude) => blinkHooks.requestBlink?.(0.4 + 0.6 * magnitude),
    })
    const gazeSocial = gazeDirector.social

    // Cursor gaze-follow (pet + Tauri only): polls pet_cursor_context and
    // writes normalized offsets into cursorGazeStateRef, which the gaze
    // director above reads every frame as its lowest-priority input. A
    // plain browser tab or the main window never starts this — see
    // startCursorGazePolling's own isTauri() guard, duplicated here so the
    // main window doesn't even attempt the dynamic import.
    // Arc-driven warmth (pet variant only, see ../vrm/arcWarmth): fetches
    // the relationship overview once at mount + hourly thereafter. Started
    // unconditionally for variant==='pet' (not gated on isTauri like cursor
    // gaze — the memory API is a plain HTTP endpoint, reachable from a
    // browser-hosted pet view too, unlike the Tauri-only cursor IPC command
    // below). `.get()` is read every frame by cursorGaze's engagement bias
    // and by the ambient-idle wiring inside the VRM-load callback.
    // Declared BEFORE cursorGazePoller so the poller can read arcWarmth's
    // getter from its very first poll.
    const arcWarmth: ArcWarmth | null = variant === 'pet' ? startArcWarmth() : null

    const cursorGazePoller =
      variant === 'pet' && isTauri()
        ? startCursorGazePolling(cursorGazeStateRef.current, {
            getWarmth: () => arcWarmth?.get() ?? 1,
          })
        : null

    // Facial expressions: eased set/hold/release, the mood-EMA resting
    // face, and blinks — see ../vrm/expressionDriver. Kurisu-flavored gaze
    // semantics ride the same per-sentence signal via onEmotionCue.
    const expressionDriver = createExpressionDriver({
      getVrm: () => vrmRef.current,
      expressionStateRef,
      moodBaselineStateRef,
      blinkStateRef,
      onEmotionCue: gazeDirector.applyEmotionGaze,
      // Deliberately lazy: isLipSyncAudible is destructured from
      // createLipSyncController a few lines below — the driver only calls
      // this from the render loop, long after both exist. While audible,
      // the driver yields the 'relaxed' expression to the lip-sync tick's
      // jaw-slack duck instead of stomping it every frame.
      isLipSyncActive: () => isLipSyncAudible(),
    })
    const setKurisuExpression = expressionDriver.set
    blinkHooks.requestBlink = expressionDriver.requestBlink

    // Populated once the animation mixer sets up (below, inside the VRM
    // load callback) so beginStream/stopLipSync — defined here, before that
    // callback runs — can crossfade to a "talk" role clip without a direct
    // reference to playAnimation/idleNames, which don't exist until then.
    const talkAnimationHooks: { onSpeakingChange: ((speaking: boolean) => void) | null } = {
      onSpeakingChange: null,
    }

    // Voice playback + lip sync live in the controller module; the deps
    // close over effect-locals (talkAnimationHooks is assigned later inside
    // the load callback). getReverb reads activeEnvironmentRef LIVE (not a
    // fixed local captured at scene-mount time) so an environment swap's
    // reverb tuning reaches the lip-sync graph immediately without this
    // effect — and therefore the AudioContext/lip-sync graph it owns — ever
    // needing to re-run.
    const {
      stopLipSync,
      fadeOutAndStopLipSync,
      startLipSync,
      beginStream,
      enqueueLipSyncChunk,
      endLipSyncStream,
      // Per-frame viseme analysis tick — driven from the render loop's
      // onFrame below (the controller no longer runs its own concurrent
      // rAF loop).
      update: updateLipSyncVisemes,
      // Covers one-shot presence clips as well as streams — the bridge's
      // isSpeaking below must use this, not lipSyncStateRef's `streaming`.
      isAudible: isLipSyncAudible,
      // App-wide volume slider (item 4) — re-reads localStorage
      // `kurisu_volume` and pushes it to the post-analyser gain stage; see
      // the 'kurisu-volume-changed' listener below.
      applyAppVolume,
    } = createLipSyncController({
      stateRef: lipSyncStateRef,
      getVrm: () => vrmRef.current,
      getReverb: () => activeEnvironmentRef.current.audio?.reverb,
      onSpeakingChange: (speaking) => {
        window.dispatchEvent(new CustomEvent('kurisu:speaking', { detail: { speaking } }))
        talkAnimationHooks.onSpeakingChange?.(speaking)
        if (speaking) {
          // A reply just started speaking — counts as user engagement for the
          // "only gesture on a timer if she's been idle a while" gate (item 1).
          lastInteractionAtRef.current = performance.now()
        } else {
          // Event-coupled blink at speech end (VMagicMirror heuristic):
          // people blink when they finish talking.
          expressionDriver.requestBlink(0.9)
        }
      },
    })

    // Camera direction: follow-camera, 20s-idle auto-return, aspect
    // reframe, double-click reframe tween — see ../vrm/cameraDirector.
    const cameraDirector = createCameraDirector({
      camera,
      controls,
      getCanvas: () => canvasRef.current,
      getVrm: () => vrmRef.current,
    })
    const followState = cameraDirector.followState
    debugLog('✅ Camera controls configured')

    // Environment-swap reframe (see reframeForEnvironmentRef's doc comment):
    // recompute the bone-derived framing at her new anchor and snap to it —
    // the same computeDefaultFraming + adoptDefaultFraming path the initial
    // load uses, never a second framing implementation. A snap (not a tween)
    // is deliberate: the whole set just changed, and tweening would fly the
    // camera through the incoming geometry. No-op before the VRM loads (the
    // load callback computes the initial framing itself).
    reframeForEnvironmentRef.current = () => {
      const vrm = vrmRef.current
      const canvas = canvasRef.current
      if (!vrm || !canvas) return
      const aspect =
        canvas.clientHeight > 0 ? canvas.clientWidth / canvas.clientHeight : camera.aspect
      const framingMode = variant === 'pet' ? petPresetToFramingMode(loadPetPreset()) : 'medium'
      const framing = computeDefaultFraming(vrm, aspect, camera.fov, framingMode)
      if (!framing) return
      cameraDirector.adoptDefaultFraming(framing)
      controls.update()
    }

    // Poke raycast + double-click reframe + B4 pointer-gaze tracking —
    // see ../vrm/pointerInteractions.
    const pointerInteractions = createPointerInteractions({
      canvas: canvasEl,
      camera,
      getVrm: () => vrmRef.current,
      getMixer: () => animationMixerRef.current,
      gazeStateRef,
      followState,
      lastInteractionAtRef,
      onReframeRequest: cameraDirector.startReframeTween,
    })
    pointerInteractions.attach()

    // "Idle" window shared by the render loop's throttle (below) and the
    // pet ambient idle beat (inside the VRM-load callback, which runs
    // later but needs this constant already initialized — declared here,
    // ahead of startVrmLoad, rather than down by isIdleNow where it used
    // to live) — any user activity within this window disqualifies idle.
    const IDLE_ACTIVITY_WINDOW_MS = 3000

    // Environment GLB loading is owned by the separate environment effect
    // below (keyed on environmentReloadToken, not this effect) — it adds
    // the GLB to this same `scene` once this effect has published sceneRef.
    // See that effect for the load/unload logic that used to live inline
    // here; splitting it out is what lets an environment swap avoid
    // rebuilding everything below (VRM fetch, renderer, animation loop).

    // Load VRM model. The outfit registry is fetched (and cached module-
    // level — see models.ts) before resolving which file to load: on a
    // registry-miss (stale localStorage id, deleted entry) resolveModelFile
    // already falls back to DEFAULT_MODEL.file, and we also persist that
    // fallback back to localStorage so picker state and reality agree.
    const loader = createVrmLoader()

    const startVrmLoad = (fileToLoad: string) => {
      debugLog('\n=== LOADING KURISU VRM MODEL ===')
      debugLog('Model file:', fileToLoad)
      loader.load(
        resolveAssetPath(fileToLoad),
      async (gltf) => {
        if (disposed) return
        setLoadStatus({ phase: 'ready' })
        const vrm = gltf.userData.vrm as VRM
        prepareLoadedVrm(vrm, gltf.scene)

        // Anchor placement, rim lighting, scene insertion, the debug-harness
        // bridge, shader precompile, the camera-follow gaze proxy, and the
        // initial bone-derived camera framing all live in
        // ../vrm/modelLoadFlow's placeAndFrameLoadedVrm — see there for the
        // per-step rationale (unchanged, moved verbatim).
        const gazeProxy = placeAndFrameLoadedVrm(vrm, {
          scene,
          camera,
          renderer,
          cameraDirector,
          gazeStateRef,
          getActiveEnvironment: () => activeEnvironmentRef.current,
          getCanvas: () => canvasRef.current,
          mainLightColor: rig.mainLight.color,
          fillLightColor: rig.fillLight.color,
          variant,
          fileToLoad,
          debugLog,
        })
        vrmRef.current = vrm
        // The scene's only shadow caster just appeared — make sure the next
        // rendered frame refreshes the frozen shadow map even if the user
        // has gone idle while the model was downloading.
        shadowMapNeedsRefresh = true
        gazeProxyForCleanup = gazeProxy
        controls.update()

        debugLog('✅ VRM model loaded successfully')
        debugLog('VRM object:', vrm)

        // Log humanoid info
        if (vrm.humanoid) {
          const boneNames = Object.keys(vrm.humanoid.humanBones)
          debugLog('Available bones:', boneNames)
        }

        // Log expressions
        if (vrm.expressionManager) {
          debugLog('Available expressions:', Object.keys(vrm.expressionManager.expressionMap))
        }
        // Item 1: blush reaction beat — expression-first, morph-fallback
        // discovery (faceMorphs). No point layering a blush *clip* without
        // the model being able to show one; and per the degrade-loudly
        // convention, a model that can't blush must COUNT, not silently
        // skip (this exact feature was dead fleet-wide for weeks because
        // no bundled model names a blush expression).
        blushChannel = findFaceChannel(vrm, [/blush|embarrass|shy/i], BLUSH_PATTERNS)
        if (blushChannel) {
          debugLog(`😳 Blush channel: "${blushChannel.name}"`)
        } else {
          debugLog('😳 No blush expression/morph on this VRM; blush beat skipped')
          diagnostics.increment('avatar.blush_unavailable', fileToLoad)
        }
        // Item 2: eyebrow-raise-on-peak micro-motion. Same fallback: the
        // VRoid-rigged models (Kurisu included) carry Fcl_BRW_* morphs
        // that no VRM expression maps, so the morph path is what makes
        // this feature real on the flagship model.
        browChannel = findFaceChannel(vrm, [/brow|eyebrow/i], BROW_RAISE_PATTERNS)
        if (browChannel) {
          debugLog(`🤨 Brow channel: "${browChannel.name}"`)
        } else {
          diagnostics.increment('avatar.brow_unavailable', fileToLoad)
        }

        // Load VRM animations (idle + action sets). Resolution (manifest →
        // preset fallback → optional discovery) and per-clip loading/
        // retargeting live in ../vrm/clipLoader.
        debugLog('\n=== LOADING VRM ANIMATIONS ===')

        const { idleFiles, actionFiles } = await resolveAnimationFiles({
          manifestPath,
          enableDiscovery,
          debugLog,
        })

        const { loadedClips, clipMeta, clipAliases, talkClipNames, allFiles, ensureClipLoaded } =
          await loadAnimationClips({ idleFiles, actionFiles, vrm, debugLog })

        const availableNames = Object.keys(loadedClips)
        debugLog(`\n✅ Loaded ${availableNames.length} VRM animations`)

        if (availableNames.length === 0) {
          console.warn('No VRM animations available; skipping mixer setup')
          return
        }

        animationMixerRef.current = new THREE.AnimationMixer(vrm.scene)
        const mixer = animationMixerRef.current

        // Idle/action pool derivation (with the "nothing resolved" fallback)
        // lives in ../vrm/avatarBridgeWiring.
        const { idleNames, actionNames } = deriveAutomaticPoolNames(
          idleFiles,
          actionFiles,
          loadedClips,
          clipMeta
        )
        debugLog('Idle clips:', idleNames)
        debugLog('Action clips:', actionNames)

        // Body-animation state machine (idle loops, gesture beats, emotion
        // reactions) lives in the scheduler module; it mutates this
        // component's animation refs in place so the teardown and the poke
        // handler's mid-gesture check below keep reading them directly.
        const scheduler = createAnimationScheduler({
          mixer,
          clips: loadedClips,
          clipMeta,
          clipAliases,
          idleNames,
          actionNames,
          talkClipNames,
          ensureClipLoaded,
          // Toggle read lives here (not in the scheduler) so flipping
          // kurisu_inertialization applies to the very next transition; the
          // inertializer itself declines until its pose history is warm.
          requestInertializedTransition: (options) => {
            if (!loadInertializationEnabled()) return false
            return inertializer.requestTransition(options)
          },
          isSpeechStreaming: () => lipSyncStateRef.current.streaming,
          getLastInteractionAt: () => lastInteractionAtRef.current,
          onFlusteredBeat: (_emotion, intensity) => {
            // Embarrassed-adjacent beat: layer the blush channel on top of
            // the normal expression handling (kurisuExpressions.set, driven
            // separately by ChatInterface) rather than replacing it, and only
            // if this VRM actually has a blush-like expression or morph.
            // Eased by the render loop via blushBeat (no setTimeout hard
            // reset — that bypassed easing and could strand a lit blush).
            if (blushChannel) {
              blushBeat.target = Math.min(1, intensity - 0.2)
              blushBeat.until = performance.now() + 2200
            }
            // Flustered = blush + the tsundere look-away: down and to the
            // side for the blush's duration, overriding the eye-contact
            // lock a strong happy would otherwise set. Runs even without a
            // blush channel — the look-away carries the beat on models
            // that can't blush.
            gazeSocial.lockOnUntil = 0
            gazeSocial.aversionLateral = (Math.random() < 0.5 ? -1 : 1) * 0.22
            gazeSocial.aversionVertical = -0.18
            gazeSocial.aversionUntil = performance.now() + 1800
          },
          debugLog,
          currentActionRef,
          currentClipNameRef,
          currentModeRef,
          scheduledActualTimeoutRef,
          mixerFinishedListenerRef,
        })

        talkAnimationHooks.onSpeakingChange = scheduler.onSpeakingChange
        // Mirror for the effect cleanup (see mixerFinishedListenerForCleanup
        // above) — the scheduler has just assigned the ref synchronously.
        mixerFinishedListenerForCleanup = mixerFinishedListenerRef.current
        // Perch unperch hook (see the 'kurisu-pet-perch-changed' listener
        // above, declared earlier in this same effect but only wired up now
        // that the scheduler instance exists): explicitly resumes the idle
        // rotation after a looping sit clip, since playByName's loop:true
        // sets 'idle' mode, which the mixer's 'finished' event never fires
        // for.
        perchHooks.resumeIdle = () => scheduler.playRandomIdle(0.8)

        // Dev-mode safety net (item 3a, docs/animations.md): the emotion map
        // gets the same check inside createAnimationScheduler; this covers
        // this component's own by-name clip references (poke regions/
        // reactions), which have the identical silent-failure mode — a
        // rename or stray lazy: true just makes a poke gestureless.
        if (import.meta.env.DEV) {
          warnUnresolvedGestureClipNames(
            [...POKE_REGION_CLIPS.map((region) => region.clip), ...POKE_ACTION_NAMES],
            scheduler.resolveClipName,
            new Set(Object.keys(loadedClips))
          )
        }

        // Bridge object construction lives in ../vrm/avatarBridgeWiring
        // (pure wiring over these already-constructed locals).
        kurisuBridge.vrmAnimations = buildVrmAnimationsBridge({
          loadedClips,
          mixer,
          allFiles,
          idleNames,
          playByName: (name, options) => scheduler.playByName(name, options),
          getCurrentClipName: () => currentClipNameRef.current,
        })
        // Thin backward-compat alias — AnimLab and usePetChat still read
        // window.vrmAnimations directly; everything inside src should go
        // through kurisuBridge.vrmAnimations instead (same pattern as
        // window.kurisuLipSync below).
        window.vrmAnimations = kurisuBridge.vrmAnimations
        debugLog('\n💡 Animation controls available at kurisuBridge.vrmAnimations (window.vrmAnimations alias)')
        debugLog('   Try: kurisuBridge.vrmAnimations.play("Greeting")')

        kurisuBridge.react = scheduler.kurisuReact
        kurisuBridge.reactReset = scheduler.kurisuReactReset

        scheduler.playRandomIdle(0.6)

        // Pet ambient idle behaviors (see ../vrm/petIdleBehaviors): a slow,
        // separate cadence layered on top of the scheduler's own idle-
        // variety timer, pet-variant only — the main chat window already
        // has that timer and doesn't need a second, rarer one on top.
        // Started here (inside the VRM-load callback) rather than at
        // effect scope because it needs the scheduler instance; stopped in
        // this effect's teardown via petIdleBehaviorsForCleanup, same
        // pattern as mixerFinishedListenerForCleanup above.
        if (variant === 'pet') {
          const idleBehaviors = startPetIdleBehaviors({
            playRandomSubtleGesture: scheduler.playRandomSubtleGesture,
            hasSubtlePool: scheduler.hasSubtlePool,
            isBusyOrRecentlyInteracted: () =>
              isLipSyncAudible() ||
              currentModeRef.current === 'actual' ||
              performance.now() - lastInteractionAtRef.current < IDLE_ACTIVITY_WINDOW_MS ||
              // Doze/perch both suppress the ambient idle beat outright —
              // she shouldn't shift around while asleep, and while perched
              // the (possibly looping) sit clip owns her body, not the
              // subtle-gesture pool.
              isDozing(dozeStateRef.current) ||
              perchedRef.current,
            getWarmth: () => arcWarmth?.get() ?? DEFAULT_WARMTH,
          })
          petIdleBehaviorsForCleanup = idleBehaviors
        }

        // Doze bridge surface (pet variant only, see ../vrm/petDoze +
        // kurisuBridge.ts's KurisuPetDoze). wake() always returns whether
        // she was ACTUALLY dozing (not merely drowsy) at the moment of the
        // call — wakeDoze() itself implements that exact contract, this is
        // just a thin passthrough plus the interaction-timestamp bump every
        // other wake trigger (poke, pointer-enter) already gets.
        if (variant === 'pet') {
          kurisuBridge.petDoze = {
            active: () => isDozing(dozeStateRef.current),
            wake: () => {
              const wasDozing = wakeDoze(dozeStateRef.current)
              lastInteractionAtRef.current = performance.now()
              return wasDozing
            },
          }
        }

        kurisuBridge.lipSync = buildLipSyncBridge({
          startLipSync,
          fadeOutAndStopLipSync,
          beginStream,
          enqueueLipSyncChunk,
          endLipSyncStream,
          isLipSyncAudible,
        })
        // Thin backward-compat alias — LoadingScreen still reads
        // window.kurisuLipSync to voice the greeting; everything inside src
        // goes through kurisuBridge.
        window.kurisuLipSync = kurisuBridge.lipSync
        debugLog('🎙️ Lip sync controls available at window.kurisuLipSync.play({ base64Audio, mimeType })')
        debugLog('🎙️ Streamed playback: beginStream() -> enqueue({ base64Audio, mimeType }) x N -> endStream()')

        kurisuBridge.expressions = buildExpressionsBridge({ setExpression: setKurisuExpression, vrm })
        debugLog('😊 Expression controls available at kurisuBridge.expressions.set(name, intensity, duration)')
      },
      (progress) => {
        if (disposed) return
        const percent = progressPercent(progress.loaded, progress.total)
        setLoadStatus((prev) =>
          prev.phase === 'loading' ? { phase: 'loading', progress: percent } : prev,
        )
        debugLog(`Loading VRM: ${percent.toFixed(1)}%`)
      },
      (error) => {
        console.error('❌ Error loading VRM:', error)
        if (disposed) return
        // Load-failure fallback: a broken/removed non-default outfit file
        // shouldn't strand the user on the permanent error overlay — fall
        // back to the default model once. modelRetriedRef guards against a
        // busted *default* looping forever (that keeps the existing error
        // path instead). See ../vrm/modelLoadFlow for the retry-gate logic.
        if (shouldRetryWithDefaultModel(modelIdRef.current, modelRetriedRef.current)) {
          modelRetriedRef.current = true
          console.warn(
            `⚠️ Failed to load outfit "${modelIdRef.current}"; falling back to the default model.`
          )
          saveModelId(DEFAULT_MODEL.id)
          modelIdRef.current = DEFAULT_MODEL.id
          setModelReloadToken((token) => token + 1)
          return
        }
        setLoadStatus({
          phase: 'error',
          message: error instanceof Error ? error.message : 'The model could not be fetched.',
        })
      }
      )
    }

    void fetchModelRegistry().then((registry) => {
      if (disposed) return
      const requestedId = modelIdRef.current
      // Registry-miss fallback resolution lives in ../vrm/modelLoadFlow.
      const { file: resolvedFile, isRegistryMiss } = resolveRequestedModel(
        requestedId,
        registry,
        resolveModelFile
      )
      if (isRegistryMiss) {
        // Persist the fallback so SettingsPanel's picker state agrees with
        // what's actually on screen.
        saveModelId(DEFAULT_MODEL.id)
        modelIdRef.current = DEFAULT_MODEL.id
      }
      startVrmLoad(resolvedFile)
    })

    // Adaptive quality (dynamic resolution) — see ../vrm/adaptiveQuality for the thresholds and the Qwen3-TTS GPU-contention rationale.
    const adaptiveQuality = createAdaptiveQuality({
      setPixelRatioScale: (scale) => renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2) * scale),
      debugLog,
      setPostFxEnabled: postfx.setEnabled, // sheds the composer before resolution drops
    })

    // Inertialized clip transitions (../vrm/inertialization, opt-in via
    // kurisu_inertialization): spring-decayed pose offsets that replace the
    // crossfade when the scheduler's hook accepts. Declared here in effect
    // scope; the async VRM-load callback above closes over it for the
    // scheduler wiring (it only runs after the effect body completes).
    const inertializer = createInertializer({ getVrm: () => vrmRef.current })

    // Procedural body motion: the record-and-undo additive stack, breathing
    // + noise drift, head aim, and the onset-driven speech/listening nod
    // machinery — see ../vrm/proceduralBody.
    const proceduralBody = createProceduralBody({
      getVrm: () => vrmRef.current,
      getRawSpeechEnvelope: () => lipSyncStateRef.current.speechEnvelope ?? 0,
      getBrowChannel: () => browChannel,
      getGazeProxy: () => gazeStateRef.current.proxy,
      isGazeAverted: () => performance.now() < gazeSocial.aversionUntil,
      // Doze support (pet variant only — see ../vrm/petDoze): slower, deeper
      // breathing eased in across drowsy/dozing via eyeCloseAmount (already
      // a smooth 0->1 ramp through exactly that range, so reusing it here
      // keeps breathing and eye-close in visual lockstep instead of two
      // independently-tuned curves). Non-pet variants (and the pet variant
      // before doze ever engages) read the untouched {1, 1} default.
      getBreathingScale:
        variant === 'pet'
          ? () => {
              const amount = dozeStateRef.current.eyeCloseAmount
              return { rate: 1 - amount * 0.45, amplitude: 1 + amount * 0.8 }
            }
          : undefined,
      getDroopAmount: variant === 'pet' ? () => dozeStateRef.current.eyeCloseAmount : undefined,
    })

    // Ambient springbone wind: a gentle pseudo-random breeze applied as a
    // per-frame bias on every spring joint's gravity vector, so hair (and
    // any springbone-driven clothing) keeps drifting subtly even when the
    // body is on the single still idle loop. State + the per-frame apply
    // function are imported from ../vrm/springboneWind (unit-tested there);
    // `springboneWindState` is this mount's instance, created once here.
    // `localStorage.kurisu_wind` = '0' disables, any other number scales
    // the strength — read once at creation (readWindStrengthPref).
    const springboneWindState = createSpringboneWindState()

    // Animation loop: RAF/clock/visibility wiring lives in
    // ../vrm/renderLoop; the ordered per-frame sequence stays here so the
    // frame order (undo additives -> mixer -> procedural layers -> wind ->
    // vrm.update -> face/gaze -> camera -> lighting -> render) is
    // reviewable in one place.
    // Pet-variant-only idle disqualifiers (item 3, idle render throttling):
    // the pointer sitting over the always-on-top overlay, or the cursor-
    // gaze target moving more than a small threshold, both wake the full
    // frame rate instantly. Neither applies to the main window — hovering
    // the chat pane has never disqualified idle there, and cursor gaze-
    // follow doesn't exist outside the pet variant, so both checks are
    // no-ops (constant false/0) for variant === 'full'.
    let pointerOverPetWindow = false
    if (variant === 'pet') {
      const handlePetPointerEnter = () => {
        pointerOverPetWindow = true
        // Wake trigger (spec: "pointer entering the window ... wakes her").
        // wakeDoze() is itself a no-op when already awake/waking, so this
        // is safe to call unconditionally on every pointer-enter rather
        // than gating on isDozing first.
        wakeDoze(dozeStateRef.current)
      }
      const handlePetPointerLeave = () => {
        pointerOverPetWindow = false
      }
      canvasEl.addEventListener('pointerenter', handlePetPointerEnter)
      canvasEl.addEventListener('pointerleave', handlePetPointerLeave)
      petPointerHoverCleanup = () => {
        canvasEl.removeEventListener('pointerenter', handlePetPointerEnter)
        canvasEl.removeEventListener('pointerleave', handlePetPointerLeave)
      }
    }
    // Last frame's cursor-gaze target, so a poll update that moves it more
    // than CURSOR_GAZE_WAKE_THRESHOLD wakes the full frame rate immediately
    // instead of waiting for the next throttled idle frame to notice —
    // otherwise the eased gaze offset in gazeDirector would visibly
    // stair-step at 15fps while the throttle held the accumulator back.
    // Tracker + threshold live in ../vrm/idleThrottle; the variant !== 'pet'
    // short-circuit stays here since the tracker itself is pet-agnostic.
    const cursorGazeWakeTracker = createCursorGazeWakeTracker(cursorGazeStateRef.current)
    const cursorGazeMovedBeyondThreshold = (): boolean =>
      variant === 'pet' && cursorGazeWakeTracker()

    // "Idle" for the render loop's throttle, defined CONSERVATIVELY: any
    // voice audio (isAudible covers streams, drain/fade tails, and
    // one-shot presence clips), any gesture clip beyond the base idle loop
    // (currentModeRef === 'actual'), any in-flight camera/lighting tween,
    // any user activity in the last few seconds, the pointer sitting over
    // the pet overlay, or the cursor-gaze target having just moved,
    // disqualifies it. NOTE: cursorGazeMovedBeyondThreshold has a side
    // effect (it updates the "last" trackers) — it must run unconditionally
    // every call; createIsIdleNow (../vrm/idleThrottle) preserves that
    // exact ordering internally, so something earlier already disqualifying
    // idle still just lags the trackers by a frame, which is harmless (a
    // small transient once movement resumes, never a stuck-throttled
    // frame).
    const isIdleNow = createIsIdleNow({
      isLipSyncAudible,
      isMidGesture: () => currentModeRef.current === 'actual',
      isReframing: () => cameraDirector.isReframing(),
      isLightingLerpActive: () => rig.isLightingLerpActive(),
      isUserDrivingCamera: () => followState.userDriving,
      isPointerOverPetWindow: () => pointerOverPetWindow,
      cursorGazeMovedBeyondThreshold,
      getLastUserInputAt: () => followState.lastUserInputAt,
      getLastPointerMoveAt: () => gazeStateRef.current.lastPointerMoveAt,
      getLastTypingAt: () => gazeSocial.lastTypingAt,
      getLastInteractionAt: () => lastInteractionAtRef.current,
      idleActivityWindowMs: IDLE_ACTIVITY_WINDOW_MS,
    })

    // Doze deepens the idle-frame throttle further (~15fps -> ~10fps): a
    // fully dozing avatar moving even at the normal pet idle rate has more
    // headroom to give up than an awake-but-idle one. See
    // ../vrm/idleThrottle for why this stays a function instead of a fixed
    // number.
    const idleFrameIntervalS = createIdleFrameIntervalS(dozeStateRef.current, variant)

    const renderLoop = createRenderLoop({
      isSpeaking: () => lipSyncStateRef.current.streaming,
      isMidGesture: () => currentModeRef.current === 'actual',
      isIdle: isIdleNow,
      idleFrameIntervalS,
      onResumeAfterHidden: adaptiveQuality.resetFrameClock,
      onFrame: (deltaTime, frameNow, idleNow) => {
        // Doze state machine (pet variant only, see ../vrm/petDoze): reuses
        // the SAME conservative "idle" signal (idleNow) the render-loop
        // throttle above just computed, rather than re-deriving a second
        // notion of "nothing is happening". Advanced every frame
        // (including non-idle-throttled ones) so deltaTime stays accurate
        // regardless of which cadence this particular frame ran on.
        if (variant === 'pet') {
          advanceDoze(dozeStateRef.current, deltaTime, { hour: new Date().getHours(), isIdle: idleNow })
          if (isDozing(dozeStateRef.current)) {
            // Suppress cursor gaze-follow while fully dozing (spec: "While
            // dozing: suppress cursor gaze-follow AND ambient idle
            // gestures"). The poller only ever WRITES `active = true` on a
            // successful engaged poll (see cursorGaze.ts) and never reads
            // it itself, so forcing it false here every dozing frame is a
            // safe, non-invasive override — it simply resumes reflecting
            // real polls again the instant she's no longer dozing, with no
            // special-case "resume" logic needed on either side.
            cursorGazeStateRef.current.active = false
          }
        }

        if (idleNow) {
          // The throttled idle cadence (~33ms main window, ~67ms pet — see
          // idleFrameIntervalS above) must not read as GPU load to the
          // adaptive-resolution sampler — reset it instead of feeding it,
          // so resolution never degrades from idling (and re-entry to the
          // active path just re-baselines on its first frame). Idle also
          // RESTORES full resolution: a tier degraded during a speech
          // burst otherwise freezes here (idle frames never feed
          // update()), parking the pet at 0.5x — grainy — indefinitely.
          adaptiveQuality.notifyIdle()
          adaptiveQuality.resetFrameClock()
        } else {
          adaptiveQuality.update(frameNow)
        }

        // Strip last frame's procedural additives (speech nods, breathing)
        // BEFORE the mixer runs: bones whose tracks the mixer drives get
        // overwritten anyway, and bones it never writes (head/neck — stripped
        // from every clip) are restored to their true base, so the additives
        // below can't compound frame-over-frame.
        proceduralBody.undoAdditives()
        // Inertialization offsets applied AFTER the mixer are stripped
        // AFTER the additives (LIFO — additives were applied on top of the
        // offsets last frame), restoring the raw mixer pose exactly.
        inertializer.undo()

        // Default relaxed hands: most library clips carry no finger tracks
        // (leaving flat splayed bind-pose "paddle hands"); clips that do
        // animate fingers overwrite this inside the same mixer update.
        // frameNow gives the per-finger idle noise its clock (lifelike
        // pass: resting fingers drift a few percent, phase-offset per
        // finger, instead of freezing).
        if (vrmRef.current) {
          applyRelaxedFingerPose(vrmRef.current, frameNow / 1000)
        }

        // Update animation mixer if it exists
        if (animationMixerRef.current) {
          animationMixerRef.current.update(deltaTime)
        }

        if (vrmRef.current) {
          // Inertialization rides directly on the mixer's fresh pose,
          // before every procedural layer stacks on top (consumes a pending
          // transition, decays + applies offsets, records pose history).
          inertializer.update(deltaTime)
          // Lip-sync viseme analysis rides this loop now (it used to self-
          // schedule a second, concurrent 60fps rAF): cheap no-op unless a
          // playback graph is live. Must run before proceduralBody.update
          // (whose speech micro-motion reads the speechEnvelope it writes)
          // and before vrm.update() (which bakes the viseme expression
          // values into the mesh). Riding the main loop also inherits the
          // document.hidden pause for free; audible speech blocks the idle
          // throttle above, so this always ticks at full frame rate while
          // it matters.
          updateLipSyncVisemes()
          // Speech micro-motion, head aim, then breathing (see
          // proceduralBody.update for the ordering rationale). All run after
          // mixer.update() so they layer on top of the freshly-written pose,
          // and before vrm.update() so the normalized-bone changes reach the
          // raw skeleton (and spring bones) this same frame.
          proceduralBody.update(deltaTime)
          // Wind biases springbone gravity before vrm.update() runs the
          // spring simulation, so the breeze reaches the hair this frame.
          applySpringboneWind(springboneWindState, vrmRef.current, deltaTime)

          // Blink, expression, doze eyelid, gaze, and the blush beat all
          // advance BEFORE vrm.update() so their weights bake into the
          // mesh THIS frame — they used to run after it and land one frame
          // stale relative to the visemes and the springbone sim.
          expressionDriver.advanceBlink(deltaTime)
          expressionDriver.advanceExpression(deltaTime)
          // Doze eye-close (pet variant only): sustains the `blink`
          // expression at the current drowsy/dozing amount, layered AFTER
          // advanceBlink so a genuine blink can still flutter on top of a
          // partial droop instead of being stomped by it. 0 while awake is
          // a no-op (see setEyeClose's own doc).
          if (variant === 'pet') {
            expressionDriver.setEyeClose(dozeStateRef.current.eyeCloseAmount)
          }
          gazeDirector.advanceGaze(deltaTime)
          // Flustered blush beat: hold the eased level until `until`, then
          // decay to 0. Skipped entirely (level stays 0) on models with no
          // blush channel; writing only while nonzero keeps the channel
          // untouched otherwise.
          if (blushChannel && (blushBeat.level > 0.001 || blushBeat.target > 0)) {
            const holding = performance.now() < blushBeat.until
            const target = holding ? blushBeat.target : 0
            blushBeat.level += (target - blushBeat.level) * Math.min(1, deltaTime * 6)
            if (!holding && blushBeat.level < 0.001) {
              blushBeat.level = 0
              blushBeat.target = 0
            }
            blushChannel.set(blushBeat.level)
          }

          vrmRef.current.update(deltaTime)

          // Morph-backed face channels must re-assert after vrm.update():
          // the expression manager zeroes every expression-BOUND morph at
          // the top of its pass, wiping direct writes made above. No-op
          // for expression-backed channels and while the values are 0.
          browChannel?.postUpdate?.()
          blushChannel?.postUpdate?.()

          // Gentle camera follow + 20s-idle auto-return (see
          // cameraDirector.updateFollow).
          cameraDirector.updateFollow(performance.now())
        }

        // Lighting refresh lerp (item 7): eases toward the target set by
        // refreshTimeOfDayLighting() over ~2s instead of snapping.
        rig.advanceLightingLerp(deltaTime)

        // Double-click reframe tween / aspect-aware reframe ease (see
        // cameraDirector.updateReframe).
        cameraDirector.updateReframe(deltaTime)

        controls.update()
        // Static shadows: shadowMap.autoUpdate is off (see createSceneRig)
        // — re-render the map on every active frame, or when explicitly
        // requested (first frame / VRM load), so it only freezes while the
        // pose is effectively static.
        if (!idleNow || shadowMapNeedsRefresh) {
          renderer.shadowMap.needsUpdate = true
          shadowMapNeedsRefresh = false
        }
        postfx.render(deltaTime)
      },
    })
    renderLoop.start()
    debugLog('✅ Animation loop started')

    // Handle resize
    const handleResize = () => {
      if (!canvasRef.current) return
      camera.aspect = canvasRef.current.clientWidth / canvasRef.current.clientHeight
      camera.updateProjectionMatrix()
      adaptiveQuality.applyPixelRatio()
      renderer.setSize(canvasRef.current.clientWidth, canvasRef.current.clientHeight, false)
      postfx.setSize(canvasRef.current.clientWidth, canvasRef.current.clientHeight, renderer.getPixelRatio())
      // Aspect-aware reframe (item 2): recompute the distance needed to keep full-body headroom at the
      // new aspect and ease into it (no-op if the user is currently driving the camera, or before load).
      cameraDirector.beginReframeToAspect()
    }
    window.addEventListener('resize', handleResize)
    // The pane can change size without a window resize (initial layout
    // settling behind the desktop shell's splash, media-query flips) —
    // observe the canvas itself so the render size always tracks it.
    const resizeObserver = new ResizeObserver(handleResize)
    resizeObserver.observe(canvasEl)
    document.addEventListener('visibilitychange', renderLoop.handleVisibilityChange)

    // Live voice-volume updates from the settings panel's mixer — the
    // lip-sync gain node persists for the session (built once, reused),
    // so a slider change needs to reach it directly rather than waiting
    // for the graph to be rebuilt.
    const handleVolumeChange = () => {
      const state = lipSyncStateRef.current
      if (state.gainNode) {
        state.gainNode.gain.value = loadVolumePrefs().voice
      }
    }
    window.addEventListener('kurisu:volume-change', handleVolumeChange)

    // App-wide volume slider (item 4, separate control from the settings-
    // panel voice mixer above): the sibling-owned slider writes
    // localStorage `kurisu_volume` and fires this event after every
    // change; applyAppVolume re-reads it and pushes it to the post-
    // analyser gain stage the SAME way handleVolumeChange does for its own
    // gain node (see lipSyncController's chain-order doc comment for why
    // both gains live downstream of the analyser).
    window.addEventListener(APP_VOLUME_CHANGE_EVENT, applyAppVolume)

    // Pet preset live reframe (small item, pet variant only) — see
    // ../vrm/modelLoadFlow's reframeForPetPresetChange for the framing
    // rationale (same path the initial load and the double-click-to-reframe
    // escape hatch already use).
    const handlePetPresetChange = () => {
      if (variant !== 'pet') return
      reframeForPetPresetChange({
        vrm: vrmRef.current,
        canvas: canvasRef.current,
        cameraFov: camera.fov,
        cameraDirector,
      })
    }
    if (variant === 'pet') {
      window.addEventListener(PET_PRESET_CHANGE_EVENT, handlePetPresetChange)
    }

    // Perch pose (pet variant only): the sibling-owned pet UI dispatches
    // this after an edge-snap drag. Decision logic (sit-clip search, wind
    // override) lives in ../vrm/petPerch; this wires it to the component's
    // refs/bridge exactly as the inline version did.
    // Deferred-assignment hook (same pattern as talkAnimationHooks above):
    // the scheduler instance this needs (to explicitly resume the idle
    // rotation after a looping sit clip, rather than leaving her frozen on
    // it forever — playByName's loop:true sets 'idle' mode, which never
    // naturally reverts via the mixer 'finished' event the way a one-shot
    // gesture does) only exists once the VRM-load callback runs, later in
    // this same effect.
    const perchHooks: { resumeIdle: (() => void) | null } = { resumeIdle: null }
    const petPerchHandler = createPetPerchHandler({
      describeClips: () => kurisuBridge.vrmAnimations?.describe(),
      playClip: (name, options) => kurisuBridge.vrmAnimations?.play(name, options),
      resumeIdle: () => perchHooks.resumeIdle?.(),
      springboneWindState,
      readWindStrengthPref,
      setPerched: (perched) => {
        perchedRef.current = perched
      },
    })
    const handlePetPerchChanged = (event: Event) => {
      if (variant !== 'pet') return
      const detail = (event as CustomEvent<{ perched?: boolean; edge?: string | null }>).detail
      petPerchHandler.handlePerchChanged(detail)
    }
    if (variant === 'pet') {
      window.addEventListener('kurisu-pet-perch-changed', handlePetPerchChanged)
    }

    // Attention/presence reflex event handlers (typing return, mic-speaking
    // lock-on, listening nod, cursor-linger tilt, thinking-aversion beat,
    // speech-beat gestures) — see ../vrm/attentionEvents for the full
    // per-event rationale. attach()/detach() follow the same convention as
    // pointerInteractions above.
    const attentionEvents = createAttentionEvents({
      gazeSocial,
      gazeStateRef,
      blinkStateRef,
      gazeDirector,
      proceduralBody,
    })
    attentionEvents.attach()

    // Reflex layer (see ../vrm/reflexes): declarative table of instant
    // LLM-free reactions to discrete events (barge-in, window focus,
    // ambient battery/music, pet untuck), with centralized gating and
    // per-reflex cooldowns. New one-shot reflexes are table rows there,
    // not new bespoke listeners here.
    const reflexes = createReflexEngine({
      requestBlink: (probability) => expressionDriver.requestBlink(probability),
      // Same lock-on bump attentionEvents' mic-speaking handler makes:
      // suppresses NEW glance-aways for ~1.6s so she holds eye contact.
      snapGaze: () => {
        gazeSocial.lockOnUntil = Math.max(gazeSocial.lockOnUntil, performance.now() + 1600)
      },
      queueCuriousTilt: () => proceduralBody.queueCuriousTilt(),
      expressionFlick: (name, weight, durationMs) => setKurisuExpression(name, weight, durationMs),
      isSpeaking: () => isLipSyncAudible(),
      isDozing: () => (variant === 'pet' ? isDozing(dozeStateRef.current) : false),
      // Pure localStorage read — safe on every variant (always false
      // outside a tucked pet).
      isTucked: () => isPetTucked(),
      prefersReducedMotion: () => {
        try {
          return window.matchMedia('(prefers-reduced-motion: reduce)').matches
        } catch {
          return false
        }
      },
      now: () => performance.now(),
    })
    reflexes.attach()

    debugLog('\n=== VRM VIEWER INITIALIZATION COMPLETE ===\n')

    // Cleanup
    return () => {
      disposed = true
      renderLoop.stop()
      cursorGazePoller?.stop()
      arcWarmth?.stop()
      petIdleBehaviorsForCleanup?.stop()
      petPointerHoverCleanup?.()
      window.clearInterval(lightingRefreshInterval)
      window.removeEventListener('resize', handleResize)
      resizeObserver.disconnect()
      document.removeEventListener('visibilitychange', renderLoop.handleVisibilityChange)
      window.removeEventListener('kurisu:volume-change', handleVolumeChange)
      window.removeEventListener(APP_VOLUME_CHANGE_EVENT, applyAppVolume)
      if (variant === 'pet') {
        window.removeEventListener(PET_PRESET_CHANGE_EVENT, handlePetPresetChange)
        window.removeEventListener('kurisu-pet-perch-changed', handlePetPerchChanged)
      }
      attentionEvents.detach()
      reflexes.detach()
      pointerInteractions.detach()
      postfx.dispose()
      rig.dispose()
      if (gazeProxyForCleanup) {
        scene.remove(gazeProxyForCleanup)
        gazeProxyForCleanup = null
      }
      if (animationMixerRef.current) {
        if (mixerFinishedListenerForCleanup) {
          animationMixerRef.current.removeEventListener('finished', mixerFinishedListenerForCleanup)
        }
        animationMixerRef.current.stopAllAction()
      }
      if (scheduledActualTimeoutRef.current !== null) {
        window.clearTimeout(scheduledActualTimeoutRef.current)
        scheduledActualTimeoutRef.current = null
      }
      stopLipSync()
      kurisuBridge.lipSync = null
      kurisuBridge.expressions = null
      kurisuBridge.react = null
      kurisuBridge.reactReset = null
      kurisuBridge.vrmAnimations = null
      kurisuBridge.petDoze = null
      if (window.kurisuLipSync) {
        delete window.kurisuLipSync
      }
      if (window.vrmAnimations) {
        delete window.vrmAnimations
      }
      // Clear the shared refs the environment effect reads/writes so it
      // never touches a disposed scene/renderer after this effect tears
      // down (e.g. mid-flight environment GLB load resolving after unmount).
      sceneRef.current = null
      cameraRef.current = null
      rendererRef.current = null
      mainLightRef.current = null
      shadowCatcherRef.current = null
      // The reframe hook closes over this effect's camera/controls/director
      // — a swap firing after teardown must not touch disposed objects.
      reframeForEnvironmentRef.current = null
      // A torn-down scene has no environment GLB loaded — the environment
      // effect's disposal logic on its NEXT run would otherwise try to
      // remove/dispose a group that belongs to the now-gone scene. (The
      // environment effect also depends on modelReloadToken — see its dep
      // array — specifically so it re-runs and reloads the GLB against the
      // fresh scene whenever an outfit swap remounts this effect.)
      currentEnvironmentGroupRef.current = null
    }
    // `variant` is fixed for the window's lifetime (URL-derived), included
    // only to satisfy exhaustive-deps. Intentionally does NOT depend on
    // environmentReloadToken/environmentIdRef — see the environment effect
    // below, which owns environment swaps without rebuilding this scene.
  }, [modelReloadToken, variant])

  // Environment effect: owns swapping the environment GLB + repositioning
  // the shadow-center-derived objects (mainLight/mainLight.target/shadow
  // catcher/shadow frustum) + repositioning the VRM's anchor, all on the
  // SAME live scene/renderer/VRM the scene effect above created — no
  // teardown of the WebGL scene, no VRM re-fetch, no AudioContext/lip-sync
  // reset. The swap/dispose/load mechanics live in
  // ../vrm/environmentManager (unit-tested there); this effect just wires
  // them to the component's refs. Declared AFTER the scene effect so on
  // first mount it runs once sceneRef/mainLightRef/etc. are already
  // published (React runs effects in declaration order); re-runs only on
  // environmentReloadToken, i.e. only when SettingsPanel's environment
  // listener bumps it.
  useEffect(() => {
    const scene = sceneRef.current
    if (!scene || variant === 'pet') {
      // Pet overlay never loads an environment GLB (see the scene effect's
      // "Pet variant: skipping environment GLB" branch) — nothing to swap.
      // No scene yet (still mounting, or already torn down) — the scene
      // effect's own load callback already places the VRM against
      // whichever environment is current at that time, so there's nothing
      // for this effect to do until a real swap happens after mount.
      return
    }

    const activeEnvironment = ENVIRONMENTS[environmentIdRef.current] ?? ENVIRONMENTS.time_machine
    activeEnvironmentRef.current = activeEnvironment

    // Refit the shadow rig + VRM anchor in place on the live objects.
    repositionForEnvironment(
      {
        mainLight: mainLightRef.current,
        shadowCatcher: shadowCatcherRef.current,
        vrm: vrmRef.current,
      },
      activeEnvironment
    )
    if (vrmRef.current) applyRimLighting(vrmRef.current, { main: mainLightRef.current?.color }) // ../vrm/rimLight

    // She just teleported to the new set's anchor (position AND facing can
    // both change between sets) — re-derive the camera shot from her new
    // bone positions. Without this the camera keeps framing the OLD anchor:
    // a lab→time-machine swap left her half out of frame at the screen edge
    // with the shot centered on the machine hull, and the reverse left her
    // entirely off-screen (reproduced headlessly 2026-07-28).
    reframeForEnvironmentRef.current?.()

    // Swap the environment GLB itself: dispose the old one's geometries/
    // materials (it's replaced, not reused) before loading the new one.
    const previousEnvironment = currentEnvironmentGroupRef.current
    if (previousEnvironment) {
      disposeEnvironmentGroup(scene, previousEnvironment)
      currentEnvironmentGroupRef.current = null
    }

    const load = loadEnvironmentGroup({
      scene,
      environment: activeEnvironment,
      debugLog,
      onLoaded: (group) => {
        currentEnvironmentGroupRef.current = group
      },
    })

    return () => {
      load.cancel()
    }
    // modelReloadToken: an outfit swap remounts the scene effect (fresh
    // THREE.Scene, no environment GLB loaded yet), so this effect must
    // re-run to load the environment against the NEW scene even though
    // environmentReloadToken didn't change. variant is fixed for the
    // window's lifetime (URL-derived), included only to satisfy
    // exhaustive-deps. environmentIdRef is a ref (read live, not a dep) —
    // the listener effect above is what mutates it and bumps
    // environmentReloadToken.
  }, [environmentReloadToken, modelReloadToken, variant])

  return (
    <div className="vrm-stage">
      <canvas ref={canvasRef} className="vrm-canvas" />
      {loadStatus.phase === 'loading' && (
        <div className="vrm-overlay" role="status">
          <div className="vrm-overlay-spinner" aria-hidden="true" />
          <p>
            Loading Amadeus…
            {loadStatus.progress > 0 ? ` ${Math.round(loadStatus.progress)}%` : ''}
          </p>
        </div>
      )}
      {loadStatus.phase === 'error' && (
        <div className="vrm-overlay" role="alert">
          <p>The avatar failed to load.</p>
          <p className="vrm-overlay-detail">{loadStatus.message}</p>
          <button type="button" onClick={() => setModelReloadToken((token) => token + 1)}>
            Retry
          </button>
        </div>
      )}
    </div>
  )
}

export default VRMViewer
