/**
 * Gaze direction, moved verbatim out of VRMViewer's scene effect: the
 * social gaze layer (typing attentiveness, emotion-driven gaze aversion,
 * eye-contact lock-on) and the per-frame advanceGaze step that steers the
 * lookAt proxy (random glance-aways, pointer tracking, micro-saccades).
 *
 * `social` is exposed as a plain mutable state object — the animation
 * scheduler's flustered beat, the head-aim aversion check, and the
 * typing/user-speaking window-event handlers in the component all write
 * its fields directly, exactly as they did when it was an effect-local.
 */

import * as THREE from 'three'
import { CURSOR_GAZE_STALE_MS, type CursorGazeState } from './cursorGaze'

/** Gaze micro-behavior: a small proxy Object3D near the camera that
 * lookAt.target is pointed at, so the render loop can nudge it off-camera
 * periodically instead of re-assigning lookAt.target (and therefore
 * snapping) every frame.
 *
 * B4 adds a third 'pointer' phase: while the mouse/touch pointer has moved
 * recently AND sits near her face on screen, the proxy eases toward the
 * pointer's world position instead of cycling onCamera/offset. Held on a
 * component ref (it survives scene-effect re-runs; the proxy itself is
 * recreated per VRM load). */
export interface GazeRuntimeState {
  proxy: THREE.Object3D | null
  phase: 'onCamera' | 'offset' | 'pointer'
  nextChangeAt: number
  offset: THREE.Vector3
  pointerWorld: THREE.Vector3
  lastPointerMoveAt: number
  pointerNear: boolean
}

export const createGazeRuntimeState = (): GazeRuntimeState => ({
  proxy: null,
  phase: 'onCamera',
  nextChangeAt: 2 + Math.random() * 4,
  offset: new THREE.Vector3(),
  pointerWorld: new THREE.Vector3(),
  lastPointerMoveAt: 0,
  pointerNear: false,
})

/** Social gaze layer (game-feel): typing attentiveness, emotion-driven
 * gaze aversion, eye-contact lock-on, and micro-saccades. All of it
 * feeds advanceGaze() in the render loop; timestamps are
 * performance.now() ms. Aversion stores camera-relative lateral/vertical
 * amounts (not a world vector) because the camera doesn't exist yet at
 * emotion-hook time and may move before the frame renders. */
export interface GazeSocialState {
  typingUntil: number
  lastTypingAt: number
  lockOnUntil: number
  aversionUntil: number
  aversionLateral: number
  aversionVertical: number
  saccadeOffset: THREE.Vector3
  saccadeTarget: THREE.Vector3
  nextSaccadeIn: number
  /** Thinking aversion (LLM-wait beat): performance.now() when the wait
   * began, 0 when not thinking. She holds eye contact through the
   * acknowledgment nod first (thinkingHoldMs), then looks up-and-away
   * until endThinking() re-engages — users read the aversion as
   * "considering" and wait more patiently (Andrist IVA'13). */
  thinkingSince: number
  thinkingHoldMs: number
  thinkingLateral: number
  thinkingVertical: number
  thinkingRetargetIn: number
}

export interface GazeDirector {
  social: GazeSocialState
  /** Kurisu-flavored gaze semantics per streamed emotion tag: sadness looks
   * down and away, anger throws a short sideways huff, strong happy/
   * surprised locks eye contact. Called from setKurisuExpression so it
   * rides the exact same signal ChatInterface already sends per sentence. */
  applyEmotionGaze: (emotion: string, intensity: number) => void
  /** Thinking-aversion beat (kurisu:thinking-start): the LLM wait began.
   * Eye contact holds through the acknowledgment nod, then the gaze averts
   * up-and-to-the-side until endThinking(). Self-capping (THINKING_MAX_MS)
   * so a lost end event can never wedge her staring at the ceiling. */
  beginThinking: () => void
  /** First token arrived (kurisu:thinking-end): drop the aversion and
   * re-engage with a short eye-contact lock-on. Safe to call when not
   * thinking (no-op beyond the lock-on suppression check). */
  endThinking: () => void
  advanceGaze: (deltaTime: number) => void
}

export const createGazeDirector = ({
  camera,
  gazeStateRef,
  isSpeaking,
  cursorGazeState,
  onGazeShift,
}: {
  camera: THREE.Camera
  gazeStateRef: { current: GazeRuntimeState }
  /** Whether her own speech stream is live (saccades get a touch larger). */
  isSpeaking: () => boolean
  /** Pet-window cursor tracking (cursorGaze.ts) — LOWEST priority gaze
   * input in the whole stack (see advanceGaze). Undefined outside the pet
   * variant/Tauri, where the feature doesn't exist at all. */
  cursorGazeState?: CursorGazeState
  /** Event-coupled blinks: fires when the gaze makes a deliberate shift
   * (glance-away start, emotional aversion, thinking aversion engage/
   * re-engage) with a 0-1 magnitude — humans blink on ~97% of large gaze
   * shifts (VMagicMirror heuristics), so VRMViewer wires this to the blink
   * machine's requestBlink. Optional: tests and older callers omit it. */
  onGazeShift?: (magnitude: number) => void
}): GazeDirector => {
  const gazeSocial: GazeSocialState = {
    typingUntil: 0,
    lastTypingAt: 0,
    lockOnUntil: 0,
    aversionUntil: 0,
    aversionLateral: 0,
    aversionVertical: 0,
    saccadeOffset: new THREE.Vector3(),
    saccadeTarget: new THREE.Vector3(),
    nextSaccadeIn: 0.3, // seconds
    thinkingSince: 0,
    thinkingHoldMs: 700,
    thinkingLateral: 0,
    thinkingVertical: 0,
    thinkingRetargetIn: 0,
  }

  const applyEmotionGaze = (emotion: string, intensity: number) => {
    const now = performance.now()
    const side = Math.random() < 0.5 ? -1 : 1
    if (emotion === 'sad' && intensity >= 0.4) {
      gazeSocial.aversionLateral = side * 0.16
      gazeSocial.aversionVertical = -0.22
      gazeSocial.aversionUntil = now + 2400
      gazeSocial.lockOnUntil = 0
      onGazeShift?.(0.8)
    } else if (emotion === 'angry' && intensity >= 0.5) {
      gazeSocial.aversionLateral = side * 0.26
      gazeSocial.aversionVertical = 0.04
      gazeSocial.aversionUntil = now + 1200
      gazeSocial.lockOnUntil = 0
      onGazeShift?.(0.9)
    } else if ((emotion === 'happy' || emotion === 'surprised') && intensity >= 0.6) {
      gazeSocial.aversionUntil = 0
      gazeSocial.lockOnUntil = now + 2500
    }
  }

  // Thinking-aversion beat (LLM-wait): a "recalling" look reads as up and
  // to the side, distinctly different from the sad/angry aversions above.
  // Self-capping so a lost thinking-end (dropped frame handler, wedged
  // stream) can never hold the averted gaze past the point where the
  // acknowledgment controller's own 6s timeout has long fired.
  const THINKING_MAX_MS = 12000
  let thinkingAversionEngaged = false

  const beginThinking = () => {
    const now = performance.now()
    gazeSocial.thinkingSince = now
    // Hold eye contact through the acknowledgment nod first — she heard
    // you — then avert. Randomized so consecutive turns don't metronome.
    gazeSocial.thinkingHoldMs = 550 + Math.random() * 350
    const side = Math.random() < 0.5 ? -1 : 1
    gazeSocial.thinkingLateral = side * (0.18 + Math.random() * 0.08)
    gazeSocial.thinkingVertical = 0.14 + Math.random() * 0.08
    gazeSocial.thinkingRetargetIn = 1.4 + Math.random() * 0.8
    thinkingAversionEngaged = false
  }

  const endThinking = () => {
    if (gazeSocial.thinkingSince === 0) return
    const wasAverted = thinkingAversionEngaged
    gazeSocial.thinkingSince = 0
    thinkingAversionEngaged = false
    // Re-engage: first token arrived, she comes back to you to answer.
    // The return shift is itself a large gaze movement — blink on it.
    gazeSocial.lockOnUntil = Math.max(
      gazeSocial.lockOnUntil,
      performance.now() + 1200 + Math.random() * 600,
    )
    if (wasAverted) onGazeShift?.(0.7)
  }

  // Reused scratch vectors for the per-frame gaze math so advanceGaze()
  // doesn't allocate every frame.
  const gazeScratchVec = new THREE.Vector3()
  const gazeScratchVec2 = new THREE.Vector3()
  const gazeCameraRight = new THREE.Vector3()
  const gazeCameraUp = new THREE.Vector3()

  // Cursor gaze-follow (pet + Tauri only, cursorGaze.ts): the LOWEST
  // priority gaze input. `cursorOffset` is a slowly-eased camera-relative
  // offset — separate from the fast-lerp saccade and the proxy's own
  // 0.15/frame ease, so the desktop-wide mouse tracking reads as a smooth
  // idle drift rather than snapping. Envelope is deliberately WIDER than
  // the random 'offset' glance (+/-0.3m): at the old glance-matched 0.22m
  // the eye deflection was imperceptible on a pet-sized window ("cursor
  // gaze doesn't really work", 2026-07-28) — following the mouse across
  // a whole desktop is an intentional look, not an idle glance, and the
  // head-aim layer takes its fraction of the wider turn too.
  const gazeCursorOffset = new THREE.Vector3()
  const CURSOR_GAZE_LATERAL_M = 0.45
  const CURSOR_GAZE_VERTICAL_M = 0.30

  // Gaze micro-behavior (item 6 + B4): nudges the lookAt proxy off
  // dead-center every 2-6s for 0.4-1.2s, otherwise keeps it pinned to the
  // camera. B4 adds a third 'pointer' phase: while the pointer has moved
  // within the last POINTER_GAZE_IDLE_MS and sits near her face on
  // screen, the proxy eases toward the pointer's world position instead
  // (still clamped to the existing +/-0.3m offset envelope around the
  // camera so it can't drag the gaze arbitrarily far). An in-progress
  // 'offset' glance is allowed to finish its hold before pointer tracking
  // can take over, so a random glance-away never gets visibly cut short.
  const POINTER_GAZE_IDLE_MS = 2000
  const advanceGaze = (deltaTime: number) => {
    const gaze = gazeStateRef.current
    const proxy = gaze.proxy
    if (!proxy) return

    const pointerActive =
      gaze.pointerNear && performance.now() - gaze.lastPointerMoveAt <= POINTER_GAZE_IDLE_MS

    if (pointerActive && gaze.phase !== 'offset') {
      if (gaze.phase !== 'pointer') {
        gaze.phase = 'pointer'
      }
    } else if (!pointerActive && gaze.phase === 'pointer') {
      // Pointer went idle/away — resume the normal onCamera/offset cycle
      // from onCamera rather than mid-cycle.
      gaze.phase = 'onCamera'
      gaze.nextChangeAt = 2 + Math.random() * 4
    }

    // Social overrides: while the user is actively typing, or an emotion
    // demands eye contact, she doesn't wander off into a random glance —
    // holding attention IS the signal. (An in-progress glance still
    // finishes naturally; only new glance-aways are suppressed.)
    const nowMs = performance.now()

    // Thinking-aversion lifecycle: expired thinking self-clears (defensive
    // cap — the acknowledgment controller should always end it first);
    // past the eye-contact hold it engages, firing the gaze-shift hook
    // once, and slowly re-targets so the "considering" look drifts rather
    // than freezing on one spot of the ceiling.
    if (gazeSocial.thinkingSince > 0 && nowMs - gazeSocial.thinkingSince > THINKING_MAX_MS) {
      gazeSocial.thinkingSince = 0
      thinkingAversionEngaged = false
    }
    const thinkingAverted =
      gazeSocial.thinkingSince > 0 && nowMs - gazeSocial.thinkingSince > gazeSocial.thinkingHoldMs
    if (thinkingAverted) {
      if (!thinkingAversionEngaged) {
        thinkingAversionEngaged = true
        onGazeShift?.(0.6)
      }
      gazeSocial.thinkingRetargetIn -= deltaTime
      if (gazeSocial.thinkingRetargetIn <= 0) {
        gazeSocial.thinkingRetargetIn = 1.4 + Math.random() * 0.8
        const side = Math.sign(gazeSocial.thinkingLateral) || 1
        // Drift around the same side — recrossing the face every couple of
        // seconds would read as shifty, not thoughtful.
        gazeSocial.thinkingLateral = side * (0.14 + Math.random() * 0.12)
        gazeSocial.thinkingVertical = 0.1 + Math.random() * 0.12
      }
    }

    // A pre-aversion thinking hold counts as attention too: she keeps eye
    // contact through the acknowledgment nod instead of wandering off the
    // instant the user hits send.
    const holdAttention =
      nowMs < gazeSocial.typingUntil ||
      nowMs < gazeSocial.lockOnUntil ||
      (gazeSocial.thinkingSince > 0 && !thinkingAverted)

    if (gaze.phase !== 'pointer') {
      gaze.nextChangeAt -= deltaTime
      if (gaze.nextChangeAt <= 0) {
        if (gaze.phase === 'onCamera' && !holdAttention && !thinkingAverted) {
          gaze.phase = 'offset'
          gaze.nextChangeAt = 0.4 + Math.random() * 0.8 // hold the glance 0.4-1.2s
          gazeCameraRight.set(1, 0, 0).applyQuaternion(camera.quaternion)
          gazeCameraUp.set(0, 1, 0).applyQuaternion(camera.quaternion)
          const lateral = (Math.random() * 2 - 1) * 0.3
          const vertical = (Math.random() * 2 - 1) * 0.3
          gaze.offset
            .copy(gazeCameraRight)
            .multiplyScalar(lateral)
            .addScaledVector(gazeCameraUp, vertical)
          // Glance-away is a deliberate gaze shift — candidate for an
          // event-coupled blink, scaled by how far the glance goes.
          onGazeShift?.(Math.min(1, (Math.abs(lateral) + Math.abs(vertical)) / 0.45))
        } else if (gaze.phase === 'onCamera') {
          gaze.nextChangeAt = 1 + Math.random() * 2 // attentive: re-check soon
        } else {
          gaze.phase = 'onCamera'
          gaze.nextChangeAt = 2 + Math.random() * 4 // back on-camera for 2-6s
          onGazeShift?.(0.35) // returning to eye contact is a shift too
        }
      }
    }

    gazeScratchVec.copy(camera.position)
    if (nowMs < gazeSocial.aversionUntil) {
      // Emotional gaze aversion (sad looks down-and-away, angry huffs
      // sideways, flustered breaks eye contact) beats every other phase —
      // it's the strongest social signal this system produces.
      gazeCameraRight.set(1, 0, 0).applyQuaternion(camera.quaternion)
      gazeCameraUp.set(0, 1, 0).applyQuaternion(camera.quaternion)
      gazeScratchVec
        .addScaledVector(gazeCameraRight, gazeSocial.aversionLateral)
        .addScaledVector(gazeCameraUp, gazeSocial.aversionVertical)
    } else if (thinkingAverted) {
      // Thinking aversion (LLM-wait): up-and-to-the-side "considering"
      // look. Below emotional aversion (a sad look-down mid-wait still
      // wins) but above every attention/idle phase.
      gazeCameraRight.set(1, 0, 0).applyQuaternion(camera.quaternion)
      gazeCameraUp.set(0, 1, 0).applyQuaternion(camera.quaternion)
      gazeScratchVec
        .addScaledVector(gazeCameraRight, gazeSocial.thinkingLateral)
        .addScaledVector(gazeCameraUp, gazeSocial.thinkingVertical)
    } else if (gaze.phase === 'offset') {
      gazeScratchVec.add(gaze.offset)
    } else if (gaze.phase === 'pointer') {
      // Clamp to the same +/-0.3m envelope the random 'offset' glance
      // uses, measured from the camera position, so pointer tracking
      // can't pull the gaze further than a normal glance would.
      const towardPointer = gazeScratchVec2.copy(gaze.pointerWorld).sub(camera.position)
      const maxOffset = 0.3 * Math.SQRT2 // envelope corner: lateral and vertical each up to 0.3m
      if (towardPointer.length() > maxOffset) {
        towardPointer.setLength(maxOffset)
      }
      gazeScratchVec.add(towardPointer)
    } else if (gaze.phase === 'onCamera') {
      // Cursor gaze-follow: only reachable when NOTHING else is claiming
      // attention — no emotional aversion (handled above, never reaches
      // this branch), no B4 pointer-near-face tracking or in-progress
      // random glance ('pointer'/'offset' own their branches above), and
      // gated further below by holdAttention (eye-contact lock-on/typing).
      // This is deliberately the LAST branch in the chain — every other
      // gaze behavior in the file wins outright over it.
      const cursorEligible =
        cursorGazeState?.active &&
        !holdAttention &&
        performance.now() - cursorGazeState.lastUpdateAt <= CURSOR_GAZE_STALE_MS
      if (cursorEligible && cursorGazeState) {
        gazeCameraRight.set(1, 0, 0).applyQuaternion(camera.quaternion)
        gazeCameraUp.set(0, 1, 0).applyQuaternion(camera.quaternion)
        gazeScratchVec2
          .copy(gazeCameraRight)
          .multiplyScalar(cursorGazeState.targetLateral * CURSOR_GAZE_LATERAL_M)
          .addScaledVector(gazeCameraUp, cursorGazeState.targetVertical * CURSOR_GAZE_VERTICAL_M)
      } else {
        gazeScratchVec2.set(0, 0, 0)
      }
      // Always ease (toward the live target when eligible, back toward
      // zero otherwise) rather than snapping the branch condition on/off —
      // the poll itself only updates 5x/second, and this smooths over both
      // that step function and holdAttention/eligibility toggling.
      gazeCursorOffset.lerp(gazeScratchVec2, Math.min(1, deltaTime * 2.5))
      gazeScratchVec.add(gazeCursorOffset)
    }

    // Micro-saccades: real eyes never hold perfectly still — they make
    // tiny ballistic jumps every few hundred ms. Amplitude is a few cm at
    // ~1m (sub-degree), a touch larger while she's speaking or making
    // deliberate eye contact. The fast lerp (not the proxy's slow 0.15)
    // is what makes them read as saccades rather than drift.
    gazeSocial.nextSaccadeIn -= deltaTime
    if (gazeSocial.nextSaccadeIn <= 0) {
      // Conversation-state saccade profiles (Eyes Alive): thinking scans
      // slower and wider (recalling), attentive listening holds steadier
      // (smaller jumps), speaking stays a touch larger than rest.
      gazeSocial.nextSaccadeIn = thinkingAverted
        ? 0.35 + Math.random() * 0.55
        : 0.15 + Math.random() * 0.35
      const speaking = isSpeaking()
      let amp = 0.012 + (speaking ? 0.02 : 0) + (gaze.phase === 'onCamera' ? 0.008 : 0)
      if (thinkingAverted) amp += 0.014
      else if (holdAttention) amp *= 0.6
      gazeCameraRight.set(1, 0, 0).applyQuaternion(camera.quaternion)
      gazeCameraUp.set(0, 1, 0).applyQuaternion(camera.quaternion)
      gazeSocial.saccadeTarget
        .copy(gazeCameraRight)
        .multiplyScalar((Math.random() * 2 - 1) * amp)
        .addScaledVector(gazeCameraUp, (Math.random() * 2 - 1) * amp * 0.6)
    }
    gazeSocial.saccadeOffset.lerp(gazeSocial.saccadeTarget, Math.min(1, deltaTime * 18))
    gazeScratchVec.add(gazeSocial.saccadeOffset)

    proxy.position.lerp(gazeScratchVec, 0.15)
  }

  return { social: gazeSocial, applyEmotionGaze, beginThinking, endThinking, advanceGaze }
}
