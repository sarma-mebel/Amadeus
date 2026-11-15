/**
 * Pet-window doze state machine: awake -> drowsy -> dozing -> waking -> awake.
 *
 * Goal: idle-at-2am should visibly read differently from idle-at-2pm — right
 * now the pet just idles identically around the clock (petIdleBehaviors.ts
 * already slows/calms the ambient gesture beat at night, but the AVATAR
 * ITSELF never looks sleepy). This module is a pure, frame-driven state
 * machine (no THREE/DOM/timers of its own) so it's trivially unit-testable
 * and so VRMViewer can drive it from the exact same render-loop tick that
 * already reads deltaTime — no separate setTimeout ladder to keep in sync
 * with the idle-activity trackers it depends on.
 *
 * Transition rule (see DozeInputs below): she only drifts toward sleep
 * during late-night hours AND after a sustained stretch with no activity
 * signal at all — reusing the SAME conservative "idle" definition VRMViewer
 * already computes for the render-loop throttle (isIdleNow), rather than
 * inventing a second, possibly-divergent notion of "nothing is happening".
 * The caller is expected to pass `isIdle` computed however that check
 * already works; this module never reads performance.now()/refs itself so
 * it stays pure and doesn't care how "idle" was decided upstream.
 *
 * Waking is instant-to-`waking` (not instant-to-`awake`): a real animal
 * jolted awake still blinks/stirs for a beat rather than snapping straight
 * back to full alertness — see WAKING_DURATION_S.
 */

export type DozeState = 'awake' | 'drowsy' | 'dozing' | 'waking'

export interface DozeMachineState {
  phase: DozeState
  /** Seconds spent in the current phase — drives drowsy->dozing timing and
   * the waking transition's fixed duration. Reset to 0 on every phase
   * change. */
  phaseElapsedS: number
  /** Continuous 0 (fully alert) -> 1 (eyes fully closed) drowsiness level,
   * eased rather than stepped so the eyelid droop in the expression driver
   * has something smooth to read from instead of a hard cut at the drowsy
   * boundary. Mirrors `phase` but interpolates through the drowsy stage. */
   eyeCloseAmount: number
  /** eyeCloseAmount captured at the moment the `waking` transition began —
   * the wake stir interpolates DOWN from this value, not from 1: waking out
   * of a partial drowsy droop must not snap the eyes fully closed for a
   * frame first. */
  wakeStartEyeClose: number
}

export const createDozeMachineState = (): DozeMachineState => ({
  phase: 'awake',
  phaseElapsedS: 0,
  eyeCloseAmount: 0,
  wakeStartEyeClose: 0,
})

/** Local hour (0-23) is considered "late night" on the same boundary
 * petIdleBehaviors.ts already uses for its own night-mode cadence — keeping
 * a single definition of "night" across the pet stack rather than two
 * slightly different clocks. */
const NIGHT_START_HOUR = 23
const NIGHT_END_HOUR = 6
export const isNightHour = (hour: number): boolean => hour >= NIGHT_START_HOUR || hour < NIGHT_END_HOUR

/** No activity for this long (AND it's night) before she starts drifting
 * toward drowsy. "~10 min" per spec. */
export const DROWSY_AFTER_IDLE_S = 10 * 60
/** Drowsy holds for roughly this much longer before crossing into fully
 * dozing — "~2 min later" per spec. */
export const DOZING_AFTER_DROWSY_S = 2 * 60
/** How long the post-wake "stir" transition holds before settling back to
 * fully awake — long enough for a blink-and-shift-of-weight to read, short
 * enough it never feels like a stuck state. */
export const WAKING_DURATION_S = 1.6

/** Eye-close eases toward 1 over the drowsy stage (not snapped at the
 * dozing boundary) so "a slow partial-lid drowsy stage before full close"
 * (per spec) is a real interpolation, not a step function. Reaches ~1 partway
 * through dozing and holds there. */
const EYE_CLOSE_EASE_PER_S = 1 / DOZING_AFTER_DROWSY_S

export interface DozeInputs {
  /** Local wall-clock hour, 0-23 (Date.getHours()). Caller-supplied so this
   * stays pure/testable without faking Date globally. */
  hour: number
  /** The SAME conservative "nothing is happening" signal the render loop's
   * idle-framerate throttle already computes (audio/gesture/camera-tween/
   * pointer-over-window/recent-interaction all disqualify it) — reused
   * rather than re-derived so doze and the frame throttle never disagree
   * about whether she's idle right now. */
  isIdle: boolean
}

/**
 * Advances the doze state machine by one frame. Pure function of
 * (state, deltaTime, inputs) — mutates `state` in place (same convention as
 * proceduralMotion's driftNoise) and returns nothing; callers read the
 * fields back off the same object they passed in.
 *
 * Any non-idle frame (isIdle: false) immediately falls back toward awake —
 * EXCEPT while `waking`, which is a fixed-duration transition that always
 * plays out even if activity resumes mid-stir (a jolt-awake shouldn't be
 * interruptible mid-blink; it's ~1.6s, not worth the extra branch
 * complexity of allowing a "wake during waking" case).
 */
export const advanceDoze = (state: DozeMachineState, deltaTimeS: number, inputs: DozeInputs): void => {
  state.phaseElapsedS += deltaTimeS
  const night = isNightHour(inputs.hour)

  switch (state.phase) {
    case 'awake': {
      if (night && inputs.isIdle && state.phaseElapsedS >= DROWSY_AFTER_IDLE_S) {
        state.phase = 'drowsy'
        state.phaseElapsedS = 0
      } else if (!inputs.isIdle) {
        // Any activity resets the idle clock so a burst of chat mid-evening
        // doesn't leave stale progress toward drowsy sitting around.
        state.phaseElapsedS = 0
      }
      state.eyeCloseAmount = Math.max(0, state.eyeCloseAmount - deltaTimeS * EYE_CLOSE_EASE_PER_S * 2)
      break
    }
    case 'drowsy': {
      if (!inputs.isIdle || !night) {
        // Activity resumed (or somehow it's no longer night, e.g. midnight
        // rollover doesn't apply here but a manual clock override in tests
        // might): stir back toward awake via the same waking transition a
        // real wake() would use, so the eye-close/head-droop never just
        // vanish instantly.
        state.phase = 'waking'
        state.phaseElapsedS = 0
        state.wakeStartEyeClose = state.eyeCloseAmount
        break
      }
      state.eyeCloseAmount = Math.min(1, state.eyeCloseAmount + deltaTimeS * EYE_CLOSE_EASE_PER_S)
      if (state.phaseElapsedS >= DOZING_AFTER_DROWSY_S) {
        state.phase = 'dozing'
        state.phaseElapsedS = 0
      }
      break
    }
    case 'dozing': {
      if (!inputs.isIdle || !night) {
        state.phase = 'waking'
        state.phaseElapsedS = 0
        state.wakeStartEyeClose = state.eyeCloseAmount
        break
      }
      state.eyeCloseAmount = 1
      break
    }
    case 'waking': {
      // Fixed-duration stir, always plays out in full (see doc comment).
      // Interpolates from wherever the lids were at wake entry down to 0 —
      // starting from a hardcoded 1 snapped the eyes fully SHUT for a frame
      // when waking out of a partial drowsy droop.
      state.eyeCloseAmount = Math.max(
        0,
        state.wakeStartEyeClose * (1 - state.phaseElapsedS / WAKING_DURATION_S),
      )
      if (state.phaseElapsedS >= WAKING_DURATION_S) {
        state.phase = 'awake'
        state.phaseElapsedS = 0
        state.eyeCloseAmount = 0
      }
      break
    }
  }
}

/** Explicit wake — pointer entering the window, a poke, or the bridge's
 * wake() call. Returns true iff she was actually `dozing` (not merely
 * `drowsy` or already `awake`/`waking`) at the moment this was called, per
 * the bridge contract: the pet UI uses the return value to decide whether a
 * "just woke up" groggy line is warranted at all — waking her from a light
 * drowse isn't the same beat as rousing her from a full doze. */
export const wakeDoze = (state: DozeMachineState): boolean => {
  const wasDozing = state.phase === 'dozing'
  if (state.phase === 'dozing' || state.phase === 'drowsy') {
    state.phase = 'waking'
    state.phaseElapsedS = 0
    state.wakeStartEyeClose = state.eyeCloseAmount
  }
  // 'awake' and 'waking' are both no-ops: already awake, or already mid-stir.
  return wasDozing
}

/** True for 'dozing' only — used to gate cursor-gaze suppression and the
 * ambient idle beat, and by the bridge's `active()` slot. 'drowsy' still
 * looks and behaves mostly normal (just heavier-lidded), so it does NOT
 * suppress those systems — only a fully dozing avatar does. */
export const isDozing = (state: DozeMachineState): boolean => state.phase === 'dozing'
