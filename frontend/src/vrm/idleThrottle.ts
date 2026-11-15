/**
 * Render-loop idle-throttle decision logic, moved verbatim out of
 * VRMViewer's scene effect. "Idle" here means the render loop is allowed to
 * drop to its throttled cadence (see idleFrameIntervalS below and
 * renderLoop.ts) — voice audio, mid-gesture animation, in-flight camera/
 * lighting tweens, recent user activity, the pointer sitting over the pet
 * overlay, or the cursor-gaze target having just moved, all disqualify it.
 *
 * createCursorGazeWakeTracker's returned function has a deliberate side
 * effect (it updates the "last" trackers each call) — callers must invoke
 * it unconditionally every frame's decision, not skip it via short-
 * circuiting, or the trackers lag and a real movement can go unnoticed for
 * a frame. See isIdleNow's own call site in VRMViewer for the ordering this
 * was tuned against.
 */

import type { CursorGazeState } from './cursorGaze'
import type { DozeMachineState } from './petDoze'
import { isDozing } from './petDoze'

/** Normalized [-1, 1] units; a poll update moving the cursor-gaze target by
 * more than this wakes the full frame rate immediately instead of waiting
 * for the next throttled idle frame to notice it — otherwise the eased gaze
 * offset in gazeDirector would visibly stair-step at the throttled rate
 * while the accumulator held it back. */
export const CURSOR_GAZE_WAKE_THRESHOLD = 0.04

/** Tracks the previous frame's cursor-gaze target and reports whether it
 * moved beyond CURSOR_GAZE_WAKE_THRESHOLD since the last call. Pet-variant
 * only for callers — non-pet variants should just pass a state whose
 * `active` never goes true, or gate the call at the call site as VRMViewer
 * does (variant !== 'pet' short-circuits before ever touching the state). */
export const createCursorGazeWakeTracker = (
  cursorGazeState: CursorGazeState
): (() => boolean) => {
  let lastLateral = cursorGazeState.targetLateral
  let lastVertical = cursorGazeState.targetVertical
  return () => {
    if (!cursorGazeState.active) return false
    const dLateral = Math.abs(cursorGazeState.targetLateral - lastLateral)
    const dVertical = Math.abs(cursorGazeState.targetVertical - lastVertical)
    lastLateral = cursorGazeState.targetLateral
    lastVertical = cursorGazeState.targetVertical
    return dLateral > CURSOR_GAZE_WAKE_THRESHOLD || dVertical > CURSOR_GAZE_WAKE_THRESHOLD
  }
}

export interface IsIdleNowDeps {
  isLipSyncAudible: () => boolean
  isMidGesture: () => boolean
  isReframing: () => boolean
  isLightingLerpActive: () => boolean
  isUserDrivingCamera: () => boolean
  isPointerOverPetWindow: () => boolean
  /** Side-effecting — see the module doc comment. Must be called
   * unconditionally every invocation. */
  cursorGazeMovedBeyondThreshold: () => boolean
  getLastUserInputAt: () => number
  getLastPointerMoveAt: () => number
  getLastTypingAt: () => number
  getLastInteractionAt: () => number
  idleActivityWindowMs: number
}

/** The render loop's conservative idle predicate — any of the disqualifiers
 * above being true (or too-recent) keeps the full frame rate active. */
export const createIsIdleNow = (deps: IsIdleNowDeps) => {
  return (now: number): boolean => {
    // gazeMoved has a side effect (updates the tracker's "last" values) and
    // must run every call — see the module doc comment.
    const gazeMoved = deps.cursorGazeMovedBeyondThreshold()
    return (
      !deps.isLipSyncAudible() &&
      !deps.isMidGesture() &&
      !deps.isReframing() &&
      !deps.isLightingLerpActive() &&
      !deps.isUserDrivingCamera() &&
      !deps.isPointerOverPetWindow() &&
      !gazeMoved &&
      now - deps.getLastUserInputAt() >= deps.idleActivityWindowMs &&
      now - deps.getLastPointerMoveAt() >= deps.idleActivityWindowMs &&
      now - deps.getLastTypingAt() >= deps.idleActivityWindowMs &&
      now - deps.getLastInteractionAt() >= deps.idleActivityWindowMs
    )
  }
}

/** Doze deepens the idle-frame throttle further (~15fps -> ~10fps): a fully
 * dozing avatar moving even at the normal pet idle rate has more headroom
 * to give up than an awake-but-idle one. Returns a function (not a fixed
 * number) so callers re-read it every frame instead of baking in whatever
 * was true at construction time; non-pet variants are unaffected
 * (dozeState is only ever advanced for variant === 'pet', so isDozing(...)
 * is always false there). */
export const createIdleFrameIntervalS = (
  dozeState: DozeMachineState,
  variant: 'full' | 'pet'
): (() => number) => {
  return () => (isDozing(dozeState) ? 1 / 10 : variant === 'pet' ? 1 / 15 : 1 / 30)
}
