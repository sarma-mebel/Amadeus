/**
 * Render-loop wiring, moved verbatim out of VRMViewer's scene effect: the
 * requestAnimationFrame loop with its clamped clock, and the
 * visibilitychange pause/resume (with the B5 focus-regain reaction). The
 * per-frame body itself is the `onFrame` callback — the component keeps
 * the ordered sequence of subsystem updates there, so the frame order
 * stays reviewable in one place.
 */

import * as THREE from 'three'
import { kurisuBridge } from './kurisuBridge'

export interface RenderLoop {
  start: () => void
  stop: () => void
  /** Pause rendering while the window/tab is hidden — no reason to burn
   * GPU and battery on frames nobody sees. Audio keeps playing. Wire to
   * document's 'visibilitychange'. */
  handleVisibilityChange: () => void
}

export const createRenderLoop = ({
  onFrame,
  isSpeaking,
  isMidGesture,
  isIdle,
  onResumeAfterHidden,
  idleFrameIntervalS = 1 / 30,
}: {
  /** One frame's work; receives the clamped animation delta (seconds), the
   * frame's wall-clock performance.now() timestamp, and whether this frame
   * ran on the throttled idle cadence (see `isIdle`). */
  onFrame: (deltaTime: number, frameNow: number, idleNow: boolean) => void
  isSpeaking: () => boolean
  /** True while the body state machine is mid-gesture ('actual' mode). */
  isMidGesture: () => boolean
  /** Idle framerate throttle: a motionless idle avatar does not need the
   * full 60fps pipeline (mixer, springbones, procedural motion, PCFSoft
   * render) — this matters most for the always-on-top desktop pet, which
   * sits on screen all day. The component defines "idle" CONSERVATIVELY
   * (no audible voice, no gesture clip, no camera/lighting tween, no
   * recent user activity). When idle, the WHOLE frame body runs at the
   * throttled cadence (see idleFrameIntervalS) via a time accumulator —
   * skipped frames early-return after scheduling the next rAF, and the
   * frame that does run receives the true accumulated dt (mixer/vrm/easing
   * all handle variable dt), so motion advances at real speed instead of
   * half speed. Breathing/blinking at this rate is imperceptible. */
  isIdle: (frameNow: number) => boolean
  /** Called on resume so hidden time is not counted as a slow frame
   * (VRMViewer wires this to adaptiveQuality.resetFrameClock). */
  onResumeAfterHidden: () => void
  /** Idle cadence in seconds-per-frame. Default 1/30 (main window, ~30fps)
   * — the pet variant passes 1/15 (~15fps): it's an always-on-top overlay
   * that sits on screen all day with no chat pane around it, so the extra
   * headroom is worth a slightly choppier idle animation, and item 3's
   * instant-wake triggers cover every case where that would be visible
   * (speaking, gesture, pointer over the window, gaze target movement).
   *
   * Accepts either a fixed number (existing callers, unchanged) or a
   * () => number read fresh every frame — doze support (../vrm/petDoze)
   * deepens this further to ~1/10 (~10fps) while fully dozing, since a
   * sleeping avatar moving at even 15fps has visibly more headroom to give
   * up than an awake-but-idle one. */
  idleFrameIntervalS?: number | (() => number)
}): RenderLoop => {
  const clock = new THREE.Clock()
  // Max deltaTime clamp (item 7): a stalled tab / debugger pause / slow
  // frame otherwise hands the next frame a huge delta, which the mixer,
  // breathing sine, and easing state machines would all read as a
  // big jump/lurch. 0.1s = still-smooth-looking floor of 10fps.
  const MAX_DELTA = 0.1
  let idleFrameAccumulatorS = 0

  let animationFrameId = 0
  let stopped = false
  // B5 (focus-regain reaction): wall-clock of when the tab went hidden,
  // and of the last time this reaction fired (for the 5-minute cooldown).
  // 0 means "not currently hidden" / "never fired" respectively.
  let hiddenAt = 0
  let lastFocusReactionAt = 0
  const FOCUS_REGAIN_HIDDEN_THRESHOLD_MS = 2 * 60 * 1000
  const FOCUS_REGAIN_COOLDOWN_MS = 5 * 60 * 1000

  const animate = () => {
    if (stopped) return
    animationFrameId = requestAnimationFrame(animate)
    const frameNow = performance.now()
    // Accumulate real elapsed time every rAF (getDelta must be called
    // each callback or the next read would include skipped-frame time
    // twice); the idle throttle below decides whether this frame runs.
    idleFrameAccumulatorS += Math.min(clock.getDelta(), MAX_DELTA)
    const idleNow = isIdle(frameNow)
    const currentIdleIntervalS =
      typeof idleFrameIntervalS === 'function' ? idleFrameIntervalS() : idleFrameIntervalS
    if (idleNow && idleFrameAccumulatorS < currentIdleIntervalS) return
    // True accumulated time since the last frame that actually ran —
    // includes skipped idle frames, clamped like the single-frame delta
    // so a stall still can't lurch the mixer/easing.
    const deltaTime = Math.min(idleFrameAccumulatorS, MAX_DELTA)
    idleFrameAccumulatorS = 0
    onFrame(deltaTime, frameNow, idleNow)
  }

  const handleVisibilityChange = () => {
    if (document.hidden) {
      hiddenAt = performance.now()
      cancelAnimationFrame(animationFrameId)
      animationFrameId = 0
    } else {
      if (hiddenAt > 0) {
        const hiddenForMs = performance.now() - hiddenAt
        const cooledDown = performance.now() - lastFocusReactionAt >= FOCUS_REGAIN_COOLDOWN_MS
        const speaking = isSpeaking()
        const midGesture = isMidGesture()
        if (hiddenForMs > FOCUS_REGAIN_HIDDEN_THRESHOLD_MS && cooledDown && !speaking && !midGesture) {
          lastFocusReactionAt = performance.now()
          kurisuBridge.expressions?.set('happy', 0.5, 2500)
          kurisuBridge.vrmAnimations?.play('Greeting')
        }
      }
      hiddenAt = 0
      if (animationFrameId === 0 && !stopped) {
        clock.getDelta() // discard hidden time so animations don't jump
        onResumeAfterHidden() // hidden time is not a slow frame
        animate()
      }
    }
  }

  return {
    start: () => {
      animate()
    },
    stop: () => {
      stopped = true
      cancelAnimationFrame(animationFrameId)
    },
    handleVisibilityChange,
  }
}
