/**
 * Instant acknowledgment beat: a visual mask for the silence-to-first-audio
 * gap. An audio filler (a synthesized "thinking sound") was tried for the
 * same gap and REJECTED as unnatural — see audio/fillerPlayer.ts's doc
 * comment ("read as unnatural in daily use", now opt-in-off). This is the
 * visual alternative: the instant the user finishes speaking or hits send,
 * she should look like she heard you and is considering, not stand idle
 * until the first sentence streams in.
 *
 * Deliberately reuses the exact primitives active-listening already
 * proved out (audio/listeningReactions.ts): the same
 * 'kurisu:user-speaking' event extends VRMViewer's gaze lock-on (she holds
 * eye contact instead of glancing away), and the same
 * 'kurisu:listening-nod' event queues one slow acknowledging nod through
 * proceduralBody's capped nod machinery. No new expression is invented —
 * the six-expression VRM vocabulary (happy/angry/sad/relaxed/surprised/
 * neutral, see vrm/expressionDriver.ts) has nothing that reads as
 * "considering" without looking like an emotion react, so this stays
 * gaze+nod only, exactly the same beat a good listener already gets.
 *
 * This module is a pure, framework-free, turn-scoped state machine:
 * arm(turnId) on submit, release(turnId) the moment the first token/
 * sentence/audio frame proves she's actually responding (or on error/
 * cancel), with a hard timeout so a lost release can never wedge the gaze
 * lock open. Idempotent and turn-scoped — a release for a stale turnId
 * (a superseded/cancelled turn's late callback) is ignored, and re-arming
 * while already armed just re-covers the same turn rather than double-
 * firing the beat.
 */

export const ACKNOWLEDGMENT_TIMEOUT_MS = 6000

export interface AcknowledgmentDeps {
  /** Fires the gaze-hold + nod beat — defaults to the same two window
   * events active-listening already dispatches, so VRMViewer needs no new
   * listener. Injectable for tests. */
  trigger?: () => void
  /** Thinking-aversion beat (lifelike pass 2026-07-30): the armed window IS
   * the LLM wait, so arm/release double as its start/end signals. Defaults
   * dispatch 'kurisu:thinking-start' / 'kurisu:thinking-end' window events
   * that VRMViewer routes to the gaze director — after the acknowledging
   * nod she visibly looks away to "consider", and re-engages the moment
   * the first token proves she's answering. End always fires if start did
   * (release, supersede by a newer arm, or the hard timeout), so the gaze
   * director's own self-cap is a second line of defense, not the primary. */
  onThinkingStart?: () => void
  onThinkingEnd?: () => void
  /** Reduced-motion escape hatch — checked once at arm time. Defaults to
   * the standard media query (same convention as LoadingScreen.tsx); a
   * user who asked for less motion shouldn't get an extra nod on every
   * single turn. */
  prefersReducedMotion?: () => boolean
  setTimeout?: (fn: () => void, ms: number) => number
  clearTimeout?: (id: number) => void
}

export interface AcknowledgmentController {
  /** Arms the beat for a newly-submitted turn: fires the gaze+nod trigger
   * (unless reduced-motion), and starts the hard-timeout safety release.
   * Re-arming while a previous turn is still armed simply moves the armed
   * turnId forward — the previous turn's eventual release becomes a no-op
   * (already superseded), and no duplicate beat fires for a re-arm of the
   * SAME turnId. */
  arm: (turnId: string) => void
  /** Releases the hold if turnId matches the currently-armed turn. A stale
   * turnId (an old turn's callback arriving after a newer arm) is silently
   * ignored — never cancels a different, still-in-flight turn's hold. */
  release: (turnId: string) => void
  /** True while a turn is currently armed (test/inspection seam). */
  isArmed: () => boolean
}

const defaultPrefersReducedMotion = (): boolean => {
  try {
    return window.matchMedia('(prefers-reduced-motion: reduce)').matches
  } catch {
    return false
  }
}

const defaultTrigger = (): void => {
  window.dispatchEvent(new Event('kurisu:user-speaking'))
  window.dispatchEvent(new Event('kurisu:listening-nod'))
}

// window-guarded (unlike defaultTrigger): this module is deliberately
// framework-free and its tests run in a plain node environment where only
// the deps they inject should ever fire — the thinking defaults must not
// explode when a caller injects trigger but not the thinking hooks.
const defaultThinkingStart = (): void => {
  if (typeof window === 'undefined') return
  window.dispatchEvent(new Event('kurisu:thinking-start'))
}

const defaultThinkingEnd = (): void => {
  if (typeof window === 'undefined') return
  window.dispatchEvent(new Event('kurisu:thinking-end'))
}

export const createAcknowledgmentController = (
  deps: AcknowledgmentDeps = {},
): AcknowledgmentController => {
  const trigger = deps.trigger ?? defaultTrigger
  const onThinkingStart = deps.onThinkingStart ?? defaultThinkingStart
  const onThinkingEnd = deps.onThinkingEnd ?? defaultThinkingEnd
  const prefersReducedMotion = deps.prefersReducedMotion ?? defaultPrefersReducedMotion
  const setTimeoutFn = deps.setTimeout ?? ((fn, ms) => window.setTimeout(fn, ms))
  const clearTimeoutFn = deps.clearTimeout ?? ((id) => window.clearTimeout(id))

  let armedTurnId: string | null = null
  let timeoutId: number | null = null
  // Paired-signal guard: thinking-end only ever fires if thinking-start
  // did for the currently-armed turn (reduced-motion arms skip both).
  let thinkingStarted = false

  const clearTimer = () => {
    if (timeoutId !== null) {
      clearTimeoutFn(timeoutId)
      timeoutId = null
    }
  }

  const endThinking = () => {
    if (!thinkingStarted) return
    thinkingStarted = false
    onThinkingEnd()
  }

  const arm = (turnId: string) => {
    if (armedTurnId === turnId) return // already covering this exact turn
    clearTimer()
    // A newer arm supersedes the previous turn's wait — close its thinking
    // window before opening the next (beginThinking would also implicitly
    // re-seed, but the paired events keep every consumer's state honest).
    endThinking()
    armedTurnId = turnId
    if (!prefersReducedMotion()) {
      trigger()
      thinkingStarted = true
      onThinkingStart()
    }
    timeoutId = setTimeoutFn(() => {
      timeoutId = null
      // Safety valve: something dropped every release callback for this
      // turn (onToken/onSentence/onAudioStart never fired, and neither did
      // onDone/onFailure) — never leave the gaze lock-on stuck past this.
      armedTurnId = null
      endThinking()
    }, ACKNOWLEDGMENT_TIMEOUT_MS)
  }

  const release = (turnId: string) => {
    if (armedTurnId !== turnId) return // stale release for a superseded turn
    armedTurnId = null
    clearTimer()
    endThinking()
  }

  const isArmed = () => armedTurnId !== null

  return { arm, release, isArmed }
}

/** Module singleton, mirroring the pattern of audio/listeningReactions.ts's
 * default-wired reactor — both hook call sites (useChatTurn, usePetChat)
 * share one instance since only one avatar responds to one user at a time
 * per surface, and each surface's own turnId keeps the two independent in
 * practice (the pet and main window are separate JS contexts). */
export const acknowledgment = createAcknowledgmentController()
