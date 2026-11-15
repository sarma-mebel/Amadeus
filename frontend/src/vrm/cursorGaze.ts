/**
 * Pet-window cursor gaze-follow (desktop pet, Tauri only): polls a Tauri
 * command for the OS cursor's global position + this window's outer rect,
 * converts it into a small camera-relative offset, and feeds it into
 * gazeDirector's 'cursor' phase — the LOWEST-priority gaze behavior in the
 * whole stack. Every other gaze signal (emotional aversion, eye-contact
 * lock-on, pointer-near-face tracking, an in-progress random glance-away)
 * must win outright; this only fills in the otherwise-idle "stare dead at
 * the camera" gap so she idly tracks the mouse anywhere on the desktop
 * instead of only reacting to it near her own face (that's already B4's
 * job, in pointerInteractions.ts, and is unaffected by this module).
 *
 * The command (`pet_cursor_context`, owned by a sibling agent's Tauri
 * command surface) is expected to return global cursor coordinates and this
 * window's outer rect, all in PHYSICAL pixels:
 *   { cursor_x, cursor_y, win_x, win_y, win_w, win_h, scale }
 * `scale` is the OS display scale factor (physical px per logical px) —
 * present so a future caller could convert to logical px, but this module
 * only needs ratios within the window rect, which are scale-invariant, so
 * it's accepted and otherwise unused here.
 *
 * Degrades to nothing whenever any precondition fails: not Tauri, not the
 * pet variant, the command isn't registered yet (a sibling agent may land
 * it after this module ships), or the window/screen goes hidden. A run of
 * consecutive invoke failures disarms polling entirely so a missing command
 * doesn't spam retries or the console forever.
 */
import * as THREE from 'three'
import { diagnostics } from '../diagnostics'
import { isTauri } from '../platform'

/** Poll cadence: ~5x/second per spec — cheap enough for a background IPC
 * call, fast enough that mouse tracking doesn't look laggy once eased. */
const POLL_INTERVAL_MS = 200
/** Consecutive invoke failures (command missing, IPC error, non-Tauri
 * context that slipped through) before polling gives up for the session —
 * never log-spam a feature that simply isn't wired up yet. */
const MAX_CONSECUTIVE_FAILURES = 3

export interface PetCursorContext {
  cursor_x: number
  cursor_y: number
  win_x: number
  win_y: number
  win_w: number
  win_h: number
  scale: number
}

const isPetCursorContext = (value: unknown): value is PetCursorContext => {
  if (!value || typeof value !== 'object') return false
  const v = value as Record<string, unknown>
  return (
    typeof v.cursor_x === 'number' &&
    typeof v.cursor_y === 'number' &&
    typeof v.win_x === 'number' &&
    typeof v.win_y === 'number' &&
    typeof v.win_w === 'number' &&
    typeof v.win_h === 'number'
  )
}

/**
 * Maps a cursor position relative to the pet window into a normalized
 * [-1, 1] lateral/vertical pair — same convention as NDC x/y (right/up
 * positive) — clamped to the window bounds. Pure so it's unit-testable
 * without a Tauri context.
 */
export const cursorToNormalizedOffset = (
  ctx: Pick<PetCursorContext, 'cursor_x' | 'cursor_y' | 'win_x' | 'win_y' | 'win_w' | 'win_h'>,
): { lateral: number; vertical: number } => {
  if (ctx.win_w <= 0 || ctx.win_h <= 0) return { lateral: 0, vertical: 0 }
  const relX = (ctx.cursor_x - ctx.win_x) / ctx.win_w // 0 at left, 1 at right
  const relY = (ctx.cursor_y - ctx.win_y) / ctx.win_h // 0 at top, 1 at bottom
  const lateral = THREE.MathUtils.clamp(relX * 2 - 1, -1, 1)
  const vertical = THREE.MathUtils.clamp(-(relY * 2 - 1), -1, 1) // flip: up is +1
  return { lateral, vertical }
}

/** Runtime state the gaze director reads every frame — module-free plain
 * object so gazeDirector doesn't need to import this file's polling
 * machinery, only the shape of what it produces. */
export interface CursorGazeState {
  /** True once at least one successful poll has landed this session. */
  active: boolean
  /** Normalized [-1, 1] target the gaze eases toward. */
  targetLateral: number
  targetVertical: number
  /** Wall-clock of the last successful poll — advanceGaze can treat a stale
   * reading (window hidden, tab backgrounded mid-poll) as inactive. */
  lastUpdateAt: number
}

export const createCursorGazeState = (): CursorGazeState => ({
  active: false,
  targetLateral: 0,
  targetVertical: 0,
  lastUpdateAt: 0,
})

/** Reading older than this is treated as stale (same idea as the pointer
 * phase's POINTER_GAZE_IDLE_MS) — covers the window going hidden mid-flight
 * or a stalled poll loop without a separate teardown signal. */
export const CURSOR_GAZE_STALE_MS = 2000

// --- Cursor-linger play ------------------------------------------------
//
// Desktop Mate's charming bit: park the mouse on/near her and she does
// something about it. The gaze half already exists (she tracks the
// cursor); this adds the PLAY half — a still cursor lingering inside her
// window for a beat fires `kurisu:cursor-linger`, and VRMViewer answers
// with a curious head-tilt. Client-side reflex only (no LLM, no server),
// rare by construction: pokes/clicks stay the interactive path.

/** Still cursor inside the window this long before she reacts. */
export const CURSOR_LINGER_MS = 2500
/** Movement beyond this (physical px) restarts the linger clock. */
export const CURSOR_LINGER_MOVE_PX = 32
/** Floor between reactions — a parked cursor is furniture after the first
 * acknowledgment, not a repeat prompt. */
export const CURSOR_LINGER_COOLDOWN_MS = 150_000
export const CURSOR_LINGER_EVENT = 'kurisu:cursor-linger'

export interface CursorLingerState {
  /** Wall-clock the current still-inside-the-window run began, or null
   * when no run is in progress (0 is a legitimate timestamp in tests). */
  lingerStartedAt: number | null
  lastX: number
  lastY: number
  lastFiredAt: number
}

export const createCursorLingerState = (): CursorLingerState => ({
  lingerStartedAt: null,
  lastX: Number.NaN,
  lastY: Number.NaN,
  lastFiredAt: -Infinity,
})

/** Advances the linger tracker with one poll reading; returns true when a
 * reaction should fire NOW. Pure (caller owns the event dispatch) so the
 * inside/moved/cooldown rules are unit-testable. */
export const updateCursorLinger = (
  state: CursorLingerState,
  ctx: Pick<
    PetCursorContext,
    'cursor_x' | 'cursor_y' | 'win_x' | 'win_y' | 'win_w' | 'win_h'
  >,
  nowMs: number,
): boolean => {
  const inside =
    ctx.cursor_x >= ctx.win_x &&
    ctx.cursor_x <= ctx.win_x + ctx.win_w &&
    ctx.cursor_y >= ctx.win_y &&
    ctx.cursor_y <= ctx.win_y + ctx.win_h
  const moved =
    !Number.isFinite(state.lastX) ||
    Math.hypot(ctx.cursor_x - state.lastX, ctx.cursor_y - state.lastY) >
      CURSOR_LINGER_MOVE_PX
  state.lastX = ctx.cursor_x
  state.lastY = ctx.cursor_y
  if (!inside) {
    state.lingerStartedAt = null
    return false
  }
  if (moved || state.lingerStartedAt === null) {
    state.lingerStartedAt = nowMs
    return false
  }
  if (
    nowMs - state.lingerStartedAt >= CURSOR_LINGER_MS &&
    nowMs - state.lastFiredAt >= CURSOR_LINGER_COOLDOWN_MS
  ) {
    state.lastFiredAt = nowMs
    // Restart the clock: an even-longer linger isn't a second event.
    state.lingerStartedAt = nowMs
    return true
  }
  return false
}

export interface CursorGazePoller {
  stop: () => void
}

/** Arc-driven warmth (../vrm/arcWarmth) modulates cursor gaze-follow along
 * two axes, deliberately kept subtle (spec: "must never feel like a switch
 * flipping") — both are continuous functions of warmth in [0, 1], not
 * thresholds:
 *
 *  - Engagement probability: how often a given poll cycle actually "notices"
 *    the cursor and refreshes the tracked target, and for how long that
 *    engagement lingers before needing to be re-rolled. Cold (low warmth)
 *    means rare, brief glances; warm means near-constant attentive
 *    following. Implemented as a per-poll coin flip (engageChance) gating
 *    whether this cycle's read is applied at all, plus an engagement window
 *    (engageHoldMs) that keeps state.active true a little past the last
 *    successful "notice" so a cold roll doesn't instantly snap the gaze back
 *    to center every single poll — it should read as "glances happen less
 *    often", not "flickers on and off 5x/second".
 *  - Bias strength: once engaged, how far the eyes actually travel toward
 *    the cursor — cold readings are damped toward center (a brief, small
 *    glance), warm readings pass the full normalized offset through.
 *
 * warmth=1 reproduces the pre-warmth behavior exactly (engageChance=1,
 * strength=1, engageHoldMs effectively irrelevant since every poll
 * re-engages) — so a caller that never passes getWarmth (or whose warmth
 * source always returns 1) sees byte-identical tracking to before this
 * feature existed. */
// 2026-07-28, two revisions in one day: the original cold floor (8%
// chance x 0.35 strength x 250ms holds) compounded with the small gaze
// envelope into tracking that read as broken; an intermediate raise
// (40%/0.65/900ms) still left her ignoring the mouse over half the time
// on a young arc. Owner's verdict: she should ALWAYS follow the cursor.
// The floors now equal the ceilings — every poll engages at full
// strength regardless of warmth — but the lerp machinery and getWarmth
// plumbing stay in place so a future pass can re-widen the band by
// editing constants only.
const ENGAGEMENT_CHANCE_MIN = 1.0
const ENGAGEMENT_CHANCE_MAX = 1.0
const ENGAGEMENT_HOLD_MS_MIN = 2500
const ENGAGEMENT_HOLD_MS_MAX = 2500
const BIAS_STRENGTH_MIN = 1.0
const BIAS_STRENGTH_MAX = 1.0

const lerp = (a: number, b: number, t: number): number => a + (b - a) * THREE.MathUtils.clamp(t, 0, 1)

/**
 * Starts polling `pet_cursor_context` and writing normalized offsets into
 * `state`. No-op (never starts) unless running inside Tauri — browsers have
 * no cross-window cursor API and no `invoke`. Pauses while the document is
 * hidden (`document.visibilityState !== 'visible'`) rather than polling a
 * window nobody can see, and resumes on visibilitychange.
 */
export const startCursorGazePolling = (
  state: CursorGazeState,
  deps?: {
    /** Arc-driven warmth [0, 1] (../vrm/arcWarmth), sampled fresh every
     * poll. Omitted (or returning 1) reproduces the original always-fully-
     * engaged behavior exactly — see the warmth-modulation doc above. */
    getWarmth?: () => number
    random?: () => number
  },
): CursorGazePoller => {
  if (!isTauri()) {
    return { stop: () => {} }
  }

  const getWarmth = deps?.getWarmth ?? (() => 1)
  const random = deps?.random ?? Math.random

  let stopped = false
  let timeoutId: number | null = null
  let consecutiveFailures = 0
  const lingerState = createCursorLingerState()
  // Engagement window: while now <= engagedUntil, state stays 'active' and
  // gazeDirector keeps easing toward the last-written target even on a poll
  // cycle that didn't re-engage — see the warmth doc above for why this
  // exists (cold readings shouldn't flicker active on/off every 200ms).
  let engagedUntil = 0
  // Lazily imported: the plain-browser bundle (main window, web builds)
  // should never eagerly pull in the Tauri IPC glue just because this
  // module is imported — only the pet variant under Tauri actually calls
  // startCursorGazePolling.
  let invokeFn: ((cmd: string) => Promise<unknown>) | null = null

  const poll = async () => {
    if (stopped) return
    if (document.visibilityState !== 'visible') {
      scheduleNext()
      return
    }
    try {
      if (!invokeFn) {
        const mod = await import('@tauri-apps/api/core')
        invokeFn = mod.invoke as (cmd: string) => Promise<unknown>
      }
      const result = await invokeFn('pet_cursor_context')
      if (stopped) return
      if (!isPetCursorContext(result)) {
        throw new Error('pet_cursor_context: unexpected shape')
      }
      consecutiveFailures = 0
      // Linger play rides the same successful reading; dispatch is here
      // (not in the pure tracker) so tests never need a DOM.
      if (updateCursorLinger(lingerState, result, performance.now())) {
        window.dispatchEvent(new CustomEvent(CURSOR_LINGER_EVENT))
      }
      const warmth = THREE.MathUtils.clamp(getWarmth(), 0, 1)
      const engageChance = lerp(ENGAGEMENT_CHANCE_MIN, ENGAGEMENT_CHANCE_MAX, warmth)
      const now = performance.now()
      if (random() < engageChance) {
        // This poll "notices" the cursor: refresh the target (damped by
        // bias strength) and extend the engagement window.
        const { lateral, vertical } = cursorToNormalizedOffset(result)
        const strength = lerp(BIAS_STRENGTH_MIN, BIAS_STRENGTH_MAX, warmth)
        state.active = true
        state.targetLateral = lateral * strength
        state.targetVertical = vertical * strength
        state.lastUpdateAt = now
        engagedUntil = now + lerp(ENGAGEMENT_HOLD_MS_MIN, ENGAGEMENT_HOLD_MS_MAX, warmth)
      } else if (now > engagedUntil) {
        // Didn't re-engage this cycle AND the previous engagement window
        // has lapsed — a cold glance genuinely ends, gazeDirector's cursor
        // phase goes inactive, and something else (or nothing) takes over.
        state.active = false
      }
      // else: mid-engagement-window but this particular poll didn't roll a
      // fresh notice — leave the existing target in place rather than
      // flickering; the next poll gets another chance.
    } catch {
      // Command not registered yet (sibling agent's Tauri piece may land
      // later), a transient IPC hiccup, or (defensively) a non-Tauri
      // context — all degrade the same way: count it, never log it (this
      // polls 5x/second, so a real outage would otherwise spam the
      // console), and give up entirely after a few in a row.
      consecutiveFailures += 1
      if (consecutiveFailures >= MAX_CONSECUTIVE_FAILURES) {
        state.active = false
        stopped = true
        diagnostics.increment('pet.cursor_gaze_disabled')
        return
      }
    }
    scheduleNext()
  }

  const scheduleNext = () => {
    if (stopped) return
    timeoutId = window.setTimeout(() => {
      void poll()
    }, POLL_INTERVAL_MS)
  }

  void poll()

  return {
    stop: () => {
      stopped = true
      if (timeoutId !== null) {
        window.clearTimeout(timeoutId)
        timeoutId = null
      }
    },
  }
}
