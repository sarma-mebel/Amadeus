/**
 * Desktop-roaming pet (Tauri only, opt-in): every few minutes, when nothing
 * else is going on, the pet strolls along the screen's work-area bottom edge
 * to a new perch point. The window motion itself is a Rust-side tween
 * (`animate_pet_window_to`, ~30Hz, cancellable, clamped to the work area);
 * this module owns the WHEN — cadence, eligibility, cancellation — plus the
 * avatar-layer notifications ('kurisu-pet-roam-changed' + the shared
 * 'kurisu:activity' pulse that keeps the render loop at full frame rate for
 * the stroll's duration).
 *
 * Opt-in contract: localStorage `kurisu_pet_roam`, default OFF (only the
 * literal string 'true' enables). Re-read at every attempt (not cached at
 * start) so flipping the Settings toggle — which lives in the MAIN window,
 * a separate browser context sharing the same origin — takes effect on the
 * pet's next cadence tick without any cross-window event plumbing, same
 * convention as the fullscreen-courtesy opt-out.
 *
 * Roaming yields to everything else. A stroll only starts when NONE of
 * these hold, and cancels within STROLL_WATCHDOG_MS if any begins
 * mid-stroll: a chat turn in flight (busy), her speaking, the user's mic
 * capturing (push-to-talk or hands-free listening), dozing, DND,
 * fullscreen-courtesy hidden, or any user interaction within the last
 * INTERACTION_QUIET_MS. Cancellation also fires the Rust-side
 * `cancel_pet_window_animation`, and the Rust tween independently aborts if
 * the window is moved externally (user grabs the drag bar mid-stroll).
 *
 * Persistence: the final position flows through the EXISTING Moved-event →
 * debounce → flush_pet_rect path in Rust (edge-snap before persist), so a
 * restart restores the same validated perch a manual drag would have left.
 * After a completed stroll this module re-queries `pet_perch_state` (after
 * the Rust debounce has flushed) and re-dispatches
 * 'kurisu-pet-perch-changed', so the avatar's perch pose (damped-wind calm
 * stance today, a vetted sit clip if one ever lands eager) tracks the new
 * spot.
 *
 * Avatar during a stroll — the future-walk-clip hook: there is currently NO
 * vetted EAGER walk clip (every Sit/Walk-like manifest entry is lazy), and
 * the hard repo rule is that automatic roles only ever use vetted eager
 * clips. So `findEagerWalkClip()` looks up an eager, enabled, loaded clip
 * with manifest role "walk" (or a walk-named clip as a fallback match) via
 * kurisuBridge.vrmAnimations.describe() — if a future vetting pass promotes
 * one, it plays automatically for the stroll's duration with zero code
 * changes here; until then the visuals fall back to the perch precedent
 * (calm stance; PetView adds a slight CSS lean/bob in the travel direction
 * off the roam-changed event). This module NEVER loads or promotes clips
 * itself.
 *
 * Degrade loudly: every invoke failure counts
 * `pet.roam_invoke_failed` (diagnostics), and after
 * MAX_CONSECUTIVE_ROAM_FAILURES in a row roaming disables itself for the
 * window's lifetime (`pet.roam_disabled`) — the cursorGaze
 * stop-after-consecutive-failures pattern, so an older shell without the
 * commands never spams retries forever.
 */
import { diagnostics } from '../diagnostics'
import { isTauri } from '../platform'
import { kurisuBridge, type VrmClipInfo } from './kurisuBridge'

/** localStorage opt-in flag. Default OFF — only the literal 'true' enables. */
export const PET_ROAM_KEY = 'kurisu_pet_roam'

export const isPetRoamEnabled = (): boolean => {
  try {
    return window.localStorage.getItem(PET_ROAM_KEY) === 'true'
  } catch {
    return false
  }
}

export const setPetRoamEnabled = (enabled: boolean): void => {
  try {
    window.localStorage.setItem(PET_ROAM_KEY, enabled ? 'true' : 'false')
  } catch {
    // Preference just won't persist across reloads.
  }
}

/** Dispatched on window at stroll start/end so the pet UI (CSS lean/bob)
 * and any future avatar-layer listener can react. `direction` is the travel
 * direction: -1 strolling left, +1 strolling right. */
export const PET_ROAM_CHANGE_EVENT = 'kurisu-pet-roam-changed'
export interface PetRoamChangeDetail {
  roaming: boolean
  direction: -1 | 1
}

// --- Cadence -------------------------------------------------------------

/** Randomized several-minute cadence between stroll ATTEMPTS (an attempt
 * that finds her ineligible just waits out the next full interval — "she
 * occasionally wanders", not "she wanders the moment she can"). */
export const ROAM_MIN_INTERVAL_MS = 4 * 60_000
export const ROAM_MAX_INTERVAL_MS = 9 * 60_000

/** First attempt after the driver starts (i.e. right after the user turns
 * the toggle on). The steady-state cadence is minutes long, and the click
 * that enables roaming is itself an interaction that restarts the
 * INTERACTION_QUIET_MS clock — with no shorter first tick, "toggle it on
 * and watch" showed literally nothing for 4-9+ minutes and read as a dead
 * button (reported live on v0.15.0). 90s = just past the quiet window, so
 * a user who enables it and leaves her alone sees the first stroll about
 * a minute and a half in; an attempt that finds her ineligible falls back
 * to the normal cadence. */
export const FIRST_ROAM_DELAY_MS = 90_000

export const nextRoamDelayMs = (random: () => number = Math.random): number =>
  Math.round(ROAM_MIN_INTERVAL_MS + random() * (ROAM_MAX_INTERVAL_MS - ROAM_MIN_INTERVAL_MS))

// --- Eligibility ---------------------------------------------------------

/** No user interaction (pointer/key/wheel in the pet window) for at least
 * this long before a stroll may start — and an interaction during one
 * cancels it (the same predicate gates both, via the stroll watchdog). */
export const INTERACTION_QUIET_MS = 60_000

export interface RoamEligibility {
  /** The kurisu_pet_roam opt-in itself. */
  enabled: boolean
  /** Chat turn in flight. */
  busy: boolean
  /** She is currently speaking (lipSync). */
  speaking: boolean
  /** The user's mic is capturing (push-to-talk held or hands-free listening). */
  listening: boolean
  /** Fully dozing (kurisuBridge.petDoze). Drowsy still behaves normally
   * elsewhere in the pet stack (see petDoze.ts's isDozing doc), so it does
   * not gate roaming either — but night idling long enough to matter here
   * generally crosses into a real doze anyway. */
  dozing: boolean
  /** Do-not-disturb quiet period. */
  dndActive: boolean
  /** Hidden out of courtesy for a fullscreen app. */
  courtesyHidden: boolean
  msSinceInteraction: number
}

/** Single predicate for both "may a stroll start" and "must an in-flight
 * stroll cancel" — one rule, no drift between the two moments. */
export const isEligibleToRoam = (e: RoamEligibility): boolean =>
  e.enabled &&
  !e.busy &&
  !e.speaking &&
  !e.listening &&
  !e.dozing &&
  !e.dndActive &&
  !e.courtesyHidden &&
  e.msSinceInteraction >= INTERACTION_QUIET_MS

// --- Target picking ------------------------------------------------------

/** One monitor's work area (multi-monitor roam extension). */
export interface RoamMonitorArea {
  x: number
  y: number
  w: number
  h: number
}

/** Shape returned by the Rust `pet_roam_context` command — window rect plus
 * containing monitor's work area, physical px. `monitors` (all work areas,
 * containing one included) arrived with the multi-monitor pass; optional so
 * an older shell without it keeps single-monitor strolls working. */
export interface RoamContext {
  win_x: number
  win_y: number
  win_w: number
  win_h: number
  work_x: number
  work_y: number
  work_w: number
  work_h: number
  monitors?: RoamMonitorArea[]
}

const isRoamContext = (value: unknown): value is RoamContext => {
  if (!value || typeof value !== 'object') return false
  const v = value as Record<string, unknown>
  return (
    typeof v.win_x === 'number' &&
    typeof v.win_y === 'number' &&
    typeof v.win_w === 'number' &&
    typeof v.win_h === 'number' &&
    typeof v.work_x === 'number' &&
    typeof v.work_y === 'number' &&
    typeof v.work_w === 'number' &&
    typeof v.work_h === 'number'
  )
}

/** A stroll shorter than this isn't worth getting up for. */
export const MIN_STROLL_DISTANCE_PX = 120
/** And longer than this stops reading as a casual wander. */
export const MAX_STROLL_DISTANCE_PX = 480

export interface RoamTarget {
  x: number
  y: number
  direction: -1 | 1
  distancePx: number
}

/**
 * Picks the next perch point: somewhere along the work-area BOTTOM edge
 * (the perch system's preferred resting edge — bottom outranks left/right
 * in the Rust snap logic too), at least MIN_STROLL_DISTANCE_PX away in a
 * feasible direction. Pure (injected random) so cadence-free unit tests
 * can pin the geometry. Returns null when there's no room to wander (tiny
 * work area, window wider than the free span).
 */
/** Chance a roam tick wanders to a DIFFERENT monitor when one exists —
 * rare enough that her home screen stays home, common enough that the side
 * monitor is genuinely part of her territory. */
export const CROSS_MONITOR_CHANCE = 0.2

const isUsableMonitorArea = (value: unknown): value is RoamMonitorArea => {
  if (!value || typeof value !== 'object') return false
  const v = value as Record<string, unknown>
  return (
    typeof v.x === 'number' &&
    typeof v.y === 'number' &&
    typeof v.w === 'number' &&
    typeof v.h === 'number'
  )
}

/** A cross-monitor perch point on another monitor's bottom edge, or null
 * when there's no other monitor / no room / the dice said stay. */
const pickCrossMonitorTarget = (
  ctx: RoamContext,
  random: () => number,
): RoamTarget | null => {
  const others = (ctx.monitors || [])
    .filter(isUsableMonitorArea)
    .filter((m) => m.x !== ctx.work_x || m.y !== ctx.work_y)
    .filter((m) => m.w >= ctx.win_w + MIN_STROLL_DISTANCE_PX)
  if (!others.length || random() >= CROSS_MONITOR_CHANCE) return null
  const area = others[Math.floor(random() * others.length)] || others[0]
  const minX = area.x
  const maxX = area.x + area.w - ctx.win_w
  if (maxX <= minX) return null
  const x = Math.round(minX + random() * (maxX - minX))
  const y = area.y + area.h - ctx.win_h
  const dx = x - ctx.win_x
  const distance = Math.round(Math.hypot(dx, y - ctx.win_y))
  if (distance < MIN_STROLL_DISTANCE_PX) return null
  return { x, y, direction: dx < 0 ? -1 : 1, distancePx: distance }
}

export const pickRoamTarget = (
  ctx: RoamContext,
  random: () => number = Math.random,
): RoamTarget | null => {
  // Multi-monitor: occasionally the stroll's destination is another
  // screen entirely (Rust clamps against the TARGET's monitor since the
  // same pass, so the point survives the re-clamp). Falls through to the
  // normal same-monitor pick whenever the roll or geometry says no.
  const crossTarget = pickCrossMonitorTarget(ctx, random)
  if (crossTarget) return crossTarget
  const minX = ctx.work_x
  const maxX = ctx.work_x + ctx.work_w - ctx.win_w
  if (maxX <= minX) return null
  const roomLeft = ctx.win_x - minX
  const roomRight = maxX - ctx.win_x
  const canLeft = roomLeft >= MIN_STROLL_DISTANCE_PX
  const canRight = roomRight >= MIN_STROLL_DISTANCE_PX
  if (!canLeft && !canRight) return null
  const direction: -1 | 1 = canLeft && canRight ? (random() < 0.5 ? -1 : 1) : canLeft ? -1 : 1
  const room = direction === -1 ? roomLeft : roomRight
  const distance = Math.min(
    room,
    MIN_STROLL_DISTANCE_PX + random() * (MAX_STROLL_DISTANCE_PX - MIN_STROLL_DISTANCE_PX),
  )
  return {
    x: Math.round(ctx.win_x + direction * distance),
    // Bottom work-area edge: lands exactly where the Rust perch snap wants
    // it, so the post-stroll flush re-perches without visibly nudging.
    y: ctx.work_y + ctx.work_h - ctx.win_h,
    direction,
    distancePx: Math.round(distance),
  }
}

/** Slow glide: ~45 physical px/s, clamped so even a clamped-short stroll
 * still reads as deliberate motion and a long one never drags on. */
export const STROLL_SPEED_PX_PER_S = 45
export const MIN_STROLL_MS = 2_500
export const MAX_STROLL_MS = 15_000

export const strollDurationMs = (distancePx: number): number =>
  Math.round(
    Math.min(MAX_STROLL_MS, Math.max(MIN_STROLL_MS, (distancePx / STROLL_SPEED_PX_PER_S) * 1000)),
  )

// --- Drag-reaction suppression -------------------------------------------

/** How long after roam motion ends the 'dragged' poke stays suppressed —
 * comfortably covers usePetChat's DRAG_SETTLE_MS debounce firing off the
 * tween's own Moved events, so she never protests her own walk. */
export const ROAM_SUPPRESS_AFTER_MS = 5_000

/** Module-level (not per-driver) because the consumer — usePetChat's
 * drag-settle reaction — has no handle to the driver instance. */
let roamMotionUntil = 0

/** True while a stroll is moving the window, or shortly after (the OS move
 * events it produced are indistinguishable from a user drag downstream). */
export const isRoamMotionRecent = (now: number = Date.now()): boolean => now < roamMotionUntil

/** Test seam. */
export const resetRoamMotionForTest = (): void => {
  roamMotionUntil = 0
}

/** Other PROGRAMMATIC window motion (the tuck/peek slides in petTuck.ts)
 * rides the same suppression window so the drag-settle poke never fires on
 * a slide she initiated herself. */
export const markProgrammaticPetMotion = (durationMs: number): void => {
  roamMotionUntil = Math.max(
    roamMotionUntil,
    Date.now() + durationMs + ROAM_SUPPRESS_AFTER_MS,
  )
}

// --- Future-vetted walk clip hook ----------------------------------------

/**
 * The clearly-marked upgrade hook (see module doc): an EAGER, enabled,
 * loaded clip with manifest role "walk" — or, failing a dedicated role, a
 * walk-named eager clip — is played (looping) for the stroll and handed
 * back to an eager idle afterwards. Today this always returns null (every
 * walk/sit-like library clip is lazy/unvetted, and this module never
 * promotes clips), so the CSS glide fallback is what actually ships.
 * 'walk' is not yet in the AnimationRole union — a future manifest role
 * lands as a plain string on describe(), hence the cast.
 */
export const findEagerWalkClip = (): VrmClipInfo | null =>
  kurisuBridge.vrmAnimations
    ?.describe()
    .find(
      (clip) =>
        !clip.lazy &&
        !clip.disabled &&
        clip.loaded &&
        ((clip.role as string) === 'walk' || /(^|[^a-z])walk/i.test(clip.name)),
    ) ?? null

// --- Driver ---------------------------------------------------------------

/** Mid-stroll eligibility re-check cadence (cancel-on-begin latency). */
export const STROLL_WATCHDOG_MS = 250
/** 'kurisu:activity' pulse cadence during a stroll — each pulse holds the
 * render loop's idle throttle open for IDLE_ACTIVITY_WINDOW_MS (3s), so 1s
 * keeps full frame rate for the whole glide with margin. */
export const ROAM_ACTIVITY_PULSE_MS = 1_000
export const MAX_CONSECUTIVE_ROAM_FAILURES = 3
/** Post-completion delay before re-querying perch state: the Rust side
 * flushes (snap + persist) ~800ms after the tween's last Moved event. */
export const PERCH_REFRESH_DELAY_MS = 1_500

type InvokeFn = (cmd: string, args?: Record<string, unknown>) => Promise<unknown>

export interface PetRoamDriverDeps {
  /** Fresh eligibility snapshot minus the opt-in flag (read internally from
   * localStorage each time so the Settings toggle applies live). */
  getEligibility: () => Omit<RoamEligibility, 'enabled'>
  /** Test seams — production uses the lazy @tauri-apps import / Math.random
   * / real platform detection. */
  invokeFn?: InvokeFn
  random?: () => number
  isTauriFn?: () => boolean
}

export interface PetRoamHandle {
  stop: () => void
  /** Cancel an in-flight stroll (interaction listeners call this so a
   * pointer-down cancels instantly instead of waiting out the watchdog). */
  cancelStroll: (reason?: string) => void
  strolling: () => boolean
}

const NOOP_HANDLE: PetRoamHandle = { stop: () => {}, cancelStroll: () => {}, strolling: () => false }

/**
 * Starts the roam scheduler. No-op outside Tauri (browsers cannot move
 * their own window — the Settings toggle is hidden there too). The returned
 * handle stops everything and best-effort cancels any in-flight tween.
 */
export const startPetRoam = (deps: PetRoamDriverDeps): PetRoamHandle => {
  const tauri = deps.isTauriFn ?? isTauri
  if (!tauri()) return NOOP_HANDLE
  const random = deps.random ?? Math.random

  let stopped = false
  let disabled = false
  let strolling = false
  let cancelRequested = false
  let currentDirection: -1 | 1 = 1
  let scheduleTimer: number | null = null
  let watchdogTimer: number | null = null
  let activityTimer: number | null = null
  let consecutiveFailures = 0
  let invokeFn: InvokeFn | null = deps.invokeFn ?? null

  const getInvoke = async (): Promise<InvokeFn> => {
    if (!invokeFn) {
      // Lazy like cursorGaze: the plain-browser bundle must never eagerly
      // pull in the Tauri IPC glue just because this module is imported.
      const mod = await import('@tauri-apps/api/core')
      invokeFn = mod.invoke as InvokeFn
    }
    return invokeFn
  }

  const recordFailure = (detail: string) => {
    consecutiveFailures += 1
    diagnostics.increment('pet.roam_invoke_failed', detail)
    if (consecutiveFailures >= MAX_CONSECUTIVE_ROAM_FAILURES) {
      disabled = true
      diagnostics.increment(
        'pet.roam_disabled',
        `after ${consecutiveFailures} consecutive invoke failures — roaming is off until the pet window reopens`,
      )
    }
  }

  // Spread FIRST, flag second: the opt-in must always come from the live
  // localStorage read, even if a caller's snapshot object structurally
  // carries its own `enabled` field.
  const fullEligibility = (): RoamEligibility => ({
    ...deps.getEligibility(),
    enabled: isPetRoamEnabled(),
  })

  const cancelStroll = (reason = 'cancelled') => {
    if (!strolling || cancelRequested) return
    cancelRequested = true
    void (async () => {
      try {
        await (await getInvoke())('cancel_pet_window_animation')
      } catch (error) {
        recordFailure(`cancel_pet_window_animation (${reason}): ${String(error)}`)
      }
    })()
  }

  /** Runs exactly once per stroll, whether it completed, cancelled, or
   * errored — clears the watchdog/activity pulse and tells the UI layer
   * the glide ended. */
  const endStroll = () => {
    strolling = false
    cancelRequested = false
    if (watchdogTimer !== null) {
      window.clearInterval(watchdogTimer)
      watchdogTimer = null
    }
    if (activityTimer !== null) {
      window.clearInterval(activityTimer)
      activityTimer = null
    }
    roamMotionUntil = Date.now() + ROAM_SUPPRESS_AFTER_MS
    window.dispatchEvent(
      new CustomEvent<PetRoamChangeDetail>(PET_ROAM_CHANGE_EVENT, {
        detail: { roaming: false, direction: currentDirection },
      }),
    )
  }

  /** After a completed stroll (and the Rust-side flush/snap), refresh the
   * avatar layer's perch pose off the authoritative state — mirrors
   * usePetChat's drag-settle re-query. Failures here use the existing
   * perch counter and do NOT count toward the roam-disable threshold (a
   * missing perch refresh degrades to a stale pose, not a dead feature). */
  const refreshPerchState = async () => {
    try {
      const state = await (await getInvoke())('pet_perch_state')
      if (!stopped) {
        window.dispatchEvent(new CustomEvent('kurisu-pet-perch-changed', { detail: state }))
      }
    } catch (error) {
      diagnostics.increment('pet.perch_state_failed', String(error))
    }
  }

  const attemptStroll = async () => {
    if (stopped || disabled) return
    // Tucked away (vrm/petTuck, key literal duplicated to avoid an import
    // cycle — petTuck imports this module's motion-suppression mark): a
    // deliberate off-screen park means no strolling, and pet_roam_context
    // can't even answer on a left tuck (top-left corner off every
    // monitor), so this must return BEFORE the failure counter arms.
    try {
      if (window.localStorage.getItem('kurisu_pet_tucked')) return
    } catch {
      // Storage unavailable — roam proceeds.
    }
    if (!isEligibleToRoam(fullEligibility())) return

    let invoke: InvokeFn
    try {
      invoke = await getInvoke()
    } catch (error) {
      recordFailure(`import @tauri-apps/api/core: ${String(error)}`)
      return
    }

    let ctx: unknown
    try {
      ctx = await invoke('pet_roam_context')
    } catch (error) {
      recordFailure(`pet_roam_context: ${String(error)}`)
      return
    }
    if (!isRoamContext(ctx)) {
      recordFailure('pet_roam_context: unexpected shape')
      return
    }
    consecutiveFailures = 0
    if (stopped) return

    const target = pickRoamTarget(ctx, random)
    if (!target) return
    const duration = strollDurationMs(target.distancePx)

    strolling = true
    cancelRequested = false
    currentDirection = target.direction
    roamMotionUntil = Date.now() + duration + ROAM_SUPPRESS_AFTER_MS
    window.dispatchEvent(
      new CustomEvent<PetRoamChangeDetail>(PET_ROAM_CHANGE_EVENT, {
        detail: { roaming: true, direction: target.direction },
      }),
    )
    // Roaming counts as motion: ride the existing 'kurisu:activity'
    // listener (VRMViewer's idle-throttle disqualifier + attentive gaze)
    // instead of inventing a new render-loop input this module can't reach.
    activityTimer = window.setInterval(
      () => window.dispatchEvent(new Event('kurisu:activity')),
      ROAM_ACTIVITY_PULSE_MS,
    )
    window.dispatchEvent(new Event('kurisu:activity'))
    // Cancel the moment ANY yield condition begins mid-stroll (speaking,
    // mic, doze, DND, courtesy, interaction, or the toggle flipping off).
    watchdogTimer = window.setInterval(() => {
      if (!isEligibleToRoam(fullEligibility())) cancelStroll('eligibility')
    }, STROLL_WATCHDOG_MS)

    // Future-vetted walk clip (see findEagerWalkClip's doc) — null today.
    const walkClip = findEagerWalkClip()
    if (walkClip) {
      kurisuBridge.vrmAnimations?.play(walkClip.name, { loop: true })
    }

    let completed = false
    try {
      completed =
        (await invoke('animate_pet_window_to', {
          x: target.x,
          y: target.y,
          durationMs: duration,
        })) === true
    } catch (error) {
      recordFailure(`animate_pet_window_to: ${String(error)}`)
    } finally {
      endStroll()
      if (walkClip) {
        // Hand the body back to an eager looping idle — playByName's
        // loop:true never naturally reverts via the mixer 'finished' path.
        const idleClip = kurisuBridge.vrmAnimations
          ?.describe()
          .find((clip) => !clip.lazy && !clip.disabled && clip.loaded && clip.role === 'idle')
        if (idleClip) kurisuBridge.vrmAnimations?.play(idleClip.name, { loop: true })
      }
    }
    if (completed && !stopped) {
      window.setTimeout(() => {
        if (!stopped) void refreshPerchState()
      }, PERCH_REFRESH_DELAY_MS)
    }
  }

  const schedule = (delayMs?: number) => {
    if (stopped || disabled) return
    scheduleTimer = window.setTimeout(() => {
      void attemptStroll().finally(() => schedule())
    }, delayMs ?? nextRoamDelayMs(random))
  }
  schedule(FIRST_ROAM_DELAY_MS)

  return {
    stop: () => {
      if (strolling) cancelStroll('stop')
      stopped = true
      if (scheduleTimer !== null) {
        window.clearTimeout(scheduleTimer)
        scheduleTimer = null
      }
      if (watchdogTimer !== null) {
        window.clearInterval(watchdogTimer)
        watchdogTimer = null
      }
      if (activityTimer !== null) {
        window.clearInterval(activityTimer)
        activityTimer = null
      }
    },
    cancelStroll,
    strolling: () => strolling,
  }
}
