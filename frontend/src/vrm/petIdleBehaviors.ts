/**
 * Pet-window ambient idle behaviors: while the desktop pet sits around with
 * nobody actively engaging it, occasionally play one clip from the SAME
 * vetted automatic pools the animation scheduler already draws from
 * (currently: the `subtle` idle-variety pool via
 * scheduler.playRandomSubtleGesture). This is deliberately NOT a new
 * automatic role and NOT a new pool — repo policy (see animationScheduler.ts
 * / static/animations/SOURCES.md) is that only visually-vetted, eager-loaded
 * clips already wired into an existing automatic role may fire unattended;
 * a lazy library clip can never serve one. This module is a slower,
 * separate cadence layered on TOP of the scheduler's own ~45-90s idle-
 * variety timer (scheduleActual) — it does not replace or duplicate it, it
 * just adds a second, much rarer "she shifts around while sitting on your
 * desktop all day" beat suited to an always-visible pet window rather than
 * a focused chat pane.
 *
 * Cadence: randomized ~2-5 minutes per spec, extended further (and biased
 * toward calmer clips when the pool distinguishes them) between 23:00 and
 * 06:00 local time — nobody wants a bored little companion doing idle
 * gestures at full daytime frequency at 3am on someone's desktop.
 */

import { clamp01 } from './arcWarmth'

export interface PetIdleBehaviorsDeps {
  /** scheduler.playRandomSubtleGesture — returns true if a clip actually
   * played (false when the pool is empty or every candidate is filtered
   * out, in which case this module just reschedules and tries later). */
  playRandomSubtleGesture: (fadeDuration?: number, preferCalm?: (name: string) => boolean) => boolean
  hasSubtlePool: () => boolean
  /** True while she's speaking, mid-gesture ('actual' animation mode), or a
   * chat/poke/etc. interaction landed recently — the same conservative
   * idle definition the render loop's throttle uses, reused here so the
   * ambient beat only fires when she's genuinely just sitting there. */
  isBusyOrRecentlyInteracted: () => boolean
  /** Wall-clock ms source, injectable for tests. */
  now?: () => number
  random?: () => number
  /** Arc-driven warmth [0, 1] (../vrm/arcWarmth), sampled fresh each time a
   * delay is scheduled/a clip is picked. Omitted (or returning a mid value)
   * reproduces the pre-warmth cadence/selection — see WARMTH_* below for
   * the exact (deliberately subtle) modulation. */
  getWarmth?: () => number
}

export interface PetIdleBehaviors {
  stop: () => void
}

const MIN_DELAY_MS = 2 * 60 * 1000
const MAX_DELAY_MS = 5 * 60 * 1000
// Late night: both a lower frequency (roughly half as often) and, when the
// pool has any name-matched calmer clips, a preference for those.
const NIGHT_MIN_DELAY_MS = 4 * 60 * 1000
const NIGHT_MAX_DELAY_MS = 9 * 60 * 1000
const NIGHT_START_HOUR = 23
const NIGHT_END_HOUR = 6

const isLateNight = (date: Date): boolean => {
  const hour = date.getHours()
  return hour >= NIGHT_START_HOUR || hour < NIGHT_END_HOUR
}

/** Matches the calmer end of the vetted subtle pool by name — "Relax" and
 * "Sleepy" are the two library clips in that pool clearly meant to read as
 * winding down (see static/animations/manifest.json's subtle-role clips).
 * Falls back to the full pool automatically (see playRandomSubtleGesture
 * above) if neither is loaded for a given outfit/manifest. */
export const isCalmSubtleClipName = (name: string): boolean =>
  /\b(relax|sleepy)\b/i.test(name)

/** Matches the warmer/affectionate end of the vetted subtle pool by name —
 * "Caring" is the one library clip in that pool clearly meant to read as
 * affectionate rather than merely calm (see the same manifest role). Used
 * for the warmth-driven clip preference (arc-driven warmth, ../vrm/
 * arcWarmth) — a high-warmth relationship slightly prefers this alongside
 * the calm set, same fallback-to-full-pool contract as isCalmSubtleClipName
 * if the manifest/outfit doesn't load it. */
export const isAffectionateSubtleClipName = (name: string): boolean => /\bcaring\b/i.test(name)

// Warmth modulation constants (deliberately subtle — spec: "must never feel
// like a switch flipping"). At warmth=1 the delay range shrinks by this
// fraction (a bit more frequent, never a different order of magnitude); at
// warmth=0 the range is unchanged from the base night/day range above.
const WARMTH_MAX_FREQUENCY_BOOST = 0.35

/** Starts the ambient idle timer. Call only for the pet variant — this has
 * no platform/variant gating of its own (VRMViewer decides when to call
 * it), unlike cursorGaze.ts which gates on isTauri() internally because it
 * also needs a Tauri-only IPC command. */
export const startPetIdleBehaviors = (deps: PetIdleBehaviorsDeps): PetIdleBehaviors => {
  const now = deps.now ?? (() => Date.now())
  const random = deps.random ?? Math.random
  const getWarmth = deps.getWarmth

  let stopped = false
  let timeoutId: number | null = null

  const scheduleNext = () => {
    if (stopped) return
    const night = isLateNight(new Date(now()))
    let min = night ? NIGHT_MIN_DELAY_MS : MIN_DELAY_MS
    let max = night ? NIGHT_MAX_DELAY_MS : MAX_DELAY_MS
    // Warmth: slightly higher frequency (a shrunk, not different, range) at
    // high warmth — sampled fresh per scheduling call so a slow warmth
    // change (hourly refresh) is reflected on the very next beat rather
    // than only at module init.
    const warmth = clamp01(getWarmth?.() ?? 0)
    if (warmth > 0) {
      const shrink = 1 - warmth * WARMTH_MAX_FREQUENCY_BOOST
      min *= shrink
      max *= shrink
    }
    const delay = min + random() * (max - min)
    timeoutId = window.setTimeout(fire, delay)
  }

  const fire = () => {
    if (stopped) return
    if (deps.hasSubtlePool() && !deps.isBusyOrRecentlyInteracted()) {
      const night = isLateNight(new Date(now()))
      const warmth = clamp01(getWarmth?.() ?? 0)
      // Clip preference: night still prefers calm-only (unchanged); a warm
      // relationship (independent of time of day) prefers calm-OR-
      // affectionate over the full pool — a coin flip weighted by warmth so
      // it's a soft bias, not an override, and never fights the stronger
      // night preference when both apply.
      const preferCalm = night
        ? isCalmSubtleClipName
        : warmth > 0 && random() < warmth
          ? (name: string) => isCalmSubtleClipName(name) || isAffectionateSubtleClipName(name)
          : undefined
      deps.playRandomSubtleGesture(1.2, preferCalm)
    }
    // Reschedule unconditionally (even after a skipped/busy tick) — a
    // window that stayed busy through one check just gets checked again
    // later, rather than the ambient beat going permanently silent.
    scheduleNext()
  }

  scheduleNext()

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
