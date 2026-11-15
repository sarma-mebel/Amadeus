/**
 * Presence loop for the desktop-pet window: greeting on open, due-reminder
 * and reach-out polls, and idle chatter.
 *
 * The pet previously ran NONE of these — they all lived in the main
 * window's usePresence — so desktop-pet-only usage (main window closed,
 * pet on the desktop all day) silently lost greetings, reminder
 * announcements, and reach-outs entirely. Everything here is voice-first:
 * the pet is by definition visible, so lines speak through the pet's own
 * lipSync and fade through the caption; there is no OS-notification path.
 *
 * The server does the real policing — should_greet's freshness gap,
 * idle chatter's cooldown + mid-session-only rule, and the exactly-once
 * pops for reminders/reach-out — so running alongside an open main window
 * can never double-announce; whichever window polls first speaks.
 */
import { useEffect, useRef, useState } from 'react'
import { apiFetch } from '../api'
import { diagnostics } from '../diagnostics'
import { idleChatterDelayMultiplier } from '../presenceLevel'
import { presenceRequest } from '../presenceApi'
import { useAmbientSignals } from './useAmbientSignals'
import { isPetTucked } from '../vrm/petTuck'
import { kurisuBridge } from '../vrm/kurisuBridge'

export interface PresencePayload {
  emotion?: string | null
  emotion_intensity?: number | null
  audio?: string | null
  audio_mime_type?: string | null
}

export interface PetPresenceDeps {
  /** True while a chat turn is in flight — presence stays quiet then. */
  isBusyRef: React.RefObject<boolean>
  /** Apply one presence payload: expression + audio + caption. */
  applyLine: (data: PresencePayload, text: string) => void | Promise<void>
  /**
   * Fullscreen courtesy mode (feature: another app is foreground-fullscreen)
   * — true means "hard-mute every presence fetch", exactly like DND. Reads
   * through a ref (not a plain boolean) so this long-lived effect always
   * sees the latest value without re-subscribing, same pattern as isBusyRef.
   * Optional: PetView is this hook's only caller today, but tests construct
   * deps directly, so a missing ref must read as "not courtesy-active".
   */
  isCourtesyActiveRef?: React.RefObject<boolean>
  /**
   * True while she's dozing (kurisuBridge.petDoze.active()). Idle chatter
   * — a spontaneous monologue with nobody prompting it — skips outright
   * while dozing (no fetch, matching the DND/courtesy "don't even ask"
   * rule, since idle has no server-side pop to lose). Reminders and
   * reach-out are deliberately NOT gated by this: they're time-sensitive
   * and allowed to wake her (see onBeforePresenceWake below).
   */
  isDozingRef?: React.RefObject<boolean>
  /**
   * Called immediately before a REMINDER or REACH-OUT line is applied
   * (never for greeting/idle — those aren't "wake her up" worthy). Wired to
   * `kurisuBridge.petDoze?.wake('presence')` by the caller; this hook stays
   * doze-API-agnostic so it doesn't need to know about kurisuBridge.
   */
  onBeforePresenceWake?: () => void
}

const POLL_MS = 30_000
const GREETING_DELAY_MS = 2_500
const IDLE_MIN_MS = 4 * 60_000
const IDLE_MAX_MS = 7 * 60_000
/** How long, beyond GREETING_DELAY_MS, the greeting attempt will wait for
 * kurisuBridge.lipSync to come online before firing text-first anyway. The
 * VRM scene is a lazy chunk + GLTF fetch/parse — routinely >2.5s on a cold
 * packaged launch — so kurisuBridge.lipSync is still null when the original
 * fixed-delay attempt fired, and the greeting's audio silently no-op'd
 * while still being marked delivered forever (greetedRef). */
const LIPSYNC_WAIT_TIMEOUT_MS = 15_000
const LIPSYNC_POLL_MS = 250

/** Resolves once kurisuBridge.lipSync is non-null, or after
 * LIPSYNC_WAIT_TIMEOUT_MS elapses — whichever comes first. Never rejects;
 * callers branch on whatever kurisuBridge.lipSync reads as afterward. */
const waitForLipSyncBridge = (): Promise<void> =>
  new Promise((resolve) => {
    if (kurisuBridge.lipSync) {
      resolve()
      return
    }
    const startedAt = Date.now()
    const poll = () => {
      if (kurisuBridge.lipSync || Date.now() - startedAt >= LIPSYNC_WAIT_TIMEOUT_MS) {
        resolve()
        return
      }
      window.setTimeout(poll, LIPSYNC_POLL_MS)
    }
    window.setTimeout(poll, LIPSYNC_POLL_MS)
  })

/**
 * Do-not-disturb: a quiet period the user sets from the pet controls.
 * localStorage `kurisu_pet_dnd_until` holds either an epoch-ms deadline
 * (string) or the literal "-1" for "until turned off"; absent or a past
 * timestamp both mean off. This is a plain localStorage contract (not a
 * hook of its own) because both PetView (chip + button UI) and
 * usePetPresence (the actual fetch gate) need to read/write it, and neither
 * owns the other.
 *
 * Deliberately NOT "poll and discard while DND": reminders and reach-out
 * are exactly-once pops on the server — a discarded poll response is a
 * permanently lost reminder, not a deferred one. So DND must prevent the
 * fetch itself, not just its display.
 */
export const PET_DND_KEY = 'kurisu_pet_dnd_until'
/** Sentinel stored value for "quiet until manually turned off". */
export const PET_DND_INDEFINITE = '-1'

/** Reads the current DND deadline: null (off), 'indefinite', or an epoch-ms
 * number for a timed quiet period already known to be in the future. A
 * stored-but-expired timestamp reads as null (off) — callers never need to
 * separately check staleness. */
export const readPetDndUntil = (): number | 'indefinite' | null => {
  let raw: string | null
  try {
    raw = window.localStorage.getItem(PET_DND_KEY)
  } catch {
    return null
  }
  if (!raw) return null
  if (raw === PET_DND_INDEFINITE) return 'indefinite'
  const until = Number(raw)
  if (!Number.isFinite(until) || until <= Date.now()) return null
  return until
}

export const isPetDndActive = (): boolean => readPetDndUntil() !== null

export const setPetDndUntil = (value: number | 'indefinite' | null): void => {
  try {
    if (value === null) {
      window.localStorage.removeItem(PET_DND_KEY)
    } else {
      window.localStorage.setItem(
        PET_DND_KEY,
        value === 'indefinite' ? PET_DND_INDEFINITE : String(value),
      )
    }
  } catch {
    // Preference just won't persist; the in-memory chip state still works
    // for this session via the caller's own state.
  }
}

/** Live DND status for UI: re-checks on an interval so the chip/button
 * clear themselves the moment a timed quiet period lapses, without
 * requiring a fetch attempt to notice. Cheap (one localStorage read). */
export const usePetDndStatus = (): { active: boolean; until: number | 'indefinite' | null } => {
  const [status, setStatus] = useState(() => {
    const until = readPetDndUntil()
    return { active: until !== null, until }
  })
  useEffect(() => {
    const check = () => {
      const until = readPetDndUntil()
      setStatus((prev) => {
        if (prev.until === until) return prev
        return { active: until !== null, until }
      })
    }
    check()
    const intervalId = window.setInterval(check, 5_000)
    return () => window.clearInterval(intervalId)
  }, [])
  return status
}

/**
 * Fullscreen courtesy mode: opt-out flag only. The actual foreground-
 * fullscreen poll + window hide/show live in usePetChat (they need
 * kurisuBridge for the busy/speaking check and the Tauri window handle),
 * but the localStorage contract lives here next to DND's, since PetView
 * reads both from the same module for the DND popover's checkbox row.
 * Convention matches the rest of the pet's opt-outs (screen companion,
 * listening reactions): absent or anything other than the literal string
 * "false" means enabled, so a pre-feature install defaults to ON without
 * needing a migration.
 */
export const FULLSCREEN_COURTESY_KEY = 'kurisu_fullscreen_courtesy'

/**
 * Heartbeat localStorage key: stamped with `Date.now()` on mount and on
 * every 30s poll tick (see the reminders/reach-out effect below). The main
 * window's usePresence reads this (petIsAlive, freshness 90s) to tell "a
 * pet window is open and alive" from "no pet, or a stale/crashed one" —
 * without it, a hidden main window has no way to know a visible pet is the
 * one place its greeting/reminder/reach-out lines could actually be heard,
 * and would keep competing with the pet for the server's exactly-once pops.
 * Shared contract pinned in tests/test_contracts.py.
 */
export const PET_ALIVE_KEY = 'kurisu_pet_alive'

const writePetAliveHeartbeat = (): void => {
  try {
    window.localStorage.setItem(PET_ALIVE_KEY, String(Date.now()))
  } catch {
    // Best-effort — a lost heartbeat write just means the main window can't
    // tell the pet is alive and falls back to its own (unchanged) behavior.
  }
}

export const isFullscreenCourtesyEnabled = (): boolean => {
  try {
    return window.localStorage.getItem(FULLSCREEN_COURTESY_KEY) !== 'false'
  } catch {
    return true
  }
}

export const setFullscreenCourtesyEnabled = (enabled: boolean): void => {
  try {
    // Storing only the "off" value (rather than "true"/"false" both) would
    // also work, but writing the explicit value keeps this symmetric with
    // every other boolean pref in the codebase and makes the stored state
    // greppable/debuggable without having to know the "absent = on" rule.
    window.localStorage.setItem(FULLSCREEN_COURTESY_KEY, enabled ? 'true' : 'false')
  } catch {
    // Preference just won't persist across reloads.
  }
}

// Request bodies come from the shared presenceApi builder (this hook's
// local copy was its template). Note the shared one resolves user_id the
// way chat does (minting one if missing) instead of sending a raw — and on
// a fresh install null — localStorage read.

const adoptSession = (data: { session_id?: string }) => {
  if (!data.session_id) return
  try {
    localStorage.setItem('session_id', data.session_id)
  } catch {
    // Session id just won't persist.
  }
}

/** Returned by `usePetPresence` so a caller (the unlock handler in
 * usePetChat) can trigger an out-of-band presence check — same DND/
 * courtesy/busy gates as every scheduled poll, just fired on demand instead
 * of waiting for the next timer tick. */
export interface PetPresenceHandle {
  /**
   * Requests the greeting line (server's `should_greet` freshness gap
   * decides whether the absence was long enough to deserve words — no
   * client-side wordiness heuristic here) and runs one immediate reminders
   * + reach-out poll, so anything queued while locked announces promptly
   * instead of waiting for the next 30s tick. Does NOT touch the mount
   * greeting's once-per-mount `greetedRef` — this can fire again even after
   * the mount greeting already landed, since the server gap-gates repeats
   * (a fresh unlock after a long absence is exactly the kind of gap
   * should_greet is for).
   */
  requestUnlockWakeup: () => void
}

export const usePetPresence = (deps: PetPresenceDeps): PetPresenceHandle => {
  // Effects run on empty dep arrays (they own long-lived timers); the
  // freshest callbacks are read through this ref, same pattern as
  // useHandsFree's optionsRef.
  const depsRef = useRef(deps)
  depsRef.current = deps

  // Return shape distinguishes "skipped because DND" from every other
  // not-sent outcome (busy, network failure, no line due) — the greeting
  // effect below only retries on the DND case; retrying forever on a
  // genuine failure (server down, 404) would turn a one-shot greeting
  // attempt into an infinite 30s poll loop, which is not what the original
  // single-attempt behavior did and not part of this feature's scope.
  const requestLine = async (
    path: string,
    key: string,
    options?: { wakeBeforeApply?: boolean; extraBody?: Record<string, unknown> },
  ): Promise<{ sent: boolean; skippedForDnd: boolean }> => {
    const { isBusyRef, applyLine, isCourtesyActiveRef, onBeforePresenceWake } = depsRef.current
    if (isBusyRef.current) return { sent: false, skippedForDnd: false }
    // DND gate lives here, not per-caller: every presence line (greeting,
    // reminder, reach-out, idle chatter) funnels through requestLine, and
    // reminders/reach-out are exactly-once server pops — fetching and then
    // discarding the reply on the client would silently burn the pop and
    // lose the reminder for good. So the fetch itself must not happen.
    if (isPetDndActive()) return { sent: false, skippedForDnd: true }
    // Tucked away (vrm/petTuck) is the same hard "don't even fetch" rule
    // as DND: she's deliberately parked out of the way, and a fetch here
    // would burn exactly-once server pops on a line nobody should hear.
    // Same return shape, so pending greetings retry after untucking.
    if (isPetTucked()) return { sent: false, skippedForDnd: true }
    // Fullscreen courtesy is the same hard "don't even fetch" rule as DND —
    // composes with it (either alone is enough to go quiet), and reuses the
    // DND-skip return shape so the greeting retry loop below treats a
    // courtesy-active miss exactly like a DND-active one (retry later,
    // don't burn the one-shot greeting).
    if (isCourtesyActiveRef?.current) return { sent: false, skippedForDnd: true }
    try {
      const response = await apiFetch(path, presenceRequest(options?.extraBody))
      if (!response.ok) {
        diagnostics.increment('presence.request_failed', `${path} HTTP ${response.status}`)
        return { sent: false, skippedForDnd: false }
      }
      const data = await response.json()
      adoptSession(data)
      const text = data[key]
      if (typeof text !== 'string' || !text) return { sent: false, skippedForDnd: false }
      // Reminders/reach-out are time-sensitive enough to interrupt a doze —
      // wake her right before the line lands so the line itself can read as
      // "just woke up for this" rather than an already-alert Kurisu
      // suddenly knowing about a reminder she should've been asleep for.
      if (options?.wakeBeforeApply) onBeforePresenceWake?.()
      await applyLine(data, text)
      return { sent: true, skippedForDnd: false }
    } catch (error) {
      // Best-effort, like every presence probe.
      diagnostics.increment('presence.request_failed', `${path} ${String(error)}`)
      return { sent: false, skippedForDnd: false }
    }
  }
  const requestLineRef = useRef(requestLine)
  requestLineRef.current = requestLine

  // Waits (bounded) for a just-delivered line's audio to finish. The pet's
  // lipSync.play resolves when playback STARTS, so back-to-back applyLine
  // calls tear the first clip down mid-sentence — and reminders/reach-outs
  // are exactly-once server pops, so half a sentence is the whole delivery.
  // The cap keeps a stuck isSpeaking() from starving the later line; on
  // expiry we just proceed (the old concurrent behavior, no pop is lost).
  const waitForSpeechEnd = async (capMs: number, isCancelled: () => boolean): Promise<void> => {
    const deadline = Date.now() + capMs
    while (
      !isCancelled() &&
      Date.now() < deadline &&
      kurisuBridge.lipSync?.isSpeaking?.() === true
    ) {
      await new Promise((resolve) => window.setTimeout(resolve, 250))
    }
  }

  // Reminders then reach-out, SERIALIZED: firing both fetches concurrently
  // let the second applyLine cut the first line off mid-sentence whenever
  // both were due on the same tick.
  const deliverDueLines = async (isCancelled: () => boolean): Promise<void> => {
    const reminder = await requestLineRef.current('/api/reminders/due', 'reminder', {
      wakeBeforeApply: true,
    })
    if (isCancelled()) return
    if (reminder.sent) await waitForSpeechEnd(15_000, isCancelled)
    if (isCancelled()) return
    await requestLineRef.current('/api/reachout/due', 'reach_out', { wakeBeforeApply: true })
  }
  const deliverDueLinesRef = useRef(deliverDueLines)
  deliverDueLinesRef.current = deliverDueLines
  const waitForSpeechEndRef = useRef(waitForSpeechEnd)
  waitForSpeechEndRef.current = waitForSpeechEnd

  // Greeting: once per pet-window open, shortly after mount so the avatar
  // and audio pipeline exist. Whether it's actually a fresh session is the
  // server's call (should_greet), and the session handle_lock serializes
  // with a concurrently-opening main window. If DND is active at the
  // scheduled moment, the greeting is not "used up" — it retries on the
  // same 30s cadence as the reminder poll below until DND clears and it
  // actually lands, instead of a pet opened during DND staying permanently
  // un-greeted. Any OTHER not-sent outcome (busy/network/no greeting due)
  // is left alone exactly like before — one attempt, then done.
  const greetedRef = useRef(false)
  useEffect(() => {
    let cancelled = false
    let timer = 0
    const fireGreeting = () => {
      if (cancelled || greetedRef.current) return
      void requestLineRef.current('/api/greeting', 'greeting').then(({ sent, skippedForDnd }) => {
        if (sent) {
          greetedRef.current = true
          return
        }
        if (!cancelled && skippedForDnd) {
          timer = window.setTimeout(attempt, POLL_MS)
        }
      })
    }
    const attempt = () => {
      if (cancelled || greetedRef.current) return
      // The VRM scene (lazy chunk + GLTF fetch/parse) routinely isn't ready
      // by GREETING_DELAY_MS on a cold launch — kurisuBridge.lipSync is
      // still null then, and firing through it anyway silently no-ops the
      // audio while marking the greeting delivered forever. Wait a bit
      // longer for it; if it's already ready this resolves immediately.
      void waitForLipSyncBridge().then(() => {
        if (cancelled || greetedRef.current) return
        if (!kurisuBridge.lipSync) {
          // Timed out with no bridge — still greet text-first (better than
          // nothing), but count the silent-audio drop.
          diagnostics.increment('pet.greeting_before_lipsync')
        }
        fireGreeting()
      })
    }
    timer = window.setTimeout(attempt, GREETING_DELAY_MS)
    return () => {
      cancelled = true
      window.clearTimeout(timer)
    }
  }, [])

  // Heartbeat: stamped once on mount and again on every poll tick below, so
  // the main window's usePresence can tell "pet alive right now" from "pet
  // closed a while ago" via a simple freshness check — no cross-window
  // channel needed beyond localStorage.
  useEffect(() => {
    writePetAliveHeartbeat()
  }, [])

  // Reminders + reach-out: fixed 30s poll (reminders are time-sensitive),
  // plus one immediate check to catch anything that came due while no
  // window was open ("while you were away" phrasing is server-side).
  useEffect(() => {
    let cancelled = false
    const tick = () => {
      if (cancelled) return
      writePetAliveHeartbeat()
      void deliverDueLinesRef.current(() => cancelled)
    }
    const intervalId = window.setInterval(tick, POLL_MS)
    tick()
    return () => {
      cancelled = true
      window.clearInterval(intervalId)
    }
  }, [])

  // Idle chatter: randomized long timer so it never feels scheduled. The
  // server enforces the real cooldown and only chatters mid-session
  // (non-empty short-term window), so a freshly-opened silent pet stays
  // quiet instead of monologuing at an empty room. Skipped outright while
  // dozing — she's asleep, so no unprompted monologue — or while she's
  // already speaking a presence/reply line (same isSpeaking() check the
  // read-together page loop uses — a NEW idle-chatter fetch on top of a
  // clip already playing would talk over it) — but the timer still
  // reschedules underneath either way, so chatter resumes on its own
  // cadence without needing a wake/quiet event to kick this loop back on.
  useEffect(() => {
    let cancelled = false
    let timer = 0
    const schedule = () => {
      // Presence dial: lively shortens the window; quiet is checked at
      // fire time below so flipping the dial applies without a remount.
      const delay =
        (IDLE_MIN_MS + Math.random() * (IDLE_MAX_MS - IDLE_MIN_MS)) *
        (idleChatterDelayMultiplier() ?? 1)
      timer = window.setTimeout(() => {
        if (
          idleChatterDelayMultiplier() === null ||
          depsRef.current.isDozingRef?.current ||
          kurisuBridge.lipSync?.isSpeaking?.() === true
        ) {
          if (!cancelled) schedule()
          return
        }
        void requestLineRef.current('/api/idle', 'chatter').finally(() => {
          if (!cancelled) schedule()
        })
      }, delay)
    }
    schedule()
    return () => {
      cancelled = true
      window.clearTimeout(timer)
    }
  }, [])

  // Ambient desktop awareness (../ambient.ts): the pet owns this loop
  // while it's alive (usePresence's copy stands down via petIsAlive), and
  // every resulting line rides the same requestLine funnel — DND,
  // fullscreen courtesy, and busy gates all apply exactly like idle
  // chatter. Never wakes a doze: an ambient remark isn't worth it.
  useAmbientSignals({
    deliver: async (kind, detail) => {
      await requestLineRef.current('/api/ambient', 'comment', {
        extraBody: { kind, detail },
      })
    },
  })

  // On-demand wakeup (unlock): reuses the exact same gated requestLine as
  // every scheduled poll — DND/courtesy/busy all apply identically, and the
  // greeting call goes through unconditionally (no client-side "was this
  // really a long absence" guess; should_greet already answers that
  // server-side). wakeBeforeApply is passed for the immediate reminders/
  // reach-out poll for the same reason the regular 30s poll passes it: they
  // may land while she's still dozing (the caller wakes her explicitly
  // BEFORE calling this via kurisuBridge.petDoze.wake, but a reminder/
  // reach-out line landing a beat later should still carry its own wake so
  // ordering never depends on the caller doing it right).
  const requestUnlockWakeup = () => {
    // Same serialization as the 30s tick's deliverDueLines: greeting first,
    // then reminders, then reach-out, each waiting out the previous line's
    // audio — three concurrent fetches meant whichever applyLine landed
    // second cut the first clip off mid-sentence.
    void (async () => {
      const greeting = await requestLineRef.current('/api/greeting', 'greeting')
      if (greeting.sent) await waitForSpeechEndRef.current(15_000, () => false)
      await deliverDueLinesRef.current(() => false)
    })()
  }

  return { requestUnlockWakeup }
}
