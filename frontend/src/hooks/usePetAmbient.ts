/**
 * Ambient pet-window behaviors that don't need a transcript or a chat turn:
 * fullscreen courtesy (hide/mute while another app is foreground-fullscreen)
 * and the screen-companion peek loop (occasional glance-and-comment on the
 * screen). Split out of usePetChat, which still owns the doze/DND/read-
 * together state these blocks read through refs.
 */
import { useEffect, useRef, useState } from 'react'
import { apiFetch } from '../api'
import { diagnostics } from '../diagnostics'
import { presenceRequest } from '../presenceApi'
import { isTauri } from '../platform'
import { kurisuBridge } from '../vrm/kurisuBridge'
import { isFullscreenCourtesyEnabled, isPetDndActive, type PresencePayload } from './usePetPresence'

// Fullscreen courtesy: how often we poll the Tauri command for "is some
// other app foreground-fullscreen right now". 5s balances snappiness
// (hide quickly when a game launches) against poll overhead — this fires
// for the entire pet-window lifetime whenever Tauri + the opt-out allow it.
const FULLSCREEN_POLL_MS = 5_000
// After this many consecutive invoke failures (older shell without the
// command, or any other rejection), stop polling for the rest of this
// window's life instead of retrying forever against a command that will
// never succeed.
const FULLSCREEN_MAX_CONSECUTIVE_FAILURES = 3

export interface UsePetAmbientDeps {
  /** True while a chat turn is in flight — ambient behaviors stay quiet then. */
  isBusyRef: React.RefObject<boolean>
  /** True while she's dozing (kurisuBridge.petDoze.active()). */
  isDozingRef: React.RefObject<boolean>
  /** True while read-together mode is active — pauses the screen-companion
   * peek loop (a redundant second "what's on screen" capture+comment). */
  readTogetherActiveRef: React.RefObject<boolean>
  watchTogetherActiveRef: React.RefObject<boolean>
  /** Shared tail for every unprompted line: expression + audio + caption. */
  applyPresenceLine: (data: PresencePayload, text: string) => void | Promise<void>
}

export interface UsePetAmbientResult {
  /**
   * Fullscreen-courtesy active flag: read by usePetPresence to hard-mute
   * every presence fetch (see the effect below that actually polls the
   * Tauri command and hides/shows the window). A ref, not state — this
   * needs to be read synchronously inside the presence effect's fetch gate
   * without forcing that effect to re-subscribe on every courtesy flip.
   */
  isCourtesyActiveRef: React.RefObject<boolean>
  screenCompanion: boolean
  toggleScreenCompanion: () => void
  /** Timestamp of the last successful screen CAPTURE (not the last comment). */
  lastPeekAt: number | null
}

/**
 * `usePetAmbient` owns two self-contained ambient blocks:
 *
 * 1. Fullscreen courtesy (Tauri only): when another app goes foreground-
 *    fullscreen (game/movie/presentation), get out of the way — hide the
 *    window and go quiet — and come back once it ends. The Tauri side
 *    (`is_foreground_fullscreen`) already excludes our own windows and the
 *    desktop shell, and returns false on any failure, so this effect only
 *    has to react to the boolean.
 *
 *    A hidden Tauri window's JS keeps running (unlike a closed one), which
 *    is exactly what makes this poll-while-hidden approach work: the same
 *    effect that hides the window is still alive to notice fullscreen end
 *    and call show() again.
 *
 * 2. Screen companion (opt-in, kurisu_screen_companion): every few minutes
 *    she glances at the screen and reacts with one line. The server
 *    enforces its own cooldown and never stores the image or the comment;
 *    the ethics rules live in docs/companion-ethics.md.
 */
export const usePetAmbient = (deps: UsePetAmbientDeps): UsePetAmbientResult => {
  const { isBusyRef, isDozingRef, readTogetherActiveRef, watchTogetherActiveRef, applyPresenceLine } =
    deps

  // Fullscreen-courtesy active flag: read by usePetPresence to hard-mute
  // every presence fetch (see the effect further down that actually polls
  // the Tauri command and hides/shows the window). A ref, not state — this
  // needs to be read synchronously inside the presence effect's fetch gate
  // without forcing that effect to re-subscribe on every courtesy flip.
  const isCourtesyActiveRef = useRef(false)

  const [courtesyActive, setCourtesyActive] = useState(false)
  useEffect(() => {
    isCourtesyActiveRef.current = courtesyActive
  }, [courtesyActive])
  useEffect(() => {
    if (!isTauri()) return
    let cancelled = false
    let consecutiveFailures = 0
    let disabled = false
    const poll = async () => {
      if (cancelled || disabled) return
      // Opt-out is re-checked every tick (not just at mount) so flipping
      // the Settings checkbox off immediately stops future hides — an
      // already-hidden window still needs its own restore, handled by the
      // early "was active, now disabled" branch below.
      if (!isFullscreenCourtesyEnabled()) {
        if (courtesyActive) {
          try {
            const { getCurrentWindow } = await import('@tauri-apps/api/window')
            await getCurrentWindow().show()
          } catch {
            // Best-effort restore; nothing more we can do from here.
          }
          if (!cancelled) setCourtesyActive(false)
        }
        return
      }
      try {
        const { invoke } = await import('@tauri-apps/api/core')
        const fullscreen = await invoke<boolean>('is_foreground_fullscreen')
        consecutiveFailures = 0
        if (cancelled) return
        if (fullscreen && !courtesyActive) {
          // Don't hide mid-turn or mid-sentence: cutting her off to make
          // way for a game is fine, but vanishing silently while she's
          // actively speaking reads as broken, not courteous. Simplest
          // correct rule (per spec): skip this cycle entirely and let the
          // next ~5s poll retry — she'll usually be done well before then.
          if (isBusyRef.current || kurisuBridge.lipSync?.isSpeaking?.() === true) return
          const { getCurrentWindow } = await import('@tauri-apps/api/window')
          await getCurrentWindow().hide()
          if (!cancelled) setCourtesyActive(true)
        } else if (!fullscreen && courtesyActive) {
          const { getCurrentWindow } = await import('@tauri-apps/api/window')
          await getCurrentWindow().show()
          if (!cancelled) setCourtesyActive(false)
        }
      } catch (error) {
        // Covers both the invoke rejecting (older shell without the
        // command) and the dynamic window-API import failing. Either way,
        // count it toward the disable threshold below.
        consecutiveFailures += 1
        diagnostics.increment('pet.courtesy_poll_invoke_failed', String(error))
        if (consecutiveFailures >= FULLSCREEN_MAX_CONSECUTIVE_FAILURES) {
          disabled = true
          diagnostics.increment(
            'pet.courtesy_poll_disabled',
            `after ${consecutiveFailures} consecutive invoke failures — recovery requires reopening the pet window`,
          )
        }
      }
    }
    void poll()
    const intervalId = window.setInterval(() => void poll(), FULLSCREEN_POLL_MS)
    return () => {
      cancelled = true
      window.clearInterval(intervalId)
    }
    // courtesyActive is read via closure each tick (needed to decide
    // hide-vs-show) — re-running the effect on every flip is intentional
    // here (unlike the ref-based gates elsewhere) since this effect's OWN
    // job is to own that piece of state, not just read it.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [courtesyActive])

  // Screen companion (opt-in, kurisu_screen_companion): every few minutes
  // she glances at the screen and reacts with one line. The server enforces
  // its own cooldown and never stores the image or the comment; the ethics
  // rules live in docs/companion-ethics.md.
  const [screenCompanion, setScreenCompanion] = useState(
    () => localStorage.getItem('kurisu_screen_companion') === 'true',
  )
  const toggleScreenCompanion = () => {
    setScreenCompanion((prev) => {
      const next = !prev
      try {
        localStorage.setItem('kurisu_screen_companion', next ? 'true' : 'false')
      } catch {
        // Preference just won't stick.
      }
      return next
    })
  }
  // Timestamp of the last successful screen CAPTURE (not the last comment).
  // Deliberately recorded at capture time, not "she actually said something"
  // time: capture happening at all is the honest signal that the screen
  // companion is alive and looking, regardless of whether the server found
  // anything worth a line. PetView pulses the Screen button off this.
  const [lastPeekAt, setLastPeekAt] = useState<number | null>(null)
  useEffect(() => {
    if (!screenCompanion) return
    let cancelled = false
    let timer = 0
    const PEEK_BASE_MS = 6 * 60_000
    const PEEK_JITTER_MS = 90_000
    const peek = async () => {
      // DND, fullscreen courtesy, and doze all cover screen peeks too —
      // they're an unprompted line/glance just like greeting/idle/reach-out,
      // and the capture itself has no server-side pop to lose, so it's
      // simplest to just skip the whole cycle. Doze specifically: she's
      // asleep, so no glances at the screen either. Read-together and
      // watch-together also pause this: each is a second, redundant
      // capture+LLM-comment loop aimed at the exact same "what's on
      // screen" question, and an explicit shared activity supersedes
      // ambient screen glances while it's active.
      if (
        cancelled ||
        isBusyRef.current ||
        isPetDndActive() ||
        isCourtesyActiveRef.current ||
        isDozingRef.current ||
        readTogetherActiveRef.current ||
        watchTogetherActiveRef.current
      )
        return
      try {
        const { captureScreen } = await import('../images')
        const image = await captureScreen()
        if (!image || cancelled) return
        setLastPeekAt(Date.now())
        const response = await apiFetch(
          '/api/screen_peek',
          presenceRequest({ image: { mime: image.mime, data: image.data } }),
        )
        if (!response.ok) return
        const data = await response.json()
        if (!data.comment || cancelled) return
        await applyPresenceLine(data, data.comment)
      } catch (error) {
        // Ambient flavor; stays quiet in the UI — but not in the console:
        // an ACL-refused capture_screenshot invoke looked identical to
        // "feature working, nothing to say" for a whole release.
        console.warn('[pet] screen glance failed:', error)
        diagnostics.increment('pet.screen_peek_failed', String(error))
      }
    }
    const schedule = () => {
      const jitter = (Math.random() * 2 - 1) * PEEK_JITTER_MS
      timer = window.setTimeout(() => {
        void peek().finally(() => {
          if (!cancelled) schedule()
        })
      }, PEEK_BASE_MS + jitter)
    }
    schedule()
    return () => {
      cancelled = true
      window.clearTimeout(timer)
    }
  }, [
    screenCompanion,
    applyPresenceLine,
    isBusyRef,
    isDozingRef,
    readTogetherActiveRef,
    watchTogetherActiveRef,
  ])

  return {
    isCourtesyActiveRef,
    screenCompanion,
    toggleScreenCompanion,
    lastPeekAt,
  }
}
