/**
 * "Presence" state extracted from ChatInterface: every server-initiated,
 * best-effort aside that isn't a direct reply to something the user typed —
 * greeting, farewell, poke reaction, idle chatter, the 30s reminders poll,
 * the morning brief card, milestone celebration banner, the relationship
 * gauge, and conversational-onboarding (M6) chip tracking. All of these
 * share the same shape: POST a small presence endpoint, best-effort catch,
 * and (except onboarding chips / the brief / the gauge) voice the response
 * through the same expression+lip-sync dance — applyPresenceExpressionAndAudio.
 *
 * Deliberately NOT included: llmIsDummy / probeSettings / the first-run
 * onboarding-wizard gate. Those are shared with the send path
 * (ensureRealLlmConfigured) and the Settings panel wiring, so they stay on
 * ChatInterface; this hook receives the pieces it needs (llmIsDummyRef,
 * waitForFirstRunProbe) as inputs instead of owning them.
 */
import { useCallback, useEffect, useRef, useState } from 'react'
import type { Dispatch, MutableRefObject, SetStateAction } from 'react'
import { apiFetch } from '../api'
import { isAudioEnabled } from '../audioPrefs'
import { canNotify, notify } from '../notifications'
import { localDate } from '../dates'
import { kurisuBridge } from '../vrm/kurisuBridge'
import { memoryRequest } from '../memoryApi'
import { idleChatterDelayMultiplier } from '../presenceLevel'
import { presenceRequest } from '../presenceApi'
import { resolveUserId } from './useSessions'
import { useAmbientSignals } from './useAmbientSignals'
import {
  filterUnseen,
  markSeen,
  milestoneClip,
  type MilestoneEvent,
} from '../milestones'
import { diffNewFactKeys, isIntroDone, markIntroDone } from '../introMode'
import type { Message } from '../chat/messages'

export interface PresenceAudioData {
  emotion?: string | null
  emotion_intensity?: number | null
  audio?: string | null
  audio_mime_type?: string | null
}

export interface UsePresenceOptions {
  /** False while the connection screen still overlays the app — greeting
   * and the morning-brief fetch wait for this so the voiced opener plays
   * after the tap that unlocks the AudioContext. */
  entered: boolean
  setMessages: Dispatch<SetStateAction<Message[]>>
  adoptSessionId: (sessionId: string) => void
  /** Live mirrors so async presence callbacks never act on stale state. */
  isLoadingRef: MutableRefObject<boolean>
  isStreamingRef: MutableRefObject<boolean>
  /** Reactive isLoading value (not just the ref) — the intro-mode chip
   * effect keys off the true->false transition the same way the original
   * inline effect did, via a dependency array rather than polling a ref. */
  isLoading: boolean
  /** True once the mount-time /api/settings probe has resolved a real
   * (non-dummy) provider — read via a ref so a greeting fired from a
   * still-resolving probe never races ahead of the answer. */
  llmIsDummyRef: MutableRefObject<boolean>
  /** Resolves once the mount-time first-run probe has updated llmIsDummyRef.
   * Intro-mode activation awaits this before trusting the ref. */
  waitForFirstRunProbe: () => Promise<void>
  /** True while "Her memories" is open — the gauge refetch on close is
   * skipped otherwise (see the memoryOpen-close effect below). */
  memoryOpen: boolean
}

/** Same 30s the reminders poll always used — reminders are time-sensitive,
 * unlike idle chatter's long randomized delay. */
const REMINDERS_POLL_MS = 30000

/** localStorage key usePetPresence stamps with `Date.now()` on mount and on
 * every 30s poll tick — a cheap heartbeat this hook reads to tell "pet window
 * open and alive" from "pet window closed/crashed" without any cross-window
 * channel. Shared contract pinned in tests/test_contracts.py. */
export const PET_ALIVE_KEY = 'kurisu_pet_alive'
/** Generous vs. the pet's own 30s tick — one missed tick shouldn't flip this
 * false, but a genuinely closed pet (no more writes) ages out quickly. */
const PET_ALIVE_FRESH_MS = 90_000

/** True when a pet window's heartbeat was written within the freshness
 * window. Used by the hidden main window to defer to a visible pet instead
 * of competing with it for the server's exactly-once presence pops — a
 * hidden window "winning" a pop delivers content to a window nobody can
 * hear, while the pet gets a clean (already-consumed) null. */
export const petIsAlive = (): boolean => {
  let raw: string | null
  try {
    raw = window.localStorage.getItem(PET_ALIVE_KEY)
  } catch {
    return false
  }
  if (!raw) return false
  const stamp = Number(raw)
  if (!Number.isFinite(stamp)) return false
  return Date.now() - stamp < PET_ALIVE_FRESH_MS
}

export const usePresence = ({
  entered,
  setMessages,
  adoptSessionId,
  isLoadingRef,
  isStreamingRef,
  isLoading,
  llmIsDummyRef,
  waitForFirstRunProbe,
  memoryOpen,
}: UsePresenceOptions) => {
  const greetingRequestedRef = useRef(false)

  /** Shared emotion + lip-sync handling for presence responses (poke
   * reaction, idle chatter, greeting, reminders, farewell) — same
   * expression/audio dance each one does after arriving. */
  const applyPresenceExpressionAndAudio = useCallback(
    async (data: PresenceAudioData) => {
      // Presence responses race the user: the server round-trip (LLM + TTS)
      // takes seconds, and if the user typed/sent in that window — or a
      // reply is streaming or still audibly playing — voicing this line on
      // top of it sounds like two people talking at once. The text (if any)
      // has already been appended by the caller; just stay quiet.
      if (
        isLoadingRef.current ||
        isStreamingRef.current ||
        kurisuBridge.lipSync?.isSpeaking?.() === true
      ) {
        return
      }
      if (data.emotion) {
        const intensity = data.emotion_intensity ?? 0.7
        kurisuBridge.expressions?.set?.(data.emotion, data.emotion_intensity ?? 0.6, 2500)
        // Presence lines drove face + lip-sync but never a body gesture (the
        // interactive turn path is the only other react caller — voiceTurn.ts).
        // The bridge's 14s / max-2-per-turn cooldown keeps this from overusing.
        kurisuBridge.react?.(data.emotion, intensity)
      }
      if (data.audio && isAudioEnabled()) {
        try {
          await kurisuBridge.lipSync?.play?.({
            base64Audio: data.audio,
            mimeType: data.audio_mime_type ?? 'audio/wav',
          })
        } catch {
          // Autoplay may be blocked before the first user gesture.
        }
      }
    },
    [isLoadingRef, isStreamingRef],
  )

  // ---- Milestone celebration (M2) ----------------------------------------

  const [milestoneBanner, setMilestoneBanner] = useState<MilestoneEvent | null>(null)
  const [gaugeCelebrating, setGaugeCelebrating] = useState(false)

  /** The greeting response may carry newly-true events (Nth session,
   * M-month anniversary). Only the first unseen one celebrates per greeting
   * — filterUnseen/markSeen de-dupe across app restarts via localStorage, so
   * a milestone never fires twice. */
  const handleMilestones = useCallback((events: MilestoneEvent[]) => {
    const userId = resolveUserId()
    const unseen = filterUnseen(events, userId)
    if (unseen.length === 0) return
    const [first] = unseen
    markSeen([first], userId)
    setMilestoneBanner(first)
    window.setTimeout(() => {
      kurisuBridge.vrmAnimations?.play?.(milestoneClip(first))
    }, 1500)
    setGaugeCelebrating(true)
    window.setTimeout(() => setGaugeCelebrating(false), 3000)
    window.setTimeout(() => {
      setMilestoneBanner((current) => (current === first ? null : current))
    }, 12000)
  }, [])

  // ---- Farewell -----------------------------------------------------------

  /** The user said goodbye — ask the server for one warm send-off line
   * (anti-dark-pattern: the server never guilts or begs). Returns true when
   * the farewell handled the message; false falls back to a normal turn. */
  const requestFarewell = useCallback(
    async (message: string): Promise<boolean> => {
      try {
        const response = await apiFetch(
          '/api/farewell',
          presenceRequest({ message, reason: 'goodbye' }),
        )
        if (!response.ok) return false
        const data = await response.json()
        if (!data.farewell) return false
        if (data.session_id) adoptSessionId(data.session_id)
        setMessages((prev) => [
          ...prev,
          { role: 'user', content: message, timestamp: Date.now() },
          { role: 'assistant', content: data.farewell, timestamp: Date.now() },
        ])
        await applyPresenceExpressionAndAudio(data)
        return true
      } catch {
        return false
      }
    },
    [adoptSessionId, applyPresenceExpressionAndAudio, setMessages],
  )

  // ---- Morning brief card ---------------------------------------------

  // The greeting speaks the brief's substance; this is the dismissible
  // reference copy. Shown for today's brief only; dismissal is remembered
  // per brief date.
  const [morningBrief, setMorningBrief] = useState<{ date: string; text: string } | null>(null)
  useEffect(() => {
    if (!entered) return
    ;(async () => {
      try {
        const userId = localStorage.getItem('user_id')
        const response = await apiFetch(
          `/api/brief${userId ? `?user_id=${encodeURIComponent(userId)}` : ''}`,
        )
        if (!response.ok) return
        const data = await response.json()
        const brief = data.brief as { date: string; text: string } | null
        if (!brief?.text) return
        // Local date, not toISOString (UTC): the backend stamps brief dates
        // from the LOCAL day, so the UTC date hid the card all morning east
        // of Greenwich.
        const today = localDate()
        if (brief.date !== today) return
        if (localStorage.getItem('kurisu_brief_dismissed') === brief.date) return
        setMorningBrief({ date: brief.date, text: brief.text })
      } catch {
        // Brief card is best-effort.
      }
    })()
  }, [entered])

  const dismissMorningBrief = useCallback(() => {
    setMorningBrief((current) => {
      if (current) {
        try {
          localStorage.setItem('kurisu_brief_dismissed', current.date)
        } catch {
          // Storage failures just mean the card reappears next load.
        }
      }
      return null
    })
  }, [])

  // ---- Relationship gauge -------------------------------------------------

  // Read from the memory overview so the header gauge doesn't need its own
  // endpoint. Best-effort — a dummy provider or long-term-memory-off backend
  // just leaves the gauge hidden (relationship stays null). Returns the
  // parsed `overview` object (or null on any failure) so the entry effect
  // can also feed it to maybeActivateIntro without firing a second request.
  const [relationshipTrust, setRelationshipTrust] = useState<number | null>(null)
  const [relationshipSessions, setRelationshipSessions] = useState<number | null>(null)
  const refreshRelationship = useCallback(async (): Promise<unknown | null> => {
    try {
      const response = await memoryRequest('overview')
      if (!response.ok) return null
      const data = await response.json()
      const overview = data?.overview
      const trust = overview?.relationship?.trust_level
      const sessionCount = overview?.profile?.session_count
      setRelationshipTrust(
        typeof trust === 'number' && Number.isFinite(trust) && trust >= 0 && trust <= 1 ? trust : null,
      )
      setRelationshipSessions(typeof sessionCount === 'number' ? sessionCount : null)
      return overview ?? null
    } catch {
      // Best-effort; the gauge just stays at its previous value.
      return null
    }
  }, [])

  // ---- Conversational onboarding (M6) ------------------------------------

  const [introActive, setIntroActive] = useState(false)
  const [synapseChips, setSynapseChips] = useState<Array<{ key: string; value: string; id: number }>>([])
  const introKnownFactsRef = useRef<Record<string, string>>({})
  const introFactCountRef = useRef(0)
  const introTurnCountRef = useRef(0)
  const introTimeoutRef = useRef<number | null>(null)
  const chipIdRef = useRef(0)

  // Brand-new profile check + chip seeding, run right after the
  // entered-overview fetch so it shares the same request instead of firing
  // a second one. Awaits waitForFirstRunProbe (the mount-time /api/settings
  // probe) before trusting llmIsDummyRef, so a dummy-provider profile that
  // hasn't resolved yet never races into activating intro mode.
  const maybeActivateIntro = useCallback(
    async (overview: unknown) => {
      await waitForFirstRunProbe()
      if (isIntroDone(resolveUserId()) || llmIsDummyRef.current) return
      const record = overview as {
        profile?: { session_count?: unknown; facts?: unknown }
        facts?: unknown
        fact_records?: unknown
      }
      const sessionCount = record?.profile?.session_count
      const profileFacts =
        record?.profile?.facts && typeof record.profile.facts === 'object'
          ? (record.profile.facts as Record<string, unknown>)
          : {}
      const factCount = Array.isArray(record?.fact_records)
        ? record.fact_records.length
        : Array.isArray(record?.facts)
          ? record.facts.length
          : 0
      const isFresh =
        (typeof sessionCount !== 'number' || sessionCount <= 1) &&
        Object.keys(profileFacts).length === 0 &&
        factCount === 0
      if (!isFresh) return
      introKnownFactsRef.current = {}
      introFactCountRef.current = 0
      introTurnCountRef.current = 0
      setIntroActive(true)
    },
    [llmIsDummyRef, waitForFirstRunProbe],
  )

  // One overview fetch on entry, shared by the gauge and intro-mode
  // activation: refreshRelationship both updates the gauge state and
  // returns the parsed overview, which maybeActivateIntro consumes
  // directly.
  useEffect(() => {
    if (!entered) return
    void (async () => {
      const overview = await refreshRelationship()
      await maybeActivateIntro(overview)
    })()
  }, [entered, refreshRelationship, maybeActivateIntro])

  // Relationship gauge: refetch whenever "Her memories" closes (a pin/edit/
  // reset in there can change trust/session count immediately). Skipped on
  // the very first render (memoryOpen starts false, so a naive `!memoryOpen`
  // effect would double-fire on mount alongside the `entered` effect above)
  // — it only matters once the panel has actually been opened.
  const memoryEverOpenedRef = useRef(false)
  useEffect(() => {
    if (memoryOpen) {
      memoryEverOpenedRef.current = true
      return
    }
    if (!memoryEverOpenedRef.current) return
    void refreshRelationship()
  }, [memoryOpen, refreshRelationship])

  // On each isLoading true→false transition while intro mode is active,
  // wait 4s (the summarizer/profile-extractor runs async after the turn
  // commits) then refetch the overview; any profile-fact key not seen
  // before becomes a chip. Never polls while the tab is hidden. After 3
  // total facts or 12 completed turns, the mode retires for good.
  const wasLoadingRef = useRef(false)
  useEffect(() => {
    if (!introActive) {
      wasLoadingRef.current = isLoading
      return
    }
    const justFinished = wasLoadingRef.current && !isLoading
    wasLoadingRef.current = isLoading
    if (!justFinished) return
    introTurnCountRef.current += 1
    if (introTimeoutRef.current !== null) window.clearTimeout(introTimeoutRef.current)
    introTimeoutRef.current = window.setTimeout(() => {
      introTimeoutRef.current = null
      if (typeof document !== 'undefined' && document.hidden) return
      ;(async () => {
        try {
          const response = await memoryRequest('overview')
          if (!response.ok) return
          const data = await response.json()
          const profileFacts = data?.overview?.profile?.facts
          const nextFacts: Record<string, string> =
            profileFacts && typeof profileFacts === 'object'
              ? Object.fromEntries(
                  Object.entries(profileFacts as Record<string, unknown>).filter(
                    (entry): entry is [string, string] => typeof entry[1] === 'string',
                  ),
                )
              : {}
          const newKeys = diffNewFactKeys(introKnownFactsRef.current, nextFacts)
          introKnownFactsRef.current = nextFacts
          if (newKeys.length > 0) {
            introFactCountRef.current += newKeys.length
            const newChips = newKeys.map((key) => ({
              key,
              value: nextFacts[key],
              id: (chipIdRef.current += 1),
            }))
            setSynapseChips((prev) => [...prev, ...newChips])
            newChips.forEach((chip) => {
              window.setTimeout(() => {
                setSynapseChips((prev) => prev.filter((item) => item.id !== chip.id))
              }, 8000)
            })
          }
          if (introFactCountRef.current >= 3 || introTurnCountRef.current >= 12) {
            markIntroDone(resolveUserId())
            setIntroActive(false)
            const finalId = (chipIdRef.current += 1)
            setSynapseChips((prev) => [
              ...prev,
              { key: '__final__', value: 'her memory of you has begun — ❖ to view', id: finalId },
            ])
            window.setTimeout(() => {
              setSynapseChips((prev) => prev.filter((item) => item.id !== finalId))
            }, 8000)
          }
        } catch {
          // Best-effort; the chip just doesn't appear this turn.
        }
      })()
    }, 4000)
  }, [isLoading, introActive])

  // Clear the pending intro-mode timeout on unmount so it never fires (and
  // touches state) after the component is gone.
  useEffect(() => {
    return () => {
      if (introTimeoutRef.current !== null) window.clearTimeout(introTimeoutRef.current)
    }
  }, [])

  // ---- Greeting -------------------------------------------------------

  /** Fires the actual /api/greeting request — no guards here beyond "is the
   * provider real"; callers are responsible for the once-only/in-flight
   * checks (see tryRequestGreeting) so greetingRequestedRef stays the single
   * source of truth for "has a request actually fired". */
  const requestGreeting = useCallback(async () => {
    try {
      const response = await apiFetch('/api/greeting', presenceRequest())
      if (!response.ok) return
      const data = await response.json()
      if (data.session_id) adoptSessionId(data.session_id)
      if (data.greeting) {
        setMessages((prev) => [
          ...prev,
          { role: 'assistant', content: data.greeting, timestamp: Date.now() },
        ])
        await applyPresenceExpressionAndAudio(data)
      }
      handleMilestones(Array.isArray(data.milestones) ? data.milestones : [])
    } catch {
      // Greeting is best-effort; silence any failure.
    }
  }, [adoptSessionId, applyPresenceExpressionAndAudio, handleMilestones, setMessages])

  /** Single gate for firing a greeting: greetingRequestedRef is set true
   * ONLY here, and ONLY once every guard has passed — so re-running this
   * from a different call site (mount effect, post-onboarding regreet) can
   * never double-fire just because llmIsDummy changed identity. Never fires
   * over an in-flight turn (a greeting racing a real reply reads as two
   * messages landing at once). */
  const tryRequestGreeting = useCallback(() => {
    if (greetingRequestedRef.current) return
    if (llmIsDummyRef.current) return
    if (isLoadingRef.current || isStreamingRef.current) return
    greetingRequestedRef.current = true
    void requestGreeting()
  }, [isLoadingRef, isStreamingRef, llmIsDummyRef, requestGreeting])

  // Ask the server whether Kurisu wants to open the session herself (one
  // short greeting; server-gated to fresh sessions only). Waits for the
  // connection screen's tap so the greeting can be voiced.
  //
  // Visibility-guarded like every other presence check in this file (idle
  // chatter, screen-peek, reminders, reach-out, daily nudges) EXCEPT this
  // one used to have no guard at all — the launch race: a pet-default Tauri
  // launch auto-connects the main window (entered flips true) milliseconds
  // before it's hidden, and the hidden main window would win the server's
  // exactly-once greeting pop, leaving the audible pet with a clean null.
  // Skips outright while hidden with a live pet (petIsAlive) — the pet is
  // the one place a greeting can actually be heard then. Hidden with no pet
  // alive still greets on entry (nothing else would ever hear it either);
  // becoming visible later fires the attempt once via visibilitychange.
  useEffect(() => {
    if (!entered) return
    if (document.hidden && petIsAlive()) return
    if (document.hidden) return
    tryRequestGreeting()
  }, [entered, tryRequestGreeting])

  useEffect(() => {
    if (!entered) return
    const handleVisibilityChange = () => {
      if (document.hidden) return
      tryRequestGreeting()
    }
    document.addEventListener('visibilitychange', handleVisibilityChange)
    return () => {
      document.removeEventListener('visibilitychange', handleVisibilityChange)
    }
  }, [entered, tryRequestGreeting])

  // ---- Poke -----------------------------------------------------------

  // VRMViewer detects a click directly on the avatar (as opposed to a
  // drag-to-orbit or a click on empty space/environment) and calls this via
  // the kurisuBridge.poke registry. Reaction is a short spoken/animated
  // aside near the avatar — never appended to the chat transcript, and the
  // server deliberately doesn't commit it to conversation memory either.
  // Most pokes resolve to a null reaction (feature off by default, or the
  // server-side cooldown hasn't elapsed) and should do nothing visible.
  useEffect(() => {
    let pokeInFlight = false
    kurisuBridge.poke = (region?: string) => {
      if (pokeInFlight) return
      pokeInFlight = true
      ;(async () => {
        try {
          const response = await apiFetch('/api/poke', presenceRequest({ region: region ?? null }))
          if (!response.ok) return
          const data = await response.json()
          if (data.session_id) adoptSessionId(data.session_id)
          if (!data.reaction) return
          await applyPresenceExpressionAndAudio(data)
        } catch {
          // Poke reaction is best-effort; silence any failure.
        } finally {
          pokeInFlight = false
        }
      })()
    }
    return () => {
      kurisuBridge.poke = null
    }
  }, [adoptSessionId, applyPresenceExpressionAndAudio])

  // ---- Idle chatter -----------------------------------------------------

  // After a stretch of user inactivity (typing/sending reset the timer),
  // ask the server if Kurisu wants to riff on the conversation unprompted.
  // Randomized 4-7 minute delay so it doesn't feel mechanical; only runs
  // while the tab is visible and the user isn't mid-conversation. Unlike
  // poke, a non-null response IS a normal transcript message (the server
  // already commits it to memory).
  const idleTimeoutRef = useRef<number | null>(null)

  useEffect(() => {
    // Presence dial: quiet suppresses unprompted chatter outright (checked
    // at fire time so flipping the dial applies without a remount), lively
    // shortens the 4-7 minute window.
    const randomDelayMs = () =>
      (4 * 60 + Math.random() * 3 * 60) * 1000 * (idleChatterDelayMultiplier() ?? 1)

    const fireIdleChatter = async () => {
      try {
        if (idleChatterDelayMultiplier() === null) return
        if (document.hidden || isLoadingRef.current || isStreamingRef.current) return
        const response = await apiFetch('/api/idle', presenceRequest())
        if (!response.ok) return
        const data = await response.json()
        if (data.session_id) adoptSessionId(data.session_id)
        if (!data.chatter) return
        setMessages((prev) => [
          ...prev,
          { role: 'assistant', content: data.chatter, timestamp: Date.now() },
        ])
        await applyPresenceExpressionAndAudio(data)
      } catch {
        // Idle chatter is best-effort; silence any failure.
      }
    }

    const scheduleIdleTimer = () => {
      if (idleTimeoutRef.current !== null) {
        window.clearTimeout(idleTimeoutRef.current)
      }
      idleTimeoutRef.current = window.setTimeout(() => {
        void (async () => {
          await fireIdleChatter()
          scheduleIdleTimer()
        })()
      }, randomDelayMs())
    }

    const resetIdleTimer = () => {
      if (document.hidden) return
      scheduleIdleTimer()
    }

    const handleVisibilityChange = () => {
      if (document.hidden) {
        if (idleTimeoutRef.current !== null) {
          window.clearTimeout(idleTimeoutRef.current)
          idleTimeoutRef.current = null
        }
      } else {
        scheduleIdleTimer()
      }
    }

    scheduleIdleTimer()
    document.addEventListener('visibilitychange', handleVisibilityChange)
    window.addEventListener('kurisu:activity', resetIdleTimer)

    return () => {
      if (idleTimeoutRef.current !== null) {
        window.clearTimeout(idleTimeoutRef.current)
        idleTimeoutRef.current = null
      }
      document.removeEventListener('visibilitychange', handleVisibilityChange)
      window.removeEventListener('kurisu:activity', resetIdleTimer)
    }
  }, [adoptSessionId, applyPresenceExpressionAndAudio, isLoadingRef, isStreamingRef, setMessages])

  // ---- Ambient desktop awareness ------------------------------------------

  // Tauri-only zero-cost signals (focused app, battery, now-playing) via
  // useAmbientSignals; the pure transition rules live in ../ambient.ts.
  // The PET owns this loop while alive — this main-window copy stands down
  // via petIsAlive so the two windows never race the server's ambient
  // cooldown. Voice-only delivery, like poke reactions: an ambient remark
  // is throwaway flavor, not transcript material (the server never commits
  // it either).
  useAmbientSignals({
    enabled: () => !petIsAlive(),
    deliver: async (kind, detail) => {
      if (isLoadingRef.current || isStreamingRef.current) return
      if (kurisuBridge.lipSync?.isSpeaking?.() === true) return
      try {
        const response = await apiFetch('/api/ambient', presenceRequest({ kind, detail }))
        if (!response.ok) return
        const data = await response.json()
        if (data.session_id) adoptSessionId(data.session_id)
        if (!data.comment) return
        await applyPresenceExpressionAndAudio(data)
      } catch {
        // Best-effort, like every presence probe.
      }
    },
  })

  // ---- Screen companion ---------------------------------------------------

  // Opt-in (shared kurisu_screen_companion flag, toggled from the pet): every
  // few minutes she glances at the screen and reacts with one line. Ported
  // from usePetChat's peek loop — the main window never ran it, so a
  // main-window-only user got no screen awareness. Same capture path
  // (images.captureScreen: Tauri command or getDisplayMedia), same randomized
  // timer, same mid-turn/mid-speech suppression. The server enforces its own
  // cooldown and never stores the image or the comment (docs/companion-
  // ethics.md). Reads the flag live per mount; a change takes effect on the
  // next entry rather than mid-session, matching how the pet reads it.
  useEffect(() => {
    if (localStorage.getItem('kurisu_screen_companion') !== 'true') return
    let cancelled = false
    let timer = 0
    const PEEK_BASE_MS = 6 * 60_000
    const PEEK_JITTER_MS = 90_000
    const peek = async () => {
      if (
        cancelled ||
        document.hidden ||
        isLoadingRef.current ||
        isStreamingRef.current ||
        kurisuBridge.lipSync?.isSpeaking?.() === true
      ) {
        return
      }
      try {
        const { captureScreen } = await import('../images')
        const image = await captureScreen()
        if (!image || cancelled) return
        const response = await apiFetch(
          '/api/screen_peek',
          presenceRequest({ image: { mime: image.mime, data: image.data } }),
        )
        if (!response.ok) return
        const data = await response.json()
        if (data.session_id) adoptSessionId(data.session_id)
        if (!data.comment || cancelled) return
        setMessages((prev) => [
          ...prev,
          { role: 'assistant', content: data.comment, timestamp: Date.now() },
        ])
        await applyPresenceExpressionAndAudio(data)
      } catch (error) {
        // Ambient flavor; stays quiet in the UI — but not the console (an
        // ACL-refused capture looks identical to "nothing to say" otherwise).
        console.warn('[presence] screen glance failed:', error)
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
  }, [adoptSessionId, applyPresenceExpressionAndAudio, isLoadingRef, isStreamingRef, setMessages])

  // ---- Reminders poll -----------------------------------------------------

  // A fixed 30s poll (unlike idle chatter's random long timer — reminders
  // are time-sensitive). The server pops due reminders exactly once, so
  // overlapping tabs can't double-announce; the line rides the same
  // presence audio path as greeting/poke/idle. Paused while hidden —
  // catch-up phrasing covers what fired in the background.
  useEffect(() => {
    let cancelled = false
    const checkReminders = async () => {
      if (cancelled || isLoadingRef.current || isStreamingRef.current) return
      const hidden = document.hidden
      // A hidden main window with a live pet defers entirely — the pet is
      // visible and polling the same exactly-once endpoint, and a hidden
      // window "winning" the pop leaves the pet with nothing to say while
      // nobody heard the main window's copy either. Hidden with no pet
      // alive is unchanged: notifications are a real delivery path then.
      if (hidden && petIsAlive()) return
      // Hidden windows normally skip the poll (catch-up phrasing covers what
      // fired meanwhile) — unless opt-in notifications can deliver it now.
      if (hidden && !canNotify('reminder')) return
      try {
        const response = await apiFetch('/api/reminders/due', presenceRequest())
        if (!response.ok) return
        const data = await response.json()
        if (data.session_id) adoptSessionId(data.session_id)
        if (!data.reminder || cancelled) return
        setMessages((prev) => [
          ...prev,
          { role: 'assistant', content: data.reminder, timestamp: Date.now() },
        ])
        if (hidden) {
          // Notification instead of voice — she doesn't talk to an empty room.
          notify('reminder', 'Kurisu', data.reminder)
        } else {
          await applyPresenceExpressionAndAudio(data)
        }
      } catch {
        // Best-effort, like the other presence probes.
      }
    }
    // Proactive reach-out: at most one server-generated line per day, popped
    // exactly once (same contract as reminders). Visible → she says it out
    // loud; hidden → the line itself is the notification (unlike the brief/
    // diary "it exists" nudges, the content IS short and hers to say).
    const checkReachOut = async () => {
      if (cancelled || isLoadingRef.current || isStreamingRef.current) return
      const hidden = document.hidden
      // Same pet-deference rule as checkReminders above.
      if (hidden && petIsAlive()) return
      if (hidden && !canNotify('reachout')) return
      try {
        const response = await apiFetch('/api/reachout/due', presenceRequest())
        if (!response.ok) return
        const data = await response.json()
        if (data.session_id) adoptSessionId(data.session_id)
        if (!data.reach_out || cancelled) return
        setMessages((prev) => [
          ...prev,
          { role: 'assistant', content: data.reach_out, timestamp: Date.now() },
        ])
        if (hidden) {
          notify('reachout', 'Kurisu', data.reach_out)
        } else {
          await applyPresenceExpressionAndAudio(data)
        }
      } catch {
        // Best-effort, like the other presence probes.
      }
    }
    // Once-daily "it exists" nudges for the brief and her diary, only while
    // hidden and only when the daily cap in canNotify still allows it (so
    // these usually cost zero network). Content stays in the app.
    const checkDailyNudges = async () => {
      if (cancelled || !document.hidden) return
      const userId = localStorage.getItem('user_id')
      const query = userId ? `?user_id=${encodeURIComponent(userId)}` : ''
      const todayStamp = localDate()
      if (canNotify('brief')) {
        try {
          const response = await apiFetch(`/api/brief${query}`)
          const data = response.ok ? await response.json() : null
          if (data?.brief?.text && data.brief.date === todayStamp) {
            notify('brief', 'Kurisu', 'Your morning brief is ready.')
          }
        } catch {
          // Best-effort.
        }
      }
      if (canNotify('diary')) {
        try {
          const response = await apiFetch(`/api/diary${query}`)
          const data = response.ok ? await response.json() : null
          if (data?.diary?.text && data.diary.date === todayStamp) {
            notify('diary', 'Kurisu', 'She wrote something in her diary today.')
          }
        } catch {
          // Best-effort.
        }
      }
    }
    const intervalId = window.setInterval(() => {
      void checkReminders()
      void checkReachOut()
      void checkDailyNudges()
    }, REMINDERS_POLL_MS)
    // One immediate check on mount picks up reminders that came due while
    // the app was closed ("while you were away" phrasing server-side) and
    // any reach-out waiting from earlier today — she brings it up as you
    // sit down rather than three ticks later.
    void checkReminders()
    void checkReachOut()
    return () => {
      cancelled = true
      window.clearInterval(intervalId)
    }
  }, [adoptSessionId, applyPresenceExpressionAndAudio, isLoadingRef, isStreamingRef, setMessages])

  return {
    applyPresenceExpressionAndAudio,
    requestFarewell,
    tryRequestGreeting,
    greetingRequestedRef,
    morningBrief,
    dismissMorningBrief,
    relationshipTrust,
    relationshipSessions,
    refreshRelationship,
    milestoneBanner,
    setMilestoneBanner,
    gaugeCelebrating,
    introActive,
    synapseChips,
  }
}
