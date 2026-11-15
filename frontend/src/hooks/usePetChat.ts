/**
 * Minimal voice-first chat loop for the desktop-pet window.
 *
 * The pet webview is a separate browser context from the main window: it
 * has its own kurisuBridge/lipSync singletons, so it must run its own WS
 * turn to animate its own avatar. Reuses the exact same building blocks as
 * ChatInterface — the shared chat/voiceTurn core, buildChatPayload (via
 * runVoiceTurn), useMic, useHandsFree, the presence endpoints — minus the
 * transcript UI. Session/user ids ride the shared same-origin localStorage,
 * so the conversation is the SAME one the main window shows when reopened.
 */
import { useCallback, useEffect, useRef, useState } from 'react'
import { apiFetch } from '../api'
import { isAudioEnabled } from '../audioPrefs'
import { describeStreamFailure } from '../chat/errors'
import { appendLiveReply, petAckTurnId, resolveTurnImages, stripInterruptedMarker } from '../chat/petTurn'
import { runVoiceTurn } from '../chat/voiceTurn'
import { diagnostics } from '../diagnostics'
import { isTauri } from '../platform'
import { presenceRequest } from '../presenceApi'
import { type StreamChatHandle } from '../streaming'
import { acknowledgment } from '../vrm/acknowledgment'
import { kurisuBridge } from '../vrm/kurisuBridge'
import { isRoamMotionRecent } from '../vrm/petRoam'
import { useFocusMode } from './useFocusMode'
import { useHandsFree } from './useHandsFree'
import { useMic } from './useMic'
import { isPetDndActive, usePetPresence, type PresencePayload } from './usePetPresence'
import { usePetAmbient } from './usePetAmbient'
import { usePetRoam } from './usePetRoam'
import { useReadTogether } from './useReadTogether'
import { resolveUserId } from './useSessions'
import { useVoiceHonestyHint } from './useVoiceHonestyHint'
import { useWatchTogether } from './useWatchTogether'

/**
 * `kurisuBridge.petDoze` is registered by a sibling agent (owns vrm/**,
 * kurisuBridge.ts is off-limits to this file); its type isn't threaded
 * onto KurisuBridge from here. Read it through this narrow, defensively-
 * typed accessor instead of widening the shared bridge type — every call
 * site already optional-chains through it, so a not-yet-landed (or
 * removed/renamed) slot degrades to "no doze integration", never a crash.
 */
interface PetDozeSlot {
  active: () => boolean
  /** Returns true iff she was actually dozing (i.e. this call woke her). */
  wake: (reason?: string) => boolean
}
const getPetDoze = (): PetDozeSlot | null | undefined =>
  (kurisuBridge as unknown as { petDoze?: PetDozeSlot | null }).petDoze

// Drag-gesture debounce: a burst of window-move events settles into ONE
// reaction after this much quiet, so dragging across the screen doesn't
// fire a poke per pixel.
const DRAG_SETTLE_MS = 600
// Cooldown shared by drag reactions specifically (poke-by-click has no
// cooldown of its own — clicking is already a deliberate, rate-limiting
// act; a dragged/click-through region string riding the same endpoint
// without one would let an accidental window nudge spam lines).
const DRAG_REACTION_COOLDOWN_MS = 2 * 60_000
// Welcome-back perk-up (session-unlocked event): at most one nonverbal
// reaction per this window, so a string of quick lock/unlock cycles doesn't
// spam the gesture. Whether she SPEAKS is unthrottled — that's entirely the
// server's should_greet freshness gap, not this client-side cap.
const PERK_UP_THROTTLE_MS = 5 * 60_000

export const usePetChat = (textareaRef: React.RefObject<HTMLTextAreaElement | null>) => {
  const [isBusy, setIsBusy] = useState(false)
  const [lastReply, setLastReply] = useState<string | null>(null)
  // Echo of the last message the user got through (voice transcript or
  // typed) — the pet has no transcript, so without this there is zero
  // feedback that a message was actually sent. Fresh object per send so
  // repeating the same words still re-triggers the echo.
  const [lastSent, setLastSent] = useState<{ text: string } | null>(null)
  // Most recent emotion tag from the stream (or a poke) — drives the pet
  // window's mood glow; the VRM expression itself is set via kurisuBridge.
  const [lastEmotion, setLastEmotion] = useState<string | null>(null)
  const cancelRef = useRef<StreamChatHandle | null>(null)
  // Turn counter feeding acknowledgment.ts's turn-scoped arm/release (see
  // useChatTurn's identical epoch pattern) — every user-initiated send gets
  // its own id so a stale release from a superseded turn can't cancel a
  // newer one's gaze-hold.
  const ackTurnRef = useRef(0)
  const isBusyRef = useRef(false)
  useEffect(() => {
    isBusyRef.current = isBusy
  }, [isBusy])

  // Voice-honesty chip (extracted hook): explains a silent line — TTS still
  // warming, or a reported synthesis failure — instead of letting it read
  // as "the audio is broken". See useVoiceHonestyHint's doc comment for the
  // re-arm semantics.
  const { voiceWarmingHint, markAudioPlayed, probeSilentLine, noteTtsFailure } =
    useVoiceHonestyHint()

  const stopCurrentStream = () => {
    cancelRef.current?.cancel()
    cancelRef.current = null
    kurisuBridge.lipSync?.stop?.()
    setIsBusy(false)
    // Sync the ref immediately: it only tracks state on the next render, so
    // a barge-in followed straight by a new utterance would still see
    // isBusyRef true in sendVoiceTurn and silently swallow the send.
    isBusyRef.current = false
  }

  // Shared tail for every unprompted line (poke, greeting, reminders,
  // reach-out, idle, screen peek): expression, then audio through the
  // pet's own lipSync, then the fading caption. Stable identity so the
  // long-lived presence/poke effects can close over it safely.
  const applyPresenceLine = useCallback(
    async (data: PresencePayload, text: string) => {
      if (data.emotion) {
        setLastEmotion(data.emotion)
        kurisuBridge.expressions?.set(data.emotion, data.emotion_intensity ?? 0.6, 2500)
        // Gesture too, matching the interactive turn path (voiceTurn.ts) — the
        // bridge's own cooldown throttles a chatty presence loop.
        kurisuBridge.react?.(data.emotion, data.emotion_intensity ?? 0.7)
      }
      if (data.audio && isAudioEnabled()) {
        if (!kurisuBridge.lipSync?.play) {
          // The historical silent killer, same class as the poke 422 that
          // used to go unnoticed for a week: audio arrived, the pet MEANT
          // to voice it, but the bridge slot isn't registered — reads
          // exactly like "nothing to say" without this counter.
          diagnostics.increment('pet.presence_audio_no_lipsync')
        } else {
          try {
            await kurisuBridge.lipSync.play({
              base64Audio: data.audio,
              mimeType: data.audio_mime_type ?? 'audio/wav',
            })
            markAudioPlayed()
          } catch (error) {
            // Autoplay may be blocked before the first click — still best-
            // effort, but counted so a genuinely broken playback path is
            // distinguishable from an expected autoplay-gate rejection.
            diagnostics.increment('pet.presence_audio_failed', String(error))
          }
        }
      } else if (!data.audio && isAudioEnabled()) {
        // No audio arrived even though audio is on — maybe explain why
        // (the hook gates re-probing internally).
        probeSilentLine()
      }
      setLastReply(text)
    },
    [markAudioPlayed, probeSilentLine],
  )

  // Doze active flag, mirrored from kurisuBridge.petDoze.active() by the
  // effect below into a ref for the same synchronous-read reason as the
  // courtesy ref below.
  const isDozingRef = useRef(false)

  // Read-together mode reads this ref inside usePetAmbient's screen-
  // companion peek effect (declared here, ahead of usePetAmbient, so
  // there's no temporal-dead-zone issue) — mirrored from useReadTogether's
  // `readTogether` state further down (same pattern as watchTogetherActiveRef
  // below it).
  const readTogetherActiveRef = useRef(false)
  // Same pattern for watch-together: the peek gate pauses ambient screen
  // glances while EITHER explicit shared activity runs (see usePetAmbient).
  const watchTogetherActiveRef = useRef(false)

  // Fullscreen courtesy + screen-companion peek — see usePetAmbient.ts for
  // both blocks (constants, the courtesy-active ref, the poll/hide/show
  // effect, and the localStorage-backed screen-companion toggle + peek
  // scheduling effect).
  const { isCourtesyActiveRef, screenCompanion, toggleScreenCompanion, lastPeekAt } =
    usePetAmbient({
      isBusyRef,
      isDozingRef,
      readTogetherActiveRef,
      watchTogetherActiveRef,
      applyPresenceLine,
    })

  // Watch together (opt-in per activation, NOT persisted, same spec as read
  // together): periodic screen frames instead of page-turn-triggered ones —
  // see useWatchTogether.ts (extracted to keep this file under its line
  // budget; usePetAmbient is the same pattern).
  const { watchTogether, toggleWatchTogether, lastWatchFrameAt } = useWatchTogether({
    isBusyRef,
    isDozingRef,
    isCourtesyActiveRef,
    applyPresenceLine,
  })
  useEffect(() => {
    watchTogetherActiveRef.current = watchTogether
  }, [watchTogether])

  // Read together (opt-in per activation, NOT persisted, same spec as watch
  // together): page-turn-triggered screen frames instead of periodic ones —
  // see useReadTogether.ts (extracted to keep this file under its line
  // budget; usePetAmbient/useWatchTogether are the same pattern).
  const { readTogether, toggleReadTogether, lastPageAt, lastPageFrameRef } = useReadTogether({
    isBusyRef,
    isDozingRef,
    isCourtesyActiveRef,
    applyPresenceLine,
  })
  useEffect(() => {
    readTogetherActiveRef.current = readTogether
  }, [readTogether])

  // Presence loop: greeting on open, reminder/reach-out polls, idle
  // chatter — previously main-window-only, which left a solo pet deaf to
  // all of them.
  const petPresence = usePetPresence({
    isBusyRef,
    applyLine: applyPresenceLine,
    isCourtesyActiveRef,
    isDozingRef,
    // Reminders/reach-out are time-sensitive enough to interrupt a doze —
    // wake her right before the line lands (rule (b) of the doze contract).
    onBeforePresenceWake: () => {
      getPetDoze()?.wake('presence')
    },
  })
  const petPresenceRef = useRef(petPresence)
  petPresenceRef.current = petPresence

  // Focus/co-working mode (amadeus/focus_mode.py): check-in rides the same
  // presence voice path; server suppresses idle/ambient/reach-out itself.
  const focus = useFocusMode({ applyLine: applyPresenceLine })

  // Mirror doze's active() into the ref usePetPresence reads. Polled on the
  // same short interval as the courtesy check below rather than wired to a
  // change event — petDoze doesn't publish one, and a 1s poll is cheap
  // (this is a single function call, not a fetch).
  useEffect(() => {
    const check = () => {
      isDozingRef.current = getPetDoze()?.active() === true
    }
    check()
    const intervalId = window.setInterval(check, 1_000)
    return () => window.clearInterval(intervalId)
  }, [])

  // Welcome-back reaction: Rust's lock poller (main.rs's spawn_lock_watcher)
  // emits `session-unlocked` on a Windows locked->unlocked transition. The
  // nonverbal perk-up (wake + gesture) always plays, throttled so a string
  // of quick lock/unlock cycles doesn't spam it; whether she actually SAYS
  // anything is entirely the server's call via the existing greeting path
  // (should_greet's freshness gap) — no client-side "was this a long
  // absence" heuristic here.
  const lastPerkUpRef = useRef(0)
  useEffect(() => {
    if (!isTauri()) return
    let cancelled = false
    let unlisten: (() => void) | null = null
    void (async () => {
      try {
        const { listen } = await import('@tauri-apps/api/event')
        const off = await listen('session-unlocked', () => {
          // Busy/DND/courtesy: an unprompted nonverbal reaction is exactly
          // the kind of thing those modes exist to suppress — same "don't
          // even act" rule as idle chatter/screen peeks, not just "don't
          // fetch". Busy specifically: a chat turn is already animating her,
          // and requestUnlockWakeup's own fetches would no-op on isBusyRef
          // anyway (requestLine's own gate), so skip the whole reaction
          // rather than firing a gesture that fights the active turn.
          if (isBusyRef.current || isPetDndActive() || isCourtesyActiveRef.current) return
          const now = Date.now()
          if (now - lastPerkUpRef.current < PERK_UP_THROTTLE_MS) return
          lastPerkUpRef.current = now
          // Wake FIRST, then fetch — doze contract ordering (b): waking to
          // see you is the point of the perk-up itself, and any words that
          // follow (via requestUnlockWakeup below) should land on an
          // already-awake avatar, not race the doze transition.
          getPetDoze()?.wake('unlock')
          kurisuBridge.react?.('happy', 0.5)
          kurisuBridge.expressions?.set('happy', 0.5, 2000)
          // Words are the server's call: the greeting path's should_greet
          // freshness gap decides "lunch break silence" vs "new session
          // greeting" — this just asks, same as the mount-greeting request
          // does, and also runs one immediate reminders/reach-out poll so
          // anything queued while locked announces promptly rather than
          // waiting up to 30s for the next scheduled tick.
          petPresenceRef.current.requestUnlockWakeup()
        })
        if (cancelled) {
          off()
        } else {
          unlisten = off
        }
      } catch (error) {
        // Older shell without the event, or the dynamic import failing —
        // the pet just never perks up on unlock.
        diagnostics.increment('pet.session_unlocked_listen_failed', String(error))
      }
    })()
    return () => {
      cancelled = true
      unlisten?.()
    }
    // isCourtesyActiveRef comes back from usePetAmbient as a stable ref
    // object (useRef identity never changes across its renders) — read via
    // closure like isBusyRef/isDozingRef, not a reactive dependency this
    // mount-once listener should re-subscribe on.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  // Live caption accumulator: the pet has no transcript, so without this
  // the caption only appeared at onDone — AFTER the voice had already been
  // speaking for seconds. Sentences append as they stream instead.
  const liveReplyRef = useRef('')

  const sendVoiceTurn = (text: string) => {
    const trimmed = text.trim()
    if (!trimmed || isBusyRef.current) return
    setIsBusy(true)
    // Same next-render lag as in stopCurrentStream: claim the busy slot now
    // so a second send racing this one can't also pass the guard above.
    isBusyRef.current = true
    setLastReply(null)
    liveReplyRef.current = ''
    setLastSent({ text: trimmed })
    // Doze wake (c): this is the single entry point for BOTH a hands-free
    // transcript and a typed send, so waking here covers both without
    // duplicating the call at each caller. Deliberately no groggy poke
    // here even if this actually woke her — a full chat message already
    // gets a real reply through the normal turn below, and the reply
    // itself carries the grogginess (persona-side); firing the poke too
    // would have her speak twice for one interaction.
    getPetDoze()?.wake('interaction')
    // Clear the per-reply gesture throttle — the pet window runs its own
    // scheduler instance, so ChatInterface's reset never reaches it.
    kurisuBridge.reactReset?.()
    // Instant acknowledgment (see vrm/acknowledgment.ts): the same visual
    // "I heard you" beat useChatTurn arms for the main window, released on
    // first sentence/audio or on done/failure below.
    const ackTurnId = petAckTurnId(++ackTurnRef.current)
    acknowledgment.arm(ackTurnId)
    const audioEnabled = isAudioEnabled()

    const handle = runVoiceTurn(
      {
        message: trimmed,
        sessionId: localStorage.getItem('session_id') || null,
        userId: resolveUserId(),
        audioEnabled,
        // Auto-attach (read-together): while the mode is active, the most
        // recent full-res page frame rides along on every turn so "what do
        // you think of this panel?" just works without a manual attach —
        // runVoiceTurn/buildChatPayload already thread `images` through
        // unchanged. Deliberately re-read every send rather than cleared
        // after one use: a follow-up question about the SAME page ("and
        // this panel?" two messages later) should still see it, and the
        // ref naturally moves on to the next page once one is detected.
        images: resolveTurnImages(readTogetherActiveRef.current, lastPageFrameRef.current),
      },
      {
        onSession: (sessionId) => {
          try {
            localStorage.setItem('session_id', sessionId)
          } catch {
            // Session id just won't persist.
          }
        },
        onSentence: (frame) => {
          // First real content proved the reply is alive.
          acknowledgment.release(ackTurnId)
          // Emotion dispatch (expression + gesture) already happened in
          // runVoiceTurn; the pet only needs the tag itself for the mood glow.
          if (frame.emotion) setLastEmotion(frame.emotion)
          // Caption keeps pace with the reply sentence by sentence — text
          // must never trail the voice, which starts on the first audio
          // frame long before the done frame lands.
          if (frame.en) {
            liveReplyRef.current = appendLiveReply(liveReplyRef.current, frame.en)
            setLastReply(liveReplyRef.current)
          }
        },
        onAudioStart: () => {
          // Real audio actually started for this turn — FIX 5: feeds the
          // re-arm check above so a LATER silent line (this window's next
          // turn, or a presence line) is recognized as a fresh event rather
          // than a continuation of an already-explained mute.
          markAudioPlayed()
        },
        onDone: (frame, ttsFailed) => {
          // Safety net for a token/sentence-less done (e.g. cancelled before
          // any sentence streamed) — see useChatTurn's identical comment.
          acknowledgment.release(ackTurnId)
          if (!frame.cancelled && frame.response.animations?.[0]) {
            kurisuBridge.vrmAnimations?.play(frame.response.animations[0])
          }
          // The server's reply text on a cancelled turn carries the internal
          // " [interrupted]" memory marker — the pet caption is the one
          // surface that renders response.reply raw, so strip it here.
          setLastReply(stripInterruptedMarker(frame.response.reply))
          // Partial or total server-side TTS failure this turn: the reply
          // text still landed (above), but some/all of it went out silent.
          // Reuse the same honesty-chip surface as the warming hint (FIX 3)
          // with a distinct message, rather than staying quiet the way a
          // dropped-audio turn always used to.
          if (!frame.cancelled && ttsFailed > 0 && isAudioEnabled()) {
            noteTtsFailure()
          }
          cancelRef.current = null
          setIsBusy(false)
          // Sync the ref immediately (same reason as stopCurrentStream): a
          // hands-free transcript arriving in the render gap would still see
          // isBusyRef true and be silently swallowed.
          isBusyRef.current = false
        },
        onFailure: (failure) => {
          acknowledgment.release(ackTurnId)
          // Unlike ChatInterface (which keeps a transcript bubble to append
          // an interrupted marker to), the pet has no transcript — a
          // mid-stream death must surface as a fresh in-character caption
          // instead of silently stranding the turn (previously: isBusy just
          // dropped to false with no explanation). A pre-stream connect
          // failure gets the same treatment: the pet never falls back to
          // POST (voice-first, no POST TTS path here), so both failure
          // kinds are equally terminal for this turn. When the server said
          // the PROVIDER failed (rate limit / model error), the caption
          // says that instead of falsely claiming the connection dropped.
          setLastReply(
            describeStreamFailure(failure.kind === 'mid-stream' ? failure.cause : undefined),
          )
          cancelRef.current = null
          setIsBusy(false)
          // Same immediate ref sync as onDone above.
          isBusyRef.current = false
        },
      },
    )
    cancelRef.current = handle
  }

  const mic = useMic({
    isStreaming: isBusy,
    onTranscript: sendVoiceTurn,
    stopCurrentStream,
    textareaRef,
  })

  const handsFree = useHandsFree({
    micAvailable: mic.micAvailable,
    micPermissionDenied: mic.micPermissionDenied,
    manualMicActive: mic.micStatus !== 'idle',
    isBusy,
    onTranscript: sendVoiceTurn,
    showHint: mic.showMicHint,
    onBargeIn: () => {
      const wasSpeaking = kurisuBridge.lipSync?.isSpeaking?.() === true
      stopCurrentStream()
      if (wasSpeaking) kurisuBridge.expressions?.set('angry', 0.35, 1600)
    },
  })

  // Desktop roaming (opt-in kurisu_pet_roam, Tauri only — see vrm/petRoam
  // for the scheduler and usePetRoam for the wiring): occasional slow
  // strolls along the work-area edge. Mounted here because this hook owns
  // every yield signal roaming composes with (busy/doze/courtesy refs, mic
  // + hands-free status); mic listening goes through a getter since those
  // are per-render hook results, not refs.
  usePetRoam({
    isBusyRef,
    isDozingRef,
    isCourtesyActiveRef,
    isListening: () =>
      mic.micStatus === 'recording' || handsFree.handsFreeStatus === 'listening',
  })

  // Shared /api/poke fetch: raycast-click pokes (region from VRMViewer),
  // drag reactions ('dragged'), and the click-through parting line
  // ('click_through') all ride this one helper so the in-flight guard,
  // busy/DND gating, and error handling live in exactly one place instead
  // of being copy-pasted per caller (poke used to be the only one; this
  // used to be inline in the effect below).
  const pokeInFlightRef = useRef(false)
  const firePoke = useCallback(
    async (region?: string): Promise<void> => {
      if (pokeInFlightRef.current || isBusyRef.current || isPetDndActive()) return
      pokeInFlightRef.current = true
      try {
        const response = await apiFetch('/api/poke', presenceRequest({ region: region ?? null }))
        if (!response.ok) {
          // The historical silent killer: a drag/region poke 422'ing here
          // looks IDENTICAL to "she had nothing to say" without this.
          diagnostics.increment('pet.poke_failed', `HTTP ${response.status}`)
          return
        }
        const data = await response.json()
        if (!data.reaction) return
        await applyPresenceLine(data, data.reaction)
      } catch (error) {
        // Poke is throwaway flavor; ignore failures — but still surface them.
        diagnostics.increment('pet.poke_failed', String(error))
      } finally {
        pokeInFlightRef.current = false
      }
    },
    [applyPresenceLine],
  )
  const firePokeRef = useRef(firePoke)
  firePokeRef.current = firePoke

  // Doze wake (c), mic-press branch: pressing the mic button is an
  // interaction that ISN'T itself a chat message (no transcript exists
  // yet — that only lands after release), so unlike sendVoiceTurn this one
  // DOES fire the groggy 'woke_from_doze' poke when the wake actually
  // interrupted a doze. PetView calls this from the mic button's
  // onPointerDown instead of duplicating the getPetDoze() accessor there.
  const wakeOnMicPress = useCallback(() => {
    if (getPetDoze()?.wake('interaction') === true) {
      void firePokeRef.current('woke_from_doze')
    }
  }, [])

  // Poke: VRMViewer raycasts the click and calls kurisuBridge.poke; the pet
  // window owns the fetch here (ChatInterface does in the main window).
  useEffect(() => {
    kurisuBridge.poke = (region?: string) => {
      void firePokeRef.current(region)
    }
    return () => {
      kurisuBridge.poke = null
    }
  }, [])

  // Drag reaction (Tauri only): the window itself is the drag handle
  // (data-tauri-drag-region on the drag bar), so "the user dragged the
  // pet" is exactly Tauri's own move-event stream — no pointer tracking of
  // our own needed. A drag is a BURST of onMoved events (every intermediate
  // frame of the OS-level move), so this debounces to one reaction per
  // burst instead of firing continuously while the cursor moves, and caps
  // to at most one every DRAG_REACTION_COOLDOWN_MS so flicking the window
  // around repeatedly doesn't turn into a line every couple of seconds.
  const lastDragReactionRef = useRef(0)
  useEffect(() => {
    if (!isTauri()) return
    let cancelled = false
    let settleTimer = 0
    let unlisten: (() => void) | null = null
    void (async () => {
      try {
        const { getCurrentWindow } = await import('@tauri-apps/api/window')
        const off = await getCurrentWindow().onMoved(() => {
          window.clearTimeout(settleTimer)
          settleTimer = window.setTimeout(() => {
            void (async () => {
              // Perch state after a drag settles: the Rust side may have
              // edge-snapped the window during/after the drag, so this is
              // queried fresh every settle rather than cached. try/catch
              // tolerates an older shell that doesn't have the command yet
              // — the drag reaction below still runs unperched in that case.
              let perched = false
              try {
                const { invoke } = await import('@tauri-apps/api/core')
                const state = await invoke<{ perched: boolean; edge: string | null }>(
                  'pet_perch_state',
                )
                perched = state?.perched === true
                if (!cancelled) {
                  window.dispatchEvent(
                    new CustomEvent('kurisu-pet-perch-changed', { detail: state }),
                  )
                }
              } catch (error) {
                // Command may not exist (older shell) — treat as unperched.
                diagnostics.increment('pet.perch_state_failed', String(error))
              }
              if (cancelled) return
              // Settling onto a perch replaces the protest: sitting down is
              // the reaction, so the 'dragged' poke would be redundant (and
              // read as complaining about a landing she just accepted).
              if (perched) return
              // An autonomous roam stroll rides the SAME OS move-event
              // stream as a user drag — she must not protest her own walk.
              if (isRoamMotionRecent()) return
              const now = Date.now()
              if (now - lastDragReactionRef.current < DRAG_REACTION_COOLDOWN_MS) return
              lastDragReactionRef.current = now
              void firePokeRef.current('dragged')
            })()
          }, DRAG_SETTLE_MS)
        })
        if (cancelled) {
          off()
        } else {
          unlisten = off
        }
      } catch (error) {
        // Not fatal — the pet just won't react to being dragged.
        diagnostics.increment('pet.drag_listen_failed', String(error))
      }
    })()
    return () => {
      cancelled = true
      window.clearTimeout(settleTimer)
      unlisten?.()
    }
  }, [])

  // Perch state on mount (Tauri only): after a restore, the window may
  // already be sitting on an edge from a previous session — the avatar
  // layer needs that BEFORE any drag happens, not just after the next one.
  // try/catch tolerates an older shell without the command.
  useEffect(() => {
    if (!isTauri()) return
    let cancelled = false
    void (async () => {
      try {
        const { invoke } = await import('@tauri-apps/api/core')
        const state = await invoke<{ perched: boolean; edge: string | null }>('pet_perch_state')
        if (!cancelled) {
          window.dispatchEvent(new CustomEvent('kurisu-pet-perch-changed', { detail: state }))
        }
      } catch (error) {
        // Command may not exist yet — the avatar layer just stays unperched.
        diagnostics.increment('pet.perch_state_failed', String(error))
      }
    })()
    return () => {
      cancelled = true
    }
  }, [])

  return {
    isBusy,
    lastReply,
    lastSent,
    lastEmotion,
    stopCurrentStream,
    // Typed input rides the exact same turn pipeline as voice transcripts.
    sendText: sendVoiceTurn,
    mic,
    handsFree,
    screenCompanion,
    toggleScreenCompanion,
    // Timestamp of the last successful screen capture — PetView pulses the
    // Screen button off changes to this (see the capture-time comment above).
    lastPeekAt,
    // Read-together mode: active flag + toggle, and the timestamp of the
    // last page actually POSTed (PetView pulses the 📖 button off changes
    // to this, same pattern as lastPeekAt/peekPulsing).
    readTogether,
    toggleReadTogether,
    lastPageAt,
    // Watch-together mode: active flag + toggle, and the timestamp of the
    // last frame actually POSTED (PetView can pulse its toggle button off
    // changes to this, same pattern as lastPageAt/lastPeekAt).
    watchTogether,
    toggleWatchTogether,
    lastWatchFrameAt,
    // Shared poke helper — PetView's click-through handler fires it with
    // 'click_through' for the parting line (item 6b); the drag reaction
    // (above) and the raycast poke (kurisuBridge.poke) already use it.
    firePoke,
    // Doze wake (c), mic-press branch — PetView calls this from the mic
    // button's onPointerDown, before any transcript exists.
    wakeOnMicPress,
    // Fullscreen courtesy active flag — PetView could use it to suppress
    // rendering while hidden, but today it's exposed mainly so the
    // click-through/close handlers (or future UI) can reflect it; the
    // hide/show itself already happens inside this hook regardless of
    // whether a caller reads this. usePetAmbient owns the underlying state
    // and mirrors it into isCourtesyActiveRef synchronously on the same
    // render that flips it, so reading .current here is current-as-of-render.
    courtesyActive: isCourtesyActiveRef.current,
    // Voice-honesty chip (useVoiceHonestyHint) — PetView renders it through
    // the same transient pet-caption/pet-hint styling as mic.micHint.
    voiceWarmingHint,
    // Focus/co-working mode: active flag + remaining time + start/end.
    focus,
  }
}
