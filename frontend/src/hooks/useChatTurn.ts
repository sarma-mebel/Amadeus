/**
 * The chat send path extracted from ChatInterface: sendMessage (goodbye/
 * memory-command/dummy-provider gates, barge-in, the WS-stream-first with
 * POST-fallback decision), sendMessageViaStream (transcript token
 * rendering, thought ghost lines, tool activity, latency logging on top of
 * chat/voiceTurn's shared core), sendMessageViaPost (+ the progressive
 * playDeferredSentences audio loop), and the turn-scoped cancel machinery
 * (stopCurrentStream, cancelDeferredTts, the turn epoch, the stream
 * cooldown). Owns isLoading/isStreaming AND their live ref mirrors —
 * presence, mic, hands-free and the voice-state HUD all consume them.
 *
 * Deliberately NOT owned: the transcript (messages/setMessages), the input
 * box value, pinnedToBottomRef (the view's scroll pin — sends re-pin it),
 * turnImageRef (useVision documents that the send path only *consumes* it),
 * audioEnabled (root state — the POST response can write it back), and the
 * regenerate/edit bookkeeping (completeTurnMessages /
 * takeRestoredVariantMessage read pendingVariantsRef, which useTurnActions
 * owns — both are injected, lazily, since useTurnActions is wired after
 * this hook and needs sendMessage from it).
 */
import { useEffect, useRef, useState } from 'react'
import type { Dispatch, MutableRefObject, SetStateAction } from 'react'
import { apiFetch } from '../api'
import { t } from '../i18n'
import { looksLikeGoodbye } from '../farewell'
import type { DoneFrame } from '../streaming'
import {
  HttpError,
  classifySendError,
  describeSendError,
  describeStreamFailure,
  type StreamFailureCause,
} from '../chat/errors'
import { saveAudioEnabled } from '../audioPrefs'
import { loadAmbienceEnabled, soundscape } from '../soundscape'
import { buildChatPayload, type ChatImageAttachment } from '../chat/payload'
import { runVoiceTurn } from '../chat/voiceTurn'
import { kurisuBridge } from '../vrm/kurisuBridge'
import { acknowledgment } from '../vrm/acknowledgment'
import type { Message } from '../chat/messages'
import type { PendingImage } from '../images'
import { resolveUserId } from './useSessions'
import { describeToolActivity } from '../chat/uiHelpers'

// Tool-requested avatar flourish (play_animation): the reply carries the
// vetted clip names; play the first once the turn settles. One-shots return
// to the idle rotation on their own.
const playRequestedAnimations = (animations?: string[] | null) => {
  const name = animations?.[0]
  if (name) kurisuBridge.vrmAnimations?.play(name)
}

export interface UseChatTurnOptions {
  setMessages: Dispatch<SetStateAction<Message[]>>
  /** Root-owned input box value/setter — sendMessage reads the draft when
   * called without an override and clears it once the turn is committed. */
  input: string
  setInput: Dispatch<SetStateAction<string>>
  /** Root-owned: the POST response can write audio_enabled back. */
  audioEnabled: boolean
  setAudioEnabled: Dispatch<SetStateAction<boolean>>
  setRequiresAudioUnlock: Dispatch<SetStateAction<boolean>>
  adoptSessionId: (sessionId: string) => void
  /** The transcript view's scroll pin — a new send always re-pins. */
  pinnedToBottomRef: MutableRefObject<boolean>
  /** Vision: pending chip state + the per-turn attachment ref. The send
   * path owns *consuming* the pending image into the turn (see
   * useVision.ts's doc comment) — the refs themselves live there. */
  pendingImageRef: MutableRefObject<PendingImage | null>
  turnImageRef: MutableRefObject<ChatImageAttachment[] | undefined>
  setPendingImage: Dispatch<SetStateAction<PendingImage | null>>
  /** useOnboardingGate's dummy-provider send gate. */
  ensureRealLlmConfigured: () => Promise<boolean>
  /** Root-owned /remember-/forget-/memory command handler. */
  handleMemoryCommand: (trimmed: string) => Promise<boolean>
  /** Optional: fired whenever a turn ends in a stream or POST failure (not
   * on a plain barge-in/cancel). Event-driven refresh for the LLM-health
   * banner (useBackendStatus.refreshBackendStatus) — a failure is exactly
   * when llm_health may have just flipped away from 'ok', so re-fetching
   * /api/status here catches that without a dedicated poll loop. Best-
   * effort: omit to leave the existing behavior (status only refreshes on
   * its own poll cadence) unchanged. */
  onTurnFailure?: () => void
  /** usePresence's goodbye path — injected lazily (usePresence is wired
   * after this hook, consuming isLoadingRef/isStreamingRef from it). */
  requestFarewell: (message: string) => Promise<boolean>
  /** useMic's transient hint line — injected lazily for the same reason. */
  showMicHint: (hint: string) => void
  /** useTurnActions' regenerate/variant bookkeeping — injected lazily
   * (useTurnActions needs sendMessage from this hook). */
  completeTurnMessages: (prev: Message[], assistantId: string, turnId: string) => Message[]
  takeRestoredVariantMessage: () => Message | null
}

export const useChatTurn = ({
  setMessages,
  input,
  setInput,
  audioEnabled,
  setAudioEnabled,
  setRequiresAudioUnlock,
  adoptSessionId,
  pinnedToBottomRef,
  pendingImageRef,
  turnImageRef,
  setPendingImage,
  ensureRealLlmConfigured,
  handleMemoryCommand,
  requestFarewell,
  showMicHint,
  completeTurnMessages,
  takeRestoredVariantMessage,
  onTurnFailure,
}: UseChatTurnOptions) => {
  const [isLoading, setIsLoading] = useState(false)
  const [isStreaming, setIsStreaming] = useState(false)
  const [lastTimings, setLastTimings] = useState<string | null>(null)
  // Write-only: kept for the header's hover-free a11y label; the visible
  // "Last reply" chip was dropped (redundant with per-message timestamps).
  const [, setLastReplyAt] = useState<number | null>(null)
  // Friendly label for the in-flight tool call ('Searching the web…'), fed
  // by 'tool_progress' WS frames; null hides the strip. Cleared when the
  // first token arrives or the turn ends (see the isLoading effect below).
  const [toolActivity, setToolActivity] = useState<string | null>(null)

  // Live mirrors of isLoading/isStreaming for async callbacks (presence
  // responses, idle-chatter timer) that fire long after the closure that
  // captured the state values went stale.
  const isLoadingRef = useRef(isLoading)
  const isStreamingRef = useRef(isStreaming)
  useEffect(() => {
    isLoadingRef.current = isLoading
    isStreamingRef.current = isStreaming
  }, [isLoading, isStreaming])

  // The tool strip only makes sense mid-turn;
  // clearing on isLoading=false covers done/error/cancel/POST-fallback.
  useEffect(() => {
    if (!isLoading) setToolActivity(null)
  }, [isLoading])

  // Aborts the progressive per-sentence /api/tts loop of the POST fallback
  // path (barge-in, new turn, conversation switch).
  const postTtsAbortRef = useRef<AbortController | null>(null)
  // "Voice unavailable" hint is shown once per app run, not per sentence.
  const voiceUnavailableNotifiedRef = useRef(false)
  // Streaming gate read on every send. Nothing sets this false anymore —
  // the old disable-for-the-rest-of-the-session behavior was replaced by
  // the streamCooldownRef below (skip a few turns, then re-probe); the ref
  // survives only as the seam where a future hard disable would plug in.
  const streamingEnabledRef = useRef(true)
  // Transient connect failures shouldn't disable streaming forever — skip
  // the next few turns (falling back to POST) then re-probe the WebSocket.
  const streamCooldownRef = useRef(0)
  const cancelStreamRef = useRef<(() => void) | null>(null)
  // Monotonic turn counter: only the latest turn may clear loading state
  // (a superseded turn's done frame must not clobber the new turn's UI).
  const turnEpochRef = useRef(0)
  const cancelRequestedRef = useRef(false)

  /** Cancels the POST path's progressive per-sentence /api/tts loop and
   * silences anything it already scheduled. Safe to call when idle. */
  const cancelDeferredTts = () => {
    if (!postTtsAbortRef.current) return
    postTtsAbortRef.current.abort()
    postTtsAbortRef.current = null
    kurisuBridge.lipSync?.stop?.()
  }

  /** Barge-in: tell the server to stop generating and silence every
   * already-scheduled audio chunk immediately. */
  const stopCurrentStream = () => {
    postTtsAbortRef.current?.abort()
    if (cancelRequestedRef.current) return
    cancelRequestedRef.current = true
    cancelStreamRef.current?.()
    kurisuBridge.lipSync?.stop?.()
  }

  /** Emotion → avatar: expression blend + (throttled) one-shot body gesture.
   * Shared by the stream per-sentence path, the POST single-emotion path,
   * and (via applyPresenceExpressionAndAudio in usePresence) presence
   * lines. */
  const dispatchEmotion = (
    emotion: string | null | undefined,
    intensity?: number | null,
    duration = 2500,
    emotionRaw?: string | null,
  ) => {
    if (!emotion) return
    const value = typeof intensity === 'number' ? intensity : 0.8
    // The face vocabulary is the six VRM expressions; body gestures may key
    // off the richer raw tag when the model used one.
    kurisuBridge.expressions?.set(emotion, value, duration)
    kurisuBridge.react?.(emotionRaw ?? emotion, value)
  }

  const sendMessage = async (overrideText?: string, opts?: { echoUser?: boolean }) => {
    // overrideText: voice input auto-sends its transcript directly instead
    // of routing through the input box (a spoken turn shouldn't need a
    // second confirm click to land). echoUser:false is the retry path — the
    // user bubble for that text is already in the transcript.
    const echoUser = opts?.echoUser ?? true
    const trimmed = (overrideText ?? input).trim()
    if (!trimmed) return
    // A live stream can be superseded (barge-in); anything else in
    // flight (POST) still blocks sending.
    if (isLoading && !isStreaming) return

    window.dispatchEvent(new Event('kurisu:activity'))
    soundscape.play('send')
    if (loadAmbienceEnabled()) soundscape.startAmbience()

    if (trimmed.startsWith('/')) {
      const handled = await handleMemoryCommand(trimmed)
      if (handled) {
        setInput('')
        return
      }
    }

    if (!(await ensureRealLlmConfigured())) {
      return
    }

    // Goodbye intent → the farewell path: one warm send-off line instead of
    // a full chat turn. Only when nothing is in flight (interrupting her to
    // say goodbye reads better as a normal barge-in turn), and the server
    // re-validates — `not_goodbye` falls through to the regular send below.
    if (!isLoading && !isStreaming && looksLikeGoodbye(trimmed)) {
      const handled = await requestFarewell(trimmed)
      if (handled) {
        setInput('')
        return
      }
    }

    if (isStreaming) {
      // Barge-in while she's mid-sentence gets a brief, low-intensity huff —
      // in character, and it acknowledges the interruption instead of her
      // going silent as if nothing happened. Intensity 0.35 stays well under
      // the 0.68 body-gesture threshold, so it's face-only.
      const wasSpeaking = kurisuBridge.lipSync?.isSpeaking?.() === true
      stopCurrentStream()
      if (wasSpeaking) kurisuBridge.expressions?.set('angry', 0.35, 1600)
    }
    // A previous POST turn's progressive TTS may still be speaking — a new
    // turn always silences it (same rule as the WS barge-in above).
    cancelDeferredTts()
    // Vision: consume the pending image into this turn (the ref survives a
    // WS->POST fallback within the turn) and clear the chip immediately.
    const attachedImage = pendingImageRef.current
    turnImageRef.current = attachedImage
      ? [{ data: attachedImage.data, mime: attachedImage.mime }]
      : undefined
    if (attachedImage) setPendingImage(null)
    const epoch = ++turnEpochRef.current
    const ackTurnId = `chat-${epoch}`
    cancelRequestedRef.current = false
    // Item 1: a new reply/turn is starting — clear the "first strong
    // emotion wins" gesture throttle so this turn can react again.
    kurisuBridge.reactReset?.()
    // Instant acknowledgment: mask the silence-to-first-audio gap with a
    // visual "I heard you" beat (gaze hold + one nod) instead of standing
    // idle until the first sentence streams in. Released the moment the
    // stream proves alive (onToken/onSentence) or the POST reply lands, on
    // any failure/cancel, or after a hard timeout — see vrm/acknowledgment.ts.
    acknowledgment.arm(ackTurnId)

    pinnedToBottomRef.current = true
    if (echoUser) {
      const userMessage: Message = { role: 'user', content: trimmed, timestamp: Date.now() }
      setMessages((prev) => [...prev, userMessage])
    }
    setInput('')
    setIsLoading(true)
    // Normal turns clear
    // any leftover strip from a superseded turn.
    // A leftover tool strip from a superseded turn never applies to this one.
    setToolActivity(null)

    if (streamingEnabledRef.current && streamCooldownRef.current === 0) {
      const streamed = await sendMessageViaStream(trimmed, epoch)
      if (epoch === turnEpochRef.current) {
        setIsLoading(false)
        setIsStreaming(false)
      }
      if (streamed) return
      // WS connect failed or the server reported streaming is off — fall
      // back to the POST endpoint for this turn and skip a few more turns
      // before re-probing the WebSocket.
      streamCooldownRef.current = 3
      if (epoch !== turnEpochRef.current) return
      setIsLoading(true)
    } else if (streamCooldownRef.current > 0) {
      streamCooldownRef.current -= 1
    }

    await sendMessageViaPost(trimmed, epoch)
    if (epoch === turnEpochRef.current) {
      setIsLoading(false)
      setIsStreaming(false)
    }
  }

  /** Streams a reply over WebSocket. Returns false if the connection never
   * produced a usable stream (caller should fall back to POST). The
   * sentence/audio demux, emotion dispatch, and cancel-vs-endStream/error
   * classification live in the shared chat/voiceTurn core (also used by
   * usePetChat); this function supplies ChatInterface's own concerns —
   * transcript rendering, thought ghost lines, tool activity strips,
   * latency logging, and the WS->POST fallback decision. */
  const sendMessageViaStream = (trimmed: string, epoch: number): Promise<boolean> => {
    return new Promise((resolve) => {
      const ackTurnId = `chat-${epoch}`
      const turnStart = performance.now()
      let clientTtft: number | null = null
      let clientTtfa: number | null = null
      // Sentences collected for the zero-audio rescue below: when server-side
      // streamed synthesis dies (e.g. CUDA OOM), the turn completes with text
      // but no audio frames — instead of silently losing her voice, fetch the
      // audio through the deferred /api/tts path and SAY that it happened.
      const wsSentences: Array<{ ja: string; emotion: string }> = []
      const userId = resolveUserId()
      const assistantId = `assistant-${Date.now()}-${Math.random().toString(36).slice(2)}`
      let assistantAdded = false

      const ensureAssistantMessage = () => {
        if (assistantAdded) return
        assistantAdded = true
        setMessages((prev) => [...prev, { role: 'assistant', content: '', timestamp: Date.now(), id: assistantId }])
      }

      // Tokens arrive faster than frames paint; coalesce them so each frame
      // costs one setMessages (one array copy + reconcile) instead of one per
      // token. setTimeout rather than rAF: rAF stalls in hidden tabs.
      let pendingTokens = ''
      let flushTimer: number | null = null
      const flushPendingTokens = () => {
        if (flushTimer !== null) {
          window.clearTimeout(flushTimer)
          flushTimer = null
        }
        if (!pendingTokens) return
        const text = pendingTokens
        pendingTokens = ''
        setMessages((prev) =>
          prev.map((msg) => (msg.id === assistantId ? { ...msg, content: msg.content + text } : msg)),
        )
      }

      const appendToAssistant = (text: string) => {
        ensureAssistantMessage()
        pendingTokens += text
        if (flushTimer === null) flushTimer = window.setTimeout(flushPendingTokens, 16)
      }

      // Mid-stream failure: keep whatever partial reply already streamed in,
      // append the interrupted marker, and offer a retry of the same turn.
      // `cause` (from the server's error frame) swaps the connection-flavored
      // marker for an honest rate-limit/model-error one — the connection was
      // fine in those cases.
      const markStreamFailure = (cause?: StreamFailureCause) => {
        ensureAssistantMessage()
        flushPendingTokens()
        setMessages((prev) =>
          prev.map((msg) =>
            msg.id === assistantId
              ? {
                  ...msg,
                  content: `${msg.content}\n\n${describeStreamFailure(cause)}`,
                  isError: true,
                  errorKind: 'stream' as const,
                  retryText: trimmed,
                }
              : msg,
          ),
        )
      }

      const handle = runVoiceTurn(
        {
          message: trimmed,
          sessionId: localStorage.getItem('session_id') || null,
          userId,
          audioEnabled,
          images: turnImageRef.current,
        },
        {
          onSession: (sessionId) => {
            streamCooldownRef.current = 0
            adoptSessionId(sessionId)
            if (epoch === turnEpochRef.current) setIsStreaming(true)
          },
          onToken: (text) => {
            if (clientTtft === null) {
              clientTtft = performance.now() - turnStart
              soundscape.play('receive')
              // First real content proved the reply is alive — release the
              // acknowledgment hold (gaze/nod) the same moment the "she's
              // thinking" tool strip would clear.
              acknowledgment.release(ackTurnId)
              // The reply is starting — tool work (if any) is behind us.
              if (epoch === turnEpochRef.current) setToolActivity(null)
            }
            appendToAssistant(text)
          },
          onSentence: (frame) => {
            // Emotion dispatch (expression + gesture) and emotionBySeq
            // bookkeeping already happened in runVoiceTurn; this only
            // accumulates the Japanese (what she actually speaks) as a
            // subtitle under the English reply.
            if (frame.ja) {
              wsSentences.push({ ja: frame.ja, emotion: frame.emotion || 'neutral' })
              setMessages((prev) =>
                prev.map((msg) =>
                  msg.id === assistantId
                    ? { ...msg, japanese: (msg.japanese ? msg.japanese + ' ' : '') + frame.ja }
                    : msg,
                ),
              )
            }
          },
          onThought: (frame) => {
            // Unspoken ghost line attached to the reply — display-only.
            if (!frame.text) return
            ensureAssistantMessage()
            setMessages((prev) =>
              prev.map((msg) =>
                msg.id === assistantId ? { ...msg, thought: frame.text } : msg,
              ),
            )
          },
          onToolProgress: (frame) => {
            if (epoch !== turnEpochRef.current) return
            if (frame.status === 'done') {
              setToolActivity(null)
            } else {
              setToolActivity(describeToolActivity(frame.tool))
            }
          },
          onAudioStart: () => {
            if (clientTtfa === null) clientTtfa = performance.now() - turnStart
            setRequiresAudioUnlock(false)
            soundscape.play('speak-start')
          },
          onDone: (frame: DoneFrame, ttsFailed: number) => {
            // Safety net: a tool-only or otherwise token-less turn (or a
            // cancel before any token) never hit the onToken release above —
            // done always means the hold is no longer warranted either way.
            acknowledgment.release(ackTurnId)
            if (!frame.cancelled) playRequestedAnimations(frame.response.animations)
            console.info('[kurisu latency]', {
              client: {
                ttft_ms: clientTtft === null ? null : Math.round(clientTtft),
                ttfa_ms: clientTtfa === null ? null : Math.round(clientTtfa),
                total_ms: Math.round(performance.now() - turnStart),
              },
              server: frame.timings ?? null,
            })
            if (frame.timings) {
              setLastTimings(
                `TTFT ${frame.timings.ttft_ms ?? '–'}ms · TTFA ${frame.timings.ttfa_ms ?? '–'}ms · total ${frame.timings.total_ms}ms`,
              )
            }
            // The assistant message and per-sentence audio/expressions were
            // already applied incrementally via onToken/onSentence/onAudio;
            // terminally — still needs to be folded in here.
            ensureAssistantMessage()
            flushPendingTokens()
            const turnId = frame.response.turn_id
            if (typeof turnId === 'string' && turnId) {
              setMessages((prev) => completeTurnMessages(prev, assistantId, turnId))
            }
            // The turn is over — release the barge-in latch (see
            // stopCurrentStream: the guard only makes repeat stop-clicks
            // idempotent DURING teardown, it must not latch across turns,
            // or every later stop becomes a silent no-op). Epoch-scoped so
            // a superseded turn's late done can't clear a newer turn's
            // in-progress cancel.
            if (epoch === turnEpochRef.current) cancelRequestedRef.current = false
            // Zero-audio rescue: text streamed but no audio frame ever came
            // (server-side synthesis failure). Fetch the voice through the
            // deferred path and surface the degradation instead of going
            // silently mute.
            if (!frame.cancelled && audioEnabled && clientTtfa === null && wsSentences.length > 0) {
              showMicHint(t('Streamed voice failed — recovering audio…'))
              void playDeferredSentences([...wsSentences])
            } else if (!frame.cancelled && audioEnabled && ttsFailed > 0 && clientTtfa !== null) {
              // Partial failure: some audio played (clientTtfa set), so the
              // zero-audio rescue above didn't fire, but the backend still
              // reported sentence(s) that failed synthesis — reuse the same
              // hint surface with an honest, distinct message instead of
              // leaving the gap unexplained.
              showMicHint(t('Part of that reply lost its voice'))
            }
            resolve(true)
          },
          onFailure: (failure) => {
            if (failure.kind === 'connect') {
              // Falls all the way back to the POST path for this same turn
              // (see sendMessage below) — the stash must survive so the
              // POST completion can fold it onto the eventual reply. Never
              // clear pendingVariantsRef here.
              resolve(false)
              return
            }
            // Terminal failure — release the acknowledgment hold same as a
            // normal done (idempotent no-op if onToken already released it).
            acknowledgment.release(ackTurnId)
            // Terminal (non-connect) failure: restore the stashed variant
            // (takeRestoredVariantMessage) — inserted right before the
            // broken in-progress reply, so it reads as "her previous
            // answer, then this turn's failure" rather than landing after
            // the error.
            const restored = takeRestoredVariantMessage()
            if (restored) {
              setMessages((prev) => {
                const brokenIndex = prev.findIndex((msg) => msg.id === assistantId)
                if (brokenIndex === -1) return [...prev, restored]
                return [...prev.slice(0, brokenIndex), restored, ...prev.slice(brokenIndex)]
              })
            }
            // Failed mid-stream after already showing partial content —
            // don't silently re-send via POST (would duplicate the turn;
            // and for a provider error, POST would hit the same failure).
            soundscape.play('error')
            markStreamFailure(failure.cause)
            // The provider may have just flipped llm_health away from 'ok'
            // (rate limit / bad key / generic error) — re-check /api/status
            // now instead of waiting for its own poll, so the banner (see
            // ChatInterface) can react the moment it would actually help.
            onTurnFailure?.()
            // Terminal end of the turn — release the barge-in latch (same
            // rule as onDone; not done on the connect-failure path above,
            // where the POST fallback continues this same turn and its
            // completion/catch resets instead).
            if (epoch === turnEpochRef.current) cancelRequestedRef.current = false
            resolve(true)
          },
        },
      )

      cancelStreamRef.current = () => {
        handle.cancel()
      }
    })
  }

  /** Progressive audio for the POST fallback path: fetch per-sentence TTS
   * sequentially from POST /api/tts and schedule each chunk through the
   * same lip-sync stream scheduler the WS path uses, so lip sync runs
   * continuously and per-sentence emotions drive the avatar. Fire-and-
   * forget: the text is already rendered when this starts. Degrades to
   * subtitle-only (one transient hint per app run) when /api/tts is
   * unavailable (503 voice not ready, 404 older backend, network error). */
  const playDeferredSentences = async (sentences: Array<{ ja: string; emotion: string }>) => {
    postTtsAbortRef.current?.abort()
    const controller = new AbortController()
    postTtsAbortRef.current = controller
    const lipSync = kurisuBridge.lipSync
    if (!lipSync?.enqueue) return
    let begun = false
    const notifyVoiceUnavailable = () => {
      if (voiceUnavailableNotifiedRef.current) return
      voiceUnavailableNotifiedRef.current = true
      showMicHint(t('Voice unavailable — replies are text-only for now'))
    }
    try {
      for (let index = 0; index < sentences.length; index += 1) {
        if (controller.signal.aborted) return
        const { ja, emotion } = sentences[index]
        let chunk: { mime?: unknown; data?: unknown }
        try {
          const response = await apiFetch('/api/tts', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            // Backend caps text around 500 chars — trim defensively.
            body: JSON.stringify({ text: ja.slice(0, 500), emotion }),
            signal: controller.signal,
          })
          if (!response.ok) {
            notifyVoiceUnavailable()
            break
          }
          chunk = await response.json()
        } catch {
          if (controller.signal.aborted) return
          notifyVoiceUnavailable()
          break
        }
        if (controller.signal.aborted) return
        if (typeof chunk?.data !== 'string' || !chunk.data) {
          notifyVoiceUnavailable()
          break
        }
        if (!begun) {
          begun = true
          lipSync.beginStream?.()
          setRequiresAudioUnlock(false)
          soundscape.play('speak-start')
        }
        // The sentence's emotion fires as its audio is scheduled — mirrors
        // the WS path's onSentence/onAudio pairing.
        dispatchEmotion(emotion)
        lipSync
          .enqueue({
            base64Audio: chunk.data,
            mimeType: typeof chunk.mime === 'string' && chunk.mime ? chunk.mime : 'audio/wav',
            seq: index,
            emotion,
          })
          .catch((err) => {
            console.error('Deferred TTS chunk failed to schedule:', err)
          })
      }
      if (begun && !controller.signal.aborted) {
        // Drain whatever is scheduled; a barge-in instead goes through
        // lipSync.stop() via cancelDeferredTts/stopCurrentStream.
        lipSync.endStream?.()
      }
    } finally {
      if (postTtsAbortRef.current === controller) postTtsAbortRef.current = null
    }
  }

  const sendMessageViaPost = async (trimmed: string, epoch: number) => {
    const ackTurnId = `chat-${epoch}`
    const turnStart = performance.now()
    try {
      const userId = resolveUserId()

      const response = await apiFetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          ...buildChatPayload({
            message: trimmed,
            sessionId: localStorage.getItem('session_id') || null,
            userId,
            audioEnabled,
            images: turnImageRef.current,
          }),
          // Ask the backend to return the reply immediately (text +
          // sentences) and skip inline TTS; audio is fetched progressively
          // per sentence below. Older backends ignore this and still return
          // inline audio — detected below and played via the legacy path.
          defer_audio: true,
        }),
      })

      if (!response.ok) throw new HttpError(response.status)

      const data = await response.json()
      // The POST path has no incremental tokens — the whole reply landing
      // at once IS the first response, so release right here.
      acknowledgment.release(ackTurnId)
      soundscape.play('receive')
      console.info('[kurisu latency]', {
        client: { total_ms: Math.round(performance.now() - turnStart) },
        server: data.timings ?? null,
      })
      if (data.timings) {
        setLastTimings(
          `LLM ${data.timings.llm_ms ?? '–'}ms · TTS ${data.timings.tts_ms ?? '–'}ms · total ${data.timings.total_ms}ms`,
        )
      }

      // Save session info
      if (data.session_id) {
        adoptSessionId(data.session_id)
      }

      const replyTimestampBase = Date.now()

      // Sentences ({en, ja, emotion}) arrive when the backend honored
      // defer_audio; they drive the ja subtitle now and the progressive
      // per-sentence TTS below. Empty on older backends.
      const sentences = (Array.isArray(data.sentences) ? data.sentences : [])
        .map((entry: unknown) => {
          const record = (entry ?? {}) as { ja?: unknown; emotion?: unknown }
          return {
            ja: typeof record.ja === 'string' ? record.ja.trim() : '',
            emotion:
              typeof record.emotion === 'string' && record.emotion ? record.emotion : 'neutral',
          }
        })
        .filter((sentence: { ja: string }) => sentence.ja.length > 0)

      const assistantTimestamp = replyTimestampBase
      const assistantId = `assistant-${Date.now()}-${Math.random().toString(36).slice(2)}`
      const assistantMessage: Message = {
        role: 'assistant',
        content: data.reply,
        timestamp: assistantTimestamp,
        id: assistantId,
        japanese:
          sentences.length > 0
            ? sentences.map((sentence: { ja: string }) => sentence.ja).join(' ')
            : undefined,
        thought: typeof data.thought === 'string' && data.thought ? data.thought : undefined,
      }
      playRequestedAnimations(Array.isArray(data.animations) ? data.animations : null)

      setMessages((prev) => {
        const updates = [...prev, assistantMessage]
        const turnId = typeof data.turn_id === 'string' && data.turn_id ? data.turn_id : null
        if (!turnId) return updates
        return completeTurnMessages(updates, assistantId, turnId)
      })

      setLastReplyAt(assistantTimestamp)

      dispatchEmotion(data.emotion, data.emotion_intensity)

      let shouldPlayAudio = false
      if (typeof data.audio_enabled === 'boolean') {
        setAudioEnabled(data.audio_enabled)
        saveAudioEnabled(data.audio_enabled)
        shouldPlayAudio = data.audio_enabled
      } else {
        shouldPlayAudio = audioEnabled
      }
      if (shouldPlayAudio) {
        setRequiresAudioUnlock(false)
      }

      // Audio. Older backends ignore defer_audio and still return one big
      // inline blob — play it through the legacy path unchanged. A backend
      // that honored defer_audio returns audio-less sentences instead;
      // fetch their TTS progressively (fire-and-forget: the text above is
      // already rendered, audio catches up sentence by sentence).
      if (data.audio && shouldPlayAudio) {
        const lipSync = kurisuBridge.lipSync
        const mimeType = data.audio_mime_type || 'audio/wav'
        if (lipSync?.play) {
          try {
            await Promise.resolve(lipSync.play({ base64Audio: data.audio, mimeType }))
          } catch (err) {
            console.error('Lip sync playback failed, falling back to direct audio:', err)
            const audio = new Audio(`data:${mimeType};base64,${data.audio}`)
            audio.play().catch((fallbackErr) => console.error('Audio playback failed:', fallbackErr))
          }
        } else {
          const audio = new Audio(`data:${mimeType};base64,${data.audio}`)
          audio.play().catch((err) => console.error('Audio playback failed:', err))
        }
      } else if (shouldPlayAudio && sentences.length > 0) {
        void playDeferredSentences(sentences)
      }
    } catch (error) {
      // Terminal failure — release the hold (no-op if already released above).
      acknowledgment.release(ackTurnId)
      console.error('Error:', error)
      soundscape.play('error')
      // Distinguish "backend unreachable" (fetch threw) from "server
      // error" (HTTP 4xx/5xx) and offer a retry of the same text.
      const { kind, status } = classifySendError(error)
      const errorMessage: Message = {
        role: 'assistant',
        content: describeSendError(kind, status),
        timestamp: Date.now(),
        isError: true,
        errorKind: kind,
        retryText: trimmed,
      }
      // Terminal failure of a regenerate turn: restore the stashed variant
      // (takeRestoredVariantMessage) as a real message before the error
      // bubble, so the user's previous good reply is never silently
      // destroyed.
      const restored = takeRestoredVariantMessage()
      setMessages((prev) =>
        restored ? [...prev, restored, errorMessage] : [...prev, errorMessage],
      )
      // Same event-driven /api/status re-check as the stream failure path
      // above — a 'server' kind here is frequently the same upstream
      // provider failure (502 from ProviderCallError), just surfaced
      // through the POST path instead of a WS error frame.
      onTurnFailure?.()
    } finally {
      setIsLoading(false)
      // The POST turn (success or failure) is over — release the barge-in
      // latch, same rule as the stream terminals. A POST turn can't be
      // superseded mid-flight (sending is blocked while isLoading &&
      // !isStreaming), so no epoch scoping is needed here.
      cancelRequestedRef.current = false
    }
  }

  return {
    isLoading,
    isStreaming,
    isLoadingRef,
    isStreamingRef,
    lastTimings,
    toolActivity,
    sendMessage,
    stopCurrentStream,
    cancelDeferredTts,
  }
}
