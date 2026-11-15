/**
 * Full-duplex call loop for ChatInterface: owns one duplex session's
 * lifecycle (chat/duplexTurn's startDuplexSession) and translates its frame
 * stream into the same transcript/avatar effects a normal streamed turn
 * produces — user bubbles from `transcript` frames, coalesced token
 * rendering into an assistant bubble, ja subtitles, turn-id folding,
 * requested-animation flourishes at done.
 *
 * Presence beats mirror useChatTurn's: the VAD endpoint (`state:
 * "processing"` — the STT+LLM wait begins) arms the instant-acknowledgment
 * controller, so she holds gaze, nods once, and drifts into the thinking
 * gaze until the first token releases it; continuous mic levels feed
 * levelBus so the listening reactions (gaze hold + slow nods) run while
 * the user is talking, exactly like the PTT path.
 *
 * Fallback contract: `onFallback` fires when duplex can't run (server has
 * web.duplex_enabled off, mic unavailable, socket died mid-call) — the
 * caller (useCallMode) downgrades the call to the classic hands-free loop
 * so the user never lands in dead air. Frontend failures count in
 * diagnostics (degrade-loudly convention).
 */
import { useEffect, useRef, useState } from 'react'
import type { Dispatch, SetStateAction } from 'react'
import { startDuplexSession, type DuplexSessionHandle, type DuplexState } from '../chat/duplexTurn'
import type { Message } from '../chat/messages'
import { acknowledgment } from '../vrm/acknowledgment'
import { kurisuBridge } from '../vrm/kurisuBridge'
import { publishMicLevel, resetMicLevel } from '../audio/levelBus'
import { loadAmbienceEnabled, soundscape } from '../soundscape'
import { diagnostics } from '../diagnostics'
import { resolveUserId } from './useSessions'

export type DuplexFallbackReason = 'unavailable' | 'mic' | 'closed'

export interface UseDuplexCallOptions {
  setMessages: Dispatch<SetStateAction<Message[]>>
  adoptSessionId: (sessionId: string) => void
  setRequiresAudioUnlock: Dispatch<SetStateAction<boolean>>
  /** useTurnActions' turn-id folding — same injection useChatTurn gets. */
  completeTurnMessages: (prev: Message[], assistantId: string, turnId: string) => Message[]
  /** Duplex can't run (or died). The caller should downgrade to the classic
   * call loop; `closed` means a session that WAS live ended unexpectedly. */
  onFallback: (reason: DuplexFallbackReason) => void
}

export const useDuplexCall = ({
  setMessages,
  adoptSessionId,
  setRequiresAudioUnlock,
  completeTurnMessages,
  onFallback,
}: UseDuplexCallOptions) => {
  const [duplexActive, setDuplexActive] = useState(false)
  const [duplexState, setDuplexState] = useState<DuplexState>('listening')
  const handleRef = useRef<DuplexSessionHandle | null>(null)
  // True between start() and stop() — distinguishes an expected onClosed
  // (user hung up) from a mid-call socket death that needs the fallback.
  const wantedActiveRef = useRef(false)
  const turnCounterRef = useRef(0)

  // Callbacks (onFallback, completeTurnMessages) can change across renders;
  // the long-lived socket handlers must always see the latest.
  const optionsRef = useRef({ onFallback, completeTurnMessages })
  useEffect(() => {
    optionsRef.current = { onFallback, completeTurnMessages }
  })

  const stop = () => {
    wantedActiveRef.current = false
    acknowledgment.release(`duplex-${turnCounterRef.current}`)
    handleRef.current?.stop()
    handleRef.current = null
    resetMicLevel()
    setDuplexActive(false)
  }

  const start = () => {
    if (handleRef.current) return
    wantedActiveRef.current = true
    setDuplexState('listening')

    // Per-turn assistant-bubble state, reset on every transcript frame.
    let assistantId: string | null = null
    let firstTokenSeen = false
    let pendingTokens = ''
    let flushTimer: number | null = null
    const flushPendingTokens = () => {
      if (flushTimer !== null) {
        window.clearTimeout(flushTimer)
        flushTimer = null
      }
      if (!pendingTokens || !assistantId) return
      const text = pendingTokens
      const id = assistantId
      pendingTokens = ''
      setMessages((prev) =>
        prev.map((msg) => (msg.id === id ? { ...msg, content: msg.content + text } : msg)),
      )
    }

    const ackId = () => `duplex-${turnCounterRef.current}`

    const handle = startDuplexSession(
      { sessionId: localStorage.getItem('session_id') || null, userId: resolveUserId() },
      {
        onSessionId: (sessionId) => adoptSessionId(sessionId),
        onCapability: (available) => {
          if (available) {
            setDuplexActive(true)
            if (loadAmbienceEnabled()) soundscape.startAmbience()
            return
          }
          diagnostics.increment('duplex.unavailable_fallback')
          wantedActiveRef.current = false
          handleRef.current = null
          optionsRef.current.onFallback('unavailable')
        },
        onState: (state) => {
          setDuplexState(state)
          if (state === 'processing') {
            // Endpoint detected — the STT+LLM wait starts NOW. Same beat
            // as a typed submit: gaze hold + one nod, then the thinking
            // aversion until the first token releases it.
            turnCounterRef.current += 1
            firstTokenSeen = false
            kurisuBridge.reactReset?.()
            acknowledgment.arm(ackId())
          } else if (state === 'user_speaking') {
            window.dispatchEvent(new Event('kurisu:activity'))
          }
        },
        onTranscript: (text) => {
          // Server-transcribed utterance — the user's bubble. A fresh
          // assistant bubble follows with the first token.
          flushPendingTokens()
          assistantId = null
          setMessages((prev) => [...prev, { role: 'user', content: text, timestamp: Date.now() }])
        },
        onToken: (text) => {
          if (!firstTokenSeen) {
            firstTokenSeen = true
            soundscape.play('receive')
            acknowledgment.release(ackId())
          }
          if (!assistantId) {
            assistantId = `assistant-${Date.now()}-${Math.random().toString(36).slice(2)}`
            const id = assistantId
            setMessages((prev) => [
              ...prev,
              { role: 'assistant', content: '', timestamp: Date.now(), id },
            ])
          }
          pendingTokens += text
          if (flushTimer === null) flushTimer = window.setTimeout(flushPendingTokens, 16)
        },
        onSentence: (frame) => {
          if (!frame.ja || !assistantId) return
          const id = assistantId
          setMessages((prev) =>
            prev.map((msg) =>
              msg.id === id
                ? { ...msg, japanese: (msg.japanese ? msg.japanese + ' ' : '') + frame.ja }
                : msg,
            ),
          )
        },
        onThought: (frame) => {
          if (!frame.text || !assistantId) return
          const id = assistantId
          setMessages((prev) =>
            prev.map((msg) => (msg.id === id ? { ...msg, thought: frame.text } : msg)),
          )
        },
        onAudioStart: () => {
          setRequiresAudioUnlock(false)
          soundscape.play('speak-start')
        },
        onMicLevel: (rms) => publishMicLevel(rms),
        onDone: (frame) => {
          acknowledgment.release(ackId())
          flushPendingTokens()
          if (!frame.cancelled) {
            const name = frame.response.animations?.[0]
            if (name) kurisuBridge.vrmAnimations?.play(name)
          }
          const turnId = frame.response.turn_id
          if (assistantId && typeof turnId === 'string' && turnId) {
            const id = assistantId
            setMessages((prev) => optionsRef.current.completeTurnMessages(prev, id, turnId))
          }
          assistantId = null
        },
        onError: (message, kind) => {
          acknowledgment.release(ackId())
          if (kind === 'mic_unavailable') {
            diagnostics.increment('duplex.mic_unavailable', message)
            wantedActiveRef.current = false
            handleRef.current = null
            resetMicLevel()
            setDuplexActive(false)
            optionsRef.current.onFallback('mic')
          }
          // Turn-scoped errors (stt_error, provider failures) keep the
          // session alive — the server returns to listening on its own.
        },
        onClosed: () => {
          acknowledgment.release(ackId())
          resetMicLevel()
          setDuplexActive(false)
          if (wantedActiveRef.current) {
            // Died mid-call (not a hang-up) — degrade to the classic loop
            // instead of leaving the user talking to a dead socket.
            diagnostics.increment('duplex.closed_mid_call')
            wantedActiveRef.current = false
            handleRef.current = null
            optionsRef.current.onFallback('closed')
          }
        },
      },
    )
    handleRef.current = handle
  }

  // Unmount safety: never leave a mic streaming after the component is gone.
  useEffect(() => () => stop(), [])

  return {
    duplexActive,
    duplexState,
    startDuplex: start,
    stopDuplex: stop,
    /** Manual barge-in (stop button) — server-side cancel. */
    cancelDuplexTurn: () => handleRef.current?.cancel(),
  }
}
