/** Full-duplex voice conversation loop (opt-in per browser — see the
 * "Experimental" section of Settings → Features; the server side is on by
 * default since the 2026-07-27 promotion). Talks to the
 * `/api/chat/duplex` WebSocket (amadeus/web/streaming/duplex.py):
 * continuous mic frames out, the shared streaming-turn frame vocabulary
 * back in (token/sentence/audio/tool_progress/thought/done/error), plus
 * duplex-only `state`/`transcript`/`backchannel` frames.
 *
 * Deliberately NOT built on top of streaming.ts's streamChat — that module
 * owns the /api/chat/stream one-shot-turn-per-socket protocol (one JSON
 * `chat` frame out, frames back, then the socket closes). Duplex is a
 * long-lived socket carrying many turns plus continuous binary audio, so it
 * needs its own connection lifecycle. What IS shared: the per-sentence
 * emotion dispatch + lipSync demux logic lives in `frameDemux.ts`, used
 * identically here and in voiceTurn.ts so the two surfaces can't drift.
 *
 * Preference: localStorage `kurisu_duplex`, default OFF. Since the
 * 2026-07-30 promotion this selects the CALL-MODE ENGINE: with it on, the
 * 📞 Call button runs this full-duplex loop (via hooks/useDuplexCall +
 * useCallMode), falling back to the classic hands-free call in place when
 * the server or mic can't support it.
 */
import { resolveWsUrl } from '../api'
import type { AudioFrame, DoneFrame, SentenceFrame, ThoughtFrame, ToolProgressFrame } from '../streaming'
import { DuplexMic, DUPLEX_SAMPLE_RATE } from '../audio/duplexMic'
import { playBackchannel } from '../audio/fillerPlayer'
import { createFrameDemuxer } from './frameDemux'

export const DUPLEX_STORAGE_KEY = 'kurisu_duplex'

export const loadDuplexEnabled = (): boolean => {
  try {
    return window.localStorage.getItem(DUPLEX_STORAGE_KEY) === 'true'
  } catch {
    return false
  }
}

export const saveDuplexEnabled = (enabled: boolean): void => {
  try {
    window.localStorage.setItem(DUPLEX_STORAGE_KEY, enabled ? 'true' : 'false')
  } catch {
    // Preference just won't persist across reloads.
  }
}

export type DuplexState = 'listening' | 'user_speaking' | 'processing' | 'assistant_speaking'

export interface DuplexHandlers {
  onSessionId?: (sessionId: string) => void
  /** Capability/handshake result: false means the server has
   * web.duplex_enabled off (or the socket failed before a session frame
   * arrived) — the caller should fall back to the normal turn path and,
   * if this is the first attempt, may want to hide the toggle next time. */
  onCapability?: (available: boolean) => void
  onState?: (state: DuplexState) => void
  onTranscript?: (text: string) => void
  onToken?: (text: string) => void
  onSentence?: (frame: SentenceFrame) => void
  onThought?: (frame: ThoughtFrame) => void
  onToolProgress?: (frame: ToolProgressFrame) => void
  onDone?: (frame: DoneFrame) => void
  onError?: (message: string, kind?: string) => void
  /** Fires exactly once per turn, right before its first audio chunk is
   * enqueued — same seam as voiceTurn's onAudioStart (audio-unlock UI,
   * soundscape beat). */
  onAudioStart?: () => void
  /** Continuous mic level tap (RMS 0..1) from the duplex capture — feed
   * levelBus.publishMicLevel so the listening reactions (gaze hold, nods)
   * run while the user talks, same as the PTT/hands-free paths. */
  onMicLevel?: (rms: number, dtMs: number) => void
  /** Socket closed (cleanly or not) — mic capture has already been torn
   * down by the time this fires. */
  onClosed?: () => void
}

export interface DuplexSessionHandle {
  /** Manual barge-in (e.g. a stop button) — same effect as server-detected
   * voiced audio during processing/assistant_speaking. */
  cancel: () => void
  /** Ends the duplex session and releases the mic. */
  stop: () => void
}

/** Opens one duplex session: connects the WebSocket, sends the open
 * handshake, starts continuous mic capture, and demuxes frames until
 * stop()/cancel() or the socket closes. userId/sessionId mirror the
 * regular chat payload (chat/payload.ts) — session_id null lets the server
 * mint or resume one, same as the POST/stream paths. */
export const startDuplexSession = (
  input: { sessionId: string | null; userId: string },
  handlers: DuplexHandlers,
): DuplexSessionHandle => {
  let socket: WebSocket
  let mic: DuplexMic | null = null
  let stopped = false
  let capabilityReported = false
  // Sentence emotion dispatch + audio demux + done stop-vs-drain: shared
  // with voiceTurn.ts via frameDemux.ts. Duplex has no filler and doesn't
  // count no-lipsync drops; onAudioStart passes through for the caller's
  // audio-unlock/soundscape concerns.
  const demux = createFrameDemuxer({
    onAudioStart: () => handlers.onAudioStart?.(),
  })

  const reportCapability = (available: boolean) => {
    if (capabilityReported) return
    capabilityReported = true
    handlers.onCapability?.(available)
  }

  const teardownMic = () => {
    mic?.stop()
    mic = null
  }

  const closeSocket = () => {
    if (stopped) return
    stopped = true
    teardownMic()
    try {
      if (socket.readyState === WebSocket.OPEN || socket.readyState === WebSocket.CONNECTING) {
        socket.close()
      }
    } catch {
      // Already closing.
    }
  }

  try {
    socket = new WebSocket(resolveWsUrl('/api/chat/duplex'))
  } catch {
    reportCapability(false)
    handlers.onClosed?.()
    return { cancel: () => undefined, stop: () => undefined }
  }
  socket.binaryType = 'arraybuffer'

  const safeParse = (raw: string): Record<string, unknown> | null => {
    try {
      return JSON.parse(raw) as Record<string, unknown>
    } catch {
      return null
    }
  }

  socket.onopen = () => {
    socket.send(
      JSON.stringify({
        type: 'start',
        session_id: input.sessionId,
        user_id: input.userId,
        sample_rate: DUPLEX_SAMPLE_RATE,
        encoding: 'pcm16',
      }),
    )
  }

  socket.onerror = () => {
    reportCapability(false)
  }

  socket.onclose = () => {
    reportCapability(false)
    teardownMic()
    handlers.onClosed?.()
  }

  const beginMicCapture = async () => {
    const captured = new DuplexMic()
    captured.onFrame = (frame) => {
      if (socket.readyState === WebSocket.OPEN) {
        socket.send(frame)
      }
    }
    if (handlers.onMicLevel) {
      const onMicLevel = handlers.onMicLevel
      captured.onLevel = (rms, dtMs) => onMicLevel(rms, dtMs)
    }
    try {
      await captured.start()
    } catch (error) {
      console.warn('Duplex mic capture failed to start:', error)
      handlers.onError?.('Could not open the microphone for full-duplex voice.', 'mic_unavailable')
      closeSocket()
      return
    }
    if (stopped) {
      captured.stop()
      return
    }
    mic = captured
  }

  socket.onmessage = (event) => {
    if (typeof event.data !== 'string') return // audio-in is binary OUT only; nothing binary comes back
    const data = safeParse(event.data)
    if (!data || typeof data.type !== 'string') return

    switch (data.type) {
      case 'session': {
        reportCapability(true)
        if (typeof data.session_id === 'string') handlers.onSessionId?.(data.session_id)
        void beginMicCapture()
        break
      }
      case 'state': {
        if (typeof data.state === 'string') {
          handlers.onState?.(data.state as DuplexState)
          if (data.state === 'listening' || data.state === 'user_speaking') {
            // Read the audio latch BEFORE interrupt() resets it: only a
            // state flip that actually cut in-flight playback is a barge-in
            // (the routine post-done return to listening isn't). The event
            // feeds the avatar reflex layer (vrm/reflexes.ts): she visibly
            // catches being talked over.
            const wasAudible = demux.hasAudioStarted()
            demux.interrupt()
            if (wasAudible) {
              try {
                window.dispatchEvent(new Event('kurisu:barge-in'))
              } catch {
                // Non-DOM test environments; cutting playback is the substance.
              }
            }
          }
        }
        break
      }
      case 'transcript': {
        if (typeof data.text === 'string') handlers.onTranscript?.(data.text)
        break
      }
      case 'backchannel': {
        // Server-detected micro-pause in the user's speech: play one soft
        // acknowledgement clip ("un", "hm") in her voice. Fire-and-forget —
        // playBackchannel() already no-ops when the clip assets don't exist
        // or she is currently making a sound. The listening-nod event rides
        // along (same slow-nod machinery the PTT listening reactor uses) so
        // the acknowledgement is visible even without generated clips.
        try {
          window.dispatchEvent(new Event('kurisu:listening-nod'))
        } catch {
          // Non-DOM test environments; the clip is the substance anyway.
        }
        void playBackchannel()
        break
      }
      case 'token': {
        handlers.onToken?.(String(data.text ?? ''))
        break
      }
      case 'sentence': {
        const frame = data as unknown as SentenceFrame
        demux.handleSentence(frame)
        handlers.onSentence?.(frame)
        break
      }
      case 'audio': {
        demux.handleAudio(data as unknown as AudioFrame, true)
        break
      }
      case 'thought': {
        handlers.onThought?.(data as unknown as ThoughtFrame)
        break
      }
      case 'tool_progress': {
        handlers.onToolProgress?.(data as unknown as ToolProgressFrame)
        break
      }
      case 'done': {
        const frame = data as unknown as DoneFrame
        demux.handleDone(frame)
        handlers.onDone?.(frame)
        break
      }
      case 'error': {
        const kind = typeof data.kind === 'string' ? data.kind : undefined
        if (kind === 'duplex_disabled' || kind === 'protocol_error') {
          reportCapability(false)
        }
        console.warn('Duplex error:', data.message)
        handlers.onError?.(String(data.message ?? 'duplex error'), kind)
        break
      }
      default:
        break
    }
  }

  return {
    cancel: () => {
      if (socket.readyState === WebSocket.OPEN) {
        try {
          socket.send(JSON.stringify({ type: 'cancel' }))
        } catch {
          // Socket already closing; nothing to do.
        }
      }
    },
    stop: () => {
      if (socket.readyState === WebSocket.OPEN) {
        try {
          socket.send(JSON.stringify({ type: 'stop' }))
        } catch {
          // Socket already closing; nothing to do.
        }
      }
      closeSocket()
    },
  }
}
