import { resolveWsUrl } from './api'

export interface StreamChatPayload {
  message: string
  session_id?: string | null
  user_id?: string | null
  audio_enabled?: boolean
  agents?: string[]
  rounds?: number
  budget_tier?: string
  /** Vision: at most one user-shared image attached to this turn. */
  images?: Array<{ data: string; mime: string }>
}

export interface SentenceFrame {
  seq: number
  en: string
  ja: string
  emotion: string
  emotion_raw?: string | null
  emotion_intensity: number
}

export interface AudioFrame {
  seq: number
  mime: string
  data: string
}

/** Tool-augmented turn liveness frame: emitted when the single-model tool
 * loop starts/finishes a tool call mid-turn (e.g. `{tool: "web_search",
 * status: "running"}`), so the UI can show what she's doing before tokens
 * arrive. POST-path turns never see these. */
export interface ToolProgressFrame {
  tool: string
  status: string
}

/** Unspoken inner-thought line (web.streaming_inner_thoughts): displayed as
 * a faint ghost line on the message, never voiced, never stored in memory.
 * At most one per reply. */
export interface ThoughtFrame {
  seq: number
  text: string
}

export interface DoneFrame {
  cancelled: boolean
  /** How many sentences produced no audio because TTS synthesis failed —
   * distinguishes a silent turn from a voice-off turn. */
  tts_failed?: number
  timings?: {
    ttft_ms: number | null
    ttfa_ms: number | null
    total_ms: number
  }
  response: {
    reply: string
    /** Server-generated id for this exchange (regenerate/edit/rewind key
     * off it); null/absent on older backends or degraded turns. */
    turn_id?: string | null
    memories?: string[]
    animations?: string[]
    summary?: string | null
    summary_strategy?: string | null
    summary_revision?: number | null
    summary_updated_at?: string | null
  }
}

export interface StreamChatHandlers {
  onSession?: (sessionId: string) => void
  onToken?: (text: string) => void
  onSentence?: (frame: SentenceFrame) => void
  onAudio?: (frame: AudioFrame) => void
  onThought?: (frame: ThoughtFrame) => void
  onToolProgress?: (frame: ToolProgressFrame) => void
  onDone?: (frame: DoneFrame) => void
  /** `kind` is the server's structured failure classification when present
   * ("rate_limited" / "provider_error"); absent on older backends and on
   * frames that predate the field. */
  onError?: (message: string, fallbackToPost: boolean, kind?: string) => void
  onConnectError?: () => void
}

export interface StreamChatHandle {
  cancel: () => void
}

/** First-frame watchdog: a socket that opens but never produces a frame
 * (server hung mid-startup, half-open TCP after sleep/resume) fires neither
 * onerror nor onclose, so without a deadline the UI would spin forever.
 * If nothing arrives within this window the socket is torn down and the
 * failure surfaces via onConnectError — the connect class, because zero
 * frames flowed, which is exactly what makes a POST fallback safe (see
 * chat/voiceTurn.ts's connect-vs-mid-stream classification). */
const FIRST_FRAME_TIMEOUT_MS = 15000

/**
 * Opens a WebSocket to /api/chat/stream, sends the initial chat frame, and
 * dispatches incoming frames to the matching handler. Connection failures
 * (server has streaming disabled, network error) surface via onConnectError
 * / onError so the caller can fall back to the POST endpoint.
 */
export const streamChat = (payload: StreamChatPayload, handlers: StreamChatHandlers): StreamChatHandle => {
  let socket: WebSocket
  let settled = false

  try {
    // resolveWsUrl appends ?token=<api token> when one is configured — the
    // WS handshake can't carry an Authorization header.
    socket = new WebSocket(resolveWsUrl('/api/chat/stream'))
  } catch {
    handlers.onConnectError?.()
    return { cancel: () => undefined }
  }

  const safeParse = (raw: string): Record<string, unknown> | null => {
    try {
      return JSON.parse(raw) as Record<string, unknown>
    } catch {
      return null
    }
  }

  // Armed from the moment the socket is created; disarmed by the first frame
  // of ANY kind (the timeout only guards "never produced a live stream" —
  // once frames flow, stalls are the mid-stream paths' problem).
  let watchdogId: number | null = null
  const clearWatchdog = () => {
    if (watchdogId !== null) {
      window.clearTimeout(watchdogId)
      watchdogId = null
    }
  }
  watchdogId = window.setTimeout(() => {
    watchdogId = null
    if (settled) return
    settled = true
    try {
      socket.close()
    } catch {
      // Already closing; nothing to do.
    }
    handlers.onConnectError?.()
  }, FIRST_FRAME_TIMEOUT_MS)

  socket.onopen = () => {
    // A cancel() while still CONNECTING settles the turn — onopen firing
    // afterwards (close() is async) must not send the payload and start a
    // reply nobody is listening to.
    if (settled) return
    socket.send(JSON.stringify({ type: 'chat', ...payload }))
  }

  socket.onerror = () => {
    clearWatchdog()
    if (!settled) {
      settled = true
      handlers.onConnectError?.()
    }
  }

  socket.onclose = () => {
    clearWatchdog()
    if (!settled) {
      // Closed before a 'done' frame arrived — treat as a connect/stream failure.
      settled = true
      handlers.onConnectError?.()
    }
  }

  socket.onmessage = (event) => {
    const data = typeof event.data === 'string' ? safeParse(event.data) : null
    if (!data || typeof data.type !== 'string') return
    // Any valid frame proves the stream is live (no-op after the first).
    clearWatchdog()

    switch (data.type) {
      case 'session':
        handlers.onSession?.(String(data.session_id))
        break
      case 'token':
        handlers.onToken?.(String(data.text ?? ''))
        break
      case 'sentence':
        handlers.onSentence?.(data as unknown as SentenceFrame)
        break
      case 'audio':
        handlers.onAudio?.(data as unknown as AudioFrame)
        break
      case 'thought':
        handlers.onThought?.(data as unknown as ThoughtFrame)
        break
      case 'tool_progress':
        handlers.onToolProgress?.(data as unknown as ToolProgressFrame)
        break
      case 'done':
        settled = true
        handlers.onDone?.(data as unknown as DoneFrame)
        break
      case 'error':
        settled = true
        handlers.onError?.(
          String(data.message ?? 'stream error'),
          Boolean(data.fallback_to_post),
          typeof data.kind === 'string' ? data.kind : undefined,
        )
        break
      default:
        break
    }
  }

  return {
    cancel: () => {
      // A cancelled turn is being abandoned either way — the watchdog must
      // not fire a spurious connect failure after the caller moved on.
      clearWatchdog()
      if (socket.readyState === WebSocket.OPEN) {
        try {
          socket.send(JSON.stringify({ type: 'cancel' }))
        } catch {
          // socket already closing; nothing to do
        }
      } else {
        // Not OPEN (usually still CONNECTING): no frames ever flowed, so
        // closing is safe — and necessary. Without it, onopen still fires,
        // sends the payload, and the cancelled turn's reply streams (and
        // speaks) anyway. Mark the turn settled so onopen/onclose no-op
        // instead of sending or reporting a spurious connect failure.
        settled = true
        try {
          socket.close()
        } catch {
          // Already closing; nothing to do.
        }
      }
    },
  }
}
