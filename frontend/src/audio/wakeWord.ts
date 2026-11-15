/** Local wake-word listener (opt-in per browser AND server-side —
 * localStorage `kurisu_wake_word` + config `wake_word.enabled`).
 *
 * Streams mic PCM to the `/api/wake` WebSocket
 * (amadeus/web/streaming/wake_word.py) and fires `onWake` when the server's
 * local openwakeword model hears the wake phrase — the caller
 * (hooks/useWakeWord) then starts the normal duplex call via useCallMode.
 * Detection is entirely local to the backend machine; audio is never
 * stored, transcribed, or forwarded.
 *
 * Reuses DuplexMic (same 16 kHz PCM16 frames as a duplex call); like the
 * duplex client, capability is reported in-band — an `error` frame or a
 * close before `ready` means the server can't do it, and the caller backs
 * off rather than reconnect-looping.
 */
import { resolveWsUrl } from '../api'
import { DuplexMic, DUPLEX_SAMPLE_RATE } from './duplexMic'
import { diagnostics } from '../diagnostics'

export const WAKE_WORD_STORAGE_KEY = 'kurisu_wake_word'
/** Broadcast on toggle so live hooks react without a reload (same pattern
 * as HANDS_FREE_EVENT in audio/handsFree.ts). */
export const WAKE_WORD_EVENT = 'kurisu:wake-word-change'

export const loadWakeWordEnabled = (): boolean => {
  try {
    return window.localStorage.getItem(WAKE_WORD_STORAGE_KEY) === 'true'
  } catch {
    return false
  }
}

export const saveWakeWordEnabled = (enabled: boolean): void => {
  try {
    window.localStorage.setItem(WAKE_WORD_STORAGE_KEY, enabled ? 'true' : 'false')
  } catch {
    // Preference just won't persist across reloads.
  }
  try {
    window.dispatchEvent(new Event(WAKE_WORD_EVENT))
  } catch {
    // Non-browser test environments.
  }
}

/** Why the listener stopped without being asked to:
 * - `disabled`/`unavailable`: the server said no (config off / no model) —
 *   don't retry until the user toggles.
 * - `mic`: getUserMedia failed (no device / permission denied).
 * - `closed`: a live socket died (backend restart, network) — retryable.
 */
export type WakeUnavailableKind = 'disabled' | 'unavailable' | 'mic' | 'closed'

export interface WakeListenerHandlers {
  onWake: (score: number) => void
  onUnavailable?: (kind: WakeUnavailableKind, message?: string) => void
}

export interface WakeListenerHandle {
  stop: () => void
}

export const startWakeListener = (handlers: WakeListenerHandlers): WakeListenerHandle => {
  let stopped = false
  let ready = false
  let mic: DuplexMic | null = null
  let socket: WebSocket | null = null

  const teardown = () => {
    if (mic) {
      mic.stop()
      mic = null
    }
    if (socket) {
      try {
        if (socket.readyState === WebSocket.OPEN || socket.readyState === WebSocket.CONNECTING) {
          socket.close()
        }
      } catch {
        // Already closing.
      }
    }
  }

  const fail = (kind: WakeUnavailableKind, message?: string) => {
    if (stopped) return
    stopped = true
    teardown()
    diagnostics.increment(`wake.${kind}`, message)
    handlers.onUnavailable?.(kind, message)
  }

  try {
    socket = new WebSocket(resolveWsUrl('/api/wake'))
  } catch {
    fail('unavailable', 'could not open wake socket')
    return { stop: () => undefined }
  }
  socket.binaryType = 'arraybuffer'

  socket.onopen = () => {
    socket?.send(JSON.stringify({ type: 'start', sample_rate: DUPLEX_SAMPLE_RATE }))
  }

  socket.onmessage = (event) => {
    if (stopped || typeof event.data !== 'string') return
    let frame: Record<string, unknown> | null = null
    try {
      frame = JSON.parse(event.data) as Record<string, unknown>
    } catch {
      return
    }
    if (!frame) return
    if (frame.type === 'ready') {
      ready = true
      void beginMicCapture()
      return
    }
    if (frame.type === 'wake') {
      handlers.onWake(typeof frame.score === 'number' ? frame.score : 0)
      return
    }
    if (frame.type === 'error') {
      // 'wake_disabled' = server flag off; 'wake_unavailable' = flag on but
      // no usable model (both from routes_chat.py / streaming/wake_word.py —
      // pinned by tests/test_contracts.py). Unknown kinds degrade the same
      // as wake_unavailable: latch off until the user re-toggles.
      const kind = frame.kind === 'wake_disabled' ? 'disabled' : 'unavailable'
      fail(kind, typeof frame.message === 'string' ? frame.message : undefined)
    }
  }

  socket.onerror = () => {
    if (!ready) fail('unavailable', 'wake socket error')
  }

  socket.onclose = () => {
    if (stopped) return
    // Before ready = capability failure; after ready = a live listener died
    // (backend restart) — the hook may retry that one.
    fail(ready ? 'closed' : 'unavailable')
  }

  const beginMicCapture = async () => {
    const captured = new DuplexMic()
    captured.onFrame = (frame) => {
      if (!stopped && socket && socket.readyState === WebSocket.OPEN) {
        socket.send(frame)
      }
    }
    try {
      await captured.start()
    } catch {
      fail('mic', 'microphone unavailable for wake word')
      return
    }
    if (stopped) {
      captured.stop()
      return
    }
    mic = captured
  }

  return {
    stop: () => {
      stopped = true
      teardown()
    },
  }
}
