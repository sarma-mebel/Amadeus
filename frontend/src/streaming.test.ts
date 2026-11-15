// @vitest-environment jsdom
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { streamChat, type StreamChatHandlers } from './streaming'

/** Minimal controllable WebSocket stub, mirroring chat/voiceTurn.test.ts's
 * FakeWebSocket: the test holds a reference (via `sockets`) to drive
 * onopen/onmessage/onclose/onerror directly instead of a real network round
 * trip. */
class FakeWebSocket {
  static CONNECTING = 0
  static OPEN = 1
  static CLOSED = 3
  readyState = FakeWebSocket.CONNECTING
  sent: string[] = []
  onopen: (() => void) | null = null
  onmessage: ((event: { data: string }) => void) | null = null
  onerror: (() => void) | null = null
  onclose: (() => void) | null = null

  constructor(public url: string) {
    sockets.push(this)
  }

  send(data: string) {
    this.sent.push(data)
  }

  close() {
    this.readyState = FakeWebSocket.CLOSED
    this.onclose?.()
  }

  // --- test-only helpers ---
  open() {
    this.readyState = FakeWebSocket.OPEN
    this.onopen?.()
  }

  emit(frame: Record<string, unknown>) {
    this.onmessage?.({ data: JSON.stringify(frame) })
  }

  emitRaw(data: string) {
    this.onmessage?.({ data })
  }

  failToConnect() {
    this.onerror?.()
  }
}

let sockets: FakeWebSocket[] = []

const basePayload = { message: 'hi', session_id: null, user_id: 'u1', audio_enabled: true }

describe('streamChat', () => {
  beforeEach(() => {
    sockets = []
    vi.stubGlobal('WebSocket', FakeWebSocket)
  })

  afterEach(() => {
    vi.unstubAllGlobals()
    vi.useRealTimers()
  })

  it('opens a socket to the resolved WS URL and sends the chat frame on open', () => {
    streamChat(basePayload, {})

    const socket = sockets[0]
    expect(socket.url).toContain('/api/chat/stream')
    expect(socket.sent).toHaveLength(0)

    socket.open()
    expect(socket.sent).toHaveLength(1)
    expect(JSON.parse(socket.sent[0])).toMatchObject({ type: 'chat', ...basePayload })
  })

  it('demuxes session, token, sentence, audio, thought, and tool_progress frames to their handlers, in order', () => {
    const calls: string[] = []
    const handlers: StreamChatHandlers = {
      onSession: (id) => calls.push(`session:${id}`),
      onToken: (text) => calls.push(`token:${text}`),
      onSentence: (frame) => calls.push(`sentence:${frame.seq}:${frame.emotion}`),
      onAudio: (frame) => calls.push(`audio:${frame.seq}`),
      onThought: (frame) => calls.push(`thought:${frame.text}`),
      onToolProgress: (frame) => calls.push(`tool:${frame.tool}:${frame.status}`),
      onDone: () => calls.push('done'),
    }
    streamChat(basePayload, handlers)
    const socket = sockets[0]
    socket.open()

    socket.emit({ type: 'session', session_id: 'sess-1' })
    socket.emit({ type: 'token', text: 'Hel' })
    socket.emit({ type: 'token', text: 'lo' })
    socket.emit({
      type: 'sentence',
      seq: 0,
      en: 'Hello.',
      ja: 'こんにちは。',
      emotion: 'happy',
      emotion_intensity: 0.7,
    })
    socket.emit({ type: 'thought', seq: 0, text: 'hmm' })
    socket.emit({ type: 'tool_progress', tool: 'web_search', status: 'running' })
    socket.emit({ type: 'audio', seq: 0, mime: 'audio/wav', data: 'QUJD' })
    socket.emit({ type: 'done', cancelled: false, response: { reply: 'Hello.' } })

    expect(calls).toEqual([
      'session:sess-1',
      'token:Hel',
      'token:lo',
      'sentence:0:happy',
      'thought:hmm',
      'tool:web_search:running',
      'audio:0',
      'done',
    ])
  })

  it('passes multiple audio frames through to onAudio in strictly ascending seq order', () => {
    const seqs: number[] = []
    streamChat(basePayload, { onAudio: (frame) => seqs.push(frame.seq) })
    const socket = sockets[0]
    socket.open()

    socket.emit({ type: 'audio', seq: 0, mime: 'audio/wav', data: 'AA' })
    socket.emit({ type: 'audio', seq: 1, mime: 'audio/wav', data: 'BB' })
    socket.emit({ type: 'audio', seq: 2, mime: 'audio/wav', data: 'CC' })

    expect(seqs).toEqual([0, 1, 2])
  })

  it('token frame with a missing text field defaults to an empty string rather than "undefined"', () => {
    const onToken = vi.fn()
    streamChat(basePayload, { onToken })
    const socket = sockets[0]
    socket.open()

    socket.emit({ type: 'token' })

    expect(onToken).toHaveBeenCalledWith('')
  })

  it('done frame: dispatches to onDone with the full frame shape and does not call onError', () => {
    const onDone = vi.fn()
    const onError = vi.fn()
    streamChat(basePayload, { onDone, onError })
    const socket = sockets[0]
    socket.open()

    socket.emit({
      type: 'done',
      cancelled: false,
      tts_failed: 1,
      response: { reply: 'Hi there.', turn_id: 't-1' },
    })

    expect(onDone).toHaveBeenCalledTimes(1)
    expect(onDone).toHaveBeenCalledWith(
      expect.objectContaining({
        cancelled: false,
        tts_failed: 1,
        response: expect.objectContaining({ reply: 'Hi there.', turn_id: 't-1' }),
      }),
    )
    expect(onError).not.toHaveBeenCalled()
  })

  it('a done frame after onclose does not double-fire onConnectError (settled latches)', () => {
    const onConnectError = vi.fn()
    const onDone = vi.fn()
    streamChat(basePayload, { onDone, onConnectError })
    const socket = sockets[0]
    socket.open()
    socket.emit({ type: 'done', cancelled: false, response: { reply: 'hi' } })
    expect(onDone).toHaveBeenCalledTimes(1)

    // The real transport now closes normally after done; must not also
    // report a connect error.
    socket.close()
    expect(onConnectError).not.toHaveBeenCalled()
  })

  it('error frame: reports message, fallback_to_post, and kind to onError', () => {
    const onError = vi.fn()
    streamChat(basePayload, { onError })
    const socket = sockets[0]
    socket.open()

    socket.emit({ type: 'error', message: 'boom', fallback_to_post: true, kind: 'rate_limited' })

    expect(onError).toHaveBeenCalledWith('boom', true, 'rate_limited')
  })

  it('error frame with missing message/fallback/kind falls back to defaults', () => {
    const onError = vi.fn()
    streamChat(basePayload, { onError })
    const socket = sockets[0]
    socket.open()

    socket.emit({ type: 'error' })

    expect(onError).toHaveBeenCalledWith('stream error', false, undefined)
  })

  it('an error frame does not also fire onConnectError on the subsequent close', () => {
    const onError = vi.fn()
    const onConnectError = vi.fn()
    streamChat(basePayload, { onError, onConnectError })
    const socket = sockets[0]
    socket.open()

    socket.emit({ type: 'error', message: 'boom' })
    socket.close()

    expect(onError).toHaveBeenCalledTimes(1)
    expect(onConnectError).not.toHaveBeenCalled()
  })

  it('malformed JSON is silently ignored (no handler fires, no throw)', () => {
    const handlers: StreamChatHandlers = {
      onSession: vi.fn(),
      onToken: vi.fn(),
      onDone: vi.fn(),
      onError: vi.fn(),
    }
    streamChat(basePayload, handlers)
    const socket = sockets[0]
    socket.open()

    expect(() => socket.emitRaw('{not valid json')).not.toThrow()

    expect(handlers.onSession).not.toHaveBeenCalled()
    expect(handlers.onToken).not.toHaveBeenCalled()
    expect(handlers.onDone).not.toHaveBeenCalled()
    expect(handlers.onError).not.toHaveBeenCalled()
  })

  it('a frame with a non-string type is ignored', () => {
    const onDone = vi.fn()
    streamChat(basePayload, { onDone })
    const socket = sockets[0]
    socket.open()

    expect(() => socket.emit({ type: 42 })).not.toThrow()
    expect(onDone).not.toHaveBeenCalled()
  })

  it('an unknown frame type is tolerated and does not throw or fire any known handler', () => {
    const handlers: StreamChatHandlers = {
      onSession: vi.fn(),
      onDone: vi.fn(),
      onError: vi.fn(),
    }
    streamChat(basePayload, handlers)
    const socket = sockets[0]
    socket.open()

    expect(() => socket.emit({ type: 'some_future_frame', foo: 'bar' })).not.toThrow()

    expect(handlers.onSession).not.toHaveBeenCalled()
    expect(handlers.onDone).not.toHaveBeenCalled()
    expect(handlers.onError).not.toHaveBeenCalled()
  })

  it('an unknown frame still disarms the first-frame watchdog (proves the stream is live)', () => {
    vi.useFakeTimers()
    const onConnectError = vi.fn()
    streamChat(basePayload, { onConnectError })
    const socket = sockets[0]
    socket.open()

    socket.emit({ type: 'some_future_frame' })
    vi.advanceTimersByTime(15_000)

    expect(onConnectError).not.toHaveBeenCalled()
  })

  it('a socket that opens but never produces a frame trips the first-frame watchdog and closes the socket', () => {
    vi.useFakeTimers()
    const onConnectError = vi.fn()
    streamChat(basePayload, { onConnectError })
    const socket = sockets[0]
    socket.open()

    vi.advanceTimersByTime(15_000)

    expect(onConnectError).toHaveBeenCalledTimes(1)
    expect(socket.readyState).toBe(FakeWebSocket.CLOSED)
  })

  it('a live frame before the watchdog deadline disarms it permanently, even across a long later pause', () => {
    vi.useFakeTimers()
    const onConnectError = vi.fn()
    const onDone = vi.fn()
    streamChat(basePayload, { onConnectError, onDone })
    const socket = sockets[0]
    socket.open()
    socket.emit({ type: 'session', session_id: 'sess-1' })

    vi.advanceTimersByTime(60_000)
    socket.emit({ type: 'done', cancelled: false, response: { reply: 'ok' } })

    expect(onConnectError).not.toHaveBeenCalled()
    expect(onDone).toHaveBeenCalledTimes(1)
  })

  it('onerror before any frame reports a connect error and does not throw closing an already-erroring socket', () => {
    const onConnectError = vi.fn()
    streamChat(basePayload, { onConnectError })
    const socket = sockets[0]
    socket.open()

    socket.failToConnect()

    expect(onConnectError).toHaveBeenCalledTimes(1)
  })

  it('onclose before any frame (no done) reports a connect error', () => {
    const onConnectError = vi.fn()
    const onDone = vi.fn()
    streamChat(basePayload, { onConnectError, onDone })
    const socket = sockets[0]
    socket.open()

    socket.close()

    expect(onConnectError).toHaveBeenCalledTimes(1)
    expect(onDone).not.toHaveBeenCalled()
  })

  it('a synchronous WebSocket constructor throw reports a connect error and returns a no-op cancel handle', () => {
    class ExplodingWebSocket {
      constructor() {
        throw new Error('no websocket here')
      }
    }
    vi.stubGlobal('WebSocket', ExplodingWebSocket)
    const onConnectError = vi.fn()

    const handle = streamChat(basePayload, { onConnectError })

    expect(onConnectError).toHaveBeenCalledTimes(1)
    expect(() => handle.cancel()).not.toThrow()
  })

  describe('cancel()', () => {
    it('sends a cancel frame when the socket is open', () => {
      const handle = streamChat(basePayload, {})
      const socket = sockets[0]
      socket.open()

      handle.cancel()

      expect(socket.sent).toHaveLength(2) // initial chat frame + cancel
      expect(JSON.parse(socket.sent[1])).toEqual({ type: 'cancel' })
    })

    it('does not send when the socket is not open, and does not throw', () => {
      const handle = streamChat(basePayload, {})
      const socket = sockets[0]
      // Never opened.

      expect(() => handle.cancel()).not.toThrow()
      expect(socket.sent).toHaveLength(0)
    })

    it('disarms the first-frame watchdog so cancelling before any frame does not later fire onConnectError', () => {
      vi.useFakeTimers()
      const onConnectError = vi.fn()
      const handle = streamChat(basePayload, { onConnectError })
      const socket = sockets[0]
      socket.open()

      handle.cancel()
      vi.advanceTimersByTime(15_000)

      expect(onConnectError).not.toHaveBeenCalled()
    })

    it('cancel is safe to call twice (second call sends another cancel frame, no throw)', () => {
      const handle = streamChat(basePayload, {})
      const socket = sockets[0]
      socket.open()

      handle.cancel()
      expect(() => handle.cancel()).not.toThrow()
      expect(socket.sent.filter((s) => JSON.parse(s).type === 'cancel')).toHaveLength(2)
    })
  })
})
