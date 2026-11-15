// @vitest-environment jsdom
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { runVoiceTurn } from './voiceTurn'
import { diagnostics } from '../diagnostics'
import { kurisuBridge } from '../vrm/kurisuBridge'
import type { DoneFrame } from '../streaming'

/** Minimal controllable WebSocket stub: the test holds a reference to the
 * instance runVoiceTurn/streamChat creates (via `sockets`) so it can drive
 * onopen/onmessage/onclose/onerror directly instead of a real network round
 * trip. Mirrors the frame shapes streaming.ts's onmessage switch parses. */
class FakeWebSocket {
  static OPEN = 1
  static CLOSED = 3
  readyState = 0
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
    // Mirrors a real close enough for the watchdog path: state flips and
    // onclose fires (streamChat's settled flag must absorb it silently).
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

  closeWithoutDone() {
    this.readyState = FakeWebSocket.CLOSED
    this.onclose?.()
  }

  failToConnect() {
    this.onerror?.()
  }
}

let sockets: FakeWebSocket[] = []

const doneFrame = (overrides?: Partial<DoneFrame>): DoneFrame => ({
  cancelled: false,
  response: { reply: 'Hmph.' },
  ...overrides,
})

describe('runVoiceTurn', () => {
  let lipSync: {
    beginStream: ReturnType<typeof vi.fn>
    enqueue: ReturnType<typeof vi.fn>
    endStream: ReturnType<typeof vi.fn>
    stop: ReturnType<typeof vi.fn>
    isSpeaking: ReturnType<typeof vi.fn>
    play: ReturnType<typeof vi.fn>
  }
  let expressionsSet: ReturnType<typeof vi.fn>
  let react: ReturnType<typeof vi.fn>

  beforeEach(() => {
    sockets = []
    vi.stubGlobal('WebSocket', FakeWebSocket)
    diagnostics.resetForTest()

    lipSync = {
      beginStream: vi.fn(),
      enqueue: vi.fn().mockResolvedValue(undefined),
      endStream: vi.fn(),
      stop: vi.fn(),
      isSpeaking: vi.fn().mockReturnValue(false),
      play: vi.fn(),
    }
    expressionsSet = vi.fn()
    react = vi.fn()
    kurisuBridge.lipSync = lipSync
    kurisuBridge.expressions = { set: expressionsSet, available: [], list: vi.fn() }
    kurisuBridge.react = react
  })

  afterEach(() => {
    vi.unstubAllGlobals()
    kurisuBridge.lipSync = null
    kurisuBridge.expressions = null
    kurisuBridge.react = null
  })

  const baseInput = {
    message: 'hi',
    sessionId: null,
    userId: 'u1',
    audioEnabled: true,
  }

  it('happy path: demuxes sentence/audio frames, dispatches emotion, and ends the stream on done', async () => {
    const onSession = vi.fn()
    const onSentence = vi.fn()
    const onAudioStart = vi.fn()
    const onDone = vi.fn()
    const onFailure = vi.fn()

    runVoiceTurn(baseInput, { onSession, onSentence, onAudioStart, onDone, onFailure })

    const socket = sockets[0]
    socket.open()
    socket.emit({ type: 'session', session_id: 'sess-1' })
    socket.emit({ type: 'sentence', seq: 0, en: 'Hi.', ja: 'こんにちは。', emotion: 'happy', emotion_intensity: 0.7 })
    socket.emit({ type: 'audio', seq: 0, mime: 'audio/wav', data: 'QUJD' })
    socket.emit({ type: 'done', cancelled: false, response: { reply: 'Hi.' } })

    expect(onSession).toHaveBeenCalledWith('sess-1')
    // Emotion dispatch happened inside runVoiceTurn, once, with the frame's
    // own intensity (not the 0.8 default).
    expect(expressionsSet).toHaveBeenCalledWith('happy', 0.7, 2500)
    expect(react).toHaveBeenCalledWith('happy', 0.7)
    expect(onSentence).toHaveBeenCalledTimes(1)
    expect(onSentence.mock.calls[0][0]).toMatchObject({ seq: 0, ja: 'こんにちは。' })

    expect(lipSync.beginStream).toHaveBeenCalledTimes(1)
    expect(onAudioStart).toHaveBeenCalledTimes(1)
    // The audio chunk carries the emotion recorded for the same seq.
    expect(lipSync.enqueue).toHaveBeenCalledWith(
      expect.objectContaining({ base64Audio: 'QUJD', seq: 0, emotion: 'happy' }),
    )

    expect(lipSync.endStream).toHaveBeenCalledTimes(1)
    expect(lipSync.stop).not.toHaveBeenCalled()
    expect(onDone).toHaveBeenCalledTimes(1)
    expect(onFailure).not.toHaveBeenCalled()
  })

  it('cancel: a cancelled done frame stops lip sync immediately instead of draining', () => {
    const onDone = vi.fn()
    runVoiceTurn(baseInput, { onDone })

    const socket = sockets[0]
    socket.open()
    socket.emit({ type: 'session', session_id: 'sess-1' })
    socket.emit({ type: 'audio', seq: 0, mime: 'audio/wav', data: 'QUJD' })
    socket.emit({ type: 'done', ...doneFrame({ cancelled: true }) })

    expect(lipSync.stop).toHaveBeenCalledTimes(1)
    expect(lipSync.endStream).not.toHaveBeenCalled()
    expect(onDone).toHaveBeenCalledWith(expect.objectContaining({ cancelled: true }), 0)
  })

  it('mid-stream error: reports a mid-stream failure once frames have already flowed, never a POST-safe one', () => {
    const onFailure = vi.fn()
    runVoiceTurn(baseInput, { onFailure })

    const socket = sockets[0]
    socket.open()
    socket.emit({ type: 'session', session_id: 'sess-1' })
    socket.emit({ type: 'audio', seq: 0, mime: 'audio/wav', data: 'QUJD' })
    // Server-sent error frame, mid-turn, even if it happens to flag
    // fallback_to_post — once content has started, a silent POST re-send
    // would duplicate the turn, so this must still classify as mid-stream.
    socket.emit({ type: 'error', message: 'boom', fallback_to_post: true })

    expect(onFailure).toHaveBeenCalledTimes(1)
    expect(onFailure).toHaveBeenCalledWith({ kind: 'mid-stream' })
    // onError (unlike onConnectError) always cuts already-scheduled audio
    // short once it has begun.
    expect(lipSync.stop).toHaveBeenCalledTimes(1)
  })

  it('provider error frame with structured kind: reports mid-stream with an honest cause, never a POST-safe connect', () => {
    const onFailure = vi.fn()
    runVoiceTurn(baseInput, { onFailure })

    const socket = sockets[0]
    socket.open()
    socket.emit({ type: 'session', session_id: 'sess-1' })
    // Backend provider failure: session frame already flowed, then the turn
    // died on a 429. fallback_to_post rides along for wire compatibility,
    // but must NOT reclassify this as 'connect' (POST would hit the same
    // rate limit).
    socket.emit({
      type: 'error',
      message: 'ResourceExhausted: 429 quota exceeded',
      fallback_to_post: true,
      kind: 'rate_limited',
    })

    expect(onFailure).toHaveBeenCalledTimes(1)
    expect(onFailure).toHaveBeenCalledWith({ kind: 'mid-stream', cause: 'rate_limited' })
  })

  it('provider error without a kind field (old backend): sniffs rate-limit phrasing from the message', () => {
    const onFailure = vi.fn()
    runVoiceTurn(baseInput, { onFailure })

    const socket = sockets[0]
    socket.open()
    socket.emit({ type: 'session', session_id: 'sess-1' })
    socket.emit({ type: 'error', message: '429 RESOURCE_EXHAUSTED: Quota exceeded', fallback_to_post: true })

    expect(onFailure).toHaveBeenCalledWith({ kind: 'mid-stream', cause: 'rate_limited' })
  })

  it('generic provider_error kind maps through; unrecognizable errors keep the plain mid-stream shape', () => {
    const onFailure = vi.fn()
    runVoiceTurn(baseInput, { onFailure })

    const socket = sockets[0]
    socket.open()
    socket.emit({ type: 'session', session_id: 'sess-1' })
    socket.emit({ type: 'error', message: 'model exploded', kind: 'provider_error' })

    expect(onFailure).toHaveBeenCalledWith({ kind: 'mid-stream', cause: 'provider_error' })
  })

  it('mid-stream connection loss: a close before done after frames flowed reports mid-stream without stopping playback', () => {
    const onFailure = vi.fn()
    runVoiceTurn(baseInput, { onFailure })

    const socket = sockets[0]
    socket.open()
    socket.emit({ type: 'session', session_id: 'sess-1' })
    socket.emit({ type: 'audio', seq: 0, mime: 'audio/wav', data: 'QUJD' })
    // Connection just dies — no 'error' frame, no 'done' frame.
    socket.closeWithoutDone()

    expect(onFailure).toHaveBeenCalledTimes(1)
    expect(onFailure).toHaveBeenCalledWith({ kind: 'mid-stream' })
    // Matches the original ChatInterface behavior: onConnectError does NOT
    // cut audio short even mid-stream (only the explicit onError path does).
    expect(lipSync.stop).not.toHaveBeenCalled()
  })

  it('pre-stream connect error: reports a POST-fallback-safe failure before any frame arrives', () => {
    const onFailure = vi.fn()
    const onSession = vi.fn()
    runVoiceTurn(baseInput, { onFailure, onSession })

    const socket = sockets[0]
    // Never opens / never sends a session frame — e.g. the server refused
    // the handshake outright.
    socket.failToConnect()

    expect(onSession).not.toHaveBeenCalled()
    expect(onFailure).toHaveBeenCalledTimes(1)
    expect(onFailure).toHaveBeenCalledWith({ kind: 'connect' })
  })

  it('pre-stream server error with fallback flag: reports a POST-fallback-safe failure', () => {
    const onFailure = vi.fn()
    runVoiceTurn(baseInput, { onFailure })

    const socket = sockets[0]
    socket.open()
    // Server reports streaming is unavailable before ever sending 'session'.
    socket.emit({ type: 'error', message: 'streaming disabled', fallback_to_post: true })

    expect(onFailure).toHaveBeenCalledWith({ kind: 'connect' })
  })

  it('WebSocket constructor throw (no WS support/blocked): reports connect, not mid-stream', () => {
    // Regression coverage: streamChat catches a synchronous constructor
    // throw and calls onConnectError with zero frames ever having flowed —
    // must classify as POST-fallback-safe 'connect', never 'mid-stream'.
    class ExplodingWebSocket {
      constructor() {
        throw new Error('no websocket here')
      }
    }
    vi.stubGlobal('WebSocket', ExplodingWebSocket)
    const onFailure = vi.fn()

    runVoiceTurn(baseInput, { onFailure })

    expect(onFailure).toHaveBeenCalledTimes(1)
    expect(onFailure).toHaveBeenCalledWith({ kind: 'connect' })
  })

  it('first-frame watchdog: a socket that opens but never produces a frame settles as connect', () => {
    // Regression coverage for the stuck-thinking bug: a hung server (or a
    // half-open connection after sleep/resume) fires neither onerror nor
    // onclose, which used to leave the turn spinning forever. The watchdog
    // must land in the POST-fallback-safe 'connect' class — zero frames
    // flowed — never 'mid-stream'.
    vi.useFakeTimers()
    try {
      const onFailure = vi.fn()
      runVoiceTurn(baseInput, { onFailure })

      const socket = sockets[0]
      socket.open()
      // No frame ever arrives.
      vi.advanceTimersByTime(15_000)

      expect(onFailure).toHaveBeenCalledTimes(1)
      expect(onFailure).toHaveBeenCalledWith({ kind: 'connect' })
      expect(socket.readyState).toBe(FakeWebSocket.CLOSED)
    } finally {
      vi.useRealTimers()
    }
  })

  it('first-frame watchdog: disarmed by the first frame — a slow turn after a live start never misfires', () => {
    vi.useFakeTimers()
    try {
      const onFailure = vi.fn()
      const onDone = vi.fn()
      runVoiceTurn(baseInput, { onFailure, onDone })

      const socket = sockets[0]
      socket.open()
      socket.emit({ type: 'session', session_id: 'sess-1' })
      // Long tool-loop pause well past the watchdog deadline: the stream is
      // live, so the timeout must not tear it down.
      vi.advanceTimersByTime(60_000)
      socket.emit({ type: 'done', ...doneFrame() })

      expect(onFailure).not.toHaveBeenCalled()
      expect(onDone).toHaveBeenCalledTimes(1)
    } finally {
      vi.useRealTimers()
    }
  })

  it('respects audioEnabled: never touches lipSync when audio is off', () => {
    const onDone = vi.fn()
    runVoiceTurn({ ...baseInput, audioEnabled: false }, { onDone })

    const socket = sockets[0]
    socket.open()
    socket.emit({ type: 'session', session_id: 'sess-1' })
    socket.emit({ type: 'sentence', seq: 0, en: 'Hi.', ja: 'a', emotion: 'happy', emotion_intensity: 0.5 })
    socket.emit({ type: 'audio', seq: 0, mime: 'audio/wav', data: 'QUJD' })
    socket.emit({ type: 'done', ...doneFrame() })

    expect(lipSync.beginStream).not.toHaveBeenCalled()
    expect(lipSync.enqueue).not.toHaveBeenCalled()
    // endStream is only called when audioStreamBegun latched true.
    expect(lipSync.endStream).not.toHaveBeenCalled()
    // Emotion dispatch is independent of audioEnabled.
    expect(expressionsSet).toHaveBeenCalledWith('happy', 0.5, 2500)
  })

  describe('tts_failed consumption', () => {
    it('passes ttsFailed=0 to onDone and never increments the counter when the frame omits/zeros it', () => {
      const onDone = vi.fn()
      runVoiceTurn(baseInput, { onDone })

      const socket = sockets[0]
      socket.open()
      socket.emit({ type: 'session', session_id: 'sess-1' })
      socket.emit({ type: 'done', ...doneFrame() })

      expect(onDone).toHaveBeenCalledTimes(1)
      expect(onDone.mock.calls[0][1]).toBe(0)
      expect(diagnostics.snapshot().counters['voice.tts_failed_sentences']).toBeUndefined()
    })

    it('surfaces a positive tts_failed count to onDone and counts it in diagnostics', () => {
      const onDone = vi.fn()
      runVoiceTurn(baseInput, { onDone })

      const socket = sockets[0]
      socket.open()
      socket.emit({ type: 'session', session_id: 'sess-1' })
      socket.emit({ type: 'done', ...doneFrame({ tts_failed: 2 }) })

      expect(onDone).toHaveBeenCalledTimes(1)
      const [frame, ttsFailed] = onDone.mock.calls[0]
      expect(frame.tts_failed).toBe(2)
      expect(ttsFailed).toBe(2)
      expect(diagnostics.snapshot().counters['voice.tts_failed_sentences']).toBe(1)
      expect(diagnostics.snapshot().lastDetail['voice.tts_failed_sentences']).toBe('2')
    })
  })

  describe('no-lipsync audio drop counter', () => {
    it('counts once per turn even when multiple audio frames arrive with no lipSync.enqueue registered', () => {
      kurisuBridge.lipSync = null
      const onDone = vi.fn()
      runVoiceTurn(baseInput, { onDone })

      const socket = sockets[0]
      socket.open()
      socket.emit({ type: 'session', session_id: 'sess-1' })
      socket.emit({ type: 'audio', seq: 0, mime: 'audio/wav', data: 'QUJD' })
      socket.emit({ type: 'audio', seq: 1, mime: 'audio/wav', data: 'QUJD' })
      socket.emit({ type: 'audio', seq: 2, mime: 'audio/wav', data: 'QUJD' })
      socket.emit({ type: 'done', ...doneFrame() })

      expect(diagnostics.snapshot().counters['voice.audio_dropped_no_lipsync']).toBe(1)
    })

    it('does not count when lipSync.enqueue is present', () => {
      const onDone = vi.fn()
      runVoiceTurn(baseInput, { onDone })

      const socket = sockets[0]
      socket.open()
      socket.emit({ type: 'session', session_id: 'sess-1' })
      socket.emit({ type: 'audio', seq: 0, mime: 'audio/wav', data: 'QUJD' })
      socket.emit({ type: 'done', ...doneFrame() })

      expect(diagnostics.snapshot().counters['voice.audio_dropped_no_lipsync']).toBeUndefined()
    })

    it('a fresh turn re-arms the once-per-turn guard (a second turn counts again)', () => {
      kurisuBridge.lipSync = null
      runVoiceTurn(baseInput, {})
      const socket1 = sockets[0]
      socket1.open()
      socket1.emit({ type: 'session', session_id: 'sess-1' })
      socket1.emit({ type: 'audio', seq: 0, mime: 'audio/wav', data: 'QUJD' })
      socket1.emit({ type: 'done', ...doneFrame() })
      expect(diagnostics.snapshot().counters['voice.audio_dropped_no_lipsync']).toBe(1)

      runVoiceTurn(baseInput, {})
      const socket2 = sockets[1]
      socket2.open()
      socket2.emit({ type: 'session', session_id: 'sess-2' })
      socket2.emit({ type: 'audio', seq: 0, mime: 'audio/wav', data: 'QUJD' })
      socket2.emit({ type: 'done', ...doneFrame() })
      expect(diagnostics.snapshot().counters['voice.audio_dropped_no_lipsync']).toBe(2)
    })
  })
})
