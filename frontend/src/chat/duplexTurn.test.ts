// @vitest-environment jsdom
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { kurisuBridge } from '../vrm/kurisuBridge'

/** Minimal controllable WebSocket stub, mirroring voiceTurn.test.ts's
 * FakeWebSocket: the test holds a reference (via `sockets`) to drive
 * onopen/onmessage/onclose/onerror directly. Duplex additionally records
 * binary sends (mic frames) separately from JSON text sends (control
 * frames), since the real protocol carries both. */
class FakeWebSocket {
  static CONNECTING = 0
  static OPEN = 1
  static CLOSED = 3
  readyState = FakeWebSocket.CONNECTING
  binaryType = 'blob'
  sentText: string[] = []
  sentBinary: ArrayBuffer[] = []
  onopen: (() => void) | null = null
  onmessage: ((event: { data: string | ArrayBuffer }) => void) | null = null
  onerror: (() => void) | null = null
  onclose: (() => void) | null = null

  constructor(public url: string) {
    sockets.push(this)
  }

  send(data: string | ArrayBuffer) {
    if (typeof data === 'string') this.sentText.push(data)
    else this.sentBinary.push(data)
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

  failToConnect() {
    this.onerror?.()
  }
}

let sockets: FakeWebSocket[] = []

// --- DuplexMic mock: the real class needs getUserMedia/AudioContext, which
// jsdom doesn't provide. Mirrors useMicRecorder.test.ts's MicRecorder mock
// pattern (vi.mock the module, expose constructed instances for assertion).
let micInstances: Array<{ start: ReturnType<typeof vi.fn>; stop: ReturnType<typeof vi.fn>; onFrame: ((f: ArrayBuffer) => void) | null }> = []

vi.mock('../audio/duplexMic', () => {
  class MockDuplexMic {
    onFrame: ((f: ArrayBuffer) => void) | null = null
    onLevel: ((rms: number, dtMs: number) => void) | null = null
    start = vi.fn().mockResolvedValue(undefined)
    stop = vi.fn()
    constructor() {
      micInstances.push(this)
    }
  }
  return { DuplexMic: MockDuplexMic, DUPLEX_SAMPLE_RATE: 16000 }
})

const playBackchannelMock = vi.fn().mockResolvedValue(true)
vi.mock('../audio/fillerPlayer', () => ({
  playBackchannel: (...args: unknown[]) => playBackchannelMock(...args),
}))

// Import after the mock is registered.
import { startDuplexSession } from './duplexTurn'

describe('startDuplexSession', () => {
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
    micInstances = []
    playBackchannelMock.mockClear()
    vi.stubGlobal('WebSocket', FakeWebSocket)

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

  const baseInput = { sessionId: null, userId: 'u1' }

  it('sends the open handshake on connect and reports capability + starts mic capture on the session frame', async () => {
    const onCapability = vi.fn()
    const onSessionId = vi.fn()
    startDuplexSession(baseInput, { onCapability, onSessionId })

    const socket = sockets[0]
    socket.open()
    expect(socket.sentText).toHaveLength(1)
    const handshake = JSON.parse(socket.sentText[0])
    expect(handshake).toMatchObject({ type: 'start', session_id: null, user_id: 'u1', encoding: 'pcm16' })

    socket.emit({ type: 'session', session_id: 'sess-1', duplex: true, sample_rate: 16000, encoding: 'pcm16' })
    expect(onCapability).toHaveBeenCalledWith(true)
    expect(onSessionId).toHaveBeenCalledWith('sess-1')

    // Mic capture starts asynchronously off the session frame.
    await Promise.resolve()
    await Promise.resolve()
    expect(micInstances).toHaveLength(1)
    expect(micInstances[0].start).toHaveBeenCalledTimes(1)
  })

  it('mic frames are forwarded as binary WebSocket sends', async () => {
    startDuplexSession(baseInput, {})
    const socket = sockets[0]
    socket.open()
    socket.emit({ type: 'session', session_id: 'sess-1' })
    await Promise.resolve()
    await Promise.resolve()

    const mic = micInstances[0]
    const frame = new ArrayBuffer(4)
    mic.onFrame?.(frame)
    expect(socket.sentBinary).toEqual([frame])
  })

  it('demuxes state/transcript/sentence/audio/done frames like the streaming turn vocabulary', async () => {
    const onState = vi.fn()
    const onTranscript = vi.fn()
    const onSentence = vi.fn()
    const onDone = vi.fn()
    startDuplexSession(baseInput, { onState, onTranscript, onSentence, onDone })

    const socket = sockets[0]
    socket.open()
    socket.emit({ type: 'session', session_id: 'sess-1' })
    socket.emit({ type: 'state', state: 'processing' })
    socket.emit({ type: 'transcript', text: 'hello there' })
    socket.emit({ type: 'sentence', seq: 0, en: 'Hi.', ja: 'こんにちは。', emotion: 'happy', emotion_intensity: 0.7 })
    socket.emit({ type: 'audio', seq: 0, mime: 'audio/wav', data: 'QUJD' })
    socket.emit({ type: 'done', cancelled: false, response: { reply: 'Hi.' } })

    expect(onState).toHaveBeenCalledWith('processing')
    expect(onTranscript).toHaveBeenCalledWith('hello there')
    expect(expressionsSet).toHaveBeenCalledWith('happy', 0.7, 2500)
    expect(react).toHaveBeenCalledWith('happy', 0.7)
    expect(onSentence.mock.calls[0][0]).toMatchObject({ seq: 0, ja: 'こんにちは。' })
    expect(lipSync.beginStream).toHaveBeenCalledTimes(1)
    expect(lipSync.enqueue).toHaveBeenCalledWith(
      expect.objectContaining({ base64Audio: 'QUJD', seq: 0, emotion: 'happy' }),
    )
    expect(lipSync.endStream).toHaveBeenCalledTimes(1)
    expect(onDone).toHaveBeenCalledTimes(1)
  })

  it('a cancelled done frame stops lip sync immediately instead of draining', () => {
    startDuplexSession(baseInput, {})
    const socket = sockets[0]
    socket.open()
    socket.emit({ type: 'session', session_id: 'sess-1' })
    socket.emit({ type: 'audio', seq: 0, mime: 'audio/wav', data: 'QUJD' })
    socket.emit({ type: 'done', cancelled: true, response: { reply: 'partial [interrupted]' } })

    expect(lipSync.stop).toHaveBeenCalledTimes(1)
    expect(lipSync.endStream).not.toHaveBeenCalled()
  })

  it('a listening/user_speaking state frame cuts off any in-progress playback (barge-in)', () => {
    startDuplexSession(baseInput, {})
    const socket = sockets[0]
    socket.open()
    socket.emit({ type: 'session', session_id: 'sess-1' })
    socket.emit({ type: 'audio', seq: 0, mime: 'audio/wav', data: 'QUJD' })
    lipSync.stop.mockClear()
    socket.emit({ type: 'state', state: 'user_speaking' })
    expect(lipSync.stop).toHaveBeenCalledTimes(1)
  })

  it('a barge-in that cut in-flight audio dispatches kurisu:barge-in (reflex layer)', () => {
    const bargeInListener = vi.fn()
    window.addEventListener('kurisu:barge-in', bargeInListener)
    try {
      startDuplexSession(baseInput, {})
      const socket = sockets[0]
      socket.open()
      socket.emit({ type: 'session', session_id: 'sess-1' })
      socket.emit({ type: 'audio', seq: 0, mime: 'audio/wav', data: 'QUJD' })
      socket.emit({ type: 'state', state: 'user_speaking' })
      expect(bargeInListener).toHaveBeenCalledTimes(1)
    } finally {
      window.removeEventListener('kurisu:barge-in', bargeInListener)
    }
  })

  it('a state flip with nothing to interrupt does NOT dispatch kurisu:barge-in', () => {
    const bargeInListener = vi.fn()
    window.addEventListener('kurisu:barge-in', bargeInListener)
    try {
      startDuplexSession(baseInput, {})
      const socket = sockets[0]
      socket.open()
      socket.emit({ type: 'session', session_id: 'sess-1' })
      // Routine post-connect transition: no audio has ever streamed.
      socket.emit({ type: 'state', state: 'listening' })
      // A fully-drained turn resets the audio latch — the normal return to
      // listening after done isn't a barge-in either.
      socket.emit({ type: 'audio', seq: 0, mime: 'audio/wav', data: 'QUJD' })
      socket.emit({ type: 'done', cancelled: false, response: { reply: 'Hi.' } })
      socket.emit({ type: 'state', state: 'listening' })
      expect(bargeInListener).not.toHaveBeenCalled()
    } finally {
      window.removeEventListener('kurisu:barge-in', bargeInListener)
    }
  })

  it('a backchannel frame plays an acknowledgement clip and queues a listening nod', () => {
    const nodListener = vi.fn()
    window.addEventListener('kurisu:listening-nod', nodListener)
    try {
      startDuplexSession(baseInput, {})
      const socket = sockets[0]
      socket.onopen?.()
      socket.emit({ type: 'session', session_id: 'sess-1' })
      socket.emit({ type: 'backchannel' })
      expect(playBackchannelMock).toHaveBeenCalledTimes(1)
      expect(nodListener).toHaveBeenCalledTimes(1)
    } finally {
      window.removeEventListener('kurisu:listening-nod', nodListener)
    }
  })

  it('a duplex_disabled error frame reports capability=false', () => {
    const onCapability = vi.fn()
    const onError = vi.fn()
    startDuplexSession(baseInput, { onCapability, onError })
    const socket = sockets[0]
    socket.open()
    socket.emit({ type: 'error', message: 'Full-duplex voice is disabled on this server.', kind: 'duplex_disabled' })

    expect(onCapability).toHaveBeenCalledWith(false)
    expect(onError).toHaveBeenCalledWith('Full-duplex voice is disabled on this server.', 'duplex_disabled')
  })

  it('a connect failure before any frame reports capability=false', () => {
    const onCapability = vi.fn()
    startDuplexSession(baseInput, { onCapability })
    const socket = sockets[0]
    socket.failToConnect()
    expect(onCapability).toHaveBeenCalledWith(false)
  })

  it('capability is only ever reported once per session', () => {
    const onCapability = vi.fn()
    startDuplexSession(baseInput, { onCapability })
    const socket = sockets[0]
    socket.open()
    socket.emit({ type: 'session', session_id: 'sess-1' })
    socket.emit({ type: 'error', message: 'late error', kind: 'protocol_error' })
    expect(onCapability).toHaveBeenCalledTimes(1)
    expect(onCapability).toHaveBeenCalledWith(true)
  })

  it('stop() sends a stop control frame and releases the mic', async () => {
    const handle = startDuplexSession(baseInput, {})
    const socket = sockets[0]
    socket.open()
    socket.emit({ type: 'session', session_id: 'sess-1' })
    await Promise.resolve()
    await Promise.resolve()

    handle.stop()
    const lastText = socket.sentText[socket.sentText.length - 1]
    expect(JSON.parse(lastText)).toMatchObject({ type: 'stop' })
    expect(micInstances[0].stop).toHaveBeenCalledTimes(1)
  })

  it('cancel() sends a cancel control frame without closing the socket', () => {
    const handle = startDuplexSession(baseInput, {})
    const socket = sockets[0]
    socket.open()
    handle.cancel()
    const lastText = socket.sentText[socket.sentText.length - 1]
    expect(JSON.parse(lastText)).toMatchObject({ type: 'cancel' })
    expect(socket.readyState).not.toBe(FakeWebSocket.CLOSED)
  })
})
