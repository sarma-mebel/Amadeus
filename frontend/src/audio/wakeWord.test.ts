// @vitest-environment jsdom
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

/** Same controllable WebSocket stub as duplexTurn.test.ts. */
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

  open() {
    this.readyState = FakeWebSocket.OPEN
    this.onopen?.()
  }

  emit(frame: Record<string, unknown>) {
    this.onmessage?.({ data: JSON.stringify(frame) })
  }
}

let sockets: FakeWebSocket[] = []

let micInstances: Array<{
  start: ReturnType<typeof vi.fn>
  stop: ReturnType<typeof vi.fn>
  onFrame: ((f: ArrayBuffer) => void) | null
}> = []
// Set true to make the next DuplexMic.start() reject (permission denied).
let micFailNext = false

vi.mock('./duplexMic', () => {
  class MockDuplexMic {
    onFrame: ((f: ArrayBuffer) => void) | null = null
    onLevel: ((rms: number, dtMs: number) => void) | null = null
    stop = vi.fn()
    start = vi.fn().mockImplementation(() => {
      if (micFailNext) {
        micFailNext = false
        return Promise.reject(new Error('denied'))
      }
      return Promise.resolve()
    })
    constructor() {
      micInstances.push(this)
    }
  }
  return { DuplexMic: MockDuplexMic, DUPLEX_SAMPLE_RATE: 16000 }
})

import {
  WAKE_WORD_EVENT,
  WAKE_WORD_STORAGE_KEY,
  loadWakeWordEnabled,
  saveWakeWordEnabled,
  startWakeListener,
} from './wakeWord'

const flushMicrotasks = async () => {
  await Promise.resolve()
  await Promise.resolve()
}

describe('wake word storage helpers', () => {
  beforeEach(() => window.localStorage.clear())

  it('defaults off and round-trips the toggle', () => {
    expect(loadWakeWordEnabled()).toBe(false)
    saveWakeWordEnabled(true)
    expect(window.localStorage.getItem(WAKE_WORD_STORAGE_KEY)).toBe('true')
    expect(loadWakeWordEnabled()).toBe(true)
    saveWakeWordEnabled(false)
    expect(loadWakeWordEnabled()).toBe(false)
  })

  it('broadcasts the change event so live hooks react', () => {
    const seen = vi.fn()
    window.addEventListener(WAKE_WORD_EVENT, seen)
    saveWakeWordEnabled(true)
    expect(seen).toHaveBeenCalledTimes(1)
    window.removeEventListener(WAKE_WORD_EVENT, seen)
  })
})

describe('startWakeListener', () => {
  beforeEach(() => {
    sockets = []
    micInstances = []
    micFailNext = false
    vi.stubGlobal('WebSocket', FakeWebSocket as unknown as typeof WebSocket)
  })

  afterEach(() => {
    vi.unstubAllGlobals()
  })

  it('handshakes, streams mic frames after ready, and fires onWake', async () => {
    const onWake = vi.fn()
    startWakeListener({ onWake })
    const socket = sockets[0]
    socket.open()
    expect(JSON.parse(socket.sentText[0]).type).toBe('start')

    socket.emit({ type: 'ready', sample_rate: 16000 })
    await flushMicrotasks()
    expect(micInstances).toHaveLength(1)
    micInstances[0].onFrame?.(new ArrayBuffer(8))
    expect(socket.sentBinary).toHaveLength(1)

    socket.emit({ type: 'wake', score: 0.91 })
    expect(onWake).toHaveBeenCalledWith(0.91)
  })

  it('reports disabled server via onUnavailable and tears down', async () => {
    const onUnavailable = vi.fn()
    startWakeListener({ onWake: vi.fn(), onUnavailable })
    const socket = sockets[0]
    socket.open()
    socket.emit({ type: 'error', kind: 'wake_disabled', message: 'off' })
    expect(onUnavailable).toHaveBeenCalledWith('disabled', 'off')
    expect(socket.readyState).toBe(FakeWebSocket.CLOSED)
  })

  it('close before ready is unavailable; close after ready is retryable', async () => {
    const first = vi.fn()
    startWakeListener({ onWake: vi.fn(), onUnavailable: first })
    sockets[0].open()
    sockets[0].close()
    expect(first).toHaveBeenCalledWith('unavailable', undefined)

    const second = vi.fn()
    startWakeListener({ onWake: vi.fn(), onUnavailable: second })
    const socket = sockets[1]
    socket.open()
    socket.emit({ type: 'ready', sample_rate: 16000 })
    await flushMicrotasks()
    socket.close()
    expect(second).toHaveBeenCalledWith('closed', undefined)
    expect(micInstances[micInstances.length - 1].stop).toHaveBeenCalled()
  })

  it('mic failure reports mic and closes the socket', async () => {
    const onUnavailable = vi.fn()
    startWakeListener({ onWake: vi.fn(), onUnavailable })
    const socket = sockets[0]
    socket.open()
    micFailNext = true
    socket.emit({ type: 'ready', sample_rate: 16000 })
    await flushMicrotasks()
    expect(onUnavailable).toHaveBeenCalledWith('mic', expect.any(String))
  })

  it('stop() tears down mic and socket without callbacks', async () => {
    const onUnavailable = vi.fn()
    const handle = startWakeListener({ onWake: vi.fn(), onUnavailable })
    const socket = sockets[0]
    socket.open()
    socket.emit({ type: 'ready', sample_rate: 16000 })
    await flushMicrotasks()
    handle.stop()
    expect(micInstances[0].stop).toHaveBeenCalled()
    expect(socket.readyState).toBe(FakeWebSocket.CLOSED)
    expect(onUnavailable).not.toHaveBeenCalled()
  })
})
