// @vitest-environment jsdom
import { beforeEach, afterEach, describe, expect, it, vi } from 'vitest'
import { MicMonitor } from './micMonitor'

interface Fakes {
  getUserMedia: ReturnType<typeof vi.fn>
  tracks: { stop: ReturnType<typeof vi.fn> }[]
  source: { connect: ReturnType<typeof vi.fn>; disconnect: ReturnType<typeof vi.fn> }
  analyser: { fftSize: number; getFloatTimeDomainData: ReturnType<typeof vi.fn> }
  context: {
    createMediaStreamSource: ReturnType<typeof vi.fn>
    createAnalyser: ReturnType<typeof vi.fn>
    close: ReturnType<typeof vi.fn>
  }
}

describe('MicMonitor', () => {
  let fakes: Fakes
  let monitor: MicMonitor

  beforeEach(() => {
    vi.useFakeTimers()
    const tracks = [{ stop: vi.fn() }, { stop: vi.fn() }]
    const stream = { getTracks: () => tracks }
    const source = { connect: vi.fn(), disconnect: vi.fn() }
    const analyser = {
      fftSize: 0,
      // Every poll reads a flat 0.5 signal -> RMS is exactly 0.5.
      getFloatTimeDomainData: vi.fn((buffer: Float32Array) => buffer.fill(0.5)),
    }
    const context = {
      createMediaStreamSource: vi.fn(() => source),
      createAnalyser: vi.fn(() => analyser),
      close: vi.fn(() => Promise.resolve()),
    }
    const getUserMedia = vi.fn().mockResolvedValue(stream)
    Object.defineProperty(navigator, 'mediaDevices', {
      value: { getUserMedia },
      configurable: true,
    })
    // Must be new-able (an arrow-implementation vi.fn is not a constructor);
    // a plain function returning the fake wins the `new` result.
    vi.stubGlobal('AudioContext', function FakeAudioContext() {
      return context
    })
    fakes = { getUserMedia, tracks, source, analyser, context }
    monitor = new MicMonitor()
  })

  afterEach(() => {
    monitor.stop()
    vi.unstubAllGlobals()
    vi.useRealTimers()
  })

  it('start requests an echo-cancelled stream and wires source -> analyser', async () => {
    await monitor.start()

    expect(fakes.getUserMedia).toHaveBeenCalledWith({
      audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true },
    })
    expect(fakes.context.createMediaStreamSource).toHaveBeenCalledTimes(1)
    expect(fakes.source.connect).toHaveBeenCalledWith(fakes.analyser)
    expect(fakes.analyser.fftSize).toBe(1024)
  })

  it('polls the analyser and reports RMS + dt to onLevel every 50ms', async () => {
    const onLevel = vi.fn()
    monitor.onLevel = onLevel
    await monitor.start()

    vi.advanceTimersByTime(50)
    expect(onLevel).toHaveBeenCalledTimes(1)
    const [rms, dtMs] = onLevel.mock.calls[0] as [number, number]
    expect(rms).toBeCloseTo(0.5, 5)
    expect(dtMs).toBe(50)

    vi.advanceTimersByTime(100)
    expect(onLevel).toHaveBeenCalledTimes(3)
  })

  it('a tick without an onLevel callback never touches the analyser', async () => {
    await monitor.start()
    vi.advanceTimersByTime(200)
    expect(fakes.analyser.getFloatTimeDomainData).not.toHaveBeenCalled()
  })

  it('stop halts polling and clears the level callback', async () => {
    const onLevel = vi.fn()
    monitor.onLevel = onLevel
    await monitor.start()
    vi.advanceTimersByTime(50)
    expect(onLevel).toHaveBeenCalledTimes(1)

    monitor.stop()
    vi.advanceTimersByTime(500)
    expect(onLevel).toHaveBeenCalledTimes(1)
    expect(monitor.onLevel).toBeNull()
  })

  it('stop releases every track, disconnects the source, and closes the context', async () => {
    await monitor.start()
    monitor.stop()

    for (const track of fakes.tracks) expect(track.stop).toHaveBeenCalledTimes(1)
    expect(fakes.source.disconnect).toHaveBeenCalledTimes(1)
    expect(fakes.context.close).toHaveBeenCalledTimes(1)
  })

  it('stop is idempotent and safe before start', async () => {
    monitor.stop() // Never started: nothing to release, must not throw.
    expect(fakes.tracks[0].stop).not.toHaveBeenCalled()

    await monitor.start()
    monitor.stop()
    monitor.stop() // Double stop must not re-release anything.
    for (const track of fakes.tracks) expect(track.stop).toHaveBeenCalledTimes(1)
    expect(fakes.context.close).toHaveBeenCalledTimes(1)
  })

  it('a throwing disconnect still releases tracks and closes the context', async () => {
    fakes.source.disconnect.mockImplementation(() => {
      throw new Error('already torn down')
    })
    await monitor.start()

    expect(() => monitor.stop()).not.toThrow()
    for (const track of fakes.tracks) expect(track.stop).toHaveBeenCalledTimes(1)
    expect(fakes.context.close).toHaveBeenCalledTimes(1)
  })
})
