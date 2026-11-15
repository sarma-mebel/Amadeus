// @vitest-environment jsdom
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import {
  APP_VOLUME_STORAGE_KEY,
  createLipSyncController,
  loadAppVolume,
  resetSharedAudioContextForTest,
  type LipSyncState,
} from './lipSyncController'
import { diagnostics } from '../diagnostics'

describe('loadAppVolume', () => {
  it('defaults to 1 when the key is absent', () => {
    window.localStorage.removeItem(APP_VOLUME_STORAGE_KEY)
    expect(loadAppVolume()).toBe(1)
  })

  it('parses a stored float string', () => {
    window.localStorage.setItem(APP_VOLUME_STORAGE_KEY, '0.35')
    expect(loadAppVolume()).toBeCloseTo(0.35)
    window.localStorage.removeItem(APP_VOLUME_STORAGE_KEY)
  })

  it('clamps out-of-range values to [0, 1]', () => {
    window.localStorage.setItem(APP_VOLUME_STORAGE_KEY, '2.5')
    expect(loadAppVolume()).toBe(1)
    window.localStorage.setItem(APP_VOLUME_STORAGE_KEY, '-0.5')
    expect(loadAppVolume()).toBe(0)
    window.localStorage.removeItem(APP_VOLUME_STORAGE_KEY)
  })

  it('falls back to 1 for junk values', () => {
    window.localStorage.setItem(APP_VOLUME_STORAGE_KEY, 'banana')
    expect(loadAppVolume()).toBe(1)
    window.localStorage.removeItem(APP_VOLUME_STORAGE_KEY)
  })

  it('treats "0" as fully muted, not falsy-default', () => {
    window.localStorage.setItem(APP_VOLUME_STORAGE_KEY, '0')
    expect(loadAppVolume()).toBe(0)
    window.localStorage.removeItem(APP_VOLUME_STORAGE_KEY)
  })
})

describe('lipSyncGeneration abandonment (startLipSync vs beginStream race)', () => {
  // Minimal controllable audio graph: startLipSync/beginStream only ever
  // call createBufferSource/createGain/createAnalyser/createConvolver/
  // createBuffer/decodeAudioData/resume plus a couple of node methods
  // (connect/disconnect/start/stop) — nothing else in the real WebAudio API
  // is touched by the code under test.
  class FakeGainNode {
    gain = { value: 0, cancelScheduledValues: vi.fn(), setTargetAtTime: vi.fn() }
    connect = vi.fn()
    disconnect = vi.fn()
  }
  class FakeAnalyserNode {
    fftSize = 1024
    frequencyBinCount = 512
    smoothingTimeConstant = 0
    connect = vi.fn()
    disconnect = vi.fn()
    getFloatTimeDomainData = vi.fn()
    getFloatFrequencyData = vi.fn()
  }
  class FakeConvolverNode {
    normalize = true
    buffer: unknown = null
    connect = vi.fn()
    disconnect = vi.fn()
  }
  class FakeBufferSourceNode {
    buffer: unknown = null
    detune = { value: 0 }
    onended: (() => void) | null = null
    connect = vi.fn()
    disconnect = vi.fn()
    start = vi.fn()
    stop = vi.fn()
  }

  let decodeQueue: Array<{ resolve: (buf: unknown) => void; reject: (err: unknown) => void }>
  let sources: FakeBufferSourceNode[]

  const makeFakeContext = () => {
    decodeQueue = []
    sources = []
    return {
      state: 'running' as AudioContextState,
      currentTime: 0,
      sampleRate: 48000,
      destination: {},
      resume: vi.fn().mockResolvedValue(undefined),
      createBufferSource: vi.fn(() => {
        const source = new FakeBufferSourceNode()
        sources.push(source)
        return source
      }),
      createGain: vi.fn(() => new FakeGainNode()),
      createAnalyser: vi.fn(() => new FakeAnalyserNode()),
      createConvolver: vi.fn(() => new FakeConvolverNode()),
      createBuffer: vi.fn(() => ({ getChannelData: () => new Float32Array(1) })),
      // Controllable: each call returns a NEW pending promise, so the test
      // can resolve them in whatever order it wants (a later beginStream's
      // decode resolving before an earlier startLipSync's stale decode).
      decodeAudioData: vi.fn(
        () =>
          new Promise((resolve, reject) => {
            decodeQueue.push({ resolve, reject })
          }),
      ),
    }
  }

  let stateRef: { current: LipSyncState }
  const makeState = (): LipSyncState => ({
    audioContext: null,
    analyser: null,
    dryGain: null,
    wetGain: null,
    convolver: null,
    gainNode: null,
    appVolumeGain: null,
    sourceNode: null,
    freqData: null,
    timeData: null,
    expressionValues: { aa: 0, ih: 0, ee: 0, ou: 0, oh: 0, relaxed: 0.4 },
    previousViseme: null,
    scheduledSources: [],
    nextChunkStartTime: 0,
    streaming: false,
    streamDraining: false,
    speechEnvelope: 0,
    sentenceStartTimeoutIds: [],
  })

  let fakeContext: ReturnType<typeof makeFakeContext>
  let controller: ReturnType<typeof createLipSyncController>
  let onSpeakingChange: ReturnType<typeof vi.fn>

  beforeEach(() => {
    diagnostics.resetForTest()
    // The controller caches ONE AudioContext at module scope for the whole
    // page lifetime (see its doc comment) — without this reset, the second
    // test in this suite would silently keep talking to the first test's
    // fakeContext instead of the fresh one built below.
    resetSharedAudioContextForTest()
    fakeContext = makeFakeContext()
    // Must be constructible (ensureAudioContext does `new AudioCtx()`) —
    // a plain vi.fn() arrow-return isn't new-able.
    function FakeAudioContext() {
      return fakeContext
    }
    vi.stubGlobal('AudioContext', FakeAudioContext)
    stateRef = { current: makeState() }
    onSpeakingChange = vi.fn()
    controller = createLipSyncController({
      stateRef,
      getVrm: () => ({ expressionManager: { setValue: vi.fn(), getValue: vi.fn(() => 0) } }) as never,
      getReverb: () => undefined,
      onSpeakingChange,
    })
  })

  afterEach(() => {
    vi.unstubAllGlobals()
  })

  it('a stale startLipSync() decode resolving AFTER a newer beginStream() does not tear the new stream down', async () => {
    // Turn 1: a one-shot play() (e.g. a presence line) starts decoding.
    const startPromise = controller.startLipSync('AAAA')
    expect(decodeQueue).toHaveLength(1)
    const staleDecode = decodeQueue[0]

    // Before that decode resolves, a NEWER turn preempts it with a real
    // streamed reply — beginStream() calls stopLipSync() itself (tearing
    // down the one-shot's not-yet-started source, which is fine — it
    // never got as far as sourceNode) and marks streaming live.
    await controller.beginStream()
    expect(stateRef.current.streaming).toBe(true)

    // NOW the stale decode resolves — this must be abandoned silently: no
    // stop(), no new source started, and critically `streaming` must stay
    // true (the bug: the stale operation used to call stopLipSync()
    // unconditionally here, cutting off the new stream and hitting the
    // `!state.streaming` guard on every subsequent chunk — a silent reply).
    staleDecode.resolve({ duration: 1 })
    await startPromise

    expect(stateRef.current.streaming).toBe(true)
    expect(onSpeakingChange).not.toHaveBeenCalledWith(false)
    expect(diagnostics.snapshot().counters['voice.lipsync_superseded']).toBe(1)

    // A chunk enqueued after this must still be scheduled onto the live
    // stream, not dropped — proves the stream is genuinely intact, not
    // just that `streaming` happens to still read true.
    const beforeSourceCount = sources.length
    const enqueuePromise = controller.enqueueLipSyncChunk('AAAA', 0, 'neutral')
    expect(decodeQueue).toHaveLength(2)
    decodeQueue[1].resolve({ duration: 1 })
    await enqueuePromise
    expect(sources.length).toBeGreaterThan(beforeSourceCount)
    expect(sources[sources.length - 1].start).toHaveBeenCalled()
  })

  it('schedules chunks in enqueue order even when a later chunk decodes first', async () => {
    await controller.beginStream()
    expect(stateRef.current.streaming).toBe(true)

    // Two chunks of one WS burst: a long sentence 1 and a short sentence 2,
    // decoding concurrently.
    const first = controller.enqueueLipSyncChunk('AAAA', 0, 'neutral')
    const second = controller.enqueueLipSyncChunk('BBBB', 1, 'neutral')
    expect(decodeQueue).toHaveLength(2)

    // The SHORT chunk's decode completes before the LONG one's — the bug:
    // scheduling in decode-completion order played the sentences swapped.
    const bufferB = { duration: 1 }
    const bufferA = { duration: 3 }
    decodeQueue[1].resolve(bufferB)
    decodeQueue[0].resolve(bufferA)
    await Promise.all([first, second])

    // Chunk 1 scheduled first (fresh-start pad), chunk 2 chained at its end.
    expect(sources).toHaveLength(2)
    expect(sources[0].buffer).toBe(bufferA)
    expect(sources[1].buffer).toBe(bufferB)
    expect(sources[0].start).toHaveBeenCalledWith(0.05)
    expect(sources[1].start).toHaveBeenCalledWith(0.05 + 3)
  })

  it('a failed decode does not wedge later chunks of the same stream', async () => {
    await controller.beginStream()

    const first = controller.enqueueLipSyncChunk('AAAA', 0, 'neutral')
    const second = controller.enqueueLipSyncChunk('BBBB', 1, 'neutral')
    expect(decodeQueue).toHaveLength(2)

    decodeQueue[0].reject(new Error('corrupt wav'))
    await expect(first).rejects.toThrow('corrupt wav')

    const bufferB = { duration: 1 }
    decodeQueue[1].resolve(bufferB)
    await second

    // The surviving chunk still schedules — the chain must not deadlock on
    // its dead predecessor (which never produced a source at all).
    expect(sources).toHaveLength(1)
    expect(sources[0].buffer).toBe(bufferB)
    expect(sources[0].start).toHaveBeenCalled()
  })

  it('a stale beginStream() resume() resolving AFTER a newer startLipSync() does not clobber the one-shot clip', async () => {
    // beginStream's only await is context.resume() (already 'running' here
    // resolves synchronously in the mock, so force 'suspended' to get a
    // real pending resume the test can control).
    fakeContext.state = 'suspended'
    let resolveResume: (() => void) | null = null
    fakeContext.resume.mockImplementation(
      () =>
        new Promise<void>((resolve) => {
          resolveResume = resolve
        }),
    )

    const beginPromise = controller.beginStream()
    // beginStream has already synchronously set streaming=true and is now
    // awaiting resume().
    expect(stateRef.current.streaming).toBe(true)

    // A newer one-shot preempts it (e.g. a presence line firing mid-stream
    // start) — startLipSync tears down and starts its own clip.
    fakeContext.state = 'running'
    const startPromise = controller.startLipSync('AAAA')
    expect(decodeQueue).toHaveLength(1)
    decodeQueue[0].resolve({ duration: 1 })
    await startPromise
    expect(stateRef.current.sourceNode).not.toBeNull()
    const liveSource = stateRef.current.sourceNode

    // NOW the stale beginStream's resume() resolves — it must not touch
    // state further (no re-marking streaming, no clobbering sourceNode).
    resolveResume?.()
    await beginPromise

    expect(stateRef.current.sourceNode).toBe(liveSource)
    expect(diagnostics.snapshot().counters['voice.lipsync_superseded']).toBe(1)
  })
})
