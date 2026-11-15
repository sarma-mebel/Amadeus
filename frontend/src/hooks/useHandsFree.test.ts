// @vitest-environment jsdom
import { act, cleanup, renderHook } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { useHandsFree } from './useHandsFree'

const { recorders, apiFetchMock, publishMock, resetMock } = vi.hoisted(() => ({
  recorders: [] as Array<{
    onLevel: ((rms: number, dtMs: number) => void) | null
    cancelled: boolean
    takes: number
    trims: number[]
  }>,
  apiFetchMock: vi.fn(),
  publishMock: vi.fn(),
  resetMock: vi.fn(),
}))

vi.mock('../api', () => ({ apiFetch: apiFetchMock }))
vi.mock('../audio/levelBus', () => ({
  publishMicLevel: publishMock,
  resetMicLevel: resetMock,
}))
vi.mock('../audio/micRecorder', () => {
  class MicRecorder {
    onLevel: ((rms: number, dtMs: number) => void) | null = null
    cancelled = false
    takes = 0
    trims: number[] = []
    constructor() {
      recorders.push(this)
    }
    async start() {}
    trimBuffer(keepMs: number) {
      this.trims.push(keepMs)
    }
    takeWav() {
      this.takes += 1
      return new Blob(['x'], { type: 'audio/wav' })
    }
    async stop() {
      return null
    }
    cancel() {
      this.cancelled = true
    }
  }
  return { MicRecorder, STT_SAMPLE_RATE: 16000 }
})

const sttResponse = (text: string) => ({ ok: true, json: async () => ({ text }) })

interface Props {
  isBusy: boolean
  manualMicActive: boolean
}

describe('useHandsFree (continuous capture)', () => {
  let transcripts: string[]
  let bargeIns: number
  let hints: string[]

  const mount = (initial: Partial<Props> = {}) =>
    renderHook(
      ({ isBusy, manualMicActive }: Props) =>
        useHandsFree({
          micAvailable: true,
          micPermissionDenied: false,
          manualMicActive,
          isBusy,
          onTranscript: (text) => transcripts.push(text),
          showHint: (hint) => hints.push(hint),
          onBargeIn: () => {
            bargeIns += 1
          },
        }),
      { initialProps: { isBusy: false, manualMicActive: false, ...initial } },
    )

  /** Lets the async mic acquisition settle. */
  const flush = () => act(async () => {})

  const lastRecorder = () => {
    const recorder = recorders[recorders.length - 1]
    if (!recorder) throw new Error('no recorder acquired')
    return recorder
  }

  /** Feeds `ms` of constant level in 50ms capture buffers. */
  const feed = (level: number, ms: number) =>
    act(() => {
      const recorder = lastRecorder()
      for (let t = 0; t < ms; t += 50) recorder.onLevel?.(level, 50)
    })

  /** One utterance loud enough and long enough to endpoint. */
  const speakUtterance = () => {
    feed(0.1, 1_000)
    feed(0.001, 900)
  }

  const setSpeaking = (speaking: boolean) =>
    act(() => {
      window.dispatchEvent(new CustomEvent('kurisu:speaking', { detail: { speaking } }))
    })

  beforeEach(() => {
    recorders.length = 0
    apiFetchMock.mockReset()
    publishMock.mockReset()
    resetMock.mockReset()
    apiFetchMock.mockResolvedValue(sttResponse('hello'))
    transcripts = []
    bargeIns = 0
    hints = []
    window.localStorage.clear()
    window.localStorage.setItem('kurisu_hands_free', 'true')
  })

  // No vitest globals in this repo, so RTL's auto-cleanup never registers —
  // unmount explicitly or hook instances (and their window listeners) leak
  // across tests and multiply mic acquisitions.
  afterEach(() => {
    cleanup()
  })

  it('acquires the mic once on mount and endpoints an utterance through STT', async () => {
    const { result } = mount()
    await flush()
    expect(recorders).toHaveLength(1)

    speakUtterance()
    expect(apiFetchMock).toHaveBeenCalledTimes(1)
    expect(apiFetchMock.mock.calls[0][0]).toBe('/api/stt')
    expect(result.current.handsFreeStatus).toBe('transcribing')
    await flush()
    expect(transcripts).toEqual(['hello'])
    // The mic was never re-acquired for the next utterance.
    expect(recorders).toHaveLength(1)
    expect(lastRecorder().cancelled).toBe(false)
    // Levels were published while listening (waveform + listening reactions).
    expect(publishMock).toHaveBeenCalled()
  })

  it('never overlaps sends: suspended while STT and the turn are in flight', async () => {
    let resolveFetch: ((response: unknown) => void) | null = null
    apiFetchMock.mockImplementation(() => new Promise((resolve) => (resolveFetch = resolve)))

    const { rerender } = mount()
    await flush()

    speakUtterance()
    expect(apiFetchMock).toHaveBeenCalledTimes(1)

    // A second full utterance while the first transcription is pending is
    // discarded, not queued.
    speakUtterance()
    expect(apiFetchMock).toHaveBeenCalledTimes(1)

    await act(async () => {
      resolveFetch?.(sttResponse('first'))
    })
    expect(transcripts).toEqual(['first'])

    // The turn is now in flight (caller flips isBusy): still suspended.
    rerender({ isBusy: true, manualMicActive: false })
    speakUtterance()
    expect(apiFetchMock).toHaveBeenCalledTimes(1)

    // Turn done: listening resumes and the next utterance sends.
    rerender({ isBusy: false, manualMicActive: false })
    speakUtterance()
    expect(apiFetchMock).toHaveBeenCalledTimes(2)
  })

  it('discards empty transcripts silently and keeps listening', async () => {
    apiFetchMock.mockResolvedValue(sttResponse('   '))
    mount()
    await flush()

    speakUtterance()
    await flush()
    expect(transcripts).toEqual([])

    // No post-transcript grace for a discard — the very next utterance sends.
    apiFetchMock.mockResolvedValue(sttResponse('real one'))
    speakUtterance()
    await flush()
    expect(apiFetchMock).toHaveBeenCalledTimes(2)
    expect(transcripts).toEqual(['real one'])
  })

  it('never endpoints her own speech; listening resumes after the grace', async () => {
    mount()
    await flush()

    setSpeaking(true)
    // Two seconds of loud audio while she speaks: no send, no level-bus
    // publish (listening reactions must not react to her own voice).
    feed(0.1, 2_000)
    expect(apiFetchMock).not.toHaveBeenCalled()
    expect(publishMock).not.toHaveBeenCalled()
    expect(bargeIns).toBe(0) // auto barge-in is off by default

    setSpeaking(false)
    feed(0.001, 400) // resume grace
    speakUtterance()
    expect(apiFetchMock).toHaveBeenCalledTimes(1)
  })

  it('runs the strict auto barge-in detector on the same stream while she speaks', async () => {
    window.localStorage.setItem('kurisu_auto_barge_in', 'true')
    mount()
    await flush()

    setSpeaking(true)
    // Warm-up window (500ms) plus a bit of the voiced run: no trigger yet.
    feed(0.1, 600)
    expect(bargeIns).toBe(0)
    // Sustained voice past warm-up + min-voiced: exactly one trigger.
    feed(0.1, 400)
    expect(bargeIns).toBe(1)
    feed(0.1, 1_000)
    expect(bargeIns).toBe(1)
    // Her audio never became an utterance send.
    expect(apiFetchMock).not.toHaveBeenCalled()
  })

  it('suspends while the manual push-to-talk mic is active', async () => {
    const { rerender } = mount()
    await flush()

    rerender({ isBusy: false, manualMicActive: true })
    speakUtterance()
    expect(apiFetchMock).not.toHaveBeenCalled()

    rerender({ isBusy: false, manualMicActive: false })
    speakUtterance()
    expect(apiFetchMock).toHaveBeenCalledTimes(1)
  })

  it('toggle off releases the mic and persists; toggle on re-acquires', async () => {
    const { result } = mount()
    await flush()
    const first = lastRecorder()

    act(() => {
      result.current.toggleHandsFree()
    })
    expect(result.current.handsFree).toBe(false)
    expect(window.localStorage.getItem('kurisu_hands_free')).toBe('false')
    expect(first.cancelled).toBe(true)

    act(() => {
      result.current.toggleHandsFree()
    })
    await flush()
    expect(result.current.handsFree).toBe(true)
    expect(recorders).toHaveLength(2)
  })

  it('reacts to the shared preference event (SettingsPanel toggle)', async () => {
    const { result } = mount()
    await flush()
    const first = lastRecorder()

    act(() => {
      window.localStorage.setItem('kurisu_hands_free', 'false')
      window.dispatchEvent(
        new CustomEvent('kurisu:hands-free', { detail: { enabled: false } }),
      )
    })
    expect(result.current.handsFree).toBe(false)
    expect(first.cancelled).toBe(true)
  })

  it('releases the mic while the tab is hidden and re-acquires on return', async () => {
    mount()
    await flush()
    const first = lastRecorder()

    const setVisibility = (state: 'hidden' | 'visible') => {
      Object.defineProperty(document, 'visibilityState', {
        configurable: true,
        get: () => state,
      })
      return act(() => {
        document.dispatchEvent(new Event('visibilitychange'))
      })
    }

    await setVisibility('hidden')
    expect(first.cancelled).toBe(true)

    await setVisibility('visible')
    await flush()
    expect(recorders).toHaveLength(2)
    expect(lastRecorder().cancelled).toBe(false)
  })

  it('releases the mic on unmount', async () => {
    const { unmount } = mount()
    await flush()
    const recorder = lastRecorder()
    unmount()
    expect(recorder.cancelled).toBe(true)
  })
})
