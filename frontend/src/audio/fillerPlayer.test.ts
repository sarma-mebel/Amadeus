// @vitest-environment jsdom
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import {
  FILLERS_STORAGE_KEY,
  armFillerForTurn,
  loadFillersEnabled,
  parseFillerManifest,
  pickFillerIndex,
  resetFillersForTest,
  saveFillersEnabled,
} from './fillerPlayer'
import { diagnostics } from '../diagnostics'
import { kurisuBridge } from '../vrm/kurisuBridge'

describe('pickFillerIndex', () => {
  it('returns null for an empty pool and 0 for a single clip', () => {
    expect(pickFillerIndex(0, null)).toBeNull()
    expect(pickFillerIndex(1, 0)).toBe(0)
  })

  it('never repeats the previous clip when there are at least two', () => {
    for (let last = 0; last < 3; last += 1) {
      for (let roll = 0; roll < 10; roll += 1) {
        const picked = pickFillerIndex(3, last, () => roll / 10)
        expect(picked).not.toBe(last)
        expect(picked).toBeGreaterThanOrEqual(0)
        expect(picked).toBeLessThan(3)
      }
    }
  })

  it('clamps a random() that returns 1', () => {
    expect(pickFillerIndex(3, null, () => 1)).toBe(2)
  })
})

describe('parseFillerManifest', () => {
  it('accepts well-formed entries and drops malformed ones', () => {
    const clips = parseFillerManifest({
      version: 1,
      fillers: [
        { file: 'hmm.wav', text: 'んー…' },
        { file: 42, text: 'bad' },
        { text: 'no file' },
        'nonsense',
      ],
    })
    expect(clips).toEqual([{ file: 'hmm.wav', text: 'んー…' }])
  })

  it('returns empty for non-manifest shapes', () => {
    expect(parseFillerManifest(null)).toEqual([])
    expect(parseFillerManifest({})).toEqual([])
    expect(parseFillerManifest({ fillers: 'nope' })).toEqual([])
  })
})

describe('fillers preference', () => {
  afterEach(() => {
    window.localStorage.removeItem(FILLERS_STORAGE_KEY)
  })

  it('defaults to disabled (opt-in) and round-trips the toggle', () => {
    expect(loadFillersEnabled()).toBe(false)
    saveFillersEnabled(true)
    expect(loadFillersEnabled()).toBe(true)
    saveFillersEnabled(false)
    expect(loadFillersEnabled()).toBe(false)
  })

  it('stays at the default (off) when storage is blocked', () => {
    // A leftover `return true` from the default-ON era force-enabled fillers
    // in storage-blocked contexts with no way to turn them off.
    const spy = vi.spyOn(Storage.prototype, 'getItem').mockImplementation(() => {
      throw new Error('storage blocked')
    })
    try {
      expect(loadFillersEnabled()).toBe(false)
    } finally {
      spy.mockRestore()
    }
  })
})

describe('armFillerForTurn', () => {
  const manifest = {
    version: 1,
    fillers: [{ file: 'hmm.wav', text: 'んー…' }],
  }
  let play: ReturnType<typeof vi.fn>
  let fetchMock: ReturnType<typeof vi.fn>

  beforeEach(() => {
    vi.useFakeTimers()
    resetFillersForTest()
    diagnostics.resetForTest()
    play = vi.fn().mockResolvedValue(undefined)
    kurisuBridge.lipSync = {
      play,
      stop: vi.fn(),
      beginStream: vi.fn(),
      enqueue: vi.fn().mockResolvedValue(undefined),
      endStream: vi.fn(),
      isSpeaking: vi.fn().mockReturnValue(false),
    }
    fetchMock = vi.fn(async (url: string) => {
      if (String(url).includes('manifest.json')) {
        return { ok: true, json: async () => manifest } as Response
      }
      return {
        ok: true,
        arrayBuffer: async () => new Uint8Array([1, 2, 3]).buffer,
      } as unknown as Response
    })
    vi.stubGlobal('fetch', fetchMock)
  })

  afterEach(() => {
    kurisuBridge.lipSync = null
    vi.unstubAllGlobals()
    vi.useRealTimers()
    window.localStorage.removeItem(FILLERS_STORAGE_KEY)
  })

  it('plays one clip after the delay (when opted in)', async () => {
    saveFillersEnabled(true)
    armFillerForTurn(600)
    await vi.advanceTimersByTimeAsync(700)
    expect(play).toHaveBeenCalledTimes(1)
    expect(play.mock.calls[0][0].mimeType).toBe('audio/wav')
    expect(typeof play.mock.calls[0][0].base64Audio).toBe('string')
  })

  it('does not play when cancelled before the delay elapses', async () => {
    const handle = armFillerForTurn(600)
    await vi.advanceTimersByTimeAsync(300)
    handle.cancel()
    await vi.advanceTimersByTimeAsync(1000)
    expect(play).not.toHaveBeenCalled()
  })

  it('is a no-op when the preference is off', async () => {
    saveFillersEnabled(false)
    armFillerForTurn(600)
    await vi.advanceTimersByTimeAsync(1000)
    expect(play).not.toHaveBeenCalled()
    expect(fetchMock).not.toHaveBeenCalled()
  })

  it('stays silent when the manifest is missing (404)', async () => {
    fetchMock.mockResolvedValue({ ok: false } as Response)
    armFillerForTurn(600)
    await vi.advanceTimersByTimeAsync(1000)
    expect(play).not.toHaveBeenCalled()
  })

  it('skips the turn when she is already speaking a presence line', async () => {
    ;(kurisuBridge.lipSync!.isSpeaking as ReturnType<typeof vi.fn>).mockReturnValue(true)
    armFillerForTurn(600)
    await vi.advanceTimersByTimeAsync(1000)
    expect(play).not.toHaveBeenCalled()
  })

  it('counts a play() rejection instead of swallowing it silently', async () => {
    saveFillersEnabled(true)
    play.mockRejectedValueOnce(new Error('autoplay blocked'))
    armFillerForTurn(600)
    await vi.advanceTimersByTimeAsync(700)
    expect(play).toHaveBeenCalledTimes(1)
    expect(diagnostics.snapshot().counters['voice.filler_play_failed']).toBe(1)
  })

  it('last-chance check: a cancel that lands during the clip-audio fetch (after the delay fired) still suppresses play()', async () => {
    saveFillersEnabled(true)
    // Replace the audio-clip fetch with a controllable promise so the test
    // can cancel() the handle WHILE armFillerForTurn's fire() is awaiting
    // fetchClipAudio — the exact race window the last-chance recheck (right
    // before the play() call) exists to close.
    let resolveClipFetch: (() => void) | null = null
    fetchMock.mockImplementation(async (url: string) => {
      if (String(url).includes('manifest.json')) {
        return { ok: true, json: async () => manifest } as Response
      }
      await new Promise<void>((resolve) => {
        resolveClipFetch = resolve
      })
      return {
        ok: true,
        arrayBuffer: async () => new Uint8Array([1, 2, 3]).buffer,
      } as unknown as Response
    })

    const handle = armFillerForTurn(600)
    await vi.advanceTimersByTimeAsync(600)
    // fire() is now inside fetchClipAudio's await, past the isSpeaking/index
    // checks but before base64 resolves — real audio (or turn end) arrives
    // right here, in-character with the described race.
    handle.cancel()
    resolveClipFetch?.()
    // Let the rest of fire()'s microtask chain (fetchClipAudio resolving,
    // the last-chance check, the would-be play() call) run to completion.
    await vi.advanceTimersByTimeAsync(0)
    await Promise.resolve()
    await Promise.resolve()

    expect(play).not.toHaveBeenCalled()
  })
})

describe('parseBackchannelManifest', () => {
  it('reads the backchannels key and tolerates its absence', async () => {
    const { parseBackchannelManifest } = await import('./fillerPlayer')
    expect(
      parseBackchannelManifest({
        version: 1,
        fillers: [{ file: 'hmm.wav', text: 'んー…' }],
        backchannels: [{ file: 'bc-un.wav', text: 'うん。' }, { file: 3 }],
      }),
    ).toEqual([{ file: 'bc-un.wav', text: 'うん。' }])
    expect(parseBackchannelManifest({ version: 1, fillers: [] })).toEqual([])
    expect(parseBackchannelManifest(null)).toEqual([])
  })
})
