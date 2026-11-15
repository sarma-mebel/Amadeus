// @vitest-environment jsdom
import { beforeEach, afterEach, describe, expect, it, vi } from 'vitest'
import { act, cleanup, renderHook } from '@testing-library/react'
import type { MutableRefObject } from 'react'
import { useWatchTogether } from './useWatchTogether'
import { startWatchTogether, type WatchTogetherHandle } from '../watchTogether'
import { diagnostics } from '../diagnostics'
import type { PresencePayload } from './usePetPresence'

const { apiFetchMock } = vi.hoisted(() => ({ apiFetchMock: vi.fn() }))
vi.mock('../api', () => ({ apiFetch: apiFetchMock }))
vi.mock('../audioPrefs', () => ({ isAudioEnabled: () => true }))
vi.mock('../diagnostics', () => ({ diagnostics: { increment: vi.fn() } }))
const { isPetDndActiveMock } = vi.hoisted(() => ({ isPetDndActiveMock: vi.fn(() => false) }))
vi.mock('./usePetPresence', () => ({ isPetDndActive: isPetDndActiveMock }))
// presenceRequest normally folds in session/user ids from localStorage —
// pass the extra body through verbatim so tests can inspect it.
vi.mock('../presenceApi', () => ({
  presenceRequest: (extra?: Record<string, unknown>) => ({
    method: 'POST',
    body: JSON.stringify(extra ?? {}),
  }),
}))
vi.mock('../vrm/kurisuBridge', () => ({ kurisuBridge: {} }))
vi.mock('../watchTogether', () => ({ startWatchTogether: vi.fn() }))

interface StartOpts {
  onFrame: (image: { mime: string; data: string }) => void
  onEnded: () => void
}

const jsonResponse = (payload: unknown, ok = true, status = 200) => ({
  ok,
  status,
  json: async () => payload,
})

/** apiFetch bodies (parsed) for calls to /api/watch_together. */
const postedBodies = (): Record<string, unknown>[] =>
  apiFetchMock.mock.calls
    .filter((call) => call[0] === '/api/watch_together')
    .map((call) => JSON.parse((call[1] as { body: string }).body) as Record<string, unknown>)

describe('useWatchTogether', () => {
  let handle: WatchTogetherHandle & { stop: ReturnType<typeof vi.fn> }
  let isBusyRef: MutableRefObject<boolean>
  let isDozingRef: MutableRefObject<boolean>
  let isCourtesyActiveRef: MutableRefObject<boolean>
  let applyPresenceLine: ReturnType<typeof vi.fn>

  const mount = () =>
    renderHook(() =>
      useWatchTogether({ isBusyRef, isDozingRef, isCourtesyActiveRef, applyPresenceLine }),
    )

  const flush = () => act(async () => {})

  /** Toggles the mode on and returns the sampler options the hook passed in. */
  const activate = async (result: { current: { toggleWatchTogether: () => void } }) => {
    act(() => result.current.toggleWatchTogether())
    await flush()
    return vi.mocked(startWatchTogether).mock.calls[0][0] as StartOpts
  }

  beforeEach(() => {
    vi.clearAllMocks()
    isBusyRef = { current: false }
    isDozingRef = { current: false }
    isCourtesyActiveRef = { current: false }
    applyPresenceLine = vi.fn()
    handle = { stop: vi.fn() }
    vi.mocked(startWatchTogether).mockResolvedValue(handle)
    isPetDndActiveMock.mockReturnValue(false)
    apiFetchMock.mockResolvedValue(jsonResponse({}))
  })

  afterEach(() => {
    cleanup()
  })

  it('toggling on starts the sampler', async () => {
    const { result } = mount()
    expect(result.current.watchTogether).toBe(false)

    await activate(result)

    expect(result.current.watchTogether).toBe(true)
    expect(startWatchTogether).toHaveBeenCalledTimes(1)
  })

  it('a sampled frame posts to /api/watch_together and a comment speaks', async () => {
    apiFetchMock.mockResolvedValue(jsonResponse({ comment: 'nice level', emotion: 'happy' }))
    const { result } = mount()
    const opts = await activate(result)

    act(() => opts.onFrame({ mime: 'image/jpeg', data: 'AAA' }))
    await flush()

    expect(postedBodies()).toEqual([{ image: { mime: 'image/jpeg', data: 'AAA' } }])
    expect(applyPresenceLine).toHaveBeenCalledWith(
      expect.objectContaining({ comment: 'nice level' } as Partial<PresencePayload>),
      'nice level',
    )
    expect(result.current.lastWatchFrameAt).not.toBeNull()
  })

  it('a busy turn gates the frame out entirely (no POST, no timestamp)', async () => {
    const { result } = mount()
    const opts = await activate(result)

    isBusyRef.current = true
    act(() => opts.onFrame({ mime: 'image/jpeg', data: 'AAA' }))
    await flush()

    expect(apiFetchMock).not.toHaveBeenCalled()
    expect(result.current.lastWatchFrameAt).toBeNull()
  })

  it('a failed frame POST counts a diagnostics increment, not a crash', async () => {
    apiFetchMock.mockResolvedValue(jsonResponse({}, false, 500))
    const { result } = mount()
    const opts = await activate(result)

    act(() => opts.onFrame({ mime: 'image/jpeg', data: 'AAA' }))
    await flush()

    expect(diagnostics.increment).toHaveBeenCalledWith(
      'pet.watch_together_frame_failed',
      'HTTP 500',
    )
    expect(applyPresenceLine).not.toHaveBeenCalled()
  })

  it('toggling off stops the engine and ends the server session after a posted frame', async () => {
    const { result } = mount()
    const opts = await activate(result)
    act(() => opts.onFrame({ mime: 'image/jpeg', data: 'AAA' }))
    await flush()

    act(() => result.current.toggleWatchTogether())
    await flush()

    expect(result.current.watchTogether).toBe(false)
    expect(handle.stop).toHaveBeenCalled()
    expect(postedBodies()).toContainEqual({ end: true })
  })

  it('toggling off with zero posted frames never fires a spurious end request', async () => {
    const { result } = mount()
    await activate(result)

    act(() => result.current.toggleWatchTogether())
    await flush()

    expect(handle.stop).toHaveBeenCalled()
    expect(apiFetchMock).not.toHaveBeenCalled()
  })

  it('a start failure counts a diagnostics increment and flips the toggle back off', async () => {
    vi.mocked(startWatchTogether).mockRejectedValue(new Error('picker dismissed'))
    const consoleWarn = vi.spyOn(console, 'warn').mockImplementation(() => {})
    try {
      const { result } = mount()
      await activate(result)

      expect(result.current.watchTogether).toBe(false)
      expect(diagnostics.increment).toHaveBeenCalledWith(
        'pet.watch_together_start_failed',
        expect.stringContaining('picker dismissed'),
      )
    } finally {
      consoleWarn.mockRestore()
    }
  })

  it('the browser "Stop sharing" chrome (onEnded) flips the toggle off', async () => {
    const { result } = mount()
    const opts = await activate(result)

    act(() => opts.onEnded())
    await flush()

    expect(result.current.watchTogether).toBe(false)
    expect(handle.stop).toHaveBeenCalled()
  })

  it('unmount stops the engine and ends an active session', async () => {
    const { result, unmount } = mount()
    const opts = await activate(result)
    act(() => opts.onFrame({ mime: 'image/jpeg', data: 'AAA' }))
    await flush()

    unmount()
    await act(async () => {})

    expect(handle.stop).toHaveBeenCalled()
    expect(postedBodies()).toContainEqual({ end: true })
  })
})
