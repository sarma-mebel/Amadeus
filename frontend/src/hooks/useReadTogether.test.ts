// @vitest-environment jsdom
// The shared toggle/gate/session/sign-off core lives in useSharedActivity.ts
// and is exercised in depth by useWatchTogether.test.ts; this file pins what
// is read-together-SPECIFIC in the adapter: the onPage->onCapture adaptation,
// the /api/read_along endpoint + counter keys, and the auto-attach
// lastPageFrameRef stash/clear semantics.
import { beforeEach, afterEach, describe, expect, it, vi } from 'vitest'
import { act, cleanup, renderHook } from '@testing-library/react'
import type { MutableRefObject } from 'react'
import { useReadTogether } from './useReadTogether'
import { startReadTogether, type ReadTogetherHandle } from '../readTogether'
import { diagnostics } from '../diagnostics'

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
vi.mock('../readTogether', () => ({ startReadTogether: vi.fn() }))

interface StartOpts {
  onPage: (image: { mime: string; data: string }) => void
  onEnded: () => void
}

const jsonResponse = (payload: unknown, ok = true, status = 200) => ({
  ok,
  status,
  json: async () => payload,
})

/** apiFetch bodies (parsed) for calls to /api/read_along. */
const postedBodies = (): Record<string, unknown>[] =>
  apiFetchMock.mock.calls
    .filter((call) => call[0] === '/api/read_along')
    .map((call) => JSON.parse((call[1] as { body: string }).body) as Record<string, unknown>)

describe('useReadTogether', () => {
  let handle: ReadTogetherHandle & { stop: ReturnType<typeof vi.fn> }
  let isBusyRef: MutableRefObject<boolean>
  let isDozingRef: MutableRefObject<boolean>
  let isCourtesyActiveRef: MutableRefObject<boolean>
  let applyPresenceLine: ReturnType<typeof vi.fn>

  const mount = () =>
    renderHook(() =>
      useReadTogether({ isBusyRef, isDozingRef, isCourtesyActiveRef, applyPresenceLine }),
    )

  const flush = () => act(async () => {})

  /** Toggles the mode on and returns the engine options the hook passed in. */
  const activate = async (result: { current: { toggleReadTogether: () => void } }) => {
    act(() => result.current.toggleReadTogether())
    await flush()
    return vi.mocked(startReadTogether).mock.calls[0][0] as StartOpts
  }

  beforeEach(() => {
    vi.clearAllMocks()
    isBusyRef = { current: false }
    isDozingRef = { current: false }
    isCourtesyActiveRef = { current: false }
    applyPresenceLine = vi.fn()
    handle = { stop: vi.fn() }
    vi.mocked(startReadTogether).mockResolvedValue(handle)
    isPetDndActiveMock.mockReturnValue(false)
    apiFetchMock.mockResolvedValue(jsonResponse({}))
  })

  afterEach(() => {
    cleanup()
  })

  it('a detected page posts to /api/read_along via the onPage adapter', async () => {
    apiFetchMock.mockResolvedValue(jsonResponse({ comment: 'oh, a plot twist' }))
    const { result } = mount()
    const opts = await activate(result)

    act(() => opts.onPage({ mime: 'image/jpeg', data: 'AAA' }))
    await flush()

    expect(postedBodies()).toEqual([{ image: { mime: 'image/jpeg', data: 'AAA' } }])
    expect(applyPresenceLine).toHaveBeenCalledWith(
      expect.objectContaining({ comment: 'oh, a plot twist' }),
      'oh, a plot twist',
    )
    expect(result.current.lastPageAt).not.toBeNull()
  })

  it('stashes the full-res frame in lastPageFrameRef for auto-attach, even if the POST fails', async () => {
    apiFetchMock.mockResolvedValue(jsonResponse({}, false, 500))
    const { result } = mount()
    const opts = await activate(result)

    act(() => opts.onPage({ mime: 'image/png', data: 'BBB' }))
    await flush()

    // The stash happens after the gate but before the POST resolves — a
    // failed page POST still leaves the frame available to auto-attach on
    // the next manual message.
    expect(result.current.lastPageFrameRef.current).toEqual({ mime: 'image/png', data: 'BBB' })
    expect(diagnostics.increment).toHaveBeenCalledWith('pet.read_along_page_failed', 'HTTP 500')
  })

  it('a gated page (busy) stashes nothing and posts nothing', async () => {
    const { result } = mount()
    const opts = await activate(result)

    isBusyRef.current = true
    act(() => opts.onPage({ mime: 'image/jpeg', data: 'AAA' }))
    await flush()

    expect(apiFetchMock).not.toHaveBeenCalled()
    expect(result.current.lastPageFrameRef.current).toBeNull()
    expect(result.current.lastPageAt).toBeNull()
  })

  it('toggling off after a posted page ends the session and clears the auto-attach frame', async () => {
    const { result } = mount()
    const opts = await activate(result)
    act(() => opts.onPage({ mime: 'image/jpeg', data: 'AAA' }))
    await flush()
    expect(result.current.lastPageFrameRef.current).not.toBeNull()

    act(() => result.current.toggleReadTogether())
    await flush()

    expect(handle.stop).toHaveBeenCalled()
    expect(result.current.lastPageFrameRef.current).toBeNull()
    expect(postedBodies()).toEqual([
      { image: { mime: 'image/jpeg', data: 'AAA' } },
      { end: true },
    ])
  })

  it('toggling off with zero pages posted sends no end request', async () => {
    const { result } = mount()
    await activate(result)

    act(() => result.current.toggleReadTogether())
    await flush()

    expect(handle.stop).toHaveBeenCalled()
    expect(postedBodies()).toEqual([])
  })

  it('a start failure counts pet.read_together_start_failed and flips the toggle back off', async () => {
    vi.mocked(startReadTogether).mockRejectedValue(new Error('picker dismissed'))
    const { result } = mount()

    act(() => result.current.toggleReadTogether())
    await flush()

    expect(result.current.readTogether).toBe(false)
    expect(diagnostics.increment).toHaveBeenCalledWith(
      'pet.read_together_start_failed',
      expect.stringContaining('picker dismissed'),
    )
  })
})
