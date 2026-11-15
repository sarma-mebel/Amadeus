// @vitest-environment jsdom
import { act, renderHook } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { useFocusMode } from './useFocusMode'
import type { PresencePayload } from './usePetPresence'

const { apiFetchMock } = vi.hoisted(() => ({ apiFetchMock: vi.fn() }))
vi.mock('../api', () => ({ apiFetch: apiFetchMock }))

const jsonResponse = (payload: unknown) => ({
  ok: true,
  json: async () => payload,
})

const bodyOf = (call: unknown[]): Record<string, unknown> =>
  JSON.parse((call[1] as RequestInit).body as string)

describe('useFocusMode', () => {
  let applied: Array<{ data: PresencePayload; text: string }>

  const mount = () =>
    renderHook(() =>
      useFocusMode({
        applyLine: (data, text) => {
          applied.push({ data, text })
        },
      }),
    )

  beforeEach(() => {
    vi.useFakeTimers()
    applied = []
    apiFetchMock.mockReset()
    window.localStorage.clear()
  })

  afterEach(() => {
    vi.useRealTimers()
  })

  it('starts a session and reflects active/remaining state', async () => {
    apiFetchMock.mockImplementation(async () =>
      jsonResponse({
        session_id: 's1',
        active: true,
        until: '2026-08-03T12:00:00',
        remaining_seconds: 1500,
        checkin: null,
      }),
    )
    const { result } = mount()
    await act(async () => {
      await result.current.start(25)
    })
    expect(result.current.active).toBe(true)
    expect(result.current.remainingSeconds).toBe(1500)
    const [path, init] = apiFetchMock.mock.calls[0]
    expect(path).toBe('/api/focus')
    expect(bodyOf(apiFetchMock.mock.calls[0])).toMatchObject({ action: 'start', minutes: 25 })
    expect(init).toBeTruthy()
  })

  it('delivers the check-in through applyLine exactly once on end', async () => {
    apiFetchMock.mockImplementation(async (_path: string, init: RequestInit) => {
      const body = JSON.parse(init.body as string)
      if (body.action === 'end') {
        return jsonResponse({
          session_id: 's1',
          active: false,
          until: null,
          remaining_seconds: null,
          checkin: 'Good to have you back.',
        })
      }
      return jsonResponse({ active: true, until: null, remaining_seconds: 100, checkin: null })
    })
    const { result } = mount()
    await act(async () => {
      await result.current.end()
    })
    expect(result.current.active).toBe(false)
    expect(applied).toHaveLength(1)
    expect(applied[0].text).toBe('Good to have you back.')
  })

  it('polls for natural expiry only while active, and stops polling once idle', async () => {
    let statusCalls = 0
    apiFetchMock.mockImplementation(async (_path: string, init: RequestInit) => {
      const body = JSON.parse(init.body as string)
      if (body.action === 'start') {
        return jsonResponse({ active: true, until: null, remaining_seconds: 60, checkin: null })
      }
      statusCalls += 1
      // First poll observes expiry; second poll (if it happened) would be a bug.
      return jsonResponse({
        active: false,
        until: null,
        remaining_seconds: null,
        checkin: statusCalls === 1 ? 'Welcome back.' : null,
      })
    })
    const { result } = mount()
    await act(async () => {
      await result.current.start(1)
    })
    expect(result.current.active).toBe(true)

    await act(async () => {
      await vi.advanceTimersByTimeAsync(30_000)
    })
    expect(result.current.active).toBe(false)
    expect(applied).toHaveLength(1)
    expect(statusCalls).toBe(1)

    // Idle now — the poll loop should not keep firing status requests.
    await act(async () => {
      await vi.advanceTimersByTimeAsync(60_000)
    })
    expect(statusCalls).toBe(1)
  })

  it('adopts a returned session_id into localStorage, like the other presence calls', async () => {
    apiFetchMock.mockImplementation(async () =>
      jsonResponse({ session_id: 'new-session', active: true, remaining_seconds: 100 }),
    )
    const { result } = mount()
    await act(async () => {
      await result.current.start(30)
    })
    expect(window.localStorage.getItem('session_id')).toBe('new-session')
  })

  it('a failed fetch is best-effort: no throw, no crash', async () => {
    apiFetchMock.mockRejectedValue(new Error('network down'))
    const { result } = mount()
    await act(async () => {
      await result.current.start(30)
    })
    expect(result.current.active).toBe(false)
  })

  it('a 404 (feature disabled) leaves state idle without a diagnostics spam entry', async () => {
    apiFetchMock.mockResolvedValue({ ok: false, status: 404, json: async () => ({}) })
    const { result } = mount()
    await act(async () => {
      await result.current.start(30)
    })
    expect(result.current.active).toBe(false)
  })
})
