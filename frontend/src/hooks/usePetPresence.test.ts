// @vitest-environment jsdom
import { act, renderHook } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import {
  PET_ALIVE_KEY,
  PET_DND_INDEFINITE,
  PET_DND_KEY,
  usePetPresence,
  type PresencePayload,
} from './usePetPresence'
import { kurisuBridge } from '../vrm/kurisuBridge'
import { diagnostics } from '../diagnostics'

const { apiFetchMock } = vi.hoisted(() => ({ apiFetchMock: vi.fn() }))
vi.mock('../api', () => ({ apiFetch: apiFetchMock }))

const jsonResponse = (payload: unknown) => ({
  ok: true,
  json: async () => payload,
})

/** apiFetch responder keyed by path; unlisted paths return empty payloads. */
const respondWith = (routes: Record<string, unknown>) => {
  apiFetchMock.mockImplementation(async (path: string) =>
    jsonResponse(routes[path] ?? {}),
  )
}

const calledPaths = () => apiFetchMock.mock.calls.map((call) => call[0])

// Matches the hook's own POLL_MS (reminders/reach-out poll cadence, also
// reused as the greeting's DND-skipped retry cadence).
const POLL_TICK_MS = 30_000
// Matches the hook's own GREETING_DELAY_MS (startup delay before the first
// mount-greeting attempt).
const GREETING_DELAY_MS = 2_500

describe('usePetPresence', () => {
  let applied: Array<{ data: PresencePayload; text: string }>
  let busy: { current: boolean }
  let courtesyActive: { current: boolean }
  let dozing: { current: boolean }
  let wakeCalls: number

  const mount = () =>
    renderHook(() =>
      usePetPresence({
        isBusyRef: busy,
        applyLine: (data, text) => {
          applied.push({ data, text })
        },
        isCourtesyActiveRef: courtesyActive,
        isDozingRef: dozing,
        onBeforePresenceWake: () => {
          wakeCalls += 1
        },
      }),
    )

  /** Drains the fire-and-forget requestUnlockWakeup fetches. */
  const flushAsync = async () => {
    await act(async () => {
      await vi.advanceTimersByTimeAsync(0)
    })
  }

  beforeEach(() => {
    vi.useFakeTimers()
    applied = []
    busy = { current: false }
    courtesyActive = { current: false }
    dozing = { current: false }
    wakeCalls = 0
    apiFetchMock.mockReset()
    respondWith({})
    window.localStorage.clear()
    // Default the bridge to already-ready so every pre-existing test (all
    // written before the lipsync-wait behavior existed) keeps its original
    // fixed-2500ms greeting timing. Tests exercising the wait/timeout path
    // explicitly null this back out.
    kurisuBridge.lipSync = {
      beginStream: vi.fn(),
      enqueue: vi.fn().mockResolvedValue(undefined),
      endStream: vi.fn(),
      stop: vi.fn(),
      isSpeaking: vi.fn().mockReturnValue(false),
      play: vi.fn().mockResolvedValue(undefined),
    }
  })

  afterEach(() => {
    vi.useRealTimers()
    kurisuBridge.lipSync = null
  })

  it('polls reminders and reach-out immediately and delivers lines', async () => {
    respondWith({
      '/api/reminders/due': { session_id: 's-1', reminder: 'Reminder: buy filters.' },
      '/api/reachout/due': { reach_out: 'Did the draft go out?', emotion: 'curious' },
    })
    mount()
    await act(async () => {
      await vi.advanceTimersByTimeAsync(0)
    })
    expect(calledPaths()).toContain('/api/reminders/due')
    expect(calledPaths()).toContain('/api/reachout/due')
    const texts = applied.map((entry) => entry.text)
    expect(texts).toContain('Reminder: buy filters.')
    expect(texts).toContain('Did the draft go out?')
    // Session id from the response is adopted for later polls.
    expect(window.localStorage.getItem('session_id')).toBe('s-1')
  })

  it('requests the greeting once after the startup delay', async () => {
    respondWith({ '/api/greeting': { greeting: 'Oh. You again.' } })
    mount()
    await act(async () => {
      await vi.advanceTimersByTimeAsync(3000)
    })
    expect(
      calledPaths().filter((path) => path === '/api/greeting'),
    ).toHaveLength(1)
    expect(applied.some((entry) => entry.text === 'Oh. You again.')).toBe(true)
    // Much later: never re-greets within this mount.
    await act(async () => {
      await vi.advanceTimersByTimeAsync(10 * 60_000)
    })
    expect(
      calledPaths().filter((path) => path === '/api/greeting'),
    ).toHaveLength(1)
  })

  it('stays quiet while a turn is in flight', async () => {
    busy.current = true
    respondWith({
      '/api/reminders/due': { reminder: 'never delivered' },
    })
    mount()
    await act(async () => {
      await vi.advanceTimersByTimeAsync(3000)
    })
    expect(apiFetchMock).not.toHaveBeenCalled()
    expect(applied).toEqual([])
  })

  it('keeps polling on the interval and fires idle chatter eventually', async () => {
    mount()
    await act(async () => {
      await vi.advanceTimersByTimeAsync(0)
    })
    const initialPolls = calledPaths().filter(
      (path) => path === '/api/reminders/due',
    ).length
    respondWith({ '/api/idle': { chatter: 'Still there?' } })
    await act(async () => {
      await vi.advanceTimersByTimeAsync(8 * 60_000)
    })
    expect(
      calledPaths().filter((path) => path === '/api/reminders/due').length,
    ).toBeGreaterThan(initialPolls)
    // Idle window is 4–7 min randomized; 8 min guarantees at least one.
    expect(calledPaths()).toContain('/api/idle')
    expect(applied.some((entry) => entry.text === 'Still there?')).toBe(true)
  })

  it('a null line (nothing due) applies nothing', async () => {
    respondWith({
      '/api/reminders/due': { reminder: null },
      '/api/reachout/due': { reach_out: null },
    })
    mount()
    await act(async () => {
      await vi.advanceTimersByTimeAsync(0)
    })
    expect(applied).toEqual([])
  })

  describe('do-not-disturb', () => {
    it('fetches nothing at all while a timed DND period is active — reminders/reach-out are exactly-once pops, so a discarded poll would lose them', async () => {
      window.localStorage.setItem(PET_DND_KEY, String(Date.now() + 60_000))
      respondWith({
        '/api/greeting': { greeting: 'Oh. You again.' },
        '/api/reminders/due': { reminder: 'buy filters' },
        '/api/reachout/due': { reach_out: 'still there?' },
      })
      mount()
      await act(async () => {
        await vi.advanceTimersByTimeAsync(3000)
      })
      expect(apiFetchMock).not.toHaveBeenCalled()
      expect(applied).toEqual([])
    })

    it('fetches nothing while indefinite DND ("-1") is set', async () => {
      window.localStorage.setItem(PET_DND_KEY, PET_DND_INDEFINITE)
      respondWith({
        '/api/reminders/due': { reminder: 'buy filters' },
      })
      mount()
      await act(async () => {
        await vi.advanceTimersByTimeAsync(POLL_TICK_MS)
      })
      expect(apiFetchMock).not.toHaveBeenCalled()
    })

    it('an expired timed DND is treated as off — polling resumes normally', async () => {
      window.localStorage.setItem(PET_DND_KEY, String(Date.now() - 1000))
      respondWith({
        '/api/reminders/due': { reminder: 'buy filters' },
      })
      mount()
      await act(async () => {
        await vi.advanceTimersByTimeAsync(0)
      })
      expect(calledPaths()).toContain('/api/reminders/due')
      expect(applied.some((entry) => entry.text === 'buy filters')).toBe(true)
    })

    it('the greeting is not consumed by a DND-active attempt — it still greets once DND clears', async () => {
      window.localStorage.setItem(PET_DND_KEY, String(Date.now() + 5000))
      respondWith({ '/api/greeting': { greeting: 'Oh. You again.' } })
      mount()
      // Scheduled greeting attempt fires during DND — skipped, not consumed.
      await act(async () => {
        await vi.advanceTimersByTimeAsync(3000)
      })
      expect(calledPaths().filter((path) => path === '/api/greeting')).toHaveLength(0)
      // DND clears; the retry (same cadence as the reminder poll) picks it up.
      window.localStorage.removeItem(PET_DND_KEY)
      await act(async () => {
        await vi.advanceTimersByTimeAsync(POLL_TICK_MS)
      })
      expect(calledPaths().filter((path) => path === '/api/greeting')).toHaveLength(1)
      expect(applied.some((entry) => entry.text === 'Oh. You again.')).toBe(true)
    })

    it('resumes all polling once DND is turned off mid-mount', async () => {
      window.localStorage.setItem(PET_DND_KEY, PET_DND_INDEFINITE)
      respondWith({
        '/api/reminders/due': { reminder: 'buy filters' },
      })
      mount()
      await act(async () => {
        await vi.advanceTimersByTimeAsync(0)
      })
      expect(apiFetchMock).not.toHaveBeenCalled()
      window.localStorage.removeItem(PET_DND_KEY)
      await act(async () => {
        await vi.advanceTimersByTimeAsync(POLL_TICK_MS)
      })
      expect(calledPaths()).toContain('/api/reminders/due')
    })
  })

  describe('fullscreen courtesy', () => {
    // Follows the DND describe block's pattern exactly: courtesy is a
    // second hard "don't even fetch" gate that must behave identically to
    // DND (same skippedForDnd-shaped retry for the greeting), just driven
    // by a ref instead of localStorage.
    it('fetches nothing at all while courtesy is active — same hard mute as DND', async () => {
      courtesyActive.current = true
      respondWith({
        '/api/greeting': { greeting: 'Oh. You again.' },
        '/api/reminders/due': { reminder: 'buy filters' },
        '/api/reachout/due': { reach_out: 'still there?' },
      })
      mount()
      await act(async () => {
        await vi.advanceTimersByTimeAsync(3000)
      })
      expect(apiFetchMock).not.toHaveBeenCalled()
      expect(applied).toEqual([])
    })

    it('the greeting is not consumed by a courtesy-active attempt — it still greets once courtesy clears', async () => {
      courtesyActive.current = true
      respondWith({ '/api/greeting': { greeting: 'Oh. You again.' } })
      mount()
      await act(async () => {
        await vi.advanceTimersByTimeAsync(3000)
      })
      expect(calledPaths().filter((path) => path === '/api/greeting')).toHaveLength(0)
      courtesyActive.current = false
      await act(async () => {
        await vi.advanceTimersByTimeAsync(POLL_TICK_MS)
      })
      expect(calledPaths().filter((path) => path === '/api/greeting')).toHaveLength(1)
      expect(applied.some((entry) => entry.text === 'Oh. You again.')).toBe(true)
    })

    it('resumes all polling once courtesy clears mid-mount', async () => {
      courtesyActive.current = true
      respondWith({
        '/api/reminders/due': { reminder: 'buy filters' },
      })
      mount()
      await act(async () => {
        await vi.advanceTimersByTimeAsync(0)
      })
      expect(apiFetchMock).not.toHaveBeenCalled()
      courtesyActive.current = false
      await act(async () => {
        await vi.advanceTimersByTimeAsync(POLL_TICK_MS)
      })
      expect(calledPaths()).toContain('/api/reminders/due')
    })

    it('composes with DND — either alone is enough to go quiet (both active is still quiet)', async () => {
      window.localStorage.setItem(PET_DND_KEY, PET_DND_INDEFINITE)
      courtesyActive.current = true
      respondWith({
        '/api/reminders/due': { reminder: 'buy filters' },
      })
      mount()
      await act(async () => {
        await vi.advanceTimersByTimeAsync(POLL_TICK_MS)
      })
      expect(apiFetchMock).not.toHaveBeenCalled()
    })
  })

  describe('doze integration', () => {
    it('wakes before applying a reminder line, but idle chatter is unaffected by wake', async () => {
      respondWith({
        '/api/reminders/due': { session_id: 's-1', reminder: 'Reminder: buy filters.' },
      })
      mount()
      await act(async () => {
        await vi.advanceTimersByTimeAsync(0)
      })
      expect(wakeCalls).toBe(1)
      expect(applied.some((entry) => entry.text === 'Reminder: buy filters.')).toBe(true)
    })

    it('wakes before applying a reach-out line', async () => {
      respondWith({
        '/api/reachout/due': { reach_out: 'Did the draft go out?' },
      })
      mount()
      await act(async () => {
        await vi.advanceTimersByTimeAsync(0)
      })
      expect(wakeCalls).toBe(1)
    })

    it('does NOT wake for the greeting', async () => {
      respondWith({ '/api/greeting': { greeting: 'Oh. You again.' } })
      mount()
      await act(async () => {
        await vi.advanceTimersByTimeAsync(3000)
      })
      expect(wakeCalls).toBe(0)
    })

    it('skips the idle-chatter fetch entirely while dozing, and does not wake for it', async () => {
      dozing.current = true
      respondWith({ '/api/idle': { chatter: 'Still there?' } })
      mount()
      await act(async () => {
        await vi.advanceTimersByTimeAsync(8 * 60_000)
      })
      expect(calledPaths()).not.toContain('/api/idle')
      expect(wakeCalls).toBe(0)
      expect(applied).toEqual([])
    })

    it('reminders/reach-out still poll and apply while dozing (only idle is gated)', async () => {
      dozing.current = true
      respondWith({
        '/api/reminders/due': { reminder: 'buy filters' },
      })
      mount()
      await act(async () => {
        await vi.advanceTimersByTimeAsync(0)
      })
      expect(calledPaths()).toContain('/api/reminders/due')
      expect(applied.some((entry) => entry.text === 'buy filters')).toBe(true)
      expect(wakeCalls).toBe(1)
    })

    it('idle chatter resumes once dozing ends, without needing a remount', async () => {
      dozing.current = true
      respondWith({ '/api/idle': { chatter: 'Still there?' } })
      mount()
      await act(async () => {
        await vi.advanceTimersByTimeAsync(4 * 60_000)
      })
      expect(calledPaths()).not.toContain('/api/idle')
      dozing.current = false
      await act(async () => {
        await vi.advanceTimersByTimeAsync(4 * 60_000)
      })
      expect(calledPaths()).toContain('/api/idle')
    })
  })

  describe('requestUnlockWakeup (session-unlocked handler)', () => {
    // The Rust lock poller emits `session-unlocked`; the actual event
    // listener lives in usePetChat (out of this hook's ownership), but the
    // gated request path it calls — requestUnlockWakeup — lives here, so
    // these tests exercise it directly via the hook's return value.

    it('triggers an immediate reminders + reach-out poll and asks for a greeting, on demand (not waiting for the 30s tick)', async () => {
      respondWith({
        '/api/reminders/due': { session_id: 's-unlock', reminder: 'Reminder: stretch.' },
        '/api/reachout/due': { reach_out: null },
        '/api/greeting': { greeting: null },
      })
      const { result } = mount()
      // Let the initial mount-time reminder/reach-out poll (fires at t=0)
      // and the greeting's startup delay pass and settle first, so the
      // assertions below are unambiguously about the on-demand call.
      await act(async () => {
        await vi.advanceTimersByTimeAsync(GREETING_DELAY_MS)
      })
      apiFetchMock.mockClear()
      applied = []

      act(() => {
        result.current.requestUnlockWakeup()
      })
      await flushAsync()

      expect(calledPaths()).toContain('/api/reminders/due')
      expect(calledPaths()).toContain('/api/reachout/due')
      expect(calledPaths()).toContain('/api/greeting')
      expect(applied.some((entry) => entry.text === 'Reminder: stretch.')).toBe(true)
    })

    it('respects DND — no fetch at all while a timed DND period is active', async () => {
      const { result } = mount()
      await act(async () => {
        await vi.advanceTimersByTimeAsync(GREETING_DELAY_MS)
      })
      window.localStorage.setItem(PET_DND_KEY, String(Date.now() + 60_000))
      apiFetchMock.mockClear()
      applied = []

      act(() => {
        result.current.requestUnlockWakeup()
      })
      await flushAsync()

      expect(apiFetchMock).not.toHaveBeenCalled()
      expect(applied).toEqual([])
    })

    it('respects fullscreen courtesy — no fetch while courtesy is active', async () => {
      const { result } = mount()
      await act(async () => {
        await vi.advanceTimersByTimeAsync(GREETING_DELAY_MS)
      })
      courtesyActive.current = true
      apiFetchMock.mockClear()

      act(() => {
        result.current.requestUnlockWakeup()
      })
      await flushAsync()

      expect(apiFetchMock).not.toHaveBeenCalled()
    })

    it('respects busy — no fetch while a turn is in flight', async () => {
      const { result } = mount()
      await act(async () => {
        await vi.advanceTimersByTimeAsync(GREETING_DELAY_MS)
      })
      busy.current = true
      apiFetchMock.mockClear()

      act(() => {
        result.current.requestUnlockWakeup()
      })
      await flushAsync()

      expect(apiFetchMock).not.toHaveBeenCalled()
    })

    it('wakes before applying reminder/reach-out lines delivered via the unlock path, same as the scheduled poll', async () => {
      respondWith({
        '/api/reminders/due': { reminder: 'Reminder: stretch.' },
      })
      const { result } = mount()
      await act(async () => {
        await vi.advanceTimersByTimeAsync(GREETING_DELAY_MS)
      })
      wakeCalls = 0

      act(() => {
        result.current.requestUnlockWakeup()
      })
      await flushAsync()

      expect(wakeCalls).toBeGreaterThan(0)
    })

    it('mount-greeting semantics are unchanged: the once-per-mount greeting still fires exactly once from the mount path, independent of any unlock-triggered greeting request', async () => {
      respondWith({ '/api/greeting': { greeting: 'Oh. You again.' } })
      const { result } = mount()
      await act(async () => {
        await vi.advanceTimersByTimeAsync(GREETING_DELAY_MS)
      })
      expect(
        calledPaths().filter((path) => path === '/api/greeting'),
      ).toHaveLength(1)
      expect(applied.some((entry) => entry.text === 'Oh. You again.')).toBe(true)

      // An unlock event arrives later — the on-demand path is allowed to
      // ask again (the server's should_greet gap decides whether it says
      // anything), and this must not disturb the mount path's own
      // greetedRef bookkeeping or double up the initial mount greeting.
      applied = []
      act(() => {
        result.current.requestUnlockWakeup()
      })
      await flushAsync()
      expect(
        calledPaths().filter((path) => path === '/api/greeting'),
      ).toHaveLength(2)

      // Long after: still no THIRD spontaneous greeting from the mount
      // timer/poll loop re-firing on its own.
      await act(async () => {
        await vi.advanceTimersByTimeAsync(10 * 60_000)
      })
      expect(
        calledPaths().filter((path) => path === '/api/greeting'),
      ).toHaveLength(2)
    })
  })

  describe('heartbeat', () => {
    it('writes kurisu_pet_alive on mount', () => {
      mount()
      expect(window.localStorage.getItem(PET_ALIVE_KEY)).not.toBeNull()
    })

    it('refreshes the heartbeat on every reminders/reach-out poll tick', async () => {
      mount()
      await act(async () => {
        await vi.advanceTimersByTimeAsync(0)
      })
      const first = Number(window.localStorage.getItem(PET_ALIVE_KEY))
      expect(Number.isFinite(first)).toBe(true)

      vi.setSystemTime(Date.now() + POLL_TICK_MS)
      await act(async () => {
        await vi.advanceTimersByTimeAsync(POLL_TICK_MS)
      })
      const second = Number(window.localStorage.getItem(PET_ALIVE_KEY))
      expect(second).toBeGreaterThan(first)
    })
  })

  describe('greeting defers for the lip-sync bridge', () => {
    // kurisuBridge.lipSync stays null until the VRM scene finishes loading
    // (lazy chunk + GLTF fetch/parse) — often well past GREETING_DELAY_MS on
    // a cold launch. The greeting must wait for it (up to a cap) rather than
    // firing through a null bridge and silently dropping the audio while
    // marking itself delivered forever.

    it('does not fire the greeting at the old fixed delay while the bridge is still null', async () => {
      kurisuBridge.lipSync = null
      respondWith({ '/api/greeting': { greeting: 'Oh. You again.' } })
      mount()
      await act(async () => {
        await vi.advanceTimersByTimeAsync(GREETING_DELAY_MS)
      })
      expect(calledPaths()).not.toContain('/api/greeting')
    })

    it('fires the greeting once the bridge becomes ready during the wait window, without incrementing the drop counter', async () => {
      kurisuBridge.lipSync = null
      respondWith({ '/api/greeting': { greeting: 'Oh. You again.' } })
      const incrementSpy = vi.spyOn(diagnostics, 'increment')
      mount()
      await act(async () => {
        await vi.advanceTimersByTimeAsync(GREETING_DELAY_MS)
      })
      expect(calledPaths()).not.toContain('/api/greeting')

      // Bridge comes online mid-wait (VRM scene finished loading).
      kurisuBridge.lipSync = {
        beginStream: vi.fn(),
        enqueue: vi.fn().mockResolvedValue(undefined),
        endStream: vi.fn(),
        stop: vi.fn(),
        isSpeaking: vi.fn().mockReturnValue(false),
        play: vi.fn().mockResolvedValue(undefined),
      }
      await act(async () => {
        await vi.advanceTimersByTimeAsync(1000)
      })
      expect(calledPaths().filter((p) => p === '/api/greeting')).toHaveLength(1)
      expect(
        incrementSpy.mock.calls.some(([name]) => name === 'pet.greeting_before_lipsync'),
      ).toBe(false)
    })

    it('still fires the greeting after the wait times out with no bridge, and counts the drop', async () => {
      kurisuBridge.lipSync = null
      respondWith({ '/api/greeting': { greeting: 'Oh. You again.' } })
      const incrementSpy = vi.spyOn(diagnostics, 'increment')
      mount()
      // Startup delay + full lipsync wait timeout.
      await act(async () => {
        await vi.advanceTimersByTimeAsync(GREETING_DELAY_MS + 15_000 + 250)
      })
      expect(calledPaths().filter((p) => p === '/api/greeting')).toHaveLength(1)
      expect(applied.some((entry) => entry.text === 'Oh. You again.')).toBe(true)
      expect(
        incrementSpy.mock.calls.some(([name]) => name === 'pet.greeting_before_lipsync'),
      ).toBe(true)
    })
  })

  describe('idle chatter defers to active speech', () => {
    it('skips the idle-chatter fetch while isSpeaking() is true, and does not wake for it', async () => {
      kurisuBridge.lipSync = {
        beginStream: vi.fn(),
        enqueue: vi.fn().mockResolvedValue(undefined),
        endStream: vi.fn(),
        stop: vi.fn(),
        isSpeaking: vi.fn().mockReturnValue(true),
        play: vi.fn().mockResolvedValue(undefined),
      }
      respondWith({ '/api/idle': { chatter: 'Still there?' } })
      mount()
      await act(async () => {
        await vi.advanceTimersByTimeAsync(8 * 60_000)
      })
      expect(calledPaths()).not.toContain('/api/idle')
      expect(applied).toEqual([])
    })

    it('resumes idle chatter once speech ends, without a remount', async () => {
      const isSpeaking = vi.fn().mockReturnValue(true)
      kurisuBridge.lipSync = {
        beginStream: vi.fn(),
        enqueue: vi.fn().mockResolvedValue(undefined),
        endStream: vi.fn(),
        stop: vi.fn(),
        isSpeaking,
        play: vi.fn().mockResolvedValue(undefined),
      }
      respondWith({ '/api/idle': { chatter: 'Still there?' } })
      mount()
      await act(async () => {
        await vi.advanceTimersByTimeAsync(4 * 60_000)
      })
      expect(calledPaths()).not.toContain('/api/idle')

      isSpeaking.mockReturnValue(false)
      await act(async () => {
        await vi.advanceTimersByTimeAsync(4 * 60_000)
      })
      expect(calledPaths()).toContain('/api/idle')
    })
  })
})
