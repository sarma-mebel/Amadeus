// @vitest-environment jsdom
import { act, renderHook } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { useRef } from 'react'
import { usePresence } from './usePresence'
import { kurisuBridge } from '../vrm/kurisuBridge'

// Presence fires real fetches through apiFetch (greeting/reminders/idle/
// screen-peek) — mock it. memoryApi.overview and the /api/brief entry effect
// also run on mount; the default {} responder keeps them harmless.
const { apiFetchMock } = vi.hoisted(() => ({ apiFetchMock: vi.fn() }))
vi.mock('../api', () => ({ apiFetch: apiFetchMock }))
const { memoryRequestMock } = vi.hoisted(() => ({ memoryRequestMock: vi.fn() }))
vi.mock('../memoryApi', () => ({ memoryRequest: memoryRequestMock }))
// captureScreen is dynamically imported inside the screen-companion peek —
// stub the module so no real getDisplayMedia/Tauri call happens.
const { captureScreenMock } = vi.hoisted(() => ({ captureScreenMock: vi.fn() }))
vi.mock('../images', () => ({ captureScreen: captureScreenMock }))
// Audio must read as enabled so the applier reaches its react/play calls.
vi.mock('../audioPrefs', () => ({ isAudioEnabled: () => true }))
// canNotify normally requires a real Notification permission grant, which
// jsdom never has — default it to true here so the hidden-window tests can
// isolate the NEW petIsAlive() gate from the pre-existing notification gate
// (both currently combine as `hidden && !canNotify(...)`, and a stale/absent
// heartbeat test must still reach the notification path to prove it).
const { canNotifyMock, notifyMock } = vi.hoisted(() => ({
  canNotifyMock: vi.fn(() => true),
  notifyMock: vi.fn(),
}))
vi.mock('../notifications', () => ({ canNotify: canNotifyMock, notify: notifyMock }))

const jsonResponse = (payload: unknown) => ({ ok: true, json: async () => payload })

const respondWith = (routes: Record<string, unknown>) => {
  apiFetchMock.mockImplementation(async (path: string) =>
    jsonResponse(routes[path] ?? {}),
  )
}

const calledPaths = () => apiFetchMock.mock.calls.map((call) => call[0])

describe('usePresence', () => {
  let react: ReturnType<typeof vi.fn>
  let expressionsSet: ReturnType<typeof vi.fn>
  let play: ReturnType<typeof vi.fn>

  // Drives the hook with live refs, the way ChatInterface passes them in.
  const mount = (entered = true) =>
    renderHook(() => {
      const isLoadingRef = useRef(false)
      const isStreamingRef = useRef(false)
      const llmIsDummyRef = useRef(false)
      return usePresence({
        entered,
        setMessages: vi.fn(),
        adoptSessionId: vi.fn(),
        isLoadingRef,
        isStreamingRef,
        isLoading: false,
        llmIsDummyRef,
        waitForFirstRunProbe: async () => {},
        memoryOpen: false,
      })
    })

  beforeEach(() => {
    vi.useFakeTimers()
    apiFetchMock.mockReset()
    respondWith({})
    memoryRequestMock.mockReset()
    memoryRequestMock.mockResolvedValue(jsonResponse({}))
    captureScreenMock.mockReset()
    canNotifyMock.mockReset()
    canNotifyMock.mockReturnValue(true)
    notifyMock.mockReset()
    window.localStorage.clear()

    react = vi.fn()
    expressionsSet = vi.fn()
    play = vi.fn().mockResolvedValue(undefined)
    kurisuBridge.react = react
    kurisuBridge.expressions = { set: expressionsSet, available: [], list: vi.fn() }
    kurisuBridge.lipSync = {
      beginStream: vi.fn(),
      enqueue: vi.fn().mockResolvedValue(undefined),
      endStream: vi.fn(),
      stop: vi.fn(),
      isSpeaking: vi.fn().mockReturnValue(false),
      play,
    }
  })

  afterEach(() => {
    vi.useRealTimers()
    kurisuBridge.react = null
    kurisuBridge.expressions = null
    kurisuBridge.lipSync = null
  })

  it('gestures on a presence line, threading the payload emotion + intensity', async () => {
    // FIX1: the greeting rides applyPresenceExpressionAndAudio, which must now
    // fire a body gesture too — not just the face expression + lip sync.
    respondWith({
      '/api/greeting': {
        greeting: 'You again.',
        emotion: 'happy',
        emotion_intensity: 0.9,
        audio: 'AAAA',
        audio_mime_type: 'audio/wav',
      },
    })
    mount()
    await act(async () => {
      await vi.advanceTimersByTimeAsync(0)
    })
    expect(expressionsSet).toHaveBeenCalledWith('happy', 0.9, 2500)
    expect(react).toHaveBeenCalledWith('happy', 0.9)
    expect(play).toHaveBeenCalled()
  })

  it('falls back to 0.7 intensity for a gesture when the payload omits it', async () => {
    respondWith({ '/api/greeting': { greeting: 'Hmph.', emotion: 'neutral' } })
    mount()
    await act(async () => {
      await vi.advanceTimersByTimeAsync(0)
    })
    expect(react).toHaveBeenCalledWith('neutral', 0.7)
  })

  it('runs the screen-companion peek when the flag is set, gesturing the comment', async () => {
    // FIX2: the pet-only screen-peek loop now runs in the main window too,
    // gated on the shared kurisu_screen_companion flag, and routes through the
    // same applier so it ALSO gestures (FIX1 + FIX2 compose).
    window.localStorage.setItem('kurisu_screen_companion', 'true')
    captureScreenMock.mockResolvedValue({ mime: 'image/jpeg', data: 'ZZZZ' })
    respondWith({
      '/api/screen_peek': { comment: 'Debugging again?', emotion: 'relaxed', emotion_intensity: 0.5 },
    })
    mount()
    // Peek timer is 6min +/- 90s; 8 min guarantees at least one fire.
    await act(async () => {
      await vi.advanceTimersByTimeAsync(8 * 60_000)
    })
    expect(captureScreenMock).toHaveBeenCalled()
    expect(calledPaths()).toContain('/api/screen_peek')
    expect(react).toHaveBeenCalledWith('relaxed', 0.5)
  })

  it('never peeks at the screen when the flag is unset', async () => {
    // Flag defaults off — the loop must not schedule or capture.
    mount()
    await act(async () => {
      await vi.advanceTimersByTimeAsync(8 * 60_000)
    })
    expect(captureScreenMock).not.toHaveBeenCalled()
    expect(calledPaths()).not.toContain('/api/screen_peek')
  })

  describe('hidden-window guards (pet-vs-main deference)', () => {
    // The hidden main window must not steal the exactly-once greeting/
    // reminder/reach-out pop from a visible pet window nobody-can-hear it —
    // and, absent a pet, must still work exactly like before.

    const hideDocument = () => {
      Object.defineProperty(document, 'hidden', {
        configurable: true,
        get: () => true,
      })
    }
    const showDocument = () => {
      Object.defineProperty(document, 'hidden', {
        configurable: true,
        get: () => false,
      })
    }

    afterEach(() => {
      showDocument()
    })

    it('skips the greeting entirely while hidden with no live pet heartbeat', async () => {
      hideDocument()
      respondWith({ '/api/greeting': { greeting: 'Oh. You again.' } })
      mount()
      await act(async () => {
        await vi.advanceTimersByTimeAsync(0)
      })
      expect(calledPaths()).not.toContain('/api/greeting')
    })

    it('fires the greeting on visibilitychange once the window becomes visible', async () => {
      hideDocument()
      respondWith({ '/api/greeting': { greeting: 'Oh. You again.' } })
      mount()
      await act(async () => {
        await vi.advanceTimersByTimeAsync(0)
      })
      expect(calledPaths()).not.toContain('/api/greeting')

      showDocument()
      await act(async () => {
        document.dispatchEvent(new Event('visibilitychange'))
        await vi.advanceTimersByTimeAsync(0)
      })
      expect(calledPaths()).toContain('/api/greeting')
    })

    it('skips the greeting while hidden with a fresh pet heartbeat, and does not fire it later either', async () => {
      hideDocument()
      window.localStorage.setItem('kurisu_pet_alive', String(Date.now()))
      respondWith({ '/api/greeting': { greeting: 'Oh. You again.' } })
      mount()
      await act(async () => {
        await vi.advanceTimersByTimeAsync(0)
      })
      expect(calledPaths()).not.toContain('/api/greeting')
    })

    it('hidden reminder/reach-out polls are skipped while the pet heartbeat is fresh', async () => {
      hideDocument()
      window.localStorage.setItem('kurisu_pet_alive', String(Date.now()))
      respondWith({
        '/api/reminders/due': { reminder: 'never delivered to a hidden window' },
        '/api/reachout/due': { reach_out: 'never delivered either' },
      })
      mount()
      await act(async () => {
        await vi.advanceTimersByTimeAsync(0)
      })
      expect(calledPaths()).not.toContain('/api/reminders/due')
      expect(calledPaths()).not.toContain('/api/reachout/due')
    })

    it('hidden reminder poll still fires when the pet heartbeat is stale', async () => {
      hideDocument()
      window.localStorage.setItem('kurisu_pet_alive', String(Date.now() - 120_000))
      respondWith({
        '/api/reminders/due': { reminder: 'delivered via notification' },
      })
      mount()
      await act(async () => {
        await vi.advanceTimersByTimeAsync(0)
      })
      expect(calledPaths()).toContain('/api/reminders/due')
    })
  })
})
