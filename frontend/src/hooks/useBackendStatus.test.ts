// @vitest-environment jsdom
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { act, cleanup, renderHook, waitFor } from '@testing-library/react'
import { useBackendStatus } from './useBackendStatus'
import { CHARACTER_SWITCH_EVENT } from '../charactersApi'
import { __resetModelRegistryCache } from '../models'

const jsonResponse = (payload: unknown, status = 200) =>
  new Response(JSON.stringify(payload), {
    status,
    headers: { 'Content-Type': 'application/json' },
  })

const statusPayload = (character?: Record<string, unknown>) => ({
  version: '0.12.0',
  demo_mode: false,
  ...(character ? { character } : {}),
  llm: { provider: 'gemini', class: 'GeminiProvider', model: 'g', is_dummy: false },
  tts: {
    requested: null,
    loaded: null,
    class: null,
    engine: null,
    cache_signature: null,
    warming: false,
    fallback_from: null,
    error: null,
  },
  stt: { provider: 'none', loaded: false, unavailable_reason: null },
  memory: {
    long_term: false,
    fact_extraction: false,
    episodes: false,
    greeting: false,
    embedder: null,
    summary_provider_active: false,
  },
})

describe('useBackendStatus', () => {
  let fetchMock: ReturnType<typeof vi.fn>

  afterEach(() => {
    cleanup()
    vi.unstubAllGlobals()
    vi.useRealTimers()
  })

  beforeEach(() => {
    window.localStorage.clear()
    __resetModelRegistryCache()
    Element.prototype.scrollIntoView = vi.fn()
    Element.prototype.scrollTo = vi.fn()

    fetchMock = vi.fn(async (url: RequestInfo | URL) => {
      const path = String(url)
      if (path.endsWith('/healthz')) return jsonResponse({ status: 'ok', voice: 'ready' })
      if (path.endsWith('/api/status')) return jsonResponse(statusPayload())
      return jsonResponse({}, 404)
    })
    vi.stubGlobal('fetch', fetchMock)
  })

  it('populates backendStatus from the mount-time healthz-driven refresh', async () => {
    const { result } = renderHook(() => useBackendStatus())

    await waitFor(() => expect(result.current.backendStatus).not.toBeNull())
    expect(result.current.backendStatus?.version).toBe('0.12.0')
    expect(result.current.voiceWarming).toBe(false)
  })

  it('flags voiceWarming while /healthz reports the voice model warming', async () => {
    fetchMock.mockImplementation(async (url: RequestInfo | URL) => {
      const path = String(url)
      if (path.endsWith('/healthz')) return jsonResponse({ status: 'ok', voice: 'warming' })
      if (path.endsWith('/api/status')) return jsonResponse(statusPayload())
      return jsonResponse({}, 404)
    })

    const { result, unmount } = renderHook(() => useBackendStatus())
    await waitFor(() => expect(result.current.voiceWarming).toBe(true))
    // Unmount cancels the pending 10s re-poll timer.
    unmount()
  })

  it('re-reads status and starts a new conversation on the character-switch event', async () => {
    const { result } = renderHook(() => useBackendStatus())
    await waitFor(() => expect(result.current.backendStatus).not.toBeNull())

    const startNew = vi.fn()
    result.current.startNewConversationRef.current = startNew
    const statusCallsBefore = fetchMock.mock.calls.filter(([url]) =>
      String(url).endsWith('/api/status'),
    ).length

    act(() => {
      window.dispatchEvent(new Event(CHARACTER_SWITCH_EVENT))
    })

    expect(startNew).toHaveBeenCalledTimes(1)
    await waitFor(() => {
      const statusCalls = fetchMock.mock.calls.filter(([url]) =>
        String(url).endsWith('/api/status'),
      ).length
      expect(statusCalls).toBe(statusCallsBefore + 1)
    })
  })

  it('auto-applies a character-declared avatar once, then respects manual picks', async () => {
    fetchMock.mockImplementation(async (url: RequestInfo | URL) => {
      const path = String(url)
      if (path.endsWith('/healthz')) return jsonResponse({ status: 'ok', voice: 'ready' })
      if (path.endsWith('/api/status')) {
        return jsonResponse(statusPayload({ name: 'Miku', model_id: 'miku_v1' }))
      }
      if (path.includes('/static/models/models.json')) {
        return jsonResponse({ models: [{ id: 'miku_v1', label: 'Miku', file: '/static/models/Miku.vrm' }] })
      }
      return jsonResponse({}, 404)
    })

    const { result } = renderHook(() => useBackendStatus())
    await waitFor(() => expect(result.current.backendStatus).not.toBeNull())

    await waitFor(() => {
      expect(window.localStorage.getItem('kurisu_character_model_v1')).toBe('miku_v1')
    })
    expect(window.localStorage.getItem('kurisu_model_v1')).toBe('miku_v1')
  })

  it('does not re-apply an avatar already marked as applied', async () => {
    window.localStorage.setItem('kurisu_character_model_v1', 'miku_v1')
    fetchMock.mockImplementation(async (url: RequestInfo | URL) => {
      const path = String(url)
      if (path.endsWith('/healthz')) return jsonResponse({ status: 'ok', voice: 'ready' })
      if (path.endsWith('/api/status')) {
        return jsonResponse(statusPayload({ name: 'Miku', model_id: 'miku_v1' }))
      }
      if (path.includes('/static/models/models.json')) {
        return jsonResponse({ models: [{ id: 'miku_v1', label: 'Miku', file: '/static/models/Miku.vrm' }] })
      }
      return jsonResponse({}, 404)
    })

    const { result } = renderHook(() => useBackendStatus())
    await waitFor(() => expect(result.current.backendStatus).not.toBeNull())

    // The registry is never consulted when the marker already matches.
    await act(async () => {
      await Promise.resolve()
    })
    const registryCalls = fetchMock.mock.calls.filter(([url]) =>
      String(url).includes('models.json'),
    )
    expect(registryCalls.length).toBe(0)
    expect(window.localStorage.getItem('kurisu_model_v1')).toBeNull()
  })
})
