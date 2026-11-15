// @vitest-environment jsdom
import { act, renderHook, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import {
  useSettingsData,
  type SettingsResponse,
  type SettingsUpdateResponse,
} from './useSettingsData'

const baseSettings: SettingsResponse = {
  is_first_run: false,
  providers: { llm: 'gemini', llm_is_dummy: false, tts: 'qwen3_tts', stt: 'dummy' },
  openai_api_key_set: false,
  gemini_api_key_set: true,
  huggingface_api_token_set: false,
  llm_model: 'gemini-2.0-flash',
  persona: { tsundere_level: 0.5, scientific_formality: 0.5, safe_mode: true },
  streaming_enabled: true,
  voice_language: 'japanese',
  audio_enable_voice: true,
  memory_enable_long_term: true,
  web_search_enabled: false,
  tools_enabled: false,
  reminders_enabled: false,
}

const jsonResponse = (payload: unknown, status = 200) =>
  new Response(JSON.stringify(payload), {
    status,
    headers: { 'Content-Type': 'application/json' },
  })

describe('useSettingsData', () => {
  let fetchMock: ReturnType<typeof vi.fn>
  let versionsResponse: unknown

  beforeEach(() => {
    window.localStorage.clear()
    versionsResponse = { versions: [], current_fingerprint: '' }
    fetchMock = vi.fn(async (url: RequestInfo | URL, init?: RequestInit) => {
      const path = String(url)
      if (path.endsWith('/api/settings') && (!init || !init.method || init.method === 'GET')) {
        return jsonResponse(baseSettings)
      }
      if (path.endsWith('/api/settings') && init?.method === 'PUT') {
        const update = JSON.parse(String(init.body))
        return jsonResponse({
          settings: { ...baseSettings, ...update },
          needs_restart: true,
          restart_reason: 'Tool settings changed',
        })
      }
      if (path.endsWith('/api/persona/versions')) {
        return jsonResponse(versionsResponse)
      }
      throw new Error(`Unexpected fetch: ${path}`)
    })
    vi.stubGlobal('fetch', fetchMock)
  })

  afterEach(() => {
    vi.unstubAllGlobals()
  })

  it('loads settings on mount and seeds lifted form state via onSettingsLoaded', async () => {
    const onSettingsLoaded = vi.fn()
    const { result } = renderHook(() => useSettingsData({ onSettingsLoaded }))

    expect(result.current.loading).toBe(true)
    expect(result.current.settings).toBeNull()

    await waitFor(() => expect(result.current.loading).toBe(false))
    expect(result.current.loadError).toBeNull()
    expect(result.current.settings).toMatchObject({ llm_model: 'gemini-2.0-flash' })
    expect(onSettingsLoaded).toHaveBeenCalledTimes(1)
    expect(onSettingsLoaded).toHaveBeenCalledWith(
      expect.objectContaining({
        providers: expect.objectContaining({ tts: 'qwen3_tts' }),
        persona: expect.objectContaining({ tsundere_level: 0.5 }),
      }),
    )
  })

  it('fetches the persona-version list on mount', async () => {
    versionsResponse = {
      versions: [
        {
          id: 'v-1',
          name: 'Original',
          created_at: '2026-07-01T10:00:00',
          pinned: false,
          fingerprint: 'fp-1',
          is_current: true,
          diff_vs_current: null,
        },
      ],
      current_fingerprint: 'fp-1',
    }
    const { result } = renderHook(() => useSettingsData())

    await waitFor(() => expect(result.current.personaVersions).toHaveLength(1))
    expect(result.current.personaVersions[0]).toMatchObject({ id: 'v-1', name: 'Original' })
    expect(result.current.versionsCurrentFp).toBe('fp-1')
  })

  it('surfaces a load error when GET /api/settings fails', async () => {
    fetchMock.mockImplementation(async (url: RequestInfo | URL, init?: RequestInit) => {
      const path = String(url)
      if (path.endsWith('/api/settings') && (!init || !init.method || init.method === 'GET')) {
        return jsonResponse({ detail: 'nope' }, 500)
      }
      if (path.endsWith('/api/persona/versions')) return jsonResponse(versionsResponse)
      throw new Error(`Unexpected fetch: ${path}`)
    })
    const onSettingsLoaded = vi.fn()
    const { result } = renderHook(() => useSettingsData({ onSettingsLoaded }))

    await waitFor(() => expect(result.current.loading).toBe(false))
    expect(result.current.settings).toBeNull()
    expect(result.current.loadError).toMatch(/could not load settings/i)
    expect(onSettingsLoaded).not.toHaveBeenCalled()
  })

  it('putSettings PUTs the payload and applyUpdateResponse surfaces the restart notice', async () => {
    const { result } = renderHook(() => useSettingsData())
    await waitFor(() => expect(result.current.loading).toBe(false))

    let data: SettingsUpdateResponse | undefined
    await act(async () => {
      data = await result.current.putSettings({ tools_enabled: true })
    })

    const put = fetchMock.mock.calls.find(([, init]) => init?.method === 'PUT')
    expect(put).toBeTruthy()
    expect(JSON.parse(String(put![1]!.body))).toEqual({ tools_enabled: true })

    act(() => {
      result.current.applyUpdateResponse(data!)
    })
    expect(result.current.settings).toMatchObject({ tools_enabled: true })
    expect(result.current.restartNotice).toBe('Tool settings changed')
  })

  it('applyUpdateResponse leaves the restart notice alone when no restart is needed', async () => {
    const { result } = renderHook(() => useSettingsData())
    await waitFor(() => expect(result.current.loading).toBe(false))

    act(() => {
      result.current.applyUpdateResponse({
        settings: { ...baseSettings, web_search_enabled: true },
        needs_restart: false,
        restart_reason: null,
      })
    })
    expect(result.current.settings).toMatchObject({ web_search_enabled: true })
    expect(result.current.restartNotice).toBeNull()
  })

  it('putSettings throws the FastAPI detail message on failure', async () => {
    fetchMock.mockImplementation(async (url: RequestInfo | URL, init?: RequestInit) => {
      const path = String(url)
      if (path.endsWith('/api/settings') && (!init || !init.method || init.method === 'GET')) {
        return jsonResponse(baseSettings)
      }
      if (path.endsWith('/api/settings') && init?.method === 'PUT') {
        // Pydantic's 422 shape: the "Value error, " prefix must be stripped.
        return jsonResponse({ detail: [{ msg: 'Value error, Invalid Gemini key' }] }, 422)
      }
      if (path.endsWith('/api/persona/versions')) return jsonResponse(versionsResponse)
      throw new Error(`Unexpected fetch: ${path}`)
    })
    const { result } = renderHook(() => useSettingsData())
    await waitFor(() => expect(result.current.loading).toBe(false))

    await expect(result.current.putSettings({ gemini_api_key: 'bad' })).rejects.toThrow(
      'Invalid Gemini key',
    )
  })
})
