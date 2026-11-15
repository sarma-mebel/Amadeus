// @vitest-environment jsdom
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import '@testing-library/jest-dom/vitest'
import { cleanup, render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'

// Hoisted so the mock factories (evaluated before the imports below run)
// can close over them. tauriFlag defaults false so every pre-existing test
// in this file — which never touches Tauri — is unaffected.
const { invokeMock, tauriFlag } = vi.hoisted(() => ({
  invokeMock: vi.fn(async (cmd: string): Promise<unknown> =>
    cmd === 'get_autostart' ? false : undefined,
  ),
  tauriFlag: { value: false },
}))
vi.mock('../platform', async (importOriginal) => {
  const actual = await importOriginal<typeof import('../platform')>()
  return { ...actual, isTauri: () => tauriFlag.value }
})
vi.mock('@tauri-apps/api/core', () => ({ invoke: invokeMock }))

import SettingsPanel, { type SettingsResponse } from './SettingsPanel'
import { __resetModelRegistryCache } from '../models'

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

const jsonResponse = (payload: unknown) =>
  new Response(JSON.stringify(payload), {
    status: 200,
    headers: { 'Content-Type': 'application/json' },
  })

describe('SettingsPanel', () => {
  let fetchMock: ReturnType<typeof vi.fn>

  // No vitest `globals` in this repo's config, so RTL's automatic
  // afterEach cleanup never registers — without this, renders stack up
  // across tests and every query starts matching twice.
  afterEach(() => {
    cleanup()
    vi.unstubAllGlobals()
    invokeMock.mockClear()
    invokeMock.mockImplementation(async (cmd: string): Promise<unknown> =>
      cmd === 'get_autostart' ? false : undefined,
    )
    tauriFlag.value = false
  })

  // Module-level cache in models.ts (fetchModelRegistry) must not leak
  // between tests — each test wants its own registry/fetch-count outcome.
  let modelsResponse: unknown = { models: [] }

  beforeEach(() => {
    window.localStorage.clear()
    __resetModelRegistryCache()
    modelsResponse = { models: [] }
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
      if (path.endsWith('/static/models/models.json')) {
        return jsonResponse(modelsResponse)
      }
      if (path.endsWith('/api/status')) {
        // DiagnosticsSection fetches this on mount; a real backend counters
        // block isn't the point of these tests, so an empty-but-valid
        // status keeps fetchBackendStatus() happy without extra fixtures.
        return jsonResponse({
          version: '0.0.0-test',
          llm: { provider: 'dummy', class: 'DummyProvider', model: null, is_dummy: true },
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
          stt: { provider: 'dummy', loaded: false, unavailable_reason: null },
          counters: {},
          memory: {
            long_term: false,
            fact_extraction: false,
            episodes: false,
            greeting: false,
            embedder: null,
            summary_provider_active: false,
          },
        })
      }
      throw new Error(`Unexpected fetch: ${path}`)
    })
    vi.stubGlobal('fetch', fetchMock)
  })

  afterEach(() => {
    __resetModelRegistryCache()
  })

  const renderPanel = () =>
    render(
      <SettingsPanel onClose={() => {}} isOnboarding={false} onSkipOnboarding={() => {}} />,
    )

  it('loads settings and shows the feature toggles', async () => {
    renderPanel()
    const memoryToggle = await screen.findByRole('checkbox', { name: /long-term memory/i })
    expect(memoryToggle).toBeChecked()
    expect(screen.getByRole('checkbox', { name: /web search/i })).not.toBeChecked()
    expect(screen.getByRole('checkbox', { name: /reminders/i })).not.toBeChecked()
  })

  it('saves feature toggles and surfaces the restart notice', async () => {
    const user = userEvent.setup()
    renderPanel()

    const toolsToggle = await screen.findByRole('checkbox', { name: /tools \(calculator/i })
    await user.click(toolsToggle)
    await user.click(screen.getByRole('checkbox', { name: /web search/i }))
    await user.selectOptions(
      screen.getByRole('combobox', { name: /voice input/i }),
      'faster_whisper',
    )
    await user.click(screen.getByRole('button', { name: /save features/i }))

    await waitFor(() => {
      const put = fetchMock.mock.calls.find(([, init]) => init?.method === 'PUT')
      expect(put).toBeTruthy()
      const body = JSON.parse(String(put![1]!.body))
      expect(body).toMatchObject({
        tools_enabled: true,
        web_search_enabled: true,
        stt_provider: 'faster_whisper',
        memory_enable_long_term: true,
        reminders_enabled: false,
      })
    })
    expect(await screen.findByText(/restart to apply/i)).toBeInTheDocument()
  })

  it('hides the outfit picker when only the default model is registered', async () => {
    modelsResponse = { models: [] }
    renderPanel()
    await screen.findByRole('checkbox', { name: /long-term memory/i })
    expect(screen.queryByRole('combobox', { name: /model/i })).not.toBeInTheDocument()
  })

  it('shows the outfit picker once a second model is registered, and switching writes localStorage + dispatches the event', async () => {
    modelsResponse = {
      models: [{ id: 'casual', label: 'Casual', file: '/static/models/casual.vrm' }],
    }
    const handler = vi.fn()
    window.addEventListener('kurisu:model-change', handler)

    const user = userEvent.setup()
    renderPanel()

    const modelSelect = await screen.findByRole('combobox', { name: /model/i })
    expect(modelSelect).toBeInTheDocument()

    await user.selectOptions(modelSelect, 'casual')

    expect(window.localStorage.getItem('kurisu_model_v1')).toBe('casual')
    expect(handler).toHaveBeenCalledTimes(1)
    const event = handler.mock.calls[0][0] as CustomEvent<string>
    expect(event.detail).toBe('casual')

    window.removeEventListener('kurisu:model-change', handler)
  })

  it('hides "Start with Windows" (and the pet-default toggle) outside Tauri', async () => {
    renderPanel()
    await screen.findByRole('checkbox', { name: /long-term memory/i })
    expect(screen.queryByRole('checkbox', { name: /start with windows/i })).not.toBeInTheDocument()
    expect(screen.queryByRole('checkbox', { name: /start as desktop pet/i })).not.toBeInTheDocument()
    expect(invokeMock).not.toHaveBeenCalled()
  })

  it('on Tauri, seeds "Start with Windows" from get_autostart and toggling calls set_autostart', async () => {
    tauriFlag.value = true
    invokeMock.mockImplementation(async (cmd: string) => (cmd === 'get_autostart' ? true : undefined))
    const user = userEvent.setup()
    renderPanel()

    const autostartToggle = await screen.findByRole('checkbox', { name: /start with windows/i })
    await waitFor(() => expect(autostartToggle).toBeChecked())

    await user.click(autostartToggle)
    await waitFor(() =>
      expect(invokeMock).toHaveBeenCalledWith('set_autostart', { enabled: false }),
    )
    expect(autostartToggle).not.toBeChecked()
  })

  it('on Tauri, a get_autostart rejection hides the toggle instead of crashing', async () => {
    tauriFlag.value = true
    invokeMock.mockImplementation(async (cmd: string) => {
      if (cmd === 'get_autostart') throw new Error('command not found')
      return undefined
    })
    renderPanel()

    await screen.findByRole('checkbox', { name: /long-term memory/i })
    // Anchor on the POSITIVE signal: the unavailable-state title only renders
    // after the mount-time get_autostart probe rejects. Waiting for the
    // toggle's absence instead was flaky — the toggle is also absent on the
    // initial render BEFORE the probe settles, so that wait could pass
    // immediately and the title assertion raced the state update (lost once
    // on a contended CI runner).
    await screen.findByTitle(/unavailable in this build/i)
    expect(
      screen.queryByRole('checkbox', { name: /start with windows/i }),
    ).not.toBeInTheDocument()
  })

  it('on Tauri, a set_autostart rejection reverts the toggle and shows the error inline', async () => {
    tauriFlag.value = true
    invokeMock.mockImplementation(async (cmd: string) => {
      if (cmd === 'get_autostart') return false
      if (cmd === 'set_autostart') throw new Error('access denied')
      return undefined
    })
    const user = userEvent.setup()
    renderPanel()

    const autostartToggle = await screen.findByRole('checkbox', { name: /start with windows/i })
    await waitFor(() => expect(autostartToggle).not.toBeChecked())

    await user.click(autostartToggle)
    // Optimistic flip happens synchronously with the click...
    await waitFor(() => expect(screen.getByText(/access denied/i)).toBeInTheDocument())
    // ...then reverts once set_autostart rejects.
    expect(autostartToggle).not.toBeChecked()
  })
})
