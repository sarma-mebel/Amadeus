// @vitest-environment jsdom
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import '@testing-library/jest-dom/vitest'
import { cleanup, render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'

// Hoisted so the module factories below (evaluated before imports run) can
// close over them; tauriFlag defaults to false so the existing browser-path
// tests above are unaffected without each one having to opt out explicitly.
const { invokeMock, tauriFlag } = vi.hoisted(() => ({
  // Default stub: every command no-ops successfully; individual tests
  // override with mockImplementation to simulate specific commands/failures.
  // Typed explicitly as (cmd: string) => ... (rather than letting it infer
  // param-less from this default body) so every later mockImplementation
  // override — which DOES branch on cmd — type-checks against the same
  // signature.
  invokeMock: vi.fn(async (cmd: string): Promise<unknown> => {
    void cmd
    return undefined
  }),
  tauriFlag: { value: false },
}))
vi.mock('@tauri-apps/api/core', () => ({ invoke: invokeMock }))
vi.mock('../platform', () => ({ isTauri: () => tauriFlag.value }))

import OnboardingWizard from './OnboardingWizard'

const jsonResponse = (payload: unknown) =>
  new Response(JSON.stringify(payload), {
    status: 200,
    headers: { 'Content-Type': 'application/json' },
  })

describe('OnboardingWizard', () => {
  let fetchMock: ReturnType<typeof vi.fn>

  // No vitest `globals` in this repo's config, so RTL's automatic
  // afterEach cleanup never registers — without this, renders stack up
  // across tests and every query starts matching twice.
  afterEach(() => {
    cleanup()
    vi.unstubAllGlobals()
    invokeMock.mockClear()
    invokeMock.mockImplementation(async (cmd: string): Promise<unknown> => {
      void cmd
      return undefined
    })
    tauriFlag.value = false
  })

  beforeEach(() => {
    window.localStorage.clear()
    fetchMock = vi.fn(async (url: RequestInfo | URL, init?: RequestInit) => {
      const path = String(url)
      if (path.endsWith('/api/settings/test') && init?.method === 'POST') {
        return jsonResponse({ ok: true, message: 'Connection OK' })
      }
      if (path.endsWith('/api/settings') && init?.method === 'PUT') {
        return jsonResponse({ settings: {}, needs_restart: false, restart_reason: null })
      }
      if (path.endsWith('/api/settings')) {
        // Mount-time prefill probe for the voice step.
        return jsonResponse({
          is_first_run: true,
          providers: { llm: 'dummy', llm_is_dummy: true, tts: 'melotts_onnx', stt: 'dummy' },
          voice_language: 'japanese',
          audio_enable_voice: true,
        })
      }
      throw new Error(`Unexpected fetch: ${path}`)
    })
    vi.stubGlobal('fetch', fetchMock)
  })

  const putBodies = () =>
    fetchMock.mock.calls
      .filter(([, init]) => (init as RequestInit | undefined)?.method === 'PUT')
      .map(([, init]) => JSON.parse(String((init as RequestInit).body)))

  it('walks the Gemini path: test, save, then toggles in one final PUT', async () => {
    const user = userEvent.setup()
    const onFinished = vi.fn()
    render(<OnboardingWizard onFinished={onFinished} onSkip={() => {}} />)

    // Step 1 — Gemini is preselected and recommended.
    expect(screen.getByRole('radio', { name: /gemini/i })).toHaveAttribute('aria-checked', 'true')
    await user.click(screen.getByRole('button', { name: /next/i }))

    // Step 2 — Next stays locked until the key is saved.
    const nextButton = screen.getByRole('button', { name: /next/i })
    expect(nextButton).toBeDisabled()

    await user.type(screen.getByLabelText(/gemini api key/i), 'test-key-123')
    await user.click(screen.getByRole('button', { name: /^test$/i }))
    expect(await screen.findByText(/connection ok/i)).toBeInTheDocument()

    const testCall = fetchMock.mock.calls.find(([url]) => String(url).endsWith('/api/settings/test'))
    expect(JSON.parse(String((testCall![1] as RequestInit).body))).toEqual({
      kind: 'llm',
      provider: 'gemini',
      api_key: 'test-key-123',
      model: null,
    })

    await user.click(screen.getByRole('button', { name: /^save$/i }))
    await waitFor(() => expect(nextButton).toBeEnabled())
    expect(putBodies()[0]).toEqual({ llm_provider: 'gemini', gemini_api_key: 'test-key-123' })
    await user.click(nextButton)

    // Step 3 — memory defaults on; flip tools on, finish.
    expect(screen.getByRole('checkbox', { name: /long-term memory/i })).toBeChecked()
    await user.click(screen.getByRole('checkbox', { name: /tools/i }))
    await user.click(screen.getByRole('button', { name: /finish/i }))

    await waitFor(() => expect(onFinished).toHaveBeenCalledTimes(1))
    expect(putBodies()[1]).toEqual({
      audio_enable_voice: true,
      memory_enable_long_term: true,
      tools_enabled: true,
      tts_provider: 'melotts_onnx',
      voice_language: 'japanese',
    })
    expect(window.localStorage.getItem('kurisu_onboarding_dismissed')).toBe('true')
  })

  it('skips the key step for the offline demo and persists the dummy provider', async () => {
    const user = userEvent.setup()
    const onFinished = vi.fn()
    render(<OnboardingWizard onFinished={onFinished} onSkip={() => {}} />)

    await user.click(screen.getByRole('radio', { name: /offline demo/i }))
    await user.click(screen.getByRole('button', { name: /next/i }))

    // Straight to step 3 — no key input anywhere.
    expect(screen.queryByLabelText(/api key/i)).not.toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: /finish/i }))

    await waitFor(() => expect(onFinished).toHaveBeenCalledTimes(1))
    expect(putBodies()[0]).toEqual({
      audio_enable_voice: true,
      memory_enable_long_term: true,
      tools_enabled: false,
      tts_provider: 'melotts_onnx',
      voice_language: 'japanese',
      llm_provider: 'dummy',
    })
  })

  it('shows the voice engine and language selects while voice is enabled', async () => {
    const user = userEvent.setup()
    render(<OnboardingWizard onFinished={() => {}} onSkip={() => {}} />)

    await user.click(screen.getByRole('radio', { name: /offline demo/i }))
    await user.click(screen.getByRole('button', { name: /next/i }))

    const engineSelect = await screen.findByLabelText(/voice engine/i)
    expect(engineSelect).toHaveValue('melotts_onnx')
    // The GPU cloned voice is not offered unless the backend already runs it.
    expect(screen.queryByRole('option', { name: /gpu/i })).not.toBeInTheDocument()
    expect(screen.getByLabelText(/spoken language/i)).toHaveValue('japanese')

    // Turning voice off hides the whole voice config block.
    await user.click(screen.getByRole('checkbox', { name: /voice replies/i }))
    expect(screen.queryByLabelText(/voice engine/i)).not.toBeInTheDocument()
  })

  it('hints when the pasted key does not match the provider shape', async () => {
    const user = userEvent.setup()
    render(<OnboardingWizard onFinished={() => {}} onSkip={() => {}} />)

    await user.click(screen.getByRole('button', { name: /next/i }))
    await user.type(screen.getByLabelText(/gemini api key/i), 'sk-looks-like-openai')
    expect(screen.getByText(/start with "AIza"/i)).toBeInTheDocument()

    // A plausible Gemini key clears the hint.
    await user.clear(screen.getByLabelText(/gemini api key/i))
    await user.type(screen.getByLabelText(/gemini api key/i), 'AIzaSyExample123')
    expect(screen.queryByText(/start with "AIza"/i)).not.toBeInTheDocument()
  })

  it('surfaces needs_restart instead of closing silently', async () => {
    const user = userEvent.setup()
    const onFinished = vi.fn()
    fetchMock.mockImplementation(async (url: RequestInfo | URL, init?: RequestInit) => {
      const path = String(url)
      if (path.endsWith('/api/settings') && init?.method === 'PUT') {
        return jsonResponse({ settings: {}, needs_restart: true, restart_reason: 'voice engine change' })
      }
      if (path.endsWith('/api/settings')) {
        return jsonResponse({
          is_first_run: true,
          providers: { llm: 'dummy', llm_is_dummy: true, tts: 'melotts_onnx', stt: 'dummy' },
          voice_language: 'japanese',
          audio_enable_voice: true,
        })
      }
      throw new Error(`Unexpected fetch: ${path}`)
    })
    render(<OnboardingWizard onFinished={onFinished} onSkip={() => {}} />)

    await user.click(screen.getByRole('radio', { name: /offline demo/i }))
    await user.click(screen.getByRole('button', { name: /next/i }))
    await user.click(screen.getByRole('button', { name: /finish/i }))

    expect(await screen.findByText(/restart amadeus to apply/i)).toBeInTheDocument()
    expect(onFinished).not.toHaveBeenCalled()
    // Onboarding is still dismissed (the save succeeded), and Close hands off.
    expect(window.localStorage.getItem('kurisu_onboarding_dismissed')).toBe('true')
    await user.click(screen.getByRole('button', { name: /^close$/i }))
    expect(onFinished).toHaveBeenCalledTimes(1)
  })

  it('offers the skip escape on every step', async () => {
    const user = userEvent.setup()
    const onSkip = vi.fn()
    render(<OnboardingWizard onFinished={() => {}} onSkip={onSkip} />)

    await user.click(screen.getByRole('button', { name: /skip for now/i }))
    expect(onSkip).toHaveBeenCalledTimes(1)

    await user.click(screen.getByRole('button', { name: /next/i }))
    await user.click(screen.getByRole('button', { name: /skip for now/i }))
    expect(onSkip).toHaveBeenCalledTimes(2)
  })

  it('does not offer "Start with Windows" or touch any Tauri command outside Tauri', async () => {
    const user = userEvent.setup()
    const onFinished = vi.fn()
    render(<OnboardingWizard onFinished={onFinished} onSkip={() => {}} />)

    await user.click(screen.getByRole('radio', { name: /offline demo/i }))
    await user.click(screen.getByRole('button', { name: /next/i }))
    expect(screen.queryByRole('checkbox', { name: /start with windows/i })).not.toBeInTheDocument()

    await user.click(screen.getByRole('button', { name: /finish/i }))
    await waitFor(() => expect(onFinished).toHaveBeenCalledTimes(1))
    expect(invokeMock).not.toHaveBeenCalled()
  })

  it('on Tauri, opens the pet window immediately and applies the autostart choice on finish', async () => {
    tauriFlag.value = true
    const user = userEvent.setup()
    const onFinished = vi.fn()
    render(<OnboardingWizard onFinished={onFinished} onSkip={() => {}} />)

    await user.click(screen.getByRole('radio', { name: /offline demo/i }))
    await user.click(screen.getByRole('button', { name: /next/i }))

    const autostartToggle = screen.getByRole('checkbox', { name: /start with windows/i })
    expect(autostartToggle).not.toBeChecked()
    await user.click(autostartToggle)

    await user.click(screen.getByRole('button', { name: /finish/i }))

    await waitFor(() => expect(onFinished).toHaveBeenCalledTimes(1))
    // Pet window opens THIS run (no relaunch needed) — additive to the
    // pet-default marker write, which onboarding.ts already covers.
    expect(invokeMock).toHaveBeenCalledWith('toggle_pet_window')
    expect(invokeMock).toHaveBeenCalledWith('set_autostart', { enabled: true })
    expect(window.localStorage.getItem('kurisu_onboarding_dismissed')).toBe('true')
  })

  it('on Tauri, leaves autostart untouched when the checkbox stays unchecked', async () => {
    tauriFlag.value = true
    const user = userEvent.setup()
    const onFinished = vi.fn()
    render(<OnboardingWizard onFinished={onFinished} onSkip={() => {}} />)

    await user.click(screen.getByRole('radio', { name: /offline demo/i }))
    await user.click(screen.getByRole('button', { name: /next/i }))
    await user.click(screen.getByRole('button', { name: /finish/i }))

    await waitFor(() => expect(onFinished).toHaveBeenCalledTimes(1))
    expect(invokeMock).toHaveBeenCalledWith('toggle_pet_window')
    expect(invokeMock).not.toHaveBeenCalledWith('set_autostart', expect.anything())
  })

  it('a failed toggle_pet_window is non-fatal — onboarding still finishes', async () => {
    tauriFlag.value = true
    invokeMock.mockImplementation(async (cmd: string) => {
      if (cmd === 'toggle_pet_window') throw new Error('window not ready')
      return undefined
    })
    const user = userEvent.setup()
    const onFinished = vi.fn()
    render(<OnboardingWizard onFinished={onFinished} onSkip={() => {}} />)

    await user.click(screen.getByRole('radio', { name: /offline demo/i }))
    await user.click(screen.getByRole('button', { name: /next/i }))
    await user.click(screen.getByRole('button', { name: /finish/i }))

    await waitFor(() => expect(onFinished).toHaveBeenCalledTimes(1))
  })

  it('a failed set_autostart shows an inline note but still finishes onboarding', async () => {
    tauriFlag.value = true
    invokeMock.mockImplementation(async (cmd: string) => {
      if (cmd === 'set_autostart') throw new Error('registry access denied')
      return undefined
    })
    const user = userEvent.setup()
    const onFinished = vi.fn()
    render(<OnboardingWizard onFinished={onFinished} onSkip={() => {}} />)

    await user.click(screen.getByRole('radio', { name: /offline demo/i }))
    await user.click(screen.getByRole('button', { name: /next/i }))
    await user.click(screen.getByRole('checkbox', { name: /start with windows/i }))
    await user.click(screen.getByRole('button', { name: /finish/i }))

    // Non-blocking: onFinished still fires despite the autostart failure.
    await waitFor(() => expect(onFinished).toHaveBeenCalledTimes(1))
    expect(screen.getByText(/registry access denied/i)).toBeInTheDocument()
  })
})
