// @vitest-environment jsdom
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { act, cleanup, renderHook, waitFor } from '@testing-library/react'
import { useOnboardingGate, type UseOnboardingGateOptions } from './useOnboardingGate'
import type { Message } from '../chat/messages'

const jsonResponse = (payload: unknown, status = 200) =>
  new Response(JSON.stringify(payload), {
    status,
    headers: { 'Content-Type': 'application/json' },
  })

const settingsPayload = (overrides: Record<string, unknown> = {}) => ({
  is_first_run: false,
  providers: { llm: 'gemini', llm_is_dummy: false },
  ...overrides,
})

describe('useOnboardingGate', () => {
  let fetchMock: ReturnType<typeof vi.fn>

  afterEach(() => {
    cleanup()
    vi.unstubAllGlobals()
  })

  beforeEach(() => {
    window.localStorage.clear()
    Element.prototype.scrollIntoView = vi.fn()
    Element.prototype.scrollTo = vi.fn()

    fetchMock = vi.fn(async (url: RequestInfo | URL) => {
      const path = String(url)
      if (path.endsWith('/api/settings')) return jsonResponse(settingsPayload())
      return jsonResponse({}, 404)
    })
    vi.stubGlobal('fetch', fetchMock)
  })

  const setup = (overrides: Partial<UseOnboardingGateOptions> = {}) => {
    let messages: Message[] = []
    const setMessages = vi.fn(
      (updater: Message[] | ((prev: Message[]) => Message[])) => {
        messages = typeof updater === 'function' ? updater(messages) : updater
      },
    )
    const tryRequestGreeting = vi.fn()
    const options: UseOnboardingGateOptions = {
      messages,
      setMessages: setMessages as unknown as UseOnboardingGateOptions['setMessages'],
      tryRequestGreeting,
      ...overrides,
    }
    const rendered = renderHook((props: UseOnboardingGateOptions) => useOnboardingGate(props), {
      initialProps: options,
    })
    return { options, tryRequestGreeting, getMessages: () => messages, ...rendered }
  }

  it('resolves the mount probe without opening settings on a configured backend', async () => {
    const { result } = setup()

    await act(async () => {
      await result.current.waitForFirstRunProbe()
    })
    expect(result.current.llmIsDummy).toBe(false)
    expect(result.current.llmIsDummyRef.current).toBe(false)
    expect(result.current.settingsOpen).toBe(false)
  })

  it('auto-opens the onboarding wizard on a first-run dummy backend', async () => {
    fetchMock.mockImplementation(async (url: RequestInfo | URL) => {
      const path = String(url)
      if (path.endsWith('/api/settings')) {
        return jsonResponse(
          settingsPayload({ is_first_run: true, providers: { llm: 'dummy', llm_is_dummy: true } }),
        )
      }
      return jsonResponse({}, 404)
    })

    const { result } = setup()
    await waitFor(() => expect(result.current.settingsOpen).toBe(true))
    expect(result.current.settingsIsOnboarding).toBe(true)
    expect(result.current.llmIsDummy).toBe(true)
    expect(result.current.llmIsDummyRef.current).toBe(true)
  })

  it('does not auto-open the wizard when onboarding was dismissed before', async () => {
    window.localStorage.setItem('kurisu_onboarding_dismissed', 'true')
    fetchMock.mockImplementation(async () =>
      jsonResponse(
        settingsPayload({ is_first_run: true, providers: { llm: 'dummy', llm_is_dummy: true } }),
      ),
    )

    const { result } = setup()
    await act(async () => {
      await result.current.waitForFirstRunProbe()
    })
    expect(result.current.settingsOpen).toBe(false)
    // The probe still resolved the dummy truth for the banner.
    expect(result.current.llmIsDummy).toBe(true)
  })

  it('ensureRealLlmConfigured passes through on a real provider', async () => {
    const { result } = setup()
    let allowed: boolean | undefined
    await act(async () => {
      allowed = await result.current.ensureRealLlmConfigured()
    })
    expect(allowed).toBe(true)
    expect(result.current.settingsOpen).toBe(false)
  })

  it('ensureRealLlmConfigured blocks the turn, opens settings and appends one system notice on an unconfigured dummy', async () => {
    fetchMock.mockImplementation(async () =>
      jsonResponse(settingsPayload({ providers: { llm: 'gemini', llm_is_dummy: true } })),
    )
    const { result, getMessages } = setup()

    let allowed: boolean | undefined
    await act(async () => {
      allowed = await result.current.ensureRealLlmConfigured()
    })
    expect(allowed).toBe(false)
    expect(result.current.settingsOpen).toBe(true)
    expect(result.current.settingsIsOnboarding).toBe(true)
    expect(getMessages().some((m) => m.content.includes('offline demo mode'))).toBe(true)

    // A second blocked send doesn't stack a duplicate notice.
    await act(async () => {
      await result.current.ensureRealLlmConfigured()
    })
    expect(getMessages().filter((m) => m.content.includes('offline demo mode')).length).toBe(1)
  })

  it('ensureRealLlmConfigured respects an explicitly chosen offline demo', async () => {
    window.localStorage.setItem('kurisu_onboarding_dismissed', 'true')
    fetchMock.mockImplementation(async () =>
      jsonResponse(settingsPayload({ providers: { llm: 'dummy', llm_is_dummy: true } })),
    )
    const { result } = setup()

    let allowed: boolean | undefined
    await act(async () => {
      allowed = await result.current.ensureRealLlmConfigured()
    })
    expect(allowed).toBe(true)
    expect(result.current.settingsOpen).toBe(false)
    // The banner state still reflects the dummy truth.
    expect(result.current.llmIsDummy).toBe(true)
  })

  it('regreets after onboarding only when the LLM is real and no assistant message landed', async () => {
    const { result, tryRequestGreeting } = setup()

    await act(async () => {
      await result.current.maybeRegreetAfterOnboarding(true)
    })
    expect(tryRequestGreeting).toHaveBeenCalledTimes(1)

    // Not an onboarding close — no probe-driven regreet.
    await act(async () => {
      await result.current.maybeRegreetAfterOnboarding(false)
    })
    expect(tryRequestGreeting).toHaveBeenCalledTimes(1)
  })

  it('does not regreet when an assistant message already landed or the LLM is still dummy', async () => {
    const assistant: Message = { role: 'assistant', content: 'yo', timestamp: Date.now() }
    const withReply = setup({ messages: [assistant] })
    await act(async () => {
      await withReply.result.current.maybeRegreetAfterOnboarding(true)
    })
    expect(withReply.tryRequestGreeting).not.toHaveBeenCalled()
    cleanup()

    fetchMock.mockImplementation(async () =>
      jsonResponse(settingsPayload({ providers: { llm: 'dummy', llm_is_dummy: true } })),
    )
    const stillDummy = setup()
    await act(async () => {
      await stillDummy.result.current.maybeRegreetAfterOnboarding(true)
    })
    expect(stillDummy.tryRequestGreeting).not.toHaveBeenCalled()
  })
})
