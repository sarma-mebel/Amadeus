// @vitest-environment jsdom
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { act, cleanup, renderHook, waitFor } from '@testing-library/react'
import type { MutableRefObject } from 'react'
import { useChatTurn, type UseChatTurnOptions } from './useChatTurn'
import type { Message } from '../chat/messages'
import type { PendingImage } from '../images'
import type { ChatImageAttachment } from '../chat/payload'

const chatReply = {
  reply: 'Hmph. It works.',
  session_id: 'sess-hook-1',
  turn_id: 't1',
  timings: { llm_ms: 5, tts_ms: null, total_ms: 9 },
  sentences: [],
}

const jsonResponse = (payload: unknown, status = 200) =>
  new Response(JSON.stringify(payload), {
    status,
    headers: { 'Content-Type': 'application/json' },
  })

describe('useChatTurn', () => {
  let fetchMock: ReturnType<typeof vi.fn>
  let wsAttempts: number

  afterEach(() => {
    cleanup()
    vi.unstubAllGlobals()
  })

  beforeEach(() => {
    window.localStorage.clear()
    wsAttempts = 0
    Element.prototype.scrollIntoView = vi.fn()
    Element.prototype.scrollTo = vi.fn()

    // Simulate an unreachable streaming endpoint: streamChat catches the
    // constructor throw and reports a connect failure, which must drop the
    // turn onto the POST path (the stream path itself is unit-tested in
    // src/chat/voiceTurn.test.ts).
    class ExplodingWebSocket {
      constructor() {
        wsAttempts += 1
        throw new Error('no websocket in this test')
      }
    }
    vi.stubGlobal('WebSocket', ExplodingWebSocket)

    fetchMock = vi.fn(async (url: RequestInfo | URL, init?: RequestInit) => {
      const path = String(url)
      if (path.endsWith('/api/chat') && init?.method === 'POST') {
        return jsonResponse(chatReply)
      }
      return jsonResponse({}, 404)
    })
    vi.stubGlobal('fetch', fetchMock)
  })

  const setup = (overrides: Partial<UseChatTurnOptions> = {}) => {
    let messages: Message[] = []
    const setMessages = vi.fn(
      (updater: Message[] | ((prev: Message[]) => Message[])) => {
        messages = typeof updater === 'function' ? updater(messages) : updater
      },
    )
    const pinnedToBottomRef: MutableRefObject<boolean> = { current: false }
    const pendingImageRef: MutableRefObject<PendingImage | null> = { current: null }
    const turnImageRef: MutableRefObject<ChatImageAttachment[] | undefined> = {
      current: undefined,
    }
    const options: UseChatTurnOptions = {
      setMessages: setMessages as unknown as UseChatTurnOptions['setMessages'],
      input: '',
      setInput: vi.fn(),
      audioEnabled: false,
      setAudioEnabled: vi.fn(),
      setRequiresAudioUnlock: vi.fn(),
      adoptSessionId: vi.fn(),
      pinnedToBottomRef,
      pendingImageRef,
      turnImageRef,
      setPendingImage: vi.fn(),
      ensureRealLlmConfigured: vi.fn(async () => true),
      handleMemoryCommand: vi.fn(async () => false),
      requestFarewell: vi.fn(async () => false),
      showMicHint: vi.fn(),
      completeTurnMessages: vi.fn((prev: Message[]) => prev),
      takeRestoredVariantMessage: vi.fn(() => null),
      ...overrides,
    }
    const rendered = renderHook((props: UseChatTurnOptions) => useChatTurn(props), {
      initialProps: options,
    })
    return {
      options,
      pinnedToBottomRef,
      pendingImageRef,
      turnImageRef,
      getMessages: () => messages,
      ...rendered,
    }
  }

  it('sends a turn via WS-fallback-to-POST: echoes the user bubble, lands the reply, adopts the session', async () => {
    const { result, options, getMessages, pinnedToBottomRef } = setup()

    await act(async () => {
      await result.current.sendMessage('Hello Kurisu')
    })

    expect(wsAttempts).toBeGreaterThan(0)
    const msgs = getMessages()
    expect(msgs[0]).toMatchObject({ role: 'user', content: 'Hello Kurisu' })
    expect(msgs.some((m) => m.role === 'assistant' && m.content === 'Hmph. It works.')).toBe(true)
    expect(options.adoptSessionId).toHaveBeenCalledWith('sess-hook-1')
    expect(pinnedToBottomRef.current).toBe(true)
    expect(result.current.isLoading).toBe(false)
    expect(result.current.isStreaming).toBe(false)

    const postCall = fetchMock.mock.calls.find(
      ([url, init]) => String(url).endsWith('/api/chat') && init?.method === 'POST',
    )
    expect(postCall).toBeTruthy()
    const body = JSON.parse(String(postCall![1]!.body))
    expect(body.message).toBe('Hello Kurisu')
    expect(body.defer_audio).toBe(true)
  })

  it('echoUser:false does not append a second user bubble (retry path)', async () => {
    const { result, getMessages } = setup()
    await act(async () => {
      await result.current.sendMessage('Hello again', { echoUser: false })
    })
    expect(getMessages().some((m) => m.role === 'user')).toBe(false)
    expect(getMessages().some((m) => m.role === 'assistant')).toBe(true)
  })

  it('exposes the isLoading edge (and its live ref) while a POST is held open', async () => {
    let resolvePost!: () => void
    const postGate = new Promise<void>((resolve) => {
      resolvePost = resolve
    })
    fetchMock.mockImplementation(async (url: RequestInfo | URL, init?: RequestInit) => {
      const path = String(url)
      if (path.endsWith('/api/chat') && init?.method === 'POST') {
        await postGate
        return jsonResponse(chatReply)
      }
      return jsonResponse({}, 404)
    })

    const { result } = setup()
    let turn!: Promise<void>
    act(() => {
      turn = result.current.sendMessage('Hello Kurisu')
    })

    await waitFor(() => expect(result.current.isLoading).toBe(true))
    expect(result.current.isLoadingRef.current).toBe(true)
    expect(result.current.isStreamingRef.current).toBe(false)

    resolvePost()
    await act(async () => {
      await turn
    })
    expect(result.current.isLoading).toBe(false)
    expect(result.current.isLoadingRef.current).toBe(false)
  })

  it('routes slash commands through handleMemoryCommand and clears the input instead of sending', async () => {
    const handleMemoryCommand = vi.fn(async () => true)
    const setInput = vi.fn()
    const { result } = setup({ handleMemoryCommand, setInput })

    await act(async () => {
      await result.current.sendMessage('/remember my birthday')
    })

    expect(handleMemoryCommand).toHaveBeenCalledWith('/remember my birthday')
    expect(setInput).toHaveBeenCalledWith('')
    const chatCalls = fetchMock.mock.calls.filter(
      ([url, init]) => String(url).endsWith('/api/chat') && init?.method === 'POST',
    )
    expect(chatCalls.length).toBe(0)
  })

  it('blocks the turn entirely when ensureRealLlmConfigured says no', async () => {
    const ensureRealLlmConfigured = vi.fn(async () => false)
    const { result, getMessages } = setup({ ensureRealLlmConfigured })

    await act(async () => {
      await result.current.sendMessage('Hello Kurisu')
    })

    expect(getMessages().length).toBe(0)
    expect(wsAttempts).toBe(0)
    const chatCalls = fetchMock.mock.calls.filter(
      ([url, init]) => String(url).endsWith('/api/chat') && init?.method === 'POST',
    )
    expect(chatCalls.length).toBe(0)
  })

  it('hands a goodbye to requestFarewell and skips the chat turn when handled', async () => {
    const requestFarewell = vi.fn(async () => true)
    const { result } = setup({ requestFarewell })

    await act(async () => {
      await result.current.sendMessage('goodbye')
    })

    expect(requestFarewell).toHaveBeenCalledWith('goodbye')
    const chatCalls = fetchMock.mock.calls.filter(
      ([url, init]) => String(url).endsWith('/api/chat') && init?.method === 'POST',
    )
    expect(chatCalls.length).toBe(0)
  })

  it('writes audio_enabled back from the POST response', async () => {
    fetchMock.mockImplementation(async (url: RequestInfo | URL, init?: RequestInit) => {
      const path = String(url)
      if (path.endsWith('/api/chat') && init?.method === 'POST') {
        return jsonResponse({ ...chatReply, audio_enabled: false })
      }
      return jsonResponse({}, 404)
    })
    const setAudioEnabled = vi.fn()
    const { result } = setup({ audioEnabled: true, setAudioEnabled })

    await act(async () => {
      await result.current.sendMessage('Hello Kurisu')
    })

    expect(setAudioEnabled).toHaveBeenCalledWith(false)
  })

  it('consumes the pending vision image into the turn and clears the chip', async () => {
    const setPendingImage = vi.fn()
    const { result, pendingImageRef, turnImageRef } = setup({ setPendingImage })
    pendingImageRef.current = {
      data: 'abc123',
      mime: 'image/png',
      previewUrl: 'blob:x',
      width: 10,
      height: 10,
    }

    await act(async () => {
      await result.current.sendMessage('what is on my screen')
    })

    expect(turnImageRef.current).toEqual([{ data: 'abc123', mime: 'image/png' }])
    expect(setPendingImage).toHaveBeenCalledWith(null)
    const postCall = fetchMock.mock.calls.find(
      ([url, init]) => String(url).endsWith('/api/chat') && init?.method === 'POST',
    )
    const body = JSON.parse(String(postCall![1]!.body))
    expect(body.images).toEqual([{ data: 'abc123', mime: 'image/png' }])
  })

  it('appends a retryable error bubble when the POST fails outright', async () => {
    fetchMock.mockImplementation(async (url: RequestInfo | URL, init?: RequestInit) => {
      const path = String(url)
      if (path.endsWith('/api/chat') && init?.method === 'POST') {
        throw new Error('network down')
      }
      return jsonResponse({}, 404)
    })
    const restored: Message = { role: 'assistant', content: 'previous good reply', timestamp: 1 }
    const takeRestoredVariantMessage = vi.fn(() => restored)
    const onTurnFailure = vi.fn()
    const { result, getMessages } = setup({ takeRestoredVariantMessage, onTurnFailure })

    await act(async () => {
      await result.current.sendMessage('Hello Kurisu')
    })

    const msgs = getMessages()
    const errorBubble = msgs[msgs.length - 1]
    expect(errorBubble.isError).toBe(true)
    expect(errorBubble.errorKind).toBe('network')
    expect(errorBubble.retryText).toBe('Hello Kurisu')
    // The stashed variant is restored right before the error bubble.
    expect(msgs[msgs.length - 2]).toBe(restored)
    expect(result.current.isLoading).toBe(false)
    // A turn failure is exactly when llm_health may have just changed —
    // the caller (ChatInterface) uses this to refresh /api/status so the
    // health banner reacts without a dedicated poll loop.
    expect(onTurnFailure).toHaveBeenCalledTimes(1)
  })

  it('does not require onTurnFailure — a successful turn never calls it', async () => {
    const onTurnFailure = vi.fn()
    const { result } = setup({ onTurnFailure })

    await act(async () => {
      await result.current.sendMessage('Hello Kurisu')
    })

    expect(onTurnFailure).not.toHaveBeenCalled()
  })

  it('ignores an empty/whitespace draft', async () => {
    const { result, getMessages } = setup({ input: '   ' })
    await act(async () => {
      await result.current.sendMessage()
    })
    expect(getMessages().length).toBe(0)
    expect(wsAttempts).toBe(0)
  })
})
