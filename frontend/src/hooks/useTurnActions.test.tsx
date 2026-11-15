// @vitest-environment jsdom
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { act, cleanup, renderHook, waitFor } from '@testing-library/react'
import { createRef } from 'react'
import type { MutableRefObject } from 'react'
import { useTurnActions, type UseTurnActionsOptions } from './useTurnActions'
import type { Message } from '../chat/messages'

const jsonResponse = (payload: unknown, status = 200) =>
  new Response(JSON.stringify(payload), {
    status,
    headers: { 'Content-Type': 'application/json' },
  })

/** Turn-aware fetch stub (same pattern as ChatInterface.test.tsx): answers
 * /rewind + /amend with simple ok responses, records every call. */
const buildFetchMock = () =>
  vi.fn(async (url: RequestInfo | URL, init?: RequestInit) => {
    const path = String(url)
    if (path.includes('/rewind') && init?.method === 'POST') {
      return jsonResponse({ session_id: 'sess-1', status: 'ok' })
    }
    if (path.includes('/amend') && init?.method === 'POST') {
      return jsonResponse({ session_id: 'sess-1', status: 'ok' })
    }
    return jsonResponse({}, 404)
  })

describe('useTurnActions', () => {
  let fetchMock: ReturnType<typeof buildFetchMock>

  afterEach(() => {
    cleanup()
    vi.unstubAllGlobals()
    vi.restoreAllMocks()
  })

  beforeEach(() => {
    window.localStorage.clear()
    window.localStorage.setItem('session_id', 'sess-1')
    Element.prototype.scrollIntoView = vi.fn()
    Element.prototype.scrollTo = vi.fn()
    fetchMock = buildFetchMock()
    vi.stubGlobal('fetch', fetchMock)
  })

  const setup = (
    initialMessages: Message[],
    overrides: Partial<UseTurnActionsOptions> = {},
  ) => {
    let messages = initialMessages
    const setMessages = vi.fn(
      (updater: Message[] | ((prev: Message[]) => Message[])) => {
        messages = typeof updater === 'function' ? updater(messages) : updater
      },
    )
    const restoredCountRef: MutableRefObject<number> = { current: 0 }
    const isLoadingRef: MutableRefObject<boolean> = { current: false }
    const isStreamingRef: MutableRefObject<boolean> = { current: false }
    const lastAssistantIndex = (() => {
      for (let i = initialMessages.length - 1; i >= 0; i -= 1) {
        if (initialMessages[i].role === 'assistant' && !initialMessages[i].isError) return i
      }
      return -1
    })()
    const options: UseTurnActionsOptions = {
      messages: initialMessages,
      setMessages: setMessages as unknown as UseTurnActionsOptions['setMessages'],
      restoredCountRef,
      isLoading: false,
      isStreaming: false,
      isLoadingRef,
      isStreamingRef,
      lastAssistantIndex,
      sendMessage: vi.fn(async () => {}),
      stopCurrentStream: vi.fn(),
      cancelDeferredTts: vi.fn(),
      setInput: vi.fn(),
      textareaRef: createRef<HTMLTextAreaElement>(),
      ...overrides,
    }
    const rendered = renderHook((props: UseTurnActionsOptions) => useTurnActions(props), {
      initialProps: options,
    })
    return { options, isLoadingRef, restoredCountRef, getMessages: () => messages, ...rendered }
  }

  const turn = (turnId: string, reply = 'First answer.'): Message[] => [
    { role: 'user', content: 'Hello Kurisu', timestamp: 1000, turnId },
    { role: 'assistant', content: reply, timestamp: 2000, turnId, id: `a-${turnId}` },
  ]

  it('retry removes a pure error bubble and re-sends without echoing the user bubble', () => {
    const errorBubble: Message = {
      role: 'assistant',
      content: 'Connection failed.',
      timestamp: 3000,
      isError: true,
      errorKind: 'network',
      retryText: 'Hello Kurisu',
    }
    const { result, options, getMessages } = setup([...turn('t1'), errorBubble])

    act(() => result.current.handleRetry(errorBubble))

    expect(getMessages()).not.toContain(errorBubble)
    expect(options.sendMessage).toHaveBeenCalledWith('Hello Kurisu', { echoUser: false })
  })

  it('retry keeps an interrupted-stream bubble (partial content) in place', () => {
    const streamBubble: Message = {
      role: 'assistant',
      content: 'partial…',
      timestamp: 3000,
      isError: true,
      errorKind: 'stream',
      retryText: 'Hello Kurisu',
    }
    const { result, options, getMessages } = setup([...turn('t1'), streamBubble])

    act(() => result.current.handleRetry(streamBubble))

    expect(getMessages()).toContain(streamBubble)
    expect(options.sendMessage).toHaveBeenCalledWith('Hello Kurisu', { echoUser: false })
  })

  it('retry is blocked while a POST turn is in flight (live ref, not stale state)', () => {
    const errorBubble: Message = {
      role: 'assistant',
      content: 'err',
      timestamp: 3000,
      isError: true,
      errorKind: 'network',
      retryText: 'Hello Kurisu',
    }
    const { result, options, isLoadingRef } = setup([errorBubble])
    isLoadingRef.current = true

    act(() => result.current.handleRetry(errorBubble))
    expect(options.sendMessage).not.toHaveBeenCalled()
  })

  it('regenerate rewinds the turn, drops the old reply, stashes it and re-sends the same prompt', async () => {
    const messages = turn('t1')
    const { result, options, getMessages } = setup(messages)

    act(() => {
      result.current.handleRegenerate(messages[1])
    })
    await waitFor(() => expect(options.sendMessage).toHaveBeenCalled())

    const rewindCall = fetchMock.mock.calls.find(([url]) => String(url).includes('/rewind'))
    expect(rewindCall).toBeTruthy()
    expect(JSON.parse(String(rewindCall![1]!.body)).turn_id).toBe('t1')
    expect(getMessages()).not.toContain(messages[1])
    expect(options.sendMessage).toHaveBeenCalledWith('Hello Kurisu', { echoUser: false })

    // The stash folds onto the replacement reply via completeTurnMessages…
    const replacement: Message[] = [
      messages[0],
      { role: 'assistant', content: 'Second answer.', timestamp: 4000, id: 'a-new' },
    ]
    const folded = result.current.completeTurnMessages(replacement, 'a-new', 't2')
    const newReply = folded.find((m) => m.id === 'a-new')
    expect(newReply?.turnId).toBe('t2')
    expect(newReply?.variants?.map((v) => v.content)).toEqual([
      'First answer.',
      'Second answer.',
    ])
    // …and the stash is cleared: a later completion folds nothing.
    const again = result.current.completeTurnMessages(replacement, 'a-new', 't3')
    expect(again.find((m) => m.id === 'a-new')?.variants).toBeUndefined()
  })

  it('does not fire two rewinds on a double-click regenerate (re-entrancy guard)', async () => {
    let resolveRewind!: () => void
    const rewindGate = new Promise<void>((resolve) => {
      resolveRewind = resolve
    })
    fetchMock.mockImplementation(async (url: RequestInfo | URL, init?: RequestInit) => {
      const path = String(url)
      if (path.includes('/rewind') && init?.method === 'POST') {
        await rewindGate
        return jsonResponse({ session_id: 'sess-1', status: 'ok' })
      }
      return jsonResponse({}, 404)
    })
    const messages = turn('t1')
    const { result, options } = setup(messages)

    act(() => {
      // Both clicks land before the first rewind resolves — a real
      // re-entrancy race, not just a fast synchronous guard check.
      result.current.handleRegenerate(messages[1])
      result.current.handleRegenerate(messages[1])
    })
    resolveRewind()
    await waitFor(() => expect(options.sendMessage).toHaveBeenCalledTimes(1))

    const rewindCalls = fetchMock.mock.calls.filter(([url]) => String(url).includes('/rewind'))
    expect(rewindCalls.length).toBe(1)
  })

  it('regenerate failure pops the stash and reports via a system bubble', async () => {
    fetchMock.mockImplementation(async () => jsonResponse({}, 500))
    const messages = turn('t1')
    const { result, options, getMessages } = setup(messages)

    act(() => {
      result.current.handleRegenerate(messages[1])
    })
    await waitFor(() =>
      expect(
        getMessages().some(
          (m) => m.role === 'system' && m.content.includes("Couldn't regenerate"),
        ),
      ).toBe(true),
    )

    expect(options.sendMessage).not.toHaveBeenCalled()
    expect(result.current.takeRestoredVariantMessage()).toBeNull()
  })

  it('swipe flips the displayed variant and amends it to the server', async () => {
    const message: Message = {
      role: 'assistant',
      content: 'Second answer.',
      timestamp: 2000,
      turnId: 't2',
      id: 'a-1',
      variants: [
        { content: 'First answer.', timestamp: 1000 },
        { content: 'Second answer.', timestamp: 2000 },
      ],
      activeVariant: 1,
    }
    const { result, getMessages } = setup([message])

    act(() => result.current.handleSwipe(message, -1))

    const swiped = getMessages()[0]
    expect(swiped.content).toBe('First answer.')
    expect(swiped.activeVariant).toBe(0)
    await waitFor(() => {
      const amendCall = fetchMock.mock.calls.find(([url]) => String(url).includes('/amend'))
      expect(amendCall).toBeTruthy()
      expect(JSON.parse(String(amendCall![1]!.body)).content).toBe('First answer.')
    })
  })

  it('edit flow: startEdit -> text change -> save rewinds, replaces the bubble, drops the stale reply and re-sends', async () => {
    const messages = turn('t1')
    const { result, options, getMessages } = setup(messages)

    act(() => result.current.handleStartEdit(messages[0]))
    expect(result.current.editing?.target).toBe(messages[0])

    act(() => result.current.handleEditTextChange('Edited question'))
    act(() => {
      result.current.handleEditSave()
    })
    await waitFor(() => expect(options.sendMessage).toHaveBeenCalled())

    const rewindCall = fetchMock.mock.calls.find(([url]) => String(url).includes('/rewind'))
    expect(JSON.parse(String(rewindCall![1]!.body)).turn_id).toBe('t1')
    const msgs = getMessages()
    expect(msgs.some((m) => m.role === 'user' && m.content === 'Edited question')).toBe(true)
    expect(msgs.some((m) => m.role === 'assistant' && m.content === 'First answer.')).toBe(false)
    expect(options.sendMessage).toHaveBeenCalledWith('Edited question', { echoUser: false })
    expect(result.current.editing).toBeNull()
  })

  it('rewind truncates the transcript, refills the input, and shrinks the restored marker', async () => {
    const confirmSpy = vi.spyOn(window, 'confirm').mockReturnValue(true)
    const messages = [...turn('t1'), ...turn('t2', 'Second answer.')]
    const { result, options, getMessages, restoredCountRef } = setup(messages)
    restoredCountRef.current = 4

    await act(async () => {
      await result.current.handleRewind(messages[0])
    })

    expect(confirmSpy).toHaveBeenCalled()
    expect(options.cancelDeferredTts).toHaveBeenCalled()
    expect(getMessages().length).toBe(0)
    expect(restoredCountRef.current).toBe(0)
    expect(options.setInput).toHaveBeenCalledWith('Hello Kurisu')
  })

  it('rewind declined by the confirm dialog is a no-op', async () => {
    const confirmSpy = vi.spyOn(window, 'confirm').mockReturnValue(false)
    const messages = turn('t1')
    const { result, getMessages } = setup(messages)

    await act(async () => {
      await result.current.handleRewind(messages[0])
    })

    expect(getMessages().length).toBe(2)
    expect(fetchMock.mock.calls.filter(([url]) => String(url).includes('/rewind')).length).toBe(0)
    confirmSpy.mockRestore()
  })

  it('takeRestoredVariantMessage rebuilds the latest stashed variant exactly once', async () => {
    const messages = turn('t1')
    // Hold the rewind open so the stash exists while we pop it (the finally
    // block hasn't cleared the in-flight guard yet — the stash survives
    // regardless; only completion/failure clears it).
    let resolveRewind!: () => void
    fetchMock.mockImplementation(async (url: RequestInfo | URL, init?: RequestInit) => {
      if (String(url).includes('/rewind') && init?.method === 'POST') {
        await new Promise<void>((resolve) => {
          resolveRewind = resolve
        })
        return jsonResponse({ session_id: 'sess-1', status: 'ok' })
      }
      return jsonResponse({}, 404)
    })
    const { result, options } = setup(messages)

    act(() => {
      result.current.handleRegenerate(messages[1])
    })

    // The stash is written synchronously, before the rewind resolves.
    const restored = result.current.takeRestoredVariantMessage()
    expect(restored).toMatchObject({ role: 'assistant', content: 'First answer.' })
    // Popped exactly once — a second take returns null.
    expect(result.current.takeRestoredVariantMessage()).toBeNull()

    resolveRewind()
    await waitFor(() => expect(options.sendMessage).toHaveBeenCalled())
  })

  it('row handlers keep a stable identity across rerenders (memoized TranscriptList contract)', () => {
    const messages = turn('t1')
    const { result, rerender, options } = setup(messages)
    const before = {
      retry: result.current.handleRetry,
      regenerate: result.current.handleRegenerate,
      swipe: result.current.handleSwipe,
      edit: result.current.handleStartEdit,
      rewind: result.current.handleRewind,
    }
    rerender({ ...options, isLoading: true })
    expect(result.current.handleRetry).toBe(before.retry)
    expect(result.current.handleRegenerate).toBe(before.regenerate)
    expect(result.current.handleSwipe).toBe(before.swipe)
    expect(result.current.handleStartEdit).toBe(before.edit)
    expect(result.current.handleRewind).toBe(before.rewind)
  })
})
