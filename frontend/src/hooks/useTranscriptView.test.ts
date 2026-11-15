// @vitest-environment jsdom
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { cleanup, renderHook } from '@testing-library/react'
import { createRef } from 'react'
import type { MutableRefObject } from 'react'
import { useTranscriptView, type UseTranscriptViewOptions } from './useTranscriptView'
import type { Message } from '../chat/messages'

const message = (overrides: Partial<Message>): Message => ({
  role: 'user',
  content: 'hi',
  timestamp: 1000,
  ...overrides,
})

describe('useTranscriptView', () => {
  afterEach(() => {
    cleanup()
    vi.unstubAllGlobals()
  })

  beforeEach(() => {
    window.localStorage.clear()
    Element.prototype.scrollIntoView = vi.fn()
    Element.prototype.scrollTo = vi.fn()
  })

  const setup = (overrides: Partial<UseTranscriptViewOptions> = {}) => {
    const pinnedToBottomRef: MutableRefObject<boolean> = { current: true }
    const options: UseTranscriptViewOptions = {
      messages: [],
      isLoading: false,
      isStreaming: false,
      input: '',
      textareaRef: createRef<HTMLTextAreaElement>(),
      pinnedToBottomRef,
      ...overrides,
    }
    const rendered = renderHook((props: UseTranscriptViewOptions) => useTranscriptView(props), {
      initialProps: options,
    })
    return { options, pinnedToBottomRef, ...rendered }
  }

  it('starts a new visual group on speaker change or a >5min same-speaker gap', () => {
    const base = 1_000_000
    const messages: Message[] = [
      message({ role: 'user', timestamp: base }),
      message({ role: 'user', timestamp: base + 1000 }),
      message({ role: 'assistant', content: 'yo', timestamp: base + 2000 }),
      message({ role: 'assistant', content: 'later', timestamp: base + 2000 + 6 * 60 * 1000 }),
    ]
    const { result } = setup({ messages })
    expect(result.current.groupStarts).toEqual([true, false, true, true])
  })

  it('keys agent groups by agentName', () => {
    const base = 1_000_000
    const messages: Message[] = [
      message({ role: 'agent', agentName: 'a', timestamp: base }),
      message({ role: 'agent', agentName: 'a', timestamp: base + 1000 }),
      message({ role: 'agent', agentName: 'b', timestamp: base + 2000 }),
    ]
    const { result } = setup({ messages })
    expect(result.current.groupStarts).toEqual([true, false, true])
  })

  it('derives the last assistant line for the VN dialogue box', () => {
    const messages: Message[] = [
      message({ role: 'assistant', content: 'first' }),
      message({ role: 'user', content: 'q' }),
      message({ role: 'assistant', content: 'latest' }),
      message({ role: 'system', content: 'notice' }),
    ]
    const { result } = setup({ messages })
    expect(result.current.lastAssistantText).toBe('latest')
  })

  it('lastAssistantText is empty with no assistant messages', () => {
    const { result } = setup({ messages: [message({ role: 'user' })] })
    expect(result.current.lastAssistantText).toBe('')
  })

  it('persists the transcript to localStorage, but not while streaming', () => {
    const messages: Message[] = [message({ role: 'user', content: 'persist me' })]
    const { rerender, options } = setup({ messages })

    const stored = () => window.localStorage.getItem('kurisu_transcript_v1') ?? ''
    expect(stored()).toContain('persist me')

    // Streaming: new content must NOT be flushed per token…
    const streaming: Message[] = [
      ...messages,
      message({ role: 'assistant', content: 'streamed-partial' }),
    ]
    rerender({ ...options, messages: streaming, isStreaming: true })
    expect(stored()).not.toContain('streamed-partial')

    // …but the flip back to isStreaming=false flushes the final transcript.
    rerender({ ...options, messages: streaming, isStreaming: false })
    expect(stored()).toContain('streamed-partial')
  })

  it('autosizes the textarea when the input value changes', () => {
    const textarea = document.createElement('textarea')
    Object.defineProperty(textarea, 'scrollHeight', { value: 64, configurable: true })
    const textareaRef = { current: textarea }
    const { rerender, options } = setup({ textareaRef })

    rerender({ ...options, input: 'a longer draft' })
    expect(textarea.style.height).toBe('64px')

    // Capped at 200px however tall the content gets.
    Object.defineProperty(textarea, 'scrollHeight', { value: 900, configurable: true })
    rerender({ ...options, input: 'a much\nlonger\ndraft' })
    expect(textarea.style.height).toBe('200px')
  })

  it('handleMessagesScroll re-pins only when the container is near the bottom', () => {
    const { result, pinnedToBottomRef } = setup()
    const container = document.createElement('div')
    Object.defineProperty(container, 'scrollHeight', { value: 1000, configurable: true })
    Object.defineProperty(container, 'clientHeight', { value: 400, configurable: true })
    ;(result.current.messageListRef as MutableRefObject<HTMLDivElement | null>).current =
      container

    container.scrollTop = 0
    result.current.handleMessagesScroll()
    expect(pinnedToBottomRef.current).toBe(false)

    container.scrollTop = 580 // 1000 - 580 - 400 = 20 < 48
    result.current.handleMessagesScroll()
    expect(pinnedToBottomRef.current).toBe(true)
  })
})
