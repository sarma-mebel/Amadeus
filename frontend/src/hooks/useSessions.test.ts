// @vitest-environment jsdom
import { beforeEach, afterEach, describe, expect, it, vi } from 'vitest'
import { act, cleanup, renderHook } from '@testing-library/react'
import type { SetStateAction } from 'react'
import { resolveUserId, useSessions } from './useSessions'
import {
  createSession,
  deleteSession,
  fetchSessionTranscript,
  listSessions,
  type SessionSummary,
  type SessionTranscriptMessage,
} from '../sessions'
import { clearStoredTranscript } from '../transcript'
import { kurisuBridge } from '../vrm/kurisuBridge'
import { diagnostics } from '../diagnostics'
import type { Message } from '../chat/messages'

vi.mock('../sessions', () => ({
  listSessions: vi.fn(),
  createSession: vi.fn(),
  deleteSession: vi.fn(),
  fetchSessionTranscript: vi.fn(),
}))
vi.mock('../transcript', () => ({ clearStoredTranscript: vi.fn() }))
vi.mock('../diagnostics', () => ({ diagnostics: { increment: vi.fn() } }))
vi.mock('../vrm/kurisuBridge', () => ({
  kurisuBridge: { lipSync: { stop: vi.fn() } },
}))

const summary = (id: string): SessionSummary => ({
  session_id: id,
  title: `Chat ${id}`,
  created_at: '2026-01-01T00:00:00Z',
  updated_at: '2026-01-02T00:00:00Z',
  message_count: 2,
})

const entry = (role: 'user' | 'assistant', content: string): SessionTranscriptMessage => ({
  role,
  content,
  ja: null,
  emotion: null,
  timestamp: '2026-01-01T00:00:00Z',
  turn_id: null,
})

describe('useSessions', () => {
  let messages: Message[]
  let restoredCountRef: { current: number }
  let cancelDeferredTts: ReturnType<typeof vi.fn>

  const mount = () =>
    renderHook(() =>
      useSessions({
        restoredCountRef,
        setMessages: (updater: SetStateAction<Message[]>) => {
          messages = typeof updater === 'function' ? updater(messages) : updater
        },
        cancelDeferredTts,
      }),
    )

  /** Flushes the mount probe / any pending async work. */
  const flush = () => act(async () => {})

  beforeEach(() => {
    vi.clearAllMocks()
    window.localStorage.clear()
    messages = []
    restoredCountRef = { current: 0 }
    cancelDeferredTts = vi.fn()
    vi.mocked(listSessions).mockResolvedValue([])
    vi.mocked(fetchSessionTranscript).mockResolvedValue([])
  })

  afterEach(() => {
    cleanup()
  })

  it('resolveUserId mints an id once and returns the stored one after', () => {
    const first = resolveUserId()
    expect(first).toBeTruthy()
    expect(window.localStorage.getItem('user_id')).toBe(first)
    expect(resolveUserId()).toBe(first)
  })

  it('mount probe restores the server transcript ahead of live messages', async () => {
    window.localStorage.setItem('session_id', 's1')
    // A live greeting already landed before the probe resolved.
    messages = [{ role: 'assistant', content: 'greeting', timestamp: Date.now() }]
    vi.mocked(listSessions).mockResolvedValue([summary('s1')])
    vi.mocked(fetchSessionTranscript).mockResolvedValue([
      entry('user', 'hi'),
      entry('assistant', 'hello'),
    ])

    const { result } = mount()
    await flush()

    expect(result.current.sessionsSupported).toBe(true)
    expect(result.current.sessions).toHaveLength(1)
    expect(restoredCountRef.current).toBe(2)
    expect(messages.map((msg) => msg.content)).toEqual(['hi', 'hello', 'greeting'])
  })

  it('a 404 probe (older backend) marks sessions unsupported and skips the transcript', async () => {
    window.localStorage.setItem('session_id', 's1')
    vi.mocked(listSessions).mockResolvedValue(null)

    const { result } = mount()
    await flush()

    expect(result.current.sessionsSupported).toBe(false)
    expect(fetchSessionTranscript).not.toHaveBeenCalled()
  })

  it('a failed probe degrades without throwing, leaving support unknown', async () => {
    vi.mocked(listSessions).mockRejectedValue(new Error('offline'))

    const { result } = mount()
    await flush()

    expect(result.current.sessionsSupported).toBeNull()
    expect(result.current.sessionsError).toBeNull()
    // Degrade-loudly: a hidden sidebar toggle from a broken probe must be
    // distinguishable from "old backend" in Diagnostics.
    expect(diagnostics.increment).toHaveBeenCalledWith(
      'sessions.probe_failed',
      expect.stringContaining('offline'),
    )
  })

  it('startNewConversation adopts the new id and clears the transcript', async () => {
    vi.mocked(createSession).mockResolvedValue('fresh')
    messages = [{ role: 'user', content: 'old', timestamp: Date.now() }]
    restoredCountRef.current = 1

    const { result } = mount()
    await flush()
    await act(async () => {
      await result.current.startNewConversation()
    })

    expect(cancelDeferredTts).toHaveBeenCalled()
    expect(window.localStorage.getItem('session_id')).toBe('fresh')
    expect(result.current.currentSessionId).toBe('fresh')
    expect(messages).toEqual([])
    expect(restoredCountRef.current).toBe(0)
    expect(clearStoredTranscript).toHaveBeenCalled()
  })

  it('startNewConversation failure surfaces sessionsError instead of throwing', async () => {
    vi.mocked(createSession).mockRejectedValue(new Error('500'))

    const { result } = mount()
    await flush()
    await act(async () => {
      await result.current.startNewConversation()
    })

    expect(result.current.sessionsError).toBe('Could not start a new conversation.')
  })

  it('switchToSession loads the transcript, adopts the id, and stops the voice', async () => {
    window.localStorage.setItem('session_id', 's1')
    const { result } = mount()
    await flush()
    vi.mocked(fetchSessionTranscript).mockResolvedValue([entry('assistant', 'from s2')])

    await act(async () => {
      await result.current.switchToSession('s2')
    })

    expect(window.localStorage.getItem('session_id')).toBe('s2')
    expect(messages.map((msg) => msg.content)).toEqual(['from s2'])
    expect(restoredCountRef.current).toBe(1)
    expect(kurisuBridge.lipSync?.stop).toHaveBeenCalled()
    expect(cancelDeferredTts).toHaveBeenCalled()
  })

  it('switchToSession failure keeps the current session and sets sessionsError', async () => {
    window.localStorage.setItem('session_id', 's1')
    const { result } = mount()
    await flush()
    vi.mocked(fetchSessionTranscript).mockRejectedValue(new Error('gone'))

    await act(async () => {
      await result.current.switchToSession('s2')
    })

    expect(window.localStorage.getItem('session_id')).toBe('s1')
    expect(result.current.sessionsError).toBe('Could not load that conversation.')
  })

  it('removeSession of a non-active conversation deletes it without starting a new one', async () => {
    window.localStorage.setItem('session_id', 's1')
    vi.mocked(listSessions).mockResolvedValue([summary('s1'), summary('s2')])

    const { result } = mount()
    await flush()
    // The post-delete refresh re-lists from the server, which no longer has s2.
    vi.mocked(listSessions).mockResolvedValue([summary('s1')])
    await act(async () => {
      await result.current.removeSession('s2')
    })

    expect(deleteSession).toHaveBeenCalledWith('s2', window.localStorage.getItem('user_id'))
    expect(result.current.sessions.map((s) => s.session_id)).not.toContain('s2')
    expect(createSession).not.toHaveBeenCalled()
  })

  it('removeSession of the active conversation starts a fresh one', async () => {
    window.localStorage.setItem('session_id', 's1')
    vi.mocked(listSessions).mockResolvedValue([summary('s1')])
    vi.mocked(createSession).mockResolvedValue('fresh')

    const { result } = mount()
    await flush()
    await act(async () => {
      await result.current.removeSession('s1')
    })

    expect(deleteSession).toHaveBeenCalledWith('s1', window.localStorage.getItem('user_id'))
    expect(createSession).toHaveBeenCalled()
    expect(window.localStorage.getItem('session_id')).toBe('fresh')
  })
})
