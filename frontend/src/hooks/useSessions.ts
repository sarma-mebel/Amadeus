/**
 * Conversation-sidebar/session state extracted from ChatInterface: the
 * GET /api/sessions support probe, session list CRUD (new/switch/delete/
 * export), and the localStorage-mirrored active session id. The chat
 * transcript itself stays with the component — this hook receives its
 * setter (plus the restored-count marker) so switching conversations can
 * replace it, exactly as the inline code did.
 */
import { useCallback, useEffect, useState } from 'react'
import type { Dispatch, MutableRefObject, SetStateAction } from 'react'
import { diagnostics } from '../diagnostics'
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
import type { Message } from '../chat/messages'

export const resolveUserId = () => {
  const existingUserId = localStorage.getItem('user_id')
  const userId = existingUserId ?? crypto.randomUUID()
  if (!existingUserId) {
    localStorage.setItem('user_id', userId)
  }
  return userId
}

/** Server transcript rows → chat Message rows (assistant markdown + ja
 * subtitle render exactly like live replies; no TTS replay). */
const transcriptToMessages = (entries: SessionTranscriptMessage[]): Message[] =>
  entries.map((entry, index) => {
    const parsed = Date.parse(entry.timestamp)
    return {
      role: entry.role,
      content: entry.content,
      japanese: entry.ja ?? undefined,
      // Unparseable timestamps degrade to "now" (monotonic via index) so
      // grouping/labels stay sane instead of showing 1970.
      timestamp: Number.isNaN(parsed) ? Date.now() + index : parsed,
      turnId: entry.turn_id ?? undefined,
    }
  })

export interface UseSessionsOptions {
  /** Marks where the restored transcript ends and the live session begins. */
  restoredCountRef: MutableRefObject<number>
  setMessages: Dispatch<SetStateAction<Message[]>>
  /** Cancels the POST path's progressive per-sentence /api/tts loop. */
  cancelDeferredTts: () => void
}

export const useSessions = ({
  restoredCountRef,
  setMessages,
  cancelDeferredTts,
}: UseSessionsOptions) => {
  // Conversation sidebar. null = still probing GET /api/sessions; false =
  // the backend 404'd (older backend) and the toggle stays hidden entirely.
  const [sessionsSupported, setSessionsSupported] = useState<boolean | null>(null)
  const [sidebarOpen, setSidebarOpen] = useState(false)
  const [sessions, setSessions] = useState<SessionSummary[]>([])
  const [sessionsLoading, setSessionsLoading] = useState(false)
  const [sessionsError, setSessionsError] = useState<string | null>(null)
  // Mirrors localStorage's session_id in React state so the sidebar's
  // "current" highlight updates as replies/switches change the session.
  const [currentSessionId, setCurrentSessionId] = useState<string | null>(() =>
    typeof window === 'undefined' ? null : window.localStorage.getItem('session_id'),
  )

  /** Single write path for the active session id: localStorage stays the
   * source every request reads from; the state mirror drives the sidebar.
   * useCallback([]) so consumers can list it in effect deps without those
   * effects ever re-running on identity churn. */
  const adoptSessionId = useCallback((sessionId: string) => {
    localStorage.setItem('session_id', sessionId)
    setCurrentSessionId(sessionId)
  }, [])

  const refreshSessions = async () => {
    setSessionsLoading(true)
    setSessionsError(null)
    try {
      const list = await listSessions(resolveUserId())
      if (list === null) {
        setSessionsSupported(false)
        return
      }
      setSessionsSupported(true)
      setSessions(list)
    } catch {
      setSessionsError('Could not load conversations.')
    } finally {
      setSessionsLoading(false)
    }
  }

  // Probe GET /api/sessions once on mount. 404 → older backend: the sidebar
  // toggle stays hidden and localStorage transcripts remain the only
  // persistence. Supported → the server transcript of the current session
  // becomes the source of truth: it replaces the localStorage-restored
  // prefix (keeping any messages that already landed live, e.g. the
  // greeting racing this probe).
  useEffect(() => {
    let cancelled = false
    ;(async () => {
      try {
        const userId = resolveUserId()
        const list = await listSessions(userId)
        if (cancelled) return
        if (list === null) {
          setSessionsSupported(false)
          return
        }
        setSessionsSupported(true)
        setSessions(list)
        const sessionId = localStorage.getItem('session_id')
        if (!sessionId) return
        try {
          const transcript = await fetchSessionTranscript(sessionId, userId)
          if (cancelled || transcript.length === 0) return
          const restored = transcriptToMessages(transcript)
          const previousRestoredCount = restoredCountRef.current
          restoredCountRef.current = restored.length
          setMessages((prev) => [...restored, ...prev.slice(previousRestoredCount)])
        } catch {
          // Session unknown to this server (or fetch failed) — keep the
          // localStorage transcript as the offline fallback.
        }
      } catch (error) {
        // Probe failed entirely (offline?) — sessionsSupported stays null
        // and the toggle stays hidden; nothing else degrades. Counted
        // (degrade-loudly): a hidden sidebar toggle from a broken probe is
        // otherwise indistinguishable from "old backend, hidden on purpose".
        diagnostics.increment('sessions.probe_failed', String(error))
      }
    })()
    return () => {
      cancelled = true
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  const startNewConversation = async () => {
    cancelDeferredTts()
    try {
      const sessionId = await createSession(resolveUserId())
      adoptSessionId(sessionId)
      restoredCountRef.current = 0
      setMessages([])
      clearStoredTranscript()
      void refreshSessions()
    } catch {
      setSessionsError('Could not start a new conversation.')
    }
  }

  const switchToSession = async (sessionId: string) => {
    if (sessionId === localStorage.getItem('session_id')) {
      setSidebarOpen(false)
      return
    }
    try {
      const transcript = await fetchSessionTranscript(sessionId, resolveUserId())
      cancelDeferredTts()
      kurisuBridge.lipSync?.stop?.()
      adoptSessionId(sessionId)
      const restored = transcriptToMessages(transcript)
      restoredCountRef.current = restored.length
      setMessages(restored)
      setSidebarOpen(false)
    } catch {
      setSessionsError('Could not load that conversation.')
    }
  }

  const removeSession = async (sessionId: string) => {
    try {
      await deleteSession(sessionId, resolveUserId())
      setSessions((prev) => prev.filter((session) => session.session_id !== sessionId))
      if (sessionId === localStorage.getItem('session_id')) {
        // Deleting the active conversation starts a fresh one.
        await startNewConversation()
      } else {
        void refreshSessions()
      }
    } catch {
      setSessionsError('Could not delete that conversation.')
    }
  }

  /** Client-side JSON export of one conversation's transcript. */
  const exportSession = async (sessionId: string) => {
    try {
      const transcript = await fetchSessionTranscript(sessionId, resolveUserId())
      const meta = sessions.find((session) => session.session_id === sessionId)
      const payload = {
        session_id: sessionId,
        title: meta?.title ?? null,
        exported_at: new Date().toISOString(),
        messages: transcript,
      }
      const blob = new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' })
      const url = URL.createObjectURL(blob)
      const anchor = document.createElement('a')
      anchor.href = url
      anchor.download = `amadeus-conversation-${sessionId}.json`
      document.body.appendChild(anchor)
      anchor.click()
      anchor.remove()
      URL.revokeObjectURL(url)
    } catch {
      setSessionsError('Could not export that conversation.')
    }
  }

  const toggleSidebar = () => {
    setSidebarOpen((prev) => {
      const next = !prev
      if (next) void refreshSessions()
      return next
    })
  }

  return {
    sessionsSupported,
    sidebarOpen,
    setSidebarOpen,
    sessions,
    sessionsLoading,
    sessionsError,
    currentSessionId,
    adoptSessionId,
    startNewConversation,
    switchToSession,
    removeSession,
    exportSession,
    toggleSidebar,
  }
}
