import { useEffect, useRef, useState } from 'react'
import { t } from '../i18n'
import { formatRelativeTime, type SessionSummary } from '../sessions'
import EmptyState from './EmptyState'

/**
 * Collapsible conversation drawer (SG-0 terminal flavored). Pure
 * presentation: the parent (ChatInterface) owns the session list, the
 * current session id, and all backend calls; this component only renders
 * and forwards intents. Delete is a two-step affordance — first click arms
 * the row ("CONFIRM?"), second click actually deletes.
 *
 * Search filters titles instantly; from three characters it also greps
 * transcript contents via the parent-supplied fetcher (fetched lazily,
 * cached per session until its updated_at changes) and shows a snippet
 * on rows matched only by content.
 */

interface SessionSidebarProps {
  sessions: SessionSummary[]
  currentSessionId: string | null
  loading: boolean
  error: string | null
  onSelect: (sessionId: string) => void
  onNew: () => void
  onDelete: (sessionId: string) => void
  onExport: (sessionId: string) => void
  onClose: () => void
  /** Full transcript text of one session, for content search. Optional so
   * the drawer still works against backends without the transcript API. */
  fetchTranscriptText?: (sessionId: string) => Promise<string>
  /** Active character's display name (character packs can swap out
   * Kurisu) — same source ChatInterface already reads for TranscriptList's
   * assistantName. Falls back to 'Kurisu' when absent (older backends). */
  assistantName?: string
}

const CONTENT_SEARCH_MIN_CHARS = 3

const SessionSidebar = ({
  sessions,
  currentSessionId,
  loading,
  error,
  onSelect,
  onNew,
  onDelete,
  onExport,
  onClose,
  fetchTranscriptText,
  assistantName = 'Kurisu',
}: SessionSidebarProps) => {
  const [pendingDeleteId, setPendingDeleteId] = useState<string | null>(null)
  const [query, setQuery] = useState('')
  const [contentMatches, setContentMatches] = useState<Record<string, string>>({})
  const [searching, setSearching] = useState(false)
  // Keyed by `${id}:${updated_at}` so a session that gained messages
  // re-fetches instead of serving a stale transcript.
  const transcriptCache = useRef<Map<string, string>>(new Map())
  const searchSeq = useRef(0)

  const q = query.trim().toLowerCase()
  const contentSearchActive = q.length >= CONTENT_SEARCH_MIN_CHARS && !!fetchTranscriptText

  useEffect(() => {
    if (!contentSearchActive) {
      setContentMatches({})
      setSearching(false)
      return
    }
    const seq = ++searchSeq.current
    let cancelled = false
    setSearching(true)
    ;(async () => {
      const matches: Record<string, string> = {}
      for (const session of sessions) {
        if (cancelled || seq !== searchSeq.current) return
        const cacheKey = `${session.session_id}:${session.updated_at}`
        let text = transcriptCache.current.get(cacheKey)
        if (text === undefined) {
          try {
            text = await fetchTranscriptText!(session.session_id)
          } catch {
            text = ''
          }
          transcriptCache.current.set(cacheKey, text)
        }
        const index = text.toLowerCase().indexOf(q)
        if (index >= 0) {
          const start = Math.max(0, index - 24)
          matches[session.session_id] =
            (start > 0 ? '…' : '') + text.slice(start, index + q.length + 40).trim() + '…'
        }
      }
      if (!cancelled && seq === searchSeq.current) {
        setContentMatches(matches)
        setSearching(false)
      }
    })()
    return () => {
      cancelled = true
    }
  }, [q, contentSearchActive, sessions, fetchTranscriptText])

  const visibleSessions = q
    ? sessions.filter(
        (session) =>
          (session.title || 'Untitled conversation').toLowerCase().includes(q) ||
          session.session_id in contentMatches,
      )
    : sessions

  const handleDeleteClick = (sessionId: string) => {
    if (pendingDeleteId === sessionId) {
      setPendingDeleteId(null)
      onDelete(sessionId)
      return
    }
    setPendingDeleteId(sessionId)
  }

  return (
    <aside className="session-sidebar" aria-label="Conversations">
      <div className="session-sidebar-header">
        <span className="session-sidebar-title">CONVERSATION LOG</span>
        <button
          type="button"
          className="session-sidebar-close"
          onClick={onClose}
          aria-label="Close conversations"
        >
          ×
        </button>
      </div>
      <button type="button" className="session-new-button" onClick={onNew}>
        + NEW CONVERSATION
      </button>
      <input
        type="search"
        className="session-search"
        placeholder="Search conversations…"
        aria-label="Search conversations"
        value={query}
        onChange={(event) => setQuery(event.target.value)}
      />
      {loading && <div className="session-sidebar-status">Loading…</div>}
      {error && <div className="session-sidebar-status error">{error}</div>}
      {!loading && !error && sessions.length === 0 && (
        <EmptyState
          icon="sessions"
          title={t('No saved conversations yet.')}
          subLine={t('Your first one with {assistantName} will show up here.', { assistantName })}
          className="session-sidebar-empty"
        />
      )}
      {!loading && !error && sessions.length > 0 && q && visibleSessions.length === 0 && (
        <div className="session-sidebar-status">
          {searching ? 'Searching transcripts…' : 'No matches.'}
        </div>
      )}
      <ul className="session-list">
        {visibleSessions.map((session) => {
          const isCurrent = session.session_id === currentSessionId
          const isPendingDelete = pendingDeleteId === session.session_id
          const relative = formatRelativeTime(session.updated_at)
          const snippet =
            q && !(session.title || 'Untitled conversation').toLowerCase().includes(q)
              ? contentMatches[session.session_id]
              : undefined
          return (
            <li
              key={session.session_id}
              className={`session-item ${isCurrent ? 'current' : ''}`}
            >
              <button
                type="button"
                className="session-item-main"
                onClick={() => onSelect(session.session_id)}
                aria-current={isCurrent ? 'true' : undefined}
                title={session.title}
              >
                <span className="session-item-title">{session.title || 'Untitled conversation'}</span>
                <span className="session-item-meta">
                  {relative}
                  {session.message_count > 0 && ` · ${session.message_count} msg`}
                  {isCurrent && ' · ACTIVE'}
                </span>
                {snippet && <span className="session-item-snippet">{snippet}</span>}
              </button>
              <span className="session-item-actions">
                <button
                  type="button"
                  className="session-item-action"
                  onClick={() => onExport(session.session_id)}
                  aria-label={`Export conversation "${session.title}" as JSON`}
                  title="Export as JSON"
                >
                  ⇩
                </button>
                <button
                  type="button"
                  className={`session-item-action delete ${isPendingDelete ? 'armed' : ''}`}
                  onClick={() => handleDeleteClick(session.session_id)}
                  onBlur={() => {
                    if (isPendingDelete) setPendingDeleteId(null)
                  }}
                  aria-label={
                    isPendingDelete
                      ? `Confirm deleting conversation "${session.title}"`
                      : `Delete conversation "${session.title}"`
                  }
                  title={isPendingDelete ? 'Click again to confirm' : 'Delete'}
                >
                  {isPendingDelete ? 'CONFIRM?' : '✕'}
                </button>
              </span>
            </li>
          )
        })}
      </ul>
    </aside>
  )
}

export default SessionSidebar
