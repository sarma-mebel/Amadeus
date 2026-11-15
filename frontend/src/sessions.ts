/**
 * Client for the conversation-management endpoints:
 *
 *   GET    /api/sessions?user_id=X                     → list (updated_at desc)
 *   POST   /api/sessions/new {user_id}                 → {session_id}
 *   DELETE /api/sessions/{id}?user_id=X                → {ok: true}
 *   GET    /api/sessions/{id}/transcript?user_id=X     → {messages: [...]}
 *
 * Older backends don't have these routes at all — `listSessions` returns
 * null on a 404 so the caller can hide the sidebar entirely instead of
 * showing a broken drawer.
 */

import { apiFetch } from './api'

export interface SessionSummary {
  session_id: string
  title: string
  created_at: string
  updated_at: string
  message_count: number
}

export interface SessionTranscriptMessage {
  role: 'user' | 'assistant'
  content: string
  ja: string | null
  emotion: string | null
  timestamp: string
  turn_id?: string | null
}

const asString = (value: unknown, fallback = ''): string =>
  typeof value === 'string' ? value : fallback

/** Lists the user's conversations, newest-updated first.
 * Returns null when the backend doesn't support sessions (404 on the
 * route) — callers should hide the conversations UI in that case.
 * Throws on other failures (network, 5xx). */
export const listSessions = async (userId: string): Promise<SessionSummary[] | null> => {
  const response = await apiFetch(`/api/sessions?user_id=${encodeURIComponent(userId)}`)
  if (response.status === 404) return null
  if (!response.ok) throw new Error(`Failed to list sessions: HTTP ${response.status}`)
  const data = await response.json()
  const raw = Array.isArray(data?.sessions) ? data.sessions : []
  return raw
    .filter(
      (entry: unknown): entry is Record<string, unknown> =>
        Boolean(entry) && typeof entry === 'object' && typeof (entry as Record<string, unknown>).session_id === 'string',
    )
    .map(
      (entry: Record<string, unknown>): SessionSummary => ({
        session_id: entry.session_id as string,
        title: asString(entry.title, 'Untitled conversation'),
        created_at: asString(entry.created_at),
        updated_at: asString(entry.updated_at),
        message_count: typeof entry.message_count === 'number' ? entry.message_count : 0,
      }),
    )
}

/** Creates a fresh conversation and returns its session_id. */
export const createSession = async (userId: string): Promise<string> => {
  const response = await apiFetch('/api/sessions/new', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ user_id: userId }),
  })
  if (!response.ok) throw new Error(`Failed to create session: HTTP ${response.status}`)
  const data = await response.json()
  if (typeof data?.session_id !== 'string' || !data.session_id) {
    throw new Error('Create session response missing session_id')
  }
  return data.session_id
}

/** Deletes a conversation. */
export const deleteSession = async (sessionId: string, userId: string): Promise<void> => {
  const response = await apiFetch(
    `/api/sessions/${encodeURIComponent(sessionId)}?user_id=${encodeURIComponent(userId)}`,
    { method: 'DELETE' },
  )
  if (!response.ok) throw new Error(`Failed to delete session: HTTP ${response.status}`)
}

/** Fetches a conversation's transcript, chronological. Defensive: skips
 * malformed entries and unknown roles rather than throwing on them. */
export const fetchSessionTranscript = async (
  sessionId: string,
  userId: string,
): Promise<SessionTranscriptMessage[]> => {
  const response = await apiFetch(
    `/api/sessions/${encodeURIComponent(sessionId)}/transcript?user_id=${encodeURIComponent(userId)}`,
  )
  if (!response.ok) throw new Error(`Failed to fetch transcript: HTTP ${response.status}`)
  const data = await response.json()
  const raw = Array.isArray(data?.messages) ? data.messages : []
  return raw
    .filter((entry: unknown): entry is Record<string, unknown> => {
      if (!entry || typeof entry !== 'object') return false
      const record = entry as Record<string, unknown>
      return (
        (record.role === 'user' || record.role === 'assistant') && typeof record.content === 'string'
      )
    })
    .map(
      (entry: Record<string, unknown>): SessionTranscriptMessage => ({
        role: entry.role as 'user' | 'assistant',
        content: entry.content as string,
        ja: typeof entry.ja === 'string' && entry.ja ? entry.ja : null,
        emotion: typeof entry.emotion === 'string' && entry.emotion ? entry.emotion : null,
        timestamp: asString(entry.timestamp),
        turn_id: typeof entry.turn_id === 'string' && entry.turn_id ? entry.turn_id : null,
      }),
    )
}

/** Short relative-time label for the sidebar ("just now", "5m ago", "3d
 * ago", then a plain date). Neutral on unparseable timestamps. */
export const formatRelativeTime = (iso: string, now: number = Date.now()): string => {
  const parsed = Date.parse(iso)
  if (Number.isNaN(parsed)) return ''
  const deltaMs = now - parsed
  if (deltaMs < 60_000) return 'just now'
  const minutes = Math.floor(deltaMs / 60_000)
  if (minutes < 60) return `${minutes}m ago`
  const hours = Math.floor(minutes / 60)
  if (hours < 24) return `${hours}h ago`
  const days = Math.floor(hours / 24)
  if (days < 7) return `${days}d ago`
  return new Date(parsed).toLocaleDateString()
}
