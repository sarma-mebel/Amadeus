/**
 * Shared request builder for the presence endpoints (/api/greeting,
 * /api/poke, /api/idle, /api/reminders/due, /api/reachout/due,
 * /api/farewell). Historically each call site rebuilt the JSON body by
 * hand, and they disagreed on identity: some sent the raw
 * `localStorage.getItem('user_id')` (null on a fresh install), others went
 * through `resolveUserId()` (which mints and persists an id). Presence and
 * chat could therefore disagree on who the user is. Every body built here
 * resolves the id the same way chat does.
 */
import { resolveUserId } from './hooks/useSessions'

export const presenceIds = (): { session_id: string | null; user_id: string } => ({
  session_id: localStorage.getItem('session_id'),
  user_id: resolveUserId(),
})

export const presenceRequest = (extra?: Record<string, unknown>): RequestInit => ({
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ ...presenceIds(), ...extra }),
})
