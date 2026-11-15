/**
 * Shared POST /api/memory helper, used by MemoryPanel and ChatInterface
 * (relationship gauge). Every call carries the current session/user id and
 * an action; text/record_id/pinned are only included when the caller
 * supplies them, so the request body stays byte-identical to before for
 * actions that don't need them (overview, graph, reset_relationship, ...).
 */
import { apiFetch } from './api'
import { resolveUserId } from './hooks/useSessions'

export interface MemoryRequestOptions {
  text?: string
  recordId?: string
  pinned?: boolean
}

export const memoryRequest = (action: string, opts?: MemoryRequestOptions) =>
  apiFetch('/api/memory', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      session_id: localStorage.getItem('session_id'),
      user_id: resolveUserId(),
      action,
      ...(opts?.text !== undefined ? { text: opts.text } : {}),
      ...(opts?.recordId !== undefined ? { record_id: opts.recordId } : {}),
      ...(opts?.pinned !== undefined ? { pinned: opts.pinned } : {}),
    }),
  })
