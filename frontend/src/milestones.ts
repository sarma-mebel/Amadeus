/**
 * Milestone celebration bookkeeping for the relationship gauge.
 *
 * The greeting response (POST /api/greeting) may carry a `milestones` array
 * of events the server has decided are newly true this session (Nth
 * session reached, M-month anniversary, ...). This module maps an event to
 * a display label + celebratory animation clip, and de-dupes across app
 * restarts via localStorage so the same milestone never celebrates twice.
 */

export interface MilestoneEvent {
  kind: 'sessions' | 'anniversary'
  value?: number
  months?: number
  label: string
  date?: string
}

const SEEN_STORAGE_BASE_KEY = 'kurisu_milestones_seen_v1'
const MAX_SEEN = 50

/** Namespaces the seen-milestones key by user id so switching the local
 * "who's using this browser" identity (a fresh profile, a shared machine)
 * doesn't inherit — or poison — another user's seen-milestone history.
 * Falls back to the bare key when no userId is supplied, so existing
 * callers/tests that don't care about multi-user isolation keep working. */
const seenStorageKey = (userId?: string): string =>
  userId ? `${SEEN_STORAGE_BASE_KEY}:${userId}` : SEEN_STORAGE_BASE_KEY

/** Stable de-dupe key for an event — session count or anniversary
 * month/date, whichever the server populated. */
export const milestoneKey = (event: MilestoneEvent): string => {
  if (event.kind === 'sessions') return `sessions:${event.value ?? ''}`
  return `anniversary:${event.date ?? event.months ?? ''}`
}

const SESSION_CLIPS: Record<number, string> = {
  10: 'Joy',
  25: 'Cheering',
  50: 'Excitement 2',
  100: 'Pride',
  250: 'Joy 2',
}

const ANNIVERSARY_CLIPS: Record<number, string> = {
  1: 'Gratitude',
  6: 'Pride',
}

/** Animation clip to play for an event; falls back to a generic clip for
 * milestone values the map above doesn't special-case. */
export const milestoneClip = (event: MilestoneEvent): string => {
  if (event.kind === 'sessions') {
    return SESSION_CLIPS[event.value ?? -1] ?? 'Joy'
  }
  const months = event.months ?? 0
  if (months >= 12) return 'Joy 3'
  return ANNIVERSARY_CLIPS[months] ?? 'Gratitude'
}

const readSeen = (userId?: string): string[] => {
  try {
    const raw = localStorage.getItem(seenStorageKey(userId))
    if (!raw) return []
    const parsed = JSON.parse(raw)
    return Array.isArray(parsed) ? parsed.filter((item): item is string => typeof item === 'string') : []
  } catch {
    return []
  }
}

/** Events not yet marked seen, in the order given. */
export const filterUnseen = (events: MilestoneEvent[], userId?: string): MilestoneEvent[] => {
  try {
    const seen = new Set(readSeen(userId))
    return events.filter((event) => !seen.has(milestoneKey(event)))
  } catch {
    return events
  }
}

/** Marks events as seen (capped at MAX_SEEN most-recent keys). */
export const markSeen = (events: MilestoneEvent[], userId?: string): void => {
  if (events.length === 0) return
  try {
    const seen = readSeen(userId)
    const next = [...seen, ...events.map(milestoneKey)]
    const capped = next.slice(Math.max(0, next.length - MAX_SEEN))
    localStorage.setItem(seenStorageKey(userId), JSON.stringify(capped))
  } catch {
    // Storage unavailable — the milestone may celebrate again next time,
    // which is harmless.
  }
}

/** Clears this user's seen-milestones latch so a relationship reset re-arms
 * every celebration (matching a fresh profile's experience). Best-effort. */
export const clearSeenMilestones = (userId?: string): void => {
  try {
    localStorage.removeItem(seenStorageKey(userId))
  } catch {
    // Storage unavailable — nothing to clear.
  }
}
