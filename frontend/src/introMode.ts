/**
 * Conversational-onboarding chip bookkeeping ("SYNAPSE FORMED · <key>").
 *
 * Intro mode celebrates the first few facts Kurisu learns about a brand new
 * user with small chips near the header instead of making them dig into
 * Her memories. Once done, `markIntroDone` latches it off forever for this
 * browser (localStorage), same pattern as onboarding.ts.
 */

const INTRO_DONE_BASE_KEY = 'kurisu_intro_done'

/** Namespaces the intro-done latch by user id, same rationale as
 * milestones.ts's seenStorageKey: a fresh profile (or a relationship reset)
 * shouldn't inherit another user's "already onboarded" state. Falls back to
 * the bare key when no userId is supplied. */
const introDoneKey = (userId?: string): string =>
  userId ? `${INTRO_DONE_BASE_KEY}:${userId}` : INTRO_DONE_BASE_KEY

export const isIntroDone = (userId?: string): boolean => {
  if (typeof window === 'undefined') return true
  try {
    return window.localStorage.getItem(introDoneKey(userId)) === 'true'
  } catch {
    return true
  }
}

export const markIntroDone = (userId?: string): void => {
  if (typeof window === 'undefined') return
  try {
    window.localStorage.setItem(introDoneKey(userId), 'true')
  } catch {
    // Storage unavailable — intro mode may run again next session, which is
    // harmless (just a few extra chips).
  }
}

/** Clears this user's intro-done latch so a relationship reset re-arms
 * conversational onboarding. Best-effort. */
export const resetIntroDone = (userId?: string): void => {
  if (typeof window === 'undefined') return
  try {
    window.localStorage.removeItem(introDoneKey(userId))
  } catch {
    // Storage unavailable — nothing to clear.
  }
}

/** Keys present in `next` but not `prev` — new profile facts learned since
 * the last check, in `next`'s iteration order. */
export const diffNewFactKeys = (
  prev: Record<string, string>,
  next: Record<string, string>,
): string[] => Object.keys(next).filter((key) => !(key in prev))
