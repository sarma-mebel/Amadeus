/**
 * Per-browser opt-in for rhythm memory (amadeus/rhythm.py).
 *
 * This is HALF of a double gate: `rhythm.enabled` must also be true in the
 * server config. Both are required before a single byte is written, for the
 * same reason screen watching is gated twice — this is the only feature that
 * keeps a durable record of how the user spends time at their machine.
 *
 * Deliberately NOT wired into the presence presets (presenceLevel.ts). A
 * Quiet/Balanced/Lively choice must never switch on a durable behavioral
 * record, same rule that keeps screen capture out of them.
 *
 * Turning it OFF erases the server-side tally rather than merely halting
 * writes — see `setRhythmEnabled`. That is a promise made in the settings
 * copy and in docs/companion-ethics.md, so the delete call lives here rather
 * than in a component that might forget it.
 */

import { apiFetch } from './api'
import { diagnostics } from './diagnostics'
import { resolveUserId } from './hooks/useSessions'

export const RHYTHM_STORAGE_KEY = 'kurisu_rhythm'

export const loadRhythmEnabled = (): boolean => {
  try {
    return window.localStorage.getItem(RHYTHM_STORAGE_KEY) === '1'
  } catch {
    // Private-mode / storage-disabled browsers default to OFF. For a
    // consent gate, unreadable must mean "no consent recorded".
    return false
  }
}

/**
 * Persist the opt-in. Disabling also asks the server to erase the tally.
 *
 * The local flag flips even if the delete request fails, so a dead backend
 * can never trap the user in an enabled state; the failure is counted rather
 * than swallowed, and the store is erased on the next successful toggle.
 */
export const setRhythmEnabled = async (enabled: boolean): Promise<void> => {
  try {
    window.localStorage.setItem(RHYTHM_STORAGE_KEY, enabled ? '1' : '0')
  } catch (error) {
    diagnostics.increment('rhythm.pref_write_failed', String(error))
  }
  if (enabled) return
  try {
    const response = await apiFetch(
      `/api/rhythm?user_id=${encodeURIComponent(resolveUserId())}`,
      { method: 'DELETE' },
    )
    if (!response.ok) throw new Error(`HTTP ${response.status}`)
  } catch (error) {
    diagnostics.increment('rhythm.clear_failed', String(error))
  }
}

/** Erase the tally without changing the opt-in — the explicit "Clear"
 * button next to the disclosure panel. */
export const clearRhythm = async (): Promise<boolean> => {
  try {
    const response = await apiFetch(
      `/api/rhythm?user_id=${encodeURIComponent(resolveUserId())}`,
      { method: 'DELETE' },
    )
    if (!response.ok) throw new Error(`HTTP ${response.status}`)
    const data = await response.json()
    return Boolean(data?.cleared)
  } catch (error) {
    diagnostics.increment('rhythm.clear_failed', String(error))
    return false
  }
}
