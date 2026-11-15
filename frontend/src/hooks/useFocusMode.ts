/**
 * Focus / co-working mode: "I'm going heads-down for N minutes." Server
 * state (amadeus/focus_mode.py) is per-session, so this hook is a thin
 * client over POST /api/focus (start/end/status) rather than a localStorage
 * contract like DND — the backend is the source of truth for whether a
 * session is active, because presence suppression (idle chatter, ambient,
 * the reach-out hold) all check the SAME session-scoped state server-side.
 *
 * Delivery of the one-shot check-in line rides the SAME presence voice path
 * as reminders/reach-out (`applyLine`): every poll response's `checkin`
 * field is treated exactly like a reminder's `reminder` field. A poll that
 * observes natural expiry (nobody called `end()`) delivers the check-in
 * exactly once, same as an explicit end.
 */
import { useCallback, useEffect, useRef, useState } from 'react'
import { apiFetch } from '../api'
import { diagnostics } from '../diagnostics'
import { presenceRequest } from '../presenceApi'
import type { PresencePayload } from './usePetPresence'

export interface FocusState {
  active: boolean
  until: string | null
  remainingSeconds: number | null
}

const IDLE_STATE: FocusState = { active: false, until: null, remainingSeconds: null }

interface FocusResponsePayload extends PresencePayload {
  session_id?: string
  active?: boolean
  until?: string | null
  remaining_seconds?: number | null
  checkin?: string | null
}

const toState = (data: FocusResponsePayload): FocusState => ({
  active: Boolean(data.active),
  until: data.until ?? null,
  remainingSeconds:
    typeof data.remaining_seconds === 'number' ? data.remaining_seconds : null,
})

export interface UseFocusModeOptions {
  /** Deliver the check-in line the same way a reminder/reach-out line is
   * delivered (expression + audio + caption). */
  applyLine: (data: PresencePayload, text: string) => void | Promise<void>
  /** Poll interval for catching a natural (un-ended) expiry — matches the
   * reminders/reach-out cadence so a check-in never waits long after the
   * timer actually runs out. */
  pollMs?: number
}

export interface UseFocusModeHandle extends FocusState {
  start: (minutes: number) => Promise<void>
  end: () => Promise<void>
}

const adoptSession = (data: { session_id?: string }) => {
  if (!data.session_id) return
  try {
    localStorage.setItem('session_id', data.session_id)
  } catch {
    // Session id just won't persist.
  }
}

export const useFocusMode = ({
  applyLine,
  pollMs = 30_000,
}: UseFocusModeOptions): UseFocusModeHandle => {
  const [state, setState] = useState<FocusState>(IDLE_STATE)
  const applyLineRef = useRef(applyLine)
  applyLineRef.current = applyLine
  // Mirrors the server's active/until so the poll loop can skip the fetch
  // entirely once we already know nothing is running — same "don't burn a
  // request for nothing" instinct as requestLine's DND gate, just without
  // an exactly-once pop to protect (status polls are idempotent when idle).
  const activeRef = useRef(false)
  activeRef.current = state.active

  const call = useCallback(
    async (action: 'start' | 'end' | 'status', minutes?: number) => {
      try {
        const response = await apiFetch(
          '/api/focus',
          presenceRequest({ action, ...(minutes ? { minutes } : {}) }),
        )
        if (!response.ok) {
          if (response.status !== 404) {
            diagnostics.increment('focus.request_failed', `${action} HTTP ${response.status}`)
          }
          return
        }
        const data = (await response.json()) as FocusResponsePayload
        adoptSession(data)
        setState(toState(data))
        if (data.checkin) {
          await applyLineRef.current(data, data.checkin)
        }
      } catch (error) {
        diagnostics.increment('focus.request_failed', `${action} ${String(error)}`)
      }
    },
    [],
  )

  const start = useCallback((minutes: number) => call('start', minutes), [call])
  const end = useCallback(() => call('end'), [call])

  // Poll for natural expiry: only while a session is (or was last known to
  // be) active — an idle focus mode costs nothing extra on top of the
  // existing reminders/reach-out 30s tick elsewhere in the presence loop.
  useEffect(() => {
    const tick = () => {
      if (activeRef.current) void call('status')
    }
    const intervalId = window.setInterval(tick, pollMs)
    return () => window.clearInterval(intervalId)
  }, [call, pollMs])

  return { ...state, start, end }
}
