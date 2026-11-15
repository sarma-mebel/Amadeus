/** Runs the wake-word listener whenever it should be running: toggle on
 * (localStorage `kurisu_wake_word`, live via WAKE_WORD_EVENT), no call in
 * progress, mic usable. A detection stops the listener and hands off to
 * `onWake` (ChatInterface starts the normal duplex call); when the call
 * ends, the dependency change restarts the listener.
 *
 * Backoff contract (mirrors the duplex capability latch): a server-side
 * "no" (disabled / no model) or a mic failure disables the listener until
 * the user toggles the setting again — no reconnect loops against a server
 * that already said no. A live socket dying (`closed`, e.g. backend
 * restart) schedules one quiet retry after RETRY_DELAY_MS instead.
 */
import { useEffect, useRef, useState } from 'react'
import {
  WAKE_WORD_EVENT,
  loadWakeWordEnabled,
  startWakeListener,
  type WakeUnavailableKind,
} from '../audio/wakeWord'

const RETRY_DELAY_MS = 15_000

export interface UseWakeWordOptions {
  /** True while any call (duplex or classic) is active — the wake listener
   * must release the mic pipeline while a call owns it. */
  callActive: boolean
  canUseMic: boolean
  onWake: () => void
  /** Surfaced once per hard failure so the caller can toast a hint. */
  onUnavailable?: (kind: WakeUnavailableKind, message?: string) => void
}

export const useWakeWord = ({
  callActive,
  canUseMic,
  onWake,
  onUnavailable,
}: UseWakeWordOptions): { wakeWordEnabled: boolean } => {
  const [enabled, setEnabled] = useState(loadWakeWordEnabled())
  // Bumped to re-run the listener effect after a retryable failure.
  const [retryTick, setRetryTick] = useState(0)
  const hardFailedRef = useRef(false)
  const onWakeRef = useRef(onWake)
  onWakeRef.current = onWake
  const onUnavailableRef = useRef(onUnavailable)
  onUnavailableRef.current = onUnavailable

  useEffect(() => {
    const onToggle = () => {
      // Re-toggling is the user's "try again" — clear the hard-failure latch.
      hardFailedRef.current = false
      setEnabled(loadWakeWordEnabled())
      setRetryTick((tick) => tick + 1)
    }
    window.addEventListener(WAKE_WORD_EVENT, onToggle)
    return () => window.removeEventListener(WAKE_WORD_EVENT, onToggle)
  }, [])

  useEffect(() => {
    if (!enabled || callActive || !canUseMic || hardFailedRef.current) return
    let retryTimer: number | null = null
    const handle = startWakeListener({
      onWake: () => {
        handle.stop()
        onWakeRef.current()
      },
      onUnavailable: (kind, message) => {
        if (kind === 'closed') {
          retryTimer = window.setTimeout(() => setRetryTick((tick) => tick + 1), RETRY_DELAY_MS)
          return
        }
        hardFailedRef.current = true
        onUnavailableRef.current?.(kind, message)
      },
    })
    return () => {
      if (retryTimer !== null) window.clearTimeout(retryTimer)
      handle.stop()
    }
  }, [enabled, callActive, canUseMic, retryTick])

  return { wakeWordEnabled: enabled }
}
