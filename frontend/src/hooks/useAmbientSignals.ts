/**
 * Ambient-awareness loop (Tauri only): polls the Rust `ambient_signals`
 * snapshot every ~15s, runs it through the pure transition tracker
 * (../ambient.ts), and hands noteworthy events to the caller's `deliver`
 * (which posts /api/ambient and voices any reply through that surface's
 * own presence path). Mounted by BOTH usePresence (main window) and
 * usePetPresence (pet); the pet owns the feature while it's alive —
 * `enabled` gates the main window off — and the server cooldown backstops
 * any remaining overlap.
 *
 * Degrade loudly: repeated invoke failures disarm the loop for this mount
 * (older shell without the command) and count in diagnostics — never a
 * silent forever-dead poll, never a crash.
 */
import { useEffect, useRef } from 'react'
import { apiFetch } from '../api'
import { createAmbientTracker, type AmbientKind, type AmbientSnapshot } from '../ambient'
import { diagnostics } from '../diagnostics'
import { isTauri } from '../platform'
import { loadRhythmEnabled } from '../rhythmPrefs'
import { createRhythmTracker, toWirePayload } from '../rhythm'
import { resolveUserId } from './useSessions'

export const AMBIENT_POLL_MS = 15_000
const MAX_CONSECUTIVE_FAILURES = 5
/** Rhythm is a rollup, not a transition: posting the running daily total
 * every 15s would be pure noise. Once every 10 minutes keeps the stored day
 * fresh enough for a greeting that fires at most weekly. */
export const RHYTHM_POST_MS = 10 * 60_000

export interface UseAmbientSignalsOptions {
  deliver: (kind: AmbientKind, detail: string) => Promise<void>
  /** Extra live gate evaluated per tick (e.g. "the pet isn't running"). */
  enabled?: () => boolean
}

export const useAmbientSignals = ({ deliver, enabled }: UseAmbientSignalsOptions): void => {
  const deliverRef = useRef(deliver)
  deliverRef.current = deliver
  const enabledRef = useRef(enabled)
  enabledRef.current = enabled

  useEffect(() => {
    if (!isTauri()) return
    const tracker = createAmbientTracker()
    // Rhythm folds into THIS poll rather than running its own: it reads the
    // same snapshot, and a second timer would double the invoke cost for no
    // additional information.
    const rhythm = createRhythmTracker()
    let cancelled = false
    let failures = 0
    let lastRhythmPost = 0
    let invokeFn: ((cmd: string) => Promise<unknown>) | null = null

    const tick = async () => {
      if (cancelled || failures >= MAX_CONSECUTIVE_FAILURES) return
      if (document.hidden) return
      if (enabledRef.current && !enabledRef.current()) return
      try {
        if (!invokeFn) {
          const mod = await import('@tauri-apps/api/core')
          invokeFn = mod.invoke as (cmd: string) => Promise<unknown>
        }
        const snapshot = (await invokeFn('ambient_signals')) as AmbientSnapshot
        failures = 0
        for (const event of tracker.ingest(snapshot, Date.now())) {
          // Instant client-side reflex tap (vrm/reflexes.ts) — fires
          // regardless of what the server delivery below does with it, and
          // a dispatch failure must never eat the delivery itself.
          try {
            window.dispatchEvent(
              new CustomEvent('kurisu:ambient', {
                detail: { kind: event.kind, detail: event.detail, severity: event.severity },
              }),
            )
          } catch (error) {
            diagnostics.increment('reflex.ambient_dispatch_failed', String(error))
          }
          void deliverRef.current(event.kind, event.detail)
        }
        // Rhythm: the per-browser opt-in is read LIVE each tick, so toggling
        // it off stops accumulation immediately rather than at next mount.
        // (The server flag is the other half of the double gate; a disabled
        // server simply answers recorded:false.)
        if (loadRhythmEnabled()) {
          const delta = rhythm.ingest(snapshot, Date.now())
          const now = Date.now()
          if (delta && now - lastRhythmPost >= RHYTHM_POST_MS) {
            lastRhythmPost = now
            void apiFetch('/api/rhythm', {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({ user_id: resolveUserId(), ...toWirePayload(delta) }),
            }).catch((error) => {
              diagnostics.increment('rhythm.post_failed', String(error))
            })
          }
        }
      } catch (error) {
        failures += 1
        diagnostics.increment('ambient.poll_failed', String(error))
        if (failures >= MAX_CONSECUTIVE_FAILURES) {
          diagnostics.increment('ambient.disarmed')
        }
      }
    }

    const timer = window.setInterval(() => void tick(), AMBIENT_POLL_MS)
    void tick()
    return () => {
      cancelled = true
      window.clearInterval(timer)
    }
  }, [])
}
