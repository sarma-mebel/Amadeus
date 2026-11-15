/**
 * Shared core for the two co-activity modes: "read together" (page-turn
 * detector -> /api/read_along) and "watch together" (slow-cadence sampler ->
 * /api/watch_together). The two hooks were extracted from usePetChat.ts as
 * verbatim near-twins — same toggle/handle-ref/server-session-ref shape, same
 * capture gate, same presence posting and sign-off pattern — differing only
 * in the capture engine, the endpoint, the diagnostics keys, and
 * read-together's auto-attach frame stash. This hook owns the shared shape;
 * useReadTogether/useWatchTogether stay the public API as thin adapters.
 *
 * Spec notes that apply to BOTH modes (moved here with the code):
 * - Opt-in per activation, NOT persisted. The server owns speak-cooldowns and
 *   a per-session capture cap — this hook only gates WHEN a capture is even
 *   sent (busy/speaking/DND/courtesy/dozing skip it outright; nothing
 *   re-queues — missing an occasional capture is fine by spec).
 * - The server session only exists after the first successful capture POST;
 *   only then is the stop-time `{ end: true }` farewell worth making. The
 *   sessionActiveRef guard means toggle-on/toggle-off with zero captures
 *   never fires a spurious end request, and it resets immediately so a rapid
 *   re-toggle can't double-send `end`.
 * - The end-comment respects mute explicitly ("if one comes back and she
 *   isn't muted") rather than silently captioning a muted session's closing
 *   remark — applyPresenceLine already no-ops audio when muted, but the
 *   caption would still show.
 * - Everything network is best-effort: failures count a diagnostics counter
 *   (degrade loudly) and the server's own session bookkeeping times out
 *   server-side regardless.
 */
import { useCallback, useEffect, useRef, useState } from 'react'
import { apiFetch } from '../api'
import { isAudioEnabled } from '../audioPrefs'
import { diagnostics } from '../diagnostics'
import { presenceRequest } from '../presenceApi'
import { kurisuBridge } from '../vrm/kurisuBridge'
import { isPetDndActive, type PresencePayload } from './usePetPresence'

export interface SharedActivityImage {
  mime: string
  data: string
}

export interface SharedActivityHandlers {
  onCapture: (image: SharedActivityImage) => void
  onEnded: () => void
}

export interface SharedActivityConfig {
  /** Server endpoint taking `{ image }` posts and a final `{ end: true }`. */
  endpoint: string
  /** Start the capture engine (page-turn detector / slow sampler). */
  start: (handlers: SharedActivityHandlers) => Promise<{ stop: () => void }>
  /** Degrade-loudly counter keys, distinct per mode. */
  counters: { startFailed: string; postFailed: string; endFailed: string }
  /** console.warn prefixes, distinct per mode for log greppability. */
  warn: { post: string; start: string }
  /** Called after the gate passes, before the POST fires (read-together
   * stashes the full-res frame for auto-attach here). */
  onCaptureSending?: (image: SharedActivityImage) => void
  /** Called when an ACTIVE server session is being closed — deliberately
   * inside the sessionActiveRef guard, matching the original inline code
   * (read-together clears its auto-attach ref here). */
  onSessionClosing?: () => void
}

export interface UseSharedActivityDeps {
  isBusyRef: React.RefObject<boolean>
  isDozingRef: React.RefObject<boolean>
  isCourtesyActiveRef: React.RefObject<boolean>
  applyPresenceLine: (data: PresencePayload, text: string) => void | Promise<void>
}

export interface UseSharedActivityResult {
  active: boolean
  toggle: () => void
  /** Timestamp of the last capture actually POSTed (not just detected) —
   * PetView pulses the mode's toggle button off changes to this, same
   * pattern as lastPeekAt. */
  lastCaptureAt: number | null
}

export const useSharedActivity = (
  deps: UseSharedActivityDeps,
  config: SharedActivityConfig,
): UseSharedActivityResult => {
  const { isBusyRef, isDozingRef, isCourtesyActiveRef, applyPresenceLine } = deps

  // Wrappers build the config object per render; read it through a ref so
  // the stable useCallback/effect closures below always see the latest
  // fields without depending on the object's identity.
  const configRef = useRef(config)
  configRef.current = config

  const [active, setActive] = useState(false)
  const handleRef = useRef<{ stop: () => void } | null>(null)
  // Whether a server session actually exists yet (the first successful
  // capture POST returns a session_id). Not persisted to localStorage:
  // unlike the main chat session_id, the co-activity session lives only for
  // this activation and is identified by the same (session_id, user_id)
  // presence identity already on every request — this flag is just "a
  // session exists to close".
  const sessionActiveRef = useRef(false)
  const [lastCaptureAt, setLastCaptureAt] = useState<number | null>(null)

  const stopEngine = useCallback(() => {
    handleRef.current?.stop()
    handleRef.current = null
  }, [])

  // Fire-and-forget session-end notice: best-effort, matches every other
  // presence call's error handling in usePetChat.
  const endSession = useCallback(() => {
    const cfg = configRef.current
    if (!sessionActiveRef.current) return
    sessionActiveRef.current = false
    cfg.onSessionClosing?.()
    void (async () => {
      try {
        const response = await apiFetch(cfg.endpoint, presenceRequest({ end: true }))
        if (!response.ok) {
          diagnostics.increment(cfg.counters.endFailed, `HTTP ${response.status}`)
          return
        }
        const data = await response.json()
        if (data.comment && isAudioEnabled()) {
          await applyPresenceLine(data, data.comment)
        }
      } catch (error) {
        diagnostics.increment(cfg.counters.endFailed, String(error))
      }
    })()
  }, [applyPresenceLine])

  const toggle = useCallback(() => {
    setActive((prev) => {
      const next = !prev
      if (!next) {
        stopEngine()
        endSession()
      }
      return next
    })
  }, [stopEngine, endSession])

  useEffect(() => {
    if (!active) return
    const cfg = configRef.current
    let cancelled = false
    void (async () => {
      try {
        const handle = await cfg.start({
          onCapture: (image) => {
            if (cancelled) return
            // Same "don't waste quota / don't step on an active turn" gate
            // as the screen-companion peek: busy, actively speaking, DND,
            // courtesy, or dozing all skip this capture outright.
            if (
              isBusyRef.current ||
              kurisuBridge.lipSync?.isSpeaking?.() === true ||
              isPetDndActive() ||
              isCourtesyActiveRef.current ||
              isDozingRef.current
            ) {
              return
            }
            cfg.onCaptureSending?.(image)
            setLastCaptureAt(Date.now())
            void (async () => {
              try {
                const response = await apiFetch(
                  cfg.endpoint,
                  presenceRequest({ image: { mime: image.mime, data: image.data } }),
                )
                if (!response.ok) {
                  diagnostics.increment(cfg.counters.postFailed, `HTTP ${response.status}`)
                  return
                }
                const data = await response.json()
                sessionActiveRef.current = true
                if (data.comment) {
                  await applyPresenceLine(data, data.comment)
                }
              } catch (error) {
                // Best-effort — one missed capture just means one fewer note.
                console.warn(cfg.warn.post, error)
                diagnostics.increment(cfg.counters.postFailed, String(error))
              }
            })()
          },
          onEnded: () => {
            // Browser path only: the user ended sharing from the browser's
            // own "Stop sharing" chrome rather than our toggle. Flip the UI
            // state back off so it doesn't claim a dead stream — this itself
            // triggers the cleanup effect below.
            if (!cancelled) setActive(false)
          },
        })
        if (cancelled) {
          handle.stop()
        } else {
          handleRef.current = handle
        }
      } catch (error) {
        // getDisplayMedia rejected (user dismissed the picker) or Tauri's
        // capture_screenshot isn't available — fail quietly back to off
        // rather than leaving the toggle stuck "on" with no engine behind it.
        console.warn(cfg.warn.start, error)
        diagnostics.increment(cfg.counters.startFailed, String(error))
        if (!cancelled) setActive(false)
      }
    })()
    return () => {
      cancelled = true
      stopEngine()
    }
    // applyPresenceLine/stopEngine are stable (useCallback); config reads go
    // through configRef — re-running only on the active flip itself is
    // intended.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [active])

  // Stop on unmount too (pet window closing mid-session) — the toggle-off
  // path only runs on an explicit user toggle, and window teardown doesn't
  // go through setActive(false).
  useEffect(() => {
    return () => {
      stopEngine()
      endSession()
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  return { active, toggle, lastCaptureAt }
}
