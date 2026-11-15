/**
 * "Read together" mode: the client detects page turns in a book/manga/
 * document on screen via cheap local frame-diffing (readTogether.ts) and
 * posts one screenshot per settled page to /api/read_along; she keeps quiet
 * notes and occasionally reacts aloud.
 *
 * Thin adapter over useSharedActivity.ts, which owns the shared
 * toggle/gate/session/sign-off shape this mode has in common with
 * useWatchTogether.ts — see that file's header for the spec notes. What is
 * read-together-specific lives here: the page-turn engine, the endpoint,
 * the diagnostics keys, and the auto-attach frame stash.
 */
import { useRef } from 'react'
import { type ChatImageAttachment } from '../chat/payload'
import { startReadTogether } from '../readTogether'
import {
  useSharedActivity,
  type UseSharedActivityDeps,
} from './useSharedActivity'

export type UseReadTogetherDeps = UseSharedActivityDeps

export interface UseReadTogetherResult {
  readTogether: boolean
  toggleReadTogether: () => void
  /** Timestamp of the last page actually POSTed (not just detected) —
   * PetView pulses the 📖 button off changes to this, same pattern as
   * lastPeekAt/lastWatchFrameAt. */
  lastPageAt: number | null
  /** Most recent FULL-RES page frame, kept only in memory for auto-attach
   * (sendVoiceTurn reads this ref and threads it through runVoiceTurn's
   * `images` param via chat/petTurn.ts's resolveTurnImages). A ref, not
   * state — the read must be synchronous inside the send path without
   * forcing a re-render subscription. Never written to
   * localStorage/state-that-persists; cleared on stop. */
  lastPageFrameRef: React.RefObject<ChatImageAttachment | null>
}

export const useReadTogether = (deps: UseReadTogetherDeps): UseReadTogetherResult => {
  const lastPageFrameRef = useRef<ChatImageAttachment | null>(null)

  const { active, toggle, lastCaptureAt } = useSharedActivity(deps, {
    endpoint: '/api/read_along',
    start: (handlers) =>
      startReadTogether({ onPage: handlers.onCapture, onEnded: handlers.onEnded }),
    counters: {
      startFailed: 'pet.read_together_start_failed',
      postFailed: 'pet.read_along_page_failed',
      endFailed: 'pet.read_along_end_failed',
    },
    warn: {
      post: '[pet] read-along page failed:',
      start: '[pet] read-together failed to start:',
    },
    onCaptureSending: (image) => {
      lastPageFrameRef.current = { mime: image.mime, data: image.data }
    },
    onSessionClosing: () => {
      lastPageFrameRef.current = null
    },
  })

  return {
    readTogether: active,
    toggleReadTogether: toggle,
    lastPageAt: lastCaptureAt,
    lastPageFrameRef,
  }
}
