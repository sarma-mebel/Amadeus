/**
 * "Watch together" mode: the client samples the screen on a slow timer
 * (watchTogether.ts, which also de-dupes near-static samples client-side via
 * shouldPostFrame) and posts frames worth sending to /api/watch_together;
 * she keeps quiet notes and occasionally reacts aloud.
 *
 * Thin adapter over useSharedActivity.ts, which owns the shared
 * toggle/gate/session/sign-off shape this mode has in common with
 * useReadTogether.ts — see that file's header for the spec notes. What is
 * watch-together-specific lives here: the sampler engine, the endpoint, and
 * the diagnostics keys.
 */
import { startWatchTogether } from '../watchTogether'
import {
  useSharedActivity,
  type UseSharedActivityDeps,
} from './useSharedActivity'

export type UseWatchTogetherDeps = UseSharedActivityDeps

export interface UseWatchTogetherResult {
  watchTogether: boolean
  toggleWatchTogether: () => void
  /** Timestamp of the last frame actually POSTed (not just sampled) —
   * PetView can pulse its toggle button off changes to this, same pattern
   * as lastPageAt/lastPeekAt. */
  lastWatchFrameAt: number | null
}

export const useWatchTogether = (deps: UseWatchTogetherDeps): UseWatchTogetherResult => {
  const { active, toggle, lastCaptureAt } = useSharedActivity(deps, {
    endpoint: '/api/watch_together',
    start: (handlers) =>
      startWatchTogether({ onFrame: handlers.onCapture, onEnded: handlers.onEnded }),
    counters: {
      startFailed: 'pet.watch_together_start_failed',
      postFailed: 'pet.watch_together_frame_failed',
      endFailed: 'pet.watch_together_end_failed',
    },
    warn: {
      post: '[pet] watch-together frame failed:',
      start: '[pet] watch-together failed to start:',
    },
  })

  return {
    watchTogether: active,
    toggleWatchTogether: toggle,
    lastWatchFrameAt: lastCaptureAt,
  }
}
