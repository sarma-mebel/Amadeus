/**
 * Latest mic RMS level, published by whichever capture path is currently
 * active (hands-free endpointer or push-to-talk) and read by
 * VoiceStatusIndicator's rAF-driven waveform bars. Deliberately NOT React
 * state and NOT an event emitter — the indicator polls it once per animation
 * frame, so there's no need to re-render on every audio buffer (they arrive
 * much faster than the UI needs to redraw).
 */

import { feedListeningLevel, resetListeningReactions } from './listeningReactions'

let level = 0
let updatedAt = 0

/** Readings older than this read back as silence — guards against a stale
 * level lingering on screen if a capture path stops publishing without
 * calling resetMicLevel (e.g. an unexpected teardown). */
const STALE_MS = 250

const clamp01 = (value: number): number => Math.max(0, Math.min(1, value))

/** Called by capture paths (useHandsFree, useMicRecorder) once per audio
 * buffer with the buffer's RMS level (0..1, may exceed 1 on hot input). */
export const publishMicLevel = (rms: number): void => {
  level = clamp01(rms)
  updatedAt = performance.now()
  // Active-listening reactions (gaze/nods/backchannels) key off the same
  // signal: a publish here means the user's capture mic is live. The
  // barge-in monitor never publishes, so she can't react to her own voice.
  feedListeningLevel(level)
}

/** Read by VoiceStatusIndicator's rAF loop. Returns 0 if the last publish is
 * older than STALE_MS, so the waveform settles rather than freezing on a
 * hot value from a capture that already stopped. */
export const readMicLevel = (): number => {
  if (performance.now() - updatedAt > STALE_MS) return 0
  return level
}

/** Called by capture paths on cancel/stop so the waveform drops to silence
 * immediately instead of waiting out STALE_MS. */
export const resetMicLevel = (): void => {
  level = 0
  updatedAt = 0
  resetListeningReactions()
}
