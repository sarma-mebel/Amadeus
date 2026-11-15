/** Auto barge-in voice-activity detector.
 *
 * (The hands-free utterance endpointer that used to live beside it moved to
 * src/audio/handsFree.ts when hands-free became a continuous open-mic mode —
 * this module now only owns the interrupt detector.)
 *
 * Runs on the echo-cancelled mic levels while Kurisu speaks (fed from the
 * hands-free capture stream, or a standalone MicMonitor). Much stricter than
 * the endpointer: a higher threshold (~3x), a minimum continuous voiced
 * duration, and a warm-up ignore window (the first moments after her speech
 * starts carry a residual-echo burst before the AEC converges) — so her own
 * voice leaking through cheap laptop speakers doesn't self-trigger an
 * interruption. */

export type BargeInState = 'monitoring' | 'triggered'

export interface BargeInOptions {
  threshold?: number
  minVoicedMs?: number
  warmupMs?: number
}

export interface BargeInDetector {
  /** Feed one RMS sample covering dtMs; 'triggered' is terminal. */
  feed: (rms: number, dtMs: number) => BargeInState
}

export const BARGE_IN_THRESHOLD = 0.045
export const BARGE_IN_MIN_VOICED_MS = 250
export const BARGE_IN_WARMUP_MS = 500

export const createBargeInDetector = ({
  threshold = BARGE_IN_THRESHOLD,
  minVoicedMs = BARGE_IN_MIN_VOICED_MS,
  warmupMs = BARGE_IN_WARMUP_MS,
}: BargeInOptions = {}): BargeInDetector => {
  let warmedMs = 0
  let voicedMs = 0
  let triggered = false

  const feed = (rms: number, dtMs: number): BargeInState => {
    if (triggered) return 'triggered'
    if (warmedMs < warmupMs) {
      warmedMs += dtMs
      return 'monitoring'
    }
    if (rms >= threshold) {
      voicedMs += dtMs
      if (voicedMs >= minVoicedMs) {
        triggered = true
        return 'triggered'
      }
    } else {
      voicedMs = 0
    }
    return 'monitoring'
  }

  return { feed }
}
