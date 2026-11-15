import { describe, expect, it } from 'vitest'
import { createBargeInDetector } from './vad'

// (The hands-free utterance endpointer and its tests moved to
// src/audio/handsFree.ts / handsFree.test.ts.)

// Detector samples arrive every ~50-85ms (MicMonitor poll / capture buffer).
const MONITOR_DT = 50

const feedDetectorFor = (
  detector: ReturnType<typeof createBargeInDetector>,
  rms: number,
  totalMs: number,
) => {
  let last: ReturnType<typeof detector.feed> = 'monitoring'
  for (let elapsed = 0; elapsed < totalMs; elapsed += MONITOR_DT) {
    last = detector.feed(rms, MONITOR_DT)
  }
  return last
}

describe('createBargeInDetector', () => {
  it('ignores everything during the warm-up window', () => {
    const detector = createBargeInDetector({ warmupMs: 500, minVoicedMs: 100 })
    // Loud residual echo right at her speech onset must not trigger.
    expect(feedDetectorFor(detector, 0.2, 500)).toBe('monitoring')
  })

  it('triggers after sustained voiced audio past warm-up', () => {
    const detector = createBargeInDetector({
      warmupMs: 500,
      minVoicedMs: 250,
      threshold: 0.045,
    })
    feedDetectorFor(detector, 0.001, 500) // warm-up
    expect(feedDetectorFor(detector, 0.1, 200)).toBe('monitoring')
    expect(feedDetectorFor(detector, 0.1, 100)).toBe('triggered')
  })

  it('resets the voiced run on a dip below threshold', () => {
    const detector = createBargeInDetector({
      warmupMs: 0,
      minVoicedMs: 250,
      threshold: 0.045,
    })
    feedDetectorFor(detector, 0.1, 200) // almost there…
    detector.feed(0.01, MONITOR_DT) // …dip resets the run
    expect(feedDetectorFor(detector, 0.1, 200)).toBe('monitoring')
    expect(feedDetectorFor(detector, 0.1, 100)).toBe('triggered')
  })

  it('never triggers on residual echo below the strict threshold', () => {
    const detector = createBargeInDetector()
    // Echo leakage at the ENDPOINTER's threshold (0.015-0.02) stays under
    // the barge-in threshold (0.045) indefinitely.
    expect(feedDetectorFor(detector, 0.02, 10000)).toBe('monitoring')
  })

  it('holds the triggered state on further feeds', () => {
    const detector = createBargeInDetector({ warmupMs: 0, minVoicedMs: 50 })
    feedDetectorFor(detector, 0.1, 100)
    expect(detector.feed(0.0, MONITOR_DT)).toBe('triggered')
  })
})
