import { describe, expect, it } from 'vitest'
import { frameDiff, PageTurnDetector } from './readTogether'

const flat = (value: number, length = 16): Uint8ClampedArray =>
  new Uint8ClampedArray(new Array(length).fill(value))

describe('frameDiff', () => {
  it('is 0 for identical frames', () => {
    expect(frameDiff(flat(120), flat(120))).toBe(0)
  })

  it('is 1 for maximally different frames (black vs white)', () => {
    expect(frameDiff(flat(0), flat(255))).toBeCloseTo(1, 5)
  })

  it('is proportional for a partial change', () => {
    // Half the pixels flip from 0 to 255: mean abs diff = 255/2, normalized ~0.5.
    const a = new Uint8ClampedArray([0, 0, 0, 0])
    const b = new Uint8ClampedArray([255, 255, 0, 0])
    expect(frameDiff(a, b)).toBeCloseTo(0.5, 5)
  })

  it('handles unequal-length arrays by comparing the overlap', () => {
    const a = new Uint8ClampedArray([0, 0, 0])
    const b = new Uint8ClampedArray([0, 0])
    expect(frameDiff(a, b)).toBe(0)
  })

  it('returns 0 for two empty frames', () => {
    expect(frameDiff(new Uint8ClampedArray(0), new Uint8ClampedArray(0))).toBe(0)
  })
})

describe('PageTurnDetector', () => {
  const LOW = flat(100)
  const HIGH = flat(220) // diff vs LOW = 120/255 ≈ 0.47, well above threshold
  const HIGH2 = flat(225) // close to HIGH — "settled" relative to HIGH

  it('does not fire on the first observed frame (nothing to compare against)', () => {
    const detector = new PageTurnDetector()
    expect(detector.observe(LOW, 0)).toBe(false)
  })

  it('does not fire while the page is unchanged', () => {
    const detector = new PageTurnDetector()
    detector.observe(LOW, 0)
    expect(detector.observe(LOW, 2_500)).toBe(false)
    expect(detector.observe(LOW, 5_000)).toBe(false)
  })

  it('fires once a change is followed by a stable frame (change -> settle)', () => {
    const detector = new PageTurnDetector()
    detector.observe(LOW, 0) // baseline
    expect(detector.observe(HIGH, 2_500)).toBe(false) // candidate change, not fired yet
    expect(detector.observe(HIGH2, 10_000)).toBe(true) // settled -> fire
  })

  it('does not fire on mid-scroll churn: repeated changes without settling never fire', () => {
    const detector = new PageTurnDetector()
    detector.observe(LOW, 0)
    // Keep alternating far-apart values every tick — never two consecutive
    // stable probes, so this must never fire despite lots of "motion".
    let now = 0
    let toggle = true
    for (let i = 0; i < 10; i++) {
      now += 2_500
      const fired = detector.observe(toggle ? flat(30) : flat(240), now)
      expect(fired).toBe(false)
      toggle = !toggle
    }
  })

  it('tracks a slow multi-tick transition and fires only once it truly settles', () => {
    const detector = new PageTurnDetector()
    detector.observe(flat(100), 0)
    // Big jump — candidate.
    expect(detector.observe(flat(200), 2_500)).toBe(false)
    // Still moving relative to the last candidate (200 -> 150 is a big
    // swing) — must not fire yet, and must keep tracking as 'changed'.
    expect(detector.observe(flat(150), 5_000)).toBe(false)
    // Now settles relative to the latest candidate (150 -> 148).
    expect(detector.observe(flat(148), 7_500)).toBe(true)
  })

  it('enforces the minimum gap between two fires', () => {
    const detector = new PageTurnDetector(0.1, 5_000)
    detector.observe(LOW, 0)
    detector.observe(HIGH, 1_000)
    expect(detector.observe(HIGH2, 2_000)).toBe(true) // first fire at t=2000

    // Immediately turn another "page" well within the gap — should settle
    // but the gap guard suppresses the fire.
    expect(detector.observe(LOW, 2_100)).toBe(false) // candidate again
    expect(detector.observe(LOW, 3_000)).toBe(false) // settled, but gap not met (3000-2000<5000)
  })

  it('fires again once the minimum gap has elapsed', () => {
    const detector = new PageTurnDetector(0.1, 5_000)
    detector.observe(LOW, 0)
    detector.observe(HIGH, 1_000)
    expect(detector.observe(HIGH2, 2_000)).toBe(true) // first fire at t=2000

    detector.observe(LOW, 8_000) // candidate, gap satisfied by the time it settles
    expect(detector.observe(LOW, 9_000)).toBe(true) // 9000-2000=7000 >= 5000
  })

  it('small jitter below threshold never enters the changed state', () => {
    const detector = new PageTurnDetector()
    detector.observe(flat(100), 0)
    expect(detector.observe(flat(105), 2_500)).toBe(false)
    expect(detector.observe(flat(98), 5_000)).toBe(false)
    expect(detector.observe(flat(102), 7_500)).toBe(false)
  })

  it('respects a custom diff threshold', () => {
    const detector = new PageTurnDetector(0.5) // very insensitive
    detector.observe(LOW, 0) // 100
    // HIGH (220) has diff ~0.47 which is below 0.5 -> should NOT even be
    // treated as a candidate change.
    expect(detector.observe(HIGH, 2_500)).toBe(false)
    expect(detector.observe(HIGH, 5_000)).toBe(false)
  })
})
