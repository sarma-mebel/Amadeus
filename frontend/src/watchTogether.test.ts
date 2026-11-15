import { describe, expect, it } from 'vitest'
import { shouldPostFrame, WATCH_TOGETHER_DIFF_THRESHOLD } from './watchTogether'

const flat = (value: number, length = 16): Uint8ClampedArray =>
  new Uint8ClampedArray(new Array(length).fill(value))

describe('shouldPostFrame', () => {
  it('always posts the first frame (nothing to compare against yet)', () => {
    expect(shouldPostFrame(flat(100), null)).toBe(true)
  })

  it('skips a static frame (diff below threshold vs the last posted probe)', () => {
    const lastPosted = flat(100)
    // Tiny jitter only — well under the 0.03 default threshold.
    expect(shouldPostFrame(flat(101), lastPosted)).toBe(false)
    expect(shouldPostFrame(flat(99), lastPosted)).toBe(false)
  })

  it('posts a changed frame (diff at/above threshold vs the last posted probe)', () => {
    const lastPosted = flat(100)
    // 100 -> 130 is diff 30/255 ≈ 0.118, comfortably above 0.03.
    expect(shouldPostFrame(flat(130), lastPosted)).toBe(true)
  })

  it('is a strict >= comparison at the threshold boundary', () => {
    // Construct a diff exactly at the default threshold: 0.03 * 255 = 7.65,
    // so a uniform delta of 8 sits just above it and one of 7 just below.
    const lastPosted = flat(100)
    expect(shouldPostFrame(flat(108), lastPosted, WATCH_TOGETHER_DIFF_THRESHOLD)).toBe(true)
    expect(shouldPostFrame(flat(107), lastPosted, WATCH_TOGETHER_DIFF_THRESHOLD)).toBe(false)
  })

  it('respects a custom diff threshold', () => {
    const lastPosted = flat(100)
    // diff vs 130 is ~0.118 — below a very insensitive 0.5 threshold.
    expect(shouldPostFrame(flat(130), lastPosted, 0.5)).toBe(false)
    // Still true when way above even a strict threshold.
    expect(shouldPostFrame(flat(255), lastPosted, 0.5)).toBe(true)
  })

  it('never posts twice for the same unchanged probe compared to itself', () => {
    const probe = flat(150)
    expect(shouldPostFrame(probe, probe)).toBe(false)
  })
})
