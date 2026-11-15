import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { publishMicLevel, readMicLevel, resetMicLevel } from './levelBus'

describe('levelBus', () => {
  beforeEach(() => {
    resetMicLevel()
  })

  afterEach(() => {
    vi.restoreAllMocks()
    resetMicLevel()
  })

  it('reads back a published level immediately', () => {
    publishMicLevel(0.42)
    expect(readMicLevel()).toBeCloseTo(0.42)
  })

  it('clamps values above 1 down to 1', () => {
    publishMicLevel(3.7)
    expect(readMicLevel()).toBe(1)
  })

  it('clamps negative values up to 0', () => {
    publishMicLevel(-0.5)
    expect(readMicLevel()).toBe(0)
  })

  it('reads back 0 once the last publish is stale', () => {
    let now = 1000
    vi.spyOn(performance, 'now').mockImplementation(() => now)

    publishMicLevel(0.8)
    expect(readMicLevel()).toBeCloseTo(0.8)

    now += 249
    expect(readMicLevel()).toBeCloseTo(0.8)

    now += 2 // crosses the 250ms staleness window
    expect(readMicLevel()).toBe(0)
  })

  it('resets to 0 and stays 0 until republished', () => {
    publishMicLevel(0.9)
    expect(readMicLevel()).toBeCloseTo(0.9)

    resetMicLevel()
    expect(readMicLevel()).toBe(0)
  })
})
