// @vitest-environment jsdom
import { describe, expect, it, vi } from 'vitest'
import { isCalmSubtleClipName, startPetIdleBehaviors } from './petIdleBehaviors'

describe('isCalmSubtleClipName', () => {
  it('matches Relax and Sleepy by name, case-insensitively', () => {
    expect(isCalmSubtleClipName('Relax')).toBe(true)
    expect(isCalmSubtleClipName('sleepy')).toBe(true)
    expect(isCalmSubtleClipName('SLEEPY')).toBe(true)
  })

  it('does not match unrelated subtle-pool clips', () => {
    expect(isCalmSubtleClipName('LookAround')).toBe(false)
    expect(isCalmSubtleClipName('Curiosity')).toBe(false)
    expect(isCalmSubtleClipName('Neck Stretching')).toBe(false)
  })
})

describe('startPetIdleBehaviors', () => {
  it('plays a subtle-pool clip when idle and the pool is non-empty', () => {
    vi.useFakeTimers()
    const playRandomSubtleGesture = vi.fn(() => true)
    const behaviors = startPetIdleBehaviors({
      playRandomSubtleGesture,
      hasSubtlePool: () => true,
      isBusyOrRecentlyInteracted: () => false,
      now: () => new Date(2026, 0, 1, 14, 0, 0).getTime(), // 2pm, daytime
      random: () => 0, // pins the delay to the minimum of its range
    })
    vi.advanceTimersByTime(2 * 60 * 1000 + 1)
    expect(playRandomSubtleGesture).toHaveBeenCalledTimes(1)
    behaviors.stop()
    vi.useRealTimers()
  })

  it('skips (but reschedules) while busy/recently interacted', () => {
    vi.useFakeTimers()
    const playRandomSubtleGesture = vi.fn(() => true)
    let busy = true
    const behaviors = startPetIdleBehaviors({
      playRandomSubtleGesture,
      hasSubtlePool: () => true,
      isBusyOrRecentlyInteracted: () => busy,
      now: () => new Date(2026, 0, 1, 14, 0, 0).getTime(),
      random: () => 0,
    })
    vi.advanceTimersByTime(2 * 60 * 1000 + 1)
    expect(playRandomSubtleGesture).not.toHaveBeenCalled()
    busy = false
    vi.advanceTimersByTime(2 * 60 * 1000 + 1)
    expect(playRandomSubtleGesture).toHaveBeenCalledTimes(1)
    behaviors.stop()
    vi.useRealTimers()
  })

  it('never plays once hasSubtlePool is false', () => {
    vi.useFakeTimers()
    const playRandomSubtleGesture = vi.fn(() => true)
    const behaviors = startPetIdleBehaviors({
      playRandomSubtleGesture,
      hasSubtlePool: () => false,
      isBusyOrRecentlyInteracted: () => false,
      now: () => new Date(2026, 0, 1, 14, 0, 0).getTime(),
      random: () => 0,
    })
    vi.advanceTimersByTime(10 * 60 * 1000)
    expect(playRandomSubtleGesture).not.toHaveBeenCalled()
    behaviors.stop()
    vi.useRealTimers()
  })

  it('passes a calm-name preference filter at night, and none in the day', () => {
    vi.useFakeTimers()
    const playRandomSubtleGesture = vi.fn(() => true)
    const dayBehaviors = startPetIdleBehaviors({
      playRandomSubtleGesture,
      hasSubtlePool: () => true,
      isBusyOrRecentlyInteracted: () => false,
      now: () => new Date(2026, 0, 1, 14, 0, 0).getTime(), // 2pm
      random: () => 0,
    })
    vi.advanceTimersByTime(2 * 60 * 1000 + 1)
    expect(playRandomSubtleGesture).toHaveBeenLastCalledWith(1.2, undefined)
    dayBehaviors.stop()
    playRandomSubtleGesture.mockClear()

    const nightBehaviors = startPetIdleBehaviors({
      playRandomSubtleGesture,
      hasSubtlePool: () => true,
      isBusyOrRecentlyInteracted: () => false,
      now: () => new Date(2026, 0, 1, 23, 30, 0).getTime(), // 11:30pm
      random: () => 0,
    })
    vi.advanceTimersByTime(4 * 60 * 1000 + 1)
    expect(playRandomSubtleGesture).toHaveBeenCalledTimes(1)
    expect(playRandomSubtleGesture.mock.calls[0][1]).toBe(isCalmSubtleClipName)
    nightBehaviors.stop()
    vi.useRealTimers()
  })

  it('uses a longer delay range at night than during the day', () => {
    vi.useFakeTimers()
    const playRandomSubtleGesture = vi.fn(() => true)
    // random() = 1 pins to the MAX of whichever range is active.
    const dayBehaviors = startPetIdleBehaviors({
      playRandomSubtleGesture,
      hasSubtlePool: () => true,
      isBusyOrRecentlyInteracted: () => false,
      now: () => new Date(2026, 0, 1, 14, 0, 0).getTime(),
      random: () => 0.999999,
    })
    // Just under the day range's max (~5min, minus an epsilon from
    // random() < 1): still nothing fired.
    vi.advanceTimersByTime(5 * 60 * 1000 - 100)
    expect(playRandomSubtleGesture).not.toHaveBeenCalled()
    vi.advanceTimersByTime(200)
    expect(playRandomSubtleGesture).toHaveBeenCalledTimes(1)
    dayBehaviors.stop()

    playRandomSubtleGesture.mockClear()
    const nightBehaviors = startPetIdleBehaviors({
      playRandomSubtleGesture,
      hasSubtlePool: () => true,
      isBusyOrRecentlyInteracted: () => false,
      now: () => new Date(2026, 0, 1, 2, 0, 0).getTime(), // 2am
      random: () => 0.999999,
    })
    // Well past the day range's max but still short of the night range's
    // max (~9min): nothing should have fired yet at this point either.
    vi.advanceTimersByTime(6 * 60 * 1000)
    expect(playRandomSubtleGesture).not.toHaveBeenCalled()
    vi.advanceTimersByTime(3 * 60 * 1000 + 100)
    expect(playRandomSubtleGesture).toHaveBeenCalledTimes(1)
    nightBehaviors.stop()
    vi.useRealTimers()
  })

  it('stop() prevents any further firing', () => {
    vi.useFakeTimers()
    const playRandomSubtleGesture = vi.fn(() => true)
    const behaviors = startPetIdleBehaviors({
      playRandomSubtleGesture,
      hasSubtlePool: () => true,
      isBusyOrRecentlyInteracted: () => false,
      now: () => new Date(2026, 0, 1, 14, 0, 0).getTime(),
      random: () => 0,
    })
    behaviors.stop()
    vi.advanceTimersByTime(10 * 60 * 1000)
    expect(playRandomSubtleGesture).not.toHaveBeenCalled()
    vi.useRealTimers()
  })
})
