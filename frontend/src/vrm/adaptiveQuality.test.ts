import { describe, expect, it, vi } from 'vitest'
import { createAdaptiveQuality, QUALITY_SCALES } from './adaptiveQuality'

const noop = () => {}

describe('createAdaptiveQuality', () => {
  it('applyPixelRatio pushes the current tier scale (1x initially)', () => {
    const setPixelRatioScale = vi.fn()
    const quality = createAdaptiveQuality({ setPixelRatioScale, debugLog: noop })
    quality.applyPixelRatio()
    expect(setPixelRatioScale).toHaveBeenCalledWith(1)
  })

  it('steps resolution down after ~1.5s of sustained over-budget frames', () => {
    const setPixelRatioScale = vi.fn()
    const quality = createAdaptiveQuality({ setPixelRatioScale, debugLog: noop })

    // Start past the 3s cooldown window (lastChangeAt is 0) and feed
    // 50ms frames (> 26ms budget) until 1.5s of over-budget accumulates.
    let now = 4000
    quality.update(now) // primes lastFrameAt
    for (let i = 0; i < 40; i += 1) {
      now += 50
      quality.update(now)
    }
    expect(setPixelRatioScale).toHaveBeenCalledWith(QUALITY_SCALES[1])
  })

  it('does not downgrade before the 3s cooldown has elapsed', () => {
    const setPixelRatioScale = vi.fn()
    const quality = createAdaptiveQuality({ setPixelRatioScale, debugLog: noop })

    let now = 0
    quality.update(now)
    // 40 x 50ms = 2s of solid over-budget time, but all before t=3000.
    for (let i = 0; i < 40; i += 1) {
      now += 50
      quality.update(now)
    }
    expect(setPixelRatioScale).not.toHaveBeenCalled()
  })

  it('recovers a tier after ~12s of solid under-budget frames', () => {
    const setPixelRatioScale = vi.fn()
    const quality = createAdaptiveQuality({ setPixelRatioScale, debugLog: noop })

    let now = 4000
    quality.update(now)
    for (let i = 0; i < 40; i += 1) {
      now += 50
      quality.update(now)
    }
    expect(setPixelRatioScale).toHaveBeenLastCalledWith(QUALITY_SCALES[1])
    setPixelRatioScale.mockClear()

    // 10ms frames (< 18ms) for > 12s, past the 3s cooldown.
    for (let i = 0; i < 1300; i += 1) {
      now += 10
      quality.update(now)
    }
    expect(setPixelRatioScale).toHaveBeenLastCalledWith(QUALITY_SCALES[0])
  })

  it('caps a single frame sample at 100ms so a tab-restore is not load', () => {
    const setPixelRatioScale = vi.fn()
    const quality = createAdaptiveQuality({ setPixelRatioScale, debugLog: noop })

    let now = 4000
    quality.update(now)
    // One 30s gap contributes at most 0.1s of over-budget time — far from
    // the 1.5s bar, so no downgrade.
    now += 30000
    quality.update(now)
    expect(setPixelRatioScale).not.toHaveBeenCalled()
  })

  it('resetFrameClock forgets the last timestamp (hidden time is not a slow frame)', () => {
    const setPixelRatioScale = vi.fn()
    const quality = createAdaptiveQuality({ setPixelRatioScale, debugLog: noop })

    let now = 4000
    quality.update(now)
    quality.resetFrameClock()
    // Next sample only primes the clock again; a huge wall-clock jump
    // after reset contributes nothing.
    now += 60000
    quality.update(now)
    expect(setPixelRatioScale).not.toHaveBeenCalled()
  })
})

describe('createAdaptiveQuality postfx hook', () => {
  it('disables postfx the moment the tier first drops to the disable threshold', () => {
    const setPixelRatioScale = vi.fn()
    const setPostFxEnabled = vi.fn()
    const quality = createAdaptiveQuality({ setPixelRatioScale, debugLog: noop, setPostFxEnabled })

    let now = 4000
    quality.update(now) // primes lastFrameAt
    for (let i = 0; i < 40; i += 1) {
      now += 50
      quality.update(now)
    }
    // Tier 0 -> 1 crossed POSTFX_DISABLE_TIER (1): postfx goes off exactly once.
    expect(setPostFxEnabled).toHaveBeenCalledTimes(1)
    expect(setPostFxEnabled).toHaveBeenLastCalledWith(false)
  })

  it('does not call the hook again for a second downgrade past the threshold', () => {
    const setPixelRatioScale = vi.fn()
    const setPostFxEnabled = vi.fn()
    const quality = createAdaptiveQuality({ setPixelRatioScale, debugLog: noop, setPostFxEnabled })

    let now = 4000
    quality.update(now)
    // Downgrade tier 0 -> 1 (crosses the threshold).
    for (let i = 0; i < 40; i += 1) {
      now += 50
      quality.update(now)
    }
    expect(setPostFxEnabled).toHaveBeenCalledTimes(1)
    setPostFxEnabled.mockClear()

    // Downgrade again, past cooldown: tier 1 -> 2. Already below the
    // threshold on both sides of this step, so the hook must not re-fire.
    now += 3100
    quality.update(now)
    for (let i = 0; i < 40; i += 1) {
      now += 50
      quality.update(now)
    }
    expect(setPostFxEnabled).not.toHaveBeenCalled()
  })

  it('re-enables postfx only once the tier fully recovers to 0', () => {
    const setPixelRatioScale = vi.fn()
    const setPostFxEnabled = vi.fn()
    const quality = createAdaptiveQuality({ setPixelRatioScale, debugLog: noop, setPostFxEnabled })

    let now = 4000
    quality.update(now)
    // Downgrade twice: tier 0 -> 1 -> 2.
    for (let i = 0; i < 40; i += 1) {
      now += 50
      quality.update(now)
    }
    now += 3100
    quality.update(now)
    for (let i = 0; i < 40; i += 1) {
      now += 50
      quality.update(now)
    }
    expect(setPostFxEnabled).toHaveBeenCalledTimes(1)
    expect(setPostFxEnabled).toHaveBeenLastCalledWith(false)
    setPostFxEnabled.mockClear()

    // Recover tier 2 -> 1: still >= POSTFX_DISABLE_TIER, hook must stay quiet.
    now += 3100
    for (let i = 0; i < 1300; i += 1) {
      now += 10
      quality.update(now)
    }
    expect(setPostFxEnabled).not.toHaveBeenCalled()

    // Recover tier 1 -> 0: crosses back below the threshold, hook fires true.
    now += 3100
    for (let i = 0; i < 1300; i += 1) {
      now += 10
      quality.update(now)
    }
    expect(setPostFxEnabled).toHaveBeenCalledTimes(1)
    expect(setPostFxEnabled).toHaveBeenLastCalledWith(true)
  })

  it('notifyIdle restoring tier 0 also re-enables postfx', () => {
    const setPixelRatioScale = vi.fn()
    const setPostFxEnabled = vi.fn()
    const quality = createAdaptiveQuality({ setPixelRatioScale, debugLog: noop, setPostFxEnabled })

    let now = 4000
    quality.update(now)
    for (let i = 0; i < 40; i += 1) {
      now += 50
      quality.update(now)
    }
    expect(setPostFxEnabled).toHaveBeenLastCalledWith(false)
    setPostFxEnabled.mockClear()

    quality.notifyIdle()
    expect(setPostFxEnabled).toHaveBeenCalledTimes(1)
    expect(setPostFxEnabled).toHaveBeenLastCalledWith(true)
  })

  it('is a no-op without the optional hook (existing callers unaffected)', () => {
    const setPixelRatioScale = vi.fn()
    const quality = createAdaptiveQuality({ setPixelRatioScale, debugLog: noop })

    let now = 4000
    quality.update(now)
    expect(() => {
      for (let i = 0; i < 40; i += 1) {
        now += 50
        quality.update(now)
      }
    }).not.toThrow()
  })
})
