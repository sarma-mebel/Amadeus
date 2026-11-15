import { describe, expect, it } from 'vitest'
import {
  advanceWeightShift,
  createDriftState,
  createWeightShiftState,
  driftNoise,
  easeInQuad,
  easeOutQuad,
  nodImpulseShape,
  WEIGHT_SHIFT_MAX_RAD,
} from './proceduralMotion'

describe('easeInQuad / easeOutQuad', () => {
  it('are identity at the endpoints', () => {
    expect(easeInQuad(0)).toBe(0)
    expect(easeInQuad(1)).toBe(1)
    expect(easeOutQuad(0)).toBe(0)
    expect(easeOutQuad(1)).toBe(1)
  })

  it('easeInQuad starts slow, easeOutQuad starts fast (mirror curves)', () => {
    expect(easeInQuad(0.5)).toBeCloseTo(0.25)
    expect(easeOutQuad(0.5)).toBeCloseTo(0.75)
    expect(easeInQuad(0.25)).toBeLessThan(easeOutQuad(0.25))
  })
})

describe('nodImpulseShape', () => {
  it('starts and ends at zero', () => {
    expect(nodImpulseShape(0)).toBeCloseTo(0)
    expect(nodImpulseShape(1)).toBeCloseTo(0)
  })

  it('peaks at t=0.4', () => {
    const peak = nodImpulseShape(0.4)
    expect(peak).toBeCloseTo(1)
    // Neighboring samples should be lower than the peak.
    expect(nodImpulseShape(0.3)).toBeLessThan(peak)
    expect(nodImpulseShape(0.5)).toBeLessThan(peak)
  })

  it('is continuous at the peak boundary (no seam)', () => {
    const justBefore = nodImpulseShape(0.4 - 1e-6)
    const justAfter = nodImpulseShape(0.4 + 1e-6)
    expect(justBefore).toBeCloseTo(justAfter, 4)
  })
})

describe('driftNoise', () => {
  it('starts at rest (pitch/yaw zero) for a freshly created state', () => {
    const state = createDriftState()
    expect(state.pitch).toBe(0)
    expect(state.yaw).toBe(0)
  })

  it('eases pitch/yaw toward a retargeted value over time rather than snapping', () => {
    const state = createDriftState()
    // Force an immediate retarget by exhausting the interval.
    state.elapsed = state.retargetInterval
    const first = driftNoise(state, 0.016, 0)
    // One small step shouldn't already be at the (nonzero, most likely)
    // target — it should have eased only partway there.
    if (state.targetPitch !== 0) {
      expect(Math.abs(first.pitch)).toBeLessThan(Math.abs(state.targetPitch))
    }
  })

  it('keeps returned pitch/yaw in sync with the mutated state object', () => {
    const state = createDriftState()
    const result = driftNoise(state, 0.05, 0.5)
    expect(result.pitch).toBe(state.pitch)
    expect(result.yaw).toBe(state.yaw)
  })

  it('larger speakingAmount allows a larger retarget amplitude bound', () => {
    // Not a statistical test — just checks the amplitude bound scales with
    // speakingAmount by sampling many retargets and taking the max.
    const sampleMaxAmplitude = (speakingAmount: number): number => {
      let maxAmp = 0
      for (let i = 0; i < 200; i += 1) {
        const state = createDriftState()
        state.elapsed = state.retargetInterval
        driftNoise(state, 0.016, speakingAmount)
        maxAmp = Math.max(maxAmp, Math.abs(state.targetPitch))
      }
      return maxAmp
    }
    const restAmp = sampleMaxAmplitude(0)
    const speakingAmp = sampleMaxAmplitude(1)
    expect(speakingAmp).toBeGreaterThan(restAmp)
  })
})

describe('advanceWeightShift', () => {
  it('starts square (zero roll) and stays bounded by the max lean', () => {
    const state = createWeightShiftState()
    expect(state.roll).toBe(0)
    // Run several minutes of simulated frames across many retargets.
    for (let i = 0; i < 20000; i += 1) {
      const roll = advanceWeightShift(state, 0.016)
      expect(Math.abs(roll)).toBeLessThanOrEqual(WEIGHT_SHIFT_MAX_RAD)
    }
  })

  it('eases toward the target instead of snapping on retarget', () => {
    const state = createWeightShiftState()
    state.elapsed = state.retargetInterval // force an immediate retarget
    advanceWeightShift(state, 0.016)
    if (state.targetRoll !== 0) {
      expect(Math.abs(state.roll)).toBeLessThan(Math.abs(state.targetRoll))
    }
  })

  it('a nonzero stance retargets to the opposite side (weight alternates)', () => {
    // Sample many retargets from a forced left lean: every non-center
    // retarget must flip to the right side.
    for (let i = 0; i < 100; i += 1) {
      const state = createWeightShiftState()
      state.targetRoll = WEIGHT_SHIFT_MAX_RAD // leaning "left"
      state.elapsed = state.retargetInterval
      advanceWeightShift(state, 0.016)
      expect(state.targetRoll).toBeLessThanOrEqual(0) // center or right, never further left
    }
  })

  it('converges onto the target over a few seconds of frames', () => {
    const state = createWeightShiftState()
    state.targetRoll = WEIGHT_SHIFT_MAX_RAD
    state.retargetInterval = 9999 // hold this target
    for (let i = 0; i < 2000; i += 1) {
      advanceWeightShift(state, 0.016) // ~32s ≈ 11 easing time-constants
    }
    expect(state.roll).toBeCloseTo(WEIGHT_SHIFT_MAX_RAD, 3)
  })
})
