// @vitest-environment jsdom
import { describe, expect, it, beforeEach, afterEach } from 'vitest'
import * as THREE from 'three'
import { computeWindSample, createSpringboneWindState, readWindStrengthPref } from './springboneWind'

describe('readWindStrengthPref', () => {
  beforeEach(() => {
    window.localStorage.clear()
  })
  afterEach(() => {
    window.localStorage.clear()
  })

  it('defaults to 1 when unset', () => {
    expect(readWindStrengthPref()).toBe(1)
  })

  it('returns 0 (disabled) when explicitly set to "0"', () => {
    window.localStorage.setItem('kurisu_wind', '0')
    expect(readWindStrengthPref()).toBe(0)
  })

  it('parses and clamps a numeric value into [0, 4]', () => {
    window.localStorage.setItem('kurisu_wind', '2.5')
    expect(readWindStrengthPref()).toBeCloseTo(2.5)
    window.localStorage.setItem('kurisu_wind', '100')
    expect(readWindStrengthPref()).toBe(4)
    window.localStorage.setItem('kurisu_wind', '-5')
    expect(readWindStrengthPref()).toBe(0)
  })

  it('falls back to 1 for garbage input', () => {
    window.localStorage.setItem('kurisu_wind', 'not-a-number')
    expect(readWindStrengthPref()).toBe(1)
  })
})

describe('computeWindSample', () => {
  it('produces zero wind when strength preference is 0', () => {
    const sample = computeWindSample(12.3, 0)
    expect(sample.strength).toBe(0)
    expect(sample.windX).toBe(0)
    expect(sample.windZ).toBe(0)
  })

  it('scales linearly with windStrengthPref at a fixed phase', () => {
    const base = computeWindSample(5, 1)
    const doubled = computeWindSample(5, 2)
    expect(doubled.strength).toBeCloseTo(base.strength * 2)
  })

  it('never returns a negative strength', () => {
    for (let phase = 0; phase < 50; phase += 0.7) {
      expect(computeWindSample(phase, 1).strength).toBeGreaterThanOrEqual(0)
    }
  })
})

describe('createSpringboneWindState', () => {
  it('reads the strength pref at creation time', () => {
    window.localStorage.setItem('kurisu_wind', '2')
    const state = createSpringboneWindState()
    expect(state.windStrengthPref).toBe(2)
    window.localStorage.clear()
  })

  it('starts with an empty joint-base WeakMap', () => {
    const state = createSpringboneWindState()
    const probe = {}
    expect(state.windJointBase.get(probe)).toBeUndefined()
  })

  it('gives each new state a THREE.Vector3 scratch vector', () => {
    const state = createSpringboneWindState()
    expect(state.windScratchVec).toBeInstanceOf(THREE.Vector3)
  })
})
