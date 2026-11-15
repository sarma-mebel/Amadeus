// @vitest-environment jsdom
import * as THREE from 'three'
import { describe, expect, it } from 'vitest'
import {
  computeHeadAimDelta,
  readHeadTrackPref,
  smoothingAlpha,
  worldDeltaToLocalAdditive,
} from './headAim'

const FORWARD = new THREE.Vector3(0, 0, 1)

const applyToForward = (
  delta: THREE.Quaternion,
  headQuat: THREE.Quaternion,
): THREE.Vector3 =>
  FORWARD.clone().applyQuaternion(headQuat).applyQuaternion(delta).normalize()

describe('computeHeadAimDelta', () => {
  it('turns the forward axis toward the target at weight 1', () => {
    const headQuat = new THREE.Quaternion() // facing +Z
    const headPos = new THREE.Vector3(0, 1.4, 0)
    const target = new THREE.Vector3(1, 1.4, 1) // 45° to the right
    const delta = computeHeadAimDelta(headQuat, headPos, target, FORWARD, {
      maxAngleRad: Math.PI,
      weight: 1,
    })
    const turned = applyToForward(delta, headQuat)
    const wanted = target.clone().sub(headPos).normalize()
    expect(turned.angleTo(wanted)).toBeLessThan(1e-6)
  })

  it('takes only the weighted fraction of the turn', () => {
    const headQuat = new THREE.Quaternion()
    const headPos = new THREE.Vector3()
    const target = new THREE.Vector3(1, 0, 0) // 90° turn
    const delta = computeHeadAimDelta(headQuat, headPos, target, FORWARD, {
      maxAngleRad: Math.PI,
      weight: 0.5,
    })
    const turned = applyToForward(delta, headQuat)
    // Half of 90° taken: 45° remaining to the target.
    expect(turned.angleTo(new THREE.Vector3(1, 0, 0))).toBeCloseTo(Math.PI / 4, 3)
  })

  it('caps the turn at maxAngleRad before weighting', () => {
    const headQuat = new THREE.Quaternion()
    const headPos = new THREE.Vector3()
    const target = new THREE.Vector3(0, 0, -1) // 180° behind
    const delta = computeHeadAimDelta(headQuat, headPos, target, FORWARD, {
      maxAngleRad: 0.5,
      weight: 1,
    })
    const turned = applyToForward(delta, headQuat)
    expect(turned.angleTo(FORWARD)).toBeLessThanOrEqual(0.5 + 1e-6)
  })

  it('is identity for a degenerate target', () => {
    const headPos = new THREE.Vector3(0, 1.4, 0)
    const delta = computeHeadAimDelta(
      new THREE.Quaternion(),
      headPos,
      headPos.clone(),
      FORWARD,
      { maxAngleRad: 0.5, weight: 1 },
    )
    expect(delta.angleTo(new THREE.Quaternion())).toBeLessThan(1e-6)
  })

  it('respects a non-default faceFront axis (VRM0-style -Z)', () => {
    const back = new THREE.Vector3(0, 0, -1)
    const headQuat = new THREE.Quaternion()
    const target = new THREE.Vector3(0, 0, -5) // straight ahead for -Z face
    const delta = computeHeadAimDelta(
      headQuat,
      new THREE.Vector3(),
      target,
      back,
      { maxAngleRad: 0.5, weight: 1 },
    )
    // Already facing the target: no turn.
    expect(delta.angleTo(new THREE.Quaternion())).toBeLessThan(1e-6)
  })
})

describe('worldDeltaToLocalAdditive', () => {
  it('right-multiplying the local additive reproduces the world delta', () => {
    const boneWorld = new THREE.Quaternion().setFromEuler(
      new THREE.Euler(0.3, -0.8, 0.1),
    )
    const delta = new THREE.Quaternion().setFromAxisAngle(
      new THREE.Vector3(0, 1, 0),
      0.4,
    )
    const local = worldDeltaToLocalAdditive(delta, boneWorld)
    const result = boneWorld.clone().multiply(local) // world * local
    const expected = delta.clone().multiply(boneWorld) // delta ∘ world
    expect(result.angleTo(expected)).toBeLessThan(1e-6)
  })
})

describe('smoothingAlpha', () => {
  it('is frame-rate independent (two half-steps ≈ one full step)', () => {
    const one = smoothingAlpha(0.032, 8)
    const half = smoothingAlpha(0.016, 8)
    const twoHalves = 1 - (1 - half) * (1 - half)
    expect(twoHalves).toBeCloseTo(one, 6)
  })
})

describe('readHeadTrackPref', () => {
  it('defaults to 1, honors 0 and scaling values, ignores junk', () => {
    window.localStorage.removeItem('kurisu_head_track')
    expect(readHeadTrackPref()).toBe(1)
    window.localStorage.setItem('kurisu_head_track', '0')
    expect(readHeadTrackPref()).toBe(0)
    window.localStorage.setItem('kurisu_head_track', '1.5')
    expect(readHeadTrackPref()).toBe(1.5)
    window.localStorage.setItem('kurisu_head_track', 'banana')
    expect(readHeadTrackPref()).toBe(1)
    window.localStorage.removeItem('kurisu_head_track')
  })
})
