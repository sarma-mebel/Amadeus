/**
 * Pure per-frame procedural-motion math for the render loop, extracted from
 * VRMViewer's animate() closure: easing curves, the nod-impulse envelope
 * shape, and the smoothed-random-walk torso drift. These take/return plain
 * numbers or small state objects and touch no THREE.Object3D/VRM/DOM state —
 * VRMViewer still owns applying the results to bones (proceduralAdditives,
 * quaternions) since that needs the live VRM humanoid + the frame's
 * additive-undo bookkeeping.
 */

import * as THREE from 'three'

export const easeInQuad = (t: number): number => t * t
export const easeOutQuad = (t: number): number => 1 - (1 - t) * (1 - t)

// One-shot damped nod impulse (item 1 rework, speech micro-motion): rises
// fast (ease-out) to the peak at 40% of the duration, then relaxes back to
// zero (ease-in-out) — reads as a single critically-damped nod rather than a
// spring that overshoots.
export const NOD_DURATION_S = 0.42

export const nodImpulseShape = (t: number): number => {
  const peakAt = 0.4
  if (t <= peakAt) {
    return easeOutQuad(t / peakAt)
  }
  const tail = (t - peakAt) / (1 - peakAt)
  return 1 - easeInQuad(tail)
}

/** Mutable state for driftNoise's smoothed random walk (see below). */
export interface DriftState {
  pitch: number
  yaw: number
  targetPitch: number
  targetYaw: number
  elapsed: number
  retargetInterval: number
}

export const createDriftState = (): DriftState => ({
  pitch: 0,
  yaw: 0,
  targetPitch: 0,
  targetYaw: 0,
  elapsed: 0,
  retargetInterval: 2.5 + Math.random() * 2.5,
})

/** Mutable state for the procedural weight shift (lifelike pass
 * 2026-07-30): a standing body slowly settles its weight onto one hip,
 * holds, and re-settles — 10-30s per stance, occasionally back to center.
 * Pure numbers; proceduralBody applies the roll to hips with a counter-
 * rotated spine/chest so the shoulders stay near level (contrapposto). */
export interface WeightShiftState {
  roll: number
  targetRoll: number
  elapsed: number
  retargetInterval: number
}

export const WEIGHT_SHIFT_MAX_RAD = THREE.MathUtils.degToRad(1.7)

export const createWeightShiftState = (): WeightShiftState => ({
  roll: 0,
  targetRoll: 0,
  elapsed: 0,
  retargetInterval: 10 + Math.random() * 20,
})

/**
 * Advance the weight-shift walk: every retargetInterval (10-30s) pick a new
 * stance — usually the OTHER hip (real standing sway alternates), sometimes
 * square/centered — and ease toward it over ~3s. Mutates `state` in place
 * and returns the current hip roll in radians.
 */
export const advanceWeightShift = (state: WeightShiftState, deltaTime: number): number => {
  state.elapsed += deltaTime
  if (state.elapsed >= state.retargetInterval) {
    state.elapsed = 0
    state.retargetInterval = 10 + Math.random() * 20
    if (Math.random() < 0.25) {
      state.targetRoll = 0 // square stance for a while
    } else {
      // Alternate hips: lean away from the current side (or pick one at
      // random from center).
      const side = state.targetRoll !== 0 ? -Math.sign(state.targetRoll) : Math.random() < 0.5 ? -1 : 1
      state.targetRoll = side * (0.4 + Math.random() * 0.6) * WEIGHT_SHIFT_MAX_RAD
    }
  }
  // ~3s settle: weight transfer is a deliberate, slow motion — anything
  // faster reads as a twitch, not a stance change.
  const ease = Math.min(1, deltaTime * 0.35)
  state.roll += (state.targetRoll - state.roll) * ease
  return state.roll
}

/**
 * Item 2 (breathing rework): slow noise-based torso/weight drift, NOT
 * sinusoidal. A simple smoothed random walk — every `retargetInterval`
 * seconds pick a new random target pitch/yaw offset and ease toward it, so
 * the motion has no fixed period at all (unlike a sine, which is trivially
 * periodic no matter how it's combined). Amplitude is tiny at rest and only
 * a little larger while speaking, per the "very low magnitude" ask. Mutates
 * `state` in place and returns the current {pitch, yaw} for convenience.
 */
export const driftNoise = (
  state: DriftState,
  deltaTime: number,
  speakingAmount: number
): { pitch: number; yaw: number } => {
  state.elapsed += deltaTime
  if (state.elapsed >= state.retargetInterval) {
    state.elapsed = 0
    state.retargetInterval = 2.5 + Math.random() * 2.5
    const maxAmp = THREE.MathUtils.degToRad(0.35 + speakingAmount * 0.45)
    state.targetPitch = (Math.random() * 2 - 1) * maxAmp
    state.targetYaw = (Math.random() * 2 - 1) * maxAmp * 0.7
  }
  // Slow ease toward the current random target — this is what keeps the
  // walk "smoothed" rather than a visible step function.
  const ease = Math.min(1, deltaTime * 0.6)
  state.pitch += (state.targetPitch - state.pitch) * ease
  state.yaw += (state.targetYaw - state.yaw) * ease
  return { pitch: state.pitch, yaw: state.yaw }
}
