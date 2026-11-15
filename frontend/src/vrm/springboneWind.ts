/**
 * Ambient springbone wind, extracted from VRMViewer's render loop: a gentle
 * pseudo-random breeze applied as a per-frame bias on every spring joint's
 * gravity vector, so hair (and any springbone-driven clothing) keeps
 * drifting subtly even when the body is on the single still idle loop. Wind
 * = layered sines (no two frequencies harmonically related, so the pattern
 * never visibly repeats) with a slowly wandering horizontal direction. The
 * joint's authored gravity settings are captured once and restored relative
 * to that base every frame — never compounded.
 *
 * `applySpringboneWind` still needs the live VRM's springBoneManager (a
 * three-vrm runtime object, not something worth faking in a pure function),
 * so it stays a small imperative function rather than returning a pure
 * value — but the wind-phase/strength bookkeeping is isolated here so it's
 * unit-testable without constructing a VRM.
 */

import * as THREE from 'three'
import type { VRM } from '@pixiv/three-vrm'

/** `localStorage.kurisu_wind` = '0' disables, any other number scales the
 * strength (clamped 0-4). Missing/invalid -> default strength 1. */
export const readWindStrengthPref = (): number => {
  if (typeof window === 'undefined') return 1
  const raw = window.localStorage.getItem('kurisu_wind')
  if (raw === null) return 1
  const parsed = Number.parseFloat(raw)
  return Number.isFinite(parsed) ? Math.max(0, Math.min(4, parsed)) : 1
}

/** Computes this frame's wind amplitude ("gust") and horizontal direction
 * (X/Z bias) from the running wind phase. Pure — no VRM/DOM access. */
export const computeWindSample = (
  windPhase: number,
  windStrengthPref: number
): { strength: number; windX: number; windZ: number } => {
  // Amplitude envelope: calm baseline with irregular soft gusts.
  const gust =
    0.5 +
    0.5 * Math.sin(windPhase * 0.31) * Math.sin(windPhase * 0.117 + 1.7) +
    0.25 * Math.sin(windPhase * 0.73 + 0.4)
  const strength = 0.012 * Math.max(0, gust) * windStrengthPref
  // Direction: mostly horizontal, wandering slowly around the model.
  const directionAngle = Math.sin(windPhase * 0.043) * Math.PI * 0.9
  const windX = Math.cos(directionAngle) * strength
  const windZ = Math.sin(directionAngle) * strength * 0.6
  return { strength, windX, windZ }
}

export interface SpringboneWindState {
  windPhase: number
  windStrengthPref: number
  windJointBase: WeakMap<object, { dir: THREE.Vector3; power: number }>
  windScratchVec: THREE.Vector3
}

export const createSpringboneWindState = (): SpringboneWindState => ({
  windPhase: Math.random() * 100,
  windStrengthPref: readWindStrengthPref(),
  windJointBase: new WeakMap(),
  windScratchVec: new THREE.Vector3(),
})

/** Advances the wind phase and applies this frame's bias to every spring
 * joint's gravity vector, restoring each joint's authored base gravity
 * relative to which the bias is computed (never compounded frame-to-frame).
 * No-op if wind is disabled (`windStrengthPref === 0`) or the VRM has no
 * springBoneManager yet. */
export const applySpringboneWind = (state: SpringboneWindState, vrm: VRM | null, deltaTime: number): void => {
  if (state.windStrengthPref === 0) return
  const springBoneManager = vrm?.springBoneManager
  if (!springBoneManager) return
  state.windPhase += deltaTime
  const { windX, windZ } = computeWindSample(state.windPhase, state.windStrengthPref)
  for (const joint of springBoneManager.joints) {
    const settings = joint.settings
    let base = state.windJointBase.get(joint)
    if (!base) {
      base = { dir: settings.gravityDir.clone(), power: settings.gravityPower }
      state.windJointBase.set(joint, base)
    }
    state.windScratchVec.set(
      base.dir.x * base.power + windX,
      base.dir.y * base.power,
      base.dir.z * base.power + windZ
    )
    const combinedPower = state.windScratchVec.length()
    settings.gravityPower = combinedPower
    if (combinedPower > 1e-6) {
      settings.gravityDir.copy(state.windScratchVec.divideScalar(combinedPower))
    }
  }
}
