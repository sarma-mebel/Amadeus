/**
 * Procedural head aim: turn the head (not just the eyes) toward the gaze
 * target, so she faces you instead of cutting her eyes sideways while the
 * idle animation leaves her head wherever it wants.
 *
 * Pure math half of the feature — VRMViewer owns reading the gaze proxy,
 * smoothing over time, and applying the result through its procedural
 * additive stack (head tracks are stripped from every clip, so the head is
 * genuinely free for this). Convention-safe: the face-forward axis comes
 * from the caller (three-vrm's `vrm.lookAt.faceFront`), never assumed.
 */
import * as THREE from 'three'

export interface HeadAimOptions {
  /** Hard cap on how far the head may turn from the animation pose. */
  maxAngleRad: number
  /** Fraction of the full turn the head takes (eyes cover the rest). */
  weight: number
}

const IDENTITY = new THREE.Quaternion()

/**
 * World-space rotation delta that turns `faceFront` (expressed in head
 * local space) toward `targetPos`, capped and weighted. Returns identity
 * when the target is degenerate (at the head position).
 */
export const computeHeadAimDelta = (
  headWorldQuat: THREE.Quaternion,
  headWorldPos: THREE.Vector3,
  targetPos: THREE.Vector3,
  faceFront: THREE.Vector3,
  { maxAngleRad, weight }: HeadAimOptions,
): THREE.Quaternion => {
  const toTarget = targetPos.clone().sub(headWorldPos)
  if (toTarget.lengthSq() < 1e-8) return IDENTITY.clone()
  toTarget.normalize()
  const currentForward = faceFront.clone().applyQuaternion(headWorldQuat).normalize()
  const full = new THREE.Quaternion().setFromUnitVectors(currentForward, toTarget)
  // Clamp the FULL turn first, then take the weighted fraction, so the cap
  // is a real ceiling regardless of weight.
  const angle = 2 * Math.acos(Math.min(1, Math.abs(full.w)))
  let capped = full
  if (angle > maxAngleRad && angle > 1e-6) {
    capped = IDENTITY.clone().slerp(full, maxAngleRad / angle)
  }
  return IDENTITY.clone().slerp(capped, Math.min(1, Math.max(0, weight)))
}

/** Convert a world-space rotation delta into the local additive VRMViewer's
 * applyProceduralAdditive expects (bone.quaternion.multiply(additive)):
 * local = worldQuat⁻¹ · delta · worldQuat. */
export const worldDeltaToLocalAdditive = (
  delta: THREE.Quaternion,
  boneWorldQuat: THREE.Quaternion,
): THREE.Quaternion => {
  const inverse = boneWorldQuat.clone().invert()
  return inverse.multiply(delta).multiply(boneWorldQuat)
}

/** Frame-rate-independent smoothing factor: fraction of the remaining
 * distance covered this frame at `speed` (1/s). */
export const smoothingAlpha = (deltaTime: number, speed: number): number =>
  1 - Math.exp(-Math.max(0, deltaTime) * speed)

/** Head-tracking strength preference: localStorage `kurisu_head_track`,
 * '0' disables, other numbers scale the base weight. Read once at mount,
 * same contract as the springbone wind pref. */
export const readHeadTrackPref = (): number => {
  try {
    const raw = window.localStorage.getItem('kurisu_head_track')
    if (raw === null) return 1
    const parsed = Number(raw)
    return Number.isFinite(parsed) && parsed >= 0 ? parsed : 1
  } catch {
    return 1
  }
}
