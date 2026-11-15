/**
 * Default relaxed finger curl, applied every frame BEFORE the animation
 * mixer runs. Most library clips (emote pack, BVH/game retargets) carry no
 * finger tracks, which leaves the hands frozen in the rig's flat splayed
 * bind pose — the "paddle hands" look whenever a gesture raises a hand.
 * Writing a gentle curl first gives trackless clips natural hands for
 * free, while any clip that does animate fingers (Peace Sign, hitareas,
 * pose clips) simply overwrites it inside the same mixer update.
 *
 * Lifelike pass 2026-07-30: the curl is no longer a frozen pose. Real
 * resting hands are never perfectly still — each finger drifts a few
 * percent around its relaxed curl on its own slow phase. Pass the frame
 * time to get that per-finger noise (phase-offset per finger AND per hand,
 * slightly different frequency per finger so the ensemble never reads as
 * one synchronized pump); omit it for the original static pose.
 *
 * Rotations are in VRM normalized-bone space (T-pose, world-aligned
 * axes): fingers point along +X (left) / -X (right) with palms facing
 * down, so a curl toward the palm is a rotation about Z — negative for
 * the left hand, positive for the right. Thumbs sit roughly 90° from the
 * fingers and get a smaller wrap about Y, mirrored the other way.
 */
import * as THREE from 'three'
import type { VRM } from '@pixiv/three-vrm'
import type { VRMHumanBoneName } from '@pixiv/three-vrm'

const FINGER_SEGMENTS: Array<[string, number]> = [
  ['Proximal', 0.35],
  ['Intermediate', 0.45],
  ['Distal', 0.22],
]
const FINGERS = ['Index', 'Middle', 'Ring', 'Little']
const THUMB_SEGMENTS: Array<[string, number]> = [
  ['Metacarpal', 0.1],
  ['Proximal', 0.16],
  ['Distal', 0.14],
]

/** Fraction of the base curl the idle noise swings by (±). Kept small —
 * this must read as circulation, not fidgeting. */
export const FINGER_NOISE_AMP = 0.1
const THUMB_NOISE_AMP = 0.05

interface FingerBoneSpec {
  bone: VRMHumanBoneName
  axis: THREE.Vector3
  baseAngle: number // signed, radians
  /** Noise channel: all segments of one finger flex together, so they
   * share a phase/frequency; fingers and hands differ. */
  phase: number
  freqHz: number
  noiseAmp: number
}

const Z_AXIS = new THREE.Vector3(0, 0, 1)
const Y_AXIS = new THREE.Vector3(0, 1, 0)

const RELAXED: FingerBoneSpec[] = (() => {
  const out: FingerBoneSpec[] = []
  for (const side of ['left', 'right'] as const) {
    const sign = side === 'left' ? -1 : 1
    const handPhase = side === 'left' ? 0 : 2.3 // hands never sync up
    FINGERS.forEach((finger, fingerIndex) => {
      // Golden-angle-ish spread keeps neighboring fingers visibly out of
      // phase; per-finger frequency spread (0.09-0.15Hz) breaks ensemble
      // periodicity the way driftNoise does for the torso.
      const phase = handPhase + fingerIndex * 2.4
      const freqHz = 0.09 + fingerIndex * 0.02
      for (const [segment, angle] of FINGER_SEGMENTS) {
        out.push({
          bone: `${side}${finger}${segment}` as VRMHumanBoneName,
          axis: Z_AXIS,
          baseAngle: sign * angle,
          phase,
          freqHz,
          noiseAmp: FINGER_NOISE_AMP,
        })
      }
    })
    for (const [segment, angle] of THUMB_SEGMENTS) {
      out.push({
        bone: `${side}Thumb${segment}` as VRMHumanBoneName,
        axis: Y_AXIS,
        baseAngle: -sign * angle,
        phase: handPhase + 4.1,
        freqHz: 0.07,
        noiseAmp: THUMB_NOISE_AMP,
      })
    }
  }
  return out
})()

/**
 * Set every finger bone to the relaxed curl (no-op for boneless rigs).
 * With `timeS` (seconds, any monotonic clock) the curl breathes per finger;
 * without it the pose is the original static one. Writes each bone's
 * quaternion in place — no allocation per frame.
 */
export const applyRelaxedFingerPose = (vrm: VRM, timeS?: number): void => {
  const humanoid = vrm.humanoid
  if (!humanoid) return
  for (const spec of RELAXED) {
    const node = humanoid.getNormalizedBoneNode(spec.bone)
    if (!node) continue
    const modulation =
      timeS === undefined
        ? 1
        : 1 + spec.noiseAmp * Math.sin(2 * Math.PI * spec.freqHz * timeS + spec.phase)
    node.quaternion.setFromAxisAngle(spec.axis, spec.baseAngle * modulation)
  }
}
