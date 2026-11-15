/**
 * Inertialization for body-clip transitions (lifelike pass 2026-07-30,
 * research item #6 — Bollo GDC'18, Holden's exact spring math).
 *
 * A linear crossfade blends two whole-body poses at partial weight for the
 * fade duration — the "crossfade mush" where both clips lose their shape
 * and any momentum the outgoing motion had dies instantly. Inertialization
 * replaces that: the scheduler hard-cuts to the new clip at full weight,
 * and the visible pose difference at the cut (plus its velocity, estimated
 * from the last two frames) becomes a per-bone offset that a critically-
 * damped spring decays to zero on top of the new clip. The new motion is
 * authoritative from frame one; the old motion's momentum carries through
 * the transition and settles out instead of being averaged away.
 *
 * Frame contract (mirrors proceduralBody's record-and-undo additive stack,
 * and composes with it LIFO — see VRMViewer's onFrame):
 *   1. proceduralBody.undoAdditives()  — strips last frame's additives
 *   2. inertializer.undo()             — strips last frame's offsets
 *   3. mixer.update()                  — writes the raw clip pose
 *   4. inertializer.update(dt)         — consumes a pending transition,
 *      decays + applies offsets, records the visible pose history
 *   5. proceduralBody.update()         — additives on top
 * Offsets therefore ride the raw mixer pose and the recorded history never
 * contains procedural additives, so breathing/nods are never double-counted
 * across a transition. Head/neck (stripped from every clip — procedural
 * territory) and fingers (applyRelaxedFingerPose owns them) are excluded.
 *
 * Gated by the kurisu_inertialization localStorage toggle (default OFF,
 * same convention as kurisu_additive_gestures): it changes how every
 * vetted transition reads, so it stays opt-in until re-vetted. The
 * scheduler falls back to the classic crossfade whenever the hook declines.
 */

import * as THREE from 'three'
import type { VRM, VRMHumanBoneName } from '@pixiv/three-vrm'

export const INERTIALIZATION_STORAGE_KEY = 'kurisu_inertialization'

export const loadInertializationEnabled = (): boolean => {
  try {
    return window.localStorage.getItem(INERTIALIZATION_STORAGE_KEY) === 'true'
  } catch {
    return false
  }
}

export const saveInertializationEnabled = (enabled: boolean): void => {
  try {
    window.localStorage.setItem(INERTIALIZATION_STORAGE_KEY, String(enabled))
  } catch {
    // Preference just doesn't persist.
  }
}

/**
 * Exact critically-damped decay toward zero (Holden's
 * decay_spring_damper_exact), applied per component. Closed-form, so any
 * frame dt — including the idle-throttled 15fps cadence — lands on the
 * same curve; nothing integrates or explodes. `halflife` is approximately
 * the time for the offset to halve.
 */
export const decaySpringVector = (
  offset: THREE.Vector3,
  velocity: THREE.Vector3,
  halflife: number,
  dt: number
): void => {
  const y = (2 * Math.LN2) / Math.max(halflife, 1e-4)
  const eydt = Math.exp(-y * dt)
  const jx = velocity.x + offset.x * y
  const jy = velocity.y + offset.y * y
  const jz = velocity.z + offset.z * y
  offset.set(eydt * (offset.x + jx * dt), eydt * (offset.y + jy * dt), eydt * (offset.z + jz * dt))
  velocity.set(
    eydt * (velocity.x - jx * y * dt),
    eydt * (velocity.y - jy * y * dt),
    eydt * (velocity.z - jz * y * dt)
  )
}

/** Rotation offsets live as scaled angle-axis vectors (axis * radians) so
 * the same spring math drives rotation and position. Shortest-path (w
 * flipped positive) so a decay never takes the long way around. */
const quatToScaledAxis = (q: THREE.Quaternion, out: THREE.Vector3): THREE.Vector3 => {
  let { x, y, z, w } = q
  if (w < 0) {
    x = -x
    y = -y
    z = -z
    w = -w
  }
  const s = Math.sqrt(x * x + y * y + z * z)
  if (s < 1e-8) return out.set(0, 0, 0)
  const k = (2 * Math.atan2(s, w)) / s
  return out.set(x * k, y * k, z * k)
}

const scaledAxisToQuat = (v: THREE.Vector3, out: THREE.Quaternion): THREE.Quaternion => {
  const angle = v.length()
  if (angle < 1e-8) return out.identity()
  const k = Math.sin(angle / 2) / angle
  return out.set(v.x * k, v.y * k, v.z * k, Math.cos(angle / 2))
}

/** Crossfade-duration → spring halflife: a spring settles visually in
 * ~3 halflifes, so a third of the old fade keeps transition pacing familiar. */
export const halflifeForFade = (fadeDuration: number): number =>
  Math.min(0.3, Math.max(0.05, fadeDuration * 0.3))

// Core skeleton the clips actually drive. Head/neck are stripped from every
// clip (withoutBoneMotionTracks — the procedural head-aim/nod systems own
// them) and fingers are posed pre-mixer by applyRelaxedFingerPose, so
// none of those belong here.
const INERTIALIZED_BONES: VRMHumanBoneName[] = [
  'hips',
  'spine',
  'chest',
  'upperChest',
  'leftShoulder',
  'leftUpperArm',
  'leftLowerArm',
  'leftHand',
  'rightShoulder',
  'rightUpperArm',
  'rightLowerArm',
  'rightHand',
  'leftUpperLeg',
  'leftLowerLeg',
  'leftFoot',
  'leftToes',
  'rightUpperLeg',
  'rightLowerLeg',
  'rightFoot',
  'rightToes',
]

// Velocity caps: a dt hiccup at capture time would otherwise turn into a
// violent spring launch. Real gesture motion sits well under both.
const MAX_ANGULAR_VELOCITY = 12 // rad/s
const MAX_LINEAR_VELOCITY = 3 // normalized-space units/s (hips position)
// Cull thresholds — springs this small are invisible; dropping them keeps
// the steady state a true no-op.
const ROT_EPSILON = 1e-4 // rad
const POS_EPSILON = 1e-4
// A pending transition whose new action never evaluates (clip failed to
// start, mixer torn down) is dropped rather than captured against a stale
// pose.
const PENDING_MAX_UPDATES = 4

interface BoneChannel {
  node: THREE.Object3D
  prev: THREE.Quaternion
  prevPrev: THREE.Quaternion
  offset: THREE.Vector3
  velocity: THREE.Vector3
  applied: THREE.Quaternion
  hasApplied: boolean
  active: boolean
}

export interface Inertializer {
  /** Arm an inertialized transition for the clip switch the caller is
   * about to make. Returns false when continuity can't be guaranteed yet
   * (no VRM bound, or fewer than two recorded frames) — the caller should
   * then keep its classic crossfade. `hasEvaluated` defers the pose
   * capture until the mixer has actually evaluated the new action at
   * least once (a switch fired from inside mixer.update — the 'finished'
   * handler — lands mid-evaluation). */
  requestTransition: (options?: { fadeDuration?: number; hasEvaluated?: () => boolean }) => boolean
  /** Strip last frame's offsets BEFORE the mixer runs — same rationale as
   * the procedural AdditiveStack: bones the mixer drives get overwritten
   * anyway, bones it doesn't are restored so offsets never compound. */
  undo: () => void
  /** Post-mixer: consume a pending transition, advance + apply the decaying
   * offsets, and record the visible pose history. */
  update: (dt: number) => void
  /** True while any offset spring is still decaying (test/debug surface). */
  isActive: () => boolean
}

export const createInertializer = ({ getVrm }: { getVrm: () => VRM | null }): Inertializer => {
  let boundVrm: VRM | null = null
  let bones: BoneChannel[] = []
  let hipsNode: THREE.Object3D | null = null
  const hipsPrev = new THREE.Vector3()
  const hipsPrevPrev = new THREE.Vector3()
  const hipsOffset = new THREE.Vector3()
  const hipsVelocity = new THREE.Vector3()
  const hipsApplied = new THREE.Vector3()
  let hipsHasApplied = false
  let hipsActive = false

  let historyFrames = 0
  let lastDt = 0
  let anyActive = false
  let halflife = 0.12
  let pending: { halflife: number; hasEvaluated: () => boolean; updatesSeen: number } | null = null

  const undoScratchQuat = new THREE.Quaternion()
  const offsetScratchQuat = new THREE.Quaternion()

  const clearSprings = () => {
    for (const bone of bones) {
      bone.active = false
      bone.hasApplied = false
    }
    hipsActive = false
    hipsHasApplied = false
    anyActive = false
  }

  const bind = (vrm: VRM) => {
    boundVrm = vrm
    bones = []
    hipsNode = null
    for (const name of INERTIALIZED_BONES) {
      const node = vrm.humanoid?.getNormalizedBoneNode(name)
      if (!node) continue
      bones.push({
        node,
        prev: node.quaternion.clone(),
        prevPrev: node.quaternion.clone(),
        offset: new THREE.Vector3(),
        velocity: new THREE.Vector3(),
        applied: new THREE.Quaternion(),
        hasApplied: false,
        active: false,
      })
      if (name === 'hips') {
        hipsNode = node
        hipsPrev.copy(node.position)
        hipsPrevPrev.copy(node.position)
      }
    }
    historyFrames = 0
    lastDt = 0
    pending = null
    clearSprings()
  }

  const requestTransition: Inertializer['requestTransition'] = (options) => {
    if (!boundVrm || bones.length === 0 || historyFrames < 2) return false
    pending = {
      halflife: halflifeForFade(options?.fadeDuration ?? 0.4),
      hasEvaluated: options?.hasEvaluated ?? (() => true),
      updatesSeen: 0,
    }
    return true
  }

  // Capture: current bone state is the new clip's raw mixer pose; `prev` is
  // last frame's visible pose (old clip + old offsets). The offset that
  // makes them meet is mixer⁻¹ * prev, and its velocity comes from the
  // prevPrev→prev step. Existing springs are folded in for free — prev
  // already includes them, so a transition mid-decay re-captures the total.
  const consumePending = (halflifeNext: number) => {
    halflife = halflifeNext
    anyActive = false
    for (const bone of bones) {
      offsetScratchQuat.copy(bone.node.quaternion).invert().multiply(bone.prev)
      quatToScaledAxis(offsetScratchQuat, bone.offset)
      if (lastDt > 1e-4) {
        offsetScratchQuat.copy(bone.prevPrev).invert().multiply(bone.prev)
        quatToScaledAxis(offsetScratchQuat, bone.velocity)
        bone.velocity.divideScalar(lastDt).clampLength(0, MAX_ANGULAR_VELOCITY)
      } else {
        bone.velocity.set(0, 0, 0)
      }
      bone.active = bone.offset.lengthSq() > ROT_EPSILON * ROT_EPSILON || bone.velocity.lengthSq() > 1e-6
      if (bone.active) anyActive = true
    }
    if (hipsNode) {
      hipsOffset.copy(hipsPrev).sub(hipsNode.position)
      if (lastDt > 1e-4) {
        hipsVelocity.copy(hipsPrev).sub(hipsPrevPrev).divideScalar(lastDt).clampLength(0, MAX_LINEAR_VELOCITY)
      } else {
        hipsVelocity.set(0, 0, 0)
      }
      hipsActive = hipsOffset.lengthSq() > POS_EPSILON * POS_EPSILON || hipsVelocity.lengthSq() > 1e-6
      if (hipsActive) anyActive = true
    }
  }

  const undo = () => {
    for (const bone of bones) {
      if (!bone.hasApplied) continue
      bone.node.quaternion.multiply(undoScratchQuat.copy(bone.applied).invert())
      bone.hasApplied = false
    }
    if (hipsHasApplied && hipsNode) {
      hipsNode.position.sub(hipsApplied)
      hipsHasApplied = false
    }
  }

  const update = (dt: number) => {
    const vrm = getVrm()
    if (!vrm) {
      if (boundVrm) {
        boundVrm = null
        bones = []
        hipsNode = null
        historyFrames = 0
        pending = null
        clearSprings()
      }
      return
    }
    if (vrm !== boundVrm) bind(vrm)

    if (pending) {
      pending.updatesSeen += 1
      if (pending.hasEvaluated()) {
        consumePending(pending.halflife)
        pending = null
      } else if (pending.updatesSeen >= PENDING_MAX_UPDATES) {
        pending = null
      }
    }

    if (anyActive && dt > 0) {
      let stillActive = false
      for (const bone of bones) {
        if (!bone.active) continue
        decaySpringVector(bone.offset, bone.velocity, halflife, dt)
        if (
          bone.offset.lengthSq() < ROT_EPSILON * ROT_EPSILON &&
          bone.velocity.lengthSq() < 1e-6
        ) {
          bone.active = false
          continue
        }
        scaledAxisToQuat(bone.offset, bone.applied)
        bone.node.quaternion.multiply(bone.applied)
        bone.hasApplied = true
        stillActive = true
      }
      if (hipsActive && hipsNode) {
        decaySpringVector(hipsOffset, hipsVelocity, halflife, dt)
        if (hipsOffset.lengthSq() < POS_EPSILON * POS_EPSILON && hipsVelocity.lengthSq() < 1e-6) {
          hipsActive = false
        } else {
          hipsNode.position.add(hipsApplied.copy(hipsOffset))
          hipsHasApplied = true
          stillActive = true
        }
      }
      anyActive = stillActive
    }

    // History records the VISIBLE pose (mixer + offsets, no procedural
    // additives — they're applied after this by frame order), so the next
    // transition's continuity target is what was actually on screen.
    for (const bone of bones) {
      bone.prevPrev.copy(bone.prev)
      bone.prev.copy(bone.node.quaternion)
    }
    if (hipsNode) {
      hipsPrevPrev.copy(hipsPrev)
      hipsPrev.copy(hipsNode.position)
    }
    historyFrames += 1
    lastDt = dt
  }

  return {
    requestTransition,
    undo,
    update,
    isActive: () => anyActive,
  }
}
