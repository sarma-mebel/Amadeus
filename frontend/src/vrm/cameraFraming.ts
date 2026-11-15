/**
 * Procedural camera framing math, extracted verbatim (functions made pure —
 * explicit params instead of closing over VRMViewer's effect-local `camera`/
 * `controls`) from VRMViewer's scene effect. The shot is derived from the
 * VRM's own bone positions + the live viewport aspect, so it stays correct
 * regardless of window shape and regardless of exactly where the model is
 * placed in whatever environment GLB is loaded — see VRMViewer for how this
 * is wired into the initial snap-to-framing and the aspect/follow/auto-return
 * camera-rig behavior.
 */

import * as THREE from 'three'
import type { VRM } from '@pixiv/three-vrm'

// --- Camera framing constants ---------------------------------------------

// Pre-load placeholder camera pose (approximates the old hardcoded shot) —
// shown only for the brief window between scene setup and VRM load; replaced
// by the computed framing the instant the model's bones are available.
export const PRE_LOAD_CAMERA_POSITION = new THREE.Vector3(-6.55, 1.42, 0.55)
export const PRE_LOAD_CAMERA_TARGET = new THREE.Vector3(-5, 1.32, 0)

// Medium-shot framing: top margin above the head, bottom cut at mid-thigh.
// The head *bone* sits near the neck/chin, not the skull apex, so this
// approximates the extra distance up to the actual top of the skull.
export const FRAMING_HEAD_TOP_MARGIN = 0.12
export const FRAMING_THIGH_FRACTION = 0.5 // fraction of the way from hips to knee used as the bottom crop line

// Compact (pet-preset "compact") framing: bust/head-and-shoulders instead
// of the medium (head-to-mid-thigh) shot above — used only by the pet
// window's "compact" preset (see VRMViewer's pet-preset listener). Bottom
// crop line sits a bit below the shoulder bones rather than at the thigh,
// scaled by the model's own vertical extent so it holds proportion across
// different VRM heights instead of a fixed meter offset.
export const FRAMING_COMPACT_CHEST_DROP_FRACTION = 0.22
// Camera sits off to one side of her facing direction rather than dead-on,
// a more flattering 3/4 angle standard in character-viewer cinematography.
export const FRAMING_YAW_OFFSET_DEG = 18 // degrees off facing direction, within the 15-20 range
// Look-at target height: fraction of the way up from hips to head-top,
// landing roughly at chest/collarbone height instead of the literal midpoint.
export const FRAMING_TARGET_HEIGHT_FRACTION = 0.6
// Extra breathing room so the computed distance doesn't crop right at the
// edges of frame — applied to both the vertical (height) and horizontal
// (aspect-driven) fit.
export const FRAMING_HEADROOM = 0.1 // 10% margin in both axes

// Orbit clamps (derived from model height at load time, see
// applyOrbitClampsForHeight below) — expressed as multipliers/degrees here.
export const ORBIT_MIN_DISTANCE_FACTOR = 0.5 // closest zoom-in: 0.5x model height
export const ORBIT_MAX_DISTANCE_FACTOR = 4 // furthest zoom-out: 4x model height
export const ORBIT_MIN_POLAR_DEG = 30 // can't tip the camera to look flat down from above
export const ORBIT_MAX_POLAR_DEG = 100 // can't dip under the floor / look straight up her skirt

// Follow-camera behavior: anchor blends hips + head so crouches/leans keep
// her face in frame, rather than tracking hips alone.
export const FOLLOW_HIPS_WEIGHT = 0.4
export const FOLLOW_HEAD_WEIGHT = 0.6
export const DEADZONE = 0.05 // meters of anchor drift ignored, so idle sway doesn't jitter the shot
export const FOLLOW_LERP = 0.02 // per-frame smoothing toward the anchor
export const RESUME_DELAY_MS = 5000 // let the user's framing hold for a bit after they let go
// How far controls.target may stray from the character's live chest anchor
// while panning — keeps "enablePan" from letting the target drift off into
// the set.
export const TARGET_CLAMP_RADIUS = 1.5 // meters
// After this long without any user camera input, the shot eases back toward
// the original cinematographic offset (distance + angle), so orbiting around
// to look at something doesn't leave the camera stuck there forever.
export const AUTO_RETURN_IDLE_MS = 20000
export const AUTO_RETURN_LERP = 0.005 // per-frame pull back toward the default offset

// Aspect-driven reframe (item 2): when the canvas resizes, the required
// distance for the new aspect is eased in rather than snapped, and only
// while the user isn't actively driving the camera.
export const REFRAME_DURATION_S = 1

// Double-click-to-reframe tween duration (item 5).
export const REFRAME_TWEEN_DURATION_S = 0.8

// Computed once the VRM's bones are available (see computeDefaultFraming
// below, invoked from the VRM load callback). Holds the "home" shot that
// orbit clamps, aspect reframing, follow-offset auto-return, and
// double-click-to-reframe all reference.
export type DefaultFraming = {
  target: THREE.Vector3 // chest-height look-at point (world space)
  offset: THREE.Vector3 // camera position relative to target (unit direction * distance baked in per-aspect)
  direction: THREE.Vector3 // unit direction from target to camera (aspect-independent)
  modelHeight: number // vertical extent used to derive orbit clamps
  topHalf: number // distance from target up to head-top (diagnostic only)
  bottomHalf: number // distance from target down to mid-thigh (diagnostic only)
  // Bone-derived points the frame must contain — kept so the aspect
  // reframe (item 2) can re-solve the exact required distance for a new
  // aspect instead of relying on a flattened vertical-extent estimate.
  mustFramePoints: THREE.Vector3[]
}

/**
 * Computes the camera distance (measured along `direction`, from `target`)
 * required so every point in `worldPoints` stays inside both the vertical
 * and horizontal FOV cones, with headroom margin.
 *
 * A closed-form "vertical extent / 2 / tan(halfFOV)" estimate (the first
 * version of this function) assumes every framed point sits exactly on the
 * target's depth plane with zero lateral/depth offset — untrue for a real
 * humanoid rig, where e.g. a bent knee or an outstretched arm sits
 * measurably off-axis in X/Z as well as Y. At the close range a medium shot
 * needs, that off-axis offset produces a noticeably larger angular size than
 * the flat vertical-only estimate predicts, so it under-shoots the required
 * distance and crops the extremity. Solving directly for the angle-from-
 * camera-forward of each actual 3D point (via binary search on distance)
 * accounts for this and is exact regardless of pose.
 */
export const solveFramingDistance = (
  target: THREE.Vector3,
  direction: THREE.Vector3,
  worldPoints: THREE.Vector3[],
  aspect: number,
  fovDeg: number
): number => {
  const vFovRad = THREE.MathUtils.degToRad(fovDeg)
  const hFovRad = 2 * Math.atan(Math.tan(vFovRad / 2) * aspect)
  // Margin shrinks the usable half-angle so the point lands inside the
  // frame with headroom, rather than exactly on the edge.
  const maxVAngle = (vFovRad / 2) / (1 + FRAMING_HEADROOM)
  const maxHAngle = (hFovRad / 2) / (1 + FRAMING_HEADROOM)

  const up = new THREE.Vector3(0, 1, 0)
  const right = new THREE.Vector3().crossVectors(direction, up).normalize()
  // Recompute a true "up" orthogonal to direction (direction may have a
  // slight vertical component from the framing lift).
  const trueUp = new THREE.Vector3().crossVectors(right, direction).normalize()

  // Camera sits at target + direction*distance (direction points from
  // target OUT to the camera), so the camera's actual view-forward axis
  // — from camera toward the subject — is the opposite of `direction`.
  const cameraForward = direction.clone().negate()
  const fits = (distance: number): boolean => {
    const camPos = target.clone().addScaledVector(direction, distance)
    return worldPoints.every((point) => {
      const toPoint = point.clone().sub(camPos)
      const forwardComponent = toPoint.dot(cameraForward)
      if (forwardComponent <= 0) return false // behind the camera
      const vComponent = toPoint.dot(trueUp)
      const hComponent = toPoint.dot(right)
      const vAngle = Math.atan2(Math.abs(vComponent), forwardComponent)
      const hAngle = Math.atan2(Math.abs(hComponent), forwardComponent)
      return vAngle <= maxVAngle && hAngle <= maxHAngle
    })
  }

  // Binary search for the minimal distance that fits every point —
  // "fits" is monotonic in distance (dollying back only shrinks angles),
  // so this converges quickly. Upper bound generous enough for any
  // reasonable humanoid scale; lower bound near zero.
  let low = 0.05
  let high = 20
  for (let i = 0; i < 30; i += 1) {
    const mid = (low + high) / 2
    if (fits(mid)) {
      high = mid
    } else {
      low = mid
    }
  }
  return high
}

/**
 * Reads the VRM's own bones to build a medium-shot framing: from head-top
 * down to mid-thigh, chest-height look-at, camera at a slight yaw offset
 * from her facing direction. Called once on load, and again (for direction/
 * target only — distance is recomputed per-aspect) never needs to re-run
 * afterward since the rig is bone-relative from here on.
 *
 * `framingMode` ('medium', default) picks the shot: 'compact' is the pet
 * window's tighter bust/head-and-shoulders preset — same bone-derived
 * approach (never a hardcoded camera offset), just a bottom crop line at
 * the shoulders (with a small drop for breathing room) instead of the
 * mid-thigh. Both modes derive facing direction from vrm.lookAt.faceFront,
 * never an assumed +Z — see the VRM0/VRM1 note below (the same rule this
 * function has always followed for the medium shot).
 */
export const computeDefaultFraming = (
  vrm: VRM,
  aspect: number,
  fovDeg: number,
  framingMode: 'medium' | 'compact' = 'medium'
): DefaultFraming | null => {
  const humanoid = vrm.humanoid
  if (!humanoid) return null
  // Bone world matrices are only current as of the last render/update
  // call; right after scene.add() (before the mixer has ever ticked)
  // they can still be stale/identity, which would make every bone read
  // back near the scene root instead of its actual rest-pose position.
  // Force a fresh matrix pass so the world positions below are correct
  // for a same-frame read.
  vrm.scene.updateMatrixWorld(true)
  const headBone = humanoid.getNormalizedBoneNode('head')
  const hipsBone = humanoid.getNormalizedBoneNode('hips')
  // "leftUpperLeg"/"rightUpperLeg" is the hip joint itself in the VRM
  // humanoid rig (sits right at the hips, not partway down the thigh) —
  // the lower-leg bone is the knee joint, which is what we actually want
  // as the bottom-of-medium-shot reference.
  const kneeBone =
    humanoid.getNormalizedBoneNode('leftLowerLeg') ?? humanoid.getNormalizedBoneNode('rightLowerLeg')
  const leftShoulderBone = humanoid.getNormalizedBoneNode('leftUpperArm')
  const rightShoulderBone = humanoid.getNormalizedBoneNode('rightUpperArm')
  if (!headBone || !hipsBone) return null

  const headWorld = headBone.getWorldPosition(new THREE.Vector3())
  const hipsWorld = hipsBone.getWorldPosition(new THREE.Vector3())
  const kneeWorld = kneeBone?.getWorldPosition(new THREE.Vector3()) ?? null
  const leftShoulderWorld = leftShoulderBone?.getWorldPosition(new THREE.Vector3()) ?? null
  const rightShoulderWorld = rightShoulderBone?.getWorldPosition(new THREE.Vector3()) ?? null

  // Head-top approximated as a bit above the head joint (joint sits near
  // the neck/chin, not the skull apex).
  const headTopWorld = headWorld.clone()
  headTopWorld.y += FRAMING_HEAD_TOP_MARGIN
  // Mid-thigh: partway from hips down to the knee (falls back to a flat
  // offset below the hips if a leg bone isn't present in this rig). Uses
  // the knee's own X/Z (not just its Y) since a bent/animated leg can
  // sit measurably forward or behind the hips.
  const thighWorld = kneeWorld
    ? hipsWorld.clone().lerp(kneeWorld, FRAMING_THIGH_FRACTION)
    : hipsWorld.clone().add(new THREE.Vector3(0, -0.35, 0))

  // Compact preset: bottom line sits a bit below the shoulder bones
  // instead of at the thigh — falls back to the medium shot's thigh line
  // if this rig has no shoulder bones (mustFramePoints already tolerates
  // that below). Scaled by the medium shot's own vertical extent so the
  // "a bit below the shoulders" drop holds proportion across VRM heights
  // rather than a fixed meter offset.
  const mediumVerticalExtent = Math.max(0.1, headTopWorld.y - thighWorld.y)
  const shoulderY =
    leftShoulderWorld && rightShoulderWorld
      ? (leftShoulderWorld.y + rightShoulderWorld.y) / 2
      : (leftShoulderWorld ?? rightShoulderWorld)?.y ?? null
  const bottomWorld =
    framingMode === 'compact' && shoulderY !== null
      ? new THREE.Vector3(
          hipsWorld.x,
          shoulderY - mediumVerticalExtent * FRAMING_COMPACT_CHEST_DROP_FRACTION,
          hipsWorld.z
        )
      : thighWorld
  const verticalExtent = Math.max(0.1, headTopWorld.y - bottomWorld.y)

  const targetY = bottomWorld.y + verticalExtent * FRAMING_TARGET_HEIGHT_FRACTION
  const target = new THREE.Vector3(hipsWorld.x, targetY, hipsWorld.z)

  // Facing direction: derive from the model root's yaw rather than
  // hardcoding an axis, so this keeps working if the model is ever
  // repositioned/rotated in the scene setup above.
  // Her true facing axis, not an assumed one: VRM1 models face root +Z but
  // VRM0 models face root -Z, and three-vrm reports which via
  // lookAt.faceFront. Hardcoding +Z made the auto-framing camera settle
  // BEHIND every VRM0 model regardless of how the root was rotated.
  const faceFront = vrm.lookAt?.faceFront ?? new THREE.Vector3(0, 0, 1)
  const forward = faceFront.clone().applyQuaternion(vrm.scene.getWorldQuaternion(new THREE.Quaternion()))
  forward.y = 0
  if (forward.lengthSq() < 1e-6) forward.set(0, 0, 1)
  forward.normalize()

  const yawOffsetRad = THREE.MathUtils.degToRad(FRAMING_YAW_OFFSET_DEG)
  const direction = forward.clone().applyAxisAngle(new THREE.Vector3(0, 1, 0), yawOffsetRad)

  // Points the frame must contain, in real 3D (not a flattened vertical
  // estimate) — see solveFramingDistance for why this matters: an
  // off-axis point (bent knee, raised arm) subtends a bigger angle at
  // close range than its Y-difference alone suggests. Compact mode drops
  // hips/thigh from the required set (a bust shot is explicitly allowed to
  // crop below the shoulders) but keeps the shoulder points themselves —
  // they ARE the compact bottom line.
  const mustFramePoints =
    framingMode === 'compact'
      ? [headTopWorld, bottomWorld]
      : [headTopWorld, thighWorld, hipsWorld]
  if (leftShoulderWorld) mustFramePoints.push(leftShoulderWorld)
  if (rightShoulderWorld) mustFramePoints.push(rightShoulderWorld)

  const distance = solveFramingDistance(target, direction, mustFramePoints, aspect, fovDeg)
  const offset = direction.clone().multiplyScalar(distance)
  // Small vertical lift so the camera isn't perfectly level with the
  // chest target — a slight downward look reads more natural. Folded
  // into a re-solve pass below so the lift doesn't reintroduce cropping.
  const liftedDirection = offset
    .clone()
    .add(new THREE.Vector3(0, verticalExtent * 0.08, 0))
    .normalize()
  const finalDistance = solveFramingDistance(target, liftedDirection, mustFramePoints, aspect, fovDeg)
  const finalOffset = liftedDirection.clone().multiplyScalar(finalDistance)

  // modelHeight backs out an estimated full-body height from the
  // head-to-thigh extent for the orbit-distance clamps (orbitClampsForHeight)
  // — always derived from the medium shot's own extent regardless of
  // framingMode, so a compact (bust-only) framing doesn't feed a wildly
  // wrong "model height" into clamps that assume a full-body scale. (The
  // pet variant that uses 'compact' also has OrbitControls disabled
  // entirely — see sceneSetup.ts — so this mostly matters for keeping the
  // returned diagnostic honest.)
  const modelHeight = mediumVerticalExtent / (1 - FRAMING_THIGH_FRACTION * 0.5)

  return {
    target,
    offset: finalOffset,
    direction: liftedDirection,
    modelHeight,
    topHalf: headTopWorld.y - targetY,
    bottomHalf: targetY - bottomWorld.y,
    mustFramePoints,
  }
}

/**
 * Orbit clamps derived from the model's own vertical extent (item 3):
 * prevents zooming to infinity, clipping through the model, tipping under
 * the floor, or looking straight down from directly overhead. Returns the
 * clamp values rather than mutating `controls` directly so callers stay in
 * charge of applying them (VRMViewer applies them to its OrbitControls
 * instance).
 */
export const orbitClampsForHeight = (
  modelHeight: number
): { minDistance: number; maxDistance: number; minPolarAngle: number; maxPolarAngle: number } => ({
  minDistance: modelHeight * ORBIT_MIN_DISTANCE_FACTOR,
  maxDistance: modelHeight * ORBIT_MAX_DISTANCE_FACTOR,
  minPolarAngle: THREE.MathUtils.degToRad(ORBIT_MIN_POLAR_DEG),
  maxPolarAngle: THREE.MathUtils.degToRad(ORBIT_MAX_POLAR_DEG),
})
