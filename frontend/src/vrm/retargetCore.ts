// Shared rest-pose retargeting math for mixamoRetarget.ts and bvhRetarget.ts.
//
// Both retargeters port the same core from the official @pixiv/three-vrm
// example (examples/loadMixamoAnimation.js in pixiv/three-vrm): for each
// source track, compose the parent's rest-world rotation with the inverse
// of a rest rotation so the animated local rotation lines up with the VRM's
// (possibly differently-posed) rest skeleton, then rebuild the keyframe
// track bound by the target VRM node's own NAME (humanoid bone names aren't
// node names, so a track named after them never binds to anything).
//
// What's genuinely different per rig (and stays in the two callers, not
// here): which rest rotation `restRotationInverse` is computed from (the
// VRM target node's own local rotation for Mixamo; the source skeleton
// bone's world rotation for BVH), BVH's extra 180°-Y-flip normalization for
// clips authored facing -Z, and BVH's "trust the first frame" hips-height
// reference-height fallback. Those live as optional hooks below so the
// shared loops stay identical to what each file had inline.
import * as THREE from 'three'

/** Scratch quaternion reused across calls — callers process one track at a
 * time (never concurrently), matching the original inline scratch objects. */
const _quatA = new THREE.Quaternion()

export interface RetargetQuaternionOptions {
  /** Rest rotation to undo (inverted) before composing with the parent's
   * rest-world rotation — Mixamo uses the VRM target node's own rest
   * rotation; BVH uses the source bone's rest-world rotation. */
  restRotationInverse: THREE.Quaternion
  /** Source rig parent's rest-world rotation, composed in first (matching
   * the pixiv example's `premultiply(parentRestWorldRotation)`). */
  parentRestWorldRotation: THREE.Quaternion
  /** BVH-only: premultiply a 180° Y-axis flip into every keyframe before
   * the rest-pose composition, for clips authored facing -Z. Omitted (or
   * false) is a no-op, matching Mixamo's clips (which never need it). */
  preFlip?: THREE.Quaternion | null
}

/**
 * Retargets one quaternion keyframe track's values: per-keyframe, optionally
 * pre-flip, then `premultiply(parentRestWorldRotation).multiply(restRotationInverse)`
 * — identical math/order to the original inline loops in both retargeters.
 */
export const retargetQuaternionValues = (
  track: THREE.QuaternionKeyframeTrack,
  { restRotationInverse, parentRestWorldRotation, preFlip }: RetargetQuaternionOptions
): Float32Array => {
  const values = new Float32Array(track.values.length)
  for (let i = 0; i < track.values.length; i += 4) {
    _quatA.fromArray(track.values, i)
    if (preFlip) _quatA.premultiply(preFlip)
    _quatA.premultiply(parentRestWorldRotation).multiply(restRotationInverse)
    _quatA.toArray(values, i)
  }
  return values
}

/**
 * Builds a VRM-bound quaternion track from retargeted values. `vrmNodeName`
 * must be the target node's own `.name` (see module docstring) — not the
 * humanoid bone name.
 */
export const buildRetargetedQuaternionTrack = (
  vrmNodeName: string,
  times: ArrayLike<number>,
  values: Float32Array
): THREE.QuaternionKeyframeTrack =>
  new THREE.QuaternionKeyframeTrack(`${vrmNodeName}.quaternion`, Array.from(times), Array.from(values))

/**
 * Rescales a position keyframe track's values by a uniform hips-height
 * ratio (both source rigs are authored at a fixed real-world/arbitrary
 * scale that has to be normalized against the target VRM's actual hips
 * height, or a hip translation track looks wrong — character floats/sinks).
 * `signX`/`signZ` let BVH mirror X/Z for clips that needed the 180° Y-flip
 * above (Mixamo never mirrors, so its caller leaves these at the default 1).
 */
export const retargetPositionValues = (
  track: THREE.VectorKeyframeTrack,
  scale: number,
  signX = 1,
  signZ = 1
): Float32Array => {
  const values = new Float32Array(track.values.length)
  for (let i = 0; i < track.values.length; i += 3) {
    values[i] = track.values[i] * scale * signX
    values[i + 1] = track.values[i + 1] * scale
    values[i + 2] = track.values[i + 2] * scale * signZ
  }
  return values
}

/** Builds a VRM-bound position track from retargeted values (see
 * `buildRetargetedQuaternionTrack` re: node name vs. humanoid bone name). */
export const buildRetargetedPositionTrack = (
  vrmNodeName: string,
  times: ArrayLike<number>,
  values: Float32Array
): THREE.VectorKeyframeTrack =>
  new THREE.VectorKeyframeTrack(`${vrmNodeName}.position`, Array.from(times), Array.from(values))

/**
 * Hips-height scale ratio: target VRM hips height (from its root) over a
 * source reference height, floored to avoid divide-by-zero on a degenerate
 * rig. Shared by both retargeters; BVH additionally picks its own
 * `referenceHeight` (see bvhRetarget.ts's "trust the first frame" fallback)
 * before calling this.
 */
export const hipsHeightScale = (vrmHipsHeight: number, referenceHeight: number): number =>
  (Math.abs(vrmHipsHeight) || 1) / (Math.abs(referenceHeight) || 1)
