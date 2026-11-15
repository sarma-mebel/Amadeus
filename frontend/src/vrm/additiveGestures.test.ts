// @vitest-environment jsdom
import { describe, expect, it } from 'vitest'
import * as THREE from 'three'
import {
  ADDITIVE_GESTURES_STORAGE_KEY,
  getAdditiveGestureClip,
  loadAdditiveGesturesEnabled,
  saveAdditiveGesturesEnabled,
} from './additiveGestures'

const makeGestureClip = (): THREE.AnimationClip => {
  // A quaternion track that starts away from identity (a raised-arm pose)
  // plus a hips position track — the two things conversion must handle.
  const q0 = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 0, 1), 0.5)
  const q1 = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 0, 1), 1.1)
  const rotTrack = new THREE.QuaternionKeyframeTrack(
    'RightArm.quaternion',
    [0, 1],
    [q0.x, q0.y, q0.z, q0.w, q1.x, q1.y, q1.z, q1.w],
  )
  const posTrack = new THREE.VectorKeyframeTrack('Hips.position', [0, 1], [0, 1, 0, 0.1, 1, 0.2])
  return new THREE.AnimationClip('Test Gesture', 1, [rotTrack, posTrack])
}

describe('getAdditiveGestureClip', () => {
  it('strips position tracks and converts rotations to additive deltas', () => {
    const source = makeGestureClip()
    const additive = getAdditiveGestureClip(source)

    expect(additive.tracks.some((track) => track.name.endsWith('.position'))).toBe(false)
    expect(additive.blendMode).toBe(THREE.AdditiveAnimationBlendMode)

    // Frame 0 is the reference frame, so the converted first keyframe is
    // the identity quaternion — the gesture ENTERS at zero delta (this is
    // what kills the pose-teleport on entry).
    const rot = additive.tracks.find((track) => track.name.endsWith('.quaternion'))!
    const first = new THREE.Quaternion().fromArray(rot.values as unknown as number[], 0)
    expect(Math.abs(first.x)).toBeLessThan(1e-6)
    expect(Math.abs(first.y)).toBeLessThan(1e-6)
    expect(Math.abs(first.z)).toBeLessThan(1e-6)
    expect(Math.abs(Math.abs(first.w) - 1)).toBeLessThan(1e-6)
  })

  it('never mutates the source clip (crossfade playback keeps working)', () => {
    const source = makeGestureClip()
    const originalValues = Array.from(source.tracks[0].values)
    getAdditiveGestureClip(source)
    expect(Array.from(source.tracks[0].values)).toEqual(originalValues)
    expect(source.tracks).toHaveLength(2)
    expect(source.blendMode).not.toBe(THREE.AdditiveAnimationBlendMode)
  })

  it('caches the conversion per source clip', () => {
    const source = makeGestureClip()
    expect(getAdditiveGestureClip(source)).toBe(getAdditiveGestureClip(source))
    // A different clip instance gets its own conversion.
    expect(getAdditiveGestureClip(makeGestureClip())).not.toBe(getAdditiveGestureClip(source))
  })
})

describe('additive gestures toggle', () => {
  it('defaults off (pending anim-lab re-vetting) and round-trips', () => {
    window.localStorage.removeItem(ADDITIVE_GESTURES_STORAGE_KEY)
    expect(loadAdditiveGesturesEnabled()).toBe(false)
    saveAdditiveGesturesEnabled(true)
    expect(loadAdditiveGesturesEnabled()).toBe(true)
    saveAdditiveGesturesEnabled(false)
    expect(loadAdditiveGesturesEnabled()).toBe(false)
  })
})
