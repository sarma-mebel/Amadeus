import { describe, expect, it } from 'vitest'
import * as THREE from 'three'
import {
  buildRetargetedPositionTrack,
  buildRetargetedQuaternionTrack,
  hipsHeightScale,
  retargetPositionValues,
  retargetQuaternionValues,
} from './retargetCore'

describe('hipsHeightScale', () => {
  it('divides the target height by the reference height', () => {
    expect(hipsHeightScale(2, 1)).toBeCloseTo(2)
    expect(hipsHeightScale(1, 2)).toBeCloseTo(0.5)
  })

  it('floors both inputs at 1 to avoid divide-by-zero on a degenerate rig', () => {
    expect(hipsHeightScale(0, 5)).toBeCloseTo(1 / 5)
    expect(hipsHeightScale(5, 0)).toBeCloseTo(5)
    expect(hipsHeightScale(0, 0)).toBeCloseTo(1)
  })

  it('takes the absolute value of negative inputs', () => {
    expect(hipsHeightScale(-2, 1)).toBeCloseTo(2)
    expect(hipsHeightScale(2, -1)).toBeCloseTo(2)
  })
})

describe('retargetQuaternionValues', () => {
  it('passes an identity keyframe through unchanged when rest rotations are identity', () => {
    const track = new THREE.QuaternionKeyframeTrack(
      'src.quaternion',
      [0],
      [0, 0, 0, 1] // identity quaternion
    )
    const values = retargetQuaternionValues(track, {
      restRotationInverse: new THREE.Quaternion(),
      parentRestWorldRotation: new THREE.Quaternion(),
    })
    expect(Array.from(values)).toEqual([0, 0, 0, 1])
  })

  it('composes parentRestWorldRotation and restRotationInverse in the documented order', () => {
    // A known case: source keyframe is a 90 deg rotation about X. The rest
    // rotation to undo is a 90 deg rotation about X too (so it should
    // cancel and leave only the parent's rest-world rotation, a 90 deg
    // rotation about Y).
    const ninetyX = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(1, 0, 0), Math.PI / 2)
    const ninetyY = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 1, 0), Math.PI / 2)
    const restRotationInverse = ninetyX.clone().invert()

    const track = new THREE.QuaternionKeyframeTrack(
      'src.quaternion',
      [0],
      [ninetyX.x, ninetyX.y, ninetyX.z, ninetyX.w]
    )
    const values = retargetQuaternionValues(track, {
      restRotationInverse,
      parentRestWorldRotation: ninetyY,
    })

    const expected = ninetyX.clone().premultiply(ninetyY).multiply(restRotationInverse)
    expect(values[0]).toBeCloseTo(expected.x)
    expect(values[1]).toBeCloseTo(expected.y)
    expect(values[2]).toBeCloseTo(expected.z)
    expect(values[3]).toBeCloseTo(expected.w)
    // With the rest rotation canceling exactly, the result collapses to the
    // parent's rest-world rotation.
    expect(values[0]).toBeCloseTo(ninetyY.x)
    expect(values[1]).toBeCloseTo(ninetyY.y)
    expect(values[2]).toBeCloseTo(ninetyY.z)
    expect(values[3]).toBeCloseTo(ninetyY.w)
  })

  it('applies an optional preFlip before the rest-pose composition (BVH-only hook)', () => {
    const yFlip = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 1, 0), Math.PI)
    const identity = new THREE.Quaternion()
    const track = new THREE.QuaternionKeyframeTrack('src.quaternion', [0], [0, 0, 0, 1])

    const withoutFlip = retargetQuaternionValues(track, {
      restRotationInverse: identity,
      parentRestWorldRotation: identity,
    })
    const withFlip = retargetQuaternionValues(track, {
      restRotationInverse: identity,
      parentRestWorldRotation: identity,
      preFlip: yFlip,
    })

    expect(Array.from(withoutFlip)).toEqual([0, 0, 0, 1])
    expect(withFlip[1]).toBeCloseTo(yFlip.y)
    expect(withFlip[3]).toBeCloseTo(yFlip.w)
  })

  it('retargets every keyframe in a multi-frame track independently', () => {
    const q0 = new THREE.Quaternion()
    const q1 = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 0, 1), Math.PI / 4)
    const track = new THREE.QuaternionKeyframeTrack(
      'src.quaternion',
      [0, 1],
      [q0.x, q0.y, q0.z, q0.w, q1.x, q1.y, q1.z, q1.w]
    )
    const values = retargetQuaternionValues(track, {
      restRotationInverse: new THREE.Quaternion(),
      parentRestWorldRotation: new THREE.Quaternion(),
    })
    expect(values.length).toBe(8)
    expect(values[4]).toBeCloseTo(q1.x)
    expect(values[5]).toBeCloseTo(q1.y)
    expect(values[6]).toBeCloseTo(q1.z)
    expect(values[7]).toBeCloseTo(q1.w)
  })
})

describe('buildRetargetedQuaternionTrack', () => {
  it('names the track after the VRM node (not the humanoid bone name)', () => {
    const values = new Float32Array([0, 0, 0, 1])
    const track = buildRetargetedQuaternionTrack('J_Bip_C_Hips', [0], values)
    expect(track.name).toBe('J_Bip_C_Hips.quaternion')
    expect(track).toBeInstanceOf(THREE.QuaternionKeyframeTrack)
    expect(Array.from(track.times)).toEqual([0])
    expect(Array.from(track.values)).toEqual([0, 0, 0, 1])
  })
})

describe('retargetPositionValues', () => {
  it('scales every component uniformly by default (no mirroring)', () => {
    const track = new THREE.VectorKeyframeTrack('hips.position', [0], [1, 2, 3])
    const values = retargetPositionValues(track, 2)
    expect(Array.from(values)).toEqual([2, 4, 6])
  })

  it('mirrors X/Z independently via signX/signZ (BVH Y-flip hook) but never Y', () => {
    const track = new THREE.VectorKeyframeTrack('hips.position', [0], [1, 2, 3])
    const values = retargetPositionValues(track, 1, -1, -1)
    expect(Array.from(values)).toEqual([-1, 2, -3])
  })

  it('scales every keyframe in a multi-frame track', () => {
    const track = new THREE.VectorKeyframeTrack('hips.position', [0, 1], [1, 1, 1, 2, 2, 2])
    const values = retargetPositionValues(track, 3)
    expect(Array.from(values)).toEqual([3, 3, 3, 6, 6, 6])
  })
})

describe('buildRetargetedPositionTrack', () => {
  it('names the track after the VRM node with a .position suffix', () => {
    const values = new Float32Array([1, 2, 3])
    const track = buildRetargetedPositionTrack('J_Bip_C_Hips', [0], values)
    expect(track.name).toBe('J_Bip_C_Hips.position')
    expect(track).toBeInstanceOf(THREE.VectorKeyframeTrack)
    expect(Array.from(track.values)).toEqual([1, 2, 3])
  })
})
