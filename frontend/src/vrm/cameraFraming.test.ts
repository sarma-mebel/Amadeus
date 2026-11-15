import { describe, expect, it } from 'vitest'
import * as THREE from 'three'
import { computeDefaultFraming, orbitClampsForHeight, solveFramingDistance } from './cameraFraming'

describe('solveFramingDistance', () => {
  it('returns a larger distance for points farther off-axis at the same FOV/aspect', () => {
    const target = new THREE.Vector3(0, 0, 0)
    const direction = new THREE.Vector3(0, 0, 1)
    const closePoints = [new THREE.Vector3(0, 0.1, 0)]
    const farPoints = [new THREE.Vector3(0, 1, 0)]
    const closeDistance = solveFramingDistance(target, direction, closePoints, 1, 45)
    const farDistance = solveFramingDistance(target, direction, farPoints, 1, 45)
    expect(farDistance).toBeGreaterThan(closeDistance)
  })

  it('a wider FOV requires a smaller distance to fit the same points', () => {
    const target = new THREE.Vector3(0, 0, 0)
    const direction = new THREE.Vector3(0, 0, 1)
    const points = [new THREE.Vector3(0, 0.5, 0)]
    const narrowFov = solveFramingDistance(target, direction, points, 1, 20)
    const wideFov = solveFramingDistance(target, direction, points, 1, 90)
    expect(wideFov).toBeLessThan(narrowFov)
  })

  it('a wider aspect ratio (more horizontal FOV) can reduce required distance for a laterally-offset point', () => {
    const target = new THREE.Vector3(0, 0, 0)
    const direction = new THREE.Vector3(0, 0, 1)
    const points = [new THREE.Vector3(0.5, 0, 0)]
    const narrowAspect = solveFramingDistance(target, direction, points, 0.5, 45)
    const wideAspect = solveFramingDistance(target, direction, points, 3, 45)
    expect(wideAspect).toBeLessThanOrEqual(narrowAspect)
  })
})

describe('orbitClampsForHeight', () => {
  it('scales min/max distance linearly with model height', () => {
    const clamps1 = orbitClampsForHeight(1)
    const clamps2 = orbitClampsForHeight(2)
    expect(clamps2.minDistance).toBeCloseTo(clamps1.minDistance * 2)
    expect(clamps2.maxDistance).toBeCloseTo(clamps1.maxDistance * 2)
  })

  it('min distance is always less than max distance for a positive height', () => {
    const clamps = orbitClampsForHeight(1.6)
    expect(clamps.minDistance).toBeLessThan(clamps.maxDistance)
  })

  it('polar angle clamps are fixed regardless of height', () => {
    const clampsA = orbitClampsForHeight(1)
    const clampsB = orbitClampsForHeight(10)
    expect(clampsA.minPolarAngle).toBeCloseTo(clampsB.minPolarAngle)
    expect(clampsA.maxPolarAngle).toBeCloseTo(clampsB.maxPolarAngle)
    // 30deg-100deg range, sanity-checked in radians.
    expect(clampsA.minPolarAngle).toBeCloseTo(THREE.MathUtils.degToRad(30))
    expect(clampsA.maxPolarAngle).toBeCloseTo(THREE.MathUtils.degToRad(100))
  })
})

/** Builds a minimal duck-typed VRM humanoid rig sufficient for
 * computeDefaultFraming: head/hips/knee/shoulder bones at plausible
 * standing-pose world positions, plus a `scene` root for the facing
 * direction + updateMatrixWorld call. */
const buildFakeVrm = () => {
  const root = new THREE.Object3D()
  const bones: Record<string, THREE.Object3D> = {
    head: new THREE.Object3D(),
    hips: new THREE.Object3D(),
    leftLowerLeg: new THREE.Object3D(),
    leftUpperArm: new THREE.Object3D(),
    rightUpperArm: new THREE.Object3D(),
  }
  bones.head.position.set(0, 1.6, 0)
  bones.hips.position.set(0, 0.9, 0)
  bones.leftLowerLeg.position.set(0.1, 0.5, 0)
  bones.leftUpperArm.position.set(-0.2, 1.4, 0)
  bones.rightUpperArm.position.set(0.2, 1.4, 0)
  Object.values(bones).forEach((bone) => root.add(bone))

  return {
    scene: root,
    humanoid: {
      getNormalizedBoneNode: (name: string) => bones[name] ?? null,
    },
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
  } as any
}

describe('computeDefaultFraming', () => {
  it('returns null when the humanoid is missing', () => {
    const vrm = { scene: new THREE.Object3D(), humanoid: null }
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    expect(computeDefaultFraming(vrm as any, 1, 45)).toBeNull()
  })

  it('returns null when head or hips bones are missing', () => {
    const vrm = {
      scene: new THREE.Object3D(),
      humanoid: { getNormalizedBoneNode: () => null },
    }
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    expect(computeDefaultFraming(vrm as any, 1, 45)).toBeNull()
  })

  it('computes a target near chest height, between hips and head', () => {
    const vrm = buildFakeVrm()
    const framing = computeDefaultFraming(vrm, 1, 45)
    expect(framing).not.toBeNull()
    if (!framing) return
    expect(framing.target.y).toBeGreaterThan(0.5) // above the knee
    expect(framing.target.y).toBeLessThan(1.6) // below head-top
  })

  it('produces a positive camera distance (offset length > 0)', () => {
    const vrm = buildFakeVrm()
    const framing = computeDefaultFraming(vrm, 16 / 9, 45)
    expect(framing).not.toBeNull()
    if (!framing) return
    expect(framing.offset.length()).toBeGreaterThan(0)
  })

  it('includes head-top, thigh, hips and both shoulder points in mustFramePoints', () => {
    const vrm = buildFakeVrm()
    const framing = computeDefaultFraming(vrm, 1, 45)
    expect(framing).not.toBeNull()
    if (!framing) return
    // headTop + thigh + hips + 2 shoulders = 5 points when all bones exist.
    expect(framing.mustFramePoints.length).toBe(5)
  })

  describe('compact framing mode', () => {
    it('crops at the shoulders instead of the thigh (fewer mustFramePoints)', () => {
      const vrm = buildFakeVrm()
      const framing = computeDefaultFraming(vrm, 1, 45, 'compact')
      expect(framing).not.toBeNull()
      if (!framing) return
      // headTop + bottom(shoulder-derived) + 2 shoulders = 4, vs 5 for medium.
      expect(framing.mustFramePoints.length).toBe(4)
    })

    it('produces a chest-height target higher up than the medium shot', () => {
      const vrm = buildFakeVrm()
      const medium = computeDefaultFraming(vrm, 1, 45, 'medium')
      const compact = computeDefaultFraming(vrm, 1, 45, 'compact')
      expect(medium).not.toBeNull()
      expect(compact).not.toBeNull()
      if (!medium || !compact) return
      expect(compact.target.y).toBeGreaterThan(medium.target.y)
    })

    it('reports the same modelHeight as medium mode (orbit clamps unaffected by preset)', () => {
      const vrm = buildFakeVrm()
      const medium = computeDefaultFraming(vrm, 1, 45, 'medium')
      const compact = computeDefaultFraming(vrm, 1, 45, 'compact')
      expect(medium).not.toBeNull()
      expect(compact).not.toBeNull()
      if (!medium || !compact) return
      expect(compact.modelHeight).toBeCloseTo(medium.modelHeight)
    })

    it('falls back to the medium bottom line when no shoulder bones exist', () => {
      const root = new THREE.Object3D()
      const bones: Record<string, THREE.Object3D> = {
        head: new THREE.Object3D(),
        hips: new THREE.Object3D(),
      }
      bones.head.position.set(0, 1.6, 0)
      bones.hips.position.set(0, 0.9, 0)
      Object.values(bones).forEach((bone) => root.add(bone))
      const vrm = {
        scene: root,
        humanoid: { getNormalizedBoneNode: (name: string) => bones[name] ?? null },
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
      } as any
      const framing = computeDefaultFraming(vrm, 1, 45, 'compact')
      expect(framing).not.toBeNull()
    })

    it('produces a positive camera distance', () => {
      const vrm = buildFakeVrm()
      const framing = computeDefaultFraming(vrm, 16 / 9, 45, 'compact')
      expect(framing).not.toBeNull()
      if (!framing) return
      expect(framing.offset.length()).toBeGreaterThan(0)
    })
  })
})
