import { describe, expect, it } from 'vitest'
import * as THREE from 'three'
import type { VRM } from '@pixiv/three-vrm'
import { applyRelaxedFingerPose, FINGER_NOISE_AMP } from './fingerPose'

const makeVrm = () => {
  const nodes = new Map<string, THREE.Object3D>()
  const vrm = {
    humanoid: {
      getNormalizedBoneNode: (name: string) => {
        let node = nodes.get(name)
        if (!node) {
          node = new THREE.Object3D()
          nodes.set(name, node)
        }
        return node
      },
    },
  } as unknown as VRM
  return { vrm, nodes }
}

/** Signed rotation angle about Z for a setFromAxisAngle(Z, θ) quaternion. */
const zAngleOf = (node: THREE.Object3D): number => 2 * Math.atan2(node.quaternion.z, node.quaternion.w)

describe('applyRelaxedFingerPose', () => {
  it('without a time argument produces the original static pose', () => {
    const { vrm, nodes } = makeVrm()
    applyRelaxedFingerPose(vrm)
    const first = zAngleOf(nodes.get('leftIndexProximal')!)
    applyRelaxedFingerPose(vrm)
    expect(zAngleOf(nodes.get('leftIndexProximal')!)).toBe(first)
    expect(first).toBeCloseTo(-0.35) // left-hand curl is negative about Z
  })

  it('with time the curl breathes around the base angle within the noise bound', () => {
    const { vrm, nodes } = makeVrm()
    const angles: number[] = []
    for (let t = 0; t < 12; t += 0.25) {
      applyRelaxedFingerPose(vrm, t)
      angles.push(zAngleOf(nodes.get('leftIndexProximal')!))
    }
    const min = Math.min(...angles)
    const max = Math.max(...angles)
    expect(max).toBeGreaterThan(min) // it moves
    // Bounded by base * (1 ± amp): base is -0.35, so all samples sit in
    // [-0.35*(1+amp), -0.35*(1-amp)].
    expect(min).toBeGreaterThanOrEqual(-0.35 * (1 + FINGER_NOISE_AMP) - 1e-9)
    expect(max).toBeLessThanOrEqual(-0.35 * (1 - FINGER_NOISE_AMP) + 1e-9)
  })

  it('fingers drift out of phase with each other (no synchronized pump)', () => {
    const { vrm, nodes } = makeVrm()
    // Normalized modulation (angle / base) per finger over time; if fingers
    // shared a phase these series would be identical.
    const seriesFor = (bone: string, base: number): number[] => {
      const out: number[] = []
      for (let t = 0; t < 12; t += 0.5) {
        applyRelaxedFingerPose(vrm, t)
        out.push(zAngleOf(nodes.get(bone)!) / base)
      }
      return out
    }
    const index = seriesFor('leftIndexProximal', -0.35)
    const ring = seriesFor('leftRingProximal', -0.35)
    const maxDiff = Math.max(...index.map((v, i) => Math.abs(v - ring[i])))
    expect(maxDiff).toBeGreaterThan(FINGER_NOISE_AMP * 0.5)
  })

  it('segments of the same finger flex together (shared phase)', () => {
    const { vrm, nodes } = makeVrm()
    applyRelaxedFingerPose(vrm, 3.7)
    const proximal = zAngleOf(nodes.get('leftIndexProximal')!) / -0.35
    const distal = zAngleOf(nodes.get('leftIndexDistal')!) / -0.22
    expect(proximal).toBeCloseTo(distal, 6)
  })

  it('no-ops cleanly on a rig without a humanoid', () => {
    const vrm = { humanoid: null } as unknown as VRM
    expect(() => applyRelaxedFingerPose(vrm, 1)).not.toThrow()
  })
})
