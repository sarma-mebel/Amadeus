// @vitest-environment jsdom
import { describe, expect, it } from 'vitest'
import * as THREE from 'three'
import type { VRM } from '@pixiv/three-vrm'
import {
  INERTIALIZATION_STORAGE_KEY,
  createInertializer,
  decaySpringVector,
  halflifeForFade,
  loadInertializationEnabled,
  saveInertializationEnabled,
} from './inertialization'

const DT = 1 / 60

describe('decaySpringVector', () => {
  it('decays an offset to (near) zero and never diverges', () => {
    const offset = new THREE.Vector3(0.5, -0.3, 0.2)
    const velocity = new THREE.Vector3()
    const start = offset.length()
    for (let i = 0; i < 120; i += 1) decaySpringVector(offset, velocity, 0.12, DT)
    // 2 seconds ≈ 16 halflifes: effectively gone.
    expect(offset.length()).toBeLessThan(start * 1e-3)
    expect(velocity.length()).toBeLessThan(1e-3)
  })

  it('an initial velocity carries the offset forward before it decays', () => {
    const offset = new THREE.Vector3(0.2, 0, 0)
    const velocity = new THREE.Vector3(2, 0, 0)
    decaySpringVector(offset, velocity, 0.15, DT)
    // First step: velocity pushes the offset up — momentum is preserved
    // through the transition instead of dying instantly (the whole point
    // vs a crossfade).
    expect(offset.x).toBeGreaterThan(0.2)
    for (let i = 0; i < 300; i += 1) decaySpringVector(offset, velocity, 0.15, DT)
    expect(Math.abs(offset.x)).toBeLessThan(1e-3)
  })

  it('is dt-robust: a large idle-throttle step still lands near the same curve', () => {
    const fine = { offset: new THREE.Vector3(0.4, 0, 0), velocity: new THREE.Vector3(1, 0, 0) }
    const coarse = { offset: new THREE.Vector3(0.4, 0, 0), velocity: new THREE.Vector3(1, 0, 0) }
    for (let i = 0; i < 8; i += 1) decaySpringVector(fine.offset, fine.velocity, 0.12, DT)
    decaySpringVector(coarse.offset, coarse.velocity, 0.12, 8 * DT)
    // Closed-form, so one 8x step equals eight small steps exactly.
    expect(coarse.offset.x).toBeCloseTo(fine.offset.x, 10)
    expect(coarse.velocity.x).toBeCloseTo(fine.velocity.x, 10)
  })
})

describe('halflifeForFade', () => {
  it('maps fade durations into the clamped halflife range', () => {
    expect(halflifeForFade(0.6)).toBeCloseTo(0.18)
    expect(halflifeForFade(0.05)).toBeCloseTo(0.05)
    expect(halflifeForFade(5)).toBeCloseTo(0.3)
  })
})

// Minimal fake VRM: three rotated bones plus the hips position — enough to
// exercise capture, decay, undo, and history without a real model.
const makeFakeVrm = () => {
  const nodes: Record<string, THREE.Object3D> = {
    hips: new THREE.Object3D(),
    spine: new THREE.Object3D(),
    chest: new THREE.Object3D(),
  }
  const vrm = {
    humanoid: {
      getNormalizedBoneNode: (name: string) => nodes[name] ?? null,
    },
  } as unknown as VRM
  return { vrm, nodes }
}

const setPose = (
  nodes: Record<string, THREE.Object3D>,
  quat: THREE.Quaternion,
  hipsPos: THREE.Vector3
) => {
  for (const node of Object.values(nodes)) node.quaternion.copy(quat)
  nodes.hips.position.copy(hipsPos)
}

/** One render frame as VRMViewer runs it: undo -> "mixer writes" -> update. */
const frame = (
  inertializer: ReturnType<typeof createInertializer>,
  nodes: Record<string, THREE.Object3D>,
  quat: THREE.Quaternion,
  hipsPos: THREE.Vector3,
  dt = DT
) => {
  inertializer.undo()
  setPose(nodes, quat, hipsPos)
  inertializer.update(dt)
}

const quatZ = (angle: number) => new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 0, 1), angle)

describe('createInertializer', () => {
  it('declines a transition until the pose history is warm', () => {
    const { vrm, nodes } = makeFakeVrm()
    const inertializer = createInertializer({ getVrm: () => vrm })
    expect(inertializer.requestTransition()).toBe(false)
    frame(inertializer, nodes, quatZ(0.4), new THREE.Vector3(0, 1, 0))
    expect(inertializer.requestTransition()).toBe(false)
    frame(inertializer, nodes, quatZ(0.4), new THREE.Vector3(0, 1, 0))
    expect(inertializer.requestTransition()).toBe(true)
  })

  it('keeps the visible pose continuous across a hard clip cut, then converges to the new clip', () => {
    const { vrm, nodes } = makeFakeVrm()
    const inertializer = createInertializer({ getVrm: () => vrm })
    const oldQuat = quatZ(0.5)
    const newQuat = quatZ(-0.3)
    const oldPos = new THREE.Vector3(0, 1, 0)
    const newPos = new THREE.Vector3(0.15, 0.9, 0)
    for (let i = 0; i < 5; i += 1) frame(inertializer, nodes, oldQuat, oldPos)

    expect(inertializer.requestTransition({ fadeDuration: 0.5 })).toBe(true)
    frame(inertializer, nodes, newQuat, newPos)
    // First frame after the cut: visible pose is still (about) the old
    // pose, not the new clip's — the 0.8 rad snap is absorbed.
    expect(nodes.chest.quaternion.angleTo(oldQuat)).toBeLessThan(0.05)
    expect(nodes.hips.position.distanceTo(oldPos)).toBeLessThan(0.02)
    expect(inertializer.isActive()).toBe(true)

    for (let i = 0; i < 240; i += 1) frame(inertializer, nodes, newQuat, newPos)
    // 4 seconds later the offset has fully decayed onto the new clip and
    // the springs culled themselves.
    expect(nodes.chest.quaternion.angleTo(newQuat)).toBeLessThan(0.001)
    expect(nodes.hips.position.distanceTo(newPos)).toBeLessThan(0.001)
    expect(inertializer.isActive()).toBe(false)
  })

  it('carries pre-cut momentum through the transition', () => {
    const { vrm, nodes } = makeFakeVrm()
    const inertializer = createInertializer({ getVrm: () => vrm })
    // Steadily rotating source motion (1 rad/s), cut to a static identity pose.
    const pos = new THREE.Vector3()
    for (let i = 0; i < 30; i += 1) frame(inertializer, nodes, quatZ(0.2 + i * DT), pos)
    const lastAngle = 0.2 + 29 * DT
    expect(inertializer.requestTransition({ fadeDuration: 0.4 })).toBe(true)
    frame(inertializer, nodes, quatZ(0), pos)
    const angleAfterCut = 2 * Math.atan2(nodes.spine.quaternion.z, nodes.spine.quaternion.w)
    // The visible rotation keeps moving in the source's direction (angle
    // still increasing past where it was) before the spring pulls it back.
    expect(angleAfterCut).toBeGreaterThan(lastAngle)
    for (let i = 0; i < 300; i += 1) frame(inertializer, nodes, quatZ(0), pos)
    expect(nodes.spine.quaternion.angleTo(quatZ(0))).toBeLessThan(0.001)
  })

  it('undo restores the raw mixer pose exactly', () => {
    const { vrm, nodes } = makeFakeVrm()
    const inertializer = createInertializer({ getVrm: () => vrm })
    const oldQuat = quatZ(0.6)
    const newQuat = quatZ(0.1)
    const pos = new THREE.Vector3(0, 1, 0)
    for (let i = 0; i < 3; i += 1) frame(inertializer, nodes, oldQuat, pos)
    inertializer.requestTransition()
    frame(inertializer, nodes, newQuat, pos)
    expect(nodes.chest.quaternion.angleTo(newQuat)).toBeGreaterThan(0.1) // offset applied
    inertializer.undo()
    expect(nodes.chest.quaternion.angleTo(newQuat)).toBeLessThan(1e-6)
    expect(nodes.hips.position.distanceTo(pos)).toBeLessThan(1e-6)
  })

  it('defers capture until the new action has evaluated, and expires a dead request', () => {
    const { vrm, nodes } = makeFakeVrm()
    const inertializer = createInertializer({ getVrm: () => vrm })
    const oldQuat = quatZ(0.5)
    const newQuat = quatZ(-0.5)
    const pos = new THREE.Vector3()
    for (let i = 0; i < 3; i += 1) frame(inertializer, nodes, oldQuat, pos)

    // hasEvaluated false: the mixer is still on the old pose this frame
    // (a 'finished'-handler switch) — nothing must be captured yet.
    let evaluated = false
    inertializer.requestTransition({ hasEvaluated: () => evaluated })
    frame(inertializer, nodes, oldQuat, pos)
    expect(inertializer.isActive()).toBe(false)
    evaluated = true
    frame(inertializer, nodes, newQuat, pos)
    // Captured against the still-old history: visible pose stays continuous.
    expect(nodes.chest.quaternion.angleTo(oldQuat)).toBeLessThan(0.05)

    // A request whose action never evaluates expires instead of capturing
    // against a stale pose much later.
    for (let i = 0; i < 240; i += 1) frame(inertializer, nodes, newQuat, pos)
    inertializer.requestTransition({ hasEvaluated: () => false })
    for (let i = 0; i < 5; i += 1) frame(inertializer, nodes, newQuat, pos)
    expect(inertializer.isActive()).toBe(false)
  })
})

describe('inertialization toggle', () => {
  it('defaults off (pending vetting) and round-trips', () => {
    window.localStorage.removeItem(INERTIALIZATION_STORAGE_KEY)
    expect(loadInertializationEnabled()).toBe(false)
    saveInertializationEnabled(true)
    expect(loadInertializationEnabled()).toBe(true)
    saveInertializationEnabled(false)
    expect(loadInertializationEnabled()).toBe(false)
  })
})
