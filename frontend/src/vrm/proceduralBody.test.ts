// @vitest-environment jsdom
import { beforeEach, describe, expect, it } from 'vitest'
import * as THREE from 'three'
import type { VRM } from '@pixiv/three-vrm'
import { createAdditiveStack, createProceduralBody } from './proceduralBody'

const quatCloseTo = (a: THREE.Quaternion, b: THREE.Quaternion, epsilon = 1e-6) =>
  Math.abs(a.x - b.x) < epsilon &&
  Math.abs(a.y - b.y) < epsilon &&
  Math.abs(a.z - b.z) < epsilon &&
  Math.abs(a.w - b.w) < epsilon

describe('createAdditiveStack', () => {
  it('undo restores the exact base pose after an apply', () => {
    const stack = createAdditiveStack()
    const bone = new THREE.Object3D()
    // Non-identity base pose so the restore is meaningful.
    bone.quaternion.setFromAxisAngle(new THREE.Vector3(0, 1, 0), 0.7)
    const base = bone.quaternion.clone()

    const additive = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(1, 0, 0), 0.2)
    stack.apply(bone, additive)
    expect(quatCloseTo(bone.quaternion, base)).toBe(false)

    stack.undo()
    expect(quatCloseTo(bone.quaternion, base)).toBe(true)
  })

  it('folds two additives on the same bone into one undo', () => {
    const stack = createAdditiveStack()
    const bone = new THREE.Object3D()
    bone.quaternion.setFromAxisAngle(new THREE.Vector3(1, 0, 0), -0.4)
    const base = bone.quaternion.clone()

    stack.apply(bone, new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(1, 0, 0), 0.15))
    stack.apply(bone, new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 0, 1), 0.1))

    stack.undo()
    expect(quatCloseTo(bone.quaternion, base)).toBe(true)
  })

  it('undo is idempotent (a second undo with an empty stack is a no-op)', () => {
    const stack = createAdditiveStack()
    const bone = new THREE.Object3D()
    const base = bone.quaternion.clone()
    stack.apply(bone, new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(1, 0, 0), 0.3))
    stack.undo()
    stack.undo()
    expect(quatCloseTo(bone.quaternion, base)).toBe(true)
  })
})

describe('createProceduralBody', () => {
  let bones: Record<string, THREE.Object3D>
  let vrm: VRM

  beforeEach(() => {
    window.localStorage.clear()
    bones = {
      head: new THREE.Object3D(),
      chest: new THREE.Object3D(),
      upperChest: new THREE.Object3D(),
      spine: new THREE.Object3D(),
    }
    vrm = {
      humanoid: {
        getNormalizedBoneNode: (name: string) => bones[name] ?? null,
      },
      expressionManager: null,
      lookAt: null,
    } as unknown as VRM
  })

  const makeBody = () =>
    createProceduralBody({
      getVrm: () => vrm,
      getRawSpeechEnvelope: () => 0,
      getBrowChannel: () => null,
      getGazeProxy: () => null, // head aim inactive without a proxy
      isGazeAverted: () => false,
    })

  it('breathing rotates the torso bones each frame and undoAdditives restores them', () => {
    const body = makeBody()
    const chestBase = bones.chest.quaternion.clone()
    const spineBase = bones.spine.quaternion.clone()

    body.update(0.1)
    expect(quatCloseTo(bones.chest.quaternion, chestBase)).toBe(false)
    expect(quatCloseTo(bones.spine.quaternion, spineBase)).toBe(false)

    body.undoAdditives()
    expect(quatCloseTo(bones.chest.quaternion, chestBase)).toBe(true)
    expect(quatCloseTo(bones.spine.quaternion, spineBase)).toBe(true)
  })

  it('a queued listening nod moves the head, then the undo pass restores base', () => {
    const body = makeBody()
    const headBase = bones.head.quaternion.clone()

    body.queueListeningNod()
    body.update(0.05)
    expect(quatCloseTo(bones.head.quaternion, headBase)).toBe(false)

    body.undoAdditives()
    expect(quatCloseTo(bones.head.quaternion, headBase)).toBe(true)
  })

  it('repeated update/undo cycles do not compound rotations on undriven bones', () => {
    // The exact failure mode the additive stack exists to prevent: a bone
    // the mixer never writes must return to base every frame, not
    // integrate the additive forever.
    const body = makeBody()
    const headBase = bones.head.quaternion.clone()
    body.queueListeningNod()
    for (let i = 0; i < 50; i += 1) {
      body.undoAdditives()
      body.update(0.016)
    }
    body.undoAdditives()
    expect(quatCloseTo(bones.head.quaternion, headBase)).toBe(true)
  })

  it('does nothing without a loaded VRM', () => {
    vrm = null as unknown as VRM
    const body = makeBody()
    expect(() => {
      body.update(0.1)
      body.undoAdditives()
    }).not.toThrow()
  })

  it('weight shift eventually rolls the hips with a counter-rotated spine', () => {
    bones.hips = new THREE.Object3D()
    const body = makeBody()
    // Run long enough (well past the 10-30s retarget window) that at least
    // one non-center stance has been eased into. Breathing never touches
    // hips, so any hips rotation must come from the weight shift.
    // Threshold well above the tiny Z cross-terms breathing's X*Y
    // composition can put on spine, so the sign comparison below is
    // unambiguously about the weight shift.
    let moved = false
    for (let i = 0; i < 8000 && !moved; i += 1) {
      body.undoAdditives()
      body.update(0.032) // ~4 minutes of simulated frames max
      moved = Math.abs(2 * Math.atan2(bones.hips.quaternion.z, bones.hips.quaternion.w)) > 0.005
    }
    expect(moved).toBe(true)
    // Counter-rotation: spine carries some opposite-signed Z component on
    // the same frame (breathing contributes X/Y there, not Z).
    const hipsZ = bones.hips.quaternion.z
    const spineZ = bones.spine.quaternion.z
    expect(Math.sign(spineZ)).toBe(-Math.sign(hipsZ))

    body.undoAdditives()
    expect(quatCloseTo(bones.hips.quaternion, new THREE.Quaternion())).toBe(true)
  })

  describe('queueBeatGesture', () => {
    it('question beat rolls the head and releases cleanly', () => {
      const body = makeBody()
      const headBase = bones.head.quaternion.clone()
      body.queueBeatGesture('question')
      body.update(0.3) // mid-envelope
      expect(quatCloseTo(bones.head.quaternion, headBase)).toBe(false)
      // Question tilt is a Z-roll (with slight X pitch), never a yaw.
      expect(Math.abs(bones.head.quaternion.z)).toBeGreaterThan(1e-4)

      // Envelope completes; the head returns to resting on the base pose.
      for (let i = 0; i < 60; i += 1) {
        body.undoAdditives()
        body.update(0.032)
      }
      body.undoAdditives()
      expect(quatCloseTo(bones.head.quaternion, headBase)).toBe(true)
    })

    it('negation beat shakes about the yaw axis only', () => {
      const body = makeBody()
      body.queueBeatGesture('negation')
      body.update(0.1)
      expect(Math.abs(bones.head.quaternion.y)).toBeGreaterThan(1e-4)
      expect(Math.abs(bones.head.quaternion.x)).toBeLessThan(1e-6)
      expect(Math.abs(bones.head.quaternion.z)).toBeLessThan(1e-6)
    })

    it('emphasis beat nods through the shared activeNod machinery', () => {
      const body = makeBody()
      const headBase = bones.head.quaternion.clone()
      body.queueBeatGesture('emphasis')
      body.update(0.1)
      expect(quatCloseTo(bones.head.quaternion, headBase)).toBe(false)
      // Nod = X pitch component.
      expect(Math.abs(bones.head.quaternion.x)).toBeGreaterThan(1e-4)
    })

    it('rate-limits beats: a second request inside the gap is dropped', () => {
      const body = makeBody()
      body.queueBeatGesture('question')
      // Let the question envelope fully finish...
      for (let i = 0; i < 40; i += 1) {
        body.undoAdditives()
        body.update(0.032)
      }
      body.undoAdditives()
      const headBase = bones.head.quaternion.clone()
      // ...then request another beat immediately: still inside the 2.5s
      // min gap (performance.now barely moved), so the head must stay put.
      body.queueBeatGesture('negation')
      body.update(0.1)
      expect(quatCloseTo(bones.head.quaternion, headBase)).toBe(true)
    })
  })
})
