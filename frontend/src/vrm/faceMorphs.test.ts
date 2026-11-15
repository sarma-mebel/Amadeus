import { describe, expect, it } from 'vitest'
import * as THREE from 'three'
import type { VRM } from '@pixiv/three-vrm'
import {
  BLUSH_PATTERNS,
  BROW_RAISE_PATTERNS,
  findExpressionName,
  findFaceChannel,
  findMorphChannel,
} from './faceMorphs'

const makeMorphMesh = (dictionary: Record<string, number>) => {
  const mesh = new THREE.Mesh(new THREE.BufferGeometry(), new THREE.MeshBasicMaterial())
  mesh.morphTargetDictionary = dictionary
  mesh.morphTargetInfluences = new Array(Object.keys(dictionary).length).fill(0)
  return mesh
}

describe('findExpressionName', () => {
  it('honors pattern priority order over list order', () => {
    // Fcl_BRW_Fun appears first in the list, but the raised-brow pattern
    // outranks it.
    const names = ['Fcl_BRW_Fun', 'Fcl_BRW_Surprised', 'Fcl_BRW_Angry']
    expect(findExpressionName(names, BROW_RAISE_PATTERNS)).toBe('Fcl_BRW_Surprised')
  })

  it('never matches angry/sorrow brow shapes', () => {
    expect(findExpressionName(['Fcl_BRW_Angry', 'Fcl_BRW_Sorrow'], BROW_RAISE_PATTERNS)).toBeNull()
  })

  it('matches Japanese blush morph names', () => {
    expect(findExpressionName(['照れ'], BLUSH_PATTERNS)).toBe('照れ')
  })
})

describe('findMorphChannel', () => {
  it('finds a VRoid brow morph and writes every mesh sharing it', () => {
    const root = new THREE.Object3D()
    const faceA = makeMorphMesh({ Fcl_BRW_Surprised: 2, Fcl_MTH_A: 0 })
    const faceB = makeMorphMesh({ Fcl_BRW_Surprised: 1 })
    root.add(faceA, faceB)

    const channel = findMorphChannel(root, BROW_RAISE_PATTERNS)
    expect(channel).not.toBeNull()
    expect(channel?.name).toBe('Fcl_BRW_Surprised')

    channel?.set(0.35)
    expect(faceA.morphTargetInfluences?.[2]).toBeCloseTo(0.35)
    expect(faceB.morphTargetInfluences?.[1]).toBeCloseTo(0.35)
    expect(channel?.get()).toBeCloseTo(0.35)
  })

  it('returns null when nothing matches', () => {
    const root = new THREE.Object3D()
    root.add(makeMorphMesh({ Fcl_MTH_A: 0 }))
    expect(findMorphChannel(root, BLUSH_PATTERNS)).toBeNull()
  })

  it('postUpdate re-asserts after an expression-manager wipe without stomping higher values', () => {
    const root = new THREE.Object3D()
    const face = makeMorphMesh({ Fcl_BRW_Surprised: 0 })
    root.add(face)
    const channel = findMorphChannel(root, BROW_RAISE_PATTERNS)

    channel?.set(0.3)
    // Simulate three-vrm zeroing the bound morph during vrm.update().
    face.morphTargetInfluences![0] = 0
    channel?.postUpdate?.()
    expect(face.morphTargetInfluences?.[0]).toBeCloseTo(0.3)
    expect(channel?.get()).toBeCloseTo(0.3)

    // A stronger expression-applied value survives (max compose).
    face.morphTargetInfluences![0] = 0.9
    channel?.postUpdate?.()
    expect(face.morphTargetInfluences?.[0]).toBeCloseTo(0.9)

    // At 0 the channel leaves the morph alone entirely.
    channel?.set(0)
    face.morphTargetInfluences![0] = 0.5
    channel?.postUpdate?.()
    expect(face.morphTargetInfluences?.[0]).toBeCloseTo(0.5)
  })
})

describe('findFaceChannel', () => {
  it('prefers an expression over a morph, falls back to the morph', () => {
    const scene = new THREE.Object3D()
    scene.add(makeMorphMesh({ Fcl_BRW_Surprised: 0 }))
    const values = new Map<string, number>()
    const withExpression = {
      scene,
      expressionManager: {
        expressionMap: { blush: {}, happy: {} },
        getValue: (name: string) => values.get(name) ?? 0,
        setValue: (name: string, value: number) => {
          values.set(name, value)
        },
      },
    } as unknown as VRM

    const blush = findFaceChannel(withExpression, [/blush/i], BLUSH_PATTERNS)
    expect(blush?.name).toBe('blush')
    blush?.set(0.5)
    expect(values.get('blush')).toBeCloseTo(0.5)

    // No brow expression on the manager -> morph fallback.
    const brow = findFaceChannel(withExpression, [/brow/i], BROW_RAISE_PATTERNS)
    expect(brow?.name).toBe('Fcl_BRW_Surprised')
  })
})
