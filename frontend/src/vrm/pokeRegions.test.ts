import { describe, expect, it } from 'vitest'
import * as THREE from 'three'
import { isNameInList, POKE_ACTION_NAMES, resolvePokedRegion } from './pokeRegions'

describe('isNameInList', () => {
  it('matches case-insensitively', () => {
    expect(isNameInList('action greeting 1', POKE_ACTION_NAMES)).toBe(true)
    expect(isNameInList('ACTION GREETING 1', POKE_ACTION_NAMES)).toBe(true)
  })

  it('trims whitespace before comparing', () => {
    expect(isNameInList('  Action Greeting 1  ', POKE_ACTION_NAMES)).toBe(true)
  })

  it('returns false for a name not in the list', () => {
    expect(isNameInList('Backflip', POKE_ACTION_NAMES)).toBe(false)
  })
})

const buildFakeVrm = (bonePositions: Record<string, [number, number, number]>) => {
  const root = new THREE.Object3D()
  const bones: Record<string, THREE.Object3D> = {}
  for (const [name, pos] of Object.entries(bonePositions)) {
    const bone = new THREE.Object3D()
    bone.position.set(...pos)
    bones[name] = bone
    root.add(bone)
  }
  root.updateMatrixWorld(true)
  return {
    humanoid: {
      getNormalizedBoneNode: (name: string) => bones[name] ?? null,
    },
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
  } as any
}

describe('resolvePokedRegion', () => {
  it('resolves to "head" when the hit point is near the head bone', () => {
    const vrm = buildFakeVrm({ head: [0, 1.6, 0], chest: [0, 1.3, 0] })
    const scratch = new THREE.Vector3()
    const hit = new THREE.Vector3(0, 1.61, 0.02)
    expect(resolvePokedRegion(vrm, hit, scratch)).toBe('head')
  })

  it('resolves to "chest" when nearest to a chest-mapped bone', () => {
    const vrm = buildFakeVrm({ head: [0, 1.6, 0], chest: [0, 1.3, 0] })
    const scratch = new THREE.Vector3()
    const hit = new THREE.Vector3(0, 1.31, 0)
    expect(resolvePokedRegion(vrm, hit, scratch)).toBe('chest')
  })

  it('returns undefined when the hit is farther than the max distance from every mapped bone', () => {
    const vrm = buildFakeVrm({ head: [0, 1.6, 0] })
    const scratch = new THREE.Vector3()
    const hit = new THREE.Vector3(5, 5, 5)
    expect(resolvePokedRegion(vrm, hit, scratch)).toBeUndefined()
  })

  it('returns undefined when the VRM has no humanoid bones mapped', () => {
    const vrm = { humanoid: { getNormalizedBoneNode: () => null } }
    const scratch = new THREE.Vector3()
    const hit = new THREE.Vector3(0, 0, 0)
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    expect(resolvePokedRegion(vrm as any, hit, scratch)).toBeUndefined()
  })

  it('picks the closest region when two candidate bones are both within range', () => {
    const vrm = buildFakeVrm({ head: [0, 1.6, 0], neck: [0, 1.5, 0], chest: [0, 1.3, 0] })
    const scratch = new THREE.Vector3()
    // Closer to neck (also mapped to 'head' region) than chest.
    const hit = new THREE.Vector3(0, 1.51, 0)
    expect(resolvePokedRegion(vrm, hit, scratch)).toBe('head')
  })
})
