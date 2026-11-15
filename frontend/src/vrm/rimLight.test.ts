// @vitest-environment jsdom
import { describe, expect, it } from 'vitest'
import * as THREE from 'three'
import type { VRM } from '@pixiv/three-vrm'
import {
  applyRimLighting,
  deriveRimColor,
  isRimCapableMaterial,
  PARAMETRIC_RIM_FRESNEL_POWER,
  PARAMETRIC_RIM_LIFT,
  type RimCapableMaterial,
} from './rimLight'

describe('deriveRimColor', () => {
  it('falls back to a soft warm white when no light data is available', () => {
    const color = deriveRimColor(null)
    // Fallback spec: rgb ~0.12-0.18 luminance, warm (r >= g >= b-ish).
    expect(color.r).toBeGreaterThan(0.1)
    expect(color.r).toBeLessThan(0.2)
    expect(color.r).toBeGreaterThanOrEqual(color.b)
  })

  it('falls back when both lights are pure black (would otherwise derive black)', () => {
    const color = deriveRimColor({ main: { r: 0, g: 0, b: 0 }, fill: { r: 0, g: 0, b: 0 } })
    expect(color.r + color.g + color.b).toBeGreaterThan(0)
  })

  it('derives a non-black tinted color from white key + blue fill (the shipped rig)', () => {
    const color = deriveRimColor({ main: { r: 1, g: 1, b: 1 }, fill: { r: 0.53, g: 0.53, b: 1 } })
    expect(color.r + color.g + color.b).toBeGreaterThan(0)
    // Blue fill dominates (70% weight) -> blue channel should be the
    // brightest of the three.
    expect(color.b).toBeGreaterThanOrEqual(color.r)
  })

  it('stays subtle: no channel is ever fully saturated for a normal-brightness input', () => {
    const color = deriveRimColor({ main: { r: 1, g: 1, b: 1 }, fill: { r: 1, g: 1, b: 1 } })
    expect(color.r).toBeLessThan(0.3)
    expect(color.g).toBeLessThan(0.3)
    expect(color.b).toBeLessThan(0.3)
  })

  it('uses whichever single light is present when the other is missing', () => {
    const mainOnly = deriveRimColor({ main: { r: 1, g: 0, b: 0 } })
    expect(mainOnly.r).toBeGreaterThan(mainOnly.g)
    expect(mainOnly.r).toBeGreaterThan(mainOnly.b)
  })
})

/** Minimal stand-in for @pixiv/three-vrm-materials-mtoon's MToonMaterial —
 * exercises the traversal/authored-skip logic without needing the real
 * shader material class (which needs a WebGL-ish environment to construct
 * cleanly). Matches the RimCapableMaterial shape exactly. */
const makeMockMToonMaterial = (rimColor: THREE.Color): RimCapableMaterial => {
  const material = new THREE.MeshBasicMaterial() as unknown as RimCapableMaterial
  material.isMToonMaterial = true
  material.parametricRimColorFactor = rimColor
  material.parametricRimFresnelPowerFactor = 1
  material.parametricRimLiftFactor = 0
  return material
}

const makeMockVrm = (materials: RimCapableMaterial[]): VRM => {
  const scene = new THREE.Group()
  materials.forEach((material, index) => {
    const mesh = new THREE.Mesh(new THREE.BoxGeometry(), material as unknown as THREE.Material)
    mesh.name = `mesh-${index}`
    scene.add(mesh)
  })
  return { scene } as unknown as VRM
}

describe('isRimCapableMaterial', () => {
  it('accepts a material carrying isMToonMaterial', () => {
    const material = makeMockMToonMaterial(new THREE.Color(0, 0, 0))
    expect(isRimCapableMaterial(material)).toBe(true)
  })

  it('rejects a plain material without the MToon flag', () => {
    const plain = new THREE.MeshBasicMaterial()
    expect(isRimCapableMaterial(plain)).toBe(false)
  })
})

describe('applyRimLighting', () => {
  it('applies a derived rim to a material with the default black rim (unauthored)', () => {
    const material = makeMockMToonMaterial(new THREE.Color(0, 0, 0))
    const vrm = makeMockVrm([material])

    applyRimLighting(vrm, { main: new THREE.Color(1, 1, 1), fill: new THREE.Color(0.5, 0.5, 1) })

    expect(material.parametricRimColorFactor.r + material.parametricRimColorFactor.g + material.parametricRimColorFactor.b).toBeGreaterThan(0)
    expect(material.parametricRimFresnelPowerFactor).toBe(PARAMETRIC_RIM_FRESNEL_POWER)
    expect(material.parametricRimLiftFactor).toBe(PARAMETRIC_RIM_LIFT)
  })

  it('never overwrites an authored (non-black) rim color', () => {
    const authoredColor = new THREE.Color(0.9, 0.1, 0.1)
    const material = makeMockMToonMaterial(authoredColor.clone())
    const vrm = makeMockVrm([material])

    applyRimLighting(vrm, { main: new THREE.Color(1, 1, 1), fill: new THREE.Color(0.5, 0.5, 1) })

    expect(material.parametricRimColorFactor.r).toBeCloseTo(authoredColor.r)
    expect(material.parametricRimColorFactor.g).toBeCloseTo(authoredColor.g)
    expect(material.parametricRimColorFactor.b).toBeCloseTo(authoredColor.b)
  })

  it('does not touch fresnel/lift on an authored material either', () => {
    const material = makeMockMToonMaterial(new THREE.Color(0.9, 0.1, 0.1))
    material.parametricRimFresnelPowerFactor = 42
    material.parametricRimLiftFactor = 0.75
    const vrm = makeMockVrm([material])

    applyRimLighting(vrm, { main: new THREE.Color(1, 1, 1), fill: new THREE.Color(0.5, 0.5, 1) })

    expect(material.parametricRimFresnelPowerFactor).toBe(42)
    expect(material.parametricRimLiftFactor).toBe(0.75)
  })

  it('re-derives an owned material rim again on a later call (environment swap)', () => {
    const material = makeMockMToonMaterial(new THREE.Color(0, 0, 0))
    const vrm = makeMockVrm([material])

    applyRimLighting(vrm, { main: new THREE.Color(1, 1, 1), fill: new THREE.Color(0.5, 0.5, 1) })
    const firstColor = material.parametricRimColorFactor.clone()

    // Different lighting palette (e.g. a swap to a warmer environment).
    applyRimLighting(vrm, { main: new THREE.Color(1, 0.6, 0.3), fill: new THREE.Color(1, 0.6, 0.3) })
    const secondColor = material.parametricRimColorFactor.clone()

    expect(secondColor.equals(firstColor)).toBe(false)
  })

  it('still skips a previously-authored material on a later call', () => {
    const authoredColor = new THREE.Color(0.9, 0.1, 0.1)
    const material = makeMockMToonMaterial(authoredColor.clone())
    const vrm = makeMockVrm([material])

    applyRimLighting(vrm, { main: new THREE.Color(1, 1, 1), fill: new THREE.Color(0.5, 0.5, 1) })
    applyRimLighting(vrm, { main: new THREE.Color(1, 0.6, 0.3), fill: new THREE.Color(1, 0.6, 0.3) })

    expect(material.parametricRimColorFactor.r).toBeCloseTo(authoredColor.r)
    expect(material.parametricRimColorFactor.g).toBeCloseTo(authoredColor.g)
    expect(material.parametricRimColorFactor.b).toBeCloseTo(authoredColor.b)
  })

  it('ignores non-MToon materials entirely', () => {
    const plain = new THREE.MeshBasicMaterial({ color: 0x123456 })
    const mesh = new THREE.Mesh(new THREE.BoxGeometry(), plain)
    const scene = new THREE.Group()
    scene.add(mesh)
    const vrm = { scene } as unknown as VRM

    expect(() =>
      applyRimLighting(vrm, { main: new THREE.Color(1, 1, 1), fill: new THREE.Color(1, 1, 1) })
    ).not.toThrow()
  })

  it('handles a mesh with a material array', () => {
    const owned = makeMockMToonMaterial(new THREE.Color(0, 0, 0))
    const authored = makeMockMToonMaterial(new THREE.Color(0.2, 0.3, 0.4))
    const mesh = new THREE.Mesh(new THREE.BoxGeometry(), [
      owned as unknown as THREE.Material,
      authored as unknown as THREE.Material,
    ])
    const scene = new THREE.Group()
    scene.add(mesh)
    const vrm = { scene } as unknown as VRM

    applyRimLighting(vrm, { main: new THREE.Color(1, 1, 1), fill: new THREE.Color(0.5, 0.5, 1) })

    expect(owned.parametricRimColorFactor.r + owned.parametricRimColorFactor.g + owned.parametricRimColorFactor.b).toBeGreaterThan(0)
    expect(authored.parametricRimColorFactor.g).toBeCloseTo(0.3)
  })

  it('falls back gracefully with no light data at all', () => {
    const material = makeMockMToonMaterial(new THREE.Color(0, 0, 0))
    const vrm = makeMockVrm([material])

    applyRimLighting(vrm, null)

    expect(material.parametricRimColorFactor.r + material.parametricRimColorFactor.g + material.parametricRimColorFactor.b).toBeGreaterThan(0)
  })
})
