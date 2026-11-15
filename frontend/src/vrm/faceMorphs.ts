/**
 * Face channels: a uniform get/set handle over a facial control that may be
 * backed by either a VRM expression or a raw mesh morph target.
 *
 * Why this exists: the blush beat and the speech brow-raise used to discover
 * their control by regex over the VRM's *expression* list only
 * (`/blush|embarrass|shy/i`, `/brow|eyebrow/i`). A 2026-07-26 audit of all 23
 * bundled models found that NO model in the fleet names an expression that
 * way — both features had been silent no-ops everywhere, forever. But the
 * flagship VRoid-rigged models DO carry the morphs, just unmapped to any
 * expression (Kurisu's face mesh has Fcl_BRW_Surprised, Fcl_BRW_Fun, etc.).
 * So discovery now falls back from the expression map to the meshes' own
 * morph target dictionaries, and callers drive whichever backing was found
 * through the same tiny FaceChannel interface.
 */

import * as THREE from 'three'
import type { VRM } from '@pixiv/three-vrm'

export interface FaceChannel {
  /** Human-readable name of the backing expression/morph (for logs). */
  name: string
  get: () => number
  set: (value: number) => void
  /** Morph-backed channels only: re-assert the last set value AFTER
   * vrm.update(). three-vrm's expression manager zeroes every morph bound
   * to ANY expression at the top of its update pass, so a direct
   * morphTargetInfluences write made before vrm.update() is wiped when the
   * morph happens to be expression-bound. Re-asserting max(current, last)
   * after the update survives the wipe without ever stomping DOWN a value
   * an expression legitimately applied (e.g. a full 'surprised' face using
   * the same brow morph at 1.0 wins over a 0.3 speech raise). Callers that
   * run before vrm.update() should invoke this once after it; no-op cheap
   * while the channel value is 0. */
  postUpdate?: () => void
}

/** Brow-raise candidates, in priority order. VRoid rigs name their raised-
 * brow morph Fcl_BRW_Surprised (Fcl_BRW_Fun is the softer pleasant arch);
 * generic rigs get the loose brow-up patterns. Deliberately avoids matching
 * angry/sorrow brow shapes — a *raise* driven by speech amplitude must not
 * read as a scowl. */
export const BROW_RAISE_PATTERNS: RegExp[] = [
  /^Fcl_BRW_Surprised$/i,
  /brow[\s_]?(up|raise|raised|surprised)/i,
  /^Fcl_BRW_Fun$/i,
  /eyebrow(?![\s_]?(angry|sad|sorrow|down|lower))/i,
]

/** Blush candidates. No bundled model currently has any (VRoid blush is a
 * texture layer, not a morph — see the audit), but future/custom packs
 * plausibly name one of these; 照れ/てれ is the common Japanese authoring
 * name ("tere", flustered). */
export const BLUSH_PATTERNS: RegExp[] = [
  /blush/i,
  /照れ|てれ/,
  /embarrass|shy/i,
  /cheek(?![\s_]?puff)/i,
]

/** Channel over a named VRM expression (the caller has already verified the
 * name exists in the expression map). */
export const createExpressionChannel = (vrm: VRM, expressionName: string): FaceChannel => ({
  name: expressionName,
  get: () => vrm.expressionManager?.getValue(expressionName) ?? 0,
  set: (value: number) => {
    vrm.expressionManager?.setValue(expressionName, value)
  },
})

/** First expression name in the manager matching any pattern (patterns are
 * tried in order — earlier patterns win over later ones, and within one
 * pattern the manager's own enumeration order decides). */
export const findExpressionName = (
  expressionNames: string[],
  patterns: RegExp[],
): string | null => {
  for (const pattern of patterns) {
    const match = expressionNames.find((name) => pattern.test(name))
    if (match) return match
  }
  return null
}

interface MorphBinding {
  mesh: THREE.Mesh
  index: number
}

/** Channel over a raw morph target, found by pattern in the scene's meshes'
 * morphTargetDictionary and written to EVERY mesh that carries the same
 * morph name (glTF meshes with several primitives load as sibling
 * THREE.Mesh objects that each need the influence set). Returns null when
 * no morph matches — callers should degrade loudly (diagnostics counter),
 * not silently. */
export const findMorphChannel = (root: THREE.Object3D, patterns: RegExp[]): FaceChannel | null => {
  const morphNames = new Map<string, MorphBinding[]>()
  root.traverse((object) => {
    const mesh = object as THREE.Mesh
    if (!mesh.isMesh || !mesh.morphTargetDictionary || !mesh.morphTargetInfluences) return
    for (const [name, index] of Object.entries(mesh.morphTargetDictionary)) {
      const bindings = morphNames.get(name)
      if (bindings) {
        bindings.push({ mesh, index })
      } else {
        morphNames.set(name, [{ mesh, index }])
      }
    }
  })
  if (morphNames.size === 0) return null

  const allNames = [...morphNames.keys()]
  const matched = findExpressionName(allNames, patterns)
  if (!matched) return null
  const bindings = morphNames.get(matched)
  if (!bindings || bindings.length === 0) return null

  let lastValue = 0
  const applyValue = (value: number) => {
    for (const { mesh, index } of bindings) {
      if (mesh.morphTargetInfluences) {
        mesh.morphTargetInfluences[index] = value
      }
    }
  }
  return {
    name: matched,
    get: () => lastValue,
    set: (value: number) => {
      lastValue = value
      applyValue(value)
    },
    postUpdate: () => {
      if (lastValue <= 0.001) return
      for (const { mesh, index } of bindings) {
        const influences = mesh.morphTargetInfluences
        if (influences) {
          influences[index] = Math.max(influences[index], lastValue)
        }
      }
    },
  }
}

/** Expression-first, morph-fallback discovery: the shape both VRMViewer
 * call sites (blush, brow) share. */
export const findFaceChannel = (
  vrm: VRM,
  expressionPatterns: RegExp[],
  morphPatterns: RegExp[],
): FaceChannel | null => {
  if (vrm.expressionManager) {
    const expressionNames = Object.keys(vrm.expressionManager.expressionMap)
    const matched = findExpressionName(expressionNames, expressionPatterns)
    if (matched) return createExpressionChannel(vrm, matched)
  }
  return findMorphChannel(vrm.scene, morphPatterns)
}
