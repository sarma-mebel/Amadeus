/**
 * Per-region poke reactions, extracted from VRMViewer's pointer-up handler.
 * The raycast hit point is matched to the nearest mapped humanoid bone, and
 * the region's vetted hitarea clip plays instead of a generic reaction.
 * Regions without a vetted clip (legs, feet, hips) and checkouts without the
 * community pack fall back to POKE_ACTION_NAMES. Bone names are VRM
 * humanoid names.
 */

import type * as THREE from 'three'
import type { VRM, VRMHumanBoneName } from '@pixiv/three-vrm'

export const POKE_REGION_CLIPS: Array<{ region: string; clip: string; bones: string[] }> = [
  { region: 'head', clip: 'Hitarea Head', bones: ['head', 'neck'] },
  { region: 'chest', clip: 'Hitarea Chest', bones: ['chest', 'upperChest', 'spine'] },
  {
    region: 'hands',
    // Surprise 2 (startled pull-back), not Hitarea Hands: the hitarea clip
    // extends an arm in a limp 6.6s handshake — vetted out 2026-07-12.
    clip: 'Surprise 2',
    bones: ['leftHand', 'rightHand', 'leftLowerArm', 'rightLowerArm'],
  },
]
// A hit farther than this from every mapped bone is "the body in general".
export const POKE_REGION_MAX_DISTANCE_M = 0.35

// Clips allowed to play as a poke reaction. These are one-shot reactions,
// not idle loops, so a wave/pose cannot repeat endlessly while idle.
// audit 2026-07-19: 'Greeting' starts turned away and root-walks into the
// camera — replaced with 'Action Greeting 1' (clean stationary wave).
// 2026-07-20: 'Thinking Pose' dropped (clip disabled in the manifest).
export const POKE_ACTION_NAMES = ['Action Greeting 1', 'Peace Sign']

export const isNameInList = (name: string, list: string[]): boolean =>
  list.some((candidate) => candidate.toLowerCase() === name.trim().toLowerCase())

/**
 * Resolves which POKE_REGION_CLIPS region a raycast hit point belongs to, by
 * finding the nearest mapped humanoid bone within POKE_REGION_MAX_DISTANCE_M.
 * `scratch` is a caller-owned Vector3 reused to avoid a per-poke allocation
 * (mirrors the pokeRegionScratch pattern in the original inline code).
 * Returns undefined when no mapped bone is close enough — "the body in
 * general", not a specific region.
 */
export const resolvePokedRegion = (
  vrm: VRM,
  hitPoint: THREE.Vector3,
  scratch: THREE.Vector3
): string | undefined => {
  let pokedRegion: string | undefined
  let bestDistance = POKE_REGION_MAX_DISTANCE_M
  for (const region of POKE_REGION_CLIPS) {
    for (const boneName of region.bones) {
      const bone = vrm.humanoid?.getNormalizedBoneNode(boneName as VRMHumanBoneName)
      if (!bone) continue
      const distance = bone.getWorldPosition(scratch).distanceTo(hitPoint)
      if (distance < bestDistance) {
        bestDistance = distance
        pokedRegion = region.region
      }
    }
  }
  return pokedRegion
}
