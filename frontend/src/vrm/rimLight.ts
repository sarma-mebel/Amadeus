/**
 * MToon parametric rim lighting: a subtle tinted edge-light keyed to the
 * active environment's lighting palette, applied on VRM load and re-applied
 * on every environment swap (see VRMViewer.tsx's reframeForEnvironmentRef
 * sibling wiring — this is called from the same two spots: the load
 * callback and the environment effect, right after repositionForEnvironment).
 *
 * "Environment lighting palette" here is the scene rig's actual light
 * colors (ambientLight/mainLight/fillLight), not a per-environment table in
 * environments.ts — no such table exists (environments.ts only carries
 * audio/reverb tuning), and the rig's lights are already the live source of
 * truth for "what color is this scene lit" (time-of-day driven, see
 * timeOfDayLighting.ts). Deriving from the rig also means a future
 * per-hour lighting change (currently pinned to the flat 'day' preset)
 * automatically reaches the rim tint with no changes here.
 *
 * RESPECT AUTHORED VALUES: a VRM whose MToon material already ships a
 * non-black parametricRimColorFactor made a deliberate artistic choice
 * (community models sometimes do) — this module records it and never
 * overwrites it. Applied/skipped state is tracked per-material (a WeakMap)
 * so a later environment swap can tell which materials it owns vs. which
 * it left alone, and re-derives only the ones it owns.
 */

import * as THREE from 'three'
import type { VRM } from '@pixiv/three-vrm'

/** Structural check — avoids an import-time dependency on
 * @pixiv/three-vrm-materials-mtoon just for a type guard. Every
 * MToonMaterial sets isMToonMaterial = true (see the package's
 * MToonMaterial class); parametricRimColorFactor etc. are the same
 * package's uniforms, always present together. */
export interface RimCapableMaterial extends THREE.Material {
  isMToonMaterial?: true
  parametricRimColorFactor: THREE.Color
  parametricRimFresnelPowerFactor: number
  parametricRimLiftFactor: number
}

export const isRimCapableMaterial = (material: THREE.Material): material is RimCapableMaterial =>
  (material as RimCapableMaterial).isMToonMaterial === true

/** Fallback rim: soft warm white at low intensity when no light data is
 * available (or all lights are literally black, which would derive a
 * black/invisible rim otherwise). ~0.12-0.18 luminance per the task spec. */
const FALLBACK_RIM_COLOR = new THREE.Color(0.16, 0.15, 0.13)

export const PARAMETRIC_RIM_FRESNEL_POWER = 5.0
export const PARAMETRIC_RIM_LIFT = 0

/** How much of the derived tint actually reaches the rim — kept well under
 * 1 so the effect stays "subtle edge light", not a colored glow outline. */
const RIM_INTENSITY = 0.16

/** Derive a soft tinted rim color from the scene's current key + fill light
 * colors. Pure function (no THREE.Object3D access) so it's unit-testable
 * without a WebGL context — takes plain {r,g,b} in [0,1], not THREE.Color,
 * so tests don't need a real THREE.Color instance either. */
export const deriveRimColor = (
  lights: { main?: { r: number; g: number; b: number }; fill?: { r: number; g: number; b: number } } | null
): THREE.Color => {
  if (!lights || (!lights.main && !lights.fill)) {
    return FALLBACK_RIM_COLOR.clone()
  }
  const main = lights.main ?? lights.fill!
  const fill = lights.fill ?? lights.main!
  // Blend key toward fill (70/30) — the rim reads as "kissed by the back/
  // fill light", which is the classic rim-light role, while still nodding
  // to the overall scene temperature via the key light.
  const r = main.r * 0.3 + fill.r * 0.7
  const g = main.g * 0.3 + fill.g * 0.7
  const b = main.b * 0.3 + fill.b * 0.7
  const luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
  if (luminance <= 0.001) {
    return FALLBACK_RIM_COLOR.clone()
  }
  // Normalize to the target intensity so a bright key light doesn't blow
  // the rim out, then scale by hue while holding overall brightness at
  // RIM_INTENSITY.
  const scale = RIM_INTENSITY / luminance
  return new THREE.Color(
    Math.min(1, r * scale),
    Math.min(1, g * scale),
    Math.min(1, b * scale)
  )
}

interface RimRecord {
  /** True when this module owns the material's rim (authored value was
   * black/default at first sight) — only owned materials get re-derived on
   * a later environment swap. */
  owned: boolean
}

/** Per-material bookkeeping so re-applying on an environment swap knows
 * which materials this module owns vs. which it left alone (authored rim).
 * Module-level WeakMap (not per-VRM state) is fine — materials are unique
 * objects per VRM instance, and an outfit swap discards the old VRM (and
 * therefore its materials) entirely, so there is nothing to leak or
 * conflate across avatar loads. */
const rimRecords = new WeakMap<RimCapableMaterial, RimRecord>()

/** True when a color is exactly black — the MToon spec's default for
 * parametricRimColorFactor when no rim is authored. Treated as "not
 * authored"; anything else (including a very dark but non-zero color) is
 * respected as a deliberate choice. */
const isBlack = (color: THREE.Color): boolean => color.r === 0 && color.g === 0 && color.b === 0

/** Walk the VRM's materials and apply/refresh the parametric rim on every
 * MToon material this module owns (see RESPECT AUTHORED VALUES above).
 * Safe to call multiple times (load, then every environment swap) — a
 * material recorded as "authored, skip" on first sight is skipped forever
 * for that VRM instance; a material this module owns gets its tint
 * refreshed to match the current lights every call. */
export const applyRimLighting = (
  vrm: VRM,
  lights: { main?: THREE.Color; fill?: THREE.Color } | null
): void => {
  const rimColor = deriveRimColor(
    lights
      ? {
          main: lights.main ? { r: lights.main.r, g: lights.main.g, b: lights.main.b } : undefined,
          fill: lights.fill ? { r: lights.fill.r, g: lights.fill.g, b: lights.fill.b } : undefined,
        }
      : null
  )

  vrm.scene.traverse((object) => {
    const mesh = object as THREE.Mesh
    if (!mesh.isMesh) return
    const materials = Array.isArray(mesh.material) ? mesh.material : [mesh.material]
    for (const material of materials) {
      if (!material || !isRimCapableMaterial(material)) continue
      let record = rimRecords.get(material)
      if (!record) {
        // First sight of this material: an already-non-black rim is an
        // authored artistic choice — record it and never touch it again.
        const authored = !isBlack(material.parametricRimColorFactor)
        record = { owned: !authored }
        rimRecords.set(material, record)
      }
      if (!record.owned) continue
      material.parametricRimColorFactor.copy(rimColor)
      material.parametricRimFresnelPowerFactor = PARAMETRIC_RIM_FRESNEL_POWER
      material.parametricRimLiftFactor = PARAMETRIC_RIM_LIFT
    }
  })
}
