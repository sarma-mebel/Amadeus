/**
 * Pet perch pose (pet variant only), moved verbatim out of VRMViewer's
 * scene effect. The sibling-owned pet UI dispatches
 * 'kurisu-pet-perch-changed' after an edge-snap drag — detection/ownership
 * of "is she docked to a screen edge" lives entirely on that side, this
 * side only reacts to the resulting boolean.
 *
 * On perch: search the EAGER vetted clip set (kurisuBridge.vrmAnimations
 * .describe() — loaded, not disabled, not lazy; NEVER promote a lazy/
 * unvetted library clip automatically, hard repo policy) for a name-matched
 * sitting pose and loop it; none exists in the current manifest (every
 * "*Sit*"/"*Sitting*" clip is lazy), so in practice this currently always
 * falls through to the fallback: calmer stance via a temporary springbone-
 * wind reduction, restored on unperch. Doze composes on top unaffected —
 * doze's visuals (eye-close, droop, breathing) run independently of
 * whichever body clip is currently playing.
 */

import type { VrmClipInfo } from './kurisuBridge'
import type { SpringboneWindState } from './springboneWind'

export const PERCH_SIT_CLIP_NAME_PATTERN = /\bsit/i

export interface PetPerchHandlerDeps {
  describeClips: () => VrmClipInfo[] | undefined
  playClip: (name: string, options?: { loop?: boolean }) => void
  /** Resumes the scheduler's own idle rotation — playByName's loop:true
   * sets 'idle' mode, which the mixer's 'finished' event never fires for,
   * so unperching after a looping sit clip needs an explicit hand-back. */
  resumeIdle: () => void
  springboneWindState: SpringboneWindState
  readWindStrengthPref: () => number
  setPerched: (perched: boolean) => void
}

export interface PetPerchHandler {
  handlePerchChanged: (detail: { perched?: boolean; edge?: string | null } | undefined) => void
}

/** Factory mirrors the closure state the inline handler used to keep at
 * effect scope: whether a sit clip is currently playing (so unperch knows
 * to hand back to idle) and the wind-strength override (so unperch restores
 * the user's actual preference instead of a hardcoded value). */
export const createPetPerchHandler = (deps: PetPerchHandlerDeps): PetPerchHandler => {
  let perchWindStrengthOverride: number | null = null
  let perchPlayedSitClip = false

  const handlePerchChanged = (detail: { perched?: boolean; edge?: string | null } | undefined) => {
    const perched = Boolean(detail?.perched)
    deps.setPerched(perched)

    if (!perched) {
      // Unperch: restore normal idle. If a sit clip had taken over the body
      // loop, hand it back to the scheduler's own idle rotation; if we'd
      // only damped wind, restore the user's actual preference.
      if (perchPlayedSitClip) {
        perchPlayedSitClip = false
        deps.resumeIdle()
      }
      if (perchWindStrengthOverride !== null) {
        deps.springboneWindState.windStrengthPref = deps.readWindStrengthPref()
        perchWindStrengthOverride = null
      }
      return
    }

    const sitClip = deps
      .describeClips()
      ?.find(
        (clip) => !clip.lazy && !clip.disabled && clip.loaded && PERCH_SIT_CLIP_NAME_PATTERN.test(clip.name)
      )
    if (sitClip) {
      perchPlayedSitClip = true
      deps.playClip(sitClip.name, { loop: true })
    } else {
      // Fallback: no vetted sit pose available in the eager set — read as
      // calmer instead, by damping the ambient springbone wind (hair/cloth
      // breeze) while perched. Restored to the user's actual preference on
      // unperch. (No eager clip currently matches — see the module-level
      // doc comment — so this path is the one that actually fires today.)
      perchWindStrengthOverride = deps.springboneWindState.windStrengthPref
      deps.springboneWindState.windStrengthPref = deps.readWindStrengthPref() * 0.3
    }
  }

  return { handlePerchChanged }
}
