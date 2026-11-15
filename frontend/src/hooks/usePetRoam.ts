/**
 * React wiring for the desktop-roaming pet (see ../vrm/petRoam for the
 * scheduler/state machine itself). This hook only does three things:
 *
 *  1. Tracks "last user interaction" via window-level pointer/key/wheel
 *     listeners — a pointerdown covers drag-bar grabs (every OS drag starts
 *     with one), keydown covers the composer, wheel covers scrolling — and
 *     cancels an in-flight stroll instantly on any of them (the scheduler's
 *     own watchdog would catch it within 250ms anyway; this just removes
 *     the latency).
 *  2. Assembles the eligibility snapshot from state the pet stack already
 *     tracks: busy/doze/courtesy refs from usePetChat, DND from the shared
 *     localStorage contract, speaking straight off kurisuBridge.lipSync,
 *     mic listening through a caller-supplied getter (mic + hands-free live
 *     as hook results in usePetChat, not refs).
 *  3. Owns the driver's lifecycle (start on mount, stop on unmount).
 *
 * Tauri-only: outside the desktop shell startPetRoam no-ops and the
 * listeners are never attached (browsers can't move their own window).
 * The kurisu_pet_roam opt-in is re-read inside the driver at every cadence
 * tick, so this hook mounts unconditionally in the pet window and the
 * Settings toggle applies without a remount.
 */
import { useEffect, useRef } from 'react'
import { isTauri } from '../platform'
import { kurisuBridge } from '../vrm/kurisuBridge'
import { startPetRoam, type PetRoamHandle } from '../vrm/petRoam'
import { isPetDndActive } from './usePetPresence'

export interface UsePetRoamDeps {
  /** True while a chat turn is in flight. */
  isBusyRef: React.RefObject<boolean>
  /** True while fully dozing (mirrored from kurisuBridge.petDoze). */
  isDozingRef: React.RefObject<boolean>
  /** True while hidden for fullscreen courtesy. */
  isCourtesyActiveRef: React.RefObject<boolean>
  /** True while the user's mic captures (push-to-talk or hands-free). */
  isListening: () => boolean
}

export const usePetRoam = (deps: UsePetRoamDeps): void => {
  // Freshest deps through a ref so the mount-once effect below never needs
  // to re-subscribe — same pattern as usePetPresence's depsRef.
  const depsRef = useRef(deps)
  depsRef.current = deps

  useEffect(() => {
    if (!isTauri()) return
    const lastInteractionAt = { current: Date.now() }
    let handle: PetRoamHandle | null = null
    const noteInteraction = () => {
      lastInteractionAt.current = Date.now()
      handle?.cancelStroll('interaction')
    }
    window.addEventListener('pointerdown', noteInteraction)
    window.addEventListener('keydown', noteInteraction)
    window.addEventListener('wheel', noteInteraction)
    handle = startPetRoam({
      getEligibility: () => ({
        busy: depsRef.current.isBusyRef.current === true,
        speaking: kurisuBridge.lipSync?.isSpeaking?.() === true,
        listening: depsRef.current.isListening(),
        // Read the bridge directly as well as the mirrored ref: the ref is
        // polled on a 1s interval in usePetChat, and a stroll starting in
        // that stale window would visibly drag a sleeping avatar around.
        dozing:
          depsRef.current.isDozingRef.current === true || kurisuBridge.petDoze?.active() === true,
        dndActive: isPetDndActive(),
        courtesyHidden: depsRef.current.isCourtesyActiveRef.current === true,
        msSinceInteraction: Date.now() - lastInteractionAt.current,
      }),
    })
    return () => {
      window.removeEventListener('pointerdown', noteInteraction)
      window.removeEventListener('keydown', noteInteraction)
      window.removeEventListener('wheel', noteInteraction)
      handle?.stop()
      handle = null
    }
  }, [])
}
