/**
 * Environment registry: the set of selectable VRM backdrops. Mirrors the
 * localStorage-preference pattern in soundscape.ts (load/save helpers, a
 * versioned key so a future shape change can be migrated/ignored cleanly).
 *
 * Each entry's `anchor` is where Kurisu's vrm.scene is placed/rotated for
 * that set, found empirically per-GLB (the sets differ wildly in scale and
 * origin placement) — see VRMViewer's environment-loading code for how
 * mainLight.target, the shadow catcher, and the shadow frustum all key off
 * the same anchor so switching sets doesn't leave shadows pointed at the
 * old spot.
 */

export type EnvironmentId = 'time_machine' | 'lab'

export interface EnvironmentAnchor {
  position: [number, number, number]
  rotationY: number
}

/** Ambience preset for soundscape.ts's startAmbience/setAmbiencePreset, plus
 * the parallel reverb send VRMViewer's lip-sync graph builds per-mount (see
 * ensureLipSyncGraph). Interior sets (small room) get a short, denser tail;
 * outdoor/large sets get a longer, sparser one. */
export interface EnvironmentAudio {
  ambience: 'machine' | 'street'
  reverb: {
    /** Impulse-response length in seconds (procedurally generated noise-burst decay). */
    irDurationSec: number
    /** Exponential decay exponent — higher decays faster/tighter. */
    decay: number
    /** Wet-path mix, 0-1, applied before the shared master (dry+wet) gain. */
    wetMix: number
  }
}

export interface EnvironmentDefinition {
  id: EnvironmentId
  label: string
  file: string
  anchor: EnvironmentAnchor
  /** Defaults to `anchor.position` when omitted — kept distinct only in
   * case a set ever needs its shadow catcher/frustum centered somewhere
   * other than Kurisu's exact standing spot. */
  shadowCenter?: [number, number, number]
  /** Optional ambience + reverb tuning for this set. Omitted entirely means
   * "use the existing machine-hum default with no reverb send" — kept
   * optional so a newly added set without tuned values still behaves like
   * the pre-B1/B2 baseline instead of silently reverting to a fallback. */
  audio?: EnvironmentAudio
}

export const ENVIRONMENTS: Record<EnvironmentId, EnvironmentDefinition> = {
  time_machine: {
    id: 'time_machine',
    label: 'Time Machine',
    file: '/static/environment/steinsgate_time_machine.glb',
    anchor: { position: [-5, 0, 0], rotationY: -Math.PI / 2 },
    // Interior of the time machine capsule — small, boxy room.
    audio: { ambience: 'machine', reverb: { irDurationSec: 0.6, decay: 2.5, wetMix: 0.22 } },
  },
  lab: {
    id: 'lab',
    label: 'Future Gadget Lab',
    file: '/static/environment/steinsgate_-_future_gadget_lab.glb',
    // Found empirically with the headless playwright harness. This GLB is
    // an exterior diorama of the lab building (no modeled room interior —
    // the 2F "lab" windows just glow from outside), so the anchor puts her
    // at street level in front of the Braun Tube Workshop storefront: CRT
    // TVs in the window, bicycle, awning and the lab above — readable set
    // dressing with no furniture clipping, and the street gives the
    // auto-framing camera unobstructed room. Street surface is y≈1.07 (the
    // whole diorama floats above y=0).
    anchor: { position: [0, 1.07, 4.0], rotationY: 0 },
    // Exterior street diorama — larger, more open space, sparser reverb tail.
    audio: { ambience: 'street', reverb: { irDurationSec: 1.1, decay: 1.5, wetMix: 0.15 } },
  },
}

export const DEFAULT_ENVIRONMENT: EnvironmentId = 'time_machine'

const ENVIRONMENT_KEY = 'kurisu_environment_v1'

export const loadEnvironmentId = (): EnvironmentId => {
  if (typeof window === 'undefined') return DEFAULT_ENVIRONMENT
  try {
    const stored = window.localStorage.getItem(ENVIRONMENT_KEY)
    if (stored === 'time_machine' || stored === 'lab') return stored
    return DEFAULT_ENVIRONMENT
  } catch {
    return DEFAULT_ENVIRONMENT
  }
}

export const saveEnvironmentId = (id: EnvironmentId): void => {
  if (typeof window === 'undefined') return
  try {
    window.localStorage.setItem(ENVIRONMENT_KEY, id)
  } catch {
    // Non-fatal — losing the environment preference isn't worth surfacing an error.
  }
}

/** Dispatched on window when the user changes the environment in
 * SettingsPanel; VRMViewer listens and hot-swaps without a reload. */
export const ENVIRONMENT_CHANGE_EVENT = 'kurisu:environment-change'

export const dispatchEnvironmentChange = (id: EnvironmentId): void => {
  if (typeof window === 'undefined') return
  window.dispatchEvent(new CustomEvent<EnvironmentId>(ENVIRONMENT_CHANGE_EVENT, { detail: id }))
}
