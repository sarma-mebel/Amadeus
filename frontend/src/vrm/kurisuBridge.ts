/**
 * Typed singleton bridge between the two sibling panes.
 *
 * ChatInterface and VRMViewer are rendered as siblings with no prop channel;
 * they used to talk through mutable `window.kurisu*` globals. This module is
 * the typed replacement: VRMViewer registers the avatar-control
 * implementations (lip sync, expressions, gestures, animation controls) once
 * its scene loads and clears them on teardown; ChatInterface registers
 * `poke` (it owns all backend fetches) and calls the rest. Every slot is
 * null until its owner registers it, so call sites use optional chaining
 * exactly as they did with the window globals.
 *
 * `window.kurisuLipSync` and `window.vrmAnimations` are still assigned as
 * thin aliases of `kurisuBridge.lipSync`/`kurisuBridge.vrmAnimations` — the
 * former for LoadingScreen (which pre-dates this module), the latter for
 * console/debug use and the anim lab's polling effect (AnimLab.tsx) — same
 * pattern as `window.kurisuLipSync`. New call sites should read
 * `kurisuBridge.vrmAnimations` directly; the window alias is back-compat
 * only, not the canonical accessor.
 */

import type * as THREE from 'three'
import type { SpeechBeatKind } from './speechBeats'

export interface VrmClipInfo {
  name: string
  filename: string
  directory: string | null
  role: 'idle' | 'gesture' | 'talk' | 'subtle'
  tags: string[]
  disabled: boolean
  additive: boolean
  isFbx: boolean
  /** Library clip loaded on first play instead of at startup. */
  lazy: boolean
  /** False for a lazy clip that hasn't been played yet (duration is 0). */
  loaded: boolean
  duration: number
  /** True when the clip is in the looping-idle rotation pool. */
  looping: boolean
}

export interface VrmAnimationControls {
  clips: Record<string, THREE.AnimationClip>
  mixer: THREE.AnimationMixer | null
  /** `additive: true` previews the clip as an additive layer over the
   * running idle (vrm/additiveGestures) regardless of the global toggle —
   * the anim-lab vetting affordance. */
  play: (name: string, options?: { fadeDuration?: number; loop?: boolean; additive?: boolean }) => void
  list: () => void
  /** Structured inventory for the anim-lab view (?view=animlab). */
  describe: () => VrmClipInfo[]
  /** Name of the clip the body state machine is currently playing. */
  current: () => string | null
}

export interface KurisuLipSync {
  play: (options: { base64Audio: string; mimeType?: string }) => Promise<void> | void
  stop: () => void
  beginStream: () => void
  // `seq`/`emotion` drive ordered karaoke timing + per-emotion pitch
  // detune; `beat` (vrm/speechBeats) rides along so the sentence's beat
  // gesture fires when its audio starts, not when its text streamed.
  enqueue: (options: {
    base64Audio: string
    mimeType?: string
    seq?: number
    emotion?: string
    beat?: SpeechBeatKind
  }) => Promise<void>
  endStream: () => void
  isSpeaking: () => boolean
}

export interface KurisuExpressions {
  set: (expression: string, intensity?: number, duration?: number) => void
  available: string[]
  list: () => void
}

/** Doze-state read/wake surface (pet variant only — see ../vrm/petDoze),
 * registered by VRMViewer alongside its other pet-only wiring and cleared
 * on teardown like every other slot. */
export interface KurisuPetDoze {
  /** True only while fully `dozing` (not merely `drowsy`) — mirrors
   * petDoze.ts's isDozing(). */
  active: () => boolean
  /** Requests an explicit wake (pointer-enter, poke, or an external
   * trigger). Returns true iff she was actually dozing (not drowsy/awake)
   * at the moment of the call — the pet UI uses that to decide whether a
   * groggy in-character line is warranted. Always safe to call regardless
   * of current phase. */
  wake: (reason?: string) => boolean
}

export interface KurisuBridge {
  // Avatar control implementations, registered by VRMViewer once the scene
  // loads (null before that, and again after its effect tears down).
  lipSync: KurisuLipSync | null
  expressions: KurisuExpressions | null
  // Emotion-driven gestures: at most one one-shot body gesture per reply;
  // reactReset clears the per-reply throttle at the start of a turn.
  react: ((emotion: string, intensity?: number) => void) | null
  reactReset: (() => void) | null
  // Poke-click on the avatar (detected by VRMViewer) → POST /api/poke
  // (owned by ChatInterface) — the same bridge pattern in reverse.
  poke: ((region?: string) => void) | null
  // Console/anim-lab clip controls (play/list/describe/current), formerly
  // window.vrmAnimations only. window.vrmAnimations stays assigned as a
  // back-compat alias of this slot (see module docstring).
  vrmAnimations: VrmAnimationControls | null
  // Pet-variant doze state (see KurisuPetDoze above) — null outside the pet
  // variant, exactly like every other avatar-control slot before its owner
  // registers it.
  petDoze: KurisuPetDoze | null
}

export const kurisuBridge: KurisuBridge = {
  lipSync: null,
  expressions: null,
  react: null,
  reactReset: null,
  poke: null,
  vrmAnimations: null,
  petDoze: null,
}
