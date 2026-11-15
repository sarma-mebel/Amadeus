/**
 * Presence intensity: one dial (quiet / balanced / lively) over the
 * client-side presence features, so "how much unprompted life does she
 * have" is a single choice instead of a scavenger hunt across toggles.
 *
 * The stored level is the authority for idle-chatter cadence (the one
 * knob with no individual toggle); picking a level also APPLIES its
 * preset to the individual feature toggles (thinking sounds, listening
 * reactions, desktop roaming), which remain the per-feature source of
 * truth and can be fine-tuned afterwards. "Balanced" matches the app's
 * shipped defaults, so an untouched install already sits on it.
 *
 * Deliberately excluded from presets: screen watching (a privacy opt-in
 * a preset must never switch on), desktop notifications (toggling them
 * fires a permission prompt), and interaction modes like hands-free or
 * duplex calls (how you talk to her, not how present she is).
 */
import { loadFillersEnabled, saveFillersEnabled } from './audio/fillerPlayer'
import {
  loadListeningReactionsEnabled,
  saveListeningReactionsEnabled,
} from './audio/listeningReactions'
import { isPetRoamEnabled, setPetRoamEnabled } from './vrm/petRoam'
import { isTauri } from './platform'

export type PresenceLevel = 'quiet' | 'balanced' | 'lively'

export const PRESENCE_LEVEL_KEY = 'kurisu_presence_level'

export interface PresencePreset {
  fillers: boolean
  listeningReactions: boolean
  petRoam: boolean
}

export const PRESENCE_PRESETS: Record<PresenceLevel, PresencePreset> = {
  quiet: { fillers: false, listeningReactions: false, petRoam: false },
  balanced: { fillers: false, listeningReactions: true, petRoam: false },
  lively: { fillers: true, listeningReactions: true, petRoam: true },
}

export const loadPresenceLevel = (): PresenceLevel => {
  try {
    const raw = window.localStorage.getItem(PRESENCE_LEVEL_KEY)
    if (raw === 'quiet' || raw === 'balanced' || raw === 'lively') return raw
  } catch {
    // Fall through to the default.
  }
  return 'balanced'
}

/** Store the level and push its preset into the individual toggles. */
export const applyPresenceLevel = (level: PresenceLevel): void => {
  try {
    window.localStorage.setItem(PRESENCE_LEVEL_KEY, level)
  } catch {
    // Preference just doesn't persist; the toggles below still apply.
  }
  const preset = PRESENCE_PRESETS[level]
  saveFillersEnabled(preset.fillers)
  saveListeningReactionsEnabled(preset.listeningReactions)
  // Roaming is desktop-only: in a browser the key would be inert, but
  // never write `true` for a feature that can't run here.
  setPetRoamEnabled(isTauri() ? preset.petRoam : false)
}

/** True when a fine-tune toggle has drifted from the stored level's
 * preset — the Settings dial shows a "customized" note instead of
 * pretending the preset still describes reality. */
export const presenceLevelCustomized = (): boolean => {
  const preset = PRESENCE_PRESETS[loadPresenceLevel()]
  if (loadFillersEnabled() !== preset.fillers) return true
  if (loadListeningReactionsEnabled() !== preset.listeningReactions) return true
  if (isTauri() && isPetRoamEnabled() !== preset.petRoam) return true
  return false
}

/** Idle-chatter cadence for the stored level: null = suppress unprompted
 * chatter entirely (quiet); otherwise a multiplier on the client's
 * randomized delay window (lively checks in a bit more often). Callers
 * read this at schedule/fire time so a change applies without a remount;
 * the server's own idle-chatter gate and cooldowns still apply. */
export const idleChatterDelayMultiplier = (): number | null => {
  const level = loadPresenceLevel()
  if (level === 'quiet') return null
  return level === 'lively' ? 0.6 : 1
}
