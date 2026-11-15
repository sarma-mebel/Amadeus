/**
 * The voice-audio mute preference, shared by the main chat, the pet window,
 * and presence lines. Historically this was six raw
 * `localStorage.getItem('kurisu_audio_enabled')` sites with the default-on
 * rule re-encoded at each one.
 *
 * Two read shapes on purpose:
 * - `isAudioEnabled()` — default-on boolean; what every playback gate wants.
 * - `loadAudioPreference()` — the raw stored value including `null` for
 *   "never set". ChatInterface needs the null case: touch devices default
 *   the toggle off and show an explicit audio-unlock affordance only until
 *   the user has chosen either way.
 */
export const AUDIO_ENABLED_KEY = 'kurisu_audio_enabled'

export const loadAudioPreference = (): string | null => {
  if (typeof window === 'undefined') return null
  try {
    return window.localStorage.getItem(AUDIO_ENABLED_KEY)
  } catch {
    return null
  }
}

export const isAudioEnabled = (): boolean => loadAudioPreference() !== 'false'

export const saveAudioEnabled = (enabled: boolean): void => {
  try {
    window.localStorage.setItem(AUDIO_ENABLED_KEY, String(enabled))
  } catch {
    // Preference just won't persist across reloads.
  }
}
