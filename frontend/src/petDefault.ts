/**
 * "Start as desktop pet" preference (Tauri only): when enabled, the desktop
 * app boots straight to the pet overlay with the main chat window hidden.
 *
 * The decision happens in Rust before any webview runs JS, so the source of
 * truth is a marker file written through the `set_pet_default` command;
 * localStorage only mirrors it so the settings toggle renders correctly.
 * Completing first-run onboarding turns it on once; the toggle rules after.
 */
import { isTauri } from './platform'

export const PET_DEFAULT_KEY = 'kurisu_pet_default'

export const loadPetDefaultEnabled = (): boolean => {
  try {
    return window.localStorage.getItem(PET_DEFAULT_KEY) === 'true'
  } catch {
    return false
  }
}

export const savePetDefaultEnabled = (enabled: boolean): void => {
  try {
    window.localStorage.setItem(PET_DEFAULT_KEY, String(enabled))
  } catch {
    // Mirror just won't persist; the Rust flag below still rules startup.
  }
  if (!isTauri()) return
  void (async () => {
    try {
      const { invoke } = await import('@tauri-apps/api/core')
      await invoke('set_pet_default', { enabled })
    } catch {
      // Best-effort: next startup keeps the previous behavior.
    }
  })()
}

/** Called once when first-run onboarding completes: desktop installs
 * default to pet-first from then on, unless the user has already touched
 * the toggle themselves. */
export const enablePetDefaultAfterOnboarding = (): void => {
  if (!isTauri()) return
  try {
    if (window.localStorage.getItem(PET_DEFAULT_KEY) !== null) return
  } catch {
    // Can't tell; enabling is the documented post-onboarding default.
  }
  savePetDefaultEnabled(true)
}
