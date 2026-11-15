/** Runtime environment detection shared across the app. */

/** True when running inside the Tauri desktop shell (vs a plain browser). */
export const isTauri = (): boolean =>
  typeof window !== 'undefined' &&
  Boolean(window.__TAURI_INTERNALS__ || window.__TAURI__)

/** True when running inside the Capacitor native shell (the Android app).
 * Uses the injected global rather than importing @capacitor/core so plain
 * web builds don't grow a dependency on the bridge. */
export const isCapacitorNative = (): boolean => {
  if (typeof window === 'undefined') return false
  const cap = (window as { Capacitor?: { isNativePlatform?: () => boolean } }).Capacitor
  try {
    return Boolean(cap?.isNativePlatform?.())
  } catch {
    return false
  }
}
