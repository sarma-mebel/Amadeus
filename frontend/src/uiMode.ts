/**
 * Immersion ("VN") mode: swaps the normal split layout (avatar + full chat
 * log) for a visual-novel style overlay (avatar full-width, only the last
 * line + input bar shown). Persisted like environments.ts's environment
 * pick — same load/save/try-catch shape, so a corrupted or foreign
 * localStorage value can't wedge the app in a broken layout.
 */
export type UiMode = 'full' | 'vn'

// VN mode is the default: she fills the window and the log stays out of
// the way. Users who prefer the split layout toggle once and the choice
// persists via localStorage.
export const DEFAULT_UI_MODE: UiMode = 'vn'

const UI_MODE_KEY = 'kurisu_ui_mode_v1'

export const loadUiMode = (): UiMode => {
  if (typeof window === 'undefined') return DEFAULT_UI_MODE
  try {
    const stored = window.localStorage.getItem(UI_MODE_KEY)
    if (stored === 'full' || stored === 'vn') return stored
    return DEFAULT_UI_MODE
  } catch {
    return DEFAULT_UI_MODE
  }
}

export const saveUiMode = (mode: UiMode): void => {
  if (typeof window === 'undefined') return
  try {
    window.localStorage.setItem(UI_MODE_KEY, mode)
  } catch {
    // Non-fatal — losing the layout preference isn't worth surfacing an error.
  }
}
