import { useEffect, useState } from 'react'
import { isTauri } from './platform'

/**
 * Fullscreen toggle that works in both runtimes:
 * - Tauri desktop shell: true OS-window fullscreen via the window API
 *   (the WebView2 HTML Fullscreen API only fills the window, not the
 *   screen, so element fullscreen is not enough there).
 * - Browser: the standard Fullscreen API on <html>.
 *
 * Module-scope key handling (same idiom as soundscape.ts): F11 toggles
 * everywhere; Escape exits only in Tauri, where — unlike the browser's
 * native fullscreen — nothing exits it for us.
 */

type Listener = (fullscreen: boolean) => void
const listeners = new Set<Listener>()
let current = false

const setCurrent = (value: boolean) => {
  if (current === value) return
  current = value
  listeners.forEach((listener) => listener(value))
}

const tauriWindow = async () => {
  const { getCurrentWindow } = await import('@tauri-apps/api/window')
  return getCurrentWindow()
}

export async function toggleFullscreen(): Promise<void> {
  if (isTauri()) {
    try {
      const win = await tauriWindow()
      const next = !(await win.isFullscreen())
      await win.setFullscreen(next)
      setCurrent(next)
    } catch (err) {
      console.warn('Fullscreen toggle failed:', err)
    }
    return
  }
  try {
    if (document.fullscreenElement) {
      await document.exitFullscreen()
    } else {
      await document.documentElement.requestFullscreen()
    }
    // state syncs via the fullscreenchange listener below
  } catch (err) {
    console.warn('Fullscreen toggle failed:', err)
  }
}

/** Current fullscreen state + toggle, for UI like the HUD button. */
export function useFullscreen(): [boolean, () => void] {
  const [fullscreen, setFullscreen] = useState(current)
  useEffect(() => {
    listeners.add(setFullscreen)
    return () => {
      listeners.delete(setFullscreen)
    }
  }, [])
  return [fullscreen, () => void toggleFullscreen()]
}

if (typeof window !== 'undefined') {
  window.addEventListener('keydown', (event) => {
    if (event.key === 'F11') {
      // In the browser this swaps native F11 fullscreen (invisible to the
      // Fullscreen API) for element fullscreen, so our state stays right.
      event.preventDefault()
      void toggleFullscreen()
    } else if (event.key === 'Escape' && current && isTauri()) {
      void toggleFullscreen()
    }
  })
  if (!isTauri()) {
    document.addEventListener('fullscreenchange', () => {
      setCurrent(Boolean(document.fullscreenElement))
    })
  } else {
    // Sync in case the window starts fullscreen (e.g. restored by the OS).
    void tauriWindow()
      .then((win) => win.isFullscreen())
      .then(setCurrent)
      .catch(() => {})
  }
}
