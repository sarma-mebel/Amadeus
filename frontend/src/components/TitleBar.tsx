import { useEffect, useState } from 'react'
import './TitleBar.css'
import { isCapacitorNative, isTauri } from '../platform'
import { diagnostics } from '../diagnostics'
import type { Window as TauriWindow } from '@tauri-apps/api/window'

/**
 * Custom in-app title bar for the Tauri MAIN window.
 *
 * The native Windows chrome (menu-less but still OS-drawn min/max/close +
 * caption) was the last visible "this is a webview app" tell once the rest
 * of the UI committed to the navy/auburn identity. This replaces it with a
 * draggable in-app bar; the actual window is created with
 * `"decorations": false` (tauri.conf.json, main window only — see
 * docs/pet-release-checklist.md for the packaged-build validation this
 * needs since none of drag/dblclick-maximize/shadow is testable from WSL).
 *
 * Renders ONLY for the desktop main window:
 *  - never in a plain browser tab (isTauri() false)
 *  - never in the Android/Capacitor app (shares this frontend bundle)
 *  - never in the pet overlay window (own frameless window, own chrome
 *    contract, platform surface frozen — see the project notes)
 *  - never in the animlab workbench view (not a real end-user window)
 *
 * Every window API call is best-effort: a rejection (missing plugin,
 * pre-1.0 shell, etc.) is caught, counted via diagnostics so a dead title
 * bar shows up in Settings → Diagnostics instead of silently doing
 * nothing, and never throws past this component.
 */

const viewParam = () =>
  typeof window !== 'undefined'
    ? new URLSearchParams(window.location.search).get('view')
    : null

/** True only for the single unlabeled/"main" desktop window — not pet, not
 * animlab, not a browser or Android tab. Mirrors App.tsx's own
 * isPetView()/isAnimLabView() query-param checks so this component doesn't
 * need those functions exported just to reuse them. */
function useIsMainDesktopWindow(): boolean {
  const [isMain, setIsMain] = useState(false)

  useEffect(() => {
    if (!isTauri() || isCapacitorNative()) return
    const view = viewParam()
    if (view === 'pet' || view === 'animlab') return

    let cancelled = false
    void (async () => {
      try {
        const { getCurrentWindow } = await import('@tauri-apps/api/window')
        const label = getCurrentWindow().label
        if (!cancelled && label === 'main') setIsMain(true)
      } catch (error) {
        diagnostics.increment('titlebar.window_api_failed', `label: ${String(error)}`)
      }
    })()
    return () => {
      cancelled = true
    }
  }, [])

  return isMain
}

function useIsMaximized(active: boolean): boolean {
  const [maximized, setMaximized] = useState(false)

  useEffect(() => {
    if (!active) return
    let cancelled = false
    let unlisten: (() => void) | undefined

    void (async () => {
      try {
        const { getCurrentWindow } = await import('@tauri-apps/api/window')
        const win = getCurrentWindow()
        const current = await win.isMaximized()
        if (!cancelled) setMaximized(current)
        unlisten = await win.onResized(async () => {
          try {
            const next = await win.isMaximized()
            if (!cancelled) setMaximized(next)
          } catch (error) {
            diagnostics.increment('titlebar.window_api_failed', `onResized: ${String(error)}`)
          }
        })
      } catch (error) {
        diagnostics.increment('titlebar.window_api_failed', `isMaximized/onResized: ${String(error)}`)
      }
    })()

    return () => {
      cancelled = true
      unlisten?.()
    }
  }, [active])

  return maximized
}

async function getWin(): Promise<TauriWindow> {
  const { getCurrentWindow } = await import('@tauri-apps/api/window')
  return getCurrentWindow()
}

async function callWindow(action: string, fn: (win: TauriWindow) => Promise<void>) {
  try {
    const win = await getWin()
    await fn(win)
  } catch (error) {
    diagnostics.increment('titlebar.window_api_failed', `${action}: ${String(error)}`)
  }
}

const MinimizeIcon = () => (
  <svg width="10" height="10" viewBox="0 0 10 10" aria-hidden="true">
    <line x1="1" y1="5" x2="9" y2="5" stroke="currentColor" strokeWidth="1" />
  </svg>
)

const MaximizeIcon = () => (
  <svg width="10" height="10" viewBox="0 0 10 10" aria-hidden="true">
    <rect x="1" y="1" width="8" height="8" stroke="currentColor" strokeWidth="1" fill="none" />
  </svg>
)

const RestoreIcon = () => (
  <svg width="10" height="10" viewBox="0 0 10 10" aria-hidden="true">
    <rect x="2.5" y="1" width="6.5" height="6.5" stroke="currentColor" strokeWidth="1" fill="none" />
    <path d="M1 3.5V9H6.5" stroke="currentColor" strokeWidth="1" fill="none" />
  </svg>
)

const CloseIcon = () => (
  <svg width="10" height="10" viewBox="0 0 10 10" aria-hidden="true">
    <line x1="1" y1="1" x2="9" y2="9" stroke="currentColor" strokeWidth="1" />
    <line x1="9" y1="1" x2="1" y2="9" stroke="currentColor" strokeWidth="1" />
  </svg>
)

export default function TitleBar() {
  const isMain = useIsMainDesktopWindow()
  const maximized = useIsMaximized(isMain)

  if (!isMain) return null

  const handleMinimize = () => void callWindow('minimize', (win) => win.minimize())
  const handleToggleMaximize = () => void callWindow('toggleMaximize', (win) => win.toggleMaximize())
  // Goes through the normal close flow (fires CloseRequested in main.rs),
  // which hides the main window while the pet is alive and otherwise kills
  // the backend sidecar + exits — same as clicking native chrome's X ever
  // did. Never call destroy()/hide() directly here.
  const handleClose = () => void callWindow('close', (win) => win.close())

  return (
    <div className="titlebar" data-tauri-drag-region>
      <span className="titlebar-title" data-tauri-drag-region>
        Amadeus
      </span>
      <div className="titlebar-controls">
        <button
          type="button"
          className="titlebar-btn titlebar-btn-min"
          aria-label="Minimize"
          onClick={handleMinimize}
        >
          <MinimizeIcon />
        </button>
        <button
          type="button"
          className="titlebar-btn titlebar-btn-max"
          aria-label={maximized ? 'Restore' : 'Maximize'}
          onClick={handleToggleMaximize}
        >
          {maximized ? <RestoreIcon /> : <MaximizeIcon />}
        </button>
        <button
          type="button"
          className="titlebar-btn titlebar-btn-close"
          aria-label="Close"
          onClick={handleClose}
        >
          <CloseIcon />
        </button>
      </div>
    </div>
  )
}
