import { lazy, Suspense, useCallback, useState } from 'react'
import './App.css'
import ChatInterface from './components/ChatInterface'
import TitleBar from './components/TitleBar'
import { isCapacitorNative, isTauri } from './platform'
import { loadBackendUrl } from './config'
import LoadingScreen from './components/LoadingScreen'
import DesktopBackendWait from './components/DesktopBackendWait'
import MobileConnect from './components/MobileConnect'
import { loadUiMode, saveUiMode, type UiMode } from './uiMode'

// three.js + @pixiv/three-vrm are the bulk of the JS bundle; splitting them
// into their own chunk lets the chat column paint (and become interactive)
// before that chunk finishes downloading, instead of both waiting on one
// ~1MB bundle. The fallback mirrors VRMViewer's own loading overlay so
// there's no visual gap between "chunk downloading" and "assets downloading".
const VRMViewer = lazy(() => import('./components/VRMViewer'))
const PetView = lazy(() => import('./components/PetView'))
const AnimLab = lazy(() => import('./components/AnimLab'))

// Desktop-pet overlay window (Tauri opens a second webview onto the same
// origin with ?view=pet). Same-origin means backend discovery, API token,
// session/user ids, and settings all ride the shared localStorage for free.
const viewParam = () =>
  typeof window !== 'undefined'
    ? new URLSearchParams(window.location.search).get('view')
    : null
const isPetView = () => viewParam() === 'pet'
// Clip-vetting workbench (docs/animations.md): full scene + inventory
// panel, no chat.
const isAnimLabView = () => viewParam() === 'animlab'

const VRMViewerFallback = () => (
  <div className="vrm-overlay" role="status">
    <div className="vrm-overlay-spinner" aria-hidden="true" />
    <p>Loading Amadeus…</p>
  </div>
)

const isWaitingForDesktopBackend = () => {
  if (typeof window === 'undefined') return false
  const { hostname, protocol } = window.location
  const isTauriBootstrapHost = hostname === 'tauri.localhost'
  if (!isTauri() && !isTauriBootstrapHost) return false
  if (protocol !== 'http:' && protocol !== 'https:') return true
  return hostname !== '127.0.0.1' && hostname !== 'localhost'
}

function App() {
  // The Amadeus connection sequence overlays the app until the user taps
  // to connect. The chat + VRM mount (and load) behind it, so by the time
  // the overlay fades the scene is ready; the greeting is gated on
  // `entered` so her voiced opener plays after the audio-unlocking tap.
  const [entered, setEntered] = useState(false)
  // Immersion ("VN") mode: avatar full-width, chat log replaced by a
  // bottom dialogue overlay. Persisted across sessions like the
  // environment picker.
  const [uiMode, setUiMode] = useState<UiMode>(() => loadUiMode())
  const toggleUiMode = useCallback(() => {
    setUiMode((prev) => {
      const next: UiMode = prev === 'vn' ? 'full' : 'vn'
      saveUiMode(next)
      return next
    })
  }, [])

  if (isPetView()) {
    // Only ever created after the backend health check, so no
    // DesktopBackendWait branch needed here.
    return (
      <Suspense fallback={null}>
        <PetView />
      </Suspense>
    )
  }

  if (isAnimLabView()) {
    return (
      <div className="app-container">
        <div className="vrm-panel">
          <Suspense fallback={<VRMViewerFallback />}>
            <VRMViewer />
          </Suspense>
        </div>
        <div className="chat-panel">
          <Suspense fallback={null}>
            <AnimLab />
          </Suspense>
        </div>
      </div>
    )
  }

  if (isWaitingForDesktopBackend()) {
    // Fullscreen pre-boot screen — the shell navigates to the real backend
    // host once the sidecar is up, so nothing behind it would ever be seen.
    // TitleBar still mounts here: decorations:false removes the native
    // close/minimize chrome at the OS-window level regardless of which
    // React branch is showing, so without this the boot screen would be
    // unclosable until the sidecar finishes starting.
    return (
      <div className="app-shell">
        <TitleBar />
        <DesktopBackendWait />
      </div>
    )
  }

  if (isCapacitorNative() && !loadBackendUrl()) {
    // The Android app ships no backend: nothing behind this gate can work
    // until a server URL is saved (MobileConnect reloads after saving).
    return <MobileConnect />
  }

  return (
    <div className="app-shell">
      <TitleBar />
      <div className="app-container" data-mode={uiMode}>
        {!entered && <LoadingScreen onConnect={() => setEntered(true)} />}
        <div className="vrm-panel">
          <Suspense fallback={<VRMViewerFallback />}>
            <VRMViewer />
          </Suspense>
        </div>
        <div className="chat-panel">
          <ChatInterface entered={entered} uiMode={uiMode} onToggleUiMode={toggleUiMode} />
        </div>
      </div>
    </div>
  )
}

export default App
