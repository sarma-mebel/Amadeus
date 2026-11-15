import NixieMeter from './NixieMeter'
import './DesktopBackendWait.css'

/**
 * Fullscreen pre-boot screen for the Tauri desktop shell while it spins up
 * its bundled Python backend (isWaitingForDesktopBackend() in App.tsx).
 * This state precedes LoadingScreen — the backend host isn't known yet, so
 * /healthz can't be probed — and the shell navigates the webview to the
 * real host once the sidecar answers. Styled as a pared-down version of
 * the same SG0 terminal so the desktop boot reads as one continuous
 * sequence instead of a generic spinner cutting to the cinematic screen.
 *
 * The Rust shell can't reach into React state, so it signals via dataset
 * flags on <html> (main.rs evals them) and the CSS swaps the copy:
 *  - data-amadeus-gpu-warmup   — GPU build warming the Qwen3 voice (long)
 *  - data-amadeus-backend-failed — sidecar never answered /healthz
 */
const DesktopBackendWait = () => (
  <div className="desktop-wait" role="status" aria-label="Starting backend">
    <div className="desktop-wait-scanlines" aria-hidden="true" />

    <div className="desktop-wait-topbar" aria-hidden="true">
      <span className="desktop-wait-topbar-lab">
        Future Gadget Lab — No. 004
        <br />
        <span className="muted">Private Terminal</span>
      </span>
      <span className="desktop-wait-topbar-net desktop-wait-when-booting">
        LAB NETWORK: LINKING <span className="desktop-wait-dot" />
      </span>
      <span className="desktop-wait-topbar-net error desktop-wait-when-failed">
        LAB NETWORK: OFFLINE
      </span>
    </div>

    <div className="desktop-wait-center">
      <h1 className="desktop-wait-wordmark">AMADEUS</h1>
      <div className="desktop-wait-subtitle">COGNITIVE MEMORY INTERFACE</div>

      <span className="desktop-wait-statusline desktop-wait-when-booting">
        ESTABLISHING LAB NETWORK···
      </span>
      <span className="desktop-wait-statusline error desktop-wait-when-failed">
        LAB NETWORK: OFFLINE
      </span>

      {/* Never settles here — this screen navigates away on success rather
          than flipping a `ready` flag, so the meter has no completion state
          of its own; it just keeps calibrating until the page changes under
          it. Hidden on failure, same as the bar it replaces. */}
      <div className="desktop-wait-when-booting">
        <NixieMeter settled={false} />
      </div>

      <p className="desktop-wait-detail desktop-wait-detail-default">
        The bundled server is booting on this machine. This can take a few
        seconds on first launch.
      </p>
      <p className="desktop-wait-detail desktop-wait-detail-gpu">
        Loading the Kurisu voice model onto the GPU. This takes a minute or
        two on each launch.
      </p>
      <p className="desktop-wait-detail desktop-wait-detail-failed">
        The backend did not start. Check the application logs in your app
        data directory.
      </p>
    </div>
  </div>
)

export default DesktopBackendWait
