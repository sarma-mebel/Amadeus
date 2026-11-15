"""Native desktop shell for the Amadeus web UI.

Runs the exact same FastAPI app the browser and Docker deployments use,
bound to a loopback port, and opens it in a native OS window (pywebview —
WebView2 on Windows, WebKit on Linux/macOS). Nothing needs to be hosted:
the SPA, the API, and the static VRM assets are all served from the local
process.

Launch behaviour is game-like: the window opens immediately with a splash
screen while the core boots in the background, then navigates to the app;
startup failures render an in-window error screen (with the log path)
instead of dying silently. A second launch detects the running instance
and opens a window onto it rather than booting a duplicate server.

Modes:
    python -m amadeus.desktop_webview              # native window (needs .[desktop])
    python -m amadeus.desktop_webview --browser    # system browser instead
    python -m amadeus.desktop_webview --no-open    # server only (containers, tests)

When pywebview isn't installed the shell degrades to ``--browser``
automatically, so the base install still works.

Under PyInstaller (``sys.frozen``) the bundled data mirrors the repo layout
(config/, static/, frontend/dist), which is exactly where ``Config.load``
and ``web.app.create_app`` already look — no special-casing required.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import os
import socket
import sys
import threading
import time
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path
from typing import Optional, Sequence

logger = logging.getLogger("amadeus.desktop_webview")

APP_IMPORT = "amadeus.web.app:app"
WINDOW_TITLE = "Amadeus — Kurisu"
DEFAULT_WINDOW_SIZE = (1280, 820)
MIN_WINDOW_SIZE = (960, 600)
STARTUP_TIMEOUT = 60.0

APP_DATA_DIR = Path.home() / ".amadeus"
LOCK_FILE = APP_DATA_DIR / "desktop.lock"
LOG_FILE = APP_DATA_DIR / "desktop.log"
WINDOW_STATE_FILE = APP_DATA_DIR / "window.json"

SPLASH_HTML = """<!doctype html>
<html><head><meta charset="utf-8"><title>Amadeus</title><style>
  html, body { height: 100%; margin: 0; }
  body {
    display: flex; flex-direction: column; align-items: center; justify-content: center;
    gap: 18px; background: radial-gradient(circle at 30% 20%, rgba(239,71,111,.18), transparent 55%),
    radial-gradient(circle at 75% 80%, rgba(0,118,255,.14), transparent 60%), #0b101c;
    color: #f9fafb; font-family: 'Segoe UI', system-ui, sans-serif;
  }
  h1 { margin: 0; font-size: 30px; letter-spacing: 10px; font-weight: 600; }
  p { margin: 0; font-size: 13px; letter-spacing: 2px; text-transform: uppercase;
      color: rgba(249,250,251,.6); animation: pulse 1.6s ease-in-out infinite; }
  .ring { width: 42px; height: 42px; border-radius: 999px;
      border: 3px solid rgba(255,255,255,.15); border-top-color: #ef476f;
      animation: spin .9s linear infinite; }
  @keyframes spin { to { transform: rotate(360deg); } }
  @keyframes pulse { 50% { opacity: .35; } }
  @media (prefers-reduced-motion: reduce) { .ring, p { animation: none; } }
</style></head><body>
  <h1>AMADEUS</h1>
  <div class="ring"></div>
  <p>Booting local core…</p>
</body></html>"""

ERROR_HTML = """<!doctype html>
<html><head><meta charset="utf-8"><title>Amadeus</title><style>
  html, body {{ height: 100%; margin: 0; }}
  body {{
    display: flex; flex-direction: column; align-items: center; justify-content: center;
    gap: 12px; background: #0b101c; color: #f9fafb;
    font-family: 'Segoe UI', system-ui, sans-serif; text-align: center; padding: 24px;
  }}
  h1 {{ margin: 0; font-size: 22px; }}
  p {{ margin: 0; font-size: 14px; color: rgba(249,250,251,.7); max-width: 520px; line-height: 1.6; }}
  code {{ background: rgba(255,255,255,.08); padding: 2px 8px; border-radius: 6px; font-size: 13px; }}
</style></head><body>
  <h1>Amadeus failed to start</h1>
  <p>{message}</p>
  <p>Details were written to <code>{log_path}</code>. Close this window and try again.</p>
</body></html>"""


def find_free_port(host: str = "127.0.0.1") -> int:
    """Ask the OS for an ephemeral port that is free right now."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return sock.getsockname()[1]


def wait_for_server(url: str, timeout: float = 30.0, interval: float = 0.25) -> bool:
    """Poll ``url`` until it answers 200 or ``timeout`` elapses."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                if response.status == 200:
                    return True
        except (urllib.error.URLError, OSError, ValueError):
            pass
        time.sleep(interval)
    return False


# --- single instance -------------------------------------------------------


def write_lock(url: str) -> None:
    try:
        APP_DATA_DIR.mkdir(parents=True, exist_ok=True)
        LOCK_FILE.write_text(
            json.dumps({"url": url, "pid": os.getpid()}), encoding="utf-8"
        )
    except OSError:
        logger.warning("Could not write instance lock at %s", LOCK_FILE)


def clear_lock() -> None:
    """Remove the lock, but only if this process owns it."""
    try:
        data = json.loads(LOCK_FILE.read_text(encoding="utf-8"))
        if data.get("pid") == os.getpid():
            LOCK_FILE.unlink()
    except (OSError, ValueError):
        pass


def find_running_instance() -> Optional[str]:
    """Return the URL of a live instance from a previous launch, if any."""
    try:
        data = json.loads(LOCK_FILE.read_text(encoding="utf-8"))
        url = data.get("url")
    except (OSError, ValueError):
        return None
    if not isinstance(url, str) or not url.startswith("http://127.0.0.1"):
        return None
    if wait_for_server(f"{url}/healthz", timeout=1.5, interval=0.3):
        return url
    return None  # stale lock from a crashed/killed instance


# --- window state -----------------------------------------------------------


class WindowState:
    """Size/position the window should reopen at — like a game remembering
    windowed-mode geometry between sessions instead of always centering."""

    __slots__ = ("width", "height", "x", "y")

    def __init__(
        self, width: int, height: int, x: Optional[int] = None, y: Optional[int] = None
    ) -> None:
        self.width = width
        self.height = height
        self.x = x
        self.y = y


def load_window_state(default_size: tuple[int, int]) -> WindowState:
    try:
        data = json.loads(WINDOW_STATE_FILE.read_text(encoding="utf-8"))
        width = int(data["width"])
        height = int(data["height"])
        x = data.get("x")
        y = data.get("y")
        return WindowState(
            width=max(width, MIN_WINDOW_SIZE[0]),
            height=max(height, MIN_WINDOW_SIZE[1]),
            x=int(x) if x is not None else None,
            y=int(y) if y is not None else None,
        )
    except (OSError, ValueError, KeyError, TypeError):
        return WindowState(width=default_size[0], height=default_size[1])


def save_window_state(state: WindowState) -> None:
    try:
        APP_DATA_DIR.mkdir(parents=True, exist_ok=True)
        WINDOW_STATE_FILE.write_text(
            json.dumps(
                {
                    "width": state.width,
                    "height": state.height,
                    "x": state.x,
                    "y": state.y,
                }
            ),
            encoding="utf-8",
        )
    except OSError:
        logger.warning("Could not save window state at %s", WINDOW_STATE_FILE)


def clamp_window_state(state: WindowState, screen_size: tuple[int, int]) -> WindowState:
    """Keep a restored window on-screen after a resolution change (e.g. a
    saved position from a since-unplugged second monitor)."""
    screen_w, screen_h = screen_size
    width = min(state.width, screen_w)
    height = min(state.height, screen_h)
    x, y = state.x, state.y
    if x is not None:
        x = max(0, min(x, max(0, screen_w - width)))
    if y is not None:
        y = max(0, min(y, max(0, screen_h - height)))
    return WindowState(width=width, height=height, x=x, y=y)


# --- shell plumbing --------------------------------------------------------


def _webview_available() -> bool:
    return importlib.util.find_spec("webview") is not None


def resolve_mode(prefer_window: bool) -> str:
    """Pick ``window`` or ``browser``, degrading when pywebview is absent."""
    if not prefer_window:
        return "browser"
    if _webview_available():
        return "window"
    logger.warning(
        'pywebview is not installed (pip install "amadeus-kurisu[desktop]"); '
        "falling back to the system browser."
    )
    return "browser"


def _configure_logging() -> None:
    """Log to stderr normally; to a file when frozen without a console."""
    handlers: list[logging.Handler] = []
    if getattr(sys, "frozen", False) and sys.stderr is None:
        try:
            APP_DATA_DIR.mkdir(parents=True, exist_ok=True)
            handlers.append(logging.FileHandler(LOG_FILE, encoding="utf-8"))
        except OSError:
            pass
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=handlers or None,
    )


def create_server(host: str, port: int):
    """Build (but don't start) a uvicorn server for the Amadeus web app.

    The app is passed as an import string so the module — and therefore
    ``Config.load`` — is only imported inside the server thread, after
    ``AMADEUS_CONFIG`` has been set from the CLI flags.
    """
    import uvicorn

    config = uvicorn.Config(
        APP_IMPORT, host=host, port=port, log_level="info", workers=1
    )
    return uvicorn.Server(config)


def _parse_window_size(value: str) -> tuple[int, int]:
    try:
        width, height = (int(part) for part in value.lower().split("x", 1))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"expected WIDTHxHEIGHT, got {value!r}"
        ) from exc
    if width < 320 or height < 240:
        raise argparse.ArgumentTypeError("window size must be at least 320x240")
    return width, height


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="amadeus-desktop",
        description="Run Amadeus as a local desktop app (no hosting required).",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to a config YAML (sets AMADEUS_CONFIG).",
    )
    parser.add_argument(
        "--host", default="127.0.0.1", help="Bind address (default: loopback only)."
    )
    parser.add_argument(
        "--port",
        type=int,
        default=0,
        help="Port to serve on (default: pick a free one).",
    )
    parser.add_argument(
        "--browser",
        action="store_true",
        help="Open the system browser instead of a native window.",
    )
    parser.add_argument(
        "--no-open",
        action="store_true",
        help="Start the server without opening any UI.",
    )
    parser.add_argument(
        "--window-size",
        type=_parse_window_size,
        default=DEFAULT_WINDOW_SIZE,
        metavar="WxH",
        help="Native window size (default: 1280x820).",
    )
    return parser


def _resolve_initial_state(state: WindowState) -> WindowState:
    """Clamp the persisted/requested window state to the primary screen.

    Best-effort: some platforms can only report screen size once the GUI
    backend is initialized, which hasn't happened yet at this point — any
    failure just means the raw (already MIN_WINDOW_SIZE-floored) state is
    used unclamped, not that window creation fails.
    """
    import webview

    try:
        screens = webview.screens()
        if screens:
            return clamp_window_state(state, (screens[0].width, screens[0].height))
    except Exception:  # pragma: no cover - platform/backend dependent
        pass
    return state


def _run_window(url: str, state: WindowState, server_ready: threading.Event) -> None:
    """Open the window immediately on a splash, then navigate when ready.

    ``server_ready`` is pre-set when attaching to an already-running
    instance, in which case the splash is skipped entirely. Window
    size/position are persisted on resize/move (debounced) and on close,
    so relaunching reopens at the same geometry — like a game remembering
    windowed-mode settings instead of always recentering.
    """
    import webview

    state = _resolve_initial_state(state)
    window_kwargs = dict(
        width=state.width, height=state.height, min_size=MIN_WINDOW_SIZE
    )
    if state.x is not None and state.y is not None:
        window_kwargs.update(x=state.x, y=state.y)

    if server_ready.is_set():
        window = webview.create_window(WINDOW_TITLE, url, **window_kwargs)
    else:
        window = webview.create_window(WINDOW_TITLE, html=SPLASH_HTML, **window_kwargs)

        def navigate_when_ready() -> None:
            if wait_for_server(f"{url}/healthz", timeout=STARTUP_TIMEOUT):
                server_ready.set()
                window.load_url(url)
            else:
                logger.error("Backend did not become healthy at %s", url)
                window.load_html(
                    ERROR_HTML.format(
                        message="The local core did not come up in time.",
                        log_path=str(LOG_FILE),
                    )
                )

    # Debounced persistence: dragging/resizing fires many events per second,
    # and each pywebview event runs on its own thread (see webview.event.Event
    # .set), so this needs its own lock rather than relying on call order.
    save_timer_lock = threading.Lock()
    save_timer: list[Optional[threading.Timer]] = [None]
    live_state = WindowState(state.width, state.height, state.x, state.y)

    def schedule_save() -> None:
        with save_timer_lock:
            if save_timer[0] is not None:
                save_timer[0].cancel()
            timer = threading.Timer(0.5, lambda: save_window_state(live_state))
            timer.daemon = True
            save_timer[0] = timer
            timer.start()

    def on_resized(width: int, height: int) -> None:
        live_state.width, live_state.height = width, height
        schedule_save()

    def on_moved(x: int, y: int) -> None:
        live_state.x, live_state.y = x, y
        schedule_save()

    window.events.resized += on_resized
    window.events.moved += on_moved
    window.events.closing += lambda: save_window_state(live_state)

    if server_ready.is_set():
        webview.start()
    else:
        webview.start(navigate_when_ready)


def _wait_forever(thread: threading.Thread) -> None:
    try:
        while thread.is_alive():
            thread.join(0.5)
    except KeyboardInterrupt:
        pass


def main(argv: Optional[Sequence[str]] = None) -> int:
    _configure_logging()
    args = build_parser().parse_args(argv)

    if args.config is not None:
        config_path = args.config.expanduser().resolve()
        if not config_path.exists():
            logger.error("Config file not found: %s", config_path)
            return 2
        os.environ["AMADEUS_CONFIG"] = str(config_path)

    mode = "server" if args.no_open else resolve_mode(prefer_window=not args.browser)

    # An explicit --window-size overrides whatever was persisted from a
    # previous session; otherwise reopen at the last size/position, like a
    # game remembering windowed-mode geometry between launches.
    if args.window_size != DEFAULT_WINDOW_SIZE:
        window_state = WindowState(
            width=args.window_size[0], height=args.window_size[1]
        )
    else:
        window_state = load_window_state(DEFAULT_WINDOW_SIZE)

    # Single instance: a second UI launch attaches to the running app
    # instead of booting a duplicate server. --no-open (server-only) is
    # exempt so containers/tests can run several deliberately.
    if mode != "server":
        existing = find_running_instance()
        if existing is not None:
            logger.info("Amadeus is already running at %s — attaching.", existing)
            if mode == "window":
                ready = threading.Event()
                ready.set()
                _run_window(existing, window_state, ready)
            else:
                webbrowser.open(existing)
            return 0

    host = args.host
    port = args.port or find_free_port(host)
    url = f"http://{host}:{port}"

    server = create_server(host, port)
    server_thread = threading.Thread(
        target=server.run, name="amadeus-uvicorn", daemon=True
    )
    server_thread.start()

    if mode != "server":
        write_lock(url)

    try:
        if mode == "window":
            # The window opens on the splash immediately; health-gating and
            # navigation happen on pywebview's background thread.
            _run_window(url, window_state, threading.Event())
        else:
            if not wait_for_server(f"{url}/healthz", timeout=STARTUP_TIMEOUT):
                logger.error(
                    "Backend did not become healthy at %s — see logs above.", url
                )
                return 1
            logger.info("Amadeus is running at %s", url)
            if mode == "browser":
                webbrowser.open(url)
                logger.info("Press Ctrl+C to stop.")
            _wait_forever(server_thread)
    finally:
        clear_lock()
        server.should_exit = True
        server_thread.join(timeout=10)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
