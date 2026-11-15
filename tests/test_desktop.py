"""Tests for the desktop shell (amadeus.desktop_webview). No GUI, no live network."""

import http.server
import socket
import threading

import pytest

from amadeus import desktop_webview as desktop


def test_find_free_port_is_bindable():
    port = desktop.find_free_port()
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", port))


def test_wait_for_server_success():
    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.end_headers()

        def log_message(self, *args):
            pass

    server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        port = server.server_address[1]
        assert desktop.wait_for_server(f"http://127.0.0.1:{port}/healthz", timeout=5)
    finally:
        server.shutdown()


def test_wait_for_server_timeout_on_dead_port():
    port = desktop.find_free_port()
    assert not desktop.wait_for_server(
        f"http://127.0.0.1:{port}/healthz", timeout=0.7, interval=0.1
    )


def test_resolve_mode_prefers_window_when_webview_present(monkeypatch):
    monkeypatch.setattr(desktop, "_webview_available", lambda: True)
    assert desktop.resolve_mode(prefer_window=True) == "window"


def test_resolve_mode_falls_back_without_webview(monkeypatch):
    monkeypatch.setattr(desktop, "_webview_available", lambda: False)
    assert desktop.resolve_mode(prefer_window=True) == "browser"


def test_resolve_mode_browser_when_requested(monkeypatch):
    monkeypatch.setattr(desktop, "_webview_available", lambda: True)
    assert desktop.resolve_mode(prefer_window=False) == "browser"


def test_parser_defaults():
    args = desktop.build_parser().parse_args([])
    assert args.host == "127.0.0.1"
    assert args.port == 0
    assert not args.browser
    assert not args.no_open
    assert args.window_size == desktop.DEFAULT_WINDOW_SIZE


def test_parser_window_size():
    args = desktop.build_parser().parse_args(["--window-size", "1024x768"])
    assert args.window_size == (1024, 768)


def test_parser_rejects_bad_window_size():
    with pytest.raises(SystemExit):
        desktop.build_parser().parse_args(["--window-size", "huge"])


def test_main_errors_on_missing_config(tmp_path):
    missing = tmp_path / "nope.yaml"
    assert desktop.main(["--config", str(missing), "--no-open"]) == 2


def _use_tmp_window_state(monkeypatch, tmp_path):
    monkeypatch.setattr(desktop, "APP_DATA_DIR", tmp_path)
    monkeypatch.setattr(desktop, "WINDOW_STATE_FILE", tmp_path / "window.json")


def test_load_window_state_defaults_when_no_file(monkeypatch, tmp_path):
    _use_tmp_window_state(monkeypatch, tmp_path)
    state = desktop.load_window_state((1280, 820))
    assert (state.width, state.height) == (1280, 820)
    assert state.x is None and state.y is None


def test_save_and_load_window_state_roundtrip(monkeypatch, tmp_path):
    _use_tmp_window_state(monkeypatch, tmp_path)
    saved = desktop.WindowState(width=1500, height=900, x=200, y=80)
    desktop.save_window_state(saved)

    loaded = desktop.load_window_state((1280, 820))
    assert (loaded.width, loaded.height, loaded.x, loaded.y) == (1500, 900, 200, 80)


def test_load_window_state_ignores_corrupt_file(monkeypatch, tmp_path):
    _use_tmp_window_state(monkeypatch, tmp_path)
    desktop.WINDOW_STATE_FILE.write_text("{not json", encoding="utf-8")
    state = desktop.load_window_state((1280, 820))
    assert (state.width, state.height) == (1280, 820)


def test_load_window_state_floors_to_minimum_size(monkeypatch, tmp_path):
    _use_tmp_window_state(monkeypatch, tmp_path)
    desktop.save_window_state(desktop.WindowState(width=10, height=10))
    state = desktop.load_window_state((1280, 820))
    assert state.width == desktop.MIN_WINDOW_SIZE[0]
    assert state.height == desktop.MIN_WINDOW_SIZE[1]


def test_clamp_window_state_shrinks_oversized_window():
    state = desktop.WindowState(width=3000, height=2000, x=100, y=100)
    clamped = desktop.clamp_window_state(state, screen_size=(1920, 1080))
    assert clamped.width == 1920
    assert clamped.height == 1080


def test_clamp_window_state_pulls_offscreen_position_back_in():
    # A position saved on a second monitor that's since been unplugged.
    state = desktop.WindowState(width=800, height=600, x=2500, y=1400)
    clamped = desktop.clamp_window_state(state, screen_size=(1920, 1080))
    assert 0 <= clamped.x <= 1920 - 800
    assert 0 <= clamped.y <= 1080 - 600


def test_clamp_window_state_leaves_unpositioned_window_alone():
    state = desktop.WindowState(width=800, height=600)
    clamped = desktop.clamp_window_state(state, screen_size=(1920, 1080))
    assert clamped.x is None and clamped.y is None


def _use_tmp_lock(monkeypatch, tmp_path):
    monkeypatch.setattr(desktop, "APP_DATA_DIR", tmp_path)
    monkeypatch.setattr(desktop, "LOCK_FILE", tmp_path / "desktop.lock")


def test_lock_roundtrip_and_owner_clear(monkeypatch, tmp_path):
    _use_tmp_lock(monkeypatch, tmp_path)
    desktop.write_lock("http://127.0.0.1:12345")
    assert desktop.LOCK_FILE.exists()
    desktop.clear_lock()  # owned by this pid -> removed
    assert not desktop.LOCK_FILE.exists()


def test_clear_lock_keeps_foreign_lock(monkeypatch, tmp_path):
    _use_tmp_lock(monkeypatch, tmp_path)
    desktop.LOCK_FILE.write_text('{"url": "http://127.0.0.1:1", "pid": -1}')
    desktop.clear_lock()
    assert desktop.LOCK_FILE.exists()


def test_find_running_instance_none_without_lock(monkeypatch, tmp_path):
    _use_tmp_lock(monkeypatch, tmp_path)
    assert desktop.find_running_instance() is None


def test_find_running_instance_ignores_stale_lock(monkeypatch, tmp_path):
    _use_tmp_lock(monkeypatch, tmp_path)
    dead_port = desktop.find_free_port()
    desktop.write_lock(f"http://127.0.0.1:{dead_port}")
    assert desktop.find_running_instance() is None


def test_find_running_instance_rejects_non_loopback(monkeypatch, tmp_path):
    _use_tmp_lock(monkeypatch, tmp_path)
    desktop.LOCK_FILE.write_text('{"url": "http://example.com", "pid": 1}')
    assert desktop.find_running_instance() is None


def test_find_running_instance_finds_live_server(monkeypatch, tmp_path):
    _use_tmp_lock(monkeypatch, tmp_path)

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.end_headers()

        def log_message(self, *args):
            pass

    server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        url = f"http://127.0.0.1:{server.server_address[1]}"
        desktop.write_lock(url)
        assert desktop.find_running_instance() == url
    finally:
        server.shutdown()


def test_splash_and_error_templates_render():
    assert "AMADEUS" in desktop.SPLASH_HTML
    rendered = desktop.ERROR_HTML.format(message="boom", log_path="/tmp/x.log")
    assert "boom" in rendered and "/tmp/x.log" in rendered
