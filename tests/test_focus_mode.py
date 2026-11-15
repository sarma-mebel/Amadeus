"""Focus / co-working mode: window math + clamping, presence suppression
(idle chatter, ambient, reach-out hold), the exactly-once check-in
(including SKIP and provider-failure paths), and the /api/focus + /api/status
HTTP surface. Structurally cloned from tests/test_sleep_schedule.py and
tests/test_reachout.py."""

from __future__ import annotations

import datetime as dt

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from amadeus.config import Config
from amadeus.engine import ChatEngine
from amadeus.focus_mode import (
    MAX_MINUTES,
    MIN_MINUTES,
    clamp_minutes,
    end_focus,
    focus_config_status,
    focus_status,
    is_focus_active,
    start_focus,
)
from amadeus.presence import ambient_comment, idle_chatter
from amadeus.providers.llm.base import ChatMessage, DummyProvider
from amadeus.reachout import ReachOut, ReachOutStore
from amadeus.web.app import create_app
from amadeus import telemetry


class _RecordingProvider:
    name = "recording"

    def __init__(self, reply: str = "Good to have you back.") -> None:
        self.reply = reply
        self.calls: list[list[ChatMessage]] = []
        self.fail = False

    def generate(self, messages, **kwargs):
        self.calls.append(list(messages))
        if self.fail:
            raise RuntimeError("provider down")
        return self.reply


def _engine(tmp_path) -> ChatEngine:
    config = Config()
    config.memory.persist_path = str(tmp_path / "memory.json")
    config.memory.embeddings_path = None
    config.memory.profile_path = str(tmp_path / "profile.json")
    config.memory.enable_idle_chatter = True
    # The sleep window check runs BEFORE the focus check in presence.py, so
    # a 02:00-08:00 UTC CI run counts presence.sleep_suppressed instead of
    # presence.focus_suppressed and every assertion here breaks. Focus tests
    # must see an awake engine at any wall-clock hour.
    config.sleep_schedule.enabled = False
    engine = ChatEngine(config, DummyProvider())
    engine.summary_provider = _RecordingProvider()
    return engine


# ---------------------------------------------------------------------------
# Clamping
# ---------------------------------------------------------------------------


def test_clamp_minutes_bounds():
    assert clamp_minutes(5, default=50) == MIN_MINUTES
    assert clamp_minutes(500, default=50) == MAX_MINUTES
    assert clamp_minutes(30, default=50) == 30
    assert clamp_minutes("not a number", default=50) == 50
    assert clamp_minutes(None, default=50) == 50


# ---------------------------------------------------------------------------
# Lifecycle: start / status / end
# ---------------------------------------------------------------------------


def test_start_clamps_and_activates(tmp_path):
    engine = _engine(tmp_path)
    now = dt.datetime(2026, 8, 3, 10, 0)
    state = start_focus(engine, 5, now=now)  # below MIN_MINUTES
    assert state["active"] is True
    assert state["remaining_seconds"] == MIN_MINUTES * 60
    assert state["checkin"] is None
    assert is_focus_active(engine, now=now) is True
    assert telemetry.snapshot().get("focus.sessions_started", 0) >= 1


def test_status_mid_session_no_checkin(tmp_path):
    engine = _engine(tmp_path)
    now = dt.datetime(2026, 8, 3, 10, 0)
    start_focus(engine, 30, now=now)
    state = focus_status(engine, now=now + dt.timedelta(minutes=5))
    assert state["active"] is True
    assert state["remaining_seconds"] == 25 * 60
    assert state["checkin"] is None


def test_end_early_delivers_one_checkin_then_none(tmp_path):
    engine = _engine(tmp_path)
    now = dt.datetime(2026, 8, 3, 10, 0)
    start_focus(engine, 50, now=now)
    ended = end_focus(engine, now=now + dt.timedelta(minutes=10))
    assert ended["active"] is False
    assert ended["until"] is None
    assert ended["checkin"] == "Good to have you back."

    # Exactly once: a second status call gets nothing further.
    again = focus_status(engine, now=now + dt.timedelta(minutes=11))
    assert again["checkin"] is None
    assert again["active"] is False


def test_end_with_no_active_session_is_a_noop(tmp_path):
    engine = _engine(tmp_path)
    state = end_focus(engine)
    assert state["active"] is False
    assert state["checkin"] is None
    assert engine.summary_provider.calls == []  # never called the model


def test_status_poll_observes_natural_expiry_exactly_once(tmp_path):
    """Nobody calls 'end' explicitly — a later status/poll call is the one
    that notices the deadline passed, and delivers the check-in exactly
    like an explicit end would."""
    engine = _engine(tmp_path)
    now = dt.datetime(2026, 8, 3, 10, 0)
    start_focus(engine, 10, now=now)
    later = now + dt.timedelta(minutes=11)

    first = focus_status(engine, now=later)
    assert first["active"] is False
    assert first["checkin"] == "Good to have you back."

    second = focus_status(engine, now=later)
    assert second["checkin"] is None


# ---------------------------------------------------------------------------
# Check-in content: SKIP, failure, disabled
# ---------------------------------------------------------------------------


def test_checkin_skip_yields_null_and_no_failure_count(tmp_path):
    engine = _engine(tmp_path)
    engine.summary_provider = _RecordingProvider(reply="SKIP")
    now = dt.datetime(2026, 8, 3, 10, 0)
    start_focus(engine, 10, now=now)
    before = telemetry.snapshot().get("focus.checkin_failures", 0)
    ended = end_focus(engine, now=now)
    assert ended["checkin"] is None
    assert telemetry.snapshot().get("focus.checkin_failures", 0) == before


def test_checkin_provider_failure_counts_and_yields_null(tmp_path):
    engine = _engine(tmp_path)
    engine.summary_provider = _RecordingProvider()
    engine.summary_provider.fail = True
    now = dt.datetime(2026, 8, 3, 10, 0)
    start_focus(engine, 10, now=now)
    before = telemetry.snapshot().get("focus.checkin_failures", 0)
    ended = end_focus(engine, now=now)
    assert ended["checkin"] is None
    assert telemetry.snapshot().get("focus.checkin_failures", 0) == before + 1


def test_checkin_disabled_in_config_yields_null(tmp_path):
    engine = _engine(tmp_path)
    engine.config.focus.checkin_enabled = False
    now = dt.datetime(2026, 8, 3, 10, 0)
    start_focus(engine, 10, now=now)
    ended = end_focus(engine, now=now)
    assert ended["checkin"] is None
    assert engine.summary_provider.calls == []


def test_checkin_prompt_bans_productivity_questions(tmp_path):
    engine = _engine(tmp_path)
    now = dt.datetime(2026, 8, 3, 10, 0)
    start_focus(engine, 10, now=now)
    end_focus(engine, now=now)
    system = engine.summary_provider.calls[0][0].content
    assert "NEVER ask whether they finished" in system
    assert "SKIP" in system


# ---------------------------------------------------------------------------
# Suppression: idle chatter, ambient, reach-out hold
# ---------------------------------------------------------------------------


def test_idle_chatter_suppressed_while_focused(tmp_path):
    engine = _engine(tmp_path)
    engine.memory.add_message("user", "hello")
    engine.memory.add_message("assistant", "hi")
    start_focus(engine, 30)
    before = telemetry.snapshot().get("presence.focus_suppressed", 0)
    assert idle_chatter(engine) is None
    assert telemetry.snapshot().get("presence.focus_suppressed", 0) == before + 1


def test_ambient_comment_suppressed_while_focused(tmp_path):
    engine = _engine(tmp_path)
    start_focus(engine, 30)
    before = telemetry.snapshot().get("presence.focus_suppressed", 0)
    assert ambient_comment(engine, "music", "New track: Foo") is None
    assert telemetry.snapshot().get("presence.focus_suppressed", 0) == before + 1


def test_idle_chatter_resumes_after_focus_ends(tmp_path):
    engine = _engine(tmp_path)
    engine.memory.add_message("user", "hello")
    engine.memory.add_message("assistant", "hi")
    now = dt.datetime(2026, 8, 3, 10, 0)
    start_focus(engine, 10, now=now)
    end_focus(engine, now=now)
    assert is_focus_active(engine, now=now) is False
    assert idle_chatter(engine) == "Good to have you back."


# ---------------------------------------------------------------------------
# HTTP surface
# ---------------------------------------------------------------------------


def _http_config(tmp_path) -> Config:
    config = Config()
    config.memory.persist_path = str(tmp_path / "memory.json")
    config.memory.embeddings_path = str(tmp_path / "embeddings.json")
    config.memory.profile_path = str(tmp_path / "profile.json")
    # Same as _engine: focus behavior must be observable at any CI hour.
    config.sleep_schedule.enabled = False
    return config


def test_endpoint_start_end_status_lifecycle(monkeypatch, tmp_path):
    config = _http_config(tmp_path)
    recorder = _RecordingProvider()
    import amadeus.web.app as app_module

    monkeypatch.setattr(app_module, "load_llm", lambda cfg: recorder)
    app = create_app(config)
    store = app.state.session_store
    store.llm = recorder
    client = TestClient(app)

    started = client.post("/api/focus", json={"action": "start", "minutes": 25}).json()
    assert started["active"] is True
    assert started["remaining_seconds"] == 25 * 60
    session_id = started["session_id"]

    status = client.post(
        "/api/focus", json={"action": "status", "session_id": session_id}
    ).json()
    assert status["active"] is True

    ended = client.post(
        "/api/focus", json={"action": "end", "session_id": session_id}
    ).json()
    assert ended["active"] is False
    assert ended["checkin"]

    again = client.post(
        "/api/focus", json={"action": "status", "session_id": session_id}
    ).json()
    assert again["checkin"] is None


def test_endpoint_clamps_minutes(monkeypatch, tmp_path):
    config = _http_config(tmp_path)
    recorder = _RecordingProvider()
    import amadeus.web.app as app_module

    monkeypatch.setattr(app_module, "load_llm", lambda cfg: recorder)
    app = create_app(config)
    app.state.session_store.llm = recorder
    client = TestClient(app)

    started = client.post("/api/focus", json={"action": "start", "minutes": 1}).json()
    assert started["remaining_seconds"] == MIN_MINUTES * 60


def test_endpoint_disabled_config_returns_404(tmp_path):
    config = _http_config(tmp_path)
    config.focus.enabled = False
    app = create_app(config)
    client = TestClient(app)
    response = client.post("/api/focus", json={"action": "status"})
    assert response.status_code == 404


def test_reachout_held_not_consumed_while_focused(monkeypatch, tmp_path):
    """Mirrors test_reachout's sleep-window hold test: a pending reach-out
    is HELD, not popped, while a focus session is active."""
    config = _http_config(tmp_path)
    config.reach_out.enabled = True
    config.reach_out.store_path = str(tmp_path / "reachout.json")
    recorder = _RecordingProvider()
    import amadeus.web.app as app_module

    monkeypatch.setattr(app_module, "load_llm", lambda cfg: recorder)
    app = create_app(config)
    store = app.state.session_store
    store.llm = recorder
    client = TestClient(app)

    today = dt.date.today()
    store.reachout_store.put(
        None,
        ReachOut(
            date=today.isoformat(),
            text="Did it work out?",
            generated_at=f"{today.isoformat()}T17:00:00",
        ),
    )

    started = client.post("/api/focus", json={"action": "start", "minutes": 30}).json()
    session_id = started["session_id"]

    held = client.post("/api/reachout/due", json={"session_id": session_id}).json()
    assert held["reach_out"] is None

    # Not consumed: ending focus lets the same poll deliver it.
    client.post("/api/focus", json={"action": "end", "session_id": session_id})
    delivered = client.post("/api/reachout/due", json={"session_id": session_id}).json()
    assert delivered["reach_out"] == "Did it work out?"


def test_status_endpoint_reports_focus_config(tmp_path):
    config = _http_config(tmp_path)
    config.focus.default_minutes = 45
    app = create_app(config)
    client = TestClient(app)
    payload = client.get("/api/status").json()
    assert payload["focus"] == {"enabled": True, "default_minutes": 45}


def test_focus_config_status_pure_function():
    config = Config()
    config.focus.enabled = False
    config.focus.default_minutes = 90
    assert focus_config_status(config) == {"enabled": False, "default_minutes": 90}
