"""Tests for the presence features: poke reactions and idle chatter.

Both reuse the greeting's server-gated, cheap-model pattern (see
tests/test_greeting_continuity.py) but add server-side cooldowns, since
these are triggered by client-side events (a click, an idle timer) that a
buggy or malicious client could otherwise fire in a tight loop.
"""

import datetime as dt

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from amadeus.config import Config
from amadeus.providers.llm.base import ChatMessage
from amadeus.web.app import create_app


class RecordingProvider:
    name = "recording"

    def __init__(self, reply: str = "Don't touch me."):
        self.reply = reply
        self.calls: list[list[ChatMessage]] = []

    def generate(self, messages, **kwargs):
        self.calls.append(list(messages))
        return self.reply


def _isolated_config(tmp_path) -> Config:
    config = Config()
    config.memory.persist_path = str(tmp_path / "memory.json")
    config.memory.embeddings_path = str(tmp_path / "embeddings.json")
    config.memory.profile_path = str(tmp_path / "profile.json")
    # Presence lines are suppressed inside the sleep window (default 1-8
    # LOCAL) — CI runners sit in UTC, so a 02:00 UTC run lands inside it
    # and every "line fires" assertion here goes dark. Tests must pass at
    # any wall-clock hour; sleep behavior is tested where it's enabled
    # explicitly (test_sleep_schedule.py).
    config.sleep_schedule.enabled = False
    return config


def _client_with_session(tmp_path, *, enable_poke=False, enable_idle=False):
    config = _isolated_config(tmp_path)
    config.memory.enable_poke_reactions = enable_poke
    config.memory.enable_idle_chatter = enable_idle
    config.memory.poke_cooldown_seconds = 5.0
    config.memory.idle_cooldown_seconds = 5.0
    app = create_app(config)
    store = app.state.session_store
    recorder = RecordingProvider()
    _, session = store.get(None, "user-1")
    session.llm = recorder
    session.summary_provider = recorder
    return TestClient(app), session, recorder


# --- poke --------------------------------------------------------------


def test_poke_disabled_by_default(tmp_path):
    client, session, _ = _client_with_session(tmp_path)
    response = client.post("/api/poke", json={"session_id": None, "user_id": "user-1"})
    assert response.status_code == 200
    assert response.json()["reaction"] is None


def test_poke_returns_reaction_when_enabled(tmp_path):
    client, session, recorder = _client_with_session(tmp_path, enable_poke=True)
    _, session_obj = list(client.app.state.session_store.sessions.items())[0]  # type: ignore[attr-defined]
    session_obj.llm = recorder
    session_obj.summary_provider = recorder

    response = client.post("/api/poke", json={"session_id": None, "user_id": "user-1"})
    body = response.json()
    assert body["reaction"] == "Don't touch me."
    assert "emotion" in body


def test_poke_prompt_does_not_ask_a_real_question(tmp_path):
    """Sanity check on the prompt itself: a poke reaction should read as
    flavor, not as the assistant fishing for the next message."""
    client, session, recorder = _client_with_session(tmp_path, enable_poke=True)
    _, session_obj = list(client.app.state.session_store.sessions.items())[0]  # type: ignore[attr-defined]
    session_obj.llm = recorder
    session_obj.summary_provider = recorder

    client.post("/api/poke", json={"session_id": None, "user_id": "user-1"})
    assert recorder.calls
    system_prompt = recorder.calls[0][0].content
    assert "poked" in system_prompt.lower()


def test_poke_region_reaches_the_prompt(tmp_path):
    """A region-aware poke (head-pat etc.) should produce a region-specific
    situation in the system prompt, not the generic poke line."""
    client, session, recorder = _client_with_session(tmp_path, enable_poke=True)
    _, session_obj = list(client.app.state.session_store.sessions.items())[0]  # type: ignore[attr-defined]
    session_obj.llm = recorder
    session_obj.summary_provider = recorder

    response = client.post(
        "/api/poke", json={"session_id": None, "user_id": "user-1", "region": "head"}
    )
    assert response.json()["reaction"] == "Don't touch me."
    system_prompt = recorder.calls[0][0].content
    assert "HEAD" in system_prompt
    assert recorder.calls[0][1].content == "(poked: head)"


def test_poke_rejects_unknown_region(tmp_path):
    client, session, recorder = _client_with_session(tmp_path, enable_poke=True)
    response = client.post(
        "/api/poke", json={"session_id": None, "user_id": "user-1", "region": "elbow"}
    )
    assert response.status_code == 422


@pytest.mark.parametrize(
    "region", ["head_pat", "dragged", "click_through", "woke_from_doze"]
)
def test_poke_accepts_pet_event_regions(tmp_path, region):
    """The pet-window event regions must clear request validation — the pet
    UI swallows poke failures by design, so a region missing from the
    PresenceRequest pattern fails silently (this bit the dragged /
    click_through reactions, which shipped 422ing on every fire)."""
    client, session, recorder = _client_with_session(tmp_path, enable_poke=True)
    _, session_obj = list(client.app.state.session_store.sessions.items())[0]  # type: ignore[attr-defined]
    session_obj.llm = recorder
    session_obj.summary_provider = recorder

    response = client.post(
        "/api/poke", json={"session_id": None, "user_id": "user-1", "region": region}
    )
    assert response.status_code == 200
    assert response.json()["reaction"] == "Don't touch me."
    # Every pet-event region has a bespoke situation, never the generic line.
    system_prompt = recorder.calls[0][0].content
    assert "poked/clicked you on screen" not in system_prompt


def test_head_pat_tone_follows_relationship_warmth(tmp_path):
    """The head-pat situation shifts with the persisted relationship arc:
    cold profiles get the indignant framing, thawed ones the begrudgingly
    pleased framing (formula mirrored in the frontend's arcWarmth)."""
    client, session, recorder = _client_with_session(tmp_path, enable_poke=True)
    _, session_obj = list(client.app.state.session_store.sessions.items())[0]  # type: ignore[attr-defined]
    session_obj.llm = recorder
    session_obj.summary_provider = recorder
    profile = session_obj.memory.profile
    assert profile is not None

    profile.trust_level = 0.0
    profile.arc_curiosity_vs_annoyance = -1.0
    client.post(
        "/api/poke",
        json={"session_id": None, "user_id": "user-1", "region": "head_pat"},
    )
    cold_prompt = recorder.calls[-1][0].content
    assert "barely know them" in cold_prompt

    session_obj._last_poke_at = None  # reset cooldown between probes
    profile.trust_level = 1.0
    profile.arc_curiosity_vs_annoyance = 1.0
    client.post(
        "/api/poke",
        json={"session_id": None, "user_id": "user-1", "region": "head_pat"},
    )
    warm_prompt = recorder.calls[-1][0].content
    assert "genuinely comfortable" in warm_prompt
    assert cold_prompt != warm_prompt


def test_woke_from_doze_prompt_mentions_dozing(tmp_path):
    client, session, recorder = _client_with_session(tmp_path, enable_poke=True)
    _, session_obj = list(client.app.state.session_store.sessions.items())[0]  # type: ignore[attr-defined]
    session_obj.llm = recorder
    session_obj.summary_provider = recorder

    client.post(
        "/api/poke",
        json={"session_id": None, "user_id": "user-1", "region": "woke_from_doze"},
    )
    assert "dozed off" in recorder.calls[0][0].content


def test_poke_without_region_keeps_generic_prompt(tmp_path):
    client, session, recorder = _client_with_session(tmp_path, enable_poke=True)
    _, session_obj = list(client.app.state.session_store.sessions.items())[0]  # type: ignore[attr-defined]
    session_obj.llm = recorder
    session_obj.summary_provider = recorder

    client.post("/api/poke", json={"session_id": None, "user_id": "user-1"})
    system_prompt = recorder.calls[0][0].content
    assert "poked/clicked you on screen" in system_prompt
    assert recorder.calls[0][1].content == "(poked)"


def test_poke_respects_cooldown(tmp_path):
    client, session, recorder = _client_with_session(tmp_path, enable_poke=True)
    _, session_obj = list(client.app.state.session_store.sessions.items())[0]  # type: ignore[attr-defined]
    session_obj.llm = recorder
    session_obj.summary_provider = recorder

    first = client.post("/api/poke", json={"session_id": None, "user_id": "user-1"})
    assert first.json()["reaction"] is not None

    second = client.post("/api/poke", json={"session_id": None, "user_id": "user-1"})
    assert second.json()["reaction"] is None
    assert len(recorder.calls) == 1  # cooldown blocked a second LLM call


def test_poke_reaction_not_committed_to_memory(tmp_path):
    """A poke is throwaway flavor, not a real turn worth summarizing later."""
    client, session, recorder = _client_with_session(tmp_path, enable_poke=True)
    _, session_obj = list(client.app.state.session_store.sessions.items())[0]  # type: ignore[attr-defined]
    session_obj.llm = recorder
    session_obj.summary_provider = recorder

    before = len(session_obj.memory.short_term)
    client.post("/api/poke", json={"session_id": None, "user_id": "user-1"})
    assert len(session_obj.memory.short_term) == before


# --- idle chatter --------------------------------------------------------


def test_idle_disabled_by_default(tmp_path):
    client, session, _ = _client_with_session(tmp_path)
    response = client.post("/api/idle", json={"session_id": None, "user_id": "user-1"})
    assert response.json()["chatter"] is None


def test_idle_skipped_on_fresh_session_with_no_history(tmp_path):
    """A brand-new, silent session should get a greeting, not idle chatter —
    idle_chatter only fires when there's an actual conversation to riff on."""
    client, session, recorder = _client_with_session(tmp_path, enable_idle=True)
    _, session_obj = list(client.app.state.session_store.sessions.items())[0]  # type: ignore[attr-defined]
    session_obj.llm = recorder
    session_obj.summary_provider = recorder

    response = client.post("/api/idle", json={"session_id": None, "user_id": "user-1"})
    assert response.json()["chatter"] is None
    assert not recorder.calls


def test_idle_fires_after_existing_conversation(tmp_path):
    client, session, recorder = _client_with_session(tmp_path, enable_idle=True)
    _, session_obj = list(client.app.state.session_store.sessions.items())[0]  # type: ignore[attr-defined]
    session_obj.llm = recorder
    session_obj.summary_provider = recorder
    recorder.reply = "Still there?"

    session_obj.memory.add_message("user", "let's talk about time travel")
    session_obj.memory.add_message("assistant", "Fine.")
    session_obj.kurisu_state.last_topic = "time travel"

    response = client.post("/api/idle", json={"session_id": None, "user_id": "user-1"})
    body = response.json()
    assert body["chatter"] == "Still there?"

    prompt_text = "\n".join(m.content for m in recorder.calls[0])
    assert "time travel" in prompt_text


def test_idle_chatter_is_committed_to_memory(tmp_path):
    client, session, recorder = _client_with_session(tmp_path, enable_idle=True)
    _, session_obj = list(client.app.state.session_store.sessions.items())[0]  # type: ignore[attr-defined]
    session_obj.llm = recorder
    session_obj.summary_provider = recorder
    session_obj.memory.add_message("user", "hello")

    before = len(session_obj.memory.short_term)
    client.post("/api/idle", json={"session_id": None, "user_id": "user-1"})
    assert len(session_obj.memory.short_term) == before + 1


def test_idle_respects_cooldown(tmp_path):
    client, session, recorder = _client_with_session(tmp_path, enable_idle=True)
    _, session_obj = list(client.app.state.session_store.sessions.items())[0]  # type: ignore[attr-defined]
    session_obj.llm = recorder
    session_obj.summary_provider = recorder
    session_obj.memory.add_message("user", "hello")

    first = client.post("/api/idle", json={"session_id": None, "user_id": "user-1"})
    assert first.json()["chatter"] is not None

    second = client.post("/api/idle", json={"session_id": None, "user_id": "user-1"})
    assert second.json()["chatter"] is None
    assert len(recorder.calls) == 1


# --- engine-level cooldown timing (no HTTP layer) -----------------------


def test_poke_cooldown_expires_after_configured_seconds(tmp_path):
    config = _isolated_config(tmp_path)
    config.memory.enable_poke_reactions = True
    config.memory.poke_cooldown_seconds = 60.0
    app = create_app(config)
    store = app.state.session_store
    recorder = RecordingProvider()
    _, session = store.get(None, "user-x")
    session.llm = recorder
    session.summary_provider = recorder

    assert session.poke_reaction() is not None
    assert session.poke_reaction() is None  # within cooldown

    # Simulate the cooldown having elapsed.
    session._last_poke_at = dt.datetime.utcnow() - dt.timedelta(seconds=61)
    assert session.poke_reaction() is not None


# --- ambient desktop awareness -----------------------------------------


def test_ambient_comment_generates_line_and_prompt_shape(tmp_path):
    client, session, recorder = _client_with_session(tmp_path)
    data = client.post(
        "/api/ambient",
        json={
            "user_id": "user-1",
            "kind": "music",
            "detail": 'The user is listening to "Hacking to the Gate".',
        },
    ).json()
    assert data["comment"] == recorder.reply
    system = recorder.calls[0][0].content
    assert "Hacking to the Gate" in system
    assert "SKIP" in system
    assert (
        "noticing often and speaking rarely" in system.lower()
        or "speaking rarely" in system
    )


def test_ambient_rejects_unknown_kind_and_blank_detail(tmp_path):
    client, session, recorder = _client_with_session(tmp_path)
    assert (
        client.post(
            "/api/ambient", json={"user_id": "user-1", "kind": "webcam", "detail": "x"}
        ).status_code
        == 422
    )
    assert (
        client.post(
            "/api/ambient", json={"user_id": "user-1", "kind": "focus", "detail": ""}
        ).status_code
        == 422
    )
    assert recorder.calls == []


def test_ambient_respects_cooldown_and_skip(tmp_path):
    client, session, recorder = _client_with_session(tmp_path)
    session.config.ambient.cooldown_seconds = 600.0
    first = client.post(
        "/api/ambient",
        json={"user_id": "user-1", "kind": "battery", "detail": "Battery at 14%."},
    ).json()
    assert first["comment"] is not None
    second = client.post(
        "/api/ambient",
        json={"user_id": "user-1", "kind": "battery", "detail": "Battery at 13%."},
    ).json()
    assert second["comment"] is None  # cooldown
    assert len(recorder.calls) == 1

    # SKIP reply → no line, but the cooldown still armed (rate limit is on
    # LLM calls, not spoken lines).
    fresh_client, fresh_session, skipper = _client_with_session(tmp_path / "b")
    skipper.reply = "SKIP"
    data = fresh_client.post(
        "/api/ambient",
        json={"user_id": "user-1", "kind": "focus", "detail": "Same app for 2 hours."},
    ).json()
    assert data["comment"] is None
    assert len(skipper.calls) == 1


def test_ambient_disabled_and_sleep_window_suppressed(tmp_path):
    import datetime as _dt

    client, session, recorder = _client_with_session(tmp_path)
    session.config.ambient.enabled = False
    data = client.post(
        "/api/ambient",
        json={"user_id": "user-1", "kind": "focus", "detail": "Same app for 2 hours."},
    ).json()
    assert data["comment"] is None
    assert recorder.calls == []

    client2, session2, recorder2 = _client_with_session(tmp_path / "sleep")
    now_hour = _dt.datetime.now().hour
    session2.config.sleep_schedule.enabled = True
    session2.config.sleep_schedule.start_hour = now_hour
    session2.config.sleep_schedule.end_hour = (now_hour + 2) % 24
    data = client2.post(
        "/api/ambient",
        json={"user_id": "user-1", "kind": "focus", "detail": "Same app for 2 hours."},
    ).json()
    assert data["comment"] is None
    assert recorder2.calls == []


def test_ambient_comment_not_committed_to_memory(tmp_path):
    client, session, recorder = _client_with_session(tmp_path)
    before = len(session.memory.short_term)
    client.post(
        "/api/ambient",
        json={"user_id": "user-1", "kind": "music", "detail": "Listening to a song."},
    )
    assert len(session.memory.short_term) == before
