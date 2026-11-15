"""Anti-dark-pattern farewell: goodbye detection, the engine's send-off
generation gates, and the /api/farewell endpoint's two shapes (goodbye vs
silent window_close)."""

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from amadeus.config import Config
from amadeus.farewell import is_goodbye
from amadeus.providers.llm.base import ChatMessage
from amadeus.web.app import create_app


class RecordingProvider:
    """Fake LLM recording every prompt so tests assert on prompt content."""

    name = "recording"

    def __init__(
        self, reply: str = "Go on. Try not to embarrass science while you're out there."
    ) -> None:
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
    config.memory.enable_farewell = True
    return config


def _session_with_recorder(app, user_id="user-1"):
    store = app.state.session_store
    recorder = RecordingProvider()
    store.llm = recorder
    session_id, session = store.get(None, user_id)
    session.llm = recorder
    session.summary_provider = recorder
    return store, session_id, session, recorder


# ---------------------------------------------------------------------------
# Goodbye detection
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "bye!",
        "Goodbye",
        "good night kurisu",
        "gotta go, talk later",
        "I'm heading to bed",
        "see ya tomorrow",
        "ttyl",
        "oyasumi",
    ],
)
def test_is_goodbye_positives(text):
    assert is_goodbye(text)


@pytest.mark.parametrize(
    "text",
    [
        "",
        "   ",
        "is goodbye a noun or an interjection in this sentence?",
        "the movie was about saying goodbye to your past and moving on with life",
        "what should I make for dinner",
        "tell me about the bayer process",
    ],
)
def test_is_goodbye_negatives(text):
    assert not is_goodbye(text)


# ---------------------------------------------------------------------------
# Engine gates
# ---------------------------------------------------------------------------


def test_farewell_prompt_forbids_retention_tactics(tmp_path):
    app = create_app(_isolated_config(tmp_path))
    _, _, session, recorder = _session_with_recorder(app)
    session.memory.add_message("user", "today was rough")
    session.memory.add_message(
        "assistant", "Then rest. That's an order backed by neuroscience."
    )

    line = session.close_session_farewell("good night")
    assert line == recorder.reply
    system = recorder.calls[0][0].content
    assert "NEVER guilt" in system
    assert "never ask them to stay" in system
    # The goodbye text itself reaches the prompt without a memory commit.
    user_content = recorder.calls[0][1].content
    assert "good night" in user_content
    assert all(m.content != "good night" for m in session.memory.short_term)


def test_farewell_requires_conversation(tmp_path):
    app = create_app(_isolated_config(tmp_path))
    _, _, session, recorder = _session_with_recorder(app)
    assert session.close_session_farewell("bye") is None
    assert recorder.calls == []


def test_farewell_disabled_flag(tmp_path):
    config = _isolated_config(tmp_path)
    config.memory.enable_farewell = False
    app = create_app(config)
    _, _, session, recorder = _session_with_recorder(app)
    session.memory.add_message("user", "hey")
    assert session.close_session_farewell("bye") is None
    assert recorder.calls == []


def test_farewell_cooldown_blocks_second_line(tmp_path):
    app = create_app(_isolated_config(tmp_path))
    _, _, session, recorder = _session_with_recorder(app)
    session.memory.add_message("user", "hey")
    assert session.close_session_farewell("bye") is not None
    # A real goodbye costs two calls: the farewell line + the next-time note.
    assert len(recorder.calls) == 2
    assert session.close_session_farewell("bye again") is None
    # The cooldown blocks BOTH: no farewell line, no second note call.
    assert len(recorder.calls) == 2


# ---------------------------------------------------------------------------
# /api/farewell endpoint
# ---------------------------------------------------------------------------


def test_farewell_endpoint_goodbye_commits_turn(tmp_path):
    app = create_app(_isolated_config(tmp_path))
    store, session_id, session, recorder = _session_with_recorder(app)
    session.memory.add_message("user", "today was rough")
    session.memory.add_message("assistant", "Then rest.")

    client = TestClient(app)
    response = client.post(
        "/api/farewell",
        json={"session_id": session_id, "user_id": "user-1", "message": "good night"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["farewell"] == recorder.reply
    contents = [m.content for m in session.memory.short_term]
    assert "good night" in contents
    assert recorder.reply in contents


def test_farewell_endpoint_not_goodbye_falls_back(tmp_path):
    app = create_app(_isolated_config(tmp_path))
    store, session_id, session, recorder = _session_with_recorder(app)
    session.memory.add_message("user", "hey")

    client = TestClient(app)
    response = client.post(
        "/api/farewell",
        json={
            "session_id": session_id,
            "user_id": "user-1",
            "message": "what is entropy exactly?",
        },
    )
    data = response.json()
    assert data["farewell"] is None
    assert data["not_goodbye"] is True
    assert recorder.calls == []


def test_farewell_endpoint_gated_none_reports_not_goodbye(tmp_path):
    """A valid goodbye that produces no line (empty session) must tell the
    client to fall back so the message still lands as a normal turn."""
    app = create_app(_isolated_config(tmp_path))
    store, session_id, session, recorder = _session_with_recorder(app)

    client = TestClient(app)
    response = client.post(
        "/api/farewell",
        json={"session_id": session_id, "user_id": "user-1", "message": "bye"},
    )
    data = response.json()
    assert data["farewell"] is None
    assert data["not_goodbye"] is True


def test_farewell_endpoint_window_close_never_creates_session(tmp_path):
    app = create_app(_isolated_config(tmp_path))
    store = app.state.session_store

    client = TestClient(app)
    response = client.post(
        "/api/farewell",
        json={"session_id": "ghost-session", "reason": "window_close"},
    )
    assert response.status_code == 200
    assert response.json()["farewell"] is None
    assert "ghost-session" not in store.sessions


def test_farewell_endpoint_window_close_finalizes_existing(tmp_path, monkeypatch):
    app = create_app(_isolated_config(tmp_path))
    store, session_id, session, _ = _session_with_recorder(app)
    calls = []
    monkeypatch.setattr(
        session.memory, "finalize_session", lambda wait=False: calls.append(wait)
    )

    client = TestClient(app)
    response = client.post(
        "/api/farewell",
        json={"session_id": session_id, "user_id": "user-1", "reason": "window_close"},
    )
    assert response.status_code == 200
    assert response.json()["farewell"] is None
    assert calls == [False]


# ---------------------------------------------------------------------------
# "Next time" thread: one line persisted at a real goodbye for the next
# session's greeting (never on window_close, never fatal to the farewell)
# ---------------------------------------------------------------------------


class ScriptedProvider:
    """Fake LLM returning scripted replies in order (farewell, then note)."""

    name = "scripted"

    def __init__(self, replies) -> None:
        self.replies = list(replies)
        self.calls: list[list[ChatMessage]] = []

    def generate(self, messages, **kwargs):
        self.calls.append(list(messages))
        return self.replies.pop(0) if self.replies else ""


def _session_with_scripted(app, replies, user_id="user-1"):
    store = app.state.session_store
    provider = ScriptedProvider(replies)
    store.llm = provider
    session_id, session = store.get(None, user_id)
    session.llm = provider
    session.summary_provider = provider
    return store, session_id, session, provider


def test_goodbye_writes_next_time_thread(tmp_path):
    app = create_app(_isolated_config(tmp_path))
    _, _, session, provider = _session_with_scripted(
        app,
        ["Go on then.", "They were about to submit the manuscript."],
    )
    session.memory.add_message("user", "one last push on the manuscript tonight")
    session.memory.add_message("assistant", "Then push. Sleep after.")

    line = session.close_session_farewell("ok heading off, bye")
    assert line == "Go on then."
    # Two calls: the farewell line, then the next-time note.
    assert len(provider.calls) == 2
    note_system = provider.calls[1][0].content
    assert "SKIP" in note_system
    assert "never about you" in note_system
    note_user = provider.calls[1][1].content
    assert "ok heading off, bye" in note_user
    thread = session.memory.profile.next_time_thread
    assert thread is not None
    assert thread["text"] == "They were about to submit the manuscript."
    # Persisted, not just in memory.
    persisted = [
        p for p in tmp_path.rglob("*.json") if "next_time_thread" in p.read_text()
    ]
    assert any("manuscript" in p.read_text() for p in persisted)


def test_next_time_thread_skip_reply_writes_nothing(tmp_path):
    from amadeus import telemetry

    app = create_app(_isolated_config(tmp_path))
    _, _, session, provider = _session_with_scripted(app, ["Off you go.", "SKIP"])
    session.memory.add_message("user", "hey")
    before = telemetry.snapshot().get("farewell.next_time_thread_failed", 0)

    assert session.close_session_farewell("bye") == "Off you go."
    assert len(provider.calls) == 2
    assert session.memory.profile.next_time_thread is None
    # SKIP is a legitimate outcome, not a failure.
    assert telemetry.snapshot().get("farewell.next_time_thread_failed", 0) == before


def test_next_time_thread_empty_reply_is_counted(tmp_path):
    from amadeus import telemetry

    app = create_app(_isolated_config(tmp_path))
    _, _, session, provider = _session_with_scripted(app, ["Later.", ""])
    session.memory.add_message("user", "hey")
    before = telemetry.snapshot().get("farewell.next_time_thread_failed", 0)

    assert session.close_session_farewell("bye") == "Later."
    assert session.memory.profile.next_time_thread is None
    assert telemetry.snapshot().get("farewell.next_time_thread_failed", 0) == before + 1


def test_next_time_thread_provider_error_counted_not_raised(tmp_path):
    from amadeus import telemetry

    class ExplodesOnSecondCall(ScriptedProvider):
        def generate(self, messages, **kwargs):
            if self.calls:
                self.calls.append(list(messages))
                raise RuntimeError("quota")
            return super().generate(messages, **kwargs)

    app = create_app(_isolated_config(tmp_path))
    store = app.state.session_store
    provider = ExplodesOnSecondCall(["Night."])
    store.llm = provider
    _, session = store.get(None, "user-1")
    session.llm = provider
    session.summary_provider = provider
    session.memory.add_message("user", "hey")
    before = telemetry.snapshot().get("farewell.next_time_thread_failed", 0)

    # The farewell line itself must survive the note's failure.
    assert session.close_session_farewell("bye") == "Night."
    assert session.memory.profile.next_time_thread is None
    assert telemetry.snapshot().get("farewell.next_time_thread_failed", 0) == before + 1


def test_window_close_writes_no_next_time_thread(tmp_path, monkeypatch):
    app = create_app(_isolated_config(tmp_path))
    store, session_id, session, recorder = _session_with_recorder(app)
    session.memory.add_message("user", "working on the manuscript")
    monkeypatch.setattr(session.memory, "finalize_session", lambda wait=False: None)

    client = TestClient(app)
    response = client.post(
        "/api/farewell",
        json={"session_id": session_id, "user_id": "user-1", "reason": "window_close"},
    )
    assert response.status_code == 200
    # Silent finalize: no LLM call, no thread written.
    assert recorder.calls == []
    assert session.memory.profile.next_time_thread is None
