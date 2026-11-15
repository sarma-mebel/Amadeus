"""Inner-thoughts channel: [thought] lines display as ghost text, are never
spoken, never leak into the token stream, never land in memory, and are
tolerantly stripped when the feature flag is off."""

from __future__ import annotations

import threading

import pytest

pytest.importorskip("fastapi")

from amadeus.config import Config
from amadeus.providers.llm.base import DummyProvider
from amadeus.providers.tts.base import TTSProvider
from amadeus.web.app import ChatRequest, create_app
from amadeus.web.streaming import (
    DisplayDelta,
    SentenceEnd,
    TaggedStreamParser,
    ThoughtEnd,
    build_tagged_instructions,
    parse_complete_reply,
    parse_complete_reply_events,
)

THOUGHT_REPLY = (
    "[happy] Fine, I'll explain. || いいわ、説明してあげる。\n"
    "[thought] Why am I even indulging this?\n"
    "[relaxed] Dopamine drives reward learning. || ドーパミンは報酬学習を駆動する。"
)


def _run(parser: TaggedStreamParser, deltas: list[str]) -> list[object]:
    events: list[object] = []
    for delta in deltas:
        events.extend(parser.feed(delta))
    events.extend(parser.finish())
    return events


def _display_text(events: list[object]) -> str:
    return "".join(e.text for e in events if isinstance(e, DisplayDelta))


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


def test_thought_line_single_delta():
    events = _run(TaggedStreamParser(), [THOUGHT_REPLY])
    thoughts = [e for e in events if isinstance(e, ThoughtEnd)]
    sentences = [e for e in events if isinstance(e, SentenceEnd)]
    assert [t.text for t in thoughts] == ["Why am I even indulging this?"]
    assert len(sentences) == 2
    assert "indulging" not in _display_text(events)


def test_thought_split_across_tiny_deltas_never_leaks_display():
    parser = TaggedStreamParser()
    events: list[object] = []
    for ch in THOUGHT_REPLY:
        events.extend(parser.feed(ch))
    events.extend(parser.finish())
    # The pre-finalize partial-display path must not stream thought text.
    assert "indulging" not in _display_text(events)
    assert [t.text for t in events if isinstance(t, ThoughtEnd)] == [
        "Why am I even indulging this?"
    ]


def test_thought_first_line_still_enters_tagged_mode():
    reply = "[thought] They're early today.\n" "[happy] You're early. || 早いじゃない。"
    parser = TaggedStreamParser()
    events = _run(parser, [reply])
    assert not parser.fallback_mode
    sentences = [e for e in events if isinstance(e, SentenceEnd)]
    thoughts = [e for e in events if isinstance(e, ThoughtEnd)]
    assert len(sentences) == 1 and len(thoughts) == 1
    # seq strictly increases across the mixed event kinds.
    assert thoughts[0].seq < sentences[0].seq


def test_thought_japanese_half_dropped():
    events = _run(
        TaggedStreamParser(),
        ["[thought] A stray thought. || 独り言。\n[happy] Hi. || やあ。"],
    )
    thoughts = [e for e in events if isinstance(e, ThoughtEnd)]
    assert thoughts[0].text == "A stray thought."


def test_unterminated_thought_finalizes_as_thought():
    events = _run(TaggedStreamParser(), ["[happy] Hi. || やあ。\n[thought] Unfinished"])
    thoughts = [e for e in events if isinstance(e, ThoughtEnd)]
    assert [t.text for t in thoughts] == ["Unfinished"]
    # And it never became a spoken neutral sentence.
    sentences = [e for e in events if isinstance(e, SentenceEnd)]
    assert all("Unfinished" not in s.en for s in sentences)


def test_parse_complete_reply_still_excludes_thoughts():
    sentences = parse_complete_reply(THOUGHT_REPLY)
    assert len(sentences) == 2
    assert all("indulging" not in s.en for s in sentences)


def test_parse_complete_reply_events_returns_both():
    sentences, thoughts = parse_complete_reply_events(THOUGHT_REPLY)
    assert len(sentences) == 2
    assert [t.text for t in thoughts] == ["Why am I even indulging this?"]


def test_build_tagged_instructions_gates_thought_grammar():
    assert "[thought]" not in build_tagged_instructions(False)
    assert "[thought]" in build_tagged_instructions(True)


# ---------------------------------------------------------------------------
# WS + POST paths
# ---------------------------------------------------------------------------


class FakeTTS(TTSProvider):
    def __init__(self) -> None:
        self.calls: list[str] = []

    def synthesize(
        self, text: str, *, speaking_rate: float = 1.0, pitch_shift: float = 0.0
    ) -> bytes:
        self.calls.append(text)
        return f"WAV:{text}".encode()


def _config(tmp_path, thoughts_enabled: bool) -> Config:
    config = Config()
    config.web.streaming_enabled = True
    config.web.streaming_tagged_output = True
    config.web.streaming_inner_thoughts = thoughts_enabled
    config.audio.enable_voice = True
    config.memory.persist_path = str(tmp_path / "memory.json")
    config.memory.embeddings_path = str(tmp_path / "embeddings.json")
    config.memory.enable_long_term = False
    return config


def _app(monkeypatch, tmp_path, thoughts_enabled: bool):
    config = _config(tmp_path, thoughts_enabled)
    scripted = DummyProvider(scripted_reply=THOUGHT_REPLY)
    fake_tts = FakeTTS()
    import amadeus.web.app as app_module

    monkeypatch.setattr(app_module, "load_llm", lambda cfg: scripted)
    monkeypatch.setattr(app_module, "load_tts", lambda cfg: fake_tts)
    return create_app(config), fake_tts


async def _run_stream(app, payload: dict):
    store = app.state.session_store
    session_id, session = store.get(payload.get("session_id"), payload.get("user_id"))
    frames: list[dict] = []
    cancel_event = threading.Event()

    async def emit(frame):
        frames.append(frame)

    request = ChatRequest.model_validate(payload)
    await session.handle_stream(request, emit, cancel_event)
    return session_id, session, frames


@pytest.mark.asyncio
async def test_ws_thought_frame_emitted_once_when_enabled(monkeypatch, tmp_path):
    app, fake_tts = _app(monkeypatch, tmp_path, thoughts_enabled=True)
    _sid, session, frames = await _run_stream(
        app, {"message": "Explain dopamine.", "user_id": "th-1"}
    )
    thought_frames = [f for f in frames if f["type"] == "thought"]
    assert len(thought_frames) == 1
    assert thought_frames[0]["text"] == "Why am I even indulging this?"
    # Never spoken: TTS calls carry no thought text.
    assert all("indulging" not in call for call in fake_tts.calls)
    # Never remembered: memory holds spoken English only.
    assistant = [m for m in session.memory.short_term if m.role == "assistant"]
    assert len(assistant) == 1
    assert "indulging" not in assistant[0].content
    # Display token frames carry no thought text either.
    token_text = "".join(f["text"] for f in frames if f["type"] == "token")
    assert "indulging" not in token_text


@pytest.mark.asyncio
async def test_ws_thought_frame_dropped_when_disabled(monkeypatch, tmp_path):
    app, fake_tts = _app(monkeypatch, tmp_path, thoughts_enabled=False)
    _sid, session, frames = await _run_stream(
        app, {"message": "Explain dopamine.", "user_id": "th-2"}
    )
    assert not [f for f in frames if f["type"] == "thought"]
    # Still cleanly stripped: not spoken, not displayed, not remembered.
    token_text = "".join(f["text"] for f in frames if f["type"] == "token")
    assert "indulging" not in token_text
    assert all("indulging" not in call for call in fake_tts.calls)


@pytest.mark.asyncio
async def test_ws_second_thought_dropped(monkeypatch, tmp_path):
    reply = (
        "[thought] First thought.\n"
        "[happy] Hello. || こんにちは。\n"
        "[thought] Second thought.\n"
        "[relaxed] Bye for now. || またね。"
    )
    config = _config(tmp_path, thoughts_enabled=True)
    scripted = DummyProvider(scripted_reply=reply)
    import amadeus.web.app as app_module

    monkeypatch.setattr(app_module, "load_llm", lambda cfg: scripted)
    monkeypatch.setattr(app_module, "load_tts", lambda cfg: FakeTTS())
    app = create_app(config)
    _sid, _session, frames = await _run_stream(
        app, {"message": "hi", "user_id": "th-3"}
    )
    thought_frames = [f for f in frames if f["type"] == "thought"]
    assert [f["text"] for f in thought_frames] == ["First thought."]


@pytest.mark.asyncio
async def test_post_response_carries_thought(monkeypatch, tmp_path):
    import httpx

    app, _fake_tts = _app(monkeypatch, tmp_path, thoughts_enabled=True)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/chat", json={"message": "Explain dopamine.", "user_id": "th-post"}
        )
    assert response.status_code == 200
    data = response.json()
    assert data["thought"] == "Why am I even indulging this?"
    assert "indulging" not in data["reply"]


@pytest.mark.asyncio
async def test_post_response_thought_none_when_disabled(monkeypatch, tmp_path):
    import httpx

    app, _fake_tts = _app(monkeypatch, tmp_path, thoughts_enabled=False)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/chat", json={"message": "Explain dopamine.", "user_id": "th-post-2"}
        )
    assert response.status_code == 200
    data = response.json()
    assert data["thought"] is None
    assert "indulging" not in data["reply"]
