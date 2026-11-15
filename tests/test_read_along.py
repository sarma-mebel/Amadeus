""" "Read together" mode: ChatEngine.read_along_page/read_along_end +
POST /api/read_along."""

from __future__ import annotations

import datetime as dt
from typing import List, Optional

import pytest

from amadeus.config import Config
from amadeus.engine import ChatEngine
from amadeus.providers.llm.base import ChatMessage, LLMProvider


class FakeVisionLLM(LLMProvider):
    """Vision-capable provider that records prompts and returns scripted
    NOTE/SAY replies (one per call, in order; the last one repeats)."""

    def __init__(self, replies: Optional[List[str]] = None):
        self.replies = replies or ["NOTE: A quiet page of exposition."]
        self.calls: List[List[ChatMessage]] = []

    def supports_vision(self) -> bool:
        return True

    def generate(self, messages, max_tokens: int = 256, **kwargs) -> str:
        self.calls.append(list(messages))
        idx = min(len(self.calls) - 1, len(self.replies) - 1)
        return self.replies[idx]

    def stream(self, messages, max_tokens: int = 256, **kwargs):
        yield self.generate(messages, max_tokens=max_tokens)


class NoVisionLLM(FakeVisionLLM):
    def supports_vision(self) -> bool:
        return False


def _engine(
    tmp_path,
    llm: Optional[LLMProvider] = None,
    *,
    summary_llm: Optional[LLMProvider] = None,
    vision_overrides: Optional[dict] = None,
    read_along_overrides: Optional[dict] = None,
) -> ChatEngine:
    config = Config()
    config.memory.persist_path = str(tmp_path / "memory.json")
    config.memory.embeddings_path = str(tmp_path / "embeddings.json")
    config.memory.profile_path = str(tmp_path / "profile.json")
    config.memory.transcripts_path = str(tmp_path / "transcripts.json")
    config.memory.enable_long_term = False
    for key, value in (vision_overrides or {}).items():
        setattr(config.vision, key, value)
    for key, value in (read_along_overrides or {}).items():
        setattr(config.read_along, key, value)
    engine = ChatEngine(config, llm or FakeVisionLLM())
    if summary_llm is not None:
        engine.summary_provider = summary_llm
    return engine


IMG = ("image/jpeg", "aGVsbG8=")


# --- read_along_page: gating ------------------------------------------


def test_read_along_disabled_by_vision(tmp_path):
    llm = FakeVisionLLM()
    engine = _engine(tmp_path, llm, vision_overrides={"enabled": False})
    assert engine.read_along_page(*IMG) is None
    assert not llm.calls


def test_read_along_needs_vision_capable_provider(tmp_path):
    engine = _engine(tmp_path, NoVisionLLM())
    assert engine.read_along_page(*IMG) is None


def test_read_along_prefers_summary_provider_when_vision_capable(tmp_path):
    """summary_provider should win over llm when it supports vision — this
    is what routes reading sessions onto the cheap model."""
    chat_llm = FakeVisionLLM(["NOTE: chat model saw this."])
    cheap_llm = FakeVisionLLM(["NOTE: cheap model saw this."])
    engine = _engine(tmp_path, chat_llm, summary_llm=cheap_llm)

    engine.read_along_page(*IMG)

    assert cheap_llm.calls
    assert not chat_llm.calls


def test_read_along_falls_back_to_llm_when_summary_provider_cant_see(tmp_path):
    chat_llm = FakeVisionLLM(["NOTE: chat model saw this."])
    non_vision_summary = NoVisionLLM(["NOTE: should never be used."])
    engine = _engine(tmp_path, chat_llm, summary_llm=non_vision_summary)

    engine.read_along_page(*IMG)

    assert chat_llm.calls
    assert not non_vision_summary.calls


# --- read_along_page: note/say parsing + cooldown -----------------------


def test_note_only_reply_speaks_nothing_but_advances_state(tmp_path):
    llm = FakeVisionLLM(["NOTE: Protagonist enters a new city."])
    engine = _engine(tmp_path, llm)

    said = engine.read_along_page(*IMG)

    assert said is None
    assert engine._read_along is not None
    assert engine._read_along.pages == 1
    assert list(engine._read_along.notes) == ["Protagonist enters a new city."]


def test_say_within_cooldown_is_swallowed_but_note_lands(tmp_path):
    llm = FakeVisionLLM(
        [
            "NOTE: A twist is revealed.\nSAY: Wait, WHAT?",
            "NOTE: The fallout continues.\nSAY: I did not see that coming.",
        ]
    )
    engine = _engine(tmp_path, llm, read_along_overrides={"speak_cooldown_seconds": 90})

    first = engine.read_along_page(*IMG)
    assert first == "Wait, WHAT?"

    second = engine.read_along_page(*IMG)
    assert second is None  # swallowed by cooldown
    assert engine._read_along.pages == 2
    assert list(engine._read_along.notes) == [
        "A twist is revealed.",
        "The fallout continues.",
    ]


def test_say_after_cooldown_expiry_is_returned(tmp_path):
    llm = FakeVisionLLM(
        [
            "NOTE: A twist is revealed.\nSAY: Wait, WHAT?",
            "NOTE: The fallout continues.\nSAY: I did not see that coming.",
        ]
    )
    engine = _engine(tmp_path, llm, read_along_overrides={"speak_cooldown_seconds": 90})

    assert engine.read_along_page(*IMG) == "Wait, WHAT?"
    # Manipulate last_spoke_at directly to simulate the cooldown elapsing.
    engine._read_along.last_spoke_at = dt.datetime.utcnow() - dt.timedelta(seconds=91)

    second = engine.read_along_page(*IMG)
    assert second == "I did not see that coming."


def test_missing_note_prefix_falls_back_to_whole_first_line(tmp_path):
    llm = FakeVisionLLM(["Just some prose without the prefix."])
    engine = _engine(tmp_path, llm)

    engine.read_along_page(*IMG)

    assert list(engine._read_along.notes) == ["Just some prose without the prefix."]


def test_private_page_gets_neutral_note_no_say(tmp_path):
    llm = FakeVisionLLM(["NOTE: Not reading material."])
    engine = _engine(tmp_path, llm)

    said = engine.read_along_page(*IMG)

    assert said is None
    assert list(engine._read_along.notes) == ["Not reading material."]


# --- notes deque cap + prompt propagation --------------------------------


def test_notes_deque_caps_at_max_notes_and_drops_oldest(tmp_path):
    replies = [f"NOTE: page {i}" for i in range(1, 6)]
    llm = FakeVisionLLM(replies)
    engine = _engine(tmp_path, llm, read_along_overrides={"max_notes": 3})

    for _ in replies:
        engine.read_along_page(*IMG)

    assert list(engine._read_along.notes) == ["page 3", "page 4", "page 5"]


def test_earlier_notes_appear_in_next_page_prompt(tmp_path):
    llm = FakeVisionLLM(["NOTE: The hero finds a sword.", "NOTE: A dragon appears."])
    engine = _engine(tmp_path, llm)

    engine.read_along_page(*IMG)
    engine.read_along_page(*IMG)

    second_system_prompt = llm.calls[1][0].content
    assert "Earlier pages" in second_system_prompt
    assert "The hero finds a sword." in second_system_prompt


# --- page cap -------------------------------------------------------------


def test_page_cap_stops_llm_calls(tmp_path):
    llm = FakeVisionLLM(["NOTE: page"] * 10)
    engine = _engine(tmp_path, llm, read_along_overrides={"max_pages_per_session": 2})

    engine.read_along_page(*IMG)
    engine.read_along_page(*IMG)
    result = engine.read_along_page(*IMG)

    assert result is None
    assert len(llm.calls) == 2
    assert engine._read_along.pages == 2


# --- read_along_end --------------------------------------------------------


def test_end_under_five_pages_returns_none_and_clears_state(tmp_path):
    llm = FakeVisionLLM(["NOTE: page"] * 3)
    engine = _engine(tmp_path, llm)

    for _ in range(3):
        engine.read_along_page(*IMG)
    assert engine._read_along is not None

    result = engine.read_along_end()

    assert result is None
    assert engine._read_along is None


def test_end_at_five_or_more_pages_returns_sign_off_and_clears_state(tmp_path):
    llm = FakeVisionLLM(["NOTE: page"] * 5)
    engine = _engine(tmp_path, llm)

    for _ in range(5):
        engine.read_along_page(*IMG)

    # summary_provider defaults to the same dummy-fallback chain; force a
    # provider capable of text generation for the sign-off call.
    text_provider = FakeVisionLLM(["See you next chapter."])
    engine.summary_provider = text_provider

    result = engine.read_along_end()

    assert result == "See you next chapter."
    assert engine._read_along is None


def test_end_with_no_active_session_returns_none(tmp_path):
    engine = _engine(tmp_path)
    assert engine.read_along_end() is None


# --- memory contract --------------------------------------------------------


def test_read_along_never_committed_to_memory(tmp_path):
    llm = FakeVisionLLM(["NOTE: A twist is revealed.\nSAY: Wait, WHAT?"] * 6)
    engine = _engine(tmp_path, llm)

    for _ in range(6):
        engine.read_along_page(*IMG)
    engine.summary_provider = FakeVisionLLM(["Good chapter."])
    engine.read_along_end()

    assert len(engine.memory.short_term) == 0


# --- endpoint ---------------------------------------------------------------


@pytest.mark.asyncio
async def test_read_along_endpoint_422_without_image_or_end(tmp_path, monkeypatch):
    pytest.importorskip("fastapi")
    import httpx

    from amadeus.web import app as app_module
    from amadeus.web.app import create_app

    monkeypatch.setenv("KURISU_FRONTEND_DIR", str(tmp_path / "no-bundle"))
    monkeypatch.setattr(app_module, "load_llm", lambda config: FakeVisionLLM())

    config = Config()
    config.audio.enable_voice = False
    config.memory.persist_path = str(tmp_path / "memory.json")
    config.memory.embeddings_path = str(tmp_path / "embeddings.json")
    config.memory.profile_path = str(tmp_path / "profile.json")
    config.memory.transcripts_path = str(tmp_path / "transcripts.json")
    config.memory.enable_long_term = False

    app = create_app(config)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
        neither = await client.post("/api/read_along", json={})
        assert neither.status_code == 422

        both = await client.post(
            "/api/read_along",
            json={"image": {"mime": "image/jpeg", "data": "aGVsbG8="}, "end": True},
        )
        assert both.status_code == 422


@pytest.mark.asyncio
async def test_read_along_endpoint_happy_path(tmp_path, monkeypatch):
    pytest.importorskip("fastapi")
    import httpx

    from amadeus.web import app as app_module
    from amadeus.web.app import create_app

    monkeypatch.setenv("KURISU_FRONTEND_DIR", str(tmp_path / "no-bundle"))
    monkeypatch.setattr(app_module, "load_llm", lambda config: FakeVisionLLM())

    config = Config()
    config.audio.enable_voice = False
    config.memory.persist_path = str(tmp_path / "memory.json")
    config.memory.embeddings_path = str(tmp_path / "embeddings.json")
    config.memory.profile_path = str(tmp_path / "profile.json")
    config.memory.transcripts_path = str(tmp_path / "transcripts.json")
    config.memory.enable_long_term = False

    app = create_app(config)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
        response = await client.post(
            "/api/read_along",
            json={"image": {"mime": "image/jpeg", "data": "aGVsbG8="}},
        )
        assert response.status_code == 200
        body = response.json()
        assert "comment" in body
        assert body["pages"] == 1
        assert body["session_id"]

        end = await client.post(
            "/api/read_along",
            json={"session_id": body["session_id"], "end": True},
        )
        assert end.status_code == 200
        assert "comment" in end.json()
