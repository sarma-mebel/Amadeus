""" "Watch together" mode: ChatEngine.watch_frame/watch_end +
POST /api/watch_together. Structural mirror of test_read_along.py —
the two features share their NOTE/SAY machinery by design."""

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
        self.replies = replies or ["NOTE: They are reading documentation."]
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
    watch_overrides: Optional[dict] = None,
) -> ChatEngine:
    config = Config()
    config.memory.persist_path = str(tmp_path / "memory.json")
    config.memory.embeddings_path = str(tmp_path / "embeddings.json")
    config.memory.profile_path = str(tmp_path / "profile.json")
    config.memory.transcripts_path = str(tmp_path / "transcripts.json")
    config.memory.enable_long_term = False
    for key, value in (vision_overrides or {}).items():
        setattr(config.vision, key, value)
    for key, value in (watch_overrides or {}).items():
        setattr(config.watch_together, key, value)
    engine = ChatEngine(config, llm or FakeVisionLLM())
    if summary_llm is not None:
        engine.summary_provider = summary_llm
    return engine


IMG = ("image/jpeg", "aGVsbG8=")


# --- watch_frame: gating -----------------------------------------------------


def test_watch_disabled_by_vision(tmp_path):
    llm = FakeVisionLLM()
    engine = _engine(tmp_path, llm, vision_overrides={"enabled": False})
    assert engine.watch_frame(*IMG) is None
    assert not llm.calls


def test_watch_needs_vision_capable_provider(tmp_path):
    engine = _engine(tmp_path, NoVisionLLM())
    assert engine.watch_frame(*IMG) is None


def test_watch_prefers_summary_provider_when_vision_capable(tmp_path):
    """Long watch sessions route to the cheap model, like read_along."""
    chat_llm = FakeVisionLLM(["NOTE: chat model saw this."])
    cheap_llm = FakeVisionLLM(["NOTE: cheap model saw this."])
    engine = _engine(tmp_path, chat_llm, summary_llm=cheap_llm)

    engine.watch_frame(*IMG)

    assert cheap_llm.calls
    assert not chat_llm.calls


# --- watch_frame: note/say + cooldown ---------------------------------------


def test_note_only_frame_speaks_nothing_but_advances_state(tmp_path):
    llm = FakeVisionLLM(["NOTE: They are debugging Python in an editor."])
    engine = _engine(tmp_path, llm)

    said = engine.watch_frame(*IMG)

    assert said is None
    assert engine._watch_together is not None
    assert engine._watch_together.frames == 1
    assert list(engine._watch_together.notes) == [
        "They are debugging Python in an editor."
    ]


def test_say_within_cooldown_is_swallowed_but_note_lands(tmp_path):
    llm = FakeVisionLLM(
        [
            "NOTE: The boss fight starts.\nSAY: Oh, this is the hard one.",
            "NOTE: The boss fight is going badly.\nSAY: You had that coming.",
        ]
    )
    engine = _engine(tmp_path, llm, watch_overrides={"speak_cooldown_seconds": 75})

    first = engine.watch_frame(*IMG)
    assert first == "Oh, this is the hard one."

    second = engine.watch_frame(*IMG)
    assert second is None  # swallowed by cooldown
    assert engine._watch_together.frames == 2
    assert list(engine._watch_together.notes) == [
        "The boss fight starts.",
        "The boss fight is going badly.",
    ]


def test_say_after_cooldown_expiry_is_returned(tmp_path):
    llm = FakeVisionLLM(
        [
            "NOTE: The boss fight starts.\nSAY: Oh, this is the hard one.",
            "NOTE: Victory screen.\nSAY: Finally. I was getting embarrassed for you.",
        ]
    )
    engine = _engine(tmp_path, llm, watch_overrides={"speak_cooldown_seconds": 75})

    assert engine.watch_frame(*IMG) == "Oh, this is the hard one."
    engine._watch_together.last_spoke_at = dt.datetime.utcnow() - dt.timedelta(
        seconds=76
    )

    assert engine.watch_frame(*IMG) == "Finally. I was getting embarrassed for you."


def test_private_screen_gets_neutral_note_no_say(tmp_path):
    llm = FakeVisionLLM(["NOTE: Private screen."])
    engine = _engine(tmp_path, llm)

    said = engine.watch_frame(*IMG)

    assert said is None
    assert list(engine._watch_together.notes) == ["Private screen."]


def test_earlier_notes_appear_in_next_frame_prompt(tmp_path):
    llm = FakeVisionLLM(["NOTE: Episode one starts.", "NOTE: The plot thickens."])
    engine = _engine(tmp_path, llm)

    engine.watch_frame(*IMG)
    engine.watch_frame(*IMG)

    second_system = llm.calls[1][0].content
    assert "Episode one starts." in second_system


def test_notes_deque_caps_at_max_notes_and_drops_oldest(tmp_path):
    llm = FakeVisionLLM([f"NOTE: Frame {i}." for i in range(5)])
    engine = _engine(tmp_path, llm, watch_overrides={"max_notes": 3})

    for _ in range(5):
        engine.watch_frame(*IMG)

    assert list(engine._watch_together.notes) == [
        "Frame 2.",
        "Frame 3.",
        "Frame 4.",
    ]


def test_frame_cap_stops_llm_calls(tmp_path):
    llm = FakeVisionLLM(["NOTE: Something."])
    engine = _engine(tmp_path, llm, watch_overrides={"max_frames_per_session": 2})

    engine.watch_frame(*IMG)
    engine.watch_frame(*IMG)
    engine.watch_frame(*IMG)

    assert len(llm.calls) == 2
    assert engine._watch_together.frames == 2


# --- watch_end ---------------------------------------------------------------


def test_end_under_five_frames_returns_none_and_clears_state(tmp_path):
    llm = FakeVisionLLM(["NOTE: Something."])
    engine = _engine(tmp_path, llm)

    engine.watch_frame(*IMG)
    assert engine.watch_end() is None
    assert engine._watch_together is None


def test_end_at_five_or_more_frames_returns_sign_off_and_clears_state(tmp_path):
    llm = FakeVisionLLM(["NOTE: Something happens."] * 5)
    engine = _engine(tmp_path, llm)

    for _ in range(5):
        engine.watch_frame(*IMG)
    engine.summary_provider = FakeVisionLLM(["Good session. Same time tomorrow?"])

    sign_off = engine.watch_end()

    assert sign_off == "Good session. Same time tomorrow?"
    assert engine._watch_together is None


def test_end_with_no_active_session_returns_none(tmp_path):
    engine = _engine(tmp_path, FakeVisionLLM())
    assert engine.watch_end() is None
    assert engine.watch_end() is None  # repeat is safe


def test_end_resets_frame_cap_logging(tmp_path):
    llm = FakeVisionLLM(["NOTE: Something."])
    engine = _engine(tmp_path, llm, watch_overrides={"max_frames_per_session": 1})

    engine.watch_frame(*IMG)
    engine.watch_frame(*IMG)  # capped
    assert engine._watch_cap_logged is True

    engine.watch_end()
    assert engine._watch_cap_logged is False


# --- memory contract ---------------------------------------------------------


def test_watch_together_never_committed_to_memory(tmp_path):
    llm = FakeVisionLLM(["NOTE: A twist on screen.\nSAY: Called it."] * 6)
    engine = _engine(tmp_path, llm)

    for _ in range(6):
        engine.watch_frame(*IMG)
    engine.summary_provider = FakeVisionLLM(["Fun watch."])
    engine.watch_end()

    assert len(engine.memory.short_term) == 0


# --- endpoint ----------------------------------------------------------------


@pytest.mark.asyncio
async def test_watch_endpoint_422_without_image_or_end(tmp_path, monkeypatch):
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
        neither = await client.post("/api/watch_together", json={})
        assert neither.status_code == 422

        both = await client.post(
            "/api/watch_together",
            json={"image": {"mime": "image/jpeg", "data": "aGVsbG8="}, "end": True},
        )
        assert both.status_code == 422


@pytest.mark.asyncio
async def test_watch_endpoint_happy_path(tmp_path, monkeypatch):
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
            "/api/watch_together",
            json={"image": {"mime": "image/jpeg", "data": "aGVsbG8="}},
        )
        assert response.status_code == 200
        body = response.json()
        assert "comment" in body
        assert body["frames"] == 1
        assert body["session_id"]

        end = await client.post(
            "/api/watch_together",
            json={"session_id": body["session_id"], "end": True},
        )
        assert end.status_code == 200
        assert "comment" in end.json()
