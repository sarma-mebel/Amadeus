"""Screen-companion peeks: ChatEngine.screen_comment + POST /api/screen_peek."""

from __future__ import annotations

from typing import List, Optional

import pytest

from amadeus.config import Config
from amadeus.engine import ChatEngine
from amadeus.providers.llm.base import ChatMessage, LLMProvider


class FakeVisionLLM(LLMProvider):
    """Vision-capable provider that records prompts and returns a canned line."""

    def __init__(self, reply: str = "Frame data again? Lab your matchups instead."):
        self.reply = reply
        self.calls: List[List[ChatMessage]] = []

    def supports_vision(self) -> bool:
        return True

    def generate(self, messages, max_tokens: int = 256, **kwargs) -> str:
        self.calls.append(list(messages))
        return self.reply

    def stream(self, messages, max_tokens: int = 256, **kwargs):
        yield self.generate(messages, max_tokens=max_tokens)


class NoVisionLLM(FakeVisionLLM):
    def supports_vision(self) -> bool:
        return False


def _engine(
    tmp_path, llm: Optional[LLMProvider] = None, **vision_overrides
) -> ChatEngine:
    config = Config()
    config.memory.persist_path = str(tmp_path / "memory.json")
    config.memory.embeddings_path = str(tmp_path / "embeddings.json")
    config.memory.profile_path = str(tmp_path / "profile.json")
    config.memory.transcripts_path = str(tmp_path / "transcripts.json")
    config.memory.enable_long_term = False
    for key, value in vision_overrides.items():
        setattr(config.vision, key, value)
    return ChatEngine(config, llm or FakeVisionLLM())


IMG = ("image/jpeg", "aGVsbG8=")


def test_screen_comment_generates_one_line(tmp_path):
    llm = FakeVisionLLM()
    engine = _engine(tmp_path, llm)

    text = engine.screen_comment(*IMG)

    assert text == llm.reply
    # The image rides the user message; the system prompt sets the rules.
    system, user = llm.calls[0]
    assert "glanced at their screen" in system.content
    assert "private or sensitive" in system.content
    assert user.images and user.images[0].mime == "image/jpeg"


def test_screen_comment_gated_by_vision_enabled(tmp_path):
    engine = _engine(tmp_path, enabled=False)
    assert engine.screen_comment(*IMG) is None


def test_screen_comment_needs_vision_capable_llm(tmp_path):
    engine = _engine(tmp_path, NoVisionLLM())
    assert engine.screen_comment(*IMG) is None


def test_screen_comment_cooldown(tmp_path):
    llm = FakeVisionLLM()
    engine = _engine(tmp_path, llm)

    assert engine.screen_comment(*IMG) is not None
    # Immediately again: cooldown (default 120s) blocks the second call.
    assert engine.screen_comment(*IMG) is None
    assert len(llm.calls) == 1


def test_screen_comment_not_committed_to_memory(tmp_path):
    engine = _engine(tmp_path)
    engine.screen_comment(*IMG)
    assert len(engine.memory.short_term) == 0


def test_screen_comment_skip_means_silence_but_still_arms_cooldown(tmp_path):
    llm = FakeVisionLLM(reply="SKIP")
    engine = _engine(tmp_path, llm)

    assert engine.screen_comment(*IMG) is None
    # The rate limit is on LLM calls, not spoken lines: an immediate retry
    # is still cooldown-blocked.
    assert engine.screen_comment(*IMG) is None
    assert len(llm.calls) == 1
    # A skip never becomes the "previous line" fed to the next prompt.
    assert engine._last_screen_comment is None


def test_screen_comment_feeds_previous_line_into_next_prompt(tmp_path):
    llm = FakeVisionLLM(reply="Dijkstra again? Bold choice.")
    engine = _engine(tmp_path, llm)

    first = engine.screen_comment(*IMG)
    assert first == llm.reply
    assert engine._last_screen_comment == llm.reply
    # First prompt offers SKIP but has no previous line yet.
    assert "SKIP" in llm.calls[0][0].content
    assert "previous glance line" not in llm.calls[0][0].content

    # Clear the cooldown stamp directly (a 0 config override falls back to
    # the 120s default via the `or`) and glance again.
    engine._last_screen_peek_at = None
    engine.screen_comment(*IMG)
    assert "Dijkstra again? Bold choice." in llm.calls[1][0].content
    assert "never repeat it" in llm.calls[1][0].content


@pytest.mark.asyncio
async def test_screen_peek_endpoint(tmp_path, monkeypatch):
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
            "/api/screen_peek",
            json={"image": {"mime": "image/jpeg", "data": "aGVsbG8="}},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["comment"]
        assert body["session_id"]

        # Cooldown: an immediate second peek on the SAME session returns a
        # null comment (the cooldown is per-session state, like poke).
        again = await client.post(
            "/api/screen_peek",
            json={
                "session_id": body["session_id"],
                "image": {"mime": "image/jpeg", "data": "aGVsbG8="},
            },
        )
        assert again.json()["comment"] is None
