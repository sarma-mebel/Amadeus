"""Token-usage telemetry: provider ``last_usage`` -> ``tokens.*`` counters.

Covers record_llm_usage directly (None/missing tolerance, dummy no-op),
the blocking web chat path (tokens.chat.*), and one engine presence path
(tokens.presence.*). Providers are stubs — no live API calls.
"""

from __future__ import annotations

import pytest

from amadeus import telemetry
from amadeus.config import Config
from amadeus.providers.llm.base import ChatMessage, DummyProvider, LLMProvider


@pytest.fixture(autouse=True)
def _clean_counters():
    telemetry.reset()
    yield
    telemetry.reset()


class UsageProvider(LLMProvider):
    """Stub whose every generate() reports fixed token usage."""

    name = "usage-stub"

    def __init__(
        self,
        reply: str = "Understood.",
        prompt: int = 100,
        completion: int = 25,
    ) -> None:
        self.reply = reply
        self._usage = {"prompt": prompt, "completion": completion}
        self.calls: list[list[ChatMessage]] = []

    def generate(self, messages, **kwargs):
        self.calls.append(list(messages))
        self.last_usage = dict(self._usage)
        return self.reply


# ---------------------------------------------------------------------------
# record_llm_usage directly
# ---------------------------------------------------------------------------


def test_record_llm_usage_increments_counters():
    provider = UsageProvider(prompt=40, completion=7)
    provider.generate([ChatMessage(role="user", content="hi")])
    telemetry.record_llm_usage(provider, "chat")
    snap = telemetry.snapshot()
    assert snap["tokens.chat.prompt"] == 40
    assert snap["tokens.chat.completion"] == 7
    assert snap["tokens.chat.calls"] == 1


def test_record_llm_usage_accumulates_across_calls():
    provider = UsageProvider(prompt=10, completion=2)
    for _ in range(3):
        provider.generate([ChatMessage(role="user", content="hi")])
        telemetry.record_llm_usage(provider, "tools")
    snap = telemetry.snapshot()
    assert snap["tokens.tools.prompt"] == 30
    assert snap["tokens.tools.completion"] == 6
    assert snap["tokens.tools.calls"] == 3


def test_record_llm_usage_tolerates_none_and_missing():
    # last_usage None (base default) -> nothing recorded.
    telemetry.record_llm_usage(LLMProvider(), "chat")
    # Object without the attribute at all -> nothing recorded.
    telemetry.record_llm_usage(object(), "chat")
    # No provider at all (text ReAct loop with a plain callable) -> nothing.
    telemetry.record_llm_usage(None, "chat")

    # Malformed usage payload (not a dict) must not raise either.
    class Weird:
        last_usage = "not-a-dict"

    telemetry.record_llm_usage(Weird(), "chat")
    assert telemetry.snapshot() == {}


def test_dummy_provider_records_nothing():
    provider = DummyProvider(scripted_reply="ok")
    provider.generate([ChatMessage(role="user", content="hi")])
    telemetry.record_llm_usage(provider, "chat")
    assert telemetry.snapshot() == {}


# ---------------------------------------------------------------------------
# Web + engine paths (stub providers wired into real sessions)
# ---------------------------------------------------------------------------


def _isolated_config(tmp_path) -> Config:
    config = Config()
    config.memory.persist_path = str(tmp_path / "memory.json")
    config.memory.embeddings_path = str(tmp_path / "embeddings.json")
    config.memory.profile_path = str(tmp_path / "profile.json")
    config.memory.enable_long_term = False
    config.audio.enable_voice = False
    return config


def test_blocking_chat_path_records_chat_tokens(tmp_path):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from amadeus.web.app import create_app

    config = _isolated_config(tmp_path)
    app = create_app(config)
    store = app.state.session_store
    session_id, session = store.get(None, "user-1")
    provider = UsageProvider(reply="[happy] Hi there. || こんにちは。")
    session.llm = provider

    client = TestClient(app)
    response = client.post(
        "/api/chat",
        json={"message": "hello", "session_id": session_id, "user_id": "user-1"},
    )
    assert response.status_code == 200
    assert provider.calls  # the stub actually served the turn

    snap = telemetry.snapshot()
    assert snap["tokens.chat.calls"] == 1
    assert snap["tokens.chat.prompt"] == 100
    assert snap["tokens.chat.completion"] == 25


def test_poke_presence_path_records_presence_tokens(tmp_path):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from amadeus.web.app import create_app

    config = _isolated_config(tmp_path)
    config.memory.enable_poke_reactions = True
    app = create_app(config)
    store = app.state.session_store
    session_id, session = store.get(None, "user-1")
    provider = UsageProvider(reply="Don't touch me.", prompt=50, completion=8)
    session.llm = provider
    session.summary_provider = provider

    client = TestClient(app)
    response = client.post(
        "/api/poke", json={"session_id": session_id, "user_id": "user-1"}
    )
    assert response.status_code == 200
    assert response.json()["reaction"] == "Don't touch me."

    snap = telemetry.snapshot()
    assert snap["tokens.presence.calls"] == 1
    assert snap["tokens.presence.prompt"] == 50
    assert snap["tokens.presence.completion"] == 8
    assert "tokens.chat.calls" not in snap  # attributed to presence, not chat


def test_fallback_provider_exposes_serving_delegates_usage():
    """FallbackLLMProvider.last_usage must reflect whichever delegate served."""
    from amadeus.providers.llm.fallback import FallbackLLMProvider

    class RateLimited(LLMProvider):
        name = "limited"

        def generate(self, messages, **kwargs):
            exc = RuntimeError("quota")
            exc.status_code = 429
            raise exc

    secondary = UsageProvider(reply="served by fallback", prompt=11, completion=4)
    composite = FallbackLLMProvider(RateLimited(), secondary)
    assert composite.generate([ChatMessage(role="user", content="hi")]) == (
        "served by fallback"
    )
    telemetry.record_llm_usage(composite, "chat")
    snap = telemetry.snapshot()
    assert snap["tokens.chat.prompt"] == 11
    assert snap["tokens.chat.completion"] == 4
    assert snap["tokens.chat.calls"] == 1
