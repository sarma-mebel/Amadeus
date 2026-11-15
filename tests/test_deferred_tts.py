"""Deferred-audio POST flow (ChatRequest.defer_audio) + POST /api/tts."""

from __future__ import annotations

import base64

import pytest

pytest.importorskip("fastapi")

import httpx

from amadeus.config import Config
from amadeus.providers.llm.base import LLMProvider
from amadeus.providers.tts.base import TTSProvider
from amadeus.web.app import create_app
from amadeus.web.sessions import _synthesize_tts


class TaggedFakeLLM(LLMProvider):
    """Always answers in the tagged bilingual grammar."""

    name = "tagged-fake"

    def generate(self, messages, **kwargs):
        return (
            "[happy] Science is amazing. || 科学は素晴らしい。\n"
            "[neutral] Trust the data. || データを信じなさい。"
        )


class FakeTTS(TTSProvider):
    def __init__(self) -> None:
        self.calls: list[str] = []

    def synthesize(
        self, text: str, *, speaking_rate: float = 1.0, pitch_shift: float = 0.0
    ) -> bytes:
        self.calls.append(text)
        return f"WAV:{text}".encode()


def _config(tmp_path) -> Config:
    config = Config()
    config.audio.enable_voice = True
    config.memory.persist_path = str(tmp_path / "memory.json")
    config.memory.embeddings_path = str(tmp_path / "embeddings.json")
    config.memory.profile_path = str(tmp_path / "profile.json")
    config.memory.transcripts_path = str(tmp_path / "transcripts.json")
    config.memory.enable_long_term = False
    return config


@pytest.fixture()
def voiced_app(monkeypatch, tmp_path):
    import amadeus.web.app as app_module

    fake_tts = FakeTTS()
    monkeypatch.setattr(app_module, "load_tts", lambda cfg: fake_tts)
    app = create_app(_config(tmp_path))
    return app, fake_tts


def _client(app):
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://testserver")


@pytest.mark.asyncio
async def test_defer_audio_skips_synthesis_entirely(voiced_app):
    app, fake_tts = voiced_app
    async with _client(app) as client:
        resp = await client.post(
            "/api/chat", json={"message": "Hello Kurisu", "defer_audio": True}
        )
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["reply"]
    assert payload["emotion"]
    assert payload["audio"] is None
    assert payload["audio_mime_type"] is None
    assert fake_tts.calls == []


@pytest.mark.asyncio
async def test_default_post_still_synthesizes(voiced_app):
    app, fake_tts = voiced_app
    async with _client(app) as client:
        resp = await client.post("/api/chat", json={"message": "Hello Kurisu"})
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["audio"] is not None
    assert payload["audio_mime_type"] == "audio/wav"
    assert fake_tts.calls, "expected the non-deferred path to synthesize"


@pytest.mark.asyncio
async def test_tagged_reply_returns_sentences(monkeypatch, tmp_path):
    import amadeus.web.app as app_module

    monkeypatch.setattr(app_module, "load_tts", lambda cfg: FakeTTS())
    monkeypatch.setattr(app_module, "load_llm", lambda cfg: TaggedFakeLLM())
    app = create_app(_config(tmp_path))
    async with _client(app) as client:
        resp = await client.post(
            "/api/chat", json={"message": "Tell me something", "defer_audio": True}
        )
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["audio"] is None
    sentences = payload["sentences"]
    assert [s["ja"] for s in sentences] == [
        "科学は素晴らしい。",
        "データを信じなさい。",
    ]
    assert sentences[0]["emotion"] == "happy"
    assert sentences[0]["en"] == "Science is amazing."


class UntaggedFakeLLM(LLMProvider):
    """Ignores the tagged grammar, like a small local model would."""

    name = "untagged-fake"

    def generate(self, messages, **kwargs):
        return "Plain English answer with no tags at all."


@pytest.mark.asyncio
async def test_untagged_defer_audio_falls_back_to_single_sentence(
    monkeypatch, tmp_path
):
    import amadeus.web.app as app_module

    # Never hit a live translator in tests; the identity-ish stub proves the
    # fallback path ran without depending on deep-translator being installed.
    monkeypatch.setattr(
        app_module, "_cached_translate", lambda translator, text: f"JA:{text}"
    )
    fake_tts = FakeTTS()
    monkeypatch.setattr(app_module, "load_tts", lambda cfg: fake_tts)
    monkeypatch.setattr(app_module, "load_llm", lambda cfg: UntaggedFakeLLM())
    app = create_app(_config(tmp_path))
    async with _client(app) as client:
        resp = await client.post(
            "/api/chat", json={"message": "Hello Kurisu", "defer_audio": True}
        )
    assert resp.status_code == 200
    payload = resp.json()
    sentences = payload["sentences"]
    assert sentences is not None and len(sentences) == 1
    assert sentences[0]["ja"]  # speech text present (translated or stripped English)
    assert fake_tts.calls == []


@pytest.mark.asyncio
async def test_tts_endpoint_roundtrip(voiced_app):
    app, fake_tts = voiced_app
    async with _client(app) as client:
        resp = await client.post(
            "/api/tts", json={"text": "こんにちは", "emotion": "happy"}
        )
    assert resp.status_code == 200
    body = resp.json()
    assert body["mime"] == "audio/wav"
    assert base64.b64decode(body["data"]) == "WAV:こんにちは".encode()
    assert fake_tts.calls == ["こんにちは"]


@pytest.mark.asyncio
async def test_tts_endpoint_emotion_optional(voiced_app):
    app, _ = voiced_app
    async with _client(app) as client:
        resp = await client.post("/api/tts", json={"text": "テスト"})
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_tts_endpoint_rejects_invalid_emotion(voiced_app):
    app, _ = voiced_app
    async with _client(app) as client:
        resp = await client.post(
            "/api/tts", json={"text": "テスト", "emotion": "ecstatic"}
        )
    assert resp.status_code == 400


@pytest.mark.asyncio
async def test_tts_endpoint_caps_text_length(voiced_app):
    app, fake_tts = voiced_app
    async with _client(app) as client:
        resp = await client.post("/api/tts", json={"text": "あ" * 501})
    assert resp.status_code == 413
    assert fake_tts.calls == []


@pytest.mark.asyncio
async def test_synthesize_tts_applies_sanitize_and_normalize():
    """_synthesize_tts is the single choke point for both audio-side text
    transforms (sanitize_speech_for_clone + normalize_ja_for_speech), so
    every entry point (POST-inline, /api/tts, routes_meta re-synthesis)
    gets both — this used to be the gap: the POST-inline full-reply path
    called the TTS provider directly with no sanitize at all."""
    fake_tts = FakeTTS()
    audio = await _synthesize_tts(
        fake_tts,
        "3時に集合ね。GPT-4は使わないで。",
        speaking_rate=1.0,
        pitch_shift=0.0,
    )
    assert fake_tts.calls == ["三時に集合ね。GPT-4は使わないで。"]
    assert audio == "WAV:三時に集合ね。GPT-4は使わないで。".encode()


@pytest.mark.asyncio
async def test_tts_endpoint_normalizes_numbers(voiced_app):
    """The deferred /api/tts endpoint no longer sanitizes at the route
    level (moved into _synthesize_tts), so this proves the transform still
    fires end-to-end through the endpoint."""
    app, fake_tts = voiced_app
    async with _client(app) as client:
        resp = await client.post("/api/tts", json={"text": "12時に会いましょう。"})
    assert resp.status_code == 200
    assert fake_tts.calls == ["十二時に会いましょう。"]


@pytest.mark.asyncio
async def test_tts_endpoint_503_when_unavailable(monkeypatch, tmp_path):
    import amadeus.web.app as app_module

    config = _config(tmp_path)
    monkeypatch.setattr(app_module, "load_tts", lambda cfg: None)
    app = create_app(config)
    async with _client(app) as client:
        resp = await client.post("/api/tts", json={"text": "テスト"})
    assert resp.status_code == 503
    assert "error" in resp.json()["detail"]
