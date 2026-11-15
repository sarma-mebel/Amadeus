"""Screenshot vision turns: images reach only the current turn's LLM
context; memory/transcripts hold the text marker and zero base64; tool
and tool paths are bypassed; non-vision providers silently drop images."""

from __future__ import annotations

import base64
import threading

import pytest

pytest.importorskip("fastapi")

import httpx

from amadeus.config import Config
from amadeus.providers.llm.base import ChatMessage, DummyProvider
from amadeus.providers.tts.base import TTSProvider
from amadeus.web.app import ChatRequest, create_app

FAKE_IMAGE_B64 = base64.b64encode(b"\x89PNG fake image bytes").decode()

TAGGED_REPLY = (
    "[surprised] That's your screen? || それがあなたの画面？\n"
    "[relaxed] The code on the left has a bug. || 左のコードにバグがあるわ。"
)


class VisionRecordingProvider(DummyProvider):
    """Dummy provider that records every context it receives and claims
    vision support."""

    def __init__(self) -> None:
        super().__init__(scripted_reply=TAGGED_REPLY)
        self.contexts: list[list[ChatMessage]] = []

    def generate(self, messages, **kwargs):
        self.contexts.append(list(messages))
        return super().generate(messages, **kwargs)

    def stream(self, messages, **kwargs):
        self.contexts.append(list(messages))
        yield from super().stream(messages, **kwargs)

    def supports_vision(self) -> bool:
        return True


class FakeTTS(TTSProvider):
    def __init__(self) -> None:
        self.calls: list[str] = []

    def synthesize(
        self, text: str, *, speaking_rate: float = 1.0, pitch_shift: float = 0.0
    ) -> bytes:
        self.calls.append(text)
        return b"WAV"


def _config(tmp_path, vision_enabled: bool = True) -> Config:
    config = Config()
    config.web.streaming_enabled = True
    config.web.streaming_tagged_output = True
    config.vision.enabled = vision_enabled
    config.memory.persist_path = str(tmp_path / "memory.json")
    config.memory.embeddings_path = str(tmp_path / "embeddings.json")
    config.memory.enable_long_term = False
    return config


def _app(monkeypatch, tmp_path, provider=None, vision_enabled: bool = True):
    config = _config(tmp_path, vision_enabled)
    provider = provider or VisionRecordingProvider()
    import amadeus.web.app as app_module

    monkeypatch.setattr(app_module, "load_llm", lambda cfg: provider)
    monkeypatch.setattr(app_module, "load_tts", lambda cfg: FakeTTS())
    return create_app(config), provider


def _image_payload(message: str = "what do you think of this?") -> dict:
    return {
        "message": message,
        "user_id": "vis-1",
        "images": [{"data": FAKE_IMAGE_B64, "mime": "image/png"}],
    }


async def _post(app, payload) -> httpx.Response:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        return await client.post("/api/chat", json=payload)


@pytest.mark.asyncio
async def test_image_reaches_current_turn_context_only(monkeypatch, tmp_path):
    app, provider = _app(monkeypatch, tmp_path)
    response = await _post(app, _image_payload())
    assert response.status_code == 200

    context = provider.contexts[-1]
    user_messages = [m for m in context if m.role == "user"]
    assert user_messages[-1].images is not None
    assert user_messages[-1].images[0].mime == "image/png"
    assert user_messages[-1].content.endswith("[user shared a screenshot]")
    # The vision system note rides the context.
    assert any(
        "attached a screenshot" in m.content for m in context if m.role == "system"
    )

    # Turn N+1 (text-only) rebuilds from memory: no images anywhere.
    response2 = await _post(
        app, {"message": "thanks, anything else?", "user_id": "vis-1"}
    )
    assert response2.status_code == 200
    context2 = provider.contexts[-1]
    assert all(m.images is None for m in context2)
    # The marker (not the bytes) is what memory carried forward.
    assert any("[user shared a screenshot]" in m.content for m in context2)


@pytest.mark.asyncio
async def test_memory_and_transcript_never_hold_base64(monkeypatch, tmp_path):
    app, provider = _app(monkeypatch, tmp_path)
    await _post(app, _image_payload())
    store = app.state.session_store
    session = next(iter(store.sessions.values()))
    for message in session.memory.short_term:
        assert FAKE_IMAGE_B64 not in (message.content or "")
    marker_messages = [
        m
        for m in session.memory.short_term
        if "[user shared a screenshot]" in m.content
    ]
    assert marker_messages


@pytest.mark.asyncio
async def test_image_turn_takes_the_plain_path(monkeypatch, tmp_path):
    """The deprecated council flag is tolerated and ignored; the image turn
    reaches the provider directly (tool loop is also gated off image turns)."""
    app, provider = _app(monkeypatch, tmp_path)
    payload = {**_image_payload(), "council": True}
    response = await _post(app, payload)
    assert response.status_code == 200
    assert provider.contexts


@pytest.mark.asyncio
async def test_ws_image_turn_attaches_images(monkeypatch, tmp_path):
    app, provider = _app(monkeypatch, tmp_path)
    store = app.state.session_store
    session_id, session = store.get(None, "vis-ws")
    frames: list[dict] = []

    async def emit(frame):
        frames.append(frame)

    request = ChatRequest.model_validate(_image_payload("look at this"))
    await session.handle_stream(request, emit, threading.Event())
    context = provider.contexts[-1]
    user_messages = [m for m in context if m.role == "user"]
    assert user_messages[-1].images is not None
    assert frames[-1]["type"] == "done"


@pytest.mark.asyncio
async def test_vision_disabled_drops_images(monkeypatch, tmp_path):
    app, provider = _app(monkeypatch, tmp_path, vision_enabled=False)
    response = await _post(app, _image_payload())
    assert response.status_code == 200
    context = provider.contexts[-1]
    assert all(m.images is None for m in context)
    assert all("[user shared a screenshot]" not in m.content for m in context)


@pytest.mark.asyncio
async def test_non_vision_provider_drops_images(monkeypatch, tmp_path):
    provider = DummyProvider(scripted_reply=TAGGED_REPLY)  # supports_vision() False
    recorded: list[list[ChatMessage]] = []
    original = provider.generate

    def recording_generate(messages, **kwargs):
        recorded.append(list(messages))
        return original(messages, **kwargs)

    provider.generate = recording_generate  # type: ignore[method-assign]
    app, _ = _app(monkeypatch, tmp_path, provider=provider)
    response = await _post(app, _image_payload())
    assert response.status_code == 200
    assert recorded
    assert all(m.images is None for m in recorded[-1])


@pytest.mark.asyncio
async def test_oversized_image_rejected(monkeypatch, tmp_path):
    app, _provider = _app(monkeypatch, tmp_path)
    huge = "A" * (7_000_000)  # > 5MB * 4/3 cap
    response = await _post(
        app,
        {
            "message": "hi",
            "user_id": "vis-big",
            "images": [{"data": huge, "mime": "image/jpeg"}],
        },
    )
    assert response.status_code in (400, 422)


@pytest.mark.asyncio
async def test_bad_mime_rejected(monkeypatch, tmp_path):
    app, _provider = _app(monkeypatch, tmp_path)
    response = await _post(
        app,
        {
            "message": "hi",
            "user_id": "vis-mime",
            "images": [{"data": FAKE_IMAGE_B64, "mime": "image/gif"}],
        },
    )
    assert response.status_code in (400, 422)


def test_openai_serialize_message_content_parts():
    """Text-only stays a plain string (byte-identical requests); images
    switch that message to content-parts with a data: URI."""
    from amadeus.providers.llm.base import ImageContent
    from amadeus.providers.llm.openai_provider import OpenAIProvider

    plain = ChatMessage(role="user", content="hello")
    assert OpenAIProvider._serialize_message(plain) == {
        "role": "user",
        "content": "hello",
    }

    with_image = ChatMessage(
        role="user",
        content="look",
        images=[ImageContent(mime="image/jpeg", data="QUJD")],
    )
    serialized = OpenAIProvider._serialize_message(with_image)
    assert serialized["role"] == "user"
    assert serialized["content"][0] == {"type": "text", "text": "look"}
    assert serialized["content"][1] == {
        "type": "image_url",
        "image_url": {"url": "data:image/jpeg;base64,QUJD"},
    }
