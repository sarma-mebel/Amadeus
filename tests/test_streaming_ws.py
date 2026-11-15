"""WebSocket integration tests for /api/chat/stream."""

from __future__ import annotations

import threading

import httpx
import pytest

from amadeus.config import Config
from amadeus.providers.llm.base import DummyProvider
from amadeus.providers.tts.base import TTSProvider
from amadeus.web.app import ChatRequest, create_app


class ScriptedProvider(DummyProvider):
    """A DummyProvider whose scripted_reply streams like a real model."""


class FakeTTS(TTSProvider):
    def __init__(self) -> None:
        self.calls: list[str] = []

    def synthesize(
        self, text: str, *, speaking_rate: float = 1.0, pitch_shift: float = 0.0
    ) -> bytes:
        self.calls.append(text)
        return f"WAV:{text}".encode()


def _streaming_config(tmp_path) -> Config:
    config = Config()
    config.web.streaming_enabled = True
    config.web.streaming_tagged_output = True
    config.audio.enable_voice = True
    # Isolate persistence per test — the default paths write into the repo
    # and accumulate across runs, corrupting message-count assertions.
    config.memory.persist_path = str(tmp_path / "memory.json")
    config.memory.embeddings_path = str(tmp_path / "embeddings.json")
    config.memory.enable_long_term = False
    return config


@pytest.fixture()
def streaming_app(monkeypatch, tmp_path):
    """Build an app whose SessionStore uses deterministic providers."""
    config = _streaming_config(tmp_path)

    scripted = ScriptedProvider(
        scripted_reply=(
            "[happy] Dopamine reinforces reward learning. || ドーパミンは報酬学習を強化する。\n"
            "[relaxed] It also supports memory consolidation. || 記憶の定着も支援する。"
        )
    )
    fake_tts = FakeTTS()

    import amadeus.web.app as app_module

    monkeypatch.setattr(app_module, "load_llm", lambda cfg: scripted)
    monkeypatch.setattr(app_module, "load_tts", lambda cfg: fake_tts)

    app = create_app(config)
    yield app, scripted, fake_tts


async def _run_stream(app, payload: dict, cancel_after: int | None = None):
    store = app.state.session_store
    session_id, session = store.get(payload.get("session_id"), payload.get("user_id"))
    frames = [{"type": "session", "session_id": session_id}]
    cancel_event = threading.Event()

    async def emit(frame):
        frames.append(frame)
        if cancel_after is not None and len(frames) >= cancel_after:
            cancel_event.set()

    request = ChatRequest.model_validate(payload)
    await session.handle_stream(request, emit, cancel_event)
    return session_id, frames


def _client(app):
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://testserver")


@pytest.mark.asyncio
async def test_ws_streaming_frame_order_and_completion(streaming_app):
    app, scripted, fake_tts = streaming_app

    _session_id, frames = await _run_stream(
        app,
        {"message": "How does dopamine work?", "user_id": "ws-1"},
    )

    types_seen = [f["type"] for f in frames]
    assert types_seen[0] == "session"
    assert types_seen[-1] == "done"

    first_token_idx = types_seen.index("token")
    audio_indices = [i for i, t in enumerate(types_seen) if t == "audio"]
    assert audio_indices, "expected at least one audio frame"
    assert first_token_idx < audio_indices[0], "tokens must arrive before audio"

    audio_seqs = [frames[i]["seq"] for i in audio_indices]
    assert audio_seqs == sorted(
        audio_seqs
    ), "audio frames must be in ascending seq order"

    sentence_frames = [f for f in frames if f["type"] == "sentence"]
    assert len(sentence_frames) == 2
    assert sentence_frames[0]["emotion"] == "happy"
    assert sentence_frames[1]["emotion"] == "relaxed"

    done_frame = frames[-1]
    assert done_frame["cancelled"] is False
    assert "Dopamine reinforces reward learning" in done_frame["response"]["reply"]
    assert isinstance(done_frame["response"]["turn_id"], str)
    assert done_frame["response"]["turn_id"]

    # Latency breakdown rides the done frame as transport metadata.
    timings = done_frame["timings"]
    assert timings["ttft_ms"] is not None
    assert timings["ttfa_ms"] is not None
    assert timings["ttft_ms"] <= timings["total_ms"]
    assert timings["ttfa_ms"] <= timings["total_ms"]

    # TTS should have received the Japanese text, not the English.
    assert any("ドーパミン" in call or "記憶" in call for call in fake_tts.calls)


@pytest.mark.asyncio
async def test_ws_turn_records_latency_telemetry(streaming_app):
    """One streamed turn accumulates the latency.* sum/count/last values that
    /api/status consumers average, and the done frame carries the milestone
    breakdown (including first_sentence_ms)."""
    from amadeus import telemetry

    app, _scripted, _fake_tts = streaming_app
    telemetry.reset()
    try:
        _session_id, frames = await _run_stream(
            app, {"message": "How fast are you?", "user_id": "ws-lat"}
        )
        done_frame = frames[-1]
        timings = done_frame["timings"]
        assert timings["first_sentence_ms"] is not None
        assert timings["ttft_ms"] <= timings["first_sentence_ms"] <= timings["total_ms"]

        snap = telemetry.snapshot()
        for phase in ("ttft", "first_sentence", "first_audio", "turn_total"):
            assert snap.get(f"latency.{phase}_count") == 1, (phase, snap)
            assert snap.get(f"latency.{phase}_ms_sum", -1) >= 0, (phase, snap)
            assert snap.get(f"latency.last_{phase}_ms", -1) >= 0, (phase, snap)
        # The pipeline also measures each whole-unit synthesis call.
        assert snap.get("latency.tts_synth_count", 0) >= 1
        assert snap.get("latency.tts_synth_ms_sum", -1) >= 0
    finally:
        telemetry.reset()


@pytest.mark.asyncio
async def test_ws_memory_committed_exactly_once(streaming_app):
    app, scripted, fake_tts = streaming_app
    store = app.state.session_store

    session_id, _frames = await _run_stream(
        app,
        {"message": "Tell me about dopamine.", "user_id": "ws-mem"},
    )

    assert session_id
    session = store.sessions[session_id]
    assistant_messages = [m for m in session.memory.short_term if m.role == "assistant"]
    assert len(assistant_messages) == 1
    assert "Dopamine reinforces reward learning" in assistant_messages[0].content
    # Tags must never leak into persisted memory.
    assert "[happy]" not in assistant_messages[0].content
    assert "||" not in assistant_messages[0].content


@pytest.mark.asyncio
async def test_ws_provider_failure_rolls_back_user_message(monkeypatch, tmp_path):
    """A provider crash mid-stream must undo the user-message commit.

    _prepare_turn records the user message before generation; without the
    rollback a retry double-records it (the POST path has the same guard in
    sessions._handle).
    """
    config = _streaming_config(tmp_path)

    class ExplodingProvider(DummyProvider):
        def supports_stream(self) -> bool:
            return True

        def stream(self, messages, **kwargs):
            yield "[happy] Partial"
            raise RuntimeError("quota exceeded")

        def generate(self, messages, **kwargs):
            raise RuntimeError("quota exceeded")

    import amadeus.web.app as app_module

    monkeypatch.setattr(app_module, "load_llm", lambda cfg: ExplodingProvider())
    monkeypatch.setattr(app_module, "load_tts", lambda cfg: None)

    app = create_app(config)
    store = app.state.session_store
    _session_id, session = store.get(None, "ws-rollback")
    request = ChatRequest.model_validate(
        {"message": "Will this survive a crash?", "user_id": "ws-rollback"}
    )

    async def emit(frame):
        pass

    with pytest.raises(RuntimeError, match="quota exceeded"):
        await session.handle_stream(request, emit, threading.Event())

    assert [m for m in session.memory.short_term if m.role == "user"] == []

    # A retry against a healthy provider records the message exactly once.
    session.llm = ScriptedProvider(
        scripted_reply="[neutral] Recovered fine. || もう大丈夫よ。"
    )
    await session.handle_stream(request, emit, threading.Event())
    user_messages = [m for m in session.memory.short_term if m.role == "user"]
    assert len(user_messages) == 1


@pytest.mark.asyncio
async def test_ws_untagged_reply_falls_back_gracefully(monkeypatch, tmp_path):
    config = _streaming_config(tmp_path)
    plain = ScriptedProvider(
        scripted_reply="This is a completely untagged plain reply from the model."
    )

    import amadeus.web.app as app_module

    monkeypatch.setattr(app_module, "load_llm", lambda cfg: plain)
    monkeypatch.setattr(app_module, "load_tts", lambda cfg: None)

    app = create_app(config)
    _session_id, frames = await _run_stream(
        app,
        {"message": "Say something plain.", "user_id": "ws-fallback"},
    )

    types_seen = [f["type"] for f in frames]
    assert "token" in types_seen
    done_frame = frames[-1]
    assert "untagged plain reply" in done_frame["response"]["reply"]


@pytest.mark.asyncio
async def test_ws_cancel_commits_partial_text(monkeypatch, tmp_path):
    config = _streaming_config(tmp_path)
    long_reply = " ".join(f"word{i}" for i in range(200))
    slow = ScriptedProvider(scripted_reply=long_reply)

    import amadeus.web.app as app_module

    monkeypatch.setattr(app_module, "load_llm", lambda cfg: slow)
    monkeypatch.setattr(app_module, "load_tts", lambda cfg: None)

    app = create_app(config)
    _session_id, frames = await _run_stream(
        app,
        {"message": "Say a lot.", "user_id": "ws-cancel"},
        cancel_after=3,
    )

    done_frame = frames[-1]
    assert done_frame["type"] == "done"
    # Timings must be present even on cancelled turns.
    assert done_frame["timings"]["total_ms"] >= 0


class _RateLimitedError(RuntimeError):
    """Duck-typed 429 the failover classifier recognizes structurally."""

    status_code = 429


def _exploding_provider(exc: Exception):
    class Exploding(DummyProvider):
        def supports_stream(self) -> bool:
            return True

        def stream(self, messages, **kwargs):
            raise exc
            yield  # pragma: no cover - makes this a generator

        def generate(self, messages, **kwargs):
            raise exc

    return Exploding()


def _ws_error_frame(app, message: str = "hi"):
    """Drive the real /api/chat/stream endpoint (routes_chat's exception
    handler emits the error frame — session.handle_stream alone re-raises)
    and return the first error frame."""
    from starlette.testclient import TestClient

    client = TestClient(app)
    with client.websocket_connect("/api/chat/stream") as ws:
        ws.send_json({"type": "chat", "message": message, "user_id": "ws-err"})
        while True:
            frame = ws.receive_json()
            if frame["type"] == "error":
                return frame


@pytest.mark.asyncio
async def test_ws_provider_429_error_frame_carries_rate_limited_kind(
    monkeypatch, tmp_path
):
    """A provider 429 must be distinguishable from a dropped connection:
    the error frame gets an additive kind='rate_limited' the frontend
    captions honestly (the WS itself was fine)."""
    config = _streaming_config(tmp_path)

    import amadeus.web.app as app_module

    monkeypatch.setattr(
        app_module,
        "load_llm",
        lambda cfg: _exploding_provider(_RateLimitedError("too many requests")),
    )
    monkeypatch.setattr(app_module, "load_tts", lambda cfg: None)

    frame = _ws_error_frame(create_app(config))
    assert frame["kind"] == "rate_limited"
    # Additive change only: the legacy fields keep their exact shape.
    assert frame["fallback_to_post"] is True
    assert "too many requests" in frame["message"]


@pytest.mark.asyncio
async def test_ws_generic_provider_failure_reports_provider_error_kind(
    monkeypatch, tmp_path
):
    config = _streaming_config(tmp_path)

    import amadeus.web.app as app_module

    monkeypatch.setattr(
        app_module,
        "load_llm",
        lambda cfg: _exploding_provider(RuntimeError("model exploded")),
    )
    monkeypatch.setattr(app_module, "load_tts", lambda cfg: None)

    frame = _ws_error_frame(create_app(config))
    assert frame["kind"] == "provider_error"
    assert frame["fallback_to_post"] is True


def test_ws_turn_limiter_frame_carries_rate_limited_kind(tmp_path):
    """The shared-instance turn limiter is a rate limit too — its error
    frame must say so instead of reading like a dropped connection."""
    from starlette.testclient import TestClient

    config = _streaming_config(tmp_path)
    app = create_app(config)

    class DenyAll:
        def allow(self, key):
            return False

    app.state.turn_limiter = DenyAll()
    client = TestClient(app)
    with client.websocket_connect("/api/chat/stream") as ws:
        ws.send_json({"type": "chat", "message": "hi", "user_id": "ws-limited"})
        frame = ws.receive_json()
    assert frame["type"] == "error"
    assert frame["kind"] == "rate_limited"
    assert "Rate limit" in frame["message"]


def test_provider_error_kind_classification():
    """Structured checks (attrs/type names, incl. the cause chain) win;
    message text is only the fallback."""
    from amadeus.web.routes_chat import _provider_error_kind

    # Structural: status_code attribute.
    assert _provider_error_kind(_RateLimitedError("nope")) == "rate_limited"

    # Structural: SDK type name (Gemini quota errors).
    ResourceExhausted = type("ResourceExhausted", (Exception,), {})
    assert _provider_error_kind(ResourceExhausted("quota")) == "rate_limited"

    # Cause chain: a wrapped 429 is still recognized.
    wrapped = RuntimeError("provider call failed")
    wrapped.__cause__ = _RateLimitedError("inner")
    assert _provider_error_kind(wrapped) == "rate_limited"

    # Text fallback for stringified errors (e.g. ProviderCallError messages).
    assert (
        _provider_error_kind(RuntimeError("HTTP 429 from upstream")) == "rate_limited"
    )
    assert (
        _provider_error_kind(RuntimeError("Quota exceeded for model")) == "rate_limited"
    )

    # Everything else is a generic provider error — never mislabeled.
    assert _provider_error_kind(RuntimeError("model exploded")) == "provider_error"


@pytest.mark.asyncio
async def test_ws_disabled_returns_fallback_error(tmp_path):
    config = _streaming_config(tmp_path)
    config.web.streaming_enabled = False
    app = create_app(config)
    assert app.state.session_store.config.web.streaming_enabled is False


@pytest.mark.asyncio
async def test_healthz_exposes_streaming_capability(tmp_path):
    config = _streaming_config(tmp_path)
    config.web.streaming_enabled = True
    app = create_app(config)
    async with _client(app) as client:
        resp = await client.get("/healthz")
    assert resp.json()["streaming"] is True

    config2 = _streaming_config(tmp_path / "2")
    config2.web.streaming_enabled = False
    app2 = create_app(config2)
    async with _client(app2) as client2:
        resp2 = await client2.get("/healthz")
    assert resp2.json()["streaming"] is False


@pytest.mark.asyncio
async def test_post_endpoint_still_works_after_ws_addition(tmp_path):
    """Regression: adding the WS route must not disturb the POST contract."""
    config = _streaming_config(tmp_path)
    app = create_app(config)
    async with _client(app) as client:
        resp = await client.post(
            "/api/chat", json={"message": "Hello", "user_id": "post-check"}
        )
    assert resp.status_code == 200
    assert resp.json()["session_id"]


@pytest.mark.asyncio
async def test_ws_tool_turn_emits_tool_progress_frames(monkeypatch, tmp_path):
    """Tool-augmented WS turns surface per-tool activity frames."""
    from amadeus.providers.llm.base import ToolTurn

    class NativeToolLLM(DummyProvider):
        def __init__(self):
            super().__init__()
            self.turns = [
                ToolTurn(tool_name="calculator", args={"expression": "2 + 2"}),
                ToolTurn(text="[neutral] The answer is four. || 答えは4よ。"),
            ]

        def supports_tools(self):
            return True

        def generate_with_tools(self, messages, tools, **kwargs):
            return self.turns.pop(0)

    config = _streaming_config(tmp_path)
    config.audio.enable_voice = False
    config.tools.enabled = True
    config.tools.gate = "always"

    import amadeus.web.app as app_module

    monkeypatch.setattr(app_module, "load_llm", lambda cfg: NativeToolLLM())
    app = create_app(config)

    _, frames = await _run_stream(app, {"message": "what is 2 + 2?"})

    tool_frames = [f for f in frames if f["type"] == "tool_progress"]
    assert [(f["tool"], f["status"]) for f in tool_frames] == [
        ("calculator", "running"),
        ("calculator", "done"),
    ]
    # Progress frames precede the streamed sentences.
    first_sentence = next(i for i, f in enumerate(frames) if f["type"] == "sentence")
    assert frames.index(tool_frames[-1]) < first_sentence
    assert frames[-1]["type"] == "done"


def test_sanitize_speech_for_clone():
    from amadeus.web.streaming import sanitize_speech_for_clone as s

    # File names and paths inside Japanese become a spoken placeholder.
    assert s(
        "そのpsilocybin_v3.docxファイルはレポートでしょう。"
    ) == "そのそのファイルファイルはレポートでしょう。" or "そのファイル" in s("そのpsilocybin_v3.docxファイルはレポートでしょう。")
    assert "dementia_manuscript" not in s("dementia_manuscript フォルダを見たわ。")
    # Long latin runs get truncated, short ones survive.
    out = s("Multiscale reorganization of functional brain geometry の話ね。")
    assert "Multiscale reorganization of functional" not in out
    assert s("DNAの構造よ。") == "DNAの構造よ。"
    # Pure English (legacy fallback path) passes through untouched.
    english = "This is the untagged fallback sentence."
    assert s(english) == english
    # Bracketed stage directions are stripped on every path — a tag repeated
    # on the Japanese side ("[amused]") was being literally read aloud.
    assert s("[amused] まあ、そういうことね。") == "まあ、そういうことね。"
    assert s("そうね、[laugh]悪くないわ。") == "そうね、悪くないわ。"
    assert s("[excited] Not bad at all.") == "Not bad at all."
    # Japanese quoting brackets are untouched — they're not stage directions.
    assert s("「まあね」と言ったのよ。") == "「まあね」と言ったのよ。"


class FakeEmotionTTS(FakeTTS):
    """FakeTTS that advertises one emotion ref and records what it receives."""

    def __init__(self) -> None:
        super().__init__()
        self.available_emotions = frozenset({"happy"})
        self.emotion_calls: list[str | None] = []

    def synthesize(
        self,
        text: str,
        *,
        speaking_rate: float = 1.0,
        pitch_shift: float = 0.0,
        emotion: str | None = None,
    ) -> bytes:
        self.emotion_calls.append(emotion)
        return super().synthesize(
            text, speaking_rate=speaking_rate, pitch_shift=pitch_shift
        )


@pytest.mark.asyncio
async def test_ws_streaming_threads_dominant_emotion_to_tts(monkeypatch, tmp_path):
    """End-to-end: parsed sentence emotions reach the provider as the
    dominant label of the synthesis unit (full-reply mode), and labels the
    provider has no reference for (relaxed) never win over ones it has."""
    config = _streaming_config(tmp_path)
    config.web.streaming_tts_full_reply = True

    scripted = ScriptedProvider(
        scripted_reply=(
            "[happy] Great news about the results. || 結果について良い知らせよ。\n"
            "[relaxed] Anyway, back to work. || まあ、仕事に戻るわ。\n"
            "[happy] I am honestly pleased. || 正直、嬉しいわ。"
        )
    )
    fake_tts = FakeEmotionTTS()

    import amadeus.web.app as app_module

    monkeypatch.setattr(app_module, "load_llm", lambda cfg: scripted)
    monkeypatch.setattr(app_module, "load_tts", lambda cfg: fake_tts)

    app = create_app(config)
    _session_id, frames = await _run_stream(
        app, {"message": "How did it go?", "user_id": "ws-emotion"}
    )

    assert frames[-1]["type"] == "done"
    # One full-reply synthesis unit, voiced with the dominant available label.
    assert fake_tts.emotion_calls == ["happy"]


@pytest.mark.asyncio
async def test_ws_streaming_applies_request_audio_toggle(streaming_app):
    """The WS path must honor request.audio_enabled on the session: a muted
    turn produces no audio frames, and unmuting on the next turn recovers
    (previously only the POST path applied it, so a session muted via POST
    stayed silent over WS forever)."""
    app, scripted, fake_tts = streaming_app

    session_id, frames = await _run_stream(
        app,
        {"message": "Quiet please.", "user_id": "ws-mute", "audio_enabled": False},
    )
    assert not [f for f in frames if f["type"] == "audio"]

    _sid, frames = await _run_stream(
        app,
        {
            "message": "Speak up again.",
            "user_id": "ws-mute",
            "session_id": session_id,
            "audio_enabled": True,
        },
    )
    assert [f for f in frames if f["type"] == "audio"]


@pytest.mark.asyncio
async def test_ws_nonverbal_line_splices_clip_between_sentences(monkeypatch, tmp_path):
    """A standalone [laugh] line plays its pre-recorded clip at its seq slot
    and never leaks into tokens, sentence frames, or the committed reply."""
    import base64

    from amadeus.web.streaming import NonVerbalLibrary
    import amadeus.web.streaming.turn as turn_module

    config = _streaming_config(tmp_path)
    scripted = ScriptedProvider(
        scripted_reply=(
            "[relaxed] Fine, you got me. || いいわ、認めるわよ。\n"
            "[laugh]\n"
            "[happy] You should see your face. || その顔、見ものね。"
        )
    )
    fake_tts = FakeTTS()

    import amadeus.web.app as app_module

    monkeypatch.setattr(app_module, "load_llm", lambda cfg: scripted)
    monkeypatch.setattr(app_module, "load_tts", lambda cfg: fake_tts)
    library = NonVerbalLibrary({"laugh": [b"LAUGH-CLIP"]})
    monkeypatch.setattr(turn_module, "get_nonverbal_library", lambda: library)

    app = create_app(config)
    _session_id, frames = await _run_stream(
        app, {"message": "Admit it.", "user_id": "ws-nv"}
    )

    audio_frames = [f for f in frames if f["type"] == "audio"]
    audio_seqs = [f["seq"] for f in audio_frames]
    assert audio_seqs == sorted(audio_seqs)
    clip_frames = [
        f for f in audio_frames if base64.b64decode(f["data"]) == b"LAUGH-CLIP"
    ]
    assert len(clip_frames) == 1
    assert clip_frames[0]["seq"] == 2  # between sentence 1 and sentence 3
    # Synthesized audio exists on both sides of the clip.
    assert any(f["seq"] < 2 for f in audio_frames)
    assert any(f["seq"] > 2 for f in audio_frames)

    sentence_frames = [f for f in frames if f["type"] == "sentence"]
    assert [f["seq"] for f in sentence_frames] == [1, 3]
    token_text = "".join(f["text"] for f in frames if f["type"] == "token")
    assert "laugh" not in token_text
    done = frames[-1]
    assert "laugh" not in done["response"]["reply"]
    assert "You should see your face." in done["response"]["reply"]


@pytest.mark.asyncio
async def test_ws_cancel_before_first_token_rolls_back_user_message(
    monkeypatch, tmp_path
):
    """A barge-in before any emitted text must roll the user message back —
    committing it reply-less left a live trailing orphan in short_term that
    only healed at the next process start."""
    config = _streaming_config(tmp_path)
    scripted = ScriptedProvider(scripted_reply="[happy] Hello. || こんにちは。")

    import amadeus.web.app as app_module

    monkeypatch.setattr(app_module, "load_llm", lambda cfg: scripted)
    monkeypatch.setattr(app_module, "load_tts", lambda cfg: None)

    app = create_app(config)
    store = app.state.session_store
    session_id, session = store.get(None, "ws-cancel-early")
    cancel_event = threading.Event()
    cancel_event.set()  # cancel lands before the provider emits anything
    frames: list[dict] = []

    async def emit(frame):
        frames.append(frame)

    request = ChatRequest.model_validate(
        {"message": "Say hi.", "user_id": "ws-cancel-early", "session_id": session_id}
    )
    await session.handle_stream(request, emit, cancel_event)

    done = frames[-1]
    assert done["type"] == "done"
    assert done["cancelled"] is True
    assert done["response"]["reply"] == ""
    assert done["response"]["turn_id"] is None
    # As if the turn never happened: no trailing orphan user message.
    assert [m for m in session.memory.short_term if m.role == "user"] == []


class NonStreamingScripted(ScriptedProvider):
    def supports_stream(self) -> bool:
        return False


@pytest.mark.asyncio
async def test_ws_tagless_line_in_tagged_reply_shows_and_commits_clean_text(
    monkeypatch, tmp_path
):
    """One line dropping its [emotion] tag mid-reply used to (a) leak the
    raw 'English || 日本語' wire format into the sentence frame and TTS and
    (b) vanish from the token stream and committed reply entirely (no
    DisplayDelta for tagless lines) — display and memory diverged from what
    was spoken. The degrade path must split the halves and keep display,
    frames, and the committed reply in agreement."""
    config = _streaming_config(tmp_path)
    scripted = ScriptedProvider(
        scripted_reply=(
            "[happy] First line fine. || 一行目。\n"
            "Second line lost its tag. || 二行目。"
        )
    )

    import amadeus.web.app as app_module

    monkeypatch.setattr(app_module, "load_llm", lambda cfg: scripted)
    monkeypatch.setattr(app_module, "load_tts", lambda cfg: None)

    app = create_app(config)
    store = app.state.session_store
    _session_id, frames = await _run_stream(
        app, {"message": "Say two things.", "user_id": "ws-tagless"}
    )

    sentence_frames = [f for f in frames if f["type"] == "sentence"]
    assert len(sentence_frames) == 2
    assert sentence_frames[1]["en"] == "Second line lost its tag."
    assert sentence_frames[1]["ja"] == "二行目。"

    token_text = "".join(f["text"] for f in frames if f["type"] == "token")
    assert "Second line lost its tag." in token_text
    assert "||" not in token_text
    assert "二行目" not in token_text

    done = frames[-1]
    reply = done["response"]["reply"]
    assert "First line fine." in reply
    assert "Second line lost its tag." in reply
    assert "||" not in reply
    assert "二行目" not in reply

    # Memory matches what was shown.
    session = store.sessions[_session_id]
    assistant = [m for m in session.memory.short_term if m.role == "assistant"]
    assert len(assistant) == 1
    assert "Second line lost its tag." in assistant[0].content
    assert "||" not in assistant[0].content


@pytest.mark.asyncio
async def test_ws_empty_uncancelled_reply_rolls_back_user_message(
    monkeypatch, tmp_path
):
    """A provider that streams ZERO deltas on an uncancelled turn used to
    reach complete_turn(msg, "") — which skips the assistant append and
    strands the user message as a trailing orphan until the next restart's
    _load healing. It must roll back like the cancelled-empty case (with
    its own counter)."""
    from amadeus import telemetry

    config = _streaming_config(tmp_path)

    class SilentProvider(DummyProvider):
        def supports_stream(self) -> bool:
            return True

        def stream(self, messages, **kwargs):
            return iter(())

        def generate(self, messages, **kwargs):
            return ""

    import amadeus.web.app as app_module

    monkeypatch.setattr(app_module, "load_llm", lambda cfg: SilentProvider())
    monkeypatch.setattr(app_module, "load_tts", lambda cfg: None)

    app = create_app(config)
    store = app.state.session_store
    _session_id, session = store.get(None, "ws-empty")
    frames: list[dict] = []

    async def emit(frame):
        frames.append(frame)

    before = telemetry.snapshot().get("web.empty_reply_rolled_back", 0)
    request = ChatRequest.model_validate(
        {"message": "Say something.", "user_id": "ws-empty"}
    )
    await session.handle_stream(request, emit, threading.Event())

    done = frames[-1]
    assert done["type"] == "done"
    assert done["cancelled"] is False
    assert done["response"]["reply"] == ""
    assert done["response"]["turn_id"] is None
    assert [m for m in session.memory.short_term if m.role == "user"] == []
    assert telemetry.snapshot().get("web.empty_reply_rolled_back", 0) == before + 1


@pytest.mark.asyncio
async def test_ws_thought_only_reply_never_leaks_and_rolls_back(monkeypatch, tmp_path):
    """A pseudo-streamed reply of ONLY [thought]/non-verbal lines parses as
    tagged with zero sentences; the untagged fallback then ran
    strip_bilingual_tails on the RAW text, showing and speaking the literal
    '[thought] ...' line. Those lines must be stripped, and the resulting
    empty turn rolls the user message back."""
    config = _streaming_config(tmp_path)
    scripted = NonStreamingScripted(
        scripted_reply="[thought] A private aside about them.\n[laugh]"
    )

    import amadeus.web.app as app_module

    monkeypatch.setattr(app_module, "load_llm", lambda cfg: scripted)
    monkeypatch.setattr(app_module, "load_tts", lambda cfg: None)

    app = create_app(config)
    store = app.state.session_store
    _session_id, frames = await _run_stream(
        app, {"message": "Anything to add?", "user_id": "ws-thought-only"}
    )

    leaked = [
        f
        for f in frames
        if f["type"] in ("token", "sentence") and "private aside" in str(f).lower()
    ]
    assert leaked == []
    done = frames[-1]
    assert done["response"]["reply"] == ""
    assert "[thought]" not in done["response"]["reply"]
    session = store.sessions[_session_id]
    assert [m for m in session.memory.short_term if m.role == "user"] == []


@pytest.mark.asyncio
async def test_ws_pre_cancelled_turn_never_calls_provider(monkeypatch, tmp_path):
    """A turn whose cancel_event is already set (duplex barge-in during the
    STT window) used to still run a full provider stream() and wait for the
    first token. It must short-circuit — no provider call — while keeping
    the done-frame/rollback flow the client expects."""
    config = _streaming_config(tmp_path)

    class CountingProvider(DummyProvider):
        def __init__(self) -> None:
            super().__init__(scripted_reply="[happy] Hi. || やあ。")
            self.stream_calls = 0
            self.generate_calls = 0

        def supports_stream(self) -> bool:
            return True

        def stream(self, messages, **kwargs):
            self.stream_calls += 1
            yield self.scripted_reply

        def generate(self, messages, **kwargs):
            self.generate_calls += 1
            return self.scripted_reply

    counting = CountingProvider()

    import amadeus.web.app as app_module

    monkeypatch.setattr(app_module, "load_llm", lambda cfg: counting)
    monkeypatch.setattr(app_module, "load_tts", lambda cfg: None)

    app = create_app(config)
    store = app.state.session_store
    _session_id, session = store.get(None, "ws-precancel")
    cancel_event = threading.Event()
    cancel_event.set()
    frames: list[dict] = []

    async def emit(frame):
        frames.append(frame)

    request = ChatRequest.model_validate(
        {"message": "Too late.", "user_id": "ws-precancel"}
    )
    await session.handle_stream(request, emit, cancel_event)

    assert counting.stream_calls == 0
    assert counting.generate_calls == 0
    done = frames[-1]
    assert done["type"] == "done"
    assert done["cancelled"] is True
    assert done["response"]["reply"] == ""
    assert [m for m in session.memory.short_term if m.role == "user"] == []


@pytest.mark.asyncio
async def test_ws_provider_crash_aborts_queued_tts_instead_of_flushing(
    monkeypatch, tmp_path
):
    """When the provider raises mid-stream, the finally used to run
    drain(cancelled=False) → pipeline.close(), which synthesized and
    EMITTED every queued sentence before the error could surface — she
    kept talking for seconds, then errored, and the heard sentences rolled
    back anyway. Exception unwinding must abort the pipeline instead."""
    config = _streaming_config(tmp_path)
    config.web.streaming_tts_full_reply = True  # sentences held until drain

    class ExplodingMidStream(DummyProvider):
        def supports_stream(self) -> bool:
            return True

        def stream(self, messages, **kwargs):
            yield "[happy] One full sentence here. || 一文目。\n"
            raise RuntimeError("provider died mid-stream")

        def generate(self, messages, **kwargs):
            raise RuntimeError("provider died mid-stream")

    fake_tts = FakeTTS()

    import amadeus.web.app as app_module

    monkeypatch.setattr(app_module, "load_llm", lambda cfg: ExplodingMidStream())
    monkeypatch.setattr(app_module, "load_tts", lambda cfg: fake_tts)

    app = create_app(config)
    store = app.state.session_store
    _session_id, session = store.get(None, "ws-crash-abort")
    frames: list[dict] = []

    async def emit(frame):
        frames.append(frame)

    request = ChatRequest.model_validate(
        {"message": "Talk to me.", "user_id": "ws-crash-abort"}
    )
    with pytest.raises(RuntimeError, match="provider died"):
        await session.handle_stream(request, emit, threading.Event())

    # Aborted, not flushed: nothing was synthesized or emitted as audio.
    assert fake_tts.calls == []
    assert [f for f in frames if f["type"] == "audio"] == []
    # And the user message rolled back (existing provider-failure contract).
    assert [m for m in session.memory.short_term if m.role == "user"] == []


@pytest.mark.asyncio
async def test_ws_cancel_pseudo_stream_commits_only_emitted_sentences(
    monkeypatch, tmp_path
):
    """The pseudo-stream path (tools/refusals/non-streaming providers) must
    commit only the sentences that were actually emitted before the cancel —
    it used to commit ALL parsed sentences, recording a reply the user never
    saw and then telling the persona not to re-deliver it."""
    config = _streaming_config(tmp_path)
    scripted = NonStreamingScripted(
        scripted_reply=(
            "[happy] First sentence here. || 一つ目。\n"
            "[relaxed] Second sentence here. || 二つ目。"
        )
    )

    import amadeus.web.app as app_module

    monkeypatch.setattr(app_module, "load_llm", lambda cfg: scripted)
    monkeypatch.setattr(app_module, "load_tts", lambda cfg: None)

    app = create_app(config)
    # Frames: [session, token, sentence, ...] — cancel after the first
    # sentence frame so the second parsed sentence is never emitted.
    _session_id, frames = await _run_stream(
        app,
        {"message": "Say two things.", "user_id": "ws-cancel-pseudo"},
        cancel_after=3,
    )

    done = frames[-1]
    assert done["type"] == "done"
    assert done["cancelled"] is True
    assert "First sentence here." in done["response"]["reply"]
    assert "Second sentence here." not in done["response"]["reply"]
    assert done["response"]["reply"].endswith("[interrupted]")
