"""Duration telemetry, LLM provider health classification, and the
RATE_LIMITED_REPLY in-character quota message.

Four areas, four sections below:
- telemetry.record_duration/duration_snapshot: the new count/total/max/last
  facility (amadeus/telemetry.py), independent of the pre-existing
  incr/set_gauge-based latency.* counters.
- provider_health: the ok/rate_limited/invalid_key/error/dummy state
  machine (amadeus/provider_health.py) and its wiring into
  ChatEngine._run_tool_loop, including the self-heal-on-success case.
- provider_health TTS: the ok/warming/degraded/failing/off state machine
  (amadeus/provider_health.py's TTS half) and its /api/status exposure.
- A WS integration test asserting turn.first_audio_ms / tts.first_audio_ms
  actually get recorded on a real streamed turn (harness copied from
  tests/test_streaming_ws.py).
"""

from __future__ import annotations

import asyncio
import threading

import pytest

from amadeus import provider_health, telemetry
from amadeus.config import Config
from amadeus.engine import ChatEngine, INVALID_KEY_REPLY, RATE_LIMITED_REPLY
from amadeus.providers.llm.base import ChatMessage, DummyProvider, LLMProvider, ToolTurn
from amadeus.web.app import ChatRequest, create_app

# ---------------------------------------------------------------------------
# telemetry.record_duration / duration_snapshot
# ---------------------------------------------------------------------------


def test_record_duration_accumulates_count_total_max_last():
    telemetry.reset()
    try:
        telemetry.record_duration("turn.first_audio_ms", 100.0)
        telemetry.record_duration("turn.first_audio_ms", 300.0)
        telemetry.record_duration("turn.first_audio_ms", 200.0)

        snap = telemetry.duration_snapshot()
        entry = snap["turn.first_audio_ms"]
        assert entry["count"] == 3
        assert entry["total_ms"] == pytest.approx(600.0)
        assert entry["max_ms"] == pytest.approx(300.0)
        # last_ms reflects the most recent call, not the largest.
        assert entry["last_ms"] == pytest.approx(200.0)
    finally:
        telemetry.reset()


def test_duration_snapshot_keys_are_independent():
    telemetry.reset()
    try:
        telemetry.record_duration("stt.request_ms", 50.0)
        telemetry.record_duration("llm.first_token_ms", 900.0)
        snap = telemetry.duration_snapshot()
        assert set(snap.keys()) == {"stt.request_ms", "llm.first_token_ms"}
        assert snap["stt.request_ms"]["count"] == 1
        assert snap["llm.first_token_ms"]["count"] == 1
    finally:
        telemetry.reset()


def test_duration_snapshot_is_a_copy_not_live_state():
    telemetry.reset()
    try:
        telemetry.record_duration("k", 1.0)
        snap = telemetry.duration_snapshot()
        snap["k"]["count"] = 999
        assert telemetry.duration_snapshot()["k"]["count"] == 1
    finally:
        telemetry.reset()


def test_reset_clears_durations_too():
    telemetry.record_duration("k", 1.0)
    telemetry.reset()
    assert telemetry.duration_snapshot() == {}


# ---------------------------------------------------------------------------
# provider_health: state machine
# ---------------------------------------------------------------------------


def test_llm_health_defaults_to_ok():
    provider_health.reset()
    snap = provider_health.llm_health_snapshot()
    assert snap["state"] == "ok"
    assert snap["detail"] is None


def test_set_llm_health_records_state_detail_and_timestamp():
    provider_health.reset()
    try:
        provider_health.set_llm_health("rate_limited", "429 too many requests")
        snap = provider_health.llm_health_snapshot()
        assert snap["state"] == "rate_limited"
        assert snap["detail"] == "429 too many requests"
        assert snap["since"]  # ISO timestamp, non-empty
    finally:
        provider_health.reset()


def test_llm_health_self_heals_from_rate_limited_to_ok():
    """A quota death that recovers must clear itself on the next good call —
    the whole point of self-healing health (no manual reset endpoint)."""
    provider_health.reset()
    try:
        provider_health.set_llm_health("rate_limited", "quota exceeded")
        assert provider_health.llm_health_snapshot()["state"] == "rate_limited"
        provider_health.set_llm_health("ok")
        snap = provider_health.llm_health_snapshot()
        assert snap["state"] == "ok"
        assert snap["detail"] is None
    finally:
        provider_health.reset()


def test_set_llm_health_rejects_unknown_state_without_raising():
    provider_health.reset()
    try:
        provider_health.set_llm_health("bogus", "whatever")
        snap = provider_health.llm_health_snapshot()
        assert snap["state"] == "error"
        assert "bogus" in snap["detail"]
    finally:
        provider_health.reset()


class _Rate429(Exception):
    """Duck-typed 429 the structured classifier recognizes (same shape as
    tests/test_llm_fallback.py's _Rate429)."""

    status_code = 429


def test_classify_and_record_rate_limit():
    provider_health.reset()
    try:
        kind = provider_health.classify_and_record(_Rate429("too many requests"))
        assert kind == "rate_limited"
        assert provider_health.llm_health_snapshot()["state"] == "rate_limited"
    finally:
        provider_health.reset()


def test_classify_and_record_invalid_key():
    provider_health.reset()
    try:
        exc = RuntimeError("400 API key not valid. [reason: API_KEY_INVALID]")
        kind = provider_health.classify_and_record(exc)
        assert kind == "invalid_key"
        assert provider_health.llm_health_snapshot()["state"] == "invalid_key"
    finally:
        provider_health.reset()


def test_classify_and_record_generic_error():
    provider_health.reset()
    try:
        kind = provider_health.classify_and_record(ValueError("boom"))
        assert kind == "error"
        assert provider_health.llm_health_snapshot()["state"] == "error"
    finally:
        provider_health.reset()


# ---------------------------------------------------------------------------
# provider_health wiring: ChatEngine init + _run_tool_loop
# ---------------------------------------------------------------------------


def _engine(tmp_path, llm, gate="always") -> ChatEngine:
    config = Config()
    config.memory.persist_path = str(tmp_path / "memory.json")
    config.memory.embeddings_path = str(tmp_path / "embeddings.json")
    config.memory.profile_path = str(tmp_path / "profile.json")
    config.tools.enabled = True
    config.tools.gate = gate
    return ChatEngine(config, llm)


def _context():
    return [
        ChatMessage(role="system", content="You are Kurisu."),
        ChatMessage(role="user", content="hello there"),
    ]


class _NativeLLM(LLMProvider):
    """A minimal, non-dummy LLMProvider that supports native tool calling —
    deliberately NOT a DummyProvider subclass, so isinstance(llm,
    DummyProvider) in ChatEngine.__init__ is False like a real provider."""

    def generate(self, messages, **kwargs) -> str:
        return "plain reply"

    def stream(self, messages, **kwargs):
        yield self.generate(messages, **kwargs)

    def supports_stream(self) -> bool:
        return True

    def supports_tools(self) -> bool:
        return True

    def generate_with_tools(self, messages, tools, **kwargs):
        raise NotImplementedError  # overridden per test


def test_dummy_provider_marks_health_dummy_at_construction(tmp_path):
    provider_health.reset()
    try:
        _engine(tmp_path, DummyProvider())
        assert provider_health.llm_health_snapshot()["state"] == "dummy"
    finally:
        provider_health.reset()


def test_dummy_turn_does_not_self_heal_health_to_ok(tmp_path):
    """A DummyProvider canned reply is not evidence of provider health: the
    turn's success path must NOT flip the dummy stamp back to ok, or one
    chat message hides the "you're running on dummies" signal /api/status
    exists to surface. (Found live 2026-07-28: health read "ok" on a
    keyless instance after a single POST /api/chat.)"""
    provider_health.reset()
    try:
        engine = _engine(tmp_path, DummyProvider())
        assert provider_health.llm_health_snapshot()["state"] == "dummy"
        engine._generate_with_health([ChatMessage(role="user", content="hello")])
        assert provider_health.llm_health_snapshot()["state"] == "dummy"
    finally:
        provider_health.reset()


def test_record_llm_success_heals_for_real_providers():
    provider_health.reset()
    try:
        provider_health.set_llm_health("rate_limited", "quota exceeded")
        provider_health.record_llm_success(_NativeLLM())
        assert provider_health.llm_health_snapshot()["state"] == "ok"
    finally:
        provider_health.reset()


def test_real_provider_does_not_mark_dummy_at_construction(tmp_path):
    provider_health.reset()
    try:
        llm = _NativeLLM()
        llm.generate_with_tools = lambda messages, tools, **kw: ToolTurn(text="hi")
        _engine(tmp_path, llm)
        # Construction alone makes no claim about a real provider's health.
        assert provider_health.llm_health_snapshot()["state"] == "ok"
    finally:
        provider_health.reset()


def test_tool_loop_auth_error_sets_invalid_key_and_replies_in_character(tmp_path):
    provider_health.reset()
    try:
        llm = _NativeLLM()

        def explode(messages, tools, **kw):
            raise RuntimeError(
                "400 API key not valid. Please pass a valid API key. "
                "[reason: API_KEY_INVALID]"
            )

        llm.generate_with_tools = explode
        engine = _engine(tmp_path, llm)
        reply = engine._run_tool_loop(_context())
        assert reply == INVALID_KEY_REPLY
        assert provider_health.llm_health_snapshot()["state"] == "invalid_key"
    finally:
        provider_health.reset()


def test_tool_loop_rate_limit_sets_rate_limited_and_replies_in_character(tmp_path):
    """Mirrors the invalid-key path: a rate-limit error shaped exactly like
    the one _is_rate_limit already recognizes (structured, HTTP 429) must
    produce RATE_LIMITED_REPLY through the same emission mechanism as
    INVALID_KEY_REPLY, not a generic 'ran out of steps' recovery line or an
    uncaught exception."""
    provider_health.reset()
    try:
        llm = _NativeLLM()

        class _Rate429(Exception):
            status_code = 429

        def explode(messages, tools, **kw):
            raise _Rate429("quota exceeded")

        llm.generate_with_tools = explode
        engine = _engine(tmp_path, llm)
        reply = engine._run_tool_loop(_context())
        assert reply == RATE_LIMITED_REPLY
        assert "ran out of steps" not in reply
        assert provider_health.llm_health_snapshot()["state"] == "rate_limited"
    finally:
        provider_health.reset()


def test_tool_loop_success_self_heals_health_to_ok(tmp_path):
    """A prior rate_limited state must clear back to ok on the next
    successful generation — the self-healing contract."""
    provider_health.reset()
    try:
        provider_health.set_llm_health("rate_limited", "stale, from an earlier turn")
        llm = _NativeLLM()
        llm.generate_with_tools = lambda messages, tools, **kw: ToolTurn(
            text="Back to normal."
        )
        engine = _engine(tmp_path, llm)
        reply = engine._run_tool_loop(_context())
        assert reply == "Back to normal."
        assert provider_health.llm_health_snapshot()["state"] == "ok"
    finally:
        provider_health.reset()


# ---------------------------------------------------------------------------
# provider_health: TTS state machine (ok/warming/degraded/failing/off)
# ---------------------------------------------------------------------------


def test_tts_health_defaults_to_ok():
    provider_health.reset_tts()
    snap = provider_health.tts_health_snapshot()
    assert snap["state"] == "ok"
    assert snap["detail"] is None
    assert snap["consecutive_failures"] == 0


def test_single_tts_failure_is_degraded_not_failing():
    provider_health.reset_tts()
    try:
        state = provider_health.record_tts_failure(RuntimeError("cuda oom"))
        assert state == "degraded"
        snap = provider_health.tts_health_snapshot()
        assert snap["state"] == "degraded"
        assert "cuda oom" in snap["detail"]
        assert snap["consecutive_failures"] == 1
    finally:
        provider_health.reset_tts()


def test_three_consecutive_tts_failures_escalate_to_failing():
    provider_health.reset_tts()
    try:
        provider_health.record_tts_failure("boom 1")
        provider_health.record_tts_failure("boom 2")
        state = provider_health.record_tts_failure("boom 3")
        assert state == "failing"
        snap = provider_health.tts_health_snapshot()
        assert snap["state"] == "failing"
        assert snap["consecutive_failures"] == 3
    finally:
        provider_health.reset_tts()


def test_tts_health_self_heals_on_success_and_resets_streak():
    """A success after a failing streak must clear back to ok AND zero the
    consecutive-failure counter — otherwise a single failure right after
    recovery would immediately read as "failing" again instead of starting
    a fresh degraded->failing climb."""
    provider_health.reset_tts()
    try:
        provider_health.record_tts_failure("boom 1")
        provider_health.record_tts_failure("boom 2")
        provider_health.record_tts_failure("boom 3")
        assert provider_health.tts_health_snapshot()["state"] == "failing"

        provider_health.record_tts_success()
        snap = provider_health.tts_health_snapshot()
        assert snap["state"] == "ok"
        assert snap["detail"] is None
        assert snap["consecutive_failures"] == 0

        # One failure right after recovery is "degraded", not "failing".
        state = provider_health.record_tts_failure("one more")
        assert state == "degraded"
    finally:
        provider_health.reset_tts()


def test_tts_health_warming_passthrough_overrides_recorded_state():
    """warming=True always wins, even over a recorded failing streak from a
    previous provider instance — the new one hasn't spoken yet, so it isn't
    "failing", it's "warming"."""
    provider_health.reset_tts()
    try:
        provider_health.record_tts_failure("boom")
        provider_health.record_tts_failure("boom")
        provider_health.record_tts_failure("boom")
        snap = provider_health.tts_health_snapshot(loaded=False, warming=True)
        assert snap["state"] == "warming"
        assert snap["detail"] is None
    finally:
        provider_health.reset_tts()


def test_tts_health_off_when_not_loaded_and_not_warming():
    """No provider loaded and not warming (voice disabled/unconfigured) is
    "off", a normal state — never "failing", even with a stale failure
    streak recorded from before voice was turned off."""
    provider_health.reset_tts()
    try:
        provider_health.record_tts_failure("boom")
        provider_health.record_tts_failure("boom")
        provider_health.record_tts_failure("boom")
        snap = provider_health.tts_health_snapshot(loaded=False, warming=False)
        assert snap["state"] == "off"
    finally:
        provider_health.reset_tts()


def test_tts_health_loaded_and_not_warming_uses_recorded_state():
    provider_health.reset_tts()
    try:
        snap = provider_health.tts_health_snapshot(loaded=True, warming=False)
        assert snap["state"] == "ok"
    finally:
        provider_health.reset_tts()


def test_reset_clears_both_llm_and_tts_health():
    provider_health.set_llm_health("rate_limited", "quota")
    provider_health.record_tts_failure("boom")
    provider_health.reset()
    assert provider_health.llm_health_snapshot()["state"] == "ok"
    assert provider_health.tts_health_snapshot()["state"] == "ok"


# ---------------------------------------------------------------------------
# provider_health: TTS wiring into _synthesize_tts (the shared choke point
# for the deferred /api/tts endpoint and presence-tail synthesis) and
# GET /api/status's tts_health exposure.
# ---------------------------------------------------------------------------


def test_synthesize_tts_records_success(tmp_path):
    from amadeus.web.sessions import _synthesize_tts

    class _OkTTS:
        def synthesize(self, text, *, speaking_rate=1.0, pitch_shift=0.0):
            return b"wav bytes"

    provider_health.reset_tts()
    try:
        provider_health.record_tts_failure("stale failure from before")
        asyncio.run(
            _synthesize_tts(_OkTTS(), "hello", speaking_rate=1.0, pitch_shift=0.0)
        )
        assert provider_health.tts_health_snapshot()["state"] == "ok"
    finally:
        provider_health.reset_tts()


def test_synthesize_tts_records_failure_and_reraises(tmp_path):
    from amadeus.web.sessions import _synthesize_tts

    class _ExplodingTTS:
        def synthesize(self, text, *, speaking_rate=1.0, pitch_shift=0.0):
            raise RuntimeError("cuda out of memory")

    provider_health.reset_tts()
    try:
        with pytest.raises(RuntimeError, match="cuda out of memory"):
            asyncio.run(
                _synthesize_tts(
                    _ExplodingTTS(), "hello", speaking_rate=1.0, pitch_shift=0.0
                )
            )
        snap = provider_health.tts_health_snapshot()
        assert snap["state"] == "degraded"
        assert "cuda out of memory" in snap["detail"]
    finally:
        provider_health.reset_tts()


def _isolated_config(tmp_path) -> Config:
    config = Config()
    config.memory.persist_path = str(tmp_path / "memory.json")
    config.memory.embeddings_path = str(tmp_path / "embeddings.json")
    config.memory.profile_path = str(tmp_path / "profile.json")
    return config


def test_status_reports_tts_health_off_when_no_provider(tmp_path):
    """No TTS provider loaded and voice disabled -> tts_health is "off", not
    "failing", even though the module default state is "ok" until a real
    synthesis runs — off must win over a stale recorded state too."""
    import httpx

    provider_health.reset_tts()
    try:
        provider_health.record_tts_failure("stale")
        provider_health.record_tts_failure("stale")
        provider_health.record_tts_failure("stale")
        config = _isolated_config(tmp_path)
        config.audio.enable_voice = False
        app = create_app(config)
        transport = httpx.ASGITransport(app=app)

        async def _get():
            async with httpx.AsyncClient(
                transport=transport, base_url="http://testserver"
            ) as client:
                return await client.get("/api/status")

        response = asyncio.run(_get())
        payload = response.json()
        assert payload["tts_health"]["state"] == "off"
    finally:
        provider_health.reset_tts()


def test_status_reports_tts_health_state_key_shape(tmp_path):
    import httpx

    provider_health.reset_tts()
    try:
        config = _isolated_config(tmp_path)
        app = create_app(config)
        transport = httpx.ASGITransport(app=app)

        async def _get():
            async with httpx.AsyncClient(
                transport=transport, base_url="http://testserver"
            ) as client:
                return await client.get("/api/status")

        response = asyncio.run(_get())
        payload = response.json()
        assert "tts_health" in payload
        assert set(payload["tts_health"].keys()) == {
            "state",
            "detail",
            "since",
            "consecutive_failures",
        }
        assert payload["tts_health"]["state"] in {
            "ok",
            "warming",
            "degraded",
            "failing",
            "off",
        }
    finally:
        provider_health.reset_tts()


# ---------------------------------------------------------------------------
# WS integration: turn.first_audio_ms / tts.first_audio_ms recorded on a
# real streamed turn. Harness copied from tests/test_streaming_ws.py.
# ---------------------------------------------------------------------------

from amadeus.providers.tts.base import TTSProvider  # noqa: E402


class _ScriptedProvider(DummyProvider):
    pass


class _FakeTTS(TTSProvider):
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
    config.memory.persist_path = str(tmp_path / "memory.json")
    config.memory.embeddings_path = str(tmp_path / "embeddings.json")
    config.memory.enable_long_term = False
    return config


@pytest.mark.asyncio
async def test_ws_turn_records_first_audio_durations(monkeypatch, tmp_path):
    """A streamed turn with voice on records turn.first_audio_ms and
    tts.first_audio_ms in the new duration facility (not just the
    pre-existing latency.* counters test_streaming_ws.py already covers)."""
    config = _streaming_config(tmp_path)
    scripted = _ScriptedProvider(
        scripted_reply=(
            "[happy] Dopamine reinforces reward learning. || "
            "ドーパミンは報酬学習を強化する。\n"
            "[relaxed] It also supports memory consolidation. || "
            "記憶の定着も支援する。"
        )
    )
    fake_tts = _FakeTTS()

    import amadeus.web.app as app_module

    monkeypatch.setattr(app_module, "load_llm", lambda cfg: scripted)
    monkeypatch.setattr(app_module, "load_tts", lambda cfg: fake_tts)

    app = create_app(config)
    store = app.state.session_store
    session_id, session = store.get(None, "ws-duration")
    request = ChatRequest.model_validate(
        {"message": "How does dopamine work?", "user_id": "ws-duration"}
    )

    frames = []

    async def emit(frame):
        frames.append(frame)

    telemetry.reset()
    try:
        await session.handle_stream(request, emit, threading.Event())

        done_frame = [f for f in frames if f["type"] == "done"][0]
        assert done_frame["timings"]["ttfa_ms"] is not None

        snap = telemetry.duration_snapshot()
        assert snap["turn.first_audio_ms"]["count"] == 1
        assert snap["turn.first_audio_ms"]["last_ms"] >= 0
        assert snap["tts.first_audio_ms"]["count"] == 1
        assert snap["tts.first_audio_ms"]["last_ms"] >= 0

        # LLM-side generation durations from the same turn.
        assert snap["llm.first_token_ms"]["count"] == 1
        assert snap["llm.stream_total_ms"]["count"] == 1
        assert (
            snap["llm.first_token_ms"]["last_ms"]
            <= snap["llm.stream_total_ms"]["last_ms"]
        )

        assert provider_health.llm_health_snapshot()["state"] == "ok"
    finally:
        telemetry.reset()
        provider_health.reset()


@pytest.mark.asyncio
async def test_ws_provider_rate_limit_marks_health_without_changing_error_frame(
    monkeypatch, tmp_path
):
    """The plain (non-tool) streamed generate/stream path keeps its existing
    502/error-frame behavior unchanged (pinned by
    test_ws_provider_429_error_frame_carries_rate_limited_kind in
    test_streaming_ws.py) — this test only asserts the NEW side effect,
    that provider_health is classified alongside the unchanged error path.
    """
    config = _streaming_config(tmp_path)

    class _RateLimitedError(RuntimeError):
        status_code = 429

    class Exploding(DummyProvider):
        def supports_stream(self) -> bool:
            return True

        def stream(self, messages, **kwargs):
            raise _RateLimitedError("too many requests")
            yield  # pragma: no cover - makes this a generator

        def generate(self, messages, **kwargs):
            raise _RateLimitedError("too many requests")

    import amadeus.web.app as app_module

    monkeypatch.setattr(app_module, "load_llm", lambda cfg: Exploding())
    monkeypatch.setattr(app_module, "load_tts", lambda cfg: None)

    app = create_app(config)
    store = app.state.session_store
    _session_id, session = store.get(None, "ws-health-429")
    request = ChatRequest.model_validate(
        {"message": "Will this hit quota?", "user_id": "ws-health-429"}
    )

    async def emit(frame):
        pass

    provider_health.reset()
    try:
        with pytest.raises(RuntimeError, match="too many requests"):
            await session.handle_stream(request, emit, threading.Event())
        assert provider_health.llm_health_snapshot()["state"] == "rate_limited"
    finally:
        provider_health.reset()
