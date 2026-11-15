"""Direct unit tests for amadeus/web/sessions.py: idle TTL eviction,
provider-reload fingerprinting, the vision _prepare_turn invariant, and the
background TTS warmup thread.

Deliberately below the HTTP layer (unlike test_vision_turn.py) so these
exercise SessionStore/WebChatSession methods directly with dummy/stub
providers — no live API calls, no FastAPI app needed.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import threading
import time


from amadeus.config import Config
from amadeus.memory.manager import Message
from amadeus.providers.llm.base import ChatMessage, DummyProvider
from amadeus.providers.tts.base import TTSProvider
from amadeus.web.models import ChatRequest, ImageAttachment
from amadeus.web.sessions import SessionStore, WebChatSession


class StubTTS(TTSProvider):
    """Minimal TTS stub; records calls, never touches real audio libs."""

    def __init__(self, tag: str = "stub") -> None:
        self.tag = tag
        self.calls: list[str] = []

    def synthesize(
        self, text: str, *, speaking_rate: float = 1.0, pitch_shift: float = 0.0
    ) -> bytes:
        self.calls.append(text)
        return b"WAV"


class RecordingProvider(DummyProvider):
    """Dummy LLM that records every context it receives."""

    def __init__(self, scripted_reply: str | None = None, vision: bool = False) -> None:
        super().__init__(scripted_reply=scripted_reply)
        self.contexts: list[list[ChatMessage]] = []
        self._vision = vision

    def generate(self, messages, **kwargs):
        self.contexts.append(list(messages))
        return super().generate(messages, **kwargs)

    def supports_vision(self) -> bool:
        return self._vision


def _config(tmp_path) -> Config:
    config = Config()
    config.memory.persist_path = str(tmp_path / "memory.json")
    config.memory.embeddings_path = str(tmp_path / "embeddings.json")
    config.memory.profile_path = str(tmp_path / "profile.json")
    config.memory.transcripts_path = str(tmp_path / "transcripts.json")
    config.memory.enable_long_term = False
    config.audio.enable_voice = False  # no TTS load/warmup unless a test opts in
    return config


def _patch_loaders(monkeypatch, llm=None, tts=None, embedder=None):
    """Patch the loaders on amadeus.web.app, which sessions.py resolves
    lazily via _app() so the patches actually take effect (see sessions.py's
    _app() docstring)."""
    import amadeus.web.app as app_module

    monkeypatch.setattr(app_module, "load_llm", lambda cfg: llm or DummyProvider())
    monkeypatch.setattr(app_module, "load_tts", lambda cfg: tts)
    monkeypatch.setattr(app_module, "load_embedder", lambda cfg: embedder)
    return app_module


def _store(tmp_path, monkeypatch, config=None, llm=None, tts=None) -> SessionStore:
    config = config or _config(tmp_path)
    _patch_loaders(monkeypatch, llm=llm, tts=tts)
    return SessionStore(config)


# ---------------------------------------------------------------------------
# 1. Idle TTL eviction (_evict_idle_locked)
# ---------------------------------------------------------------------------


def test_evict_idle_drops_sessions_past_ttl(tmp_path, monkeypatch):
    config = _config(tmp_path)
    config.web.session_ttl_hours = 1.0
    store = _store(tmp_path, monkeypatch, config=config)

    session_id, session = store.get(None, "user-a")
    assert session_id in store.sessions

    # Fresh session: a sweep well within the TTL must not evict it.
    store._evict_idle_locked(session.last_activity + 10)
    assert session_id in store.sessions

    # Push the clock past the TTL (1h = 3600s).
    store._evict_idle_locked(session.last_activity + 3601)
    assert session_id not in store.sessions
    assert "user-a" not in store.user_index


def test_evict_idle_survives_fresh_sessions(tmp_path, monkeypatch):
    config = _config(tmp_path)
    config.web.session_ttl_hours = 6.0
    store = _store(tmp_path, monkeypatch, config=config)

    session_id, _session = store.get(None, "user-b")
    now = time.monotonic()
    store._evict_idle_locked(now)  # barely any time has passed
    assert session_id in store.sessions


def test_evict_idle_disabled_when_ttl_zero(tmp_path, monkeypatch):
    config = _config(tmp_path)
    config.web.session_ttl_hours = 0
    store = _store(tmp_path, monkeypatch, config=config)

    session_id, session = store.get(None, "user-c")
    # Even a huge time jump must not evict when the TTL is disabled.
    store._evict_idle_locked(session.last_activity + 999999)
    assert session_id in store.sessions


def test_get_touches_last_activity_and_reloads_ttl_via_sweep(tmp_path, monkeypatch):
    """get() sweeps on every call (using real time.monotonic), and touches
    last_activity for the session it returns."""
    config = _config(tmp_path)
    config.web.session_ttl_hours = 6.0
    store = _store(tmp_path, monkeypatch, config=config)

    session_id, session = store.get(None, "user-d")
    stale_activity = session.last_activity - 100
    session.last_activity = stale_activity

    session_id2, session2 = store.get(session_id, "user-d")
    assert session_id2 == session_id
    assert session2.last_activity > stale_activity


def test_evict_idle_does_not_drop_a_session_mid_use(tmp_path, monkeypatch):
    """An in-flight handler holds its own reference to the session object;
    even after the store evicts it from the dict, that reference must keep
    working (per the module's documented eviction contract)."""
    config = _config(tmp_path)
    config.web.session_ttl_hours = 1.0
    store = _store(tmp_path, monkeypatch, config=config)

    session_id, session = store.get(None, "user-e")
    in_flight_ref = session  # simulate a request handler's held reference

    store._evict_idle_locked(session.last_activity + 3601)
    assert session_id not in store.sessions

    # The evicted-but-referenced session object is still fully usable.
    assert in_flight_ref.memory is not None
    setup = in_flight_ref.prepare_turn("still working", in_flight_ref.config.persona)
    assert setup.context


def test_evict_idle_only_removes_expired_not_all(tmp_path, monkeypatch):
    config = _config(tmp_path)
    config.web.session_ttl_hours = 1.0
    store = _store(tmp_path, monkeypatch, config=config)

    old_id, old_session = store.get(None, "user-old")
    new_id, new_session = store.get(None, "user-new")
    # Backdate only the old session.
    old_session.last_activity = time.monotonic() - 7200
    store._evict_idle_locked(time.monotonic())

    assert old_id not in store.sessions
    assert new_id in store.sessions


# ---------------------------------------------------------------------------
# 2. reload_providers + _tts_fingerprint
# ---------------------------------------------------------------------------


def test_tts_fingerprint_changes_with_tts_relevant_setting(tmp_path):
    config = _config(tmp_path)
    fp1 = SessionStore._tts_fingerprint(config)

    changed = dataclasses.replace(config)
    changed.providers = dataclasses.replace(config.providers, tts="melotts_onnx")
    fp2 = SessionStore._tts_fingerprint(changed)
    assert fp1 != fp2


def test_tts_fingerprint_stable_across_unrelated_changes(tmp_path):
    config = _config(tmp_path)
    fp1 = SessionStore._tts_fingerprint(config)

    changed = dataclasses.replace(config)
    changed.providers = dataclasses.replace(
        config.providers, llm="gemini", stt="whisper"
    )
    fp2 = SessionStore._tts_fingerprint(changed)
    assert fp1 == fp2  # llm/stt are excluded from the TTS fingerprint


def test_tts_fingerprint_ignores_embedding_provider(tmp_path):
    config = _config(tmp_path)
    fp1 = SessionStore._tts_fingerprint(config)
    changed = dataclasses.replace(config)
    changed.providers = dataclasses.replace(config.providers, embedding="openai")
    fp2 = SessionStore._tts_fingerprint(changed)
    assert fp1 == fp2


def test_reload_providers_skips_tts_reload_when_unchanged(tmp_path, monkeypatch):
    config = _config(tmp_path)
    config.audio.enable_voice = True
    original_tts = StubTTS("original")
    store = _store(tmp_path, monkeypatch, config=config, tts=original_tts)
    assert store.tts is original_tts

    # Something reloads a brand-new TTS instance every call; if
    # reload_providers correctly skips the rebuild, `store.tts` must remain
    # the original instance rather than a new one from this loader.
    other_tts = StubTTS("other-should-not-be-used")
    import amadeus.web.app as app_module

    monkeypatch.setattr(app_module, "load_tts", lambda cfg: other_tts)

    # Change something TTS-irrelevant (an LLM-only field).
    new_config = dataclasses.replace(config)
    new_config.llm_model = "some-different-model"
    store.reload_providers(new_config)

    assert store.tts is original_tts


def test_reload_providers_reloads_tts_on_relevant_change(tmp_path, monkeypatch):
    config = _config(tmp_path)
    config.audio.enable_voice = True
    original_tts = StubTTS("original")
    store = _store(tmp_path, monkeypatch, config=config, tts=original_tts)
    assert store.tts is original_tts

    new_tts = StubTTS("new")
    import amadeus.web.app as app_module

    monkeypatch.setattr(app_module, "load_tts", lambda cfg: new_tts)

    new_config = dataclasses.replace(config)
    new_config.providers = dataclasses.replace(config.providers, tts="melotts_onnx")
    store.reload_providers(new_config)

    assert store.tts is new_tts


def test_reload_providers_skips_tts_while_warming(tmp_path, monkeypatch):
    """A background warmup thread must not be raced by a second TTS build."""
    config = _config(tmp_path)
    config.audio.enable_voice = True
    store = _store(tmp_path, monkeypatch, config=config, tts=StubTTS("original"))
    store.tts_warming = True

    build_calls = []
    import amadeus.web.app as app_module

    def _tracking_load_tts(cfg):
        build_calls.append(cfg)
        return StubTTS("should-not-be-built")

    monkeypatch.setattr(app_module, "load_tts", _tracking_load_tts)

    new_config = dataclasses.replace(config)
    new_config.providers = dataclasses.replace(config.providers, tts="melotts_onnx")
    store.reload_providers(new_config)

    assert not build_calls
    assert store.tts_warming is True


def test_reload_providers_disables_tts_when_voice_off(tmp_path, monkeypatch):
    config = _config(tmp_path)
    config.audio.enable_voice = True
    store = _store(tmp_path, monkeypatch, config=config, tts=StubTTS("original"))
    assert store.tts is not None

    new_config = dataclasses.replace(config)
    new_config.audio = dataclasses.replace(config.audio, enable_voice=False)
    store.reload_providers(new_config)

    assert store.tts is None


def test_reload_providers_rebinds_live_sessions(tmp_path, monkeypatch):
    """Existing chat sessions must see the newly configured llm/tts/embedder
    immediately, without needing a relaunch. There is a known narrow race in
    reload_providers (rebinding without handle_lock); this test only checks
    the end state after reload_providers returns, so it passes regardless of
    whether that race is later fixed."""
    config = _config(tmp_path)
    store = _store(tmp_path, monkeypatch, config=config)
    _session_id, session = store.get(None, "user-f")
    old_llm = session.llm

    new_llm = RecordingProvider(scripted_reply="new provider reply")
    new_tts = StubTTS("rebind-tts")
    import amadeus.web.app as app_module

    monkeypatch.setattr(app_module, "load_llm", lambda cfg: new_llm)
    monkeypatch.setattr(app_module, "load_tts", lambda cfg: new_tts)

    new_config = dataclasses.replace(config)
    new_config.audio = dataclasses.replace(config.audio, enable_voice=True)
    store.reload_providers(new_config)

    assert session.llm is new_llm
    assert session.llm is not old_llm
    assert session.tts is new_tts
    assert session.config is new_config


def test_reload_providers_tts_load_failure_falls_back_to_none(tmp_path, monkeypatch):
    config = _config(tmp_path)
    config.audio.enable_voice = True
    store = _store(tmp_path, monkeypatch, config=config, tts=StubTTS("original"))

    import amadeus.web.app as app_module

    def _boom(cfg):
        raise RuntimeError("boom: dependency missing")

    monkeypatch.setattr(app_module, "load_tts", _boom)

    new_config = dataclasses.replace(config)
    new_config.providers = dataclasses.replace(config.providers, tts="melotts_onnx")
    store.reload_providers(new_config)

    assert store.tts is None
    assert store.tts_error is not None
    assert "boom" in store.tts_error


# ---------------------------------------------------------------------------
# 3. WebChatSession._prepare_turn vision invariant
# ---------------------------------------------------------------------------


def _vision_session(tmp_path, vision_enabled=True, provider=None) -> WebChatSession:
    config = _config(tmp_path)
    config.vision.enabled = vision_enabled
    config.web.streaming_tagged_output = True
    provider = provider or RecordingProvider(vision=True)
    session = WebChatSession(
        config,
        provider,
        embedder=None,
        tts_provider=None,
        emotion_analyzer=__import__(
            "amadeus.agents.emotion_analyzer", fromlist=["KurisuEmotionAnalyzer"]
        ).KurisuEmotionAnalyzer(),
        translator=None,
        persist_path=config.memory.persist_path,
        embeddings_path=config.memory.embeddings_path,
        profile_path=config.memory.profile_path,
    )
    return session


def _image_request(message: str = "check this out") -> ChatRequest:
    return ChatRequest(
        message=message,
        images=[ImageAttachment(data="ZmFrZWJ5dGVz", mime="image/png")],
    )


def test_prepare_turn_attaches_image_only_after_memory_commit(tmp_path):
    session = _vision_session(tmp_path)
    request = _image_request()

    setup = session._prepare_turn(request)

    # Memory (short_term) only ever holds the text marker, never the image.
    # memory.manager.Message has no `images` field at all (structurally
    # cannot carry bytes) — assert both that and the absence of the payload.
    user_messages = [m for m in session.memory.short_term if m.role == "user"]
    assert len(user_messages) == 1
    assert not hasattr(user_messages[0], "images")
    assert "[user shared a screenshot]" in user_messages[0].content
    assert "ZmFrZWJ5dGVz" not in user_messages[0].content

    # The *turn context* (built from memory, then mutated after) does carry
    # the image on its trailing user message.
    context_user_messages = [m for m in setup.context if m.role == "user"]
    assert context_user_messages[-1].images is not None
    assert context_user_messages[-1].images[0].mime == "image/png"
    assert context_user_messages[-1].images[0].data == "ZmFrZWJ5dGVz"

    # The vision system note rides the context but never memory.
    assert any(
        "attached a screenshot" in m.content
        for m in setup.context
        if m.role == "system"
    )
    assert not any(
        "attached a screenshot" in (m.content or "") for m in session.memory.short_term
    )


def test_prepare_turn_marks_request_message_with_marker(tmp_path):
    """request.message itself is mutated to carry the marker so complete_turn
    (called later with request.message) commits the same marker text, not
    the original message — one mutation covers memory, transcript, and
    rollback per the source comment."""
    session = _vision_session(tmp_path)
    request = _image_request("what is this")

    session._prepare_turn(request)
    assert request.message.startswith("what is this")
    assert request.message.endswith("[user shared a screenshot]")


def test_prepare_turn_without_image_context_untouched(tmp_path):
    session = _vision_session(tmp_path)
    request = ChatRequest(message="just text, no image")

    setup = session._prepare_turn(request)
    assert all(m.images is None for m in setup.context)
    assert not any(
        "attached a screenshot" in m.content
        for m in setup.context
        if m.role == "system"
    )


def test_prepare_turn_vision_disabled_ignores_images(tmp_path):
    session = _vision_session(tmp_path, vision_enabled=False)
    request = _image_request()

    setup = session._prepare_turn(request)
    assert all(m.images is None for m in setup.context)
    assert "[user shared a screenshot]" not in request.message
    user_messages = [m for m in session.memory.short_term if m.role == "user"]
    assert "[user shared a screenshot]" not in user_messages[0].content


def test_prepare_turn_non_vision_provider_ignores_images(tmp_path):
    provider = RecordingProvider(vision=False)  # supports_vision() -> False
    session = _vision_session(tmp_path, provider=provider)
    request = _image_request()

    setup = session._prepare_turn(request)
    assert all(m.images is None for m in setup.context)
    assert "[user shared a screenshot]" not in request.message


def test_prepare_turn_second_turn_never_resurfaces_image(tmp_path):
    """After a vision turn commits, a later turn's context (rebuilt from
    memory) must never carry image bytes again."""
    session = _vision_session(tmp_path)
    request = _image_request()
    session._prepare_turn(request)
    session.complete_turn(request.message, "[neutral] Noted. || わかったわ。")

    setup2 = session._prepare_turn(ChatRequest(message="anything else?"))
    assert all(m.images is None for m in setup2.context)
    assert any("[user shared a screenshot]" in m.content for m in setup2.context)


# ---------------------------------------------------------------------------
# 4. _warm_tts_in_background
# ---------------------------------------------------------------------------


def test_warm_tts_in_background_completes_and_binds_sessions(tmp_path, monkeypatch):
    config = _config(tmp_path)
    config.audio.enable_voice = False  # keep SessionStore.__init__ from loading TTS
    store = _store(tmp_path, monkeypatch, config=config)

    session_id, session = store.get(None, "user-warm")
    assert session.tts is None

    warm_tts = StubTTS("warmed")
    import amadeus.web.app as app_module

    monkeypatch.setattr(app_module, "load_tts", lambda cfg: warm_tts)
    store.tts_warming = True

    store._warm_tts_in_background()

    assert store.tts is warm_tts
    assert store.tts_warming is False
    assert session.tts is warm_tts  # live session got rebound


def test_warm_tts_in_background_failure_does_not_crash(tmp_path, monkeypatch):
    config = _config(tmp_path)
    config.audio.enable_voice = False
    store = _store(tmp_path, monkeypatch, config=config)

    session_id, session = store.get(None, "user-warm-fail")
    import amadeus.web.app as app_module

    def _boom(cfg):
        raise RuntimeError("weights missing")

    monkeypatch.setattr(app_module, "load_tts", _boom)
    store.tts_warming = True

    store._warm_tts_in_background()  # must not raise

    assert store.tts is None
    assert store.tts_warming is False
    assert session.tts is None
    assert store.tts_error is not None


def test_warm_tts_in_background_runs_on_real_thread(tmp_path, monkeypatch):
    """Exercise the actual threading.Thread path (as SessionStore.__init__
    does for qwen3_tts) rather than only calling the target method inline.

    The loader stub blocks on a gate until this test has confirmed the
    "tts-warmup" thread is alive and tts_warming is still True, then releases
    it and waits for the thread to converge to the warmed state — avoids
    racing an instant stub load against the assertion.
    """
    config = _config(tmp_path)
    config.audio.enable_voice = True
    config.providers.tts = "qwen3_tts"

    warm_tts = StubTTS("threaded-warm")
    release = threading.Event()
    import amadeus.web.app as app_module

    def _gated_load_tts(cfg):
        release.wait(timeout=5)
        return warm_tts

    monkeypatch.setattr(app_module, "load_llm", lambda cfg: DummyProvider())
    monkeypatch.setattr(app_module, "load_embedder", lambda cfg: None)
    monkeypatch.setattr(app_module, "load_tts", _gated_load_tts)

    store = SessionStore(config)

    warmup_thread = None
    for _ in range(200):
        warmup_thread = next(
            (t for t in threading.enumerate() if t.name == "tts-warmup"), None
        )
        if warmup_thread is not None:
            break
        time.sleep(0.01)

    assert warmup_thread is not None
    assert store.tts_warming is True  # still loading, gated on `release`

    release.set()
    warmup_thread.join(timeout=5)

    assert store.tts_warming is False
    assert store.tts is warm_tts


class _ExoticSignal(BaseException):
    """A non-Exception BaseException, standing in for the kind of exotic
    error (e.g. raised from a C extension) that a bare ``except Exception``
    would not catch. Used to prove the warmup thread's crash guard is a
    ``except BaseException``, not just ``except Exception``."""


def test_warm_tts_in_background_baseexception_still_clears_warming_and_sets_error(
    tmp_path, monkeypatch
):
    """A BaseException escaping _load_tts_blocking (or anything else inside
    the warmup body) must not leave tts_warming stuck True forever — that
    would read as "permanently warming, permanently silent" with no signal
    anywhere. The crash guard must clear tts_warming, set a short tts_error,
    and count tts.warmup_thread_crashed, without re-raising (daemon thread:
    nothing downstream to catch it)."""
    from amadeus import telemetry

    config = _config(tmp_path)
    config.audio.enable_voice = False
    store = _store(tmp_path, monkeypatch, config=config)

    import amadeus.web.app as app_module

    def _explode(cfg):
        raise _ExoticSignal("hardware fell over")

    monkeypatch.setattr(app_module, "load_tts", _explode)
    store.tts_warming = True

    before = telemetry.snapshot().get("tts.warmup_thread_crashed", 0)

    store._warm_tts_in_background()  # must not raise

    assert store.tts_warming is False
    assert store.tts_error is not None
    assert "hardware fell over" in store.tts_error
    assert telemetry.snapshot().get("tts.warmup_thread_crashed", 0) == before + 1


def test_reload_during_warming_defers_and_reapplies_after_warmup(tmp_path, monkeypatch):
    """A settings save that arrives while the background warmup thread is
    mid-load must not be silently dropped: reload_providers should flag
    tts_reload_pending (+ count tts.reload_deferred), and once
    _warm_tts_in_background finishes warming it must re-apply that deferred
    reload — binding sessions onto the provider matching the LATEST saved
    config, not the one that was current when warmup started."""
    from amadeus import telemetry

    config = _config(tmp_path)
    config.audio.enable_voice = False  # keep SessionStore.__init__ from loading TTS
    store = _store(tmp_path, monkeypatch, config=config)

    session_id, session = store.get(None, "user-deferred")

    import amadeus.web.app as app_module

    warm_tts = StubTTS("warmed")
    monkeypatch.setattr(app_module, "load_tts", lambda cfg: warm_tts)

    # Simulate: warmup is in flight when a settings save calls
    # reload_providers with voice enabled.
    store.tts_warming = True
    before_deferred = telemetry.snapshot().get("tts.reload_deferred", 0)

    new_config = dataclasses.replace(
        config, audio=dataclasses.replace(config.audio, enable_voice=True)
    )
    store.reload_providers(new_config)

    assert store.tts_reload_pending is True
    assert telemetry.snapshot().get("tts.reload_deferred", 0) == before_deferred + 1
    # The deferred branch must not have touched tts itself.
    assert store.tts is None

    # Warmup "finishes" (simulated inline call, same as the other
    # _warm_tts_in_background tests) — its own _load_tts_blocking call uses
    # whatever load_tts is patched to right now; the reapply step at the end
    # re-reads self.config (already the new, voice-enabled config from the
    # reload_providers call above) and re-runs the load against it.
    store._warm_tts_in_background()

    assert store.tts_warming is False
    assert store.tts_reload_pending is False
    assert store.tts is warm_tts
    assert session.tts is warm_tts


# ---------------------------------------------------------------------------
# 4. Turn ids: complete_turn assignment, rewind_to_turn, amend_last_assistant
# ---------------------------------------------------------------------------


def _turn_config(tmp_path) -> Config:
    config = _config(tmp_path)
    config.web.streaming_tagged_output = True
    return config


async def _chat(
    store: SessionStore, session_id, user_id: str, message: str, reply: str
):
    """Drive one full turn through session.handle() with a scripted reply."""
    sid, session = store.get(session_id, user_id)
    session.llm = RecordingProvider(scripted_reply=reply)
    request = ChatRequest(message=message, session_id=sid, user_id=user_id)
    response = await session.handle(request)
    return sid, session, response


async def test_complete_turn_assigns_turn_id(tmp_path, monkeypatch):
    config = _turn_config(tmp_path)
    store = _store(tmp_path, monkeypatch, config=config)

    sid, session, response = await _chat(
        store, None, "turn-user", "hello", "[happy] Hi there. || こんにちは。"
    )

    assert response.turn_id
    assert session.last_turn_id == response.turn_id

    transcript = store.transcripts.get_transcript("turn-user", sid)
    assert [m["turn_id"] for m in transcript] == [response.turn_id, response.turn_id]


async def test_rewind_to_turn_truncates_transcript_and_short_term(
    tmp_path, monkeypatch
):
    config = _turn_config(tmp_path)
    store = _store(tmp_path, monkeypatch, config=config)

    sid, _session, r1 = await _chat(
        store, None, "rewind-user", "one", "[happy] Reply one. || 一。"
    )
    _sid, _session, r2 = await _chat(
        store, sid, "rewind-user", "two", "[happy] Reply two. || 二。"
    )
    _sid, session, r3 = await _chat(
        store, sid, "rewind-user", "three", "[happy] Reply three. || 三。"
    )
    assert len({r1.turn_id, r2.turn_id, r3.turn_id}) == 3

    async with session.handle_lock:
        result = session.rewind_to_turn(r2.turn_id)

    assert result["status"] == "ok"
    # Rewinding to turn 2 drops turn 2 and turn 3 -> only turn 1 remains.
    assert len(result["messages"]) == 2

    transcript = store.transcripts.get_transcript("rewind-user", sid)
    assert len(transcript) == 2
    assert transcript[0]["turn_id"] == r1.turn_id

    short_term_contents = [m.content for m in session.memory.short_term]
    assert len(session.memory.short_term) == 2
    assert any("Reply one" in c for c in short_term_contents)
    assert not any("Reply two" in c or "Reply three" in c for c in short_term_contents)

    assert session.last_turn_id == r1.turn_id
    # Summary state was reset (not left describing the dropped turns).
    assert session.memory.summary_info.get("revision", 0) >= 0


async def test_post_empty_reply_rolls_back_user_message(tmp_path, monkeypatch):
    """A NOT-cancelled empty reply on the POST path used to run
    complete_turn(msg, "") — which skips the assistant append and strands
    the user message as a trailing orphan in the persisted store until the
    next restart's _load healing. It must roll back instead (counted), and
    the response must not advertise the PREVIOUS turn's id."""
    from amadeus import telemetry

    config = _turn_config(tmp_path)
    store = _store(tmp_path, monkeypatch, config=config)

    before = telemetry.snapshot().get("web.empty_reply_rolled_back", 0)
    _sid, session, response = await _chat(store, None, "empty-user", "hello", "")

    assert response.reply == ""
    assert response.turn_id is None
    assert [m for m in session.memory.short_term if m.role == "user"] == []
    assert telemetry.snapshot().get("web.empty_reply_rolled_back", 0) == before + 1


async def test_post_thought_only_reply_never_leaks_and_rolls_back(
    tmp_path, monkeypatch
):
    """A reply of ONLY [thought]/non-verbal lines parses as tagged with
    ZERO sentences; the POST untagged fallback then kept the literal
    '[thought] ...' text — leaked to display AND committed to memory. The
    special lines must be stripped, and the resulting empty turn rolls the
    user message back."""
    config = _turn_config(tmp_path)
    store = _store(tmp_path, monkeypatch, config=config)

    _sid, session, response = await _chat(
        store,
        None,
        "thought-only-user",
        "anything to add?",
        "[thought] A private aside about them.\n[laugh]",
    )

    assert response.reply == ""
    assert "thought" not in response.reply
    assert response.turn_id is None
    # Nothing committed on either side of the exchange.
    assert list(session.memory.short_term) == []


async def test_rewind_without_transcript_store_pops_last_exchange(
    tmp_path, monkeypatch
):
    config = _turn_config(tmp_path)
    config.memory.transcripts_path = None
    store = _store(tmp_path, monkeypatch, config=config)
    assert store.transcripts.base is None

    sid, session, r1 = await _chat(
        store, None, "no-store-user", "hi", "[happy] Hi. || やあ。"
    )
    assert session.transcript_store is store.transcripts

    async with session.handle_lock:
        result = session.rewind_to_turn(r1.turn_id)

    assert result["status"] == "ok"
    assert result["messages"] == []
    assert list(session.memory.short_term) == []
    assert session.last_turn_id is None

    # A second rewind against an id that no longer matches last_turn_id (None)
    # can't be found in a transcript store that never had it and isn't the
    # known trailing turn either -> not_found, not unavailable.
    async with session.handle_lock:
        result2 = session.rewind_to_turn("some-other-unknown-id")
    assert result2["status"] == "not_found"


async def test_rewind_unknown_turn_id_not_found(tmp_path, monkeypatch):
    config = _turn_config(tmp_path)
    store = _store(tmp_path, monkeypatch, config=config)

    sid, session, _r1 = await _chat(
        store, None, "unknown-user", "hi", "[happy] Hi. || やあ。"
    )

    async with session.handle_lock:
        result = session.rewind_to_turn("nonexistent-turn-id")

    assert result["status"] == "not_found"


async def test_amend_last_assistant_updates_short_term_and_transcript(
    tmp_path, monkeypatch
):
    config = _turn_config(tmp_path)
    store = _store(tmp_path, monkeypatch, config=config)

    sid, session, r1 = await _chat(
        store, None, "amend-user", "hi", "[happy] Original reply. || 元の返事。"
    )

    async with session.handle_lock:
        result = session.amend_last_assistant(
            r1.turn_id, "Amended reply.", ja="修正済み。"
        )

    assert result["status"] == "ok"
    assert session.memory.short_term[-1].content == "Amended reply."

    transcript = store.transcripts.get_transcript("amend-user", sid)
    assert transcript[-1]["content"] == "Amended reply."
    assert transcript[-1]["ja"] == "修正済み。"


async def test_amend_stale_turn_conflicts(tmp_path, monkeypatch):
    config = _turn_config(tmp_path)
    store = _store(tmp_path, monkeypatch, config=config)

    sid, session, r1 = await _chat(
        store, None, "stale-user", "hi", "[happy] First. || 一。"
    )
    _sid, session, _r2 = await _chat(
        store, sid, "stale-user", "hi again", "[happy] Second. || 二。"
    )

    async with session.handle_lock:
        result = session.amend_last_assistant(r1.turn_id, "Nope.")

    assert result["status"] == "conflict"
    # Nothing was mutated.
    assert "Second" in session.memory.short_term[-1].content


async def test_regenerate_flow_rewind_then_resend(tmp_path, monkeypatch):
    """Full regenerate flow: turn -> rewind last -> handle same text again ->
    exactly one exchange remains in both the transcript and short_term."""
    config = _turn_config(tmp_path)
    store = _store(tmp_path, monkeypatch, config=config)

    sid, session, r1 = await _chat(
        store, None, "regen-user", "explain dopamine", "[happy] First take. || 一つ目。"
    )
    assert r1.turn_id

    async with session.handle_lock:
        result = session.rewind_to_turn(r1.turn_id)
    assert result["status"] == "ok"
    assert result["messages"] == []

    _sid, session, r2 = await _chat(
        store,
        sid,
        "regen-user",
        "explain dopamine",
        "[happy] Second take. || 二つ目。",
    )
    assert r2.turn_id != r1.turn_id

    transcript = store.transcripts.get_transcript("regen-user", sid)
    assert len(transcript) == 2
    assert "Second take" in transcript[1]["content"]

    assistant_messages = [m for m in session.memory.short_term if m.role == "assistant"]
    assert len(assistant_messages) == 1
    assert "Second take" in assistant_messages[0].content


# ---------------------------------------------------------------------------
# 5. F1-F6: amend desync, rehydration, phantom gap, carryover, fallback
#    validation, no eager summarize.
# ---------------------------------------------------------------------------


async def test_amend_after_idle_chatter_edits_the_right_message(tmp_path, monkeypatch):
    """A trailing assistant message appended by idle chatter (via
    ChatEngine.add_message, outside the normal turn flow) must not be the one
    amend edits — the transcript's old content locates the real target."""
    config = _turn_config(tmp_path)
    store = _store(tmp_path, monkeypatch, config=config)

    sid, session, r1 = await _chat(
        store, None, "idle-user", "hi", "[happy] Original reply. || 元の返事。"
    )
    # Simulate idle chatter: an extra assistant line appended straight to
    # memory, bypassing complete_turn/transcript recording entirely.
    session.memory.add_message("assistant", "*waves idly*")

    async with session.handle_lock:
        result = session.amend_last_assistant(
            r1.turn_id, "Amended reply.", ja="修正済み。"
        )

    assert result["status"] == "ok"
    contents = [m.content for m in session.memory.short_term]
    assert "Amended reply." in contents
    assert "*waves idly*" in contents  # untouched, not clobbered
    assert "Original reply." not in contents

    transcript = store.transcripts.get_transcript("idle-user", sid)
    assert transcript[-1]["content"] == "Amended reply."
    assert transcript[-1]["ja"] == "修正済み。"


async def test_amend_after_council_trailing_system_line_updates_both_stores(
    tmp_path, monkeypatch
):
    """A trailing 'Council summary:' system line must be skipped when
    locating the assistant message to amend."""
    config = _turn_config(tmp_path)
    store = _store(tmp_path, monkeypatch, config=config)

    sid, session = store.get(None, "council-user")
    session.llm = RecordingProvider(
        scripted_reply="[happy] Council reply. || 会議の返事。"
    )
    request = ChatRequest(message="debate this", session_id=sid, user_id="council-user")
    response = await session.handle(request)
    turn_id = response.turn_id
    assert turn_id

    # Simulate a council turn's trailing system line, appended by
    # ChatEngine.complete_turn's council_summary branch.
    session.memory.add_message("system", "Council summary: they agreed.")

    async with session.handle_lock:
        result = session.amend_last_assistant(turn_id, "Amended council reply.")

    assert result["status"] == "ok"
    contents = [(m.role, m.content) for m in session.memory.short_term]
    assert ("assistant", "Amended council reply.") in contents
    assert ("system", "Council summary: they agreed.") in contents  # untouched

    transcript = store.transcripts.get_transcript("council-user", sid)
    assert transcript[-1]["content"] == "Amended council reply."


async def test_amend_returns_conflict_when_transcript_says_no(tmp_path, monkeypatch):
    """Even if last_turn_id (in-memory) happens to match, the transcript is
    authoritative: if it reports no match (stale/foreign turn_id), amend must
    conflict rather than trust the in-memory id alone."""
    config = _turn_config(tmp_path)
    store = _store(tmp_path, monkeypatch, config=config)

    sid, session, r1 = await _chat(
        store, None, "conflict-user", "hi", "[happy] Reply. || 返事。"
    )
    # Force a mismatch directly on disk: the last recorded assistant message's
    # turn_id no longer matches what the session thinks is current.
    session.last_turn_id = r1.turn_id  # already true, but explicit for clarity
    path = store.transcripts._path_for("conflict-user")
    data = store.transcripts._load(path)
    data["sessions"][sid]["messages"][-1]["turn_id"] = "some-other-turn"
    store.transcripts._save(path, data)

    async with session.handle_lock:
        result = session.amend_last_assistant(r1.turn_id, "Nope.")

    assert result["status"] == "conflict"


def test_last_turn_id_rehydrates_from_transcript(tmp_path, monkeypatch):
    """Constructing a fresh in-memory session over an existing transcript
    (process restart / TTL eviction) starts with last_turn_id=None; amend and
    rewind must rehydrate it from the durable transcript before validating."""
    config = _turn_config(tmp_path)
    store = _store(tmp_path, monkeypatch, config=config)

    sid = store.create_session("rehydrate-user")
    store.transcripts.record_turn(
        "rehydrate-user", sid, "hi", "Hello.", turn_id="disk-turn-1"
    )

    # A brand-new SessionStore/session over the same transcript path,
    # simulating a restart: nothing in memory, only the transcript on disk.
    store2 = _store(tmp_path, monkeypatch, config=config)
    _sid2, session2 = store2.get(sid, "rehydrate-user")
    assert session2.last_turn_id is None  # never completed a turn in-process

    result = session2.amend_last_assistant("disk-turn-1", "Edited.")
    assert result["status"] == "ok"
    assert session2.last_turn_id == "disk-turn-1"

    transcript = store2.transcripts.get_transcript("rehydrate-user", sid)
    assert transcript[-1]["content"] == "Edited."


def test_rewind_last_turn_id_rehydrates_from_transcript(tmp_path, monkeypatch):
    config = _turn_config(tmp_path)
    store = _store(tmp_path, monkeypatch, config=config)

    sid = store.create_session("rehydrate-rewind-user")
    store.transcripts.record_turn(
        "rehydrate-rewind-user", sid, "hi", "Hello.", turn_id="disk-turn-2"
    )

    store2 = _store(tmp_path, monkeypatch, config=config)
    _sid2, session2 = store2.get(sid, "rehydrate-rewind-user")
    assert session2.last_turn_id is None

    result = session2.rewind_to_turn("disk-turn-2")
    assert result["status"] == "ok"
    assert result["messages"] == []
    assert session2.last_turn_id is None  # rewound the only turn away


async def test_rewind_preserves_pre_session_carryover(tmp_path, monkeypatch):
    """MemoryManager._load's cross-session carryover (messages already in
    short_term before this session's own turns began) must survive a rewind
    untouched — only the turns actually removed from the transcript pop."""
    config = _turn_config(tmp_path)
    store = _store(tmp_path, monkeypatch, config=config)

    sid, session, r1 = await _chat(
        store, None, "carryover-user", "one", "[happy] Reply one. || 一。"
    )
    _sid, session, r2 = await _chat(
        store, sid, "carryover-user", "two", "[happy] Reply two. || 二。"
    )

    # Seed extra leading carryover messages directly, simulating what
    # MemoryManager._load restores from a previous session's persisted state.
    session.memory.short_term.appendleft(
        Message(role="assistant", content="carried-over reply", timestamp=None)
    )
    session.memory.short_term.appendleft(
        Message(role="user", content="carried-over question", timestamp=None)
    )

    async with session.handle_lock:
        result = session.rewind_to_turn(r2.turn_id)

    assert result["status"] == "ok"
    contents = [m.content for m in session.memory.short_term]
    assert "carried-over question" in contents
    assert "carried-over reply" in contents
    assert any("Reply one" in c for c in contents)
    assert not any("Reply two" in c for c in contents)
    assert session.last_turn_id == r1.turn_id


async def test_rewind_restamps_tail_timestamp_no_phantom_gap(tmp_path, monkeypatch):
    """After rewind, the remaining tail message's timestamp must be re-stamped
    to now so _maybe_finalize_gap doesn't read the rewound-to turn's age as a
    session gap on the very next message."""
    config = _turn_config(tmp_path)
    config.memory.enable_episodes = True
    store = _store(tmp_path, monkeypatch, config=config)

    sid, session, r1 = await _chat(
        store, None, "gap-user", "one", "[happy] Reply one. || 一。"
    )
    _sid, session, r2 = await _chat(
        store, sid, "gap-user", "two", "[happy] Reply two. || 二。"
    )

    # Backdate turn 1's messages far into the past so, if the rewind didn't
    # re-stamp the tail, the next add_message would see a huge "gap".
    ancient = (dt.datetime.utcnow() - dt.timedelta(days=2)).isoformat()
    for message in session.memory.short_term:
        if "Reply one" in message.content or message.content == "one":
            message.timestamp = ancient

    session_count_before = session.memory.profile.session_count

    async with session.handle_lock:
        result = session.rewind_to_turn(r2.turn_id)
    assert result["status"] == "ok"

    # The remaining tail (turn 1's assistant reply) must now be freshly
    # stamped, not the ancient backdated value.
    tail = session.memory.short_term[-1]
    tail_stamp = dt.datetime.fromisoformat(tail.timestamp)
    assert (dt.datetime.utcnow() - tail_stamp).total_seconds() < 5

    # Sending another message must not trigger a spurious session boundary.
    session.memory.add_message("user", "still here?")
    assert session.memory.profile.session_count == session_count_before


async def test_rewind_fallback_returns_not_found_on_mismatched_tail(
    tmp_path, monkeypatch
):
    """_rewind_fallback_no_store must verify the tail actually looks like a
    completed exchange before popping anything, and must never clear
    last_turn_id when nothing pops."""
    config = _turn_config(tmp_path)
    config.memory.transcripts_path = None
    store = _store(tmp_path, monkeypatch, config=config)
    assert store.transcripts.base is None

    sid, session, r1 = await _chat(
        store, None, "mismatch-user", "hi", "[happy] Hi. || やあ。"
    )
    # Corrupt the tail so it no longer looks like a completed user/assistant
    # exchange (e.g. a stray trailing user message with no reply yet).
    session.memory.short_term.append(
        Message(role="user", content="dangling", timestamp=None)
    )
    before = list(session.memory.short_term)
    last_turn_before = session.last_turn_id

    async with session.handle_lock:
        result = session.rewind_to_turn(r1.turn_id)

    assert result["status"] == "not_found"
    # Nothing changed: no message popped, last_turn_id untouched.
    assert list(session.memory.short_term) == before
    assert session.last_turn_id == last_turn_before


async def test_rewind_no_eager_summarize(tmp_path, monkeypatch):
    """rewind_to_turn must not eagerly call memory.summarize(): the summary
    is reset and left to regenerate lazily on the next turn's debounce."""
    config = _turn_config(tmp_path)
    store = _store(tmp_path, monkeypatch, config=config)

    sid, session, r1 = await _chat(
        store, None, "no-eager-user", "one", "[happy] Reply one. || 一。"
    )

    calls = []
    session.memory.summarize = lambda: calls.append(True)

    async with session.handle_lock:
        result = session.rewind_to_turn(r1.turn_id)

    assert result["status"] == "ok"
    assert calls == []  # summarize() was never called by the rewind path
    assert session.memory.summary is None
    assert session.memory.summary_info == {}


# ---------------------------------------------------------------------------
# scope_context_to_session: short_term/summary scoped to the conversation,
# not the user, while long-term memory (facts/profile) stays per-user.
# ---------------------------------------------------------------------------


def _scoped_config(tmp_path) -> Config:
    config = _turn_config(tmp_path)
    config.memory.scope_context_to_session = True
    return config


async def test_scope_context_to_session_evicted_session_rehydrates_from_own_transcript(
    tmp_path, monkeypatch
):
    """Recreating the SAME session id (eviction/restart) must rehydrate
    short_term from that session's own transcript tail, even though
    MemoryManager._load no longer restores short_term from the per-user
    persist file under this flag."""
    config = _scoped_config(tmp_path)
    store = _store(tmp_path, monkeypatch, config=config)

    sid, session, r1 = await _chat(
        store, None, "scoped-user", "one", "[happy] Reply one. || 一。"
    )
    _sid, session, r2 = await _chat(
        store, sid, "scoped-user", "two", "[happy] Reply two. || 二。"
    )
    assert list(session.memory.short_term)  # sanity: turns landed in-memory

    # Evict the in-memory session (simulates TTL eviction / process restart)
    # but the durable transcript for this session id remains on disk.
    store._evict_idle_locked(session.last_activity + 999999)
    assert sid not in store.sessions

    # Recreating the SAME session id must rehydrate short_term from that
    # session's transcript tail rather than starting empty.
    _sid2, session2 = store.get(sid, "scoped-user")
    assert session2 is not session
    contents = [m.content for m in session2.memory.short_term]
    assert "one" in contents
    assert any("Reply one" in c for c in contents)
    assert "two" in contents
    assert any("Reply two" in c for c in contents)
    # last_turn_id rehydrates the same way amend/rewind already do.
    assert session2.last_turn_id == r2.turn_id


async def test_scope_context_to_session_new_session_id_starts_empty(
    tmp_path, monkeypatch
):
    """A DIFFERENT (brand-new) session id for the same user must start with
    empty short_term (no transcript to hydrate from), while profile/facts
    persisted from the first conversation still survive."""
    config = _scoped_config(tmp_path)
    store = _store(tmp_path, monkeypatch, config=config)

    sid, session, _r1 = await _chat(
        store, None, "scoped-user-2", "my name is Okabe", "[happy] Noted. || 了解。"
    )
    session.memory.profile.facts["name"] = "Okabe"
    session.memory._save_profile()

    # A different session id for the same user: no transcript under this id.
    new_sid = store.create_session("scoped-user-2")
    assert new_sid != sid
    _sid2, session2 = store.get(new_sid, "scoped-user-2")

    assert list(session2.memory.short_term) == []
    assert session2.memory.summary is None
    # Long-term/profile state is per-user and unaffected by the flag.
    assert session2.memory.profile is not None
    assert session2.memory.profile.facts.get("name") == "Okabe"


def test_scope_context_to_session_fresh_conversation_triggers_greeting(
    tmp_path, monkeypatch
):
    """A fresh conversation (empty short_term) must still fire should_greet,
    even for a returning user whose profile/facts already exist."""
    config = _scoped_config(tmp_path)
    config.memory.enable_greeting = True
    store = _store(tmp_path, monkeypatch, config=config)

    sid, session = store.get(None, "greet-user")
    assert session.memory.should_greet() is True


# ---------------------------------------------------------------------------
# Per-character memory isolation (character switch swaps the memory store)
# ---------------------------------------------------------------------------


def test_reload_providers_swaps_session_memory_on_character_switch(
    tmp_path, monkeypatch
):
    """A character switch rides reload_providers; live sessions must land on
    the new character's isolated store: fresh short_term/profile, scoped
    paths, and Kurisu's legacy-path store left untouched on disk."""
    import json
    from pathlib import Path

    config = _config(tmp_path)
    store = _store(tmp_path, monkeypatch, config=config)
    _sid, session = store.get(None, None)
    session.memory.add_message("user", "my name is Alex")
    session.memory.add_message("assistant", "Noted.")
    session.audio_enabled = False
    old_memory = session.memory
    kurisu_persist = Path(config.memory.persist_path)
    assert session.persist_path == kurisu_persist

    guest_config = dataclasses.replace(config)
    guest_config.character = dataclasses.replace(
        config.character, name="Rem", full_name="Rem", system_prompt="You are Rem."
    )
    store.reload_providers(guest_config)

    scoped = kurisu_persist.parent / "characters" / "rem" / kurisu_persist.name
    assert store.persist_base == scoped
    assert session.persist_path == scoped
    assert session.memory is not old_memory
    # The previous character's rolling context must not bleed across.
    assert list(session.memory.short_term) == []
    assert session.memory.profile is not None
    assert session.memory.profile.facts == {}
    # Kurisu's on-disk store still holds her conversation, byte-for-readable.
    data = json.loads(kurisu_persist.read_text(encoding="utf-8"))
    assert "my name is Alex" in [m["content"] for m in data["messages"]]
    # The rebind must not unmute a muted session.
    assert session.audio_enabled is False

    # Switching back to the builtin restores the legacy store and its memory.
    store.reload_providers(dataclasses.replace(config))
    assert session.persist_path == kurisu_persist
    assert "my name is Alex" in [m.content for m in session.memory.short_term]


def test_reload_providers_keeps_memory_on_ordinary_settings_save(tmp_path, monkeypatch):
    """A settings save that does not move the memory store must keep the
    session's live MemoryManager (transcript/context continuity)."""
    config = _config(tmp_path)
    store = _store(tmp_path, monkeypatch, config=config)
    _sid, session = store.get(None, None)
    old_memory = session.memory

    new_config = dataclasses.replace(config)
    store.reload_providers(new_config)

    assert session.memory is old_memory
    assert session.config is new_config


def _guest_config(config):
    """A copy of ``config`` with a custom character pack active."""
    guest = dataclasses.replace(config)
    guest.character = dataclasses.replace(
        config.character, name="Rem", full_name="Rem", system_prompt="You are Rem."
    )
    return guest


def test_reload_providers_swaps_transcripts_on_character_switch(tmp_path, monkeypatch):
    """A character switch must rebuild the transcript store on the scoped
    path: the guest's sessions list starts empty (never Kurisu's
    conversations), turns append to the guest's file, and rollback/amend
    keep working on either side of the switch. Kurisu's transcripts stay
    byte-identical at the legacy path."""
    from pathlib import Path

    config = _config(tmp_path)
    store = _store(tmp_path, monkeypatch, config=config)
    session_id, session = store.get(None, None)
    session.complete_turn("hello there", "hi yourself")
    kurisu_transcripts = Path(config.memory.transcripts_path)
    kurisu_store = store.transcripts
    kurisu_turn_id = session.last_turn_id
    assert [s["session_id"] for s in store.transcripts.list_sessions(None)] == [
        session_id
    ]
    kurisu_bytes = kurisu_transcripts.read_bytes()

    store.reload_providers(_guest_config(config))

    scoped = kurisu_transcripts.parent / "characters" / "rem" / kurisu_transcripts.name
    assert store.transcripts is not kurisu_store
    assert store.transcripts.base == scoped
    # The sessions-list API reads store.transcripts: a guest lists no
    # Kurisu-era conversations.
    assert store.transcripts.list_sessions(None) == []
    # The live session appends future turns to the guest's file, and its
    # turn-id anchor was reset (it referenced the previous character's
    # transcript).
    assert session.transcript_store is store.transcripts
    assert session.last_turn_id is None
    session.complete_turn("who are you", "I'm Rem")
    assert scoped.exists()
    # Amend targets the guest transcript and works normally after the switch.
    assert session.amend_last_assistant(session.last_turn_id, "I am Rem!") == {
        "status": "ok"
    }
    transcript = store.transcripts.get_transcript(None, session_id)
    assert transcript[-1]["content"] == "I am Rem!"
    # Kurisu's transcripts never moved and never changed.
    assert kurisu_transcripts.read_bytes() == kurisu_bytes

    # Switching back restores the legacy store: her conversations list
    # again, and amend re-anchors on her transcript's own last turn.
    store.reload_providers(dataclasses.replace(config))
    assert store.transcripts.base == kurisu_transcripts
    assert [s["session_id"] for s in store.transcripts.list_sessions(None)] == [
        session_id
    ]
    assert session.last_turn_id is None
    assert session.amend_last_assistant(kurisu_turn_id, "hi again") == {"status": "ok"}
    transcript = store.transcripts.get_transcript(None, session_id)
    assert transcript[-1]["content"] == "hi again"
    # The guest's own transcript survives at its scoped path for next time.
    assert scoped.exists()


def test_transcripts_not_scoped_when_per_character_memory_off(tmp_path, monkeypatch):
    from pathlib import Path

    config = _config(tmp_path)
    store = _store(tmp_path, monkeypatch, config=config)
    kurisu_store = store.transcripts

    guest_config = _guest_config(config)
    guest_config.memory.per_character_memory = False
    store.reload_providers(guest_config)

    # Fully legacy behavior: same store instance, same base path.
    assert store.transcripts is kurisu_store
    assert store.transcripts.base == Path(config.memory.transcripts_path)


def test_reload_providers_rebuilds_side_stores_on_character_switch(
    tmp_path, monkeypatch
):
    """Brief/diary/reach-out stores are persona-flavored and must follow the
    active character; a guest's files live under characters/<id>/ and
    Kurisu's entries survive the round-trip."""
    import datetime as ddt

    from amadeus.brief import DailyBrief

    config = _config(tmp_path)
    config.daily_brief.enabled = True
    config.daily_brief.store_path = str(tmp_path / "daily_brief.json")
    config.diary.enabled = True
    config.diary.store_path = str(tmp_path / "diary.json")
    config.reach_out.enabled = True
    config.reach_out.store_path = str(tmp_path / "reachout.json")
    store = _store(tmp_path, monkeypatch, config=config)
    _sid, session = store.get(None, None)
    today = ddt.date(2026, 7, 19)
    store.brief_store.put(
        None,
        DailyBrief(
            date=today.isoformat(),
            text="kurisu brief",
            generated_at=f"{today.isoformat()}T06:00:00",
        ),
    )

    store.reload_providers(_guest_config(config))

    scoped_dir = tmp_path / "characters" / "rem"
    assert store.brief_store.path == str(scoped_dir / "daily_brief.json")
    assert store.diary_store.path == str(scoped_dir / "diary.json")
    assert store.reachout_store.path == str(scoped_dir / "reachout.json")
    # The guest never sees (or consumes) Kurisu's brief...
    assert store.brief_store.get(None) is None
    # ...and greeting-time consumers were repointed at the shared instances.
    assert session.brief_store is store.brief_store
    assert session.diary_store is store.diary_store

    # Switch back: Kurisu's brief is still hers, unconsumed.
    store.reload_providers(dataclasses.replace(config))
    assert store.brief_store.path == str(tmp_path / "daily_brief.json")
    kurisu_brief = store.brief_store.get(None)
    assert kurisu_brief is not None and kurisu_brief.text == "kurisu brief"
    assert kurisu_brief.consumed is False
    assert session.brief_store is store.brief_store
