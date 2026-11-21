"""Meta/status/presence routes: healthz, status, settings,
speech-to-text, greeting/poke/idle/reminders, and memory controls."""

from __future__ import annotations

import asyncio
import base64
import datetime as dt
import importlib.util
import io
import re
import time
import wave
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException, UploadFile

from ..config import Config
from ..logging_utils import get_logger
from ..providers.llm.base import DummyProvider
from . import self_test, settings_api
from .. import gpu_runtime, persona_versions, provider_health, self_eval, telemetry
from ..async_notify import record_client_poll
from ..sleep_schedule import is_sleep_hours, sleep_status
from ..focus_mode import focus_config_status, is_focus_active
from ..farewell import is_goodbye
from ..memory.manager import PROTECTED
from .models import (
    AmbientRequest,
    FarewellRequest,
    FocusRequest,
    GreetingRequest,
    InsightDismissRequest,
    MemoryRequest,
    PresenceRequest,
    ReadAlongRequest,
    RhythmRequest,
    ScreenPeekRequest,
    WatchTogetherRequest,
)
from .sessions import SessionStore, WebChatSession, _run_blocking, _synthesize_tts

LOGGER = get_logger(__name__)


def _app_version() -> str:
    """Version of the code actually running.

    Reads the package's own ``__version__`` constant rather than installed
    dist metadata (importlib.metadata): an editable install's metadata
    reflects whatever pyproject said at ``pip install -e`` time, so a conda
    env that pulled newer source kept reporting the stale old version —
    which broke anything comparing it against releases (the frontend's
    update nudge)."""
    try:
        from .. import __version__

        return __version__
    except Exception:  # pragma: no cover - defensive; constant always exists
        return "dev"


def _tools_status(config: Config) -> Dict[str, Any]:
    """Tool availability for /api/status.

    Builds a registry from the live config so the answer reflects runtime
    truth (missing packages, missing API keys), not just config flags. Shape
    (consumed by the frontend status panel — keep stable):
    {"enabled": bool, "registered": [name, ...],
     "unavailable": [{"name": ..., "reason": ...}, ...]}
    """
    from ..agents.factory import create_tool_registry

    try:
        # Probe registry: never executed, only listed. The memory tool needs a
        # live MemoryManager the engine wires at runtime; passing a sentinel
        # here makes the status answer reflect config-level availability
        # instead of always reporting "memory" unavailable.
        registry = create_tool_registry(config, memory=object())
        registered = sorted(registry.tools.keys())
        unavailable = list(registry.unavailable)
    except Exception as exc:  # pragma: no cover - defensive
        registered = []
        unavailable = [{"name": "registry", "reason": str(exc)}]
    return {
        "enabled": bool(getattr(getattr(config, "tools", None), "enabled", False)),
        "registered": registered,
        "unavailable": unavailable,
    }


# Kept as an alias — the implementation moved to settings_api so the
# persona-version rollback route can share the same persist tail.
_deep_merge_config = settings_api.deep_merge_config


def _persona_version_store(
    config: Config,
) -> Optional[persona_versions.PersonaVersionStore]:
    """Version store for the current config, or None when disabled/broken."""
    path = getattr(config.memory, "persona_versions_path", None)
    if not path:
        return None
    try:
        from pathlib import Path

        return persona_versions.PersonaVersionStore(Path(path))
    except Exception as exc:  # pragma: no cover - store init failure
        LOGGER.warning("Persona version store unavailable: %s", exc)
        return None


def build_router(store: SessionStore, config: Config) -> APIRouter:
    router = APIRouter()

    @router.get("/healthz")
    async def healthz() -> Dict[str, object]:
        if store.tts is not None:
            voice = "ready"
        elif store.tts_warming:
            voice = "warming"
        else:
            voice = "unavailable"
        return {
            "status": "ok",
            "streaming": bool(config.web.streaming_enabled),
            "voice": voice,
            "version": _app_version(),
        }

    @router.get("/api/status")
    async def api_status() -> Dict[str, object]:
        """Runtime provider truth, as opposed to config-shape guesses.

        Reports the provider instances actually serving requests — including
        silent degradations (dummy LLM after a construction failure, the
        qwen3→ONNX voice fallback, a missing embedder) that the settings API's
        config-derived view cannot see. The UI status chip reads this.
        """
        with store.lock:
            live_config = store.config
            llm = store.llm
            tts_wrapper = store.tts
            tts_warming = store.tts_warming
            tts_fallback_from = store.tts_fallback_from
            tts_error = store.tts_error
            stt_loaded = store.stt is not None
            stt_reason = store.stt_unavailable_reason
            embedder = store.embedder

        tts_inner = getattr(tts_wrapper, "provider", tts_wrapper)
        # The qwen3 provider records which synthesis engine it loaded:
        # "faster" (CUDA-graph, RTF < 1) or "upstream" (the silent ~10x
        # slowdown when faster-qwen3-tts isn't installed). Surface the slow
        # case loudly; additive field, always present, None when healthy.
        tts_engine = getattr(tts_inner, "_engine", None)
        engine_warning = (
            "slow fallback — faster-qwen3-tts not installed (~10x slower "
            "synthesis); pip install faster-qwen3-tts"
            if tts_engine == "upstream"
            else None
        )
        if tts_wrapper is None and not tts_warming:
            tts_loaded: Optional[str] = None
        elif tts_fallback_from is not None:
            tts_loaded = "melotts_onnx"
        else:
            tts_loaded = live_config.providers.tts
        signature = getattr(
            tts_wrapper, "_signature", getattr(tts_inner, "cache_signature", None)
        )

        memory_cfg = live_config.memory
        payload: Dict[str, object] = {
            "version": _app_version(),
            # True on hosted public demos: the settings/persona mutation
            # surface is locked (403) and the UI hides those controls.
            "demo_mode": bool(getattr(live_config.web, "demo_mode", False)),
            # BYO-character identity (config.character); defaults to the
            # built-in Kurisu. The UI can use this for titles/copy.
            "character": {
                "name": getattr(
                    getattr(live_config, "character", None), "name", "Kurisu"
                )
                or "Kurisu",
                "full_name": getattr(
                    getattr(live_config, "character", None),
                    "full_name",
                    "Makise Kurisu",
                )
                or "Makise Kurisu",
                "custom": bool(
                    getattr(getattr(live_config, "character", None), "is_custom", False)
                ),
                "model_id": getattr(
                    getattr(live_config, "character", None), "model_id", None
                ),
            },
            # Her sleep window (amadeus/sleep_schedule.py): schedule + live
            # state, so the UI can render the dozing pet / status chip
            # instead of reading "no chatter" as a dead presence loop.
            "sleep": sleep_status(live_config),
            # Focus/co-working mode (amadeus/focus_mode.py): config-only
            # availability (enabled + default length) — the live
            # active/until state is per-session and rides POST /api/focus
            # (action "status"), same as every other presence endpoint.
            "focus": focus_config_status(live_config),
            "llm": {
                "provider": live_config.providers.llm,
                "class": type(llm).__name__,
                "model": getattr(llm, "model", None),
                "is_dummy": isinstance(llm, DummyProvider),
            },
            "tts": {
                "requested": (
                    live_config.providers.tts
                    if live_config.audio.enable_voice
                    else None
                ),
                "loaded": tts_loaded,
                "class": type(tts_inner).__name__ if tts_inner else None,
                "engine": tts_engine,
                "engine_warning": engine_warning,
                "cache_signature": str(signature) if signature else None,
                "warming": tts_warming,
                "fallback_from": tts_fallback_from,
                "error": tts_error,
            },
            "stt": {
                "provider": live_config.providers.stt,
                "loaded": stt_loaded,
                "unavailable_reason": stt_reason,
            },
            "memory": {
                "long_term": bool(memory_cfg.enable_long_term),
                "fact_extraction": bool(memory_cfg.enable_fact_extraction),
                "episodes": bool(memory_cfg.enable_episodes),
                "greeting": bool(memory_cfg.enable_greeting),
                "embedder": type(embedder).__name__ if embedder else None,
                "summary_provider_active": not isinstance(llm, DummyProvider),
            },
            "tools": _tools_status(live_config),
            # Process-lifetime health counters (amadeus/telemetry.py): summary/
            # episode/embed failures, empty-recall turns, provider rollbacks,
            # tagged-repair passes. The question they answer is "has anything
            # been quietly failing?" — nonzero failure counts deserve a look.
            "counters": telemetry.snapshot(),
            # Per-key duration stats (amadeus/telemetry.py record_duration):
            # {count, total_ms, max_ms, last_ms} per key, e.g.
            # "turn.first_audio_ms" — the headline silence-to-first-audio
            # number. average = total_ms / count.
            "durations": telemetry.duration_snapshot(),
            # Per-key non-duration value stats (telemetry.record_value), same
            # shape as durations: e.g. "llm.prompt_est_tokens" — the chars/4
            # estimated size of each assembled request (p50/p95 from the
            # 64-sample ring). The token-accounting counterpart to the
            # latency chain.
            "values": telemetry.value_snapshot(),
            # LLM provider health classification (amadeus/provider_health.py):
            # {state, detail, since} — ok/rate_limited/invalid_key/error/
            # dummy. Self-healing: a rate_limited/error state clears back to
            # ok the moment a call succeeds again, so this always reflects
            # "right now", not "ever went wrong".
            "llm_health": provider_health.llm_health_snapshot(),
            # TTS provider health classification (amadeus/provider_health.py):
            # {state, detail, since, consecutive_failures} — ok/warming/
            # degraded/failing/off. Aggregates the scattered TTS signals
            # (tts_warming, tts.*_synth_failures counters) into one "is voice
            # OK right now" field, the same job llm_health does for the LLM
            # side — it does not replace any of those existing signals.
            # loaded/warming come from the same store-lock read as the rest
            # of this endpoint's tts block, above.
            "tts_health": provider_health.tts_health_snapshot(
                loaded=tts_wrapper is not None, warming=tts_warming
            ),
            # Nightly conversation-quality self-check (amadeus/self_eval.py,
            # config.self_eval, default off): {last_run_at, scores,
            # regression, regressions}. All-None/False when the job is
            # disabled or hasn't run yet — same "absent/empty means healthy"
            # contract as counters/durations.
            "self_eval": self_eval.status_snapshot(),
        }
        # Slim GPU installer's runtime-pack fetch/activation state
        # (amadeus/gpu_runtime.py) — only present once the controller has
        # actually recorded state this process (activate()/ensure_pack()
        # called, which only happens for --prefer-qwen desktop launches).
        # Omitted entirely on web/CPU deployments so the UI has no dead
        # section for a feature that build doesn't have.
        if gpu_runtime.is_initialized():
            payload["gpu_runtime"] = gpu_runtime.status()
        return payload

    @router.get("/api/settings", response_model=settings_api.SettingsResponse)
    async def get_settings() -> settings_api.SettingsResponse:
        return settings_api.build_settings_response(store.config, llm=store.llm)

    @router.get("/api/settings/openai-models")
    def openai_models() -> Dict[str, Any]:
        from openai import OpenAI

        cfg = store.config
        if not cfg.openai_api_key:
            raise HTTPException(400, "OpenAI project key is not configured")
        try:
            with OpenAI(api_key=cfg.openai_api_key, base_url=cfg.providers.openai_base_url,
                        timeout=12, max_retries=0) as client:
                catalog = client.models.list().data
        except Exception as exc:
            raise HTTPException(502, "Could not load the model list") from exc
        excluded = ("audio", "realtime", "search", "image", "transcribe", "tts", "embedd",
                    "moderation", "codex", "deep-research", "computer-use", "-pro")
        models = [m for m in catalog if re.match(r"^(gpt-[0-9]|o[1-9])", m.id)
                  and not any(tag in m.id for tag in excluded)]
        models.sort(key=lambda m: (m.created, m.id), reverse=True)
        return {"models": [m.id for m in models]}

    @router.put("/api/settings", response_model=settings_api.SettingsUpdateResponse)
    async def update_settings(
        payload: settings_api.SettingsUpdate,
    ) -> settings_api.SettingsUpdateResponse:
        path = settings_api.user_config_path()
        # Seed with whatever's actually running, not just what's on disk: a
        # key can be live via an env var (OPENAI_API_KEY) or a config file
        # this process was started with that isn't the user-settings path
        # (e.g. config.local.yaml). Without this, saving any other setting
        # would silently drop that key from the persisted file.
        raw = {
            **settings_api.effective_raw(store.config),
            **settings_api.load_raw(path),
        }
        new_raw, needs_restart, reason = settings_api.apply_update(raw, payload)
        old_identity = persona_versions.identity_payload(store.config)
        new_config = settings_api.persist_and_reload(store, new_raw)

        # Identity continuity: if this save changed the persona/voice, keep
        # the *previous* identity restorable and let the next greeting
        # acknowledge the change (never silent, always reversible).
        try:
            new_identity = persona_versions.identity_payload(new_config)
            if persona_versions.identity_fingerprint(
                old_identity
            ) != persona_versions.identity_fingerprint(new_identity):
                vstore = _persona_version_store(new_config)
                if vstore is not None:
                    vstore.snapshot(old_identity)
        except Exception as exc:  # pragma: no cover - snapshot is best-effort
            LOGGER.warning("Persona version auto-snapshot failed: %s", exc)

        return settings_api.SettingsUpdateResponse(
            # reload_providers already swapped in the new LLM instance.
            settings=settings_api.build_settings_response(new_config, llm=store.llm),
            needs_restart=needs_restart,
            restart_reason=reason,
        )

    @router.post("/api/settings/test", response_model=settings_api.ProviderTestResponse)
    async def test_provider(
        payload: settings_api.ProviderTestRequest,
    ) -> settings_api.ProviderTestResponse:
        return await asyncio.to_thread(
            settings_api.run_provider_test, store.config, payload
        )

    @router.post("/api/self_test")
    async def run_self_test_endpoint() -> Dict[str, object]:
        """End-to-end "is she actually working?" check (web/self_test.py).

        POST, not GET: it round-trips the live LLM and synthesizes real
        audio, so it costs tokens and GPU time and must never be something a
        poller or a prefetch can trigger. Unlike /api/settings/test (which
        probes a CANDIDATE config from unsaved form fields), this exercises
        the providers this session is actually running.

        Off-thread because every leg is blocking provider I/O; the module
        guarantees no leg raises, so there is no error path to map here.
        """
        return await asyncio.to_thread(self_test.run_self_test, store)

    # --- Speech-to-text (push-to-talk) ------------------------------------
    #
    # The frontend records mic audio and encodes it to a 16kHz mono 16-bit
    # PCM WAV clip in-browser (see ChatInterface.tsx), so the only parsing
    # needed server-side is a minimal WAV header read via the stdlib `wave`
    # module — no ffmpeg/codec dependency.

    def _probe_stt_importable(provider_name: str) -> bool:
        """Cheap importability check for /api/stt/status — does not
        construct or load any model, just checks the dependency exists."""
        if provider_name == "faster_whisper":
            return importlib.util.find_spec("faster_whisper") is not None
        return False

    @router.get("/api/stt/status")
    async def stt_status() -> Dict[str, object]:
        provider_name = config.providers.stt
        if provider_name == "dummy":
            return {
                "available": False,
                "reason": "Speech-to-text is disabled in config.",
            }
        importable = await asyncio.to_thread(_probe_stt_importable, provider_name)
        if not importable:
            return {
                "available": False,
                "reason": f"STT provider '{provider_name}' is not installed.",
            }
        return {"available": True}

    @router.post("/api/stt")
    async def stt_endpoint(audio: UploadFile) -> Dict[str, object]:
        stt = await asyncio.to_thread(store.get_stt)
        if stt is None:
            raise HTTPException(
                status_code=501,
                detail={
                    "available": False,
                    "reason": store.stt_unavailable_reason
                    or "Speech-to-text is not available on this server.",
                },
            )

        raw = await audio.read()
        try:
            with wave.open(io.BytesIO(raw), "rb") as wav_file:
                sample_rate = wav_file.getframerate()
                pcm_bytes = wav_file.readframes(wav_file.getnframes())
        except (wave.Error, EOFError) as exc:
            raise HTTPException(
                status_code=400, detail=f"Invalid WAV audio: {exc}"
            ) from exc

        started = time.monotonic()
        try:
            text = await _run_blocking(stt.transcribe_audio, pcm_bytes, sample_rate)
        except Exception as exc:  # pragma: no cover - runtime provider errors
            LOGGER.error("STT transcription failed: %s", exc)
            raise HTTPException(
                status_code=500, detail=f"Transcription failed: {exc}"
            ) from exc

        # Voice-to-voice latency starts here: record how long server-side
        # transcription took. Two facilities record the same measurement —
        # the pre-existing latency.stt_* sum/count/last triple (left as-is,
        # existing tests pin its exact keys) and the new record_duration
        # (stt.request_ms, part of the durations snapshot alongside
        # llm.first_token_ms/tts.first_audio_ms/turn.first_audio_ms so the
        # whole voice chain reads from one place).
        try:
            stt_ms = int(1000 * (time.monotonic() - started))
            telemetry.incr("latency.stt_ms_sum", stt_ms)
            telemetry.incr("latency.stt_count")
            telemetry.set_gauge("latency.last_stt_ms", stt_ms)
            telemetry.record_duration("stt.request_ms", stt_ms)
            LOGGER.info(
                "stt transcription: %dms (%d bytes pcm)", stt_ms, len(pcm_bytes)
            )
        except Exception:  # pragma: no cover - instrumentation must never break STT
            pass

        return {"text": text}

    async def _synthesize_presence_line(
        session: "WebChatSession", text: str, *, log_label: str
    ) -> Dict[str, Any]:
        """Shared tail for greeting/poke/idle: emotion-tag the line and
        (best-effort) synthesize audio for it. All three are short,
        unprompted in-character lines that follow the same shape."""
        emotion_data = session.emotion_analyzer.analyze(text)
        audio_payload: Optional[str] = None
        audio_mime: Optional[str] = None
        if session.tts and config.audio.enable_voice and session.audio_enabled:
            try:
                japanese_text = text
                voice_language = str(
                    getattr(config.audio, "voice_language", "japanese") or "japanese"
                ).lower()
                if session.translator and voice_language != "english":
                    try:
                        japanese_text = await _run_blocking(
                            session.translator.translate, text
                        )
                    except Exception as exc:
                        LOGGER.warning(
                            "%s translation failed, using English: %s", log_label, exc
                        )
                audio_bytes = await _synthesize_tts(
                    session.tts,
                    japanese_text,
                    speaking_rate=config.persona.speaking_rate,
                    pitch_shift=config.persona.pitch_shift,
                )
                audio_payload = base64.b64encode(audio_bytes).decode("ascii")
                audio_mime = "audio/wav"
            except Exception as exc:  # pragma: no cover - runtime TTS errors
                LOGGER.error("%s synthesis failed: %s", log_label, exc)
                telemetry.incr("tts.presence_synth_failures")

        return {
            "emotion": emotion_data.get("vrm_expression", "relaxed"),
            "emotion_intensity": emotion_data.get("intensity", 0.6),
            "audio": audio_payload,
            "audio_mime_type": audio_mime,
        }

    @router.post("/api/greeting")
    async def greeting_endpoint(payload: GreetingRequest) -> Dict[str, Any]:
        session_id, session = store.get(payload.session_id, payload.user_id)
        async with session.handle_lock:
            greeting = await _run_blocking(session.open_session_greeting)
        if not greeting:
            return {"session_id": session_id, "greeting": None}

        # Read back the exact milestone events open_session_greeting used to
        # build the prompt, rather than recomputing milestone_events() here
        # — a second independent call could diverge from what the greeting
        # text actually said (e.g. a session boundary ticking over between
        # the two calls).
        milestones = getattr(session, "last_greeting_milestones", None) or []

        extra = await _synthesize_presence_line(session, greeting, log_label="Greeting")
        return {
            "session_id": session_id,
            "greeting": greeting,
            "milestones": milestones,
            **extra,
        }

    @router.post("/api/poke")
    async def poke_endpoint(payload: PresenceRequest) -> Dict[str, Any]:
        session_id, session = store.get(payload.session_id, payload.user_id)
        async with session.handle_lock:
            reaction = await _run_blocking(session.poke_reaction, region=payload.region)
        if not reaction:
            return {"session_id": session_id, "reaction": None}

        extra = await _synthesize_presence_line(
            session, reaction, log_label="Poke reaction"
        )
        return {"session_id": session_id, "reaction": reaction, **extra}

    @router.post("/api/ambient")
    async def ambient_endpoint(payload: AmbientRequest) -> Dict[str, Any]:
        """Ambient desktop awareness: one client-detected transition in,
        at most one short in-character line out, riding the same presence
        tail as poke/idle. All gates (enabled, sleep window, cooldown,
        SKIP) live in presence.ambient_comment."""
        session_id, session = store.get(payload.session_id, payload.user_id)
        async with session.handle_lock:
            comment = await _run_blocking(
                session.ambient_comment, payload.kind, payload.detail
            )
        if not comment:
            return {"session_id": session_id, "comment": None}

        extra = await _synthesize_presence_line(
            session, comment, log_label="Ambient comment"
        )
        return {"session_id": session_id, "comment": comment, **extra}

    @router.post("/api/focus")
    async def focus_endpoint(payload: FocusRequest) -> Dict[str, Any]:
        """Focus / co-working mode: start a heads-down session, end one
        early, or poll status. Quiet company, not a productivity tracker —
        all gating/check-in logic lives in amadeus/focus_mode.py. A "status"
        poll that observes the session already expired delivers the
        one-shot check-in line exactly like an explicit "end" would."""
        if not getattr(getattr(config, "focus", None), "enabled", True):
            raise HTTPException(status_code=404, detail="Focus mode is disabled")
        session_id, session = store.get(payload.session_id, payload.user_id)
        async with session.handle_lock:
            if payload.action == "start":
                state = await _run_blocking(session.start_focus, payload.minutes)
            elif payload.action == "end":
                state = await _run_blocking(session.end_focus)
            else:
                state = await _run_blocking(session.focus_status)
        checkin = state.get("checkin")
        result: Dict[str, Any] = {"session_id": session_id, **state}
        if not checkin:
            return result
        extra = await _synthesize_presence_line(
            session, checkin, log_label="Focus check-in"
        )
        return {**result, **extra}

    @router.post("/api/screen_peek")
    async def screen_peek_endpoint(payload: ScreenPeekRequest) -> Dict[str, Any]:
        """Pet mode's opt-in screen companion: one screenshot in, one short
        in-character line out, riding the same presence tail as poke/idle.
        Server-side gates (vision.enabled, cooldown, vision-capable LLM)
        live in ChatEngine.screen_comment."""
        session_id, session = store.get(payload.session_id, payload.user_id)
        async with session.handle_lock:
            comment = await _run_blocking(
                session.screen_comment, payload.image.mime, payload.image.data
            )
        if not comment:
            return {"session_id": session_id, "comment": None}

        extra = await _synthesize_presence_line(
            session, comment, log_label="Screen comment"
        )
        return {"session_id": session_id, "comment": comment, **extra}

    @router.post("/api/read_along")
    async def read_along_endpoint(payload: ReadAlongRequest) -> Dict[str, Any]:
        """ "Read together" mode: one screenshot per page turn (image path)
        or a session-close signal (end path). Server-side gates
        (vision.enabled, vision-capable LLM, per-session page cap, the
        speak cooldown) live in ChatEngine.read_along_page/read_along_end;
        this endpoint only shapes the response and, when a line was
        actually produced, folds in the shared presence tail (emotion +
        best-effort TTS) exactly like screen_peek.
        """
        session_id, session = store.get(payload.session_id, payload.user_id)
        if payload.end:
            async with session.handle_lock:
                comment = await _run_blocking(session.read_along_end)
            if not comment:
                return {"session_id": session_id, "comment": None}
            extra = await _synthesize_presence_line(
                session, comment, log_label="Read-along"
            )
            return {"session_id": session_id, "comment": comment, **extra}

        async with session.handle_lock:
            comment = await _run_blocking(
                session.read_along_page, payload.image.mime, payload.image.data
            )
            pages = getattr(session._read_along, "pages", 0)
        if not comment:
            return {"session_id": session_id, "comment": None, "pages": pages}

        extra = await _synthesize_presence_line(
            session, comment, log_label="Read-along"
        )
        return {"session_id": session_id, "comment": comment, "pages": pages, **extra}

    @router.post("/api/watch_together")
    async def watch_together_endpoint(payload: WatchTogetherRequest) -> Dict[str, Any]:
        """ "Watch together" mode: one periodic screen frame in (image path)
        or a session-close signal (end path). Server-side gates
        (vision.enabled, vision-capable provider, frame cap, speak cooldown)
        live in ChatEngine.watch_frame/watch_end; this endpoint only shapes
        the response and folds in the shared presence tail exactly like
        read_along."""
        session_id, session = store.get(payload.session_id, payload.user_id)
        if payload.end:
            async with session.handle_lock:
                comment = await _run_blocking(session.watch_end)
            if not comment:
                return {"session_id": session_id, "comment": None}
            extra = await _synthesize_presence_line(
                session, comment, log_label="Watch-together"
            )
            return {"session_id": session_id, "comment": comment, **extra}

        async with session.handle_lock:
            comment = await _run_blocking(
                session.watch_frame, payload.image.mime, payload.image.data
            )
            frames = getattr(session._watch_together, "frames", 0)
        if not comment:
            return {"session_id": session_id, "comment": None, "frames": frames}

        extra = await _synthesize_presence_line(
            session, comment, log_label="Watch-together"
        )
        return {"session_id": session_id, "comment": comment, "frames": frames, **extra}

    @router.post("/api/idle")
    async def idle_endpoint(payload: PresenceRequest) -> Dict[str, Any]:
        session_id, session = store.get(payload.session_id, payload.user_id)
        async with session.handle_lock:
            chatter = await _run_blocking(session.idle_chatter)
        if not chatter:
            return {"session_id": session_id, "chatter": None}

        extra = await _synthesize_presence_line(
            session, chatter, log_label="Idle chatter"
        )
        return {"session_id": session_id, "chatter": chatter, **extra}

    @router.get("/api/brief")
    async def brief_endpoint(user_id: Optional[str] = None) -> Dict[str, Any]:
        """Read-only view of the day's brief (the reference copy for the UI
        card). Never consumes — only the greeting marks a brief consumed."""
        if store.brief_store is None:
            return {"brief": None}
        try:
            brief = store.brief_store.get(user_id)
        except Exception as exc:  # pragma: no cover - disk failure
            LOGGER.warning("Brief read failed: %s", exc)
            return {"brief": None}
        if brief is None:
            return {"brief": None}
        return {
            "brief": {
                "date": brief.date,
                "text": brief.text,
                "generated_at": brief.generated_at,
                "consumed": brief.consumed,
            }
        }

    @router.get("/api/diary")
    async def diary_endpoint(user_id: Optional[str] = None) -> Dict[str, Any]:
        """Read-only view of today's diary entry (the reference copy for the
        UI card). Never consumes — only the greeting/idle-chatter path marks
        an entry consumed."""
        if store.diary_store is None:
            return {"diary": None}
        try:
            entry = store.diary_store.get(user_id)
        except Exception as exc:  # pragma: no cover - disk failure
            LOGGER.warning("Diary read failed: %s", exc)
            return {"diary": None}
        if entry is None:
            return {"diary": None}
        return {
            "diary": {
                "date": entry.date,
                "text": entry.text,
                "generated_at": entry.generated_at,
                "consumed": entry.consumed,
            }
        }

    @router.get("/api/insights")
    async def insights_endpoint(user_id: Optional[str] = None) -> Dict[str, Any]:
        """Read-only view of the current standing-insight set (nightly
        consolidation, amadeus/insights.py). Transparency surface: the user
        can always see what she has concluded about how to treat them. An
        empty `insights` list with a date means "consolidated today, nothing
        well-supported to say"."""
        if store.insight_store is None:
            return {"insights": None}
        try:
            current = store.insight_store.get(user_id)
        except Exception as exc:  # pragma: no cover - disk failure
            LOGGER.warning("Insights read failed: %s", exc)
            return {"insights": None}
        if current is None:
            return {"insights": None}
        return {
            "insights": {
                "date": current.date,
                "generated_at": current.generated_at,
                "insights": [
                    {"text": entry.text, "kind": entry.kind}
                    for entry in current.insights
                ],
                "dismissed": list(current.dismissed),
            }
        }

    @router.post("/api/insights/dismiss")
    async def insights_dismiss_endpoint(
        payload: InsightDismissRequest,
    ) -> Dict[str, Any]:
        """Remove one standing insight from the user's own record.

        The removed text is remembered (``InsightSet.dismissed``) and fed to
        the next nightly consolidation as "do not restate this" — without
        that, tonight's job regenerates what they deleted this morning and
        the delete button is a lie. Best-effort: a disk failure reports
        ``dismissed: false`` rather than raising.
        """
        if store.insight_store is None:
            return {"dismissed": False}
        try:
            removed = await _run_blocking(
                store.insight_store.dismiss, payload.user_id, payload.text
            )
        except Exception as exc:  # pragma: no cover - disk failure
            LOGGER.warning("Insight dismiss failed: %s", exc)
            telemetry.incr("insights.dismiss_failed")
            return {"dismissed": False}
        if removed:
            telemetry.incr("insights.dismissed")
        return {"dismissed": bool(removed)}

    @router.post("/api/rhythm")
    async def rhythm_record_endpoint(payload: RhythmRequest) -> Dict[str, Any]:
        """Record one local day's coarse activity tally (amadeus/rhythm.py).

        Its own endpoint rather than a fourth ambient kind: this is a rollup,
        not a transition, and it deliberately wants no line back — routing it
        through ambient_comment would drag it into that path's LLM call.

        The client has already reduced to counters and app CATEGORIES before
        posting, and RhythmDay.from_dict drops anything outside the fixed
        vocabulary, so neither a modified client nor a hand-edited file can
        widen what this store holds.
        """
        if store.rhythm_store is None:
            return {"recorded": False}
        try:
            from ..rhythm import RhythmDay

            day = RhythmDay.from_dict(payload.model_dump())
            if day is None:
                return {"recorded": False}
            await _run_blocking(
                store.rhythm_store.record, payload.user_id, day, dt.date.today()
            )
        except Exception as exc:  # pragma: no cover - disk failure
            LOGGER.warning("Rhythm record failed: %s", exc)
            telemetry.incr("rhythm.write_failures")
            return {"recorded": False}
        return {"recorded": True}

    @router.get("/api/rhythm")
    async def rhythm_read_endpoint(user_id: Optional[str] = None) -> Dict[str, Any]:
        """The whole stored tally, verbatim.

        Transparency surface: the settings panel renders exactly this, so the
        user can always read the complete artifact in one screen. That
        readability is a design constraint on what may be stored, not just a
        feature.
        """
        if store.rhythm_store is None:
            return {"rhythm": None}
        try:
            days = await _run_blocking(store.rhythm_store.days, user_id)
        except Exception as exc:  # pragma: no cover - disk failure
            LOGGER.warning("Rhythm read failed: %s", exc)
            return {"rhythm": None}
        return {"rhythm": {"days": [day.to_dict() for day in days]}}

    @router.delete("/api/rhythm")
    async def rhythm_clear_endpoint(user_id: Optional[str] = None) -> Dict[str, Any]:
        """Erase the stored tally. Toggling the feature off calls this — the
        promise is that turning it off DELETES the record, not that it stops
        adding to one."""
        if store.rhythm_store is None:
            return {"cleared": False}
        try:
            await _run_blocking(store.rhythm_store.clear, user_id)
        except Exception as exc:  # pragma: no cover - disk failure
            LOGGER.warning("Rhythm clear failed: %s", exc)
            return {"cleared": False}
        return {"cleared": True}

    @router.get("/api/followup")
    async def followup_endpoint(user_id: Optional[str] = None) -> Dict[str, Any]:
        """Read-only view of the current follow-up record (note OR skip —
        the skip is included so Diagnostics can distinguish "researched,
        nothing useful" from "never ran"). Never consumes — only the
        greeting marks a note consumed."""
        if store.followup_store is None:
            return {"followup": None}
        try:
            entry = store.followup_store.get(user_id)
        except Exception as exc:  # pragma: no cover - disk failure
            LOGGER.warning("Follow-up read failed: %s", exc)
            return {"followup": None}
        if entry is None:
            return {"followup": None}
        return {
            "followup": {
                "thread": entry.thread,
                "note": entry.note,
                "kind": entry.kind,
                "generated_at": entry.generated_at,
                "sources": entry.sources,
                "consumed": entry.consumed,
            }
        }

    @router.get("/api/life_threads")
    async def life_threads_endpoint() -> Dict[str, Any]:
        """Read-only view of her ongoing pursuits (not per-user: her life
        doesn't fork per person). Never advances anything — only the
        scheduler progresses threads."""
        if store.life_store is None:
            return {"threads": []}
        try:
            threads = store.life_store.all_threads()
        except Exception as exc:  # pragma: no cover - disk failure
            LOGGER.warning("Life-thread read failed: %s", exc)
            return {"threads": []}
        return {
            "threads": [
                {
                    "topic": t.topic,
                    "summary": t.summary,
                    "status": t.status,
                    "started_at": t.started_at,
                    "last_progressed_at": t.last_progressed_at,
                    "progressions": t.progressions,
                }
                for t in threads
            ]
        }

    @router.get("/api/vision/status")
    async def vision_status() -> Dict[str, Any]:
        """Whether image attachments will actually reach the model, so the
        UI can hide the capture button on dummy/non-vision providers."""
        vision_cfg = getattr(store.config, "vision", None)
        enabled = bool(vision_cfg and getattr(vision_cfg, "enabled", False))
        supports = False
        if enabled and store.llm is not None:
            try:
                supports = bool(store.llm.supports_vision())
            except Exception:  # pragma: no cover - defensive
                supports = False
        reason = None
        if not enabled:
            reason = "Vision is disabled in config (vision.enabled)"
        elif not supports:
            reason = "The active LLM provider does not accept images"
        return {
            "available": enabled and supports,
            "reason": reason,
            "max_image_px": getattr(vision_cfg, "max_image_px", 1568),
            "jpeg_quality": getattr(vision_cfg, "jpeg_quality", 85),
        }

    @router.post("/api/farewell")
    async def farewell_endpoint(payload: FarewellRequest) -> Dict[str, Any]:
        """The user is leaving. Two shapes:

        - reason="goodbye": the user typed/said goodbye — validate the intent
          server-side, generate ONE warm anti-retention send-off line, commit
          the turn, finalize the session episode, and voice the line through
          the shared presence path.
        - reason="window_close": fired as a keepalive beacon on page hide —
          silent (no LLM, no TTS, nothing could play); best-effort session
          finalize only, and never materializes a session that doesn't exist.
        """
        if payload.reason == "window_close":
            session = store.peek(payload.session_id, payload.user_id)
            if session is not None:
                try:
                    session.memory.finalize_session(wait=False)
                except Exception as exc:  # pragma: no cover - disk failure
                    LOGGER.warning("Window-close finalize failed: %s", exc)
            return {"session_id": payload.session_id, "farewell": None}

        session_id, session = store.get(payload.session_id, payload.user_id)
        if not payload.message or not is_goodbye(payload.message):
            # Server-authoritative detection: tell the client to fall back
            # to a normal chat turn instead of a farewell.
            return {"session_id": session_id, "farewell": None, "not_goodbye": True}
        async with session.handle_lock:
            line = await _run_blocking(session.close_session_farewell, payload.message)
            if line:
                # Commit the goodbye + send-off exactly once, only now that a
                # line exists: memory (user half here, assistant half inside
                # complete_turn) plus the transcript.
                await _run_blocking(session.memory.add_message, "user", payload.message)
                await _run_blocking(session.complete_turn, payload.message, line)
                session.memory.finalize_session(wait=False)
        if not line:
            # It read as a goodbye but no line was produced (cooldown,
            # disabled, provider missing) — let the client run a normal turn.
            return {"session_id": session_id, "farewell": None, "not_goodbye": True}
        extra = await _synthesize_presence_line(session, line, log_label="Farewell")
        return {"session_id": session_id, "farewell": line, **extra}

    @router.post("/api/reminders/due")
    async def due_reminders_endpoint(payload: PresenceRequest) -> Dict[str, Any]:
        """Announce reminders that came due — the web half of the reminders
        tool (the CLI announces in its own prompt loop). Polled by the
        frontend; each reminder is announced exactly once (pop_due marks it),
        and the line speaks through the same presence path as greeting/poke.

        Also stamps the client-liveness snapshot (async_notify.py): a
        polling client is the signal the async-notify scheduler job uses to
        stay quiet, since it runs in a different process lifetime and can't
        see this request loop directly.
        """
        await _run_blocking(record_client_poll, store.config)
        session_id, session = store.get(payload.session_id, payload.user_id)
        async with session.handle_lock:
            lines = await _run_blocking(session.due_reminder_lines)
        if not lines:
            return {"session_id": session_id, "reminder": None}

        text = " ".join(lines)
        extra = await _synthesize_presence_line(session, text, log_label="Reminder")
        return {"session_id": session_id, "reminder": text, **extra}

    @router.post("/api/reachout/due")
    async def due_reachout_endpoint(payload: PresenceRequest) -> Dict[str, Any]:
        """Deliver the day's proactive reach-out line, exactly once
        (``pop_unconsumed`` marks it atomically, like reminders' pop_due).
        Polled alongside reminders; the line rides the same presence path,
        and is committed to memory so she remembers having said it. Also
        stamps the client-liveness snapshot (see the reminders endpoint's
        docstring) — this poll fires on the same 30s cadence.
        """
        await _run_blocking(record_client_poll, store.config)
        if store.reachout_store is None:
            return {"session_id": payload.session_id, "reach_out": None}
        # Sleep window: the line is HELD, not consumed — she doesn't reach
        # out in her sleep; the same poll delivers it after she wakes.
        if is_sleep_hours(store.config):
            telemetry.incr("presence.sleep_suppressed")
            return {"session_id": payload.session_id, "reach_out": None}
        session_id, session = store.get(payload.session_id, payload.user_id)
        # Focus session: HELD, not consumed, exactly like the sleep window
        # above — she doesn't reach out while the user is heads-down; the
        # same poll delivers it once the session ends.
        if is_focus_active(session):
            telemetry.incr("presence.focus_suppressed")
            return {"session_id": session_id, "reach_out": None}
        async with session.handle_lock:
            entry = await _run_blocking(
                store.reachout_store.pop_unconsumed,
                payload.user_id,
                dt.date.today(),
            )
            if entry is None:
                return {"session_id": session_id, "reach_out": None}
            # Committed like idle chatter: an unprompted line she said out
            # loud is part of the conversation, not ambient noise — without
            # this, the user's "yes, it went out!" reply would make no sense
            # in the next turn's context. presence_delivery: a reach-out is
            # by definition preceded by >= min_idle_hours of silence, so a
            # plain add_message fired the gap-finalize boundary (phantom
            # session_count/trust/arc drift) and stamped last_seen while the
            # user was away.
            await _run_blocking(
                session.memory.add_message,
                "assistant",
                entry.text,
                presence_delivery=True,
            )

        extra = await _synthesize_presence_line(
            session, entry.text, log_label="Reach-out"
        )
        return {"session_id": session_id, "reach_out": entry.text, **extra}

    _UNAVAILABLE_RESPONSE = {
        "status": "unavailable",
        "detail": "Long-term memory is disabled on this server.",
    }
    _PROTECTED_RESPONSE = {
        "status": "protected",
        "detail": "That memory is pinned — unpin it first.",
    }

    def _long_term_unavailable(session: "WebChatSession") -> bool:
        """Mirrors the gate `remember` checks internally: long-term memory
        (and therefore pin/edit/forget_id) is only live once a vector store
        was constructed, which only happens when enable_long_term is on."""
        return session.memory.vector_store is None

    @router.get("/api/memory/export")
    async def memory_export(user_id: Optional[str] = None):
        """Portable zip of the user's memory files (profile, store,
        embeddings, transcripts). The user owns her memories of them."""
        from fastapi.responses import Response

        from ..memory import backup

        data = await _run_blocking(backup.export_memory, store.config, user_id)
        if data is None:
            raise HTTPException(
                status_code=404, detail="No memory files to export yet."
            )
        stamp = dt.date.today().isoformat()
        return Response(
            content=data,
            media_type="application/zip",
            headers={
                "Content-Disposition": f'attachment; filename="amadeus-memory-{stamp}.zip"'
            },
        )

    @router.post("/api/memory/import")
    async def memory_import(
        file: UploadFile, user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Restore memory files from an export zip.

        Validates before touching anything, takes a pre-import backup of the
        current files, then swaps atomically and drops the user's live
        sessions so the next request reloads the imported state from disk.
        """
        from ..memory import backup

        data = await file.read()
        try:
            restored = await _run_blocking(
                backup.import_memory, store.config, user_id, data
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        dropped = store.drop_user_sessions(user_id)
        return {"status": "ok", "restored": restored, "sessions_reloaded": dropped}

    @router.post("/api/memory")
    async def memory_endpoint(payload: MemoryRequest) -> Dict[str, Any]:
        _, session = store.get(payload.session_id, payload.user_id)
        action = (payload.action or "").lower()

        async def _mutated(response: Dict[str, Any]) -> Dict[str, Any]:
            """A user-intent mutation succeeded on this session's manager —
            refresh sibling live sessions over the same files BEFORE
            replying, or their stale RAM copies resurrect deleted records
            (and drop added ones) on their own next persist. See
            WebSessionStore.refresh_sibling_long_term."""
            await _run_blocking(store.refresh_sibling_long_term, session)
            return response

        if action == "remember":
            # remember/forget embed text; keep the event loop free.
            stored = await _run_blocking(
                session.memory.remember,
                payload.text or "",
                pinned=bool(payload.pinned),
            )
            if stored:
                return await _mutated(
                    {
                        "status": "ok",
                        "detail": f"Committed to memory: {payload.text}",
                    }
                )
            return dict(_UNAVAILABLE_RESPONSE)
        if action in ("pin", "unpin"):
            if not payload.record_id:
                raise HTTPException(
                    status_code=400, detail="record_id is required for pin/unpin"
                )
            if _long_term_unavailable(session):
                return dict(_UNAVAILABLE_RESPONSE)
            updated = await _run_blocking(
                session.memory.set_pinned, payload.record_id, action == "pin"
            )
            if updated:
                return await _mutated({"status": "ok"})
            return {"status": "not_found", "detail": "No such memory."}
        if action == "edit":
            if not payload.record_id:
                raise HTTPException(
                    status_code=400, detail="record_id is required for edit"
                )
            if not payload.text:
                raise HTTPException(status_code=400, detail="text is required for edit")
            if _long_term_unavailable(session):
                return dict(_UNAVAILABLE_RESPONSE)
            updated = await _run_blocking(
                session.memory.edit_memory, payload.record_id, payload.text
            )
            if updated:
                return await _mutated({"status": "ok"})
            return {"status": "not_found", "detail": "No such memory."}
        if action == "forget_id":
            if not payload.record_id:
                raise HTTPException(
                    status_code=400, detail="record_id is required for forget_id"
                )
            if _long_term_unavailable(session):
                return dict(_UNAVAILABLE_RESPONSE)
            result = await _run_blocking(session.memory.forget_id, payload.record_id)
            if result is PROTECTED:
                return dict(_PROTECTED_RESPONSE)
            if result is not None:
                return await _mutated(
                    {"status": "ok", "detail": f"Forgotten: {result}"}
                )
            return {"status": "not_found", "detail": "No such memory."}
        if action == "forget":
            forgotten = await _run_blocking(session.memory.forget, payload.text or "")
            if forgotten is PROTECTED:
                return dict(_PROTECTED_RESPONSE)
            if forgotten:
                return await _mutated(
                    {"status": "ok", "detail": f"Forgotten: {forgotten}"}
                )
            return {"status": "not_found", "detail": "No matching memory found."}
        if action == "forget_fact":
            removed = await _run_blocking(
                session.memory.forget_fact, payload.text or ""
            )
            if removed:
                return await _mutated(
                    {"status": "ok", "detail": f"Forgot fact: {payload.text}"}
                )
            return {"status": "not_found", "detail": "No such fact."}
        if action == "forget_thread":
            removed = await _run_blocking(
                session.memory.forget_open_thread, payload.text or ""
            )
            if removed:
                return await _mutated(
                    {"status": "ok", "detail": "Dropped that thread."}
                )
            return {"status": "not_found", "detail": "No such open thread."}
        if action == "reset_relationship":
            reset = await _run_blocking(session.memory.reset_relationship)
            if reset:
                return await _mutated(
                    {"status": "ok", "detail": "Arc and tone history reset."}
                )
            return {
                "status": "unavailable",
                "detail": "Long-term memory is disabled on this server.",
            }
        if action == "overview":
            return {"status": "ok", "overview": session.memory.memory_overview()}
        if action == "graph":
            # O(n^2) cosine pass over the (bounded) record set — off the loop.
            graph = await _run_blocking(session.memory.memory_graph)
            return {"status": "ok", "graph": graph}
        raise HTTPException(
            status_code=400, detail=f"Unknown memory action: {payload.action}"
        )

    return router
