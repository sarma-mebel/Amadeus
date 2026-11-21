"""Chat routes: POST /api/chat, WS /api/chat/stream, /api/tts, /api/reset."""

from __future__ import annotations

import asyncio
import base64
import re
import threading
from typing import Callable, Dict, Optional

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect

from .. import telemetry
from ..config import Config
from ..logging_utils import get_logger
from ..providers.llm.fallback import _is_rate_limit
from .models import ChatRequest, ChatResponse, ResetRequest, TTSRequest
from .ratelimit import client_key
from .sessions import ProviderCallError, SessionStore, _synthesize_tts
from .streaming import VALID_EMOTIONS
from .streaming.duplex import DuplexSocketRunner

LOGGER = get_logger(__name__)

# Last-resort textual hints for a rate-limit failure, used only when the
# structured checks below can't see the original exception (e.g. it was
# stringified into a ProviderCallError message).
_RATE_LIMIT_HINTS = (
    "429",
    "rate limit",
    "rate-limit",
    "ratelimit",
    "quota",
    "resource exhausted",
    "resource_exhausted",
)


def _provider_error_kind(exc: BaseException) -> str:
    """Classify a turn-failing exception for the client.

    Returns ``"rate_limited"`` for 429/quota-style provider errors and
    ``"provider_error"`` for everything else, so the frontend can show an
    honest caption instead of a misleading "(connection lost)" — the
    connection is fine, the upstream model refused. Structured SDK checks
    (via the failover provider's classifier) walk the cause chain first;
    matching English error text is only the fallback.
    """
    seen: set = set()
    node: Optional[BaseException] = exc
    while node is not None and id(node) not in seen:
        seen.add(id(node))
        if _is_rate_limit(node):
            return "rate_limited"
        node = node.__cause__ or node.__context__
    text = str(exc).lower()
    if any(hint in text for hint in _RATE_LIMIT_HINTS):
        return "rate_limited"
    return "provider_error"


def build_router(
    store: SessionStore,
    config: Config,
    *,
    api_token: Optional[str],
    token_ok: Callable[[Optional[str]], bool],
) -> APIRouter:
    router = APIRouter()

    @router.post("/api/chat", response_model=ChatResponse)
    async def chat_endpoint(payload: ChatRequest) -> ChatResponse:
        return await handle_chat(payload)

    @router.post("/api/chat/mobile", response_model=ChatResponse)
    async def mobile_chat(payload: ChatRequest, model: str = "gpt-6.2-astra") -> ChatResponse:
        if not api_token or config.web.demo_mode:
            raise HTTPException(403, "Model selection requires a private, token-protected server")
        if not re.fullmatch(r"(?:gpt-[0-9]|o[1-9])[a-z0-9.-]{0,72}", model):
            raise HTTPException(422, "Invalid OpenAI chat model ID")
        return await handle_chat(payload, model)

    async def handle_chat(payload: ChatRequest, model: Optional[str] = None) -> ChatResponse:
        session_id, session = store.get(payload.session_id, payload.user_id)
        try:
            if model:
                response = await session.handle(payload, openai_model=model)
            else:
                response = await session.handle(payload)
        except ProviderCallError as exc:
            # Rate limits, quota, and network blips shouldn't surface as a
            # bare 500 with a stack trace. 502: the upstream provider failed.
            raise HTTPException(
                status_code=502,
                detail={
                    "error": f"Language model call failed: {exc}",
                    # Additive: lets clients distinguish "rate_limited" from
                    # a generic "provider_error" without parsing the message.
                    "kind": _provider_error_kind(exc),
                },
            ) from exc
        response.session_id = session_id
        return response

    # --- Standalone TTS (deferred-audio flow) ------------------------------

    _TTS_MAX_CHARS = 500

    @router.post("/api/tts")
    async def tts_endpoint(payload: TTSRequest) -> Dict[str, str]:
        """Synthesize one clip of (Japanese) speech outside a chat turn.

        Pairs with ChatRequest.defer_audio: the client shows the reply text
        immediately and fetches audio here afterwards.
        """
        text = payload.text.strip()
        if not text:
            raise HTTPException(status_code=400, detail="text must not be empty")
        # sanitize_speech_for_clone + normalize_ja_for_speech now run inside
        # _synthesize_tts (single choke point for all three TTS entry
        # points), so this endpoint no longer applies them itself — the
        # _TTS_MAX_CHARS check below is therefore against the raw payload
        # text rather than the sanitized text (reviewed/accepted: sanitizing
        # only ever shortens/replaces content, never lengthens it past the
        # cap it would have passed before).
        if len(text) > _TTS_MAX_CHARS:
            raise HTTPException(
                status_code=413,
                detail=f"text exceeds the {_TTS_MAX_CHARS}-character limit",
            )
        emotion = (payload.emotion or "neutral").lower()
        if emotion not in VALID_EMOTIONS:
            raise HTTPException(
                status_code=400,
                detail=f"emotion must be one of {sorted(VALID_EMOTIONS)}",
            )

        with store.lock:
            tts = store.tts
            warming = store.tts_warming
        if tts is None or not config.audio.enable_voice:
            raise HTTPException(
                status_code=503,
                detail={
                    "error": (
                        "TTS is still warming up."
                        if warming
                        else "TTS is unavailable on this server."
                    )
                },
            )

        try:
            audio_bytes = await _synthesize_tts(
                tts,
                text,
                speaking_rate=config.persona.speaking_rate,
                pitch_shift=config.persona.pitch_shift,
                # _synthesize_tts drops labels the provider has no reference
                # clip for, so this is a no-op everywhere but emotion-ref'd
                # voices (the deferred client already sends the sentence's
                # emotion here).
                emotion=emotion,
            )
        except Exception as exc:  # pragma: no cover - runtime TTS errors
            LOGGER.error("Standalone TTS synthesis failed: %s", exc)
            telemetry.incr("tts.deferred_synth_failures")
            raise HTTPException(
                status_code=503, detail={"error": f"Synthesis failed: {exc}"}
            ) from exc

        return {
            "mime": "audio/wav",
            "data": base64.b64encode(audio_bytes).decode("ascii"),
        }

    @router.websocket("/api/chat/stream")
    async def chat_stream_endpoint(websocket: WebSocket) -> None:
        await websocket.accept()
        if api_token is not None and not token_ok(websocket.query_params.get("token")):
            # 1008 = policy violation; the token rides the query string
            # because browsers can't set headers on WebSocket connects.
            await websocket.close(code=1008, reason="Missing or invalid token")
            return
        if not config.web.streaming_enabled:
            await websocket.send_json(
                {
                    "type": "error",
                    "message": "Streaming is disabled on this server.",
                    "fallback_to_post": True,
                }
            )
            await websocket.close()
            return

        try:
            raw = await websocket.receive_json()
        except Exception:
            await websocket.close()
            return
        if raw.get("type") != "chat":
            await websocket.send_json(
                {
                    "type": "error",
                    "message": "First frame must be {type: 'chat', ...}",
                    "fallback_to_post": True,
                }
            )
            await websocket.close()
            return

        payload = dict(raw)
        payload.pop("type", None)
        try:
            request = ChatRequest.model_validate(payload)
        except Exception as exc:
            await websocket.send_json(
                {
                    "type": "error",
                    "message": f"Invalid chat payload: {exc}",
                    "fallback_to_post": True,
                }
            )
            await websocket.close()
            return

        # WebSockets bypass HTTP middleware, so the shared per-IP turn
        # limiter (app.state.turn_limiter, set in create_app) is checked
        # here — one WS connection carries exactly one chat turn.
        limiter = getattr(websocket.app.state, "turn_limiter", None)
        if limiter is not None:
            key = client_key(
                websocket.headers,
                websocket.client.host if websocket.client else None,
            )
            if not limiter.allow(key):
                await websocket.send_json(
                    {
                        "type": "error",
                        "message": "Rate limit reached — give it a minute, this instance is shared.",
                        # Additive: honest caption hint for the client (this
                        # is a rate limit, not a dropped connection).
                        "kind": "rate_limited",
                    }
                )
                await websocket.close(code=1013)
                return

        session_id, session = store.get(request.session_id, request.user_id)
        await websocket.send_json({"type": "session", "session_id": session_id})

        cancel_event = threading.Event()
        send_lock = asyncio.Lock()

        async def emit(frame: Dict[str, object]) -> None:
            async with send_lock:
                try:
                    await websocket.send_json(frame)
                except Exception:  # pragma: no cover - client already gone
                    # A send failure self-cancels the turn exactly like a
                    # barge-in; count it so a flaky transport masquerading as
                    # "users keep interrupting" is visible in /api/status.
                    if not cancel_event.is_set():
                        telemetry.incr("web.ws_send_error_cancels")
                    cancel_event.set()

        async def recv_loop() -> None:
            try:
                while True:
                    msg = await websocket.receive_json()
                    if msg.get("type") == "cancel":
                        cancel_event.set()
                        return
            except (WebSocketDisconnect, RuntimeError):
                cancel_event.set()
            except Exception:  # pragma: no cover - malformed client frame
                cancel_event.set()

        turn_task = asyncio.create_task(
            session.handle_stream(request, emit, cancel_event)
        )
        listener_task = asyncio.create_task(recv_loop())
        try:
            await turn_task
        except Exception as exc:
            LOGGER.error("Streaming turn failed: %s", exc)
            # `kind` is additive (old clients ignore it): a provider failure
            # is NOT a connection loss, and the frontend caption should say
            # so. fallback_to_post stays True for wire compatibility, but by
            # this point the session frame has already flowed, so up-to-date
            # clients never re-send via POST (it would hit the same error).
            await emit(
                {
                    "type": "error",
                    "message": str(exc),
                    "kind": _provider_error_kind(exc),
                    "fallback_to_post": True,
                }
            )
        finally:
            listener_task.cancel()
            try:
                await websocket.close()
            except Exception:  # pragma: no cover - already closed
                pass

    @router.websocket("/api/chat/duplex")
    async def chat_duplex_endpoint(websocket: WebSocket) -> None:
        """Full-duplex voice conversation loop (PROTOTYPE, default off).

        Strictly additive: the /api/chat and /api/chat/stream paths above
        are completely untouched by this route. Gated by
        ``web.duplex_enabled`` — with the flag off (the default) the socket
        is accepted only long enough to send one error frame, mirroring the
        streaming_enabled gate on /api/chat/stream. See
        amadeus/web/streaming/duplex.py for the wire protocol, the
        turn-taking state machine, and how cancel-truthfulness is inherited
        from the shared streaming turn orchestrator.
        """
        await websocket.accept()
        if not config.web.duplex_enabled:
            await websocket.send_json(
                {
                    "type": "error",
                    "message": "Full-duplex voice is disabled on this server.",
                    "kind": "duplex_disabled",
                }
            )
            await websocket.close()
            return
        if api_token is not None and not token_ok(websocket.query_params.get("token")):
            await websocket.close(code=1008, reason="Missing or invalid token")
            return

        # Same shared per-IP turn limiter as /api/chat/stream: a duplex
        # session can run many turns, but the first one still has to clear
        # the same bar a POST/stream turn would.
        limiter = getattr(websocket.app.state, "turn_limiter", None)
        if limiter is not None:
            key = client_key(
                websocket.headers,
                websocket.client.host if websocket.client else None,
            )
            if not limiter.allow(key):
                await websocket.send_json(
                    {
                        "type": "error",
                        "message": "Rate limit reached — give it a minute, this instance is shared.",
                        "kind": "rate_limited",
                    }
                )
                await websocket.close(code=1013)
                return

        runner = DuplexSocketRunner(websocket, store, config)
        try:
            await runner.run()
        except (
            Exception
        ) as exc:  # pragma: no cover - defensive; run() degrades loudly internally
            LOGGER.error("Duplex session failed: %s", exc)
        finally:
            try:
                await websocket.close()
            except Exception:  # pragma: no cover - already closed
                pass

    @router.websocket("/api/wake")
    async def wake_word_endpoint(websocket: WebSocket) -> None:
        """Local wake-word listener (default off; ``wake_word.enabled``).

        Additive like the duplex route: streams mic PCM in, emits one
        ``{"type": "wake"}`` frame per detection, and the CLIENT reacts by
        starting the normal duplex call — this socket never runs a turn
        itself. Gate/token handling mirrors /api/chat/duplex; protocol and
        degrade behavior live in streaming/wake_word.py. No turn limiter:
        no LLM/TTS cost is reachable from here.
        """
        await websocket.accept()
        wake_settings = getattr(config, "wake_word", None)
        if wake_settings is None or not wake_settings.enabled:
            await websocket.send_json(
                {
                    "type": "error",
                    "message": "Wake word is disabled on this server.",
                    "kind": "wake_disabled",
                }
            )
            await websocket.close()
            return
        if api_token is not None and not token_ok(websocket.query_params.get("token")):
            await websocket.close(code=1008, reason="Missing or invalid token")
            return

        from .streaming.wake_word import WakeSocketRunner

        runner = WakeSocketRunner(websocket, config)
        try:
            await runner.run()
        except WebSocketDisconnect:  # pragma: no cover - normal client close
            pass
        except Exception as exc:  # pragma: no cover - defensive
            LOGGER.error("Wake-word session failed: %s", exc)
        finally:
            try:
                await websocket.close()
            except Exception:  # pragma: no cover - already closed
                pass

    @router.post("/api/reset")
    async def reset_endpoint(payload: ResetRequest) -> Dict[str, str]:
        try:
            store.reset(
                payload.session_id,
                forget=payload.forget,
                user_id=payload.user_id,
                full_forget=payload.full_forget,
            )
        except KeyError as exc:  # pragma: no cover - network race
            raise HTTPException(
                status_code=404, detail=f"Unknown session: {payload.session_id}"
            ) from exc
        return {"status": "reset"}

    return router
