"""Web chat session and session-store machinery for the Amadeus web app."""

from __future__ import annotations

import asyncio
import base64
import copy
import dataclasses
import datetime as dt
import functools
import threading
import time
import uuid
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .. import provider_health, telemetry
from ..agents.emotion_analyzer import KurisuEmotionAnalyzer
from ..config import Config, PersonaSettings
from ..logging_utils import get_logger
from ..engine import ChatEngine, TurnSetup, build_summary_provider
from ..memory.manager import EmbeddingProvider, MemoryManager, Message
from ..memory.profile import familiarity_guardedness
from ..persona import apply_style_modifiers
from ..providers.llm.base import DummyProvider
from ..providers.llm.openai_provider import OpenAIProvider
from ..providers.stt.base import STTProvider
from . import streaming
from .blocking import run_blocking as _run_blocking
from .models import ChatRequest, ChatResponse, SentencePayload
from .streaming import (
    build_tagged_instructions,
    get_nonverbal_library,
    normalize_ja_for_speech,
    parse_complete_reply,
    parse_complete_reply_events,
    sanitize_speech_for_clone,
)
from .transcripts import TranscriptStore
from ..providers.llm.base import ChatMessage, ImageContent, LLMProvider
from ..providers.tts.base import TTSProvider

# Import translation for Japanese TTS
try:
    from deep_translator import GoogleTranslator

    TRANSLATOR_AVAILABLE = True
except ImportError:
    TRANSLATOR_AVAILABLE = False


LOGGER = get_logger(__name__)


def _app() -> Any:
    """The ``amadeus.web.app`` module, imported lazily.

    The provider loaders (``load_llm``/``load_tts``/``load_stt``/
    ``load_embedder``) and ``_cached_translate`` are re-exported by
    ``app.py``, and the test suite monkeypatches them *there*
    (``monkeypatch.setattr(app_module, "load_tts", ...)``). Resolving those
    names through the app module at call time keeps the patches effective
    now that this code lives outside ``app.py``. Lazy import also avoids a
    circular import: ``app.py`` imports this module at its top.
    """
    from . import app

    return app


def _warn_if_llm_degraded(config: Config, llm: Any) -> None:
    """Shout when a configured real LLM silently fell back to the dummy.

    The loaders swap in DummyProvider when a provider or its API key is
    missing, so a broken deployment still "works" — on canned replies. A
    single loud banner at startup turns that from a confusing quality
    regression into an obvious config error. (/api/status and the frontend
    status chip report the same truth per-request.)
    """
    if config.providers.llm != "dummy" and isinstance(llm, DummyProvider):
        LOGGER.warning(
            "\n%s\n"
            "DEGRADED: providers.llm='%s' could not be loaded — replies come\n"
            "from the canned DUMMY provider. Check the API key / extras\n"
            "(pip install '.[%s]'), then restart.\n%s",
            "=" * 66,
            config.providers.llm,
            config.providers.llm,
            "=" * 66,
        )


class ProviderCallError(RuntimeError):
    """An upstream LLM/tool call failed during a chat turn.

    Distinguishes provider failures (rate limits, quota, network) from bugs
    so the chat endpoint can answer with a structured 502 instead of a raw
    500 stack trace.
    """


async def _synthesize_tts(
    tts: TTSProvider,
    text: str,
    *,
    speaking_rate: float,
    pitch_shift: float,
    emotion: Optional[str] = None,
) -> bytes:
    # Single choke point for audio-side text transforms: the zero-shot-clone
    # off-timbre guard, then Japanese number/date normalization (both CJK-
    # gated no-ops on English text). Every caller — POST-inline full reply,
    # the deferred /api/tts endpoint, and routes_meta's TTS re-synthesis —
    # goes through here, so this used to be the one path (POST-inline) that
    # skipped sanitize entirely; it's covered now too.
    #
    # Also the single choke point for tts_health (provider_health.py):
    # instrumenting here means all three callers self-heal/degrade together
    # instead of needing the same try/record pair copy-pasted three times.
    # The streamed-turn path (TTSPipeline) doesn't route through this
    # function — it's instrumented directly in tts_pipeline.py instead.
    text = sanitize_speech_for_clone(text)
    text = normalize_ja_for_speech(text)
    kwargs: Dict[str, Any] = {
        "speaking_rate": speaking_rate,
        "pitch_shift": pitch_shift,
    }
    # Only providers advertising the label ever receive the kwarg — the
    # rest keep the exact pre-emotion call signature.
    if emotion and emotion in (getattr(tts, "available_emotions", None) or ()):
        kwargs["emotion"] = emotion
    synthesize_async = getattr(tts, "synthesize_async", None)
    try:
        if synthesize_async is not None:
            audio_bytes = await synthesize_async(text, **kwargs)
        else:
            audio_bytes = await _run_blocking(tts.synthesize, text, **kwargs)
    except Exception as exc:
        provider_health.record_tts_failure(exc)
        raise
    provider_health.record_tts_success()
    return audio_bytes


class _SpeculativeSTT:
    """Transparent STT wrapper: after a successful transcription, pre-start
    the long-term recall lookup for the transcript
    (``memory.speculative_recall``) so its embed round trip overlaps the
    client's transcript->chat round trip instead of sitting inside the turn.
    Also runs vocal-emotion analysis (``vocal_emotion.enabled``) on the same
    audio bytes, stashed for the next chat turn (``amadeus/vocal_emotion.py``
    / ``amadeus/vocal_tone_stash.py``) — cheap, and never allowed to slow
    down or break transcription.

    Both the plain ``/api/stt`` endpoint and the duplex loop
    (``web/streaming/duplex.py``) reach STT exclusively through
    ``SessionStore.get_stt()``, so wiring both hooks here covers both paths
    with no duplex-specific code.

    Everything else delegates to the wrapped provider unchanged; with both
    flags off (or no live session to target) transcription behavior is
    byte-identical.
    """

    def __init__(self, provider: STTProvider, store: "SessionStore") -> None:
        self.provider = provider
        self._store = store

    def __getattr__(self, name: str) -> Any:
        return getattr(self.provider, name)

    def transcribe_audio(self, audio_data: bytes, sample_rate: int) -> str:
        text = self.provider.transcribe_audio(audio_data, sample_rate)
        try:
            self._store.speculate_recall(text)
        except Exception:  # pragma: no cover - speculation must never break STT
            # Losing speculation only means the turn runs the normal serial
            # recall — but count it (degrade loudly), or a broken hook would
            # read as "feature working, recall just never overlaps".
            telemetry.incr("memory.speculative_recall_start_failures")
            LOGGER.warning("Speculative recall start failed", exc_info=True)
        try:
            self._store.analyze_vocal_tone(audio_data, sample_rate)
        except Exception:  # pragma: no cover - analysis must never break STT
            telemetry.incr("vocal_emotion.analyze_failures")
            LOGGER.warning("Vocal emotion analysis hook failed", exc_info=True)
        return text


class WebChatSession(ChatEngine):
    """Stateful chat session scoped to a browser client.

    Turn orchestration (memory, persona prompt, guardrails, tools) lives in
    :class:`amadeus.engine.ChatEngine`; this adapter adds the web-only parts:
    TTS/base64 audio, emotion/translation, request-level persona overrides,
    and per-session locking for concurrent HTTP/WS turns.
    """

    # Set by SessionStore when the session is registered; complete_turn uses
    # them to append each finished exchange to the on-disk transcript exactly
    # once (both the POST and WS paths finalize through complete_turn).
    session_id: Optional[str] = None
    transcript_store: Optional[TranscriptStore] = None
    # Per-turn extras (ja/emotion) stashed by _handle just before
    # complete_turn; the WS path doesn't stash, so it records nulls.
    _pending_turn_meta: Optional[Dict[str, Any]] = None
    # Server-generated id of the most recently completed turn; drives
    # rewind_to_turn/amend_last_assistant so the UI can target "the last
    # exchange" without racing a concurrent turn on the same session.
    last_turn_id: Optional[str] = None

    def __init__(
        self,
        config: Config,
        llm_provider: LLMProvider,
        embedder: Optional[EmbeddingProvider],
        tts_provider: Optional[TTSProvider],
        emotion_analyzer: KurisuEmotionAnalyzer,
        translator: Optional["GoogleTranslator"],
        *,
        user_id: Optional[str] = None,
        persist_path: Optional[Path] = None,
        embeddings_path: Optional[Path] = None,
        profile_path: Optional[Path] = None,
    ) -> None:
        self.persist_path = persist_path
        self.embeddings_path = embeddings_path
        self.profile_path = profile_path
        super().__init__(
            config,
            llm_provider,
            embedder=embedder,
            persist_path=persist_path,
            embeddings_path=embeddings_path,
            profile_path=profile_path,
        )
        self.tts = tts_provider
        self.emotion_analyzer = emotion_analyzer
        self.translator = translator
        self.user_id = user_id
        self.audio_enabled = True
        # Touched on every SessionStore.get(); drives idle eviction.
        self.last_activity = time.monotonic()
        # Old managers finishing a session-end episode job; kept alive so GC
        # can't shut their executor down mid-flight.
        self._retiring_memories: List[MemoryManager] = []
        # Serializes handle() so concurrent requests for the same session cannot
        # interleave memory mutations or race on the persist file.
        self.handle_lock = asyncio.Lock()
        # Provider swap staged by SessionStore.reload_providers while a turn
        # was mid-flight; applied at the next turn boundary under handle_lock.
        self._pending_rebind: Optional[tuple] = None

    def rebind_providers(
        self,
        config: Config,
        llm: LLMProvider,
        embedder: Optional[EmbeddingProvider],
        tts: Optional[TTSProvider],
        memory_paths: Optional[tuple] = None,
        transcript_store: Optional[TranscriptStore] = None,
    ) -> None:
        """Swap this session onto freshly loaded providers.

        Called from the event loop (settings save). If a turn currently holds
        ``handle_lock`` its blocking work may be touching ``self.memory`` from
        an executor thread, so the swap is staged and applied at the next turn
        boundary instead of yanking providers out from under it. With the lock
        free the swap applies immediately (the check and the apply share one
        event-loop tick, so no turn can slip in between).

        ``memory_paths`` is a ``(persist, embeddings, profile)`` tuple passed
        by the store when a character switch moved this user's memory store
        location (per-character isolation); None means "memory unchanged".
        ``transcript_store`` is the store's rebuilt character-scoped
        TranscriptStore under the same circumstances; None means "transcripts
        unchanged".
        """
        self._pending_rebind = (
            config,
            llm,
            embedder,
            tts,
            memory_paths,
            transcript_store,
        )
        if not self.handle_lock.locked():
            self._apply_pending_rebind()

    def _apply_pending_rebind(self) -> None:
        pending = self._pending_rebind
        if pending is None:
            return
        self._pending_rebind = None
        config, llm, embedder, tts, memory_paths, transcript_store = pending
        self.config = config
        self.llm = llm
        self.embedder = embedder
        self.tts = tts
        self.summary_provider = build_summary_provider(config, llm)
        transcripts_swapped = (
            transcript_store is not None
            and transcript_store is not self.transcript_store
        )
        if transcripts_swapped:
            # Character switch moved the durable transcripts: future turns
            # append to the new character's file. last_turn_id belonged to
            # the previous character's transcript — clearing it makes
            # rollback/amend re-anchor on the new store's actual last turn
            # (their existing None-rehydrate path) instead of dangling.
            self.transcript_store = transcript_store
            self.last_turn_id = None
        if memory_paths is not None and tuple(memory_paths) != (
            self.persist_path,
            self.embeddings_path,
            self.profile_path,
        ):
            # Character switch: this session's memory store location moved.
            # The OLD manager is retired untouched — reset(forget=False)
            # finalizes it in the background and its async episode/summary
            # jobs keep writing through its own baked-in paths/providers, so
            # nothing from the previous character can land in the new store.
            # The fresh manager (built from the paths set here + the new
            # config/providers assigned above) loads the NEW character's
            # stores — empty on a first switch — so the previous character's
            # rolling short-term context never bleeds across.
            self.persist_path, self.embeddings_path, self.profile_path = tuple(
                memory_paths
            )
            audio_enabled = self.audio_enabled
            self.reset(forget=False)
            # reset() defaults audio back on; a provider rebind must not
            # silently unmute a muted session.
            self.audio_enabled = audio_enabled
        else:
            self.memory.set_summary_provider(self.summary_provider)
            self.memory.set_embedder(embedder)
        if transcripts_swapped:
            # With scope_context_to_session on, the fresh manager left
            # short_term empty; if the NEW character's transcript already
            # has this session id (switching back to a character mid-
            # conversation), restore that character's own tail — never the
            # previous character's. A no-op when the flag is off, short_term
            # is populated, or the session id is new to this store.
            self.hydrate_short_term_from_transcript()

    def _build_persona(self, request: ChatRequest) -> PersonaSettings:
        persona = replace(self.config.persona)
        if request.safe_mode is not None:
            persona.safe_mode = request.safe_mode
        if request.tsundere_level is not None:
            persona.tsundere_level = request.tsundere_level
        if request.scientific_formality is not None:
            persona.scientific_formality = request.scientific_formality
        # Familiarity warms Kurisu up over sessions; the request-level
        # tsundere_level still sets the baseline.
        if persona.emotional_guardedness is None and self.memory.profile is not None:
            persona.emotional_guardedness = familiarity_guardedness(
                persona.tsundere_level, self.memory.profile.session_count
            )
        return persona

    def complete_turn(self, user_text: str, reply: str) -> None:
        """Engine commit + one transcript append (shared by POST and WS)."""
        super().complete_turn(user_text, reply)
        meta = self._pending_turn_meta or {}
        self._pending_turn_meta = None
        turn_id = uuid.uuid4().hex
        # Set last_turn_id even if the transcript write below fails: the
        # in-memory ops (short_term/summary) already committed above via
        # super().complete_turn(), so this turn is real regardless of disk.
        # amend_last_assistant treats the transcript as authoritative when
        # one is present, so a failed write here just means a subsequent
        # amend degrades to "conflict" (transcript_store.replace_last_assistant
        # returns None for this turn_id) rather than silently corrupting
        # state — it does not make last_turn_id itself unsafe to set.
        self.last_turn_id = turn_id
        if self.transcript_store is not None and self.session_id:
            try:
                self.transcript_store.record_turn(
                    self.user_id,
                    self.session_id,
                    user_text,
                    reply,
                    ja=meta.get("ja"),
                    emotion=meta.get("emotion"),
                    turn_id=turn_id,
                )
            except Exception as exc:  # pragma: no cover - disk failure
                LOGGER.warning("Failed to persist transcript turn: %s", exc)

    def rewind_to_turn(self, turn_id: str) -> Dict[str, Any]:
        """Truncate this session's history to *before* ``turn_id``.

        Blocking (callers hold ``handle_lock`` and dispatch this through
        ``_run_blocking``): it touches the transcript file and rebuilds
        in-memory short-term state, mirroring what a slow disk write or a
        big deque rebuild would cost on the event loop.

        Council "Council summary:" system lines living in short_term have no
        counterpart in the durable transcript (it only ever records
        user/assistant turns) — ``_rebuild_short_term`` below pops them
        alongside the matched messages rather than trying to replay them.
        Trust/arc/mood are session-boundary state (advanced once per session
        start, not per turn) and need no rewind of their own.
        """
        # Barrier any in-flight async summarize/extract job before mutating
        # short_term/summary state out from under it.
        self.memory.flush()

        if self.transcript_store is not None and self.session_id:
            if self.last_turn_id is None:
                # Fresh in-memory session over an existing transcript (process
                # restart, TTL eviction) never had a turn complete in this
                # process — rehydrate from disk so the validation below (and
                # the no-match fallback path) has something to compare against.
                self.last_turn_id = self.transcript_store.last_turn_info(
                    self.user_id, self.session_id
                )
            truncated = self.transcript_store.truncate_from_turn(
                self.user_id, self.session_id, turn_id
            )
            if truncated is None:
                if turn_id == self.last_turn_id:
                    # Known turn, but no (or no longer any) transcript record
                    # of it — fall back to the no-store path below. A failed
                    # verification there (tail doesn't actually contain the
                    # exchange) means the turn can't be found, same as above.
                    remaining = self._rewind_fallback_no_store()
                    if remaining is None:
                        return {"status": "not_found"}
                else:
                    return {"status": "not_found"}
            else:
                remaining, removed = truncated
                self._rebuild_short_term(remaining, removed)
        else:
            if turn_id != self.last_turn_id:
                return {"status": "not_found"}
            remaining = self._rewind_fallback_no_store()
            if remaining is None:
                return {"status": "not_found"}

        self._reset_summary_state()
        self.memory._persist()
        return {"status": "ok", "messages": remaining}

    def _rewind_fallback_no_store(self) -> Optional[List[Dict[str, Any]]]:
        """No-transcript-store rewind: only the last completed turn is
        addressable (there's no durable record to find an arbitrary turn
        id in — callers must already have verified ``turn_id ==
        self.last_turn_id`` before calling this). Pops the trailing system
        lines (council summary), then the assistant, then the user
        short_term message — but only after verifying the tail actually
        looks like a completed exchange. Never mutates state (including
        ``last_turn_id``) when nothing pops.
        """
        short_term = self.memory.short_term
        # Work on a snapshot index so a verification failure leaves the
        # deque completely untouched.
        messages = list(short_term)
        i = len(messages)
        while i > 0 and messages[i - 1].role == "system":
            i -= 1
        if i == 0 or messages[i - 1].role != "assistant":
            return None
        i -= 1
        if i == 0 or messages[i - 1].role != "user":
            return None
        user_index = i - 1
        # Verified: pop everything from user_index onward.
        for _ in range(len(messages) - user_index):
            short_term.pop()
        self.last_turn_id = None
        return []

    def _rebuild_short_term(
        self,
        remaining: List[Dict[str, Any]],
        removed: List[Dict[str, Any]],
    ) -> None:
        """Pop only the short_term entries that correspond to ``removed``
        (the durable transcript messages truncated by this rewind), leaving
        everything earlier — including cross-session carryover that
        ``MemoryManager._load`` restored — untouched.

        Right-aligns by comparing role+content against the short_term tail,
        walking backward from the end of ``removed``. Council "Council
        summary:" system lines have no transcript counterpart, so any system
        message sitting between/after matched entries is popped too — its
        turn is gone regardless. Falls back to popping until the short_term
        tail matches the new last remaining transcript message (or short_term
        is empty) if the tail doesn't line up cleanly.
        """
        short_term = self.memory.short_term
        removed_msgs = [m for m in removed if m.get("role") in ("user", "assistant")]

        def _matches(entry, removed_msg) -> bool:
            return entry.role == removed_msg.get(
                "role"
            ) and entry.content == removed_msg.get("content", "")

        popped_ok = True
        cursor = list(removed_msgs)
        while cursor and short_term:
            target = cursor[-1]
            # Drop any trailing system lines (council summaries) first —
            # they ride along with whichever turn precedes them and have no
            # transcript record of their own.
            while short_term and short_term[-1].role == "system":
                short_term.pop()
            if not short_term:
                popped_ok = False
                break
            if _matches(short_term[-1], target):
                short_term.pop()
                cursor.pop()
            else:
                popped_ok = False
                break

        if not popped_ok or cursor:
            # Tail didn't line up cleanly (e.g. short_term was truncated by
            # maxlen and no longer holds the full removed run). Fall back:
            # pop from the end until the short_term tail matches the last
            # remaining transcript message, or short_term is empty.
            remaining_msgs = [
                m for m in remaining if m.get("role") in ("user", "assistant")
            ]
            target_tail = remaining_msgs[-1] if remaining_msgs else None
            while short_term:
                if short_term[-1].role == "system":
                    short_term.pop()
                    continue
                if target_tail is not None and _matches(short_term[-1], target_tail):
                    break
                short_term.pop()

        self.last_turn_id = None
        for message in reversed(remaining):
            if message.get("role") == "assistant":
                self.last_turn_id = message.get("turn_id")
                break

        # Re-stamp the last remaining short_term message's timestamp to now:
        # otherwise _maybe_finalize_gap would read the rewound-to turn's
        # (possibly long-past) age as a session gap on the very next message
        # and spuriously start a new session. The durable transcript keeps
        # the real timestamps — only the in-memory copy is touched.
        if short_term:
            short_term[-1].timestamp = dt.datetime.utcnow().isoformat()

    def hydrate_short_term_from_transcript(self) -> None:
        """Rehydrate an empty short_term from THIS session's own transcript.

        Only meaningful when ``memory.scope_context_to_session`` is on:
        MemoryManager._load then deliberately skips restoring short_term/
        summary (they're conversation state, not user state), so a session
        recreated over an existing session id after eviction/process-restart
        would otherwise start with no context even though the conversation
        itself is still going. This restores the tail of *that specific*
        session's durable transcript — never another conversation's — so
        scoping-to-session and mid-conversation resilience both hold at once.

        A brand-new session id has no transcript (``get_transcript`` returns
        None) and stays empty, which is the point of the feature.
        """
        if not getattr(self.config.memory, "scope_context_to_session", False):
            return
        if self.memory.short_term:
            return  # already has state (flag off, or a live process handoff)
        if self.transcript_store is None or not self.session_id:
            return
        transcript = self.transcript_store.get_transcript(self.user_id, self.session_id)
        if not transcript:
            return
        maxlen = self.memory.short_term.maxlen or len(transcript)
        tail = transcript[-maxlen:]
        for entry in tail:
            role = entry.get("role") or "assistant"
            content = entry.get("content", "")
            timestamp = entry.get("timestamp")
            iso_timestamp = (
                dt.datetime.utcfromtimestamp(float(timestamp)).isoformat()
                if timestamp
                else dt.datetime.utcnow().isoformat()
            )
            self.memory.short_term.append(
                Message(role=role, content=content, timestamp=iso_timestamp)
            )
        # Re-stamp the final message's timestamp to now — same phantom-gap
        # concern _rebuild_short_term documents above: an old transcript
        # timestamp on the tail would otherwise read as a huge session gap
        # on the very next message and spuriously start a new session.
        if self.memory.short_term:
            self.memory.short_term[-1].timestamp = dt.datetime.utcnow().isoformat()
        # The last assistant message in the tail is the most recent completed
        # turn — rehydrate last_turn_id the same way amend/rewind already do
        # for a fresh in-memory session over an existing transcript.
        for message in reversed(tail):
            if message.get("role") == "assistant":
                self.last_turn_id = message.get("turn_id")
                break

    def _reset_summary_state(self) -> None:
        """Clear the running summary so it's rebuilt lazily from the (now
        truncated) short_term on the next turn's debounce, instead of
        describing dropped messages. Deliberately does not eagerly call
        summarize() here — build_context tolerates summary=None, and an
        eager call would spend an LLM call on a summary that may never be
        read before the next real turn regenerates it anyway.
        """
        self.memory.summary = None
        self.memory.summary_info = {}
        self.memory._messages_since_summary = len(self.memory.short_term)

    def amend_last_assistant(
        self, turn_id: str, content: str, ja: Optional[str] = None
    ) -> Dict[str, Any]:
        """Replace the latest assistant reply's text in place (edit/redo).

        Only ever targets the most recent turn — an amend against a stale
        turn_id (a newer turn has since completed) is rejected as a conflict
        rather than silently rewriting older history.

        The transcript is authoritative when present: idle-chatter turns can
        append an extra assistant message after complete_turn (via
        ``ChatEngine.add_message`` outside the normal turn flow), and council
        turns leave a trailing "Council summary:" system line — so the
        target short_term message is never reliably ``short_term[-1]``. The
        transcript's old content (returned by ``replace_last_assistant``) is
        used to locate the matching message by scanning from the right,
        skipping non-assistant lines.
        """
        if self.transcript_store is not None and self.session_id:
            if self.last_turn_id is None:
                # Fresh in-memory session over an existing transcript —
                # rehydrate before validating (see rewind_to_turn).
                self.last_turn_id = self.transcript_store.last_turn_info(
                    self.user_id, self.session_id
                )
            old_content = self.transcript_store.replace_last_assistant(
                self.user_id, self.session_id, turn_id, content, ja=ja
            )
            if old_content is None:
                return {"status": "conflict"}
            # Scan from the right for the last assistant-role message whose
            # content equals the transcript's old content, skipping
            # system/other roles (idle-chatter assistant lines appended
            # after complete_turn, or a trailing council summary line) along
            # the way. If it's fallen out of the deque (maxlen eviction),
            # that's fine — the transcript already has the authoritative edit.
            for message in reversed(self.memory.short_term):
                if message.role == "assistant" and message.content == old_content:
                    message.content = content
                    break
            self.memory._persist()
            return {"status": "ok"}

        # No transcript store: last_turn_id is the only signal available.
        if turn_id != self.last_turn_id:
            return {"status": "conflict"}
        for message in reversed(self.memory.short_term):
            if message.role == "system":
                continue
            if message.role == "assistant":
                message.content = content
            break
        self.memory._persist()
        return {"status": "ok"}

    async def handle(self, request: ChatRequest, *, openai_model: Optional[str] = None) -> ChatResponse:
        async with self.handle_lock:
            self._apply_pending_rebind()
            if openai_model:
                configured = getattr(self.llm, "primary", self.llm)
                if not isinstance(configured, OpenAIProvider):
                    raise ProviderCallError("This model requires a configured OpenAI provider.")
                provider = copy.copy(configured)
                provider.model = openai_model
                return await self._handle(request, turn_llm=provider)
            return await self._handle(request)

    async def handle_stream(
        self,
        request: ChatRequest,
        emit: Callable[[Dict[str, Any]], Any],
        cancel_event: threading.Event,
    ) -> None:
        """WebSocket counterpart to :meth:`handle`. Acquires the same lock so
        a WS turn and a POST turn on the same session can never interleave
        memory writes.
        """
        async with self.handle_lock:
            self._apply_pending_rebind()
            await streaming.run_streaming_turn(self, request, emit, cancel_event)

    # Prompt guidance for image turns, appended as a system message AFTER
    # prepare_turn so it never lands in memory (context is rebuilt from
    # memory text next turn — same reason the image bytes can't persist).
    VISION_TURN_NOTE = (
        "The user attached a screenshot/image to this message. Examine it "
        "carefully and react in character — reference specific on-screen "
        "details, tease or analyze as {name} would. Keep the required "
        "tagged output format."
    )

    def _vision_ok(self) -> bool:
        vision_cfg = getattr(self.config, "vision", None)
        if not vision_cfg or not getattr(vision_cfg, "enabled", False):
            return False
        try:
            return bool(self.llm.supports_vision())
        except Exception:  # pragma: no cover - defensive
            return False

    def _prepare_turn(self, request: ChatRequest) -> TurnSetup:
        """Web-flavored prepare: request-level persona overrides + tagged output.

        Vision turns: the image is attached to the turn context AFTER
        prepare_turn (which commits the user message to memory) — memory,
        transcripts, and next-turn contexts only ever see the text marker,
        so image bytes are structurally never persisted or re-sent.
        """
        persona = self._build_persona(request)
        LOGGER.info(
            "Processing web request with persona safe_mode=%s", persona.safe_mode
        )
        # Apply the client's audio preference here, on the path BOTH the POST
        # and WS turns funnel through. _handle also applies it (harmless
        # double-set), but the WS path previously never did — so a session
        # whose audio_enabled had been set False by an earlier POST could
        # never be unmuted over the default streaming path, and vice versa.
        if request.audio_enabled is not None:
            self.audio_enabled = request.audio_enabled
        use_images = bool(request.images) and self._vision_ok()
        if use_images:
            base = request.message.strip() or "(shared a screenshot)"
            # One mutation covers memory commit, transcript, and rollback —
            # they all key off request.message.
            request.message = f"{base}\n[user shared a screenshot]"
        # Advertise non-verbal tags only when a clip could actually play
        # this turn: voice on, and a generated clip library present. A model
        # told about [laugh] with no clips would emit silent dead lines.
        nonverbal_kinds = None
        if (
            self.tts is not None
            and self.audio_enabled
            and bool(getattr(self.config.web, "streaming_nonverbals", True))
        ):
            library = get_nonverbal_library()
            if library:
                nonverbal_kinds = library.kinds
        setup = self.prepare_turn(
            request.message,
            persona,
            tagged_output=bool(
                getattr(self.config.web, "streaming_tagged_output", True)
            ),
            tagged_instructions=build_tagged_instructions(
                bool(getattr(self.config.web, "streaming_inner_thoughts", False)),
                nonverbal_kinds=nonverbal_kinds,
                char_name=getattr(getattr(self.config, "character", None), "name", None)
                or "Kurisu",
            ),
        )
        if use_images:
            attachments = [
                ImageContent(mime=image.mime, data=image.data)
                for image in request.images or []
            ]
            for message in reversed(setup.context):
                if message.role == "user":
                    message.images = attachments
                    break
            _char = getattr(self.config, "character", None)
            _char_name = (getattr(_char, "name", None) or "Kurisu").strip() or "Kurisu"
            setup.context.append(
                ChatMessage(
                    role="system",
                    content=self.VISION_TURN_NOTE.format(name=_char_name),
                )
            )
        return setup

    async def _handle(self, request: ChatRequest, *, turn_llm: Optional[LLMProvider] = None) -> ChatResponse:
        provider = self.llm if turn_llm is None else turn_llm
        turn_started = time.monotonic()
        # Off the event loop: recall runs an embedding + vector scan (same
        # reason the streaming path wraps _prepare_turn in _run_blocking).
        setup = await _run_blocking(self._prepare_turn, request)
        persona = setup.persona
        context = setup.context
        recall_items = setup.recall_items
        refusal = setup.refusal
        use_tagged_output = setup.use_tagged_output

        llm_ms: Optional[int] = None
        tts_ms: Optional[int] = None
        if refusal:
            reply = refusal
        else:
            llm_started = time.monotonic()
            # Raising before complete_turn() below means a failed provider
            # call never commits the turn to memory — the user can simply
            # retry.
            try:
                if (
                    not request.images
                    and self._tools_active()
                    and self._should_use_tools(request.message)
                ):
                    # Tool-augmented turn: run the (multi-step, blocking) ReAct loop
                    # off the event loop; it returns the final text like generate().
                    tool_loop = self._run_tool_loop if turn_llm is None else functools.partial(self._run_tool_loop, llm=provider)
                    reply = await _run_blocking(tool_loop, context)
                    if use_tagged_output and reply and not parse_complete_reply(reply):
                        # Native function-calling rounds tend to drop the tagged
                        # grammar (markdown lists, no Japanese) — one repair pass
                        # restores emotion + speech text.
                        reply = await _run_blocking(
                            streaming.ensure_tagged_reply,
                            provider,
                            setup.system_prompt,
                            reply,
                        )
                else:
                    # Blocking provider call — keep it off the event loop.
                    # frequency_penalty breaks the model's opener ruts; only
                    # the chat reply gets it (summaries/JSON must not).
                    reply = await _run_blocking(
                        functools.partial(
                            provider.generate,
                            frequency_penalty=self.config.llm_frequency_penalty,
                        ),
                        context,
                    )
                    # Only the plain-generate branch records here — tool
                    # turns record their own calls (tokens.tools.*) inside
                    # the loop, and the repair pass records tokens.repair.*.
                    telemetry.record_llm_usage(provider, "chat")
                    provider_health.record_llm_success(provider)
            except Exception as exc:
                LOGGER.exception("Provider call failed for this turn")
                telemetry.incr("web.provider_failures")
                # Health classification only — the reply text/status-code
                # behavior here is unchanged (pinned by
                # test_chat_endpoint_provider_failure_returns_502): a plain
                # (non-tool) turn still surfaces as a 502 with a `kind`
                # field, it does not get the in-character RATE_LIMITED_REPLY/
                # INVALID_KEY_REPLY substitution — that only happens inside
                # the tool loop (engine.py::_run_tool_loop), which already
                # returns those as a normal successful reply above.
                provider_health.classify_and_record(exc)
                self.rollback_user_message(request.message)
                raise ProviderCallError(str(exc)) from exc
            llm_ms = int(1000 * (time.monotonic() - llm_started))

        # Try the tagged "[emotion] English || Japanese" grammar first: it
        # replaces the VADER pass and the deep-translator network hop with
        # data the model already produced. Falls back automatically (empty
        # list) when the model didn't follow the format, e.g. small local
        # models, or when the flag is off.
        tagged_sentences: list = []
        thought: Optional[str] = None
        if use_tagged_output:
            tagged_sentences, tagged_thoughts = parse_complete_reply_events(reply)
            if tagged_thoughts and bool(
                getattr(self.config.web, "streaming_inner_thoughts", False)
            ):
                # Display-only; excluded from memory/transcript/TTS because
                # display_reply_raw is rebuilt from sentences below.
                thought = tagged_thoughts[0].text or None
        if tagged_sentences:
            display_reply_raw = " ".join(s.en for s in tagged_sentences if s.en)
            japanese_text = (
                " ".join(s.ja for s in tagged_sentences if s.ja) or display_reply_raw
            )
            # Dominant emotion: most frequent non-neutral tag, else neutral.
            counts: Dict[str, int] = {}
            for s in tagged_sentences:
                if s.emotion != "neutral":
                    counts[s.emotion] = counts.get(s.emotion, 0) + 1
            emotion = max(counts, key=counts.get) if counts else "neutral"
            emotion_intensity = 0.8
        else:
            # Half-followed grammar: strip [thought]/non-verbal lines (a
            # thoughts-only reply parses as tagged with ZERO sentences and
            # lands here — the literal "[thought] ..." text must not reach
            # display, TTS, or memory), then '|| japanese' tails so the
            # display never carries the raw wire format either.
            display_reply_raw = streaming.strip_bilingual_tails(
                streaming.strip_special_tag_lines(reply)
            )
            japanese_text = None  # translate lazily below only if TTS needed
            emotion_data = self.emotion_analyzer.analyze(reply)
            emotion = emotion_data["vrm_expression"]
            emotion_intensity = emotion_data.get("intensity", 0.8)

        committed = bool(display_reply_raw.strip())
        if committed:
            # Rich transcript fields for this exchange (POST path only; the
            # WS path finalizes through the same complete_turn but records
            # nulls).
            self._pending_turn_meta = {"ja": japanese_text, "emotion": emotion}
            self.complete_turn(request.message, display_reply_raw)
        else:
            # Empty reply (provider returned nothing, or only [thought]/
            # non-verbal lines): complete_turn("") skips the assistant
            # append and would strand the user message as a trailing orphan
            # in the persisted store until the next restart's _load healing.
            # Roll it back instead — same contract as the WS path's
            # _finalize_streamed_turn.
            self.rollback_user_message(request.message)
            telemetry.incr("web.empty_reply_rolled_back")

        # Generate audio with Japanese translation
        audio_payload: Optional[str] = None
        audio_mime: Optional[str] = None
        if request.audio_enabled is not None:
            self.audio_enabled = request.audio_enabled

        # Per-sentence payload for the client subtitle + deferred TTS path.
        sentence_payloads: Optional[list[SentencePayload]] = None
        if tagged_sentences:
            sentence_payloads = [
                SentencePayload(
                    en=s.en,
                    ja=s.ja or s.en,
                    emotion=s.emotion,
                    emotion_raw=getattr(s, "emotion_raw", None),
                )
                for s in tagged_sentences
            ]
        elif (
            request.defer_audio
            and self.tts
            and self.config.audio.enable_voice
            and self.audio_enabled
        ):
            # Untagged reply on a deferred-audio turn: the client will fetch
            # audio itself via /api/tts, so hand it speech text the same way
            # the inline branch below would have produced it.
            speech_text = streaming.strip_markdown_for_speech(display_reply_raw)
            if self.translator:
                try:
                    speech_text = await _run_blocking(
                        _app()._cached_translate, self.translator, speech_text
                    )
                except Exception as exc:
                    LOGGER.warning("Translation failed, using English: %s", exc)
            sentence_payloads = [
                SentencePayload(en=display_reply_raw, ja=speech_text, emotion=emotion)
            ]

        if (
            committed  # an empty (rolled-back) turn has nothing to voice
            and self.tts
            and self.config.audio.enable_voice
            and self.audio_enabled
            and not request.defer_audio
        ):
            tts_started = time.monotonic()
            try:
                if japanese_text is None:
                    # No tagged translation available — fall back to the
                    # deep-translator hop as before. Strip markdown first so
                    # TTS never reads asterisks/bullets aloud (display keeps
                    # the original text).
                    japanese_text = streaming.strip_markdown_for_speech(
                        display_reply_raw
                    )
                    if self.translator:
                        try:
                            # Blocking HTTP call — keep it off the event loop.
                            # Cached: this fallback path only runs for
                            # untagged replies, and translation is
                            # deterministic per input text.
                            japanese_text = await _run_blocking(
                                _app()._cached_translate, self.translator, japanese_text
                            )
                            LOGGER.debug("Translated for TTS: %s", japanese_text[:50])
                        except Exception as exc:
                            LOGGER.warning("Translation failed, using English: %s", exc)

                # Voice the whole-reply clip with the dominant emotion the
                # provider has a reference for (None → base voice; mirrors
                # the WS path's per-unit resolution in _TurnAudio).
                tts_emotions = frozenset(
                    getattr(self.tts, "available_emotions", ()) or ()
                )
                tts_emotion = streaming.dominant_emotion(
                    [
                        streaming.resolve_tts_emotion(
                            getattr(s, "emotion_raw", None), s.emotion, tts_emotions
                        )
                        for s in tagged_sentences
                    ]
                )
                # Blocking model inference — keep it off the event loop.
                audio_bytes = await _synthesize_tts(
                    self.tts,
                    japanese_text,  # Use Japanese for voice synthesis
                    speaking_rate=persona.speaking_rate,
                    pitch_shift=persona.pitch_shift,
                    emotion=tts_emotion,
                )
                audio_payload = base64.b64encode(audio_bytes).decode("ascii")
                audio_mime = "audio/wav"
            except Exception as exc:  # pragma: no cover - runtime TTS errors
                LOGGER.error("Voice synthesis failed: %s", exc)
                telemetry.incr("tts.post_synth_failures")
            # Includes the translation hop when it ran — this is the real
            # time-to-audio cost the user experiences.
            tts_ms = int(1000 * (time.monotonic() - tts_started))

        # Stage directions are display-only: memory and TTS keep the clean text.
        display_reply = apply_style_modifiers(display_reply_raw, self.kurisu_state)

        timings = {
            "llm_ms": llm_ms,
            "tts_ms": tts_ms,
            "total_ms": int(1000 * (time.monotonic() - turn_started)),
        }
        LOGGER.info(
            "turn timings: llm=%sms tts=%sms total=%sms",
            llm_ms,
            tts_ms,
            timings["total_ms"],
        )

        return ChatResponse(
            session_id=request.session_id or "",  # Placeholder replaced by store
            reply=display_reply,  # English text for display
            # A rolled-back (empty) turn has no turn of its own — exposing
            # the PREVIOUS turn's id would let a rewind target the wrong one.
            turn_id=self.last_turn_id if committed else None,
            memories=recall_items,
            summary=self.memory.summary,
            summary_strategy=self.memory.summary_info.get("strategy"),
            summary_revision=self.memory.summary_info.get("revision"),
            summary_updated_at=self.memory.summary_info.get("updated_at"),
            emotion=emotion,
            emotion_intensity=emotion_intensity,
            animations=self.pop_pending_animations(),
            audio=audio_payload,  # Japanese audio
            audio_mime_type=audio_mime,
            audio_enabled=self.audio_enabled,
            sentences=sentence_payloads,
            thought=thought,
            timings=timings,
        )

    def reset(self, forget: bool = True, full_forget: bool = False) -> None:
        LOGGER.info(
            "Resetting web chat session (forget=%s, full_forget=%s)",
            forget,
            full_forget,
        )
        if forget:
            # Don't memorialize a conversation the user asked to erase.
            self.memory.clear(full_forget=full_forget)
        else:
            # "New conversation, keep memory": consolidate the finished session
            # in the background and keep the old manager alive until it's done
            # (GC would shut its executor down mid-job otherwise).
            retiring = self.memory
            retiring.finalize_session(wait=False)
            if retiring._summary_executor is not None:
                self._retiring_memories.append(retiring)
                done = retiring._summary_executor.submit(lambda: None)
                done.add_done_callback(
                    lambda _f, m=retiring: (
                        self._retiring_memories.remove(m)
                        if m in self._retiring_memories
                        else None
                    )
                )
        self.audio_enabled = True
        self.memory = MemoryManager(
            self.config,
            embedder=self.embedder,
            persist_path=self.persist_path,
            embeddings_path=self.embeddings_path,
            profile_path=self.profile_path,
            summary_provider=self.summary_provider,
        )
        # The fresh manager reloaded (or, after full_forget, dropped) the
        # profile — carry mood/trust back into the session state to match.
        self._hydrate_state_from_profile()
        self.profile_block = self.memory.profile_block()


class SessionStore:
    """Track active web chat sessions keyed by ID."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.llm = _app().load_llm(config)
        self.embedder: Optional[EmbeddingProvider] = _app().load_embedder(config)
        self.tts: Optional[TTSProvider] = None
        self.tts_warming = False
        # Runtime truth for /api/status: which provider was *requested* vs
        # which one is actually serving. `tts_fallback_from` records that the
        # configured provider failed to load and a substitute is speaking;
        # `tts_error` keeps the last load failure so "unavailable" is
        # diagnosable without grepping logs.
        self.tts_fallback_from: Optional[str] = None
        self.tts_error: Optional[str] = None
        # Set by reload_providers when a settings save arrives while the
        # background warmup thread is mid-load (see the tts_warming branch
        # there): the TTS side of that reload is skipped in the moment
        # rather than raced against the in-flight load, and this flag tells
        # _warm_tts_in_background to re-apply it, against the config current
        # at that time, once warming finishes.
        self.tts_reload_pending = False
        if self.config.audio.enable_voice and config.providers.tts != "qwen3_tts":
            self.tts = self._load_tts_blocking(config)

        # Initialize emotion analyzer for VRM expressions
        self.emotion_analyzer = KurisuEmotionAnalyzer()
        LOGGER.info("Emotion analyzer initialized (VADER-based, <1ms)")

        # Initialize Japanese translator if available
        self.translator = None
        if TRANSLATOR_AVAILABLE:
            try:
                self.translator = GoogleTranslator(source="en", target="ja")
                LOGGER.info("Japanese translator initialized")
            except Exception as exc:
                LOGGER.warning("Translation disabled: %s", exc)

        # STT is loaded lazily on first /api/stt request (model load is slow
        # and voice input may never be used in a given session) and cached
        # here so subsequent requests reuse the same model instance.
        self.stt: Optional[STTProvider] = None
        self.stt_load_attempted = False
        self.stt_unavailable_reason: Optional[str] = None
        # Lazily built wrapper handed out by get_stt (speculative recall).
        self._stt_speculative: Optional[_SpeculativeSTT] = None

        self.sessions: Dict[str, WebChatSession] = {}
        self.user_index: Dict[str, str] = {}
        # Evicted sessions whose memory is still consolidating on its
        # executor; referenced here so GC can't kill the job mid-flight.
        self._retiring_sessions: List[WebChatSession] = []
        (
            self.persist_base,
            self.embeddings_base,
            self.profile_base,
            self.transcripts_base,
        ) = self._memory_bases(config)
        # Durable per-session chat history for the sessions API. Follows the
        # same per-user file layout as the memory tiers, but is NOT cleared
        # by TTL eviction — only by DELETE /api/sessions/{id}. Character-
        # scoped like the memory tiers: a guest character must not list (or
        # resume, and thereby rehydrate) Kurisu's conversations.
        self.transcripts = TranscriptStore(self.transcripts_base)
        self._side_paths = self._side_store_paths(config)
        self._init_side_stores(config)
        self.lock = threading.Lock()

        # The Qwen3 cloned voice takes ~2 minutes to load + capture CUDA
        # graphs. Loading it here would block uvicorn from serving at all,
        # so the GPU desktop build showed a blank "waking up" wall instead
        # of a usable chat. Warm it on a background thread instead: the app
        # opens in seconds (replies are subtitle-only) and the voice joins
        # as soon as it's ready. Must start last — the thread touches
        # self.lock/self.sessions, which have to exist first.
        if self.config.audio.enable_voice and config.providers.tts == "qwen3_tts":
            self.tts_warming = True
            threading.Thread(
                target=self._warm_tts_in_background,
                name="tts-warmup",
                daemon=True,
            ).start()

    @staticmethod
    def _character_scope(config: Config):
        """Per-character path scoper for this config (identity when builtin)."""
        scope = getattr(config, "character_memory_path", None)
        if callable(scope):
            return scope
        return lambda value: value

    @classmethod
    def _memory_bases(
        cls, config: Config
    ) -> tuple[Optional[Path], Optional[Path], Optional[Path], Optional[Path]]:
        """(persist, embeddings, profile, transcripts) base paths.

        Applies per-character memory scoping (Config.character_memory_path):
        a custom character's stores live under characters/<id>/ while the
        builtin Kurisu keeps the legacy locations untouched. Scoping happens
        here — at consumption — because the config's own path fields must
        keep the base values (settings merges round-trip asdict(config)).
        """
        scope = cls._character_scope(config)

        def _base(attr: str) -> Optional[Path]:
            raw = scope(getattr(config.memory, attr, None))
            return Path(raw) if raw else None

        return (
            _base("persist_path"),
            _base("embeddings_path"),
            _base("profile_path"),
            _base("transcripts_path"),
        )

    @classmethod
    def _side_store_paths(cls, config: Config) -> tuple[
        Optional[str],
        Optional[str],
        Optional[str],
        Optional[str],
        Optional[str],
        Optional[str],
        Optional[str],
    ]:
        """Character-scoped (brief, diary, reachout, followup, life,
        insights, rhythm) store paths, None when the feature is disabled.
        Used both to construct the stores and to detect (in
        ``reload_providers``) that a character switch moved them.

        These are persona-flavored artifacts — the brief/diary/reach-out/
        follow-up are written in the active character's voice from the active
        character's view of the user — so a guest gets its own files instead
        of overwriting (or delivering) Kurisu's. Reminders are deliberately
        NOT scoped anywhere: they are the user's real-world timers, not
        persona memory (see BackgroundScheduler).
        """
        scope = cls._character_scope(config)

        def _path(section: str) -> Optional[str]:
            cfg = getattr(config, section, None)
            if not cfg or not getattr(cfg, "enabled", False):
                return None
            return scope(getattr(cfg, "store_path", None))

        return (
            _path("daily_brief"),
            _path("diary"),
            _path("reach_out"),
            _path("follow_through"),
            _path("life_threads"),
            _path("insights"),
            _path("rhythm"),
        )

    def _init_side_stores(self, config: Config) -> None:
        """(Re)build the brief/diary/reach-out stores from ``_side_paths``.

        Each is one SHARED instance per store: the scheduler's put() and the
        greeting/presence consume must serialize on the same lock instance,
        so consumers always read them through this SessionStore (or the
        per-session refs the reload loop keeps in sync).
        """
        (
            brief_path,
            diary_path,
            reachout_path,
            followup_path,
            life_path,
            insights_path,
            rhythm_path,
        ) = self._side_paths
        self.brief_store = None
        if brief_path:
            try:
                from ..brief import DailyBriefStore

                self.brief_store = DailyBriefStore(brief_path)
            except Exception as exc:  # pragma: no cover - store init failure
                LOGGER.warning("Daily brief store unavailable: %s", exc)
        self.diary_store = None
        if diary_path:
            try:
                from ..diary import DiaryStore

                self.diary_store = DiaryStore(diary_path)
            except Exception as exc:  # pragma: no cover - store init failure
                LOGGER.warning("Diary store unavailable: %s", exc)
        self.reachout_store = None
        if reachout_path:
            try:
                from ..reachout import ReachOutStore

                self.reachout_store = ReachOutStore(reachout_path)
            except Exception as exc:  # pragma: no cover - store init failure
                LOGGER.warning("Reach-out store unavailable: %s", exc)
        self.followup_store = None
        if followup_path:
            try:
                from ..followup import FollowUpStore

                self.followup_store = FollowUpStore(followup_path)
            except Exception as exc:  # pragma: no cover - store init failure
                LOGGER.warning("Follow-up store unavailable: %s", exc)
        self.life_store = None
        if life_path:
            try:
                from ..life_threads import LifeThreadStore

                self.life_store = LifeThreadStore(life_path)
            except Exception as exc:  # pragma: no cover - store init failure
                LOGGER.warning("Life-thread store unavailable: %s", exc)
        self.insight_store = None
        if insights_path:
            try:
                from ..insights import InsightStore

                self.insight_store = InsightStore(insights_path)
            except Exception as exc:  # pragma: no cover - store init failure
                LOGGER.warning("Insight store unavailable: %s", exc)

        self.rhythm_store = None
        if rhythm_path:
            try:
                from ..rhythm import RhythmStore

                self.rhythm_store = RhythmStore(rhythm_path)
            except Exception as exc:  # pragma: no cover - store init failure
                LOGGER.warning("Rhythm store unavailable: %s", exc)

    def _load_tts_blocking(self, config: Config) -> Optional[TTSProvider]:
        """Construct (and pre-warm) the configured TTS provider, or None."""
        self.tts_fallback_from = None
        self.tts_error = None
        try:
            tts = _app().load_tts(config)
            # Pre-warm local Melo providers so the first user-visible
            # turn does not pay tokenizer/session initialization.
            if config.providers.tts in {"melotts", "melotts_onnx"} and tts:
                if config.providers.tts == "melotts":
                    LOGGER.info("Pre-warming MeloTTS model (this takes ~50s)...")
                else:
                    LOGGER.info("Pre-warming MeloTTS ONNX model...")
                try:
                    tts.synthesize("テスト", speaking_rate=1.0, pitch_shift=0.0)
                    LOGGER.info("MeloTTS model ready (pre-warmed)")
                except Exception as warmup_exc:
                    LOGGER.warning(
                        "MeloTTS warmup failed (will be slow on first request): %s",
                        warmup_exc,
                    )
            return tts
        except Exception as exc:  # pragma: no cover - optional dependency failure
            # The GPU desktop build defaults to the Qwen3 cloned voice, which
            # can fail at load time (weights not yet downloaded, CUDA OOM,
            # missing sox). Rather than go silent, fall back to the always-
            # bundled ONNX voice so the app still speaks.
            self.tts_error = str(exc)
            if config.providers.tts == "qwen3_tts":
                LOGGER.warning(
                    "Qwen3 voice failed to load (%s); falling back to ONNX voice.",
                    exc,
                )
                try:
                    fallback = dataclasses.replace(config.providers, tts="melotts_onnx")
                    fallback_config = dataclasses.replace(config, providers=fallback)
                    provider = _app().load_tts(fallback_config)
                    self.tts_fallback_from = "qwen3_tts"
                    return provider
                except Exception as fallback_exc:
                    LOGGER.warning(
                        "ONNX fallback voice also failed to load: %s",
                        fallback_exc,
                    )
                    self.tts_error = (
                        f"{exc} (ONNX fallback also failed: {fallback_exc})"
                    )
            else:
                LOGGER.warning("Voice playback disabled for web UI: %s", exc)
            return None

    def _warm_tts_in_background(self) -> None:
        """Daemon-thread target: load the (typically qwen3) voice off the
        request path, then apply it.

        Wrapped in a bare ``except BaseException`` (not just ``Exception``):
        this runs with no supervisor, so anything that escapes it — even an
        exotic non-Exception (e.g. a C-extension-raised BaseException
        subclass) — would otherwise kill the daemon thread with
        ``tts_warming`` stuck True forever: permanently "warming", permanently
        silent, with no error surfaced anywhere. The handler clears
        ``tts_warming`` and records a short ``tts_error`` unconditionally so
        callers degrade the same way a normal load failure would, then
        re-raises nothing (a daemon thread has no one to propagate to).
        """
        try:
            provider = self._load_tts_blocking(self.config)
            with self.lock:
                self.tts = provider
                self.tts_warming = False
                for session in self.sessions.values():
                    if session.tts is None:
                        session.tts = provider
                reload_pending = self.tts_reload_pending
                self.tts_reload_pending = False
            if provider is not None:
                LOGGER.info("Voice ready (background warmup complete)")
            else:
                LOGGER.warning("Voice unavailable (background warmup failed)")
            if reload_pending:
                # A settings save arrived mid-warmup and deferred its TTS
                # reload (see reload_providers' tts_warming branch). Re-apply
                # it now, directly on this thread — never spawn another
                # warmup thread here, or a rapid string of saves during a
                # slow load could pile up concurrent multi-GB loads (the
                # exact double-load VRAM blowout tts_warming exists to
                # prevent). _reapply_tts_reload reads self.config under its
                # own lock, so it always sees the latest saved config even if
                # another save landed after this warmup finished but before
                # this check ran.
                self._reapply_tts_reload()
        except BaseException as exc:  # noqa: BLE001 - see docstring
            with self.lock:
                self.tts_warming = False
                self.tts_error = f"warmup thread crashed: {exc}"
            telemetry.incr("tts.warmup_thread_crashed")
            LOGGER.error("TTS warmup thread crashed: %s", exc, exc_info=True)

    def _reapply_tts_reload(self) -> None:
        """Re-run the TTS-provider load ``reload_providers`` skipped while
        warmup was in flight, on the calling (background warmup) thread,
        against whatever config is current now.

        By the time this runs, ``reload_providers`` has already overwritten
        ``self.config`` with the new config (it does that unconditionally,
        before checking ``tts_warming``) — the pre-save config is gone, so
        there's no "old vs new" left to diff for the fingerprint-skip or
        voice-rebind fast paths ``reload_providers`` itself uses. This is a
        rare, already-deferred path (a settings save landing in the ~2-minute
        qwen3 load window), so it deliberately skips those optimizations and
        just does a plain load — correctness over shaving a reload here.
        Must never be called while holding ``self.lock`` (it takes the lock
        itself, and ``rebind_providers``, further down, can call into a
        session's own ``handle_lock`` — the same reason ``reload_providers``
        releases ``self.lock`` before its own rebind loop).
        """
        with self.lock:
            config = self.config
            if config.audio.enable_voice:
                self.tts_fallback_from = None
                self.tts_error = None
                try:
                    self.tts = _app().load_tts(config)
                except (
                    Exception
                ) as exc:  # pragma: no cover - optional dependency failure
                    LOGGER.warning(
                        "Voice playback disabled after deferred settings change: %s",
                        exc,
                    )
                    self.tts = None
                    self.tts_error = str(exc)
            else:
                self.tts = None
                self.tts_fallback_from = None
                self.tts_error = None
            tts = self.tts
            sessions = list(self.sessions.values())
        for session in sessions:
            session.rebind_providers(config, session.llm, session.embedder, tts)

    @staticmethod
    def _tts_fingerprint(config: Config) -> tuple:
        """Everything the TTS provider construction depends on.

        Used to skip the TTS rebuild on settings saves that don't touch the
        voice: reloading can mean re-initializing a multi-GB GPU model
        (qwen3_tts takes ~2 minutes and re-captures CUDA graphs), which must
        not happen just because the user saved an LLM API key.
        """
        providers = dataclasses.asdict(config.providers)
        for key in ("llm", "stt", "embedding"):
            providers.pop(key, None)
        return (
            providers,
            config.audio.enable_voice,
            config.openai_api_key,
            config.huggingface_api_token,
        )

    # Voice fields a loaded qwen3 provider can swap in place (one ref-audio
    # embed) instead of the multi-GB model reload — the character-switch
    # fast path.
    _VOICE_REBIND_KEYS = frozenset(
        {"qwen3_tts_ref_audio", "qwen3_tts_ref_text", "qwen3_tts_language"}
    )

    def _voice_rebind_eligible(self, old_config: Config, new_config: Config) -> bool:
        """True when the TTS change is confined to the reference
        clip/text/language, i.e. an in-place rebind can handle it. Must be
        called with the OLD config still current."""
        inner = getattr(self.tts, "provider", self.tts)
        if not hasattr(inner, "rebind_voice"):
            return False
        old_fp = self._tts_fingerprint(old_config)
        new_fp = self._tts_fingerprint(new_config)
        old_providers = dict(old_fp[0])
        new_providers = dict(new_fp[0])
        for key in self._VOICE_REBIND_KEYS:
            old_providers.pop(key, None)
            new_providers.pop(key, None)
        return (old_providers, *old_fp[1:]) == (new_providers, *new_fp[1:])

    def _try_voice_rebind(self, new_config: Config) -> bool:
        """Rebind the loaded TTS voice in place. Returns True when handled;
        False falls through to the normal full reload."""
        inner = getattr(self.tts, "provider", self.tts)
        try:
            inner.rebind_voice(
                new_config.providers.qwen3_tts_ref_audio,
                new_config.providers.qwen3_tts_ref_text,
                getattr(new_config.providers, "qwen3_tts_language", None),
            )
        except Exception as exc:
            LOGGER.warning("In-place voice rebind failed (%s); full reload", exc)
            return False
        refresh = getattr(self.tts, "refresh_signature", None)
        if callable(refresh):
            refresh()
        return True

    def reload_providers(self, config: Config) -> None:
        """Rebuild the LLM/TTS providers after a settings change.

        Existing chat sessions keep their transcript/memory, but future turns
        use the newly configured providers immediately. This is important for
        first-run setup: saving an API key should not leave the open chat stuck
        on the dummy provider until a relaunch.
        """
        with self.lock:
            tts_unchanged = self.tts is not None and self._tts_fingerprint(
                self.config
            ) == self._tts_fingerprint(config)
            voice_rebindable = (
                not tts_unchanged
                and self.tts is not None
                and self._voice_rebind_eligible(self.config, config)
            )
            # Per-character memory: a character switch changes the scoped
            # store locations. Recompute the bases and, when they moved,
            # hand every live session its new per-user paths below so its
            # MemoryManager is swapped (safely, at a turn boundary) instead
            # of the new character inheriting the old one's memory.
            new_bases = self._memory_bases(config)
            memory_moved = new_bases[:3] != (
                self.persist_base,
                self.embeddings_base,
                self.profile_base,
            )
            transcripts_moved = new_bases[3] != self.transcripts_base
            (
                self.persist_base,
                self.embeddings_base,
                self.profile_base,
                self.transcripts_base,
            ) = new_bases
            if transcripts_moved:
                # New character, new sessions list: rebuild the transcript
                # store on the scoped path so the UI lists (and resumes)
                # only the active character's conversations. Kurisu's
                # transcripts stay untouched at the legacy path and come
                # back verbatim on switch-back.
                self.transcripts = TranscriptStore(self.transcripts_base)
            new_side_paths = self._side_store_paths(config)
            side_stores_moved = new_side_paths != self._side_paths
            if side_stores_moved:
                self._side_paths = new_side_paths
                self._init_side_stores(config)
            self.config = config
            self.llm = _app().load_llm(config)
            self.embedder = _app().load_embedder(config)
            if self.tts_warming:
                # The background warmup thread is mid-load; racing a second
                # multi-GB model construction against it is exactly the
                # double-load VRAM blowout we fixed. Let it finish — but
                # don't just drop the reload: flag it so
                # _warm_tts_in_background re-applies it (against whatever
                # config is current at that point) once warming clears,
                # instead of the settings change silently never taking
                # effect until the next unrelated reload or process restart.
                self.tts_reload_pending = True
                telemetry.incr("tts.reload_deferred")
                LOGGER.info(
                    "TTS warmup in progress; settings change will not reload voice "
                    "(deferred until warmup completes)"
                )
            elif tts_unchanged:
                LOGGER.info(
                    "Settings change does not affect TTS; keeping loaded provider"
                )
            elif voice_rebindable and self._try_voice_rebind(config):
                LOGGER.info("Voice reference rebound in place; model kept loaded")
            elif config.audio.enable_voice:
                self.tts_fallback_from = None
                self.tts_error = None
                try:
                    self.tts = _app().load_tts(config)
                except (
                    Exception
                ) as exc:  # pragma: no cover - optional dependency failure
                    LOGGER.warning(
                        "Voice playback disabled after settings change: %s", exc
                    )
                    self.tts = None
                    self.tts_error = str(exc)
            else:
                self.tts = None
                self.tts_fallback_from = None
                self.tts_error = None
            for session in self.sessions.values():
                memory_paths = None
                if memory_moved:
                    memory_paths = (
                        self._user_path(self.persist_base, session.user_id),
                        self._user_path(self.embeddings_base, session.user_id),
                        self._user_path(self.profile_base, session.user_id),
                    )
                if side_stores_moved:
                    # Greeting-time consumers hold per-session refs to the
                    # brief/diary stores; keep them on the shared instances
                    # (a plain atomic ref swap — these are only ever read at
                    # greeting/idle-chatter time, never mid-turn, so no
                    # turn-boundary staging is needed).
                    session.brief_store = self.brief_store
                    session.diary_store = self.diary_store
                    session.followup_store = self.followup_store
                    session.life_store = self.life_store
                    session.insight_store = self.insight_store
                    session.rhythm_store = self.rhythm_store
                session.rebind_providers(
                    config,
                    self.llm,
                    self.embedder,
                    self.tts,
                    memory_paths=memory_paths,
                    transcript_store=self.transcripts if transcripts_moved else None,
                )

    def get_stt(self) -> Optional[STTProvider]:
        """Lazily construct (and cache) the STT provider on first use.

        Model load can take real time (and, for faster-whisper, download
        weights on first run) so it's deliberately deferred until a request
        actually needs it rather than paid at server startup.
        """
        with self.lock:
            if self.stt is not None:
                if self._stt_speculative is None:
                    self._stt_speculative = _SpeculativeSTT(self.stt, self)
                return self._stt_speculative
            if self.stt_load_attempted:
                return None
            self.stt_load_attempted = True
            try:
                provider = _app().load_stt(self.config)
            except Exception as exc:  # pragma: no cover - defensive
                LOGGER.warning("Failed to load STT provider: %s", exc)
                self.stt_unavailable_reason = str(exc)
                return None
            if provider is None:
                self.stt_unavailable_reason = (
                    f"STT provider '{self.config.providers.stt}' is unavailable "
                    "(missing dependency or not configured)."
                )
                return None
            self.stt = provider
            # Handed out wrapped so a finished transcription can pre-start
            # recall (speculative_recall); a pure pass-through otherwise.
            self._stt_speculative = _SpeculativeSTT(provider, self)
            return self._stt_speculative

    def speculate_recall(self, text: str) -> None:
        """Pre-start recall for a fresh STT transcript (see _SpeculativeSTT).

        The STT endpoint carries no session identity, so the target is the
        most-recently-active live session — the one about to receive the
        chat turn in the single-user voice flow. A wrong guess is safe by
        construction: the future lands on that session's own MemoryManager
        and is consumed only on a matching query, so at worst one embed is
        wasted and the real turn falls back to serial recall.
        """
        if not text or not text.strip():
            return
        if not bool(getattr(self.config.memory, "speculative_recall", True)):
            return
        with self.lock:
            session = max(
                self.sessions.values(),
                key=lambda s: s.last_activity,
                default=None,
            )
        if session is None:
            # Nothing live to speculate for (voice input before any chat
            # request after a restart): the turn will run the serial path.
            telemetry.incr("memory.speculative_recall_no_session")
            return
        session.memory.speculate_recall(text)

    def analyze_vocal_tone(self, audio_data: bytes, sample_rate: int) -> None:
        """Run vocal-emotion analysis and stash it on the target session
        (see ``_SpeculativeSTT``, ``amadeus/vocal_tone_stash.py``).

        Same "most recently active session" targeting as
        ``speculate_recall`` — the STT endpoint carries no session identity,
        and a wrong guess is safe by construction: the stash is per-session,
        TTL-bounded, and consumed at most once by that session's own next
        turn, so at worst one analysis is wasted on a session that never
        reads it.
        """
        if not bool(getattr(self.config.vocal_emotion, "enabled", True)):
            return
        with self.lock:
            session = max(
                self.sessions.values(),
                key=lambda s: s.last_activity,
                default=None,
            )
        if session is None:
            return  # nothing live to target; same no-session case as recall
        from ..vocal_tone_stash import analyze_and_stash

        analyze_and_stash(session, audio_data, sample_rate)

    def _user_path(
        self, base: Optional[Path], user_id: Optional[str]
    ) -> Optional[Path]:
        # Single implementation shared with backup/export (memory/backup.py).
        from ..memory.backup import user_scoped_path

        return user_scoped_path(base, user_id)

    def _clear_user_memory(self, user_id: str, full_forget: bool = False) -> None:
        persist_path = self._user_path(self.persist_base, user_id)
        if persist_path:
            try:
                persist_path.unlink()
            except FileNotFoundError:
                pass
        embeddings_path = self._user_path(self.embeddings_base, user_id)
        if embeddings_path:
            try:
                embeddings_path.unlink()
            except FileNotFoundError:
                pass
        # The long-lived profile ("who you are") survives a normal forget.
        if full_forget:
            profile_path = self._user_path(self.profile_base, user_id)
            if profile_path:
                try:
                    profile_path.unlink()
                except FileNotFoundError:
                    pass

    def _evict_idle_locked(self, now: float) -> None:
        """Drop sessions idle past the TTL. Caller holds ``self.lock``.

        The store's dict was unbounded — every anonymous visitor pinned a
        session (provider refs + a summary-executor thread) forever. Swept on
        access, no background thread. In-flight handlers keep their own
        reference, so an evicted-but-running session finishes safely; a
        returning user gets a fresh session that reloads their on-disk memory.
        """
        ttl_hours = float(getattr(self.config.web, "session_ttl_hours", 6.0) or 0)
        if ttl_hours <= 0:
            return
        ttl_seconds = ttl_hours * 3600
        expired = [
            sid
            for sid, session in self.sessions.items()
            if now - session.last_activity > ttl_seconds
        ]
        for sid in expired:
            session = self.sessions.pop(sid)
            for user_id, mapped in list(self.user_index.items()):
                if mapped == sid:
                    del self.user_index[user_id]
            # Same drain pattern as WebChatSession.reset: memorialize the
            # session and keep it referenced until its executor is idle.
            session.memory.finalize_session(wait=False)
            executor = session.memory._summary_executor
            if executor is not None:
                self._retiring_sessions.append(session)
                done = executor.submit(lambda: None)
                done.add_done_callback(
                    lambda _f, s=session: (
                        self._retiring_sessions.remove(s)
                        if s in self._retiring_sessions
                        else None
                    )
                )
            LOGGER.info("Evicted idle web session %s", sid)

    def get(
        self, session_id: Optional[str], user_id: Optional[str]
    ) -> tuple[str, WebChatSession]:
        with self.lock:
            self._evict_idle_locked(time.monotonic())
            if user_id:
                existing_id = self.user_index.get(user_id)
                if existing_id and existing_id not in self.sessions:
                    del self.user_index[user_id]
                    existing_id = None
                # Only steer requests that didn't name a session toward the
                # user's current one. An explicit session_id must be honored
                # (recreated if evicted/restarted) so the multi-session UI
                # can resume any conversation from the sessions list.
                if existing_id and session_id is None:
                    session_id = existing_id
            if session_id and session_id in self.sessions:
                session = self.sessions[session_id]
                if user_id and session.user_id and session.user_id != user_id:
                    session_id = None
                else:
                    if session.tts is None and self.tts is not None:
                        session.tts = self.tts
                    if user_id and session.user_id is None:
                        session.user_id = user_id
                        session.persist_path = self._user_path(
                            self.persist_base, user_id
                        )
                        session.embeddings_path = self._user_path(
                            self.embeddings_base, user_id
                        )
                        session.profile_path = self._user_path(
                            self.profile_base, user_id
                        )
                        session.reset(forget=False)
                    if user_id:
                        self.user_index[user_id] = session_id
                    session.last_activity = time.monotonic()
                    return session_id, session
            session_id = session_id or uuid.uuid4().hex
            return session_id, self._new_session_locked(session_id, user_id)

    def drop_user_sessions(self, user_id: Optional[str]) -> int:
        """Detach a user's live sessions so the next request reloads from disk.

        Used after a memory import: the imported files are the new truth, and
        any live MemoryManager still holding pre-import state would clobber
        them on its next save. For ``user_id=None`` (the base single-user
        files) every session not owned by an identified user is dropped.
        Returns how many sessions were dropped.
        """

        with self.lock:
            if user_id is not None:
                sids = [
                    sid
                    for sid, session in self.sessions.items()
                    if session.user_id == user_id
                ]
                self.user_index.pop(user_id, None)
            else:
                sids = [
                    sid
                    for sid, session in self.sessions.items()
                    if session.user_id is None
                ]
            for sid in sids:
                self.sessions.pop(sid, None)
            return len(sids)

    def refresh_sibling_long_term(self, source: WebChatSession) -> int:
        """After a user-intent memory mutation on ``source`` (synapse-map
        delete/edit/pin, "remember this"), refresh every OTHER live session
        sharing the same long-term files from disk.

        The JSON vector store and profile are rewrite-whole-file backends: a
        sibling still holding the pre-mutation state in RAM resurrects
        deleted memories (or drops added ones) on its own next persist —
        last-writer-wins, reproduced 2026-07-28. Multi-session chat, the pet
        window, and retiring sessions (which persist during their background
        finalize) are all such siblings. Lighter cousin of
        :meth:`drop_user_sessions` (the import path's hammer): sessions stay
        live, only their long-term RAM state re-reads disk. Returns how many
        siblings were refreshed.
        """
        with self.lock:
            siblings = [
                session
                for session in [
                    *self.sessions.values(),
                    *self._retiring_sessions,
                ]
                if session is not source
                and session.embeddings_path == source.embeddings_path
                and session.profile_path == source.profile_path
            ]
        refreshed = 0
        for session in siblings:
            try:
                session.memory.reload_long_term_from_disk()
                refreshed += 1
            except Exception as exc:  # pragma: no cover - defensive
                # Degrade loudly: a sibling we failed to refresh is a
                # sibling that can still resurrect the deleted record.
                LOGGER.warning("Sibling long-term refresh failed: %s", exc)
                telemetry.incr("memory.sibling_refresh_failed")
        if refreshed:
            telemetry.incr("memory.sibling_stores_refreshed", refreshed)
        return refreshed

    def peek(
        self, session_id: Optional[str], user_id: Optional[str]
    ) -> Optional[WebChatSession]:
        """Return an existing live session without ever creating one.

        Used by close-time paths (window-close farewell beacons): a beacon
        from a tab whose session was already evicted must not materialize a
        fresh session just to finalize it.
        """
        with self.lock:
            if not session_id and user_id:
                session_id = self.user_index.get(user_id)
            if not session_id:
                return None
            session = self.sessions.get(session_id)
            if session is None:
                return None
            if user_id and session.user_id and session.user_id != user_id:
                return None
            return session

    def _new_session_locked(
        self, session_id: str, user_id: Optional[str]
    ) -> WebChatSession:
        """Construct and register a fresh session. Caller holds ``self.lock``."""
        # Rolling daily backup of this user's memory files, taken before the
        # session touches them. Never raises; at most one per 24h per user.
        from ..memory.backup import maybe_daily_backup

        maybe_daily_backup(self.config, user_id)
        persist_path = self._user_path(self.persist_base, user_id)
        embeddings_path = self._user_path(self.embeddings_base, user_id)
        profile_path = self._user_path(self.profile_base, user_id)
        session = WebChatSession(
            self.config,
            self.llm,
            self.embedder,
            self.tts,
            self.emotion_analyzer,
            self.translator,
            user_id=user_id,
            persist_path=persist_path,
            embeddings_path=embeddings_path,
            profile_path=profile_path,
        )
        session.session_id = session_id
        session.transcript_store = self.transcripts
        session.brief_store = self.brief_store
        session.diary_store = self.diary_store
        session.followup_store = self.followup_store
        session.life_store = self.life_store
        session.insight_store = self.insight_store
        session.rhythm_store = self.rhythm_store
        # scope_context_to_session: MemoryManager._load intentionally left
        # short_term empty (it's conversation state now). If this session id
        # already has a durable transcript (eviction/restart recreation of
        # an existing conversation, not a brand-new one), rehydrate the tail
        # from it so the conversation itself doesn't lose context — a no-op
        # when the flag is off or there's no prior transcript for this id.
        session.hydrate_short_term_from_transcript()
        self.sessions[session_id] = session
        if user_id:
            self.user_index[user_id] = session_id
        return session

    def create_session(self, user_id: Optional[str]) -> str:
        """Materialize a brand-new session (POST /api/sessions/new)."""
        with self.lock:
            self._evict_idle_locked(time.monotonic())
            session_id = uuid.uuid4().hex
            self._new_session_locked(session_id, user_id)
        # Register an empty transcript so the session appears in listings
        # immediately, before the first message.
        self.transcripts.ensure_session(user_id, session_id)
        return session_id

    def drop_session(self, session_id: str, user_id: Optional[str]) -> None:
        """Remove a session's in-memory state and persisted transcript.

        Deliberately does not finalize/memorialize: deleting a conversation
        is a "forget this" gesture, not a session end.
        """
        with self.lock:
            self.sessions.pop(session_id, None)
            for uid, mapped in list(self.user_index.items()):
                if mapped == session_id:
                    del self.user_index[uid]
        self.transcripts.delete_session(user_id, session_id)

    def reset(
        self,
        session_id: str,
        forget: bool = True,
        user_id: Optional[str] = None,
        full_forget: bool = False,
    ) -> None:
        with self.lock:
            if session_id in self.sessions:
                self.sessions[session_id].reset(forget=forget, full_forget=full_forget)
                return
            if forget and user_id:
                self._clear_user_memory(user_id, full_forget=full_forget)
                if user_id in self.user_index:
                    del self.user_index[user_id]
                return
            raise KeyError(session_id)
