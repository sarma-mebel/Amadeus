"""Streaming turn orchestrator."""

from __future__ import annotations

import asyncio
import threading
import time
from typing import Any, Awaitable, Callable, Dict, List, Optional

from ..blocking import cached_translate as _cached_translate
from ..blocking import run_blocking as _run_blocking
from ... import provider_health, telemetry
from ...logging_utils import get_logger
from .bridge import iter_in_thread
from .nonverbals import get_nonverbal_library
from .parser import (
    DisplayDelta,
    strip_bilingual_tails,
    NonVerbalEnd,
    SentenceEnd,
    TaggedStreamParser,
    ThoughtEnd,
    ensure_tagged_reply,
    parse_complete_reply,
    parse_complete_reply_full,
    speech_text,
    strip_markdown_for_speech,
    strip_special_tag_lines,
)
from .think_delay import compute_think_delay, signals_for_turn
from .tts_pipeline import (
    _SentenceAudioBatcher,
    _build_tts_pipeline,
    _submit_sentence_audio,
    resolve_tts_emotion,
)

LOGGER = get_logger(__name__)


class StreamCancelled(Exception):
    """Raised internally to unwind a turn after client cancellation."""


async def _cancellable_sleep(seconds: float, cancel_event: threading.Event) -> bool:
    """Sleep up to ``seconds`` but wake early if ``cancel_event`` fires.

    Returns True if the sleep was interrupted by cancellation. The wait is
    polled in short slices so a barge-in during the think-delay aborts within
    ~40ms instead of holding the turn open for the whole beat. Never blocks the
    event loop (each slice is an ``asyncio.sleep``) and never raises.
    """
    if seconds <= 0:
        return cancel_event.is_set()
    slice_s = 0.04
    remaining = seconds
    while remaining > 0:
        if cancel_event.is_set():
            return True
        await asyncio.sleep(min(slice_s, remaining))
        remaining -= slice_s
    return cancel_event.is_set()


def _think_delay_for(session: Any, request: Any, emotion: Optional[str]) -> float:
    """Compute this turn's bounded think-delay (0.0 when the feature is off).

    Best-effort: any failure returns 0.0 so the beat never breaks a turn.
    """
    web_cfg = session.config.web
    if not bool(getattr(web_cfg, "think_delay_enabled", False)):
        return 0.0
    try:
        signals = signals_for_turn(getattr(request, "message", None), emotion)
        return compute_think_delay(
            signals,
            float(getattr(web_cfg, "think_delay_min_seconds", 0.0)),
            float(getattr(web_cfg, "think_delay_max_seconds", 0.0)),
        )
    except Exception:  # pragma: no cover - defensive; the beat is cosmetic
        LOGGER.debug("think-delay computation failed", exc_info=True)
        return 0.0


async def run_streaming_turn(
    session: Any,
    request: Any,
    emit: Callable[[Dict[str, Any]], Awaitable[None]],
    cancel_event: threading.Event,
) -> None:
    """Drive one chat turn over a WebSocket, emitting protocol frames via
    ``emit`` as they become available.

    ``session`` is a ``WebChatSession`` (duck-typed here to avoid an app.py
    import cycle: this module is imported *by* app.py). ``request`` is a
    ``ChatRequest``. Frame shapes (see the WS protocol in the modernization
    plan): token / sentence / audio / tool_progress / done / error.
    """
    # Wrap emit so every frame passes one timing tap: first token/audio frames
    # stamp TTFT/TTFA, and the terminal done frame carries the breakdown.
    timings: Dict[str, float] = {"start": time.monotonic()}
    emit = _wrap_emit_with_timings(emit, timings)

    # _prepare_turn does memory/recall work (short-term context assembly,
    # optional vector recall) synchronously; keep it off the event loop so a
    # slow recall on one session can't stall every other in-flight turn.
    setup = await _run_blocking(session._prepare_turn, request)

    if cancel_event.is_set():
        # Pre-cancelled turn (duplex barge-in during STT seeds the event
        # before this coroutine ever runs): the provider must not be called
        # at all — a full stream()/generate() would run only to be thrown
        # away, holding the provider (and, for local models, its process-wide
        # lock) for the next turn's whole latency budget. Still finalize
        # through the normal seam so the client gets the done frame and the
        # user message rolls back exactly as a post-first-token cancel would.
        await _finalize_streamed_turn(session, request, "", True, emit, tts_failed=0)
        return

    if setup.refusal:
        await _stream_final_text(
            session, request, setup, setup.refusal, emit, cancel_event
        )
        return

    # Mirror the POST path (sessions._handle): _prepare_turn committed the
    # user message, so a provider failure in any generation branch must pop
    # it again or the client's retry double-records the turn. The rollback
    # only matches a trailing user message, so failures after the turn was
    # finalized (TTS drain, done emit) are a safe no-op.
    try:
        if (
            not getattr(request, "images", None)
            and session._tools_active()
            and session._should_use_tools(request.message)
        ):
            # Tool-augmented turn: the ReAct loop is inherently multi-step and
            # non-streaming, so run it to completion (off the event loop) and
            # then pseudo-stream the final text through the same parser/TTS
            # path. Per-tool progress frames keep the multi-second wait from
            # looking like a hang (emitted thread-safely — the loop may run
            # in an executor thread).
            loop = asyncio.get_running_loop()

            def report_tool_progress(event: Dict[str, Any]) -> None:
                frame = {
                    "type": "tool_progress",
                    "tool": event.get("tool"),
                    "status": event.get("status", "running"),
                }
                asyncio.run_coroutine_threadsafe(emit(frame), loop)

            reply = await _run_blocking(
                session._run_tool_loop, setup.context, report_tool_progress
            )
            if setup.use_tagged_output and reply and not parse_complete_reply(reply):
                # Native function-calling rounds tend to drop the tagged grammar
                # (markdown lists, no Japanese) — one repair pass restores it.
                reply = await _run_blocking(
                    ensure_tagged_reply, session.llm, setup.system_prompt, reply
                )
            await _stream_final_text(session, request, setup, reply, emit, cancel_event)
            return

        if not setup.use_tagged_output or not session.llm.supports_stream():
            # Provider can't stream, or tagged output disabled: generate the full
            # reply, then pseudo-stream it through the same parser/TTS path.
            reply = await _run_blocking(session.llm.generate, setup.context)
            telemetry.record_llm_usage(session.llm, "chat")
            provider_health.record_llm_success(session.llm)
            await _stream_final_text(session, request, setup, reply, emit, cancel_event)
            return

        await _stream_tagged_generation(session, request, setup, emit, cancel_event)
    except Exception as exc:
        LOGGER.exception("Streaming turn failed; rolling back user message")
        telemetry.incr("web.provider_failures")
        # Health classification only — the WS error-frame kind (rate_limited
        # vs provider_error, routes_chat.py::_provider_error_kind) and status
        # behavior are unchanged (pinned by
        # test_ws_provider_429_error_frame_carries_rate_limited_kind): a
        # plain (non-tool) streamed turn still surfaces as an error frame,
        # not the in-character RATE_LIMITED_REPLY/INVALID_KEY_REPLY
        # substitution — that only happens inside the tool loop
        # (engine.py::_run_tool_loop, reached via _run_blocking above),
        # which already returns those as a normal successful reply.
        provider_health.classify_and_record(exc)
        session.rollback_user_message(request.message)
        raise


def _wrap_emit_with_timings(
    emit: Callable[[Dict[str, Any]], Awaitable[None]],
    timings: Dict[str, float],
) -> Callable[[Dict[str, Any]], Awaitable[None]]:
    """Stamp time-to-first-token/-sentence/-audio and attach them to the done
    frame.

    Server-side emit time is the measurement point: TTFT/TTFA as the client
    would see them minus network. Values ride the ``done`` frame as transport
    metadata (sibling of ``cancelled``), not inside ``response``, and are also
    recorded process-wide (telemetry ``latency.*`` sum/count pairs plus
    ``latency.last_*`` gauges) so ``/api/status`` can answer "how slow are
    turns lately?" without log-grepping. Recording is best-effort: a timing
    bug must never break the turn itself.
    """

    async def timed_emit(frame: Dict[str, Any]) -> None:
        frame_type = frame.get("type")
        if frame_type == "token":
            timings.setdefault("first_token", time.monotonic())
        elif frame_type == "sentence":
            timings.setdefault("first_sentence", time.monotonic())
        elif frame_type == "audio":
            timings.setdefault("first_audio", time.monotonic())
        elif frame_type == "done":
            start = timings["start"]
            first_token = timings.get("first_token")
            first_sentence = timings.get("first_sentence")
            first_audio = timings.get("first_audio")
            frame = dict(frame)
            frame["timings"] = {
                "ttft_ms": int(1000 * (first_token - start)) if first_token else None,
                "first_sentence_ms": (
                    int(1000 * (first_sentence - start)) if first_sentence else None
                ),
                "ttfa_ms": int(1000 * (first_audio - start)) if first_audio else None,
                "total_ms": int(1000 * (time.monotonic() - start)),
            }
            LOGGER.info(
                "turn latency: ttft=%sms first_sentence=%sms first_audio=%sms "
                "total=%sms cancelled=%s",
                frame["timings"]["ttft_ms"],
                frame["timings"]["first_sentence_ms"],
                frame["timings"]["ttfa_ms"],
                frame["timings"]["total_ms"],
                frame.get("cancelled"),
            )
            _record_turn_latency(frame["timings"], bool(frame.get("cancelled")))
        await emit(frame)

    return timed_emit


def _record_turn_latency(turn_timings: Dict[str, Any], cancelled: bool) -> None:
    """Accumulate one streamed turn's milestones into process telemetry.

    Sum+count per phase (average = sum/count for /api/status consumers) plus
    a ``last_`` gauge. Phases that never happened this turn (voice off -> no
    first_audio) are skipped so they don't drag averages toward zero;
    ``total`` is skipped on cancelled turns (it measures the interruption,
    not the pipeline). Never raises.
    """
    try:
        phases = {
            "ttft": turn_timings.get("ttft_ms"),
            "first_sentence": turn_timings.get("first_sentence_ms"),
            "first_audio": turn_timings.get("ttfa_ms"),
            "turn_total": turn_timings.get("total_ms") if not cancelled else None,
        }
        for phase, ms in phases.items():
            if ms is None:
                continue
            telemetry.incr(f"latency.{phase}_ms_sum", int(ms))
            telemetry.incr(f"latency.{phase}_count")
            telemetry.set_gauge(f"latency.last_{phase}_ms", int(ms))
    except Exception:  # pragma: no cover - instrumentation must never break a turn
        LOGGER.debug("turn latency telemetry recording failed", exc_info=True)

    # New record_duration facility (count/total/max/last, no separate
    # sum-vs-count bookkeeping needed by the caller): the headline
    # silence-to-first-audio number, plus the TTS-only slice of it.
    try:
        ttfa_ms = turn_timings.get("ttfa_ms")
        if ttfa_ms is not None:
            # turn.first_audio_ms: WS message received (turn start, stamped
            # before _prepare_turn) to first ordered audio frame emitted —
            # the single number that best represents the felt "she started
            # talking" latency, voice-off turns simply never record it.
            telemetry.record_duration("turn.first_audio_ms", ttfa_ms)
            first_sentence_ms = turn_timings.get("first_sentence_ms")
            if first_sentence_ms is not None:
                # tts.first_audio_ms: first SENTENCE parsed to first ordered
                # audio frame — isolates TTS's own contribution from the
                # LLM's time-to-first-sentence, so a regression can be
                # attributed to the right stage instead of just "turns feel
                # slower". Never negative in practice (sentence always
                # precedes its audio), but max(0, ...) keeps a clock skew
                # from ever recording garbage.
                telemetry.record_duration(
                    "tts.first_audio_ms", max(0.0, ttfa_ms - first_sentence_ms)
                )
    except Exception:  # pragma: no cover - instrumentation must never break a turn
        LOGGER.debug("duration telemetry recording failed", exc_info=True)


def _sentence_frame(evt: SentenceEnd) -> Dict[str, Any]:
    """The one canonical shape of a ``sentence`` protocol frame."""
    return {
        "type": "sentence",
        "seq": evt.seq,
        "en": evt.en,
        "ja": evt.ja,
        "emotion": evt.emotion,
        "emotion_raw": evt.emotion_raw,
        "emotion_intensity": evt.emotion_intensity,
    }


class _TurnAudio:
    """Per-turn TTS wiring shared by the live-stream and pseudo-stream paths.

    Owns the pipeline + sentence batcher and the config knobs both paths used
    to read (and drift on) independently: first-clause split, voice language,
    full-reply batching, unit-size bounds.
    """

    def __init__(self, session: Any, setup: Any, emit: Any) -> None:
        web_cfg = session.config.web
        self.pipeline, self._chunked_tts = _build_tts_pipeline(session, setup, emit)
        # Emotion labels the provider can voice distinctly (empty for every
        # provider without emotion refs — sentence emotions then resolve to
        # None and the whole feature stays out of the synthesis path).
        self._tts_emotions = frozenset(
            getattr(session.tts, "available_emotions", ()) or ()
        )
        self._first_clause_split = bool(
            getattr(web_cfg, "streaming_first_clause_split", True)
        )
        self.voice_language = str(
            getattr(session.config.audio, "voice_language", "japanese") or "japanese"
        ).lower()
        self._full_reply_tts = bool(getattr(web_cfg, "streaming_tts_full_reply", True))
        self.batcher: Optional[_SentenceAudioBatcher] = None
        if self.pipeline:
            self.batcher = _SentenceAudioBatcher(
                self.submit,
                min_chars=int(getattr(web_cfg, "streaming_tts_min_unit_chars", 24)),
                hold_all=self._full_reply_tts,
                max_chars=int(getattr(web_cfg, "streaming_tts_max_unit_chars", 360)),
            )
        self._nonverbals = (
            get_nonverbal_library()
            if self.pipeline and bool(getattr(web_cfg, "streaming_nonverbals", True))
            else None
        )

    def submit(self, seq: float, text: str, emotion: Optional[str] = None) -> None:
        """Send one speech unit straight to the pipeline (batcher callback)."""
        _submit_sentence_audio(
            self.pipeline,
            seq,
            text,
            # Full-reply mode wants exactly one generation unit; the
            # first-clause split would cut it back in two.
            first_clause_split=self._first_clause_split and not self._full_reply_tts,
            chunked_tts=self._chunked_tts,
            emotion=emotion,
        )

    def add_sentence(self, evt: SentenceEnd) -> None:
        if self.batcher and (evt.ja or evt.en):
            self.batcher.add(
                evt.seq,
                speech_text(evt.en, evt.ja, self.voice_language),
                resolve_tts_emotion(evt.emotion_raw, evt.emotion, self._tts_emotions),
            )

    def add_nonverbal(self, evt: NonVerbalEnd) -> None:
        """Splice a pre-recorded clip into the audio stream at its seq.

        The batcher may be holding coalesced sentences that precede this
        clip; flush them first so the pipeline's submit order (which IS its
        emit order) matches the reply order. In full-reply batching mode
        this splits the reply into units at the non-verbal — a voice-reset
        boundary is inaudible there because a different sound intervenes
        anyway. No clip for the kind = the tag stays silently stripped.
        """
        if not self.pipeline or not self._nonverbals:
            return
        clip = self._nonverbals.pick(evt.kind)
        if clip is None:
            return
        if self.batcher:
            self.batcher.flush()
        self.pipeline.submit_prerendered(evt.seq, clip)

    async def drain(self, cancelled: bool) -> int:
        """Flush or abort the pipeline; returns how many seqs never produced
        audio (0 when voice is off), for the done frame's ``tts_failed``."""
        if not self.pipeline:
            return 0
        if cancelled:
            self.pipeline.cancel()
        else:
            if self.batcher:
                self.batcher.flush()
            await self.pipeline.close()
        return self.pipeline.failed_count


async def _stream_tagged_generation(
    session: Any,
    request: Any,
    setup: Any,
    emit: Callable[[Dict[str, Any]], Awaitable[None]],
    cancel_event: threading.Event,
) -> None:
    parser = TaggedStreamParser()
    audio = _TurnAudio(session, setup, emit)

    english_parts: List[str] = []
    # Seed from the event: a cancel that lands before generation starts makes
    # iter_in_thread's worker stop before yielding a single delta, so the
    # in-loop check never runs and the turn would otherwise finish
    # "uncancelled" with an empty reply — committing an orphan user message.
    cancelled = cancel_event.is_set()
    tts_failed = 0
    thoughts_enabled = bool(
        getattr(session.config.web, "streaming_inner_thoughts", False)
    )
    thought_emitted = False
    beat_done = False
    # True only while draining the provider stream. iter_in_thread SETS
    # cancel_event in its own teardown ``finally`` when the generator is
    # exhausted (it's how the worker thread is told to stop), so once the
    # ``async for`` has ended, cancel_event no longer distinguishes a real
    # barge-in from normal completion. The think-delay only honours
    # cancel_event as a barge-in while this is set; a sentence emitted from
    # parser.finish() (the whole reply arrived in the final delta — common
    # with fast providers) still gets its beat but can't be spuriously
    # "cancelled" by teardown.
    stream_active = True

    async def handle_event(evt: Any) -> None:
        nonlocal thought_emitted, beat_done, cancelled
        if isinstance(evt, DisplayDelta):
            english_parts.append(evt.text)
            await emit({"type": "token", "text": evt.text})
        elif isinstance(evt, SentenceEnd):
            # Variable "think delay": pause a beat before the FIRST sentence's
            # frame/audio, keyed off the input signals plus this sentence's
            # emotion, so TTFA isn't robotically constant. Cancel-safe while
            # the stream is live: a barge-in during the beat flips
            # ``cancelled`` and we skip the frame; the outer loop then unwinds
            # and drain() aborts the pipeline. Fires once per turn, before any
            # audio is submitted.
            if not beat_done:
                beat_done = True
                delay = _think_delay_for(session, request, evt.emotion)
                if delay > 0:
                    interrupted = await _cancellable_sleep(delay, cancel_event)
                    if interrupted and stream_active:
                        cancelled = True
                        return
            await emit(_sentence_frame(evt))
            audio.add_sentence(evt)
        elif isinstance(evt, ThoughtEnd):
            # Consumer-side enforcement: flag off or a second thought ->
            # silently dropped (never displayed, never spoken) — the parser
            # stays mechanical and always recognizes the tag.
            if thoughts_enabled and not thought_emitted and evt.text:
                thought_emitted = True
                await emit({"type": "thought", "seq": evt.seq, "text": evt.text})
        elif isinstance(evt, NonVerbalEnd):
            # Audio-only splice; drops silently when no clip library exists.
            audio.add_nonverbal(evt)

    def gen_factory():
        # frequency_penalty breaks opener ruts on the streamed chat reply
        # (summaries/JSON extraction never pass it).
        return session.llm.stream(
            setup.context,
            frequency_penalty=session.config.llm_frequency_penalty,
        )

    # Generation-start clock for llm.first_token_ms / llm.stream_total_ms.
    # Stamped immediately before iter_in_thread starts pulling from the
    # provider's stream() generator — this is real wall time to the
    # provider, not wall time since the WS message arrived (that also
    # includes _prepare_turn's memory/recall work, measured separately by
    # turn.first_audio_ms's "start" timestamp). The think-delay beat below
    # never sits inside the first-token window: it only runs when handling a
    # SentenceEnd event, which the parser can only emit strictly after the
    # DisplayDelta(s) that produced this sentence's text already fired their
    # "token" frames — so first_token is always the honest, undelayed value.
    gen_started = time.monotonic()
    first_token_recorded = False
    # Assume the stream fails until it provably drains: on exception
    # unwinding the finally must ABORT the audio pipeline (cancel path), not
    # close-and-flush it — close() waits for and emits every queued sentence
    # before the caller can send its error frame, so she'd keep talking for
    # seconds, then error, and the heard sentences would roll back anyway.
    stream_failed = True

    try:
        async for delta in iter_in_thread(gen_factory, cancel_event):
            if not first_token_recorded:
                first_token_recorded = True
                telemetry.record_duration(
                    "llm.first_token_ms", 1000 * (time.monotonic() - gen_started)
                )
            if cancel_event.is_set():
                cancelled = True
                break
            for evt in parser.feed(delta):
                await handle_event(evt)
                # handle_event may itself flip ``cancelled`` (a barge-in that
                # lands during the pre-first-sentence think delay); stop
                # emitting immediately rather than draining the rest of this
                # chunk's events.
                if cancelled:
                    break
            if cancelled:
                break
        # The stream is drained; iter_in_thread's teardown sets cancel_event as
        # part of stopping its worker, so from here a set cancel_event no longer
        # means "the client barged in". Tell the think-delay to stop treating it
        # as one (a beat on a finish()-emitted sentence still runs, just can't
        # be spuriously cancelled by teardown).
        stream_active = False
        if not cancelled:
            for evt in parser.finish():
                await handle_event(evt)
        stream_failed = False
    finally:
        # Recorded even on a cancelled/exception-unwound stream: "how long
        # was the provider actually generating for" is meaningful either way,
        # and finally guarantees exactly one sample per attempted stream.
        telemetry.record_duration(
            "llm.stream_total_ms", 1000 * (time.monotonic() - gen_started)
        )
        tts_failed = await audio.drain(cancelled or stream_failed)

    # Once per turn, after the provider generator is done (never per chunk):
    # usage rides the stream's final chunk, so it's populated exactly now. A
    # cancelled stream records whatever the provider saw before the cut
    # (those tokens were still consumed), or nothing when usage never arrived.
    telemetry.record_llm_usage(session.llm, "chat")
    # Reaching here means iter_in_thread drained the provider stream without
    # raising — a genuine successful generation (cancellation is a client
    # barge-in, not a provider failure, so it still counts as healthy).
    provider_health.record_llm_success(session.llm)

    reply_text = "".join(english_parts).strip()

    await _finalize_streamed_turn(
        session, request, reply_text, cancelled, emit, tts_failed=tts_failed
    )


async def _stream_final_text(
    session: Any,
    request: Any,
    setup: Any,
    reply: str,
    emit: Callable[[Dict[str, Any]], Awaitable[None]],
    cancel_event: threading.Event,
) -> None:
    """Send a complete (non-streamed-from-provider) reply through the same
    per-sentence token/sentence/audio framing, for the refusal and tool
    branches (and providers without stream()).
    """
    tagged: List[SentenceEnd] = []
    thoughts: List[ThoughtEnd] = []
    nonverbals: List[NonVerbalEnd] = []
    if setup.use_tagged_output:
        tagged, thoughts, nonverbals = parse_complete_reply_full(reply)
    audio = _TurnAudio(session, setup, emit)

    cancelled = cancel_event.is_set()

    if tagged:
        # Variable "think delay": one beat before the first spoken frame, keyed
        # off the input signals plus the first sentence's emotion. This path
        # (tools / refusals / non-streaming providers) pseudo-streams an already
        # complete reply, so the beat is the only source of TTFA variability
        # here. Cancel-safe: an interrupted beat marks the turn cancelled.
        first_emotion = tagged[0].emotion if tagged else None
        beat = _think_delay_for(session, request, first_emotion)
        if beat > 0 and not cancelled:
            if await _cancellable_sleep(beat, cancel_event):
                cancelled = True
        # Interleave sentences and non-verbal clips by seq so a [laugh]
        # spliced mid-reply plays exactly where the model put it.
        spoken = sorted(tagged + nonverbals, key=lambda e: e.seq)
        emitted: List[SentenceEnd] = []
        for event in spoken:
            if cancelled or cancel_event.is_set():
                cancelled = True
                break
            if isinstance(event, NonVerbalEnd):
                audio.add_nonverbal(event)
                continue
            sentence = event
            await emit({"type": "token", "text": sentence.en + " "})
            await emit(_sentence_frame(sentence))
            audio.add_sentence(sentence)
            emitted.append(sentence)
        if (
            thoughts
            and not cancelled
            and bool(getattr(session.config.web, "streaming_inner_thoughts", False))
            and thoughts[0].text
        ):
            # Max one per reply; never touches reply_text or the TTS batcher.
            await emit(
                {"type": "thought", "seq": thoughts[0].seq, "text": thoughts[0].text}
            )
        # Emitted-only, mirroring _stream_tagged_generation's english_parts: a
        # cancel mid-walk must not commit sentences the client never received —
        # memory would record a reply the user never saw, and the
        # [interrupted] prompt rule would then tell her not to re-deliver
        # content she never delivered.
        reply_text = " ".join(s.en for s in emitted)
    else:
        # No tags: emit as a single token + sentence using the VADER-less
        # default emotion; caller's session.emotion_analyzer already runs in
        # the POST path — for streaming we keep it simple (neutral) since
        # per-sentence emotion is the tagged path's job.
        # A reply of ONLY [thought]/non-verbal lines parses as tagged with
        # zero sentences and lands here — strip those lines first or the
        # literal "[thought] ..." text is shown, spoken, and committed. An
        # empty result then flows into the empty-reply rollback below. And a
        # model that dropped its tags but kept '||' separators must not
        # show (or remember) the raw wire format either.
        reply = strip_bilingual_tails(strip_special_tag_lines(reply))
        emotion = "neutral"
        if hasattr(session, "emotion_analyzer"):
            try:
                data = session.emotion_analyzer.analyze(reply)
                emotion = data.get("vrm_expression", "neutral")
            except Exception:  # pragma: no cover - defensive
                pass
        # This branch previously ignored cancellation entirely: an untagged
        # reply was emitted, synthesized, and committed in full even when the
        # cancel landed before any of it.
        if cancelled or cancel_event.is_set():
            cancelled = True
            reply = ""
        elif reply:
            await emit({"type": "token", "text": reply})
            await emit(
                _sentence_frame(SentenceEnd(seq=1, en=reply, ja="", emotion=emotion))
            )
        if audio.pipeline and reply.strip():
            # Untagged fallback: what goes to the translator/TTS must be
            # speakable prose — display keeps the original markdown.
            japanese_text = strip_markdown_for_speech(reply)
            if session.translator and audio.voice_language != "english":
                try:
                    japanese_text = await _run_blocking(
                        _cached_translate, session.translator, japanese_text
                    )
                except Exception as exc:  # pragma: no cover - network failure
                    LOGGER.warning(
                        "Streaming translation failed, using English: %s", exc
                    )
            audio.submit(1, japanese_text)
        reply_text = reply

    tts_failed = await audio.drain(cancelled)

    await _finalize_streamed_turn(
        session,
        request,
        reply_text,
        cancelled,
        emit,
        tts_failed=tts_failed,
    )


async def _finalize_streamed_turn(
    session: Any,
    request: Any,
    reply_text: str,
    cancelled: bool,
    emit: Callable[[Dict[str, Any]], Awaitable[None]],
    *,
    tts_failed: int = 0,
) -> None:
    """Commit the assistant message exactly once and send the terminal
    ``done`` frame. Mirrors the memory-write shape of WebChatSession._handle.

    ``tts_failed`` rides the done frame (sibling of ``cancelled``) so a
    silent turn is distinguishable client-side from a voice-off turn.
    """
    final_text = reply_text
    rolled_back = False
    if cancelled and final_text:
        final_text = f"{final_text} [interrupted]"
    if not final_text:
        # No reply exists — either a barge-in before the first emitted token
        # (think-delay beat, recall, or the provider's TTFT window), or a
        # NOT-cancelled empty generation (provider streamed zero deltas, or
        # only [thought]/non-verbal lines). Both have the same memory shape:
        # complete_turn("") would skip the assistant append but leave the
        # user message as a live trailing orphan in short_term for the rest
        # of the process — _load's healing only runs at the next start. Roll
        # it back instead, as if the turn never happened; distinct counters
        # keep the two causes tellable apart.
        session.rollback_user_message(request.message)
        telemetry.incr(
            "web.cancelled_before_reply" if cancelled else "web.empty_reply_rolled_back"
        )
        rolled_back = True
    else:
        session.complete_turn(request.message, final_text)

    response = {
        "reply": final_text,
        "turn_id": None if rolled_back else getattr(session, "last_turn_id", None),
        "memories": [],
        "animations": session.pop_pending_animations(),
        "summary": session.memory.summary,
        "summary_strategy": session.memory.summary_info.get("strategy"),
        "summary_revision": session.memory.summary_info.get("revision"),
        "summary_updated_at": session.memory.summary_info.get("updated_at"),
    }
    await emit(
        {
            "type": "done",
            "cancelled": cancelled,
            "tts_failed": tts_failed,
            "response": response,
        }
    )
