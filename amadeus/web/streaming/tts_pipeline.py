"""Ordered, concurrent sentence-level TTS synthesis."""

from __future__ import annotations

import asyncio
import base64
import inspect
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, List, Optional

from ... import provider_health, telemetry
from ...logging_utils import get_logger
from .bridge import iter_in_thread
from .parser import (
    normalize_ja_for_speech,
    sanitize_speech_for_clone,
    split_first_clause,
)

LOGGER = get_logger(__name__)


def _record_latency(phase: str, started: float) -> None:
    """Best-effort telemetry for one synthesis phase: sum+count (average =
    sum/count for /api/status consumers) plus a ``last_`` gauge. Never raises
    — instrumentation must not be able to break audio."""
    try:
        ms = int(1000 * (time.monotonic() - started))
        telemetry.incr(f"latency.{phase}_ms_sum", ms)
        telemetry.incr(f"latency.{phase}_count")
        telemetry.set_gauge(f"latency.last_{phase}_ms", ms)
    except Exception:  # pragma: no cover - defensive
        pass


class _SynthesisCancelled(Exception):
    """Raised by ``_synthesize_if_not_cancelled`` when the pipeline was
    cancelled before a queued-then-dispatched synthesis actually started.
    Caught separately from real provider errors so a cooperative cancel skip
    is never counted as (or logged like) a synthesis failure."""


@dataclass
class AudioChunk:
    seq: float
    mime: str
    data_b64: str


@dataclass
class _Prerendered:
    """A queue item whose audio already exists (non-verbal clip splice)."""

    seq: float
    data: bytes


def resolve_tts_emotion(
    raw: Optional[str],
    basic: Optional[str],
    available: frozenset[str],
) -> Optional[str]:
    """Pick the emotion label to hand the TTS provider for one sentence.

    ``available`` is the provider's ``available_emotions`` set (empty for
    providers without emotion refs — the common case — which short-circuits
    the whole feature to None). The rich tag (``emotion_raw``) wins when the
    provider has a ref for it, then the basic VRM-6 emotion; anything else
    means the base voice.
    """
    if not available:
        return None
    for label in (raw, basic):
        if label and label.lower() in available:
            return label.lower()
    return None


def dominant_emotion(labels: List[Optional[str]]) -> Optional[str]:
    """Most frequent non-None label, ties broken toward earliest occurrence.

    Synthesis units can span several sentences (short-sentence coalescing,
    full-reply mode), each with its own tag; one unit gets one voice, so the
    unit speaks with whatever emotion dominates it.
    """
    counts: Dict[str, int] = {}
    first_seen: Dict[str, int] = {}
    for index, label in enumerate(labels):
        if not label:
            continue
        counts[label] = counts.get(label, 0) + 1
        first_seen.setdefault(label, index)
    if not counts:
        return None
    return max(counts, key=lambda label: (counts[label], -first_seen[label]))


class TTSPipeline:
    """Synthesizes queued sentences to audio and emits them strictly in
    ascending ``seq`` order, even though synthesis work can complete out of
    order (workers, provider latency jitter).

    A single worker task pulls ``(seq, japanese_text)`` off an internal
    queue, runs the blocking ``tts.synthesize`` in a thread, and buffers the
    result. An ordered emitter releases buffered results to the caller's
    ``emit`` callback only when the next expected seq is available; a
    failure at a given seq logs and advances past it rather than stalling
    every later sentence.

    ``seq`` is normally the sentence's integer sequence number, but the
    first-clause fast path (``split_first_clause``) submits *two* chunks for
    sentence 1 to get audio playing sooner: ``1`` for the short first clause
    and ``1.5`` for the remainder, both before sentence 2's ``2``. Because
    ordering here is driven by an explicit expected-keys queue (populated in
    ``submit()`` call order) rather than by incrementing an integer, any
    sortable, distinct keys work as long as callers submit in the order they
    should be emitted.

    When constructed with ``synthesize_stream`` (a ``text -> Iterator[bytes]``
    callable yielding WAV-encoded sub-sentence chunks, e.g.
    ``Qwen3TTSProvider.synthesize_stream``), each submitted seq is
    synthesized via that generator instead of one blocking ``synthesize``
    call, and every chunk it yields is emitted as its own audio frame at
    ``seq, seq + SUB_SEQ_STEP, seq + 2*SUB_SEQ_STEP, ...`` (capped below
    the next integer seq) rather than waiting for the whole sentence.
    Sub-chunks for one seq are always emitted in generator order, and only
    once every seq/sub-chunk ahead of them in submission order has already
    emitted — the same strict global ordering ``_expected_order`` already
    guarantees for whole-sentence chunks. Because the sub-chunk count isn't
    known ahead of time, each streamed seq occupies exactly one slot in
    ``_expected_order`` (as usual) while its chunks are buffered separately
    in ``_stream_chunks`` and drained through that slot one at a time.
    """

    # Sub-chunk keys for one streamed seq are seq + n*_SUB_SEQ_STEP, capped
    # below _MAX_SUB_SEQ_OFFSET so they always sort before the next integer
    # seq (and stay clear of the first-clause split's seq+0.5 convention,
    # which this pipeline never combines with streaming for the same
    # submission).
    _SUB_SEQ_STEP = 0.001
    _MAX_SUB_SEQ_OFFSET = 0.49

    def __init__(
        self,
        synthesize: Callable[[str], bytes | Awaitable[bytes]],
        emit: Callable[[AudioChunk], Awaitable[None]],
        *,
        workers: int = 1,
        mime: str = "audio/wav",
        synthesize_stream: Optional[Callable[[str], Any]] = None,
    ) -> None:
        # NOTE: raising `workers` (config.web.tts_stream_workers) does not
        # parallelize synthesis when the active provider is
        # MeloTTSONNXProvider. Its synthesize_async
        # (melotts_onnx_provider.py) runs the ONNX call synchronously on the
        # event loop rather than handing it to this pool, and ONNX Runtime
        # claims all physical cores per inference call regardless — extra
        # workers just contend for the same cores. Investigated and
        # rejected as a TTFA fix; don't "fix" this back to workers>1 without
        # first changing the ONNX provider's threading model.
        self._synthesize = synthesize
        self._synthesize_stream = synthesize_stream
        # Set by this pipeline's own cancel() to abort in-flight streaming
        # generators — deliberately NOT the turn-level cancel_event, which
        # iter_in_thread unconditionally sets in its own `finally` once any
        # one generator finishes (even on normal completion); sharing it
        # would make every *other* in-flight sentence's stream look
        # cancelled the instant one sentence's stream ended. Each
        # _run_streaming() call passes iter_in_thread a fresh private Event
        # instead (since iter_in_thread mutates whatever it's given) and
        # polls this pipeline-wide flag itself to honor cancel().
        self._pipeline_cancel_event = threading.Event()
        self._emit = emit
        self._mime = mime
        self._queue: (
            "asyncio.Queue[Optional[tuple[float, str, Optional[str]] | _Prerendered]]"
        ) = asyncio.Queue()
        self._pending: Dict[float, bytes] = {}
        self._failed: set[float] = set()
        # Streaming-only bookkeeping: per-seq FIFO of chunks generated but
        # not yet emitted, whether the generator for that seq has finished
        # producing (successfully or not), and how many chunks have already
        # been emitted (for computing each chunk's outgoing sub-key). Keyed
        # by the seq submit() originally registered — never rewritten.
        self._stream_chunks: Dict[float, List[bytes]] = {}
        self._stream_done: set[float] = set()
        self._stream_emitted: Dict[float, int] = {}
        # Explicit FIFO of keys in the order submit() was called, so the
        # emitter doesn't need to assume integer +1 stepping between seqs
        # (the first-clause split submits two chunks for one sentence seq,
        # e.g. 1 then 1.5, both ahead of sentence 2's plain 2).
        self._expected_order: List[float] = []
        # Monotonic count of seqs that produced no audio at all (entries in
        # self._failed are discarded as the emitter advances past them, so
        # they can't be counted after the fact). Read via failed_count to
        # surface "this turn was silent because TTS failed" to the client.
        self._failed_total = 0
        self._lock = asyncio.Lock()
        # Held across the pop-ready + emit pair (never by state-only
        # mutations): see _apply_and_emit for the workers>1 ordering race
        # it closes.
        self._emit_order_lock = asyncio.Lock()
        self._workers = [
            asyncio.create_task(self._worker_loop()) for _ in range(max(1, workers))
        ]
        self._executor = ThreadPoolExecutor(
            max_workers=max(1, workers), thread_name_prefix="amadeus-tts-pipeline"
        )
        self._closed = False

    def _mark_failed_locked(self, seq: float) -> None:
        """Record a seq that will never produce audio (caller holds _lock)."""
        self._failed.add(seq)
        self._failed_total += 1

    async def _apply_and_emit(self, mutate: Callable[[], None]) -> None:
        """Apply one state mutation, then drain every now-ready chunk to the
        ``emit`` callback while holding ``_emit_order_lock`` across the whole
        pop+emit pair.

        With ``workers > 1``, popping under ``_lock`` but emitting after its
        release let two workers interleave: worker A pops [seq 1], suspends
        at the first ``await emit``, worker B (whose synthesis just finished)
        pops [seq 2] — seq 1 is already off ``_expected_order`` — and its
        emit completes first, so the client sees audio out of order despite
        the documented strictly-ascending contract. Holding the separate
        emit-order lock from pop through the last emit means no worker can
        pop until every previously-popped chunk has finished emitting.
        ``_lock`` still guards the state dicts and is never held across an
        emit await, so a slow emit can't block state bookkeeping.
        """
        async with self._lock:
            mutate()
        async with self._emit_order_lock:
            async with self._lock:
                ready = self._pop_ready_locked()
            for chunk in ready:
                await self._emit(chunk)

    def _synthesize_if_not_cancelled(self, text: str, emotion: Optional[str]):
        """Runs inside the executor thread. A full abort of an in-flight
        provider call isn't possible (cancel() can't reach into a call
        already handed to the executor), but this bounds the damage for a
        task that was queued-then-started right as cancel() fired: checking
        the pipeline's cancel flag here, before touching the provider, means
        that task exits immediately instead of running a full generation and
        holding the process-wide provider lock (qwen3) — which would
        otherwise delay the NEXT turn's first audio by however long that
        generation takes, well after the barge-in that cancelled this one.
        """
        if self._pipeline_cancel_event.is_set():
            raise _SynthesisCancelled(f"cancelled before synthesis started: {text!r}")
        if emotion is not None:
            return self._synthesize(text, emotion)
        return self._synthesize(text)

    @property
    def failed_count(self) -> int:
        """How many submitted seqs produced no audio at all this turn."""
        return self._failed_total

    def submit(
        self, seq: float, japanese_text: str, emotion: Optional[str] = None
    ) -> None:
        """Queue one speech unit. ``emotion`` (already resolved against the
        provider's ``available_emotions`` by the caller) rides along to the
        synthesize callable; None — the overwhelmingly common case — keeps
        the exact pre-emotion call shape, so plain ``text -> bytes``
        callables keep working unchanged."""
        if self._closed:
            return
        # Audio-side only: keeps the cloned voice from drifting off-timbre
        # on latin-heavy content (file names, paths), then normalizes bare
        # digit runs/dates/times into kanji-numeral readings (Japanese is
        # the TTS model's weakest language; text-side normalization avoids
        # misreadings). Display text already went out in the sentence frame
        # untouched — both transforms are audio-side only.
        japanese_text = sanitize_speech_for_clone(japanese_text)
        japanese_text = normalize_ja_for_speech(japanese_text)
        self._expected_order.append(seq)
        self._queue.put_nowait((seq, japanese_text, emotion))

    def submit_prerendered(self, seq: float, audio_bytes: bytes) -> None:
        """Queue already-encoded audio (a non-verbal clip) at ``seq``.

        Takes a normal slot in ``_expected_order``, so it plays exactly
        where it was submitted relative to synthesized sentences — after
        everything submitted before it, before everything after. The bytes
        must match the pipeline's ``mime`` (WAV for every current caller).
        """
        if self._closed:
            return
        self._expected_order.append(seq)
        self._queue.put_nowait(_Prerendered(seq, audio_bytes))

    async def close(self) -> None:
        """Signal no more submissions and wait for in-flight work to drain."""
        if self._closed:
            return
        self._closed = True
        for _ in self._workers:
            self._queue.put_nowait(None)
        await asyncio.gather(*self._workers, return_exceptions=True)
        self._executor.shutdown(wait=True, cancel_futures=True)

    def cancel(self) -> None:
        """Abort immediately: drop queued and in-flight results."""
        self._closed = True
        self._pipeline_cancel_event.set()
        for task in self._workers:
            task.cancel()
        self._executor.shutdown(wait=False, cancel_futures=True)
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break

    async def _worker_loop(self) -> None:
        while True:
            item = await self._queue.get()
            if item is None:
                return
            if isinstance(item, _Prerendered):
                # No synthesis to do: buffer the bytes and drain in order.
                def _buffer_prerendered(item: _Prerendered = item) -> None:
                    self._pending[item.seq] = item.data

                await self._apply_and_emit(_buffer_prerendered)
                continue
            seq, text, emotion = item
            if self._synthesize_stream is not None:
                await self._run_streaming(seq, text, emotion)
                continue
            if self._pipeline_cancel_event.is_set():
                # Checked immediately before dispatch, not just once at the
                # top of the loop: cancel() can arrive while several items
                # already sat queued, and each one only becomes "about to
                # run" right here. Skipping now means a cancelled turn's
                # later, not-yet-started sentences never touch the provider
                # (and its process-wide lock) at all — cancel() itself only
                # skips futures the executor hasn't started yet, so this is
                # the only guard for items still in this async queue.
                await self._apply_and_emit(
                    lambda seq=seq: self._mark_failed_locked(seq)
                )
                continue
            started = time.monotonic()
            try:
                # A set emotion is passed positionally; None keeps the
                # one-arg call so pre-emotion callables never see it.
                if inspect.iscoroutinefunction(self._synthesize):
                    if emotion is not None:
                        audio_bytes = await self._synthesize(text, emotion)
                    else:
                        audio_bytes = await self._synthesize(text)
                else:
                    loop = asyncio.get_running_loop()
                    if emotion is not None:
                        audio_bytes = await loop.run_in_executor(
                            self._executor,
                            self._synthesize_if_not_cancelled,
                            text,
                            emotion,
                        )
                    else:
                        audio_bytes = await loop.run_in_executor(
                            self._executor,
                            self._synthesize_if_not_cancelled,
                            text,
                            None,
                        )
            except _SynthesisCancelled:
                # Cooperative cancel caught inside the executor thread right
                # before it would have touched the provider — not a real
                # synthesis failure, so no telemetry/log noise for it (the
                # pre-dispatch check above already accounts for the cancelled
                # turn in _failed_total/failed_count).
                await self._apply_and_emit(
                    lambda seq=seq: self._mark_failed_locked(seq)
                )
                continue
            except Exception as exc:  # pragma: no cover - provider runtime errors
                LOGGER.warning("TTS synthesis failed for seq=%s: %s", seq, exc)
                telemetry.incr("tts.synth_failures")
                provider_health.record_tts_failure(exc)
                await self._apply_and_emit(
                    lambda seq=seq: self._mark_failed_locked(seq)
                )
                continue
            # Whole-unit synthesis wall time; the direct measurement of the
            # provider's real-time factor (and of the ~10x qwen3 slow-engine
            # fallback) in live use.
            _record_latency("tts_synth", started)
            provider_health.record_tts_success()

            def _buffer_result(seq: float = seq, audio_bytes: bytes = audio_bytes):
                self._pending[seq] = audio_bytes

            await self._apply_and_emit(_buffer_result)

    async def _run_streaming(
        self, seq: float, text: str, emotion: Optional[str] = None
    ) -> None:
        """Handle one submitted seq via the provider's chunked streaming API.

        ``seq`` keeps exactly the one slot ``submit()`` already registered
        in ``_expected_order`` — it is never rewritten or re-searched.
        Instead, each arriving chunk is appended to ``self._stream_chunks[seq]``
        (a per-seq FIFO of not-yet-emitted chunks) and assigned its display
        sub-key as ``seq + n*_SUB_SEQ_STEP`` purely for the *outgoing*
        ``AudioChunk.seq`` the client sees; ``_pop_ready_locked`` drains that
        FIFO one chunk at a time whenever ``seq`` is at the front of
        ``_expected_order``, only removing ``seq`` from ``_expected_order``
        once the generator has signaled completion (``_stream_done``) and
        every chunk has been drained. This avoids ever searching
        ``_expected_order`` for a value that a concurrent pop may have
        already removed.
        """
        n = 0
        # A private event for this one generator: iter_in_thread always
        # sets whatever event it's given once the generator finishes (even
        # on normal completion), so it cannot be the shared
        # _pipeline_cancel_event or every other in-flight sentence's stream
        # would look cancelled the instant this one ended.
        call_cancel_event = threading.Event()

        def gen_factory():
            if emotion is not None:
                return self._synthesize_stream(text, emotion)
            return self._synthesize_stream(text)

        started = time.monotonic()
        try:
            async for audio_bytes in iter_in_thread(gen_factory, call_cancel_event):
                if self._pipeline_cancel_event.is_set():
                    break
                if n == 0:
                    # Chunked synthesis: what matters for perceived latency
                    # is how fast the FIRST chunk arrives, not the whole unit.
                    _record_latency("tts_first_chunk", started)

                def _buffer_chunk(audio_bytes: bytes = audio_bytes) -> None:
                    self._stream_chunks.setdefault(seq, []).append(audio_bytes)

                await self._apply_and_emit(_buffer_chunk)
                n += 1
        except Exception as exc:  # pragma: no cover - provider runtime errors
            LOGGER.warning("Streaming TTS synthesis failed for seq=%s: %s", seq, exc)
            telemetry.incr("tts.stream_synth_failures")
            provider_health.record_tts_failure(exc)

            def _mark_stream_failed() -> None:
                self._stream_done.add(seq)
                if not self._stream_chunks.get(seq):
                    self._mark_failed_locked(seq)

            await self._apply_and_emit(_mark_stream_failed)
            return

        def _mark_stream_done() -> None:
            self._stream_done.add(seq)
            if n == 0:
                # Generator produced no chunks at all (empty text or
                # provider quirk); treat as failed so downstream keys aren't
                # blocked waiting on a chunk that will never arrive.
                self._mark_failed_locked(seq)
                provider_health.record_tts_failure(
                    f"streaming synthesis produced no audio for seq={seq}"
                )
            else:
                provider_health.record_tts_success()

        await self._apply_and_emit(_mark_stream_done)

    def _pop_ready_locked(self) -> List[AudioChunk]:
        """Release buffered results in submit() order, front of the queue
        first, stopping at the first key that hasn't synthesized (or
        failed) yet.

        A streaming seq (present in ``self._stream_chunks`` or
        ``self._stream_done``) drains at most one buffered chunk per call —
        its slot in ``_expected_order`` only advances past it once
        ``_stream_done`` is set for that seq *and* its chunk FIFO is empty,
        so later seqs can never emit while an earlier streaming seq still
        has chunks in flight.
        """
        ready: List[AudioChunk] = []
        while self._expected_order:
            next_expected = self._expected_order[0]
            if next_expected in self._failed:
                self._failed.discard(next_expected)
                self._stream_done.discard(next_expected)
                self._stream_chunks.pop(next_expected, None)
                self._expected_order.pop(0)
                continue
            if (
                next_expected in self._stream_chunks
                or next_expected in self._stream_done
            ):
                chunks = self._stream_chunks.get(next_expected)
                if chunks:
                    audio_bytes = chunks.pop(0)
                    n = self._stream_emitted.get(next_expected, 0)
                    self._stream_emitted[next_expected] = n + 1
                    sub_key = min(
                        next_expected + n * self._SUB_SEQ_STEP,
                        next_expected + self._MAX_SUB_SEQ_OFFSET,
                    )
                    ready.append(
                        AudioChunk(
                            seq=sub_key,
                            mime=self._mime,
                            data_b64=base64.b64encode(audio_bytes).decode("ascii"),
                        )
                    )
                    if not chunks and next_expected in self._stream_done:
                        self._stream_chunks.pop(next_expected, None)
                        self._stream_done.discard(next_expected)
                        self._stream_emitted.pop(next_expected, None)
                        self._expected_order.pop(0)
                    # Either more chunks may still arrive for this seq, or we
                    # just fully drained+removed it — either way, stop after
                    # emitting one chunk this pass so a later seq's pending
                    # results can't be evaluated out of turn.
                    continue
                if next_expected in self._stream_done:
                    # Done, and the FIFO drained to empty on a previous pass
                    # (no chunks at all were ever produced but not marked
                    # failed — defensive; normally caught above).
                    self._stream_chunks.pop(next_expected, None)
                    self._stream_done.discard(next_expected)
                    self._stream_emitted.pop(next_expected, None)
                    self._expected_order.pop(0)
                    continue
                # Streaming but no chunk ready yet and not done: stop.
                break
            if next_expected not in self._pending:
                break
            audio_bytes = self._pending.pop(next_expected)
            self._expected_order.pop(0)
            chunk = AudioChunk(
                seq=next_expected,
                mime=self._mime,
                data_b64=base64.b64encode(audio_bytes).decode("ascii"),
            )
            ready.append(chunk)
        return ready


def _submit_sentence_audio(
    tts_pipeline: TTSPipeline,
    seq: int,
    text: str,
    *,
    first_clause_split: bool,
    chunked_tts: bool = False,
    emotion: Optional[str] = None,
) -> None:
    """Submit a sentence's speech text to the TTS pipeline.

    For sentence 1 only, when ``first_clause_split`` is enabled, splits the
    text into a short first clause (seq) + remainder (seq + 0.5) so the
    first TTS call is much shorter and audio can start playing well before
    the rest of the sentence has synthesized. Later sentences, and turns
    with the flag off, submit unchanged (seq only).

    ``chunked_tts`` means the pipeline itself was built with a
    ``synthesize_stream`` callable, so every submitted sentence (not just
    the first) already gets broken into sub-sentence audio frames by the
    pipeline. In that case the first-clause split is redundant — it only
    exists to shrink the first synthesis *unit*, which chunking already
    does more finely — so it's skipped even when the flag is on.
    """
    if seq == 1 and first_clause_split and not chunked_tts:
        first, rest = split_first_clause(text)
        tts_pipeline.submit(seq, first, emotion)
        if rest:
            tts_pipeline.submit(seq + 0.5, rest, emotion)
        return
    tts_pipeline.submit(seq, text, emotion)


class _SentenceAudioBatcher:
    """Coalesce short sentences into the next one before TTS submission.

    Every submitted text is an independent zero-shot voice-clone generation,
    and very short sentences are both the most drift-prone (the clone can
    land on a different speaker for one interjection) and create audible
    voice-reset boundaries mid-reply. Holding a short sentence and prepending
    it to the next keeps each generation unit above ``min_chars``, so the
    voice stays consistent across more of the reply without giving up
    per-sentence streaming for normal-length sentences.

    Sentence 1 is never held — first-audio latency beats consistency there.
    ``flush()`` must be called at end of turn so a short trailing sentence
    isn't dropped. ``min_chars=0`` disables batching entirely.

    ``hold_all=True`` is full-reply mode: every sentence (including the
    first) is held and the whole reply is submitted as one unit at
    ``flush()``, so the entire turn is a single zero-shot generation with no
    voice-reset boundaries at all. ``max_chars`` caps the unit size — once
    the accumulated text reaches it, the batch is submitted and a new one
    starts — so pathologically long replies split into a few large
    generations instead of one huge one (0 = no cap).
    """

    def __init__(
        self,
        submit: Callable[[float, str, Optional[str]], None],
        min_chars: int,
        *,
        hold_all: bool = False,
        max_chars: int = 0,
    ) -> None:
        self._submit = submit
        self._min_chars = min_chars
        self._hold_all = hold_all
        self._max_chars = max_chars
        self._pending_seq: Optional[float] = None
        self._pending_text: str = ""
        # One resolved emotion label (or None) per sentence coalesced into
        # the pending unit — the unit is voiced with the dominant one.
        self._pending_emotions: List[Optional[str]] = []

    def add(self, seq: float, text: str, emotion: Optional[str] = None) -> None:
        emotions: List[Optional[str]] = [emotion]
        if self._pending_text:
            text = f"{self._pending_text} {text}"
            seq = self._pending_seq if self._pending_seq is not None else seq
            emotions = self._pending_emotions + emotions
            self._pending_seq = None
            self._pending_text = ""
            self._pending_emotions = []
        if self._hold_all:
            if self._max_chars > 0 and len(text) >= self._max_chars:
                self._submit(seq, text, dominant_emotion(emotions))
                return
            self._pending_seq = seq
            self._pending_text = text
            self._pending_emotions = emotions
            return
        if seq != 1 and self._min_chars > 0 and len(text) < self._min_chars:
            self._pending_seq = seq
            self._pending_text = text
            self._pending_emotions = emotions
            return
        self._submit(seq, text, dominant_emotion(emotions))

    def flush(self) -> None:
        if self._pending_text:
            self._submit(
                self._pending_seq or 0,
                self._pending_text,
                dominant_emotion(self._pending_emotions),
            )
            self._pending_seq = None
            self._pending_text = ""
            self._pending_emotions = []


def _build_tts_pipeline(
    session: Any,
    setup: Any,
    emit: Callable[[Dict[str, Any]], Awaitable[None]],
) -> tuple[Optional[TTSPipeline], bool]:
    """Construct the per-turn TTSPipeline, if audio is enabled for this turn.

    Returns ``(pipeline_or_None, chunked_tts)`` — ``chunked_tts`` tells the
    caller whether the pipeline was built with sub-sentence streaming
    synthesis (so it should skip the redundant first-clause split; see
    ``_submit_sentence_audio``).
    """
    audio_ready = (
        session.tts is not None
        and session.config.audio.enable_voice
        and session.audio_enabled
    )
    if not audio_ready:
        return None, False

    workers = max(1, int(getattr(session.config.web, "tts_stream_workers", 1)))
    persona = setup.persona

    async def emit_audio(chunk: AudioChunk) -> None:
        await emit(
            {
                "type": "audio",
                "seq": chunk.seq,
                "mime": chunk.mime,
                "data": chunk.data_b64,
            }
        )

    # Providers advertise emotion refs via available_emotions; anything else
    # never gets the kwarg (submit() only carries labels already resolved
    # against this same set, so the guard here is belt-and-braces).
    tts_emotions = frozenset(getattr(session.tts, "available_emotions", ()) or ())

    def _synthesis_kwargs(emotion: Optional[str]) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {
            "speaking_rate": persona.speaking_rate,
            "pitch_shift": persona.pitch_shift,
        }
        if emotion and emotion in tts_emotions:
            kwargs["emotion"] = emotion
        return kwargs

    synthesize_async = getattr(session.tts, "synthesize_async", None)
    if synthesize_async is not None:

        async def synthesize_audio(text: str, emotion: Optional[str] = None) -> bytes:
            return await synthesize_async(text, **_synthesis_kwargs(emotion))

    else:

        def synthesize_audio(text: str, emotion: Optional[str] = None) -> bytes:
            return session.tts.synthesize(text, **_synthesis_kwargs(emotion))

    chunked_tts_enabled = bool(
        getattr(session.config.web, "streaming_chunked_tts", True)
    )
    inner_synthesize_stream = getattr(session.tts, "synthesize_stream", None)
    synthesize_stream = None
    if chunked_tts_enabled and inner_synthesize_stream is not None:

        def synthesize_stream(text: str, emotion: Optional[str] = None):
            return inner_synthesize_stream(text, **_synthesis_kwargs(emotion))

    tts_pipeline = TTSPipeline(
        synthesize_audio,
        emit_audio,
        workers=workers,
        synthesize_stream=synthesize_stream,
    )
    return tts_pipeline, synthesize_stream is not None
