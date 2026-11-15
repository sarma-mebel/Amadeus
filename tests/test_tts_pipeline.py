"""Unit tests for amadeus.web.streaming.TTSPipeline ordered emission."""

from __future__ import annotations

import asyncio
import threading

import pytest

from amadeus import telemetry
from amadeus.web.streaming import AudioChunk, TTSPipeline


@pytest.mark.asyncio
async def test_emits_strictly_in_seq_order_under_latency_inversion():
    """seq 3 finishes fastest, seq 1 slowest — output must still be 1,2,3."""
    delays = {1: 0.06, 2: 0.03, 3: 0.0}

    def synth(text: str) -> bytes:
        seq = int(text)
        import time

        time.sleep(delays[seq])
        return f"audio-{seq}".encode()

    emitted: list[AudioChunk] = []

    async def emit(chunk: AudioChunk) -> None:
        emitted.append(chunk)

    pipeline = TTSPipeline(synth, emit, workers=3)
    pipeline.submit(1, "1")
    pipeline.submit(2, "2")
    pipeline.submit(3, "3")
    await pipeline.close()

    assert [c.seq for c in emitted] == [1, 2, 3]


@pytest.mark.asyncio
async def test_multi_worker_emit_order_survives_suspended_emit():
    """The pop+emit pair must be atomic with respect to other workers.

    Pre-fix, _pop_ready_locked ran under _lock but the emit loop ran after
    its release: with workers=2, worker A pops [seq 1] and suspends inside
    its emit await; worker B (whose synthesis finished meanwhile) pops
    [seq 2] — seq 1 is already off _expected_order — and its emit COMPLETES
    first, violating the strictly-ascending contract. Measured with this
    exact scenario: old code emitted [2, 1]. The emit-order lock held across
    pop+emit forces seq 2's emit to wait for seq 1's to finish.
    """
    import time

    def synth(text: str) -> bytes:
        seq = int(text)
        # seq 1 synthesizes instantly (pops first), seq 2 shortly after —
        # while seq 1's emit is still suspended below.
        time.sleep({1: 0.0, 2: 0.05}[seq])
        return f"audio-{seq}".encode()

    completed: list[float] = []

    async def emit(chunk: AudioChunk) -> None:
        if chunk.seq == 1:
            await asyncio.sleep(0.2)  # a slow send: the emit await suspends
        completed.append(chunk.seq)

    pipeline = TTSPipeline(synth, emit, workers=2)
    pipeline.submit(1, "1")
    pipeline.submit(2, "2")
    await pipeline.close()

    assert completed == [1, 2]


@pytest.mark.asyncio
async def test_failed_sentence_is_skipped_without_stalling_later_ones():
    def synth(text: str) -> bytes:
        seq = int(text)
        if seq == 2:
            raise RuntimeError("synth boom")
        return f"audio-{seq}".encode()

    emitted: list[AudioChunk] = []

    async def emit(chunk: AudioChunk) -> None:
        emitted.append(chunk)

    pipeline = TTSPipeline(synth, emit, workers=1)
    pipeline.submit(1, "1")
    pipeline.submit(2, "2")
    pipeline.submit(3, "3")
    await pipeline.close()

    assert [c.seq for c in emitted] == [1, 3]


@pytest.mark.asyncio
async def test_single_worker_processes_in_submission_order():
    order: list[int] = []

    def synth(text: str) -> bytes:
        order.append(int(text))
        return b"audio"

    async def emit(chunk: AudioChunk) -> None:
        pass

    pipeline = TTSPipeline(synth, emit, workers=1)
    for seq in range(1, 6):
        pipeline.submit(seq, str(seq))
    await pipeline.close()

    assert order == [1, 2, 3, 4, 5]


@pytest.mark.asyncio
async def test_close_waits_for_in_flight_work():
    async def emit(chunk: AudioChunk) -> None:
        pass

    completed = {"count": 0}

    def synth(text: str) -> bytes:
        completed["count"] += 1
        return b"audio"

    pipeline = TTSPipeline(synth, emit, workers=2)
    for seq in range(1, 5):
        pipeline.submit(seq, str(seq))
    await pipeline.close()

    assert completed["count"] == 4


@pytest.mark.asyncio
async def test_submit_normalizes_japanese_numbers_before_synth():
    """The WS streaming path bypasses _synthesize_tts entirely, so
    TTSPipeline.submit must apply normalize_ja_for_speech itself (alongside
    the existing sanitize_speech_for_clone call) rather than relying on a
    shared choke point."""
    received: list[str] = []

    def synth(text: str) -> bytes:
        received.append(text)
        return b"audio"

    async def emit(chunk: AudioChunk) -> None:
        pass

    pipeline = TTSPipeline(synth, emit, workers=1)
    pipeline.submit(1, "3時に集合ね。")
    await pipeline.close()

    assert received == ["三時に集合ね。"]


# ---- Emotion threading ------------------------------------------------------


@pytest.mark.asyncio
async def test_emotion_reaches_synthesize_only_when_set():
    calls: list[tuple[str, str | None]] = []

    def synth(text: str, emotion: str | None = None) -> bytes:
        calls.append((text, emotion))
        return b"audio"

    async def emit(chunk: AudioChunk) -> None:
        pass

    pipeline = TTSPipeline(synth, emit, workers=1)
    pipeline.submit(1, "plain")
    pipeline.submit(2, "voiced", "happy")
    await pipeline.close()

    assert calls == [("plain", None), ("voiced", "happy")]


def test_resolve_tts_emotion_prefers_raw_over_basic():
    from amadeus.web.streaming import resolve_tts_emotion

    available = frozenset({"happy", "excited"})
    assert resolve_tts_emotion("excited", "happy", available) == "excited"
    assert resolve_tts_emotion("amused", "happy", available) == "happy"
    assert resolve_tts_emotion(None, "Happy", available) == "happy"
    assert resolve_tts_emotion("annoyed", "angry", available) is None
    # Empty set (any provider without emotion refs) short-circuits to None.
    assert resolve_tts_emotion("excited", "happy", frozenset()) is None


def test_dominant_emotion_majority_and_ties():
    from amadeus.web.streaming import dominant_emotion

    assert dominant_emotion([]) is None
    assert dominant_emotion([None, None]) is None
    assert dominant_emotion(["sad", "happy", "happy"]) == "happy"
    # Tie breaks toward the earliest-seen label.
    assert dominant_emotion(["sad", "happy"]) == "sad"
    assert dominant_emotion([None, "angry", None]) == "angry"


def test_batcher_coalesced_unit_gets_dominant_emotion():
    from amadeus.web.streaming import _SentenceAudioBatcher

    submitted: list[tuple[float, str, str | None]] = []

    def submit(seq: float, text: str, emotion: str | None) -> None:
        submitted.append((seq, text, emotion))

    # Full-reply mode: every sentence is held; flush() emits one unit voiced
    # with the emotion that dominated the reply.
    batcher = _SentenceAudioBatcher(submit, min_chars=0, hold_all=True)
    batcher.add(1, "one.", "happy")
    batcher.add(2, "two.", None)
    batcher.add(3, "three.", "sad")
    batcher.add(4, "four.", "sad")
    batcher.flush()

    assert len(submitted) == 1
    assert submitted[0] == (1, "one. two. three. four.", "sad")


def test_batcher_short_sentence_carries_emotion_forward():
    from amadeus.web.streaming import _SentenceAudioBatcher

    submitted: list[tuple[float, str, str | None]] = []

    def submit(seq: float, text: str, emotion: str | None) -> None:
        submitted.append((seq, text, emotion))

    batcher = _SentenceAudioBatcher(submit, min_chars=10)
    batcher.add(1, "a long enough first sentence.", "happy")
    batcher.add(2, "short.", "angry")  # held (under min_chars)
    batcher.add(3, "and a long enough closer sentence.", None)
    batcher.flush()

    assert [entry[2] for entry in submitted] == ["happy", "angry"]
    assert submitted[1][0] == 2
    assert submitted[1][1] == "short. and a long enough closer sentence."


# ---- Degrade-loudly telemetry + cooperative cancel -------------------------


@pytest.mark.asyncio
async def test_synthesis_failure_increments_synth_failures_counter():
    """The non-streaming _worker_loop except-branch must count a swallowed
    synthesis failure (tts.synth_failures), not just log it — otherwise a
    provider that starts failing every call degrades silently."""

    def synth(text: str) -> bytes:
        raise RuntimeError("provider boom")

    async def emit(chunk: AudioChunk) -> None:
        pass

    before = telemetry.snapshot().get("tts.synth_failures", 0)

    pipeline = TTSPipeline(synth, emit, workers=1)
    pipeline.submit(1, "1")
    await pipeline.close()

    assert telemetry.snapshot().get("tts.synth_failures", 0) == before + 1
    assert pipeline.failed_count == 1


@pytest.mark.asyncio
async def test_cancel_before_dispatch_never_calls_provider():
    """cancel() only skips futures the executor hasn't started; a sentence
    still sitting in the internal asyncio queue when cancel() fires must
    also be skipped — checked immediately before dispatch in _worker_loop —
    so the provider (and its process-wide lock, for real providers like
    qwen3) is never touched for it."""
    calls: list[str] = []
    started = threading.Event()
    release = threading.Event()

    def synth(text: str) -> bytes:
        calls.append(text)
        started.set()
        release.wait(timeout=5)
        return b"audio"

    async def emit(chunk: AudioChunk) -> None:
        pass

    # workers=1 so the second submission is guaranteed to still be sitting
    # in the queue (not yet dispatched to the executor) while the first is
    # in flight.
    pipeline = TTSPipeline(synth, emit, workers=1)
    pipeline.submit(1, "first")
    pipeline.submit(2, "second")

    # Wait for the first synthesis to actually start on the executor thread,
    # then cancel while the second is still queued (never dispatched).
    await asyncio.get_event_loop().run_in_executor(None, started.wait, 5)
    pipeline.cancel()
    release.set()

    # Give the executor a moment to finish the in-flight first call; the
    # second must never have reached synth() at all.
    await asyncio.sleep(0.1)
    assert calls == ["first"]
