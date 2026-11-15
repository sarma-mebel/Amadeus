"""Tests for sub-sentence chunked TTS streaming (time-to-first-audio).

Covers:
- TTSPipeline emitting multiple audio frames per submitted sentence when a
  synthesize_stream callable is supplied, preserving strict global ordering
  even when synthesis completes out of order across concurrent sentences.
- Fallback to the existing whole-sentence behavior for providers that don't
  support streaming (no synthesize_stream attribute at all).
- CachedTTSProvider.synthesize_stream: cache hit -> single assembled chunk
  with no provider call; cache miss -> chunks pass through live and the
  assembled WAV is written to disk for next time.
- WS integration: chunk-capable provider produces multiple audio frames per
  sentence, and the first-clause split is skipped when chunked TTS is active
  (chunking already subsumes it), but stays intact when chunking is off/the
  provider can't stream.
"""

from __future__ import annotations

import asyncio
import io
import threading
import wave

import pytest

from amadeus.config import Config
from amadeus.providers.llm.base import DummyProvider
from amadeus.providers.tts.base import TTSProvider
from amadeus.providers.tts.cache import CachedTTSProvider
from amadeus.web.app import ChatRequest, create_app
from amadeus.web.streaming import AudioChunk, TTSPipeline


def _wav_bytes(text: str, *, framerate: int = 24000) -> bytes:
    """Build a tiny valid mono 16-bit WAV whose payload encodes ``text``."""
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(framerate)
        payload = text.encode("utf-8")
        # Pad to an even length so it's valid 16-bit PCM frame data.
        if len(payload) % 2:
            payload += b"\x00"
        wav_file.writeframes(payload)
    return buffer.getvalue()


# ---------------------------------------------------------------------------
# TTSPipeline: chunked streaming ordering behavior
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_streaming_provider_emits_multiple_chunks_per_sentence():
    """A single submit() with a streaming synthesize_stream produces one
    audio frame per chunk yielded by the generator, all sharing the seq's
    integer part but with ascending fractional sub-keys.
    """
    emitted: list[AudioChunk] = []

    def synthesize_stream(text: str):
        for i in range(3):
            yield _wav_bytes(f"{text}-{i}")

    async def emit(chunk: AudioChunk) -> None:
        emitted.append(chunk)

    def synthesize(text: str) -> bytes:
        raise AssertionError("whole-sentence synthesize should not be used")

    pipeline = TTSPipeline(synthesize, emit, synthesize_stream=synthesize_stream)
    pipeline.submit(1, "hello")
    await pipeline.close()

    assert len(emitted) == 3
    seqs = [c.seq for c in emitted]
    assert seqs == sorted(seqs)
    assert all(1 <= s < 2 for s in seqs)
    assert seqs[0] == 1  # first sub-chunk reuses the sentence's own seq


@pytest.mark.asyncio
async def test_streaming_chunks_never_interleave_across_sentences():
    """Sentence 2's chunks must never emit before sentence 1's chunks have
    all been released, even if sentence 2's generator produces faster.
    """
    emitted: list[AudioChunk] = []
    release_first = asyncio.Event()

    def synthesize_stream(text: str):
        if text == "first":
            # Block until released, then yield two chunks.
            import time

            while not release_first.is_set():
                time.sleep(0.005)
            yield _wav_bytes("first-0")
            yield _wav_bytes("first-1")
        else:
            yield _wav_bytes("second-0")
            yield _wav_bytes("second-1")

    async def emit(chunk: AudioChunk) -> None:
        emitted.append(chunk)

    def synthesize(text: str) -> bytes:
        raise AssertionError("whole-sentence synthesize should not be used")

    pipeline = TTSPipeline(
        synthesize, emit, workers=2, synthesize_stream=synthesize_stream
    )
    pipeline.submit(1, "first")
    pipeline.submit(2, "second")

    await asyncio.sleep(0.05)
    assert emitted == []  # sentence 2 finished synthesizing but can't emit yet

    release_first.set()
    await pipeline.close()

    seqs = [c.seq for c in emitted]
    assert seqs == sorted(seqs)
    # First two chunks belong to seq 1 (values in [1, 2)); last two to seq 2.
    assert all(1 <= s < 2 for s in seqs[:2])
    assert all(2 <= s < 3 for s in seqs[2:])


@pytest.mark.asyncio
async def test_streaming_generator_failure_does_not_stall_later_sentences():
    def synthesize_stream(text: str):
        if text == "boom":
            raise RuntimeError("synthesis blew up")
        yield _wav_bytes(text)

    emitted: list[AudioChunk] = []

    async def emit(chunk: AudioChunk) -> None:
        emitted.append(chunk)

    def synthesize(text: str) -> bytes:
        raise AssertionError("whole-sentence synthesize should not be used")

    pipeline = TTSPipeline(synthesize, emit, synthesize_stream=synthesize_stream)
    pipeline.submit(1, "boom")
    pipeline.submit(2, "ok")
    await pipeline.close()

    assert [c.seq for c in emitted] == [2]


@pytest.mark.asyncio
async def test_streaming_empty_generator_advances_past_seq():
    def synthesize_stream(text: str):
        if text == "empty":
            return
            yield  # pragma: no cover - never reached, makes this a generator

        yield _wav_bytes(text)

    emitted: list[AudioChunk] = []

    async def emit(chunk: AudioChunk) -> None:
        emitted.append(chunk)

    def synthesize(text: str) -> bytes:
        raise AssertionError("whole-sentence synthesize should not be used")

    pipeline = TTSPipeline(synthesize, emit, synthesize_stream=synthesize_stream)
    pipeline.submit(1, "empty")
    pipeline.submit(2, "ok")
    await pipeline.close()

    assert [c.seq for c in emitted] == [2]


@pytest.mark.asyncio
async def test_pipeline_without_synthesize_stream_uses_whole_sentence_path():
    """No synthesize_stream supplied -> exactly the pre-existing behavior:
    one call to `synthesize`, one emitted chunk per submitted seq.
    """
    calls: list[str] = []

    def synthesize(text: str) -> bytes:
        calls.append(text)
        return _wav_bytes(text)

    emitted: list[AudioChunk] = []

    async def emit(chunk: AudioChunk) -> None:
        emitted.append(chunk)

    pipeline = TTSPipeline(synthesize, emit)
    pipeline.submit(1, "only")
    pipeline.submit(2, "second")
    await pipeline.close()

    assert calls == ["only", "second"]
    assert [c.seq for c in emitted] == [1, 2]


# ---------------------------------------------------------------------------
# CachedTTSProvider.synthesize_stream
# ---------------------------------------------------------------------------


class FakeStreamingTTSProvider:
    """Fake provider exposing synthesize_stream, mirroring Qwen3TTSProvider."""

    def __init__(self, cache_signature: str = "fake-streaming-provider"):
        self.cache_signature = cache_signature
        self.stream_calls: list[str] = []

    def synthesize_stream(
        self, text: str, *, speaking_rate: float = 1.0, pitch_shift: float = 0.0
    ):
        self.stream_calls.append(text)
        yield _wav_bytes(f"{text}-a")
        yield _wav_bytes(f"{text}-b")


def test_cache_miss_streams_chunks_and_writes_assembled_cache(tmp_path):
    inner = FakeStreamingTTSProvider()
    cached = CachedTTSProvider(inner, cache_dir=str(tmp_path))

    assert hasattr(cached, "synthesize_stream")

    chunks = list(cached.synthesize_stream("hello"))
    assert len(chunks) == 2
    assert inner.stream_calls == ["hello"]

    cache_files = list(tmp_path.glob("*.wav"))
    assert len(cache_files) == 1
    with wave.open(str(cache_files[0]), "rb") as wav_file:
        assert wav_file.getnframes() > 0


def test_cache_hit_yields_single_assembled_chunk_without_provider_call(tmp_path):
    inner = FakeStreamingTTSProvider()
    cached = CachedTTSProvider(inner, cache_dir=str(tmp_path))

    # First call: cache miss, streams and writes the cache.
    first_chunks = list(cached.synthesize_stream("hello"))
    assert len(inner.stream_calls) == 1

    # Second call: should hit cache, yield exactly one chunk, and not touch
    # the wrapped provider's streaming API again.
    second_chunks = list(cached.synthesize_stream("hello"))
    assert len(inner.stream_calls) == 1
    assert len(second_chunks) == 1

    # The single cached chunk should contain the same audio content as the
    # assembled version of the original two chunks.
    with wave.open(io.BytesIO(second_chunks[0]), "rb") as wav_file:
        cached_frames = wav_file.readframes(wav_file.getnframes())
    original_frames = b"".join(
        wave.open(io.BytesIO(c), "rb").readframes(
            wave.open(io.BytesIO(c), "rb").getnframes()
        )
        for c in first_chunks
    )
    assert cached_frames == original_frames


def test_long_text_bypasses_cache_for_streaming(tmp_path):
    inner = FakeStreamingTTSProvider()
    cached = CachedTTSProvider(inner, cache_dir=str(tmp_path), max_chars=5)

    long_text = "this text is definitely longer than five characters"
    list(cached.synthesize_stream(long_text))
    list(cached.synthesize_stream(long_text))

    assert len(inner.stream_calls) == 2
    assert list(tmp_path.glob("*.wav")) == []


def test_synthesize_stream_absent_when_inner_lacks_it(tmp_path):
    class FakeTTSProvider:
        def __init__(self):
            self.cache_signature = "no-stream"

        def synthesize(self, text, *, speaking_rate=1.0, pitch_shift=0.0):
            return _wav_bytes(text)

    cached = CachedTTSProvider(FakeTTSProvider(), cache_dir=str(tmp_path))
    assert not hasattr(cached, "synthesize_stream")
    assert getattr(cached, "synthesize_stream", None) is None


# ---------------------------------------------------------------------------
# WS integration
# ---------------------------------------------------------------------------


class ScriptedProvider(DummyProvider):
    """A DummyProvider whose scripted_reply streams like a real model."""


class ChunkCapableFakeTTS(TTSProvider):
    """Fake TTS exposing both synthesize and synthesize_stream, so tests can
    assert on how many audio frames each sentence produces.
    """

    def __init__(self, chunks_per_sentence: int = 3) -> None:
        self.chunks_per_sentence = chunks_per_sentence
        self.synthesize_calls: list[str] = []
        self.stream_calls: list[str] = []

    def synthesize(
        self, text: str, *, speaking_rate: float = 1.0, pitch_shift: float = 0.0
    ) -> bytes:
        self.synthesize_calls.append(text)
        return _wav_bytes(text)

    def synthesize_stream(
        self, text: str, *, speaking_rate: float = 1.0, pitch_shift: float = 0.0
    ):
        self.stream_calls.append(text)
        for i in range(self.chunks_per_sentence):
            yield _wav_bytes(f"{text}-{i}")


class NonStreamingFakeTTS(TTSProvider):
    """Fake TTS with only the plain synthesize() method (no streaming)."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def synthesize(
        self, text: str, *, speaking_rate: float = 1.0, pitch_shift: float = 0.0
    ) -> bytes:
        self.calls.append(text)
        return _wav_bytes(text)


def _streaming_config(
    tmp_path,
    *,
    chunked_tts: bool,
    first_clause_split: bool = True,
    full_reply: bool = False,
) -> Config:
    config = Config()
    config.web.streaming_enabled = True
    config.web.streaming_tagged_output = True
    config.web.streaming_first_clause_split = first_clause_split
    config.web.streaming_chunked_tts = chunked_tts
    # Most tests here exercise the per-sentence pipeline; full-reply mode
    # (the default) coalesces the whole turn into one unit, so tests opt in
    # explicitly.
    config.web.streaming_tts_full_reply = full_reply
    config.audio.enable_voice = True
    config.memory.persist_path = str(tmp_path / "memory.json")
    config.memory.embeddings_path = str(tmp_path / "embeddings.json")
    config.memory.enable_long_term = False
    return config


async def _run_stream(app, payload: dict):
    store = app.state.session_store
    session_id, session = store.get(payload.get("session_id"), payload.get("user_id"))
    frames = [{"type": "session", "session_id": session_id}]
    cancel_event = threading.Event()

    async def emit(frame):
        frames.append(frame)

    request = ChatRequest.model_validate(payload)
    await session.handle_stream(request, emit, cancel_event)
    return session_id, frames


def _build_app(monkeypatch, config, scripted, fake_tts):
    import amadeus.web.app as app_module

    monkeypatch.setattr(app_module, "load_llm", lambda cfg: scripted)
    monkeypatch.setattr(app_module, "load_tts", lambda cfg: fake_tts)
    return create_app(config)


@pytest.mark.asyncio
async def test_chunked_tts_produces_multiple_audio_frames_per_sentence(
    monkeypatch, tmp_path
):
    config = _streaming_config(tmp_path, chunked_tts=True)
    scripted = ScriptedProvider(
        scripted_reply=(
            "[happy] Hmph, fine. || はいはい、わかったわよ。\n"
            "[relaxed] It also supports memory consolidation. || 記憶の定着も支援する。"
        )
    )
    fake_tts = ChunkCapableFakeTTS(chunks_per_sentence=3)
    app = _build_app(monkeypatch, config, scripted, fake_tts)

    _session_id, frames = await _run_stream(app, {"message": "Hi", "user_id": "ct-1"})

    audio_frames = [f for f in frames if f["type"] == "audio"]
    # 2 sentences x 3 chunks each = 6 audio frames, strictly ascending seq.
    assert len(audio_frames) == 6
    seqs = [f["seq"] for f in audio_frames]
    assert seqs == sorted(seqs)
    # First sentence's chunks are all in [1, 2); second's in [2, 3).
    assert all(1 <= s < 2 for s in seqs[:3])
    assert all(2 <= s < 3 for s in seqs[3:])

    # The streaming API was used, not the whole-sentence synthesize().
    assert fake_tts.stream_calls == [
        "はいはい、わかったわよ。",
        "記憶の定着も支援する。",
    ]
    assert fake_tts.synthesize_calls == []


@pytest.mark.asyncio
async def test_chunked_tts_records_first_chunk_latency_telemetry(monkeypatch, tmp_path):
    """Streaming synthesis measures time-to-first-chunk (the number that
    matters for perceived latency), not the whole-unit synth duration."""
    from amadeus import telemetry

    telemetry.reset()
    try:
        config = _streaming_config(tmp_path, chunked_tts=True)
        scripted = ScriptedProvider(
            scripted_reply="[happy] Hmph, fine. || はいはい、わかったわよ。"
        )
        fake_tts = ChunkCapableFakeTTS(chunks_per_sentence=2)
        app = _build_app(monkeypatch, config, scripted, fake_tts)

        await _run_stream(app, {"message": "Hi", "user_id": "ct-lat"})

        snap = telemetry.snapshot()
        assert snap.get("latency.tts_first_chunk_count", 0) >= 1
        assert snap.get("latency.tts_first_chunk_ms_sum", -1) >= 0
        assert snap.get("latency.last_tts_first_chunk_ms", -1) >= 0
        # The whole-unit counter belongs to the non-streaming path only.
        assert "latency.tts_synth_count" not in snap
    finally:
        telemetry.reset()


@pytest.mark.asyncio
async def test_chunked_tts_suppresses_first_clause_split(monkeypatch, tmp_path):
    """When both streaming_chunked_tts and streaming_first_clause_split are
    on and the provider can stream, chunking wins and the first-clause split
    never fires (it would be redundant).
    """
    config = _streaming_config(tmp_path, chunked_tts=True, first_clause_split=True)
    scripted = ScriptedProvider(
        scripted_reply=(
            "[happy] Hmph, fine. || はいはい、わかったわよ。\n"
            "[relaxed] It also supports memory consolidation. || 記憶の定着も支援する。"
        )
    )
    fake_tts = ChunkCapableFakeTTS(chunks_per_sentence=2)
    app = _build_app(monkeypatch, config, scripted, fake_tts)

    await _run_stream(app, {"message": "Hi", "user_id": "ct-2"})

    # The whole first sentence went through synthesize_stream as one call,
    # not split into a clause + remainder pair.
    assert fake_tts.stream_calls[0] == "はいはい、わかったわよ。"


@pytest.mark.asyncio
async def test_chunked_tts_disabled_falls_back_to_whole_sentence(monkeypatch, tmp_path):
    # Also disable the first-clause split so this test isolates the chunked
    # TTS flag: with both off, a single sentence must resolve to exactly one
    # whole-sentence synthesize() call and one audio frame.
    config = _streaming_config(tmp_path, chunked_tts=False, first_clause_split=False)
    scripted = ScriptedProvider(
        scripted_reply="[happy] Hmph, fine. || はいはい、わかったわよ。"
    )
    fake_tts = ChunkCapableFakeTTS(chunks_per_sentence=3)
    app = _build_app(monkeypatch, config, scripted, fake_tts)

    _session_id, frames = await _run_stream(app, {"message": "Hi", "user_id": "ct-3"})

    audio_frames = [f for f in frames if f["type"] == "audio"]
    # Flag off -> whole-sentence synthesize(), even though the provider is
    # capable of streaming.
    assert fake_tts.stream_calls == []
    assert fake_tts.synthesize_calls == ["はいはい、わかったわよ。"]
    assert len(audio_frames) == 1


@pytest.mark.asyncio
async def test_non_streaming_provider_unaffected_by_chunked_tts_flag(
    monkeypatch, tmp_path
):
    """Provider without synthesize_stream: flag on or off, behavior is
    identical to the pre-existing whole-sentence path.
    """
    config = _streaming_config(tmp_path, chunked_tts=True, first_clause_split=True)
    scripted = ScriptedProvider(
        scripted_reply=(
            "[happy] Hmph, fine. || はいはい、わかったわよ。\n"
            "[relaxed] Second sentence here. || 二番目の文です。"
        )
    )
    fake_tts = NonStreamingFakeTTS()
    app = _build_app(monkeypatch, config, scripted, fake_tts)

    _session_id, frames = await _run_stream(app, {"message": "Hi", "user_id": "ct-4"})

    audio_frames = [f for f in frames if f["type"] == "audio"]
    # First-clause split still applies (provider can't stream, so chunking
    # doesn't suppress it): sentence 1 splits into 1 + 1.5, sentence 2 is 2.
    assert [f["seq"] for f in audio_frames] == [1, 1.5, 2]
    assert fake_tts.calls[0] == "はいはい、"
    assert fake_tts.calls[1] == "わかったわよ。"


@pytest.mark.asyncio
async def test_full_reply_tts_synthesizes_whole_turn_as_one_unit(monkeypatch, tmp_path):
    """streaming_tts_full_reply (the default): the whole reply is one
    zero-shot generation — no per-sentence voice-reset boundaries — and the
    first-clause split never fires.
    """
    config = _streaming_config(
        tmp_path, chunked_tts=True, first_clause_split=True, full_reply=True
    )
    scripted = ScriptedProvider(
        scripted_reply=(
            "[happy] Hmph, fine. || はいはい、わかったわよ。\n"
            "[relaxed] It also supports memory consolidation. || 記憶の定着も支援する。"
        )
    )
    fake_tts = ChunkCapableFakeTTS(chunks_per_sentence=3)
    app = _build_app(monkeypatch, config, scripted, fake_tts)

    _session_id, frames = await _run_stream(app, {"message": "Hi", "user_id": "fr-1"})

    # Exactly one generation over the joined reply text.
    assert fake_tts.stream_calls == ["はいはい、わかったわよ。 記憶の定着も支援する。"]
    assert fake_tts.synthesize_calls == []
    # Audio still streams out in chunks from that single generation.
    audio_frames = [f for f in frames if f["type"] == "audio"]
    assert len(audio_frames) == 3
    seqs = [f["seq"] for f in audio_frames]
    assert seqs == sorted(seqs)
    # Sentence frames remain per-sentence for display even though audio is
    # a single unit.
    sentence_frames = [f for f in frames if f["type"] == "sentence"]
    assert [f["seq"] for f in sentence_frames] == [1, 2]


@pytest.mark.asyncio
async def test_full_reply_tts_max_unit_chars_splits_long_replies(monkeypatch, tmp_path):
    """The safety cap: a reply longer than streaming_tts_max_unit_chars is
    submitted as multiple units, split at sentence boundaries.
    """
    config = _streaming_config(
        tmp_path, chunked_tts=True, first_clause_split=False, full_reply=True
    )
    config.web.streaming_tts_max_unit_chars = 20
    scripted = ScriptedProvider(
        scripted_reply=(
            "[happy] One. || はいはい、わかったわよ。\n"
            "[relaxed] Two. || 記憶の定着も支援する。\n"
            "[neutral] Three. || 三番目の文です。"
        )
    )
    fake_tts = ChunkCapableFakeTTS(chunks_per_sentence=1)
    app = _build_app(monkeypatch, config, scripted, fake_tts)

    await _run_stream(app, {"message": "Hi", "user_id": "fr-2"})

    # Sentences 1+2 exceed the 20-char cap together, so they flush as one
    # unit; sentence 3 flushes at close.
    assert fake_tts.stream_calls == [
        "はいはい、わかったわよ。 記憶の定着も支援する。",
        "三番目の文です。",
    ]
