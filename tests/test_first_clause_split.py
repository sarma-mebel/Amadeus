"""Tests for the first-clause TTS fast path (time-to-first-audio reduction).

Covers the pure splitter function (``split_first_clause``) and the
``TTSPipeline`` / WS integration behavior when the first sentence of a turn
is split into a short first chunk + remainder for faster playback start.
"""

from __future__ import annotations

import asyncio
import threading

import pytest

from amadeus.config import Config
from amadeus.providers.llm.base import DummyProvider
from amadeus.providers.tts.base import TTSProvider
from amadeus.web.app import ChatRequest, create_app
from amadeus.web.streaming import AudioChunk, TTSPipeline, split_first_clause

# ---------------------------------------------------------------------------
# split_first_clause() unit tests
# ---------------------------------------------------------------------------


def test_split_boundary_within_range():
    first, rest = split_first_clause("はいはい、わかったわよ。")
    assert first == "はいはい、"
    assert rest == "わかったわよ。"


def test_split_no_early_boundary_falls_back_to_length_split():
    # No punctuation at all, long enough to force the length-based split.
    text = "あ" * 20
    first, rest = split_first_clause(text)
    assert first == "あ" * 12
    assert rest == "あ" * 8


def test_split_short_sentence_not_split():
    text = "短い返事です。"  # 8 chars total, boundary is trailing punctuation
    first, rest = split_first_clause(text)
    assert first == text
    assert rest == ""


def test_split_short_sentence_no_punctuation_not_split():
    text = "みじかい"  # 4 chars, below the split threshold
    first, rest = split_first_clause(text)
    assert first == text
    assert rest == ""


def test_split_empty_input():
    assert split_first_clause("") == ("", "")
    assert split_first_clause("   ") == ("   ", "")


def test_split_boundary_too_early_is_ignored_if_remainder_is_only_punctuation():
    # A boundary that lands exactly at the end (nothing left to speak after
    # it) must not produce an empty remainder chunk.
    text = "それでいい。"
    first, rest = split_first_clause(text)
    assert first == text
    assert rest == ""


def test_split_boundary_beyond_max_len_uses_length_fallback():
    # Comma only appears after the max clause-boundary window; long sentence
    # should still hard-split at the fallback length rather than waiting for
    # the far-away boundary.
    text = "あ" * 25 + "、" + "い" * 5
    first, rest = split_first_clause(text)
    assert first == "あ" * 12
    assert rest == "あ" * 13 + "、" + "い" * 5


# ---------------------------------------------------------------------------
# TTSPipeline: fractional-seq ordering behavior
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_pipeline_orders_fractional_seq_chunks_correctly():
    """A first-clause split (seq=1, seq=1.5) must be emitted before seq=2,
    and in the right internal order, even if synthesis completes out of
    order (simulated by making seq=1.5 resolve before seq=1).
    """
    emitted: list[AudioChunk] = []
    release_first = asyncio.Event()

    async def synthesize(text: str) -> bytes:
        if text == "first":
            await release_first.wait()
        return text.encode()

    async def emit(chunk: AudioChunk) -> None:
        emitted.append(chunk)

    pipeline = TTSPipeline(synthesize, emit, workers=2)
    pipeline.submit(1, "first")
    pipeline.submit(1.5, "second")
    pipeline.submit(2, "third")

    # Give the "second"/"third" workers a chance to finish first.
    await asyncio.sleep(0.05)
    assert emitted == []  # nothing released yet: seq=1 hasn't resolved

    release_first.set()
    await pipeline.close()

    assert [c.seq for c in emitted] == [1, 1.5, 2]
    assert [c.data_b64 for c in emitted] == [
        __import__("base64").b64encode(b"first").decode("ascii"),
        __import__("base64").b64encode(b"second").decode("ascii"),
        __import__("base64").b64encode(b"third").decode("ascii"),
    ]


@pytest.mark.asyncio
async def test_pipeline_single_seq_unaffected_by_fractional_support():
    """Sentences that aren't split (no seq+0.5 submitted) behave exactly as
    before: one submit, one emitted chunk, integer seq.
    """
    emitted: list[AudioChunk] = []

    async def synthesize(text: str) -> bytes:
        return text.encode()

    async def emit(chunk: AudioChunk) -> None:
        emitted.append(chunk)

    pipeline = TTSPipeline(synthesize, emit, workers=1)
    pipeline.submit(1, "only")
    pipeline.submit(2, "second sentence")
    await pipeline.close()

    assert [c.seq for c in emitted] == [1, 2]


# ---------------------------------------------------------------------------
# WS integration: first sentence produces two audio frames, later sentences
# unaffected, config flag off restores old behavior.
# ---------------------------------------------------------------------------


class ScriptedProvider(DummyProvider):
    """A DummyProvider whose scripted_reply streams like a real model."""


class SlowFakeTTS(TTSProvider):
    """Fake TTS that tags calls so tests can assert on chunk boundaries."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def synthesize(
        self, text: str, *, speaking_rate: float = 1.0, pitch_shift: float = 0.0
    ) -> bytes:
        self.calls.append(text)
        return f"WAV:{text}".encode()


def _streaming_config(tmp_path, *, first_clause_split: bool) -> Config:
    config = Config()
    config.web.streaming_enabled = True
    config.web.streaming_tagged_output = True
    config.web.streaming_first_clause_split = first_clause_split
    # These tests exercise the per-sentence clause-split machinery, which
    # full-reply TTS (the default) supersedes — opt out of it here.
    config.web.streaming_tts_full_reply = False
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
async def test_first_sentence_splits_into_two_audio_frames(monkeypatch, tmp_path):
    config = _streaming_config(tmp_path, first_clause_split=True)
    scripted = ScriptedProvider(
        scripted_reply=(
            "[happy] Hmph, fine. || はいはい、わかったわよ。\n"
            "[relaxed] It also supports memory consolidation. || 記憶の定着も支援する。"
        )
    )
    fake_tts = SlowFakeTTS()
    app = _build_app(monkeypatch, config, scripted, fake_tts)

    _session_id, frames = await _run_stream(app, {"message": "Hi", "user_id": "fc-1"})

    audio_frames = [f for f in frames if f["type"] == "audio"]
    # First sentence's Japanese has an early clause boundary -> splits into
    # two chunks (seq 1 and 1.5); second sentence submits as plain seq=2.
    assert [f["seq"] for f in audio_frames] == [1, 1.5, 2]

    # TTS received the split halves as separate calls, in order, followed by
    # the unsplit second sentence.
    assert fake_tts.calls[0] == "はいはい、"
    assert fake_tts.calls[1] == "わかったわよ。"
    assert fake_tts.calls[2] == "記憶の定着も支援する。"

    # Audio frames must still arrive strictly in ascending seq order overall.
    seqs = [f["seq"] for f in audio_frames]
    assert seqs == sorted(seqs)


@pytest.mark.asyncio
async def test_short_first_sentence_not_split(monkeypatch, tmp_path):
    config = _streaming_config(tmp_path, first_clause_split=True)
    scripted = ScriptedProvider(
        scripted_reply=(
            "[happy] Sure. || うん。\n"
            "[relaxed] It also supports memory consolidation. || 記憶の定着も支援する。"
        )
    )
    fake_tts = SlowFakeTTS()
    app = _build_app(monkeypatch, config, scripted, fake_tts)

    _session_id, frames = await _run_stream(app, {"message": "Hi", "user_id": "fc-2"})

    audio_frames = [f for f in frames if f["type"] == "audio"]
    assert [f["seq"] for f in audio_frames] == [1, 2]
    assert fake_tts.calls[0] == "うん。"


@pytest.mark.asyncio
async def test_config_flag_off_restores_old_behavior(monkeypatch, tmp_path):
    config = _streaming_config(tmp_path, first_clause_split=False)
    scripted = ScriptedProvider(
        scripted_reply=(
            "[happy] Hmph, fine. || はいはい、わかったわよ。\n"
            "[relaxed] It also supports memory consolidation. || 記憶の定着も支援する。"
        )
    )
    fake_tts = SlowFakeTTS()
    app = _build_app(monkeypatch, config, scripted, fake_tts)

    _session_id, frames = await _run_stream(app, {"message": "Hi", "user_id": "fc-3"})

    audio_frames = [f for f in frames if f["type"] == "audio"]
    assert [f["seq"] for f in audio_frames] == [1, 2]
    assert fake_tts.calls[0] == "はいはい、わかったわよ。"


@pytest.mark.asyncio
async def test_later_sentences_never_split_even_if_short_clause_early(
    monkeypatch, tmp_path
):
    """Only sentence seq==1 is eligible for the split; a second sentence
    with an early clause boundary must still go through as a single chunk.
    """
    config = _streaming_config(tmp_path, first_clause_split=True)
    scripted = ScriptedProvider(
        scripted_reply=(
            "[happy] Hmph, fine. || はいはい、わかったわよ。\n"
            "[relaxed] Yes, indeed. || そうね、その通りよ。"
        )
    )
    fake_tts = SlowFakeTTS()
    app = _build_app(monkeypatch, config, scripted, fake_tts)

    _session_id, frames = await _run_stream(app, {"message": "Hi", "user_id": "fc-4"})

    audio_frames = [f for f in frames if f["type"] == "audio"]
    # First sentence splits (1, 1.5); second sentence, despite having an
    # early clause boundary too, stays a single seq=2 chunk.
    assert [f["seq"] for f in audio_frames] == [1, 1.5, 2]
    assert fake_tts.calls[2] == "そうね、その通りよ。"
