"""Speculative recall (memory.speculative_recall) + recall duration telemetry.

Covers the MemoryManager stash semantics (hit / miss / flag-off / TTL /
replace), augmentation parity with assemble_turn_context, the SessionStore
STT-wrapper wiring, and the memory.recall_ms duration recorded around
per-turn context assembly. Everything runs on local fakes — no live API
calls, no model loads.
"""

from __future__ import annotations

import time
from typing import List, Sequence

import pytest

from amadeus import telemetry
from amadeus.config import Config
from amadeus.engine import ChatEngine
from amadeus.memory.context import assemble_turn_context
from amadeus.memory.manager import EmbeddingProvider, MemoryManager
from amadeus.memory.vector_store import VectorRecord
from amadeus.providers.llm.base import DummyProvider


class CountingEmbedder(EmbeddingProvider):
    """Deterministic 2-d embedder that records every embed_text call."""

    def __init__(self) -> None:
        self.calls: List[str] = []

    def embed_text(self, text: str) -> List[float]:
        self.calls.append(text)
        return [1.0, 0.0]

    def similarity(self, a: Sequence[float], b: Sequence[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5
        return dot / ((norm_a * norm_b) or 1.0)


def _config(tmp_path, **overrides) -> Config:
    config = Config()
    config.memory.persist_path = None
    config.memory.enable_long_term = True
    config.memory.embeddings_path = str(tmp_path / "vectors.json")
    config.memory.dedupe_threshold = 0.0
    config.memory.recall_min_query_chars = 0
    # Extraction ON (with no summary provider it never runs an LLM call and,
    # with profile_path=None, never builds a profile) so add_message skips
    # the legacy raw-message long-term capture — otherwise every user
    # message costs a background embed and the embed-count assertions below
    # get racy. The only embeds these tests see are recall query embeds.
    config.memory.enable_fact_extraction = True
    config.memory.enable_episodes = False
    config.memory.profile_path = None
    config.memory.recall_summary_dedupe = 0.0
    config.memory.recall_recent_dedupe = 0.0
    for key, value in overrides.items():
        setattr(config.memory, key, value)
    return config


def _seed_fact(manager: MemoryManager, text: str = "the user loves cats") -> None:
    manager.vector_store.add(
        VectorRecord(
            vector=[1.0, 0.0],
            text=text,
            metadata={"type": "fact", "strength": 1.0},
        )
    )


@pytest.fixture(autouse=True)
def _clean_telemetry():
    telemetry.reset()
    yield
    telemetry.reset()


# ---------------------------------------------------------------------------
# MemoryManager stash semantics
# ---------------------------------------------------------------------------


def test_speculative_hit_consumes_future_one_recall(tmp_path) -> None:
    embedder = CountingEmbedder()
    manager = MemoryManager(_config(tmp_path), embedder=embedder)
    _seed_fact(manager)

    query = "what do you think about my cats today"
    future = manager.speculate_recall(query)
    assert future is not None
    future.result(timeout=5)
    assert len(embedder.calls) == 1  # speculative query embed ran

    matches = manager.recall_detailed(query)
    assert [m.text for m in matches] == ["the user loves cats"]
    # Consumed the pre-started future: no second embed, one hit, no misses.
    assert len(embedder.calls) == 1
    snap = telemetry.snapshot()
    assert snap.get("memory.speculative_recall_started") == 1
    assert snap.get("memory.speculative_recall_hits") == 1
    assert "memory.speculative_recall_misses" not in snap
    assert manager._speculative is None


def test_speculative_miss_on_different_text_runs_serial_recall(tmp_path) -> None:
    embedder = CountingEmbedder()
    manager = MemoryManager(_config(tmp_path), embedder=embedder)
    _seed_fact(manager)

    future = manager.speculate_recall("what do you think about my cats today")
    future.result(timeout=5)

    matches = manager.recall_detailed("something entirely different came up")
    assert [m.text for m in matches] == ["the user loves cats"]
    # Serial path ran its own embed on top of the speculative one.
    assert len(embedder.calls) == 2
    snap = telemetry.snapshot()
    assert snap.get("memory.speculative_recall_misses") == 1
    assert "memory.speculative_recall_hits" not in snap


def test_speculative_mismatch_leaves_stash_for_the_real_turn(tmp_path) -> None:
    """An unrelated recall (forget path, memory endpoints) between STT and
    the chat turn must not eat the chat turn's future."""
    embedder = CountingEmbedder()
    manager = MemoryManager(_config(tmp_path), embedder=embedder)
    _seed_fact(manager)

    query = "what do you think about my cats today"
    manager.speculate_recall(query).result(timeout=5)

    manager.recall_detailed("unrelated interleaved lookup", limit=1)  # miss
    matches = manager.recall_detailed(query)  # still a hit
    assert [m.text for m in matches] == ["the user loves cats"]
    snap = telemetry.snapshot()
    assert snap.get("memory.speculative_recall_misses") == 1
    assert snap.get("memory.speculative_recall_hits") == 1


def test_speculative_flag_off_is_inert(tmp_path) -> None:
    embedder = CountingEmbedder()
    manager = MemoryManager(
        _config(tmp_path, speculative_recall=False), embedder=embedder
    )
    _seed_fact(manager)

    assert manager.speculate_recall("what do you think about my cats") is None
    assert manager._speculative is None

    matches = manager.recall_detailed("what do you think about my cats")
    assert [m.text for m in matches] == ["the user loves cats"]
    snap = telemetry.snapshot()
    assert "memory.speculative_recall_started" not in snap
    assert "memory.speculative_recall_hits" not in snap
    assert "memory.speculative_recall_misses" not in snap


def test_speculative_expired_entry_counts_miss_and_falls_back(tmp_path) -> None:
    embedder = CountingEmbedder()
    manager = MemoryManager(
        _config(tmp_path, speculative_recall_ttl_seconds=0.01), embedder=embedder
    )
    _seed_fact(manager)

    query = "what do you think about my cats today"
    manager.speculate_recall(query).result(timeout=5)
    time.sleep(0.05)

    matches = manager.recall_detailed(query)
    assert [m.text for m in matches] == ["the user loves cats"]
    assert len(embedder.calls) == 2  # serial re-run after expiry
    snap = telemetry.snapshot()
    assert snap.get("memory.speculative_recall_misses") == 1
    assert "memory.speculative_recall_hits" not in snap
    assert manager._speculative is None


def test_speculative_new_transcript_replaces_previous(tmp_path) -> None:
    embedder = CountingEmbedder()
    manager = MemoryManager(_config(tmp_path), embedder=embedder)
    _seed_fact(manager)

    manager.speculate_recall("first attempt at saying something")
    second = manager.speculate_recall("second and final transcription here")
    second.result(timeout=5)
    assert manager._speculative is not None
    assert manager._speculative.future is second

    matches = manager.recall_detailed("second and final transcription here")
    assert [m.text for m in matches] == ["the user loves cats"]
    assert telemetry.snapshot().get("memory.speculative_recall_hits") == 1
    assert manager._speculative is None


def test_speculative_skips_when_long_term_disabled(tmp_path) -> None:
    manager = MemoryManager(
        _config(tmp_path, enable_long_term=False), embedder=CountingEmbedder()
    )
    assert manager.speculate_recall("anything at all really") is None


# ---------------------------------------------------------------------------
# Augmentation parity with assemble_turn_context (the real consume path)
# ---------------------------------------------------------------------------


def test_assemble_turn_context_consumes_speculative_future(tmp_path) -> None:
    embedder = CountingEmbedder()
    manager = MemoryManager(_config(tmp_path), embedder=embedder)
    _seed_fact(manager)

    text = "what should I feed my cats"
    manager.speculate_recall(text).result(timeout=5)

    context, recall_items = assemble_turn_context(manager, "sys prompt", text)
    assert recall_items == ["the user loves cats"]
    assert len(embedder.calls) == 1
    assert telemetry.snapshot().get("memory.speculative_recall_hits") == 1
    # The recall framing block still rides the context as usual.
    assert any("the user loves cats" in m.content for m in context)


def test_assemble_parity_with_query_augmentation(tmp_path) -> None:
    """A short follow-up gets the previous user turn prepended to the recall
    query on BOTH sides (speculation runs before add_message; assembly runs
    after), so the keys still match."""
    embedder = CountingEmbedder()
    manager = MemoryManager(_config(tmp_path), embedder=embedder)
    _seed_fact(manager)
    manager.add_message("user", "I adopted three cats from the shelter last month")
    manager.add_message("assistant", "Noted. Chaos incoming.")

    short_followup = "did it work"
    manager.speculate_recall(short_followup).result(timeout=5)
    assert len(embedder.calls) == 1
    # The speculative query embedded the augmented form.
    assert "I adopted three cats" in embedder.calls[0]

    _, recall_items = assemble_turn_context(manager, "sys", short_followup)
    assert recall_items == ["the user loves cats"]
    assert len(embedder.calls) == 1
    assert telemetry.snapshot().get("memory.speculative_recall_hits") == 1


# ---------------------------------------------------------------------------
# SessionStore / STT wrapper wiring
# ---------------------------------------------------------------------------


class FakeSTT:
    def __init__(self, text: str) -> None:
        self.text = text
        self.calls: List[tuple] = []

    def transcribe_audio(self, audio_data: bytes, sample_rate: int) -> str:
        self.calls.append((audio_data, sample_rate))
        return self.text


def _store_config(tmp_path) -> Config:
    config = _config(tmp_path)
    config.memory.persist_path = str(tmp_path / "memory.json")
    config.memory.profile_path = str(tmp_path / "profile.json")
    config.memory.transcripts_path = str(tmp_path / "transcripts.json")
    config.audio.enable_voice = False
    return config


def _patched_store(tmp_path, monkeypatch, config, stt=None):
    import amadeus.web.app as app_module

    from amadeus.web.sessions import SessionStore

    monkeypatch.setattr(app_module, "load_llm", lambda cfg: DummyProvider())
    monkeypatch.setattr(app_module, "load_tts", lambda cfg: None)
    monkeypatch.setattr(app_module, "load_embedder", lambda cfg: CountingEmbedder())
    monkeypatch.setattr(app_module, "load_stt", lambda cfg: stt)
    return SessionStore(config)


def test_stt_wrapper_speculates_for_most_recent_session(tmp_path, monkeypatch):
    fake = FakeSTT("hello what about my cats")
    store = _patched_store(tmp_path, monkeypatch, _store_config(tmp_path), stt=fake)
    _, session = store.get(None, None)

    stt = store.get_stt()
    assert stt.transcribe_audio(b"\x00\x00", 16000) == "hello what about my cats"
    assert fake.calls == [(b"\x00\x00", 16000)]

    stash = session.memory._speculative
    assert stash is not None
    assert stash.query_key == "hello what about my cats"
    stash.future.result(timeout=5)
    assert telemetry.snapshot().get("memory.speculative_recall_started") == 1


def test_stt_wrapper_flag_off_leaves_no_stash(tmp_path, monkeypatch):
    config = _store_config(tmp_path)
    config.memory.speculative_recall = False
    fake = FakeSTT("hello what about my cats")
    store = _patched_store(tmp_path, monkeypatch, config, stt=fake)
    _, session = store.get(None, None)

    stt = store.get_stt()
    assert stt.transcribe_audio(b"\x00\x00", 16000) == "hello what about my cats"
    assert session.memory._speculative is None
    snap = telemetry.snapshot()
    assert "memory.speculative_recall_started" not in snap


def test_stt_wrapper_no_live_session_counts_skip(tmp_path, monkeypatch):
    fake = FakeSTT("talking before any chat exists")
    store = _patched_store(tmp_path, monkeypatch, _store_config(tmp_path), stt=fake)

    stt = store.get_stt()
    assert stt.transcribe_audio(b"\x00\x00", 16000) == (
        "talking before any chat exists"
    )
    snap = telemetry.snapshot()
    assert snap.get("memory.speculative_recall_no_session") == 1
    assert "memory.speculative_recall_started" not in snap


def test_stt_wrapper_speculation_failure_never_breaks_stt(tmp_path, monkeypatch):
    fake = FakeSTT("hello what about my cats")
    store = _patched_store(tmp_path, monkeypatch, _store_config(tmp_path), stt=fake)
    store.get(None, None)

    def _boom(text):
        raise RuntimeError("speculation exploded")

    monkeypatch.setattr(store, "speculate_recall", _boom)
    stt = store.get_stt()
    assert stt.transcribe_audio(b"\x00\x00", 16000) == "hello what about my cats"
    assert telemetry.snapshot().get("memory.speculative_recall_start_failures") == 1


# ---------------------------------------------------------------------------
# memory.recall_ms duration telemetry (QW)
# ---------------------------------------------------------------------------


def test_prepare_turn_records_recall_duration(tmp_path) -> None:
    config = Config()
    config.memory.persist_path = str(tmp_path / "memory.json")
    config.memory.embeddings_path = str(tmp_path / "embeddings.json")
    config.memory.profile_path = str(tmp_path / "profile.json")
    config.memory.enable_long_term = False
    engine = ChatEngine(config, DummyProvider())

    engine.prepare_turn("hello there, how has your day been?", config.persona)

    snap = telemetry.duration_snapshot()
    assert "memory.recall_ms" in snap
    assert snap["memory.recall_ms"]["count"] == 1
    assert snap["memory.recall_ms"]["last_ms"] >= 0
