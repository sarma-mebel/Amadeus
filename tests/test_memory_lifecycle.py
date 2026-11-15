"""Fact store lifecycle: archive-not-delete, provenance, cap eviction,
truncation, the model-facing contract methods (find_facts/archive_fact/
store_user_fact), the summary/eviction race fix, extraction prompt
hardening, and the pre-augmentation min_query_chars gate.

All paths isolated to tmp_path (per conftest's repo-root leak guard); no
live provider calls — everything rides the local fakes already used by
test_memory.py (VectorEmbedder, FakeLLMProvider).
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import List

import pytest

from amadeus import telemetry
from amadeus.config import Config
from amadeus.memory.manager import MemoryManager
from amadeus.memory.vector_store import VectorRecord

from test_memory import (
    EXTRACTION_JSON,
    FakeLLMProvider,
    VectorEmbedder,
    _extraction_config,
    _iso_days_ago,
    _wait_for,
)


@pytest.fixture(autouse=True)
def _clean_telemetry():
    telemetry.reset()
    yield
    telemetry.reset()


def _archive_lines(manager: MemoryManager) -> List[dict]:
    from amadeus.memory.fact_archive import archive_path

    path = archive_path(manager)
    assert path is not None
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


# ---------------------------------------------------------------------------
# A. Archive-not-delete
# ---------------------------------------------------------------------------


def test_forget_archives_instead_of_hard_deleting(tmp_path: Path) -> None:
    embedder = VectorEmbedder({"cilantro": (1.0, 0.0), "tea": (0.0, 1.0)})
    manager = MemoryManager(_extraction_config(tmp_path), embedder=embedder)
    manager.remember("User hates cilantro")

    forgotten = manager.forget("cilantro")
    assert forgotten == "User hates cilantro"
    # Hot store shrinks...
    assert list(manager.vector_store.dump()) == []
    # ...but the fact survives in the archive with the right reason.
    lines = _archive_lines(manager)
    assert len(lines) == 1
    assert lines[0]["text"] == "User hates cilantro"
    assert lines[0]["reason"] == "user_forget"
    assert lines[0]["archived_at"]
    assert telemetry.snapshot().get("memory.fact_archived") == 1


def test_forget_id_archives_instead_of_hard_deleting(tmp_path: Path) -> None:
    embedder = VectorEmbedder({"hiking": (1.0, 0.0), "tea": (0.0, 1.0)})
    manager = MemoryManager(_extraction_config(tmp_path), embedder=embedder)
    manager.remember("User likes hiking")
    record_id = list(manager.vector_store.dump())[0].id

    manager.forget_id(record_id)
    assert list(manager.vector_store.dump()) == []
    lines = _archive_lines(manager)
    assert len(lines) == 1
    assert lines[0]["id"] == record_id
    assert lines[0]["reason"] == "user_forget"


def test_archive_write_failure_still_removes_and_counts(
    tmp_path: Path, monkeypatch
) -> None:
    """Archive is best-effort provenance: a write failure must not block the
    removal, and must be visible via telemetry (degrade-loudly convention)."""
    from amadeus.memory import fact_archive

    embedder = VectorEmbedder({"tea": (1.0, 0.0)})
    manager = MemoryManager(_extraction_config(tmp_path), embedder=embedder)
    manager.remember("User likes tea")
    record_id = list(manager.vector_store.dump())[0].id

    # Point the archive sidecar at a path whose parent can never be created
    # (a file sitting where a directory is needed) — narrowly breaks only
    # the archive write, leaving the vector store's own persist() (a
    # different path) untouched.
    blocker = tmp_path / "blocked_file"
    blocker.write_text("occupied")
    bogus_path = blocker / "archive.jsonl"
    monkeypatch.setattr(fact_archive, "archive_path", lambda mgr: bogus_path)

    forgotten = manager.forget_id(record_id)
    assert forgotten == "User likes tea"
    # Removal proceeded despite the archive write failure.
    assert list(manager.vector_store.dump()) == []
    assert telemetry.snapshot().get("memory.fact_archive_write_failed") == 1


# ---------------------------------------------------------------------------
# C/D. Supersede provenance + hint threshold
# ---------------------------------------------------------------------------


def test_supersede_duplicate_records_previous_fact_and_archives(
    tmp_path: Path,
) -> None:
    embedder = VectorEmbedder({"likes": (1.0, 0.0)})
    manager = MemoryManager(_extraction_config(tmp_path), embedder=embedder)
    manager._store_fact("User likes tea", 0.5)
    manager._store_fact("User likes coffee now", 0.6)

    records = list(manager.vector_store.dump())
    assert len(records) == 1
    assert records[0].text == "User likes coffee now"
    assert records[0].metadata.get("supersedes")
    assert records[0].metadata.get("previous_fact") == "User likes tea"

    lines = _archive_lines(manager)
    assert len(lines) == 1
    assert lines[0]["text"] == "User likes tea"
    assert lines[0]["reason"] == "superseded_duplicate"
    assert lines[0]["id"] == records[0].metadata["supersedes"]


def test_hint_supersede_respects_configured_threshold(tmp_path: Path) -> None:
    """The hint-driven scan must use fact_supersede_hint_threshold (not a
    hardcoded 0.6): a candidate scoring between the two must supersede only
    when the configured threshold is loose enough to admit it."""

    # Three distinct texts, three distinct vectors: the NEW fact's own text
    # (embeds orthogonal to the existing candidate, so the near-duplicate
    # scan at fact_supersede_threshold never fires), the existing candidate,
    # and the "replaces" hint text — chosen so cosine(candidate, hint) is
    # exactly 0.8, letting the threshold sit cleanly above or below it.
    embedder = VectorEmbedder(
        {
            "NEWFACTMARKER": (0.0, 1.0),
            "CANDIDATEMARKER": (1.0, 0.0),
            "HINTMARKER": (0.8, 0.6),
        },
        default=(0.0, 1.0),
    )

    # High threshold: the hint candidate (cosine 0.8 against the "replaces"
    # text) must NOT supersede when the bar is set above its score.
    config = _extraction_config(tmp_path / "high", fact_supersede_hint_threshold=0.95)
    manager = MemoryManager(config, embedder=embedder)
    manager._store_fact("CANDIDATEMARKER User loves Nolan movies", 0.5)
    manager._store_fact(
        "NEWFACTMARKER User is bored of Nolan movies now",
        0.6,
        replaces="HINTMARKER User loves Nolan movies",
    )
    records = list(manager.vector_store.dump())
    assert len(records) == 2, "threshold too high to supersede; both must survive"

    # Now loosen the threshold below the hint candidate's score: must supersede.
    # Separate tmp subdirectory — reusing the same embeddings_path would let
    # manager2 load manager's already-persisted records back in.
    config2 = _extraction_config(tmp_path / "low", fact_supersede_hint_threshold=0.5)
    manager2 = MemoryManager(config2, embedder=embedder)
    manager2._store_fact("CANDIDATEMARKER User loves Nolan movies", 0.5)
    manager2._store_fact(
        "NEWFACTMARKER User is bored of Nolan movies now",
        0.6,
        replaces="HINTMARKER User loves Nolan movies",
    )
    records2 = list(manager2.vector_store.dump())
    assert len(records2) == 1
    assert (
        records2[0].metadata.get("previous_fact")
        == "CANDIDATEMARKER User loves Nolan movies"
    )
    assert telemetry.snapshot().get("memory.fact_superseded_by_hint") == 1


# ---------------------------------------------------------------------------
# E. Fact cap eviction
# ---------------------------------------------------------------------------


def test_prune_facts_over_cap_evicts_weakest_unpinned_spares_pinned(
    tmp_path: Path,
) -> None:
    embedder = VectorEmbedder()
    config = _extraction_config(tmp_path)
    config.memory.max_facts = 2
    manager = MemoryManager(config, embedder=embedder)

    # A pinned fact with the weakest strength — must survive the cap despite
    # ranking lowest, because pinned facts are exempt and don't count.
    manager.vector_store.add(
        VectorRecord(
            vector=[0.0, 1.0],
            text="pinned weak fact",
            metadata={
                "type": "fact",
                "pinned": True,
                "strength": 0.01,
                "timestamp": _iso_days_ago(400),
            },
        )
    )
    # Three unpinned facts with distinct strength.
    for name, strength in [("weakest", 0.1), ("mid", 0.5), ("strongest", 1.0)]:
        manager.vector_store.add(
            VectorRecord(
                vector=[0.0, 1.0],
                text=f"{name} fact",
                metadata={
                    "type": "fact",
                    "strength": strength,
                    "timestamp": _iso_days_ago(1),
                },
            )
        )

    manager._prune_facts_over_cap()

    remaining = {
        rec.text
        for rec in manager.vector_store.dump()
        if (rec.metadata or {}).get("type") == "fact"
    }
    assert "pinned weak fact" in remaining, "pinned fact must survive the cap"
    assert "weakest fact" not in remaining, "weakest unpinned fact must be evicted"
    assert "mid fact" in remaining
    assert "strongest fact" in remaining
    assert telemetry.snapshot().get("memory.facts_evicted") == 1

    lines = _archive_lines(manager)
    assert len(lines) == 1
    assert lines[0]["text"] == "weakest fact"
    assert lines[0]["reason"] == "evicted_cap"


def test_fact_cap_disabled_when_zero(tmp_path: Path) -> None:
    embedder = VectorEmbedder()
    config = _extraction_config(tmp_path)
    config.memory.max_facts = 0
    manager = MemoryManager(config, embedder=embedder)
    for i in range(5):
        manager.vector_store.add(
            VectorRecord(
                vector=[0.0, 1.0],
                text=f"fact {i}",
                metadata={"type": "fact", "strength": 0.01},
            )
        )
    manager._prune_facts_over_cap()
    assert manager.long_term_count == 5


def test_extraction_batch_triggers_prune_pass(tmp_path: Path) -> None:
    """After the extraction job stores a batch of new facts, one prune pass
    runs automatically when the store is over cap."""
    embedder = VectorEmbedder()
    config = _extraction_config(tmp_path, max_facts=1)
    summarizer = FakeLLMProvider(EXTRACTION_JSON)
    manager = MemoryManager(config, embedder=embedder, summary_provider=summarizer)
    # Seed an existing weak fact so the batch pushes the store over cap=1.
    manager.vector_store.add(
        VectorRecord(
            vector=[0.0, 1.0],
            text="pre-existing weak fact",
            metadata={"type": "fact", "strength": 0.01, "timestamp": _iso_days_ago(30)},
        )
    )
    manager.add_message("user", "I study neuroscience")
    _wait_for(lambda: manager.summary_info.get("strategy") == "llm")
    manager.flush()

    fact_records = [
        rec
        for rec in manager.vector_store.dump()
        if (rec.metadata or {}).get("type") == "fact"
    ]
    assert len(fact_records) <= 1
    assert telemetry.snapshot().get("memory.facts_evicted", 0) >= 1


# ---------------------------------------------------------------------------
# F. Truncation
# ---------------------------------------------------------------------------


def test_store_fact_truncates_to_fact_max_chars(tmp_path: Path) -> None:
    embedder = VectorEmbedder()
    config = _extraction_config(tmp_path, fact_max_chars=10)
    manager = MemoryManager(config, embedder=embedder)
    manager._store_fact("this text is definitely longer than ten characters", 0.5)
    records = list(manager.vector_store.dump())
    assert len(records) == 1
    assert len(records[0].text) == 10
    assert telemetry.snapshot().get("memory.fact_truncated") == 1


def test_store_fact_truncation_disabled_when_zero(tmp_path: Path) -> None:
    embedder = VectorEmbedder()
    config = _extraction_config(tmp_path, fact_max_chars=0)
    manager = MemoryManager(config, embedder=embedder)
    long_text = "x" * 5000
    manager._store_fact(long_text, 0.5)
    records = list(manager.vector_store.dump())
    assert records[0].text == long_text
    assert "memory.fact_truncated" not in telemetry.snapshot()


# ---------------------------------------------------------------------------
# B. Public contract methods
# ---------------------------------------------------------------------------


def test_find_facts_returns_scored_sorted_dicts(tmp_path: Path) -> None:
    embedder = VectorEmbedder({"neuroscience": (1.0, 0.0)}, default=(0.0, 1.0))
    manager = MemoryManager(_extraction_config(tmp_path), embedder=embedder)
    manager.remember("User studies neuroscience")
    manager.remember("User's birthday is July 25", pinned=True)
    # A non-fact record must be excluded.
    manager.vector_store.add(
        VectorRecord(
            vector=[1.0, 0.0],
            text="an episode, not a fact",
            metadata={"type": "episode", "strength": 1.0},
        )
    )

    results = manager.find_facts("neuroscience", k=3)
    assert isinstance(results, list)
    texts = {r["text"] for r in results}
    assert "User studies neuroscience" in texts
    assert "an episode, not a fact" not in texts
    for r in results:
        assert set(r.keys()) == {"id", "text", "score", "pinned"}
        assert isinstance(r["id"], str)
        assert isinstance(r["score"], float)
        assert isinstance(r["pinned"], bool)
    # Sorted score descending.
    scores = [r["score"] for r in results]
    assert scores == sorted(scores, reverse=True)
    by_text = {r["text"]: r for r in results}
    assert by_text["User's birthday is July 25"]["pinned"] is True


def test_find_facts_empty_on_embed_failure(tmp_path: Path) -> None:
    class BoomEmbedder(VectorEmbedder):
        def embed_text(self, text):
            raise RuntimeError("embedder down")

    manager = MemoryManager(_extraction_config(tmp_path), embedder=BoomEmbedder())
    assert manager.find_facts("anything") == []
    assert telemetry.snapshot().get("memory.find_facts_embed_failed") == 1


def test_find_facts_empty_without_store_or_text(tmp_path: Path) -> None:
    manager = MemoryManager(_extraction_config(tmp_path), embedder=VectorEmbedder())
    assert manager.find_facts("") == []

    bare_config = Config()
    bare_config.memory.persist_path = None
    bare_config.memory.enable_long_term = False
    bare = MemoryManager(bare_config)
    assert bare.find_facts("something") == []


def test_archive_fact_returns_text_and_removes(tmp_path: Path) -> None:
    embedder = VectorEmbedder()
    manager = MemoryManager(_extraction_config(tmp_path), embedder=embedder)
    manager.remember("User likes climbing")
    record_id = list(manager.vector_store.dump())[0].id

    text = manager.archive_fact(record_id, "user_forget")
    assert text == "User likes climbing"
    assert list(manager.vector_store.dump()) == []
    lines = _archive_lines(manager)
    assert lines[0]["reason"] == "user_forget"


def test_archive_fact_none_for_missing_or_pinned(tmp_path: Path) -> None:
    embedder = VectorEmbedder()
    manager = MemoryManager(_extraction_config(tmp_path), embedder=embedder)
    manager.remember("User likes climbing", pinned=True)
    record_id = list(manager.vector_store.dump())[0].id

    assert manager.archive_fact("nonexistent-id", "user_forget") is None
    assert manager.archive_fact(record_id, "user_forget") is None
    # Pinned record untouched.
    assert len(list(manager.vector_store.dump())) == 1


def test_store_user_fact_success_sets_source_metadata(tmp_path: Path) -> None:
    embedder = VectorEmbedder()
    manager = MemoryManager(_extraction_config(tmp_path), embedder=embedder)
    ok = manager.store_user_fact("User's favorite color is blue")
    assert ok is True
    records = list(manager.vector_store.dump())
    assert len(records) == 1
    assert records[0].metadata.get("source") == "user_explicit"
    assert records[0].metadata.get("salience") == 0.9


def test_store_user_fact_with_previous_fact_supersedes(tmp_path: Path) -> None:
    embedder = VectorEmbedder(
        {"blue": (1.0, 0.0)},
        default=(0.9, 0.1),
    )
    config = _extraction_config(tmp_path, fact_supersede_hint_threshold=0.5)
    manager = MemoryManager(config, embedder=embedder)
    manager.store_user_fact("User's favorite color is blue")
    ok = manager.store_user_fact(
        "User's favorite color is green now",
        previous_fact="User's favorite color is blue",
    )
    assert ok is True
    records = [r for r in manager.vector_store.dump() if r.text.endswith("green now")]
    assert records
    assert records[0].metadata.get("previous_fact") == "User's favorite color is blue"


def test_store_user_fact_failure_counts_and_returns_false(tmp_path: Path) -> None:
    bare_config = Config()
    bare_config.memory.persist_path = None
    bare_config.memory.enable_long_term = False
    manager = MemoryManager(bare_config)
    assert manager.store_user_fact("anything") is False
    assert telemetry.snapshot().get("memory.store_user_fact_failed") == 1

    telemetry.reset()

    class BoomEmbedder(VectorEmbedder):
        def embed_text(self, text):
            raise RuntimeError("embedder down")

    manager2 = MemoryManager(_extraction_config(tmp_path), embedder=BoomEmbedder())
    assert manager2.store_user_fact("anything") is False
    # Both the underlying _store_fact embed-failure counter and the
    # store_user_fact-level counter increment on this path.
    snap = telemetry.snapshot()
    assert snap.get("memory.store_user_fact_failed") == 1
    assert snap.get("memory.store_fact_embed_failed") == 1


# ---------------------------------------------------------------------------
# G. Summary/eviction race
# ---------------------------------------------------------------------------


class _BlockingLLMProvider:
    """Summary provider that blocks in generate() until released, so a test
    can simulate messages arriving (and evicting from the bounded short_term
    deque) WHILE a summary job is in flight."""

    name = "fake-blocking"

    def __init__(self, release_event, result_text: str) -> None:
        self._release = release_event
        self._result_text = result_text
        self.calls = 0

    def generate(self, messages, **kwargs):
        self.calls += 1
        self._release.wait(timeout=5)
        return self._result_text

    def last_usage(self):
        return None


def test_summary_delta_survives_short_term_eviction_during_in_flight_job(
    tmp_path: Path,
) -> None:
    """G: short_term is a bounded deque (maxlen=max_context_messages) that
    silently evicts. Previously the incremental-summary delta sliced
    short_term by count, so a message that fell off the front while a
    summary job was in flight was permanently lost from the summary. The
    fix (_unsummarized, drained only on success) must cover it instead."""
    import threading

    release = threading.Event()
    summarizer = _BlockingLLMProvider(release, "first summary")
    config = _extraction_config(tmp_path, enable_fact_extraction=False)
    config.memory.max_context_messages = 4
    config.memory.summary_trigger_messages = 2
    manager = MemoryManager(
        config, embedder=VectorEmbedder(), summary_provider=summarizer
    )

    # Kick off the first summary job (blocks inside generate()).
    manager.add_message("user", "message one")
    manager.add_message("assistant", "reply one")
    _wait_for(lambda: summarizer.calls >= 1)
    assert summarizer.calls == 1
    assert manager._summary_future is not None and not manager._summary_future.done()

    # Flood in more messages than max_context_messages while the job is
    # in flight — short_term evicts the earliest of these.
    for i in range(2, 10):
        manager.add_message("user" if i % 2 == 0 else "assistant", f"message {i}")
    assert len(manager.short_term) == 4  # deque evicted down to maxlen

    # The unsummarized buffer, unlike short_term, kept everything.
    assert len(manager._unsummarized) >= 8

    # Release the first job; it lands with strategy "retained"/"llm" — either
    # way _unsummarized must NOT have been drained of messages the job never
    # actually saw (it only summarized the first two).
    release.set()
    _wait_for(lambda: summarizer.calls >= 1 and manager._summary_future is None)
    manager.flush()

    # A second summarize() call must still be able to fold in every message
    # that arrived during the blocked job — none silently dropped.
    manager.summarize()
    manager.flush()
    _wait_for(
        lambda: manager._summary_future is None and not manager._resummarize_queued
    )
    manager.flush()  # drain any chained resummarize-queue re-submit fully

    assert "memory.unsummarized_dropped" not in telemetry.snapshot()


def test_unsummarized_buffer_caps_and_counts_drops(tmp_path: Path) -> None:
    """A dead/hung summary provider must not grow the buffer unboundedly."""
    import threading

    # Bounded "hang" (not truly infinite) so the background thread always
    # has a defined end — released explicitly at the end of the test either
    # way, to avoid a stray _on_summary_ready callback firing during a LATER
    # test or interpreter shutdown.
    release = threading.Event()
    summarizer = _BlockingLLMProvider(release, "unused")
    config = _extraction_config(tmp_path, enable_fact_extraction=False)
    config.memory.max_context_messages = 200
    config.memory.summary_trigger_messages = 2
    manager = MemoryManager(
        config, embedder=VectorEmbedder(), summary_provider=summarizer
    )

    try:
        manager.add_message("user", "trigger")
        manager.add_message("assistant", "trigger the job")
        assert (
            manager._summary_future is not None and not manager._summary_future.done()
        )

        for i in range(100):
            manager.add_message("user", f"flood {i}")

        assert len(manager._unsummarized) == manager._unsummarized_cap
        snap = telemetry.snapshot()
        assert snap.get("memory.unsummarized_dropped", 0) > 0
    finally:
        release.set()
        manager.flush()


def test_finalize_session_short_session_uses_unsummarized_buffer(
    tmp_path: Path,
) -> None:
    """finalize_session's >=2-messages-since-summary gate must still fire
    off the new buffer (regression check for the delta-source swap)."""
    summarizer = FakeLLMProvider(EXTRACTION_JSON)
    config = _extraction_config(tmp_path, summary_trigger_messages=20)
    config.memory.enable_episodes = False
    manager = MemoryManager(
        config, embedder=VectorEmbedder(), summary_provider=summarizer
    )
    manager.add_message("user", "Quick update: I switched labs")
    manager.add_message("assistant", "Noted.")
    manager.flush()
    assert manager.long_term_count == 0

    manager.finalize_session(wait=True)
    manager.flush()
    records = list(manager.vector_store.dump())
    assert [r.metadata.get("type") for r in records] == ["fact"]


# ---------------------------------------------------------------------------
# H. Extraction prompt hardening
# ---------------------------------------------------------------------------


def test_extraction_prompt_excludes_sensitive_data(tmp_path: Path) -> None:
    from amadeus.memory import extraction_jobs

    manager = MemoryManager(_extraction_config(tmp_path), embedder=VectorEmbedder())
    captured = {}

    class CapturingProvider(FakeLLMProvider):
        def generate(self, messages, **kwargs):
            captured["system"] = messages[0].content
            return super().generate(messages, **kwargs)

    manager.summary_provider = CapturingProvider(EXTRACTION_JSON)
    extraction_jobs.run_extraction_job(manager, "hi there", 1, None, None, 2)

    system_text = captured["system"].lower()
    assert "password" in system_text or "credential" in system_text
    assert "api key" in system_text or "api keys" in system_text
    assert "financial" in system_text or "card" in system_text
    assert "significant topic" in system_text or "passing mention" in system_text


# ---------------------------------------------------------------------------
# I. min_query_chars pre-augmentation
# ---------------------------------------------------------------------------


def test_short_raw_query_skips_recall_even_when_augmentation_would_lengthen(
    tmp_path: Path,
) -> None:
    from amadeus.memory.context import assemble_turn_context

    calls: List[str] = []

    class CountingEmbedder(VectorEmbedder):
        def embed_text(self, text):
            calls.append(text)
            return super().embed_text(text)

    embedder = CountingEmbedder({"job interview": (1.0, 0.0)}, default=(0.0, 1.0))
    # enable_fact_extraction=True with no summary_provider keeps extraction
    # inert (no LLM call) while disabling the legacy raw-message long-term
    # capture path — otherwise add_message("user", ...) inside
    # assemble_turn_context would itself embed and store every turn's raw
    # text, contaminating the embed-call assertions below.
    config = _extraction_config(tmp_path)
    config.memory.enable_episodes = False
    config.memory.profile_path = None
    config.memory.recall_min_query_chars = 10
    manager = MemoryManager(config, embedder=embedder)
    manager.vector_store.add(
        VectorRecord(
            vector=[1.0, 0.0],
            text="User has a job interview coming up",
            metadata={"type": "fact", "timestamp": _iso_days_ago(0), "strength": 1.0},
        )
    )
    # Seed a previous turn long enough to lengthen the short follow-up past
    # the gate if augmentation ran first.
    manager.add_message("user", "I have a job interview next week, wish me luck")
    manager.add_message("assistant", "Good luck!")
    calls.clear()

    # "did it work?" is under 10 raw chars... no, it's not; use a genuinely
    # short raw turn that would ALSO trigger augmentation.
    _, recall_items = assemble_turn_context(manager, "system prompt", "ok?")
    assert recall_items == []
    assert calls == [], "raw-too-short turn must skip the embed round-trip entirely"


def test_speculative_recall_pre_check_mirrors_min_query_chars(tmp_path: Path) -> None:
    """speculative.py's duplicated augmentation logic must respect the same
    raw-text gate as assemble_turn_context, or the speculative cache key can
    mismatch what the real turn does."""
    embedder = VectorEmbedder({"job interview": (1.0, 0.0)}, default=(0.0, 1.0))
    config = _extraction_config(tmp_path)
    config.memory.enable_episodes = False
    config.memory.profile_path = None
    config.memory.recall_min_query_chars = 10
    manager = MemoryManager(config, embedder=embedder)
    manager.vector_store.add(
        VectorRecord(
            vector=[1.0, 0.0],
            text="User has a job interview coming up",
            metadata={"type": "fact", "timestamp": _iso_days_ago(0), "strength": 1.0},
        )
    )
    manager.add_message("user", "I have a job interview next week, wish me luck")
    manager.add_message("assistant", "Good luck!")

    future = manager.speculate_recall("ok?")
    assert future is None, "raw-too-short text must not start speculation at all"
    assert manager._speculative is None


def test_min_query_chars_zero_disables_pre_check(tmp_path: Path) -> None:
    from amadeus.memory.context import assemble_turn_context

    embedder = VectorEmbedder({"cats": (1.0, 0.0)}, default=(0.0, 1.0))
    config = _extraction_config(tmp_path)
    config.memory.enable_episodes = False
    config.memory.profile_path = None
    config.memory.recall_min_query_chars = 0
    manager = MemoryManager(config, embedder=embedder)
    manager.vector_store.add(
        VectorRecord(
            vector=[1.0, 0.0],
            text="User loves cats",
            metadata={"type": "fact", "timestamp": _iso_days_ago(0), "strength": 1.0},
        )
    )
    _, recall_items = assemble_turn_context(manager, "sys", "cats")
    assert recall_items == ["User loves cats"]
