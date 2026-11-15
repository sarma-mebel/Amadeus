"""Fact consolidation (amadeus/memory/consolidation.py): merging clusters
of near-duplicate facts into one stronger record at session end.

The tests that matter most pin the lifecycle contract: archive-not-delete
(every merged-away original lands in the fact-archive sidecar), strength
carries corroboration instead of resetting, the OLDEST timestamp survives
(a fresh stamp would silently reset the forgetting clock), and the
eligibility rules that keep pinned / user-explicit / event-anchored records
out of the merger entirely.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import List, Optional

import pytest

from amadeus import telemetry
from amadeus.config import Config
from amadeus.memory import consolidation
from amadeus.memory.consolidation import (
    _clean_merged_line,
    _find_clusters,
    _introduces_numbers,
    _merge_prompt,
    maybe_consolidate_facts,
)
from amadeus.memory.manager import EmbeddingProvider, MemoryManager
from amadeus.memory.vector_store import VectorRecord
from amadeus.providers.llm.base import ChatMessage, LLMProvider


@pytest.fixture(autouse=True)
def _clean_telemetry():
    telemetry.reset()
    yield
    telemetry.reset()


class VectorEmbedder(EmbeddingProvider):
    """Substring-keyed deterministic embedder (relational-test pattern)."""

    def __init__(self, mapping: Optional[dict] = None, default=(1.0, 0.0)) -> None:
        self.mapping = mapping or {}
        self.default = default

    def embed_text(self, text: str) -> List[float]:
        for key, vec in self.mapping.items():
            if key in text:
                return list(vec)
        return list(self.default)

    def similarity(self, a: List[float], b: List[float]) -> float:
        return sum(x * y for x, y in zip(a, b))


class ScriptedLLM(LLMProvider):
    name = "scripted"

    def __init__(self, replies: Optional[List[str]] = None):
        self.replies = replies or ["User drinks coffee every morning to start the day"]
        self.calls: List[List[ChatMessage]] = []
        self.fail = False

    def generate(self, messages, max_tokens: int = 256, **kwargs) -> str:
        self.calls.append(list(messages))
        if self.fail:
            raise RuntimeError("provider down")
        idx = min(len(self.calls) - 1, len(self.replies) - 1)
        return self.replies[idx]

    def stream(self, messages, max_tokens: int = 256, **kwargs):
        yield self.generate(messages, max_tokens=max_tokens)


# Unit vectors: "alpha"/"beta"/"gamma" sit ~0.84 cosine from each other's
# hub (alpha), i.e. inside the merge band but below the 0.86 store-time
# supersede bar; "other" is orthogonal.
_VECTORS = {
    "alpha": (1.0, 0.0),
    "beta": (0.84, 0.5426),
    "gamma": (0.84, -0.5426),
    "other": (0.0, 1.0),
}


def _manager(
    tmp_path: Path,
    llm: Optional[ScriptedLLM] = None,
    *,
    enabled: bool = True,
    **overrides,
) -> MemoryManager:
    config = Config()
    config.memory.persist_path = str(tmp_path / "memory.json")
    config.memory.embeddings_path = str(tmp_path / "embeddings.json")
    config.memory.profile_path = str(tmp_path / "profile.json")
    config.memory.enable_long_term = True
    config.memory.enable_fact_extraction = False
    config.memory.enable_episodes = False
    config.memory.summary_trigger_messages = 10_000
    config.memory.enable_consolidation = enabled
    config.memory.consolidation_min_facts = 3
    for key, value in overrides.items():
        setattr(config.memory, key, value)
    return MemoryManager(
        config,
        embedder=VectorEmbedder(_VECTORS, default=(0.6, 0.8)),
        summary_provider=llm or ScriptedLLM(),
    )


def _seed(manager: MemoryManager, text: str, *, days_old: int = 0, **metadata) -> str:
    """Add one fact record directly (bypasses the store-time supersede)."""
    stamp = (dt.datetime.utcnow() - dt.timedelta(days=days_old)).isoformat()
    base = {
        "type": "fact",
        "timestamp": stamp,
        "salience": 0.6,
        "strength": 1.0,
        "pinned": False,
    }
    base.update(metadata)
    record = VectorRecord(
        vector=manager.embedder.embed_text(text), text=text, metadata=base
    )
    manager.vector_store.add(record)
    return record.id


def _facts(manager: MemoryManager) -> list:
    return [
        rec
        for rec in manager.vector_store.dump()
        if (rec.metadata or {}).get("type") == "fact"
    ]


def _archive_lines(tmp_path: Path) -> list:
    path = tmp_path / "embeddings_archive.jsonl"
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


# ---------------------------------------------------------------------------
# Gates
# ---------------------------------------------------------------------------


def test_disabled_by_default_makes_no_call(tmp_path: Path) -> None:
    assert Config().memory.enable_consolidation is False
    llm = ScriptedLLM()
    manager = _manager(tmp_path, llm, enabled=False)
    for name in ("alpha", "beta", "gamma"):
        _seed(manager, f"{name} coffee note")

    maybe_consolidate_facts(manager)

    assert llm.calls == []


def test_enabled_without_store_counts_unavailable(tmp_path: Path) -> None:
    # Degrade loudly: enabled-but-cannot-run must be visible in telemetry,
    # not a silent return.
    llm = ScriptedLLM()
    manager = _manager(tmp_path, llm)
    manager.vector_store = None

    maybe_consolidate_facts(manager)

    assert llm.calls == []
    assert telemetry.snapshot().get("memory.consolidation_unavailable") == 1


def test_enabled_without_provider_counts_unavailable(tmp_path: Path) -> None:
    manager = _manager(tmp_path)
    manager.summary_provider = None
    for name in ("alpha", "beta", "gamma"):
        _seed(manager, f"{name} coffee note")

    maybe_consolidate_facts(manager)

    assert len(_facts(manager)) == 3
    assert telemetry.snapshot().get("memory.consolidation_unavailable") == 1


def test_small_store_skips_and_stamps(tmp_path: Path) -> None:
    llm = ScriptedLLM()
    manager = _manager(tmp_path, llm, consolidation_min_facts=10)
    _seed(manager, "alpha coffee note")

    maybe_consolidate_facts(manager)

    assert llm.calls == []
    assert (tmp_path / "embeddings_consolidation.json").exists()


# ---------------------------------------------------------------------------
# The merge itself
# ---------------------------------------------------------------------------


def test_merge_archives_originals_and_carries_lineage(tmp_path: Path) -> None:
    llm = ScriptedLLM(["User drinks coffee every morning to start the day"])
    manager = _manager(tmp_path, llm)
    id_a = _seed(manager, "alpha: likes coffee", days_old=30)
    _seed(manager, "beta: drinks coffee daily", days_old=10, strength=1.3)
    _seed(manager, "gamma: mentioned coffee again", days_old=2)
    _seed(manager, "other: has a cat")

    maybe_consolidate_facts(manager)

    assert len(llm.calls) == 1
    facts = _facts(manager)
    texts = {rec.text for rec in facts}
    assert "User drinks coffee every morning to start the day" in texts
    assert "other: has a cat" in texts
    assert len(facts) == 2  # 3 originals collapsed into 1, unrelated intact

    merged = next(rec for rec in facts if "every morning" in rec.text)
    meta = merged.metadata
    # Corroboration carries: max strength (1.3) + one bump per absorbed dup.
    assert meta["strength"] == pytest.approx(1.3 + 0.15 * 2)
    assert meta["consolidated_from"] == 3
    # Oldest timestamp survives so forgetting/confidence stay honest.
    oldest = min(dt.datetime.utcnow() - dt.timedelta(days=d) for d in (30, 10, 2))
    kept = dt.datetime.fromisoformat(meta["timestamp"])
    assert abs((kept - oldest).total_seconds()) < 5
    assert meta["supersedes"] == id_a
    assert "previous_fact" in meta

    reasons = [line["reason"] for line in _archive_lines(tmp_path)]
    assert reasons.count("consolidated") == 3
    counters = telemetry.snapshot()
    assert counters.get("memory.consolidation_merged") == 1
    assert counters.get("memory.consolidation_absorbed") == 2


def test_skip_reply_leaves_store_untouched(tmp_path: Path) -> None:
    llm = ScriptedLLM(["SKIP"])
    manager = _manager(tmp_path, llm)
    for name in ("alpha", "beta", "gamma"):
        _seed(manager, f"{name} adjacent note")

    maybe_consolidate_facts(manager)

    assert len(_facts(manager)) == 3
    assert _archive_lines(tmp_path) == []
    assert telemetry.snapshot().get("memory.consolidation_skipped") == 1


def test_number_hallucination_is_rejected(tmp_path: Path) -> None:
    llm = ScriptedLLM(["User drinks 42 cups of coffee daily"])
    manager = _manager(tmp_path, llm)
    for name in ("alpha", "beta", "gamma"):
        _seed(manager, f"{name} coffee note")

    maybe_consolidate_facts(manager)

    assert len(_facts(manager)) == 3
    assert telemetry.snapshot().get("memory.consolidation_rejected") == 1


# ---------------------------------------------------------------------------
# Cadence + failure retry
# ---------------------------------------------------------------------------


def test_cadence_gate_blocks_second_run(tmp_path: Path) -> None:
    llm = ScriptedLLM(["merged coffee fact"])
    manager = _manager(tmp_path, llm)
    for name in ("alpha", "beta", "gamma"):
        _seed(manager, f"{name} coffee note")

    maybe_consolidate_facts(manager)
    maybe_consolidate_facts(manager)

    assert len(llm.calls) == 1


def test_provider_failure_does_not_stamp_so_it_retries(tmp_path: Path) -> None:
    llm = ScriptedLLM()
    llm.fail = True
    manager = _manager(tmp_path, llm)
    for name in ("alpha", "beta", "gamma"):
        _seed(manager, f"{name} coffee note")

    maybe_consolidate_facts(manager)
    assert telemetry.snapshot().get("memory.consolidation_failures") == 1
    assert not (tmp_path / "embeddings_consolidation.json").exists()

    llm.fail = False
    maybe_consolidate_facts(manager)
    assert len(_facts(manager)) < 3  # retry merged


def test_midrun_failure_persists_completed_merges(tmp_path: Path) -> None:
    """Regression: on inline-vector configs (no embeddings_path, so
    vector_store.persist() is a no-op) a failure on cluster 2 returned
    before manager._persist(), leaving cluster 1's rewrite RAM-only while
    its archive lines were already on disk — a process exit resurrected
    originals the archive also holds."""

    class FailAfterFirst(ScriptedLLM):
        def generate(self, messages, max_tokens: int = 256, **kwargs) -> str:
            if len(self.calls) >= 1:
                self.calls.append(list(messages))
                raise RuntimeError("provider died mid-run")
            return super().generate(messages, max_tokens=max_tokens, **kwargs)

    merged_line = "other pair merged into one cat note"
    llm = FailAfterFirst([merged_line])
    manager = _manager(tmp_path, llm, embeddings_path=None)
    # Cluster A: the identical "other" pair (internal similarity 1.0, so it
    # spends the merge budget first). Cluster B: the alpha/beta/gamma hub.
    _seed(manager, "other: has a cat")
    _seed(manager, "other: is a cat person")
    for name in ("alpha", "beta", "gamma"):
        _seed(manager, f"{name} coffee note")
    manager._persist()  # originals on disk, as in normal operation

    maybe_consolidate_facts(manager)

    counters = telemetry.snapshot()
    assert counters.get("memory.consolidation_merged") == 1
    assert counters.get("memory.consolidation_failures") == 1

    # A fresh manager over the same files sees the merge, not the originals.
    reloaded = _manager(tmp_path, ScriptedLLM(), embeddings_path=None)
    texts = {rec.text for rec in reloaded.vector_store.dump()}
    assert merged_line in texts
    assert "other: has a cat" not in texts
    assert "other: is a cat person" not in texts


# ---------------------------------------------------------------------------
# Eligibility
# ---------------------------------------------------------------------------


def test_pinned_sourced_dated_and_moment_records_are_untouchable(
    tmp_path: Path,
) -> None:
    llm = ScriptedLLM(["merged line"])
    manager = _manager(tmp_path, llm, consolidation_min_facts=2)
    _seed(manager, "alpha pinned", pinned=True)
    _seed(manager, "beta explicit", source="user_explicit")
    _seed(manager, "gamma dated", valid_at="2026-07-01T00:00:00")
    _seed(manager, "alpha moment", type="moment")
    _seed(manager, "beta plain")

    maybe_consolidate_facts(manager)

    # Only one eligible record — no cluster, no call, store intact.
    assert llm.calls == []
    assert len(list(manager.vector_store.dump())) == 5


def test_finalize_session_submits_the_job(tmp_path: Path, monkeypatch) -> None:
    manager = _manager(tmp_path)
    called = []
    monkeypatch.setattr(
        consolidation, "maybe_consolidate_facts", lambda mgr: called.append(mgr)
    )
    manager.add_message("user", "hello")
    manager.add_message("assistant", "hi")
    manager.finalize_session(wait=True)
    assert called == [manager]


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def test_merge_prompt_contract(tmp_path: Path) -> None:
    manager = _manager(tmp_path)
    prompt = _merge_prompt(manager, ["oldest note", "newest note"])
    system = prompt[0].content
    assert "SKIP" in system
    assert "invent NOTHING" in system
    assert "oldest first" in system
    assert prompt[1].content == "1. oldest note\n2. newest note"


def test_clean_merged_line() -> None:
    assert _clean_merged_line('  "merged fact."\nextra') == "merged fact."
    assert _clean_merged_line("SKIP") is None
    assert _clean_merged_line("skip.") is None
    assert _clean_merged_line("") is None


def test_introduces_numbers() -> None:
    assert _introduces_numbers("wakes at 6", ["wakes early"]) is True
    assert _introduces_numbers("wakes at 6", ["wakes at 6 usually"]) is False
    assert _introduces_numbers("no digits", ["also none"]) is False


def test_clusters_respect_max_size_and_threshold() -> None:
    def rec(name: str, vec, stamp: str) -> VectorRecord:
        return VectorRecord(
            vector=list(vec),
            text=name,
            metadata={"type": "fact", "timestamp": stamp},
        )

    records = [
        rec("a", _VECTORS["alpha"], "2026-01-01"),
        rec("b", _VECTORS["beta"], "2026-01-02"),
        rec("c", _VECTORS["gamma"], "2026-01-03"),
        rec("d", _VECTORS["other"], "2026-01-04"),
    ]
    clusters = _find_clusters(records, 0.83, max_cluster=4)
    assert len(clusters) == 1
    assert {r.text for r in clusters[0]} == {"a", "b", "c"}

    capped = _find_clusters(records, 0.83, max_cluster=2)
    assert len(capped[0]) == 2
