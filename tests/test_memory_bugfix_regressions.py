"""Regression pins for a batch of verified memory-layer bug fixes.

- recall.py: the rolling-summary embed rides the deadline-bounded helper
  (it was the one remaining unbounded network call on the recall hot path);
  a timeout degrades to skipping summary-dedupe, never a stalled turn.
- extraction_jobs.py: episode supersede/cap-prune archive-not-delete, with
  counters — they used to hard-delete with no paper trail.
- vector_store.py / profile.py: valid-JSON-wrong-shape files follow the
  same ".bak-corrupt + fresh start, never a crash" quarantine contract as
  unparseable JSON (they used to raise out of load).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

import pytest

from amadeus import telemetry
from amadeus.config import Config
from amadeus.memory.extraction_jobs import (
    prune_episodes_over_cap,
    supersede_near_duplicate_episode,
)
from amadeus.memory.manager import EmbeddingProvider, MemoryManager, _EmbedTimeout
from amadeus.memory.profile import UserProfile
from amadeus.memory.vector_store import InMemoryVectorStore, VectorRecord


@pytest.fixture(autouse=True)
def _clean_counters():
    telemetry.reset()
    yield
    telemetry.reset()


class VectorEmbedder(EmbeddingProvider):
    def embed_text(self, text: str) -> List[float]:
        return [1.0, 0.0]

    def similarity(self, a: List[float], b: List[float]) -> float:
        return sum(x * y for x, y in zip(a, b))


def _manager(tmp_path: Path, **overrides) -> MemoryManager:
    config = Config()
    config.memory.persist_path = None
    config.memory.profile_path = None
    config.memory.enable_long_term = True
    config.memory.embeddings_path = str(tmp_path / "vectors.json")
    config.memory.enable_fact_extraction = False
    config.memory.enable_episodes = False
    config.memory.recall_min_query_chars = 0
    config.memory.recall_min_score = 0.0
    for key, value in overrides.items():
        setattr(config.memory, key, value)
    return MemoryManager(config, embedder=VectorEmbedder())


def _add_record(manager: MemoryManager, text: str, metadata: dict) -> None:
    manager.vector_store.add(
        VectorRecord(vector=[1.0, 0.0], text=text, metadata=metadata)
    )


def _archive_lines(tmp_path: Path) -> List[dict]:
    path = tmp_path / "vectors_archive.jsonl"
    if not path.is_file():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


# ---------------------------------------------------------------------------
# recall.py: bounded summary embed
# ---------------------------------------------------------------------------


def test_summary_embed_timeout_degrades_to_no_dedupe(tmp_path, monkeypatch):
    manager = _manager(tmp_path, recall_summary_dedupe=0.9)
    manager.summary = "the rolling summary of the conversation"
    _add_record(manager, "the user loves control theory", {"type": "fact"})

    original = manager._embed_query_bounded

    def bounded(text: str):
        if text == manager.summary:
            raise _EmbedTimeout()
        return original(text)

    monkeypatch.setattr(manager, "_embed_query_bounded", bounded)

    results = manager.recall_detailed("what does the user love?", limit=3)

    # The candidate survives (skipping dedupe beats losing recall) and the
    # timeout is counted — the turn neither stalled nor crashed.
    assert [m.text for m in results] == ["the user loves control theory"]
    assert telemetry.snapshot().get("memory.recall_dedupe_embed_timeouts") == 1


def test_summary_embed_goes_through_bounded_helper(tmp_path, monkeypatch):
    """The summary embed must never call embedder.embed_text directly."""
    manager = _manager(tmp_path, recall_summary_dedupe=0.9)
    manager.summary = "summary text"
    _add_record(manager, "a stored fact", {"type": "fact"})

    seen: List[str] = []
    original = manager._embed_query_bounded

    def bounded(text: str):
        seen.append(text)
        return original(text)

    monkeypatch.setattr(manager, "_embed_query_bounded", bounded)
    manager.recall_detailed("tell me about the fact", limit=3)

    assert manager.summary in seen


# ---------------------------------------------------------------------------
# extraction_jobs.py: episodes archive-not-delete
# ---------------------------------------------------------------------------


def test_episode_supersede_archives_instead_of_hard_delete(tmp_path):
    manager = _manager(tmp_path, episode_supersede_threshold=0.9)
    _add_record(
        manager,
        "2026-08-01: the user greeted her",
        {"type": "episode", "timestamp": "2026-08-01T00:00:00", "salience": 0.5},
    )

    with manager._state_lock:
        supersede_near_duplicate_episode(manager, [1.0, 0.0])

    assert manager.vector_store.count() == 0
    lines = _archive_lines(tmp_path)
    assert len(lines) == 1
    assert lines[0]["reason"] == "episode_superseded"
    assert lines[0]["text"] == "2026-08-01: the user greeted her"
    assert telemetry.snapshot().get("memory.episodes_superseded") == 1


def test_episode_cap_prune_archives_instead_of_hard_delete(tmp_path):
    manager = _manager(tmp_path, max_episodes=1)
    for i in range(3):
        _add_record(
            manager,
            f"2026-08-0{i + 1}: episode {i}",
            {
                "type": "episode",
                "timestamp": f"2026-08-0{i + 1}T00:00:00",
                "salience": 0.1 * (i + 1),
            },
        )

    with manager._state_lock:
        prune_episodes_over_cap(manager)

    assert manager.vector_store.count() == 1
    lines = _archive_lines(tmp_path)
    assert len(lines) == 2
    assert all(line["reason"] == "episode_cap" for line in lines)
    assert telemetry.snapshot().get("memory.episodes_pruned") == 2


# ---------------------------------------------------------------------------
# vector_store.py / profile.py: wrong-shape quarantine
# ---------------------------------------------------------------------------


def test_vector_store_quarantines_valid_json_wrong_shape(tmp_path):
    path = tmp_path / "vectors.json"
    path.write_text(json.dumps({"a": 1}), encoding="utf-8")

    store = InMemoryVectorStore(path=path)  # must not raise

    assert store.count() == 0
    assert path.with_name(path.name + ".bak-corrupt").is_file()
    assert telemetry.snapshot().get("memory.vector_store_load_corrupt") == 1


def test_vector_store_quarantines_non_numeric_vectors(tmp_path):
    path = tmp_path / "vectors.json"
    path.write_text(
        json.dumps([{"vector": ["not-a-number"], "text": "x"}]), encoding="utf-8"
    )

    store = InMemoryVectorStore(path=path)

    assert store.count() == 0
    assert path.with_name(path.name + ".bak-corrupt").is_file()
    assert telemetry.snapshot().get("memory.vector_store_load_corrupt") == 1


def test_profile_quarantines_valid_json_wrong_shape(tmp_path):
    path = tmp_path / "profile.json"
    path.write_text(json.dumps({"session_count": "garbage"}), encoding="utf-8")

    profile = UserProfile.load(path)  # must not raise

    assert profile.session_count == 0  # blank profile
    assert path.with_name(path.name + ".bak-corrupt").is_file()
    assert telemetry.snapshot().get("memory.profile_load_corrupt") == 1


def test_profile_still_loads_good_files_unchanged(tmp_path):
    path = tmp_path / "profile.json"
    path.write_text(
        json.dumps({"session_count": 4, "facts": {"name": "Alex"}}),
        encoding="utf-8",
    )

    profile = UserProfile.load(path)

    assert profile.session_count == 4
    assert profile.facts == {"name": "Alex"}
    assert not path.with_name(path.name + ".bak-corrupt").exists()
