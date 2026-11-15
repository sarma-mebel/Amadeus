"""Bitemporal memory stamps: ``timestamp`` (learn time) vs ``valid_at``
(event time).

The split is additive-only by design: every decay/sort/cap consumer keeps
reading ``timestamp`` unchanged, while ``valid_at`` — an optional flat
naive-UTC ISO string (Chroma metadata must stay scalar) — is consumed only
by the recall renderer (event clause, future-capable) and the introspection
payloads. Existing stores need no migration; records without ``valid_at``
render exactly as before.
"""

import datetime as dt
import json
from pathlib import Path
from typing import List

from amadeus import telemetry
from amadeus.config import Config
from amadeus.memory.extraction_jobs import run_extraction_job
from amadeus.memory.manager import EmbeddingProvider, MemoryManager
from amadeus.memory.text_utils import _event_clause, _normalize_event_time
from amadeus.memory.vector_store import VectorRecord
from amadeus.providers.llm.base import LLMProvider


class VectorEmbedder(EmbeddingProvider):
    def __init__(self, mapping: dict | None = None, default=(1.0, 0.0)) -> None:
        self.mapping = mapping or {}
        self.default = default

    def embed_text(self, text: str) -> List[float]:
        for key, vec in self.mapping.items():
            if key in text:
                return list(vec)
        return list(self.default)

    def similarity(self, a: List[float], b: List[float]) -> float:
        return sum(x * y for x, y in zip(a, b))


class ScriptedProvider(LLMProvider):
    name = "fake-scripted"

    def __init__(self, text: str) -> None:
        self.text = text

    def generate(self, messages, **kwargs):  # type: ignore[override]
        return self.text


def _config(tmp_path: Path, **overrides) -> Config:
    config = Config()
    config.memory.persist_path = None
    config.memory.enable_long_term = True
    config.memory.embeddings_path = str(tmp_path / "vectors.json")
    config.memory.dedupe_threshold = 0.0
    config.memory.recall_min_query_chars = 0
    config.memory.enable_fact_extraction = True
    config.memory.enable_episodes = False
    config.memory.profile_path = str(tmp_path / "profile.json")
    config.memory.recall_summary_dedupe = 0.0
    for key, value in overrides.items():
        setattr(config.memory, key, value)
    return config


def _iso_days_from_now(days: float) -> str:
    return (dt.datetime.utcnow() + dt.timedelta(days=days)).isoformat()


def test_event_clause_buckets() -> None:
    now = dt.datetime(2026, 7, 20, 12, 0, 0)

    def at(**kwargs):
        return (now + dt.timedelta(**kwargs)).isoformat()

    # Past events reuse the learn-time buckets with a "happened" prefix.
    assert _event_clause(at(days=-21), now) == "happened about 3 weeks ago"
    assert _event_clause(at(hours=-3), now) == "happened earlier today"
    # Unlike _age_phrase, future stamps are first-class ("happens ..."), not
    # eaten by the clock-skew guard — upcoming events are the point.
    assert _event_clause(at(hours=6), now) == "happens later today"
    assert _event_clause(at(days=1, hours=6), now) == "happens tomorrow"
    assert _event_clause(at(days=4), now) == "happens in 4 days"
    assert _event_clause(at(days=10), now) == "happens in about a week"
    assert _event_clause(at(days=21), now) == "happens in about 3 weeks"
    assert _event_clause(at(days=90), now) == "happens in about 3 months"
    assert _event_clause(at(days=400), now) == "happens over a year from now"
    assert _event_clause(None, now) is None
    assert _event_clause("not-a-date", now) is None


def test_normalize_event_time() -> None:
    # Date-only strings are the extraction prompt's requested format.
    assert _normalize_event_time("2026-08-15") == "2026-08-15T00:00:00"
    # Aware stamps normalize to naive UTC so downstream naive arithmetic
    # (recall decay, _age_phrase-style subtraction) can never TypeError.
    assert _normalize_event_time("2026-08-15T10:00:00+02:00") == ("2026-08-15T08:00:00")
    assert _normalize_event_time("not-a-date") is None
    assert _normalize_event_time(None) is None
    assert _normalize_event_time(20260815) is None


def test_extraction_parses_when_into_fourth_tuple_slot(tmp_path: Path) -> None:
    payload = json.dumps(
        {
            "summary": "User talked about their defense.",
            "new_facts": [
                {
                    "text": "User defends their thesis on 2026-08-15",
                    "salience": 0.9,
                    "when": "2026-08-15",
                },
                {"text": "User likes coffee", "salience": 0.5},
            ],
        }
    )
    manager = MemoryManager(
        _config(tmp_path),
        embedder=VectorEmbedder(),
        summary_provider=ScriptedProvider(payload),
    )
    result = run_extraction_job(manager, "transcript", 1, None, None, 2)
    assert result["facts"] == [
        ("User defends their thesis on 2026-08-15", 0.9, None, "2026-08-15T00:00:00"),
        ("User likes coffee", 0.5, None, None),
    ]


def test_extraction_rejects_malformed_when_and_counts(tmp_path: Path) -> None:
    payload = json.dumps(
        {
            "summary": "s",
            "new_facts": [{"text": "User moved", "salience": 0.5, "when": "sometime"}],
        }
    )
    manager = MemoryManager(
        _config(tmp_path),
        embedder=VectorEmbedder(),
        summary_provider=ScriptedProvider(payload),
    )
    before = telemetry.snapshot().get("memory.valid_at_rejected", 0)
    result = run_extraction_job(manager, "transcript", 1, None, None, 2)
    # The junk "when" is dropped, never allowed to block the fact itself.
    assert result["facts"] == [("User moved", 0.5, None, None)]
    assert telemetry.snapshot()["memory.valid_at_rejected"] == before + 1


def test_store_fact_persists_valid_at(tmp_path: Path) -> None:
    manager = MemoryManager(_config(tmp_path), embedder=VectorEmbedder())
    assert manager._store_fact("User defends their thesis", 0.9, valid_at="2026-08-15")
    records = list(manager.vector_store.dump())
    assert len(records) == 1
    assert records[0].metadata["valid_at"] == "2026-08-15T00:00:00"
    # Learn time stays the untouched decay/sort key.
    assert "timestamp" in records[0].metadata


def test_store_fact_rejects_bad_valid_at_but_stores_fact(tmp_path: Path) -> None:
    manager = MemoryManager(_config(tmp_path), embedder=VectorEmbedder())
    before = telemetry.snapshot().get("memory.valid_at_rejected", 0)
    assert manager._store_fact("User moved recently", 0.5, valid_at="last spring")
    records = list(manager.vector_store.dump())
    assert len(records) == 1
    assert "valid_at" not in records[0].metadata
    assert telemetry.snapshot()["memory.valid_at_rejected"] == before + 1


def test_recall_renders_event_clause_past_and_future(tmp_path: Path) -> None:
    from amadeus.memory.context import assemble_turn_context

    manager = MemoryManager(_config(tmp_path), embedder=VectorEmbedder())
    manager.vector_store.add(
        VectorRecord(
            vector=[1.0, 0.0],
            text="User ran the Prague marathon",
            metadata={
                "type": "fact",
                "timestamp": _iso_days_from_now(-2),
                "valid_at": _iso_days_from_now(-21),
            },
        )
    )
    manager.vector_store.add(
        VectorRecord(
            vector=[1.0, 0.0],
            text="User defends their thesis soon",
            metadata={
                "type": "fact",
                "timestamp": _iso_days_from_now(-2),
                "valid_at": _iso_days_from_now(21),
            },
        )
    )
    manager.vector_store.add(
        VectorRecord(
            vector=[1.0, 0.0],
            text="plain fact without event time",
            metadata={"type": "fact", "timestamp": _iso_days_from_now(-21)},
        )
    )

    context, _ = assemble_turn_context(manager, "system", "prague marathon thesis?")
    body = context[-1].content
    assert (
        "- User ran the Prague marathon (2 days ago; happened about 3 weeks ago)"
        in body
    )
    assert (
        "- User defends their thesis soon (2 days ago; happens in about 3 weeks)"
        in body
    )
    # Records without valid_at keep the exact pre-bitemporal line shape.
    assert "- plain fact without event time (about 3 weeks ago)" in body


def test_memory_tool_remember_threads_when(tmp_path: Path) -> None:
    from amadeus.agents.memory_tool import MemoryTool

    manager = MemoryManager(_config(tmp_path), embedder=VectorEmbedder())
    tool = MemoryTool(manager)
    result = tool.execute(
        action="remember",
        text="User's wedding is on 2026-10-03",
        when="2026-10-03",
    )
    assert result.success
    records = list(manager.vector_store.dump())
    assert len(records) == 1
    assert records[0].metadata["valid_at"] == "2026-10-03T00:00:00"
    assert records[0].metadata["source"] == "user_explicit"


def test_memory_overview_exposes_valid_at(tmp_path: Path) -> None:
    manager = MemoryManager(_config(tmp_path), embedder=VectorEmbedder())
    assert manager._store_fact("User graduates", 0.8, valid_at="2026-09-01")
    overview = manager.memory_overview()
    assert overview["fact_records"][0]["valid_at"] == "2026-09-01T00:00:00"
    # Un-stamped records surface None, not a missing key.
    assert manager._store_fact("Completely different topic entirely", 0.5)


def test_valid_at_rides_into_archive_on_supersede(tmp_path: Path) -> None:
    manager = MemoryManager(
        _config(tmp_path, fact_supersede_threshold=0.9), embedder=VectorEmbedder()
    )
    assert manager._store_fact("User loves Nolan movies", 0.6, valid_at="2026-07-01")
    assert manager._store_fact("User loves Nolan movies", 0.6)
    archive_path = tmp_path / "vectors_archive.jsonl"
    entries = [
        json.loads(line)
        for line in archive_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(entries) == 1
    assert entries[0]["reason"] == "superseded_duplicate"
    assert entries[0]["metadata"]["valid_at"] == "2026-07-01T00:00:00"
