"""Shared-moments memory (amadeus/moments.py): opt-in distillation of a
finished read-along/watch-together session into one durable, recallable
fact. Covers the default-off contract, the happy path, SKIP, the <2-notes
gate, the daily cap, recall surfacing kind=moment, and archive/forget
working on a moment exactly like any other fact.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import List, Optional

import pytest

from amadeus import telemetry
from amadeus.config import Config
from amadeus.engine import ChatEngine
from amadeus.memory.manager import EmbeddingProvider, MemoryManager
from amadeus.providers.llm.base import ChatMessage, LLMProvider


@pytest.fixture(autouse=True)
def _clean_telemetry():
    telemetry.reset()
    yield
    telemetry.reset()


class VectorEmbedder(EmbeddingProvider):
    """Same fixed-vector fake used across the memory test suite: distinct
    keyword buckets get distinct unit vectors so cosine scores are graded
    instead of degenerate."""

    def __init__(self, mapping: Optional[dict] = None, default=(1.0, 0.0)) -> None:
        self.mapping = mapping or {}
        self.default = default

    def embed_text(self, text: str) -> List[float]:
        for key, vec in self.mapping.items():
            if key in text:
                return list(vec)
        return list(self.default)

    def similarity(self, a: List[float], b: List[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        return dot


class ScriptedLLM(LLMProvider):
    """Vision-capable-or-not fake that also supports plain generate() for
    the moments distillation call; returns each entry in ``replies`` in
    order (last one repeats), and records every prompt it saw."""

    name = "scripted"

    def __init__(self, replies: Optional[List[str]] = None, vision: bool = True):
        self.replies = replies or ["a distilled moment"]
        self.calls: List[List[ChatMessage]] = []
        self._vision = vision
        self.fail = False

    def supports_vision(self) -> bool:
        return self._vision

    def generate(self, messages, max_tokens: int = 256, **kwargs) -> str:
        self.calls.append(list(messages))
        if self.fail:
            raise RuntimeError("provider down")
        idx = min(len(self.calls) - 1, len(self.replies) - 1)
        return self.replies[idx]

    def stream(self, messages, max_tokens: int = 256, **kwargs):
        yield self.generate(messages, max_tokens=max_tokens)


def _engine(
    tmp_path: Path,
    llm: Optional[LLMProvider] = None,
    *,
    embedder: Optional[EmbeddingProvider] = None,
    moments_overrides: Optional[dict] = None,
    long_term: bool = True,
) -> ChatEngine:
    config = Config()
    config.memory.persist_path = str(tmp_path / "memory.json")
    config.memory.embeddings_path = str(tmp_path / "embeddings.json")
    config.memory.profile_path = str(tmp_path / "profile.json")
    config.memory.transcripts_path = str(tmp_path / "transcripts.json")
    config.memory.enable_long_term = long_term
    config.memory.enable_fact_extraction = False
    config.memory.enable_episodes = False
    config.vision.enabled = True
    for key, value in (moments_overrides or {}).items():
        setattr(config.moments, key, value)
    engine = ChatEngine(
        config, llm or ScriptedLLM(), embedder=embedder or VectorEmbedder()
    )
    return engine


IMG = ("image/jpeg", "aGVsbG8=")


def _run_read_along(engine: ChatEngine, pages: int, note_llm: LLMProvider) -> None:
    """Drive read_along_page enough times to accumulate ``pages`` notes,
    using ``note_llm`` (must be vision-capable) as the page-note provider."""
    engine.llm = note_llm
    engine.summary_provider = note_llm
    for _ in range(pages):
        engine.read_along_page(*IMG)


# ---------------------------------------------------------------------------
# Default-off contract
# ---------------------------------------------------------------------------


def test_disabled_by_default_stores_nothing(tmp_path: Path) -> None:
    note_llm = ScriptedLLM(["NOTE: page one", "NOTE: page two", "NOTE: page three"])
    engine = _engine(tmp_path, note_llm)  # moments.enabled defaults False
    assert engine.config.moments.enabled is False

    _run_read_along(engine, 3, note_llm)
    sign_off_llm = ScriptedLLM(["Good session."])
    engine.summary_provider = sign_off_llm
    engine.read_along_end()

    assert engine.memory.long_term_count == 0
    assert telemetry.snapshot().get("moments.stored") is None


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_enabled_happy_path_stores_one_moment_record(tmp_path: Path) -> None:
    note_llm = ScriptedLLM(
        [
            "NOTE: The finale begins.",
            "NOTE: A huge reveal happens.",
            "NOTE: They cry at the ending.",
        ]
    )
    engine = _engine(tmp_path, note_llm, moments_overrides={"enabled": True})
    _run_read_along(engine, 3, note_llm)

    moment_llm = ScriptedLLM(["we read the finale together and it wrecked us"])
    engine.summary_provider = moment_llm
    engine.read_along_end()

    assert engine.memory.long_term_count == 1
    records = list(engine.memory.vector_store.dump())
    assert len(records) == 1
    assert records[0].metadata["type"] == "moment"
    assert records[0].metadata.get("timestamp")
    assert records[0].text == "we read the finale together and it wrecked us"
    assert telemetry.snapshot().get("moments.stored") == 1


def test_stored_moment_respects_max_chars(tmp_path: Path) -> None:
    note_llm = ScriptedLLM(["NOTE: page one", "NOTE: page two"])
    long_line = "we watched " + ("x" * 500)
    engine = _engine(
        tmp_path,
        note_llm,
        moments_overrides={"enabled": True, "max_chars": 40},
    )
    _run_read_along(engine, 2, note_llm)
    engine.summary_provider = ScriptedLLM([long_line])
    engine.read_along_end()

    records = list(engine.memory.vector_store.dump())
    assert len(records) == 1
    assert len(records[0].text) <= 40


# ---------------------------------------------------------------------------
# SKIP verdict
# ---------------------------------------------------------------------------


def test_skip_verdict_stores_nothing_and_counts(tmp_path: Path) -> None:
    note_llm = ScriptedLLM(["NOTE: page one", "NOTE: page two"])
    engine = _engine(tmp_path, note_llm, moments_overrides={"enabled": True})
    _run_read_along(engine, 2, note_llm)

    engine.summary_provider = ScriptedLLM(["SKIP"])
    engine.read_along_end()

    assert engine.memory.long_term_count == 0
    assert telemetry.snapshot().get("moments.skipped") == 1
    assert telemetry.snapshot().get("moments.stored") is None


# ---------------------------------------------------------------------------
# <2 notes: no call at all
# ---------------------------------------------------------------------------


def test_fewer_than_two_notes_makes_no_moment_call(tmp_path: Path) -> None:
    note_llm = ScriptedLLM(["NOTE: only page"])
    engine = _engine(tmp_path, note_llm, moments_overrides={"enabled": True})
    _run_read_along(engine, 1, note_llm)

    moment_llm = ScriptedLLM(["should never be called"])
    engine.summary_provider = moment_llm
    # read_along_end also declines the sign-off itself below 5 pages, but
    # the moments call is gated independently on note count — assert no
    # call happened via the moment-shaped provider at all.
    engine.read_along_end()

    assert moment_llm.calls == []
    assert engine.memory.long_term_count == 0
    assert telemetry.snapshot().get("moments.stored") is None
    assert telemetry.snapshot().get("moments.skipped") is None


# ---------------------------------------------------------------------------
# Daily cap
# ---------------------------------------------------------------------------


def test_daily_cap_skips_and_counts(tmp_path: Path) -> None:
    engine = _engine(
        tmp_path,
        moments_overrides={"enabled": True, "max_per_day": 2},
    )

    now = dt.datetime.utcnow()
    for i in range(2):
        engine.memory._store_fact(
            f"we did thing {i} together",
            salience=0.6,
            record_type="moment",
            source="co_activity",
        )
    assert engine.memory.count_moments_since(now - dt.timedelta(hours=24)) == 2

    note_llm = ScriptedLLM(["NOTE: page one", "NOTE: page two"])
    _run_read_along(engine, 2, note_llm)
    moment_llm = ScriptedLLM(["we did a third thing together"])
    engine.summary_provider = moment_llm
    engine.read_along_end()

    assert moment_llm.calls == []  # cap short-circuits before the LLM call
    assert engine.memory.long_term_count == 2
    assert telemetry.snapshot().get("moments.daily_cap_skipped") == 1


def test_daily_cap_only_counts_last_24h(tmp_path: Path) -> None:
    engine = _engine(
        tmp_path,
        moments_overrides={"enabled": True, "max_per_day": 1},
    )
    engine.memory._store_fact(
        "an old shared moment", salience=0.6, record_type="moment"
    )
    # Backdate it outside the 24h window directly on the stored record.
    records = list(engine.memory.vector_store.dump())
    records[0].metadata["timestamp"] = (
        dt.datetime.utcnow() - dt.timedelta(days=2)
    ).isoformat()
    engine.memory.vector_store.persist()

    now = dt.datetime.utcnow()
    assert engine.memory.count_moments_since(now - dt.timedelta(hours=24)) == 0

    note_llm = ScriptedLLM(["NOTE: page one", "NOTE: page two"])
    _run_read_along(engine, 2, note_llm)
    moment_llm = ScriptedLLM(["we shared something new"])
    engine.summary_provider = moment_llm
    engine.read_along_end()

    assert engine.memory.long_term_count == 2
    assert telemetry.snapshot().get("moments.daily_cap_skipped") is None


def test_daily_cap_ignores_relational_moments(tmp_path: Path) -> None:
    """The caps are independent by contract (memory/relational.py): a
    relational-memory moment stored today must NOT eat the co-activity
    budget — counting it here silenced a genuine shared-activity memory."""
    engine = _engine(
        tmp_path,
        moments_overrides={"enabled": True, "max_per_day": 1},
    )
    engine.memory._store_fact(
        "something shifted between them today",
        salience=0.6,
        record_type="moment",
        source="relational",
    )

    note_llm = ScriptedLLM(["NOTE: page one", "NOTE: page two"])
    _run_read_along(engine, 2, note_llm)
    moment_llm = ScriptedLLM(["we read something great together"])
    engine.summary_provider = moment_llm
    engine.read_along_end()

    # The co-activity moment still stored despite the relational one.
    assert engine.memory.long_term_count == 2
    assert telemetry.snapshot().get("moments.stored") == 1
    assert telemetry.snapshot().get("moments.daily_cap_skipped") is None


# ---------------------------------------------------------------------------
# Watch-together parity
# ---------------------------------------------------------------------------


def test_watch_together_end_also_stores_a_moment(tmp_path: Path) -> None:
    note_llm = ScriptedLLM(["NOTE: frame one", "NOTE: frame two"])
    engine = _engine(tmp_path, note_llm, moments_overrides={"enabled": True})
    engine.llm = note_llm
    engine.summary_provider = note_llm
    for _ in range(2):
        engine.watch_frame(*IMG)

    moment_llm = ScriptedLLM(["we watched something together"])
    engine.summary_provider = moment_llm
    engine.watch_end()

    assert engine.memory.long_term_count == 1
    records = list(engine.memory.vector_store.dump())
    assert records[0].metadata["type"] == "moment"


# ---------------------------------------------------------------------------
# Failure path never raises
# ---------------------------------------------------------------------------


def test_provider_failure_counts_and_does_not_raise(tmp_path: Path) -> None:
    note_llm = ScriptedLLM(["NOTE: page one", "NOTE: page two"])
    engine = _engine(tmp_path, note_llm, moments_overrides={"enabled": True})
    _run_read_along(engine, 2, note_llm)

    failing = ScriptedLLM()
    failing.fail = True
    engine.summary_provider = failing

    result = engine.read_along_end()  # must not raise

    assert result is None  # sign-off ALSO failed (<5 pages), independent gate
    assert engine.memory.long_term_count == 0
    assert telemetry.snapshot().get("moments.failures") == 1


def test_store_failure_counts_and_does_not_raise(tmp_path: Path, monkeypatch) -> None:
    note_llm = ScriptedLLM(["NOTE: page one", "NOTE: page two"])
    engine = _engine(tmp_path, note_llm, moments_overrides={"enabled": True})
    _run_read_along(engine, 2, note_llm)

    engine.summary_provider = ScriptedLLM(["we did something nice together"])
    monkeypatch.setattr(engine.memory, "store_moment", lambda text, **k: False)

    engine.read_along_end()  # must not raise

    assert telemetry.snapshot().get("moments.failures") == 1
    assert telemetry.snapshot().get("moments.stored") is None


# ---------------------------------------------------------------------------
# Recall surfaces kind=moment
# ---------------------------------------------------------------------------


def test_recall_surfaces_moment_records(tmp_path: Path) -> None:
    config = Config()
    config.memory.persist_path = None
    config.memory.enable_long_term = True
    config.memory.embeddings_path = str(tmp_path / "vectors.json")
    config.memory.enable_fact_extraction = False
    config.memory.enable_episodes = False
    config.memory.profile_path = None
    config.memory.recall_min_query_chars = 0
    embedder = VectorEmbedder({"Evangelion": (1.0, 0.0)}, default=(0.0, 1.0))
    manager = MemoryManager(config, embedder=embedder)

    assert manager.store_moment("we watched the Evangelion finale together") is True
    records = list(manager.vector_store.dump())
    assert records[0].metadata["type"] == "moment"

    hits = manager.recall("Evangelion")
    assert hits and "Evangelion finale" in hits[0]


def test_recall_ages_moment_records_temporally(tmp_path: Path) -> None:
    from amadeus.memory.context import _age_phrase

    config = Config()
    config.memory.persist_path = None
    config.memory.enable_long_term = True
    config.memory.embeddings_path = str(tmp_path / "vectors.json")
    config.memory.enable_fact_extraction = False
    config.memory.enable_episodes = False
    config.memory.profile_path = None
    embedder = VectorEmbedder({"movie": (1.0, 0.0)}, default=(0.0, 1.0))
    manager = MemoryManager(config, embedder=embedder)
    manager.store_moment("we watched a movie together")

    records = list(manager.vector_store.dump())
    stamp = records[0].metadata["timestamp"]
    phrase = _age_phrase(stamp, dt.datetime.utcnow())
    assert phrase  # a humanized age string was produced, same as any fact


# ---------------------------------------------------------------------------
# Archive/forget works on a moment
# ---------------------------------------------------------------------------


def test_forget_archives_a_moment_like_any_fact(tmp_path: Path) -> None:
    embedder = VectorEmbedder({"beach": (1.0, 0.0)}, default=(0.0, 1.0))
    config = Config()
    config.memory.persist_path = None
    config.memory.enable_long_term = True
    config.memory.embeddings_path = str(tmp_path / "vectors.json")
    config.memory.enable_fact_extraction = False
    config.memory.enable_episodes = False
    config.memory.profile_path = None
    config.memory.recall_min_query_chars = 0
    manager = MemoryManager(config, embedder=embedder)
    manager.store_moment("we watched the beach episode together")

    forgotten = manager.forget("beach")
    assert forgotten == "we watched the beach episode together"
    assert list(manager.vector_store.dump()) == []

    from amadeus.memory.fact_archive import archive_path
    import json

    path = archive_path(manager)
    lines = [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]
    assert len(lines) == 1
    assert lines[0]["metadata"]["type"] == "moment"
    assert lines[0]["reason"] == "user_forget"


def test_archive_fact_works_on_a_moment_by_id(tmp_path: Path) -> None:
    embedder = VectorEmbedder({"picnic": (1.0, 0.0)}, default=(0.0, 1.0))
    config = Config()
    config.memory.persist_path = None
    config.memory.enable_long_term = True
    config.memory.embeddings_path = str(tmp_path / "vectors.json")
    config.memory.enable_fact_extraction = False
    config.memory.enable_episodes = False
    config.memory.profile_path = None
    manager = MemoryManager(config, embedder=embedder)
    manager.store_moment("we had a picnic together in the story")
    record_id = list(manager.vector_store.dump())[0].id

    text = manager.archive_fact(record_id, "user_forget")
    assert text == "we had a picnic together in the story"
    assert list(manager.vector_store.dump()) == []


# ---------------------------------------------------------------------------
# Moments never treated as a user-stated fact for supersede-hint purposes
# ---------------------------------------------------------------------------


def test_moment_not_superseded_by_unrelated_fact_hint(tmp_path: Path) -> None:
    embedder = VectorEmbedder({"movie": (1.0, 0.0)}, default=(0.0, 1.0))
    config = Config()
    config.memory.persist_path = None
    config.memory.enable_long_term = True
    config.memory.embeddings_path = str(tmp_path / "vectors.json")
    config.memory.enable_fact_extraction = False
    config.memory.enable_episodes = False
    config.memory.profile_path = None
    config.memory.fact_supersede_hint_threshold = 0.0  # would match anything
    manager = MemoryManager(config, embedder=embedder)
    manager.store_moment("we watched a movie together")

    # A hint-driven fact store that "replaces" the moment's text must not
    # touch it — moments are excluded from the supersede scan entirely.
    manager._store_fact(
        "user corrected something unrelated",
        salience=0.9,
        replaces="we watched a movie together",
    )

    records = list(manager.vector_store.dump())
    types = sorted((r.metadata or {}).get("type") for r in records)
    assert types == ["fact", "moment"]  # both survive, neither archived
