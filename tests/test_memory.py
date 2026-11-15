from typing import List

import time
from pathlib import Path

import pytest

from amadeus.config import Config

from amadeus.memory.manager import PROTECTED, EmbeddingProvider, MemoryManager
from amadeus.providers.llm.base import LLMProvider


def test_memory_summary_trigger() -> None:
    config = Config()
    config.memory.summary_trigger_messages = 2
    config.memory.persist_path = None
    config.memory.summary_max_chars = 10
    manager = MemoryManager(config)
    manager.add_message("user", "Hello")
    assert manager.summary is None
    manager.add_message("assistant", "Hi")
    assert manager.summary is not None
    assert manager.summary_info["message_count"] == 2
    assert manager.summary_info["strategy"] == "truncate"


def test_memory_persist_and_clear(tmp_path: Path) -> None:
    config = Config()
    config.memory.persist_path = str(tmp_path / "memory.json")
    config.memory.embeddings_path = None
    manager = MemoryManager(config)
    manager.add_message("user", "Remember this")
    manager.add_message("assistant", "Noted")

    restored = MemoryManager(config)
    restored_messages = [msg.content for msg in list(restored.short_term)]
    assert "Remember this" in restored_messages

    restored.clear()
    assert not (tmp_path / "memory.json").exists()

    fresh = MemoryManager(config)
    assert list(fresh.short_term) == []


def test_scope_context_to_session_default_off_still_restores(tmp_path: Path) -> None:
    """Explicit control for the flag-off case: unchanged legacy behavior."""
    config = Config()
    config.memory.persist_path = str(tmp_path / "memory.json")
    config.memory.embeddings_path = None
    assert config.memory.scope_context_to_session is False

    manager = MemoryManager(config)
    manager.add_message("user", "Remember this")
    manager.add_message("assistant", "Noted")
    manager.summary = "a running summary"
    manager._persist()

    restored = MemoryManager(config)
    assert "Remember this" in [m.content for m in restored.short_term]
    assert restored.summary == "a running summary"


def test_scope_context_to_session_skips_short_term_and_summary_on_reload(
    tmp_path: Path,
) -> None:
    """flag on: a new MemoryManager over the same paths starts with empty
    short_term/summary, while long-term facts and profile fields survive."""
    config = _extraction_config(tmp_path)
    config.memory.scope_context_to_session = True
    manager = MemoryManager(config, embedder=VectorEmbedder())
    manager.add_message("user", "Hello there")
    manager.add_message("assistant", "Hi")
    manager.summary = "a running summary"
    manager.summary_info = {"strategy": "truncate", "message_count": 2}
    manager.profile.facts["name"] = "Alex"
    manager._save_profile()
    # Seed a long-term vector fact directly (avoids depending on the async
    # extraction job's timing) to isolate what this test is actually about:
    # long-term recall surviving reload regardless of the new flag.
    from amadeus.memory.vector_store import VectorRecord

    manager.vector_store.add(
        VectorRecord(vector=[1.0, 0.0], text="User loves cats", metadata={})
    )
    manager.vector_store.persist()
    manager._persist()

    restored = MemoryManager(config, embedder=VectorEmbedder())
    assert list(restored.short_term) == []
    assert restored.summary is None
    assert restored.summary_info == {}
    # Long-term memory is unaffected by the flag.
    assert restored.profile is not None
    assert restored.profile.facts.get("name") == "Alex"
    recall = restored.recall("cats")
    assert recall == ["User loves cats"]


class DummyEmbedder(EmbeddingProvider):
    def embed_text(self, text: str) -> List[float]:
        return [float(len(text))]

    def similarity(self, a: List[float], b: List[float]) -> float:
        if not a or not b:
            return 0.0
        return 1.0 if round(a[0], 5) == round(b[0], 5) else 0.0


def test_memory_long_term_dedupe(tmp_path: Path) -> None:
    config = Config()
    config.memory.persist_path = None
    config.memory.enable_long_term = True
    config.memory.embeddings_path = str(tmp_path / "vectors.json")
    config.memory.dedupe_threshold = 0.9
    # Raw-message storage only runs with fact extraction (default on) off.
    config.memory.enable_fact_extraction = False
    config.memory.enable_episodes = False
    config.memory.profile_path = None
    # Legacy raw-message (untyped) recall path; facts-only filters it by default.
    config.memory.recall_facts_only = False
    config.memory.recall_min_query_chars = 0  # recalls with short queries
    manager = MemoryManager(config, embedder=DummyEmbedder())

    # Embedding runs on the background executor now; flush before asserting.
    manager.add_message("user", "hello")
    manager.flush()
    assert manager.long_term_count == 1

    manager.add_message("user", "hello")  # duplicate vector
    manager.flush()
    assert manager.long_term_count == 1

    recall = manager.recall("hello")
    assert recall and recall[0] == "hello"


class FakeLLMProvider(LLMProvider):
    name = "fake-summary"

    def __init__(self, text: str) -> None:
        self.text = text
        self.calls = 0
        self.last_kwargs: dict = {}

    def generate(self, messages, **kwargs):  # type: ignore[override]
        self.calls += 1
        self.last_kwargs = kwargs
        return self.text


def test_memory_llm_summary(tmp_path: Path) -> None:
    config = Config()
    config.memory.summary_trigger_messages = 1
    config.memory.persist_path = str(tmp_path / "memory.json")
    config.memory.embeddings_path = None
    summarizer = FakeLLMProvider("Concise summary")
    manager = MemoryManager(config, summary_provider=summarizer)
    manager.add_message("user", "Hello world")

    for _ in range(20):
        if manager.summary_info.get("strategy") == "llm":
            break
        time.sleep(0.05)

    assert manager.summary == "Concise summary"
    assert manager.summary_info["strategy"] == "llm"
    assert summarizer.calls >= 1
    manager.clear()


class VectorEmbedder(EmbeddingProvider):
    """Maps texts to fixed 2-D unit vectors so cosine scores are graded
    (1-D fakes give degenerate +/-1 similarities)."""

    def __init__(self, mapping: dict | None = None, default=(1.0, 0.0)) -> None:
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


class FailingLLMProvider(LLMProvider):
    name = "fake-broken"

    def generate(self, messages, **kwargs):  # type: ignore[override]
        raise RuntimeError("provider down")


def _wait_for(predicate, attempts: int = 40) -> None:
    for _ in range(attempts):
        if predicate():
            return
        time.sleep(0.05)


def _iso_days_ago(days: float) -> str:
    import datetime as dt

    return (dt.datetime.utcnow() - dt.timedelta(days=days)).isoformat()


def _long_term_config(tmp_path: Path, **overrides) -> Config:
    config = Config()
    config.memory.persist_path = None
    config.memory.enable_long_term = True
    config.memory.embeddings_path = str(tmp_path / "vectors.json")
    config.memory.dedupe_threshold = 0.0
    # These tests recall with short synthetic queries ("cats"); the
    # production trivial-query gate would skip them, so pin it off here and
    # test the gate explicitly in test_recall_skips_trivial_queries.
    config.memory.recall_min_query_chars = 0
    # These tests exercise the legacy raw-message long-term path; fact
    # extraction (now on by default) replaces it and would also load a
    # profile from the default repo-root profile_path — pin both off.
    # (_extraction_config re-enables extraction with an isolated profile.)
    config.memory.enable_fact_extraction = False
    config.memory.enable_episodes = False
    config.memory.profile_path = None
    # These tests exercise the legacy raw-message (untyped) recall path, which
    # the facts-only default now filters out; keep it visible here so the
    # ranking/decay assertions still see those records. Facts-only + summary
    # dedupe have their own dedicated tests below.
    config.memory.recall_facts_only = False
    config.memory.recall_summary_dedupe = 0.0
    for key, value in overrides.items():
        setattr(config.memory, key, value)
    return config


def test_recall_threshold_filters_low_similarity(tmp_path: Path) -> None:
    embedder = VectorEmbedder({"cats": (1.0, 0.0), "weather": (0.0, 1.0)})
    config = _long_term_config(tmp_path, recall_min_score=0.35)
    manager = MemoryManager(config, embedder=embedder)
    manager.add_message("user", "I love cats")
    manager.add_message("user", "the weather is grim")
    manager.flush()
    assert manager.long_term_count == 2

    recall = manager.recall("tell me about cats")
    assert recall == ["I love cats"]


def test_recall_embedder_relevance_floor_overrides_config(tmp_path: Path) -> None:
    """The embedder's declared noise floor governs when stricter than config.

    Semantic embedders (gemini-embedding-2) score completely unrelated text
    ~0.45-0.59, so the 0.35 config default admitted top-k memories on every
    turn regardless of relevance. The recall path takes
    max(recall_min_score, embedder.relevance_floor).
    """
    from amadeus import telemetry

    class SemanticEmbedder(VectorEmbedder):
        relevance_floor = 0.9

    # Stored record embeds to (0.8, 0.6); an unrelated query embeds to the
    # (1.0, 0.0) default -> cosine 0.8: above the 0.35 config threshold,
    # below the embedder's 0.9 floor.
    embedder = SemanticEmbedder({"cats": (0.8, 0.6)})
    config = _long_term_config(tmp_path, recall_min_score=0.35)
    manager = MemoryManager(config, embedder=embedder)
    manager.add_message("user", "I love cats")
    manager.flush()
    assert manager.long_term_count == 1

    before = telemetry.snapshot().get("memory.recall_floor_filtered", 0)
    assert manager.recall("something unrelated") == []
    assert telemetry.snapshot().get("memory.recall_floor_filtered", 0) == before + 1

    # A genuinely relevant match (cosine 1.0 via the same "cats" mapping)
    # still clears the floor — the gate kills noise, not recall itself.
    assert manager.recall("tell me about cats") == ["I love cats"]


def test_reload_long_term_prevents_cross_manager_resurrection(tmp_path: Path) -> None:
    """Two live managers over the same JSON store: a delete in one must not
    be resurrected by the other's next whole-file persist.

    This is the synapse-map bug (2026-07-28): deletions landed on disk, then
    a sibling session's next store write rewrote its stale RAM copy back —
    last-writer-wins. reload_long_term_from_disk is the sibling-side half of
    the fix (the web layer calls it on every live sibling after a mutation).
    """
    embedder = VectorEmbedder({"cilantro": (0.8, 0.6)})

    def make_manager() -> MemoryManager:
        config = _long_term_config(tmp_path)
        config.memory.persist_path = str(tmp_path / "st.json")
        return MemoryManager(config, embedder=embedder)

    a = make_manager()
    assert a.remember("user hates cilantro")
    b = make_manager()
    assert b.long_term_count == 1

    record = b.vector_store.dump()[0]
    assert b.forget_id(record.id) == "user hates cilantro"

    # Without the reload, A's next write resurrects the deleted record.
    a.reload_long_term_from_disk()
    assert a.long_term_count == 0
    assert a.remember("user likes trains")

    # Disk state after A's write (a fresh load) must NOT contain the
    # deleted record — before the fix it did.
    fresh = make_manager()
    assert [rec.text for rec in fresh.vector_store.dump()] == ["user likes trains"]


def test_gemini_embedding_declares_relevance_floor() -> None:
    """Pin the measured floor so a refactor can't silently drop it back to
    the config-only 0.35 (which filters nothing on this embedder)."""
    from amadeus.memory.embedders import EmbeddingProvider, GeminiEmbedding

    assert EmbeddingProvider.relevance_floor == 0.0
    assert GeminiEmbedding.relevance_floor == 0.60


def test_recall_recency_decay_prefers_recent(tmp_path: Path) -> None:
    from amadeus.memory.vector_store import VectorRecord

    embedder = VectorEmbedder()
    config = _long_term_config(tmp_path, recency_half_life_days=30.0)
    manager = MemoryManager(config, embedder=embedder)
    manager.vector_store.add(
        VectorRecord(
            vector=[1.0, 0.0],
            text="old fact",
            metadata={"type": "fact", "timestamp": _iso_days_ago(90), "strength": 1.0},
        )
    )
    manager.vector_store.add(
        VectorRecord(
            vector=[1.0, 0.0],
            text="new fact",
            metadata={"type": "fact", "timestamp": _iso_days_ago(0), "strength": 1.0},
        )
    )

    matches = manager.recall_detailed("anything", limit=1)
    assert [m.text for m in matches] == ["new fact"]

    # Records without a timestamp (legacy) get neutral recency, not buried.
    manager.vector_store.add(
        VectorRecord(vector=[1.0, 0.0], text="legacy", metadata={})
    )
    texts = [m.text for m in manager.recall_detailed("anything", limit=3)]
    assert "legacy" in texts


def test_recall_facts_only_excludes_episodes_and_untyped(tmp_path: Path) -> None:
    from amadeus.memory.vector_store import VectorRecord

    embedder = VectorEmbedder()
    config = _long_term_config(tmp_path)
    config.memory.recall_facts_only = True
    manager = MemoryManager(config, embedder=embedder)
    manager.vector_store.add(
        VectorRecord(
            vector=[1.0, 0.0],
            text="durable fact",
            metadata={"type": "fact", "timestamp": _iso_days_ago(0), "strength": 1.0},
        )
    )
    manager.vector_store.add(
        VectorRecord(
            vector=[1.0, 0.0],
            text="2026: the user greeted and mentioned Debbie",
            metadata={
                "type": "episode",
                "timestamp": _iso_days_ago(0),
                "strength": 1.0,
            },
        )
    )
    manager.vector_store.add(
        VectorRecord(
            vector=[1.0, 0.0],
            text="raw message capture",
            metadata={"role": "user", "timestamp": _iso_days_ago(0)},
        )
    )

    texts = [m.text for m in manager.recall_detailed("anything", limit=5)]
    assert texts == ["durable fact"]


def test_recall_facts_only_keeps_pinned_episode(tmp_path: Path) -> None:
    from amadeus.memory.vector_store import VectorRecord

    embedder = VectorEmbedder()
    config = _long_term_config(tmp_path)
    config.memory.recall_facts_only = True
    manager = MemoryManager(config, embedder=embedder)
    manager.vector_store.add(
        VectorRecord(
            vector=[1.0, 0.0],
            text="pinned episode the user asked to keep",
            metadata={
                "type": "episode",
                "timestamp": _iso_days_ago(0),
                "strength": 1.0,
                "pinned": True,
            },
        )
    )
    texts = [m.text for m in manager.recall_detailed("anything", limit=5)]
    assert texts == ["pinned episode the user asked to keep"]


def test_recall_summary_dedupe_drops_near_copy_of_summary(tmp_path: Path) -> None:
    from amadeus.memory.vector_store import VectorRecord

    # Both facts are retrievable by the query (both share the (1,0) direction so
    # they clear min_score), but only the "restated" fact embeds parallel to the
    # summary and is dropped as a near-copy; the other survives.
    embedder = VectorEmbedder(
        {
            "query": (1.0, 0.0),
            "summary-topic": (1.0, 0.0),
            "other unrelated": (0.6, 0.8),
        }
    )
    config = _long_term_config(tmp_path)
    config.memory.recall_facts_only = True
    config.memory.recall_summary_dedupe = 0.9
    manager = MemoryManager(config, embedder=embedder)
    manager.summary = "summary-topic recap already in context"
    manager.vector_store.add(
        VectorRecord(
            vector=[1.0, 0.0],
            text="summary-topic restated as a fact",
            metadata={"type": "fact", "timestamp": _iso_days_ago(0), "strength": 1.0},
        )
    )
    manager.vector_store.add(
        VectorRecord(
            vector=[0.6, 0.8],
            text="other unrelated fact",
            metadata={"type": "fact", "timestamp": _iso_days_ago(0), "strength": 1.0},
        )
    )

    texts = [m.text for m in manager.recall_detailed("query about the topic", limit=5)]
    assert "summary-topic restated as a fact" not in texts
    assert "other unrelated fact" in texts


def test_assistant_messages_not_stored_long_term(tmp_path: Path) -> None:
    config = _long_term_config(tmp_path)
    manager = MemoryManager(config, embedder=VectorEmbedder())
    manager.add_message("assistant", "my own words")
    manager.flush()
    assert manager.long_term_count == 0


def test_persist_atomic_leaves_no_tmp(tmp_path: Path) -> None:
    config = Config()
    config.memory.persist_path = str(tmp_path / "memory.json")
    config.memory.embeddings_path = None
    manager = MemoryManager(config)
    manager.add_message("user", "hello")
    assert (tmp_path / "memory.json").exists()
    assert not list(tmp_path.glob("*.tmp"))


EXTRACTION_JSON = (
    '{"summary": "Talked about the lab.", '
    '"new_facts": [{"text": "User studies neuroscience", "salience": 0.8}], '
    '"profile_updates": {"name": "Alex"}, '
    '"open_threads": ["fMRI pipeline bug"], '
    '"tone": 0.4}'
)


def _extraction_config(tmp_path: Path, **overrides) -> Config:
    config = _long_term_config(tmp_path)
    config.memory.summary_trigger_messages = 1
    config.memory.enable_fact_extraction = True
    config.memory.profile_path = str(tmp_path / "profile.json")
    for key, value in overrides.items():
        setattr(config.memory, key, value)
    return config


def test_extraction_parses_facts_and_profile(tmp_path: Path) -> None:
    import json

    summarizer = FakeLLMProvider(EXTRACTION_JSON)
    manager = MemoryManager(
        _extraction_config(tmp_path),
        embedder=VectorEmbedder(),
        summary_provider=summarizer,
    )
    manager.add_message("user", "I study neuroscience")
    _wait_for(lambda: manager.summary_info.get("strategy") == "llm")
    manager.flush()

    assert manager.summary == "Talked about the lab."
    assert summarizer.last_kwargs.get("max_tokens") == 400
    # The fact (not the raw message) is what landed in long-term memory.
    assert manager.long_term_count == 1
    records = list(manager.vector_store.dump())
    assert records[0].text == "User studies neuroscience"
    assert records[0].metadata["type"] == "fact"
    assert records[0].metadata["salience"] == 0.8

    profile_data = json.loads((tmp_path / "profile.json").read_text())
    assert profile_data["facts"]["name"] == "Alex"
    # Threads are persisted with the date they were first noted, so ages can
    # be phrased ("a week ago") and stale threads expired.
    assert [t["text"] for t in profile_data["open_threads"]] == ["fMRI pipeline bug"]
    assert profile_data["open_threads"][0]["noted"]

    block = manager.profile_block()
    assert block is not None and "Alex" in block


def test_finalize_session_extracts_facts_from_short_session(tmp_path: Path) -> None:
    # A session shorter than the summary trigger must still get one fact-
    # extraction pass at session end — per-turn recall is facts-only, so
    # without it a durable fact from a quick chat ("I switched labs") would
    # never become recallable at all.
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
    assert records[0].text == "User studies neuroscience"


def test_extraction_malformed_json_retains_summary(tmp_path: Path) -> None:
    summarizer = FakeLLMProvider('{"summary": "trunca')  # broken JSON
    manager = MemoryManager(
        _extraction_config(tmp_path),
        embedder=VectorEmbedder(),
        summary_provider=summarizer,
    )
    manager.summary = "established summary"
    manager.summary_info = {"revision": 1}
    manager.add_message("user", "hello there")
    _wait_for(lambda: summarizer.calls >= 1)
    manager.flush()

    assert manager.summary == "established summary"
    assert manager.summary_info["strategy"] == "retained"
    assert manager.long_term_count == 0


def test_extraction_plain_prose_falls_back_to_summary(tmp_path: Path) -> None:
    summarizer = FakeLLMProvider("They talked about science all evening.")
    manager = MemoryManager(
        _extraction_config(tmp_path),
        embedder=VectorEmbedder(),
        summary_provider=summarizer,
    )
    manager.add_message("user", "hello there")
    _wait_for(lambda: manager.summary_info.get("strategy") == "llm")
    manager.flush()
    assert manager.summary == "They talked about science all evening."


def test_fact_supersede_replaces_contradiction(tmp_path: Path) -> None:
    embedder = VectorEmbedder({"likes": (1.0, 0.0)})
    manager = MemoryManager(_extraction_config(tmp_path), embedder=embedder)
    manager._store_fact("User likes tea", 0.5)
    manager._store_fact("User likes coffee now", 0.6)

    records = list(manager.vector_store.dump())
    assert len(records) == 1
    assert records[0].text == "User likes coffee now"
    assert records[0].metadata.get("supersedes")


def test_profile_survives_clear_but_not_full_forget(tmp_path: Path) -> None:
    manager = MemoryManager(_extraction_config(tmp_path), embedder=VectorEmbedder())
    manager.profile.facts["name"] = "Alex"
    manager._save_profile()
    profile_file = tmp_path / "profile.json"
    assert profile_file.exists()

    manager.clear()
    assert profile_file.exists()

    manager.clear(full_forget=True)
    assert not profile_file.exists()


def test_message_timestamp_backcompat(tmp_path: Path) -> None:
    import json

    persist = tmp_path / "memory.json"
    # A completed pair: a lone trailing user message would be healed away as
    # a failed-turn orphan before this test's timestamp assertion ran.
    persist.write_text(
        json.dumps(
            {
                "messages": [
                    {"role": "user", "content": "old message"},
                    {"role": "assistant", "content": "old reply"},
                ],
                "summary": None,
                "summary_metadata": {},
            }
        )
    )
    config = Config()
    config.memory.persist_path = str(persist)
    config.memory.embeddings_path = None
    manager = MemoryManager(config)
    assert manager.short_term[0].timestamp is None
    manager.add_message("user", "new message")
    assert manager.short_term[-1].timestamp is not None


def test_gap_creates_episode_and_bumps_session(tmp_path: Path) -> None:
    summarizer = FakeLLMProvider("We debugged the pipeline; user was upbeat.")
    config = _extraction_config(tmp_path, enable_fact_extraction=False)
    config.memory.enable_episodes = True
    config.memory.session_gap_minutes = 30
    config.memory.summary_trigger_messages = 50  # keep summaries out of the way
    manager = MemoryManager(
        config, embedder=VectorEmbedder(), summary_provider=summarizer
    )
    sessions_before = manager.profile.session_count

    manager.add_message("user", "hello")
    manager.flush()
    manager.short_term[-1].timestamp = _iso_days_ago(1)  # simulate a long silence
    manager.add_message("user", "back again")
    manager.flush()

    episodes = [
        rec
        for rec in manager.vector_store.dump()
        if rec.metadata.get("type") == "episode"
    ]
    assert len(episodes) == 1
    assert "debugged the pipeline" in episodes[0].text
    assert manager.profile.session_count == sessions_before + 1
    assert manager.profile.last_episode


def _episode_count(manager: MemoryManager) -> int:
    return sum(
        1
        for rec in manager.vector_store.dump()
        if (rec.metadata or {}).get("type") == "episode"
    )


def test_episode_supersede_replaces_near_duplicate(tmp_path: Path) -> None:
    from amadeus.memory.manager import Message

    # All episodes embed to the same direction -> perfect similarity -> the
    # newer episode supersedes the older one instead of accreting.
    embedder = VectorEmbedder()
    config = _extraction_config(tmp_path, enable_fact_extraction=False)
    config.memory.enable_episodes = True
    config.memory.episode_supersede_threshold = 0.9
    summarizer = FakeLLMProvider("The user greeted the assistant.")
    manager = MemoryManager(config, embedder=embedder, summary_provider=summarizer)

    for _ in range(4):
        manager._run_episode_job([Message(role="user", content="hi")], None)

    assert _episode_count(manager) == 1


def test_episode_supersede_disabled_accretes(tmp_path: Path) -> None:
    from amadeus.memory.manager import Message

    embedder = VectorEmbedder()
    config = _extraction_config(tmp_path, enable_fact_extraction=False)
    config.memory.enable_episodes = True
    config.memory.episode_supersede_threshold = 0.0  # disabled
    config.memory.max_episodes = 0  # also disable the cap
    summarizer = FakeLLMProvider("The user greeted the assistant.")
    manager = MemoryManager(config, embedder=embedder, summary_provider=summarizer)

    for _ in range(3):
        manager._run_episode_job([Message(role="user", content="hi")], None)

    assert _episode_count(manager) == 3


def test_episode_supersede_keeps_pinned(tmp_path: Path) -> None:
    from amadeus.memory.manager import Message
    from amadeus.memory.vector_store import VectorRecord

    embedder = VectorEmbedder()
    config = _extraction_config(tmp_path, enable_fact_extraction=False)
    config.memory.enable_episodes = True
    config.memory.episode_supersede_threshold = 0.9
    summarizer = FakeLLMProvider("The user greeted the assistant.")
    manager = MemoryManager(config, embedder=embedder, summary_provider=summarizer)
    manager.vector_store.add(
        VectorRecord(
            vector=[1.0, 0.0],
            text="a pinned episode to keep",
            metadata={"type": "episode", "pinned": True, "strength": 1.0},
        )
    )

    manager._run_episode_job([Message(role="user", content="hi")], None)

    texts = [
        rec.text
        for rec in manager.vector_store.dump()
        if (rec.metadata or {}).get("type") == "episode"
    ]
    assert "a pinned episode to keep" in texts  # pinned survives
    assert _episode_count(manager) == 2  # pinned + the new one


def test_max_episodes_prunes_lowest_salience_oldest(tmp_path: Path) -> None:
    from amadeus.memory.vector_store import VectorRecord

    embedder = VectorEmbedder()
    config = _extraction_config(tmp_path, enable_fact_extraction=False)
    config.memory.max_episodes = 2
    config.memory.episode_supersede_threshold = 0.0  # isolate the cap behavior
    manager = MemoryManager(config, embedder=embedder)

    # Three existing episodes with distinct salience; keep the two highest.
    for name, salience in [("low", 0.2), ("mid", 0.5), ("high", 0.9)]:
        manager.vector_store.add(
            VectorRecord(
                vector=[0.0, 1.0],
                text=f"{name} salience episode",
                metadata={
                    "type": "episode",
                    "salience": salience,
                    "timestamp": _iso_days_ago(1),
                    "strength": 1.0,
                },
            )
        )
    with manager._state_lock:
        manager._prune_episodes_over_cap()

    texts = {
        rec.text
        for rec in manager.vector_store.dump()
        if (rec.metadata or {}).get("type") == "episode"
    }
    assert "low salience episode" not in texts
    assert texts == {"mid salience episode", "high salience episode"}


def test_finalize_session_writes_episode(tmp_path: Path) -> None:
    summarizer = FakeLLMProvider("A quiet chat about thesis stress.")
    config = _extraction_config(tmp_path, enable_fact_extraction=False)
    config.memory.enable_episodes = True
    config.memory.summary_trigger_messages = 50
    manager = MemoryManager(
        config, embedder=VectorEmbedder(), summary_provider=summarizer
    )
    manager.add_message("user", "my thesis is due soon")
    manager.finalize_session(wait=True)

    episodes = [
        rec
        for rec in manager.vector_store.dump()
        if rec.metadata.get("type") == "episode"
    ]
    assert len(episodes) == 1
    assert manager.profile.last_episode and "thesis" in manager.profile.last_episode


def test_episode_llm_failure_falls_back_to_summary(tmp_path: Path) -> None:
    config = _extraction_config(tmp_path, enable_fact_extraction=False)
    config.memory.enable_episodes = True
    config.memory.summary_trigger_messages = 50
    manager = MemoryManager(
        config, embedder=VectorEmbedder(), summary_provider=FailingLLMProvider()
    )
    manager.summary = "running summary of the chat"
    manager.add_message("user", "hello")
    manager.finalize_session(wait=True)

    episodes = [
        rec
        for rec in manager.vector_store.dump()
        if rec.metadata.get("type") == "episode"
    ]
    assert len(episodes) == 1
    assert "running summary of the chat" in episodes[0].text


def test_reflection_stored_at_finalize(tmp_path: Path) -> None:
    summarizer = FakeLLMProvider("I believe they push themselves too hard.")
    config = _extraction_config(tmp_path)
    config.memory.enable_reflection = True
    config.memory.summary_trigger_messages = 50
    manager = MemoryManager(
        config, embedder=VectorEmbedder(), summary_provider=summarizer
    )
    manager.profile.facts["studies"] = "neuroscience"
    manager.finalize_session(wait=True)

    reflections = [
        rec
        for rec in manager.vector_store.dump()
        if rec.metadata.get("type") == "reflection"
    ]
    assert len(reflections) == 1
    assert reflections[0].metadata["salience"] == 0.8


def test_forgetting_excludes_decayed_records(tmp_path: Path) -> None:
    from amadeus.memory.vector_store import VectorRecord

    config = _long_term_config(
        tmp_path,
        enable_forgetting=True,
        strength_half_life_days=90.0,
        strength_floor=0.25,
        recency_half_life_days=0.0,  # isolate strength decay from recency
    )
    manager = MemoryManager(config, embedder=VectorEmbedder())
    manager.vector_store.add(
        VectorRecord(
            vector=[1.0, 0.0],
            text="ancient trivia",
            metadata={"type": "fact", "timestamp": _iso_days_ago(400), "strength": 1.0},
        )
    )
    manager.vector_store.add(
        VectorRecord(
            vector=[1.0, 0.0],
            text="fresh fact",
            metadata={"type": "fact", "timestamp": _iso_days_ago(0), "strength": 1.0},
        )
    )

    texts = [m.text for m in manager.recall_detailed("anything", limit=3)]
    assert texts == ["fresh fact"]


def test_reinforcement_bumps_strength(tmp_path: Path) -> None:
    from amadeus.memory.vector_store import VectorRecord

    config = _long_term_config(
        tmp_path, enable_forgetting=True, retrieval_reinforcement=0.15
    )
    manager = MemoryManager(config, embedder=VectorEmbedder())
    manager.vector_store.add(
        VectorRecord(
            vector=[1.0, 0.0],
            text="a fact",
            metadata={"type": "fact", "timestamp": _iso_days_ago(0), "strength": 1.0},
        )
    )
    # Legacy untyped record must never be reinforced.
    manager.vector_store.add(
        VectorRecord(vector=[1.0, 0.0], text="legacy", metadata={})
    )

    manager.recall_detailed("anything", limit=3)
    records = {rec.text: rec for rec in manager.vector_store.dump()}
    assert records["a fact"].metadata["strength"] == 1.15
    assert "strength" not in records["legacy"].metadata


def test_recall_framing_in_context(tmp_path: Path) -> None:
    from amadeus.memory.context import RECALL_FRAMING, assemble_turn_context

    config = _long_term_config(tmp_path)
    manager = MemoryManager(config, embedder=VectorEmbedder())
    manager.add_message("user", "seed memory about the lab")
    manager.flush()

    context, recall_items = assemble_turn_context(manager, "system prompt", "the lab?")
    assert recall_items == ["seed memory about the lab"]
    assert context[0].content == "system prompt"
    assert context[-1].content.startswith(RECALL_FRAMING)
    assert "Memory:" not in context[-1].content
    assert manager.short_term[-1].content == "the lab?"


def test_recall_recent_dedupe_suppresses_discussed_facts(tmp_path: Path) -> None:
    from amadeus.memory.vector_store import VectorRecord

    config = _long_term_config(tmp_path)
    config.memory.recall_recent_dedupe = 0.6
    config.memory.recall_facts_only = True
    manager = MemoryManager(config, embedder=VectorEmbedder())
    manager.vector_store.add(
        VectorRecord(
            vector=[1.0, 0.0],
            text="User is presenting a dementia poster at a Prague conference",
            metadata={"type": "fact", "timestamp": _iso_days_ago(1)},
        )
    )

    # Her own earlier reply already covered the fact this session, so
    # re-recalling it on an unrelated turn is pure repetition fuel.
    manager.add_message("user", "hello there")
    manager.add_message(
        "assistant",
        "How is the poster for the Prague dementia conference coming along? "
        "Presenting there is a big deal.",
    )
    manager.add_message("user", "unrelated question about tea brands")
    assert manager.recall_detailed("unrelated question about tea brands") == []

    # 0 disables the guard.
    config.memory.recall_recent_dedupe = 0.0
    assert [m.text for m in manager.recall_detailed("tea brands again")] == [
        "User is presenting a dementia poster at a Prague conference"
    ]


def test_recall_recent_dedupe_ignores_current_query(tmp_path: Path) -> None:
    from amadeus.memory.vector_store import VectorRecord

    config = _long_term_config(tmp_path)
    config.memory.recall_recent_dedupe = 0.6
    config.memory.recall_facts_only = True
    manager = MemoryManager(config, embedder=VectorEmbedder())
    manager.vector_store.add(
        VectorRecord(
            vector=[1.0, 0.0],
            text="User is presenting a dementia poster at a Prague conference",
            metadata={"type": "fact", "timestamp": _iso_days_ago(1)},
        )
    )

    # The just-committed user message IS the recall query — asking about a
    # topic directly must never count as "already discussed".
    query = "Remind me about my Prague conference poster on dementia?"
    manager.add_message("user", query)
    assert [m.text for m in manager.recall_detailed(query)] == [
        "User is presenting a dementia poster at a Prague conference"
    ]


def test_age_phrase_buckets() -> None:
    import datetime as dt

    from amadeus.memory.context import _age_phrase

    now = dt.datetime(2026, 7, 20, 12, 0, 0)

    def ago(**kwargs):
        return (now - dt.timedelta(**kwargs)).isoformat()

    assert _age_phrase(ago(hours=3), now) == "earlier today"
    assert _age_phrase(ago(hours=30), now) == "yesterday"
    assert _age_phrase(ago(days=4), now) == "4 days ago"
    assert _age_phrase(ago(days=10), now) == "about a week ago"
    assert _age_phrase(ago(days=21), now) == "about 3 weeks ago"
    assert _age_phrase(ago(days=90), now) == "about 3 months ago"
    assert _age_phrase(ago(days=400), now) == "over a year ago"
    assert _age_phrase(None, now) is None
    assert _age_phrase("not-a-date", now) is None
    # A future stamp (clock skew) must not produce nonsense.
    assert _age_phrase((now + dt.timedelta(days=2)).isoformat(), now) is None


def test_recall_annotates_age_in_context(tmp_path: Path) -> None:
    from amadeus.memory.context import assemble_turn_context
    from amadeus.memory.vector_store import VectorRecord

    config = _long_term_config(tmp_path)
    manager = MemoryManager(config, embedder=VectorEmbedder())
    manager.vector_store.add(
        VectorRecord(
            vector=[1.0, 0.0],
            text="user presents a poster in Prague",
            metadata={"type": "fact", "timestamp": _iso_days_ago(21)},
        )
    )
    manager.vector_store.add(
        VectorRecord(
            vector=[1.0, 0.0],
            text="stampless legacy fact",
            metadata={"type": "fact"},
        )
    )

    context, recall_items = assemble_turn_context(manager, "system", "prague trip?")
    # The prompt line carries the humanized age; recall_items stay raw text.
    body = context[-1].content
    assert "- user presents a poster in Prague (about 3 weeks ago)" in body
    assert "- stampless legacy fact (" not in body
    assert "- stampless legacy fact" in body
    # The legacy raw-message path also captures the query itself; only the
    # seeded records' presence matters here.
    assert {"user presents a poster in Prague", "stampless legacy fact"} <= set(
        recall_items
    )


def test_remember_stores_high_salience_fact(tmp_path: Path) -> None:
    manager = MemoryManager(_extraction_config(tmp_path), embedder=VectorEmbedder())
    assert manager.remember("User's birthday is July 25") is True

    records = list(manager.vector_store.dump())
    assert len(records) == 1
    assert records[0].metadata["type"] == "fact"
    assert records[0].metadata["salience"] == 0.9


def test_remember_unavailable_without_long_term(tmp_path: Path) -> None:
    config = Config()
    config.memory.persist_path = None
    config.memory.enable_long_term = False
    manager = MemoryManager(config)
    assert manager.remember("anything") is False


def test_remember_pinned_sets_metadata(tmp_path: Path) -> None:
    manager = MemoryManager(_extraction_config(tmp_path), embedder=VectorEmbedder())
    assert manager.remember("User's birthday is July 25", pinned=True) is True

    records = list(manager.vector_store.dump())
    assert len(records) == 1
    assert records[0].metadata["pinned"] is True


def test_pinned_fact_survives_strength_floor(tmp_path: Path) -> None:
    from amadeus.memory.vector_store import VectorRecord

    config = _long_term_config(
        tmp_path,
        enable_forgetting=True,
        strength_half_life_days=90.0,
        strength_floor=0.25,
        recency_half_life_days=0.0,
    )
    manager = MemoryManager(config, embedder=VectorEmbedder())
    manager.vector_store.add(
        VectorRecord(
            vector=[1.0, 0.0],
            text="ancient pinned fact",
            metadata={
                "type": "fact",
                "timestamp": _iso_days_ago(400),
                "strength": 1.0,
                "pinned": True,
            },
        )
    )
    manager.vector_store.add(
        VectorRecord(
            vector=[1.0, 0.0],
            text="ancient unpinned fact",
            metadata={"type": "fact", "timestamp": _iso_days_ago(400), "strength": 1.0},
        )
    )

    texts = [m.text for m in manager.recall_detailed("anything", limit=3)]
    assert texts == ["ancient pinned fact"]


def test_store_fact_does_not_supersede_pinned_near_duplicate(tmp_path: Path) -> None:
    embedder = VectorEmbedder(default=(1.0, 0.0))
    manager = MemoryManager(_extraction_config(tmp_path), embedder=embedder)
    manager.remember("User likes black tea", pinned=True)
    manager.remember("User likes black tea a lot")

    records = list(manager.vector_store.dump())
    # Both records survive: the pinned near-duplicate is never deleted.
    assert len(records) == 2
    pinned_records = [r for r in records if r.metadata.get("pinned")]
    assert len(pinned_records) == 1
    assert pinned_records[0].text == "User likes black tea"


def test_store_fact_skips_pinned_and_supersedes_next_unpinned_candidate(
    tmp_path: Path,
) -> None:
    """G2 regression: when the NEAREST candidate is pinned, the supersede
    scan must keep walking (not stop/break) and dedup the first UNPINNED
    candidate above threshold further down the similarity-ranked list —
    the old code's unconditional trailing `break` meant only the single
    nearest candidate was ever examined, so a pinned nearest neighbor
    silently defeated dedup for everything behind it."""
    embedder = VectorEmbedder(default=(1.0, 0.0))
    manager = MemoryManager(_extraction_config(tmp_path), embedder=embedder)

    # Nearest candidate (highest similarity, identical default vector):
    # pinned. Must be skipped, never superseded.
    manager.remember("User likes black tea", pinned=True)
    # Second candidate: same default vector (same similarity tier under this
    # embedder), unpinned — this is the one that should get superseded by
    # the new fact below.
    manager.remember("User likes black tea a lot")

    records_before = list(manager.vector_store.dump())
    assert len(records_before) == 2

    # A third, near-duplicate fact arrives: the pinned record must survive
    # untouched, and the unpinned near-duplicate must be superseded (not
    # left to accumulate) since the scan no longer stops at the pinned
    # candidate.
    manager.remember("User likes black tea very much")

    records_after = list(manager.vector_store.dump())
    texts = {rec.text: rec for rec in records_after}
    assert "User likes black tea" in texts, "pinned record must survive"
    assert texts["User likes black tea"].metadata["pinned"] is True
    assert "User likes black tea a lot" not in texts, (
        "unpinned near-duplicate behind the pinned one must still be "
        "superseded, not permanently protected by the pinned candidate "
        "blocking the scan"
    )
    assert "User likes black tea very much" in texts
    # Exactly the pinned original plus the newly-stored fact — the
    # unpinned duplicate was deduped away.
    assert len(records_after) == 2


def test_set_pinned_round_trip(tmp_path: Path) -> None:
    manager = MemoryManager(_extraction_config(tmp_path), embedder=VectorEmbedder())
    manager.remember("User likes hiking")
    record_id = list(manager.vector_store.dump())[0].id

    assert manager.set_pinned(record_id, True) is True
    assert manager.vector_store.get(record_id).metadata["pinned"] is True

    assert manager.set_pinned(record_id, False) is True
    assert manager.vector_store.get(record_id).metadata["pinned"] is False

    assert manager.set_pinned("nonexistent-id", True) is False


def test_edit_memory_preserves_id_changes_text_and_embedding(tmp_path: Path) -> None:
    embedder = VectorEmbedder({"hiking": (1.0, 0.0), "swimming": (0.0, 1.0)})
    manager = MemoryManager(_extraction_config(tmp_path), embedder=embedder)
    manager.remember("User likes hiking")
    record_id = list(manager.vector_store.dump())[0].id

    assert manager.edit_memory(record_id, "User likes swimming") is True

    records = list(manager.vector_store.dump())
    assert len(records) == 1
    assert records[0].id == record_id
    assert records[0].text == "User likes swimming"
    assert records[0].vector == [0.0, 1.0]
    assert "edited_at" in records[0].metadata

    assert manager.edit_memory("nonexistent-id", "whatever") is False


def test_forget_id_deletes_exactly_that_record(tmp_path: Path) -> None:
    embedder = VectorEmbedder({"hiking": (1.0, 0.0), "tea": (0.0, 1.0)})
    manager = MemoryManager(_extraction_config(tmp_path), embedder=embedder)
    manager.remember("User likes hiking")
    manager.remember("User likes tea")
    records = list(manager.vector_store.dump())
    target = next(r for r in records if r.text == "User likes hiking")

    forgotten = manager.forget_id(target.id)
    assert forgotten == "User likes hiking"
    remaining = [rec.text for rec in manager.vector_store.dump()]
    assert remaining == ["User likes tea"]

    assert manager.forget_id("nonexistent-id") is None


def test_forget_id_pinned_record_returns_protected(tmp_path: Path) -> None:
    """G4: deleting a pinned record BY ID must not silently succeed — the
    intended UI flow is unpin-then-delete."""
    embedder = VectorEmbedder({"hiking": (1.0, 0.0), "tea": (0.0, 1.0)})
    manager = MemoryManager(_extraction_config(tmp_path), embedder=embedder)
    manager.remember("User likes hiking", pinned=True)
    records = list(manager.vector_store.dump())
    target_id = records[0].id

    result = manager.forget_id(target_id)
    assert result is PROTECTED
    # Untouched — still present and still pinned.
    remaining = list(manager.vector_store.dump())
    assert len(remaining) == 1
    assert remaining[0].id == target_id
    assert remaining[0].metadata["pinned"] is True

    # Unpin, then forget_id succeeds normally.
    assert manager.set_pinned(target_id, False) is True
    assert manager.forget_id(target_id) == "User likes hiking"
    assert manager.vector_store.dump() == []


def test_forget_query_pinned_record_returns_protected(tmp_path: Path) -> None:
    """A pinned best-match is protected, not silently skipped — and MUST
    NOT fall through to the profile-facts scan (G3): the sentinel signals
    'don't touch anything' rather than looking elsewhere for something to
    delete."""
    embedder = VectorEmbedder({"cilantro": (1.0, 0.0), "tea": (0.0, 1.0)})
    manager = MemoryManager(_extraction_config(tmp_path), embedder=embedder)
    manager.remember("User hates cilantro", pinned=True)
    # A same-substring profile fact that forget() must NOT fall through to
    # and delete when the vector match is protected.
    manager.profile.facts["cilantro_opinion"] = "hates it"

    forgotten = manager.forget("cilantro")
    assert forgotten is PROTECTED
    remaining = [rec.text for rec in manager.vector_store.dump()]
    assert remaining == ["User hates cilantro"]
    assert manager.profile.facts.get("cilantro_opinion") == "hates it"


def test_forget_removes_matched_record(tmp_path: Path) -> None:
    embedder = VectorEmbedder({"cilantro": (1.0, 0.0), "tea": (0.0, 1.0)})
    manager = MemoryManager(_extraction_config(tmp_path), embedder=embedder)
    manager.remember("User hates cilantro")
    manager.remember("User likes tea")

    forgotten = manager.forget("cilantro")
    assert forgotten == "User hates cilantro"
    remaining = [rec.text for rec in manager.vector_store.dump()]
    assert remaining == ["User likes tea"]


def test_forget_no_match_returns_none(tmp_path: Path) -> None:
    embedder = VectorEmbedder({"tea": (1.0, 0.0), "quantum": (0.0, 1.0)})
    config = _extraction_config(tmp_path)
    config.memory.recall_min_score = 0.5
    manager = MemoryManager(config, embedder=embedder)
    manager.remember("User likes tea")
    assert manager.forget("quantum physics") is None


def test_forget_pops_profile_fact(tmp_path: Path) -> None:
    embedder = VectorEmbedder({"vector-only": (1.0, 0.0)}, default=(0.0, 1.0))
    config = _extraction_config(tmp_path)
    config.memory.recall_min_score = 0.99  # keep the vector path from matching
    manager = MemoryManager(config, embedder=embedder)
    manager.profile.facts["favorite_drink"] = "black tea"
    manager._save_profile()

    forgotten = manager.forget("favorite_drink")
    assert forgotten == "favorite_drink: black tea"
    assert "favorite_drink" not in manager.profile.facts


def test_forget_fact_removes_by_exact_key(tmp_path: Path) -> None:
    manager = MemoryManager(_extraction_config(tmp_path), embedder=VectorEmbedder())
    manager.profile.facts["current_project"] = "the manuscript"
    manager.profile.facts["interests"] = "neuroscience"
    manager._save_profile()

    assert manager.forget_fact("current_project") is True
    assert "current_project" not in manager.profile.facts
    assert "interests" in manager.profile.facts  # untouched
    assert manager.forget_fact("nonexistent") is False
    assert manager.profile_block_dirty is True


def test_forget_open_thread_removes_by_text(tmp_path: Path) -> None:
    manager = MemoryManager(_extraction_config(tmp_path), embedder=VectorEmbedder())
    manager.profile.set_open_threads(
        ["reviewing the manuscript", "philosophy of consciousness"], max_threads=3
    )
    manager._save_profile()

    assert manager.forget_open_thread("reviewing the manuscript") is True
    remaining = [
        t["text"] if isinstance(t, dict) else t for t in manager.profile.open_threads
    ]
    assert remaining == ["philosophy of consciousness"]
    assert manager.forget_open_thread("not a real thread") is False


def test_memory_overview_shape(tmp_path: Path) -> None:
    embedder = VectorEmbedder({"neuroscience": (1.0, 0.0), "birthday": (0.0, 1.0)})
    manager = MemoryManager(_extraction_config(tmp_path), embedder=embedder)
    manager.profile.facts["name"] = "Alex"
    manager.remember("User studies neuroscience")
    manager.remember("User's birthday is July 25", pinned=True)

    overview = manager.memory_overview()
    assert overview["profile"]["facts"]["name"] == "Alex"
    assert set(overview["facts"]) == {
        "User studies neuroscience",
        "User's birthday is July 25",
    }
    assert overview["counts"] == {
        "facts": 2,
        "episodes": 0,
        "reflections": 0,
        "moments": 0,
    }
    assert len(overview["fact_records"]) == 2
    by_text = {rec["text"]: rec for rec in overview["fact_records"]}
    assert by_text["User studies neuroscience"]["pinned"] is False
    assert by_text["User's birthday is July 25"]["pinned"] is True
    assert all("id" in rec and "timestamp" in rec for rec in overview["fact_records"])
    assert overview["episode_records"] == []

    # Degrades to empty shapes without profile/store.
    bare_config = Config()
    bare_config.memory.persist_path = None
    bare_config.memory.enable_long_term = False
    bare_config.memory.enable_fact_extraction = False
    bare_config.memory.enable_episodes = False
    bare_config.memory.profile_path = None
    bare = MemoryManager(bare_config)
    empty = bare.memory_overview()
    assert empty["profile"] == {} and empty["facts"] == []
    assert empty["fact_records"] == [] and empty["episode_records"] == []


def test_should_greet_gating(tmp_path: Path) -> None:
    config = Config()
    config.memory.persist_path = None
    config.memory.enable_fact_extraction = False
    config.memory.enable_episodes = False
    config.memory.profile_path = None
    config.memory.enable_greeting = False
    manager = MemoryManager(config)
    assert manager.should_greet() is False  # greeting disabled

    config.memory.enable_greeting = True
    config.memory.session_gap_minutes = 30
    assert manager.should_greet() is True  # empty session

    manager.add_message("user", "hi")
    assert manager.should_greet() is False  # live conversation

    manager.short_term[-1].timestamp = _iso_days_ago(1)
    assert manager.should_greet() is True  # long silence = fresh session


def test_familiarity_guardedness_curve() -> None:
    import pytest

    from amadeus.memory.profile import familiarity_guardedness

    assert familiarity_guardedness(0.6, 0) == 0.6
    # Warms up with sessions but never below the 0.25 floor.
    assert familiarity_guardedness(0.8, 4) < 0.70
    assert familiarity_guardedness(0.8, 15) == pytest.approx(0.35)
    assert familiarity_guardedness(0.5, 100) == 0.25
    # Warming stops after 15 sessions.
    assert familiarity_guardedness(0.8, 40) == familiarity_guardedness(0.8, 15)


def test_profile_render_block(tmp_path: Path) -> None:
    from amadeus.memory.profile import UserProfile

    empty = UserProfile()
    assert empty.render_block() is None

    profile = UserProfile(
        facts={"name": "Alex"},
        first_met="2026-05-01",
        session_count=14,
        open_threads=["MeloTTS latency"],
        last_episode="2026-07-01: talked about thesis stress",
    )
    block = profile.render_block()
    assert "name: Alex" in block
    assert "14 times since 2026-05-01" in block
    assert "MeloTTS latency" in block
    assert "never recite" in block


def test_profile_populates_without_embedder(tmp_path: Path) -> None:
    """Personality memory (profile facts/open threads) must not depend on an
    embedder — the packaged desktop build ships no embedder, and
    _on_summary_ready's profile-merge path calls only the summary LLM."""

    config = Config()
    config.memory.persist_path = None
    config.memory.embeddings_path = None
    config.memory.enable_long_term = False
    config.memory.enable_fact_extraction = True
    config.memory.enable_episodes = True
    config.memory.summary_trigger_messages = 1
    config.memory.profile_path = str(tmp_path / "profile.json")

    summarizer = FakeLLMProvider(EXTRACTION_JSON)
    manager = MemoryManager(config, embedder=None, summary_provider=summarizer)
    assert manager.embedder is None
    assert manager.vector_store is None

    manager.add_message("user", "I study neuroscience")
    _wait_for(lambda: summarizer.calls >= 1)
    manager.flush()

    assert manager.summary == "Talked about the lab."
    assert manager.profile.facts.get("name") == "Alex"
    assert manager.profile.open_threads == ["fMRI pipeline bug"]
    # The extracted fact itself has nowhere to be stored without an embedder —
    # that's the accepted trade-off, not a bug.
    assert manager.long_term_count == 0


def test_set_summary_provider_activates_dormant_executor(tmp_path: Path) -> None:
    """Regression: hot-swapping in a summary provider after construction
    (as SessionStore.reload_providers does when a dummy LLM is replaced by a
    real one) must actually enable summarization, not silently no-op forever.
    """
    config = Config()
    config.memory.summary_trigger_messages = 1
    config.memory.persist_path = str(tmp_path / "memory.json")
    config.memory.embeddings_path = None
    # No summary_provider and no long-term memory at construction time, so
    # _summary_executor starts out None — mirrors a session started on the
    # dummy LLM fallback.
    manager = MemoryManager(config)
    assert manager._summary_executor is None

    manager.add_message("user", "Hello before key is configured")
    assert manager.summary is None or manager.summary_info.get("strategy") != "llm"

    summarizer = FakeLLMProvider("Concise summary")
    manager.set_summary_provider(summarizer)
    assert manager._summary_executor is not None

    manager.add_message("user", "Hello after key is configured")
    _wait_for(lambda: summarizer.calls >= 1)
    manager.flush()

    assert manager.summary == "Concise summary"
    assert manager.summary_info["strategy"] == "llm"
    manager.clear()


def test_set_embedder_activates_dormant_executor(tmp_path: Path) -> None:
    """Same lazy-executor gap, but for hot-swapping the embedder in with
    long-term memory already enabled."""
    config = Config()
    config.memory.persist_path = None
    config.memory.enable_long_term = True
    config.memory.embeddings_path = str(tmp_path / "vectors.json")
    config.memory.enable_fact_extraction = False
    config.memory.enable_episodes = False
    config.memory.profile_path = None
    manager = MemoryManager(config)
    assert manager._summary_executor is None

    manager.set_embedder(DummyEmbedder())
    assert manager._summary_executor is not None

    manager.add_message("user", "hello")
    manager.flush()
    assert manager.long_term_count == 1


def test_hashing_embedding_deterministic_and_lexical() -> None:
    from amadeus.memory.manager import HashingEmbedding

    emb = HashingEmbedding()

    # Deterministic across calls (required for persisted vectors to stay valid).
    assert emb.embed_text("I love coffee") == emb.embed_text("I love coffee")

    # Exact keyword overlap ranks above unrelated text.
    query = emb.embed_text("coffee")
    related = emb.embed_text("the user loves coffee in the morning")
    unrelated = emb.embed_text("quantum physics equations on the board")
    assert emb.similarity(query, related) > emb.similarity(query, unrelated)
    assert emb.similarity(query, related) > 0.0

    # Empty text yields a zero vector without error.
    assert all(v == 0.0 for v in emb.embed_text(""))


def test_early_extraction_triggers_on_empty_profile_facts(tmp_path: Path) -> None:
    """First-meeting fast path: with fact extraction enabled and an empty
    profile, the 5th message (2 user + 2 assistant already in, then one more
    user message) should trigger summarize() early — well before the normal
    debounced trigger window (summary_trigger_messages) fills up."""
    config = _extraction_config(tmp_path, summary_trigger_messages=50)
    manager = MemoryManager(config, embedder=VectorEmbedder())
    assert not manager.profile.facts

    calls = []
    manager.summarize = lambda: calls.append(1)  # type: ignore[assignment]

    manager.add_message("user", "hello")
    manager.add_message("assistant", "hi there")
    assert calls == []  # only 2 messages so far; short_term < 4

    manager.add_message("user", "how are you")
    manager.add_message("assistant", "fine")
    assert calls == []  # short_term == 4, but this message wasn't role=="user"

    manager.add_message("user", "what's my name")
    assert calls == [1], "5th message should trigger early summarize"


def test_early_extraction_does_not_fire_with_existing_profile_facts(
    tmp_path: Path,
) -> None:
    """Same message shape, but the profile already has facts — the fast
    path must not fire (there's nothing urgent left to learn early)."""
    config = _extraction_config(tmp_path, summary_trigger_messages=50)
    manager = MemoryManager(config, embedder=VectorEmbedder())
    manager.profile.facts["name"] = "Alex"

    calls = []
    manager.summarize = lambda: calls.append(1)  # type: ignore[assignment]

    manager.add_message("user", "hello")
    manager.add_message("assistant", "hi there")
    manager.add_message("user", "how are you")
    manager.add_message("assistant", "fine")
    manager.add_message("user", "what's my name")

    assert calls == []


def test_early_extraction_requires_extraction_enabled(tmp_path: Path) -> None:
    """Same empty-profile shape, but fact extraction is disabled — the fast
    path must not fire even though every other condition holds."""
    config = _extraction_config(
        tmp_path, summary_trigger_messages=50, enable_fact_extraction=False
    )
    manager = MemoryManager(config, embedder=VectorEmbedder())
    assert manager.profile is None or not manager.profile.facts

    calls = []
    manager.summarize = lambda: calls.append(1)  # type: ignore[assignment]

    manager.add_message("user", "hello")
    manager.add_message("assistant", "hi there")
    manager.add_message("user", "how are you")
    manager.add_message("assistant", "fine")
    manager.add_message("user", "what's my name")

    assert calls == []


def test_early_extraction_latch_caps_attempts_at_two(tmp_path: Path) -> None:
    """G1 regression: without a separate attempt latch, summarize() resets
    _messages_since_summary on every call and only profile.merge_updates
    (i.e. extraction actually yielding facts) disarms the `not
    self.profile.facts` guard — so a run of extraction calls that keep
    yielding zero facts would re-fire roughly every other user message for
    the entire conversation. _early_extract_attempts must cap it at 2 fires
    total, regardless of how many more qualifying user messages follow."""
    config = _extraction_config(tmp_path, summary_trigger_messages=50)
    manager = MemoryManager(config, embedder=VectorEmbedder())
    assert not manager.profile.facts

    calls = []
    manager.summarize = lambda: calls.append(1)  # type: ignore[assignment]

    # First qualifying window: 4 short_term messages + messages_since_summary
    # >= 2 fires the first attempt.
    manager.add_message("user", "hello")
    manager.add_message("assistant", "hi there")
    manager.add_message("user", "how are you")
    manager.add_message("assistant", "fine")
    manager.add_message("user", "what's my name")
    assert calls == [1]
    assert manager._early_extract_attempts == 1

    # profile.facts is STILL empty (summarize is mocked, never actually
    # extracts) — every subsequent pair of user messages would have
    # re-qualified under the old code once _messages_since_summary rebuilds
    # past the debounce floor.
    manager.add_message("assistant", "fine again")
    manager.add_message("user", "still no facts learned")
    assert calls == [1, 1], "second attempt should still fire (latch allows 2)"
    assert manager._early_extract_attempts == 2

    # From here on the latch must block every further attempt, no matter
    # how many more qualifying user messages arrive.
    for i in range(6):
        manager.add_message("assistant", f"reply {i}")
        manager.add_message("user", f"message {i}")
    assert calls == [1, 1], "latch must block all further early-extraction attempts"
    assert manager._early_extract_attempts == 2


def test_memory_overview_lists_recent_episodes(tmp_path: Path) -> None:
    summarizer = FakeLLMProvider("We debugged the pipeline; user was upbeat.")
    config = _extraction_config(tmp_path, enable_fact_extraction=False)
    config.memory.enable_episodes = True
    config.memory.session_gap_minutes = 30
    config.memory.summary_trigger_messages = 50
    manager = MemoryManager(
        config, embedder=VectorEmbedder(), summary_provider=summarizer
    )

    manager.add_message("user", "hello")
    manager.flush()
    manager.short_term[-1].timestamp = _iso_days_ago(1)
    manager.add_message("user", "back again")
    manager.flush()

    overview = manager.memory_overview()
    assert overview["counts"]["episodes"] == 1
    assert any("debugged the pipeline" in text for text in overview["episodes"])


def test_memory_overview_shows_moments(tmp_path: Path) -> None:
    """docs/companion-ethics.md promises a stored moment "shows up in the
    memory panel" — the one opt-in exception to the co-activity forgetting
    contract is only defensible if the user can actually see and delete
    what got stored. `memory_overview` filtered on type=="fact" exactly,
    so every moment ever stored was invisible there: a live doc-vs-code
    violation, not a cosmetic gap. Moments ride the fact view because they
    are durable/recallable/editable in every respect a user cares about.
    """
    embedder = VectorEmbedder({"hiking": (1.0, 0.0), "finale": (0.0, 1.0)})
    manager = MemoryManager(_extraction_config(tmp_path), embedder=embedder)
    manager.remember("User likes hiking")
    assert manager.store_moment("we watched the finale together")

    overview = manager.memory_overview()
    texts = [rec["text"] for rec in overview["fact_records"]]
    assert "we watched the finale together" in texts
    assert overview["counts"]["moments"] == 1
    assert overview["counts"]["facts"] == 1
    # Derived legacy view must stay consistent with the records it mirrors.
    assert overview["facts"] == texts


def test_memory_overview_legacy_lists_derive_from_records(tmp_path: Path) -> None:
    """K2: `facts`/`episodes` must be byte-identical to
    `[r["text"] for r in fact_records/episode_records]` — built once and
    derived, not computed by a second independent pass that could drift."""
    embedder = VectorEmbedder({"hiking": (1.0, 0.0), "tea": (0.0, 1.0)})
    manager = MemoryManager(_extraction_config(tmp_path), embedder=embedder)
    manager.remember("User likes hiking", pinned=True)
    manager.remember("User likes tea")

    overview = manager.memory_overview()
    assert overview["facts"] == [r["text"] for r in overview["fact_records"]]
    assert overview["episodes"] == [r["text"] for r in overview["episode_records"]]
    assert set(overview["facts"]) == {"User likes hiking", "User likes tea"}


def test_in_memory_vector_store_get_by_id(tmp_path: Path) -> None:
    """K1: VectorStore.get(record_id) is a targeted lookup, not a
    dump-then-scan — exercised directly against InMemoryVectorStore."""
    from amadeus.memory.vector_store import InMemoryVectorStore, VectorRecord

    store = InMemoryVectorStore(path=tmp_path / "vectors.json")
    record = VectorRecord(vector=[1.0, 0.0], text="hello", metadata={"type": "fact"})
    store.add(record)
    other = VectorRecord(vector=[0.0, 1.0], text="world", metadata={"type": "fact"})
    store.add(other)

    fetched = store.get(record.id)
    assert fetched is not None
    assert fetched.id == record.id
    assert fetched.text == "hello"

    assert store.get("nonexistent-id") is None
    assert store.get("") is None


def test_chroma_vector_store_get_by_id(tmp_path: Path) -> None:
    """K1: same contract as InMemoryVectorStore.get(), against the Chroma
    backend (collection.get(ids=[...])) — skipped when chromadb isn't
    installed in this environment."""
    pytest.importorskip("chromadb")
    from amadeus.memory.vector_store import ChromaVectorStore, VectorRecord

    store = ChromaVectorStore(path=tmp_path / "chroma")
    record = VectorRecord(vector=[1.0, 0.0], text="hello", metadata={"type": "fact"})
    store.add(record)

    fetched = store.get(record.id)
    assert fetched is not None
    assert fetched.id == record.id
    assert fetched.text == "hello"
    assert fetched.vector == [1.0, 0.0]

    assert store.get("nonexistent-id") is None
    assert store.get("") is None


def test_load_heals_orphaned_trailing_user_messages(tmp_path: Path) -> None:
    """A store ending on unanswered user message(s) is a failed/crashed turn:
    restoring the orphan pollutes the next turn's context and its fresh
    timestamp suppresses the greeting gap check (the v0.12.0 shakedown bug)."""
    config = Config()
    config.memory.persist_path = str(tmp_path / "memory.json")
    config.memory.embeddings_path = None

    manager = MemoryManager(config)
    manager.add_message("user", "Hello")
    manager.add_message("assistant", "Hi.")
    manager.add_message("user", "This turn's provider crashed")
    manager.add_message("user", "So did the retry")

    restored = MemoryManager(config)
    roles = [msg.role for msg in restored.short_term]
    assert roles == ["user", "assistant"]
    # With the orphan gone, a stale-enough window greets normally again
    # (the orphan carried the freshest timestamp in the store).
    assert [m.content for m in restored.short_term] == ["Hello", "Hi."]


def test_rollback_user_message_persists_the_pop(tmp_path: Path) -> None:
    from amadeus.engine import ChatEngine
    from amadeus.providers.llm.base import DummyProvider

    config = Config()
    config.memory.persist_path = str(tmp_path / "memory.json")
    config.memory.embeddings_path = None
    config.memory.profile_path = str(tmp_path / "profile.json")

    engine = ChatEngine(config, DummyProvider())
    engine.memory.add_message("user", "Doomed question")
    engine.rollback_user_message("Doomed question")
    assert [m for m in engine.memory.short_term if m.role == "user"] == []

    # The pop must reach disk, not just RAM (pre-fix, the store kept the
    # orphan until some unrelated later persist).
    restored = MemoryManager(config)
    assert list(restored.short_term) == []


# --- Per-character memory isolation -----------------------------------------


def _guest_character(config: Config, full_name: str = "Rem") -> None:
    """Make the config describe an active custom character pack."""
    config.character.name = full_name.split()[0]
    config.character.full_name = full_name
    config.character.system_prompt = f"You are {full_name}."


def test_character_scoped_path_derivation(tmp_path: Path) -> None:
    from amadeus.paths import character_scoped_path

    base = str(tmp_path / "memory_store.json")
    scoped = character_scoped_path(base, "rem")
    assert scoped == str(tmp_path / "characters" / "rem" / "memory_store.json")
    # Idempotent: re-scoping an already scoped path is a no-op (settings
    # saves re-derive paths from the runtime config).
    assert character_scoped_path(scoped, "rem") == scoped
    # A different character nests under its own id.
    other = character_scoped_path(base, "megumin")
    assert other == str(tmp_path / "characters" / "megumin" / "memory_store.json")
    # Builtin (None id) and empty values pass through untouched.
    assert character_scoped_path(base, None) == base
    assert character_scoped_path(None, "rem") is None


def test_memory_character_id_builtin_guest_and_flag() -> None:
    config = Config()
    # Builtin Kurisu: no custom prompt resolves -> legacy shared paths.
    assert config.memory_character_id() is None

    _guest_character(config, "Rin Tohsaka")
    assert config.memory_character_id() == "rin-tohsaka"

    # The kill switch restores the shared-store behavior.
    config.memory.per_character_memory = False
    assert config.memory_character_id() is None


def test_memory_character_id_non_latin_name_still_isolates() -> None:
    config = Config()
    _guest_character(config, "レム")
    char_id = config.memory_character_id()
    assert char_id  # deterministic digest fallback, never empty/None
    assert config.memory_character_id() == char_id


def test_guest_manager_isolated_from_kurisu_store(tmp_path: Path) -> None:
    """A guest character reads/writes characters/<id>/ files and starts
    empty; Kurisu's legacy-path files stay byte-identical."""
    from amadeus.memory.vector_store import VectorRecord

    config = _extraction_config(tmp_path)
    config.memory.persist_path = str(tmp_path / "memory.json")
    manager = MemoryManager(config, embedder=VectorEmbedder())
    manager.add_message("user", "My name is Alex")
    manager.add_message("assistant", "Noted, Alex.")
    manager.profile.facts["name"] = "Alex"
    manager._save_profile()
    manager.vector_store.add(
        VectorRecord(
            vector=[1.0, 0.0], text="User loves cats", metadata={"type": "fact"}
        )
    )
    manager.vector_store.persist()
    manager._persist()
    kurisu_store = (tmp_path / "memory.json").read_bytes()

    guest_config = _extraction_config(tmp_path)
    guest_config.memory.persist_path = str(tmp_path / "memory.json")
    _guest_character(guest_config)
    guest = MemoryManager(guest_config, embedder=VectorEmbedder())
    assert guest.persist_path == tmp_path / "characters" / "rem" / "memory.json"
    assert list(guest.short_term) == []
    assert guest.recall("cats") == []
    assert guest.profile.facts == {}
    guest.add_message("user", "I'm someone new")
    guest.add_message("assistant", "Nice to meet you.")
    assert (tmp_path / "characters" / "rem" / "memory.json").exists()
    # Kurisu's store never moved and never changed.
    assert (tmp_path / "memory.json").read_bytes() == kurisu_store

    # Switching back: Kurisu reloads her own memory, not the guest's.
    restored = MemoryManager(config, embedder=VectorEmbedder())
    assert "My name is Alex" in [m.content for m in restored.short_term]
    assert restored.recall("cats") == ["User loves cats"]
    assert restored.profile.facts.get("name") == "Alex"


def test_per_character_memory_disabled_shares_store(tmp_path: Path) -> None:
    config = _extraction_config(tmp_path)
    config.memory.persist_path = str(tmp_path / "memory.json")
    manager = MemoryManager(config)
    manager.add_message("user", "Hello")
    manager.add_message("assistant", "Hi")

    guest_config = _extraction_config(tmp_path)
    guest_config.memory.persist_path = str(tmp_path / "memory.json")
    _guest_character(guest_config)
    guest_config.memory.per_character_memory = False
    guest = MemoryManager(guest_config)
    assert guest.persist_path == tmp_path / "memory.json"
    assert "Hello" in [m.content for m in guest.short_term]


def test_recall_skips_trivial_queries(tmp_path: Path) -> None:
    """Sub-threshold queries ("hi", "ok") skip the embed round-trip entirely."""
    calls: List[str] = []

    class CountingEmbedder(EmbeddingProvider):
        def embed_text(self, text: str) -> List[float]:
            calls.append(text)
            return [float(len(text))]

        def similarity(self, a: List[float], b: List[float]) -> float:
            return 1.0

    config = _long_term_config(tmp_path)
    config.memory.recall_min_query_chars = 10
    manager = MemoryManager(config, embedder=CountingEmbedder())
    calls.clear()
    assert manager.recall("hi") == []
    assert calls == []
    manager.recall("a much longer meaningful query")
    assert calls  # long queries still embed


def test_recall_embed_timeout_degrades_to_empty(tmp_path: Path) -> None:
    """A network embedder that blows the deadline yields empty recall, not a
    stalled or crashed turn; local embedders ignore the timeout."""
    import time as _time

    class SlowNetworkEmbedder(EmbeddingProvider):
        network = True

        def embed_text(self, text: str) -> List[float]:
            _time.sleep(0.4)
            return [float(len(text))]

        def similarity(self, a: List[float], b: List[float]) -> float:
            return 1.0

    config = _long_term_config(tmp_path)
    config.memory.recall_embed_timeout_seconds = 0.05
    manager = MemoryManager(config, embedder=SlowNetworkEmbedder())
    assert manager.recall("a sufficiently long recall query") == []

    class SlowLocalEmbedder(SlowNetworkEmbedder):
        network = False

    config.memory.recall_embed_timeout_seconds = 0.05
    manager = MemoryManager(config, embedder=SlowLocalEmbedder())
    # Local embedder: timeout not applied, the (slow) embed completes.
    manager.recall("a sufficiently long recall query")


# --- M1: contradiction-aware fact supersede ---------------------------------


def test_store_fact_replaces_hint_supersedes_at_moderate_similarity(
    tmp_path: Path,
) -> None:
    """A `replaces` hint lets a paraphrased contradiction (moderate cosine,
    well below the near-duplicate threshold) supersede the earlier fact."""
    embedder = VectorEmbedder(
        {
            "loves Nolan": (1.0, 0.0),
            "bored of Nolan": (0.7, 0.7135),  # dot with (1,0) ~= 0.7
        },
        default=(0.0, 1.0),
    )
    manager = MemoryManager(_extraction_config(tmp_path), embedder=embedder)
    manager._store_fact("User loves Nolan movies", 0.6)
    manager._store_fact(
        "User is bored of Nolan movies now",
        0.6,
        replaces="User loves Nolan movies",
    )

    records = list(manager.vector_store.dump())
    assert len(records) == 1
    assert records[0].text == "User is bored of Nolan movies now"
    assert records[0].metadata.get("supersedes")


def test_store_fact_without_hint_does_not_supersede_at_moderate_similarity(
    tmp_path: Path,
) -> None:
    """Without a `replaces` hint, a fact at only moderate similarity (below
    the near-duplicate threshold) must NOT be superseded — the hint path is
    additive, not a threshold loosening for the default path."""
    embedder = VectorEmbedder(
        {
            "loves Nolan": (1.0, 0.0),
            "bored of Nolan": (0.7, 0.7135),
        },
        default=(0.0, 1.0),
    )
    manager = MemoryManager(_extraction_config(tmp_path), embedder=embedder)
    manager._store_fact("User loves Nolan movies", 0.6)
    manager._store_fact("User is bored of Nolan movies now", 0.6)

    records = list(manager.vector_store.dump())
    assert len(records) == 2
    texts = {r.text for r in records}
    assert "User loves Nolan movies" in texts
    assert "User is bored of Nolan movies now" in texts


def test_store_fact_replaces_hint_never_supersedes_pinned(tmp_path: Path) -> None:
    embedder = VectorEmbedder(
        {
            "loves Nolan": (1.0, 0.0),
            "bored of Nolan": (0.7, 0.7135),
        },
        default=(0.0, 1.0),
    )
    manager = MemoryManager(_extraction_config(tmp_path), embedder=embedder)
    manager.remember("User loves Nolan movies", pinned=True)
    manager._store_fact(
        "User is bored of Nolan movies now",
        0.6,
        replaces="User loves Nolan movies",
    )

    records = list(manager.vector_store.dump())
    # The pinned original survives untouched; the new fact is stored too.
    assert len(records) == 2
    pinned = [r for r in records if r.metadata.get("pinned")]
    assert len(pinned) == 1
    assert pinned[0].text == "User loves Nolan movies"


def test_store_fact_replaces_hint_below_similarity_floor_keeps_both(
    tmp_path: Path,
) -> None:
    """A `replaces` hint that doesn't actually match anything in the store
    (cosine similarity below the 0.6 floor) must not delete an unrelated
    fact. Cosine similarity normalizes vectors, so the mapping is chosen so
    both the primary near-dup check (new-fact-vs-existing) and the hint
    check (replaces-text-vs-existing) land well under their thresholds."""
    embedder = VectorEmbedder(
        {
            "unrelated": (1.0, 0.0),
            "totally different topic": (0.0, 1.0),
            "something never actually stored": (0.2, 0.98),
        },
        default=(0.5, 0.5),
    )
    manager = MemoryManager(_extraction_config(tmp_path), embedder=embedder)
    manager._store_fact("User likes unrelated hobby", 0.5)
    manager._store_fact(
        "User started a totally different topic",
        0.5,
        replaces="something never actually stored",
    )
    records = list(manager.vector_store.dump())
    assert len(records) == 2


def test_extraction_parses_replaces_field(tmp_path: Path) -> None:
    summarizer = FakeLLMProvider(
        '{"summary": "Tastes update.", '
        '"new_facts": [{"text": "User is bored of Nolan movies now", '
        '"salience": 0.6, "replaces": "User loves Nolan movies"}], '
        '"profile_updates": {}, "open_threads": [], "tone": 0.0}'
    )
    manager = MemoryManager(
        _extraction_config(tmp_path),
        embedder=VectorEmbedder(),
        summary_provider=summarizer,
    )
    manager._store_fact("User loves Nolan movies", 0.6)
    manager.add_message("user", "I'm kind of over Nolan movies now")
    _wait_for(lambda: manager.summary_info.get("strategy") == "llm")
    manager.flush()

    records = list(manager.vector_store.dump())
    assert len(records) == 1
    assert records[0].text == "User is bored of Nolan movies now"
    assert records[0].metadata.get("supersedes")


def test_extraction_absent_replaces_field_parses_as_none(tmp_path: Path) -> None:
    # EXTRACTION_JSON's new_facts item has no "replaces" key at all.
    summarizer = FakeLLMProvider(EXTRACTION_JSON)
    manager = MemoryManager(
        _extraction_config(tmp_path),
        embedder=VectorEmbedder(),
        summary_provider=summarizer,
    )
    manager.add_message("user", "I study neuroscience")
    _wait_for(lambda: manager.summary_info.get("strategy") == "llm")
    manager.flush()

    records = list(manager.vector_store.dump())
    assert len(records) == 1
    assert "supersedes" not in records[0].metadata


# --- v4: felt-tone extraction + profile persistence --------------------------


FELT_EXTRACTION_JSON = (
    '{"summary": "Talked about the paper.", '
    '"new_facts": [], "profile_updates": {}, "open_threads": [], '
    '"tone": 0.6, '
    '"felt": {"label": "Excited", "subject": "the paper deadline"}}'
)


def test_extraction_persists_felt_onto_profile(tmp_path: Path) -> None:
    summarizer = FakeLLMProvider(FELT_EXTRACTION_JSON)
    manager = MemoryManager(
        _extraction_config(tmp_path),
        embedder=VectorEmbedder(),
        summary_provider=summarizer,
    )
    manager.add_message("user", "The deadline is close and I'm pumped")
    _wait_for(lambda: manager.summary_info.get("strategy") == "llm")
    manager.flush()

    felt = manager.profile.last_felt
    assert felt is not None
    assert felt["label"] == "excited"  # lowercased on the way in
    assert felt["subject"] == "the paper deadline"
    assert felt["date"]  # dated by record_felt


def test_extraction_without_felt_leaves_slot_untouched(tmp_path: Path) -> None:
    # EXTRACTION_JSON has no "felt" key: a prior felt-tone must not be wiped by
    # a later session that simply didn't surface one.
    summarizer = FakeLLMProvider(EXTRACTION_JSON)
    manager = MemoryManager(
        _extraction_config(tmp_path),
        embedder=VectorEmbedder(),
        summary_provider=summarizer,
    )
    manager.profile.last_felt = {"label": "reflective", "date": "2026-01-01"}
    manager.add_message("user", "I study neuroscience")
    _wait_for(lambda: manager.summary_info.get("strategy") == "llm")
    manager.flush()
    assert manager.profile.last_felt == {"label": "reflective", "date": "2026-01-01"}


# --- M2: salience in recall ranking -----------------------------------------


def test_recall_ranking_prefers_higher_salience_at_equal_score_and_recency(
    tmp_path: Path,
) -> None:
    from amadeus.memory.vector_store import VectorRecord

    embedder = VectorEmbedder()
    config = _long_term_config(tmp_path)
    manager = MemoryManager(config, embedder=embedder)
    manager.vector_store.add(
        VectorRecord(
            vector=[1.0, 0.0],
            text="low salience fact",
            metadata={
                "type": "fact",
                "timestamp": _iso_days_ago(0),
                "strength": 1.0,
                "salience": 0.1,
            },
        )
    )
    manager.vector_store.add(
        VectorRecord(
            vector=[1.0, 0.0],
            text="high salience fact",
            metadata={
                "type": "fact",
                "timestamp": _iso_days_ago(0),
                "strength": 1.0,
                "salience": 0.9,
            },
        )
    )

    matches = manager.recall_detailed("anything", limit=2)
    assert [m.text for m in matches] == ["high salience fact", "low salience fact"]


def test_recall_ranking_missing_salience_defaults_neutral(tmp_path: Path) -> None:
    """A record with no salience metadata (legacy) ranks like salience=0.5,
    not like the floor — it must not be pushed below an explicit low-salience
    record."""
    from amadeus.memory.vector_store import VectorRecord

    embedder = VectorEmbedder()
    config = _long_term_config(tmp_path)
    manager = MemoryManager(config, embedder=embedder)
    manager.vector_store.add(
        VectorRecord(
            vector=[1.0, 0.0],
            text="legacy no-salience fact",
            metadata={"type": "fact", "timestamp": _iso_days_ago(0), "strength": 1.0},
        )
    )
    manager.vector_store.add(
        VectorRecord(
            vector=[1.0, 0.0],
            text="low salience fact",
            metadata={
                "type": "fact",
                "timestamp": _iso_days_ago(0),
                "strength": 1.0,
                "salience": 0.1,
            },
        )
    )
    matches = manager.recall_detailed("anything", limit=2)
    assert [m.text for m in matches] == [
        "legacy no-salience fact",
        "low salience fact",
    ]


# --- M3: forgetting on by default -------------------------------------------


def test_forgetting_enabled_by_default() -> None:
    config = Config()
    assert config.memory.enable_forgetting is True


# --- M4: retrospective-intent gate -------------------------------------------


def test_is_retrospective_positive_examples() -> None:
    from amadeus.memory.retrospective import is_retrospective

    positives = [
        "remember when we talked about the fMRI pipeline?",
        "Do you remember my sister's name?",
        "what did we talk about last week?",
        "What did I tell you about my thesis?",
        "last time we spoke you mentioned a bug",
        "we discussed this before, right?",
        "we talked about my new job",
        "you mentioned something about it before",
        "you mentioned that earlier",
        "a while back we chatted about cats",
    ]
    for text in positives:
        assert is_retrospective(text), text


def test_is_retrospective_negative_examples() -> None:
    from amadeus.memory.retrospective import is_retrospective

    negatives = [
        "hi there",
        "what's the weather like today?",
        "did it work out?",
        "I remember to lock the door every night",
        "can you remember my birthday going forward?",
        "",
        "   ",
    ]
    for text in negatives:
        assert not is_retrospective(text), text


def test_recall_detailed_include_episodes_admits_episodes_only_when_set(
    tmp_path: Path,
) -> None:
    from amadeus.memory.vector_store import VectorRecord

    embedder = VectorEmbedder()
    config = _long_term_config(tmp_path)
    config.memory.recall_facts_only = True
    manager = MemoryManager(config, embedder=embedder)
    manager.vector_store.add(
        VectorRecord(
            vector=[1.0, 0.0],
            text="an episode about last week",
            metadata={
                "type": "episode",
                "timestamp": _iso_days_ago(0),
                "strength": 1.0,
            },
        )
    )
    manager.vector_store.add(
        VectorRecord(
            vector=[1.0, 0.0],
            text="a reflection record",
            metadata={
                "type": "reflection",
                "timestamp": _iso_days_ago(0),
                "strength": 1.0,
            },
        )
    )

    default_texts = [m.text for m in manager.recall_detailed("anything", limit=5)]
    assert default_texts == []

    with_episodes_texts = [
        m.text
        for m in manager.recall_detailed("anything", limit=5, include_episodes=True)
    ]
    # Episode admitted; reflection still excluded even with the flag set.
    assert with_episodes_texts == ["an episode about last week"]


def test_assemble_turn_context_admits_episode_on_retrospective_query(
    tmp_path: Path,
) -> None:
    from amadeus.memory.context import assemble_turn_context
    from amadeus.memory.vector_store import VectorRecord

    embedder = VectorEmbedder()
    config = _long_term_config(tmp_path)
    config.memory.recall_facts_only = True
    manager = MemoryManager(config, embedder=embedder)
    manager.vector_store.add(
        VectorRecord(
            vector=[1.0, 0.0],
            text="an episode about last week",
            metadata={
                "type": "episode",
                "timestamp": _iso_days_ago(0),
                "strength": 1.0,
            },
        )
    )

    _, recall_items = assemble_turn_context(
        manager, "system prompt", "do you remember what we talked about?"
    )
    assert recall_items == ["an episode about last week"]

    manager2 = MemoryManager(config, embedder=embedder)
    manager2.vector_store.add(
        VectorRecord(
            vector=[1.0, 0.0],
            text="an episode about last week",
            metadata={
                "type": "episode",
                "timestamp": _iso_days_ago(0),
                "strength": 1.0,
            },
        )
    )
    _, recall_items_plain = assemble_turn_context(
        manager2, "system prompt", "how's your day going?"
    )
    assert recall_items_plain == []


# --- M5: query augmentation for short/pronoun follow-ups --------------------


def test_needs_query_augmentation_short_and_pronoun_heavy() -> None:
    from amadeus.memory.context import _needs_query_augmentation

    assert _needs_query_augmentation("did it work out?")
    assert _needs_query_augmentation("what about it")
    # Contractions of function words are not content words.
    assert _needs_query_augmentation("didn't it work?")
    assert _needs_query_augmentation("what's it about?")
    # Empty text has no previous-turn context worth prepending.
    assert not _needs_query_augmentation("")


def test_needs_query_augmentation_normal_message_not_augmented() -> None:
    from amadeus.memory.context import _needs_query_augmentation

    assert not _needs_query_augmentation(
        "I've been working on the fMRI segmentation pipeline all week"
    )


def test_needs_query_augmentation_is_content_gated_not_length_gated() -> None:
    """A short but self-contained question must NOT be augmented: the old
    <40-chars shortcut concatenated the unrelated previous turn onto exactly
    the focused questions recall serves best, shifting the query embedding
    (gap-audit 2026-08-03 open item)."""
    from amadeus.memory.context import _needs_query_augmentation

    assert not _needs_query_augmentation("what's a tensor?")
    assert not _needs_query_augmentation("how did the interview go?")


def test_assemble_turn_context_augments_short_followup_query(tmp_path: Path) -> None:
    from amadeus.memory.context import assemble_turn_context
    from amadeus.memory.vector_store import VectorRecord

    # Embedder only resolves the target fact when the query contains both
    # the previous turn's topic word AND the recall is actually attempted;
    # this simulates "did it work out?" being noise on its own but
    # meaningful once the prior turn's topic is prefixed.
    embedder = VectorEmbedder(
        {"job interview": (1.0, 0.0)},
        default=(0.0, 1.0),
    )
    config = _long_term_config(tmp_path)
    manager = MemoryManager(config, embedder=embedder)
    manager.vector_store.add(
        VectorRecord(
            vector=[1.0, 0.0],
            text="User has a job interview coming up",
            metadata={"type": "fact", "timestamp": _iso_days_ago(0), "strength": 1.0},
        )
    )
    # Seed the previous user turn into short-term context.
    manager.add_message("user", "I have a job interview next week")
    manager.add_message("assistant", "Good luck!")

    _, recall_items = assemble_turn_context(
        manager, "system prompt", "did it work out?"
    )
    assert recall_items == ["User has a job interview coming up"]


def test_assemble_turn_context_normal_query_not_augmented(tmp_path: Path) -> None:
    """A normal, self-contained message must recall using its own text —
    verified by asserting recall_items are unaffected by an unrelated
    previous turn that would otherwise pollute a naive concatenation."""
    from amadeus.memory.context import assemble_turn_context
    from amadeus.memory.vector_store import VectorRecord

    # "weather" listed first: embed_text returns the first key found, so a
    # query wrongly augmented with the weather turn embeds to the weather
    # vector and the fact recall below fails — that ordering is what makes
    # this test actually detect pollution.
    embedder = VectorEmbedder(
        {"weather": (0.0, 1.0), "neuroscience": (1.0, 0.0)},
        default=(0.5, 0.5),
    )
    # facts_only: assemble_turn_context commits the user message, and the
    # async raw-message store can land it in the vector store before recall
    # runs (seen on fast CI runners). The raw echo of the CURRENT message is
    # exempt from the recall_recent_dedupe fixation guard by design, so
    # filter untyped records instead — this test is about query
    # augmentation, not the legacy raw-record ranking path.
    config = _long_term_config(tmp_path, recall_facts_only=True)
    manager = MemoryManager(config, embedder=embedder)
    manager.vector_store.add(
        VectorRecord(
            vector=[1.0, 0.0],
            text="User studies neuroscience",
            metadata={"type": "fact", "timestamp": _iso_days_ago(0), "strength": 1.0},
        )
    )
    manager.add_message("user", "the weather has been grim lately")
    manager.add_message("assistant", "Sounds gray out there.")

    _, recall_items = assemble_turn_context(
        manager,
        "system prompt",
        "tell me about current neuroscience research directions",
    )
    assert recall_items == ["User studies neuroscience"]


def test_assemble_turn_context_short_selfcontained_query_not_augmented(
    tmp_path: Path,
) -> None:
    """The gap-audit pollution case: a sub-40-char but self-contained
    question ("neuroscience news?") must recall on its own text. Under the
    old length gate it was augmented with the unrelated weather turn, which
    (with "weather" listed first in the embedder mapping, same trick as the
    test above) embeds to the weather vector and loses the fact."""
    from amadeus.memory.context import assemble_turn_context
    from amadeus.memory.vector_store import VectorRecord

    embedder = VectorEmbedder(
        {"weather": (0.0, 1.0), "neuroscience": (1.0, 0.0)},
        default=(0.5, 0.5),
    )
    config = _long_term_config(tmp_path, recall_facts_only=True)
    manager = MemoryManager(config, embedder=embedder)
    manager.vector_store.add(
        VectorRecord(
            vector=[1.0, 0.0],
            text="User studies neuroscience",
            metadata={"type": "fact", "timestamp": _iso_days_ago(0), "strength": 1.0},
        )
    )
    manager.add_message("user", "the weather has been grim lately")
    manager.add_message("assistant", "Sounds gray out there.")

    _, recall_items = assemble_turn_context(
        manager, "system prompt", "any neuroscience news?"
    )
    assert recall_items == ["User studies neuroscience"]


def test_summarize_queues_rerun_while_job_in_flight(tmp_path: Path) -> None:
    """A summarize() landing while a job runs must queue a re-run, not drop —
    dropping meant a session-final extraction pass could silently never
    happen and the affected messages were never summarized."""
    from concurrent.futures import Future

    config = Config()
    config.memory.summary_trigger_messages = 50  # keep auto-triggers quiet
    config.memory.persist_path = str(tmp_path / "memory.json")
    config.memory.embeddings_path = None
    summarizer = FakeLLMProvider("Concise summary")
    manager = MemoryManager(config, summary_provider=summarizer)
    manager.add_message("user", "hello")
    manager.add_message("assistant", "hi")

    pending: Future = Future()  # never resolves: simulates an in-flight job
    manager._summary_future = pending
    before = manager._messages_since_summary
    manager.summarize()
    assert manager._resummarize_queued is True
    assert manager._messages_since_summary == before  # counter not consumed

    # When the in-flight job lands, the queued request re-submits and the
    # summary reflects the messages that arrived during the first job.
    manager._summary_future = None
    manager._process_resummarize_queue()
    manager.flush()
    assert manager.summary == "Concise summary"
    assert manager._resummarize_queued is False


def test_corrupt_vector_store_is_quarantined(tmp_path: Path) -> None:
    from amadeus import telemetry
    from amadeus.memory.vector_store import InMemoryVectorStore

    path = tmp_path / "embeddings.json"
    path.write_text("[ not valid json", encoding="utf-8")
    before = telemetry.snapshot().get("memory.vector_store_load_corrupt", 0)
    store = InMemoryVectorStore(path=path)
    assert store.is_empty()
    assert (tmp_path / "embeddings.json.bak-corrupt").exists()
    assert telemetry.snapshot().get("memory.vector_store_load_corrupt", 0) == before + 1


# --- Calibrated recall ------------------------------------------------------


def test_recall_hedges_a_stale_volatile_fact_in_context(tmp_path: Path) -> None:
    """An old job fact renders with a hedge clause; a fresh one does not.

    The end-to-end shape of calibrated recall: the confidence is computed in
    recall_detailed and rendered by _recall_line, so this exercises both
    seams at once.

    120 days, not 400: with `enable_forgetting` on (the default), strength
    decays past `strength_floor` at ~130 days and the fact is dropped from
    the candidate pool entirely — hedging only ever applies to facts young
    enough to survive forgetting, which is the real operating range of the
    bottom bands.
    """
    from amadeus.memory.context import assemble_turn_context
    from amadeus.memory.vector_store import VectorRecord

    config = _long_term_config(tmp_path)
    manager = MemoryManager(config, embedder=VectorEmbedder())
    manager.vector_store.add(
        VectorRecord(
            vector=[1.0, 0.0],
            text="user started a new job at a startup",
            metadata={"type": "fact", "timestamp": _iso_days_ago(120)},
        )
    )
    manager.vector_store.add(
        VectorRecord(
            vector=[1.0, 0.0],
            text="user drinks dark roast coffee",
            metadata={"type": "fact", "timestamp": _iso_days_ago(2)},
        )
    )

    context, _ = assemble_turn_context(manager, "system", "how is work going?")
    body = context[-1].content
    assert "user started a new job at a startup" in body
    stale_line = next(
        line for line in body.splitlines() if "new job at a startup" in line
    )
    assert "sure of this" in stale_line or "certain of this" in stale_line
    fresh_line = next(line for line in body.splitlines() if "dark roast coffee" in line)
    assert "sure of this" not in fresh_line
    assert "certain of this" not in fresh_line
    assert "misremembering" not in fresh_line


def test_recall_does_not_hedge_a_pinned_fact(tmp_path: Path) -> None:
    """A pinned fact is a user-asserted one — she states it plainly however
    old it is. Pinned records skip the confidence computation entirely."""
    from amadeus.memory.context import assemble_turn_context
    from amadeus.memory.vector_store import VectorRecord

    config = _long_term_config(tmp_path)
    manager = MemoryManager(config, embedder=VectorEmbedder())
    manager.vector_store.add(
        VectorRecord(
            vector=[1.0, 0.0],
            text="user's deadline for the project is in March",
            metadata={
                "type": "fact",
                "pinned": True,
                "timestamp": _iso_days_ago(500),
            },
        )
    )

    context, _ = assemble_turn_context(manager, "system", "what about the deadline?")
    line = next(line for line in context[-1].content.splitlines() if "deadline" in line)
    assert "sure of this" not in line
    assert "certain of this" not in line


def test_recall_confidence_flag_off_restores_flat_rendering(tmp_path: Path) -> None:
    """`memory.recall_confidence: false` is the escape hatch — no clause is
    computed or rendered, whatever the fact's age."""
    from amadeus.memory.context import assemble_turn_context
    from amadeus.memory.vector_store import VectorRecord

    config = _long_term_config(tmp_path)
    config.memory.recall_confidence = False
    manager = MemoryManager(config, embedder=VectorEmbedder())
    manager.vector_store.add(
        VectorRecord(
            vector=[1.0, 0.0],
            text="user started a new job at a startup",
            metadata={"type": "fact", "timestamp": _iso_days_ago(120)},
        )
    )

    context, _ = assemble_turn_context(manager, "system", "how is work going?")
    body = context[-1].content
    assert "new job at a startup" in body
    assert "sure of this" not in body
    assert "certain of this" not in body


def test_recall_confidence_is_not_written_back_to_the_store(tmp_path: Path) -> None:
    """Transient by contract: recall may attach confidence to the returned
    match, but the persisted metadata must never gain the field."""
    from amadeus.memory.vector_store import VectorRecord

    config = _long_term_config(tmp_path)
    manager = MemoryManager(config, embedder=VectorEmbedder())
    manager.vector_store.add(
        VectorRecord(
            vector=[1.0, 0.0],
            text="user started a new job at a startup",
            metadata={"type": "fact", "timestamp": _iso_days_ago(120)},
        )
    )

    matches = manager.recall_detailed("how is work going?", limit=3)
    assert matches and matches[0].confidence is not None
    for record in manager.vector_store.dump():
        assert "confidence" not in (record.metadata or {})
