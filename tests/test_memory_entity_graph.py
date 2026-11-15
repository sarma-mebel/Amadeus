"""Entity-graph memory layer (memory.graph_enabled, default off).

Note: NOT test_memory_graph.py — that file covers the pre-existing
"synapse map" visualization (MemoryManager.memory_graph(), nodes+edges over
the vector store). This file covers the separate typed entity/relation
layer in amadeus/memory/graph.py (person/project/place/date/preference,
supersede provenance, recall integration).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from amadeus import telemetry
from amadeus.config import Config
from amadeus.memory import graph as graph_module
from amadeus.memory.graph import EntityGraph, extract_into_graph, parse_fact
from amadeus.memory.manager import MemoryManager


class DeterministicEmbedder:
    def embed_text(self, text: str):
        import hashlib

        digest = hashlib.sha256(text.encode("utf-8")).digest()
        return [b / 255.0 for b in digest[:8]]

    def similarity(self, a, b) -> float:
        return sum(x * y for x, y in zip(a, b))


def _config(
    tmp_path: Path, *, graph_enabled: bool = True, backend: str = "json"
) -> Config:
    config = Config()
    config.memory.graph_enabled = graph_enabled
    config.memory.storage_backend = backend
    config.memory.enable_long_term = True
    config.memory.enable_fact_extraction = True
    config.memory.persist_path = str(tmp_path / "memory_store.json")
    config.memory.embeddings_path = str(tmp_path / "memory_embeddings.json")
    config.memory.profile_path = str(tmp_path / "memory_profile.json")
    return config


# --- parse_fact: well-formed + malformed -----------------------------------


def test_parse_fact_extracts_person_and_project() -> None:
    entities, relations = parse_fact(
        "User works with Alice on the CMB segmentation project", "2026-01-01T00:00:00"
    )
    keys = {e.key for e in entities}
    assert "alice" in keys
    kinds = {e.key: e.kind for e in entities}
    assert kinds["alice"] == "person"
    assert relations  # two entities mentioned together -> relation edges


def test_parse_fact_extracts_date() -> None:
    entities, _ = parse_fact("User's birthday is March 3", "2026-01-01T00:00:00")
    dates = [e for e in entities if e.kind == "date"]
    assert dates


def test_parse_fact_extracts_preference() -> None:
    entities, _ = parse_fact("User loves hiking on weekends", "2026-01-01T00:00:00")
    prefs = [e for e in entities if e.kind == "preference"]
    assert prefs


def test_parse_fact_malformed_input_never_raises() -> None:
    # Garbage / empty / non-string-ish text must degrade to empty, not crash.
    assert parse_fact("", "2026-01-01T00:00:00") == ([], [])
    assert parse_fact("   ", "2026-01-01T00:00:00") == ([], [])
    assert parse_fact("lowercase text no signal", "2026-01-01T00:00:00")[0] == []


def test_extract_into_graph_counts_malformed_facts(tmp_path: Path) -> None:
    before = telemetry.snapshot().get("memory.graph_extract_failed", 0)
    graph = EntityGraph()
    extract_into_graph(
        graph, [("no capitalized names or dates or keywords here", 0.5, None)]
    )
    after = telemetry.snapshot().get("memory.graph_extract_failed", 0)
    assert after == before + 1


# --- supersede chain ---------------------------------------------------


def test_supersede_chain_keeps_provenance() -> None:
    graph = EntityGraph()
    extract_into_graph(
        graph, [("User works with Alice on the CMB segmentation project", 0.8, None)]
    )
    extract_into_graph(
        graph,
        [
            (
                "User is now bored of working with Alice on the CMB segmentation project",
                0.7,
                "User works with Alice on the CMB segmentation project",
            )
        ],
    )
    alice = graph.entities["alice"]
    assert "bored" in alice.facts[-1]
    assert (
        alice.previous_fact == "User works with Alice on the CMB segmentation project"
    )


def test_supersede_only_touches_entities_in_new_fact() -> None:
    """An unrelated entity sharing wording with the replaces hint must not
    be superseded by a fact that never mentions it."""
    graph = EntityGraph()
    extract_into_graph(graph, [("User works with Alice on Project Nova", 0.8, None)])
    extract_into_graph(graph, [("User loves hiking", 0.6, None)])
    extract_into_graph(
        graph,
        [("User now prefers running", 0.6, "User works with Alice on Project Nova")],
    )
    # "Alice"/"Project Nova" entities must be untouched by a replace hint
    # that only textually overlaps but whose NEW fact never mentions them.
    assert graph.entities["alice"].previous_fact is None


# --- recall merge + dedupe ----------------------------------------------


def test_manager_recall_from_graph_merges_and_disabled_is_inert(tmp_path: Path) -> None:
    config = _config(tmp_path, graph_enabled=True)
    manager = MemoryManager(config, embedder=DeterministicEmbedder())
    manager._update_graph(
        [("User works with Alice on the CMB segmentation project", 0.8, None)]
    )
    lines = manager.recall_from_graph("How is Alice doing?")
    assert lines
    assert any("Alice" in line for line in lines)

    # No match -> empty, not an error.
    assert manager.recall_from_graph("completely unrelated message") == []


def test_recall_from_graph_disabled_flag_is_inert(tmp_path: Path) -> None:
    config = _config(tmp_path, graph_enabled=False)
    manager = MemoryManager(config, embedder=DeterministicEmbedder())
    manager._update_graph([("User works with Alice on Project Nova", 0.8, None)])
    # Even though facts were passed to _update_graph, it must no-op when the
    # flag is off — including not writing the sidecar file.
    assert manager.recall_from_graph("Alice") == []
    assert not (tmp_path / "memory_store_graph.json").exists()


def test_assemble_turn_context_merges_graph_lines_ahead_of_vector_recall(
    tmp_path: Path,
) -> None:
    from amadeus.memory.context import assemble_turn_context

    config = _config(tmp_path, graph_enabled=True)
    manager = MemoryManager(config, embedder=DeterministicEmbedder())
    manager._update_graph(
        [("User works with Alice on the CMB segmentation project", 0.8, None)]
    )
    context, recall_items = assemble_turn_context(
        manager, "system prompt", "How is Alice doing on the project?"
    )
    combined = "\n".join(msg.content for msg in context)
    assert "Alice" in combined


def test_assemble_turn_context_dedupes_graph_against_vector_recall(
    tmp_path: Path,
) -> None:
    """A graph line that's near-identical to something vector recall already
    surfaced must not be double-injected (fixation-guard convention)."""
    from amadeus.memory.context import assemble_turn_context

    config = _config(tmp_path, graph_enabled=True)
    config.memory.recall_min_query_chars = 0
    manager = MemoryManager(config, embedder=DeterministicEmbedder())
    fact = "User works with Alice on the CMB segmentation project this year"
    manager._store_fact(fact, 0.8)
    manager._update_graph([(fact, 0.8, None)])

    context, recall_items = assemble_turn_context(
        manager, "system prompt", "Tell me about Alice and the project"
    )
    combined = "\n".join(msg.content for msg in context)
    # The fact text should appear at most once across the merged recall block.
    assert combined.count("CMB segmentation project") <= 1


# --- storage parity: JSON sidecar vs SQLite ---------------------------


@pytest.mark.parametrize("backend", ["json", "sqlite"])
def test_graph_storage_roundtrip_parity(tmp_path: Path, backend: str) -> None:
    config = _config(tmp_path, graph_enabled=True, backend=backend)
    manager = MemoryManager(config, embedder=DeterministicEmbedder())
    manager._update_graph([("User works with Bob on Project Nova", 0.7, None)])

    reloaded = MemoryManager(config, embedder=DeterministicEmbedder())
    lines = reloaded.recall_from_graph("what about Bob")
    assert lines
    assert any("Bob" in line for line in lines)


def test_graph_json_sidecar_path(tmp_path: Path) -> None:
    config = _config(tmp_path, graph_enabled=True, backend="json")
    manager = MemoryManager(config, embedder=DeterministicEmbedder())
    manager._update_graph([("User works with Bob on Project Nova", 0.7, None)])
    sidecar = tmp_path / "memory_store_graph.json"
    assert sidecar.exists()
    doc = json.loads(sidecar.read_text())
    assert doc["entities"]


def test_graph_sqlite_no_sidecar_file(tmp_path: Path) -> None:
    config = _config(tmp_path, graph_enabled=True, backend="sqlite")
    manager = MemoryManager(config, embedder=DeterministicEmbedder())
    manager._update_graph([("User works with Bob on Project Nova", 0.7, None)])
    assert not (tmp_path / "memory_store_graph.json").exists()
    from amadeus.memory import sqlite_store

    doc = sqlite_store.load_graph_doc(sqlite_store.db_path(manager.persist_path))
    assert doc and doc.get("entities")


def test_graph_corrupt_sidecar_is_quarantined(tmp_path: Path) -> None:
    config = _config(tmp_path, graph_enabled=True, backend="json")
    manager = MemoryManager(config, embedder=DeterministicEmbedder())
    sidecar = tmp_path / "memory_store_graph.json"
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    sidecar.write_text("{not valid json")

    before = telemetry.snapshot().get("memory.graph_load_corrupt", 0)
    graph = graph_module.load_graph(manager)
    after = telemetry.snapshot().get("memory.graph_load_corrupt", 0)

    assert graph.entities == {}
    assert after == before + 1
    assert sidecar.with_name(sidecar.name + ".bak-corrupt").exists()


def test_on_summary_ready_updates_graph_from_extraction_output(tmp_path: Path) -> None:
    """The graph piggybacks on the extraction job's facts with no extra LLM
    call — verify the actual _on_summary_ready wiring, not just the direct
    _update_graph call."""
    from concurrent.futures import Future

    config = _config(tmp_path, graph_enabled=True)
    manager = MemoryManager(config, embedder=DeterministicEmbedder())
    future: Future = Future()
    future.set_result(
        {
            "summary": "talked about work",
            "strategy": "llm",
            "message_count": 2,
            "revision": 1,
            "facts": [
                ("User works with Alice on the CMB segmentation project", 0.8, None)
            ],
            "profile_updates": {},
            "open_threads": None,
            "tone": None,
            "felt": None,
        }
    )
    manager._on_summary_ready(future)
    assert manager.recall_from_graph("Alice") != []
