"""Synapse-map graph: nodes/edges from the vector store + profile."""

import pytest

from amadeus.config import Config
from amadeus.memory.manager import MemoryManager


class FirstWordEmbedder:
    """Deterministic toy embedder: identical first words -> identical
    vectors, so similarity edges are predictable."""

    def embed_text(self, text: str):
        seed = sum(ord(ch) for ch in text.split()[0].lower()) if text.strip() else 1
        return [((seed * (i + 3)) % 97) / 97.0 for i in range(8)]


def _manager(tmp_path) -> MemoryManager:
    config = Config()
    config.memory.enable_long_term = True
    config.memory.persist_path = str(tmp_path / "memory.json")
    config.memory.embeddings_path = str(tmp_path / "embeddings.json")
    config.memory.profile_path = str(tmp_path / "profile.json")
    return MemoryManager(config, embedder=FirstWordEmbedder())


def test_graph_nodes_and_similarity_edges(tmp_path):
    # Distinct first words: manager.remember() dedupes near-identical
    # vectors, so identical-seed memories would collapse to one record.
    manager = _manager(tmp_path)
    manager.remember("coffee is best black")
    manager.remember("trains run on time")
    manager.remember("zebras run fast")

    graph = manager.memory_graph()
    rec_nodes = [n for n in graph["nodes"] if n["id"].startswith("rec:")]
    assert len(rec_nodes) == 3
    assert all(n["kind"] == "fact" for n in rec_nodes)
    # Positive-valued toy vectors are mutually similar enough to clear the
    # edge floor; weights are valid cosines.
    assert graph["edges"]
    assert all(0.0 < e["weight"] <= 1.0 for e in graph["edges"])
    node_ids = {n["id"] for n in graph["nodes"]}
    assert all(
        e["source"] in node_ids and e["target"] in node_ids for e in graph["edges"]
    )


def test_graph_nodes_carry_pinned_flag(tmp_path):
    manager = _manager(tmp_path)
    manager.remember("coffee is best black", pinned=True)
    manager.remember("trains run on time")

    graph = manager.memory_graph()
    rec_nodes = {n["label"]: n for n in graph["nodes"] if n["id"].startswith("rec:")}
    assert rec_nodes["coffee is best black"]["pinned"] is True
    assert rec_nodes["trains run on time"]["pinned"] is False


def test_graph_includes_profile_and_threads(tmp_path):
    manager = _manager(tmp_path)
    manager.remember("coffee is best black")
    assert manager.profile is not None
    manager.profile.facts["name"] = "Alex"
    manager.profile.open_threads.append("coffee tasting next week")

    graph = manager.memory_graph()
    kinds = {n["kind"] for n in graph["nodes"]}
    assert {"fact", "profile", "thread"} <= kinds
    profile_node = next(n for n in graph["nodes"] if n["kind"] == "profile")
    assert profile_node["label"] == "name: Alex"
    # The thread shares its first word with the memory -> attached edge.
    thread_node = next(n for n in graph["nodes"] if n["kind"] == "thread")
    assert any(e["source"] == thread_node["id"] for e in graph["edges"])


def test_graph_endpoint(tmp_path):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from amadeus.web.app import create_app

    config = Config()
    config.memory.persist_path = str(tmp_path / "memory.json")
    config.memory.embeddings_path = str(tmp_path / "embeddings.json")
    config.memory.profile_path = str(tmp_path / "profile.json")
    app = create_app(config)
    client = TestClient(app)
    response = client.post("/api/memory", json={"action": "graph", "user_id": "user-1"})
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert set(body["graph"].keys()) == {"nodes", "edges"}
