"""SQLite persistence backend (memory.storage_backend: sqlite).

Covers the low-level sqlite_store module (state/vector roundtrip, corrupt-db
quarantine, WAL concurrent reads) and its wiring into MemoryManager (backend
parity with the JSON store, per-character path scoping, clear()). No live
API calls; a tiny deterministic embedder stands in for a real one.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from amadeus import telemetry
from amadeus.config import Config
from amadeus.memory import sqlite_store
from amadeus.memory.manager import MemoryManager


class DeterministicEmbedder:
    """No network, no model: stable hash-based vectors."""

    def embed_text(self, text: str):
        import hashlib

        digest = hashlib.sha256(text.encode("utf-8")).digest()
        return [b / 255.0 for b in digest[:8]]

    def similarity(self, a, b) -> float:
        return sum(x * y for x, y in zip(a, b))


# --- sqlite_store.py low-level ---------------------------------------------


def test_db_path_swaps_suffix(tmp_path: Path) -> None:
    assert sqlite_store.db_path(tmp_path / "memory_store.json") == (
        tmp_path / "memory_store.db"
    )
    assert sqlite_store.db_path(tmp_path / "already.db") == tmp_path / "already.db"
    assert sqlite_store.db_path(None) is None


def test_state_save_load_roundtrip(tmp_path: Path) -> None:
    path = sqlite_store.db_path(tmp_path / "store.json")
    payload = {
        "messages": [
            {"role": "user", "content": "hi", "timestamp": "2026-01-01T00:00:00"},
            {
                "role": "assistant",
                "content": "hello!",
                "timestamp": "2026-01-01T00:00:01",
            },
        ],
        "summary": "a summary",
        "summary_metadata": {"revision": 3},
    }
    assert sqlite_store.save_state(path, payload) is True
    loaded = sqlite_store.load_state(path)
    assert loaded == payload


def test_state_load_missing_returns_none(tmp_path: Path) -> None:
    assert sqlite_store.load_state(tmp_path / "nope.db") is None


def test_vector_save_load_roundtrip(tmp_path: Path) -> None:
    path = sqlite_store.db_path(tmp_path / "embeddings.json")
    records = [
        {
            "id": "a",
            "vector": [0.1, 0.2],
            "text": "fact one",
            "metadata": {"type": "fact"},
        },
        {"id": "b", "vector": [0.3, 0.4], "text": "fact two", "metadata": {}},
    ]
    assert sqlite_store.save_vectors(path, records) is True
    store = sqlite_store.SqliteVectorStore(path=path)
    assert store.count() == 2
    dumped = {r.id: r for r in store.dump()}
    assert dumped["a"].text == "fact one"
    assert dumped["a"].metadata == {"type": "fact"}


def test_graph_doc_roundtrip(tmp_path: Path) -> None:
    path = sqlite_store.db_path(tmp_path / "store.json")
    doc = {"version": 1, "entities": [{"key": "alice"}], "relations": []}
    assert sqlite_store.save_graph_doc(path, doc) is True
    assert sqlite_store.load_graph_doc(path) == doc


def test_corrupt_db_is_quarantined_and_counted(tmp_path: Path) -> None:
    path = sqlite_store.db_path(tmp_path / "store.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"not a sqlite database at all")
    before = telemetry.snapshot().get("memory.sqlite_load_corrupt", 0)

    loaded = sqlite_store.load_state(path)

    assert loaded == {"messages": []}  # fresh start, not a crash
    backup = path.with_name(path.name + ".bak-corrupt")
    assert backup.exists()
    assert backup.read_bytes() == b"not a sqlite database at all"
    after = telemetry.snapshot().get("memory.sqlite_load_corrupt", 0)
    assert after == before + 1


def test_wal_mode_allows_concurrent_read_during_write(tmp_path: Path) -> None:
    path = sqlite_store.db_path(tmp_path / "store.json")
    sqlite_store.save_state(
        path, {"messages": [], "summary": None, "summary_metadata": {}}
    )
    assert path.exists()

    writer = sqlite_store._connect(path)
    writer.execute("BEGIN IMMEDIATE")
    writer.execute(
        "INSERT INTO messages (role, content, timestamp) VALUES ('user', 'x', NULL)"
    )
    try:
        # A second connection can still READ while the writer holds an
        # uncommitted write transaction under WAL — the whole point of
        # WAL mode for the scheduler's disk-snapshot reads.
        reader = sqlite3.connect(str(path), timeout=1.0)
        rows = reader.execute("SELECT COUNT(*) FROM meta").fetchall()
        assert rows is not None
        reader.close()
    finally:
        writer.rollback()
        writer.close()


# --- MemoryManager wiring ----------------------------------------------


def _config(tmp_path: Path, backend: str = "json") -> Config:
    config = Config()
    config.memory.storage_backend = backend
    config.memory.enable_long_term = True
    config.memory.persist_path = str(tmp_path / "memory_store.json")
    config.memory.embeddings_path = str(tmp_path / "memory_embeddings.json")
    config.memory.profile_path = str(tmp_path / "memory_profile.json")
    return config


def test_sqlite_backend_default_is_json() -> None:
    assert Config().memory.storage_backend == "json"


@pytest.mark.parametrize("backend", ["json", "sqlite"])
def test_manager_persist_reload_parity(tmp_path: Path, backend: str) -> None:
    config = _config(tmp_path, backend)
    manager = MemoryManager(config, embedder=DeterministicEmbedder())
    manager.add_message("user", "remember this fact")
    manager.add_message("assistant", "noted")
    manager._persist()

    reloaded = MemoryManager(config, embedder=DeterministicEmbedder())
    contents = [m.content for m in reloaded.short_term]
    assert "remember this fact" in contents
    assert "noted" in contents


@pytest.mark.parametrize("backend", ["json", "sqlite"])
def test_manager_vector_recall_parity(tmp_path: Path, backend: str) -> None:
    config = _config(tmp_path, backend)
    manager = MemoryManager(config, embedder=DeterministicEmbedder())
    manager.remember("the sky is blue today")
    manager.remember("trains run on schedule")

    reloaded = MemoryManager(config, embedder=DeterministicEmbedder())
    assert reloaded.vector_store is not None
    assert reloaded.vector_store.count() == 2


def test_sqlite_backend_creates_db_not_json(tmp_path: Path) -> None:
    config = _config(tmp_path, "sqlite")
    manager = MemoryManager(config, embedder=DeterministicEmbedder())
    manager.add_message("user", "hi")
    manager.add_message("assistant", "hello")
    manager._persist()
    manager.remember("a durable fact")

    assert (tmp_path / "memory_store.db").exists()
    assert not (tmp_path / "memory_store.json").exists()
    assert (tmp_path / "memory_embeddings.db").exists()
    assert not (tmp_path / "memory_embeddings.json").exists()


def test_sqlite_backend_clear_removes_db_and_wal_sidecars(tmp_path: Path) -> None:
    config = _config(tmp_path, "sqlite")
    manager = MemoryManager(config, embedder=DeterministicEmbedder())
    manager.add_message("user", "hi")
    manager.add_message("assistant", "hello")
    manager._persist()
    manager.remember("a fact")

    state_db = tmp_path / "memory_store.db"
    vector_db = tmp_path / "memory_embeddings.db"
    assert state_db.exists()
    assert vector_db.exists()

    manager.clear()

    assert not state_db.exists()
    assert not Path(f"{state_db}-wal").exists()
    assert not Path(f"{state_db}-shm").exists()


def test_sqlite_orphan_user_message_healed_on_load(tmp_path: Path) -> None:
    config = _config(tmp_path, "sqlite")
    manager = MemoryManager(config, embedder=DeterministicEmbedder())
    manager.add_message("user", "orphan question")
    manager._persist()  # crashed turn: persisted state ends on a user msg

    before = telemetry.snapshot().get("memory.orphaned_user_messages_healed", 0)
    reloaded = MemoryManager(config, embedder=DeterministicEmbedder())
    after = telemetry.snapshot().get("memory.orphaned_user_messages_healed", 0)

    assert list(reloaded.short_term) == []
    assert after == before + 1


def test_sqlite_per_character_paths_scope_identically(tmp_path: Path) -> None:
    """character_memory_path re-parents into characters/<id>/ the same way
    regardless of backend — only the file suffix should differ."""
    from amadeus.paths import character_scoped_path

    json_scoped = character_scoped_path(str(tmp_path / "memory_store.json"), "rem")
    assert json_scoped == str(tmp_path / "characters" / "rem" / "memory_store.json")
    db_scoped = sqlite_store.db_path(Path(json_scoped))
    assert db_scoped == tmp_path / "characters" / "rem" / "memory_store.db"


def test_sqlite_vector_store_corrupt_quarantines_and_returns_empty(
    tmp_path: Path,
) -> None:
    path = sqlite_store.db_path(tmp_path / "embeddings.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"garbage")

    store = sqlite_store.SqliteVectorStore(path=path)

    assert store.count() == 0
    assert path.with_name(path.name + ".bak-corrupt").exists()
