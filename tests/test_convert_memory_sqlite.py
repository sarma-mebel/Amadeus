"""scripts/convert_memory_sqlite.py: one-way JSON -> SQLite converter.

Covers discovery (base + per-user + per-character guest stores, same layout
as migrate_memory.py), idempotency (.bak-migrate preserved on re-run,
converted .db content matches source), dry-run (no writes), and the graph
JSON sidecar folding into the converted db. No live API calls.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest

_SCRIPT = (
    Path(__file__).resolve().parent.parent / "scripts" / "convert_memory_sqlite.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "convert_memory_sqlite_under_test", _SCRIPT
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_state_store(path: Path, messages) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"messages": messages, "summary": None, "summary_metadata": {}}),
        encoding="utf-8",
    )


def _write_vector_store(path: Path, texts) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    records = [
        {"id": f"id-{i}", "text": t, "vector": [0.1, 0.2], "metadata": {}}
        for i, t in enumerate(texts)
    ]
    path.write_text(json.dumps(records), encoding="utf-8")


def _run(module, monkeypatch, config, *, dry_run: bool = False) -> int:
    monkeypatch.setattr(
        module, "Config", types.SimpleNamespace(load=lambda path: config)
    )
    argv = ["convert_memory_sqlite.py", "--config", "unused.yaml"]
    if dry_run:
        argv.append("--dry-run")
    monkeypatch.setattr(sys, "argv", argv)
    return module.main()


@pytest.fixture()
def store_layout(tmp_path):
    from amadeus.config import Config

    config = Config()
    config.memory.persist_path = str(tmp_path / "memory_store.json")
    config.memory.embeddings_path = str(tmp_path / "memory_embeddings.json")

    _write_state_store(
        tmp_path / "memory_store.json",
        [{"role": "user", "content": "hi base", "timestamp": "2026-01-01T00:00:00"}],
    )
    _write_vector_store(tmp_path / "memory_embeddings.json", ["base fact"])
    _write_state_store(
        tmp_path / "memory_store_users" / "alice.json",
        [{"role": "user", "content": "hi alice", "timestamp": "2026-01-01T00:00:00"}],
    )
    _write_state_store(
        tmp_path / "characters" / "rem" / "memory_store.json",
        [{"role": "user", "content": "hi rem", "timestamp": "2026-01-01T00:00:00"}],
    )
    return config


def test_dry_run_writes_nothing(tmp_path, monkeypatch, store_layout):
    module = _load_module()
    rc = _run(module, monkeypatch, store_layout, dry_run=True)
    assert rc == 0
    assert not (tmp_path / "memory_store.db").exists()
    assert not (tmp_path / "memory_embeddings.db").exists()


def test_converts_base_and_per_user_and_character_stores(
    tmp_path, monkeypatch, store_layout
):
    module = _load_module()
    rc = _run(module, monkeypatch, store_layout)
    assert rc == 0

    from amadeus.memory import sqlite_store

    base_db = tmp_path / "memory_store.db"
    assert base_db.exists()
    loaded = sqlite_store.load_state(base_db)
    assert loaded["messages"][0]["content"] == "hi base"

    user_db = tmp_path / "memory_store_users" / "alice.db"
    assert user_db.exists()
    assert sqlite_store.load_state(user_db)["messages"][0]["content"] == "hi alice"

    char_db = tmp_path / "characters" / "rem" / "memory_store.db"
    assert char_db.exists()
    assert sqlite_store.load_state(char_db)["messages"][0]["content"] == "hi rem"

    vector_db = tmp_path / "memory_embeddings.db"
    assert vector_db.exists()
    vs = sqlite_store.SqliteVectorStore(path=vector_db)
    assert vs.count() == 1

    # Originals preserved.
    assert (tmp_path / "memory_store.json").exists()
    assert (tmp_path / "memory_store.json.bak-migrate").exists()


def test_idempotent_second_run_keeps_first_backup(tmp_path, monkeypatch, store_layout):
    module = _load_module()
    _run(module, monkeypatch, store_layout)
    backup = tmp_path / "memory_store.json.bak-migrate"
    first_backup_bytes = backup.read_bytes()

    # Mutate the source after first conversion (simulating ongoing JSON use)
    # and reconvert -- backup must stay the FIRST pre-conversion snapshot.
    _write_state_store(
        tmp_path / "memory_store.json",
        [{"role": "user", "content": "hi base v2", "timestamp": "2026-01-01T00:00:02"}],
    )
    rc = _run(module, monkeypatch, store_layout)
    assert rc == 0
    assert backup.read_bytes() == first_backup_bytes

    from amadeus.memory import sqlite_store

    loaded = sqlite_store.load_state(tmp_path / "memory_store.db")
    assert loaded["messages"][0]["content"] == "hi base v2"


def test_converts_graph_sidecar_into_db(tmp_path, monkeypatch, store_layout):
    sidecar = tmp_path / "memory_store_graph.json"
    sidecar.write_text(
        json.dumps({"version": 1, "entities": [{"key": "alice"}], "relations": []}),
        encoding="utf-8",
    )
    module = _load_module()
    rc = _run(module, monkeypatch, store_layout)
    assert rc == 0

    from amadeus.memory import sqlite_store

    doc = sqlite_store.load_graph_doc(tmp_path / "memory_store.db")
    assert doc is not None
    assert doc["entities"] == [{"key": "alice"}]
    assert (tmp_path / "memory_store_graph.json.bak-migrate").exists()


def test_no_stores_found_returns_zero(tmp_path, monkeypatch):
    from amadeus.config import Config

    config = Config()
    config.memory.persist_path = str(tmp_path / "nowhere.json")
    config.memory.embeddings_path = str(tmp_path / "nowhere_embeddings.json")
    module = _load_module()
    rc = _run(module, monkeypatch, config)
    assert rc == 0
