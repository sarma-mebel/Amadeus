"""Rolling backups + export/import of memory files (amadeus/memory/backup.py)."""

from __future__ import annotations

import io
import json
import zipfile

import pytest

from amadeus import telemetry
from amadeus.config import Config
from amadeus.memory import backup, sqlite_store


def _config(tmp_path) -> Config:
    config = Config()
    config.memory.persist_path = str(tmp_path / "memory_store.json")
    config.memory.embeddings_path = str(tmp_path / "memory_embeddings.json")
    config.memory.profile_path = str(tmp_path / "memory_profile.json")
    config.memory.transcripts_path = str(tmp_path / "memory_transcripts.json")
    return config


def _seed_files(tmp_path, user_id=None) -> None:
    config = _config(tmp_path)
    for kind, path in backup.memory_paths(config, user_id).items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"kind": kind}), encoding="utf-8")


def test_create_backup_copies_files_and_writes_manifest(tmp_path):
    config = _config(tmp_path)
    _seed_files(tmp_path)

    dest = backup.create_backup(config, reason="test")

    assert dest is not None and dest.is_dir()
    manifest = json.loads((dest / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["reason"] == "test"
    assert set(manifest["files"]) == {"store", "embeddings", "profile", "transcripts"}
    for name in manifest["files"].values():
        assert (dest / name).is_file()


def test_create_backup_returns_none_when_nothing_exists(tmp_path):
    assert backup.create_backup(_config(tmp_path)) is None


def test_backups_are_pruned_to_keep(tmp_path):
    config = _config(tmp_path)
    _seed_files(tmp_path)
    dirs = []
    for i in range(4):
        dest = backup.create_backup(config, reason=f"r{i}", keep=2)
        dirs.append(dest)
        # Distinct timestamps: rename to force ordering rather than sleeping.
    user_dir = dirs[-1].parent
    remaining = sorted(p.name for p in user_dir.iterdir() if p.is_dir())
    assert len(remaining) <= 2


def test_maybe_daily_backup_fires_once(tmp_path):
    config = _config(tmp_path)
    _seed_files(tmp_path)

    first = backup.maybe_daily_backup(config)
    second = backup.maybe_daily_backup(config)

    assert first is not None
    assert second is None  # newest backup is fresh, no second one


def test_export_import_roundtrip(tmp_path):
    config = _config(tmp_path)
    _seed_files(tmp_path)

    exported = backup.export_memory(config)
    assert exported is not None

    # Wipe and restore.
    for path in backup.memory_paths(config).values():
        path.unlink()
    restored = backup.import_memory(config, None, exported)

    assert restored == ["embeddings", "profile", "store", "transcripts"]
    for kind, path in backup.memory_paths(config).items():
        assert json.loads(path.read_text(encoding="utf-8")) == {"kind": kind}


def test_import_takes_a_pre_import_backup(tmp_path):
    config = _config(tmp_path)
    _seed_files(tmp_path)
    exported = backup.export_memory(config)

    backup.import_memory(config, None, exported)

    newest = backup.latest_backup(config)
    assert newest is not None and newest.name.endswith("pre-import")


def test_import_rejects_garbage(tmp_path):
    config = _config(tmp_path)
    with pytest.raises(ValueError, match="not a zip"):
        backup.import_memory(config, None, b"definitely not a zip")


def test_import_rejects_missing_manifest(tmp_path):
    config = _config(tmp_path)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("store.json", "{}")
    with pytest.raises(ValueError, match="manifest"):
        backup.import_memory(config, None, buffer.getvalue())


def test_import_rejects_invalid_json_section(tmp_path):
    config = _config(tmp_path)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr(
            "manifest.json",
            json.dumps({"version": 1, "files": {"store": "store.json"}}),
        )
        archive.writestr("store.json", "not json at all")
    with pytest.raises(ValueError, match="store"):
        backup.import_memory(config, None, buffer.getvalue())
    # Nothing was written.
    assert not any(p.exists() for p in backup.memory_paths(config).values())


def test_import_rejects_path_traversal_names(tmp_path):
    config = _config(tmp_path)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr(
            "manifest.json",
            json.dumps({"version": 1, "files": {"store": "../evil.json"}}),
        )
        archive.writestr("../evil.json", "{}")
    with pytest.raises(ValueError, match="invalid"):
        backup.import_memory(config, None, buffer.getvalue())


def _sqlite_config(tmp_path) -> Config:
    config = _config(tmp_path)
    config.memory.storage_backend = "sqlite"
    return config


def _seed_sqlite_files(tmp_path) -> Config:
    """Real .db files at the paths the sqlite backend actually reads."""
    config = _sqlite_config(tmp_path)
    paths = backup.memory_paths(config)
    assert sqlite_store.save_state(
        paths["store"],
        {"messages": [{"role": "user", "content": "hi"}], "summary": "a summary"},
    )
    assert sqlite_store.save_vectors(
        paths["embeddings"],
        [{"id": "r1", "vector": [1.0, 0.0], "text": "a fact", "metadata": {}}],
    )
    for kind in ("profile", "transcripts"):
        paths[kind].parent.mkdir(parents=True, exist_ok=True)
        paths[kind].write_text(json.dumps({"kind": kind}), encoding="utf-8")
    return config


def test_sqlite_backend_paths_point_at_db_files(tmp_path):
    paths = backup.memory_paths(_sqlite_config(tmp_path))
    assert paths["store"].suffix == ".db"
    assert paths["embeddings"].suffix == ".db"
    # Profile/transcripts stay JSON regardless of the backend.
    assert paths["profile"].suffix == ".json"
    assert paths["transcripts"].suffix == ".json"


def test_sqlite_backend_backup_includes_db_files(tmp_path):
    """Regression: backups used to enumerate the configured .json paths only,
    silently omitting a sqlite user's live message and vector stores."""
    config = _seed_sqlite_files(tmp_path)

    dest = backup.create_backup(config, reason="test")

    assert dest is not None
    manifest = json.loads((dest / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["files"]["store"] == "store.db"
    assert manifest["files"]["embeddings"] == "embeddings.db"
    assert set(manifest["files"]) == {"store", "embeddings", "profile", "transcripts"}
    for name in manifest["files"].values():
        assert (dest / name).is_file()
    # The copied databases are real SQLite files, not empty shells.
    assert (dest / "store.db").read_bytes().startswith(b"SQLite format 3\x00")


def test_sqlite_backend_export_import_roundtrip(tmp_path):
    config = _seed_sqlite_files(tmp_path)

    exported = backup.export_memory(config)
    assert exported is not None

    for path in backup.memory_paths(config).values():
        path.unlink()
    restored = backup.import_memory(config, None, exported)

    assert restored == ["embeddings", "profile", "store", "transcripts"]
    state = sqlite_store.load_state(backup.memory_paths(config)["store"])
    assert state is not None and state["summary"] == "a summary"
    assert state["messages"][0]["content"] == "hi"


def test_import_rejects_cross_backend_export(tmp_path):
    """A JSON-backend export must not be written over live .db files (it
    would be quarantined as corruption on the next load)."""
    json_dir = tmp_path / "json"
    json_dir.mkdir()
    _seed_files(json_dir)
    exported = backup.export_memory(_config(json_dir))
    assert exported is not None

    sqlite_dir = tmp_path / "sqlite"
    sqlite_dir.mkdir()
    with pytest.raises(ValueError, match="different storage backend"):
        backup.import_memory(_sqlite_config(sqlite_dir), None, exported)


def test_backup_with_no_files_counts(tmp_path):
    telemetry.reset()
    try:
        assert backup.create_backup(_config(tmp_path)) is None
        assert telemetry.snapshot().get("memory.backup_no_files") == 1
    finally:
        telemetry.reset()


def test_user_scoped_paths_isolate_users(tmp_path):
    config = _config(tmp_path)
    _seed_files(tmp_path, user_id="alice")
    _seed_files(tmp_path)

    alice_files = backup.memory_paths(config, "alice")
    assert all("_users" in str(p) for p in alice_files.values())

    exported = backup.export_memory(config, "alice")
    assert exported is not None
    # Importing alice's export as bob writes bob's files only.
    restored = backup.import_memory(config, "bob", exported)
    assert restored
    bob_store = backup.memory_paths(config, "bob")["store"]
    assert bob_store.is_file()
