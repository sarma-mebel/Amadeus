"""scripts/migrate_memory.py: store discovery + re-embedding.

Covers the per-character extension: guest stores under
``<memory_dir>/characters/<id>/`` (base file plus their own ``*_users/``
directories) must be discovered and migrated with the same idempotent
``*.bak-migrate`` backup behavior as the legacy base store — otherwise an
embedder switch leaves every guest's vectors at the old dimension and
recall's mismatch guard silently disables recall for them. All embedder
calls are faked; no live APIs.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "migrate_memory.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("migrate_memory_under_test", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeEmbedder:
    """Deterministic 4-d embedder; counts calls (no network, no models)."""

    dim = 4

    def __init__(self) -> None:
        self.calls = 0

    def embed_text(self, text: str):
        self.calls += 1
        return [float(len(text) % 7) + 1.0] * self.dim


def _write_store(path: Path, texts, dim: int = 2) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    records = [
        {"text": text, "vector": [1.0] * dim, "metadata": {"type": "fact"}}
        for text in texts
    ]
    path.write_text(json.dumps(records), encoding="utf-8")


def _dims(path: Path):
    records = json.loads(path.read_text(encoding="utf-8"))
    return sorted({len(rec["vector"]) for rec in records})


def _run(module, monkeypatch, config) -> int:
    monkeypatch.setattr(
        module, "Config", types.SimpleNamespace(load=lambda path: config)
    )
    monkeypatch.setattr(module, "load_embedder", lambda cfg: FakeEmbedder())
    monkeypatch.setattr(sys, "argv", ["migrate_memory.py", "--config", "unused.yaml"])
    return module.main()


@pytest.fixture()
def store_layout(tmp_path):
    """Legacy base store + per-user file + two character-scoped stores (one
    with its own per-user file), all at the old 2-d dimension."""
    from amadeus.config import Config

    config = Config()
    config.memory.embeddings_path = str(tmp_path / "embeddings.json")
    _write_store(tmp_path / "embeddings.json", ["kurisu base"])
    _write_store(tmp_path / "embeddings_users" / "alice.json", ["kurisu alice"])
    _write_store(tmp_path / "characters" / "rem" / "embeddings.json", ["rem base"])
    _write_store(
        tmp_path / "characters" / "rem" / "embeddings_users" / "bob.json", ["rem bob"]
    )
    _write_store(
        tmp_path / "characters" / "megumin" / "embeddings.json", ["megumin base"]
    )
    files = [
        tmp_path / "embeddings.json",
        tmp_path / "embeddings_users" / "alice.json",
        tmp_path / "characters" / "rem" / "embeddings.json",
        tmp_path / "characters" / "rem" / "embeddings_users" / "bob.json",
        tmp_path / "characters" / "megumin" / "embeddings.json",
    ]
    return config, files


def test_migrate_discovers_and_reembeds_character_stores(
    tmp_path, monkeypatch, capsys, store_layout
):
    config, files = store_layout
    module = _load_module()

    assert _run(module, monkeypatch, config) == 0

    for path in files:
        # Every store — including nested guest stores — is now 4-d...
        assert _dims(path) == [FakeEmbedder.dim], path
        # ...with the original 2-d state preserved as the backup.
        backup = path.with_name(path.name + ".bak-migrate")
        assert backup.exists(), backup
        assert _dims(backup) == [2], backup
    # Per-store logging keeps the character subdirectory visible.
    out = capsys.readouterr().out
    assert "characters/rem/embeddings.json" in out.replace("\\", "/")
    assert "characters/megumin/embeddings.json" in out.replace("\\", "/")
    assert "characters/rem/embeddings_users/bob.json" in out.replace("\\", "/")


def test_migrate_is_idempotent_and_keeps_first_backup(
    tmp_path, monkeypatch, store_layout
):
    config, files = store_layout
    module = _load_module()
    assert _run(module, monkeypatch, config) == 0
    after_first = [p.read_bytes() for p in files]

    # Second run: dimensions already match, nothing rewritten, and the
    # .bak-migrate files still hold the FIRST pre-migration state.
    assert _run(module, monkeypatch, config) == 0
    assert [p.read_bytes() for p in files] == after_first
    for path in files:
        assert _dims(path.with_name(path.name + ".bak-migrate")) == [2]


def test_migrate_skips_empty_and_missing_guest_stores(
    tmp_path, monkeypatch, capsys, store_layout
):
    """An empty guest store file is skipped with a summary line (never
    rewritten, no backup); a character directory with no store file at all is
    simply not a candidate — neither aborts the run."""
    config, files = store_layout
    empty = tmp_path / "characters" / "darkness" / "embeddings.json"
    empty.parent.mkdir(parents=True, exist_ok=True)
    empty.write_text("[]", encoding="utf-8")
    # Character dir with no embeddings file at all.
    (tmp_path / "characters" / "aqua").mkdir(parents=True, exist_ok=True)
    module = _load_module()

    assert _run(module, monkeypatch, config) == 0

    out = capsys.readouterr().out.replace("\\", "/")
    assert "characters/darkness/embeddings.json: empty, skipped" in out
    assert "aqua" not in out
    assert empty.read_text(encoding="utf-8") == "[]"
    assert not empty.with_name(empty.name + ".bak-migrate").exists()
    # The real stores still migrated around the skips.
    for path in files:
        assert _dims(path) == [FakeEmbedder.dim]


def test_migrate_flag_off_skips_character_stores(tmp_path, monkeypatch, store_layout):
    """per_character_memory off = fully legacy: only the base store and its
    per-user files are touched."""
    config, files = store_layout
    config.memory.per_character_memory = False
    module = _load_module()

    assert _run(module, monkeypatch, config) == 0

    legacy = files[:2]
    character_scoped = files[2:]
    for path in legacy:
        assert _dims(path) == [FakeEmbedder.dim]
    for path in character_scoped:
        assert _dims(path) == [2]
        assert not path.with_name(path.name + ".bak-migrate").exists()
