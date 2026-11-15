#!/usr/bin/env python3
"""One-way converter: JSON memory stores -> SQLite (``memory.storage_backend:
sqlite``).

Converts, for every store this config knows about:

* the message/summary store (``memory.persist_path``, .json -> .db rows)
* the vector store (``memory.embeddings_path``, when ``vector_backend`` is
  the default JSON-backed one — chroma is already its own persistent
  backend and is left untouched)
* the entity-graph JSON sidecar (``<persist_path stem>_graph.json``), if one
  exists — folded into the same state db as a single JSON document

across the base store, every per-user ``<stem>_users/<id>.json`` file, and
(when ``memory.per_character_memory`` is on) every per-character guest store
under ``characters/<id>/``. Idempotent: converting an already-converted
store just rewrites the .db from the .json (safe to re-run), and every
original .json is preserved as ``<name>.bak-migrate`` (first run only, same
convention as ``migrate_memory.py`` — a second run does not clobber the
first backup).

This only writes the new .db files; it does not flip ``storage_backend`` in
your config or delete the .json originals — do that yourself once you've
confirmed the app runs correctly against the converted store.

    python scripts/convert_memory_sqlite.py --config config.local.yaml --dry-run
    python scripts/convert_memory_sqlite.py --config config.local.yaml
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from amadeus.config import Config  # noqa: E402
from amadeus.memory import sqlite_store  # noqa: E402
from amadeus.paths import resolve_user_data_path  # noqa: E402


def _store_files(store_base: Path) -> list[Path]:
    """One store's files: the base file plus its `<stem>_users/*.json`."""
    found = []
    if store_base.exists():
        found.append(store_base)
    users_dir = store_base.with_name(store_base.stem + "_users")
    if users_dir.is_dir():
        found.extend(sorted(users_dir.glob("*.json")))
    return found


def _discover(base: Path, per_character: bool) -> list[Path]:
    candidates = _store_files(base)
    if per_character:
        characters_root = base.parent / "characters"
        if characters_root.is_dir():
            for char_dir in sorted(characters_root.iterdir()):
                if char_dir.is_dir():
                    candidates.extend(_store_files(char_dir / base.name))
    return [p for p in candidates if not p.name.endswith((".bak-migrate", ".tmp"))]


def _label(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return path.name


def _backup(path: Path) -> None:
    backup = path.with_name(path.name + ".bak-migrate")
    if not backup.exists():  # keep the FIRST pre-conversion state
        shutil.copy2(path, backup)


def _convert_state_store(path: Path, *, dry_run: bool, root: Path) -> str:
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        return f"SKIP {_label(path, root)}: unreadable ({exc})"
    if not isinstance(data, dict):
        return f"SKIP {_label(path, root)}: not a state store"
    db_target = sqlite_store.db_path(path)
    if dry_run:
        n = len(data.get("messages") or [])
        return f"DRY  {_label(path, root)}: would convert {n} message(s) -> {db_target.name}"
    _backup(path)
    ok = sqlite_store.save_state(db_target, data)
    if not ok:
        return f"FAIL {_label(path, root)}: sqlite write failed"
    # Fold the entity-graph JSON sidecar (if present) into the same db.
    graph_sidecar = path.with_name(path.stem + "_graph.json")
    if graph_sidecar.exists():
        try:
            with graph_sidecar.open("r", encoding="utf-8") as handle:
                graph_doc = json.load(handle)
            _backup(graph_sidecar)
            sqlite_store.save_graph_doc(db_target, graph_doc)
        except (OSError, json.JSONDecodeError) as exc:
            return (
                f"DONE {_label(path, root)}: {len(data.get('messages') or [])} "
                f"message(s) -> {db_target.name} (graph sidecar unreadable: {exc})"
            )
    n = len(data.get("messages") or [])
    return f"DONE {_label(path, root)}: {n} message(s) -> {db_target.name}"


def _convert_vector_store(path: Path, *, dry_run: bool, root: Path) -> str:
    try:
        with path.open("r", encoding="utf-8") as handle:
            records = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        return f"SKIP {_label(path, root)}: unreadable ({exc})"
    if not isinstance(records, list):
        return f"SKIP {_label(path, root)}: not a vector store"
    db_target = sqlite_store.db_path(path)
    if dry_run:
        return f"DRY  {_label(path, root)}: would convert {len(records)} record(s) -> {db_target.name}"
    _backup(path)
    ok = sqlite_store.save_vectors(db_target, records)
    if not ok:
        return f"FAIL {_label(path, root)}: sqlite write failed"
    return f"DONE {_label(path, root)}: {len(records)} record(s) -> {db_target.name}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.local.yaml")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = Config.load(Path(args.config))
    per_character = bool(getattr(config.memory, "per_character_memory", True))

    state_base = Path(resolve_user_data_path(config.memory.persist_path))
    state_candidates = _discover(state_base, per_character)

    vector_backend = getattr(config.memory, "vector_backend", "json")
    vector_candidates: list[Path] = []
    if vector_backend in {"json", "memory"}:
        vector_base = Path(resolve_user_data_path(config.memory.embeddings_path))
        vector_candidates = _discover(vector_base, per_character)
    else:
        print(
            f"vector_backend={vector_backend!r} is already its own persistent "
            "backend; only the message/summary store will be converted."
        )

    if not state_candidates and not vector_candidates:
        print("No JSON store files found — nothing to convert.")
        return 0

    root = state_base.parent
    print(f"Converting {len(state_candidates)} state store(s):")
    for path in state_candidates:
        print("  " + _convert_state_store(path, dry_run=args.dry_run, root=root))

    if vector_candidates:
        print(f"Converting {len(vector_candidates)} vector store(s):")
        for path in vector_candidates:
            print("  " + _convert_vector_store(path, dry_run=args.dry_run, root=root))

    if args.dry_run:
        print("\nDry run only — nothing written.")
    else:
        print(
            "\nDone. Originals preserved as *.bak-migrate. Set "
            "memory.storage_backend: sqlite in your config to switch over "
            "(re-run this converter first if you keep using the app on JSON "
            "in the meantime — it is idempotent and safe to re-run)."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
