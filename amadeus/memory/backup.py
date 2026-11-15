"""Rolling backups and export/import of a user's memory files.

Her memories are the emotional core of the app, and they live in a handful
of JSON files. One corrupted or accidentally deleted file must never mean
total loss, so: a cheap daily rolling backup fires at session start, every
import takes a pre-import backup first, and export/import gives users a
portable copy they own.
"""

from __future__ import annotations

import datetime as dt
import io
import json
import os
import shutil
import sqlite3
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from .. import telemetry
from ..logging_utils import get_logger
from .sqlite_store import db_path as _sqlite_db_path

LOGGER = get_logger(__name__)

# kind -> MemorySettings attribute holding the (already data-dir-resolved) path.
MEMORY_KINDS: Dict[str, str] = {
    "store": "persist_path",
    "embeddings": "embeddings_path",
    "profile": "profile_path",
    "transcripts": "transcripts_path",
}

# Kinds whose live file is a ``.db`` sibling of the configured JSON path when
# ``memory.storage_backend`` is "sqlite" (see sqlite_store.db_path). Profile
# and transcripts stay JSON regardless of the backend.
_SQLITE_KINDS = ("store", "embeddings")
_SQLITE_MAGIC = b"SQLite format 3\x00"
_SQLITE_SIDECAR_EXTS = ("-wal", "-shm")

EXPORT_FORMAT_VERSION = 1
MANIFEST_NAME = "manifest.json"
# Zip-bomb guard: reject imports whose declared decompressed size exceeds
# this. Real stores are a few MB; embeddings dominate and stay well under.
MAX_IMPORT_UNCOMPRESSED_BYTES = 512 * 1024 * 1024
DEFAULT_KEEP_BACKUPS = 10


def user_scoped_path(base: Optional[Path], user_id: Optional[str]) -> Optional[Path]:
    """Per-user variant of a memory file path.

    Mirrors the web SessionStore convention: ``memory_store.json`` for the
    base user, ``memory_store_users/<user_id>.json`` for identified users.
    """

    if base is None:
        return None
    if user_id is None:
        return base
    if base.is_dir():
        directory = base
        suffix = base.suffix or ".json"
    elif base.suffix:
        directory = base.parent / f"{base.stem}_users"
        suffix = base.suffix
    else:
        directory = base
        suffix = ".json"
    return directory / f"{user_id}{suffix}"


def _character_scope(config: Any):
    """Per-character path scoper for this config (identity when builtin).

    Backups/exports must follow the ACTIVE character's store files: a guest
    character's memory — transcripts included, since 0.13 they are scoped
    like the memory tiers — lives under characters/<id>/ (see
    Config.character_memory_path) while the builtin Kurisu keeps the legacy
    locations.
    """

    scope = getattr(config, "character_memory_path", None)
    if callable(scope):
        return scope
    return lambda value: value


def _is_sqlite_backend(config: Any) -> bool:
    memory_cfg = getattr(config, "memory", None)
    return str(getattr(memory_cfg, "storage_backend", "json") or "json") == "sqlite"


def memory_paths(config: Any, user_id: Optional[str] = None) -> Dict[str, Path]:
    """The user's memory files by kind (configured paths; may not exist).

    Backend-aware: with ``storage_backend: sqlite`` the message store and
    vector store live in ``.db`` siblings of the configured ``.json`` paths
    (same directory/stem — sqlite_store.db_path — so per-user/per-character
    scoping is unaffected). Returning the configured ``.json`` paths here
    made daily backups, exports, and pre-import safety backups silently omit
    the files that actually hold a sqlite user's memory.
    """

    out: Dict[str, Path] = {}
    memory_cfg = getattr(config, "memory", None)
    scope = _character_scope(config)
    sqlite_backend = _is_sqlite_backend(config)
    for kind, attr in MEMORY_KINDS.items():
        raw = getattr(memory_cfg, attr, None)
        if raw:
            raw = scope(raw)
        if not raw:
            continue
        path = user_scoped_path(Path(raw), user_id)
        if path is None:
            continue
        if sqlite_backend and kind in _SQLITE_KINDS:
            path = _sqlite_db_path(path)
        out[kind] = path
    return out


def backup_root(config: Any) -> Path:
    """Backups live next to the profile/store files, never inside them."""

    scope = _character_scope(config)
    for attr in ("profile_path", "persist_path"):
        raw = getattr(getattr(config, "memory", None), attr, None)
        if raw:
            return Path(scope(raw)).parent / "memory_backups"
    from ..paths import default_data_dir

    return default_data_dir() / "memory_backups"


def _user_backup_dir(config: Any, user_id: Optional[str]) -> Path:
    return backup_root(config) / (user_id or "default")


def _existing_files(config: Any, user_id: Optional[str]) -> Dict[str, Path]:
    return {k: p for k, p in memory_paths(config, user_id).items() if p.is_file()}


def _checkpoint_sqlite(path: Path) -> None:
    """Fold a database's WAL sidecar into the main file so a plain file copy
    is a complete snapshot. Best-effort: on failure the copy paths below
    still grab any surviving ``-wal``/``-shm`` sidecars alongside the .db.
    """
    try:
        conn = sqlite3.connect(str(path), timeout=5.0)
        try:
            conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        finally:
            conn.close()
    except Exception as exc:
        LOGGER.warning("WAL checkpoint before backup failed for %s: %s", path, exc)
        telemetry.incr("memory.backup_wal_checkpoint_failed")


def create_backup(
    config: Any,
    user_id: Optional[str] = None,
    *,
    reason: str = "manual",
    keep: int = DEFAULT_KEEP_BACKUPS,
) -> Optional[Path]:
    """Copy the user's existing memory files into a timestamped backup dir.

    Returns the backup directory, or None when there is nothing to back up.
    Oldest backups beyond ``keep`` are pruned.
    """

    files = _existing_files(config, user_id)
    if not files:
        # A backup that finds nothing on an install that HAS memory files
        # (e.g. a paths/backend mismatch) would be silent data-loss
        # exposure — count it so /api/status shows the gap. All-zeros
        # stays the healthy steady state; fresh installs bump it once.
        telemetry.incr("memory.backup_no_files")
        return None
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    dest = _user_backup_dir(config, user_id) / f"{stamp}-{reason}"
    dest.mkdir(parents=True, exist_ok=True)
    manifest: Dict[str, Any] = {
        "version": EXPORT_FORMAT_VERSION,
        "created_at": dt.datetime.now().isoformat(timespec="seconds"),
        "reason": reason,
        "user_id": user_id,
        "files": {},
    }
    for kind, path in files.items():
        name = f"{kind}{path.suffix or '.json'}"
        if path.suffix == ".db":
            _checkpoint_sqlite(path)
        shutil.copy2(path, dest / name)
        if path.suffix == ".db":
            # Sidecars surviving a failed checkpoint hold committed writes;
            # copy them beside the .db (recovery extras, not manifest kinds).
            for ext in _SQLITE_SIDECAR_EXTS:
                sidecar = path.with_name(path.name + ext)
                if sidecar.is_file():
                    shutil.copy2(sidecar, dest / (name + ext))
        manifest["files"][kind] = name
    (dest / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    _prune_backups(_user_backup_dir(config, user_id), keep)
    return dest


def _prune_backups(user_dir: Path, keep: int) -> None:
    if keep <= 0 or not user_dir.is_dir():
        return
    backups = sorted(p for p in user_dir.iterdir() if p.is_dir())
    for stale in backups[:-keep] if len(backups) > keep else []:
        shutil.rmtree(stale, ignore_errors=True)


def latest_backup(config: Any, user_id: Optional[str] = None) -> Optional[Path]:
    user_dir = _user_backup_dir(config, user_id)
    if not user_dir.is_dir():
        return None
    backups = sorted(p for p in user_dir.iterdir() if p.is_dir())
    return backups[-1] if backups else None


def maybe_daily_backup(
    config: Any, user_id: Optional[str] = None, *, max_age_hours: float = 24.0
) -> Optional[Path]:
    """Rolling backup: take one if the newest is older than ``max_age_hours``.

    Called at session start (web and CLI). Never raises — a failed backup
    must not break the conversation it was protecting.
    """

    try:
        newest = latest_backup(config, user_id)
        if newest is not None:
            age = dt.datetime.now().timestamp() - newest.stat().st_mtime
            if age < max_age_hours * 3600:
                return None
        return create_backup(config, user_id, reason="daily")
    except Exception as exc:  # pragma: no cover - defensive
        LOGGER.warning("Daily memory backup failed: %s", exc)
        return None


def export_memory(config: Any, user_id: Optional[str] = None) -> Optional[bytes]:
    """A portable zip of the user's memory files, or None when empty."""

    files = _existing_files(config, user_id)
    if not files:
        return None
    manifest: Dict[str, Any] = {
        "version": EXPORT_FORMAT_VERSION,
        "exported_at": dt.datetime.now().isoformat(timespec="seconds"),
        "user_id": user_id,
        "files": {},
    }
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for kind, path in files.items():
            name = f"{kind}{path.suffix or '.json'}"
            if path.suffix == ".db":
                _checkpoint_sqlite(path)
            archive.write(path, name)
            manifest["files"][kind] = name
        archive.writestr(MANIFEST_NAME, json.dumps(manifest, indent=2))
    return buffer.getvalue()


def import_memory(config: Any, user_id: Optional[str], data: bytes) -> List[str]:
    """Restore memory files from an export zip. Returns the kinds restored.

    Validates the archive (manifest version, known file names, size caps,
    JSON well-formedness) and takes a pre-import backup of whatever the
    user currently has before overwriting anything. Raises ValueError on
    any validation failure, leaving the live files untouched.
    """

    try:
        archive = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile as exc:
        raise ValueError("Not a valid memory export (not a zip file).") from exc

    total_uncompressed = sum(info.file_size for info in archive.infolist())
    if total_uncompressed > MAX_IMPORT_UNCOMPRESSED_BYTES:
        raise ValueError("Memory export is too large to import.")

    names = set(archive.namelist())
    if MANIFEST_NAME not in names:
        raise ValueError("Not a valid memory export (missing manifest.json).")
    try:
        manifest = json.loads(archive.read(MANIFEST_NAME).decode("utf-8"))
    except (ValueError, UnicodeDecodeError) as exc:
        raise ValueError("Memory export manifest is unreadable.") from exc
    if manifest.get("version") != EXPORT_FORMAT_VERSION:
        raise ValueError(
            f"Unsupported memory export version: {manifest.get('version')!r}."
        )

    targets = memory_paths(config, user_id)
    payloads: Dict[str, bytes] = {}
    for kind, name in (manifest.get("files") or {}).items():
        if kind not in MEMORY_KINDS:
            raise ValueError(f"Memory export contains unknown section: {kind!r}.")
        if name not in names or "/" in name or "\\" in name or name.startswith("."):
            raise ValueError(f"Memory export file entry is invalid: {name!r}.")
        if kind not in targets:
            # This install doesn't persist that kind (path unset); skip it.
            continue
        if (targets[kind].suffix == ".db") != name.endswith(".db"):
            # Restoring JSON bytes into a live .db (or vice versa) would be
            # quarantined as corruption on the next load — refuse loudly.
            raise ValueError(
                f"Memory export section {kind!r} was made under a different "
                "storage backend; convert it with "
                "scripts/convert_memory_sqlite.py first."
            )
        raw = archive.read(name)
        if name.endswith(".db"):
            # SQLite sections can't be JSON-validated; check the file magic
            # so a mangled export still fails before overwriting anything.
            if not raw.startswith(_SQLITE_MAGIC):
                raise ValueError(
                    f"Memory export section {kind!r} is not a SQLite database."
                )
        else:
            try:
                json.loads(raw.decode("utf-8"))
            except (ValueError, UnicodeDecodeError) as exc:
                raise ValueError(
                    f"Memory export section {kind!r} is not valid JSON."
                ) from exc
        payloads[kind] = raw
    if not payloads:
        raise ValueError("Memory export contains no importable sections.")

    # Everything validated: protect what's there now, then swap in the new
    # files atomically (tmp + os.replace per file).
    create_backup(config, user_id, reason="pre-import")
    restored: List[str] = []
    for kind, raw in payloads.items():
        target = targets[kind]
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_name(target.name + ".tmp")
        tmp.write_bytes(raw)
        os.replace(tmp, target)
        if target.suffix == ".db":
            # The imported database is self-contained (exports are taken
            # after a checkpoint); stale WAL sidecars from the replaced db
            # would corrupt it on the next open.
            for ext in _SQLITE_SIDECAR_EXTS:
                try:
                    os.remove(target.with_name(target.name + ext))
                except OSError:
                    pass
        restored.append(kind)
    return sorted(restored)
