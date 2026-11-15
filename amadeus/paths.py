"""Platform data-directory resolution for user-data files.

Memory stores, profiles, transcripts, and reminders historically defaulted to
bare filenames, which resolve against the process CWD — running from a
different directory silently split (or appeared to lose) a user's memory.
Relative user-data paths now anchor to a per-platform data directory instead;
absolute paths (e.g. the desktop build's explicit overrides) pass through
untouched.
"""

from __future__ import annotations

import logging
import os
import shutil
import sys
from pathlib import Path
from typing import Optional, Union

_LOGGER = logging.getLogger(__name__)

# Set by the test suite (and available to embedders) to disable the one-time
# CWD migration — moving real files out of the working directory is an
# app-startup behavior, never a side effect of merely loading a config.
NO_MIGRATION_ENV = "AMADEUS_NO_LEGACY_MIGRATION"

DATA_DIR_ENV = "AMADEUS_DATA_DIR"

# User-data files that used to live in the CWD and are eligible for the
# one-time migration into the data dir. Per-user variants live in a sibling
# "<stem>_users" directory and migrate together with their base file.
_LEGACY_USER_DATA_NAMES = (
    "memory_store.json",
    "memory_embeddings.json",
    "memory_profile.json",
    "memory_transcripts.json",
    "memory_vectors",
    "reminders.json",
)


def default_data_dir() -> Path:
    """Per-platform user-data directory (override with AMADEUS_DATA_DIR)."""

    override = os.getenv(DATA_DIR_ENV)
    if override:
        return Path(override).expanduser()
    if sys.platform == "win32":
        base = os.getenv("LOCALAPPDATA")
        root = Path(base) if base else Path.home() / "AppData" / "Local"
        return root / "Amadeus"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "Amadeus"
    base = os.getenv("XDG_DATA_HOME")
    root = Path(base) if base else Path.home() / ".local" / "share"
    return root / "amadeus"


def character_scoped_path(
    value: Optional[Union[str, Path]], character_id: Optional[str]
) -> Optional[str]:
    """Re-parent a memory-file path into a per-character subdirectory.

    ``<dir>/memory_store.json`` + ``"rem"`` -> ``<dir>/characters/rem/
    memory_store.json``. The builtin/default character passes
    ``character_id=None`` and gets the path back untouched — the legacy
    store location is never moved or renamed. Idempotent: a path already
    scoped to this character is returned as-is (pure, no I/O).
    """

    if not value or not character_id:
        return value if value is None else str(value)
    path = Path(value)
    if len(path.parts) >= 3 and path.parts[-3:-1] == ("characters", character_id):
        return str(path)
    return str(path.parent / "characters" / character_id / path.name)


def resolve_user_data_path(
    value: Optional[Union[str, Path]], data_dir: Optional[Path] = None
) -> Optional[str]:
    """Anchor a relative user-data path to the data dir (pure, no I/O).

    None/empty and absolute paths pass through unchanged.
    """

    if not value:
        return value if value is None else str(value)
    raw = Path(value).expanduser()
    if raw.is_absolute():
        return str(raw)
    return str((data_dir or default_data_dir()) / raw)


def migrate_legacy_cwd_files(config) -> None:
    """One-time move of legacy CWD user-data files into the data dir.

    Only touches the well-known default filenames, and only when the config
    points at the default data-dir location and no file exists there yet —
    explicitly configured paths never trigger a move. Called from app
    entrypoints (CLI / web startup), deliberately not from Config.load.
    """

    if os.getenv(NO_MIGRATION_ENV):
        return
    data_dir = default_data_dir()
    configured = {
        str(p)
        for p in (
            config.memory.persist_path,
            config.memory.embeddings_path,
            config.memory.profile_path,
            config.memory.transcripts_path,
            config.memory.vector_path,
            config.reminders.store_path,
        )
        if p
    }
    cwd = Path.cwd()
    for name in _LEGACY_USER_DATA_NAMES:
        target = data_dir / name
        if str(target) not in configured:
            continue
        _move_if_present(cwd / name, target)
        legacy_users = cwd / f"{Path(name).stem}_users"
        target_users = data_dir / f"{Path(name).stem}_users"
        _move_if_present(legacy_users, target_users)


def _move_if_present(legacy: Path, target: Path) -> None:
    try:
        if not legacy.exists() or target.exists():
            return
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(legacy), str(target))
        _LOGGER.warning("Migrated user data: %s -> %s", legacy, target)
    except OSError as exc:  # never let migration break startup
        _LOGGER.warning("Could not migrate %s to %s: %s", legacy, target, exc)
