"""Opt-in SQLite persistence backend for conversation memory.

``memory.storage_backend: "sqlite"`` swaps the two rewrite-whole-file JSON
stores for SQLite databases at the same derived paths with the suffix
changed to ``.db``: the message/summary state store (``memory_store.db``)
and the vector store (``memory_embeddings.db``). Per-user
(``<stem>_users/``) and per-character (``characters/<id>/``) path scoping
is untouched — only the file format changes.

Design notes:

- stdlib ``sqlite3`` only, no new dependencies.
- WAL journal mode, so concurrent readers (the scheduler's disk-snapshot
  jobs) never block on — or get a torn view from — the app's writes.
- Writes are single transactions replacing the JSON stores' atomic
  tmp+``os.replace`` pattern: a crash mid-write leaves the previous
  committed state, never a half-written store.
- Connections are opened per operation and closed immediately. SQLite
  checkpoints and removes the ``-wal``/``-shm`` sidecars when the last
  connection closes cleanly, so callers that unlink the ``.db`` file
  (``MemoryManager.clear``) keep working without special-casing.
- A corrupt database follows the JSON stores' quarantine convention:
  moved aside to ``<name>.bak-corrupt``, ``memory.sqlite_load_corrupt``
  incremented, fresh start — never silently overwritten, never a crash.
"""

from __future__ import annotations

import json
import os
import sqlite3
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .. import telemetry
from ..logging_utils import get_logger
from .vector_store import InMemoryVectorStore, VectorRecord

LOGGER = get_logger(__name__)

SCHEMA_VERSION = 1

# State database: the short-term message window as rows, everything else
# (summary, summary_metadata, legacy inline long_term) as JSON docs in
# ``meta``. ``graph_doc`` holds the entity graph (memory/graph.py) as one
# JSON document — transactional and colocated with the state it describes.
STATE_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS messages (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    timestamp TEXT
);
CREATE TABLE IF NOT EXISTS graph_doc (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    payload TEXT NOT NULL
);
"""

# Vector database: one row per record, vector/metadata JSON-encoded (simple
# beats clever at these store sizes — a few thousand rows at most).
VECTOR_SCHEMA = """
CREATE TABLE IF NOT EXISTS vectors (
    id TEXT PRIMARY KEY,
    vector TEXT NOT NULL,
    text TEXT NOT NULL,
    metadata TEXT NOT NULL
);
"""

# Keys of the state payload persisted as JSON docs in ``meta`` (everything
# except "messages", which gets real rows).
_META_KEYS = ("summary", "summary_metadata", "long_term")


def db_path(path: Optional[Union[str, Path]]) -> Optional[Path]:
    """The SQLite variant of a configured store path: same directory and
    stem, ``.db`` suffix — so per-user and per-character scoping (which act
    on directories and stems) behave identically."""
    if path is None:
        return None
    path = Path(path)
    if path.suffix == ".db":
        return path
    return path.with_name(path.stem + ".db")


def _quarantine(path: Path, exc: BaseException) -> None:
    """Corruption = potential loss of memory; keep the evidence, count it."""
    telemetry.incr("memory.sqlite_load_corrupt")
    backup = path.with_name(path.name + ".bak-corrupt")
    try:
        os.replace(path, backup)
        for sidecar in (f"{path}-wal", f"{path}-shm"):
            try:
                os.remove(sidecar)
            except OSError:
                pass
        LOGGER.error(
            "SQLite store %s is corrupt (%s); starting fresh. "
            "The unreadable file was kept at %s",
            path,
            exc,
            backup,
        )
    except OSError:
        LOGGER.error(
            "SQLite store %s is corrupt (%s) and could not be backed up; "
            "starting fresh",
            path,
            exc,
        )


def _connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), timeout=5.0)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


def open_db(path: Path, schema: str) -> Optional[sqlite3.Connection]:
    """Connect and ensure the schema. A corrupt database is quarantined to
    ``.bak-corrupt`` (+ ``memory.sqlite_load_corrupt``) and recreated fresh;
    returns None only when even a fresh database cannot be opened."""
    for attempt in (0, 1):
        conn: Optional[sqlite3.Connection] = None
        try:
            conn = _connect(path)
            conn.executescript(schema)
            if "CREATE TABLE IF NOT EXISTS meta" in schema:
                conn.execute(
                    "INSERT OR IGNORE INTO meta (key, value) VALUES (?, ?)",
                    ("schema_version", str(SCHEMA_VERSION)),
                )
            conn.commit()
            return conn
        except sqlite3.DatabaseError as exc:
            if conn is not None:
                try:
                    conn.close()
                except sqlite3.Error:  # pragma: no cover - close best effort
                    pass
            if attempt == 0 and path.exists():
                _quarantine(path, exc)
                continue
            LOGGER.error("Could not open SQLite store %s: %s", path, exc)
            telemetry.incr("memory.sqlite_open_failed")
            return None
    return None  # pragma: no cover - loop always returns


def save_state(path: Path, payload: Dict[str, Any]) -> bool:
    """Persist the manager's serialized state dict in one transaction.

    Accepts exactly the dict shape ``MemoryManager._persist`` builds for the
    JSON file, so the two backends stay drop-in interchangeable.
    """
    conn = open_db(path, STATE_SCHEMA)
    if conn is None:
        telemetry.incr("memory.sqlite_save_failed")
        return False
    try:
        with conn:
            conn.execute("DELETE FROM messages")
            conn.executemany(
                "INSERT INTO messages (role, content, timestamp) VALUES (?, ?, ?)",
                [
                    (
                        str(msg.get("role", "")),
                        str(msg.get("content", "")),
                        msg.get("timestamp"),
                    )
                    for msg in payload.get("messages", [])
                ],
            )
            for key in _META_KEYS:
                if key in payload and payload[key] is not None:
                    conn.execute(
                        "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
                        (key, json.dumps(payload[key], ensure_ascii=False)),
                    )
                else:
                    conn.execute("DELETE FROM meta WHERE key = ?", (key,))
        return True
    except sqlite3.Error as exc:
        LOGGER.warning("SQLite state save failed for %s: %s", path, exc)
        telemetry.incr("memory.sqlite_save_failed")
        return False
    finally:
        conn.close()


def load_state(path: Path) -> Optional[Dict[str, Any]]:
    """Load the state dict back in the JSON store's exact shape, or None
    when the database is missing/corrupt/unreadable (corruption is
    quarantined via :func:`open_db`)."""
    if not path.exists():
        return None
    conn = open_db(path, STATE_SCHEMA)
    if conn is None:
        return None
    try:
        data: Dict[str, Any] = {
            "messages": [
                {"role": role, "content": content, "timestamp": timestamp}
                for role, content, timestamp in conn.execute(
                    "SELECT role, content, timestamp FROM messages ORDER BY seq"
                )
            ]
        }
        for key, value in conn.execute(
            "SELECT key, value FROM meta WHERE key IN (?, ?, ?)", _META_KEYS
        ):
            try:
                data[key] = json.loads(value)
            except json.JSONDecodeError:
                telemetry.incr("memory.sqlite_row_invalid")
        return data
    except sqlite3.DatabaseError as exc:
        conn.close()
        conn = None
        _quarantine(path, exc)
        return None
    finally:
        if conn is not None:
            conn.close()


def save_graph_doc(path: Path, payload: Dict[str, Any]) -> bool:
    """Persist the entity graph as a single JSON document, transactionally."""
    conn = open_db(path, STATE_SCHEMA)
    if conn is None:
        telemetry.incr("memory.sqlite_save_failed")
        return False
    try:
        with conn:
            conn.execute(
                "INSERT OR REPLACE INTO graph_doc (id, payload) VALUES (1, ?)",
                (json.dumps(payload, ensure_ascii=False),),
            )
        return True
    except sqlite3.Error as exc:
        LOGGER.warning("SQLite graph save failed for %s: %s", path, exc)
        telemetry.incr("memory.sqlite_save_failed")
        return False
    finally:
        conn.close()


def load_graph_doc(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    conn = open_db(path, STATE_SCHEMA)
    if conn is None:
        return None
    try:
        row = conn.execute("SELECT payload FROM graph_doc WHERE id = 1").fetchone()
        if row is None:
            return None
        try:
            data = json.loads(row[0])
        except json.JSONDecodeError:
            telemetry.incr("memory.sqlite_row_invalid")
            return None
        return data if isinstance(data, dict) else None
    except sqlite3.DatabaseError as exc:
        conn.close()
        conn = None
        _quarantine(path, exc)
        return None
    finally:
        if conn is not None:
            conn.close()


def save_vectors(path: Path, records: List[Dict[str, Any]]) -> bool:
    """Replace the vector table with ``records`` (dicts in the JSON store's
    row shape) in one transaction. Shared by the store and the converter."""
    conn = open_db(path, VECTOR_SCHEMA)
    if conn is None:
        telemetry.incr("memory.sqlite_save_failed")
        return False
    try:
        with conn:
            conn.execute("DELETE FROM vectors")
            conn.executemany(
                "INSERT INTO vectors (id, vector, text, metadata) "
                "VALUES (?, ?, ?, ?)",
                [
                    (
                        str(rec.get("id") or uuid.uuid4().hex),
                        json.dumps(rec.get("vector") or []),
                        str(rec.get("text") or ""),
                        json.dumps(rec.get("metadata") or {}, ensure_ascii=False),
                    )
                    for rec in records
                ],
            )
        return True
    except sqlite3.Error as exc:
        LOGGER.warning("SQLite vector save failed for %s: %s", path, exc)
        telemetry.incr("memory.sqlite_save_failed")
        return False
    finally:
        conn.close()


class SqliteVectorStore(InMemoryVectorStore):
    """The default vector store with its persistence swapped to SQLite.

    Records live in memory exactly like :class:`InMemoryVectorStore` (same
    similarity search, same semantics); only ``_load``/``persist`` differ,
    so recall behavior is byte-identical across backends.
    """

    def __init__(self, path: Optional[Path] = None) -> None:
        self.path = db_path(Path(path)) if path else None
        self._records: List[VectorRecord] = []
        if self.path and self.path.exists():
            self._load()

    def persist(self) -> None:
        if not self.path:
            return
        save_vectors(
            self.path,
            [
                {
                    "id": rec.id,
                    "vector": rec.vector,
                    "text": rec.text,
                    "metadata": rec.metadata,
                }
                for rec in self._records
            ],
        )

    def _load(self) -> None:
        self._records = []
        conn = open_db(self.path, VECTOR_SCHEMA)
        if conn is None:
            return
        try:
            for record_id, vector, text, metadata in conn.execute(
                "SELECT id, vector, text, metadata FROM vectors"
            ):
                try:
                    vector_list = [float(x) for x in json.loads(vector)]
                    metadata_dict = json.loads(metadata) or {}
                except (json.JSONDecodeError, TypeError, ValueError):
                    telemetry.incr("memory.sqlite_row_invalid")
                    continue
                self._records.append(
                    VectorRecord(
                        vector=vector_list,
                        text=str(text),
                        metadata=metadata_dict,
                        id=str(record_id or uuid.uuid4().hex),
                    )
                )
        except sqlite3.DatabaseError as exc:
            conn.close()
            conn = None
            _quarantine(self.path, exc)
            self._records = []
        finally:
            if conn is not None:
                conn.close()
