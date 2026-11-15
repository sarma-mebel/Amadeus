"""Fact store lifecycle: archive sidecar + the fact write/lookup paths.

Supersede and forget used to be hard deletes — the record vanished from the
vector store with no trace, and a superseding fact's ``metadata["supersedes"]``
pointed at an id that no longer resolved to anything. This module gives every
removal path a paper trail: before a fact record leaves the hot store it is
appended as one JSON line to a sidecar file next to the vector store, so
"what did she used to think before I corrected her" stays answerable. It also
owns the write/lookup side of the fact lifecycle (``store_fact``,
``find_facts``, ``archive_fact``, ``store_user_fact``) — grouped here rather
than left on ``MemoryManager`` for the same line-budget reason as
``recall.py``/``extraction_jobs.py``.

Free functions taking the ``MemoryManager`` as first argument — the same
sibling-module convention as ``introspection.py``/``context.py``. The archive
path rides the same per-user/per-character scoping as the vector store
automatically: it is derived from whichever resolved path the manager already
uses to persist vectors (works identically for the sqlite backend, which is
cold storage for this file — it never reads it, only ever appends).

Archive writes are best-effort provenance, never a gate: a write failure logs
a warning, counts ``memory.fact_archive_write_failed``, and the caller
proceeds with the removal regardless (see ``archive_record``'s docstring).
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from .. import telemetry
from ..logging_utils import get_logger
from .text_utils import _normalize_event_time
from .vector_store import VectorRecord

if TYPE_CHECKING:  # pragma: no cover - import cycle guard
    from .manager import MemoryManager

LOGGER = get_logger(__name__)


def archive_path(manager: "MemoryManager") -> Optional[Path]:
    """Resolve the JSONL sidecar path for this manager's fact archive.

    Derived from whichever resolved path the manager persists vectors under
    (``embeddings_path`` first — the file path used by both the JSON and
    SQLite vector backends — falling back to ``vector_path`` for a bare
    Chroma-only setup, then ``persist_path`` as a last resort), so per-user
    and per-character scoping (already baked into those paths at
    ``MemoryManager.__init__``) carries over automatically. Returns ``None``
    when the manager has no persistence configured at all (in-memory only —
    nothing to archive to).
    """
    base = manager.embeddings_path or manager.vector_path or manager.persist_path
    if not base:
        return None
    base = Path(base)
    return base.with_name(f"{base.stem}_archive.jsonl")


def archive_record(
    manager: "MemoryManager",
    *,
    record_id: Optional[str],
    text: str,
    metadata: Optional[Dict[str, Any]],
    reason: str,
) -> None:
    """Append one archived-fact line. Best-effort: never raises, never
    blocks the removal it accompanies.

    ``reason`` is one of "superseded_duplicate", "superseded_hint",
    "evicted_cap", "user_forget", "user_correct".
    """
    path = archive_path(manager)
    if path is None:
        return
    entry = {
        "id": record_id,
        "text": text,
        "metadata": metadata or {},
        "archived_at": dt.datetime.utcnow().isoformat(),
        "reason": reason,
    }
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except OSError as exc:
        LOGGER.warning("Failed to write fact archive entry to %s: %s", path, exc)
        telemetry.incr("memory.fact_archive_write_failed")
        return
    telemetry.incr("memory.fact_archived")


def archive_and_delete_locked(
    manager: "MemoryManager",
    record_id: Optional[str],
    text: str,
    metadata: Optional[Dict[str, object]],
    reason: str,
) -> None:
    """Archive a record (best-effort provenance) then remove it from the
    hot vector store. Caller MUST hold ``manager._state_lock`` and is
    responsible for ``vector_store.persist()`` afterward — this only
    mutates the in-memory store, matching every call site's existing shape.
    """
    archive_record(
        manager, record_id=record_id, text=text, metadata=metadata, reason=reason
    )
    if record_id and manager.vector_store:
        manager.vector_store.delete(record_id)


def store_fact(
    manager: "MemoryManager",
    text: str,
    salience: float,
    *,
    pinned: bool = False,
    replaces: Optional[str] = None,
    source: Optional[str] = None,
    record_type: str = "fact",
    valid_at: Optional[str] = None,
) -> bool:
    """Store an extracted fact, superseding a near-duplicate/contradiction.

    Two independent supersede paths: the existing near-duplicate scan
    (nearest neighbor to the NEW fact's own text, high threshold — catches
    near-identical restatements), and an optional hint-driven scan when the
    extractor names an earlier fact this one contradicts/updates
    (paraphrased contradictions sit well below the near-duplicate threshold
    on real embeddings, e.g. "loves Nolan" vs "bored of Nolan now" — the hint
    lets us dedup those too, without loosening the threshold for everything
    else). Both paths ARCHIVE the superseded record instead of hard-deleting
    it, and the new record's metadata carries both the archived record's id
    (``supersedes``, now resolvable against the archive instead of dangling)
    and a truncated copy of its text (``previous_fact`` — same provenance
    convention as the entity graph, see graph.py). Returns True on success,
    False (best-effort, never raises) if the embed or store write failed.

    ``record_type`` defaults to ``"fact"`` (the extraction/user-explicit
    path). A non-"fact" caller (moments.py stores ``"moment"``) skips BOTH
    supersede scans entirely — a shared-activity memory is a standalone
    event, not a correction to or duplicate of a user-stated fact, and must
    never be treated as one for supersede-hint purposes; the candidate scans
    below are also scoped to ``type in (None, "fact")`` so an existing
    moment can never be superseded by an unrelated fact either.
    """
    if not (manager.embedder and manager.vector_store):
        return False
    max_chars = int(getattr(manager.config.memory, "fact_max_chars", 0) or 0)
    if max_chars and len(text) > max_chars:
        text = text[:max_chars]
        telemetry.incr("memory.fact_truncated")
    threshold = float(
        getattr(manager.config.memory, "fact_supersede_threshold", 0.86) or 0.86
    )
    hint_threshold = float(
        getattr(manager.config.memory, "fact_supersede_hint_threshold", 0.75) or 0.75
    )
    try:
        vector = manager.embedder.embed_text(text)
    except Exception as exc:  # pragma: no cover - embedder runtime errors
        LOGGER.warning("Fact embedding failed: %s", exc)
        telemetry.incr("memory.store_fact_embed_failed")
        return False
    with manager._state_lock:
        superseded_id: Optional[str] = None
        superseded_text: Optional[str] = None
        if record_type == "fact" and manager.vector_store.count():
            # Walk candidates in similarity order; pinned records are never
            # superseded (skip, keep looking) but still block nothing else —
            # the first unpinned candidate above the threshold is the one we
            # dedup against, then we stop.
            for nearest in manager.vector_store.similarity_search(vector, limit=3):
                metadata = nearest.metadata or {}
                if metadata.get("type") not in (None, "fact"):
                    continue
                if manager._is_protected(metadata):
                    continue
                if nearest.score >= threshold and nearest.id:
                    superseded_id = nearest.id
                    superseded_text = nearest.text
                    archive_and_delete_locked(
                        manager,
                        nearest.id,
                        nearest.text,
                        metadata,
                        "superseded_duplicate",
                    )
                break
        if record_type == "fact" and not superseded_id and replaces:
            try:
                replaces_vector = manager.embedder.embed_text(replaces)
            except Exception:  # pragma: no cover - embedder best effort
                replaces_vector = None
            if replaces_vector is not None and manager.vector_store.count():
                for nearest in manager.vector_store.similarity_search(
                    replaces_vector, limit=3
                ):
                    metadata = nearest.metadata or {}
                    if metadata.get("type") not in (None, "fact"):
                        continue
                    if manager._is_protected(metadata):
                        continue
                    if nearest.score >= hint_threshold and nearest.id:
                        superseded_id = nearest.id
                        superseded_text = nearest.text
                        LOGGER.info(
                            "Superseding fact %r (hint %r matched at score >= %.2f)",
                            nearest.text,
                            replaces,
                            hint_threshold,
                        )
                        archive_and_delete_locked(
                            manager,
                            nearest.id,
                            nearest.text,
                            metadata,
                            "superseded_hint",
                        )
                        telemetry.incr("memory.fact_superseded_by_hint")
                        telemetry.incr("memory.hint_supersedes")
                    break
        metadata: Dict[str, object] = {
            "type": record_type,
            "timestamp": dt.datetime.utcnow().isoformat(),
            "salience": float(salience),
            "strength": 1.0,
            "pinned": bool(pinned),
        }
        # Bitemporal split: ``timestamp`` stays learn time (all decay/sort/
        # cap math reads it, unchanged), ``valid_at`` is the optional EVENT
        # time — flat naive-UTC ISO string (Chroma metadata must be scalar),
        # consumed only by the recall renderer and introspection payloads.
        if valid_at:
            normalized = _normalize_event_time(valid_at)
            if normalized is not None:
                metadata["valid_at"] = normalized
            else:
                telemetry.incr("memory.valid_at_rejected")
        if source:
            metadata["source"] = source
        if superseded_id:
            metadata["supersedes"] = superseded_id
        if superseded_text:
            metadata["previous_fact"] = superseded_text[:200]
        manager.vector_store.add(
            VectorRecord(vector=list(vector), text=text, metadata=metadata)
        )
        manager.vector_store.persist()
        manager._vector_dirty = False
    return True


def find_facts(manager: "MemoryManager", text: str, k: int = 3) -> List[Dict[str, Any]]:
    """Look up fact/moment records relevant to ``text`` — the model-facing
    memory tool's lookup primitive, also `forget`'s query path (moments are
    admitted so "forget that we watched X" works — they inherit forget like
    any fact; see relational.store_moment). (Safe to call mid-turn: uses the same
    embed-then-search shape as recall, under ``_state_lock`` only for the
    search itself). Returns up to ``k`` dicts
    ``{"id", "text", "score", "pinned"}`` sorted by score descending; an
    empty list on empty input, no vector store, or an embed failure
    (counted, never raised).
    """
    text = (text or "").strip()
    if not text or not (manager.embedder and manager.vector_store):
        return []
    try:
        vector = manager.embedder.embed_text(text)
    except Exception as exc:  # pragma: no cover - embedder runtime errors
        LOGGER.warning("find_facts embedding failed: %s", exc)
        telemetry.incr("memory.find_facts_embed_failed")
        return []
    with manager._state_lock:
        matches = manager.vector_store.similarity_search(vector, max(k, 1) * 4)
    results: List[Dict[str, Any]] = []
    for match in matches:
        metadata = match.metadata or {}
        if metadata.get("type") not in (None, "fact", "moment"):
            continue
        if not match.id:
            continue
        results.append(
            {
                "id": match.id,
                "text": match.text,
                "score": match.score,
                "pinned": manager._is_protected(metadata),
            }
        )
    results.sort(key=lambda item: item["score"], reverse=True)
    return results[:k]


def archive_fact(
    manager: "MemoryManager",
    record_id: str,
    reason: str,
    allow_pinned: bool = False,
) -> Optional[str]:
    """Archive-then-remove a fact (or any vector-store record) by id.

    Public contract counterpart of ``forget_id`` for callers (the
    model-facing memory tool) that already know the id and want a plain
    text-or-None result without the ``PROTECTED`` sentinel. Pinned records
    are refused by default (returns None; archive_fact is not the
    pin-then-delete UI flow forget_id is) — ``allow_pinned=True`` archives
    them anyway and exists ONLY for direct, explicit user instructions
    ("forget that" via the memory tool: explicit forget beats pins).
    Automatic paths (cap eviction, supersede hints, consolidation) must
    never pass it. Returns the archived text, or None when the id doesn't
    exist or is pinned and ``allow_pinned`` is False.
    """
    if not manager.vector_store or not record_id:
        return None
    with manager._state_lock:
        record = manager._find_record_locked(record_id)
        if record is None:
            return None
        if manager._is_protected(record.metadata) and not allow_pinned:
            return None
        text = record.text
        archive_and_delete_locked(manager, record_id, text, record.metadata, reason)
        manager.vector_store.persist()
    manager._persist()
    return text


def store_user_fact(
    manager: "MemoryManager",
    text: str,
    salience: float = 0.9,
    previous_fact: Optional[str] = None,
    valid_at: Optional[str] = None,
) -> bool:
    """Explicit user-fact store path for the model-facing memory tool.

    Applies the same ``fact_max_chars`` truncation and supersede machinery
    as extracted facts (via ``store_fact``), tags the record ``source:
    "user_explicit"`` so it's distinguishable from extraction-derived facts,
    and — when ``previous_fact`` is given — threads it through as the
    hint-driven supersede's ``replaces`` text (same provenance convention:
    the new record gets ``previous_fact``/``supersedes`` metadata). Returns
    False (counted) on embed/store failure; never raises.
    """
    text = (text or "").strip()
    if not text or not (
        manager.config.memory.enable_long_term
        and manager.embedder
        and manager.vector_store
    ):
        telemetry.incr("memory.store_user_fact_failed")
        return False
    ok = store_fact(
        manager,
        text,
        salience=salience,
        replaces=(previous_fact or "").strip() or None,
        source="user_explicit",
        valid_at=valid_at,
    )
    if not ok:
        telemetry.incr("memory.store_user_fact_failed")
    return ok
