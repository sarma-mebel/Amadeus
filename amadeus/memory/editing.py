"""User-initiated memory edits: forget / edit / pin / relationship reset.

Line-budget extraction from manager.py (same free-function-over-owner
pattern as recall.py / introspection.py / fact_archive.py): every function
takes the owning :class:`MemoryManager` as its first argument and the
manager keeps one-line delegates, so the public API is unchanged. These are
the levers a USER pulls (CLI commands, memory-panel actions) as opposed to
the model-facing tool surface in fact_archive.py — they share its archive
helpers but own the match/protect/persist choreography.
"""

from __future__ import annotations

import datetime as dt
import logging
from typing import TYPE_CHECKING, Optional, Union

from . import fact_archive
from .vector_store import VectorRecord

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .manager import MemoryManager, _Protected

LOGGER = logging.getLogger(__name__)


def forget(manager: "MemoryManager", query: str) -> Union[str, "_Protected", None]:
    """User-directed "forget that": remove the best-matching memory.

    Returns the forgotten text, ``PROTECTED`` when the best vector match
    is pinned (never falls through to the profile-facts scan in that
    case — a pinned record blocking the match must not cause an
    unrelated profile fact to be deleted instead), or ``None`` when
    nothing matched at all. Checks the vector store first, then profile
    facts by substring.
    """
    from .manager import PROTECTED

    query = (query or "").strip()
    if not query:
        return None
    # Lookup via the UNFILTERED find_facts path (like the model-facing
    # memory tool), not conversational recall: recall's fixation guard
    # drops facts whose content words appear in the recent short-term
    # window, and "forget what I said about X" almost always follows
    # talk about X — so the best match was filtered and forget silently
    # missed. Recall's relevance floor is kept, so a nonsense query
    # still can't delete the nearest unrelated fact.
    min_score = max(
        float(getattr(manager.config.memory, "recall_min_score", 0.0) or 0.0),
        float(getattr(manager.embedder, "relevance_floor", 0.0) or 0.0),
    )
    matches = manager.find_facts(query, k=1)
    if matches and matches[0]["score"] >= min_score:
        if matches[0]["pinned"]:
            return PROTECTED
        with manager._state_lock:
            # Re-find under the mutation lock (TOCTOU: the record could
            # have been pinned/deleted since the lookup above).
            record = manager._find_record_locked(matches[0]["id"])
            removed: Optional[str] = None
            if record is not None:
                if manager._is_protected(record.metadata):
                    return PROTECTED
                manager._archive_and_delete_locked(
                    record.id, record.text, record.metadata, "user_forget"
                )
                manager.vector_store.persist()
                removed = record.text
        if removed is not None:
            manager._persist()
            return removed
    if manager.profile is not None:
        needle = query.lower()
        with manager._state_lock:
            for key, value in list(manager.profile.facts.items()):
                if needle in key.lower() or needle in value.lower():
                    manager.profile.facts.pop(key)
                    manager._save_profile()
                    manager.profile_block_dirty = True
                    return f"{key}: {value}"
    return None


def set_pinned(manager: "MemoryManager", record_id: str, pinned: bool) -> bool:
    """Pin/unpin a memory record by id. Returns False if not found."""
    if not manager.vector_store:
        return False
    with manager._state_lock:
        record = manager._find_record_locked(record_id)
        if record is None:
            return False
        updated = dict(record.metadata or {})
        updated["pinned"] = bool(pinned)
        manager.vector_store.update_metadata(record_id, updated)
        manager.vector_store.persist()
    manager._persist()
    return True


def edit_memory(manager: "MemoryManager", record_id: str, text: str) -> bool:
    """Edit a memory record's text in place, re-embedding it."""
    if not (manager.vector_store and manager.embedder):
        return False
    text = (text or "").strip()
    if not text:
        return False
    # Embed BEFORE the lock: this is the only slow (LLM/model) step, and
    # holding _state_lock across it would block every other memory
    # operation on the same manager for the embedding call's duration.
    vector = manager.embedder.embed_text(text)
    with manager._state_lock:
        # Re-find under the same lock that performs the mutation, so a
        # concurrent delete/edit between "find" and "mutate" can't race
        # us into resurrecting a record that no longer exists.
        record = manager._find_record_locked(record_id)
        if record is None:
            return False
        updated_metadata = dict(record.metadata or {})
        updated_metadata["edited_at"] = dt.datetime.utcnow().isoformat()
        # Provenance for the pre-edit text (id is preserved, so this is a
        # rewrite rather than a removal — archived best-effort so "what
        # did she used to think" stays answerable, never blocking the edit).
        fact_archive.archive_record(
            manager,
            record_id=record_id,
            text=record.text,
            metadata=record.metadata,
            reason="user_correct",
        )
        manager.vector_store.delete(record_id)
        manager.vector_store.add(
            VectorRecord(
                id=record_id,
                vector=list(vector),
                text=text,
                metadata=updated_metadata,
            )
        )
        manager.vector_store.persist()
    manager._persist()
    return True


def forget_id(
    manager: "MemoryManager", record_id: str
) -> Union[str, "_Protected", None]:
    """Delete a specific memory record by id.

    Returns the forgotten text, ``PROTECTED`` when the targeted record is
    pinned (pin-then-delete is the intended UI flow — deleting a pinned
    record by id must not silently succeed), or ``None`` when not found.
    """
    from .manager import PROTECTED

    if not manager.vector_store:
        return None
    with manager._state_lock:
        record = manager._find_record_locked(record_id)
        if record is None:
            return None
        if manager._is_protected(record.metadata):
            return PROTECTED
        manager._archive_and_delete_locked(
            record_id, record.text, record.metadata, "user_forget"
        )
        manager.vector_store.persist()
    manager._persist()
    return record.text


def forget_fact(manager: "MemoryManager", key: str) -> bool:
    """Delete one profile fact by its exact key. Returns True if removed."""
    if manager.profile is None:
        return False
    with manager._state_lock:
        if key in manager.profile.facts:
            manager.profile.facts.pop(key)
            manager._save_profile()
            manager.profile_block_dirty = True
            return True
    return False


def forget_open_thread(manager: "MemoryManager", text: str) -> bool:
    """Drop one open thread by its exact text. Returns True if removed.

    Open threads drive the "let's continue X" framing in greetings and
    context, so this is the lever a user pulls to make Kurisu stop
    circling a finished topic.
    """
    if manager.profile is None:
        return False
    with manager._state_lock:
        threads = manager.profile.open_threads
        for i, entry in enumerate(threads):
            entry_text = entry.get("text") if isinstance(entry, dict) else entry
            if entry_text == text:
                threads.pop(i)
                manager._save_profile()
                manager.profile_block_dirty = True
                return True
    return False


def reset_relationship(manager: "MemoryManager") -> bool:
    """User-directed reset of the long-arc dials and tone history.

    Trust level is left untouched — it has its own floor semantics and
    isn't part of what this action promises to clear. Returns False
    without a profile.
    """
    if manager.profile is None:
        return False
    with manager._state_lock:
        manager.profile.arc_curiosity_vs_annoyance = 0.0
        manager.profile.arc_logic_vs_emotion = 0.0
        manager.profile.arc_pride_vs_vulnerability = 0.0
        manager.profile.tone_history = []
        manager.profile.last_felt = None
        manager._save_profile()
        manager.profile_block_dirty = True
    return True
