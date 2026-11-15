"""On-disk per-session chat transcripts backing the web sessions API.

In-memory ``WebChatSession`` state is TTL-evicted and lost on restart; the
transcripts here are the durable record the sessions UI lists and reopens.
Storage follows the existing per-user memory-file convention: one JSON file
per user under ``<stem>_users/<user_id><suffix>`` next to the configured base
path (``memory.transcripts_path``), or the base file itself for anonymous
(no ``user_id``) clients.

File shape::

    {"sessions": {"<session_id>": {
        "created_at": <epoch float>,
        "updated_at": <epoch float>,
        "messages": [{"role": "user"|"assistant", "content": str,
                      "ja": str|null, "emotion": str|null,
                      "timestamp": <epoch float>, "turn_id": str|null}, ...]
    }}}

Transcripts persist until explicitly deleted via the DELETE endpoint —
session TTL eviction only drops the in-memory state.
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..logging_utils import get_logger

LOGGER = get_logger(__name__)

_TITLE_MAX_CHARS = 60


class TranscriptStore:
    """Thread-safe JSON-file store for per-session chat transcripts.

    Constructed with the configured base path (or ``None`` to disable, in
    which case every method is a harmless no-op returning empty data).
    """

    def __init__(self, base: Optional[Path]) -> None:
        self.base = base
        self._lock = threading.Lock()

    # -- path resolution (mirrors SessionStore._user_path) ------------------

    def _path_for(self, user_id: Optional[str]) -> Optional[Path]:
        base = self.base
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

    # -- raw file IO ---------------------------------------------------------

    @staticmethod
    def _load(path: Path) -> Dict[str, Any]:
        try:
            with path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
        except FileNotFoundError:
            return {"sessions": {}}
        except (OSError, json.JSONDecodeError) as exc:
            LOGGER.warning("Transcript store unreadable (%s); starting empty", exc)
            return {"sessions": {}}
        if not isinstance(data, dict) or not isinstance(data.get("sessions"), dict):
            return {"sessions": {}}
        return data

    @staticmethod
    def _save(path: Path, data: Dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        # Atomic replace so a crash mid-write can't corrupt the whole store.
        fd, tmp_name = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(data, handle, ensure_ascii=False)
            os.replace(tmp_name, path)
        except BaseException:
            try:
                os.unlink(tmp_name)
            except OSError:
                pass
            raise

    # -- public API ----------------------------------------------------------

    def ensure_session(self, user_id: Optional[str], session_id: str) -> None:
        """Register an (empty) session so it shows up in listings at once."""
        path = self._path_for(user_id)
        if path is None or not session_id:
            return
        with self._lock:
            data = self._load(path)
            if session_id not in data["sessions"]:
                now = time.time()
                data["sessions"][session_id] = {
                    "created_at": now,
                    "updated_at": now,
                    "messages": [],
                }
                self._save(path, data)

    def record_turn(
        self,
        user_id: Optional[str],
        session_id: str,
        user_text: str,
        reply: str,
        *,
        ja: Optional[str] = None,
        emotion: Optional[str] = None,
        turn_id: Optional[str] = None,
    ) -> None:
        """Append one user/assistant exchange to the session transcript."""
        path = self._path_for(user_id)
        if path is None or not session_id:
            return
        now = time.time()
        with self._lock:
            data = self._load(path)
            record = data["sessions"].setdefault(
                session_id, {"created_at": now, "updated_at": now, "messages": []}
            )
            if user_text:
                record["messages"].append(
                    {
                        "role": "user",
                        "content": user_text,
                        "ja": None,
                        "emotion": None,
                        "timestamp": now,
                        "turn_id": turn_id,
                    }
                )
            if reply:
                record["messages"].append(
                    {
                        "role": "assistant",
                        "content": reply,
                        "ja": ja,
                        "emotion": emotion,
                        "timestamp": now,
                        "turn_id": turn_id,
                    }
                )
            record["updated_at"] = now
            self._save(path, data)

    def list_sessions(self, user_id: Optional[str]) -> List[Dict[str, Any]]:
        """Session summaries, newest activity first."""
        path = self._path_for(user_id)
        if path is None:
            return []
        with self._lock:
            data = self._load(path)
        summaries: List[Dict[str, Any]] = []
        for session_id, record in data["sessions"].items():
            messages = record.get("messages") or []
            title = "New conversation"
            for message in messages:
                if message.get("role") == "user" and message.get("content"):
                    title = self._truncate_title(str(message["content"]))
                    break
            summaries.append(
                {
                    "session_id": session_id,
                    "title": title,
                    "created_at": float(record.get("created_at") or 0.0),
                    "updated_at": float(record.get("updated_at") or 0.0),
                    "message_count": len(messages),
                }
            )
        summaries.sort(key=lambda item: item["updated_at"], reverse=True)
        return summaries

    def get_transcript(
        self, user_id: Optional[str], session_id: str
    ) -> Optional[List[Dict[str, Any]]]:
        """Chronological messages for one session, or None if unknown."""
        path = self._path_for(user_id)
        if path is None:
            return None
        with self._lock:
            data = self._load(path)
        record = data["sessions"].get(session_id)
        if record is None:
            return None
        return [
            {
                "role": message.get("role", "assistant"),
                "content": message.get("content", ""),
                "ja": message.get("ja"),
                "emotion": message.get("emotion"),
                "timestamp": float(message.get("timestamp") or 0.0),
                "turn_id": message.get("turn_id"),
            }
            for message in (record.get("messages") or [])
        ]

    def truncate_from_turn(
        self, user_id: Optional[str], session_id: str, turn_id: str
    ) -> Optional[Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]]:
        """Drop every message from ``turn_id``'s user message onward.

        Returns ``(remaining, removed)`` (copies), or ``None`` when the
        session or the turn id is unknown — callers use that to distinguish
        "nothing to truncate" from "truncated to empty". ``removed`` lets
        callers right-align match against in-memory short_term so unrelated
        carryover earlier in the deque is never destroyed.
        """
        path = self._path_for(user_id)
        if path is None or not session_id:
            return None
        with self._lock:
            data = self._load(path)
            record = data["sessions"].get(session_id)
            if record is None:
                return None
            messages = record.get("messages") or []
            index = None
            for i, message in enumerate(messages):
                if message.get("turn_id") == turn_id:
                    index = i
                    break
            if index is None:
                return None
            removed = list(messages[index:])
            del messages[index:]
            record["updated_at"] = time.time()
            self._save(path, data)
            return list(messages), removed

    def last_turn_info(self, user_id: Optional[str], session_id: str) -> Optional[str]:
        """Turn id of the final assistant message, or ``None``.

        Used to rehydrate ``WebChatSession.last_turn_id`` when a fresh
        in-memory session is constructed over an existing durable transcript
        (process restart, TTL eviction) — without this, amend/rewind have no
        way to validate a turn id against a session that has never completed
        a turn in this process's lifetime.
        """
        path = self._path_for(user_id)
        if path is None or not session_id:
            return None
        with self._lock:
            data = self._load(path)
            record = data["sessions"].get(session_id)
            if record is None:
                return None
            messages = record.get("messages") or []
            for message in reversed(messages):
                if message.get("role") == "assistant":
                    return message.get("turn_id")
            return None

    def replace_last_assistant(
        self,
        user_id: Optional[str],
        session_id: str,
        turn_id: str,
        content: str,
        ja: Optional[str] = None,
    ) -> Optional[str]:
        """Overwrite the latest assistant message's text, if it matches
        ``turn_id``.

        Returns the message's OLD content on success, or ``None`` on any
        mismatch (unknown session, empty transcript, non-assistant tail, or
        stale turn id). The transcript is the authoritative record for
        amend: returning the old content lets the caller locate the matching
        in-memory short_term message by content rather than assuming it sits
        at a fixed offset (idle chatter and legacy council turns can append trailing
        messages after the one being amended).
        """
        path = self._path_for(user_id)
        if path is None or not session_id:
            return None
        with self._lock:
            data = self._load(path)
            record = data["sessions"].get(session_id)
            if record is None:
                return None
            messages = record.get("messages") or []
            if not messages:
                return None
            last = messages[-1]
            if last.get("role") != "assistant" or last.get("turn_id") != turn_id:
                return None
            old_content = last.get("content", "")
            last["content"] = content
            if ja is not None:
                last["ja"] = ja
            record["updated_at"] = time.time()
            self._save(path, data)
            return old_content

    def delete_session(self, user_id: Optional[str], session_id: str) -> bool:
        """Drop a session's persisted transcript. Idempotent."""
        path = self._path_for(user_id)
        if path is None:
            return False
        with self._lock:
            data = self._load(path)
            if session_id not in data["sessions"]:
                return False
            del data["sessions"][session_id]
            self._save(path, data)
        return True

    @staticmethod
    def _truncate_title(text: str) -> str:
        cleaned = " ".join(text.split())
        if len(cleaned) <= _TITLE_MAX_CHARS:
            return cleaned
        return cleaned[: _TITLE_MAX_CHARS - 1].rstrip() + "…"
