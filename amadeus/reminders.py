"""Local, network-free reminders/timers for Amadeus.

Poll-driven only: no scheduler thread. `pop_due()` is the single due-check API,
used both for live delivery (while the app runs) and session-start catch-up
(reminders that came due while the app was closed). Stored as a small JSON file
alongside memory_store.json.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import os
import threading
from dataclasses import dataclass
from typing import Dict, List, Optional
from uuid import uuid4

LOGGER = logging.getLogger(__name__)

_PRUNE_DAYS = 7  # drop non-pending entries older than this on load


@dataclass
class Reminder:
    id: str
    text: str
    due_at: dt.datetime
    created_at: dt.datetime
    status: str = "pending"  # pending | announced | cancelled

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "text": self.text,
            "due_at": self.due_at.isoformat(),
            "created_at": self.created_at.isoformat(),
            "status": self.status,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Reminder":
        return cls(
            id=str(d["id"]),
            text=str(d["text"]),
            due_at=dt.datetime.fromisoformat(d["due_at"]),
            created_at=dt.datetime.fromisoformat(d["created_at"]),
            status=str(d.get("status", "pending")),
        )


class ReminderStore:
    """JSON-backed reminder store. Local naive timestamps (single machine).

    Concurrency: unlike the sibling stores (one long-lived instance each),
    this file is read-modify-written by several live instances at once — the
    engine's cached store, the reminders tool, and the brief/async-notify
    jobs' throwaway instances. A per-PATH class-level lock (instance locks
    can't serialize across instances) plus reload-before-mutate in every
    mutating method keeps a concurrent add from being lost to
    last-writer-wins, and a pop from double-announcing.
    """

    _PATH_LOCKS: Dict[str, threading.Lock] = {}
    _PATH_LOCKS_GUARD = threading.Lock()

    def __init__(self, path: str = "reminders.json") -> None:
        self.path = path
        key = os.path.abspath(str(path))
        with ReminderStore._PATH_LOCKS_GUARD:
            self._lock = ReminderStore._PATH_LOCKS.setdefault(key, threading.Lock())
        self._reminders: List[Reminder] = []
        with self._lock:
            self._load()

    # --- persistence ---------------------------------------------------------

    def _load(self) -> None:
        self._reminders = []
        if not os.path.exists(self.path):
            return
        try:
            with open(self.path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            for entry in data.get("reminders", []):
                try:
                    self._reminders.append(Reminder.from_dict(entry))
                except Exception:  # skip a corrupt row, keep the rest
                    continue
        except Exception as exc:  # corrupt/unreadable file — start clean
            LOGGER.warning("Could not read reminders file %s: %s", self.path, exc)
            self._reminders = []
        self._prune()

    def _prune(self) -> None:
        cutoff = dt.datetime.now() - dt.timedelta(days=_PRUNE_DAYS)
        self._reminders = [
            r
            for r in self._reminders
            if r.status == "pending" or r.created_at >= cutoff
        ]

    def _save(self) -> None:
        payload = {"version": 1, "reminders": [r.to_dict() for r in self._reminders]}
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        os.replace(tmp, self.path)

    # --- API -----------------------------------------------------------------

    def add(self, text: str, due_at: dt.datetime) -> Reminder:
        r = Reminder(
            id=uuid4().hex[:6],
            text=text.strip(),
            due_at=due_at,
            created_at=dt.datetime.now(),
        )
        with self._lock:
            self._load()  # reload-before-mutate: keep concurrent adds
            self._reminders.append(r)
            self._save()
        return r

    def list_pending(self) -> List[Reminder]:
        with self._lock:
            self._load()  # separate instances (CLI + tool) share one file
            return sorted(
                (r for r in self._reminders if r.status == "pending"),
                key=lambda r: r.due_at,
            )

    def cancel(self, reminder_id: str) -> bool:
        with self._lock:
            self._load()
            rid = reminder_id.strip().lower()
            for r in self._reminders:
                if r.status == "pending" and (
                    r.id == rid or r.text.lower().startswith(rid)
                ):
                    r.status = "cancelled"
                    self._save()
                    return True
            return False

    def pop_due(self, now: Optional[dt.datetime] = None) -> List[Reminder]:
        """Return pending reminders due at/<= now, marking them announced.
        The single due-check API — serves live delivery and catch-up alike."""
        now = now or dt.datetime.now()
        with self._lock:
            self._load()
            due = [
                r for r in self._reminders if r.status == "pending" and r.due_at <= now
            ]
            if due:
                for r in due:
                    r.status = "announced"
                self._save()
        return sorted(due, key=lambda r: r.due_at)

    def restore_pending(self, reminders: List[Reminder]) -> None:
        """Undo ``pop_due`` for these reminders after a delivery that failed
        AFTER the pop (async-notify's Discord DM). Only flips announced rows
        back to pending — anything cancelled/re-announced meanwhile is left
        alone. The caller only restores when nothing was actually delivered,
        so exactly-once announcement holds."""
        ids = {r.id for r in reminders}
        if not ids:
            return
        with self._lock:
            self._load()
            changed = False
            for r in self._reminders:
                if r.id in ids and r.status == "announced":
                    r.status = "pending"
                    changed = True
            if changed:
                self._save()
