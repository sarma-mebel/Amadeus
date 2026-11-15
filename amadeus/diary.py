"""Kurisu's diary: a scheduler-generated daily in-character note about HER
day — a paper she read, a half-formed lab idea, dry grumbling — so the
companion reads as having a life of her own, not just reacting to the user.

Structurally cloned from ``amadeus/brief.py`` (see that module for the full
rationale on disk-snapshot reads and why no live MemoryManager is touched
here). The diary differs from the brief in one key way: it is not gated on
having anything to report. A daily brief with nothing due is a no-op; a
diary entry is written even when the user's profile is empty, because
Kurisu's day happens regardless of what she knows about the user.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import telemetry
from .logging_utils import get_logger

LOGGER = get_logger(__name__)

_DEFAULT_USER_KEY = "default"


@dataclass
class DiaryEntry:
    date: str  # ISO date the entry is FOR
    text: str
    generated_at: str  # ISO datetime
    consumed: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "date": self.date,
            "text": self.text,
            "generated_at": self.generated_at,
            "consumed": self.consumed,
        }

    @classmethod
    def from_dict(cls, data: Any) -> Optional["DiaryEntry"]:
        if not isinstance(data, dict) or not data.get("date") or not data.get("text"):
            return None
        return cls(
            date=str(data["date"]),
            text=str(data["text"]),
            generated_at=str(data.get("generated_at") or ""),
            consumed=bool(data.get("consumed", False)),
        )


class DiaryStore:
    """One JSON file, one entry per user (yesterday's entry is worthless, so
    each day's entry overwrites the last). threading.Lock + reload-before-
    mutate + atomic writes, same stance as ReminderStore/DailyBriefStore. Not
    designed for multi-worker uvicorn (neither is the rest of the
    persistence layer); ``put``'s reload-and-refuse narrows the
    double-generation window.

    The path handed in is already character-scoped by the caller
    (SessionStore applies Config.character_memory_path — the diary is the
    active character's OWN day, so a guest's entries must not overwrite
    Kurisu's); this class never scopes.
    """

    def __init__(self, path: str) -> None:
        self.path = str(path)
        self._lock = threading.Lock()
        self._entries: Dict[str, DiaryEntry] = {}
        self._load()

    @staticmethod
    def _key(user_id: Optional[str]) -> str:
        return user_id or _DEFAULT_USER_KEY

    def _load(self) -> None:
        self._entries = {}
        if not os.path.exists(self.path):
            return
        try:
            with open(self.path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            for key, entry in (data.get("entries") or {}).items():
                diary_entry = DiaryEntry.from_dict(entry)
                if diary_entry is not None:
                    self._entries[str(key)] = diary_entry
        except Exception as exc:  # corrupt/unreadable — start clean
            LOGGER.warning("Could not read diary file %s: %s", self.path, exc)
            self._entries = {}

    def _save(self) -> None:
        payload = {
            "version": 1,
            "entries": {key: entry.to_dict() for key, entry in self._entries.items()},
        }
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        os.replace(tmp, self.path)

    def get(self, user_id: Optional[str]) -> Optional[DiaryEntry]:
        with self._lock:
            self._load()
            return self._entries.get(self._key(user_id))

    def has_for_date(self, user_id: Optional[str], date: dt.date) -> bool:
        entry = self.get(user_id)
        return entry is not None and entry.date == date.isoformat()

    def put(self, user_id: Optional[str], entry: DiaryEntry) -> bool:
        """Store an entry; refuses (returns False) when a same-date entry
        already exists — the double-generation guard."""
        with self._lock:
            self._load()
            existing = self._entries.get(self._key(user_id))
            if existing is not None and existing.date == entry.date:
                return False
            self._entries[self._key(user_id)] = entry
            self._save()
            return True

    def peek_unconsumed(
        self, user_id: Optional[str], today: dt.date
    ) -> Optional[DiaryEntry]:
        """Today's entry if it hasn't been delivered yet (stale entries from
        previous days are never delivered late)."""
        entry = self.get(user_id)
        if entry is None or entry.consumed or entry.date != today.isoformat():
            return None
        return entry

    def mark_consumed(self, user_id: Optional[str], date: str) -> None:
        with self._lock:
            self._load()
            entry = self._entries.get(self._key(user_id))
            if entry is not None and entry.date == date:
                entry.consumed = True
                self._save()


# ---------------------------------------------------------------------------
# Diary content
# ---------------------------------------------------------------------------


def collect_diary_seeds(
    *,
    profile: Any,
    now: dt.datetime,
    life_seeds: Optional[List[str]] = None,
) -> List[str]:
    """Assemble the seed lines the diary generation call sees. ``profile``
    is a UserProfile or None; ``life_seeds`` are the pre-rendered
    life-thread context lines (life_threads.life_thread_seeds) — her
    ongoing pursuits, so consecutive diary entries develop instead of
    inventing an unrelated day each time. Pure, no I/O."""
    seeds: List[str] = [f"Now: {now:%A %H:%M}."]
    for line in life_seeds or []:
        seeds.append(line)
    if profile is not None:
        facts_values = [v for v in (profile.facts or {}).values() if v][:3]
        for value in facts_values:
            seeds.append(f"Something on your mind about the user: {value}")
        if profile.open_threads:
            seeds.append(
                "Open threads: "
                + "; ".join(
                    profile.describe_thread(text) for text in profile.open_threads
                )
            )
        if profile.last_episode:
            seeds.append(f"Last session: {profile.last_episode}")
    return seeds


def generate_diary_entry(
    *,
    provider: Any,
    facts: List[str],
    max_tokens: int = 180,
    character_name: str = "Makise Kurisu",
    character: Any = None,
) -> Optional[str]:
    """One cheap-model call producing the diary text. None on any failure —
    the scheduler treats that as 'retry next tick'. Unlike the daily brief,
    this does NOT bail on timestamp-only facts — Kurisu has a day even with
    an empty profile.

    ``character`` (CharacterSettings) supplies name, voice capsule, and the
    built-in test — the old exact string compare against "Makise Kurisu"
    silently sent any differently-spelled pack down the generic branch."""
    from .persona import persona_capsule
    from .providers.llm.base import ChatMessage

    if provider is None:
        return None
    name = (getattr(character, "full_name", None) or character_name).strip()
    if character is not None:
        is_builtin = character.resolved_system_prompt() is None
    else:
        is_builtin = character_name == "Makise Kurisu"
    prompt = [
        ChatMessage(
            role="system",
            content=(
                f"You are {name}, writing a short private diary note "
                + (
                    "about YOUR day — a paper you read, a half-formed lab idea, "
                    "something that annoyed you, dry in-character grumbling. "
                    if is_builtin
                    else "about YOUR day — something you did, a half-formed idea, "
                    "something that annoyed you, in-character grumbling. "
                )
                + persona_capsule(character)
                + " First person, at most 70 words, plain prose, no greetings, "
                "no addressing the reader. The listed context is what's been "
                "on your mind lately; you may let one item color the entry, "
                "but the entry is about you. Never invent facts about the "
                "user."
            ),
        ),
        ChatMessage(role="user", content="\n".join(facts)),
    ]
    try:
        text = (provider.generate(prompt, max_tokens=int(max_tokens)) or "").strip()
        telemetry.record_llm_usage(provider, "diary")
    except Exception as exc:
        LOGGER.warning("Diary generation failed: %s", exc)
        telemetry.incr("diary.generation_failed")
        return None
    return text or None
