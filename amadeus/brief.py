"""Daily morning brief: overnight synthesis from what Amadeus already knows.

The Pulse-style proactivity feature: a background job (see
``amadeus/web/scheduler.py``) generates, once per day, a short Kurisu-voiced
brief from disk snapshots — the user profile (open threads, last episode),
pending/overdue reminders, and optionally one arXiv search — and stores it
here. Delivery is greeting-first: ``ChatEngine.open_session_greeting`` folds
the unconsumed brief into the first greeting of the day and marks it
consumed; ``GET /api/brief`` serves a read-only reference copy.

Deliberately reads NO live MemoryManager: constructing one per user from a
scheduler would fire session-boundary bookkeeping (session_count/trust
inflation) daily. UserProfile and ReminderStore are atomic-write files that
are safe to snapshot-read concurrently — and ``list_pending`` (never the
destructive ``pop_due``) keeps reminder announcements intact.
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
class DailyBrief:
    date: str  # ISO date the brief is FOR
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
    def from_dict(cls, data: Any) -> Optional["DailyBrief"]:
        if not isinstance(data, dict) or not data.get("date") or not data.get("text"):
            return None
        return cls(
            date=str(data["date"]),
            text=str(data["text"]),
            generated_at=str(data.get("generated_at") or ""),
            consumed=bool(data.get("consumed", False)),
        )


class DailyBriefStore:
    """One JSON file, one brief per user (yesterday's brief is worthless, so
    each day's entry overwrites the last). threading.Lock + reload-before-
    mutate + atomic writes, same stance as ReminderStore. Not designed for
    multi-worker uvicorn (neither is the rest of the persistence layer);
    ``put``'s reload-and-refuse narrows the double-generation window.

    The path handed in is already character-scoped by the caller
    (SessionStore applies Config.character_memory_path, so a guest
    character's briefs never overwrite Kurisu's); this class never scopes.
    """

    def __init__(self, path: str) -> None:
        self.path = str(path)
        self._lock = threading.Lock()
        self._briefs: Dict[str, DailyBrief] = {}
        self._load()

    @staticmethod
    def _key(user_id: Optional[str]) -> str:
        return user_id or _DEFAULT_USER_KEY

    def _load(self) -> None:
        self._briefs = {}
        if not os.path.exists(self.path):
            return
        try:
            with open(self.path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            for key, entry in (data.get("briefs") or {}).items():
                brief = DailyBrief.from_dict(entry)
                if brief is not None:
                    self._briefs[str(key)] = brief
        except Exception as exc:  # corrupt/unreadable — start clean
            LOGGER.warning("Could not read daily brief file %s: %s", self.path, exc)
            self._briefs = {}

    def _save(self) -> None:
        payload = {
            "version": 1,
            "briefs": {key: brief.to_dict() for key, brief in self._briefs.items()},
        }
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        os.replace(tmp, self.path)

    def get(self, user_id: Optional[str]) -> Optional[DailyBrief]:
        with self._lock:
            self._load()
            return self._briefs.get(self._key(user_id))

    def has_for_date(self, user_id: Optional[str], date: dt.date) -> bool:
        brief = self.get(user_id)
        return brief is not None and brief.date == date.isoformat()

    def put(self, user_id: Optional[str], brief: DailyBrief) -> bool:
        """Store a brief; refuses (returns False) when a same-date entry
        already exists — the double-generation guard."""
        with self._lock:
            self._load()
            existing = self._briefs.get(self._key(user_id))
            if existing is not None and existing.date == brief.date:
                return False
            self._briefs[self._key(user_id)] = brief
            self._save()
            return True

    def peek_unconsumed(
        self, user_id: Optional[str], today: dt.date
    ) -> Optional[DailyBrief]:
        """Today's brief if it hasn't been delivered yet (stale briefs from
        previous days are never delivered late)."""
        brief = self.get(user_id)
        if brief is None or brief.consumed or brief.date != today.isoformat():
            return None
        return brief

    def mark_consumed(self, user_id: Optional[str], date: str) -> None:
        with self._lock:
            self._load()
            brief = self._briefs.get(self._key(user_id))
            if brief is not None and brief.date == date:
                brief.consumed = True
                self._save()


# ---------------------------------------------------------------------------
# Brief content
# ---------------------------------------------------------------------------


def collect_brief_facts(
    *,
    profile: Any,
    pending_reminders: List[Any],
    arxiv_hits: List[Dict[str, str]],
    now: dt.datetime,
    calendar_events: Optional[List[Any]] = None,
) -> List[str]:
    """Assemble the fact lines the brief generation call sees. ``profile``
    is a UserProfile or None; reminders are ``Reminder`` rows;
    ``calendar_events`` (optional, default None) are ``CalendarEvent`` rows
    from ``amadeus.calendar_feed.upcoming_events`` — omitted entirely (not
    even an empty-line placeholder) when the feature is off or nothing is
    upcoming, so the brief is byte-identical to before this feature existed
    in that case."""
    facts: List[str] = [f"Now: {now.strftime('%A %H:%M')}."]
    overdue = [r for r in pending_reminders if r.due_at <= now]
    upcoming = [
        r for r in pending_reminders if now < r.due_at <= now + dt.timedelta(hours=24)
    ]
    if overdue:
        facts.append(
            "Overdue reminders: "
            + "; ".join(f"{r.text} (was due {r.due_at:%a %H:%M})" for r in overdue[:4])
        )
    if upcoming:
        facts.append(
            "Due today: "
            + "; ".join(f"{r.text} (at {r.due_at:%H:%M})" for r in upcoming[:4])
        )
    if calendar_events:
        facts.append(
            "Today's calendar: "
            + "; ".join(event.human_line() for event in calendar_events)
        )
    if profile is not None:
        if profile.open_threads:
            facts.append(
                "Open threads: "
                + "; ".join(
                    profile.describe_thread(text) for text in profile.open_threads
                )
            )
        if profile.last_episode:
            facts.append(f"Last session: {profile.last_episode}")
    for hit in arxiv_hits[:2]:
        title = hit.get("title") or ""
        date = hit.get("date") or ""
        if title:
            facts.append(f"Possibly relevant new paper: {title} ({date}).")
    return facts


def arxiv_topic_from_profile(profile: Any) -> Optional[str]:
    """A single search topic for the optional literature lookup: the first
    open thread, else the first profile fact value."""
    if profile is None:
        return None
    if profile.open_threads:
        return str(profile.open_threads[0])
    for value in (profile.facts or {}).values():
        if value:
            return str(value)
    return None


def generate_daily_brief(
    *,
    provider: Any,
    facts: List[str],
    max_tokens: int = 220,
    character_name: str = "Makise Kurisu",
    character: Any = None,
) -> Optional[str]:
    """One cheap-model call producing the brief text. None on any failure —
    the scheduler treats that as 'retry next tick'.

    Pass ``character`` (CharacterSettings) when available: it supplies both
    the name and the voice capsule, so a custom pack's brief doesn't default
    to Kurisu's identity or register. ``character_name`` remains for older
    callers."""
    from .persona import persona_capsule
    from .providers.llm.base import ChatMessage

    if provider is None or len(facts) <= 1:
        # Only the timestamp fact: nothing worth briefing.
        return None
    name = (getattr(character, "full_name", None) or character_name).strip()
    prompt = [
        ChatMessage(
            role="system",
            content=(
                f"You are {name}, preparing a short private morning "
                "brief for someone you know. Write at most 90 words of plain "
                "prose (no lists, no markdown, no greetings) summarizing "
                "what deserves their attention today, in priority order: "
                "overdue reminders first, then today's calendar events, "
                "then open threads, then at most one paper mention. "
                + persona_capsule(character)
                + " Never invent facts that are not listed."
            ),
        ),
        ChatMessage(role="user", content="\n".join(facts)),
    ]
    try:
        text = (provider.generate(prompt, max_tokens=int(max_tokens)) or "").strip()
        telemetry.record_llm_usage(provider, "brief")
    except Exception as exc:
        LOGGER.warning("Daily brief generation failed: %s", exc)
        telemetry.incr("brief.generation_failed")
        return None
    return text or None
