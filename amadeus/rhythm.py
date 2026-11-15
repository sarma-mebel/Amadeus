"""Rhythm memory: the coarse shape of the user's week.

Ambient signals (``amadeus/presence.py::ambient_comment``) are deliberately
forgetful — one throwaway line about a single transition, never stored. That
makes her reactive but pattern-blind: she can notice you just opened the
laptop, never that this is the third late night this week.

This module adds an OPT-IN durable tally so a pattern is noticeable. What it
stores is the whole privacy argument:

- Counters only. How MUCH, roughly WHEN (hour buckets), and which coarse
  CATEGORY of app. Never a window title, never a document or track name,
  never a per-event timestamp.
- The category vocabulary is fixed and small (editor/browser/terminal/chat/
  media/other) and the client maps to it BEFORE the POST, so a raw title
  never crosses the network at all (see frontend/src/rhythm.ts for why that
  boundary is where it is).
- 14-day rolling retention, so it can answer "this week vs last week" and
  nothing longer.

Consent is doubled: ``rhythm.enabled`` in config AND a per-browser
localStorage opt-in, both required before a byte is written. Turning it off
ERASES the file rather than merely halting writes.

Surfacing is deliberately dull: at most one mention a week, chosen by the
pure ``notable_rhythm`` predicate below (no LLM), delivered as greeting
COLOR — never an ask, never mid-session. A rhythm remark in idle chatter
would be a jump-scare, and per-turn prompt presence would make her a
productivity tracker. She notices; she does not measure.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import telemetry
from .logging_utils import get_logger

LOGGER = get_logger(__name__)

#: The complete stored app vocabulary. An unrecognized app becomes "other"
#: client-side; anything outside this set is dropped on ingest, so the store
#: can never accumulate a list of the user's software.
APP_BUCKETS = ("editor", "browser", "terminal", "chat", "media", "other")

#: Two weeks is the shortest window supporting "this week vs. last week",
#: and nothing here needs longer.
RETENTION_DAYS = 14

_MAX_HOUR_BUCKETS = 24


@dataclass
class RhythmDay:
    """One local day's tally. This is the ENTIRE durable artifact."""

    date: str
    late_night_blocks: int = 0
    long_focus_stretches: int = 0
    longest_focus_hours: float = 0.0
    active_hour_buckets: List[int] = field(default_factory=list)
    music_events: int = 0
    battery_low_events: int = 0
    app_buckets: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "date": self.date,
            "late_night_blocks": self.late_night_blocks,
            "long_focus_stretches": self.long_focus_stretches,
            "longest_focus_hours": self.longest_focus_hours,
            "active_hour_buckets": list(self.active_hour_buckets),
            "music_events": self.music_events,
            "battery_low_events": self.battery_low_events,
            "app_buckets": dict(self.app_buckets),
        }

    @classmethod
    def from_dict(cls, data: Any) -> Optional["RhythmDay"]:
        """Coerce one stored/posted day into the canonical bounded shape.

        Never raises: a malformed entry yields None and is dropped. Every
        numeric field is clamped non-negative and every app bucket outside
        APP_BUCKETS is discarded — a client (or a hand-edited file) must not
        be able to widen what this store holds.
        """
        if not isinstance(data, dict):
            return None
        date = str(data.get("date") or "").strip()[:10]
        if not date:
            return None

        def _int(key: str) -> int:
            try:
                return max(0, int(data.get(key) or 0))
            except (TypeError, ValueError):
                return 0

        try:
            longest = max(0.0, float(data.get("longest_focus_hours") or 0.0))
        except (TypeError, ValueError):
            longest = 0.0

        hours: List[int] = []
        for raw in data.get("active_hour_buckets") or []:
            try:
                hour = int(raw)
            except (TypeError, ValueError):
                continue
            if 0 <= hour <= 23 and hour not in hours:
                hours.append(hour)
        hours.sort()

        buckets: Dict[str, int] = {}
        for name, raw in (data.get("app_buckets") or {}).items():
            if str(name) not in APP_BUCKETS:
                continue
            try:
                count = max(0, int(raw))
            except (TypeError, ValueError):
                continue
            if count:
                buckets[str(name)] = count

        return cls(
            date=date,
            late_night_blocks=_int("late_night_blocks"),
            long_focus_stretches=_int("long_focus_stretches"),
            longest_focus_hours=round(longest, 1),
            active_hour_buckets=hours[:_MAX_HOUR_BUCKETS],
            music_events=_int("music_events"),
            battery_low_events=_int("battery_low_events"),
            app_buckets=buckets,
        )


class RhythmStore:
    """One JSON file per user, 14 rolling days. Lock + reload-before-mutate
    + atomic write, same stance as the sibling stores. The path handed in is
    already character-scoped by the caller."""

    def __init__(self, path: str) -> None:
        self.path = str(path)
        self._lock = threading.Lock()
        self._days: Dict[str, Dict[str, RhythmDay]] = {}
        self._last_surfaced: Dict[str, str] = {}
        self._load()

    @staticmethod
    def _key(user_id: Optional[str]) -> str:
        return user_id or "default"

    def _load(self) -> None:
        self._days = {}
        self._last_surfaced = {}
        if not os.path.exists(self.path):
            return
        try:
            with open(self.path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            for key, entries in (data.get("days") or {}).items():
                parsed: Dict[str, RhythmDay] = {}
                for entry in entries or []:
                    day = RhythmDay.from_dict(entry)
                    if day is not None:
                        parsed[day.date] = day
                self._days[str(key)] = parsed
            for key, stamp in (data.get("last_surfaced") or {}).items():
                self._last_surfaced[str(key)] = str(stamp)
        except Exception as exc:  # corrupt/unreadable — start clean, loudly
            LOGGER.warning("Could not read rhythm file %s: %s", self.path, exc)
            telemetry.incr("rhythm.store_load_failed")
            self._days = {}
            self._last_surfaced = {}

    def _save(self) -> None:
        payload = {
            "version": 1,
            "days": {
                key: [
                    day.to_dict() for day in sorted(days.values(), key=lambda d: d.date)
                ]
                for key, days in self._days.items()
            },
            "last_surfaced": dict(self._last_surfaced),
        }
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        os.replace(tmp, self.path)

    def _trim(self, days: Dict[str, RhythmDay], today: dt.date) -> Dict[str, RhythmDay]:
        cutoff = (today - dt.timedelta(days=RETENTION_DAYS)).isoformat()
        return {date: day for date, day in days.items() if date > cutoff}

    def record(self, user_id: Optional[str], day: RhythmDay, today: dt.date) -> None:
        """Upsert one day's tally (the client posts the running total for the
        current day, so a later post for the same date replaces the earlier)."""
        with self._lock:
            self._load()
            key = self._key(user_id)
            days = self._days.setdefault(key, {})
            days[day.date] = day
            self._days[key] = self._trim(days, today)
            self._save()

    def days(self, user_id: Optional[str]) -> List[RhythmDay]:
        with self._lock:
            self._load()
            return sorted(
                self._days.get(self._key(user_id), {}).values(), key=lambda d: d.date
            )

    def clear(self, user_id: Optional[str]) -> None:
        """Erase this user's tally. Toggling the feature off calls this — the
        promise is that turning it off deletes the record, not that it stops
        adding to one."""
        with self._lock:
            self._load()
            key = self._key(user_id)
            self._days.pop(key, None)
            self._last_surfaced.pop(key, None)
            self._save()
        telemetry.incr("rhythm.cleared")

    def surfaced_recently(
        self, user_id: Optional[str], today: dt.date, min_days: int
    ) -> bool:
        with self._lock:
            self._load()
            stamp = self._last_surfaced.get(self._key(user_id), "")
        if not stamp:
            return False
        try:
            last = dt.date.fromisoformat(stamp[:10])
        except ValueError:
            return False
        return (today - last).days < max(1, min_days)

    def mark_surfaced(self, user_id: Optional[str], today: dt.date) -> None:
        with self._lock:
            self._load()
            self._last_surfaced[self._key(user_id)] = today.isoformat()
            self._save()
        telemetry.incr("rhythm.surfaced")


def notable_rhythm(days: List[RhythmDay], today: dt.date) -> Optional[str]:
    """One short phrasing hint about the week's shape, or None.

    Deterministic and LLM-free by design: an LLM asked "is anything notable
    here" will always find something, and a companion who comments on your
    habits every week is a productivity tracker. None is the correct and
    common answer, and the thresholds below are what "notable" means.
    """
    if not days:
        return None
    week_cutoff = (today - dt.timedelta(days=7)).isoformat()
    recent = [day for day in days if day.date > week_cutoff]
    if len(recent) < 3:
        # Too little history to call anything a pattern.
        return None

    late_nights = sum(1 for day in recent if day.late_night_blocks > 0)
    if late_nights >= 3:
        return "They have been up past midnight most nights this week."

    stretches = sum(day.long_focus_stretches for day in recent)
    if stretches >= 5:
        dominant = _dominant_bucket(recent)
        if dominant and dominant != "other":
            return (
                "They have had a lot of long uninterrupted stretches at the "
                f"keyboard this week, mostly in a {dominant}."
            )
        return (
            "They have had a lot of long uninterrupted stretches at the "
            "keyboard this week."
        )

    prior = [
        day
        for day in days
        if (today - dt.timedelta(days=14)).isoformat() < day.date <= week_cutoff
    ]
    shift = _median_start_shift(recent, prior)
    if shift is not None and shift >= 3:
        return "Their days have started noticeably later this week than last."
    return None


def _dominant_bucket(days: List[RhythmDay]) -> Optional[str]:
    totals: Dict[str, int] = {}
    for day in days:
        for name, count in day.app_buckets.items():
            totals[name] = totals.get(name, 0) + count
    if not totals:
        return None
    return max(totals.items(), key=lambda pair: pair[1])[0]


def _median_start_shift(
    recent: List[RhythmDay], prior: List[RhythmDay]
) -> Optional[float]:
    """Hours later the median day now starts, or None without both weeks."""
    if len(prior) < 3:
        return None

    def _median_start(days: List[RhythmDay]) -> Optional[float]:
        starts = [
            min(day.active_hour_buckets) for day in days if day.active_hour_buckets
        ]
        if not starts:
            return None
        starts.sort()
        mid = len(starts) // 2
        if len(starts) % 2:
            return float(starts[mid])
        return (starts[mid - 1] + starts[mid]) / 2.0

    now = _median_start(recent)
    before = _median_start(prior)
    if now is None or before is None:
        return None
    return now - before


def rhythm_greeting_fact(hint: Optional[str]) -> Optional[str]:
    """Wrap a notable_rhythm hint as a greeting fact, or None.

    Color, never an ask: the greeting's ask slot is arbitrated for the
    USER's material (next-time thread > follow-up > open thread >
    early acquaintance), and rhythm must not compete for it.
    """
    if not hint:
        return None
    return (
        "Something you've quietly noticed about their week (mention it in at "
        "most one short clause, or not at all — never as a statistic, never "
        "as concern, never twice, and never as advice): " + hint
    )
