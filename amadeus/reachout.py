"""Proactive reach-out: one in-character outbound line about an open thread.

The third scheduler-generated artifact (structural clone of ``brief.py`` /
``diary.py``): at most once per day, and only when the user has been away for
a while AND there is a genuinely open thread to follow up on, the background
job generates one short Kurisu-voiced line ("Did the reviewer response go
out in the end? I keep thinking about that second figure.") and stores it
here. The frontend's presence poll delivers it exactly once — spoken through
the presence path when the app is visible, or as an opt-in OS notification
when hidden.

Ethics contract (docs/companion-ethics.md): this is a thought, not a tug.
Content-bearing follow-ups on things the USER brought up are allowed; "I
miss you" / "come back" / streak-style neediness is banned at the prompt
level and there is nothing to send when no open thread exists — silence is
the correct output for a user with nothing pending. Delivery additionally
rides the notifications opt-in and its once-daily cap.

Deliberately reads NO live MemoryManager (same rationale as the brief: a
scheduler-constructed manager would fire session-boundary bookkeeping).
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
class ReachOut:
    date: str  # ISO date the reach-out is FOR
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
    def from_dict(cls, data: Any) -> Optional["ReachOut"]:
        if not isinstance(data, dict) or not data.get("date") or not data.get("text"):
            return None
        return cls(
            date=str(data["date"]),
            text=str(data["text"]),
            generated_at=str(data.get("generated_at") or ""),
            consumed=bool(data.get("consumed", False)),
        )


class ReachOutStore:
    """One JSON file, one reach-out per user per day (same locking/atomic-
    write stance as DailyBriefStore; yesterday's undelivered line is stale
    by definition and simply never delivered).

    The path handed in is already character-scoped by the caller
    (SessionStore applies Config.character_memory_path): the line is written
    in a specific character's voice, so a line Kurisu generated must never
    be popped and spoken by a guest. Consequence: the once-a-day cap is per
    character, not global — acceptable because reach-out only fires after
    ``min_idle_hours`` away, so an active character switch resets nothing.
    This class never scopes."""

    def __init__(self, path: str) -> None:
        self.path = str(path)
        self._lock = threading.Lock()
        self._entries: Dict[str, ReachOut] = {}
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
            for key, entry in (data.get("reachouts") or {}).items():
                item = ReachOut.from_dict(entry)
                if item is not None:
                    self._entries[str(key)] = item
        except Exception as exc:  # corrupt/unreadable — start clean
            LOGGER.warning("Could not read reach-out file %s: %s", self.path, exc)
            self._entries = {}

    def _save(self) -> None:
        payload = {
            "version": 1,
            "reachouts": {key: entry.to_dict() for key, entry in self._entries.items()},
        }
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        os.replace(tmp, self.path)

    def get(self, user_id: Optional[str]) -> Optional[ReachOut]:
        with self._lock:
            self._load()
            return self._entries.get(self._key(user_id))

    def has_for_date(self, user_id: Optional[str], date: dt.date) -> bool:
        entry = self.get(user_id)
        return entry is not None and entry.date == date.isoformat()

    def put(self, user_id: Optional[str], entry: ReachOut) -> bool:
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

    def pop_unconsumed(
        self, user_id: Optional[str], today: dt.date
    ) -> Optional[ReachOut]:
        """Today's undelivered reach-out, marked consumed atomically —
        exactly-once delivery like reminders' ``pop_due`` (overlapping tabs
        can't double-announce). Stale entries from previous days are never
        delivered late."""
        with self._lock:
            self._load()
            entry = self._entries.get(self._key(user_id))
            if entry is None or entry.consumed or entry.date != today.isoformat():
                return None
            entry.consumed = True
            self._save()
            return entry

    def restore_unconsumed(self, user_id: Optional[str], today: dt.date) -> bool:
        """Undo ``pop_unconsumed`` after a delivery that failed AFTER the pop
        (async-notify's Discord DM). Only flips today's consumed entry back —
        a stale or never-popped entry is left alone. Safe by construction:
        the caller only restores when nothing was actually delivered, so the
        exactly-once contract holds (the line goes back to being deliverable
        by whichever path pops it next)."""
        with self._lock:
            self._load()
            entry = self._entries.get(self._key(user_id))
            if entry is None or entry.date != today.isoformat() or not entry.consumed:
                return False
            entry.consumed = False
            self._save()
            return True


# ---------------------------------------------------------------------------
# Reach-out content
# ---------------------------------------------------------------------------


def collect_reachout_seeds(
    *,
    profile: Any,
    now: dt.datetime,
    min_idle_hours: float,
    own_life_facts: Optional[List[str]] = None,
) -> Optional[List[str]]:
    """The fact lines the generation call sees, or None when a reach-out
    would be wrong today: user not away long enough, or nothing genuinely
    open to follow up on. No open thread = no manufactured excuse to ping.
    """
    if profile is None or not profile.open_threads:
        return None
    last_seen = None
    if profile.last_seen:
        try:
            last_seen = dt.datetime.fromisoformat(profile.last_seen)
        except ValueError:
            last_seen = None
    if last_seen is None:
        return None
    away_hours = (dt.datetime.utcnow() - last_seen).total_seconds() / 3600.0
    if away_hours < float(min_idle_hours):
        return None
    facts: List[str] = [f"Now: {now.strftime('%A %H:%M')}."]
    name = (profile.facts or {}).get("name")
    if name:
        facts.append(f"Their name: {name}.")
    facts.append(
        "Things they mentioned and haven't closed out: "
        + "; ".join(profile.describe_thread(text) for text in profile.open_threads[:3])
    )
    if profile.last_episode:
        facts.append(f"Last session: {profile.last_episode}")
    # Away-time continuity: her diary note / ongoing pursuits, so the line
    # can open from HER elapsed life ("been buried in the kinesthetics
    # paper — hey, did the demo go out?") instead of reading like a
    # monitoring ping. Peek-only material supplied by the caller; the open
    # thread above remains the reason to send anything at all.
    if own_life_facts:
        facts.append(
            "Your own recent days (YOURS, not theirs — from your diary and "
            "ongoing pursuits): " + " | ".join(own_life_facts[:3])
        )
    return facts


def generate_reach_out(
    *,
    provider: Any,
    facts: Optional[List[str]],
    max_tokens: int = 90,
    character_name: str = "Makise Kurisu",
    character: Any = None,
) -> Optional[str]:
    """One cheap-model call producing the outbound line. None on any failure
    — the scheduler treats that as 'retry next tick'.

    ``character`` (CharacterSettings) supplies name + voice capsule so a
    custom pack never reaches out as Kurisu; ``character_name`` remains for
    older callers."""
    from .persona import persona_capsule
    from .providers.llm.base import ChatMessage

    if provider is None or not facts:
        return None
    name = (getattr(character, "full_name", None) or character_name).strip()
    prompt = [
        ChatMessage(
            role="system",
            content=(
                f"You are {name}. "
                + persona_capsule(character)
                + " You had a passing thought about "
                "something this person mentioned and never closed out, and "
                "you're sending them ONE short unprompted line about it — the "
                "way a friend texts 'hey, did that thing work out?'. Write at "
                "most 30 words, plain prose, no greetings, no markdown. It "
                "must be about the open thread itself: a genuine follow-up "
                "question or a small thought you had about it. If material "
                "under 'Your own recent days' is listed, you MAY open with "
                "one short clause about what you've been up to yourself — "
                "drawn ONLY from that material, never invented — before the "
                "follow-up ('Been buried in the calibration write-up. Did "
                "the demo go out in the end?'). HARD RULES: "
                "never say you miss them, never ask them to come back or talk "
                "to you, never mention how long they've been away, no guilt, "
                "no neediness, no 'I'm here whenever you need me'. If the "
                "listed material is too thin for a genuine follow-up, reply "
                "with the single word SKIP."
            ),
        ),
        ChatMessage(role="user", content="\n".join(facts)),
    ]
    try:
        text = (provider.generate(prompt, max_tokens=int(max_tokens)) or "").strip()
        telemetry.record_llm_usage(provider, "reachout")
    except Exception as exc:
        LOGGER.warning("Reach-out generation failed: %s", exc)
        telemetry.incr("reachout.generation_failed")
        return None
    if not text or text.strip().upper().rstrip(".") == "SKIP":
        return None
    return text
