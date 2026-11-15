"""Persistent user profile ("core memory") for long-lived familiarity.

Small, always-in-prompt facts about the user plus relationship state.
No LLM calls happen here; the profile is updated by the extraction job in
:mod:`amadeus.memory.manager` and rendered into the system prompt's static
region so provider prompt caching keeps its marginal cost near zero.
"""

from __future__ import annotations

import calendar
import datetime as dt
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .. import telemetry
from ..logging_utils import get_logger

LOGGER = get_logger(__name__)

# v2 adds persisted relationship state (mood, trust_level). v3 adds arc
# floats + tone_history. v4 adds last_felt (the most recent session's
# affective tag). v5 adds last_session_tone. v6 adds next_time_thread (the
# one open topic stashed at farewell for the next greeting). v7 adds
# friction_topics (subjects that went badly, so a misunderstanding can
# persist across sessions and then heal). Older files load with defaults.
PROFILE_VERSION = 7

DEFAULT_TRUST = 0.5

# tone_history is trimmed to the most recent N entries on load/record so the
# profile file (and the prompt-facing trend computation) stays bounded.
TONE_HISTORY_CAP = 20

# Bounds on the affective tag persisted in last_felt — kept short so it reads
# as a phrase in the greeting prompt, never a paragraph the model recites.
FELT_LABEL_MAX_CHARS = 24
FELT_SUBJECT_MAX_CHARS = 80

# The "next time" thread persisted at farewell (v6). The char cap keeps it a
# single line; past STALE_DAYS the greeting phrases it tentatively; past
# MAX_AGE_DAYS it is dropped unconsumed — a month-old "they were about to
# submit" delivered as fresh news would read as a script, not memory.
NEXT_TIME_THREAD_MAX_CHARS = 160
NEXT_TIME_THREAD_STALE_DAYS = 7
NEXT_TIME_THREAD_MAX_AGE_DAYS = 30

# Friction memory (v7): subjects that went badly BETWEEN the two of them, so
# a misunderstanding can outlive the session it happened in and then heal.
#
# The cap is deliberately tiny and the expiry deliberately unconditional.
# A companion that accumulates grievances and holds them until the user
# performs a repair has manufactured a debt, which is the retention pattern
# docs/companion-ethics.md exists to refuse. Three live topics, three weeks
# each, cleared by time alone — matching OPEN_THREAD max age, the existing
# precedent for "she stops carrying stale relational state".
FRICTION_TOPICS_CAP = 3
FRICTION_TOPIC_MAX_CHARS = 60
FRICTION_MAX_AGE_DAYS = 21


def _sanitize_felt(raw: Any) -> Optional[Dict[str, str]]:
    """Coerce a stored/extracted affective tag into the bounded canonical shape.

    Returns None unless there is a non-empty ``label`` (a felt-tone with no
    label carries nothing worth surfacing). ``subject`` and ``date`` are
    optional; everything is length-capped and lowercased-label so the greeting
    fact reads as a phrase. Never raises on garbage — bad input yields None.
    """
    if not isinstance(raw, dict):
        return None
    label = str(raw.get("label") or "").strip().lower()[:FELT_LABEL_MAX_CHARS]
    if not label:
        return None
    result: Dict[str, str] = {"label": label}
    subject = str(raw.get("subject") or "").strip()[:FELT_SUBJECT_MAX_CHARS]
    if subject and subject.lower() not in {"none", "null", "unknown", "n/a"}:
        result["subject"] = subject
    date = str(raw.get("date") or "").strip()[:10]
    if date:
        result["date"] = date
    return result


def _sanitize_friction(raw: Any) -> Optional[Dict[str, Any]]:
    """Coerce one stored friction entry into the bounded canonical shape.

    Returns None unless there is a non-empty ``topic``. Never raises on
    garbage — a corrupt entry is dropped, not fatal, so one bad write can
    never make a profile unloadable.
    """
    if not isinstance(raw, dict):
        return None
    topic = str(raw.get("topic") or "").strip()[:FRICTION_TOPIC_MAX_CHARS]
    if not topic:
        return None
    result: Dict[str, Any] = {"topic": topic}
    noted = str(raw.get("noted") or "").strip()[:10]
    if noted:
        result["noted"] = noted
    try:
        count = int(raw.get("count") or 1)
    except (TypeError, ValueError):
        count = 1
    result["count"] = max(1, count)
    return result


@dataclass
class UserProfile:
    """Durable facts about the user and the state of the relationship."""

    facts: Dict[str, str] = field(default_factory=dict)
    first_met: Optional[str] = None
    session_count: int = 0
    last_seen: Optional[str] = None
    open_threads: List[str] = field(default_factory=list)
    # thread text -> ISO date it was first noted; lets render_block phrase age
    # ("from last week") and lets prune_stale_threads expire threads the
    # extraction job keeps carrying but nobody has touched in weeks.
    open_thread_dates: Dict[str, str] = field(default_factory=dict)
    last_episode: Optional[str] = None
    # The most recent session's *felt* tone (v4): how the last conversation
    # FELT, kept deliberately separate from last_episode (which stays neutral,
    # events-only, on purpose — see extraction_jobs.run_episode_job). Shape:
    # {"label": "excited", "subject": "the paper deadline", "date": "<ISO>"}.
    # "label" is a one-word affect, "subject" an optional short phrase. Surfaced
    # into the greeting as a structured fact for the LLM to voice, never recited.
    last_felt: Optional[Dict[str, str]] = None
    # Sore subjects (v7): topics that went BADLY between the two of them, so a
    # misunderstanding outlives the session and can then heal. Shape:
    # [{"topic": "the funding thing", "noted": "<ISO date>", "count": int}].
    #
    # Deliberately NOT merged into last_felt, which is how the USER felt. This
    # is how a TOPIC went between them, and merging the two makes "the user was
    # frustrated about X" indistinguishable from "talking about X went badly" —
    # those want opposite prompts (empathy vs. care). Note there is no field
    # here that could hold "she is upset with the user": friction is
    # information she HAS, never a mood she is IN, and that is enforced by the
    # data model rather than by prompt wording.
    friction_topics: List[Dict[str, Any]] = field(default_factory=list)
    # Relationship state carried across sessions (v2). `mood` is stored
    # raw — persona.KurisuState.normalized_mood() sanitizes at hydration, so
    # this module stays ignorant of the persona layer. `trust_level` only
    # changes at session boundaries via updated_trust().
    mood: Optional[str] = None
    trust_level: float = DEFAULT_TRUST
    # Long-arc relationship dials (v3), raw floats — persona-ignorant, same
    # rationale as `mood` above: this module never interprets them, it just
    # persists what updated_arc() computes at session boundaries.
    arc_curiosity_vs_annoyance: float = 0.0
    arc_logic_vs_emotion: float = 0.0
    arc_pride_vs_vulnerability: float = 0.0
    # Per-session tone samples, newest last: [{"date": "<ISO date>", "tone": float}].
    # Capped at TONE_HISTORY_CAP entries; feeds tone_trend() for the prompt.
    tone_history: List[Dict[str, Any]] = field(default_factory=list)
    # The ending tone of the most recent session, awaiting consumption at the
    # next session boundary (v5): {"tone": float, "date": "<ISO date>"}.
    # Session boundaries mostly fire at the NEXT process construction, where
    # the in-memory tone from the extraction job is long gone — without this
    # slot the tone-driven trust/arc drift and tone_history never moved on
    # that (dominant) path. Cleared exactly once when a boundary consumes it.
    last_session_tone: Optional[Dict[str, Any]] = None
    # The one open topic noted as the user said goodbye (v6), awaiting the
    # next greeting: {"text": "they were about to submit the manuscript",
    # "noted": "<ISO datetime>"}. Written by the farewell path, folded into
    # the next greeting, and cleared exactly once AFTER a greeting actually
    # lands (consume-after-success, same contract as the daily brief).
    next_time_thread: Optional[Dict[str, str]] = None
    # Threads the greeting has already asked about once — asking about the
    # same interview five sessions running reads as a script, not care.
    followed_up_threads: List[str] = field(default_factory=list)

    @classmethod
    def load(cls, path: Path) -> "UserProfile":
        try:
            with path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
        except FileNotFoundError:
            # Normal on a fresh install — not a failure.
            return cls()
        except (json.JSONDecodeError, OSError) as exc:
            # A corrupt profile silently returned as blank is an identity
            # wipe: the next save overwrites the evidence, stamps a fresh
            # first_met, and re-triggers the first-meeting greeting. Keep the
            # unreadable file and degrade loudly instead.
            cls._quarantine_corrupt(path, exc)
            return cls()
        if not isinstance(data, dict):
            cls._quarantine_corrupt(path, "not a JSON object")
            return cls()
        try:
            return cls._from_data(data)
        except (TypeError, AttributeError, ValueError, KeyError) as exc:
            # Valid JSON with a garbage shape (a string session_count, a
            # scalar where a dict is expected, ...) used to raise out of
            # load and crash construction — same quarantine contract as
            # unparseable JSON: keep the evidence, start blank.
            cls._quarantine_corrupt(path, exc)
            return cls()

    @classmethod
    def _from_data(cls, data: Dict[str, Any]) -> "UserProfile":
        """Build a profile from a loaded JSON dict. Field-level tolerance
        (bad tone entries skipped, arcs clamped, ...) stays here; anything
        that still raises is treated as corruption by ``load``."""
        facts = data.get("facts") or {}
        if not isinstance(facts, dict):
            facts = {}
        threads = data.get("open_threads") or []
        if not isinstance(threads, list):
            threads = []
        # Accept both the legacy plain-string form and the dated
        # {"text": ..., "noted": ...} form, so old profiles keep loading.
        texts: List[str] = []
        dates: Dict[str, str] = {}
        for entry in threads:
            if isinstance(entry, dict):
                text = str(entry.get("text") or "").strip()
                noted = str(entry.get("noted") or "").strip()
            else:
                text = str(entry).strip()
                noted = ""
            if not text:
                continue
            texts.append(text)
            if noted:
                dates[text] = noted
        try:
            trust = float(data.get("trust_level", DEFAULT_TRUST))
        except (TypeError, ValueError):
            trust = DEFAULT_TRUST
        mood = data.get("mood")

        def _clamped_arc(key: str) -> float:
            try:
                value = float(data.get(key, 0.0))
            except (TypeError, ValueError):
                return 0.0
            return max(-1.0, min(1.0, value))

        tone_entries = data.get("tone_history") or []
        if not isinstance(tone_entries, list):
            tone_entries = []
        tone_history: List[Dict[str, Any]] = []
        for entry in tone_entries:
            if not isinstance(entry, dict):
                continue
            date = entry.get("date")
            if not isinstance(date, str) or not date:
                continue
            try:
                tone_value = float(entry.get("tone"))
            except (TypeError, ValueError):
                continue
            tone_history.append({"date": date, "tone": max(-1.0, min(1.0, tone_value))})
        tone_history = tone_history[-TONE_HISTORY_CAP:]

        last_felt = _sanitize_felt(data.get("last_felt"))

        friction: List[Dict[str, Any]] = []
        for entry in data.get("friction_topics") or []:
            cleaned = _sanitize_friction(entry)
            if cleaned is not None:
                friction.append(cleaned)
        friction = friction[-FRICTION_TOPICS_CAP:]

        session_tone: Optional[Dict[str, Any]] = None
        raw_session_tone = data.get("last_session_tone")
        if isinstance(raw_session_tone, dict):
            try:
                tone_val = max(-1.0, min(1.0, float(raw_session_tone.get("tone"))))
            except (TypeError, ValueError):
                tone_val = None
            if tone_val is not None:
                session_tone = {"tone": tone_val}
                date = raw_session_tone.get("date")
                if isinstance(date, str) and date:
                    session_tone["date"] = date[:10]

        next_time: Optional[Dict[str, str]] = None
        raw_next_time = data.get("next_time_thread")
        if isinstance(raw_next_time, dict):
            thread_text = str(raw_next_time.get("text") or "").strip()[
                :NEXT_TIME_THREAD_MAX_CHARS
            ]
            if thread_text:
                next_time = {"text": thread_text}
                noted = str(raw_next_time.get("noted") or "").strip()
                if noted:
                    next_time["noted"] = noted[:19]

        return cls(
            facts={str(k): str(v) for k, v in facts.items()},
            first_met=data.get("first_met"),
            session_count=int(data.get("session_count") or 0),
            last_seen=data.get("last_seen"),
            open_threads=texts,
            open_thread_dates=dates,
            last_episode=data.get("last_episode"),
            last_felt=last_felt,
            friction_topics=friction,
            mood=str(mood) if mood else None,
            trust_level=max(0.0, min(1.0, trust)),
            arc_curiosity_vs_annoyance=_clamped_arc("arc_curiosity_vs_annoyance"),
            arc_logic_vs_emotion=_clamped_arc("arc_logic_vs_emotion"),
            arc_pride_vs_vulnerability=_clamped_arc("arc_pride_vs_vulnerability"),
            tone_history=tone_history,
            last_session_tone=session_tone,
            next_time_thread=next_time,
            followed_up_threads=[
                str(t)
                for t in (data.get("followed_up_threads") or [])
                if str(t).strip()
            ],
        )

    @staticmethod
    def _quarantine_corrupt(path: Path, reason: Any) -> None:
        telemetry.incr("memory.profile_load_corrupt")
        backup = path.with_name(path.name + ".bak-corrupt")
        try:
            os.replace(path, backup)
            LOGGER.error(
                "User profile %s is unreadable (%s); starting with a blank "
                "profile. The corrupt file was kept at %s",
                path,
                reason,
                backup,
            )
        except OSError:
            LOGGER.error(
                "User profile %s is unreadable (%s) and could not be backed "
                "up; starting with a blank profile",
                path,
                reason,
            )

    def save_atomic(self, path: Path) -> None:
        if self.first_met is None:
            self.first_met = dt.date.today().isoformat()
        payload = {
            "version": PROFILE_VERSION,
            "first_met": self.first_met,
            "session_count": self.session_count,
            "last_seen": self.last_seen,
            "facts": self.facts,
            "open_threads": [
                {"text": text, "noted": self.open_thread_dates.get(text, "")}
                for text in self.open_threads
            ],
            "last_episode": self.last_episode,
            "last_felt": self.last_felt,
            "friction_topics": self.friction_topics,
            "mood": self.mood,
            "trust_level": self.trust_level,
            "arc_curiosity_vs_annoyance": self.arc_curiosity_vs_annoyance,
            "arc_logic_vs_emotion": self.arc_logic_vs_emotion,
            "arc_pride_vs_vulnerability": self.arc_pride_vs_vulnerability,
            "tone_history": self.tone_history,
            "last_session_tone": self.last_session_tone,
            "next_time_thread": self.next_time_thread,
            "followed_up_threads": self.followed_up_threads,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_name(path.name + ".tmp")
        with tmp_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        os.replace(tmp_path, path)

    def merge_updates(self, updates: Dict[str, str], max_facts: int) -> None:
        """Apply extraction-job profile updates, keeping the fact list small."""
        for key, value in updates.items():
            key = str(key).strip()
            value = str(value).strip()
            if not key or not value:
                continue
            if value.lower() in {"none", "null", "unknown", "n/a"}:
                self.facts.pop(key, None)
                continue
            if key not in self.facts and len(self.facts) >= max_facts:
                # Drop the oldest insertion to stay in budget (dict preserves
                # insertion order).
                oldest = next(iter(self.facts))
                self.facts.pop(oldest)
            self.facts[key] = value

    def set_open_threads(self, threads: List[str], max_threads: int) -> None:
        """Replace the thread list with the extraction job's updated view.

        An empty list is a real update ("everything got resolved"), not a
        no-op — the extraction prompt sees the current threads and returns
        the full carried-over list, so whatever it drops is resolved.
        Carried-over threads keep their original noted date (matched
        case-insensitively) so age phrasing stays truthful.
        """
        cleaned = [str(t).strip() for t in threads if str(t).strip()][:max_threads]
        previous = {text.lower(): date for text, date in self.open_thread_dates.items()}
        today = dt.date.today().isoformat()
        self.open_threads = cleaned
        self.open_thread_dates = {
            text: previous.get(text.lower(), today) for text in cleaned
        }
        # A resolved (dropped) thread frees its follow-up slot; if the same
        # text ever reappears it is genuinely new again.
        still_open = {text.lower() for text in cleaned}
        self.followed_up_threads = [
            t for t in self.followed_up_threads if t.lower() in still_open
        ]

    def prune_stale_threads(self, max_age_days: int) -> None:
        """Drop threads first noted more than ``max_age_days`` ago — asking
        about a weeks-dead topic reads as creepy, not attentive."""
        if max_age_days <= 0 or not self.open_threads:
            return
        cutoff = dt.date.today() - dt.timedelta(days=max_age_days)
        kept: List[str] = []
        for text in self.open_threads:
            noted = self._parse_noted(self.open_thread_dates.get(text))
            if noted is None or noted >= cutoff:
                kept.append(text)
            else:
                self.open_thread_dates.pop(text, None)
        self.open_threads = kept

    def thread_age_days(self, text: str) -> Optional[int]:
        noted = self._parse_noted(self.open_thread_dates.get(text))
        if noted is None:
            return None
        return max(0, (dt.date.today() - noted).days)

    @staticmethod
    def _parse_noted(raw: Optional[str]) -> Optional[dt.date]:
        if not raw:
            return None
        try:
            return dt.date.fromisoformat(raw[:10])
        except ValueError:
            return None

    def followup_thread(self) -> Optional[str]:
        """The ripest open thread to actively ask about, or None.

        Eligible: noted at least a day ago (same-day threads are still the
        current conversation, not a follow-up) and not already asked about.
        Oldest first — the longer something has hung open, the more the
        ask lands as genuine attention.
        """
        best: Optional[str] = None
        best_age = 0
        for text in self.open_threads:
            if any(text.lower() == t.lower() for t in self.followed_up_threads):
                continue
            age = self.thread_age_days(text)
            if age is None or age < 1:
                continue
            if age > best_age:
                best, best_age = text, age
        return best

    def mark_thread_followed_up(self, text: str) -> None:
        if not any(text.lower() == t.lower() for t in self.followed_up_threads):
            self.followed_up_threads.append(text)

    def describe_thread(self, text: str) -> str:
        """Thread text plus a coarse age tag when it isn't fresh."""
        age = self.thread_age_days(text)
        if age is None or age <= 1:
            return text
        if age < 7:
            return f"{text} ({age} days ago)"
        if age < 14:
            return f"{text} (about a week ago)"
        return f"{text} (a while ago)"

    def record_tone(self, tone: float, when: Optional[dt.date] = None) -> None:
        """Append a per-session tone sample, trimmed to TONE_HISTORY_CAP.

        Pure state mutation — no disk I/O; the caller is responsible for
        calling save_atomic() afterward.
        """
        try:
            clamped = max(-1.0, min(1.0, float(tone)))
        except (TypeError, ValueError):
            clamped = 0.0
        day = when or dt.date.today()
        self.tone_history.append({"date": day.isoformat(), "tone": clamped})
        if len(self.tone_history) > TONE_HISTORY_CAP:
            self.tone_history = self.tone_history[-TONE_HISTORY_CAP:]

    def set_session_tone(self, tone: float, when: Optional[dt.date] = None) -> None:
        """Stash the session's current ending tone for the next boundary.

        Overwritten as the session's tone estimate refines; consumed (and
        cleared) exactly once by the next session-boundary update. Pure state
        mutation — the caller owns save_atomic().
        """
        try:
            clamped = max(-1.0, min(1.0, float(tone)))
        except (TypeError, ValueError):
            return
        day = when or dt.date.today()
        self.last_session_tone = {"tone": clamped, "date": day.isoformat()}

    def pop_session_tone(self) -> Tuple[Optional[float], Optional[dt.date]]:
        """Consume the stashed session tone (value, date) — clears the slot."""
        stored = self.last_session_tone
        self.last_session_tone = None
        if not isinstance(stored, dict):
            return None, None
        try:
            tone = max(-1.0, min(1.0, float(stored.get("tone"))))
        except (TypeError, ValueError):
            return None, None
        when: Optional[dt.date] = None
        raw_date = stored.get("date")
        if isinstance(raw_date, str):
            try:
                when = dt.date.fromisoformat(raw_date[:10])
            except ValueError:
                when = None
        return tone, when

    def set_next_time_thread(
        self, text: Optional[str], when: Optional[dt.datetime] = None
    ) -> None:
        """Stash the one open topic to pick up next session (blank clears).

        Written by the farewell path after a real goodbye. Single slot —
        only the most recent farewell's note survives. Pure state mutation;
        the caller owns persistence.
        """
        cleaned = " ".join(str(text or "").split())[:NEXT_TIME_THREAD_MAX_CHARS]
        if not cleaned:
            self.next_time_thread = None
            return
        stamp = (when or dt.datetime.utcnow()).isoformat(timespec="seconds")
        self.next_time_thread = {"text": cleaned, "noted": stamp}

    def take_next_time_thread(
        self, now: Optional[dt.datetime] = None
    ) -> Optional[Dict[str, Any]]:
        """The pending next-time thread for the greeting, age-classified.

        Returns ``{"text": ..., "stale": bool}`` — ``stale`` past
        NEXT_TIME_THREAD_STALE_DAYS (or when the timestamp is unverifiable),
        meaning the greeting should phrase it tentatively. Entries older than
        NEXT_TIME_THREAD_MAX_AGE_DAYS are dropped unconsumed: the slot is
        cleared and None returned (the caller persists the drop). A live
        entry is NOT cleared here — consumption is
        :meth:`clear_next_time_thread`, run only after a greeting actually
        landed (consume-after-success).
        """
        entry = self.next_time_thread
        if not isinstance(entry, dict) or not str(entry.get("text") or "").strip():
            return None
        text = str(entry["text"]).strip()
        noted: Optional[dt.datetime] = None
        try:
            noted = dt.datetime.fromisoformat(str(entry.get("noted") or "")[:19])
        except ValueError:
            noted = None
        if noted is None:
            # Unverifiable age can't confirm freshness: deliver once, but
            # tentatively (fail toward caution, never toward false recency).
            return {"text": text, "stale": True}
        age_days = ((now or dt.datetime.utcnow()) - noted).total_seconds() / 86400.0
        if age_days > NEXT_TIME_THREAD_MAX_AGE_DAYS:
            self.next_time_thread = None
            return None
        return {"text": text, "stale": age_days > NEXT_TIME_THREAD_STALE_DAYS}

    def clear_next_time_thread(self) -> None:
        """Consume the next-time thread (after a greeting delivered it)."""
        self.next_time_thread = None

    def record_felt(
        self,
        label: Optional[str],
        subject: Optional[str] = None,
        when: Optional[dt.date] = None,
    ) -> None:
        """Store the most recent session's affective tag, overwriting the last.

        Only the newest session's felt-tone is kept (a single slot, unlike the
        rolling tone_history): the greeting surfaces "that last talk where you
        were …", not a history. A blank/unusable label clears the slot. Pure
        state mutation — the caller owns save_atomic().
        """
        day = when or dt.date.today()
        self.last_felt = _sanitize_felt(
            {"label": label, "subject": subject, "date": day.isoformat()}
        )

    def felt_fact(
        self, today: Optional[dt.date] = None, max_age_days: int = 30
    ) -> Optional[str]:
        """One structured greeting fact for how the last session FELT, or None.

        Facts-not-scripts: this hands the LLM the affect + subject and the
        anti-guilt constraint, never a canned line. Suppressed when the tag is
        missing or older than ``max_age_days`` (a weeks-stale "you were excited"
        reads as canned, not attentive). The date is best-effort — an
        unparseable/absent date is treated as current rather than dropped.
        """
        felt = self.last_felt
        if not felt or not felt.get("label"):
            return None
        raw_date = felt.get("date")
        if raw_date:
            noted = self._parse_noted(raw_date)
            if noted is not None:
                reference = today or dt.date.today()
                if (reference - noted).days > max(0, max_age_days):
                    return None
        label = felt["label"]
        subject = felt.get("subject")
        core = (
            f"the user seemed {label} about {subject}"
            if subject
            else f"the user seemed {label}"
        )
        return (
            "Emotional continuity — last time you talked, "
            f"{core}. You may acknowledge that felt-tone in ONE light, "
            'specific touch if it fits the moment ("that talk where you '
            'were…"), then move on. Never guilt them, never dwell, never '
            "recite this."
        )

    def record_friction(self, topic: str, when: Optional[dt.date] = None) -> None:
        """Note that a topic went badly between them.

        Re-noting an existing topic (matched case-insensitively) bumps its
        count and date rather than appending a second entry — a recurring
        sore spot is one topic, not three. Evicts the oldest when over
        FRICTION_TOPICS_CAP. A blank topic is a no-op. Pure state mutation;
        the caller owns save_atomic().
        """
        cleaned = str(topic or "").strip()[:FRICTION_TOPIC_MAX_CHARS]
        if not cleaned:
            return
        day = (when or dt.date.today()).isoformat()
        needle = cleaned.lower()
        for entry in self.friction_topics:
            if str(entry.get("topic", "")).strip().lower() == needle:
                entry["count"] = int(entry.get("count") or 1) + 1
                entry["noted"] = day
                return
        self.friction_topics.append({"topic": cleaned, "noted": day, "count": 1})
        if len(self.friction_topics) > FRICTION_TOPICS_CAP:
            dropped = len(self.friction_topics) - FRICTION_TOPICS_CAP
            del self.friction_topics[:dropped]
            telemetry.incr("memory.friction_evicted_cap", dropped)

    def clear_friction(self, topic: str) -> bool:
        """Drop a repaired topic. True when something was actually cleared.

        Repair is never transactional: the topic going FINE once is enough,
        and no apology is required or recorded anywhere.
        """
        needle = str(topic or "").strip().lower()
        if not needle:
            return False
        before = len(self.friction_topics)
        self.friction_topics = [
            entry
            for entry in self.friction_topics
            if str(entry.get("topic", "")).strip().lower() != needle
        ]
        return len(self.friction_topics) < before

    def prune_stale_friction(
        self,
        max_age_days: int = FRICTION_MAX_AGE_DAYS,
        today: Optional[dt.date] = None,
    ) -> int:
        """Drop friction older than ``max_age_days``, returning how many went.

        Time alone repairs, and this is not optional. A sore topic that only
        the user can clear is a debt she is holding them to, which is exactly
        the retention pattern docs/companion-ethics.md refuses. An entry with
        an unparseable/absent date is treated as current rather than dropped,
        matching felt_fact's best-effort stance.
        """
        reference = today or dt.date.today()
        kept: List[Dict[str, Any]] = []
        for entry in self.friction_topics:
            noted = self._parse_noted(str(entry.get("noted") or ""))
            if noted is not None and (reference - noted).days > max(0, max_age_days):
                continue
            kept.append(entry)
        dropped = len(self.friction_topics) - len(kept)
        self.friction_topics = kept
        return dropped

    def friction_fact(
        self,
        today: Optional[dt.date] = None,
        max_age_days: int = FRICTION_MAX_AGE_DAYS,
    ) -> Optional[str]:
        """One structured greeting fact about a live sore topic, or None.

        Facts-not-scripts, same contract as felt_fact: hands the LLM the topic
        plus the hard anti-guilt rails, never a canned line. The most recently
        noted live topic wins; only one ever surfaces.

        Every DO NOT below maps to a named dark pattern, and the first one is
        the load-bearing rule: friction is REACTIVE. She does not raise the
        sore subject — she is better at it when they do.
        """
        reference = today or dt.date.today()
        live: List[Dict[str, Any]] = []
        for entry in self.friction_topics:
            if not str(entry.get("topic") or "").strip():
                continue
            noted = self._parse_noted(str(entry.get("noted") or ""))
            if noted is not None and (reference - noted).days > max(0, max_age_days):
                continue
            live.append(entry)
        if not live:
            return None
        newest = max(live, key=lambda e: str(e.get("noted") or ""))
        topic = str(newest["topic"]).strip()
        return (
            f"Continuity — last time you two talked about {topic}, it did not "
            "go well between you (a real misunderstanding, not just a hard "
            "subject). You remember that. If it comes up, be a little more "
            "careful and a little more direct about it than usual — you are "
            "willing to get it right this time. Do NOT bring it up yourself "
            "unless they do. Do NOT be cold, distant, short, or wary with "
            "them. Do NOT reference it as something they did to you, do NOT "
            "ask for or imply an apology, do NOT sulk or make them manage "
            "your feelings. They owe you nothing. If they seem fine, you are "
            "fine — open warm exactly as you normally would."
        )

    def render_block(self, max_facts: int = 12) -> Optional[str]:
        """Render the <=150-word prompt block, or None when nothing is known."""
        if not self.facts and not self.last_episode and not self.open_threads:
            return None
        lines: List[str] = [
            "What you know about this user (long-term memory — use it naturally, never recite this list):"
        ]
        for key, value in list(self.facts.items())[:max_facts]:
            lines.append(f"- {key}: {value}")
        if self.session_count and self.first_met:
            lines.append(
                f"You have talked {self.session_count} times since {self.first_met}."
            )
        if self.last_episode:
            lines.append(f"Last time: {self.last_episode}")
        if self.open_threads:
            lines.append(
                "Unresolved threads you may bring up naturally if the moment fits (never force it): "
                + "; ".join(self.describe_thread(text) for text in self.open_threads)
            )
        return "\n".join(lines)


def early_acquaintance_active(
    session_count: int,
    fact_count: int,
    *,
    max_sessions: int,
    max_facts: int,
) -> bool:
    """Whether the getting-to-know-you prompt nudge should still run.

    Active only while BOTH counters are under their caps: the fact cap ends
    the phase the natural way (she now knows enough), and the session cap is
    the consent valve — a user who deflects every question stops being asked
    after max_sessions conversations even with an empty profile. Either cap
    at 0 disables the feature outright.
    """
    try:
        sessions = int(session_count)
        facts = int(fact_count)
        cap_sessions = int(max_sessions)
        cap_facts = int(max_facts)
    except (TypeError, ValueError):
        return False
    if cap_sessions <= 0 or cap_facts <= 0:
        return False
    return sessions < cap_sessions and facts < cap_facts


def familiarity_guardedness(base_tsundere: float, session_count: int) -> float:
    """Map relationship familiarity to emotional guardedness.

    Starts at the configured tsundere level and warms up ~0.03 per session,
    floored at 0.25 so Kurisu never fully drops her reserve. With a 0.6-0.8
    base this crosses the 0.70 warmth threshold used by the prompt builder
    within the first handful of sessions.
    """
    try:
        base = float(base_tsundere)
    except (TypeError, ValueError):
        base = 0.5
    sessions = max(0, int(session_count))
    return max(0.25, min(1.0, base - 0.03 * min(sessions, 15)))


def updated_trust(
    current: float, session_count: int, last_tone: Optional[float]
) -> float:
    """Once-per-session trust update from existing signals — no LLM calls.

    Trust gates the prompt builder's teasing budget (thresholds at 0.60/0.80),
    so it ramps deliberately slower than familiarity_guardedness: a floor that
    grows ~0.015/session (capped at +0.25 ≈ 17 sessions), a small bonus for a
    warm previous session and a milder penalty for a hostile one, clamped to
    [floor, 0.9] — deep trust stays asymptotic. Stored at full precision so
    the small tone bonuses accumulate; quantize_trust() coarsens it at the
    prompt boundary.
    """
    try:
        trust = float(current)
    except (TypeError, ValueError):
        trust = DEFAULT_TRUST
    floor = 0.5 + min(0.25, 0.015 * max(0, int(session_count)))
    bonus = 0.0
    if last_tone is not None:
        if last_tone >= 0.15:
            bonus = 0.01
        elif last_tone <= -0.5:
            bonus = -0.005
    trust = max(max(trust + bonus, floor), 0.0)
    return min(trust, 0.9)


def quantize_trust(trust: float) -> float:
    """Coarsen trust to 0.05 steps for the prompt, so the dynamic prompt tail
    is stable within a conversation instead of churning on float noise."""
    try:
        value = float(trust)
    except (TypeError, ValueError):
        value = DEFAULT_TRUST
    return round(max(0.0, min(1.0, value)) * 20) / 20


def greeting_register_facts(
    local_now: dt.datetime,
    utc_now: dt.datetime,
    last_seen: Optional[dt.datetime],
    asleep: bool = False,
) -> List[str]:
    """Time-of-day and welcome-back register lines for the opening greeting.

    Always emits one time-of-day line so the greeting's tone can take a light
    cue from the hour without reciting it. When ``last_seen`` shows a gap of
    48h or more, also emits an absence line — deliberately blunt about the
    anti-guilt constraint, since this fact is fed to an LLM that will
    otherwise reach for "I missed you" as the easy warm beat. Pure function,
    no I/O: callers own how ``local_now``/``utc_now`` are sourced.
    """
    hour = local_now.hour
    if 5 <= hour <= 8:
        bucket = "early morning"
    elif 9 <= hour <= 11:
        bucket = "morning"
    elif 12 <= hour <= 17:
        bucket = "afternoon"
    elif 18 <= hour <= 22:
        bucket = "evening"
    else:
        bucket = "late night"
    time_of_day_line = (
        f"Register: it is {bucket}. Let the time of day color your tone, "
        "at most one light nod to it."
    )
    if bucket == "late night":
        time_of_day_line += (
            " If they are up unusually late, one dry observation is fine — "
            "never lecture them about sleep."
        )
    facts = [time_of_day_line]
    # Sleep window (amadeus/sleep_schedule.py, caller-evaluated): opening a
    # session mid-window means the user's arrival woke her. One groggy beat,
    # then normal — and explicitly never a lecture or a guilt line.
    if asleep:
        facts.append(
            "Register: it is the middle of YOUR night and you were asleep — "
            "their arrival just woke you. Open visibly groggy and "
            "half-awake (drowsy, a little disoriented, dry about it), then "
            "engage normally. Never lecture them about the hour and never "
            "make them feel bad for waking you."
        )
    if last_seen is not None:
        gap_hours = (utc_now - last_seen).total_seconds() / 3600
        if gap_hours >= 48:
            days = round(gap_hours / 24)
            if days < 14:
                away = f"{days} days"
            else:
                weeks = round(days / 7)
                away = f"about {weeks} weeks"
            facts.append(
                f"Register: the user is returning after {away} away. Open "
                "warm and glad to see them. NEVER guilt them, never mention "
                "being lonely or waiting, never imply they owe you their "
                "time."
            )
    return facts


def updated_arc(
    curiosity: float,
    logic: float,
    pride: float,
    *,
    session_count: int,
    trust_level: float,
    last_tone: Optional[float],
) -> Tuple[float, float, float]:
    """Once-per-session drift of the long-arc dials — "the tsundere thaw".

    Unlike trust/mood, these three floats never reset within a conversation;
    they only move at session boundaries, and by small clamped steps, so the
    character arc reads as a slow multi-week thaw rather than a mood swing.

    - A warm previous session (``last_tone >= 0.15``) nudges her toward
      curiosity (+0.02) and softens the logic/pride armor a touch (-0.01
      each, drifting toward emotion/vulnerability).
    - A hostile previous session (``last_tone <= -0.5``) pushes the other
      way: curiosity down (-0.02) and pride back up (+0.01) — the guard
      goes back up.
    - Independently of tone, sustained high trust (``trust_level >= 0.7``)
      slowly melts the armor regardless of how the last session went:
      pride -0.005, logic -0.005. This stacks with the tone deltas above.
    - With no tone signal (``last_tone is None``) and trust below 0.7,
      nothing moves — there's no signal to drift on.

    All three values are clamped to [-1.0, 1.0]; malformed inputs are
    coerced via ``float()`` and default to 0.0 on failure.
    """

    def _coerced(value: float) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    curiosity = _coerced(curiosity)
    logic = _coerced(logic)
    pride = _coerced(pride)

    try:
        trust = float(trust_level)
    except (TypeError, ValueError):
        trust = DEFAULT_TRUST

    if last_tone is not None:
        if last_tone >= 0.15:
            curiosity += 0.02
            logic -= 0.01
            pride -= 0.01
        elif last_tone <= -0.5:
            curiosity -= 0.02
            pride += 0.01

    if trust >= 0.7:
        pride -= 0.005
        logic -= 0.005

    def _clamped(value: float) -> float:
        return max(-1.0, min(1.0, value))

    return _clamped(curiosity), _clamped(logic), _clamped(pride)


def tone_trend(
    history: List[Dict[str, Any]],
    *,
    today: Optional[dt.date] = None,
    window_days: int = 7,
    min_entries: int = 3,
    low_threshold: float = -0.15,
    high_threshold: float = 0.30,
) -> Optional[str]:
    """Summarize recent tone_history into "low" / "positive" / None.

    Filters entries to those with a parseable ISO date within
    ``window_days`` of ``today`` (inclusive), ignoring anything malformed.
    Returns None when fewer than ``min_entries`` survive the filter (not
    enough signal to call a trend), otherwise buckets the mean tone against
    the two thresholds. Never raises on garbage input — unparseable entries
    are simply skipped.
    """
    reference = today or dt.date.today()
    cutoff = reference - dt.timedelta(days=max(0, window_days))
    values: List[float] = []
    for entry in history or []:
        if not isinstance(entry, dict):
            continue
        raw_date = entry.get("date")
        if not isinstance(raw_date, str):
            continue
        try:
            entry_date = dt.date.fromisoformat(raw_date[:10])
        except ValueError:
            continue
        if entry_date < cutoff or entry_date > reference:
            continue
        try:
            tone = float(entry.get("tone"))
        except (TypeError, ValueError):
            continue
        values.append(tone)

    if len(values) < min_entries:
        return None

    mean_tone = sum(values) / len(values)
    if mean_tone <= low_threshold:
        return "low"
    if mean_tone >= high_threshold:
        return "positive"
    return None


_SESSION_MILESTONES = (10, 25, 50, 100, 250)


def _months_after(date: dt.date, months: int) -> dt.date:
    """Add ``months`` calendar months to ``date``, clamping to the last valid
    day of the target month (e.g. Jan 31 + 1 month -> Feb 28/29)."""
    total_months = date.month - 1 + months
    year = date.year + total_months // 12
    month = total_months % 12 + 1
    last_day = calendar.monthrange(year, month)[1]
    day = min(date.day, last_day)
    return dt.date(year, month, day)


# Month names for the tolerant birthday parser: full and three-letter
# forms, English only (fact extraction writes English values).
_MONTH_NAMES = {
    name.lower(): index for index, name in enumerate(calendar.month_name) if name
}
_MONTH_NAMES.update(
    (name.lower(), index) for index, name in enumerate(calendar.month_abbr) if name
)

# Fact keys that plausibly hold the user's birthday. Fact keys come from
# free-form LLM extraction, so this is substring matching, not equality.
_BIRTHDAY_KEY_TOKENS = ("birthday", "birthdate", "birth date", "date of birth")


def parse_birthday(text: Optional[str]) -> Optional[Tuple[int, int]]:
    """Best-effort (month, day) from a birthday-ish string, else None.

    Accepts "07-25", "7/25", ISO dates ("1992-07-25" — the year is
    ignored), and English month-name forms ("July 25", "25 July",
    "July 25th, 1992"). Values come from character packs and from LLM
    fact extraction, so this never raises — anything unparseable or not a
    real calendar date returns None.
    """
    if not text:
        return None
    cleaned = str(text).strip().lower()
    if not cleaned:
        return None
    month: Optional[int] = None
    day: Optional[int] = None
    # ISO-style or numeric: take the last two of Y-M-D, or M-D as given.
    numeric = re.findall(r"\d{1,4}", cleaned)
    for token in re.findall(r"[a-z]+", cleaned):
        if token in _MONTH_NAMES:
            month = _MONTH_NAMES[token]
            break
    if month is not None:
        # Month came from a name: the day is the first 1-2 digit number
        # (skips a 4-digit year wherever it sits).
        for value in numeric:
            if len(value) <= 2:
                day = int(value)
                break
    elif len(numeric) >= 3 and len(numeric[0]) == 4:
        month, day = int(numeric[1]), int(numeric[2])  # Y-M-D
    elif len(numeric) >= 2:
        month, day = int(numeric[0]), int(numeric[1])  # M-D
    if month is None or day is None:
        return None
    try:
        dt.date(2000, month, day)  # leap year, so Feb 29 validates
    except ValueError:
        return None
    return (month, day)


def user_birthday_from_facts(
    facts: Optional[Dict[str, str]],
) -> Optional[Tuple[int, int]]:
    """The user's (month, day) birthday from extracted profile facts.

    Scans fact KEYS for birthday-ish tokens and parses the first value
    that yields a real date — fact keys are free-form LLM output, so
    exact-key lookup would silently miss "user birthday" vs "birthday".
    """
    if not facts:
        return None
    for key, value in facts.items():
        lowered = str(key).lower()
        if any(token in lowered for token in _BIRTHDAY_KEY_TOKENS):
            parsed = parse_birthday(value)
            if parsed is not None:
                return parsed
    return None


def milestone_events(
    session_count: int,
    first_met: Optional[str],
    today: Optional[dt.date] = None,
    *,
    character_birthday: Optional[Tuple[int, int]] = None,
    user_birthday: Optional[Tuple[int, int]] = None,
) -> List[Dict[str, Any]]:
    """Return zero or more structured milestone events for today.

    Session-count milestones fire on exact equality (never raises even if
    session_count skips past one). Anniversary milestones are computed from
    ``first_met`` (an ISO date, optionally with a time component — only the
    date prefix is parsed; unparseable values are skipped, never raised):
    the 1-month and 6-month marks, plus every whole year, each checked
    against ``today`` for an exact match.

    Birthday events fire when ``today`` matches the given (month, day) —
    the caller resolves whose birthdays these are (engine: built-in Kurisu
    gets July 25, custom packs only what their manifest sets; the user's
    comes from extracted facts). Like anniversaries they fire on every
    greeting that day; the greeting freshness gap keeps that to ~one.

    Each event is one of:
    - {"kind": "sessions", "value": N, "label": <human phrase>}
    - {"kind": "anniversary", "months": M, "label": ..., "date": <iso date>}
    - {"kind": "user_birthday" | "character_birthday", "label": ...}
    """
    events: List[Dict[str, Any]] = []
    reference = today or dt.date.today()
    today_pair = (reference.month, reference.day)
    if user_birthday == today_pair:
        events.append(
            {
                "kind": "user_birthday",
                "label": (
                    "today is the user's BIRTHDAY — wish them a happy "
                    "birthday once, briefly, in your own voice"
                ),
            }
        )
    if character_birthday == today_pair:
        events.append(
            {
                "kind": "character_birthday",
                "label": (
                    "today is your OWN birthday — you may mention it at "
                    "most once, offhandedly; deflect if congratulated, "
                    "but let it visibly please you"
                ),
            }
        )
    try:
        count = int(session_count)
    except (TypeError, ValueError):
        count = None
    if count is not None and count in _SESSION_MILESTONES:
        events.append(
            {
                "kind": "sessions",
                "value": count,
                "label": f"this is conversation number {count} together",
            }
        )

    if not first_met:
        return events
    try:
        met_date = dt.date.fromisoformat(str(first_met)[:10])
    except ValueError:
        return events

    if _months_after(met_date, 1) == reference:
        events.append(
            {
                "kind": "anniversary",
                "months": 1,
                "label": "today marks 1 month since you first met",
                "date": reference.isoformat(),
            }
        )
    if _months_after(met_date, 6) == reference:
        events.append(
            {
                "kind": "anniversary",
                "months": 6,
                "label": "today marks 6 months since you first met",
                "date": reference.isoformat(),
            }
        )

    if reference.year > met_date.year or (
        reference.year == met_date.year
        and (reference.month, reference.day) >= (met_date.month, met_date.day)
    ):
        years = reference.year - met_date.year
        if (reference.month, reference.day) < (met_date.month, met_date.day):
            years -= 1
        if years >= 1:
            anniversary = _months_after(met_date, years * 12)
            if anniversary == reference:
                label = f"{years} year" if years == 1 else f"{years} years"
                events.append(
                    {
                        "kind": "anniversary",
                        "months": years * 12,
                        "label": f"today marks {label} since you first met",
                        "date": reference.isoformat(),
                    }
                )

    return events


def milestone_facts(
    session_count: int,
    first_met: Optional[str],
    today: Optional[dt.date] = None,
    *,
    character_birthday: Optional[Tuple[int, int]] = None,
    user_birthday: Optional[Tuple[int, int]] = None,
) -> List[str]:
    """Return zero or more one-off milestone lines for today's prompt.

    Thin wrapper over :func:`milestone_events` — kept for the greeting prompt
    assembly, which only wants the flat phrasing, not the structured shape.
    """
    return [
        f"Milestone: {event['label']}."
        for event in milestone_events(
            session_count,
            first_met,
            today,
            character_birthday=character_birthday,
            user_birthday=user_birthday,
        )
    ]
