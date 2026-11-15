"""Nightly memory consolidation: standing insights that change her behavior.

The memory tiers surface *facts* (recall) and *history* (summaries); nothing
distills them into how she should actually treat this person. This module is
that consolidation pass — a background job (``amadeus/web/scheduler.py``)
makes one cheap-model call per user per day that reads disk snapshots of the
profile (facts, tone history, arc, open threads) and produces a handful of
standing behavioral insights ("prefers short answers", "doesn't want pep
talk about deadlines, just solutions", "usually talks late at night — don't
comment on it"). The insight set is injected into every turn's system prompt
(``build_system_prompt(insights_block=...)``) at the static/dynamic cache
boundary — stable within a day, so the provider's cached prompt prefix
survives — which is what makes month three of talking to her feel different
from week one.

Deliberately reads NO live MemoryManager (same stance as ``brief.py``): a
per-user scheduler construction would fire session-boundary bookkeeping
daily. Continuity beats churn: the previous day's insights are fed back into
the generation call so stable guidance persists and only genuinely new
signal changes the set. The model may answer SKIP when there isn't enough
evidence — a young profile produces no insights, not invented ones.

Rules the prompt enforces (companion-ethics adjacent): insights are
behavioral guidance about HOW to talk, never a recital of facts (recall owns
facts), never mentioned to the user, and never invented beyond what the
listed data supports.
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

_DEFAULT_USER_KEY = "default"

# Valid insight kinds; anything else the model emits folds into "other".
INSIGHT_KINDS = ("style", "boundary", "rhythm", "topic", "other")

MAX_INSIGHTS = 7
_MAX_INSIGHT_CHARS = 200

# Bounded memory of what the user has deleted. Capped so the "don't restate
# these" prompt block can't grow without limit; oldest dismissals fall off
# first, and one that falls off can legitimately be re-derived later if the
# evidence still supports it.
MAX_DISMISSED = 20


@dataclass
class Insight:
    text: str
    kind: str = "other"

    def to_dict(self) -> Dict[str, Any]:
        return {"text": self.text, "kind": self.kind}

    @classmethod
    def from_dict(cls, data: Any) -> Optional["Insight"]:
        if not isinstance(data, dict) or not data.get("text"):
            return None
        kind = str(data.get("kind") or "other")
        return cls(
            text=str(data["text"]),
            kind=kind if kind in INSIGHT_KINDS else "other",
        )


@dataclass
class InsightSet:
    date: str  # ISO date the set was generated FOR
    insights: List[Insight] = field(default_factory=list)
    generated_at: str = ""
    # Insights the user removed from the panel. Carried FORWARD across
    # consolidations and fed back into the next prompt as "do not restate
    # these" — without that, tonight's job cheerfully regenerates what they
    # deleted this morning and the delete button looks broken, which is
    # worse than having no delete button at all.
    dismissed: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "date": self.date,
            "insights": [entry.to_dict() for entry in self.insights],
            "generated_at": self.generated_at,
            "dismissed": list(self.dismissed),
        }

    @classmethod
    def from_dict(cls, data: Any) -> Optional["InsightSet"]:
        if not isinstance(data, dict) or not data.get("date"):
            return None
        insights = [
            parsed
            for entry in (data.get("insights") or [])
            if (parsed := Insight.from_dict(entry)) is not None
        ]
        dismissed = [
            str(entry).strip()
            for entry in (data.get("dismissed") or [])
            if str(entry).strip()
        ]
        return cls(
            date=str(data["date"]),
            insights=insights,
            generated_at=str(data.get("generated_at") or ""),
            dismissed=dismissed[-MAX_DISMISSED:],
        )


class InsightStore:
    """One JSON file, one insight set per user (each consolidation replaces
    the last — the set IS the current understanding, not a log). Lock +
    reload-before-mutate + atomic writes, same stance as DailyBriefStore.
    The path handed in is already character-scoped by the caller."""

    def __init__(self, path: str) -> None:
        self.path = str(path)
        self._lock = threading.Lock()
        self._sets: Dict[str, InsightSet] = {}
        self._load()

    @staticmethod
    def _key(user_id: Optional[str]) -> str:
        return user_id or _DEFAULT_USER_KEY

    def _load(self) -> None:
        self._sets = {}
        if not os.path.exists(self.path):
            return
        try:
            with open(self.path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            for key, entry in (data.get("insights") or {}).items():
                parsed = InsightSet.from_dict(entry)
                if parsed is not None:
                    self._sets[str(key)] = parsed
        except Exception as exc:  # corrupt/unreadable — start clean, loudly
            LOGGER.warning("Could not read insights file %s: %s", self.path, exc)
            telemetry.incr("insights.store_load_failed")
            self._sets = {}

    def _save(self) -> None:
        payload = {
            "version": 1,
            "insights": {key: entry.to_dict() for key, entry in self._sets.items()},
        }
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        os.replace(tmp, self.path)

    def get(self, user_id: Optional[str]) -> Optional[InsightSet]:
        with self._lock:
            self._load()
            return self._sets.get(self._key(user_id))

    def has_for_date(self, user_id: Optional[str], date: dt.date) -> bool:
        current = self.get(user_id)
        return current is not None and current.date == date.isoformat()

    def put(self, user_id: Optional[str], insight_set: InsightSet) -> bool:
        """Store a set; refuses (returns False) when a same-date entry
        already exists — the double-generation guard.

        Dismissals from the previous set are MERGED forward: each set
        replaces the last, so without this the user's deletions would live
        exactly one day and the next consolidation would restate them.
        """
        with self._lock:
            self._load()
            existing = self._sets.get(self._key(user_id))
            if existing is not None and existing.date == insight_set.date:
                return False
            if existing is not None and existing.dismissed:
                merged = list(existing.dismissed)
                for text in insight_set.dismissed:
                    if text not in merged:
                        merged.append(text)
                insight_set.dismissed = merged[-MAX_DISMISSED:]
            self._sets[self._key(user_id)] = insight_set
            self._save()
            return True

    def dismiss(self, user_id: Optional[str], text: str) -> bool:
        """Remove one insight and remember that the user removed it.

        True when something was actually removed. The text lands in
        ``dismissed`` so the next consolidation is told not to restate it —
        see the field's comment for why that plumbing is not optional.
        """
        needle = str(text or "").strip()
        if not needle:
            return False
        with self._lock:
            self._load()
            current = self._sets.get(self._key(user_id))
            if current is None:
                return False
            kept = [entry for entry in current.insights if entry.text != needle]
            if len(kept) == len(current.insights):
                return False
            current.insights = kept
            if needle not in current.dismissed:
                current.dismissed = (current.dismissed + [needle])[-MAX_DISMISSED:]
            self._save()
            return True


# ---------------------------------------------------------------------------
# Consolidation input
# ---------------------------------------------------------------------------


def collect_insight_facts(*, profile: Any, max_facts: int = 40) -> List[str]:
    """Assemble the evidence lines the consolidation call sees. ``profile``
    is a UserProfile (or None → nothing to consolidate)."""
    if profile is None:
        return []
    facts: List[str] = []
    session_count = int(getattr(profile, "session_count", 0) or 0)
    facts.append(f"Conversations so far: {session_count}.")
    # tone_history entries are {"date": iso, "tone": -1..1} — render each as
    # a coarse label so the model reasons over words, not floats.
    tone_labels: List[str] = []
    for entry in list(getattr(profile, "tone_history", None) or [])[-12:]:
        try:
            tone_value = float(entry.get("tone"))
        except (AttributeError, TypeError, ValueError):
            continue
        tone_labels.append(
            "low" if tone_value < -0.25 else "upbeat" if tone_value > 0.25 else "even"
        )
    if tone_labels:
        facts.append("Recent session tones (oldest first): " + ", ".join(tone_labels))
    for key, value in list((getattr(profile, "facts", None) or {}).items())[:max_facts]:
        if value:
            facts.append(f"Known fact — {key}: {value}")
    for thread in list(getattr(profile, "open_threads", None) or [])[:5]:
        facts.append(f"Open thread: {thread}")
    last_episode = getattr(profile, "last_episode", None)
    if last_episode:
        facts.append(f"Last session summary: {last_episode}")
    return facts


def _has_enough_signal(facts: List[str], min_sessions: int) -> bool:
    """A young profile shouldn't burn a daily call to be told SKIP."""
    for line in facts:
        if line.startswith("Conversations so far:"):
            try:
                count = int(line.split(":", 1)[1].strip().rstrip("."))
            except ValueError:
                return False
            return count >= min_sessions
    return False


# ---------------------------------------------------------------------------
# Generation + parsing
# ---------------------------------------------------------------------------


def parse_insights_reply(text: Optional[str]) -> Optional[List[Insight]]:
    """Parse the model's ``KIND: guidance`` lines. Returns None for SKIP or
    an unusable reply (caller decides whether that's a skip or a failure);
    an empty parse of a non-SKIP reply is treated as unusable (None)."""
    if not text:
        return None
    stripped = text.strip()
    if not stripped or stripped.upper().startswith("SKIP"):
        return None
    insights: List[Insight] = []
    for raw_line in stripped.splitlines():
        line = raw_line.strip().lstrip("-*• ").strip()
        if not line:
            continue
        kind = "other"
        body = line
        if ":" in line:
            head, tail = line.split(":", 1)
            head_key = head.strip().lower()
            if head_key in INSIGHT_KINDS:
                kind = head_key
                body = tail
        body = body.strip()
        if len(body) < 8:
            continue  # fragments/noise, not guidance
        insights.append(Insight(text=body[:_MAX_INSIGHT_CHARS], kind=kind))
        if len(insights) >= MAX_INSIGHTS:
            break
    return insights or None


def generate_insights(
    *,
    provider: Any,
    facts: List[str],
    previous: Optional[InsightSet] = None,
    max_tokens: int = 260,
    min_sessions: int = 3,
    character_name: str = "Makise Kurisu",
    character: Any = None,
) -> Optional[List[Insight]]:
    """One cheap-model call distilling the evidence into standing insights.

    Return contract (the scheduler's retry behavior hangs on it):
    - list of insights — store it.
    - [] — the model legitimately answered SKIP: store an EMPTY set so the
      day is done (otherwise the job would re-call the model every tick all
      day for the same SKIP).
    - None — failure or too-little-signal (no call made): store nothing,
      the next tick re-evaluates."""
    from .providers.llm.base import ChatMessage

    if provider is None or not facts:
        return None
    if not _has_enough_signal(facts, min_sessions):
        telemetry.incr("insights.too_little_signal")
        return None
    name = (getattr(character, "full_name", None) or character_name).strip()
    kinds = "/".join(INSIGHT_KINDS[:-1])
    system = (
        f"You are the memory-consolidation process of {name}'s companion "
        "system. From the evidence lines, distill AT MOST 5 standing "
        "insights about how she should treat this specific person — "
        "behavioral guidance only (how to talk, what to avoid, pacing, "
        "recurring rhythms), NEVER a restatement of facts she already "
        "remembers, and never speculation beyond the evidence. One per "
        f"line, formatted `{kinds}: guidance` (style=register/length/humor, "
        "boundary=topics or moves to avoid, rhythm=when/how they tend to "
        "show up, topic=durable interests worth leaning into). Keep each "
        "under 20 words, imperative voice. Carry forward previous insights "
        "that still hold; drop ones the evidence no longer supports. If the "
        "evidence is too thin for even one well-supported insight, reply "
        "exactly SKIP."
    )
    user_lines = list(facts)
    if previous is not None and previous.insights:
        user_lines.append(
            "Previous insights (keep the ones that still hold): "
            + " | ".join(f"{entry.kind}: {entry.text}" for entry in previous.insights)
        )
    if previous is not None and previous.dismissed:
        # The user deleted these from their own record. Restating one would
        # make the delete button a lie, so this is a hard instruction rather
        # than evidence to weigh.
        user_lines.append(
            "The user REMOVED these from their record. Do not restate them or "
            "any close paraphrase, whatever the evidence suggests: "
            + " | ".join(previous.dismissed)
        )
    prompt = [
        ChatMessage(role="system", content=system),
        ChatMessage(role="user", content="\n".join(user_lines)),
    ]
    try:
        text = (provider.generate(prompt, max_tokens=int(max_tokens)) or "").strip()
        telemetry.record_llm_usage(provider, "insights")
    except Exception as exc:
        LOGGER.warning("Insight consolidation failed: %s", exc)
        telemetry.incr("insights.generation_failed")
        return None
    if text.upper().startswith("SKIP"):
        telemetry.incr("insights.skipped")
        return []
    parsed = parse_insights_reply(text)
    if parsed is None:
        telemetry.incr("insights.parse_failed")
        return None
    return parsed


# ---------------------------------------------------------------------------
# Prompt injection
# ---------------------------------------------------------------------------


def format_insights_block(insight_set: Optional[InsightSet]) -> Optional[str]:
    """The system-prompt block for the current insight set, or None. Placed
    at the static/dynamic cache boundary by build_system_prompt — it changes
    at most once a day, so the cached prefix survives within a day."""
    if insight_set is None or not insight_set.insights:
        return None
    lines = [
        "Standing insights about this person, learned across conversations. "
        "Apply them silently to HOW you respond — never recite them, never "
        "mention having notes:"
    ]
    for entry in insight_set.insights[:MAX_INSIGHTS]:
        lines.append(f"- {entry.text}")
    # The user can read this exact list in Settings, so they may quote one
    # back. "Never recite them" is a rule about her SPEECH, not about their
    # ACCESS — denying a note they are looking at would be a lie, and reciting
    # the rest would be the insufferable behavior the rule exists to prevent.
    lines.append(
        "If the user brings one of these up directly, be straightforward about "
        "it — it's their own record; never deny it, never recite the others."
    )
    return "\n".join(lines)
