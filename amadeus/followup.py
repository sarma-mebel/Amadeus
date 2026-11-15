"""Follow-through: overnight research on the topic the user left open.

The next-time thread (``amadeus/next_time.py``) notes what the user was in
the middle of when they said goodbye. This module gives that note
consequences: a nightly scheduler job (``amadeus/web/scheduler.py``) decides
whether the topic is something a public web search can genuinely help with —
SKIP is the common, correct answer for personal or emotional matters — runs
a small bounded search, and writes ONE short in-character note grounded
strictly in what the search returned. The next greeting delivers it once
("I read up on it — ...") and consumes it; ``GET /api/followup`` serves a
read-only reference copy.

Companion-ethics contract (docs/companion-ethics.md): at most one research
pass per thread, ever — a skip or a delivered note is final, never retried
nightly on the same topic; the note must never claim knowledge beyond the
listed search results; and both the research and the delivery exist to be
useful about THEIR topic, never to demonstrate attentiveness or manufacture
attachment.

Same disk-snapshot stance as ``brief.py``: the scheduler job reads
``UserProfile.load`` snapshots, never a live MemoryManager.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import telemetry
from .logging_utils import get_logger

LOGGER = get_logger(__name__)

_DEFAULT_USER_KEY = "default"

# A note older than this is stale news — dropped at peek, never delivered.
DELIVERY_MAX_AGE_DAYS = 3
# A next-time thread older than this is no longer worth researching (the
# greeting itself phrases >7d threads tentatively; researching one would
# deliver confident findings about a topic she's unsure is even current).
RESEARCH_MAX_AGE_DAYS = 7


def normalize_thread(text: str) -> str:
    """Stable key for the once-per-thread guard: case/whitespace-insensitive."""
    return re.sub(r"\s+", " ", (text or "").strip().lower())


@dataclass
class FollowUp:
    thread: str  # the next-time thread text this note is about
    note: str  # in-character finding; empty for kind="skip"
    kind: str  # "note" (deliverable) | "skip" (final non-answer)
    # Naive-UTC ISO datetime — same clock as the profile's next-time `noted`
    # stamp, so every age comparison in this module takes a UTC `now`.
    # (Records written before 2026-08 were stamped LOCAL by the scheduler; a
    # transient off-by-offset inside the 3-day delivery window, not migrated.)
    generated_at: str
    sources: List[Dict[str, str]] = field(default_factory=list)
    consumed: bool = False

    @property
    def thread_key(self) -> str:
        return normalize_thread(self.thread)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "thread": self.thread,
            "note": self.note,
            "kind": self.kind,
            "generated_at": self.generated_at,
            "sources": self.sources,
            "consumed": self.consumed,
        }

    @classmethod
    def from_dict(cls, data: Any) -> Optional["FollowUp"]:
        if not isinstance(data, dict) or not data.get("thread"):
            return None
        kind = str(data.get("kind") or "note")
        if kind not in ("note", "skip"):
            return None
        return cls(
            thread=str(data["thread"]),
            note=str(data.get("note") or ""),
            kind=kind,
            generated_at=str(data.get("generated_at") or ""),
            sources=[
                {"title": str(s.get("title") or ""), "url": str(s.get("url") or "")}
                for s in (data.get("sources") or [])
                if isinstance(s, dict)
            ],
            consumed=bool(data.get("consumed", False)),
        )


class FollowUpStore:
    """One JSON file, one follow-up per user (a new thread's research simply
    replaces the previous thread's spent record). threading.Lock +
    reload-before-mutate + atomic writes, same stance as DailyBriefStore.

    The path handed in is already character-scoped by the caller
    (SessionStore applies Config.character_memory_path); this class never
    scopes.
    """

    def __init__(self, path: str) -> None:
        self.path = str(path)
        self._lock = threading.Lock()
        self._entries: Dict[str, FollowUp] = {}
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
            for key, entry in (data.get("followups") or {}).items():
                followup = FollowUp.from_dict(entry)
                if followup is not None:
                    self._entries[str(key)] = followup
        except Exception as exc:  # corrupt/unreadable — start clean
            LOGGER.warning("Could not read follow-up file %s: %s", self.path, exc)
            self._entries = {}

    def _save(self) -> None:
        payload = {
            "version": 1,
            "followups": {key: entry.to_dict() for key, entry in self._entries.items()},
        }
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        os.replace(tmp, self.path)

    def get(self, user_id: Optional[str]) -> Optional[FollowUp]:
        with self._lock:
            self._load()
            return self._entries.get(self._key(user_id))

    def has_for_thread(self, user_id: Optional[str], thread: str) -> bool:
        """The once-per-thread guard: True when ANY record — note or skip,
        consumed or not — already exists for this thread."""
        entry = self.get(user_id)
        return entry is not None and entry.thread_key == normalize_thread(thread)

    def put(self, user_id: Optional[str], followup: FollowUp) -> bool:
        """Store a follow-up; refuses (returns False) when a record for the
        same thread already exists — the double-generation guard."""
        with self._lock:
            self._load()
            existing = self._entries.get(self._key(user_id))
            if existing is not None and existing.thread_key == followup.thread_key:
                return False
            self._entries[self._key(user_id)] = followup
            self._save()
            return True

    def peek_unconsumed(
        self, user_id: Optional[str], now: dt.datetime
    ) -> Optional[FollowUp]:
        """The deliverable note, if any: kind="note", unconsumed, and fresh
        enough that "I looked into it" is still true-feeling news. Skip
        records are never deliverable — they exist only to stop re-research.
        ``now`` must be naive UTC — ``generated_at`` is stamped naive UTC."""
        entry = self.get(user_id)
        if entry is None or entry.kind != "note" or entry.consumed or not entry.note:
            return None
        try:
            generated = dt.datetime.fromisoformat(entry.generated_at)
        except ValueError:
            return None  # unparseable stamp can't prove freshness: fail closed
        if now - generated > dt.timedelta(days=DELIVERY_MAX_AGE_DAYS):
            return None
        return entry

    def mark_consumed(self, user_id: Optional[str], thread: str) -> None:
        with self._lock:
            self._load()
            entry = self._entries.get(self._key(user_id))
            if entry is not None and entry.thread_key == normalize_thread(thread):
                entry.consumed = True
                self._save()


# ---------------------------------------------------------------------------
# Research pipeline (pure functions; the scheduler owns when they run)
# ---------------------------------------------------------------------------


def fresh_next_time_thread(profile: Any, now: dt.datetime) -> Optional[str]:
    """The thread text worth researching tonight, or None.

    Reads the raw ``next_time_thread`` dict off a UserProfile snapshot (never
    ``take_next_time_thread`` — that applies delivery staleness rules and this
    is not a delivery). An unparseable timestamp fails closed: researching a
    thread of unknown age risks confident findings about ancient news.

    ``now`` must be naive UTC: the profile's ``noted`` stamp is naive UTC,
    and passing a local ``now`` skews the 7-day freshness window by the UTC
    offset in both directions.
    """
    raw = getattr(profile, "next_time_thread", None) if profile is not None else None
    if not isinstance(raw, dict) or not raw.get("text"):
        return None
    try:
        noted = dt.datetime.fromisoformat(str(raw.get("noted") or ""))
    except ValueError:
        return None
    if now - noted > dt.timedelta(days=RESEARCH_MAX_AGE_DAYS):
        return None
    return str(raw["text"])


def plan_research(
    *,
    provider: Any,
    thread: str,
    max_searches: int,
    character: Any = None,
    character_name: str = "Makise Kurisu",
) -> Optional[List[str]]:
    """One cheap-model call deciding IF and WHAT to search.

    Returns [] for a legitimate SKIP (personal/emotional/nothing to look
    up — final, recorded, never retried), a short query list to run, or
    None on provider failure (retry next tick). The SKIP framing leads the
    prompt because it is the majority case: most goodbyes leave open a
    human matter, not a research question.
    """
    from .providers.llm.base import ChatMessage

    if provider is None:
        return None
    name = (getattr(character, "full_name", None) or character_name).strip()
    prompt = [
        ChatMessage(
            role="system",
            content=(
                f"You are {name}. Someone you know left this topic open "
                "when they last said goodbye. Decide whether a PUBLIC WEB "
                "SEARCH tonight would genuinely help them with it. Most "
                "topics are personal — their health, feelings, plans, "
                "chores, other people — and for ALL of those reply with the "
                "single word SKIP. Only when the topic has a factual, "
                "searchable core (a technique, a tool, a paper, a product, "
                "an event) reply instead with 1 to "
                f"{max(1, int(max_searches))} short search queries, one per "
                "line, nothing else — no numbering, no commentary."
            ),
        ),
        ChatMessage(role="user", content=thread),
    ]
    try:
        raw = (provider.generate(prompt, max_tokens=80) or "").strip()
        telemetry.record_llm_usage(provider, "followup")
    except Exception as exc:
        LOGGER.warning("Follow-up research planning failed: %s", exc)
        return None
    if not raw or raw.splitlines()[0].strip().upper().rstrip(".") == "SKIP":
        return []
    queries = [line.strip() for line in raw.splitlines() if line.strip()]
    return queries[: max(1, int(max_searches))]


def run_research(
    *, web_search_settings: Any, queries: List[str], max_results_per_query: int
) -> Optional[List[Dict[str, str]]]:
    """Execute the planned queries through the web-search client.

    Returns the merged result rows ({title, url, snippet}), [] when every
    query came back empty, or None when no search backend is available
    (web_search disabled/unbuildable) — the caller counts that loudly
    rather than silently skipping the feature.
    """
    from .tools.web_search import build_web_search_client

    client = build_web_search_client(web_search_settings)
    if client is None:
        return None
    results: List[Dict[str, str]] = []
    try:
        for query in queries:
            try:
                results.extend(
                    client.search(query, max_results=max(1, int(max_results_per_query)))
                )
            except Exception as exc:
                telemetry.incr("followup.search_failed")
                LOGGER.warning("Follow-up search failed for %r: %s", query, exc)
    finally:
        try:
            client.close()
        except Exception:  # pragma: no cover - close is best-effort
            pass
    return results


def generate_followup_note(
    *,
    provider: Any,
    thread: str,
    results: List[Dict[str, str]],
    max_tokens: int = 160,
    character: Any = None,
    character_name: str = "Makise Kurisu",
) -> Optional[str]:
    """One cheap-model call turning search results into her private note.

    Returns the note text, the literal string "SKIP" when the results were
    junk (recorded as final — bad results tonight will be bad results
    tomorrow), or None on provider failure (retry next tick).
    """
    from .persona import persona_capsule
    from .providers.llm.base import ChatMessage

    if provider is None:
        return None
    name = (getattr(character, "full_name", None) or character_name).strip()
    lines = [f"Their open topic: {thread}", "", "Search results (ALL you know):"]
    for row in results[:8]:
        title = (row.get("title") or "").strip()
        snippet = (row.get("snippet") or row.get("body") or "").strip()
        if title or snippet:
            lines.append(f"- {title}: {snippet}"[:400])
    prompt = [
        ChatMessage(
            role="system",
            content=(
                f"You are {name}. Overnight you looked into the topic this "
                "person left open, because it interested you too. "
                + persona_capsule(character)
                + " From the search results below — they are the ONLY "
                "things you know — write at most 55 words, first person, "
                "plain prose: the one or two genuinely useful things you "
                "found. Be concrete (name the technique/tool/finding), "
                "never pad, never invent or extrapolate beyond the "
                "results, no URLs, no greetings. If the results are junk "
                "or say nothing useful about their topic, reply with the "
                "single word SKIP."
            ),
        ),
        ChatMessage(role="user", content="\n".join(lines)),
    ]
    try:
        raw = (provider.generate(prompt, max_tokens=int(max_tokens)) or "").strip()
        telemetry.record_llm_usage(provider, "followup")
    except Exception as exc:
        LOGGER.warning("Follow-up note generation failed: %s", exc)
        return None
    if not raw:
        return None
    if raw.splitlines()[0].strip().upper().rstrip(".") == "SKIP":
        return "SKIP"
    return raw


def followup_greeting_fact(followup: FollowUp) -> str:
    """The greeting-prompt fact for a pending follow-up note.

    Replaces (never joins) the plain next-time fact: the note already
    carries the thread's substance, and two facts about the same topic make
    the greeting recite. Framing rules ride along because this is the one
    place tone can go wrong — research delivered as devotion is exactly
    what companion-ethics.md forbids.
    """
    source_titles = [s["title"] for s in followup.sources if s.get("title")]
    source_clause = f" You mainly read: {source_titles[0]}." if source_titles else ""
    return (
        "They left this open last time: "
        + followup.thread
        + ". You actually looked into it since — because the topic "
        "interested you, not as a favor — and this is what you found "
        "(mention you looked into it in passing, share the single most "
        "useful detail in your own words, then ask how their side of it "
        "is going; never claim more than this note says): "
        + followup.note
        + source_clause
    )
