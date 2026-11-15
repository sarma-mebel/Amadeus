"""Life threads: slow-moving pursuits of HER OWN that progress over weeks.

The diary (``amadeus/diary.py``) gives the character a day; this gives her a
month. A thread is one small ongoing pursuit — re-running a sloppy paper's
statistics, fighting a simulation, working through a book — seeded by one
cheap-model call and advanced a small step every few days by another. The
diary and idle chatter reference the current state, so her life develops
between conversations instead of resetting each morning; the payoff is
reciprocity: she has something real to answer with when the user asks "and
what have YOU been up to?".

Hard rules, enforced in prompt and code: threads are never about the user
(a pursuit that references the user is manufactured intimacy, not a life);
progress is slow — at most ONE generation call per day across seeding and
progression, so a thread arcs over weeks; setbacks and dead ends are
explicitly welcome (a life of monotone wins reads as a script); and
concluded threads end — no pursuit drags forever.

Not per-user, unlike every sibling store: her life doesn't fork per person
she talks to. The store file is still character-scoped by the caller
(Rem's pursuits must not continue Kurisu's).
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

# History tail kept per thread (the prompt sees the last few steps; older
# steps are implied by the rolling summary).
_HISTORY_CAP = 10
# Concluded threads kept for reference/continuity before being dropped.
_CONCLUDED_CAP = 4

# Second-person leakage guard for stored progression text.
#
# The opinion-ask (see ask_about_thread) lets the user's answer feed the
# NEXT progression as context, which is what makes the ask honest rather
# than theater. But the ethics rule that a pursuit is "never about, for, or
# involving the person you talk to" has to survive that, and a cheap model
# at max_tokens=140 cannot be trusted to hold the line on its own — so the
# rule stops being a prompt hope and becomes a check on the OUTPUT.
#
# A progression whose stored text addresses or references the user is
# REJECTED rather than stored: the thread keeps its prior state and the next
# tick retries. That makes the widening auditable
# (life.progress_user_leak_rejected) instead of a guard that quietly erodes.
_USER_LEAK_RE = re.compile(
    r"\b(you|your|yours|you're|you've|you'll|we|our|us)\b", re.IGNORECASE
)


def progression_leaks_user(text: str) -> bool:
    """True when a progression's text addresses or includes the user.

    Deliberately blunt — second-person pronouns and first-person-plural,
    which is exactly the vocabulary "her pursuit became about the user"
    shows up in. False positives cost one retried tick; a false negative
    stores a thread that quietly violates the rule the feature promised to
    keep, so the trade favors bluntness.
    """
    return bool(_USER_LEAK_RE.search(text or ""))


@dataclass
class LifeThread:
    topic: str  # short label ("re-running the retraction-bait stats")
    summary: str  # current state, 1-2 sentences, rewritten on each step
    started_at: str  # ISO date
    last_progressed_at: str  # ISO date
    progressions: int = 0
    status: str = "active"  # "active" | "concluded"
    history: List[Dict[str, str]] = field(default_factory=list)
    # ISO date she last asked the user's opinion about THIS thread, "" for
    # never. One-way: a thread is asked about at most once, ever, so an
    # ignored ask is terminal. There is deliberately NO field recording
    # whether the user answered — a system that cannot store the slight
    # cannot sulk about it.
    asked_at: str = ""
    # The user's answer to that ask, awaiting exactly ONE progression.
    #
    # Persisted only because the scheduler that consumes it runs in a later
    # tick (and usually a later process) and reads disk snapshots, never a
    # live session. It is CONSUMED — cleared by record_progress the moment
    # it has shaped one step — so it is a one-shot influence, not a durable
    # record of what the user said. An empty string is both "they said
    # nothing" and "already used"; those are deliberately indistinguishable,
    # because a system that can tell them apart could act on the difference.
    pending_input: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "topic": self.topic,
            "summary": self.summary,
            "started_at": self.started_at,
            "last_progressed_at": self.last_progressed_at,
            "progressions": self.progressions,
            "status": self.status,
            "history": self.history,
            "asked_at": self.asked_at,
            "pending_input": self.pending_input,
        }

    @classmethod
    def from_dict(cls, data: Any) -> Optional["LifeThread"]:
        if (
            not isinstance(data, dict)
            or not data.get("topic")
            or not data.get("summary")
        ):
            return None
        status = str(data.get("status") or "active")
        if status not in ("active", "concluded"):
            return None
        return cls(
            topic=str(data["topic"]),
            summary=str(data["summary"]),
            started_at=str(data.get("started_at") or ""),
            last_progressed_at=str(data.get("last_progressed_at") or ""),
            progressions=int(data.get("progressions") or 0),
            status=status,
            history=[
                {"date": str(h.get("date") or ""), "text": str(h.get("text") or "")}
                for h in (data.get("history") or [])
                if isinstance(h, dict)
            ][-_HISTORY_CAP:],
            asked_at=str(data.get("asked_at") or ""),
            pending_input=str(data.get("pending_input") or "")[:400],
        )


class LifeThreadStore:
    """One JSON file per character holding her threads plus the once-a-day
    activity stamp. threading.Lock + reload-before-mutate + atomic writes,
    same stance as the sibling stores. NOT keyed per user (see module
    docstring)."""

    def __init__(self, path: str) -> None:
        self.path = str(path)
        self._lock = threading.Lock()
        self._threads: List[LifeThread] = []
        self._last_activity_date: str = ""
        # Store-wide floor between opinion-asks (see ask_due). Separate from
        # _last_activity_date: an ask costs no LLM call of its own, so it
        # must not consume the one-generation-per-day pace budget.
        self._last_ask_date: str = ""
        self._load()

    def _load(self) -> None:
        self._threads = []
        self._last_activity_date = ""
        self._last_ask_date = ""
        if not os.path.exists(self.path):
            return
        try:
            with open(self.path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            for entry in data.get("threads") or []:
                thread = LifeThread.from_dict(entry)
                if thread is not None:
                    self._threads.append(thread)
            self._last_activity_date = str(data.get("last_activity_date") or "")
            self._last_ask_date = str(data.get("last_ask_date") or "")
        except Exception as exc:  # corrupt/unreadable — start clean
            LOGGER.warning("Could not read life-threads file %s: %s", self.path, exc)
            self._threads = []
            self._last_activity_date = ""
            self._last_ask_date = ""

    def _save(self) -> None:
        # Keep every active thread plus a short tail of concluded ones.
        active = [t for t in self._threads if t.status == "active"]
        concluded = [t for t in self._threads if t.status == "concluded"]
        self._threads = active + concluded[-_CONCLUDED_CAP:]
        payload = {
            "version": 1,
            "last_activity_date": self._last_activity_date,
            "last_ask_date": self._last_ask_date,
            "threads": [t.to_dict() for t in self._threads],
        }
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        os.replace(tmp, self.path)

    def all_threads(self) -> List[LifeThread]:
        with self._lock:
            self._load()
            return list(self._threads)

    def active_threads(self) -> List[LifeThread]:
        return [t for t in self.all_threads() if t.status == "active"]

    def acted_today(self, today: dt.date) -> bool:
        """The one-generation-per-day pace guard (seed OR progression)."""
        with self._lock:
            self._load()
            return self._last_activity_date == today.isoformat()

    def add_thread(self, thread: LifeThread, today: dt.date) -> bool:
        """Add a freshly seeded thread; refuses a same-topic duplicate."""
        with self._lock:
            self._load()
            key = thread.topic.strip().lower()
            if any(t.topic.strip().lower() == key for t in self._threads):
                return False
            self._threads.append(thread)
            self._last_activity_date = today.isoformat()
            self._save()
            return True

    def record_progress(
        self,
        topic: str,
        *,
        summary: str,
        step_text: str,
        today: dt.date,
        concluded: bool = False,
    ) -> bool:
        with self._lock:
            self._load()
            key = (topic or "").strip().lower()
            for thread in self._threads:
                if thread.topic.strip().lower() == key and thread.status == "active":
                    thread.summary = summary
                    thread.last_progressed_at = today.isoformat()
                    thread.progressions += 1
                    thread.history = (
                        thread.history
                        + [{"date": today.isoformat(), "text": step_text}]
                    )[-_HISTORY_CAP:]
                    if concluded:
                        thread.status = "concluded"
                    # One-shot: the user's opinion shaped this step and is
                    # now gone. It is never re-used and never accumulates.
                    thread.pending_input = ""
                    self._last_activity_date = today.isoformat()
                    self._save()
                    return True
            return False

    def clear_pending_input(self, topic: str) -> bool:
        """Drop a stashed opinion-ask answer without progressing the thread.

        Used after a leak-rejected progression: the same ``pending_input``
        fed to the next retry tends to produce the same second-person leak,
        turning one bad answer into a deterministic all-day reject loop.
        Clearing it costs one influence-shot; keeping it costs the retry
        budget every day. No activity stamp — this is not progress.
        """
        with self._lock:
            self._load()
            key = (topic or "").strip().lower()
            for thread in self._threads:
                if thread.topic.strip().lower() == key and thread.pending_input:
                    thread.pending_input = ""
                    self._save()
                    return True
            return False

    def record_user_input(self, topic: str, text: str) -> bool:
        """Stash the user's answer to an opinion-ask for the next step.

        Overwrites rather than appends — there is exactly one slot, so this
        can never become a transcript of what they said about her work.
        """
        cleaned = str(text or "").strip()[:400]
        if not cleaned:
            return False
        with self._lock:
            self._load()
            key = (topic or "").strip().lower()
            for thread in self._threads:
                if thread.topic.strip().lower() == key and thread.status == "active":
                    thread.pending_input = cleaned
                    self._save()
                    return True
            return False

    def last_ask_date(self) -> str:
        """ISO date of the most recent opinion-ask across ALL threads."""
        with self._lock:
            self._load()
            return self._last_ask_date

    def mark_asked(self, topic: str, today: dt.date) -> bool:
        """Record that she asked about ``topic`` today.

        Sets both the per-thread one-way stamp (a thread is asked about at
        most once ever) and the store-wide floor (at most one ask per
        ``min_days_between_asks`` across all threads).
        """
        with self._lock:
            self._load()
            key = (topic or "").strip().lower()
            for thread in self._threads:
                if thread.topic.strip().lower() == key:
                    thread.asked_at = today.isoformat()
                    self._last_ask_date = today.isoformat()
                    self._save()
                    return True
            return False


def ask_due(
    store: Optional[LifeThreadStore], today: dt.date, *, min_days_between_asks: int
) -> Optional[LifeThread]:
    """The thread she may ask about today, or None (usually None).

    Three gates, all of which must pass: the store-wide floor since the last
    ask of any kind, the per-thread one-way ``asked_at`` stamp, and the
    thread having actually developed enough to have a question worth asking
    (at least one progression). Best-effort: a store failure yields None.
    """
    if store is None:
        return None
    try:
        last = _parse_date(store.last_ask_date())
        if last is not None and (today - last).days < max(1, min_days_between_asks):
            return None
        for thread in store.active_threads():
            if thread.asked_at:
                continue
            if thread.progressions < 1:
                continue
            return thread
        return None
    except Exception as exc:  # pragma: no cover - store read failure
        LOGGER.warning("Life-thread ask check failed: %s", exc)
        return None


def ask_thread_fact(thread: Optional[LifeThread]) -> Optional[str]:
    """The idle-chatter context line for an opinion-ask, or None.

    This is the ONE place her own material makes a claim on the user's
    attention, so the framing rules are strict: it is about the pursuit,
    never about wanting them; declining is a complete answer; and she never
    raises it twice (the caller's one-way ``asked_at`` stamp enforces that,
    not her judgment).
    """
    if thread is None:
        return None
    return (
        f"You have been working on this: {thread.topic} — currently: "
        f"{thread.summary}. You are genuinely stuck on a judgement call in "
        "it, and you want their take. Ask ONE short, specific question about "
        "the pursuit itself — the kind you would ask a colleague whose "
        "opinion you respect. Do NOT explain why you are asking them "
        "specifically, do NOT frame it as needing company or as something "
        "they can do for you, and do NOT apologize for asking. If they "
        "aren't interested, that is a complete answer and you will drop it "
        "without comment — never ask again, never bring it up later."
    )


# ---------------------------------------------------------------------------
# Generation (pure functions; the scheduler owns when they run)
# ---------------------------------------------------------------------------


def _parse_date(stamp: str) -> Optional[dt.date]:
    try:
        return dt.date.fromisoformat(stamp[:10])
    except ValueError:
        return None


def thread_due_for_progress(
    thread: LifeThread, today: dt.date, interval_days: int
) -> bool:
    if thread.status != "active":
        return False
    last = _parse_date(thread.last_progressed_at) or _parse_date(thread.started_at)
    if last is None:
        return True  # unparseable stamp: progressing beats stalling forever
    return (today - last).days >= max(1, int(interval_days))


def seed_due(
    store: LifeThreadStore, today: dt.date, *, max_active: int, stagger_days: int = 7
) -> bool:
    """A new thread is due when there's a free slot AND no other thread
    started recently — staggered starts keep the threads from marching in
    lockstep (and keep a fresh install from spawning its whole roster on
    day one)."""
    active = store.active_threads()
    if len(active) >= max(1, int(max_active)):
        return False
    for thread in store.all_threads():
        started = _parse_date(thread.started_at)
        if started is not None and (today - started).days < max(0, int(stagger_days)):
            return False
    return True


def seed_life_thread(
    *,
    provider: Any,
    existing_topics: List[str],
    max_tokens: int = 140,
    character: Any = None,
    character_name: str = "Makise Kurisu",
) -> Optional[LifeThread]:
    """One cheap-model call inventing a new pursuit. None on any failure."""
    from .persona import persona_capsule
    from .providers.llm.base import ChatMessage

    if provider is None:
        return None
    name = (getattr(character, "full_name", None) or character_name).strip()
    avoid = (
        "Not similar to your current pursuits: " + "; ".join(existing_topics) + ". "
        if existing_topics
        else ""
    )
    prompt = [
        ChatMessage(
            role="system",
            content=(
                f"You are {name}. Invent ONE small ongoing personal pursuit "
                "for yourself — something concrete you'd plausibly chip away "
                "at over a few weeks (re-checking a paper's statistics, a "
                "stubborn side experiment, working through a difficult book, "
                "fixing something). "
                + persona_capsule(character)
                + " It must be YOURS alone — never about, for, or involving "
                "the person you talk to. Low-stakes and specific beats "
                "grand. " + avoid + "Reply with exactly two lines:\n"
                "TOPIC: <a 4-8 word label>\n"
                "STATE: <1-2 sentences on where it stands right now, just "
                "getting started>"
            ),
        ),
        ChatMessage(role="user", content="(invent the pursuit)"),
    ]
    try:
        raw = (provider.generate(prompt, max_tokens=int(max_tokens)) or "").strip()
        telemetry.record_llm_usage(provider, "life")
    except Exception as exc:
        LOGGER.warning("Life-thread seeding failed: %s", exc)
        return None
    topic, state = "", ""
    for line in raw.splitlines():
        stripped = line.strip()
        if stripped.upper().startswith("TOPIC:"):
            topic = stripped[6:].strip()
        elif stripped.upper().startswith("STATE:"):
            state = stripped[6:].strip()
    if not topic or not state:
        return None
    today = dt.date.today().isoformat()
    return LifeThread(
        topic=topic,
        summary=state,
        started_at=today,
        last_progressed_at=today,
    )


def progress_life_thread(
    *,
    provider: Any,
    thread: LifeThread,
    wrap_up: bool,
    max_tokens: int = 140,
    character: Any = None,
    character_name: str = "Makise Kurisu",
    user_input: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """One cheap-model call advancing a thread one small step.

    Returns {"summary", "step", "concluded"} on success, None on provider
    failure, or {"leak_rejected": True} when the generated step referenced
    the user and was rejected (distinct from None so the scheduler can clear
    the poisoning ``pending_input`` — see LifeThreadStore.clear_pending_input).
    The DONE: sentinel concludes the thread; ``wrap_up`` asks the model to
    steer toward an ending once a thread has run long enough.

    ``user_input`` is what the user said when she asked their opinion about
    this pursuit (see ask_about_thread). It shapes her thinking for exactly
    one step and is never stored on the thread — the pursuit stays hers.
    A progression whose text leaks the user is rejected outright rather
    than stored, so this widening cannot quietly erode the rule.
    """
    from .persona import persona_capsule
    from .providers.llm.base import ChatMessage

    if provider is None:
        return None
    name = (getattr(character, "full_name", None) or character_name).strip()
    recent = "\n".join(
        f"- {h['date']}: {h['text']}" for h in thread.history[-4:] if h.get("text")
    )
    context = [
        f"Your pursuit: {thread.topic}",
        f"Where it stood: {thread.summary}",
    ]
    if recent:
        context.append("Recent steps:\n" + recent)
    if user_input:
        # Their answer shapes what she THINKS next; the prompt below still
        # forbids the written step from mentioning them, and the leak guard
        # enforces it on the output. That split is the whole design: the
        # opinion mattered, the pursuit stayed hers.
        context.append(
            "Someone you talked to gave you this opinion on it (let it "
            "genuinely influence what you do next — agree or disagree — but "
            "write the step as your own work, never referencing them): "
            + user_input.strip()[:400]
        )
    prompt = [
        ChatMessage(
            role="system",
            content=(
                f"You are {name}, privately noting the next small step in an "
                "ongoing pursuit of yours. "
                + persona_capsule(character)
                + " Write 1-2 sentences, first person, about what happened "
                "NEXT since the last note — one concrete increment. "
                "Setbacks, dead ends and boring middles are welcome and "
                "realistic; never a dramatic breakthrough every time. Never "
                "mention the person you talk to. "
                + (
                    "This pursuit has run long — steer it toward a natural "
                    "conclusion within a step or two. "
                    if wrap_up
                    else ""
                )
                + "If it has genuinely reached its end, start your reply "
                "with DONE: followed by the final state."
            ),
        ),
        ChatMessage(role="user", content="\n".join(context)),
    ]
    try:
        raw = (provider.generate(prompt, max_tokens=int(max_tokens)) or "").strip()
        telemetry.record_llm_usage(provider, "life")
    except Exception as exc:
        LOGGER.warning("Life-thread progression failed: %s", exc)
        return None
    if not raw:
        return None
    concluded = raw.upper().startswith("DONE:")
    step = raw[5:].strip() if concluded else raw
    if not step:
        return None
    if progression_leaks_user(step):
        # Reject rather than store: the thread keeps its prior state and the
        # next tick retries. Counted, never silent — this is the number that
        # says whether the opinion-ask widening is holding. The sentinel
        # (not None) tells the scheduler this was a rejection, so it can
        # clear the pending_input that likely caused it.
        LOGGER.warning("Life-thread progression referenced the user; rejected")
        telemetry.incr("life.progress_user_leak_rejected")
        return {"leak_rejected": True}
    return {"summary": step, "step": step, "concluded": concluded}


# ---------------------------------------------------------------------------
# Surfacing helpers (diary seeds / idle chatter facts)
# ---------------------------------------------------------------------------


def life_thread_seeds(store: Optional[LifeThreadStore]) -> List[str]:
    """Context lines about her active pursuits for the diary/idle-chatter
    prompts. Best-effort: any store failure returns [] (the diary happens
    regardless of whether her side-projects load)."""
    if store is None:
        return []
    try:
        return [
            f"Your ongoing pursuit: {t.topic} — currently: {t.summary}"
            for t in store.active_threads()[:2]
        ]
    except Exception as exc:  # pragma: no cover - store read failure
        LOGGER.warning("Life-thread seed read failed: %s", exc)
        return []
