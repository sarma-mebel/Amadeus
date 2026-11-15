"""Relational memory: durable records of moments that changed something
between her and the user.

Widens the shared-moments concept (amadeus/moments.py) from co-activity-only
into a general relational-event store. Same record type (``"moment"``),
tagged ``source="relational"`` so the two triggers keep independent caps,
telemetry and prompts while sharing recall, pruning, memory-panel
visibility, archive/forget and the supersede exemption.

ANTI-FIXATION CONTRACT — the reason this module is so careful. Episodes and
reflections are deliberately neutral third-person, because writing session
records in her voice made her persona compound across sessions until she
fixated (see ``extraction_jobs.run_episode_job``). This module deliberately
reintroduces a FIRST-PERSON session-end record, so it carries four guards:

1. A hard 1-per-week cap. This is the REAL guard: episodes fixated because
   every session wrote one, at a 200-record ceiling. A record at ~1/week
   landing in a store that already holds hundreds of facts cannot dominate
   recall the way a per-session record did.
2. A SKIP-default prompt whose bar is "did this change something between
   us", not "what happened". SKIP is the expected answer for a normal
   session, and the healthy telemetry signal is skipped >> stored.
3. A ban on evaluative language about the user. It records the EVENT, never
   her opinion of them — that is the precise line episodes crossed.
4. Low salience, so it never routes as eagerly as a stated fact. The
   existing ``recall_recent_dedupe`` fixation guard then applies for free.

Free-function-over-owner, taking the MANAGER (not the engine) as first
argument: the trigger is ``MemoryManager.finalize_session``, which is the
one seam every session-end path goes through — farewell, window close, gap
detection and session eviction alike — and it has no engine reference.
"""

from __future__ import annotations

import datetime as dt
from typing import TYPE_CHECKING, List, Optional, Sequence

from .. import telemetry
from ..logging_utils import get_logger
from ..persona import persona_capsule
from ..providers.llm.base import ChatMessage

if TYPE_CHECKING:  # pragma: no cover - import cycle guard
    from .manager import Message
    from .manager import MemoryManager

LOGGER = get_logger(__name__)

_SOURCE = "relational"


# ---------------------------------------------------------------------------
# Moment-store primitives
#
# Bodies for MemoryManager.store_moment/count_moments_since, which keep thin
# delegates. They live here rather than in manager.py because the `source`
# split (co_activity vs relational) is this module's concept, and manager.py
# is a budgeted god-file whose fix is extraction, not a bigger number.
# ---------------------------------------------------------------------------


def store_moment(
    manager: "MemoryManager",
    text: str,
    salience: float = 0.6,
    source: str = "co_activity",
) -> bool:
    """Store one distilled durable memory of something SHARED.

    Two writers, distinguished by ``source`` rather than by record type:
    ``co_activity`` (amadeus/moments.py — read-along/watch-together) and
    ``relational`` (this module — a session that changed something between
    them). They deliberately share ``type: "moment"`` so both inherit
    prune-over-cap, memory-panel visibility, archive/forget, recall admission
    and the supersede exemption for free; a separate record type would have
    needed each of those taught about it individually, and every one of those
    omissions fails silently.

    Rides ``_store_fact``'s machinery (truncation, archive-on-evict, vector
    persistence) tagged ``type: "moment"`` instead of ``"fact"`` so it is
    excluded from the supersede-hint scan (a moment is a standalone event,
    never a correction to a user-stated fact) while still surfacing through
    recall like any other durable record. Salience defaults lower than a
    user-stated fact (0.9): a shared moment is pleasant color, not something
    she should route to as eagerly as an explicit fact.
    """
    text = (text or "").strip()
    if not text:
        return False
    if not (
        manager.config.memory.enable_long_term
        and manager.embedder
        and manager.vector_store
    ):
        return False
    return manager._store_fact(
        text, salience=salience, record_type="moment", source=source
    )


def count_moments_since(
    manager: "MemoryManager", cutoff: dt.datetime, source: Optional[str] = None
) -> int:
    """Count stored moments with a timestamp at or after ``cutoff``.

    The cap check for both moment writers. ``source=None`` counts every
    moment (the original behavior); passing a source scopes the count to that
    writer, so co-activity's per-day cap and relational memory's per-week cap
    stay independent — one busy reading week must not silence a genuine
    relational record, and vice versa. A cheap full scan (both caps keep
    totals tiny; this never runs per-turn).
    """
    from .manager import _parse_iso

    if not manager.vector_store:
        return 0
    with manager._state_lock:
        records = list(manager.vector_store.dump())
    count = 0
    for rec in records:
        metadata = rec.metadata or {}
        if metadata.get("type") != "moment":
            continue
        if source is not None and metadata.get("source") != source:
            continue
        stamped = _parse_iso(metadata.get("timestamp"))
        if stamped is not None and stamped >= cutoff:
            count += 1
    return count


def maybe_store_relational_moment(
    manager: "MemoryManager", messages: Sequence["Message"]
) -> None:
    """Distill a just-ended session into at most one relational memory.

    Called from ``MemoryManager.finalize_session`` as a submitted job — a
    pure side effect that never blocks or changes a return value.
    Best-effort by contract: every failure counts a telemetry counter and
    returns, never raising into the session-end path.
    """
    try:
        _maybe_store_relational_moment(manager, messages)
    except Exception as exc:  # never break session finalization
        LOGGER.warning("Relational moment distillation failed: %s", exc)
        telemetry.incr("memory.relational_failures")


def _maybe_store_relational_moment(
    manager: "MemoryManager", messages: Sequence["Message"]
) -> None:
    cfg = getattr(manager.config, "relational_memory", None)
    if cfg is None or not getattr(cfg, "enabled", False):
        return
    if not getattr(manager.config.memory, "enable_long_term", False):
        return

    message_list = list(messages or [])
    if len(message_list) < int(getattr(cfg, "min_messages", 6) or 6):
        telemetry.incr("memory.relational_too_short")
        return

    if manager.vector_store is None or manager.embedder is None:
        return

    # Cap check BEFORE the provider call, so a capped period costs zero
    # tokens rather than generating a line only to throw it away.
    max_per_days = max(1, int(getattr(cfg, "max_per_days", 7) or 7))
    max_per_period = max(1, int(getattr(cfg, "max_per_period", 1) or 1))
    cutoff = dt.datetime.utcnow() - dt.timedelta(days=max_per_days)
    try:
        stored_recently = manager.count_moments_since(cutoff, source=_SOURCE)
    except Exception as exc:  # pragma: no cover - store best-effort
        LOGGER.warning("Relational cap check failed: %s", exc)
        telemetry.incr("memory.relational_failures")
        return
    if stored_recently >= max_per_period:
        telemetry.incr("memory.relational_cap_skipped")
        return

    provider = manager.summary_provider
    if provider is None:
        return

    transcript = manager._build_transcript(message_list)
    max_chars = int(getattr(cfg, "max_chars", 200) or 200)
    prompt = _relational_prompt(manager, transcript, max_chars)
    try:
        raw = (
            provider.generate(
                prompt, max_tokens=int(getattr(cfg, "max_tokens", 120) or 120)
            )
            or ""
        )
        telemetry.record_llm_usage(provider, "relational")
    except Exception as exc:  # pragma: no cover - provider runtime errors
        LOGGER.warning("Relational moment generation failed: %s", exc)
        telemetry.incr("memory.relational_failures")
        return

    line = _clean_relational_line(raw, max_chars)
    if line is None:
        # The common, correct outcome: most sessions change nothing.
        telemetry.incr("memory.relational_skipped")
        return

    salience = float(getattr(cfg, "salience", 0.55) or 0.55)
    if not manager.store_moment(line, salience=salience, source=_SOURCE):
        telemetry.incr("memory.relational_failures")
        return
    telemetry.incr("memory.relational_stored")


def _relational_prompt(
    manager: "MemoryManager", transcript: str, max_chars: int
) -> List[ChatMessage]:
    """The SKIP-default distillation prompt.

    Split out so the hard bans are assertable in tests without running a
    provider — they are the anti-fixation contract, not prompt decoration.
    """
    name = getattr(manager, "_char_name", None) or "Kurisu"
    character = getattr(manager.config, "character", None)
    return [
        ChatMessage(
            role="system",
            content=(
                f"You are {name}, privately noting whether anything in this "
                "conversation changed something BETWEEN you and this person.\n\n"
                "The bar is high and SKIP is the normal answer. Most "
                "conversations — however pleasant, however long, however much "
                "got discussed — change nothing between you, and for those the "
                "correct answer is the single word SKIP.\n\n"
                f"Answer with ONE sentence (under {max_chars} characters) ONLY "
                "if one of these genuinely happened: they told you something "
                "they had clearly not told you before; you disagreed about "
                "something that mattered and worked through it; they trusted "
                "you with something difficult; something shifted in how you "
                "two talk to each other.\n\n"
                'Write it in first person plural as a shared event — "we '
                'finally talked about why you left the lab", "you told me '
                'about your father for the first time".\n\n'
                "HARD RULES:\n"
                "- Record what HAPPENED between you, never your opinion of "
                "them. No adjectives judging the user (difficult, needy, "
                "distant, wonderful). No feelings about them at all — just "
                "the event.\n"
                "- Never about what you did FOR them, never anything they owe "
                "you.\n"
                "- Ground it only in this conversation; invent nothing.\n"
                "- No quotes, no stage directions, no preamble.\n\n"
                + persona_capsule(character)
                + "\n\nIf nothing meets the bar, reply with exactly: SKIP"
            ),
        ),
        ChatMessage(role="user", content=transcript),
    ]


def _clean_relational_line(raw: str, max_chars: int) -> Optional[str]:
    """First line, de-quoted, length-capped. None for empty or SKIP.

    Pure function — the parsing contract is tested without a provider.
    """
    text = (raw or "").strip()
    if not text:
        return None
    text = text.splitlines()[0].strip().strip('"').strip("'").strip()
    if not text:
        return None
    if text.upper().rstrip(".").startswith("SKIP"):
        return None
    return text[:max_chars]
