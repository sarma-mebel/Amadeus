"""Shared-moments memory: opt-in, one-line durable memories distilled from a
finished co-activity session (read-along or watch-together).

``amadeus/co_activity.py`` keeps read-along/watch-together deliberately
forgetful: per-page/per-frame screenshots and the rolling in-RAM notes buffer
never touch memory or disk (see docs/companion-ethics.md). That contract is
unchanged. This module adds exactly one additional, OPT-IN step at session
end: when ``moments.enabled`` and the session accumulated enough notes to be
worth remembering, one cheap-model call distills the whole session into a
single durable line ("we watched the Evangelion finale together — user loved
the ending") and stores it as a recallable memory. The frames are still never
stored; only that one sentence is, and only when the feature is on.

Free-function-over-owner pattern (same as ``next_time.py``/``presence.py``):
takes the :class:`~amadeus.engine.ChatEngine` as first argument, all state
(the daily-cap check) is read live off ``engine.memory``, nothing new is
held on the engine itself.
"""

from __future__ import annotations

import datetime as dt
from typing import TYPE_CHECKING, Iterable, Optional

from . import telemetry
from .logging_utils import get_logger
from .persona import persona_capsule
from .providers.llm.base import ChatMessage

if TYPE_CHECKING:  # pragma: no cover - import cycle guard
    from .engine import ChatEngine

LOGGER = get_logger(__name__)

# Below this many notes, a co-activity session is too thin to be worth a
# durable memory (a two-page glance or a one-frame check-in isn't "we did
# something together" yet) — cheaper than a wasted LLM call, and avoids
# clutter from every incidental screen check accreting a memory record.
_MIN_NOTES = 2


def _min_notes_met(notes: Optional[Iterable[str]]) -> bool:
    if not notes:
        return False
    count = 0
    for _ in notes:
        count += 1
        if count >= _MIN_NOTES:
            return True
    return False


def maybe_store_moment(
    engine: "ChatEngine",
    *,
    activity: str,
    notes: Optional[Iterable[str]],
) -> None:
    """Distill a just-ended co-activity session into one durable memory.

    Called from ``co_activity.read_along_end``/``watch_end`` after the
    session's sign-off has already been generated — this never blocks or
    changes that return value, it is a pure side effect. Best-effort by
    contract: any failure (missing config, no long-term memory, provider
    error, malformed output) counts a telemetry counter and returns, never
    raising into the end-of-session path. ``activity`` is a short label
    ("reading session", "watch-together session") used only in the prompt.
    """
    try:
        _maybe_store_moment(engine, activity=activity, notes=notes)
    except Exception as exc:  # never break the co-activity end path
        LOGGER.warning("Moment distillation failed: %s", exc)
        telemetry.incr("moments.failures")


def _maybe_store_moment(
    engine: "ChatEngine",
    *,
    activity: str,
    notes: Optional[Iterable[str]],
) -> None:
    moments_cfg = getattr(engine.config, "moments", None)
    if moments_cfg is None or not getattr(moments_cfg, "enabled", False):
        return
    if not getattr(engine.config.memory, "enable_long_term", False):
        return
    notes_list = [n for n in (notes or []) if n]
    if len(notes_list) < _MIN_NOTES:
        return

    memory = engine.memory
    if memory is None or not getattr(memory, "vector_store", None):
        return

    max_per_day = int(getattr(moments_cfg, "max_per_day", 0) or 0)
    if max_per_day > 0:
        cutoff = dt.datetime.utcnow() - dt.timedelta(hours=24)
        try:
            # Scoped to THIS writer's moments: relational-memory moments
            # share type "moment" but carry source="relational" and their
            # own independent weekly cap (memory/relational.py) — counting
            # them here would let a relational record silently eat the
            # co-activity budget. "co_activity" is store_moment's default
            # source, stamped on every moment this module writes.
            stored_today = memory.count_moments_since(cutoff, source="co_activity")
        except Exception as exc:  # pragma: no cover - store best-effort
            LOGGER.warning("Moment daily-cap check failed: %s", exc)
            telemetry.incr("moments.failures")
            return
        if stored_today >= max_per_day:
            telemetry.incr("moments.daily_cap_skipped")
            return

    provider = engine.summary_provider or engine.llm
    if provider is None:
        telemetry.incr("moments.failures")
        return

    max_chars = int(getattr(moments_cfg, "max_chars", 200) or 200)
    notes_block = "\n".join(f"- {n}" for n in notes_list)
    prompt = [
        ChatMessage(
            role="system",
            content=(
                f"You are {engine.character_full_name}. You and the user "
                f"just finished a {activity} together. Below are your "
                "private notes from the session. Distill them into exactly "
                "ONE sentence, under "
                f"{max_chars} characters, written in first-person-plural "
                "('we watched...', 'we read...') as a memory of something "
                "you shared together — not a summary of the notes, a "
                "memory of the experience. "
                + persona_capsule(getattr(engine.config, "character", None))
                + " Ground it ONLY in the notes below; never invent details "
                "that are not there. If the session was too thin or "
                "uneventful to be worth remembering, reply with the single "
                "word SKIP. No quotes, no stage directions."
            ),
        ),
        ChatMessage(role="user", content=notes_block),
    ]
    try:
        raw = (provider.generate(prompt, max_tokens=120) or "").strip()
        telemetry.record_llm_usage(provider, "moments")
    except Exception as exc:  # pragma: no cover - provider runtime errors
        LOGGER.warning("Moment generation failed: %s", exc)
        telemetry.incr("moments.failures")
        return

    line = raw.splitlines()[0].strip().strip('"').strip() if raw else ""
    if not line or line.upper().rstrip(".") == "SKIP":
        telemetry.incr("moments.skipped")
        return
    if max_chars and len(line) > max_chars:
        line = line[:max_chars].rstrip()

    ok = memory.store_moment(line)
    if not ok:
        telemetry.incr("moments.failures")
        return
    telemetry.incr("moments.stored")
