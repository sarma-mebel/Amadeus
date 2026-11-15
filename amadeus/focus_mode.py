"""Focus / co-working mode: "I'm going heads-down for a while."

The user tells her they're focusing for N minutes; she goes quiet for the
duration — the same suppression shape as her sleep window (idle chatter
off, ambient comments off, a pending reach-out HELD not consumed) — and
gives exactly one warm check-in line when the session ends, rather than a
report card ("nice stretch of work", never "did you finish?"). Chat always
works throughout; this mutes unprompted lines only, never the user turning
to her directly.

State lives on the :class:`~amadeus.engine.ChatEngine` instance
(``_focus_until``, ``_focus_checkin_pending``) — same free-function-over-
owner pattern as ``presence.py`` and ``sleep_schedule.py``: this module is
stateless, every mutable bit is an attribute on the engine. Unlike sleep
(a recurring daily window computed from the clock), a focus session is an
explicit, one-shot, per-session interval the user opts into and can end
early — closer in shape to a timer than a schedule.
"""

from __future__ import annotations

import datetime as dt
from typing import Any, Dict, Optional, TYPE_CHECKING

from . import telemetry
from .logging_utils import get_logger
from .persona import persona_capsule
from .providers.llm.base import ChatMessage

if TYPE_CHECKING:  # pragma: no cover - import cycle guard (engine imports us)
    from .engine import ChatEngine

LOGGER = get_logger(__name__)

MIN_MINUTES = 10
MAX_MINUTES = 180


def clamp_minutes(minutes: Any, default: int) -> int:
    """Clamp a requested focus duration to [MIN_MINUTES, MAX_MINUTES].

    Never raises — a garbage/missing value degrades to ``default`` (itself
    clamped), matching sleep_schedule's "malformed config degrades, never
    crashes the presence loop" stance.
    """
    try:
        value = int(minutes)
    except (TypeError, ValueError):
        value = int(default)
    return max(MIN_MINUTES, min(MAX_MINUTES, value))


def focus_config_status(config: Any) -> Dict[str, Any]:
    """The config-availability half for ``/api/status``: whether the
    feature is on and the default duration, so the UI can render the
    control before any session-scoped state exists. Unlike
    ``sleep_status`` this carries no live active/until — a focus session
    is per-session state (an explicit interval the user starts), not a
    clock-derived window computable from config alone; the live state
    rides ``/api/focus`` (action "status"), which is session-scoped like
    every other presence endpoint."""
    cfg = getattr(config, "focus", None)
    return {
        "enabled": bool(getattr(cfg, "enabled", True)) if cfg else True,
        "default_minutes": int(getattr(cfg, "default_minutes", 50)) if cfg else 50,
    }


def is_focus_active(engine: "ChatEngine", now: Optional[dt.datetime] = None) -> bool:
    """True while a focus session is currently running on this engine."""
    until = getattr(engine, "_focus_until", None)
    if until is None:
        return False
    return (now or dt.datetime.utcnow()) < until


def _clear(engine: "ChatEngine") -> None:
    engine._focus_until = None


def start_focus(
    engine: "ChatEngine", minutes: Any, *, now: Optional[dt.datetime] = None
) -> Dict[str, Any]:
    """Begin a focus session ``minutes`` long (clamped), returning status."""
    cfg = getattr(engine.config, "focus", None)
    default_minutes = int(getattr(cfg, "default_minutes", 50) or 50)
    now = now or dt.datetime.utcnow()
    clamped = clamp_minutes(minutes, default_minutes)
    engine._focus_until = now + dt.timedelta(minutes=clamped)
    engine._focus_started_at = now
    engine._focus_checkin_pending = False
    telemetry.incr("focus.sessions_started")
    return focus_status(engine, now=now)


def end_focus(
    engine: "ChatEngine", *, now: Optional[dt.datetime] = None
) -> Dict[str, Any]:
    """End the session early (or no-op if none was running) and, when a
    session really was active, arm exactly one check-in for this response.
    """
    now = now or dt.datetime.utcnow()
    was_active = is_focus_active(engine, now=now)
    until_before = getattr(engine, "_focus_until", None)
    _clear(engine)
    if was_active and until_before is not None:
        engine._focus_checkin_pending = True
    return focus_status(engine, now=now, consume_checkin=True)


def focus_status(
    engine: "ChatEngine",
    *,
    now: Optional[dt.datetime] = None,
    consume_checkin: bool = True,
) -> Dict[str, Any]:
    """Current state, as returned by every focus endpoint action.

    A status/poll call that OBSERVES expiry (the deadline already passed,
    nobody called ``end_focus`` explicitly) is exactly-once too: the first
    caller to notice gets the check-in text, later callers get ``active:
    False, checkin: None`` — sleep_schedule/reach-out's exactly-once pop
    shape, reused for a client-driven poll instead of a server-driven one.
    """
    now = now or dt.datetime.utcnow()
    until = getattr(engine, "_focus_until", None)
    active = is_focus_active(engine, now=now)
    if not active and until is not None:
        # The deadline passed since we last looked (nobody called
        # end_focus explicitly) — clear it so we don't keep re-detecting
        # expiry on every subsequent poll, and arm the one-shot check-in.
        _clear(engine)
        if consume_checkin:
            engine._focus_checkin_pending = True

    checkin: Optional[str] = None
    if consume_checkin and getattr(engine, "_focus_checkin_pending", False):
        checkin = _generate_checkin(
            engine, started_at=getattr(engine, "_focus_started_at", None), now=now
        )
        engine._focus_checkin_pending = False
        engine._focus_started_at = None

    remaining = None
    if active and until is not None:
        remaining = max(0, int((until - now).total_seconds()))

    return {
        "active": active,
        "until": until.isoformat() if (active and until is not None) else None,
        "remaining_seconds": remaining,
        "checkin": checkin,
    }


def _generate_checkin(
    engine: "ChatEngine",
    *,
    started_at: Optional[dt.datetime],
    now: dt.datetime,
) -> Optional[str]:
    """One warm, SKIP-allowed line for the end of a focus session.

    Never committed to memory (mirrors poke/idle/ambient's throwaway-flavor
    stance) — a focus check-in is a greeting-adjacent flourish, not
    something worth recalling weeks later. Failures degrade to None and
    count `focus.checkin_failures`; a model SKIP degrades to None too and
    is not counted as a failure (it's a legitimate verdict, like reach-out's
    SKIP).
    """
    cfg = getattr(engine.config, "focus", None)
    if not getattr(cfg, "checkin_enabled", True):
        return None
    provider = getattr(engine, "summary_provider", None)
    if provider is None:
        return None
    minutes = None
    if started_at is not None:
        minutes = max(1, int((now - started_at).total_seconds() // 60))
    time_note = f"roughly {minutes} minutes" if minutes else "a while"
    character = getattr(engine.config, "character", None)
    prompt = [
        ChatMessage(
            role="system",
            content=(
                f"You are {engine.character_full_name}. "
                + persona_capsule(character)
                + " The user just finished a focused work/study stretch "
                "they told you about in advance — you have been quiet the "
                "whole time, keeping them company without interrupting, "
                "like sitting in the same room while they work. Write "
                "exactly ONE short, warm line (under 20 words) welcoming "
                "them back. Roughly acknowledge the stretch of time if it "
                "feels natural, but NEVER ask whether they finished, "
                "whether it went well, or what they got done — that turns "
                "quiet company into a performance review. No praise for "
                "productivity, no guilt if they stopped early, no "
                "'hope you got a lot done'. Just warm and glad they're "
                "back. No stage directions, no quotation marks. If nothing "
                "genuine comes to mind, reply with exactly SKIP."
            ),
        ),
        ChatMessage(
            role="user",
            content=f"(focus session ended, {time_note})",
        ),
    ]
    try:
        text = (
            (
                provider.generate(
                    prompt,
                    max_tokens=int(getattr(cfg, "checkin_max_tokens", 60) or 60),
                )
                or ""
            )
            .strip()
            .strip('"')
            .strip()
        )
        telemetry.record_llm_usage(provider, "presence")
    except Exception as exc:  # pragma: no cover - provider runtime errors
        LOGGER.warning("Focus check-in generation failed: %s", exc)
        telemetry.incr("focus.checkin_failures")
        return None
    if not text or text.upper().startswith("SKIP"):
        return None
    telemetry.incr("focus.checkins")
    return text
