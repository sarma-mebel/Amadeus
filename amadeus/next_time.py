"""The "next time" thread: farewell writes one line, the next greeting picks
it up.

At a real goodbye (never the silent window_close finalize) the engine makes
one extra cheap-model call that notes the USER's open topic — "they were
about to submit the manuscript" — onto ``UserProfile.next_time_thread``. The
next session's greeting folds it in and consumes it exactly once AFTER a
greeting actually lands (the brief/diary consume-after-success contract);
stale entries are phrased tentatively and entries older than
``NEXT_TIME_THREAD_MAX_AGE_DAYS`` are dropped unconsumed (see
``UserProfile.take_next_time_thread``).

Free-function-over-owner pattern (same as ``amadeus/presence.py`` /
``memory/introspection.py``): the ChatEngine passes itself in and all state
stays on the engine/profile — this module exists for engine.py's line
budget, not as a second owner.
"""

from __future__ import annotations

from typing import Any, Dict, List

from . import telemetry
from .logging_utils import get_logger
from .persona import persona_capsule
from .providers.llm.base import ChatMessage

LOGGER = get_logger(__name__)


def note_next_time_thread(engine: Any, provider: Any, recent: List[str]) -> None:
    """Persist ONE line about the user's open topic for the next greeting.

    Runs only after a farewell line was actually produced. Best-effort by
    contract: any failure — missing profile, provider error, malformed
    output — counts ``farewell.next_time_thread_failed`` and never breaks
    the farewell path. SKIP is a legitimate answer (nothing genuinely open),
    not a failure. The note is about the USER's open topic, never about
    missing them (companion-ethics contract).
    """
    try:
        profile = engine.memory.profile
        if profile is None:
            # No profile store: the note has nowhere durable to live.
            telemetry.incr("farewell.next_time_thread_failed")
            return
        prompt = [
            ChatMessage(
                role="system",
                content=(
                    f"You are {engine.character_full_name}, privately noting "
                    "what to pick up next time with this person as they "
                    "leave. "
                    + persona_capsule(getattr(engine.config, "character", None))
                    + " From the exchange, write exactly ONE short line "
                    "(under 15 words) in third person about the USER's own "
                    "open topic — what they were doing or about to do, "
                    "e.g. 'they were about to submit the manuscript'. It "
                    "is a note about their life, never about you, never "
                    "about missing them or wanting them back. If nothing "
                    "is genuinely open, reply with the single word SKIP. "
                    "No quotes, no greetings."
                ),
            ),
            ChatMessage(role="user", content="\n".join(recent) or "(saying goodbye)"),
        ]
        raw = (provider.generate(prompt, max_tokens=60) or "").strip()
        telemetry.record_llm_usage(provider, "presence")
        line = ""
        if raw:
            line = raw.splitlines()[0].strip().strip('"').strip()
        if line.upper().rstrip(".") == "SKIP":
            return
        if not line:
            telemetry.incr("farewell.next_time_thread_failed")
            return
        profile.set_next_time_thread(line)
        engine.memory.persist_profile()
    except Exception as exc:  # never break the farewell path
        telemetry.incr("farewell.next_time_thread_failed")
        LOGGER.warning("Next-time thread note failed: %s", exc)


def next_time_greeting_fact(pending: Dict[str, Any]) -> str:
    """The greeting-prompt fact for a pending next-time thread.

    Fresh entries claim the follow-up ask slot ("ask about THIS"); stale
    ones get at most one tentative mention — a week-old note delivered as
    fresh news reads as a script, not memory.
    """
    if pending.get("stale"):
        return (
            "When you last spoke — a while ago now — this was left open; "
            "if you mention it, phrase it tentatively, things may have "
            "moved on since: " + str(pending["text"])
        )
    return (
        "Unresolved since last time (they left mid-thing — ask about "
        "THIS): " + str(pending["text"])
    )
