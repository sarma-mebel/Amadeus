"""Unprompted, server-gated presence lines: pokes, idle chatter, and screen
peeks.

Free functions taking the :class:`~amadeus.engine.ChatEngine` (the
sibling-module convention already used by ``memory/introspection.py`` for
``MemoryManager``) — ``ChatEngine`` keeps thin delegating methods with the
same names/signatures, so every existing caller and test is untouched.

What belongs here: the family of features that fire WITHOUT the user having
just sent a message — a click on the avatar, an idle timer, an ambient
screen glance. All share the same shape: gated behind a config flag,
throttled by a per-engine cooldown timestamp, generated with a short
single-purpose prompt, and (except idle chatter) never committed to memory
— they are throwaway flavor, not a real conversational turn.

What does NOT belong here: turn orchestration (``prepare_turn``/
``complete_turn``) and the greeting/farewell pair. Those two look
presence-shaped (also unprompted, also cooldown-gated via
``_presence_cooldown_ready``) but are entangled with session-boundary state
— memory summarization, the daily brief, the diary, persona-version
change notes — that lives in ``engine.py`` alongside the rest of turn
lifecycle, not in this module. The stateful shared-activity pair ("read
together" pages, "watch together" frames) lives in ``co_activity.py`` —
same conventions, extracted for the line budget.

State discipline: exactly like ``introspection.py`` over ``MemoryManager``,
this module is stateless — every mutable bit (cooldown timestamps) stays
an attribute on the engine instance itself, read and written in place.
``_presence_cooldown_ready`` also stays defined on the engine (not here)
because the greeting/farewell methods in engine.py use it too.
"""

from __future__ import annotations

import datetime as dt
from typing import TYPE_CHECKING, Optional

from . import telemetry
from .logging_utils import get_logger
from .focus_mode import is_focus_active
from .persona import persona_capsule
from .providers.llm.base import ChatMessage, ImageContent
from .sleep_schedule import is_sleep_hours

if TYPE_CHECKING:  # pragma: no cover - import cycle guard (engine imports us)
    from .engine import ChatEngine

LOGGER = get_logger(__name__)


_POKE_REGION_CONTEXT = {
    "head": (
        "poked/patted you on the HEAD, like you're a child or a pet. "
        "React flustered and indignant about the head-pat specifically"
    ),
    "chest": (
        "poked you in the CHEST. React scandalized and snappy about " "personal space"
    ),
    "hands": (
        "poked at your HAND. React with dry exasperation about them "
        "fiddling with your hand"
    ),
    # Desktop-pet-only regions: the overlay window fires these when the
    # user drags the window around or flips on click-through mode.
    "dragged": (
        "grabbed your window and DRAGGED you across the screen. React "
        "indignant about being hauled around like furniture"
    ),
    "click_through": (
        "made your window click-through — clicks now pass right past "
        "you to the desktop. React with a dry parting jab about being "
        "reduced to decoration"
    ),
    "woke_from_doze": (
        "woke you up — you'd dozed off on screen. React groggy and "
        "mildly embarrassed, half-denying you were asleep"
    ),
}

# 'head_pat' (press-and-hold on the head, vs a quick 'head' poke) is
# deliberately NOT in the static map: its tone follows the relationship
# arc, so it's built per-call in _head_pat_context().
_HEAD_PAT_BY_WARMTH = [
    (
        0.35,
        "patted you on the HEAD, resting their hand there. You barely "
        "know them. React flustered and indignant — personal space, "
        "you're not a pet",
    ),
    (
        0.65,
        "patted you on the HEAD, resting their hand there. You know "
        "them fairly well by now. React flustered, protesting on "
        "principle, but without any real bite",
    ),
    (
        float("inf"),
        "patted you on the HEAD, resting their hand there. You've grown "
        "genuinely comfortable with them. React begrudgingly pleased — "
        "deflect, but let a hint of warmth slip through",
    ),
]


def _relationship_warmth(engine: "ChatEngine") -> float:
    """0..1 warmth scalar derived from the persisted relationship state.

    Mirrors the frontend's arcWarmth formula exactly (0.6*trust +
    0.4*normalized curiosity-vs-annoyance) so her on-screen
    attentiveness and her verbal tone thaw in lockstep, and defaults to
    the same 0.3 the client assumes when no profile exists yet.
    """
    profile = getattr(engine.memory, "profile", None)
    if profile is None:
        return 0.3
    trust = float(getattr(profile, "trust_level", 0.0) or 0.0)
    curiosity = float(getattr(profile, "arc_curiosity_vs_annoyance", 0.0) or 0.0)
    warmth = 0.6 * trust + 0.4 * (curiosity + 1.0) / 2.0
    return max(0.0, min(1.0, warmth))


def _head_pat_context(engine: "ChatEngine") -> str:
    warmth = _relationship_warmth(engine)
    for ceiling, context in _HEAD_PAT_BY_WARMTH:
        if warmth < ceiling:
            return context
    return _HEAD_PAT_BY_WARMTH[-1][1]


def poke_reaction(engine: "ChatEngine", region: Optional[str] = None) -> Optional[str]:
    """One short in-character line reacting to being "poked" (the user
    clicking/tapping the avatar) — a companion-app staple this assistant
    was missing entirely; clicking the model just orbited the camera.

    Server-gated the same way as the greeting: disabled by default, and
    cooldown-limited so a client bug or deliberate spam can't turn a
    click into a burst of LLM calls.
    """
    if not getattr(engine.config.memory, "enable_poke_reactions", False):
        return None
    cooldown = float(getattr(engine.config.memory, "poke_cooldown_seconds", 8.0) or 8.0)
    if not engine._presence_cooldown_ready(engine._last_poke_at, cooldown):
        return None
    provider = engine.summary_provider
    if provider is None:
        return None
    region_key = (region or "").strip().lower() or None
    if region_key == "head_pat":
        region_context = _head_pat_context(engine)
    else:
        region_context = _POKE_REGION_CONTEXT.get(region_key or "")
    if region_context:
        situation = f"The user just {region_context}."
    else:
        situation = (
            "The user just poked/clicked you on screen, interrupting "
            "nothing in particular. React mildly annoyed"
        ) + "."
    # Sleep window: a poke still gets a line (she's sleepy, not gone), but
    # the register is groggy-half-awake instead of the usual reaction.
    if is_sleep_hours(engine.config):
        situation += (
            " IMPORTANT: it is the middle of your night and you were "
            "ASLEEP — the poke just woke you. React groggy and half-awake: "
            "drowsy, mumbling, mildly indignant at being woken, maybe "
            "threatening to go right back to sleep. Never fully alert."
        )
    # Focus session: a poke is the USER reaching out (never suppressed —
    # only unprompted lines are held during focus), but the register
    # softens to quiet-company instead of the usual annoyed/teasing beat.
    elif is_focus_active(engine):
        situation += (
            " IMPORTANT: the user told you they're heads-down working "
            "right now and you've been quietly keeping them company. "
            "React soft and brief — glad they looked over, not annoyed at "
            "the interruption. Something like 'still here' in spirit. "
            "Never ask if they're done or how it's going."
        )
    prompt = [
        ChatMessage(
            role="system",
            content=(
                f"You are {engine.character_full_name}. "
                + situation
                + " Reply with exactly ONE short line (under 15 words), in "
                "character. "
                + persona_capsule(getattr(engine.config, "character", None))
                + " No stage directions, no quotation "
                "marks, never actually helpful or informative."
            ),
        ),
        ChatMessage(
            role="user",
            content=f"(poked: {region_key})" if region_key else "(poked)",
        ),
    ]
    try:
        text = (
            (
                provider.generate(
                    prompt,
                    max_tokens=int(
                        getattr(engine.config.memory, "poke_max_tokens", 40) or 40
                    ),
                )
                or ""
            )
            .strip()
            .strip('"')
            .strip()
        )
        telemetry.record_llm_usage(provider, "presence")
    except Exception as exc:  # pragma: no cover - provider runtime errors
        LOGGER.warning("Poke reaction generation failed: %s", exc)
        telemetry.incr("presence.poke_failures")
        return None
    engine._last_poke_at = dt.datetime.utcnow()
    if not text:
        return None
    # Deliberately not committed to memory (unlike the greeting/idle
    # chatter): a poke reaction is throwaway flavor, not a real
    # conversational turn worth recalling or summarizing later.
    return text


def screen_comment(
    engine: "ChatEngine", image_mime: str, image_data: str
) -> Optional[str]:
    """One short unprompted line about what's on the user's screen —
    pet mode's opt-in screen-companion feature.

    Poke-shaped: server-gated (vision.enabled + a vision-capable chat
    LLM), cooldown-limited, and never committed to memory. The image
    lives only in this one request, like every vision turn.
    """
    if not getattr(getattr(engine.config, "vision", None), "enabled", False):
        return None
    cooldown = float(
        getattr(engine.config.vision, "screen_peek_cooldown_seconds", 120.0) or 120.0
    )
    if not engine._presence_cooldown_ready(engine._last_screen_peek_at, cooldown):
        return None
    supports = getattr(engine.llm, "supports_vision", None)
    if not callable(supports) or not supports():
        return None
    # Her previous line rides along so an unchanged screen reads as "nothing
    # new" instead of prompting a near-verbatim repeat — without it she
    # commented on the same file every glance, freshly each time (live
    # finding, 2026-08-03: two consecutive glances at one article produced
    # the same "of all things..." line twice).
    previous_line = (
        f"\nYour previous glance line was: {engine._last_screen_comment} "
        "— never repeat it or reuse its subject."
        if engine._last_screen_comment
        else ""
    )
    prompt = [
        ChatMessage(
            role="system",
            content=(
                f"You are {engine.character_full_name}, keeping the user "
                "company on their desktop, and you just glanced at their "
                "screen. React with exactly ONE short line (under 20 "
                "words) about what they seem to be doing, in character. "
                + persona_capsule(getattr(engine.config, "character", None))
                + " Be specific to what you actually see, never a summary "
                "or unsolicited advice. If the screen looks essentially "
                "unchanged from your previous glance, or nothing on it "
                "genuinely earns a remark, reply with exactly SKIP — "
                "glancing often and speaking rarely is the point. If the "
                "screen looks private or sensitive (messages, banking, "
                "credentials), tease that you saw nothing instead of "
                "describing it. No stage directions, no quotation marks."
                + previous_line
            ),
        ),
        ChatMessage(
            role="user",
            content="(a glance at the user's screen)",
            images=[ImageContent(mime=image_mime, data=image_data)],
        ),
    ]
    try:
        text = (
            (
                engine.llm.generate(
                    prompt,
                    max_tokens=int(
                        getattr(engine.config.vision, "screen_peek_max_tokens", 800)
                        or 800
                    ),
                )
                or ""
            )
            .strip()
            .strip('"')
            .strip()
        )
        telemetry.record_llm_usage(engine.llm, "presence")
    except Exception as exc:  # pragma: no cover - provider runtime errors
        LOGGER.warning("Screen comment generation failed: %s", exc)
        telemetry.incr("presence.screen_peek_failures")
        return None
    engine._last_screen_peek_at = dt.datetime.utcnow()
    # SKIP: she looked and chose silence (unchanged/unremarkable screen).
    # Still arms the cooldown above — the rate limit is on LLM calls, not
    # on spoken lines — and counted so a too-quiet feel is diagnosable.
    if text.upper().startswith("SKIP"):
        telemetry.incr("presence.screen_peek_skips")
        return None
    # Like poke reactions, deliberately not committed to memory: an
    # ambient glance is throwaway flavor, and persisting screen contents
    # (even as text) would violate the vision privacy contract.
    if text:
        engine._last_screen_comment = text
    return text or None


#: Whitelisted ambient signal kinds (client-reported transitions from the
#: Tauri `ambient_signals` poll — anything else is dropped unanswered).
_AMBIENT_KINDS = {"focus", "battery", "music"}


def ambient_comment(engine: "ChatEngine", kind: str, detail: str) -> Optional[str]:
    """One short line about an ambient desktop signal — the cheap
    continuous-awareness layer (focused-app stretches, battery state,
    now-playing). Poke-shaped: server-gated, cooldown-limited, never
    committed to memory (ambient glances are throwaway flavor). The client
    owns signal *detection*; this owns whether and how she reacts.
    """
    cfg = getattr(engine.config, "ambient", None)
    if not getattr(cfg, "enabled", True):
        return None
    if is_sleep_hours(engine.config):
        telemetry.incr("presence.sleep_suppressed")
        return None
    # Focus session: quiet company, not a ping generator — mirrors the
    # sleep-window check right above it (amadeus/focus_mode.py).
    if is_focus_active(engine):
        telemetry.incr("presence.focus_suppressed")
        return None
    kind_key = (kind or "").strip().lower()
    detail_text = " ".join(str(detail or "").split())[:200]
    if kind_key not in _AMBIENT_KINDS or not detail_text:
        return None
    cooldown = float(getattr(cfg, "cooldown_seconds", 600.0) or 600.0)
    if not engine._presence_cooldown_ready(engine._last_ambient_at, cooldown):
        return None
    provider = engine.summary_provider
    if provider is None:
        return None
    prompt = [
        ChatMessage(
            role="system",
            content=(
                f"You are {engine.character_full_name}, keeping the user "
                "company on their desktop. You just noticed: "
                + detail_text
                + " React with exactly ONE short line (under 20 words), in "
                "character. "
                + persona_capsule(getattr(engine.config, "character", None))
                + " Casual and observational — a flatmate glancing over, "
                "not a monitor. Low battery may get one practical nudge; "
                "everything else gets no advice. If it doesn't genuinely "
                "earn a remark, reply with exactly SKIP — noticing often "
                "and speaking rarely is the point. No stage directions, no "
                "quotation marks."
            ),
        ),
        ChatMessage(role="user", content=f"(ambient: {kind_key})"),
    ]
    try:
        text = (
            (
                provider.generate(
                    prompt, max_tokens=int(getattr(cfg, "max_tokens", 60) or 60)
                )
                or ""
            )
            .strip()
            .strip('"')
            .strip()
        )
        telemetry.record_llm_usage(provider, "presence")
    except Exception as exc:  # pragma: no cover - provider runtime errors
        LOGGER.warning("Ambient comment generation failed: %s", exc)
        telemetry.incr("presence.ambient_failures")
        return None
    # SKIP still arms the cooldown: the rate limit is on LLM calls.
    engine._last_ambient_at = dt.datetime.utcnow()
    if not text or text.upper().startswith("SKIP"):
        telemetry.incr("presence.ambient_skips")
        return None
    return text


def idle_chatter(engine: "ChatEngine") -> Optional[str]:
    """One short unprompted line after a quiet stretch — the idle-chatter
    half of presence (the greeting covers session *starts*; this covers
    mid-session silence). Reuses should_greet's session-freshness check
    inverted: only fires when there IS recent conversation to riff on,
    not on a fresh/empty session (that's the greeting's job).
    """
    if not getattr(engine.config.memory, "enable_idle_chatter", False):
        return None
    # Sleep window: she doesn't monologue in her sleep. Counted (not a
    # failure, but "why has she gone quiet" must be diagnosable from the
    # status counters rather than looking like a dead presence loop).
    if is_sleep_hours(engine.config):
        telemetry.incr("presence.sleep_suppressed")
        return None
    # Focus session: she doesn't monologue while the user is heads-down —
    # same suppression shape as the sleep window (amadeus/focus_mode.py).
    if is_focus_active(engine):
        telemetry.incr("presence.focus_suppressed")
        return None
    if not engine.memory.short_term:
        return None  # nothing to riff on; a fresh session gets a greeting instead
    cooldown = float(
        getattr(engine.config.memory, "idle_cooldown_seconds", 240.0) or 240.0
    )
    if not engine._presence_cooldown_ready(engine._last_idle_chatter_at, cooldown):
        return None
    provider = engine.summary_provider
    if provider is None:
        return None
    facts = [f"Now: {dt.datetime.now().strftime('%A %H:%M')}."]
    if engine.kurisu_state.last_topic:
        facts.append(f"Last topic discussed: {engine.kurisu_state.last_topic}.")
    # Peek-only, never consume: idle chatter isn't the delivery moment,
    # only the greeting consumes the diary.
    if engine.diary_store is not None:
        try:
            entry = engine.diary_store.get(getattr(engine, "user_id", None))
            if entry is not None and entry.date == dt.date.today().isoformat():
                facts.append(
                    f"What you did earlier today (your own note): {entry.text}"
                )
        except Exception as exc:  # pragma: no cover - store read failure
            LOGGER.warning("Diary peek for idle chatter failed: %s", exc)
    # Same peek-only stance for her ongoing pursuits (life_threads.py):
    # idle chatter may mention one, never advances one.
    # An opinion-ask outranks the plain peek when one is due: it is the same
    # thread either way, and asking is the more interesting version of
    # mentioning it. `ask_due` answers None for almost every idle beat (a
    # 21-day floor plus a one-way per-thread stamp), so this stays rare.
    ask_thread = None
    if getattr(engine, "life_store", None) is not None:
        from .life_threads import (
            ask_due,
            ask_thread_fact,
            life_thread_seeds,
        )

        life_cfg = getattr(engine.config, "life_threads", None)
        if life_cfg is not None and getattr(life_cfg, "enable_opinion_asks", False):
            ask_thread = ask_due(
                engine.life_store,
                dt.date.today(),
                min_days_between_asks=int(
                    getattr(life_cfg, "min_days_between_asks", 21) or 21
                ),
            )
        ask_fact = ask_thread_fact(ask_thread)
        if ask_fact:
            facts.append(ask_fact)
        else:
            facts.extend(life_thread_seeds(engine.life_store)[:1])
    prompt = [
        ChatMessage(
            role="system",
            content=(
                f"You are {engine.character_full_name}. The user has gone quiet mid-"
                "conversation for a while. Write exactly ONE short, "
                "unprompted line (under 20 words) — in character, never "
                "neediness. "
                + persona_capsule(getattr(engine.config, "character", None))
                + " You may nod to the last topic if "
                "natural. You may mention one thing from your own day "
                "if listed. No stage directions, no quotation marks, "
                "never more than one question."
            ),
        ),
        ChatMessage(role="user", content="\n".join(facts)),
    ]
    try:
        text = (
            (
                provider.generate(
                    prompt,
                    max_tokens=int(
                        getattr(engine.config.memory, "idle_max_tokens", 60) or 60
                    ),
                )
                or ""
            )
            .strip()
            .strip('"')
            .strip()
        )
        telemetry.record_llm_usage(provider, "presence")
    except Exception as exc:  # pragma: no cover - provider runtime errors
        LOGGER.warning("Idle chatter generation failed: %s", exc)
        telemetry.incr("presence.idle_failures")
        return None
    engine._last_idle_chatter_at = dt.datetime.utcnow()
    if not text:
        return None
    if ask_thread is not None:
        # Mark only after the line exists (same consume-after-success
        # ordering as the greeting's brief/diary/follow-up). One-way: this
        # thread is never asked about again, whatever the user says or
        # doesn't say back.
        try:
            engine.life_store.mark_asked(ask_thread.topic, dt.date.today())
            telemetry.incr("life.ask_surfaced")
            # Arm the ONE next user turn to be read as their answer
            # (engine.prepare_turn consumes and disarms it). A single-slot
            # flag, so a user who never replies simply leaves it to be
            # overwritten or ignored — nothing tracks the non-answer.
            engine._pending_life_ask = ask_thread.topic
        except Exception as exc:  # pragma: no cover - store write failure
            LOGGER.warning("Life-thread ask mark failed: %s", exc)
    # presence_delivery: idle chatter fires while the user is away, so a
    # plain add_message could trip the gap-finalize session boundary
    # (phantom session_count/trust/arc drift) and stamp last_seen for a
    # line she said to an empty room.
    engine.memory.add_message("assistant", text, presence_delivery=True)
    return text
