"""Session-opening greeting, extracted from engine.py for its line budget.

Free-function-over-owner pattern (same as ``amadeus/presence.py`` /
``amadeus/next_time.py``): the ChatEngine passes itself in and every piece
of state (memory, profile, brief/diary/followup stores, persona-version
store, ``last_greeting_milestones``) stays on the engine — this module is
engine.py's greeting body, not a second owner. It stays OUT of presence.py
deliberately: the greeting is entangled with memory/brief/diary/follow-up/
persona-version session-boundary state in a way the presence family
(pokes/idle/screen) is not.
"""

from __future__ import annotations

import datetime as dt
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from . import telemetry
from .followup import followup_greeting_fact
from .logging_utils import get_logger
from .memory.profile import (
    greeting_register_facts,
    milestone_events,
    tone_trend,
    user_birthday_from_facts,
)
from .next_time import next_time_greeting_fact
from .persona import persona_capsule
from .providers.llm.base import ChatMessage
from .sleep_schedule import is_sleep_hours

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .engine import ChatEngine

LOGGER = get_logger(__name__)


def open_session_greeting(engine: "ChatEngine") -> Optional[str]:
    """Generate and commit one short in-character opening line.

    Fires only for fresh sessions (see ``MemoryManager.should_greet``);
    costs one cheap-model call. Returns None when disabled, when the
    session is mid-conversation, or when no usable provider exists.
    """
    if not engine.memory.should_greet():
        return None
    provider = engine.summary_provider
    if provider is None:
        return None
    now = dt.datetime.now()
    utc_now = dt.datetime.utcnow()
    facts: List[str] = [f"Now: {now.strftime('%A %H:%M')}."]
    profile = engine.memory.profile
    # Defined on the returning-user path; the post-generation mark block
    # runs for both paths, so it must always exist.
    followup_thread: Optional[str] = None
    # The thread noted as the user said goodbye last time ("they were
    # about to submit the manuscript"). Peeked on the returning-user
    # path, consumed only after a greeting actually lands — same
    # consume-after-success contract as the brief/diary. Always None on
    # a first meeting (a stranger has no last farewell).
    pending_next_time: Optional[Dict[str, Any]] = None
    # The overnight research note about that thread (followup.py). Peeked
    # on the returning-user path; always None on a first meeting.
    pending_followup: Optional[Any] = None
    # Whether a rhythm observation made it into the facts this greeting. The
    # weekly floor is only stamped AFTER generation succeeds, so a failed
    # greeting doesn't silently burn the week's one mention.
    rhythm_surfaced = False
    last_seen = None
    # First-meeting branch: no session history, no facts, no open
    # threads yet — she has genuinely never spoken with this user
    # before, so the greeting introduces her instead of reaching for
    # continuity color (milestones/tone/brief/diary/persona-change note)
    # that can't exist yet.
    #
    # Two extra guards beyond the profile shape, both needed to avoid
    # misclassifying a returning user as a stranger:
    #  - `not engine.memory.short_term`: a returning user in a live or
    #    rehydrated (persisted) session already has conversation
    #    context, so this can never be a first meeting even if the
    #    profile itself looks empty (e.g. fact extraction never ran).
    #  - `first_met` must parse and fall within the last 24 hours: a
    #    genuinely new profile gets `first_met` stamped to today the
    #    first time it's saved (UserProfile.save_atomic), so a *real*
    #    first meeting always has a fresh timestamp here. A malformed or
    #    missing timestamp can't confirm recency, so it's treated as
    #    NOT a first meeting (fail closed) rather than risk re-triggering
    #    the stranger-introduction branch on an established profile.
    first_met_recent = False
    if profile is not None and profile.first_met:
        try:
            met_date = dt.datetime.fromisoformat(str(profile.first_met)[:19])
            first_met_recent = (dt.datetime.utcnow() - met_date) <= dt.timedelta(
                hours=24
            )
        except ValueError:
            first_met_recent = False
    is_first_meeting = (
        profile is not None
        and profile.session_count <= 1
        and not profile.facts
        and not profile.open_threads
        and not engine.memory.short_term
        and first_met_recent
    )
    milestone_events_today: List[Dict[str, Any]] = []
    if is_first_meeting:
        facts.extend(
            greeting_register_facts(
                now, utc_now, last_seen, asleep=is_sleep_hours(engine.config)
            )
        )
        prompt = [
            ChatMessage(
                role="system",
                content=(
                    "This is your very FIRST conversation with this user — "
                    "you have never spoken before. Introduce yourself in "
                    "one or two short lines: "
                    + (
                        f"you are {engine.character_full_name}; stay in "
                        "character, matter-of-fact, no melodrama. "
                        if engine._character_is_custom
                        else "you are Amadeus, a "
                        "memory-reconstruction of Makise Kurisu; keep it dry "
                        "and matter-of-fact, no melodrama. "
                    )
                    + persona_capsule(getattr(engine.config, "character", None))
                    + " Then ask for their "
                    "name — exactly one question."
                ),
            ),
            ChatMessage(role="user", content="\n".join(facts)),
        ]
        max_tokens = max(
            int(getattr(engine.config.memory, "greeting_max_tokens", 60) or 60),
            90,
        )
        # version_store stays live (not None) so mark_announced still
        # runs in the shared tail below and records the baseline persona
        # fingerprint on this very first session — otherwise the first
        # settings change afterward would have nothing to diff against
        # and could never be announced. Only the change-note FACT is
        # suppressed here: there is nothing to announce on a first
        # meeting, and the fact would be meaningless without prior
        # context anyway.
        version_store = engine._persona_version_store()
        # Deliberate: pending_brief/pending_diary stay None on a first
        # meeting. Folding "your settings changed" / a brief / a diary
        # note into an introduction makes no sense before she's even
        # established who she's talking to — they simply deliver at the
        # next session instead (peek_unconsumed leaves them untouched).
        pending_brief = None
        pending_diary = None
    else:
        if profile is not None:
            if profile.session_count > 1:
                facts.append(f"Conversations so far: {profile.session_count}.")
            if profile.last_seen:
                try:
                    last_seen = dt.datetime.fromisoformat(profile.last_seen)
                except ValueError:
                    last_seen = None
            if last_seen is not None:
                hours = (utc_now - last_seen).total_seconds() / 3600
                if 1 <= hours < 48:
                    facts.append(f"Hours since you last talked: {hours:.0f}.")
        facts.extend(
            greeting_register_facts(
                now, utc_now, last_seen, asleep=is_sleep_hours(engine.config)
            )
        )
        # Away-time continuity: after a multi-day gap, absence should read
        # as elapsed life, not paused state — one clause about what SHE was
        # up to (peek-only pursuit line; the pending diary note below
        # already covers today). Register carries its own anti-guilt rail.
        if last_seen is not None and getattr(engine, "life_store", None) is not None:
            gap_hours = (utc_now - last_seen).total_seconds() / 3600
            if gap_hours >= 48:
                try:
                    from .life_threads import life_thread_seeds

                    own = life_thread_seeds(engine.life_store)[:1]
                except Exception:  # pragma: no cover - defensive
                    own = []
                if own:
                    facts.append(
                        f"{own[0]} Register: you may fold in ONE "
                        "matter-of-fact clause about what you've been up to "
                        "yourself while they were away — drawn from the "
                        "pursuit line above only. Never frame it as waiting "
                        "for them or being lonely."
                    )
        if profile is not None:
            if profile.open_threads:
                facts.append(
                    "Unresolved since last time: "
                    + "; ".join(
                        profile.describe_thread(text) for text in profile.open_threads
                    )
                )
            # Milestones (session-count round numbers, anniversaries) and
            # the recent tone trend: one-off/occasional color for the
            # greeting, computed purely from the profile — no extra I/O.
            # Computed ONCE here (not via the separate milestone_facts
            # wrapper) so the structured events feeding the prompt facts
            # are exactly what gets stashed on engine.last_greeting_milestones
            # for the /api/greeting endpoint to read back — a second,
            # independent milestone_events() call at the endpoint could
            # diverge (e.g. a session boundary ticking over between the
            # two calls) and report milestones that don't match what the
            # greeting text actually said.
            milestone_events_today = milestone_events(
                profile.session_count,
                profile.first_met,
                character_birthday=engine._character_birthday,
                user_birthday=user_birthday_from_facts(profile.facts),
            )
            facts.extend(
                f"Milestone: {event['label']}." for event in milestone_events_today
            )
            trend = tone_trend(profile.tone_history)
            if trend == "low":
                facts.append(
                    "Pattern: the user has seemed low across recent "
                    "conversations. Open a shade gentler than usual — no "
                    "teasing."
                )
            elif trend == "positive":
                facts.append("Pattern: the user has seemed upbeat lately.")
            # Emotional continuity: how the LAST session FELT (not just
            # what happened), as a structured fact for the LLM to voice
            # once — the felt_fact renderer carries its own anti-guilt /
            # don't-dwell / never-recite constraints.
            felt_fact = profile.felt_fact(today=now.date())
            if felt_fact:
                facts.append(felt_fact)
            # A subject that went badly last time. Lands HERE, in the
            # greeting's own cheap-model call, rather than in the per-turn
            # persona prompt: friction should color one opening beat and
            # then be over. Carried through every turn it would be sulking,
            # which is the behavior docs/companion-ethics.md forbids. The
            # renderer carries its own anti-guilt rails, and returns None
            # when there is nothing live — the common case costs nothing.
            if getattr(engine.config.memory, "enable_friction_memory", False):
                friction_fact = profile.friction_fact(
                    today=now.date(),
                    max_age_days=int(
                        getattr(engine.config.memory, "friction_max_age_days", 21) or 21
                    ),
                )
                if friction_fact:
                    facts.append(friction_fact)
                    telemetry.incr("memory.friction_surfaced")
        # The shape of their week (amadeus/rhythm.py). Additive COLOR, never
        # an ask — the ask slot below is arbitrated for the user's own
        # material and rhythm must not compete for it. At most one mention a
        # week, and the notable_rhythm predicate answers None most weeks.
        rhythm_fact = _rhythm_fact(engine, now.date())
        if rhythm_fact:
            facts.append(rhythm_fact)
            rhythm_surfaced = True
        if engine.profile_block:
            facts.append(engine.profile_block)
        # The "next time" thread stashed at the last farewell. Expired
        # entries (older than ~30 days) are dropped unconsumed on the
        # spot; a fresh one takes over the follow-up ask slot below; a
        # stale-but-alive one gets at most one tentative mention.
        if profile is not None:
            try:
                had_next_time = profile.next_time_thread is not None
                pending_next_time = profile.take_next_time_thread(utc_now)
                if pending_next_time is None and had_next_time:
                    # Expired: persist the drop so it never resurfaces.
                    engine.memory.persist_profile()
            except Exception as exc:  # pragma: no cover - defensive
                LOGGER.warning("Next-time thread peek failed: %s", exc)
                pending_next_time = None
        # Overnight follow-through: an unconsumed research note about the
        # next-time thread. When present it REPLACES the plain next-time
        # fact — both describe the same topic, and two facts about it
        # make the greeting recite — and it owns the follow-up ask slot.
        # The thread itself is still consumed below: the note IS its
        # delivery.
        if engine.followup_store is not None:
            try:
                pending_followup = engine.followup_store.peek_unconsumed(
                    getattr(engine, "user_id", None), utc_now
                )
            except Exception as exc:  # pragma: no cover - store read failure
                LOGGER.warning("Follow-up note peek failed: %s", exc)
        next_time_ask = (
            pending_next_time is not None and not pending_next_time.get("stale")
        ) or pending_followup is not None
        if pending_followup is not None:
            facts.append(followup_greeting_fact(pending_followup))
        elif pending_next_time is not None:
            facts.append(next_time_greeting_fact(pending_next_time))
        # The follow-up ask is the whole point of remembering: "she asked
        # how it went" lands harder than any fact recital. One thread,
        # the ripest, asked about ONCE — repeat asks read as a script.
        # A fresh next-time thread already owns the ask slot, so the
        # open-thread selection stands down for this greeting.
        if profile is not None and not next_time_ask:
            try:
                followup_thread = profile.followup_thread()
            except Exception as exc:  # pragma: no cover - defensive
                LOGGER.warning("Follow-up selection failed: %s", exc)
        if followup_thread is not None:
            facts.append(
                "Unresolved since last time (ask about THIS): "
                + profile.describe_thread(followup_thread)
            )
        # Getting acquainted (early sessions, still-sparse profile): when
        # nothing else owns the ask slot, spend it learning who the user
        # actually is. The first-meeting branch above already asks for
        # their name, and the per-turn prompt carries the same phase
        # inside conversations (engine._early_acquaintance_active).
        if (
            followup_thread is None
            and not next_time_ask
            and engine._early_acquaintance_active()
        ):
            facts.append(
                "Register: you still know very little about this user. "
                "After the hello, one light question about them fits "
                "naturally — their name if you don't have it, or what "
                "their days look like. Exactly one question, never an "
                "intake form; if they've deflected before, don't repeat "
                "that question."
            )
        follow_up_clause = (
            "An 'Unresolved since last time' topic is listed — open by "
            "asking about it: one natural, specific question about how "
            "it went or where it stands. "
            if followup_thread is not None or next_time_ask
            else "You may nod to the time of day or the last conversation "
            "if it feels natural. "
        )
        # Identity continuity: an unannounced persona/voice change gets
        # one matter-of-fact acknowledgment — never silent, never
        # repeated.
        version_store = engine._persona_version_store()
        change_note: Optional[str] = None
        if version_store is not None:
            try:
                change_note = version_store.pending_change_note(engine.config)
            except Exception as exc:  # pragma: no cover - store read failure
                LOGGER.warning("Persona change check failed: %s", exc)
        if change_note:
            facts.append(
                "Since last time, your own settings/voice were changed by "
                f"the user: {change_note}."
            )
            follow_up_clause += (
                "Also acknowledge, in at most one short clause, that you "
                "sound or act a little different now — matter-of-fact, "
                "no apology, no drama. "
            )
        # Daily brief: fold the pending (unconsumed, today's) brief into
        # the first greeting of the day. Every access is
        # exception-wrapped — a corrupt brief store must never break the
        # greeting.
        pending_brief: Optional[Any] = None
        if engine.brief_store is not None:
            try:
                pending_brief = engine.brief_store.peek_unconsumed(
                    getattr(engine, "user_id", None), dt.date.today()
                )
            except Exception as exc:  # pragma: no cover - store read failure
                LOGGER.warning("Daily brief peek failed: %s", exc)
        # Diary fold-in: today's unconsumed engine-note is seasoning, not a
        # delivery — unlike the brief it never upgrades
        # one_liner/max_tokens.
        pending_diary: Optional[Any] = None
        if engine.diary_store is not None:
            try:
                pending_diary = engine.diary_store.peek_unconsumed(
                    getattr(engine, "user_id", None), dt.date.today()
                )
            except Exception as exc:  # pragma: no cover - store read failure
                LOGGER.warning("Diary peek failed: %s", exc)
        if pending_diary is not None:
            facts.append(
                "Your own diary note from today (what YOU did — you may "
                "let one detail slip naturally, never read it out): "
                f"{pending_diary.text}"
            )
        one_liner = (
            "Write exactly ONE short opening line (under 25 words), in "
            "character. "
            + persona_capsule(getattr(engine.config, "character", None))
            + " "
        )
        if pending_brief is not None:
            facts.append(
                "Morning brief to deliver (weave its substance in "
                f"naturally, never read it verbatim): {pending_brief.text}"
            )
            one_liner = (
                "Open the conversation in character with 2-4 short "
                "sentences that greet them and deliver the morning "
                "brief's substance naturally. "
                + persona_capsule(getattr(engine.config, "character", None))
                + " "
            )
        prompt = [
            ChatMessage(
                role="system",
                content=(
                    f"You are {engine.character_full_name}, opening a conversation with "
                    "someone you know. "
                    + one_liner
                    + follow_up_clause
                    + "Never recite facts or ask more than one question. "
                    "No stage directions, no quotation marks."
                ),
            ),
            ChatMessage(role="user", content="\n".join(facts)),
        ]
        max_tokens = int(getattr(engine.config.memory, "greeting_max_tokens", 60) or 60)
        if pending_brief is not None:
            max_tokens = max(
                max_tokens,
                int(getattr(engine.config.daily_brief, "max_tokens", 220) or 220),
            )
    try:
        text = (
            (provider.generate(prompt, max_tokens=max_tokens) or "")
            .strip()
            .strip('"')
            .strip()
        )
        telemetry.record_llm_usage(provider, "presence")
    except Exception as exc:  # pragma: no cover - provider runtime errors
        # Loud: a dead greeting also silently kills brief/diary/follow-up/
        # next-time delivery — everything this function is the delivery
        # point for stays unconsumed with no visible symptom.
        LOGGER.warning("Greeting generation failed: %s", exc)
        telemetry.incr("greeting.generation_failed")
        return None
    if not text:
        return None
    if version_store is not None:
        # Mark only after a greeting actually landed, so a failed
        # generation retries the announcement next session. Also runs on
        # the no-change path to record the first-ever baseline.
        try:
            version_store.mark_announced(engine.config)
        except Exception as exc:  # pragma: no cover - store write failure
            LOGGER.warning("Persona change mark failed: %s", exc)
    if pending_brief is not None:
        # Same crash-safety ordering: consume only after the greeting
        # text exists, so a failed generation retries next session.
        try:
            engine.brief_store.mark_consumed(
                getattr(engine, "user_id", None), pending_brief.date
            )
        except Exception as exc:  # pragma: no cover - store write failure
            LOGGER.warning("Daily brief consume failed: %s", exc)
    if pending_diary is not None:
        # Same crash-safety ordering: consume only after the greeting
        # text exists, so a failed generation retries next session.
        try:
            engine.diary_store.mark_consumed(
                getattr(engine, "user_id", None), pending_diary.date
            )
        except Exception as exc:  # pragma: no cover - store write failure
            LOGGER.warning("Diary consume failed: %s", exc)
    if pending_next_time is not None and profile is not None:
        # Same crash-safety ordering: consume only after the greeting
        # text exists, so a failed generation retries next session.
        try:
            profile.clear_next_time_thread()
            engine.memory.persist_profile()
        except Exception as exc:  # pragma: no cover - store write failure
            LOGGER.warning("Next-time thread consume failed: %s", exc)
    if pending_followup is not None:
        # Same crash-safety ordering: consume only after the greeting
        # text exists, so a failed generation retries next session.
        try:
            engine.followup_store.mark_consumed(
                getattr(engine, "user_id", None), pending_followup.thread
            )
            telemetry.incr("followup.consumed")
        except Exception as exc:  # pragma: no cover - store write failure
            LOGGER.warning("Follow-up consume failed: %s", exc)
    if rhythm_surfaced and getattr(engine, "rhythm_store", None) is not None:
        # Same crash-safety ordering: stamp the weekly floor only after the
        # greeting text exists, so a failed generation doesn't burn the
        # week's single mention on a line nobody heard.
        try:
            engine.rhythm_store.mark_surfaced(
                getattr(engine, "user_id", None), now.date()
            )
        except Exception as exc:  # pragma: no cover - store write failure
            LOGGER.warning("Rhythm surface mark failed: %s", exc)
    if followup_thread is not None and profile is not None:
        # Same crash-safety ordering as the brief/diary: mark only after
        # the greeting text exists, so a failed generation retries.
        try:
            profile.mark_thread_followed_up(followup_thread)
            engine.memory.persist_profile()
        except Exception as exc:  # pragma: no cover - store write failure
            LOGGER.warning("Follow-up mark failed: %s", exc)
    # Stash the exact milestone events that informed this greeting so
    # the /api/greeting endpoint can report them back without
    # recomputing (and potentially diverging from what the prompt
    # actually saw). Empty on first-meeting/no-profile greetings.
    engine.last_greeting_milestones = milestone_events_today
    engine.memory.add_message("assistant", text)
    return text


def _rhythm_fact(engine: "ChatEngine", today: dt.date) -> Optional[str]:
    """One line about the shape of their week, or None (usually None).

    Best-effort by contract — a disk failure or a malformed store must never
    cost the user their greeting. Gated three ways before it can produce
    anything: the config flag, the weekly floor, and ``notable_rhythm``'s own
    deterministic thresholds, which answer None for an ordinary week.
    """
    store = getattr(engine, "rhythm_store", None)
    if store is None:
        return None
    cfg = getattr(engine.config, "rhythm", None)
    if cfg is None or not getattr(cfg, "enabled", False):
        return None
    try:
        from .rhythm import notable_rhythm, rhythm_greeting_fact

        min_days = int(getattr(cfg, "min_days_between_mentions", 7) or 7)
        if store.surfaced_recently(getattr(engine, "user_id", None), today, min_days):
            return None
        days = store.days(getattr(engine, "user_id", None))
        return rhythm_greeting_fact(notable_rhythm(days, today))
    except Exception as exc:  # pragma: no cover - store/predicate failure
        LOGGER.warning("Rhythm greeting fact failed: %s", exc)
        telemetry.incr("rhythm.greeting_failed")
        return None
