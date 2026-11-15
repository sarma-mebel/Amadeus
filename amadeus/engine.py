"""Shared chat-turn orchestration for the CLI and the web app.

Both front-ends drive the same core: build the system prompt, assemble the
request context (committing the user message to memory), run guardrails,
optionally run the tool loop, then commit the reply. Generation
itself stays with the caller — it is the one genuinely channel-specific part
(stdout streaming vs WebSocket frames vs blocking POST).

This module must stay free of web imports so the CLI never pulls FastAPI.

The poke/idle-chatter/screen-comment presence family lives in
``presence.py`` (and the stateful read-along/watch-together pair in
``co_activity.py``) instead of here — see presence.py's docstring for the
line: unprompted lines gated by a config flag and a per-engine cooldown,
almost never persisted to memory. The greeting and farewell look
presence-shaped too (also unprompted, also cooldown-gated) but stay in
this file because they're entangled with session-boundary state — memory
summarization, the daily brief, the diary, persona-version change notes
— that belongs with the rest of turn lifecycle, not with the stateless
presence functions.
"""

from __future__ import annotations

import datetime as dt
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from . import (
    co_activity,
    greeting,
    presence,
    provider_health,
    telemetry,
    vocal_tone_stash,
)
from .agents.factory import create_tool_registry
from .config import Config, PersonaSettings
from .logging_utils import get_logger
from .memory.context import assemble_turn_context
from .memory.manager import EmbeddingProvider, MemoryManager
from .memory.profile import (
    early_acquaintance_active,
    parse_birthday,
    quantize_trust,
)
from .persona import (
    KurisuArc,
    KurisuState,
    build_system_prompt,
    derive_topic,
    is_late_night,
    persona_capsule,
)

# Canned provider-failure replies live in canned_replies.py (extracted for
# the line budget); the Kurisu constants are re-exported here because
# tool_loop/sessions/turn/tests import them from engine.
from .canned_replies import (  # noqa: F401  (re-exports)
    INVALID_KEY_REPLY,
    LOST_TOOL_REPLY,
    RATE_LIMITED_REPLY,
    canned_failure_reply,
)
from .next_time import note_next_time_thread
from .co_activity import _ReadAlongState, _WatchTogetherState
from .providers.llm.base import ChatMessage, DummyProvider, LLMProvider
from .providers.llm.gemini_provider import GeminiProvider
from .providers.llm.local_provider import LocalTransformersProvider
from .providers.llm.openai_provider import OpenAIProvider
from .safety import guardrail_response

LOGGER = get_logger(__name__)


def mood_from_tone(tone: float, current: str) -> str:
    """Map the conversation's emotional tone onto Kurisu's mood.

    Positive tone earns engagement and, when strong, visible warmth. A
    strongly negative tone also maps to "affectionate" — supportive warmth
    with teasing dropped is the right register when the user is upset.
    Neutral tone settles back to "focused" rather than sticking.
    """
    if tone >= 0.5:
        return "affectionate"
    if tone >= 0.15:
        return "curious"
    if tone <= -0.5:
        return "affectionate"
    if -0.5 < tone <= -0.15:
        return "focused"
    return "focused" if current in {"curious", "affectionate"} else current


# How a strong fresh vocal-tone label biases a still-"focused" mood (see
# prepare_turn). Only the labels that map onto an existing KurisuState mood
# are here — "neutral"/"flat" deliberately have no entry, so a shrug of a
# voice clip never overrides the default. The second block is the model
# (speech-emotion) label set from amadeus/vocal_emotion.py: an angry/tense
# user puts an edge in the room, a clearly happy one invites energy, a sad
# or anxious one pulls warmth forward; "surprised"/"disgusted" stay
# unmapped — neither implies a mood SHE should shift into.
_MOOD_FROM_VOCAL_TONE: Dict[str, str] = {
    "tense": "annoyed",
    "low": "tired",
    "bright": "curious",
    "angry": "annoyed",
    "happy": "curious",
    "sad": "affectionate",
    "fearful": "affectionate",
}


def build_summary_provider(
    config: Config, base_llm: LLMProvider
) -> Optional[LLMProvider]:
    """Build the cheap summarization LLM, reusing the chat provider when apt.

    ``memory.summary_model: null`` means "reuse the chat model" — the memory
    tiers should just work with whatever provider is configured, without the
    old gpt-3.5-turbo default 404ing on Gemini. With a dummy chat provider
    this still returns None, so every tier no-ops harmlessly offline.
    """
    if isinstance(base_llm, DummyProvider):
        return None
    base_model = getattr(base_llm, "model", None)
    summary_model = getattr(config.memory, "summary_model", None) or base_model
    if not summary_model:
        return base_llm
    if base_model == summary_model:
        return base_llm
    try:
        if isinstance(base_llm, OpenAIProvider) and (
            config.openai_api_key or config.providers.openai_base_url
        ):
            return OpenAIProvider(
                model=summary_model,
                api_key=config.openai_api_key or "not-needed",
                base_url=config.providers.openai_base_url,
            )
        if isinstance(base_llm, GeminiProvider) and config.gemini_api_key:
            return GeminiProvider(model=summary_model, api_key=config.gemini_api_key)
        if isinstance(base_llm, LocalTransformersProvider):
            return LocalTransformersProvider(model_name=summary_model)
    except Exception as exc:  # pragma: no cover - provider init errors
        LOGGER.warning(
            "Failed to initialize dedicated summary provider (%s); using chat provider.",
            exc,
        )
        return base_llm
    return base_llm


@dataclass
class TurnSetup:
    """Shared per-turn state built once and consumed by the caller."""

    persona: PersonaSettings
    system_prompt: str
    context: List[ChatMessage]
    recall_items: List[str]
    refusal: Optional[str]
    use_tagged_output: bool


class ChatEngine:
    """Core assistant runtime: persona + memory + guardrails + tools."""

    def __init__(
        self,
        config: Config,
        llm: LLMProvider,
        *,
        embedder: Optional[EmbeddingProvider] = None,
        persist_path: Optional[Union[str, Path]] = None,
        embeddings_path: Optional[Union[str, Path]] = None,
        profile_path: Optional[Union[str, Path]] = None,
    ) -> None:
        self.config = config
        self.llm = llm
        # A dummy LLM means no real provider is configured/loaded at all —
        # every "call" is canned text, so rate_limited/invalid_key/error can
        # never apply. Stamped at construction time (both the CLI's
        # ChatEngine and WebChatSession share this __init__) so /api/status
        # reports "dummy" immediately, not only after the first failed
        # generation. A real provider's first successful call overwrites
        # this with "ok" (see _generate_with_health and the streaming/POST
        # call sites); a real provider's first failure overwrites it with
        # the appropriate failure state.
        if isinstance(llm, DummyProvider):
            provider_health.set_llm_health("dummy")
        self.embedder = embedder
        self.summary_provider = build_summary_provider(config, llm)
        memory_kwargs: Dict[str, Any] = {}
        if persist_path is not None:
            memory_kwargs["persist_path"] = persist_path
        if embeddings_path is not None:
            memory_kwargs["embeddings_path"] = embeddings_path
        if profile_path is not None:
            memory_kwargs["profile_path"] = profile_path
        self.memory = MemoryManager(
            config,
            embedder=embedder,
            summary_provider=self.summary_provider,
            **memory_kwargs,
        )
        self.kurisu_state = KurisuState()
        self.kurisu_arc: Optional[KurisuArc] = None
        # Character identity used by the presence prompts (greeting, farewell,
        # poke, idle chatter). Defaults reproduce the built-in Kurisu; a
        # custom character pack (config.character) swaps the name.
        char = getattr(config, "character", None)
        self.character_full_name = (
            getattr(char, "full_name", None) or "Makise Kurisu"
        ).strip() or "Makise Kurisu"
        self._character_is_custom = bool(char is not None and char.is_custom)
        # Character birthday as a (month, day) for the greeting milestone.
        # Built-in Kurisu gets July 25 (Steins;Gate canon) without any
        # config; a custom pack only gets a birthday its manifest set —
        # never Kurisu's by inheritance (CharacterSettings.birthday defaults
        # to None for exactly this reason).
        raw_birthday = getattr(char, "birthday", None)
        if raw_birthday:
            self._character_birthday = parse_birthday(raw_birthday)
        elif not self._character_is_custom:
            self._character_birthday = (7, 25)
        else:
            self._character_birthday = None
        self._hydrate_state_from_profile()
        # Frozen per session for prompt-prefix caching; prepare_turn refreshes
        # it only when a session gap semantically invalidated the prefix.
        self.profile_block = self.memory.profile_block()
        self.tool_registry = create_tool_registry(config, memory=self.memory)
        # Presence cooldowns: enforced here (not just client-side debouncing)
        # so a poke/idle endpoint can't be spammed into runaway LLM calls.
        self._last_poke_at: Optional[dt.datetime] = None
        self._last_idle_chatter_at: Optional[dt.datetime] = None
        self._last_ambient_at: Optional[dt.datetime] = None
        self._last_farewell_at: Optional[dt.datetime] = None
        self._last_screen_peek_at: Optional[dt.datetime] = None
        # Focus / co-working mode (amadeus/focus_mode.py): the deadline of
        # the currently running session (None = not focusing), when it
        # started (for the check-in's "roughly N minutes" note), and
        # whether a check-in line is owed the next status/poll call.
        # Per-session state, like the cooldown timestamps above.
        self._focus_until: Optional[dt.datetime] = None
        self._focus_started_at: Optional[dt.datetime] = None
        self._focus_checkin_pending: bool = False
        # Her previous spoken glance line (in-RAM only, like every vision
        # artifact): fed back into the next screen_comment prompt so an
        # unchanged screen gets a SKIP instead of a near-verbatim repeat.
        self._last_screen_comment: Optional[str] = None
        # "Read together" mode: lazily created on the first page posted,
        # cleared by read_along_end(). Never persisted — see
        # _ReadAlongState's docstring and docs/companion-ethics.md.
        self._read_along: Optional[_ReadAlongState] = None
        # Logged once (not per page) when a session first hits
        # read_along.max_pages_per_session, so a long chapter doesn't spam
        # the log with an INFO line on every subsequent page post.
        self._read_along_cap_logged = False
        # "Watch together" mode: read_along's screen-shaped sibling, same
        # lazily-created never-persisted in-RAM state and same once-only
        # cap logging. See _WatchTogetherState and docs/companion-ethics.md.
        self._watch_together: Optional[_WatchTogetherState] = None
        self._watch_cap_logged = False
        # Lazy ReminderStore for due-reminder announcements (web presence
        # path; the CLI runs its own copy of this logic in-loop).
        self._reminder_store: Optional[Any] = None
        # Lazy PersonaVersionStore for the "your settings changed since last
        # time" greeting announcement (identity continuity).
        self._persona_versions: Optional[Any] = None
        # Daily-brief store, injected by the web SessionStore (the CLI leaves
        # it None). The greeting folds in an unconsumed brief when present.
        self.brief_store: Optional[Any] = None
        # Diary store, injected by the web SessionStore (the CLI leaves it
        # None). The greeting folds in today's unconsumed entry; idle chatter
        # peeks it without consuming.
        self.diary_store: Optional[Any] = None
        # Follow-up store (overnight research on the next-time thread),
        # injected by the web SessionStore (the CLI leaves it None). The
        # greeting delivers an unconsumed note exactly once, replacing the
        # plain next-time fact for that greeting.
        self.followup_store: Optional[Any] = None
        # Life-thread store (her own ongoing pursuits), injected by the web
        # SessionStore (the CLI leaves it None). Idle chatter and the diary
        # peek it; nothing here ever consumes — threads progress only via
        # the scheduler.
        self.life_store: Optional[Any] = None
        # Insight store (nightly consolidation, amadeus/insights.py),
        # injected by the web SessionStore (the CLI leaves it None). Read at
        # prepare_turn into a cached per-day prompt block — see
        # _refresh_insights_block.
        self.insight_store: Optional[Any] = None
        # Rhythm store (opt-in coarse tally of the user's week,
        # amadeus/rhythm.py), injected by the web SessionStore (the CLI
        # leaves it None). Read ONLY by the greeting, at most once a week —
        # never per-turn, never mid-session.
        self.rhythm_store: Optional[Any] = None
        # Topic of a life-thread opinion-ask awaiting the user's reply, set
        # by presence.idle_chatter and consumed by the very NEXT user turn
        # in prepare_turn. One slot, cleared on read: an unanswered ask
        # simply expires, and nothing anywhere records that it went
        # unanswered (see life_threads.LifeThread.pending_input).
        self._pending_life_ask: Optional[str] = None
        # Cached block + last read time: the set changes at most once a day,
        # so a short TTL keeps the hot path off disk while a freshly
        # consolidated set still lands within minutes. Re-reads of an
        # unchanged set produce the identical string, so the provider's
        # cached prompt prefix is unaffected.
        self._insights_block: Optional[str] = None
        self._insights_block_read_at: float = 0.0
        # Structured milestone events computed by the most recent
        # open_session_greeting() call (empty on first-meeting/no-profile
        # greetings, or before any greeting has run). The web greeting
        # endpoint reads this instead of recomputing milestone_events() a
        # second time from the (possibly since-changed) profile, so the
        # endpoint's response always matches what actually informed the
        # greeting text.
        self.last_greeting_milestones: List[Dict[str, Any]] = []
        # Stashed vocal-emotion analysis of the user's most recent voice
        # clip (amadeus/vocal_tone_stash.py), written by the STT endpoints
        # (plain /api/stt and the duplex loop) and consumed by exactly the
        # next prepare_turn call within vocal_emotion.ttl_seconds. Always
        # None on the CLI, which has no audio input path.
        self._vocal_tone: Optional[Any] = None

    def _persona_version_store(self) -> Optional[Any]:
        if self._persona_versions is None:
            path = getattr(self.config.memory, "persona_versions_path", None)
            if not path:
                return None
            try:
                from .persona_versions import PersonaVersionStore

                self._persona_versions = PersonaVersionStore(Path(path))
            except Exception as exc:  # pragma: no cover - store init failure
                LOGGER.warning("Persona version store unavailable: %s", exc)
                return None
        return self._persona_versions

    def _hydrate_state_from_profile(self) -> None:
        """Carry mood/trust across sessions via the persisted UserProfile.

        Without a profile (memory tiers off, or full_forget) the state stays
        at KurisuState() defaults — exactly the old fresh-every-session
        behavior. Trust is quantized at this boundary so the prompt tail is
        stable within a conversation.
        """
        profile = getattr(self.memory, "profile", None)
        if profile is None:
            self.kurisu_state = KurisuState()
            self.kurisu_arc = None
            return
        if profile.mood:
            self.kurisu_state.mood = profile.mood
        self.kurisu_state.trust_level = quantize_trust(profile.trust_level)
        # Arc dials move only at session boundaries (here), same rationale as
        # trust: the dynamic prompt tail stays cache-stable within a
        # conversation. Without a profile, kurisu_arc stays None so
        # persona._arc_line renders its default "hidden arc" fallback.
        self.kurisu_arc = KurisuArc(
            curiosity_vs_annoyance=profile.arc_curiosity_vs_annoyance,
            logic_vs_emotion=profile.arc_logic_vs_emotion,
            pride_vs_vulnerability=profile.arc_pride_vs_vulnerability,
        ).normalized()

    # ------------------------------------------------------------------
    # Turn lifecycle
    # ------------------------------------------------------------------

    def _recent_reply_openers(self, count: int = 2, words: int = 4) -> List[str]:
        """First few words of her recent replies, to steer opener variety.

        The persona prompt uses these to avoid reopening the same way turn
        after turn ("Oh, really…"). Cheap: reads the short-term window.
        """
        openers: List[str] = []
        for message in reversed(self.memory.short_term):
            if message.role != "assistant":
                continue
            head = " ".join((message.content or "").split()[:words]).strip()
            if head:
                openers.append(head)
            if len(openers) >= count:
                break
        return openers

    def _early_acquaintance_active(self) -> bool:
        """Whether the getting-to-know-you prompt nudge applies right now.

        Re-read per turn: extraction merges facts mid-conversation, and the
        nudge should stop the moment she knows enough — one static-prefix
        invalidation when it flips, same cadence class as profile_block.
        Profile-less setups (long-term memory off) are always inactive: with
        nowhere for answers to land the curiosity phase could never end.
        """
        cfg = self.config.memory
        if not getattr(cfg, "enable_early_acquaintance", True):
            return False
        profile = getattr(self.memory, "profile", None)
        if profile is None:
            return False
        return early_acquaintance_active(
            profile.session_count,
            len(profile.facts),
            max_sessions=getattr(cfg, "early_acquaintance_max_sessions", 8),
            max_facts=getattr(cfg, "early_acquaintance_max_facts", 5),
        )

    def _consume_life_ask_reply(self, user_text: str) -> None:
        """Read the next user turn as the answer to a life-thread ask.

        Fires at most once per ask and disarms unconditionally — including
        when the reply is empty or the store write fails — so an unanswered
        ask expires silently and is never retried. Best-effort: her pursuit
        advancing is never worth failing a turn over.
        """
        topic = self._pending_life_ask
        if topic is None:
            return
        self._pending_life_ask = None
        store = getattr(self, "life_store", None)
        if store is None or not (user_text or "").strip():
            return
        try:
            if store.record_user_input(topic, user_text):
                telemetry.incr("life.ask_answered")
        except Exception as exc:  # pragma: no cover - store write failure
            LOGGER.warning("Life-thread ask reply capture failed: %s", exc)

    def _refresh_insights_block(self) -> Optional[str]:
        """Standing-insights prompt block (amadeus/insights.py), re-read
        from the store at most every few minutes. Exception-wrapped — a
        corrupt insight store must never break a turn (and counts, per the
        degrade-loudly convention)."""
        store = self.insight_store
        if store is None:
            return None
        now = time.monotonic()
        if (
            now - self._insights_block_read_at < 300
            and self._insights_block_read_at > 0
        ):
            return self._insights_block
        block: Optional[str] = None
        try:
            from .insights import format_insights_block

            block = format_insights_block(store.get(getattr(self, "user_id", None)))
        except Exception as exc:  # pragma: no cover - store read failure
            LOGGER.warning("Insight block read failed: %s", exc)
            telemetry.incr("insights.block_read_failed")
        self._insights_block = block
        self._insights_block_read_at = now
        return block

    def prepare_turn(
        self,
        user_text: str,
        persona: PersonaSettings,
        *,
        tagged_output: bool = False,
        tagged_instructions: Optional[str] = None,
    ) -> TurnSetup:
        """Build the request context and commit the user message to memory."""
        self._consume_life_ask_reply(user_text)
        if self.memory.profile_block_dirty:
            self.profile_block = self.memory.profile_block()
        # Cross-turn emotional state: let the extraction job's tone nudge
        # Kurisu's mood so the register persists between turns.
        tone = self.memory.last_tone
        if tone is not None and getattr(self.config.memory, "mood_from_tone", True):
            self.kurisu_state.mood = mood_from_tone(
                tone, self.kurisu_state.normalized_mood()
            )
        # Clock-driven energy: past midnight an otherwise-neutral mood reads
        # as tired. Only the "focused" default is overridden, so tone-driven
        # moods (affectionate/curious/annoyed) always win over the clock.
        if self.kurisu_state.normalized_mood() == "focused" and is_late_night():
            self.kurisu_state.mood = "tired"
        # Vocal-emotion bias: a strong fresh signal from the user's voice
        # (not their words) can nudge a still-neutral "focused" mood the
        # same way the clock does above — tone-driven moods (set from what
        # they SAID) always win over what their voice merely sounded like.
        # Consumed here so it never survives into a second turn even when
        # the prompt line itself is dropped (e.g. use_tagged_output branches
        # that skip build_system_prompt entirely never happen — this call
        # always runs — so exactly-once consumption is safe here).
        vocal_tone_result = vocal_tone_stash.consume_vocal_tone(self)
        vocal_tone_phrase = (
            vocal_tone_result.phrase() if vocal_tone_result is not None else None
        )
        if (
            vocal_tone_result is not None
            and self.kurisu_state.normalized_mood() == "focused"
        ):
            biased = _MOOD_FROM_VOCAL_TONE.get(vocal_tone_result.label)
            if biased:
                self.kurisu_state.mood = biased
        # Trust only moves at session boundaries (MemoryManager's gap
        # handling), and it's quantized — so this re-read is a no-op within a
        # conversation and the prompt prefix stays cache-friendly.
        profile = getattr(self.memory, "profile", None)
        if profile is not None:
            self.kurisu_state.trust_level = quantize_trust(profile.trust_level)
            # Same contract for the arc dials: they only move at session
            # boundaries, so re-reading per turn is a no-op within a
            # conversation but keeps a long-lived engine from serving a
            # construction-time snapshot forever (an in-process gap boundary
            # used to drift the profile without the prompt ever seeing it).
            self.kurisu_arc = KurisuArc(
                curiosity_vs_annoyance=profile.arc_curiosity_vs_annoyance,
                logic_vs_emotion=profile.arc_logic_vs_emotion,
                pride_vs_vulnerability=profile.arc_pride_vs_vulnerability,
            ).normalized()
        system_prompt = build_system_prompt(
            persona,
            state=self.kurisu_state,
            arc=self.kurisu_arc,
            user_text=user_text,
            profile_block=self.profile_block,
            insights_block=self._refresh_insights_block(),
            early_acquaintance=self._early_acquaintance_active(),
            recent_openers=self._recent_reply_openers(),
            character=getattr(self.config, "character", None),
            vocal_tone=vocal_tone_phrase,
        )
        if tagged_output and tagged_instructions:
            system_prompt = f"{system_prompt}\n\n{tagged_instructions}"
        # Recall + context assembly wall time (includes the recall query
        # embed): recall has silently re-entered the critical path before
        # (the 0.35s embed-deadline bug), so it's timed like the rest of the
        # voice chain (stt.request_ms, llm.first_token_ms, ...).
        _recall_started = time.monotonic()
        context, recall_items = assemble_turn_context(
            self.memory, system_prompt, user_text
        )
        telemetry.record_duration(
            "memory.recall_ms", 1000 * (time.monotonic() - _recall_started)
        )
        refusal = guardrail_response(user_text)
        return TurnSetup(
            persona=persona,
            system_prompt=system_prompt,
            context=context,
            recall_items=recall_items,
            refusal=refusal,
            use_tagged_output=bool(tagged_output),
        )

    def open_session_greeting(self) -> Optional[str]:
        """Generate and commit one short in-character opening line.

        Fires only for fresh sessions (see ``MemoryManager.should_greet``);
        costs one cheap-model call. Returns None when disabled, when the
        session is mid-conversation, or when no usable provider exists.
        Body lives in ``amadeus/greeting.py`` (line-budget extraction, same
        free-function-over-owner pattern as the presence family).
        """
        return greeting.open_session_greeting(self)

    def close_session_farewell(
        self, goodbye_text: Optional[str] = None
    ) -> Optional[str]:
        """One short warm send-off line when the user says goodbye.

        The inverse of :meth:`open_session_greeting`: fires only when there
        IS a conversation to close. Deliberately anti-dark-pattern — the
        prompt forbids guilt, begging, or FOMO; she lets the user go gladly.

        Unlike the greeting, this is commit-free: callers commit the goodbye
        turn themselves only after a line was actually produced (web via
        ``complete_turn`` so the transcript gets it too) — so a failed
        generation can fall back to a normal chat turn without the goodbye
        being double-committed to memory. ``goodbye_text`` rides in via the
        prompt for the same reason.
        """
        if not self.memory.should_bid_farewell():
            return None
        cooldown = float(
            getattr(self.config.memory, "farewell_cooldown_seconds", 120.0) or 120.0
        )
        if not self._presence_cooldown_ready(self._last_farewell_at, cooldown):
            return None
        provider = self.summary_provider
        if provider is None:
            return None
        # The last exchange shapes the send-off ("good night" vs "gotta run
        # to work" deserve different lines).
        recent = [
            f"{m.role}: {m.content}"
            for m in list(self.memory.short_term)[-2:]
            if (m.content or "").strip()
        ]
        if goodbye_text and goodbye_text.strip():
            recent.append(f"user: {goodbye_text.strip()}")
        prompt = [
            ChatMessage(
                role="system",
                content=(
                    f"You are {self.character_full_name}. The user is saying goodbye and "
                    "ending the conversation. Write exactly ONE short warm "
                    "send-off line (under 20 words), in character — a "
                    "little fond is fine. "
                    + persona_capsule(getattr(self.config, "character", None))
                    + " Let them go gladly. NEVER guilt them for "
                    "leaving, never ask them to stay or come back, never say "
                    "you'll be lonely or miss them in a needy way — their "
                    "life outside this app matters more than this chat. "
                    "No stage directions, no quotation marks."
                ),
            ),
            ChatMessage(role="user", content="\n".join(recent) or "(saying goodbye)"),
        ]
        try:
            text = (
                (
                    provider.generate(
                        prompt,
                        max_tokens=int(
                            getattr(self.config.memory, "farewell_max_tokens", 40) or 40
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
            LOGGER.warning("Farewell generation failed: %s", exc)
            telemetry.incr("farewell.generation_failed")
            return None
        if not text:
            return None
        self._last_farewell_at = dt.datetime.utcnow()
        # Only a REAL goodbye reaches this point (window_close never calls
        # close_session_farewell) — stash one line about the user's open
        # topic for the next greeting. Best-effort; see amadeus/next_time.py.
        note_next_time_thread(self, provider, recent)
        return text

    def _presence_cooldown_ready(
        self, last_at: Optional[dt.datetime], cooldown_seconds: float
    ) -> bool:
        if last_at is None:
            return True
        elapsed = (dt.datetime.utcnow() - last_at).total_seconds()
        return elapsed >= cooldown_seconds

    def due_reminder_lines(self, catch_up: bool = False) -> List[str]:
        """Spoken lines for reminders that just came due, one per reminder.

        Mirrors the CLI's `_announce_due_reminders` phrasing so both
        front-ends sound the same; the web presence endpoint speaks these
        through the same TTS path as greeting/poke/idle. Empty list when the
        feature is off or nothing is due — safe to poll frequently
        (`pop_due` marks announced, so each reminder fires exactly once).
        """
        reminders_cfg = getattr(self.config, "reminders", None)
        if not reminders_cfg or not getattr(reminders_cfg, "enabled", False):
            return []
        if self._reminder_store is None:
            try:
                from .reminders import ReminderStore

                self._reminder_store = ReminderStore(reminders_cfg.store_path)
            except Exception as exc:  # pragma: no cover - store init failure
                LOGGER.error("Reminder store unavailable: %s", exc)
                return []
        try:
            due = self._reminder_store.pop_due()
        except Exception as exc:  # pragma: no cover - disk failure
            LOGGER.error("Reminder check failed: %s", exc)
            return []
        if not due:
            return []
        now = dt.datetime.now()
        lines: List[str] = []
        for reminder in due:
            overdue = (now - reminder.due_at).total_seconds() > 120
            if catch_up or overdue:
                lines.append(
                    f"While you were away: at {reminder.due_at:%I:%M %p}, "
                    f"{reminder.text}."
                )
            else:
                lines.append(f"Reminder: {reminder.text}.")
        return lines

    def poke_reaction(self, region: Optional[str] = None) -> Optional[str]:
        """One short in-character line reacting to being "poked". See
        presence.poke_reaction."""
        return presence.poke_reaction(self, region)

    def screen_comment(self, image_mime: str, image_data: str) -> Optional[str]:
        """One short unprompted line about the user's screen. See
        presence.screen_comment."""
        return presence.screen_comment(self, image_mime, image_data)

    def read_along_page(self, image_mime: str, image_data: str) -> Optional[str]:
        """Process one page of a "read together" session. See
        co_activity.read_along_page."""
        return co_activity.read_along_page(self, image_mime, image_data)

    def read_along_end(self) -> Optional[str]:
        """Close out a "read together" session. See co_activity.read_along_end."""
        return co_activity.read_along_end(self)

    def watch_frame(self, image_mime: str, image_data: str) -> Optional[str]:
        """Process one frame of a "watch together" session. See
        co_activity.watch_frame."""
        return co_activity.watch_frame(self, image_mime, image_data)

    def watch_end(self) -> Optional[str]:
        """Close out a "watch together" session. See co_activity.watch_end."""
        return co_activity.watch_end(self)

    def idle_chatter(self) -> Optional[str]:
        """One short unprompted line after a quiet stretch. See
        presence.idle_chatter."""
        return presence.idle_chatter(self)

    def ambient_comment(self, kind: str, detail: str) -> Optional[str]:
        """One short line about an ambient desktop signal (focused app,
        battery, now-playing). See presence.ambient_comment."""
        return presence.ambient_comment(self, kind, detail)

    def start_focus(self, minutes: Any) -> Dict[str, Any]:
        """Begin a focus/co-working session. See focus_mode.start_focus."""
        from . import focus_mode

        return focus_mode.start_focus(self, minutes)

    def end_focus(self) -> Dict[str, Any]:
        """End the focus session early. See focus_mode.end_focus."""
        from . import focus_mode

        return focus_mode.end_focus(self)

    def focus_status(self) -> Dict[str, Any]:
        """Current focus session state (also delivers a due check-in
        exactly once). See focus_mode.focus_status."""
        from . import focus_mode

        return focus_mode.focus_status(self)

    # --- Text ReAct tool-calling (shared by CLI and web frontends) -----------

    # Keyword intent gate (tools.gate: "keyword", and the text-ReAct half of
    # "auto"): only pay the extra LLM call(s) + latency of the tool loop when
    # the message plausibly needs a lookup. Casual chat ("hi", "how are you")
    # stays a single generate() call — important for a voice daily driver on
    # free-tier quotas. Native function-calling providers under "auto" skip
    # this entirely: they get tools every turn and the model decides (a
    # no-tool turn is still exactly one LLM call).
    _TOOL_HINT_WORDS = (
        "file",
        "files",
        "document",
        "manuscript",
        "pdf",
        "docx",
        "folder",
        "desktop",
        "find",
        "search",
        "look up",
        "lookup",
        "read",
        "open",
        "email",
        "mail",
        "calendar",
        "schedule",
        "what's on",
        "whats on",
        "latest",
        "news",
        "who is",
        "what is",
        "when is",
        "how many",
        "cover letter",
        "abstract",
        "paper",
        "poster",
        "pubmed",
        "arxiv",
        "papers",
        "study",
        "studies",
        "publication",
        "journal",
        "remind",
        "reminder",
        "timer",
        "alarm",
        # Explicit memory instructions (the "memory" tool: remember/correct/forget).
        "remember",
        "forget",
        "memorize",
        "memorise",
        # Avatar flourishes (play_animation): physical requests should enter
        # the tool loop even on text-ReAct providers.
        "dance",
        "wave",
        "pose",
        "spin",
        "clap",
        "gesture",
        "animation",
        # Follow-up / drill-in words: after a search, "who wrote it / what are
        # the methods" must ALSO route through tools so the answer is grounded
        # (via fetch) rather than answered from the model's memory.
        "who wrote",
        "author",
        "authors",
        "method",
        "methods",
        "methodology",
        "abstract",
        "explain",
        "detail",
        "details",
        "title",
        "citation",
        "doi",
        "published",
        "which one",
        "that one",
        "first one",
        "tell me more",
    )

    def _tools_active(self) -> bool:
        """True when text ReAct tool-calling is configured and available."""
        return (
            getattr(self.config, "tools", None) is not None
            and self.config.tools.enabled
            and bool(self.tool_registry.tools)
        )

    # Set once a turn actually uses tools, so short follow-ups in the same
    # conversation ("who wrote it?", "explain it") stay tool-eligible and get
    # grounded answers instead of falling back to memory. Reset on session reset.
    _tools_used_this_session: bool = False

    def _llm_supports_native_tools(self) -> bool:
        return bool(getattr(self.llm, "supports_tools", lambda: False)())

    def _should_use_tools(self, user_input: str) -> bool:
        gate = (getattr(self.config.tools, "gate", "auto") or "auto").lower()
        if gate == "always":
            return True
        if gate == "auto" and self._llm_supports_native_tools():
            # Native function calling: pass tools every turn, let the model
            # decide. A turn where it calls nothing costs exactly 1 LLM call.
            return True
        # "keyword" mode, or "auto" on a text-ReAct provider.
        text = (user_input or "").lower().strip()
        if len(text) < 3:
            return False
        if any(hint in text for hint in self._TOOL_HINT_WORDS):
            return True
        # Sticky: after tools have been used, treat a follow-up question as
        # tool-eligible too (it's usually drilling into the prior result). Only
        # questions — casual acknowledgements ("thanks", "cool") stay cheap.
        if self._tools_used_this_session and "?" in text and len(text.split()) >= 3:
            return True
        return False

    # Animation requests recorded by the play_animation tool during the last
    # tool-loop turn; drained into ChatResponse.animations by the web session.
    _pending_animations: List[str] = []

    def pop_pending_animations(self) -> List[str]:
        drained, self._pending_animations = self._pending_animations, []
        return drained

    def _drain_animation_requests(self) -> None:
        tool = self.tool_registry.tools.get("play_animation")
        pop = getattr(tool, "pop_requested", None)
        if callable(pop):
            self._pending_animations = pop()

    def _run_tool_loop(self, context: List[ChatMessage], progress_callback=None) -> str:
        """Run the non-council ReAct tool loop, augmenting the system prompt
        (context[0]) with the available-tools instructions. Blocking (may make
        several sequential model calls); callers off the event loop should wrap
        it appropriately."""
        from .agents.tool_loop import (
            ToolLoopAuthError,
            ToolLoopRateLimitError,
            build_tool_system_prompt,
            run_tool_loop,
            run_native_tool_loop,
        )

        # Mark the conversation tool-oriented so short follow-ups stay grounded.
        self._tools_used_this_session = True
        self._pending_animations = []
        # Prefer native function calling when the provider supports it (no
        # JSON-in-text to malform). Falls back to text ReAct otherwise.
        native = getattr(self.llm, "supports_tools", lambda: False)()
        tool_prompt = build_tool_system_prompt(
            self.tool_registry, allowed=self.config.tools.allowed, native=native
        )
        augmented = list(context)
        if augmented and augmented[0].role == "system":
            augmented[0] = ChatMessage(
                role="system",
                content=augmented[0].content + tool_prompt,
            )
        else:
            augmented.insert(0, ChatMessage(role="system", content=tool_prompt))
        try:
            if native:
                reply = run_native_tool_loop(
                    self.llm,
                    self.tool_registry,
                    augmented,
                    max_iters=self.config.tools.max_iterations,
                    allowed=self.config.tools.allowed,
                    block_network_after_file_read=self.config.tools.block_network_after_file_read,
                    network_arg_guard=getattr(
                        self.config.tools, "network_arg_guard", True
                    ),
                    progress_callback=progress_callback,
                )
            else:
                reply = run_tool_loop(
                    self.llm.generate,
                    self.tool_registry,
                    augmented,
                    max_iters=self.config.tools.max_iterations,
                    allowed=self.config.tools.allowed,
                    block_network_after_file_read=self.config.tools.block_network_after_file_read,
                    network_arg_guard=getattr(
                        self.config.tools, "network_arg_guard", True
                    ),
                    progress_callback=progress_callback,
                )
        except ToolLoopAuthError as exc:
            # Retrying plain generate() would 401/403 identically; the generic
            # "ran out of steps" recovery line actively misleads here (this is
            # exactly how an invalid saved key manifested in the wild). Answer
            # with a pointed diagnosis instead, same pattern as the dummy
            # provider's NO_KEY_REPLY.
            LOGGER.error("Provider rejected the API key during a tool turn: %s", exc)
            provider_health.set_llm_health("invalid_key", str(exc))
            self._drain_animation_requests()
            return self._canned_failure_reply("invalid_key")
        except ToolLoopRateLimitError as exc:
            # Mirrors the auth-error branch exactly, but for quota/429s:
            # retrying (plain generate() or a fallback provider) would just
            # 429 again on the same exhausted window, so answer with the
            # honest quota line immediately. Caught before the generic
            # handler below because the original exception's structured
            # shape (status_code, SDK class name) doesn't survive being
            # re-wrapped into ToolLoopRateLimitError(str(exc)) — re-running
            # classify_and_record on the wrapper could miss it.
            LOGGER.warning("Tool loop rate-limited: %s", exc)
            provider_health.set_llm_health("rate_limited", str(exc))
            self._drain_animation_requests()
            return self._canned_failure_reply("rate_limited")
        except Exception as exc:  # pragma: no cover - provider/tool errors
            # Classify before falling back: a rate-limited provider will
            # 429 identically on the plain generate() below, so answer with
            # the honest quota line up front instead of masking it behind a
            # second failure (or, worse, a second success on a fallback
            # provider that then silently clears the health flag).
            kind = provider_health.classify_and_record(exc)
            if kind == "rate_limited":
                LOGGER.warning("Tool loop rate-limited: %s", exc)
                self._drain_animation_requests()
                return self._canned_failure_reply("rate_limited")
            LOGGER.error("Tool loop failed, falling back to plain generate(): %s", exc)
            self._drain_animation_requests()
            reply = self._generate_with_health(context)
            return reply
        self._drain_animation_requests()
        if not (reply or "").strip():
            # An empty reply usually isn't a confused loop — the native loop
            # swallows provider errors (free-tier 429s land here) and returns
            # nothing. Retry once WITHOUT tools before admitting defeat: a
            # rate-limited burst often clears within the retry, and a plain
            # answer beats the canned "ran out of steps" line. If the retry
            # itself hits the rate limit, answer with the same honest quota
            # line rather than letting the exception fall through uncaught.
            LOGGER.warning(
                "Tool loop returned an empty reply; retrying once without tools"
            )
            try:
                reply = self._generate_with_health(context)
            except Exception as exc:  # pragma: no cover - provider errors
                if provider_health.classify_and_record(exc) == "rate_limited":
                    return self._canned_failure_reply("rate_limited")
                raise
            if (reply or "").strip():
                return reply
            # Genuinely nothing to say: a confused loop (cap, refusals) gets
            # an in-character recovery line instead of an empty bubble.
            return self._canned_failure_reply("lost")
        provider_health.record_llm_success(self.llm)
        return reply

    def _canned_failure_reply(self, kind: str) -> str:
        return canned_failure_reply(kind, is_custom_character=self._character_is_custom)

    def _generate_with_health(self, context: List[ChatMessage]) -> str:
        """Plain ``llm.generate`` + usage telemetry + health self-heal, the
        shared tail every non-streaming fallback/retry path in this class
        wants. A successful call clears any prior rate_limited/error/
        invalid_key state — the self-healing half of provider health."""
        reply = self.llm.generate(context)
        telemetry.record_llm_usage(self.llm, "chat")
        provider_health.record_llm_success(self.llm)
        return reply

    def rollback_user_message(self, user_text: str) -> None:
        """Undo the user-message commit after a failed provider call.

        prepare_turn records the user's message before the LLM runs; when the
        provider dies mid-turn, popping it leaves memory as if the turn never
        happened, so a retry doesn't double-record the message. Delegates to
        :meth:`MemoryManager.rollback_user_message`, which also rewinds the
        ``_unsummarized`` delta buffer — popping short_term alone let a
        cancelled turn's retracted text still reach summarization/extraction.
        """
        self.memory.rollback_user_message(user_text)

    def complete_turn(self, user_text: str, reply: str) -> None:
        """Commit the assistant reply and update conversational state."""
        if reply:
            self.memory.add_message("assistant", reply)
        self.kurisu_state.last_topic = derive_topic(
            user_text, self.kurisu_state.last_topic
        )
        # Mood carries across sessions: stash it on the profile (saved on the
        # next existing profile save trigger, not per turn).
        self.memory.record_mood(self.kurisu_state.normalized_mood())
