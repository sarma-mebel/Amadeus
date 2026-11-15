"""Background LLM job bodies for the memory manager.

Free functions taking the manager (the sibling-module convention here —
see context.py/backup.py/introspection.py): the summary/extraction job,
the session-end episode and reflection jobs, and the episode supersede/
prune helpers. They mutate state only via manager attributes under
``manager._state_lock``. The submission/callback plumbing (``summarize``,
``_on_summary_ready``, ``_submit_job``, the single-worker executor
lifecycle) stays on MemoryManager, which keeps thin private methods
delegating here — executor submits reference those methods.
"""

from __future__ import annotations

import datetime as dt
import json
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple

from .. import telemetry
from ..logging_utils import get_logger
from ..providers.llm.base import ChatMessage
from .text_utils import (
    _as_float,
    _clamp,
    _normalize_event_time,
    _parse_extraction_json,
    _vader_tone,
)
from .vector_store import VectorRecord

if TYPE_CHECKING:  # pragma: no cover - import cycle guard (manager imports us)
    from .manager import MemoryManager, Message

LOGGER = get_logger(__name__)


def run_summary_job(
    manager: "MemoryManager",
    messages: Sequence["Message"],
    revision: int,
    previous_summary: Optional[str] = None,
    profile_snapshot: Optional[str] = None,
) -> Dict[str, Any]:
    transcript = manager._build_transcript(messages)
    extraction = bool(getattr(manager.config.memory, "enable_fact_extraction", False))
    if extraction:
        return run_extraction_job(
            manager,
            transcript,
            revision,
            previous_summary,
            profile_snapshot,
            len(messages),
        )
    if previous_summary:
        system_content = (
            f"You maintain a running summary of a conversation between a user and {manager._char_name}. "
            "Update the previous summary with the new messages, keeping established facts. "
            "Focus on factual details, key questions, and promises. Keep it under 120 words."
        )
        user_content = (
            f"Previous summary:\n{previous_summary}\n\nNew messages:\n{transcript}"
        )
    else:
        system_content = (
            f"Summarize the following conversation between a user and {manager._char_name}. "
            "Focus on factual details, key questions, and promises. "
            "Keep it under 120 words."
        )
        user_content = transcript
    prompt = [
        ChatMessage(role="system", content=system_content),
        ChatMessage(role="user", content=user_content),
    ]
    try:
        summary_text = (
            manager.summary_provider.generate(prompt)
            if manager.summary_provider
            else ""
        )
        telemetry.record_llm_usage(manager.summary_provider, "memory")
    except Exception as exc:  # pragma: no cover - provider runtime errors
        LOGGER.warning("LLM summarization failed: %s", exc)
        return {
            # Keep the existing summary rather than replacing it with a
            # truncated fragment of only the newest messages.
            "summary": previous_summary or transcript[: manager.summary_max_chars],
            "strategy": "retained" if previous_summary else "truncate",
            "error": str(exc),
            "message_count": len(messages),
            "revision": revision,
        }
    summary_text = (summary_text or "").strip()
    if not summary_text:
        LOGGER.debug("LLM summarization returned empty text; using fallback.")
        return {
            "summary": previous_summary or transcript[: manager.summary_max_chars],
            "strategy": "retained" if previous_summary else "truncate",
            "message_count": len(messages),
            "revision": revision,
        }
    return {
        "summary": summary_text[: manager.summary_max_chars],
        "strategy": "llm",
        "message_count": len(messages),
        "revision": revision,
    }


def run_extraction_job(
    manager: "MemoryManager",
    transcript: str,
    revision: int,
    previous_summary: Optional[str],
    profile_snapshot: Optional[str],
    message_count: int,
) -> Dict[str, Any]:
    """One combined call: running summary + durable memory extraction."""
    max_threads = int(getattr(manager.config.memory, "max_open_threads", 3) or 3)
    # Friction rides this existing call rather than adding one of its own
    # (the `felt` field is the precedent). Gated so a disabled install pays
    # zero prompt characters, and so the schema and the parse below can
    # never disagree about whether the key is expected.
    friction_on = bool(getattr(manager.config.memory, "enable_friction_memory", False))
    friction_schema = (
        ',\n "friction": {"topic": "<short phrase for a subject that went BADLY '
        "between the user and you this conversation — a real misunderstanding, "
        "a disagreement left unresolved, a subject that visibly stung. Omit "
        "the whole friction object for a normal conversation, INCLUDING one "
        'where the user was merely upset ABOUT something else>", '
        '"state": "new" | "repaired" (use "repaired" when a subject listed as '
        "sore in the Known profile came up again and went FINE)}"
        if friction_on
        else ""
    )
    system_content = (
        f"You maintain {manager._char_name}'s memory of the conversation with the user. "
        "Update the running summary and extract durable memory. Return ONLY JSON:\n"
        '{"summary": "<updated conversation summary, under 120 words>",\n'
        ' "new_facts": [{"text": "<one-line durable fact about the user>", '
        '"salience": 0.0-1.0, "replaces": "<optional: the earlier fact this '
        "updates or contradicts, e.g. 'User loves Nolan movies'>\", "
        '"when": "<optional ISO date the fact\'s event happened or will '
        "happen, ONLY when the user stated or clearly implied it, e.g. "
        "'2026-08-15'>\"}],\n"
        ' "profile_updates": {"<key>": "<short value>"},\n'
        ' "open_threads": ["<unresolved topic the user may want to revisit>"],\n'
        ' "tone": <emotional tone of the user\'s recent messages, -1.0 to 1.0>,\n'
        ' "felt": {"label": "<one word for how the USER seemed this '
        'conversation, e.g. excited/frustrated/anxious/reflective/content>", '
        '"subject": "<optional short phrase for what it was about, e.g. \'the '
        "paper deadline'>\"}" + friction_schema + "}\n"
        "Only extract facts that will still matter in a week (preferences, biography, "
        f"projects, relationships). No facts about {manager._char_name}. Empty lists are fine. "
        "Never store credentials, passwords, API keys, one-time codes, or financial "
        "account/card numbers, even if the user shares them. Store a third party's "
        "personal details or sensitive health details only when the user made them a "
        "clear, significant topic of conversation, not a passing mention. "
        "Keep established facts in the summary. "
        "Set replaces ONLY when a new fact updates or contradicts something already "
        "in the Known profile or previous summary (e.g. new_facts text 'User is bored "
        "of Nolan movies now' with replaces 'User loves Nolan movies'); omit replaces "
        "otherwise. "
        f"Today is {dt.date.today().isoformat()}. Set when ONLY for facts tied to "
        "a dated event the user actually placed in time ('my defense is on the "
        "15th', 'we moved last March') — resolve relative phrasing against "
        "today's date if you can, and omit when otherwise; never guess. "
        "open_threads is the FULL updated list, not additions: carry over any known "
        "open thread that is still unresolved, DROP threads that got resolved or "
        "answered in the new messages, and add new pending things the user will "
        "plausibly have news about later (an upcoming event, something they were "
        "worried about, work in progress). Phrase each thread so it still makes "
        "sense days later. Return [] when nothing is pending."
    )
    parts: List[str] = []
    if previous_summary:
        parts.append(f"Previous summary:\n{previous_summary}")
    if profile_snapshot:
        parts.append(f"Known profile:\n{profile_snapshot}")
    parts.append(f"New messages:\n{transcript}")
    prompt = [
        ChatMessage(role="system", content=system_content),
        ChatMessage(role="user", content="\n\n".join(parts)),
    ]
    fallback = {
        "summary": previous_summary or transcript[: manager.summary_max_chars],
        "strategy": "retained" if previous_summary else "truncate",
        "message_count": message_count,
        "revision": revision,
    }
    try:
        raw = (
            manager.summary_provider.generate(
                prompt,
                max_tokens=int(
                    getattr(manager.config.memory, "extraction_max_tokens", 400) or 400
                ),
            )
            if manager.summary_provider
            else ""
        )
        telemetry.record_llm_usage(manager.summary_provider, "memory")
    except Exception as exc:  # pragma: no cover - provider runtime errors
        LOGGER.warning("Memory extraction failed: %s", exc)
        fallback["error"] = str(exc)
        return fallback
    raw = (raw or "").strip()
    if not raw:
        return fallback
    data = _parse_extraction_json(raw)
    if data is None:
        if "{" not in raw:
            # Model ignored the format and returned prose; treat it as a
            # plain summary (legacy behavior) rather than losing the call.
            return {
                "summary": raw[: manager.summary_max_chars],
                "strategy": "llm",
                "message_count": message_count,
                "revision": revision,
            }
        LOGGER.warning(
            "Memory extraction returned malformed JSON; retaining previous summary."
        )
        # Loses the whole combined call (facts, profile updates, threads,
        # tone, felt) — count it or a misbehaving model is invisible.
        telemetry.incr("memory.extraction_malformed_json")
        return fallback
    summary_text = str(data.get("summary") or "").strip()
    if not summary_text:
        return fallback
    # 4-tuples since the bitemporal pass: (text, salience, replaces, valid_at).
    # Downstream consumers (manager's store loop, graph.update_graph) unpack
    # tolerantly so older 3-tuple fixtures keep working.
    facts: List[Tuple[str, float, Optional[str], Optional[str]]] = []
    for item in data.get("new_facts") or []:
        if isinstance(item, dict):
            text = str(item.get("text") or "").strip()
            salience = _clamp(_as_float(item.get("salience"), 0.5), 0.0, 1.0)
            replaces = str(item.get("replaces") or "").strip() or None
            valid_at = _normalize_event_time(item.get("when"))
            if item.get("when") and valid_at is None:
                # A malformed model-supplied event time is dropped, never
                # allowed to block the fact itself — but count it so a model
                # that keeps emitting junk "when" values is visible.
                telemetry.incr("memory.valid_at_rejected")
        else:
            text = str(item).strip()
            salience = 0.5
            replaces = None
            valid_at = None
        if text:
            facts.append((text, salience, replaces, valid_at))
    raw_updates = data.get("profile_updates")
    profile_updates = (
        {str(k): str(v) for k, v in raw_updates.items()}
        if isinstance(raw_updates, dict)
        else {}
    )
    raw_threads = data.get("open_threads")
    # None = "model didn't address threads" (leave them alone); an empty
    # list is a real answer ("everything resolved") and must clear them.
    open_threads = (
        [str(t).strip() for t in raw_threads if str(t).strip()][:max_threads]
        if isinstance(raw_threads, list)
        else None
    )
    tone = data.get("tone")
    raw_felt = data.get("felt")
    # Affective tag (v4): a narrowly-scoped label+subject about how the USER
    # seemed, kept SEPARATE from the neutral episode logger on purpose (an
    # opinionated episode log made the persona fixate — see run_episode_job).
    # None when the model omitted it or it carried no label.
    felt: Optional[Dict[str, str]] = None
    if isinstance(raw_felt, dict):
        label = str(raw_felt.get("label") or "").strip()
        if label:
            felt = {"label": label}
            subject = str(raw_felt.get("subject") or "").strip()
            if subject:
                felt["subject"] = subject
    # Friction signal (v7): a TOPIC that went badly between them, deliberately
    # narrower than `felt`/`tone` above (which carry how the USER was doing —
    # a user upset about their own deadline is not friction). Only parsed when
    # the feature asked for the key; None when omitted or unusable.
    friction: Optional[Dict[str, str]] = None
    if friction_on:
        raw_friction = data.get("friction")
        if isinstance(raw_friction, dict):
            topic = str(raw_friction.get("topic") or "").strip()
            if topic:
                state = str(raw_friction.get("state") or "new").strip().lower()
                friction = {
                    "topic": topic,
                    "state": state if state in ("new", "repaired") else "new",
                }
    return {
        "summary": summary_text[: manager.summary_max_chars],
        "strategy": "llm",
        "message_count": message_count,
        "revision": revision,
        "facts": facts,
        "profile_updates": profile_updates,
        "open_threads": open_threads,
        "tone": (_clamp(_as_float(tone, 0.0), -1.0, 1.0) if tone is not None else None),
        "felt": felt,
        "friction": friction,
    }


def apply_relational_signals(manager: "MemoryManager", result: Dict[str, Any]) -> None:
    """Persist the extraction call's two relational side-channels.

    Both are narrow, profile-only signals that ride the same call as the
    summary/facts, and both are consumed exactly once here (from
    ``MemoryManager._on_summary_ready``):

    - ``felt``: how the last conversation FELT to the USER (label + optional
      subject), kept in its own slot rather than folded into the neutral
      episode log, which must never accumulate opinion.
    - ``friction``: a TOPIC that went badly BETWEEN them, or one the model
      reports came up again and went fine. Repair needs no apology — going
      fine IS the repair, and there is deliberately nowhere to record that
      the user made amends.

    Caller holds no lock; this takes ``manager._state_lock`` per write.
    """
    profile = manager.profile
    if profile is None:
        return

    felt = result.get("felt")
    if isinstance(felt, dict) and felt.get("label"):
        with manager._state_lock:
            profile.record_felt(felt.get("label"), felt.get("subject"))
            manager._save_profile()

    friction = result.get("friction")
    if isinstance(friction, dict) and friction.get("topic"):
        with manager._state_lock:
            if friction.get("state") == "repaired":
                if profile.clear_friction(friction["topic"]):
                    telemetry.incr("memory.friction_repaired")
            else:
                profile.record_friction(friction["topic"])
                telemetry.incr("memory.friction_noted")
            manager._save_profile()


def run_episode_job(
    manager: "MemoryManager", messages: Sequence["Message"], summary: Optional[str]
) -> None:
    """Write one dated autobiographical record for a finished session."""
    if not messages:
        return
    date = dt.date.today().isoformat()
    transcript = manager._build_transcript(messages)
    text: Optional[str] = None
    if manager.summary_provider:
        prompt = [
            ChatMessage(
                role="system",
                content=(
                    # NEUTRAL on purpose: these summaries are recalled into "
                    # future prompts, so writing them in Kurisu's voice made her "
                    # persona (and any irritation) compound across sessions until "
                    # she fixated. Record events, not attitude.
                    "You are a neutral session logger. In 2-3 past-tense "
                    "sentences, factually summarize what the user and assistant "
                    "discussed and what the user was working on. Third person, "
                    "no personality, no opinions or judgments about the user, "
                    "no preamble. Refer to the parties as 'the user' and 'the "
                    "assistant'."
                ),
            ),
            ChatMessage(role="user", content=transcript),
        ]
        try:
            text = (
                manager.summary_provider.generate(
                    prompt,
                    max_tokens=int(
                        getattr(manager.config.memory, "episode_max_tokens", 200) or 200
                    ),
                )
                or ""
            ).strip()
            telemetry.record_llm_usage(manager.summary_provider, "memory")
        except Exception as exc:  # pragma: no cover - provider runtime errors
            LOGGER.warning("Episode summarization failed: %s", exc)
            telemetry.incr("memory.episode_failures")
            text = None
    if not text:
        # Free fallback: reuse the running summary instead of a new call.
        text = (summary or transcript[: manager.summary_max_chars]).strip()
    if not text:
        return
    episode_line = text if text.startswith(date) else f"{date}: {text}"
    tone = manager._last_tone
    if tone is None:
        tone = _vader_tone(transcript)
    salience = _clamp(abs(tone), 0.3, 1.0) if tone is not None else 0.5
    with manager._state_lock:
        if manager.embedder and manager.vector_store:
            metadata = {
                "type": "episode",
                "timestamp": dt.datetime.utcnow().isoformat(),
                "date": date,
                "salience": salience,
                "strength": 1.0,
            }
            vector = manager.embedder.embed_text(episode_line)
            # Supersede a near-duplicate episode (repeated low-content
            # sessions otherwise accrete "the user greeted..." forever).
            supersede_near_duplicate_episode(manager, vector)
            manager.vector_store.add(
                VectorRecord(vector=list(vector), text=episode_line, metadata=metadata)
            )
            prune_episodes_over_cap(manager)
            manager.vector_store.persist()
            manager._vector_dirty = False
        if manager.profile is not None:
            manager.profile.last_episode = episode_line[:280]
            manager._save_profile()
    manager.profile_block_dirty = True


def supersede_near_duplicate_episode(
    manager: "MemoryManager", vector: Sequence[float]
) -> None:
    """Archive-and-remove an existing episode this vector nearly duplicates.

    Called under ``_state_lock`` with the vector of the episode about to be
    added. Repeated low-content sessions (greeting-only, "hi again") would
    otherwise each write a near-identical episode; keeping only the newest
    stops the store from accreting them. Pinned episodes are never removed.
    """
    threshold = float(
        getattr(manager.config.memory, "episode_supersede_threshold", 0.0) or 0.0
    )
    if threshold <= 0 or not manager.vector_store or not manager.vector_store.count():
        return
    # Overfetch: similarity_search returns all record types; scan for the
    # nearest that is an (unpinned) episode.
    for match in manager.vector_store.similarity_search(vector, limit=5):
        if match.score < threshold:
            break  # results are score-descending; nothing closer remains
        metadata = match.metadata or {}
        if metadata.get("type") != "episode" or manager._is_protected(metadata):
            continue
        if match.id:
            # Archive-not-delete, same paper-trail contract as the fact
            # supersede paths (fact_archive.py) — this used to hard-delete.
            manager._archive_and_delete_locked(
                match.id, match.text, metadata, "episode_superseded"
            )
            telemetry.incr("memory.episodes_superseded")
        return


def prune_episodes_over_cap(manager: "MemoryManager") -> None:
    """Prune the oldest, lowest-salience episodes beyond ``max_episodes``.

    Called under ``_state_lock`` after adding an episode. Pinned episodes
    are exempt and don't count against the cap.
    """
    cap = int(getattr(manager.config.memory, "max_episodes", 0) or 0)
    if cap <= 0 or not manager.vector_store:
        return
    episodes = [
        rec
        for rec in manager.vector_store.dump()
        if (rec.metadata or {}).get("type") == "episode"
        and not manager._is_protected(rec.metadata)
    ]
    if len(episodes) <= cap:
        return

    def _rank(rec: VectorRecord) -> Tuple[float, str]:
        meta = rec.metadata or {}
        salience = _as_float(meta.get("salience"), 0.5)
        stamp = str(meta.get("timestamp") or meta.get("date") or "")
        return (salience, stamp)

    # Lowest salience first, then oldest, are dropped — archived first
    # (archive-not-delete, mirroring prune_facts_over_cap's "evicted_cap").
    episodes.sort(key=_rank)
    pruned = 0
    for rec in episodes[: len(episodes) - cap]:
        if rec.id:
            manager._archive_and_delete_locked(
                rec.id, rec.text, rec.metadata, "episode_cap"
            )
            pruned += 1
    if pruned:
        telemetry.incr("memory.episodes_pruned", pruned)


def prune_facts_over_cap(manager: "MemoryManager") -> None:
    """Archive the weakest unpinned facts beyond ``max_facts``.

    Mirrors ``prune_episodes_over_cap``'s cadence (run once after a batch of
    new facts lands) but ranks by EFFECTIVE strength — ``strength * 0.5 **
    (age_days / strength_half_life_days)``, the exact decay formula
    ``recall.py`` uses to rank/filter — rather than salience: facts don't
    carry the episode job's tone-derived salience signal, and effective
    strength is what already determines which facts survive recall's
    forgetting filter, so eviction order matches what recall would drop
    first anyway. Ties broken oldest-first. Pinned facts are exempt and
    don't count against the cap. Unlike the episode pruner (called under the
    caller's lock), this acquires its own lock — it runs standalone from
    ``_on_summary_ready``, after that method's own locked block has already
    released. 0 disables. Counter: memory.facts_evicted (once per evicted
    fact).
    """
    cap = int(getattr(manager.config.memory, "max_facts", 0) or 0)
    if cap <= 0 or not manager.vector_store:
        return
    half_life = float(
        getattr(manager.config.memory, "strength_half_life_days", 90.0) or 90.0
    )
    now = dt.datetime.utcnow()

    def _effective_strength(meta: Dict[str, Any]) -> float:
        strength = _as_float(meta.get("strength"), 1.0)
        stamp = meta.get("timestamp")
        if not stamp or half_life <= 0:
            return strength
        try:
            parsed = dt.datetime.fromisoformat(str(stamp))
        except ValueError:
            return strength
        age_days = max(0.0, (now - parsed).total_seconds() / 86400.0)
        return strength * (0.5 ** (age_days / half_life))

    with manager._state_lock:
        # Moments (amadeus/moments.py) ride the same store_fact machinery
        # and the same cap — they're bounded far more tightly day-to-day by
        # moments.max_per_day, but a very long-lived install should still
        # have them age out on the same effective-strength schedule as
        # everything else rather than growing unbounded forever.
        facts = [
            rec
            for rec in manager.vector_store.dump()
            if (rec.metadata or {}).get("type") in ("fact", "moment")
            and not manager._is_protected(rec.metadata)
        ]
        if len(facts) <= cap:
            return

        def _rank(rec: VectorRecord) -> Tuple[float, str]:
            meta = rec.metadata or {}
            return (_effective_strength(meta), str(meta.get("timestamp") or ""))

        # Weakest effective strength first, then oldest, are evicted.
        facts.sort(key=_rank)
        evicted = 0
        for rec in facts[: len(facts) - cap]:
            manager._archive_and_delete_locked(
                rec.id, rec.text, rec.metadata, "evicted_cap"
            )
            evicted += 1
        if evicted:
            manager.vector_store.persist()
            telemetry.incr("memory.facts_evicted", evicted)


def run_reflection_job(manager: "MemoryManager") -> None:
    """Session-end reflection: beliefs Kurisu forms about the user."""
    if not manager.summary_provider:
        return
    episodes: List[str] = []
    if manager.vector_store:
        with manager._state_lock:
            records = list(manager.vector_store.dump())
        episodes = [
            rec.text for rec in records if (rec.metadata or {}).get("type") == "episode"
        ][-5:]
    facts = dict(manager.profile.facts) if manager.profile else {}
    if not episodes and not facts:
        return
    parts: List[str] = []
    if facts:
        parts.append("Known facts:\n" + json.dumps(facts, ensure_ascii=False))
    if episodes:
        parts.append("Recent sessions:\n" + "\n".join(f"- {e}" for e in episodes))
    prompt = [
        ChatMessage(
            role="system",
            content=(
                # Neutral like the episode logger: recalled into future
                # prompts, so keep it factual observation, not in-character
                # commentary that compounds tone across sessions.
                "You are a neutral analyst noting durable observations about "
                "the user from the material below. State 1-2 short, factual "
                "observations about their interests, patterns, or what they "
                "are working on. Third person ('the user'), one line each, no "
                "opinions or judgments, no preamble."
            ),
        ),
        ChatMessage(role="user", content="\n\n".join(parts)),
    ]
    try:
        text = (
            manager.summary_provider.generate(
                prompt,
                max_tokens=int(
                    getattr(manager.config.memory, "reflection_max_tokens", 200) or 200
                ),
            )
            or ""
        ).strip()
        telemetry.record_llm_usage(manager.summary_provider, "memory")
    except Exception as exc:  # pragma: no cover - provider runtime errors
        LOGGER.warning("Reflection failed: %s", exc)
        telemetry.incr("memory.reflection_failures")
        return
    if not text or not (manager.embedder and manager.vector_store):
        return
    with manager._state_lock:
        metadata = {
            "type": "reflection",
            "timestamp": dt.datetime.utcnow().isoformat(),
            "salience": 0.8,
            "strength": 1.0,
        }
        vector = manager.embedder.embed_text(text)
        manager.vector_store.add(
            VectorRecord(vector=list(vector), text=text, metadata=metadata)
        )
        manager.vector_store.persist()
        manager._vector_dirty = False
