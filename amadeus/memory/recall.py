"""Per-turn long-term recall ranking, split out of ``memory/manager.py``.

Same free-function-over-owner pattern as :mod:`amadeus.memory.introspection`
(and the presence family): :func:`recall_detailed` takes the ``MemoryManager``
as its first argument and reads/writes state on it — ``MemoryManager`` keeps a
one-line delegate so callers and tests are unaffected. Extracted verbatim for
the manager's line budget (tests/test_file_budgets.py); the ranking logic and
its telemetry are unchanged.
"""

from __future__ import annotations

import datetime as dt
from typing import TYPE_CHECKING, List, Tuple

from .. import telemetry
from ..logging_utils import get_logger
from . import confidence, speculative
from .text_utils import _already_discussed, _as_float, _parse_iso
from .vector_store import VectorMatch

if TYPE_CHECKING:  # pragma: no cover - import cycle guard
    from .manager import MemoryManager

LOGGER = get_logger(__name__)


def recall_detailed(
    manager: "MemoryManager",
    query: str,
    limit: int = 3,
    *,
    include_episodes: bool = False,
    _from_speculative: bool = False,
) -> List[VectorMatch]:
    """Relevance x recency (x salience) ranked long-term recall.

    Strength is NOT a ranking factor — it only gates survival, as a
    forgetting-curve filter (``strength * decay < strength_floor`` drops a
    record from the candidate pool below, when ``enable_forgetting`` is on).
    The weight formula that actually orders survivors is relevance (cosine
    score) x recency x salience.

    ``_from_speculative`` marks the internal pre-started call submitted
    by :meth:`MemoryManager.speculate_recall`, which must not consume its
    own pending stash entry (it would deadlock waiting on itself).
    """
    # Imported here, not at module top: _EmbedTimeout lives in manager.py,
    # which imports this module — a top-level import would be circular.
    from .manager import _EmbedTimeout

    if not _from_speculative:
        hit, result = speculative.consume_speculative(
            manager, query, limit, include_episodes
        )
        if hit:
            return result

    if not (
        manager.config.memory.enable_long_term
        and manager.embedder
        and manager.vector_store
    ):
        return []

    memory = manager.config.memory
    config_min_score = float(getattr(memory, "recall_min_score", 0.0) or 0.0)
    # Per-embedder noise floor: semantic embedders score unrelated text
    # far above the config default (gemini-embedding-2 puts pure noise at
    # ~0.45-0.59, where recall_min_score's 0.35 filters nothing — the
    # "she keeps routing to memories on irrelevant questions" bug). The
    # embedder declares where ITS noise band ends; the stricter of the
    # two governs. See EmbeddingProvider.relevance_floor.
    embedder_floor = float(getattr(manager.embedder, "relevance_floor", 0.0) or 0.0)
    min_score = max(config_min_score, embedder_floor)
    half_life = float(getattr(memory, "recency_half_life_days", 0.0) or 0.0)
    forgetting = bool(getattr(memory, "enable_forgetting", False))
    strength_half_life = float(getattr(memory, "strength_half_life_days", 90.0) or 90.0)
    strength_floor = float(getattr(memory, "strength_floor", 0.25) or 0.25)
    facts_only = bool(getattr(memory, "recall_facts_only", True))
    confidence_enabled = bool(getattr(memory, "recall_confidence", True))
    summary_dedupe = float(getattr(memory, "recall_summary_dedupe", 0.0) or 0.0)

    # Trivial turns ("hi", "ok", "thanks") carry no searchable meaning;
    # skipping recall saves the query-embed round-trip on exactly the
    # turns where responsiveness matters most.
    min_chars = int(getattr(memory, "recall_min_query_chars", 0) or 0)
    if min_chars and len(query.strip()) < min_chars:
        return []

    try:
        query_vec = manager._embed_query_bounded(query)
    except _EmbedTimeout:
        LOGGER.info("Recall query embedding timed out; proceeding without recall")
        telemetry.incr("memory.recall_embed_timeouts")
        return []
    except Exception as exc:
        # A network embedder (Gemini) can fail transiently; a chat turn
        # must survive that with empty recall, not crash.
        LOGGER.warning("Recall query embedding failed: %s", exc)
        telemetry.incr("memory.embed_failures")
        return []
    if not manager._check_embedding_dimension(len(query_vec)):
        return []
    # A summary-shaped record recalled alongside the rolling summary already
    # in context is pure repetition; embed the summary once so near-copies
    # drop below. Bounded like the query embed (this was the one remaining
    # unbounded network call on the recall path): timeout/failure degrades
    # to skipping summary-dedupe — never a stalled or crashed turn.
    summary_vec = None
    if summary_dedupe > 0 and manager.summary:
        try:
            summary_vec = manager._embed_query_bounded(manager.summary)
        except _EmbedTimeout:
            telemetry.incr("memory.recall_dedupe_embed_timeouts")
        except Exception:  # pragma: no cover - embedder best effort
            pass
    with manager._state_lock:
        # Overfetch so type-filtering and decay re-ranking have candidates
        # to work with; episode-heavy stores would otherwise starve facts.
        fetch = limit * 6 if facts_only else limit * 3
        matches = manager.vector_store.similarity_search(query_vec, fetch)
    now = dt.datetime.utcnow()
    ranked: List[Tuple[float, VectorMatch]] = []
    floor_filtered = 0
    for match in matches:
        if match.score < min_score or match.score <= 0:
            # Visibility for the embedder noise floor (degrade-loudly):
            # candidates the config threshold alone would have admitted.
            # A high value alongside empty-recall complaints means the
            # floor is set too strict for this store's score spread.
            if embedder_floor > config_min_score and match.score >= config_min_score:
                floor_filtered += 1
            continue
        metadata = match.metadata or {}
        pinned = manager._is_protected(metadata)
        if facts_only and not pinned:
            # Episodes/reflections re-narrate whole sessions; keep them for
            # the overview, out of per-turn recall. Untyped legacy records
            # (raw message captures) are conversational, not durable facts.
            # A retrospective-intent turn ("remember when...") is the one
            # case a per-turn query can only be answered from an episode,
            # so include_episodes admits type=="episode" only — reflections
            # and untyped records stay excluded either way. type=="moment"
            # (amadeus/moments.py, opt-in shared-activity memories) is
            # explicitly a durable-fact-shaped record and always admitted
            # here alongside plain facts — it rides the same temporal-aging
            # and fixation-guard treatment below, unchanged.
            record_type = metadata.get("type")
            if record_type == "episode" and include_episodes:
                pass
            elif record_type in ("episode", "reflection") or not record_type:
                continue
        age_days = None
        stamped = _parse_iso(metadata.get("timestamp"))
        if stamped is not None:
            age_days = max(0.0, (now - stamped).total_seconds() / 86400.0)
        recency = 1.0
        if age_days is not None and half_life > 0:
            recency = 0.5 ** (age_days / half_life)
        strength = _as_float(metadata.get("strength"), 1.0)
        if forgetting and not pinned:
            effective_strength = strength
            if age_days is not None and strength_half_life > 0:
                effective_strength = strength * (0.5 ** (age_days / strength_half_life))
            if effective_strength < strength_floor:
                continue
        # Calibrated recall (confidence.py): how sure she should SOUND about
        # this fact, as opposed to how highly it ranks. Not a ranking input.
        # Called here because `strength` and `age_days` are already in hand.
        if confidence_enabled:
            confidence.annotate_confidence(
                match,
                metadata=metadata,
                strength=strength,
                age_days=age_days,
                pinned=pinned,
            )
        # Canonical retrieval scoring: relevance x recency x importance.
        # Salience is stored on every fact (_store_fact metadata) but was
        # previously dropped in ranking; neutral at salience 0.5 (weight
        # x1.0) so untouched records keep their old ranking, scaling down
        # to x0.7 at 0 and up to x1.3 at 1.
        weight = (
            match.score
            * recency
            * (0.7 + 0.6 * _as_float(metadata.get("salience"), 0.5))
        )
        if pinned:
            weight *= 1.1
        ranked.append((weight, match))
    ranked.sort(key=lambda pair: pair[0], reverse=True)
    # "Already discussed this conversation" window: a fact whose content
    # is sitting in short-term context doesn't need recalling, and
    # re-injecting it every turn is what makes her hang onto an old topic
    # instead of following the user to the present one. The current user
    # message is excluded — it's the thing recall is answering.
    recent_dedupe = float(getattr(memory, "recall_recent_dedupe", 0.0) or 0.0)
    recent_text = ""
    if recent_dedupe > 0:
        with manager._state_lock:
            window = list(manager.short_term)[-12:]
        if window and window[-1].role == "user":
            window = window[:-1]
        recent_text = " ".join(msg.content.lower() for msg in window)
    # Summary dedupe runs lazily over the weight-ranked list: each check
    # costs a network embed (LRU-cached per text) for a network embedder,
    # so checking all `fetch` candidates up front would stack sequential
    # round-trips. Walking in rank order and stopping once `limit`
    # survivors are found selects the exact same records.
    selected: List[VectorMatch] = []
    for _, match in ranked:
        if len(selected) >= limit:
            break
        if (
            recent_text
            and match.text
            and _already_discussed(match.text, recent_text, recent_dedupe)
        ):
            continue
        if summary_vec is not None and match.text:
            # Bounded like the query embed: this per-candidate embed used
            # to be the one recall-path network call with NO deadline, so
            # a slow embedder stalled the whole turn here even though the
            # query embed was deadline-guarded. On timeout/failure the
            # candidate survives (skipping dedupe beats losing recall).
            try:
                sim = manager.embedder.similarity(
                    summary_vec, manager._embed_query_bounded(match.text)
                )
            except _EmbedTimeout:
                telemetry.incr("memory.recall_dedupe_embed_timeouts")
                sim = 0.0
            except Exception:  # pragma: no cover - embedder best effort
                sim = 0.0
            if sim >= summary_dedupe:
                continue
        selected.append(match)
    if floor_filtered:
        telemetry.incr("memory.recall_floor_filtered", floor_filtered)
    if confidence_enabled and selected:
        confidence.record_confidence_telemetry(selected)
    if forgetting and selected:
        manager._reinforce(selected)
    return selected
