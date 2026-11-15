"""Speculative (STT-time pre-started) per-turn recall.

When the web STT endpoint finishes transcribing a voice clip, the per-turn
long-term recall for that transcript can be pre-started in the background,
so the query-embed round trip (~0.45s warm on a network embedder) overlaps
the client's transcript->chat round trip instead of sitting serially inside
the turn (``MemorySettings.speculative_recall``, default on).

Free functions taking the ``MemoryManager`` as first argument — the same
free-function-over-owner pattern as ``memory/introspection.py`` — so the
manager keeps only a thin delegate plus its two state slots
(``_speculative_lock``/``_speculative``). Correctness story: the stashed
future is consumed only when the arriving recall query matches the
transcript's (whitespace-normalized) with identical limit/include_episodes,
and the background job runs the exact same ``recall_detailed`` body (same
config, dedupe, embed deadlines) on the same manager — so a wrong guess can
never change what a turn recalls, only waste one embed. Health:
``memory.speculative_recall_hits``/``_misses`` counters in ``/api/status``.
"""

from __future__ import annotations

import time
from concurrent.futures import CancelledError as FuturesCancelled
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

from .. import telemetry
from ..logging_utils import get_logger

LOGGER = get_logger(__name__)

# Worker for speculative recall jobs — separate from manager._EMBED_EXECUTOR
# so a speculative job can never queue behind, or delay, a live turn's
# deadline-bounded query embed.
_SPECULATIVE_EXECUTOR = ThreadPoolExecutor(
    max_workers=1, thread_name_prefix="amadeus-speculative-recall"
)


def _normalize_query(text: str) -> str:
    """Whitespace-collapsed form used to match a speculative recall future
    against the chat message that eventually arrives."""
    return " ".join((text or "").split())


@dataclass
class _SpeculativeRecall:
    """One pre-started recall lookup (see :func:`speculate_recall`)."""

    query_key: str
    limit: int
    include_episodes: bool
    expires_at: float
    future: Future


def speculate_recall(manager: Any, user_text: str, limit: int = 3) -> Optional[Future]:
    """Fire-and-forget: pre-start the per-turn recall for ``user_text``.

    Called (via the manager's delegate) from the web STT path right after
    transcription. A newer call replaces (and best-effort cancels) the
    previous stash; entries expire after
    ``memory.speculative_recall_ttl_seconds``. Returns the started future
    (for tests), or None when speculation didn't start.
    """
    memory_cfg = manager.config.memory
    if not bool(getattr(memory_cfg, "speculative_recall", True)):
        return None
    text = (user_text or "").strip()
    if not text:
        return None
    if not (memory_cfg.enable_long_term and manager.embedder and manager.vector_store):
        return None  # recall_detailed would return [] instantly anyway
    # Mirror assemble_turn_context's query augmentation. The transcript has
    # NOT been added to short_term yet, so _last_user_text sees the same
    # "previous user turn" the real call will see after add_message appends
    # the current one. If the two ever diverge (consecutive user messages,
    # an edited transcript) the key simply won't match and the turn runs the
    # normal path. Lazy imports: context.py imports manager.py at top level.
    from .context import _last_user_text, _needs_query_augmentation
    from .retrospective import is_retrospective
    from .text_utils import _raw_query_too_short

    min_chars = int(getattr(memory_cfg, "recall_min_query_chars", 0) or 0)
    if _raw_query_too_short(text, min_chars):
        # Same raw-text gate as assemble_turn_context, checked BEFORE
        # augmentation for the same reason (see _raw_query_too_short's
        # docstring): a short raw transcript that would get lengthened by
        # the previous-turn prefix must still skip recall, so nothing is
        # worth pre-starting here — the real turn will also skip it.
        return None
    recall_query = text
    if _needs_query_augmentation(text):
        previous_user_text = _last_user_text(manager)
        if previous_user_text:
            recall_query = f"{previous_user_text}\n{text}"
    include_episodes = is_retrospective(text)
    ttl = float(getattr(memory_cfg, "speculative_recall_ttl_seconds", 10.0) or 10.0)
    entry = _SpeculativeRecall(
        query_key=_normalize_query(recall_query),
        limit=limit,
        include_episodes=include_episodes,
        expires_at=time.monotonic() + ttl,
        future=_SPECULATIVE_EXECUTOR.submit(
            manager.recall_detailed,
            recall_query,
            limit,
            include_episodes=include_episodes,
            _from_speculative=True,
        ),
    )
    with manager._speculative_lock:
        prior, manager._speculative = manager._speculative, entry
    if prior is not None:
        prior.future.cancel()
    telemetry.incr("memory.speculative_recall_started")
    return entry.future


def _take_speculative(
    manager: Any, query: str, limit: int, include_episodes: bool
) -> Optional[Future]:
    """Pop the stashed speculative future when it matches this recall call.

    An expired entry is dropped (and counted as a miss); a live entry that
    doesn't match is counted as a miss but left in place — an unrelated
    recall (the forget path, memory endpoints) between STT and the chat
    turn must not eat the chat turn's future.
    """
    with manager._speculative_lock:
        entry = manager._speculative
        if entry is None:
            return None
        if time.monotonic() > entry.expires_at:
            manager._speculative = None
            entry.future.cancel()
            telemetry.incr("memory.speculative_recall_misses")
            return None
        if (
            entry.query_key != _normalize_query(query)
            or entry.limit != limit
            or entry.include_episodes != include_episodes
        ):
            telemetry.incr("memory.speculative_recall_misses")
            return None
        manager._speculative = None
    return entry.future


def consume_speculative(
    manager: Any, query: str, limit: int, include_episodes: bool
) -> Tuple[bool, Optional[List[Any]]]:
    """Try to answer a ``recall_detailed`` call from the pre-started future.

    Returns ``(hit, result)``: ``(True, matches)`` when the stash matched
    and its result (possibly an empty list — still a hit) is usable;
    ``(False, None)`` when the caller must run the normal serial path.
    """
    future = _take_speculative(manager, query, limit, include_episodes)
    if future is None:
        return False, None
    wait_s = float(
        getattr(manager.config.memory, "speculative_recall_ttl_seconds", 10.0) or 10.0
    )
    try:
        result = future.result(timeout=wait_s)
    except (Exception, FuturesCancelled):
        # Timeout, cancellation, or an unexpected error inside the
        # background job: fall back to the normal serial path — speculation
        # must never lose a recall.
        telemetry.incr("memory.speculative_recall_misses")
        return False, None
    telemetry.incr("memory.speculative_recall_hits")
    return True, result
