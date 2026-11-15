"""Conversation memory management."""

from __future__ import annotations

import dataclasses
import datetime as dt
import json
import os
import shutil
import threading
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeout
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Sequence, Tuple, Union

from ..config import Config
from ..logging_utils import get_logger
from .. import telemetry
from ..providers.llm.base import LLMProvider

# Re-exported for external callers (provider_loader, cli, engine, web/sessions,
# tests) that import the embedders from amadeus.memory.manager; the noqa keeps
# ruff's unused-import autofix from stripping them.
from . import (
    extraction_jobs,
    fact_archive,
    graph as graph_module,
    introspection,
    speculative,
)
from . import recall as recall_module
from .embedders import EmbeddingProvider  # noqa: F401
from .embedders import GeminiEmbedding  # noqa: F401
from .embedders import HashingEmbedding  # noqa: F401
from .embedders import SentenceTransformerEmbedding  # noqa: F401
from .profile import UserProfile, updated_arc, updated_trust
from .text_utils import (
    _as_float,
    _parse_iso,
)
from .vector_store import (
    ChromaVectorStore,
    InMemoryVectorStore,
    VectorMatch,
    VectorRecord,
    VectorStore,
)

LOGGER = get_logger(__name__)

# Worker pool for deadline-bounded recall-query embeds (see
# MemoryManager._embed_query_bounded). Two workers so one embed that has
# already blown its deadline can't also delay the NEXT turn's embed while it
# finishes in the background.
_EMBED_EXECUTOR = ThreadPoolExecutor(max_workers=2, thread_name_prefix="recall-embed")


class _EmbedTimeout(Exception):
    """Recall query embed exceeded recall_embed_timeout_seconds."""


@dataclass
class Message:
    role: str
    content: str
    timestamp: Optional[str] = None


@dataclass
class MemoryState:
    """Serializable memory state."""

    messages: List[Message] = field(default_factory=list)
    summary: Optional[str] = None
    summary_metadata: Dict[str, Any] = field(default_factory=dict)
    long_term: List[Tuple[List[float], str]] = field(default_factory=list)


class _Protected:
    """Sentinel returned by ``forget``/``forget_id`` when the best/targeted
    match is pinned, so callers can distinguish "protected, don't touch it"
    from "nothing matched" (``None``) or "here's what was removed" (a str).
    A dedicated type (rather than reusing None or a string) keeps this
    unambiguous even if forgotten text is ever empty/falsy."""

    __slots__ = ()

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return "PROTECTED"


PROTECTED = _Protected()


class MemoryManager:
    """Tracks short-term chat context and optional long-term recall."""

    def __init__(
        self,
        config: Config,
        embedder: Optional["EmbeddingProvider"] = None,
        *,
        persist_path: Optional[Union[str, Path]] = None,
        embeddings_path: Optional[Union[str, Path]] = None,
        profile_path: Optional[Union[str, Path]] = None,
        summary_provider: Optional[LLMProvider] = None,
    ) -> None:
        self.config = config
        self.embedder = embedder
        # Character short-name for the summary/extraction prompts
        # (BYO-character packs rename the assistant side of the transcript).
        _char = getattr(config, "character", None)
        self._char_name = (getattr(_char, "name", None) or "Kurisu").strip() or "Kurisu"
        self.short_term: Deque[Message] = deque(
            maxlen=config.memory.max_context_messages
        )
        self.summary: Optional[str] = None
        self.summary_info: Dict[str, Any] = {}
        self.summary_max_chars: int = getattr(config.memory, "summary_max_chars", 500)
        self.dedupe_threshold: float = float(
            getattr(config.memory, "dedupe_threshold", 0.0) or 0.0
        )
        self.summary_provider = summary_provider
        self._summary_executor: Optional[ThreadPoolExecutor] = None
        self._summary_future: Optional[Future] = None
        self._messages_since_summary: int = 0
        # Messages appended since the last successful summary, kept
        # SEPARATELY from the bounded short_term deque: short_term evicts
        # silently once it hits max_context_messages (12 by default), and
        # slicing it by count to build the incremental-summary delta lost
        # any message that fell off the front while a summary job was still
        # in flight (or simply hadn't run yet) — a real data-loss bug, not
        # just a display quirk, since a lost message never reaches the
        # summarizer at all. This list is the delta source instead; drained
        # on successful summary completion in _on_summary_ready. Capped so a
        # dead/hung summary provider can't grow it without bound.
        self._unsummarized: List[Message] = []
        self._unsummarized_cap: int = 60
        # How many _unsummarized entries the in-flight summary job was
        # submitted with: the done-callback drains exactly this many off the
        # front (clear() also swallowed messages appended WHILE the job ran,
        # silently losing them from summarization/extraction). Single slot is
        # safe — at most one summary job is ever in flight.
        self._unsummarized_submitted: int = 0
        # Set when summarize() is requested while a job is in flight; the
        # done-callback consumes it and re-submits (see _on_summary_ready).
        self._resummarize_queued: bool = False
        # Bounded attempt counter for the first-meeting fast path in
        # add_message: caps how many times the early-extraction summarize()
        # can fire before profile.facts is populated, so a run of extraction
        # calls that keep yielding zero facts can't re-fire forever (the
        # elif's own guards reset _messages_since_summary each time, which
        # would otherwise let it retrigger roughly every other user message).
        self._early_extract_attempts: int = 0
        # Serializes state mutation/persistence between the chat thread and
        # the background executor thread.
        self._state_lock = threading.RLock()
        # At most one pre-started recall future (speculative.py); its own
        # lock because the STT thread writes it while a turn thread consumes.
        self._speculative_lock = threading.Lock()
        self._speculative: Optional[speculative._SpeculativeRecall] = None
        self._last_tone: Optional[float] = None
        self._vector_dirty = False
        # Embedder/store dimension compatibility, checked lazily on first
        # recall (see _check_embedding_dimension).
        self._dim_check_done = False
        self._dim_check_ok = True
        # Per-character memory isolation: paths that fall back to the config
        # are scoped to the active custom character (Config.character_memory_
        # path — a no-op for the builtin Kurisu, so legacy stores stay put).
        # Explicitly passed paths are used as-is: the web SessionStore
        # already scopes its per-user bases before handing them down.
        _scope = getattr(config, "character_memory_path", None)
        if not callable(_scope):

            def _scope(value):
                return value

        raw_persist = (
            persist_path
            if persist_path is not None
            else _scope(config.memory.persist_path)
        )
        raw_embeddings = (
            embeddings_path
            if embeddings_path is not None
            else _scope(config.memory.embeddings_path)
        )
        raw_vector_path = _scope(getattr(config.memory, "vector_path", None))
        self.persist_path = Path(raw_persist) if raw_persist else None
        self.embeddings_path = Path(raw_embeddings) if raw_embeddings else None
        self.vector_path = Path(raw_vector_path) if raw_vector_path else None
        # The single-worker executor is load-bearing: it serializes summary,
        # fact, episode, and reflection jobs so memory writes never interleave.
        if self.summary_provider or (config.memory.enable_long_term and embedder):
            self._summary_executor = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="amadeus-summary"
            )
        # Persistence backend for the message/summary store and (when
        # vector_backend is json/memory) the vector store: "json" (default,
        # the legacy rewrite-whole-file stores) or "sqlite" (see
        # sqlite_store.py). Chroma is unaffected either way — it's already
        # its own persistent backend.
        self.storage_backend: str = str(
            getattr(config.memory, "storage_backend", "json") or "json"
        )
        self.vector_store: Optional[VectorStore] = None
        if self.config.memory.enable_long_term:
            backend = getattr(config.memory, "vector_backend", "json")
            if backend in {"json", "memory"}:
                if self.storage_backend == "sqlite":
                    from .sqlite_store import SqliteVectorStore

                    self.vector_store = SqliteVectorStore(path=self.embeddings_path)
                else:
                    self.vector_store = InMemoryVectorStore(path=self.embeddings_path)
            elif backend == "chroma":
                chroma_path = self.vector_path or self.embeddings_path
                try:
                    self.vector_store = ChromaVectorStore(path=chroma_path)
                except RuntimeError as exc:
                    LOGGER.error("%s Falling back to JSON vector store.", exc)
                    self.vector_store = InMemoryVectorStore(path=self.embeddings_path)
            else:
                LOGGER.warning(
                    "Unknown vector backend '%s'; defaulting to JSON-backed store",
                    backend,
                )
                self.vector_store = InMemoryVectorStore(path=self.embeddings_path)
        # User profile ("core memory") is only active with the new features so
        # default installs keep byte-identical prompts.
        self.profile: Optional[UserProfile] = None
        self.profile_path: Optional[Path] = None
        self.profile_block_dirty = False
        if self._profile_features_enabled():
            raw_profile = (
                profile_path
                if profile_path is not None
                else _scope(getattr(config.memory, "profile_path", None))
            )
            self.profile_path = Path(raw_profile) if raw_profile else None
            self.profile = (
                UserProfile.load(self.profile_path)
                if self.profile_path
                else UserProfile()
            )
            self.profile.prune_stale_threads(
                int(getattr(config.memory, "open_thread_max_age_days", 21) or 0)
            )
            self._maybe_begin_session()
        if self.persist_path and self._state_store_path().exists():
            self._load()

    def __del__(self) -> None:  # pragma: no cover - best effort cleanup
        if self._summary_executor:
            self._summary_executor.shutdown(wait=False)

    def set_summary_provider(self, provider: Optional[LLMProvider]) -> None:
        """Hot-swap the summarization LLM (e.g. after a settings change).

        A manager built with no summary provider (dummy LLM at startup) never
        allocates ``_summary_executor``, so summarize()/finalize_session() are
        permanently no-ops even after a real key is configured later — unless
        we lazily create the executor here when a provider first shows up.
        """
        self.summary_provider = provider
        self._ensure_summary_executor()

    def set_embedder(self, embedder: Optional["EmbeddingProvider"]) -> None:
        """Hot-swap the embedding provider (e.g. after a settings change).

        Same lazy gap as the summary provider: long-term storage jobs are
        submitted to ``_summary_executor``, which a manager built without an
        embedder never allocated.
        """
        self.embedder = embedder
        # New embedder, new dimension: re-run the store-compatibility check on
        # the next recall.
        self._dim_check_done = False
        self._ensure_summary_executor()

    def _ensure_summary_executor(self) -> None:
        """Create the shared background executor if it's newly needed.

        Mirrors the gating in ``__init__``: the executor backs summary, fact,
        episode, and reflection jobs, and is required as soon as either a
        summary provider or long-term memory (embedder) is available.
        """
        if self._summary_executor is not None:
            return
        if self.summary_provider or (
            self.config.memory.enable_long_term and self.embedder
        ):
            self._summary_executor = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="amadeus-summary"
            )

    def _profile_features_enabled(self) -> bool:
        memory = self.config.memory
        return bool(
            getattr(memory, "enable_fact_extraction", False)
            or getattr(memory, "enable_episodes", False)
        )

    def _apply_session_boundary_state(self) -> None:
        """Trust + arc drift + tone-history append at a session boundary.

        Caller holds any needed lock and saves. The tone comes from the live
        in-process value when available (in-process gap boundary) or from the
        profile's persisted ``last_session_tone`` (fresh-process boundary —
        the dominant path, where the extraction job's tone died with the old
        process). Either way the persisted slot is cleared, so one session's
        tone never drifts trust/arc twice.
        """
        stored_tone, stored_date = self.profile.pop_session_tone()
        tone = self._last_tone if self._last_tone is not None else stored_tone
        tone_date = None if self._last_tone is not None else stored_date
        # One-shot, like the persisted slot: in a long-lived process every
        # LATER boundary where extraction produced no fresh tone would
        # otherwise re-consume this stale value — duplicate trust/arc drift
        # plus a tone_history entry stamped today for an old session.
        self._last_tone = None
        self.profile.trust_level = updated_trust(
            self.profile.trust_level,
            self.profile.session_count,
            tone,
        )
        (
            self.profile.arc_curiosity_vs_annoyance,
            self.profile.arc_logic_vs_emotion,
            self.profile.arc_pride_vs_vulnerability,
        ) = updated_arc(
            self.profile.arc_curiosity_vs_annoyance,
            self.profile.arc_logic_vs_emotion,
            self.profile.arc_pride_vs_vulnerability,
            session_count=self.profile.session_count,
            trust_level=self.profile.trust_level,
            last_tone=tone,
        )
        if tone is not None:
            self.profile.record_tone(tone, when=tone_date)
            telemetry.incr("memory.boundary_tone_applied")
        # Time repairs friction unconditionally. This runs at every session
        # boundary regardless of the feature flag, so disabling the feature
        # still lets already-noted topics age out rather than freezing them
        # in the profile forever.
        max_age = int(getattr(self.config.memory, "friction_max_age_days", 21) or 21)
        dropped = self.profile.prune_stale_friction(max_age_days=max_age)
        if dropped:
            telemetry.incr("memory.friction_expired", dropped)

    def _maybe_begin_session(self) -> None:
        """Count a new session when we come back after a real gap.

        scope_context_to_session interaction: this is driven entirely by
        ``self.profile.last_seen`` (per-user, disk-persisted) vs.
        ``session_gap_minutes`` — it never reads short_term, so it is
        unaffected by whether short_term was just restored, hydrated from a
        transcript tail, or left empty. Two conversations (session ids)
        started back-to-back for the same user already only count once
        today (the second construction sees a ``last_seen`` from moments ago
        and the gap check fails) — that dynamic is unchanged by this flag.
        Scoping context to a conversation does not scope "session" in the
        trust/arc/relationship sense to a conversation; those remain
        per-user and gated purely by wall-clock gap, by design.
        """
        if not self.profile:
            return
        gap_minutes = float(
            getattr(self.config.memory, "session_gap_minutes", 30) or 30
        )
        last_seen = _parse_iso(self.profile.last_seen)
        if (
            last_seen is None
            or (dt.datetime.utcnow() - last_seen).total_seconds() / 60 > gap_minutes
        ):
            self.profile.session_count += 1
            # Session boundary = the one place trust, arc drift, and tone
            # history move. Fresh-process starts read the previous session's
            # ending tone from the profile's persisted last_session_tone slot.
            self._apply_session_boundary_state()
            self.profile.last_seen = dt.datetime.utcnow().isoformat()
            self._save_profile()

    def add_message(
        self, role: str, content: str, *, presence_delivery: bool = False
    ) -> None:
        """Commit one message to short-term context.

        ``presence_delivery`` marks her own unprompted lines committed while
        the user is away (reach-out pop, idle chatter): the commit itself is
        intentional context, but it must not fire the gap-finalize session
        boundary — a reach-out is guaranteed preceded by hours of silence, so
        the gap check minted a phantom session (session_count increment plus
        trust/arc/tone drift for a "session" of one line she said to an empty
        room) — and must not stamp ``profile.last_seen``, which belongs to
        the user actually showing up.
        """
        now = dt.datetime.utcnow()
        # Unconditional (for real turns): the gap check also owns the
        # in-process session boundary (trust/arc/tone drift). Gating it on
        # enable_episodes silently disabled all tone-driven relationship
        # drift when episodes were off; the episode job itself is still
        # episodes-gated inside.
        if not presence_delivery:
            self._maybe_finalize_gap(now)
        message = Message(role=role, content=content, timestamp=now.isoformat())
        # Mutations under _state_lock: executor done-callbacks iterate
        # short_term/_unsummarized under this lock (_persist, the summary
        # drain in _on_summary_ready), and a deque mutated during iteration
        # inside a done-callback dies silently. Kept small — the summarize /
        # long-term submissions below stay outside it.
        with self._state_lock:
            self.short_term.append(message)
            self._unsummarized.append(message)
            if len(self._unsummarized) > self._unsummarized_cap:
                overflow = len(self._unsummarized) - self._unsummarized_cap
                del self._unsummarized[:overflow]
                telemetry.incr("memory.unsummarized_dropped", overflow)
            self._messages_since_summary += 1
            if self.profile is not None and not presence_delivery:
                self.profile.last_seen = now.isoformat()
        # Debounce: once the window is full, re-summarizing on every single
        # message doubles LLM traffic for no benefit. Refresh every few
        # messages instead.
        debounce = max(1, self.config.memory.summary_trigger_messages // 2)
        if (
            len(self.short_term) >= self.config.memory.summary_trigger_messages
            and self._messages_since_summary >= debounce
        ):
            self.summarize()
        elif (
            role == "user"
            and self.profile is not None
            and not self.profile.facts
            and getattr(self.config.memory, "enable_fact_extraction", False)
            and len(self.short_term) >= 4
            and self._messages_since_summary >= 2
            and self._early_extract_attempts < 2
        ):
            # First-meeting fast path: don't wait for the normal debounced
            # window to fill before extracting facts — "she learns your
            # name" needs to be visible within the first conversation, since
            # the onboarding UI polls for new facts. Bounded to at most 2
            # attempts: summarize() resets _messages_since_summary on every
            # call, and only profile.merge_updates (i.e. extraction actually
            # yielding facts) disarms this branch via `not self.profile.facts`
            # — without a separate attempt latch a string of empty-fact
            # extractions would re-fire this roughly every other user
            # message for the entire conversation.
            self._early_extract_attempts += 1
            self.summarize()
        # Raw-message storage is the legacy long-term path; fact extraction
        # replaces it with far fewer, denser records.
        if (
            self.config.memory.enable_long_term
            and self.embedder
            and self.vector_store
            and role == "user"
            and not getattr(self.config.memory, "enable_fact_extraction", False)
        ):
            if self._summary_executor:
                self._summary_executor.submit(self._store_long_term, role, content)
            else:
                self._store_long_term(role, content)
        self._persist()

    def rollback_user_message(self, user_text: str) -> bool:
        """Undo the most recent user-message commit (failed/cancelled turn).

        Pops the matching trailing user message from short_term AND from the
        ``_unsummarized`` delta buffer (rewinding ``_messages_since_summary``)
        — popping short_term alone left the retracted text queued for the
        next summarization/extraction pass, so a cancelled turn could still
        mint durable facts from words the user took back. Persists the pop
        (a disk error here must never mask the provider error that triggered
        the rollback). Returns True when a message was removed.
        """
        with self._state_lock:
            if not (
                self.short_term
                and self.short_term[-1].role == "user"
                and self.short_term[-1].content == user_text
            ):
                return False
            self.short_term.pop()
            # Match by role+content from the tail: the same Message was
            # appended to both lists, but compare fields in case the buffers
            # were rebuilt independently (restore/hydration paths).
            if (
                self._unsummarized
                and self._unsummarized[-1].role == "user"
                and self._unsummarized[-1].content == user_text
            ):
                self._unsummarized.pop()
            if self._messages_since_summary > 0:
                self._messages_since_summary -= 1
        try:
            self._persist()
        except Exception:  # pragma: no cover - disk full/permission
            telemetry.incr("memory.rollback_persist_failed")
            LOGGER.exception("Failed to persist user-message rollback")
        return True

    def summarize(self) -> None:
        """Summarize the conversation so far using a naive heuristic."""

        if not self.short_term:
            return
        full_snapshot = list(self.short_term)
        if self.summary_provider and self._summary_executor:
            if self._summary_future and not self._summary_future.done():
                # Queue instead of drop: _on_summary_ready re-runs summarize()
                # once the in-flight job lands, so these messages still get
                # summarized/extracted (the counter keeps accumulating).
                LOGGER.debug("Summary job already running; queueing a re-run.")
                self._resummarize_queued = True
                return
            revision = int(self.summary_info.get("revision", 0)) + 1
            # Incremental: once a summary exists, send only the messages added
            # since the last refresh plus the previous summary, instead of the
            # whole window every time. Source is _unsummarized, NOT a
            # count-based slice of short_term: short_term is a bounded deque
            # (max_context_messages, default 12) that evicts silently, so
            # slicing it by _messages_since_summary undercounted — and
            # silently dropped — any message that fell off the front while
            # this job was pending. _unsummarized holds exactly the messages
            # not yet folded into a summary, independent of short_term's
            # eviction.
            previous_summary = self.summary
            # Snapshot under the lock: the previous job's done-callback
            # drains _unsummarized under _state_lock, and copying a list
            # while another thread deletes from it is the same silent race
            # as the deque case in add_message.
            with self._state_lock:
                if previous_summary and self._unsummarized:
                    snapshot = list(self._unsummarized)
                else:
                    snapshot = full_snapshot
                # Everything pending right now is covered by this job; only
                # this many entries get drained on success, so messages
                # appended while it runs stay queued (see _on_summary_ready).
                self._unsummarized_submitted = len(self._unsummarized)
                self._messages_since_summary = 0
            profile_snapshot: Optional[str] = None
            if self.profile is not None and getattr(
                self.config.memory, "enable_fact_extraction", False
            ):
                # Include current open threads (with ages) so the extraction
                # model can carry unresolved ones forward and drop resolved
                # ones — without seeing them it can only ever append.
                snapshot_payload: Dict[str, Any] = {
                    "facts": self.profile.facts,
                    "open_threads": [
                        self.profile.describe_thread(text)
                        for text in self.profile.open_threads
                    ],
                }
                # Sore topics go in only when friction is on, and only so the
                # model can report state="repaired" when one comes up and goes
                # fine. This snapshot feeds the EXTRACTION call, never the
                # per-turn persona prompt — friction must not color every turn.
                if getattr(self.config.memory, "enable_friction_memory", False):
                    sore = [
                        str(entry.get("topic"))
                        for entry in self.profile.friction_topics
                        if str(entry.get("topic") or "").strip()
                    ]
                    if sore:
                        snapshot_payload["sore_topics"] = sore
                profile_snapshot = json.dumps(snapshot_payload, ensure_ascii=False)
            self._summary_future = self._summary_executor.submit(
                self._run_summary_job,
                snapshot,
                revision,
                previous_summary,
                profile_snapshot,
            )
            self._summary_future.add_done_callback(self._on_summary_ready)
        else:
            with self._state_lock:
                self._messages_since_summary = 0
                self._unsummarized.clear()
            self._update_summary_truncate(full_snapshot)

    def build_context(self) -> List[Dict[str, str]]:
        """Return messages ready to send to an LLM."""

        context: List[Dict[str, str]] = [
            {"role": msg.role, "content": msg.content} for msg in self.short_term
        ]
        if self.summary:
            # Trailing, not leading: the summary is rewritten every few turns,
            # and placing it between the static persona prompt and the history
            # would truncate the provider's cached prompt prefix each rewrite.
            context.append(
                {
                    "role": "system",
                    "content": (
                        "Summary of the earlier conversation (before the "
                        f"messages above): {self.summary}"
                    ),
                }
            )
        return context

    def recall(self, query: str, limit: int = 3) -> List[str]:
        """Return long-term memories relevant to ``query``."""

        return [match.text for match in self.recall_detailed(query, limit)]

    def _embed_query_bounded(self, query: str) -> List[float]:
        """Embed the recall query, bounded by recall_embed_timeout_seconds
        for network embedders (a Gemini round-trip measured ~0.5s — paid
        sequentially before the LLM can start). On timeout the worker keeps
        running and lands its result in the embedder's LRU cache, so an
        immediate retry of the same text is free. Local embedders (hashing,
        sentence-transformers) skip the machinery entirely: they are either
        instant or pay a one-time model load the timeout would misread."""
        timeout = float(
            getattr(self.config.memory, "recall_embed_timeout_seconds", 0.0) or 0.0
        )
        if timeout <= 0 or not getattr(self.embedder, "network", False):
            return self.embedder.embed_text(query)
        future = _EMBED_EXECUTOR.submit(self.embedder.embed_text, query)
        try:
            return future.result(timeout=timeout)
        except FuturesTimeout:
            raise _EmbedTimeout() from None

    def _check_embedding_dimension(self, query_dim: int) -> bool:
        """Detect an embedder/store dimension mismatch (once per instance).

        After switching embedders (e.g. hashing 512-d -> Gemini 768-d) the
        persisted vectors are incomparable: naive cosine over zipped values
        silently truncates and returns garbage scores. Refuse recall and point
        at the migration script instead of serving noise.
        """
        if self._dim_check_done:
            return self._dim_check_ok
        self._dim_check_done = True
        self._dim_check_ok = True
        try:
            with self._state_lock:
                first = next(iter(self.vector_store.dump()), None)
            if first is not None and first.vector and len(first.vector) != query_dim:
                self._dim_check_ok = False
                telemetry.incr("memory.dim_mismatch_recall_disabled")
                LOGGER.warning(
                    "Embedding dimension mismatch: store has %d-d vectors but "
                    "the configured embedder produces %d-d. Long-term recall is "
                    "disabled until the store is re-embedded — run "
                    "scripts/migrate_memory.py.",
                    len(first.vector),
                    query_dim,
                )
        except Exception:  # pragma: no cover - dump() backend quirks
            pass
        return self._dim_check_ok

    def speculate_recall(self, user_text: str, limit: int = 3) -> Optional[Future]:
        """Fire-and-forget: pre-start the per-turn recall for ``user_text``
        so its embed round trip overlaps the client's STT->chat round trip.
        Full mechanics (match rule, TTL, counters) live in
        :mod:`amadeus.memory.speculative`."""
        return speculative.speculate_recall(self, user_text, limit)

    def recall_detailed(
        self,
        query: str,
        limit: int = 3,
        *,
        include_episodes: bool = False,
        _from_speculative: bool = False,
    ) -> List[VectorMatch]:
        """Relevance x recency (x salience) ranked long-term recall.

        Strength is NOT a ranking factor — it only gates survival (a
        forgetting-curve filter under ``enable_forgetting``); see
        :func:`amadeus.memory.recall.recall_detailed`'s docstring for the
        full breakdown. Body lives in :mod:`amadeus.memory.recall`
        (line-budget extraction, same free-function-over-owner pattern as
        introspection) — this delegate keeps the public surface on the
        manager. ``_from_speculative`` marks the internal pre-started call
        submitted by :meth:`speculate_recall`, which must not consume its
        own pending stash entry (it would deadlock waiting on itself).
        """
        return recall_module.recall_detailed(
            self,
            query,
            limit,
            include_episodes=include_episodes,
            _from_speculative=_from_speculative,
        )

    def _reinforce(self, matches: Sequence[VectorMatch]) -> None:
        """Retrieval strengthens a memory (Ebbinghaus-style reinforcement)."""
        bump = float(
            getattr(self.config.memory, "retrieval_reinforcement", 0.15) or 0.0
        )
        if bump <= 0 or not self.vector_store:
            return
        with self._state_lock:
            for match in matches:
                metadata = match.metadata or {}
                # Legacy id-less/untyped records are never reinforced.
                if not match.id or not metadata.get("type"):
                    continue
                updated = dict(metadata)
                updated["strength"] = min(
                    2.0, _as_float(metadata.get("strength"), 1.0) + bump
                )
                self.vector_store.update_metadata(match.id, updated)
            self._vector_dirty = True

    @property
    def long_term_count(self) -> int:
        if not self.vector_store:
            return 0
        return self.vector_store.count()

    @property
    def last_tone(self) -> Optional[float]:
        """Latest conversational tone (-1..1) from the extraction job, if any."""
        return self._last_tone

    def record_mood(self, mood: str) -> None:
        """Stash Kurisu's current mood on the profile for cross-session carry.

        Deliberately does NOT save to disk — mood changes every few turns and
        is cosmetic; it rides along on the next existing profile save trigger
        (extraction job, session gap, forget). No-op without a profile.
        """
        if self.profile is None or not mood:
            return
        with self._state_lock:
            self.profile.mood = mood

    def should_greet(self) -> bool:
        """True when a session is fresh enough to open with a greeting.

        scope_context_to_session interaction: a brand-new conversation
        always starts with empty short_term (whether because the flag left
        it unrestored, or because there was never any persisted state), so
        this returns True immediately — every new conversation opens with a
        greeting, which is the intended behavior. A session id recreated
        over an existing transcript (WebChatSession.hydrate_short_term_from_
        transcript) instead re-populates short_term with a freshly-stamped
        tail, so this falls through to the gap check below and correctly
        does NOT greet — it's the same conversation continuing, not a new
        one.
        """
        if not getattr(self.config.memory, "enable_greeting", False):
            return False
        if not self.short_term:
            return True
        last_stamp = _parse_iso(self.short_term[-1].timestamp)
        if last_stamp is None:
            return True
        gap_minutes = float(
            getattr(self.config.memory, "session_gap_minutes", 30) or 30
        )
        return (dt.datetime.utcnow() - last_stamp).total_seconds() / 60 > gap_minutes

    def should_bid_farewell(self) -> bool:
        """True when a goodbye deserves a send-off line — the inverse of
        :meth:`should_greet`: only when there IS a conversation to close."""
        if not getattr(self.config.memory, "enable_farewell", False):
            return False
        return bool(self.short_term)

    def profile_block(self) -> Optional[str]:
        """Render the core-memory prompt block (frozen per session by callers)."""
        self.profile_block_dirty = False
        if not self.profile:
            return None
        max_facts = int(getattr(self.config.memory, "profile_max_facts", 12) or 12)
        with self._state_lock:
            return self.profile.render_block(max_facts)

    def remember(self, text: str, *, pinned: bool = False) -> bool:
        """User-directed "remember this": store a high-salience durable fact."""
        text = (text or "").strip()
        if not text:
            return False
        if not (
            self.config.memory.enable_long_term and self.embedder and self.vector_store
        ):
            return False
        return self._store_fact(text, salience=0.9, pinned=pinned)

    def find_facts(self, text: str, k: int = 3) -> List[Dict[str, Any]]:
        """Look up fact-type records relevant to ``text`` — the model-facing
        memory tool's lookup primitive. Delegates to
        :func:`amadeus.memory.fact_archive.find_facts` (line-budget
        extraction, same free-function-over-owner pattern as recall.py)."""
        return fact_archive.find_facts(self, text, k)

    def archive_fact(
        self, record_id: str, reason: str, *, allow_pinned: bool = False
    ) -> Optional[str]:
        """Archive-then-remove a fact (or any vector-store record) by id.
        Delegates to :func:`amadeus.memory.fact_archive.archive_fact`.
        ``allow_pinned`` is reserved for direct user instructions (explicit
        forget/correct) — automatic paths must leave it False."""
        return fact_archive.archive_fact(
            self, record_id, reason, allow_pinned=allow_pinned
        )

    def store_user_fact(
        self,
        text: str,
        salience: float = 0.9,
        previous_fact: Optional[str] = None,
        valid_at: Optional[str] = None,
    ) -> bool:
        """Explicit user-fact store path for the model-facing memory tool.
        Delegates to :func:`amadeus.memory.fact_archive.store_user_fact`."""
        return fact_archive.store_user_fact(
            self, text, salience, previous_fact, valid_at
        )

    def store_moment(
        self,
        text: str,
        salience: float = 0.6,
        source: str = "co_activity",
    ) -> bool:
        """Delegates to :func:`amadeus.memory.relational.store_moment`."""
        from . import relational

        return relational.store_moment(self, text, salience=salience, source=source)

    def count_moments_since(
        self, cutoff: dt.datetime, source: Optional[str] = None
    ) -> int:
        """Delegates to :func:`amadeus.memory.relational.count_moments_since`."""
        from . import relational

        return relational.count_moments_since(self, cutoff, source=source)

    @staticmethod
    def _is_protected(metadata: Optional[Dict[str, object]]) -> bool:
        """True when a vector-store record's metadata marks it pinned.

        Single source of truth for the pin check — used everywhere a pinned
        record must be treated as protected: recall ranking/reinforcement,
        forget/forget_id, and the supersede scan in _store_fact.
        """
        return bool((metadata or {}).get("pinned"))

    def _archive_and_delete_locked(
        self,
        record_id: Optional[str],
        text: str,
        metadata: Optional[Dict[str, object]],
        reason: str,
    ) -> None:
        """Archive a record (best-effort provenance) then remove it from the
        hot vector store. Caller MUST hold ``_state_lock`` and is responsible
        for ``vector_store.persist()`` afterward — this only mutates the
        in-memory store, matching every call site's existing shape.
        Delegates to :func:`amadeus.memory.fact_archive.archive_and_delete_locked`.
        """
        fact_archive.archive_and_delete_locked(self, record_id, text, metadata, reason)

    def forget(self, query: str) -> Union[str, "_Protected", None]:
        """User-directed "forget that": remove the best-matching memory.
        Delegates to :func:`amadeus.memory.editing.forget` (line-budget
        extraction; PROTECTED / floor / TOCTOU semantics unchanged)."""
        from . import editing

        return editing.forget(self, query)

    def _find_record_locked(self, record_id: str) -> Optional[VectorRecord]:
        """Locate a vector-store record by id. Caller MUST hold
        ``_state_lock`` — this only reads, it never mutates or persists, so
        callers can find-then-act atomically under one lock acquisition
        instead of racing a separate find against a later mutate (TOCTOU:
        another thread could pin/delete/edit the record in between)."""
        if not self.vector_store or not record_id:
            return None
        return self.vector_store.get(record_id)

    def set_pinned(self, record_id: str, pinned: bool) -> bool:
        """Delegates to :func:`amadeus.memory.editing.set_pinned`."""
        from . import editing

        return editing.set_pinned(self, record_id, pinned)

    def edit_memory(self, record_id: str, text: str) -> bool:
        """Delegates to :func:`amadeus.memory.editing.edit_memory`."""
        from . import editing

        return editing.edit_memory(self, record_id, text)

    def forget_id(self, record_id: str) -> Union[str, "_Protected", None]:
        """Delegates to :func:`amadeus.memory.editing.forget_id`
        (PROTECTED-on-pinned semantics unchanged)."""
        from . import editing

        return editing.forget_id(self, record_id)

    def forget_fact(self, key: str) -> bool:
        """Delegates to :func:`amadeus.memory.editing.forget_fact`."""
        from . import editing

        return editing.forget_fact(self, key)

    def reload_long_term_from_disk(self) -> None:
        """RAM-only refresh of the vector store + profile from disk, for
        SIBLING live managers after another manager over the same files
        performed a user-intent mutation (synapse-map delete/edit/pin,
        "remember this").

        The JSON vector store and the profile are rewrite-whole-file
        backends: a sibling that keeps stale records in RAM resurrects
        deleted memories (and drops newly added ones) on its own next
        persist — last-writer-wins, reproduced 2026-07-28. Disk is
        authoritative here because every mutation on these stores persists
        immediately; a reload therefore only discards state that another
        manager has already superseded. Sqlite and Chroma are per-record
        backends without the rewrite hazard, so only the in-memory JSON
        store is swapped. Never persists — the mutating manager already did.
        """
        with self._state_lock:
            if (
                isinstance(self.vector_store, InMemoryVectorStore)
                and self.embeddings_path
            ):
                self.vector_store = InMemoryVectorStore(path=self.embeddings_path)
            if self.profile is not None and self.profile_path:
                self.profile = UserProfile.load(self.profile_path)
                self.profile_block_dirty = True

    def forget_open_thread(self, text: str) -> bool:
        """Delegates to :func:`amadeus.memory.editing.forget_open_thread`."""
        from . import editing

        return editing.forget_open_thread(self, text)

    def reset_relationship(self) -> bool:
        """Delegates to :func:`amadeus.memory.editing.reset_relationship`
        (trust deliberately untouched)."""
        from . import editing

        return editing.reset_relationship(self)

    def memory_overview(self) -> Dict[str, Any]:
        """Everything Kurisu remembers, for "what do you remember about me?".

        Delegates to :func:`amadeus.memory.introspection.memory_overview`.
        """
        return introspection.memory_overview(self)

    def memory_graph(self) -> Dict[str, Any]:
        """Nodes+edges view of long-term memory for the UI "synapse map".

        Delegates to :func:`amadeus.memory.introspection.memory_graph`.
        """
        return introspection.memory_graph(self)

    def _update_graph(self, facts: List[tuple]) -> None:
        """Mine newly extracted facts into the entity-graph memory layer
        (``memory.graph_enabled``, default off). Delegates to
        :func:`amadeus.memory.graph.update_graph`; a no-op, and never
        raises, when the flag is off."""
        graph_module.update_graph(self, facts)

    def recall_from_graph(self, user_text: str, *, max_entities: int = 3) -> List[str]:
        """Entity-graph recall lines for entities mentioned in
        ``user_text`` (name/alias match against the persisted entity
        graph), current facts + one-hop relations only. Delegates to
        :func:`amadeus.memory.graph.recall_from_graph`; returns ``[]``
        when ``memory.graph_enabled`` is off, nothing matches, or the
        graph is empty/unreadable — never raises."""
        return graph_module.recall_from_graph(
            self, user_text, max_entities=max_entities
        )

    def flush(self, timeout: float = 5.0) -> None:
        """Wait for pending background memory jobs (mainly for tests/shutdown)."""
        if not self._summary_executor:
            return
        try:
            self._summary_executor.submit(lambda: None).result(timeout=timeout)
        except Exception:  # pragma: no cover - best effort barrier
            pass

    def finalize_session(self, wait: bool = False) -> None:
        """Consolidate the current session: episode summary + reflection."""
        memory = self.config.memory
        episodes_on = bool(getattr(memory, "enable_episodes", False))
        reflection_on = bool(getattr(memory, "enable_reflection", False))
        futures: List[Future] = []
        # Fact extraction rides the summary job, whose trigger needs the
        # short-term window to fill (summary_trigger_messages). A short
        # session can end before that ever happens — and per-turn recall is
        # facts-only, so without a final extraction pass here a durable fact
        # from a quick conversation ("I switched labs") would never become
        # recallable at all.
        if (
            getattr(memory, "enable_fact_extraction", False)
            and self._messages_since_summary >= 2
            and self.short_term
        ):
            if wait and self._summary_future and not self._summary_future.done():
                # Let the in-flight job land first so this final pass runs
                # directly instead of riding the requeue flag (whose re-submit
                # would race the shutdown of a quitting CLI process).
                try:
                    self._summary_future.result(timeout=15)
                except Exception:  # pragma: no cover - job errors already logged
                    pass
            self.summarize()
            if self._summary_future is not None:
                futures.append(self._summary_future)
        if episodes_on and self.short_term:
            future = self._submit_job(
                self._run_episode_job, list(self.short_term), self.summary
            )
            if future:
                futures.append(future)
        if reflection_on:
            future = self._submit_job(self._run_reflection_job)
            if future:
                futures.append(future)
        # Relational memory (memory/relational.py): a separate job from the
        # episode above, deliberately — the episode stays neutral
        # third-person to keep her persona from compounding, and this one is
        # first-person by design, so they must fail and be disabled
        # independently.
        if getattr(getattr(self.config, "relational_memory", None), "enabled", False):
            future = self._submit_job(self._run_relational_job, list(self.short_term))
            if future:
                futures.append(future)
        # Fact consolidation (memory/consolidation.py): cadence-gated inside.
        if getattr(memory, "enable_consolidation", False):
            future = self._submit_job(self._run_consolidation_job)
            if future:
                futures.append(future)
        with self._state_lock:
            if self._vector_dirty and self.vector_store:
                self.vector_store.persist()
                self._vector_dirty = False
        if wait:
            for future in futures:
                try:
                    future.result(timeout=15)
                except Exception as exc:  # pragma: no cover - job errors already logged
                    LOGGER.debug("Session finalization job failed: %s", exc)

    def _submit_job(self, fn, *args) -> Optional[Future]:
        if self._summary_executor:
            return self._summary_executor.submit(fn, *args)
        fn(*args)
        return None

    def _maybe_finalize_gap(self, now: dt.datetime) -> None:
        """Lazy session-end: a long silence means the last exchange was a
        separate session — memorialize it before starting the new one."""
        if not self.short_term:
            return
        last_stamp = _parse_iso(self.short_term[-1].timestamp)
        if last_stamp is None:
            return
        gap_minutes = float(
            getattr(self.config.memory, "session_gap_minutes", 30) or 30
        )
        if (now - last_stamp).total_seconds() / 60 <= gap_minutes:
            return
        LOGGER.debug(
            "Session gap detected; consolidating previous session as an episode."
        )
        if getattr(self.config.memory, "enable_episodes", False):
            self._submit_job(self._run_episode_job, list(self.short_term), self.summary)
        if self.profile is not None:
            with self._state_lock:
                self.profile.session_count += 1
                # In-process gap: the ending session's tone is available, so
                # this is the main channel for the warm/hostile trust nudge,
                # arc drift, and tone-history append.
                self._apply_session_boundary_state()
                self._save_profile()
            self.profile_block_dirty = True

    def _build_transcript(self, messages: Sequence[Message]) -> str:
        return "\n".join(f"{msg.role}: {msg.content}" for msg in messages)

    def _update_summary_truncate(self, messages: Sequence[Message]) -> None:
        transcript = self._build_transcript(messages)
        limit = max(self.summary_max_chars, 1)
        with self._state_lock:
            self.summary = transcript[:limit]
            self.summary_info = {
                "updated_at": dt.datetime.utcnow().isoformat(),
                "message_count": len(messages),
                "strategy": "truncate",
                "revision": int(self.summary_info.get("revision", 0)) + 1,
            }
        LOGGER.debug(
            "Updated memory summary via truncation (%d chars)", len(self.summary)
        )

    def _run_summary_job(
        self,
        messages: Sequence[Message],
        revision: int,
        previous_summary: Optional[str] = None,
        profile_snapshot: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Delegates to :func:`amadeus.memory.extraction_jobs.run_summary_job`."""
        return extraction_jobs.run_summary_job(
            self, messages, revision, previous_summary, profile_snapshot
        )

    def _run_extraction_job(
        self,
        transcript: str,
        revision: int,
        previous_summary: Optional[str],
        profile_snapshot: Optional[str],
        message_count: int,
    ) -> Dict[str, Any]:
        """Delegates to :func:`amadeus.memory.extraction_jobs.run_extraction_job`."""
        return extraction_jobs.run_extraction_job(
            self,
            transcript,
            revision,
            previous_summary,
            profile_snapshot,
            message_count,
        )

    def _on_summary_ready(self, future: Future) -> None:
        try:
            result = future.result()
        except Exception as exc:  # pragma: no cover - executor errors
            LOGGER.error("Summary job crashed: %s", exc)
            telemetry.incr("memory.summary_failures")
            # Snapshot + drain under the lock: this callback runs on the
            # executor thread while the request thread may be appending, and
            # iterating the live deque here is the silent-death race
            # add_message's locking exists to prevent.
            with self._state_lock:
                fallback_snapshot = list(self.short_term)
                # The truncate fallback folds in everything currently in
                # short_term, so the pending buffer is covered too (best
                # effort — matches this path's existing behavior of not
                # preserving exact per-message fidelity).
                self._unsummarized.clear()
            self._update_summary_truncate(fallback_snapshot)
            self._persist()
            self._summary_future = None
            return
        with self._state_lock:
            self.summary = result["summary"]
            self.summary_info = {
                "updated_at": dt.datetime.utcnow().isoformat(),
                "message_count": result.get("message_count", len(self.short_term)),
                "strategy": result.get("strategy", "truncate"),
                "revision": result.get(
                    "revision", int(self.summary_info.get("revision", 0))
                ),
            }
            if result.get("error"):
                self.summary_info["error"] = result["error"]
                telemetry.incr("memory.summary_failures")
            # Drain the unsummarized buffer only when this job actually
            # folded its messages into the summary text — a "retained"
            # strategy means the LLM call failed/returned empty and the
            # PREVIOUS summary was kept unchanged, so these messages are
            # still pending and must stay queued for the next attempt
            # (add_message already keeps appending to the same list).
            # Drain only what the job was SUBMITTED with: clear() also
            # swallowed messages appended while it ran, losing them from
            # summarization/extraction entirely.
            if self.summary_info["strategy"] != "retained":
                del self._unsummarized[: self._unsummarized_submitted]
        new_facts = result.get("facts") or []
        # Tolerant unpack: extraction emits 4-tuples (text, salience,
        # replaces, valid_at) since the bitemporal pass; older callers and
        # fixtures still pass 3-tuples.
        for fact in new_facts:
            text, salience, replaces = fact[0], fact[1], fact[2]
            valid_at = fact[3] if len(fact) > 3 else None
            self._store_fact(text, salience, replaces=replaces, valid_at=valid_at)
        if new_facts:
            # One prune pass after the whole batch (not per-fact): mirrors
            # prune_episodes_over_cap's cadence — cheap, and avoids evicting
            # a fact this same batch just stored ahead of one stored later.
            self._prune_facts_over_cap()
        # Entity-graph mining piggybacks on the same extraction output — no
        # extra LLM call (see graph.py's module docstring for why that's
        # enough); no-ops instantly when memory.graph_enabled is off.
        self._update_graph(new_facts)
        profile_updates = result.get("profile_updates")
        open_threads = result.get("open_threads")
        if self.profile is not None and (profile_updates or open_threads is not None):
            max_facts = int(getattr(self.config.memory, "profile_max_facts", 12) or 12)
            max_threads = int(getattr(self.config.memory, "max_open_threads", 3) or 3)
            max_age = int(
                getattr(self.config.memory, "open_thread_max_age_days", 21) or 0
            )
            with self._state_lock:
                if profile_updates:
                    self.profile.merge_updates(profile_updates, max_facts)
                if open_threads is not None:
                    self.profile.set_open_threads(open_threads, max_threads)
                self.profile.prune_stale_threads(max_age)
                self._save_profile()
        if result.get("tone") is not None:
            self._last_tone = float(result["tone"])
            # Persist the session's ending tone so the next session boundary
            # (usually a fresh process) can apply the tone-driven trust/arc
            # drift — the in-memory value dies with this process.
            if self.profile is not None:
                with self._state_lock:
                    self.profile.set_session_tone(self._last_tone)
                    self._save_profile()
        extraction_jobs.apply_relational_signals(self, result)
        LOGGER.debug(
            "Updated memory summary via %s (%d chars)",
            self.summary_info["strategy"],
            len(self.summary or ""),
        )
        # A mid-conversation extraction can add profile facts / open threads;
        # without dirtying the block the prompt keeps serving the stale
        # version until an unrelated dirty-setter fires.
        if self.profile is not None:
            self.profile_block_dirty = True
        self._persist()
        self._summary_future = None
        self._process_resummarize_queue()

    def _process_resummarize_queue(self) -> None:
        """Run a summarize() that was queued while a job was in flight.

        Queued-instead-of-dropped: without this, a summarize() request landing
        during a running job was silently discarded — including the
        session-final extraction pass, whose messages then never got
        summarized or fact-extracted at all.
        """
        if self._resummarize_queued:
            self._resummarize_queued = False
            if self.short_term and self._messages_since_summary:
                self.summarize()

    def _store_long_term(self, role: str, content: str) -> None:
        if not (self.embedder and self.vector_store):
            return
        vector = self.embedder.embed_text(content)
        with self._state_lock:
            if self.dedupe_threshold > 0 and self.vector_store.count():
                nearest = self.vector_store.similarity_search(vector, limit=1)
                if nearest and nearest[0].score >= self.dedupe_threshold:
                    LOGGER.debug(
                        "Skipping long-term memory append (score %.3f >= threshold %.3f)",
                        nearest[0].score,
                        self.dedupe_threshold,
                    )
                    return
            metadata = {"role": role, "timestamp": dt.datetime.utcnow().isoformat()}
            record = VectorRecord(vector=list(vector), text=content, metadata=metadata)
            self.vector_store.add(record)
            self.vector_store.persist()

    def _store_fact(
        self,
        text: str,
        salience: float,
        *,
        pinned: bool = False,
        replaces: Optional[str] = None,
        source: Optional[str] = None,
        record_type: str = "fact",
        valid_at: Optional[str] = None,
    ) -> bool:
        """Store an extracted fact, superseding a near-duplicate/contradiction
        (archive-not-delete; see fact_archive.py). Delegates to
        :func:`amadeus.memory.fact_archive.store_fact` (line-budget
        extraction, same free-function-over-owner pattern as recall.py).
        """
        return fact_archive.store_fact(
            self,
            text,
            salience,
            pinned=pinned,
            replaces=replaces,
            source=source,
            record_type=record_type,
            valid_at=valid_at,
        )

    def _run_episode_job(
        self, messages: Sequence[Message], summary: Optional[str]
    ) -> None:
        """Delegates to :func:`amadeus.memory.extraction_jobs.run_episode_job`."""
        extraction_jobs.run_episode_job(self, messages, summary)

    def _run_relational_job(self, messages: Sequence[Message]) -> None:
        """Delegates to :func:`amadeus.memory.relational.maybe_store_relational_moment`."""
        from . import relational

        relational.maybe_store_relational_moment(self, messages)

    def _run_consolidation_job(self) -> None:
        """Delegates to :func:`amadeus.memory.consolidation.maybe_consolidate_facts`."""
        from . import consolidation

        consolidation.maybe_consolidate_facts(self)

    def _supersede_near_duplicate_episode(self, vector: Sequence[float]) -> None:
        """Delegates to
        :func:`amadeus.memory.extraction_jobs.supersede_near_duplicate_episode`."""
        extraction_jobs.supersede_near_duplicate_episode(self, vector)

    def _prune_episodes_over_cap(self) -> None:
        """Delegates to
        :func:`amadeus.memory.extraction_jobs.prune_episodes_over_cap`."""
        extraction_jobs.prune_episodes_over_cap(self)

    def _prune_facts_over_cap(self) -> None:
        """Delegates to
        :func:`amadeus.memory.extraction_jobs.prune_facts_over_cap`."""
        extraction_jobs.prune_facts_over_cap(self)

    def _run_reflection_job(self) -> None:
        """Delegates to :func:`amadeus.memory.extraction_jobs.run_reflection_job`."""
        extraction_jobs.run_reflection_job(self)

    def persist_profile(self) -> None:
        """Public save hook for callers that mutate profile state directly
        (e.g. the greeting marking a thread as followed-up)."""
        self._save_profile()

    def _save_profile(self) -> None:
        if self.profile is None or self.profile_path is None:
            return
        try:
            self.profile.save_atomic(self.profile_path)
        except OSError as exc:
            # Degrade loudly: a profile that silently stops persisting means
            # trust/arc/facts quietly reset on the next process start.
            telemetry.incr("memory.profile_save_failed")
            LOGGER.warning("Failed to save user profile: %s", exc)

    def _vector_store_self_persists(self) -> bool:
        """True when the vector store writes its own persistence file/dir."""
        if self.vector_store is None:
            return False
        if isinstance(self.vector_store, ChromaVectorStore):
            return True
        return bool(getattr(self.vector_store, "path", None))

    def _state_store_path(self) -> Path:
        """The state store path for the configured backend: ``persist_path``
        unchanged for "json", the ``.db`` sibling for "sqlite" (see
        ``sqlite_store.db_path``) — same directory/stem either way, so
        per-user and per-character scoping (which act on those) is
        unaffected by the backend choice."""
        if self.storage_backend == "sqlite":
            from .sqlite_store import db_path

            return db_path(self.persist_path)
        return self.persist_path

    def _persist(self) -> None:
        if not self.persist_path:
            return
        with self._state_lock:
            # scope_context_to_session (see _load): short_term/summary are
            # still written here unconditionally even when the flag is on.
            # Skipping the write would need no new state (_load already does
            # all the gating), but keeping one unconditional persist shape is
            # simpler than branching this dict — and harmless, since _load
            # just never reads "messages"/"summary" back while the flag is
            # on. Long-term vectors and the profile persist the
            # same way regardless of this flag.
            serialized: Dict[str, Any] = {
                "messages": [dataclasses.asdict(msg) for msg in self.short_term],
                "summary": self.summary,
                "summary_metadata": self.summary_info,
            }
            # Long-term vectors normally live in the vector store's own file
            # (embeddings/vector path); duplicating them here made every message
            # rewrite every stored vector. Only inline them when the store has
            # nowhere else to persist.
            if self.vector_store and not self._vector_store_self_persists():
                long_term_serialized: List[Tuple[List[float], str]] = [
                    (list(record.vector), record.text)
                    for record in self.vector_store.dump()
                ]
                if long_term_serialized:
                    serialized["long_term"] = long_term_serialized
            if self.storage_backend == "sqlite":
                from .sqlite_store import save_state

                save_state(self._state_store_path(), serialized)
                return
            self.persist_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = self.persist_path.with_name(self.persist_path.name + ".tmp")
            with tmp_path.open("w", encoding="utf-8") as handle:
                json.dump(serialized, handle, indent=2)
            os.replace(tmp_path, self.persist_path)

    def _load(self) -> None:
        if not self.persist_path:
            return
        if self.storage_backend == "sqlite":
            from .sqlite_store import load_state

            data = load_state(self._state_store_path())
            if data is None:
                return
        else:
            try:
                with self.persist_path.open("r", encoding="utf-8") as handle:
                    data = json.load(handle)
            except (FileNotFoundError, json.JSONDecodeError):
                return
        # scope_context_to_session: short_term + the rolling summary are
        # CONVERSATION state, not user state — a new conversation must not
        # carry the previous one's tail into context, so skip restoring them
        # here. Long-term vectors and the profile are user
        # state and always restore below regardless of this flag.
        # _persist() deliberately still WRITES short_term/summary either way
        # (see its docstring note) — simpler than threading the flag through
        # every persist call site, and harmless: the file just holds a tail
        # nothing will ever read back while the flag stays on.
        if not getattr(self.config.memory, "scope_context_to_session", False):
            messages = [Message(**msg) for msg in data.get("messages", [])]
            # A persisted window ending on user message(s) is a failed or
            # crashed turn — every completed turn appends the assistant
            # reply right after, and rollback_user_message persists its pop.
            # Restoring the orphan would make the next turn answer a stale
            # question and its fresh-looking timestamp suppresses the
            # greeting gap check, so heal it here (also repairs stores
            # written before rollback persisted).
            while messages and messages[-1].role == "user":
                orphan = messages.pop()
                telemetry.incr("memory.orphaned_user_messages_healed")
                LOGGER.info(
                    "Dropped orphaned trailing user message from %s: %.60r",
                    self.persist_path,
                    orphan.content,
                )
            self.short_term.extend(messages)
            self.summary = data.get("summary")
            self.summary_info = data.get("summary_metadata", {}) or {}
        legacy_long_term = data.get("long_term", [])
        if legacy_long_term and self.vector_store and self.vector_store.is_empty():
            for vector, text in legacy_long_term:
                try:
                    vector_list = [float(x) for x in vector]
                except (TypeError, ValueError):
                    continue
                record = VectorRecord(
                    vector=vector_list,
                    text=str(text),
                    metadata={"legacy": True},
                )
                self.vector_store.add(record)
            self.vector_store.persist()

    @staticmethod
    def _unlink_store_file(path: Optional[Path]) -> None:
        """Remove ``path`` and, for a SQLite store, its ``-wal``/``-shm``
        checkpoint sidecars (harmless no-ops for a JSON path)."""
        if not path:
            return
        for candidate in (path, Path(f"{path}-wal"), Path(f"{path}-shm")):
            try:
                candidate.unlink()
            except FileNotFoundError:
                pass

    def clear(self, *, full_forget: bool = False) -> None:
        """Forget all tracked messages and delete persisted state.

        The user profile survives by default (wiping a conversation should not
        make Kurisu forget who you are); pass ``full_forget=True`` to erase it.
        """

        self.short_term.clear()
        self.summary = None
        self.summary_info = {}
        if self.persist_path:
            self._unlink_store_file(self._state_store_path())
        if self._summary_executor:
            self._summary_executor.shutdown(wait=False)
            self._summary_executor = None
            self._summary_future = None
        vector_store_path = (
            getattr(self.vector_store, "path", None)
            if self.storage_backend == "sqlite"
            else self.embeddings_path
        )
        if self.vector_store:
            self.vector_store.clear()
            if vector_store_path:
                self._unlink_store_file(Path(vector_store_path))
            if self.vector_path and self.vector_path.exists():
                shutil.rmtree(self.vector_path, ignore_errors=True)
        elif self.embeddings_path:
            self._unlink_store_file(self.embeddings_path)
        if full_forget:
            self.profile = UserProfile() if self.profile is not None else None
            if self.profile_path:
                try:
                    self.profile_path.unlink()
                except FileNotFoundError:
                    pass
