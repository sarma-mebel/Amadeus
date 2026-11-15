"""Fact consolidation: merge clusters of near-duplicate facts into one.

The store-time supersede scan (fact_archive.store_fact) only ever inspects
the top-3 neighbors of a NEW fact, so restatements that score just under
``fact_supersede_threshold`` (0.86) — or that arrive when a closer neighbor
shadows them — accumulate as separate records: "likes coffee", "drinks
coffee every morning", "mentioned needing coffee to start the day". Each
copy competes for recall slots and dilutes ranking without ever adding
information. This module is the periodic counterpart: at session end, at
most once per ``consolidation_min_days``, it clusters stored fact vectors
(no re-embedding — the vectors already exist), asks the cheap summary model
to merge each cluster into ONE sentence, and replaces the originals.

Design contract:

- ARCHIVE-NOT-DELETE: every merged-away original goes to the fact-archive
  sidecar with reason ``"consolidated"``, and the merged record carries
  ``consolidated_from`` (count) plus ``previous_fact`` provenance.
- CORROBORATION CARRIES: the merged record's ``strength`` is the cluster's
  max plus one retrieval-reinforcement bump per absorbed duplicate (capped
  at the same 2.0 ceiling ``_reinforce`` uses) — k restatements of a fact
  are k observations, and flattening them to 1.0 would make the calibrated
  recall layer hedge a well-corroborated fact.
- TIME STAYS HONEST: the merged record keeps the OLDEST original's
  ``timestamp``. A fresh stamp would reset the forgetting clock and the
  confidence staleness term for content the user may not have mentioned in
  months; the cost (slightly earlier cap-eviction eligibility) is the
  correct side of that trade.
- CONSERVATIVE ELIGIBILITY: only unpinned ``type: "fact"`` records with no
  ``source`` and no ``valid_at`` are candidates. Pinned is user intent;
  ``user_explicit`` preserves the user's own phrasing; ``relational``/
  ``co_activity`` moments are events, not restateable facts; a
  ``valid_at``-anchored fact records a dated event and merging two events
  fabricates history.
- SKIP-DEFAULT: the model may answer SKIP per cluster ("these are adjacent,
  not the same fact") and a hallucination guard rejects merged lines that
  introduce numbers absent from every input. The healthy telemetry shape is
  ``memory.consolidation_skipped``/``_rejected`` being nonzero — a merger
  that never declines is not exercising judgment.

Runs as a submitted job from ``MemoryManager.finalize_session`` on the
single-worker summary executor — the same seam and serialization guarantee
as extraction/relational jobs, and deliberately NOT a scheduler job: the
scheduler contract is disk snapshots and non-destructive reads, while this
must rewrite the hot store, which only the owning manager can do without
the JSON backend's last-writer-wins resurrection hazard.

Free functions taking the ``MemoryManager`` as first argument (the
relational.py / fact_archive.py convention).
"""

from __future__ import annotations

import datetime as dt
import json
import math
import re
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence

from .. import telemetry
from ..logging_utils import get_logger
from ..providers.llm.base import ChatMessage
from .fact_archive import archive_and_delete_locked
from .vector_store import VectorRecord

if TYPE_CHECKING:  # pragma: no cover - import cycle guard
    from .manager import MemoryManager
    from .vector_store import VectorRecord as _VR  # noqa: F401

LOGGER = get_logger(__name__)

# Cap on how many recent candidate facts enter the O(n^2) similarity scan.
# At 768 dims and 400 records the pure-Python scan measures ~2.2s on a
# development machine — acceptable only because it runs at most once per
# cadence window on the single-worker summary executor, off the turn path,
# and finalize_session bounds the job at its 15s future timeout. The newest
# facts are also where mergeable restatements live (older near-copies were
# the store-time scan's job).
_MAX_SCAN = 400


def state_path(manager: "MemoryManager") -> Optional[Path]:
    """Sidecar holding the last-run stamp, next to the vector store.

    Same derivation as ``fact_archive.archive_path`` so per-user and
    per-character scoping carries over automatically.
    """
    base = manager.embeddings_path or manager.vector_path or manager.persist_path
    if not base:
        return None
    base = Path(base)
    return base.with_name(f"{base.stem}_consolidation.json")


def maybe_consolidate_facts(manager: "MemoryManager") -> None:
    """Session-end entry point. Best-effort by contract: every failure
    counts a telemetry counter and returns, never raising into the
    session-end path."""
    try:
        _consolidate(manager)
    except Exception as exc:  # never break session finalization
        LOGGER.warning("Fact consolidation failed: %s", exc)
        telemetry.incr("memory.consolidation_failures")


def _consolidate(manager: "MemoryManager") -> None:
    cfg = manager.config.memory
    if not getattr(cfg, "enable_consolidation", False):
        return
    if not getattr(cfg, "enable_long_term", False):
        return
    if manager.vector_store is None or manager.embedder is None:
        # Enabled but cannot run — degrade loudly, not silently.
        telemetry.incr("memory.consolidation_unavailable")
        return

    # Cadence gate BEFORE any work: this runs at every session end, but a
    # store only drifts into mergeable shape over days, not sessions.
    path = state_path(manager)
    min_days = float(getattr(cfg, "consolidation_min_days", 7.0) or 7.0)
    last_run = _read_last_run(path)
    now = dt.datetime.utcnow()
    if last_run is not None and (now - last_run) < dt.timedelta(days=min_days):
        return

    provider = manager.summary_provider
    if provider is None:
        # Enabled but no cheap model to merge with — same loud degrade.
        telemetry.incr("memory.consolidation_unavailable")
        return

    with manager._state_lock:
        records = list(manager.vector_store.dump())
    candidates = [rec for rec in records if _eligible(manager, rec)]
    min_facts = int(getattr(cfg, "consolidation_min_facts", 12) or 12)
    if len(candidates) < min_facts:
        # Too small to bother; stamp so the scan itself stays rare.
        _write_last_run(path, now)
        return

    threshold = float(getattr(cfg, "consolidation_similarity", 0.83) or 0.83)
    max_cluster = max(2, int(getattr(cfg, "consolidation_max_cluster", 4) or 4))
    clusters = _find_clusters(candidates, threshold, max_cluster)

    max_merges = max(1, int(getattr(cfg, "consolidation_max_merges", 3) or 3))
    merged_any = False
    for cluster in clusters[:max_merges]:
        try:
            if _merge_cluster(manager, provider, cluster):
                merged_any = True
        except Exception as exc:  # provider/store runtime errors
            LOGGER.warning("Consolidation merge failed: %s", exc)
            telemetry.incr("memory.consolidation_failures")
            # Don't stamp: a provider outage should retry at the next
            # session end rather than waiting out the full cadence. But DO
            # persist any cluster that already merged — on inline-vector
            # configs vector_store.persist() is a no-op and the hot-store
            # rewrite lives only in RAM until manager._persist(); the
            # archive lines are already on disk, so bailing without it
            # would resurrect originals the archive also holds.
            if merged_any:
                manager._persist()
            return
    _write_last_run(path, now)
    if merged_any:
        manager._persist()


def _eligible(manager: "MemoryManager", record: "VectorRecord") -> bool:
    metadata = record.metadata or {}
    if metadata.get("type") != "fact":
        return False
    if manager._is_protected(metadata):
        return False
    if metadata.get("source"):
        return False
    if metadata.get("valid_at"):
        return False
    return bool(record.id) and bool((record.text or "").strip())


def _find_clusters(
    candidates: Sequence["VectorRecord"], threshold: float, max_cluster: int
) -> List[List["VectorRecord"]]:
    """Greedy similarity clustering over the stored vectors.

    Pairs above ``threshold`` are taken in similarity order; a record joins
    at most one cluster and clusters stop growing at ``max_cluster``.
    Returns clusters sorted by their strongest internal similarity, so the
    most obvious duplicates spend the merge budget first. Pure function of
    the stored vectors — no embedding calls.
    """
    recent = sorted(
        candidates,
        key=lambda rec: str((rec.metadata or {}).get("timestamp") or ""),
        reverse=True,
    )[:_MAX_SCAN]
    n = len(recent)
    if n < 2:
        return []
    norms: List[float] = []
    for rec in recent:
        norms.append(math.sqrt(sum(v * v for v in rec.vector)) or 1.0)
    pairs: List[tuple] = []
    for i in range(n):
        vec_i = recent[i].vector
        for j in range(i + 1, n):
            vec_j = recent[j].vector
            if len(vec_i) != len(vec_j):
                continue
            dot = sum(a * b for a, b in zip(vec_i, vec_j))
            score = dot / (norms[i] * norms[j])
            if score >= threshold:
                pairs.append((score, i, j))
    pairs.sort(reverse=True)
    assigned: Dict[int, int] = {}
    clusters: List[List[int]] = []
    best_score: List[float] = []
    for score, i, j in pairs:
        ci, cj = assigned.get(i), assigned.get(j)
        if ci is None and cj is None:
            assigned[i] = assigned[j] = len(clusters)
            clusters.append([i, j])
            best_score.append(score)
        elif ci is not None and cj is None:
            if len(clusters[ci]) < max_cluster:
                assigned[j] = ci
                clusters[ci].append(j)
        elif cj is not None and ci is None:
            if len(clusters[cj]) < max_cluster:
                assigned[i] = cj
                clusters[cj].append(i)
        # both already clustered: leave them; merging clusters would let
        # chains of pairwise-similar facts drag unrelated ends together.
    ordered = sorted(range(len(clusters)), key=lambda k: best_score[k], reverse=True)
    return [[recent[idx] for idx in clusters[k]] for k in ordered]


def _merge_cluster(
    manager: "MemoryManager", provider, cluster: Sequence["VectorRecord"]
) -> bool:
    """Merge one cluster: one cheap-model call, guards, archive + rewrite.

    Returns True when a merged record was stored, False on SKIP/rejection.
    Raises provider/store errors to the caller (which counts and aborts).
    """
    cfg = manager.config.memory
    # Oldest first, so "prefer the newest wording" is expressible in the
    # prompt and the oldest timestamp is simply the first record's.
    ordered = sorted(
        cluster, key=lambda rec: str((rec.metadata or {}).get("timestamp") or "")
    )
    texts = [rec.text for rec in ordered]
    prompt = _merge_prompt(manager, texts)
    max_tokens = int(getattr(cfg, "consolidation_max_tokens", 160) or 160)
    raw = provider.generate(prompt, max_tokens=max_tokens) or ""
    telemetry.record_llm_usage(provider, "consolidation")

    line = _clean_merged_line(raw)
    if line is None:
        telemetry.incr("memory.consolidation_skipped")
        return False
    if _introduces_numbers(line, texts):
        # A merged fact must not contain a number none of the inputs had —
        # the classic shape of a confabulated detail.
        telemetry.incr("memory.consolidation_rejected")
        return False
    max_chars = int(getattr(cfg, "fact_max_chars", 0) or 0)
    if max_chars and len(line) > max_chars:
        line = line[:max_chars]

    vector = manager.embedder.embed_text(line)
    with manager._state_lock:
        alive = [
            rec
            for rec in ordered
            if rec.id and manager.vector_store.get(rec.id) is not None
        ]
        if len(alive) < 2:
            # A concurrent supersede already thinned the cluster; nothing
            # left worth rewriting.
            return False
        oldest_meta = alive[0].metadata or {}
        strengths = [
            _as_float((rec.metadata or {}).get("strength"), 1.0) for rec in alive
        ]
        saliences = [
            _as_float((rec.metadata or {}).get("salience"), 0.5) for rec in alive
        ]
        bump = float(getattr(cfg, "retrieval_reinforcement", 0.15) or 0.15)
        metadata: Dict[str, object] = {
            "type": "fact",
            "timestamp": str(oldest_meta.get("timestamp") or now_iso()),
            "salience": max(saliences),
            "strength": min(2.0, max(strengths) + bump * (len(alive) - 1)),
            "pinned": False,
            "consolidated_from": len(alive),
            "supersedes": alive[0].id,
            "previous_fact": "; ".join(rec.text for rec in alive)[:200],
        }
        for rec in alive:
            archive_and_delete_locked(
                manager, rec.id, rec.text, rec.metadata, "consolidated"
            )
        manager.vector_store.add(
            VectorRecord(vector=list(vector), text=line, metadata=metadata)
        )
        manager.vector_store.persist()
        manager._vector_dirty = False
    telemetry.incr("memory.consolidation_merged")
    telemetry.incr("memory.consolidation_absorbed", len(alive) - 1)
    return True


def _merge_prompt(manager: "MemoryManager", texts: Sequence[str]) -> List[ChatMessage]:
    """The SKIP-default merge prompt. Split out so the hard rules are
    assertable in tests without running a provider."""
    name = getattr(manager, "_char_name", None) or "Kurisu"
    numbered = "\n".join(f"{i + 1}. {text}" for i, text in enumerate(texts))
    return [
        ChatMessage(
            role="system",
            content=(
                f"You maintain {name}'s long-term memory notes about the "
                "person she talks to. The notes below appear to record the "
                "SAME underlying fact, stored oldest first.\n\n"
                "If they truly are the same fact, reply with ONE sentence "
                "that merges them: keep every concrete detail any note "
                "contains, prefer the newest wording where they conflict, "
                "and invent NOTHING that no note states. No preamble, no "
                "quotes, no list — just the sentence.\n\n"
                "If they are adjacent but genuinely DIFFERENT facts (two "
                "hobbies, two people, an event vs a preference), reply with "
                "exactly: SKIP"
            ),
        ),
        ChatMessage(role="user", content=numbered),
    ]


def _clean_merged_line(raw: str) -> Optional[str]:
    """First line, de-quoted. None for empty or SKIP. Pure function."""
    text = (raw or "").strip()
    if not text:
        return None
    text = text.splitlines()[0].strip().strip('"').strip("'").strip()
    if not text:
        return None
    if text.upper().rstrip(".").startswith("SKIP"):
        return None
    return text


def _introduces_numbers(line: str, texts: Sequence[str]) -> bool:
    """True when ``line`` contains a digit-run absent from every input."""
    seen = set()
    for text in texts:
        seen.update(re.findall(r"\d+", text))
    return any(token not in seen for token in re.findall(r"\d+", line))


def _read_last_run(path: Optional[Path]) -> Optional[dt.datetime]:
    if path is None or not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        stamp = data.get("last_run")
        return dt.datetime.fromisoformat(stamp) if stamp else None
    except (OSError, ValueError, AttributeError) as exc:
        LOGGER.warning("Consolidation state unreadable at %s: %s", path, exc)
        telemetry.incr("memory.consolidation_state_load_failed")
        return None


def _write_last_run(path: Optional[Path], now: dt.datetime) -> None:
    if path is None:
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps({"last_run": now.isoformat()}), encoding="utf-8")
        tmp.replace(path)
    except OSError as exc:  # best-effort: worst case is an early re-run
        LOGGER.warning("Consolidation state write failed at %s: %s", path, exc)
        telemetry.incr("memory.consolidation_state_write_failed")


def now_iso() -> str:
    return dt.datetime.utcnow().isoformat()


def _as_float(value, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
