"""Calibrated recall: how sure she should SOUND about a recalled fact.

Split out of ``recall.py`` for its line budget (tests/test_file_budgets.py),
and because hedging is a genuinely separate concern from ranking: ranking
decides WHICH memories surface, this decides how confidently she states the
ones that did. A shaky memory that answers the question is still the right
thing to surface — it just gets hedged at the render seam
(``memory/context.py::_recall_line``).

The math itself lives in ``text_utils.py`` alongside the other pure
text/time helpers (``_effective_confidence`` and friends), since
``graph.py`` and the calibration harness both need it without importing the
recall path. This module is only the recall-side wiring.

Two hard rules, both pinned in tests/test_gotcha_contracts.py:

- **Transient by contract.** Confidence rides on the returned
  ``VectorMatch`` and is NEVER written back to record metadata. It is
  derived from ``strength``, which ``MemoryManager._reinforce`` owns and
  bumps on every retrieval; a stored copy would drift out of sync
  immediately.
- **Mechanical only.** Every input is something the store already observes
  (retrieval count, learn-time age, fact wording). No model grades its own
  extraction here — see ``scripts/calibrate_recall.py`` for the measurement
  that has to pass before an LLM-supplied term is stacked on top.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

from .. import telemetry
from .text_utils import (
    _confidence_clause,
    _effective_confidence,
    _volatility_class,
)
from .vector_store import VectorMatch

# Bottom-band edge: items below this actively invite correction, and a store
# where many recalls land here is the signal that the volatility half-lives
# are mis-tuned (or the store is genuinely stale). Kept beside the telemetry
# that reports it rather than inlined at the call site.
_LOW_CONFIDENCE_EDGE = 0.35


def annotate_confidence(
    match: VectorMatch,
    *,
    metadata: Dict[str, Any],
    strength: float,
    age_days: Optional[float],
    pinned: bool,
) -> None:
    """Attach effective confidence to ``match`` in place.

    ``pinned`` records get None: a pinned fact is a user-asserted one, so
    she states it plainly however old it is. Callers pass ``strength`` and
    ``age_days`` because the recall loop already computed both for the
    forgetting filter — recomputing them here would duplicate the parsing.
    """
    if pinned:
        match.confidence = None
        return
    match.confidence = _effective_confidence(
        strength=strength,
        age_days=age_days,
        volatility=_volatility_class(match.text, metadata.get("valid_at")),
    )


def record_confidence_telemetry(selected: Sequence[VectorMatch]) -> None:
    """Count how much of what surfaced this turn will be hedged.

    Degrade-loudly for calibration: ``memory.recall_hedged_low`` is the
    smoke alarm — a high count alongside "she keeps second-guessing
    herself" means the half-lives are too aggressive for this user's facts.
    """
    hedged = 0
    low = 0
    for match in selected:
        confidence = match.confidence
        if confidence is None:
            continue
        if _confidence_clause(confidence) is not None:
            hedged += 1
        if confidence < _LOW_CONFIDENCE_EDGE:
            low += 1
    if hedged:
        telemetry.incr("memory.recall_hedged", hedged)
    if low:
        telemetry.incr("memory.recall_hedged_low", low)
