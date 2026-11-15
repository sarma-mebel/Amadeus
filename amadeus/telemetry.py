"""Process-wide runtime health counters.

The unit tests catch regressions you introduce in code; these counters catch
degradations that only happen in live use — a summary model silently failing
every call, recall coming back empty for a week, the tagged-format repair pass
firing on every tool turn. They are aggregated across all sessions/users,
reset on process restart, and exposed read-only via ``/api/status`` under
``counters`` so the UI (or a curl) can answer "has anything been quietly
failing?" without grepping logs.

Deliberately minimal: a lock + dict, no deps, no persistence, no metrics
server. ``incr`` for monotonic counts, ``set_gauge`` for last-value readings
(gauges are prefixed with ``last_`` by convention).

``record_duration``/``duration_snapshot`` below are a second, parallel
facility for the same reason: counters answer "how often", durations answer
"how long". Pre-existing call sites (``web/streaming/turn.py``'s
``_record_turn_latency``, ``tts_pipeline.py``'s ``_record_latency``, the STT
endpoint) already hand-roll sum/count/last triples on top of ``incr``/
``set_gauge`` under the ``latency.*`` prefix — that shape stays exactly as-is
(tests pin it) rather than being migrated, so as not to change behavior for a
plumbing refactor. New instrumentation should prefer ``record_duration``.

``record_value``/``value_snapshot`` are a third facility, structurally
identical to ``record_duration``/``duration_snapshot`` (count/total/max/last
plus a bounded ring for p50/p95) but for non-duration numeric measurements —
e.g. ``llm.prompt_est_tokens``, the estimated per-turn prompt size from
``amadeus.token_budget``.
"""

from __future__ import annotations

import math
import threading
from collections import deque
from typing import Deque, Dict, Union

_LOCK = threading.Lock()
_VALUES: Dict[str, Union[int, float]] = {}

# One entry per duration key: {"count": int, "total_ms": float, "max_ms":
# float, "last_ms": float}. Percentiles come from a BOUNDED window, not a
# full history: each key also keeps a fixed-size ring of the most recent
# ``_SAMPLE_WINDOW`` samples (``deque(maxlen=...)`` — O(1) append, zero
# allocation growth in a long-lived process), and ``duration_snapshot``
# computes p50/p95 over that window on read. Recording stays O(1); the sort
# only happens on snapshot reads (/api/status polls), which are rare and
# tiny (≤64 floats per key). So: count/total/last answer "is this seam slow
# right now", max_ms answers "was it EVER slow" (all-time, outlives the
# window), and p50_ms/p95_ms answer "how slow is it typically/at the tail
# lately".
_SAMPLE_WINDOW = 64
_DURATIONS: Dict[str, Dict[str, float]] = {}
_SAMPLES: Dict[str, Deque[float]] = {}

# Same shape as _DURATIONS/_SAMPLES but for arbitrary numeric measurements
# that aren't a duration (e.g. estimated prompt token counts) — kept as a
# separate pair of dicts rather than folded into the duration ones so the
# two facilities' keys can never collide and duration-specific naming
# ("_ms") doesn't leak onto non-duration values.
_VALUE_STATS: Dict[str, Dict[str, float]] = {}
_VALUE_SAMPLES: Dict[str, Deque[float]] = {}


def incr(name: str, amount: int = 1) -> None:
    """Add to a monotonic counter (creates it at 0)."""
    with _LOCK:
        _VALUES[name] = _VALUES.get(name, 0) + amount


def set_gauge(name: str, value: Union[int, float]) -> None:
    """Record a last-value reading (overwrites)."""
    with _LOCK:
        _VALUES[name] = value


def snapshot() -> Dict[str, Union[int, float]]:
    """Copy of all counters/gauges, for /api/status."""
    with _LOCK:
        return dict(_VALUES)


def record_duration(key: str, ms: float) -> None:
    """Accumulate one timing sample under ``key``.

    Cheap enough to call unconditionally on every turn: O(1) — a running
    count/total/max/last update plus one append into a fixed-size ring
    (``deque(maxlen=_SAMPLE_WINDOW)``, which evicts the oldest sample for
    free; no sorting, no growth). ``total_ms / count`` gives the mean for
    consumers; ``max_ms`` catches the rare disaster a mean would hide,
    ``last_ms`` shows whether it's happening right now; percentiles over the
    ring are computed lazily in ``duration_snapshot``. Never raises on bad
    input (a negative or NaN ``ms`` is still recorded as given — the
    caller's clock, not this function, is the source of truth).
    """
    with _LOCK:
        entry = _DURATIONS.setdefault(
            key, {"count": 0, "total_ms": 0.0, "max_ms": 0.0, "last_ms": 0.0}
        )
        entry["count"] += 1
        entry["total_ms"] += ms
        entry["max_ms"] = max(entry["max_ms"], ms)
        entry["last_ms"] = ms
        _SAMPLES.setdefault(key, deque(maxlen=_SAMPLE_WINDOW)).append(ms)


def _percentile(ordered: list, q: float) -> float:
    """Nearest-rank percentile of an already-sorted non-empty list."""
    rank = max(1, math.ceil(q * len(ordered)))
    return float(ordered[min(rank, len(ordered)) - 1])


def duration_snapshot() -> Dict[str, Dict[str, float]]:
    """Copy of all recorded durations, for /api/status's ``durations`` key.

    Each entry carries ``p50_ms``/``p95_ms`` — nearest-rank percentiles over
    the last ``_SAMPLE_WINDOW`` samples for that key (a recent-window view,
    unlike the all-time ``max_ms``). The sort happens here, on read, so the
    per-turn recording path never pays for it.
    """
    with _LOCK:
        snap = {}
        for key, value in _DURATIONS.items():
            entry = dict(value)
            window = _SAMPLES.get(key)
            if window:
                ordered = sorted(window)
                entry["p50_ms"] = _percentile(ordered, 0.50)
                entry["p95_ms"] = _percentile(ordered, 0.95)
            snap[key] = entry
        return snap


def record_value(key: str, value: float) -> None:
    """Accumulate one arbitrary numeric sample under ``key``.

    Mirrors ``record_duration`` exactly (count/total/max/last plus a
    bounded 64-sample ring for percentiles) but for measurements that
    aren't a duration — e.g. ``llm.prompt_est_tokens``. Kept as a distinct
    facility rather than reusing ``record_duration`` under the hood: the
    duration functions' docstrings and field names (``*_ms``) are
    duration-specific, and contorting them to also mean "generic value"
    would make both harder to read. Cheap enough to call unconditionally
    every turn — same O(1) shape as ``record_duration``.
    """
    with _LOCK:
        entry = _VALUE_STATS.setdefault(
            key, {"count": 0, "total": 0.0, "max": 0.0, "last": 0.0}
        )
        entry["count"] += 1
        entry["total"] += value
        entry["max"] = max(entry["max"], value)
        entry["last"] = value
        _VALUE_SAMPLES.setdefault(key, deque(maxlen=_SAMPLE_WINDOW)).append(value)


def value_snapshot() -> Dict[str, Dict[str, float]]:
    """Copy of all recorded values, mirroring ``duration_snapshot``.

    Each entry carries ``p50``/``p95`` — nearest-rank percentiles over the
    last ``_SAMPLE_WINDOW`` samples for that key.
    """
    with _LOCK:
        snap = {}
        for key, value in _VALUE_STATS.items():
            entry = dict(value)
            window = _VALUE_SAMPLES.get(key)
            if window:
                ordered = sorted(window)
                entry["p50"] = _percentile(ordered, 0.50)
                entry["p95"] = _percentile(ordered, 0.95)
            snap[key] = entry
        return snap


def reset() -> None:
    """Clear everything (tests only)."""
    with _LOCK:
        _VALUES.clear()
        _DURATIONS.clear()
        _SAMPLES.clear()
        _VALUE_STATS.clear()
        _VALUE_SAMPLES.clear()


def record_llm_usage(provider: object, purpose: str) -> None:
    """Accumulate one LLM call's token usage under ``tokens.{purpose}.*``.

    Reads the additive ``last_usage`` attribute providers populate after a
    completed call ({"prompt": int, "completion": int}). Dummy/local
    providers — and calls whose backend reported no usage metadata — leave
    it None, and then nothing is recorded. Best-effort by contract:
    instrumentation must never be able to break a turn, so this swallows
    everything (mirroring ``_record_turn_latency`` in web/streaming/turn.py).
    """
    try:
        usage = getattr(provider, "last_usage", None)
        if not usage:
            return
        incr(f"tokens.{purpose}.prompt", int(usage.get("prompt") or 0))
        incr(f"tokens.{purpose}.completion", int(usage.get("completion") or 0))
        incr(f"tokens.{purpose}.calls")
    except Exception:  # pragma: no cover - instrumentation must never raise
        pass
