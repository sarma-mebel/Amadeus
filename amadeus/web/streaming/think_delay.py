"""Variable "think delay" before the first sentence/audio of a streamed turn.

The rest of the streaming pipeline optimizes toward minimum time-to-first-audio,
which makes latency robotically constant regardless of how heavy the question
is. A real person takes a beat on a hard or emotional prompt and snaps back on a
quip. This module computes a small, bounded pause to inject before the first
frame, sampled from a distribution keyed off signals that ALREADY exist upstream
(the length-analysis hint, the user-input-length bucket, and the first
sentence's emotion). The computation is a pure function so it is deterministic
and unit-testable; the actual ``asyncio.sleep`` lives in the turn orchestrator.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Optional

from ...persona import _user_len_bucket, kurisu_analysis_hint

__all__ = ["ThinkDelaySignals", "signals_for_turn", "compute_think_delay"]


# Base "weight" each signal contributes to the pre-jitter delay fraction, in
# seconds. These are intentionally small; the final value is clamped into the
# configured [min, max] window, so the ceiling is the real cap. Tuned so a bare
# quip lands near min and a long emotional analytical turn approaches max.
_LEN_BUCKET_WEIGHT = {
    "empty": 0.0,
    "short": 0.05,
    "medium": 0.35,
    "long": 0.6,
}

# The analysis hint is a proxy for "this needs actual thought": any hint OTHER
# than the generic fallback (the causal / procedural / evidentiary hints) earns
# a beat. We identify the fallback by asking the persona helper for the
# empty-input hint, so this stays in sync if its wording changes.
_GENERIC_HINT = kurisu_analysis_hint("")

# Emotionally loaded first sentences earn a beat of hesitation; neutral/quick
# affect earns none. Keyed off the tagged-grammar emotion label.
_EMOTION_WEIGHT = {
    "sad": 0.45,
    "angry": 0.3,
    "surprised": 0.2,
    "happy": 0.05,
    "relaxed": 0.0,
    "neutral": 0.0,
}


@dataclass(frozen=True)
class ThinkDelaySignals:
    """The already-existing signals the delay is keyed off.

    ``emotion`` is optional because it only becomes known once the first
    sentence is parsed; the input-derived signals are known up front.
    """

    len_bucket: str = "empty"
    heavy_hint: bool = False
    emotion: Optional[str] = None


def signals_for_turn(
    user_text: Optional[str], emotion: Optional[str] = None
) -> ThinkDelaySignals:
    """Derive the delay signals from the user's message (and, when known, the
    first sentence emotion), reusing the persona helpers so we key off the same
    length/analysis signals the prompt does."""
    hint = kurisu_analysis_hint(user_text or "")
    heavy = hint != _GENERIC_HINT
    return ThinkDelaySignals(
        len_bucket=_user_len_bucket(user_text),
        heavy_hint=heavy,
        emotion=(emotion or None),
    )


def compute_think_delay(
    signals: ThinkDelaySignals,
    min_seconds: float,
    max_seconds: float,
    rng: Optional[random.Random] = None,
) -> float:
    """Pure: map the turn's signals to a bounded think-delay in seconds.

    The weights for length, "heavy" analysis, and emotion sum into a base
    fraction of the [min, max] span, a little multiplicative jitter is applied
    so identical prompts don't produce a metronomic identical pause, and the
    result is clamped into [min, max]. Inject ``rng`` for deterministic tests.

    ``min >= max`` collapses to a fixed delay of ``max`` (a disabled/pinned
    range). Non-finite or negative bounds degrade to 0.0 — never raises.
    """
    try:
        lo = float(min_seconds)
        hi = float(max_seconds)
    except (TypeError, ValueError):
        return 0.0
    if not (lo == lo and hi == hi):  # NaN guard
        return 0.0
    lo = max(0.0, lo)
    hi = max(0.0, hi)
    if hi <= 0.0:
        return 0.0
    if lo >= hi:
        return hi

    rng = rng or random.Random()

    weight = _LEN_BUCKET_WEIGHT.get(signals.len_bucket, 0.0)
    if signals.heavy_hint:
        weight += 0.3
    if signals.emotion:
        weight += _EMOTION_WEIGHT.get(signals.emotion.strip().lower(), 0.0)

    # Multiplicative jitter in [0.8, 1.2] so the beat feels organic without
    # letting a low-weight quip balloon into a long pause.
    weight *= 0.8 + 0.4 * rng.random()

    # Map the (unbounded-above) weight into the span, then clamp. weight is
    # already scaled so the interesting range is roughly [0, ~1].
    frac = min(1.0, max(0.0, weight))
    delay = lo + frac * (hi - lo)
    return min(hi, max(lo, delay))
