"""Pure text/parsing helpers shared by the memory modules."""

from __future__ import annotations

import datetime as dt
import json
import re
from typing import Any, Dict, Optional

# Words too generic to signal that a fact's content was already discussed
# (the token regex already drops anything shorter than 4 characters). "user"
# is here because extracted facts are phrased "User <does something>".
_GENERIC_TOKENS = frozenset(
    "user that this with from have been were will would could should about "
    "into their they them then than when what where which your yours really "
    "very just like some more most other after before because while does "
    "doing going want wants also there here still".split()
)


def _content_tokens(text: str) -> frozenset:
    """Distinctive lowercase word tokens of a fact (4+ chars, no filler)."""
    return frozenset(
        token
        for token in re.findall(r"[a-z0-9][a-z0-9'-]{3,}", text.lower())
        if token not in _GENERIC_TOKENS
    )


def _already_discussed(fact_text: str, recent_text: str, threshold: float) -> bool:
    """True when enough of a fact's content words appear in recent chat.

    Lexical on purpose: recent replies restate facts nearly verbatim, and a
    substring scan costs nothing on the recall hot path (the alternative —
    embedding every recent message — is a network round-trip each).
    """
    tokens = _content_tokens(fact_text)
    if len(tokens) < 3:
        # Too few distinctive words for the ratio to mean anything.
        return False
    hits = sum(1 for token in tokens if token in recent_text)
    return hits / len(tokens) >= threshold


def _age_phrase(timestamp: object, now: Optional[dt.datetime] = None) -> Optional[str]:
    """Humanize a record timestamp into the rough phrase a person would use.

    Buckets on purpose — "about 3 weeks ago" is how memory feels; day-exact
    ages would push the model toward mechanical date recitals. (Lives here so
    both context.py and graph.py can render ages without an import cycle.)
    """
    days = _days_since(timestamp, now)
    if days is None or days < 0:
        return None
    return _past_days_phrase(days)


def _days_since(
    timestamp: object, now: Optional[dt.datetime] = None
) -> Optional[float]:
    """Fractional days between ``timestamp`` and ``now`` (negative = future),
    or None when unparseable. Aware stamps are normalized to naive UTC — the
    rest of the memory code does naive-UTC arithmetic throughout."""
    stamped = _parse_iso(timestamp if isinstance(timestamp, str) else None)
    if stamped is None:
        return None
    if stamped.tzinfo is not None:
        stamped = stamped.astimezone(dt.timezone.utc).replace(tzinfo=None)
    now = now or dt.datetime.utcnow()
    return (now - stamped).total_seconds() / 86400.0


def _past_days_phrase(days: float) -> str:
    if days < 1:
        return "earlier today"
    if days < 2:
        return "yesterday"
    if days < 7:
        return f"{int(days)} days ago"
    if days < 14:
        return "about a week ago"
    if days < 60:
        return f"about {max(2, round(days / 7))} weeks ago"
    if days < 365:
        return f"about {max(2, round(days / 30.4))} months ago"
    return "over a year ago"


def _future_days_phrase(days_ahead: float) -> str:
    if days_ahead < 1:
        return "later today"
    if days_ahead < 2:
        return "tomorrow"
    if days_ahead < 7:
        return f"in {int(days_ahead)} days"
    if days_ahead < 14:
        return "in about a week"
    if days_ahead < 60:
        return f"in about {max(2, round(days_ahead / 7))} weeks"
    if days_ahead < 365:
        return f"in about {max(2, round(days_ahead / 30.4))} months"
    return "over a year from now"


def _event_clause(valid_at: object, now: Optional[dt.datetime] = None) -> Optional[str]:
    """Humanized clause for a fact's EVENT time (``valid_at`` metadata) —
    the bitemporal counterpart of ``_age_phrase``, which speaks learn time.

    Unlike ``_age_phrase`` it also speaks future stamps ("happens in about
    2 weeks"): stored events can be upcoming ("user's wedding is in
    October"), and the future-guard that protects learn-time rendering from
    clock skew would silently eat exactly those facts' event times.
    """
    days = _days_since(valid_at, now)
    if days is None:
        return None
    if days >= 0:
        return f"happened {_past_days_phrase(days)}"
    return f"happens {_future_days_phrase(-days)}"


def _normalize_event_time(value: object) -> Optional[str]:
    """Validate/normalize a model- or user-supplied event time into the
    store's naive-UTC ISO format. None for anything unparseable — callers
    count the rejection; a bad ``when`` must never block storing the fact
    itself. Accepts date-only strings ("2026-08-15") and normalizes aware
    stamps to naive UTC so every downstream naive-arithmetic consumer stays
    TypeError-free.
    """
    stamped = _parse_iso(value if isinstance(value, str) else None)
    if stamped is None:
        return None
    if stamped.tzinfo is not None:
        stamped = stamped.astimezone(dt.timezone.utc).replace(tzinfo=None)
    return stamped.isoformat()


# --- Recall confidence (calibrated recall) -------------------------------
#
# How sure she should sound about a recalled fact. Deliberately MECHANICAL:
# every input below is something the store already observes (retrieval
# reinforcement, learn-time age, fact wording), never a model's self-report.
# An LLM-supplied confidence term was designed alongside this and left OUT
# until scripts/calibrate_recall.py shows it separates true from outdated
# facts on a real store — the emotion2vec+ lesson (most certain when wrong)
# applies directly to a model grading its own extractions.
#
# Volatility classes: age erodes different facts at different rates. Keyword
# matched on purpose, same spirit as _needs_query_augmentation — a real
# classifier is not worth an LLM call on the recall hot path. Half-lives are
# PROVISIONAL until the calibration pass sets them.
_VOLATILE_MARKERS = frozenset(
    "job jobs work works working role roles title employer company startup "
    "intern internship contract deadline deadlines due sprint milestone "
    "project projects paper thesis draft submission review revision "
    "apartment flat house moving moved move relocating relocated city "
    "currently now these lately recently temporarily semester rotation "
    "manager team lab supervisor advisor course class exam".split()
)

_PERMANENT_MARKERS = frozenset(
    "born birthday birthdate hometown nationality raised grew childhood "
    "graduated degree phd masters bachelor alma mother father parent "
    "brother sister sibling son daughter married wedding anniversary "
    "surname maiden name named".split()
)

# The store-supplied prior every fact starts from. A constant while the
# LLM-supplied confidence term stays unshipped: it makes effective
# confidence a pure function of retrieval count and age.
#
# 1.0, NOT 0.75. The multipliers below are both <= 1.0, so the base is the
# CEILING of the whole formula — a 0.75 base caps effective confidence at
# 0.75, puts the no-hedge band (>= 0.80) out of reach, and makes her hedge
# about a fact the user stated yesterday. That is the degenerate-heuristic
# failure the vocal-emotion pass was built to catch (16/17 clips read
# "bright" at identical confidence), so it is pinned in the tests: a fresh,
# once-stored fact MUST render with no clause at all.
_BASE_CONFIDENCE = 1.0

# Half-life in days for the staleness term, per volatility class. None means
# "no staleness term at all" — a birthday must never be hedged for age.
_VOLATILITY_HALF_LIVES: Dict[str, Optional[float]] = {
    "volatile": 90.0,
    "stable": 730.0,
    "permanent": None,
}


def _volatility_class(text: str, valid_at: object = None) -> str:
    """Classify how fast a fact's truth decays: volatile/stable/permanent.

    ``valid_at`` (bitemporal event time) is checked first: a fact pinned to
    an event that already happened is a historical record and cannot go
    stale, whatever its wording ("user's defense was on the 3rd").
    """
    if valid_at is not None:
        days = _days_since(valid_at)
        if days is not None and days >= 0:
            return "permanent"
    tokens = _content_tokens(text or "")
    if tokens & _PERMANENT_MARKERS:
        return "permanent"
    if tokens & _VOLATILE_MARKERS:
        return "volatile"
    return "stable"


def _corroboration(strength: float) -> float:
    """Retrieval reinforcement, normalized to [0, 1].

    ``strength`` starts at 1.0 and _reinforce bumps it +0.15 per retrieval
    (capped 2.0). Repeated retrieval is weak evidence that a fact is real
    and still in play, so it reads as corroboration here — the one place
    the forgetting-curve signal is reused for confidence. It is a BONUS
    only: see _effective_confidence's multiplier floor.
    """
    return _clamp((strength - 1.0) / 0.75, 0.0, 1.0)


def _staleness(age_days: Optional[float], volatility: str) -> float:
    """Decay in [0, 1] of a fact's truth given its age and volatility."""
    half_life = _VOLATILITY_HALF_LIVES.get(volatility, 730.0)
    if half_life is None or age_days is None or half_life <= 0:
        return 1.0
    return 0.5 ** (max(0.0, age_days) / half_life)


def _effective_confidence(
    *,
    strength: float,
    age_days: Optional[float],
    volatility: str,
    base: float = _BASE_CONFIDENCE,
) -> float:
    """How sure she should sound, in [0, 1].

    ``base`` is the store-supplied prior; with the LLM term deliberately
    unshipped it is the constant default for every fact, which makes the
    output a pure function of retrieval count and age. The multiplier
    SHAPES matter:

    - corroboration is a pure BONUS: it runs 1.0 -> 1.25, so a
      never-re-retrieved fact is never punished for it (most facts are
      stored once and never retrieved again — penalizing that would hedge
      almost everything). It only buys back confidence that age took.
    - staleness floors at 0.6, so an old fact becomes "I might be behind
      on this", never "I have no idea". Total erasure is what forgetting
      is for; this is tone.

    The clamp to [0, 1] is what makes the bonus behave: a fresh fact is
    already at the ceiling, so corroboration changes nothing until age has
    pulled it down.
    """
    corroborated = 1.0 + 0.25 * _corroboration(strength)
    fresh = 0.6 + 0.4 * _staleness(age_days, volatility)
    return _clamp(base * corroborated * fresh, 0.0, 1.0)


# Confidence bands, rendered as words rather than numbers: a model shown
# "0.62" either ignores it or says "sixty-two percent", and both are wrong.
# The top band emits NOTHING — the common case must cost zero prompt tokens
# and read exactly as it did before this feature. Edges and strings are
# pinned in tests/test_gotcha_contracts.py; the persona register is tuned
# against this wording, so drifting it silently changes her voice.
_CONFIDENCE_BANDS = (
    (0.80, None),
    (0.55, "you seem fairly sure of this"),
    (0.35, "you are not certain of this"),
    (0.0, "you may be misremembering this"),
)


def _confidence_clause(confidence: Optional[float]) -> Optional[str]:
    """Hedging clause for an effective confidence, or None to say it plainly."""
    if confidence is None:
        return None
    for edge, clause in _CONFIDENCE_BANDS:
        if confidence >= edge:
            return clause
    return None


def _parse_iso(value: Any) -> Optional[dt.datetime]:
    if not value or not isinstance(value, str):
        return None
    try:
        return dt.datetime.fromisoformat(value)
    except ValueError:
        return None


def _as_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _clamp(value: float, low: float, high: float) -> float:
    return min(high, max(low, value))


def _raw_query_too_short(raw_text: str, min_chars: int) -> bool:
    """True when ``raw_text`` (the user's actual turn, BEFORE any previous-
    turn augmentation) doesn't clear ``recall_min_query_chars``.

    The trivial-turn latency gate ("hi", "ok", "thanks") is meant to skip
    recall's embed round trip on turns that carry no searchable meaning of
    their own. Checking it against the augmented query instead (as
    recall_detailed's internal gate does, since that's the only text it
    ever sees) lets a 5-char turn that got the previous turn prepended slip
    past the gate it was designed for. Callers that augment (context.py,
    speculative.py) must call this against the RAW text first and skip
    recall entirely on a hit, before augmenting or embedding anything —
    kept here (rather than duplicated inline) specifically so the two call
    sites can't drift out of sync with each other.
    """
    if min_chars <= 0:
        return False
    return len((raw_text or "").strip()) < min_chars


def _parse_extraction_json(raw: str) -> Optional[Dict[str, Any]]:
    """Best-effort parse of the extraction JSON; None means unusable."""
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
        text = text.strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end <= start:
            return None
        try:
            data = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return None
    return data if isinstance(data, dict) else None


def _vader_tone(text: str) -> Optional[float]:
    """Local, free emotional-tone fallback (compound score in [-1, 1])."""
    try:
        from ..agents.emotion_analyzer import KurisuEmotionAnalyzer

        analyzer = KurisuEmotionAnalyzer()
        scores = analyzer.analyze(text).get("vader_scores")
        if isinstance(scores, dict) and "compound" in scores:
            return float(scores["compound"])
    except Exception:  # pragma: no cover - optional dependency path
        return None
    return None
