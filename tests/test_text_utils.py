"""amadeus/memory/text_utils.py — the shared normalization/parsing helpers.

context.py and speculative.py must use the SAME gate implementation
(``_raw_query_too_short``) or the speculative-recall cache key drifts from
the real turn's gate (see the helper's docstring). Beyond behavior tests,
this file pins that sharing with identity/source checks so a future
copy-paste fork fails loudly.
"""

from __future__ import annotations

import datetime as dt
import inspect

import amadeus.memory.context as context
import amadeus.memory.speculative as speculative
import amadeus.memory.text_utils as text_utils
from amadeus.memory.text_utils import (
    _age_phrase,
    _already_discussed,
    _as_float,
    _clamp,
    _content_tokens,
    _parse_extraction_json,
    _parse_iso,
    _raw_query_too_short,
)

# ---------------------------------------------------------------------------
# Drift guards: one implementation, shared
# ---------------------------------------------------------------------------


def test_context_reexports_the_same_function_objects():
    assert context._raw_query_too_short is text_utils._raw_query_too_short
    assert context._already_discussed is text_utils._already_discussed
    assert context._age_phrase is text_utils._age_phrase


def test_speculative_imports_the_gate_and_defines_no_local_copy():
    # speculative.py imports _raw_query_too_short lazily inside a function,
    # so an identity check on module attributes can't see it — pin the
    # source instead: it must import from text_utils and never grow a
    # shadowing local definition.
    src = inspect.getsource(speculative)
    assert "from .text_utils import _raw_query_too_short" in src
    assert "def _raw_query_too_short" not in src


def test_context_defines_no_local_copy_of_shared_helpers():
    src = inspect.getsource(context)
    assert "def _raw_query_too_short" not in src
    assert "def _already_discussed" not in src
    assert "def _age_phrase" not in src


# ---------------------------------------------------------------------------
# _content_tokens
# ---------------------------------------------------------------------------


def test_content_tokens_empty_and_whitespace():
    assert _content_tokens("") == frozenset()
    assert _content_tokens("   \n\t ") == frozenset()


def test_content_tokens_lowercases_and_drops_short_and_filler():
    tokens = _content_tokens("User Lives in BERLIN and loves Quantum physics")
    # "user" is filler, "in"/"and" are under 4 chars.
    assert tokens == {"lives", "berlin", "loves", "quantum", "physics"}


def test_content_tokens_keeps_apostrophes_and_hyphens():
    tokens = _content_tokens("she's re-training the model")
    assert "she's" in tokens
    assert "re-training" in tokens


def test_content_tokens_japanese_yields_nothing():
    # The token regex is ASCII-only by design (facts are stored in English).
    assert _content_tokens("時間跳躍について話した") == frozenset()


def test_content_tokens_long_input_is_bounded_by_vocabulary():
    text = ("neuroscience dopamine cortex " * 500).strip()
    assert _content_tokens(text) == {"neuroscience", "dopamine", "cortex"}


# ---------------------------------------------------------------------------
# _already_discussed
# ---------------------------------------------------------------------------


def test_already_discussed_true_when_content_words_restated():
    fact = "User lives in Berlin and studies quantum physics"
    recent = "yeah, berlin is great — how goes the quantum physics? it lives on"
    assert _already_discussed(fact, recent, 0.6) is True


def test_already_discussed_false_for_unrelated_recent_text():
    fact = "User lives in Berlin and studies quantum physics"
    assert _already_discussed(fact, "we talked about dinner plans", 0.6) is False


def test_already_discussed_needs_three_distinctive_tokens():
    # Fewer than 3 content words: the ratio is meaningless, always False —
    # even at threshold 0 with a full lexical match.
    assert _already_discussed("User is fine", "fine fine fine", 0.0) is False


# ---------------------------------------------------------------------------
# _raw_query_too_short (the speculative-cache-critical gate)
# ---------------------------------------------------------------------------


def test_raw_query_gate_disabled_at_zero_min_chars():
    assert _raw_query_too_short("", 0) is False
    assert _raw_query_too_short("hi", 0) is False
    assert _raw_query_too_short("anything", -1) is False


def test_raw_query_gate_trims_whitespace_before_measuring():
    assert _raw_query_too_short("   hi   ", 12) is True
    assert _raw_query_too_short("        ", 12) is True
    assert _raw_query_too_short("", 12) is True


def test_raw_query_gate_boundary_is_strictly_less_than():
    assert _raw_query_too_short("a" * 11, 12) is True
    assert _raw_query_too_short("a" * 12, 12) is False


def test_raw_query_gate_counts_japanese_characters():
    assert _raw_query_too_short("こんにちは", 12) is True  # 5 chars
    assert _raw_query_too_short("タイムマシンの実験について教えて", 12) is False


def test_raw_query_gate_tolerates_none():
    assert _raw_query_too_short(None, 5) is True


# ---------------------------------------------------------------------------
# _age_phrase / _parse_iso
# ---------------------------------------------------------------------------

_NOW = dt.datetime(2026, 8, 1, 12, 0, 0)


def _stamp(days_ago: float) -> str:
    return (_NOW - dt.timedelta(days=days_ago)).isoformat()


def test_age_phrase_buckets():
    assert _age_phrase(_stamp(0.5), now=_NOW) == "earlier today"
    assert _age_phrase(_stamp(1.5), now=_NOW) == "yesterday"
    assert _age_phrase(_stamp(3), now=_NOW) == "3 days ago"
    assert _age_phrase(_stamp(10), now=_NOW) == "about a week ago"
    assert _age_phrase(_stamp(21), now=_NOW) == "about 3 weeks ago"
    assert _age_phrase(_stamp(14), now=_NOW) == "about 2 weeks ago"
    assert _age_phrase(_stamp(90), now=_NOW) == "about 3 months ago"
    assert _age_phrase(_stamp(400), now=_NOW) == "over a year ago"


def test_age_phrase_future_timestamp_is_none():
    assert _age_phrase(_stamp(-2), now=_NOW) is None


def test_age_phrase_handles_timezone_aware_stamps():
    aware = (_NOW.replace(tzinfo=dt.timezone.utc) - dt.timedelta(days=3)).isoformat()
    assert _age_phrase(aware, now=_NOW) == "3 days ago"


def test_age_phrase_garbage_inputs_are_none():
    assert _age_phrase(None, now=_NOW) is None
    assert _age_phrase("", now=_NOW) is None
    assert _age_phrase("not-a-date", now=_NOW) is None
    assert _age_phrase(12345, now=_NOW) is None  # non-string object


def test_parse_iso():
    assert _parse_iso("2026-08-01T12:00:00") == dt.datetime(2026, 8, 1, 12, 0, 0)
    assert _parse_iso("nope") is None
    assert _parse_iso(None) is None
    assert _parse_iso(1234) is None
    assert _parse_iso("") is None


# ---------------------------------------------------------------------------
# _as_float / _clamp
# ---------------------------------------------------------------------------


def test_as_float():
    assert _as_float("1.5", 0.0) == 1.5
    assert _as_float(2, 0.0) == 2.0
    assert _as_float(None, 3.3) == 3.3
    assert _as_float("abc", 3.3) == 3.3


def test_clamp():
    assert _clamp(0.5, 0.0, 1.0) == 0.5
    assert _clamp(-4.0, 0.0, 1.0) == 0.0
    assert _clamp(9.0, 0.0, 1.0) == 1.0


# ---------------------------------------------------------------------------
# _parse_extraction_json
# ---------------------------------------------------------------------------


def test_extraction_json_plain():
    assert _parse_extraction_json('{"facts": []}') == {"facts": []}


def test_extraction_json_fenced_with_language_tag():
    raw = '```json\n{"facts": ["a"]}\n```'
    assert _parse_extraction_json(raw) == {"facts": ["a"]}


def test_extraction_json_fenced_without_language_tag():
    raw = '```\n{"tone": "warm"}\n```'
    assert _parse_extraction_json(raw) == {"tone": "warm"}


def test_extraction_json_embedded_in_prose():
    raw = 'Sure! Here you go: {"facts": [], "tone": null} hope that helps'
    assert _parse_extraction_json(raw) == {"facts": [], "tone": None}


def test_extraction_json_rejects_non_dict_and_garbage():
    assert _parse_extraction_json("[1, 2, 3]") is None
    assert _parse_extraction_json("no json here") is None
    assert _parse_extraction_json("") is None
    assert _parse_extraction_json("{broken: }") is None
