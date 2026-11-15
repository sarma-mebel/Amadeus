"""amadeus/utils/simple_yaml.py — the PyYAML-free fallback parser.

It is the config reader whenever PyYAML is missing (config.py,
characters.py, desktop/runtime.py, settings_api.py all fall back to it),
so a silent parsing bug here changes user configs without a traceback.
These tests pin the supported grammar (scalar coercions, nested maps via
indentation, comments/blank lines, quoted values) plus how the known
UNSUPPORTED constructs degrade (lists, inline comments, empty scalars) so
any behavior change is loud.
"""

from __future__ import annotations

import io
from pathlib import Path

import pytest

from amadeus.utils.simple_yaml import safe_dump, safe_load

# ---------------------------------------------------------------------------
# Scalar coercions
# ---------------------------------------------------------------------------


def test_scalar_types_coerce():
    text = "\n".join(
        [
            "count: 3",
            "ratio: 0.75",
            "on_flag: true",
            "on_word: yes",
            "off_flag: false",
            "off_word: no",
            "nothing: null",
            "also_nothing: none",
            "name: Kurisu",
        ]
    )
    data = safe_load(text)
    assert data == {
        "count": 3,
        "ratio": 0.75,
        "on_flag": True,
        "on_word": True,
        "off_flag": False,
        "off_word": False,
        "nothing": None,
        "also_nothing": None,
        "name": "Kurisu",
    }
    assert isinstance(data["count"], int)
    assert isinstance(data["ratio"], float)


def test_bool_and_null_are_case_insensitive():
    data = safe_load("a: True\nb: FALSE\nc: Null\nd: YES")
    assert data == {"a": True, "b": False, "c": None, "d": True}


def test_version_like_string_stays_a_string():
    # "1.2.3" contains a dot so the parser tries float() first — it must
    # fall through to a string, not crash or truncate.
    assert safe_load("version: 1.2.3") == {"version": "1.2.3"}


def test_quoted_value_with_colon_keeps_text_after_first_colon():
    # partition(":") splits on the FIRST colon only; surrounding double
    # quotes are stripped by the scalar parser.
    assert safe_load('title: "Steins;Gate: Zero"') == {"title": "Steins;Gate: Zero"}


def test_unquoted_value_with_colon_survives():
    assert safe_load("url: http://localhost:8000") == {"url": "http://localhost:8000"}


def test_quotes_are_stripped_from_plain_strings():
    assert safe_load('greeting: "hello"') == {"greeting": "hello"}


def test_hash_comment_rules_match_yaml():
    # A hash at the value start (or after whitespace) opens a comment, same
    # as real YAML — so an unquoted hex color is a comment (quote it, as in
    # real YAML), the emptied line reads as a section opener, a mid-token
    # hash stays data, and a quoted hash stays data.
    assert safe_load("color: #ff0000") == {"color": {}}
    assert safe_load("tag: a#b") == {"tag": "a#b"}
    assert safe_load('color: "#ff0000"') == {"color": "#ff0000"}


# ---------------------------------------------------------------------------
# Structure: nesting, comments, blank lines
# ---------------------------------------------------------------------------


def test_nested_maps_and_dedent():
    text = "\n".join(
        [
            "providers:",
            "  llm: gemini",
            "  tts: qwen3_tts",
            "memory:",
            "  enable_long_term: true",
            "  recall:",
            "    limit: 3",
            "top_level: back",
        ]
    )
    assert safe_load(text) == {
        "providers": {"llm": "gemini", "tts": "qwen3_tts"},
        "memory": {"enable_long_term": True, "recall": {"limit": 3}},
        "top_level": "back",
    }


def test_full_line_comments_and_blank_lines_are_skipped():
    text = "\n".join(
        [
            "# top comment",
            "",
            "a: 1",
            "   # indented comment",
            "b: 2",
            "",
        ]
    )
    assert safe_load(text) == {"a": 1, "b": 2}


def test_crlf_input_parses_like_lf():
    # Windows-authored configs arrive with \r\n; splitlines() must absorb it.
    assert safe_load("a: 1\r\nnested:\r\n  b: 2\r\n") == {"a": 1, "nested": {"b": 2}}


def test_empty_input_returns_empty_dict():
    assert safe_load("") == {}
    assert safe_load("\n# only a comment\n") == {}


def test_file_like_input_is_read():
    assert safe_load(io.StringIO("a: 1")) == {"a": 1}


def test_duplicate_keys_last_wins():
    assert safe_load("a: 1\na: 2") == {"a": 2}


# ---------------------------------------------------------------------------
# Intended degradations of unsupported constructs
# ---------------------------------------------------------------------------


def test_key_with_empty_value_opens_a_nested_map():
    # The grammar's rule for section headers: "key:" means "map follows".
    # A trailing "key:" with no children is therefore an empty dict, never
    # None or "" — callers must not rely on empty-string scalars.
    assert safe_load("section:") == {"section": {}}


def test_scalar_lists_parse_flow_and_block():
    # Since 2026-08-03: flow sequences and block sequences of SCALARS parse
    # to real lists (the example config uses both — tools.allowed,
    # cors_origins, self_eval.scenarios). Sequences of maps stay
    # unsupported, and duplicate block items collapse (they ride dict keys).
    assert safe_load("items:\n  - alpha\n  - beta") == {"items": ["alpha", "beta"]}
    assert safe_load("allowed: [a, b, c]") == {"allowed": ["a", "b", "c"]}
    assert safe_load('origins: ["*"]') == {"origins": ["*"]}
    assert safe_load("empty: []") == {"empty": []}
    assert safe_load("kwargs: {}") == {"kwargs": {}}
    assert safe_load("nums: [1, 2.5, true]") == {"nums": [1, 2.5, True]}
    assert safe_load("items:\n  - alpha  # note") == {"items": ["alpha"]}


def test_example_config_parses_identically_to_pyyaml():
    # The whole point of this module: a PyYAML-less install must read the
    # SHIPPED example config exactly the way PyYAML would. This walked diff
    # is the guard that keeps the fallback honest as the example grows.
    yaml = pytest.importorskip("yaml")
    example = Path(__file__).resolve().parents[1] / "config" / "config.example.yaml"
    text = example.read_text(encoding="utf-8")
    assert safe_load(text) == yaml.safe_load(text)


def test_inline_comments_are_stripped_with_coercion_intact():
    # The example config uses inline comments; before 2026-08-03 this
    # loaded as the STRING "5  # not a comment here" on PyYAML-less
    # installs — type coercion must survive the strip.
    data = safe_load("count: 5  # a note")
    assert data["count"] == 5
    assert safe_load("flag: true # on by default") == {"flag": True}
    # Hashes inside a quoted value are data, comment after the quote drops.
    assert safe_load('note: "5 # kept"  # dropped') == {"note": "5 # kept"}


# ---------------------------------------------------------------------------
# safe_dump + round-trips
# ---------------------------------------------------------------------------


def _roundtrip(data):
    handle = io.StringIO()
    safe_dump(data, handle)
    return safe_load(handle.getvalue())


def test_roundtrip_representative_config():
    data = {
        "name": "Kurisu",
        "level": 3,
        "ratio": 0.5,
        "enabled": True,
        "disabled": False,
        "nothing": None,
        "nested": {"deep": {"value": 7}, "label": "hi"},
    }
    assert _roundtrip(data) == data


def test_roundtrip_value_with_colon():
    data = {"url": "http://localhost:8000"}
    assert _roundtrip(data) == data


def test_dump_sort_keys_orders_top_level():
    handle = io.StringIO()
    safe_dump({"b": 1, "a": 2}, handle, sort_keys=True)
    assert handle.getvalue() == "a: 2\nb: 1\n"


def test_dump_formats_none_and_bools_reloadably():
    handle = io.StringIO()
    safe_dump({"x": None, "y": True, "z": False}, handle)
    assert handle.getvalue() == "x: null\ny: true\nz: false\n"


def test_roundtrip_empty_string_survives():
    # "" dumps quoted ('k: ""') so it no longer reloads as a section header
    # (the pre-2026-08-03 asymmetry).
    assert _roundtrip({"k": ""}) == {"k": ""}
