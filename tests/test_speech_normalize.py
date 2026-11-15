"""Unit tests for amadeus.web.streaming.normalize_ja_for_speech.

Japanese number/date normalization pre-TTS: converts bare Arabic-digit runs,
dates, times, percents, and decimals into kanji-numeral Japanese so the
(Japanese-weakest-language) Qwen3 clone isn't left to guess a reading.
Audio-side only, mirroring sanitize_speech_for_clone's CJK gate.
"""

from __future__ import annotations

import pytest

from amadeus.web.streaming import normalize_ja_for_speech

# --- CJK gate / passthrough ------------------------------------------------


def test_non_cjk_text_passes_through_unchanged():
    english = "This is pure English with 42 apples and a 2026 deadline."
    assert normalize_ja_for_speech(english) == english


def test_empty_string_passes_through():
    assert normalize_ja_for_speech("") == ""


def test_plain_japanese_with_no_digits_unchanged():
    text = "ただの日本語の文章です。"
    assert normalize_ja_for_speech(text) == text


# --- digit runs -------------------------------------------------------------


@pytest.mark.parametrize(
    "text,expected",
    [
        ("これは3つです。", "これは三つです。"),
        ("12個あります。", "十二個あります。"),
        ("250円です。", "二百五十円です。"),
        ("1000円です。", "千円です。"),
    ],
)
def test_digit_runs_convert_to_kanji_numerals(text, expected):
    assert normalize_ja_for_speech(text) == expected


def test_long_digit_run_falls_back_to_digit_by_digit():
    # > 8 digits: safer to read digit-by-digit than invent a huge-number
    # pronunciation.
    out = normalize_ja_for_speech("999999999個です。")
    assert out == "九九九九九九九九九個です。"


def test_eight_digit_run_still_gets_place_value_reading():
    out = normalize_ja_for_speech("99999999個です。")
    assert out == "九千九百九十九万九千九百九十九個です。"


# --- year / date / time / percent / decimal --------------------------------


def test_year_context_reads_digit_place_not_date_grouped():
    assert normalize_ja_for_speech("2026年です。") == "二千二十六年です。"


def test_slash_date_between_cjk_context():
    assert normalize_ja_for_speech("7/20に行きます。") == "七月二十日に行きます。"


def test_already_kanji_date_is_left_alone():
    text = "7月20日は晴れです。"
    assert normalize_ja_for_speech(text) == "七月二十日は晴れです。"


def test_time_hh_mm_converts_to_ji_fun():
    assert (
        normalize_ja_for_speech("15:30に会いましょう。")
        == "十五時三十分に会いましょう。"
    )


def test_time_on_the_hour_omits_zero_minutes():
    assert normalize_ja_for_speech("10:00に会いましょう。") == "十時に会いましょう。"


def test_percent_converts_to_kanji_plus_katakana_unit():
    assert normalize_ja_for_speech("50%です。") == "五十パーセントです。"


def test_decimal_point_reads_as_ten():
    assert normalize_ja_for_speech("3.5メートルです。") == "三点五メートルです。"


def test_decimal_leading_zero_whole_part():
    assert normalize_ja_for_speech("0.5です。") == "〇点五です。"


# --- latin-identifier digits stay untouched ---------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "GPT-4を使うわ。",
        "v0.12をインストールして。",
        "item2を選んで。",
        "Python3のコードよ。",
    ],
)
def test_digits_attached_to_latin_identifiers_are_left_alone(text):
    assert normalize_ja_for_speech(text) == text


def test_standalone_digit_next_to_latin_word_is_still_untouched_but_cjk_number_converts():
    # The GPT-4 token stays alone; a separate, CJK-adjacent number elsewhere
    # in the same sentence still converts.
    out = normalize_ja_for_speech("GPT-4は3年前に出たわ。")
    assert "GPT-4" in out
    assert "三年前" in out


# --- idempotence -------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "これは3つです。",
        "12個あります。",
        "250円です。",
        "1000円です。",
        "2026年です。",
        "7/20に行きます。",
        "15:30に会いましょう。",
        "50%です。",
        "3.5メートルです。",
        "GPT-4は3年前に出たわ。",
        "v0.12をインストールして。",
        "99999999個です。",
        "999999999個です。",
        "ただの日本語文です。",
        "",
    ],
)
def test_idempotent(text):
    once = normalize_ja_for_speech(text)
    twice = normalize_ja_for_speech(once)
    assert once == twice
