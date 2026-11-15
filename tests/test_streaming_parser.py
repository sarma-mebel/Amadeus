"""Unit tests for amadeus.web.streaming: TaggedStreamParser and SentenceSegmenter."""

from __future__ import annotations

from amadeus.web.streaming import (
    DisplayDelta,
    SentenceEnd,
    SentenceSegmenter,
    TaggedStreamParser,
    parse_complete_reply,
)


def _run(parser: TaggedStreamParser, deltas: list[str]) -> list[object]:
    events: list[object] = []
    for delta in deltas:
        events.extend(parser.feed(delta))
    events.extend(parser.finish())
    return events


def _sentences(events: list[object]) -> list[SentenceEnd]:
    return [e for e in events if isinstance(e, SentenceEnd)]


def _display_text(events: list[object]) -> str:
    return "".join(e.text for e in events if isinstance(e, DisplayDelta))


class TestTaggedStreamParserWellFormed:
    def test_single_sentence_single_delta(self):
        parser = TaggedStreamParser()
        events = _run(parser, ["[happy] Hello there. || こんにちは。\n"])
        sentences = _sentences(events)
        assert len(sentences) == 1
        assert sentences[0].en == "Hello there."
        assert sentences[0].ja == "こんにちは。"
        assert sentences[0].emotion == "happy"
        assert sentences[0].seq == 1
        assert not parser.fallback_mode

    def test_multiple_sentences_one_delta(self):
        parser = TaggedStreamParser()
        text = (
            "[relaxed] First point. || 最初の点。\n"
            "[surprised] Second point! || 二番目の点！\n"
        )
        events = _run(parser, [text])
        sentences = _sentences(events)
        assert [s.emotion for s in sentences] == ["relaxed", "surprised"]
        assert [s.seq for s in sentences] == [1, 2]
        assert sentences[1].en == "Second point!"

    def test_tag_split_across_deltas(self):
        parser = TaggedStreamParser()
        events = _run(
            parser,
            ["[hap", "py] Hel", "lo. || こん", "にちは。\n"],
        )
        sentences = _sentences(events)
        assert len(sentences) == 1
        assert sentences[0].emotion == "happy"
        assert sentences[0].en == "Hello."
        assert sentences[0].ja == "こんにちは。"

    def test_double_pipe_split_across_deltas(self):
        parser = TaggedStreamParser()
        events = _run(
            parser,
            ["[neutral] A statement.", " |", "| 一つの文。\n"],
        )
        sentences = _sentences(events)
        assert len(sentences) == 1
        assert sentences[0].en == "A statement."
        assert sentences[0].ja == "一つの文。"

    def test_missing_japanese_segment(self):
        parser = TaggedStreamParser()
        events = _run(parser, ["[sad] Only English here.\n"])
        sentences = _sentences(events)
        assert len(sentences) == 1
        assert sentences[0].en == "Only English here."
        assert sentences[0].ja == ""

    def test_unknown_emotion_becomes_neutral(self):
        parser = TaggedStreamParser()
        events = _run(parser, ["[zesty] Wait what. || ちょっと待って。\n"])
        sentences = _sentences(events)
        assert sentences[0].emotion == "neutral"

    def test_rich_emotion_maps_to_basic_expression(self):
        # "confused" is a rich body-language tag: face gets the mapped basic
        # expression, the raw tag rides along for the gesture pools.
        parser = TaggedStreamParser()
        events = _run(parser, ["[confused] Wait what. || ちょっと待って。\n"])
        sentences = _sentences(events)
        assert sentences[0].emotion == "surprised"
        assert sentences[0].emotion_raw == "confused"

    def test_finish_flushes_unterminated_line(self):
        parser = TaggedStreamParser()
        events = _run(parser, ["[relaxed] No trailing newline. || 改行なし。"])
        sentences = _sentences(events)
        assert len(sentences) == 1
        assert sentences[0].en == "No trailing newline."

    def test_english_streams_as_display_deltas_before_line_completes(self):
        parser = TaggedStreamParser()
        events: list[object] = []
        events.extend(parser.feed("[happy] Hello "))
        events.extend(parser.feed("world"))
        events.extend(parser.feed(". || こんにちは世界。\n"))
        events.extend(parser.finish())

        display_before_sentence = _display_text(events)
        assert "Hello" in display_before_sentence
        assert "world" in display_before_sentence
        # Japanese must never appear in a display delta.
        assert "こんにちは" not in display_before_sentence
        sentences = _sentences(events)
        assert len(sentences) == 1


class TestTaggedStreamParserFallback:
    def test_no_leading_tag_triggers_fallback(self):
        parser = TaggedStreamParser()
        events = _run(parser, ["Just a plain English reply with no tags at all here."])
        assert parser.fallback_mode
        assert not _sentences(events)
        assert "plain English reply" in _display_text(events)

    def test_fallback_mode_persists_across_feeds(self):
        parser = TaggedStreamParser()
        events = list(parser.feed("No tag to start with, "))
        assert parser.fallback_mode
        events.extend(parser.feed("and more text keeps flowing in."))
        events.extend(parser.finish())
        assert not _sentences(events)
        assert "keeps flowing" in _display_text(events)

    def test_short_ambiguous_prefix_waits_for_more_data(self):
        parser = TaggedStreamParser()
        # Very short, no "]" yet — parser should not have decided.
        events = parser.feed("Hi")
        assert events == []
        assert not parser.fallback_mode


class TestParseCompleteReply:
    def test_well_formed_complete_reply(self):
        reply = (
            "[relaxed] I have analyzed the data. || データを分析しました。\n"
            "[happy] The results are promising. || 結果は有望です。"
        )
        sentences = parse_complete_reply(reply)
        assert len(sentences) == 2
        assert sentences[0].en == "I have analyzed the data."
        assert sentences[1].emotion == "happy"

    def test_untagged_reply_returns_empty(self):
        sentences = parse_complete_reply("This is just a normal untagged reply.")
        assert sentences == []


class TestSentenceSegmenter:
    def test_splits_on_terminal_punctuation(self):
        seg = SentenceSegmenter()
        sentences = seg.feed("First sentence. Second sentence! Third one?")
        # Last sentence has no trailing space/EOF marker fed yet in this call,
        # but since input ends exactly at "?" with no trailing content, it
        # still matches (end of buffer counts as boundary).
        assert sentences == ["First sentence.", "Second sentence!", "Third one?"]

    def test_avoids_splitting_abbreviations(self):
        seg = SentenceSegmenter()
        sentences = seg.feed("Dr. Makise made a breakthrough. It was huge.")
        assert sentences == ["Dr. Makise made a breakthrough.", "It was huge."]

    def test_delta_split_sentence_end(self):
        seg = SentenceSegmenter()
        first = seg.feed("This sentence is split across ")
        assert first == []
        second = seg.feed("two deltas. And a second sentence.")
        assert second == [
            "This sentence is split across two deltas.",
            "And a second sentence.",
        ]

    def test_finish_flushes_remainder(self):
        seg = SentenceSegmenter()
        seg.feed("Trailing text with no terminal punctuation")
        remainder = seg.finish()
        assert remainder == "Trailing text with no terminal punctuation"

    def test_finish_returns_none_when_empty(self):
        seg = SentenceSegmenter()
        seg.feed("Complete sentence.")
        assert seg.finish() is None


# --- speech_text (voice_language TTS pick) ------------------------------


def test_speech_text_japanese_default_prefers_ja():
    from amadeus.web.streaming import speech_text

    assert speech_text("Hello.", "こんにちは。", "japanese") == "こんにちは。"
    assert speech_text("Hello.", "", "japanese") == "Hello."


def test_speech_text_english_prefers_en():
    from amadeus.web.streaming import speech_text

    assert speech_text("Hello.", "こんにちは。", "english") == "Hello."
    assert speech_text("", "こんにちは。", "english") == "こんにちは。"


# --- rich emotion tags ---------------------------------------------------


def test_rich_emotion_maps_to_expression_and_keeps_raw():
    from amadeus.web.streaming import parse_complete_reply

    sentences = parse_complete_reply(
        "[embarrassed] I-it's not like that! || べ、別にそういうんじゃないから！"
    )
    assert sentences
    assert sentences[0].emotion == "happy"  # face vocabulary stays VRM-basic
    assert sentences[0].emotion_raw == "embarrassed"


def test_basic_emotion_has_no_raw_and_unknown_falls_back():
    from amadeus.web.streaming import parse_complete_reply

    basic = parse_complete_reply("[angry] Enough. || もういい。")
    assert basic[0].emotion == "angry"
    assert basic[0].emotion_raw is None

    unknown = parse_complete_reply("[zesty] What. || なに。")
    assert unknown[0].emotion == "neutral"
    assert unknown[0].emotion_raw is None


# --- non-verbal vocalization tags ----------------------------------------


def test_nonverbal_line_becomes_event_not_sentence():
    from amadeus.web.streaming import NonVerbalEnd

    parser = TaggedStreamParser()
    events = _run(
        parser,
        [
            "[relaxed] Fine, I'll admit it. || いいわ、認めるわよ。\n"
            "[laugh]\n"
            "[happy] You should see your face. || その顔、見ものね。\n"
        ],
    )
    sentences = _sentences(events)
    nonverbals = [e for e in events if isinstance(e, NonVerbalEnd)]
    assert [s.seq for s in sentences] == [1, 3]
    assert len(nonverbals) == 1
    assert nonverbals[0].kind == "laugh"
    assert nonverbals[0].seq == 2  # occupies its own slot in audio ordering
    assert "laugh" not in _display_text(events)


def test_nonverbal_trailing_text_is_dropped_and_never_displayed():
    from amadeus.web.streaming import NonVerbalEnd

    parser = TaggedStreamParser()
    events = _run(
        parser,
        ["[sigh] haa... || はぁ…\n[neutral] Anyway. || とにかく。\n"],
    )
    nonverbals = [e for e in events if isinstance(e, NonVerbalEnd)]
    assert len(nonverbals) == 1 and nonverbals[0].kind == "sigh"
    assert "haa" not in _display_text(events)


def test_nonverbal_split_across_deltas_does_not_leak_tokens():
    from amadeus.web.streaming import NonVerbalEnd

    parser = TaggedStreamParser()
    events = _run(
        parser,
        ["[neutral] Okay. || ええ。\n[la", "ugh]", "\n"],
    )
    assert [e.kind for e in events if isinstance(e, NonVerbalEnd)] == ["laugh"]
    assert _display_text(events).strip() == "Okay."


def test_parse_complete_reply_excludes_nonverbals_but_full_returns_them():
    from amadeus.web.streaming import parse_complete_reply_full

    text = "[laugh]\n[happy] Told you so. || 言ったでしょ。\n"
    assert len(parse_complete_reply(text)) == 1  # memory/transcripts contract
    sentences, thoughts, nonverbals = parse_complete_reply_full(text)
    assert len(sentences) == 1 and not thoughts
    assert [n.kind for n in nonverbals] == ["laugh"]
    assert nonverbals[0].seq < sentences[0].seq


def test_build_tagged_instructions_advertises_only_given_kinds():
    from amadeus.web.streaming import build_tagged_instructions

    plain = build_tagged_instructions(False)
    assert "[laugh]" not in plain

    with_clips = build_tagged_instructions(False, nonverbal_kinds=["laugh", "sigh"])
    assert "[laugh]" in with_clips and "[sigh]" in with_clips
    assert "[gasp]" not in with_clips
    # The base grammar and thought grammar are unaffected.
    assert with_clips.startswith(plain)


# --- half-followed grammar: '||' without tags ------------------------------


def test_fallback_strips_bilingual_tails_from_display():
    """Seen live: the model dropped its [emotion] tags but kept the '||'
    separators, and the fallback path displayed the raw wire format."""
    from amadeus.web.streaming import strip_bilingual_tails

    parser = TaggedStreamParser()
    events = _run(
        parser,
        [
            "That sounds familiar. || 退屈な講義ね。\n",
            "What were you working on? || 何をしていたの？\n",
        ],
    )
    assert parser.fallback_mode
    display = _display_text(events)
    assert "||" not in display
    assert "退屈" not in display
    assert "That sounds familiar." in display
    assert "What were you working on?" in display

    # The helper is code-safe: a genuine logical OR stays untouched.
    assert strip_bilingual_tails("use a || b in the guard") == "use a || b in the guard"


def test_fallback_segmenter_splits_on_japanese_enders():
    segmenter = SentenceSegmenter()
    sentences = segmenter.feed("英語じゃない文。 Second sentence. ")
    assert sentences == ["英語じゃない文。", "Second sentence."]


def test_tagged_mode_tagless_line_splits_bilingual_and_emits_display():
    """A completed line WITHOUT a leading tag, after the parser already
    decided on tagged mode, used to degrade to SentenceEnd(en=<raw line>,
    ja="") — leaking 'English || 日本語' wire format to display and TTS —
    while _flush_line_display returned [] so the sentence never appeared in
    the token stream at all (display/committed text diverged from the
    frame). The degrade path must split the halves like a tagged line AND
    emit the display delta."""
    parser = TaggedStreamParser()
    events = _run(
        parser,
        [
            "[happy] First line fine. || 一行目。\n",
            "Second line lost its tag. || 二行目。\n",
        ],
    )
    assert not parser.fallback_mode
    sentences = _sentences(events)
    assert len(sentences) == 2
    degraded = sentences[1]
    assert degraded.en == "Second line lost its tag."
    assert degraded.ja == "二行目。"
    assert degraded.emotion == "neutral"
    display = _display_text(events)
    assert "Second line lost its tag." in display
    assert "||" not in display
    assert "二行目" not in display


def test_tagged_mode_tagless_final_line_without_newline_degrades_cleanly():
    """Same degrade path via finish() (no trailing newline)."""
    parser = TaggedStreamParser()
    events = _run(
        parser,
        ["[happy] Fine. || いいわ。\nUntagged closer. || 締めの文。"],
    )
    sentences = _sentences(events)
    assert len(sentences) == 2
    assert sentences[1].en == "Untagged closer."
    assert sentences[1].ja == "締めの文。"
    display = _display_text(events)
    assert "Untagged closer." in display
    assert "||" not in display


def test_strip_special_tag_lines_removes_thoughts_and_nonverbals():
    """A reply of ONLY [thought]/non-verbal lines parses as tagged with zero
    sentences; the untagged fallbacks must strip those lines instead of
    showing/speaking/committing the literal '[thought] ...' text."""
    from amadeus.web.streaming import strip_special_tag_lines

    assert strip_special_tag_lines("[thought] A private aside.") == ""
    assert strip_special_tag_lines("[thought] Hm.\n[laugh]\n[sigh]") == ""
    mixed = "[thought] Private.\nReal prose stays.\n[giggle]"
    assert strip_special_tag_lines(mixed) == "Real prose stays."
    # Ordinary tagged/emotion lines are untouched — only the special kinds.
    tagged = "[happy] Hello. || こんにちは。"
    assert strip_special_tag_lines(tagged) == tagged
    plain = "No tags anywhere here."
    assert strip_special_tag_lines(plain) == plain
