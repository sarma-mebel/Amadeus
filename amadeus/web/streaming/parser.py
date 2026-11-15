"""Structured bilingual output parsing for Kurisu's streaming replies.

The LLM is asked to emit one line per sentence:

    [emotion] English sentence text. || 日本語の文。

This module turns a stream of raw text deltas (or a single complete reply)
into English display text plus per-sentence (english, japanese, emotion)
tuples, driving TTS input and avatar expressions without a VADER pass or a
deep-translator network hop. If the model does not follow the format, parsing
falls back to plain sentence segmentation with no translation/emotion tag —
callers are expected to run the existing VADER + translator path in that case.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, List, Optional

from ...logging_utils import get_logger
from ...providers.llm.base import ChatMessage

LOGGER = get_logger(__name__)


VALID_EMOTIONS = {"happy", "angry", "sad", "relaxed", "surprised", "neutral"}

# Richer tags the model may emit for BODY language. Each maps onto one of the
# six VRM face expressions above (the face vocabulary is fixed by the model's
# blend shapes); the raw tag rides along as SentenceEnd.emotion_raw so the
# frontend's emotion-gesture pools can react to the nuance.
RICH_EMOTION_TO_EXPRESSION = {
    "embarrassed": "happy",
    "proud": "happy",
    "grateful": "happy",
    "excited": "happy",
    "amused": "happy",
    "curious": "surprised",
    "confused": "surprised",
    "annoyed": "angry",
    "disappointed": "sad",
    "fearful": "sad",
    "relieved": "relaxed",
}

# Optional unspoken-thought line tag: displayed as a ghost line, never
# voiced, never stored in memory. The parser always recognizes it (so a
# model that emits one with the feature off gets cleanly stripped rather
# than coerced to a spoken "neutral" sentence); the consumer enforces the
# config flag and the max-one-per-reply rule.
THOUGHT_TAG = "thought"

# Non-verbal vocalization tags: a standalone "[laugh]" line becomes a
# pre-recorded one-shot clip spliced into the ordered audio stream instead
# of a synthesized sentence. Same defensive contract as [thought]: the
# parser ALWAYS recognizes these (never display text, never memory, never
# a coerced "neutral" sentence); whether any audio actually plays is the
# consumer's job (streaming turn + NonVerbalLibrary — no clips, no sound).
NONVERBAL_KINDS = frozenset({"laugh", "giggle", "sigh", "gasp", "hmm"})

# Matches a leading "[emotion]" tag at the start of a (stripped) line.
_TAG_RE = re.compile(r"^\[([a-zA-Z]+)\]\s*")

# Sentence-ending punctuation for the fallback segmenter. Deliberately avoids
# splitting on single periods after short abbreviation-like tokens (Dr., Mr.,
# e.g., etc.) by requiring the sentence-ending punctuation be followed by
# whitespace/EOF and at least a couple of preceding non-space characters.
# Includes the full-width Japanese enders: fallback text from a model that
# half-followed the tagged grammar interleaves 'English. || 日本語。' pairs,
# and without 。！？ the Japanese half glues onto the NEXT English sentence
# in one segment, defeating the per-segment bilingual-tail strip.
_SENTENCE_END_RE = re.compile(r"[.!?…。！？]+(?=\s|$)")
_ABBREVIATIONS = {
    "dr",
    "mr",
    "mrs",
    "ms",
    "prof",
    "vs",
    "etc",
    "e.g",
    "i.e",
    "sr",
    "jr",
}

# Clause-boundary punctuation (full- and half-width) used by
# split_first_clause() to find a natural place to cut the first sentence of a
# turn short, so TTS only has to synthesize ~1s of speech before playback can
# start. 、 and 。 are the common Japanese comma/period; ， and , are their
# full/half-width Latin equivalents (some models mix scripts).
_CLAUSE_BOUNDARY_RE = re.compile(r"[、。，,]")
_CLAUSE_MIN_LEN = 4
_CLAUSE_MAX_LEN = 20
_CLAUSE_SPLIT_THRESHOLD = 14
_CLAUSE_FALLBACK_LEN = 12


def split_first_clause(text: str) -> tuple[str, str]:
    """Split ``text`` into a short first TTS chunk and the remainder.

    Used only for the first sentence of a streamed turn: synthesizing a
    short first chunk lets audio playback start well before the whole
    sentence has been generated, which matters when the TTS provider runs
    slower than realtime.

    Rules:
    - If a clause-boundary punctuation mark (、 。 ， ,) occurs within the
      first ``_CLAUSE_MIN_LEN``.._CLAUSE_MAX_LEN`` characters, split right
      after it (boundary punctuation stays with the first chunk).
    - Otherwise, if the text is longer than ``_CLAUSE_SPLIT_THRESHOLD``
      characters, hard-split at ``_CLAUSE_FALLBACK_LEN`` characters.
    - Otherwise (short text with no early boundary), don't split — return
      the whole text as the first chunk and an empty remainder.

    Returns ``(first_chunk, remainder)``; ``remainder`` is ``""`` when no
    split happened.
    """
    if not text:
        return "", ""
    stripped = text.strip()
    if not stripped:
        return text, ""

    for match in _CLAUSE_BOUNDARY_RE.finditer(stripped):
        end = match.end()
        if _CLAUSE_MIN_LEN <= end <= _CLAUSE_MAX_LEN:
            first = stripped[:end]
            rest = stripped[end:].strip()
            if not rest:
                return stripped, ""
            return first, rest
        if end > _CLAUSE_MAX_LEN:
            # No boundary appeared early enough; stop scanning.
            break

    if len(stripped) > _CLAUSE_SPLIT_THRESHOLD:
        first = stripped[:_CLAUSE_FALLBACK_LEN]
        rest = stripped[_CLAUSE_FALLBACK_LEN:].strip()
        if not rest:
            return stripped, ""
        return first, rest

    return stripped, ""


@dataclass
class DisplayDelta:
    """A chunk of English display text as it streams in."""

    text: str


@dataclass
class SentenceEnd:
    """A finalized sentence: display text, speech text, and emotion."""

    seq: int
    en: str
    ja: str
    emotion: str
    emotion_intensity: float = 0.8
    # The model's original tag when it used one of the rich body-language
    # emotions (RICH_EMOTION_TO_EXPRESSION); None when it used a basic one.
    emotion_raw: str | None = None


@dataclass
class ThoughtEnd:
    """A finalized [thought] line: displayed unspoken, never sent to TTS.

    Shares the parser's sentence counter, so sentence/audio pairing sees a
    seq gap where a thought occurred — harmless, ordering is preserved.
    """

    seq: int
    text: str


@dataclass
class NonVerbalEnd:
    """A finalized non-verbal line ("[laugh]"): audio-only, no display text.

    Shares the sentence counter like ThoughtEnd, so a spliced clip lands at
    its own seq slot in the strictly-ordered audio stream.
    """

    seq: int
    kind: str


StreamEvent = "DisplayDelta | SentenceEnd | ThoughtEnd | NonVerbalEnd"


class SentenceSegmenter:
    """Fallback sentence splitter for untagged model output.

    Used both when the tagged parser detects the model isn't following the
    format, and by any caller that just wants plain English sentence
    boundaries (e.g. for VADER + translator per-sentence dispatch).
    """

    def __init__(self) -> None:
        self._buffer = ""

    def feed(self, delta: str) -> List[str]:
        if not delta:
            return []
        self._buffer += delta
        return self._extract_complete_sentences()

    def finish(self) -> Optional[str]:
        remainder = self._buffer.strip()
        self._buffer = ""
        return remainder or None

    def _extract_complete_sentences(self) -> List[str]:
        sentences: List[str] = []
        while True:
            match = _SENTENCE_END_RE.search(self._buffer)
            if not match:
                break
            end = match.end()
            candidate = self._buffer[:end].strip()
            preceding_word = (
                candidate.rsplit(None, 1)[-1].rstrip(".!?…").lower()
                if candidate
                else ""
            )
            if preceding_word in _ABBREVIATIONS and end < len(self._buffer):
                # Don't split on "Dr." etc. unless it's the end of the stream;
                # extend the search past this punctuation mark.
                probe = self._buffer[end:]
                next_match = _SENTENCE_END_RE.search(probe)
                if next_match is None:
                    break
                end = end + next_match.end()
                candidate = self._buffer[:end].strip()
            if candidate:
                sentences.append(candidate)
            self._buffer = self._buffer[end:].lstrip()
            if not self._buffer:
                break
        return sentences


class TaggedStreamParser:
    """Incremental parser for the "[emotion] English || Japanese" grammar.

    Feed raw text deltas via ``feed()``; call ``finish()`` once the stream
    ends to flush any trailing partial sentence. Events are returned in
    arrival order. If the very first non-whitespace content does not begin
    with a valid ``[emotion]`` tag, the parser permanently switches to
    fallback mode: all subsequent text (including what was already buffered)
    is emitted as ``DisplayDelta`` only, with no ``SentenceEnd`` events, and
    ``fallback_mode`` becomes True so the caller can drive translation/emotion
    itself via ``SentenceSegmenter``.
    """

    def __init__(self) -> None:
        self._raw_buffer = ""
        self._decided = False
        self.fallback_mode = False
        self._seq = 0
        self._fallback_segmenter = SentenceSegmenter()

    def feed(self, delta: str) -> List[object]:
        if not delta:
            return []
        self._raw_buffer += delta
        return self._process()

    def finish(self) -> List[object]:
        events: List[object] = []
        if not self._decided:
            # Never saw enough to decide; treat whatever remains as fallback.
            self._enter_fallback_mode()
        if self.fallback_mode:
            remainder = self._fallback_segmenter.finish()
            if remainder:
                cleaned = strip_bilingual_tails(remainder)
                if cleaned:
                    events.append(DisplayDelta(cleaned))
            return events

        remaining = self._raw_buffer.strip()
        self._raw_buffer = ""
        if remaining:
            events.extend(self._flush_line_display(remaining))
            events.append(self._finalize_line(remaining))
        return events

    def _process(self) -> List[object]:
        events: List[object] = []

        if not self._decided:
            stripped = self._raw_buffer.lstrip()
            if not stripped:
                return events
            # Need at least a closing bracket to decide, or enough characters
            # to conclude there's no tag at all.
            if "]" in stripped:
                if _TAG_RE.match(stripped):
                    self._decided = True
                else:
                    self._enter_fallback_mode()
                    events.extend(self._drain_fallback())
                    return events
            elif len(stripped) > 12 and not stripped.startswith("["):
                # Long enough run of text with no opening bracket at all.
                self._enter_fallback_mode()
                events.extend(self._drain_fallback())
                return events
            else:
                return events  # wait for more data

        if self.fallback_mode:
            events.extend(self._drain_fallback())
            return events

        # Tagged mode: process complete lines; keep the last partial line
        # buffered (it may still be accumulating the Japanese half).
        while "\n" in self._raw_buffer:
            line, self._raw_buffer = self._raw_buffer.split("\n", 1)
            line = line.strip()
            if line:
                # Any English that arrived in the same chunk as the line
                # terminator was never streamed as tokens; flush it (plus a
                # separator so consecutive sentences don't concatenate).
                events.extend(self._flush_line_display(line))
                events.append(self._finalize_line(line))

        # Stream English display text as it arrives, even before the line is
        # complete, so the client isn't stuck waiting for the whole sentence.
        # Only do this for the *current* (still-buffering) line, and only for
        # the portion before "||" (Japanese half is never streamed as tokens).
        events.extend(self._peek_partial_display())
        return events

    def _enter_fallback_mode(self) -> None:
        self.fallback_mode = True
        self._decided = True

    def _drain_fallback(self) -> List[object]:
        pending = self._raw_buffer
        self._raw_buffer = ""
        events: List[object] = []
        for sentence in self._fallback_segmenter.feed(pending):
            # Half-followed grammar (separators without tags) must never
            # reach the display as raw '|| japanese' wire format.
            cleaned = strip_bilingual_tails(sentence)
            if cleaned:
                events.append(DisplayDelta(cleaned + " "))
        return events

    _emitted_partial_len: int = 0

    def _flush_line_display(self, line: str) -> List[object]:
        """Emit the not-yet-streamed English tail of a completed line.

        The partial-display peek only streams what has arrived *before* the
        line terminator shows up; whatever came with the terminator (often
        the sentence's end plus the whole Japanese half) was silently
        dropped from the token stream. Always ends with a single space so
        the next sentence never concatenates onto this one.
        """
        tag_match = _TAG_RE.match(line.lstrip())
        if not tag_match:
            # Tagless degrade path (mirrors _finalize_line): the line still
            # becomes a SentenceEnd, so its English half must still reach
            # the token stream — returning [] here left the sentence out of
            # the display/committed reply while the frame and TTS carried
            # it. Only the part before '||' is display text.
            split_at = line.find("||")
            display_part = line if split_at == -1 else line[:split_at]
            remainder = display_part[self._emitted_partial_len :].rstrip()
            return [DisplayDelta(remainder + " " if remainder else " ")]
        if tag_match.group(1).lower() == THOUGHT_TAG:
            # Thought lines never stream as display tokens.
            return []
        if tag_match.group(1).lower() in NONVERBAL_KINDS:
            # Non-verbal lines are audio-only; nothing to display.
            return []
        after_tag = line.lstrip()[tag_match.end() :]
        split_at = after_tag.find("||")
        display_part = after_tag if split_at == -1 else after_tag[:split_at]
        remainder = display_part[self._emitted_partial_len :].rstrip()
        return [DisplayDelta(remainder + " " if remainder else " ")]

    def _peek_partial_display(self) -> List[object]:
        """Emit newly-available English text from the in-progress line."""
        current_line = self._raw_buffer
        tag_match = _TAG_RE.match(current_line.lstrip())
        if not tag_match:
            return []
        if tag_match.group(1).lower() == THOUGHT_TAG:
            # An in-progress thought line must not leak into the token
            # stream either — this is the second (pre-finalize) leak path.
            return []
        if tag_match.group(1).lower() in NONVERBAL_KINDS:
            # Same for an in-progress non-verbal line (any stray text a
            # model appends after the tag must never stream as tokens).
            return []
        after_tag = current_line.lstrip()[tag_match.end() :]
        split_at = after_tag.find("||")
        display_part = after_tag if split_at == -1 else after_tag[:split_at]
        if len(display_part) <= self._emitted_partial_len:
            return []
        new_text = display_part[self._emitted_partial_len :]
        # Don't emit a dangling partial word/tag fragment right at the
        # boundary; hold back if we might be mid "|" of "||".
        if display_part.endswith("|"):
            new_text = new_text[:-1]
        if not new_text:
            return []
        self._emitted_partial_len = len(display_part) - (
            1 if display_part.endswith("|") else 0
        )
        return [DisplayDelta(new_text)]

    def _finalize_line(self, line: str) -> object:
        self._emitted_partial_len = 0
        match = _TAG_RE.match(line)
        if not match:
            # Shouldn't happen once decided, but degrade gracefully: the
            # model dropped this line's tag while (possibly) keeping the
            # '||' separator, so split the halves apart exactly like a
            # tagged line — the raw 'English || 日本語' wire format must
            # reach neither the display/memory nor TTS.
            self._seq += 1
            if "||" in line:
                en_part, ja_part = line.split("||", 1)
            else:
                en_part, ja_part = line, ""
            return SentenceEnd(
                seq=self._seq,
                en=strip_bilingual_tails(en_part.strip()),
                ja=ja_part.strip(),
                emotion="neutral",
            )

        emotion = match.group(1).lower()
        if emotion == THOUGHT_TAG:
            self._seq += 1
            # Defensively drop a "|| japanese" half — thoughts are
            # English-only and never spoken.
            return ThoughtEnd(
                seq=self._seq, text=line[match.end() :].split("||", 1)[0].strip()
            )
        if emotion in NONVERBAL_KINDS:
            self._seq += 1
            # Any trailing text on the line is dropped — the grammar says a
            # non-verbal tag stands alone, and there's no clip for prose.
            return NonVerbalEnd(seq=self._seq, kind=emotion)
        emotion_raw: str | None = None
        if emotion in RICH_EMOTION_TO_EXPRESSION:
            emotion_raw = emotion
            emotion = RICH_EMOTION_TO_EXPRESSION[emotion]
        elif emotion not in VALID_EMOTIONS:
            emotion = "neutral"
        rest = line[match.end() :]
        if "||" in rest:
            en_part, ja_part = rest.split("||", 1)
        else:
            en_part, ja_part = rest, ""
        self._seq += 1
        return SentenceEnd(
            seq=self._seq,
            en=en_part.strip(),
            ja=ja_part.strip(),
            emotion=emotion,
            emotion_raw=emotion_raw,
        )


TAGGED_OUTPUT_INSTRUCTIONS = (
    "Output format (strict): one line per sentence, no other text:\n"
    "[emotion] English sentence. || Japanese translation of that sentence.\n"
    "emotion is normally one of: happy, angry, sad, relaxed, surprised, neutral.\n"
    "When one of these fits clearly better, you may use it instead: embarrassed,\n"
    "proud, grateful, excited, amused, curious, confused, annoyed, disappointed,\n"
    "fearful, relieved. Never invent tags outside these two lists.\n"
    "The very first line must be a very short reaction — just a few words, like an\n"
    'interjection or brief acknowledgement (e.g. "Hmph.", "Oh, right.", "Wait.") — with\n'
    "the actual substance of your answer starting from the second line onward. This is\n"
    "because the first line is spoken aloud before the rest of your reply finishes\n"
    "generating. Later sentences can run normal length.\n"
    "Example:\n"
    "[relaxed] Hmph, fine. || はいはい、わかったわよ。\n"
    "[relaxed] That is a reasonable question. || それは妥当な質問ですね。\n"
    "[surprised] Wait, that changes everything. || 待って、それで全部変わるじゃない。"
)

_THOUGHT_OUTPUT_TEMPLATE = (
    "Optionally, you may include AT MOST ONE private inner-thought line anywhere "
    "after the first line, formatted as:\n"
    "[thought] A short unspoken thought, English only, no || translation.\n"
    "It is shown faintly on screen but never spoken aloud, and it never replaces "
    "your spoken reply — a reply must still contain normal spoken lines. Use it "
    "sparingly, for what {char_name} would think but not say."
)

# Back-compat constant (Kurisu wording) — prefer build_tagged_instructions,
# which names the active character instead of hardcoding Kurisu inside a
# custom pack's own output-format block.
THOUGHT_OUTPUT_INSTRUCTIONS = _THOUGHT_OUTPUT_TEMPLATE.format(char_name="Kurisu")


def _nonverbal_instructions(kinds: List[str]) -> str:
    tags = ", ".join(f"[{k}]" for k in sorted(kinds))
    return (
        "You may also insert a non-verbal reaction as its own line: exactly "
        f"one of {tags}, alone on the line, nothing else on it. It plays as a "
        "real vocal sound (a laugh, a sigh) in your voice, so place it where "
        "you would genuinely make that sound — a [laugh] line right before "
        "delivering a teasing remark, a [sigh] before conceding a point. Use "
        "at most one or two per reply, and only when the moment truly calls "
        "for it; most replies need none. A reply must still contain normal "
        "spoken lines."
    )


def build_tagged_instructions(
    include_thoughts: bool,
    nonverbal_kinds: Optional[List[str]] = None,
    char_name: str = "Kurisu",
) -> str:
    """The tagged-output system-prompt block, with the optional thought
    grammar appended when ``web.streaming_inner_thoughts`` is enabled and
    the non-verbal grammar appended when the caller has clips to play
    (``nonverbal_kinds`` — never advertise a sound that can't be made).
    ``char_name`` personalizes the thought grammar for custom packs."""
    parts = [TAGGED_OUTPUT_INSTRUCTIONS]
    if include_thoughts:
        parts.append(_THOUGHT_OUTPUT_TEMPLATE.format(char_name=char_name or "Kurisu"))
    if nonverbal_kinds:
        parts.append(_nonverbal_instructions(list(nonverbal_kinds)))
    return "\n".join(parts)


# Markdown constructs that read terribly when spoken: emphasis/bold markers,
# inline code, headings, list bullets, and [text](url) links. Display keeps
# the original text — this only cleans what goes to the translator/TTS.
_SPEECH_LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]*\)")
_SPEECH_EMPHASIS_RE = re.compile(r"(\*{1,3}|_{1,3}|`{1,3})(.+?)\1", re.DOTALL)
_SPEECH_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s+", re.MULTILINE)
_SPEECH_BULLET_RE = re.compile(r"^\s*(?:[-*+•]|\d+[.)])\s+", re.MULTILINE)


def strip_markdown_for_speech(text: str) -> str:
    """Flatten markdown into speakable prose for the untagged TTS fallback.

    Tool-augmented turns in particular tend to come back as markdown lists;
    without this the translator/TTS pipeline reads asterisks and bullet
    punctuation out loud.
    """
    cleaned = _SPEECH_LINK_RE.sub(r"\1", text)
    # Bullets/headings first: a leading "* " would otherwise pair with a
    # later emphasis marker and corrupt the unwrap.
    cleaned = _SPEECH_HEADING_RE.sub("", cleaned)
    cleaned = _SPEECH_BULLET_RE.sub("", cleaned)
    # Run twice so nested markers (***bold italic***) fully unwrap.
    cleaned = _SPEECH_EMPHASIS_RE.sub(r"\2", cleaned)
    cleaned = _SPEECH_EMPHASIS_RE.sub(r"\2", cleaned)
    return re.sub(r"[ \t]{2,}", " ", cleaned).strip()


_CJK_RE = re.compile(r"[぀-ヿ一-鿿]")


def strip_bilingual_tails(text: str) -> str:
    """Remove '|| japanese' halves from text that HALF-followed the tagged
    grammar — [emotion] tags dropped but the separators kept. The fallback
    paths would otherwise show the raw wire format to the user (seen live:
    'English sentence. || 日本語の文。 Next sentence. || ...'). Operates per
    line/segment: everything after '||' is dropped when it contains CJK, so
    prose about code ('a || b') is untouched. Callers on the streaming path
    feed segmenter output, whose sentence enders include 。！？ — one
    bilingual pair per segment.
    """
    out: List[str] = []
    for line in text.split("\n"):
        index = line.find("||")
        if index != -1 and _CJK_RE.search(line[index + 2 :]):
            line = line[:index].rstrip()
        out.append(line)
    return "\n".join(out).strip()


def strip_special_tag_lines(text: str) -> str:
    """Drop [thought]/non-verbal lines from text headed for the UNTAGGED
    fallback paths.

    A reply consisting of ONLY such lines parses as tagged with zero
    sentences, so both fallbacks (WS ``_stream_final_text`` and the POST
    half-followed-grammar branch) used to run ``strip_bilingual_tails`` on
    the RAW reply and keep the literal ``[thought] ...`` text — leaked to
    display/TTS and, on POST, committed to memory. Special lines have no
    spoken/displayed form outside the tagged event flow, so they're removed
    line-wise; genuine prose never starts with these tags. An all-special
    reply strips to empty, which the empty-reply rollback then handles.
    """
    special = NONVERBAL_KINDS | {THOUGHT_TAG}
    kept: List[str] = []
    for line in text.split("\n"):
        match = _TAG_RE.match(line.strip())
        if match and match.group(1).lower() in special:
            continue
        kept.append(line)
    return "\n".join(kept).strip()


_LATIN_FILEISH_RE = re.compile(
    # Path-ish/file-ish runs: contain a dot-extension, slash, or underscore.
    r"[A-Za-z0-9_\-]*(?:[/\\.][A-Za-z0-9_\-]+)+|[A-Za-z0-9]+(?:_[A-Za-z0-9]+)+"
)
_LATIN_LONG_RUN_RE = re.compile(r"[A-Za-z][A-Za-z0-9 \-]{15,}[A-Za-z0-9]")

# Bracketed stage directions must never be SPOKEN. The tagged-output grammar
# puts exactly one [emotion] tag at line START, which the parser consumes —
# but models occasionally repeat a tag on the Japanese side of the '||' or
# drop one mid-sentence ("[amused]"), and the voice then literally reads
# "amused" aloud. Any short ASCII-bracketed run is speech-hostile (emotion
# tags, [laugh]-style non-verbals that weren't standalone, [3]-style
# citations), so strip them all on the audio side; display text is
# untouched, and Japanese quoting brackets 「」 are a different character
# entirely. Applied BEFORE the CJK gate so the English-fallback speech path
# (untagged replies, empty-ja sentences) is covered too.
_BRACKET_TAG_RE = re.compile(r"\[[^\[\]\n]{1,24}\]")


def sanitize_speech_for_clone(text: str) -> str:
    """Keep the zero-shot voice clone on-timbre.

    The Qwen3 clone is prompted with a Japanese reference clip; long latin
    runs inside a sentence (file names, paths, English titles) reliably pull
    the generation off-voice — listeners hear it flip to a different (often
    male) speaker mid-reply. Replace file-ish tokens with a spoken
    placeholder and truncate other long latin runs. Applied only to the
    audio side of Japanese sentences; display text is untouched, and pure
    English text (legacy fallback path) passes through unchanged.
    """
    text = _BRACKET_TAG_RE.sub("", text).strip()
    if not _CJK_RE.search(text):
        return text
    cleaned = _LATIN_FILEISH_RE.sub("そのファイル", text)
    cleaned = _LATIN_LONG_RUN_RE.sub(lambda m: m.group(0)[:12], cleaned)
    return re.sub(r"(そのファイル)(?:[、 ]*そのファイル)+", r"\1", cleaned)


# --- Japanese number/date normalization for TTS --------------------------
#
# The Qwen3-TTS technical report flags Japanese as the model's weakest
# supported language (WER 3.8-6.4 vs. <2 for English/Chinese); numbers and
# dates left as Arabic digits are a common misreading source since the
# "correct" reading depends on Japanese numeral grammar the model doesn't
# reliably apply. Converting to kanji numerals text-side sidesteps that.
#
# A digit run adjacent to a latin identifier (GPT-4, v0.12) must NOT be
# touched — that's still meant to be read as an English/alphanumeric token,
# not a Japanese number. Everything here is audio-side only, mirroring
# sanitize_speech_for_clone: display text is untouched.

_KANJI_DIGITS = "〇一二三四五六七八九"
_KANJI_SMALL_UNITS = ("", "十", "百", "千")
_KANJI_BIG_UNITS = ("", "万", "億")

# A digit run is "latin-attached" (leave it alone) when immediately preceded
# or followed by a latin letter, or preceded by a latin letter+punctuation
# combo like "v0.12" or "GPT-4". Checked against the surrounding characters
# at match time rather than baked into one mega-regex, to keep each pattern
# readable.
_LATIN_ADJACENT_RE = re.compile(r"[A-Za-z]")

_MAX_KANJI_CONVERSION_DIGITS = 8


def _int_to_kanji(n: int) -> str:
    """Convert a non-negative integer to kanji numerals with proper
    Japanese reading structure (man/sen/hyaku/juu place values), e.g.
    3 -> 三, 12 -> 十二, 250 -> 二百五十, 1000 -> 千, 2026 -> 二千二十六.
    """
    if n == 0:
        return _KANJI_DIGITS[0]

    def _four_digit_block(block: int) -> str:
        # block is 0..9999
        out = []
        s = str(block).zfill(4)
        for i, ch in enumerate(s):
            d = int(ch)
            unit = _KANJI_SMALL_UNITS[3 - i]
            if d == 0:
                continue
            if d == 1 and unit:  # bare "十"/"百"/"千", not "一十"/"一百"/"一千"
                out.append(unit)
            else:
                out.append(_KANJI_DIGITS[d] + unit)
        return "".join(out)

    # Split into 4-digit groups from the least significant end (man = 10^4).
    groups: List[int] = []
    remainder = n
    while remainder > 0:
        groups.append(remainder % 10000)
        remainder //= 10000

    parts: List[str] = []
    for i in range(len(groups) - 1, -1, -1):
        block = groups[i]
        if block == 0:
            continue
        block_text = _four_digit_block(block)
        if i < len(_KANJI_BIG_UNITS):
            parts.append(block_text + _KANJI_BIG_UNITS[i])
        else:  # pragma: no cover - guarded by _MAX_KANJI_CONVERSION_DIGITS
            parts.append(block_text)
    return "".join(parts) if parts else _KANJI_DIGITS[0]


def _digits_to_kanji(digits: str) -> str:
    """Digit-by-digit kanji reading, used for runs too long to confidently
    read as a single place-value number (phone-number-shaped, etc.)."""
    return "".join(_KANJI_DIGITS[int(c)] for c in digits)


def _number_to_kanji(digits: str) -> str:
    """Convert a plain digit run to kanji, capping place-value conversion at
    ``_MAX_KANJI_CONVERSION_DIGITS`` digits; longer runs fall back to a safer
    digit-by-digit reading rather than inventing a huge-number pronunciation.
    """
    stripped = digits.lstrip("0")
    if not stripped:
        return _KANJI_DIGITS[0] * len(digits) if len(digits) > 1 else _KANJI_DIGITS[0]
    if len(stripped) > _MAX_KANJI_CONVERSION_DIGITS:
        return _digits_to_kanji(digits)
    leading_zeros = len(digits) - len(stripped)
    return (_KANJI_DIGITS[0] * leading_zeros) + _int_to_kanji(int(stripped))


def _is_latin_attached(text: str, start: int, end: int) -> bool:
    """True when the digit run at ``text[start:end]`` is glued to a latin
    identifier (GPT-4, v0.12, item2) and should be left as Arabic digits."""
    before = text[start - 1] if start > 0 else ""
    after = text[end] if end < len(text) else ""
    if _LATIN_ADJACENT_RE.match(before) or _LATIN_ADJACENT_RE.match(after):
        return True
    # "v0.12" style: a latin letter, then some digits/dots, then our run.
    prefix = text[:start]
    if re.search(r"[A-Za-z][A-Za-z0-9._\-]*$", prefix):
        return True
    return False


# Order matters: more specific patterns (percent/time/date/year/decimal) run
# before the generic digit-run fallback so they consume their digits first.
_PERCENT_RE = re.compile(r"(\d{1,3})\s*%")
_TIME_RE = re.compile(r"(?<![\d.])([01]?\d|2[0-3]):([0-5]\d)(?![\d:])")
_DATE_SLASH_RE = re.compile(
    r"(?<![\d/])(1[0-2]|0?[1-9])/([12]\d|3[01]|0?[1-9])(?![\d/])"
)
_YEAR_RE = re.compile(r"(?<!\d)([12]\d{3})年")
_DECIMAL_RE = re.compile(r"(?<!\d)(\d{1,8})\.(\d{1,4})(?!\d)")
_DIGIT_RUN_RE = re.compile(r"\d+")


def normalize_ja_for_speech(text: str) -> str:
    """Normalize Arabic numerals/dates/times into kanji-numeral Japanese for
    TTS input, so the (Japanese-weakest-language) clone isn't left guessing
    how to read a bare digit run.

    CJK-gated exactly like :func:`sanitize_speech_for_clone`: non-Japanese
    text (the English voice-language fallback) passes through unchanged.
    Conservative and idempotent — running it twice equals running it once,
    since every substitution replaces Arabic digits with kanji and none of
    the patterns match kanji output. Digit runs glued to a latin identifier
    (``GPT-4``, ``v0.12``) are left untouched.
    """
    if not _CJK_RE.search(text):
        return text

    # 1) Percent: "50%" -> "五十パーセント". Must run before the generic
    #    digit-run pass consumes the same digits.
    def _percent_sub(m: re.Match) -> str:
        if _is_latin_attached(text, m.start(1), m.end(1)):
            return m.group(0)
        return _number_to_kanji(m.group(1)) + "パーセント"

    text = _PERCENT_RE.sub(_percent_sub, text)

    # 2) Time "HH:MM" -> "H時M分".
    def _time_sub(m: re.Match) -> str:
        hour, minute = m.group(1), m.group(2)
        hour_kanji = _number_to_kanji(hour)
        if minute == "00":
            return f"{hour_kanji}時"
        return f"{hour_kanji}時{_number_to_kanji(minute)}分"

    text = _TIME_RE.sub(_time_sub, text)

    # 3) Date "M/D" -> "M月D日", but only in Japanese-text context (the CJK
    #    gate above already ensures the string as a whole has CJK content;
    #    slash dates are otherwise ambiguous with fractions/ratios, so this
    #    stays deliberately narrow — valid month/day ranges only).
    def _date_sub(m: re.Match) -> str:
        month, day = m.group(1), m.group(2)
        return f"{_number_to_kanji(month)}月{_number_to_kanji(day)}日"

    text = _DATE_SLASH_RE.sub(_date_sub, text)

    # 4) Year "2026年" -> "二千二十六年" (read digit-place, not date-grouped
    #    like "20-26"; that's how Japanese actually reads calendar years).
    def _year_sub(m: re.Match) -> str:
        return _number_to_kanji(m.group(1)) + "年"

    text = _YEAR_RE.sub(_year_sub, text)

    # 5) Decimal "3.5" -> "三点五" (each side read digit-by-digit/place-value
    #    the way Japanese decimal reading works: integer part normal, digits
    #    after the point read individually via the same kanji table).
    def _decimal_sub(m: re.Match) -> str:
        if _is_latin_attached(text, m.start(1), m.end(1) + 1 + len(m.group(2))):
            return m.group(0)
        whole, frac = m.group(1), m.group(2)
        return f"{_number_to_kanji(whole)}点{_digits_to_kanji(frac)}"

    text = _DECIMAL_RE.sub(_decimal_sub, text)

    # 6) Whatever plain digit runs remain: convert unless latin-attached.
    def _plain_sub(m: re.Match) -> str:
        if _is_latin_attached(text, m.start(), m.end()):
            return m.group(0)
        return _number_to_kanji(m.group(0))

    text = _DIGIT_RUN_RE.sub(_plain_sub, text)

    return text


_RETAG_PROMPT = (
    "Rewrite your draft reply below as your final spoken answer, following the "
    "required output format EXACTLY (one line per sentence: [emotion] English "
    "sentence. || Japanese sentence.). Keep the key substance, but say it the "
    "way you would out loud: no markdown, no bullet points, no headings, at "
    "most six sentences. Never add [thought] lines.\n\nDRAFT:\n"
)


def ensure_tagged_reply(llm: Any, system_prompt: str, reply: str) -> str:
    """One repair pass for replies that ignored the tagged grammar.

    Tool-augmented turns are the main offender: after native function-calling
    rounds the model tends to answer in plain markdown, which loses the
    per-sentence emotion and native-Japanese speech text. One cheap rewrite
    against the same system prompt (which already carries the format
    instructions) restores them; on any failure the original reply is
    returned and the VADER+translator fallback handles it as before.
    """
    from ... import telemetry

    telemetry.incr("web.tagged_repairs")
    try:
        candidate = llm.generate(
            [
                ChatMessage(role="system", content=system_prompt),
                ChatMessage(role="user", content=f"{_RETAG_PROMPT}{reply}"),
            ]
        )
        telemetry.record_llm_usage(llm, "repair")
        if candidate and parse_complete_reply(candidate):
            return candidate
        LOGGER.info("Re-tag pass did not produce tagged output; keeping draft")
        telemetry.incr("web.tagged_repair_failures")
    except Exception as exc:  # pragma: no cover - provider failure
        LOGGER.warning("Re-tag pass failed (%s); keeping draft reply", exc)
        telemetry.incr("web.tagged_repair_failures")
    return reply


def parse_complete_reply(text: str) -> List[SentenceEnd]:
    """Parse a full (non-streaming) reply through the tagged grammar.

    Returns an empty list if the reply doesn't match the tagged format at
    all (caller should fall back to VADER + translator on the raw text).
    """
    parser = TaggedStreamParser()
    events: List[object] = []
    events.extend(parser.feed(text))
    events.extend(parser.finish())
    if parser.fallback_mode:
        return []
    return [e for e in events if isinstance(e, SentenceEnd)]


def parse_complete_reply_events(
    text: str,
) -> tuple[List[SentenceEnd], List[ThoughtEnd]]:
    """Like :func:`parse_complete_reply` but also returns thought lines.

    ``parse_complete_reply`` keeps its sentences-only contract so every
    existing call site (memory commit, transcripts, TTS, repair validation)
    automatically excludes thoughts; only display-side consumers that want
    the thought call this variant.
    """
    parser = TaggedStreamParser()
    events: List[object] = []
    events.extend(parser.feed(text))
    events.extend(parser.finish())
    if parser.fallback_mode:
        return [], []
    return (
        [e for e in events if isinstance(e, SentenceEnd)],
        [e for e in events if isinstance(e, ThoughtEnd)],
    )


def parse_complete_reply_full(
    text: str,
) -> tuple[List[SentenceEnd], List[ThoughtEnd], List[NonVerbalEnd]]:
    """Like :func:`parse_complete_reply_events` plus non-verbal lines.

    Used by the pseudo-stream path (tool/non-streaming replies) so
    a [laugh] in a full reply still splices its clip into the audio stream.
    """
    parser = TaggedStreamParser()
    events: List[object] = []
    events.extend(parser.feed(text))
    events.extend(parser.finish())
    if parser.fallback_mode:
        return [], [], []
    return (
        [e for e in events if isinstance(e, SentenceEnd)],
        [e for e in events if isinstance(e, ThoughtEnd)],
        [e for e in events if isinstance(e, NonVerbalEnd)],
    )


def speech_text(en: str, ja: str, voice_language: str) -> str:
    """Pick which half of a bilingual sentence drives TTS.

    "japanese" (default) keeps the historical behavior: the tagged-output
    Japanese line, falling back to English when the model omitted it.
    "english" speaks the display text directly (no translation call on any
    path), falling back to Japanese only if English is somehow empty.
    """
    if voice_language == "english":
        return en or ja
    return ja or en
