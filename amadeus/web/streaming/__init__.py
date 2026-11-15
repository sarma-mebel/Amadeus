"""Structured bilingual streaming output: parsing, TTS pipeline, and the WS
turn orchestrator, split into submodules:

- ``parser``: the tagged-output grammar and text machinery (parsing,
  emotion mapping, speech-text helpers).
- ``bridge``: the async/sync generator bridge for provider streaming.
- ``tts_pipeline``: ordered concurrent sentence-level TTS synthesis.
- ``turn``: the WebSocket turn orchestrator that ties the above together.

This package replaces the former flat ``amadeus/web/streaming.py`` module.
Everything that used to be importable from that module — every public name
and every underscore-prefixed name other modules/tests reached into — is
re-exported here unchanged, so existing ``from .streaming import X`` /
``from amadeus.web.streaming import X`` / ``streaming.X`` attribute access
keeps working without modification.
"""

from __future__ import annotations

from .bridge import _SENTINEL, iter_in_thread
from .nonverbals import (
    NonVerbalLibrary,
    get_nonverbal_library,
    reset_nonverbal_library_for_test,
)
from .parser import (
    NONVERBAL_KINDS,
    RICH_EMOTION_TO_EXPRESSION,
    TAGGED_OUTPUT_INSTRUCTIONS,
    THOUGHT_OUTPUT_INSTRUCTIONS,
    THOUGHT_TAG,
    VALID_EMOTIONS,
    DisplayDelta,
    NonVerbalEnd,
    SentenceEnd,
    SentenceSegmenter,
    StreamEvent,
    TaggedStreamParser,
    ThoughtEnd,
    build_tagged_instructions,
    ensure_tagged_reply,
    normalize_ja_for_speech,
    parse_complete_reply,
    parse_complete_reply_events,
    parse_complete_reply_full,
    sanitize_speech_for_clone,
    speech_text,
    split_first_clause,
    strip_bilingual_tails,
    strip_markdown_for_speech,
    strip_special_tag_lines,
)
from .tts_pipeline import (
    AudioChunk,
    TTSPipeline,
    _build_tts_pipeline,
    _SentenceAudioBatcher,
    _submit_sentence_audio,
    dominant_emotion,
    resolve_tts_emotion,
)
from .think_delay import (
    ThinkDelaySignals,
    compute_think_delay,
    signals_for_turn,
)
from .turn import (
    StreamCancelled,
    _finalize_streamed_turn,
    _sentence_frame,
    _stream_final_text,
    _stream_tagged_generation,
    _TurnAudio,
    _wrap_emit_with_timings,
    run_streaming_turn,
)

__all__ = [
    "VALID_EMOTIONS",
    "RICH_EMOTION_TO_EXPRESSION",
    "THOUGHT_TAG",
    "TAGGED_OUTPUT_INSTRUCTIONS",
    "THOUGHT_OUTPUT_INSTRUCTIONS",
    "DisplayDelta",
    "SentenceEnd",
    "ThoughtEnd",
    "StreamEvent",
    "SentenceSegmenter",
    "TaggedStreamParser",
    "build_tagged_instructions",
    "ensure_tagged_reply",
    "parse_complete_reply",
    "parse_complete_reply_events",
    "parse_complete_reply_full",
    "NONVERBAL_KINDS",
    "NonVerbalEnd",
    "NonVerbalLibrary",
    "get_nonverbal_library",
    "reset_nonverbal_library_for_test",
    "sanitize_speech_for_clone",
    "normalize_ja_for_speech",
    "strip_bilingual_tails",
    "strip_markdown_for_speech",
    "strip_special_tag_lines",
    "speech_text",
    "split_first_clause",
    "_SENTINEL",
    "iter_in_thread",
    "AudioChunk",
    "TTSPipeline",
    "dominant_emotion",
    "resolve_tts_emotion",
    "_build_tts_pipeline",
    "_SentenceAudioBatcher",
    "_submit_sentence_audio",
    "StreamCancelled",
    "run_streaming_turn",
    "_wrap_emit_with_timings",
    "_sentence_frame",
    "_TurnAudio",
    "_stream_tagged_generation",
    "_stream_final_text",
    "_finalize_streamed_turn",
    "ThinkDelaySignals",
    "compute_think_delay",
    "signals_for_turn",
]
