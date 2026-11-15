"""Vocal emotion: prosody heuristic analyzer, stash/consume/TTL lifecycle,
prompt-line rendering, mood bias, and STT-endpoint wiring (both the plain
/api/stt path and the duplex loop share one wrapper — see sessions.py's
_SpeculativeSTT). No live model download, no live API calls: the heuristic
path is deterministic numpy over synthetic audio, and the ONNX model path
is exercised only through a fake that skips onnxruntime entirely.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import List

import numpy as np
import pytest

from amadeus import telemetry
from amadeus.config import Config
from amadeus.engine import ChatEngine
from amadeus.persona import build_system_prompt
from amadeus.providers.llm.base import DummyProvider
from amadeus.vocal_emotion import VocalEmotionResult, analyze_heuristic
from amadeus.vocal_tone_stash import analyze_and_stash, consume_vocal_tone

SAMPLE_RATE = 16000


def _pcm16(signal: np.ndarray) -> bytes:
    clipped = np.clip(signal, -1.0, 1.0)
    return (clipped * 32767).astype("<i2").tobytes()


def _tone(freq: float, seconds: float, amplitude: float, sr: int = SAMPLE_RATE):
    t = np.arange(int(sr * seconds)) / sr
    return amplitude * np.sin(2 * np.pi * freq * t)


# ---------------------------------------------------------------------------
# Heuristic analyzer on synthetic fixtures
# ---------------------------------------------------------------------------


def test_heuristic_abstains_on_silence():
    silence = _pcm16(np.zeros(int(SAMPLE_RATE * 1.0)))
    assert analyze_heuristic(silence, SAMPLE_RATE) is None


def test_heuristic_abstains_on_too_short_clip():
    short = _pcm16(np.zeros(int(SAMPLE_RATE * 0.1)))
    assert analyze_heuristic(short, SAMPLE_RATE) is None


def test_heuristic_abstains_on_empty_bytes():
    assert analyze_heuristic(b"", SAMPLE_RATE) is None


def test_heuristic_flat_steady_tone_reads_flat():
    flat = _tone(150.0, 1.5, 0.08)
    result = analyze_heuristic(_pcm16(flat), SAMPLE_RATE)
    assert result is not None
    assert result.label == "flat"
    assert result.source == "heuristic"
    assert 0.0 <= result.confidence <= 1.0


def _speechlike(pitch: float, amplitude: float, modulation: float, seconds=1.5):
    """A crude voiced-speech stand-in: a modulated tone plus one harmonic."""
    t = np.arange(int(SAMPLE_RATE * seconds)) / SAMPLE_RATE
    envelope = 0.5 * (1 + modulation * np.sin(2 * np.pi * 3.5 * t))
    return (
        amplitude
        * envelope
        * (np.sin(2 * np.pi * pitch * t) + 0.4 * np.sin(2 * np.pi * 2 * pitch * t))
    )


def _calibrated_baseline(pitch=180.0, amplitude=0.06, modulation=0.5):
    """A SpeakerBaseline that has seen enough clips to offer verdicts."""
    from amadeus.vocal_emotion import SpeakerBaseline

    baseline = SpeakerBaseline()
    for _ in range(SpeakerBaseline.MIN_SAMPLES + 1):
        analyze_heuristic(
            _pcm16(_speechlike(pitch, amplitude, modulation)),
            SAMPLE_RATE,
            baseline=baseline,
        )
    assert baseline.ready
    return baseline


def test_heuristic_without_baseline_never_claims_a_deviation_label():
    """The uncalibrated path may only say flat/neutral.

    Regression pin for the 2026-08-05 finding: absolute pitch thresholds
    made 16 of 17 voice-acted reference clips read "bright" at an identical
    confidence — the analyzer was reporting vocal range (who is speaking),
    not affect. With no baseline there is no reference point, so a
    deviation verdict would be a guess.
    """
    for pitch, amplitude in [(300.0, 0.2), (100.0, 0.02), (170.0, 0.25)]:
        result = analyze_heuristic(
            _pcm16(_speechlike(pitch, amplitude, 0.9)), SAMPLE_RATE
        )
        assert result is None or result.label in {"flat", "neutral"}


def test_heuristic_reads_low_below_the_speakers_own_energy_norm():
    baseline = _calibrated_baseline()
    quiet = _speechlike(175.0, 0.012, 0.1)
    result = analyze_heuristic(_pcm16(quiet), SAMPLE_RATE, baseline=baseline)
    assert result is not None
    assert result.label == "low"


def test_heuristic_reads_bright_above_the_speakers_own_pitch_norm():
    baseline = _calibrated_baseline()
    bright = _speechlike(300.0, 0.09, 0.9)
    result = analyze_heuristic(_pcm16(bright), SAMPLE_RATE, baseline=baseline)
    assert result is not None
    assert result.label == "bright"


def test_heuristic_reads_tense_when_loud_and_variable_for_this_speaker():
    baseline = _calibrated_baseline()
    tense = _speechlike(170.0, 0.20, 1.0)
    result = analyze_heuristic(_pcm16(tense), SAMPLE_RATE, baseline=baseline)
    assert result is not None
    assert result.label == "tense"


def test_same_clip_reads_differently_for_two_different_speaker_norms():
    """The point of the baseline: one clip is 'bright' for a low-pitched
    speaker and unremarkable for a naturally high-pitched one."""
    clip = _pcm16(_speechlike(300.0, 0.09, 0.9))
    low_voice = _calibrated_baseline(pitch=150.0)
    high_voice = _calibrated_baseline(pitch=300.0, amplitude=0.09, modulation=0.9)
    for_low = analyze_heuristic(clip, SAMPLE_RATE, baseline=low_voice)
    for_high = analyze_heuristic(clip, SAMPLE_RATE, baseline=high_voice)
    assert for_low is not None and for_low.label == "bright"
    assert for_high is None or for_high.label != "bright"


def test_heuristic_deterministic_across_repeated_calls():
    flat = _tone(150.0, 1.5, 0.08)
    pcm = _pcm16(flat)
    first = analyze_heuristic(pcm, SAMPLE_RATE)
    second = analyze_heuristic(pcm, SAMPLE_RATE)
    assert first == second


def test_heuristic_min_confidence_gate_can_abstain():
    flat = _tone(150.0, 1.5, 0.08)
    pcm = _pcm16(flat)
    permissive = analyze_heuristic(pcm, SAMPLE_RATE, min_confidence=0.0)
    assert permissive is not None
    strict = analyze_heuristic(pcm, SAMPLE_RATE, min_confidence=0.99)
    assert strict is None


def test_heuristic_result_phrase_matches_label_table():
    result = VocalEmotionResult(label="tense", confidence=0.6, source="heuristic")
    assert result.phrase() == "tense"
    result2 = VocalEmotionResult(label="bright", confidence=0.6, source="heuristic")
    assert "energetic" in result2.phrase()


def test_speaker_baseline_is_thread_safe_under_concurrent_clips():
    """The process-wide baseline is mutated from STT thread-executor calls;
    concurrent observe/read must never interleave the append/trim pair
    (a skewed rolling norm) or raise. Pins the lock added to
    SpeakerBaseline: hammer it from several threads and check the window
    stays bounded and both series stay in lockstep."""
    import threading

    from amadeus.vocal_emotion import SpeakerBaseline

    baseline = SpeakerBaseline()
    errors: List[BaseException] = []

    def worker(seed: int) -> None:
        try:
            for i in range(200):
                baseline.observe(180.0 + (seed + i) % 9, 0.05 + ((seed + i) % 5) / 100)
                baseline.pitch_z(200.0)
                baseline.energy_z(0.06)
                _ = baseline.ready
        except BaseException as exc:  # noqa: BLE001 - the assertion target
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(n,)) for n in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)
    assert not errors
    assert baseline.ready
    # Every observe here carried voiced pitch AND energy, so a consistent
    # window means both lists trimmed to exactly WINDOW entries.
    assert len(baseline._pitches) == SpeakerBaseline.WINDOW
    assert len(baseline._energies) == SpeakerBaseline.WINDOW


# ---------------------------------------------------------------------------
# Stash / consume / TTL lifecycle
# ---------------------------------------------------------------------------


def _engine(tmp_path: Path) -> ChatEngine:
    config = Config()
    config.memory.persist_path = str(tmp_path / "memory.json")
    config.memory.embeddings_path = None
    return ChatEngine(config, DummyProvider())


def test_analyze_and_stash_then_consume_round_trip(tmp_path):
    engine = _engine(tmp_path)
    flat = _pcm16(_tone(150.0, 1.5, 0.08))
    before = telemetry.snapshot().get("vocal_emotion.analyzed", 0)
    analyze_and_stash(engine, flat, SAMPLE_RATE)
    assert telemetry.snapshot()["vocal_emotion.analyzed"] == before + 1

    consumed_before = telemetry.snapshot().get("vocal_emotion.consumed", 0)
    result = consume_vocal_tone(engine)
    assert result is not None
    assert result.label == "flat"
    assert telemetry.snapshot()["vocal_emotion.consumed"] == consumed_before + 1


def test_consume_is_exactly_once(tmp_path):
    engine = _engine(tmp_path)
    flat = _pcm16(_tone(150.0, 1.5, 0.08))
    analyze_and_stash(engine, flat, SAMPLE_RATE)
    first = consume_vocal_tone(engine)
    assert first is not None
    second = consume_vocal_tone(engine)
    assert second is None


def test_consume_with_nothing_stashed_returns_none(tmp_path):
    engine = _engine(tmp_path)
    assert consume_vocal_tone(engine) is None


def test_stale_entry_is_dropped_and_counted(tmp_path):
    engine = _engine(tmp_path)
    engine.config.vocal_emotion.ttl_seconds = 0.05
    flat = _pcm16(_tone(150.0, 1.5, 0.08))
    analyze_and_stash(engine, flat, SAMPLE_RATE)
    time.sleep(0.1)
    before = telemetry.snapshot().get("vocal_emotion.stale_dropped", 0)
    result = consume_vocal_tone(engine)
    assert result is None
    assert telemetry.snapshot()["vocal_emotion.stale_dropped"] == before + 1


def test_analyze_and_stash_disabled_config_is_noop(tmp_path):
    engine = _engine(tmp_path)
    engine.config.vocal_emotion.enabled = False
    flat = _pcm16(_tone(150.0, 1.5, 0.08))
    analyze_and_stash(engine, flat, SAMPLE_RATE)
    assert consume_vocal_tone(engine) is None


def test_analyze_and_stash_low_confidence_leaves_nothing_stashed(tmp_path):
    engine = _engine(tmp_path)
    engine.config.vocal_emotion.min_confidence = 0.99
    flat = _pcm16(_tone(150.0, 1.5, 0.08))
    analyze_and_stash(engine, flat, SAMPLE_RATE)
    assert consume_vocal_tone(engine) is None


def test_analyze_and_stash_survives_analyzer_exception(tmp_path, monkeypatch):
    engine = _engine(tmp_path)

    def _boom(*_args, **_kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr("amadeus.vocal_tone_stash.analyze", _boom)
    before = telemetry.snapshot().get("vocal_emotion.analyze_failures", 0)
    analyze_and_stash(engine, b"\x00\x00", SAMPLE_RATE)  # must not raise
    assert telemetry.snapshot()["vocal_emotion.analyze_failures"] == before + 1
    assert consume_vocal_tone(engine) is None


# ---------------------------------------------------------------------------
# Prompt line rendering
# ---------------------------------------------------------------------------


def test_prompt_omits_vocal_tone_line_by_default():
    prompt = build_system_prompt(Config().persona)
    assert "sounded" not in prompt


def test_prompt_renders_vocal_tone_line_when_provided():
    prompt = build_system_prompt(Config().persona, vocal_tone="tense")
    assert "sounded tense" in prompt


def test_prompt_vocal_tone_line_uses_exact_phrase():
    prompt = build_system_prompt(Config().persona, vocal_tone="bright and energetic")
    assert "sounded bright and energetic" in prompt


# ---------------------------------------------------------------------------
# Mood bias in prepare_turn (mirrors is_late_night's override-only-default)
# ---------------------------------------------------------------------------


def test_prepare_turn_biases_focused_mood_from_strong_vocal_tone(tmp_path, monkeypatch):
    # Pin the clock: past local midnight prepare_turn's is_late_night bias
    # flips "focused" to "tired" FIRST, and the vocal bias (which only
    # nudges a still-"focused" mood) never runs — a 02:00 UTC CI run must
    # not change what this test observes.
    monkeypatch.setattr("amadeus.engine.is_late_night", lambda: False)
    engine = _engine(tmp_path)
    assert engine.kurisu_state.normalized_mood() == "focused"
    # Stash a verdict directly: what's under test is the mood bias, not the
    # analyzer (which now needs a calibrated speaker baseline to say
    # "tense" at all — see the heuristic tests above).
    from amadeus.vocal_tone_stash import _VocalToneEntry

    engine._vocal_tone = _VocalToneEntry(
        result=VocalEmotionResult(label="tense", confidence=0.6, source="heuristic"),
        expires_at=time.monotonic() + 30,
    )
    engine.prepare_turn("hey", engine.config.persona)
    assert engine.kurisu_state.mood == "annoyed"


def test_prepare_turn_biases_mood_from_model_emotion_labels(tmp_path, monkeypatch):
    """The model label set maps onto moods too (opt-in mode: "model")."""
    monkeypatch.setattr("amadeus.engine.is_late_night", lambda: False)
    from amadeus.vocal_tone_stash import _VocalToneEntry

    for label, expected in [("angry", "annoyed"), ("sad", "affectionate")]:
        engine = _engine(tmp_path)
        engine._vocal_tone = _VocalToneEntry(
            result=VocalEmotionResult(label=label, confidence=0.9, source="model"),
            expires_at=time.monotonic() + 30,
        )
        engine.prepare_turn("hey", engine.config.persona)
        assert engine.kurisu_state.mood == expected


def test_prepare_turn_vocal_tone_never_overrides_nonfocused_mood(tmp_path):
    engine = _engine(tmp_path)
    engine.kurisu_state.mood = "affectionate"
    tense_pcm = _pcm16(
        0.45
        * (np.sin(2 * np.pi * 4 * np.arange(int(SAMPLE_RATE * 1.5)) / SAMPLE_RATE) > 0)
        * np.random.default_rng(0).uniform(-1, 1, int(SAMPLE_RATE * 1.5))
    )
    analyze_and_stash(engine, tense_pcm, SAMPLE_RATE)
    engine.prepare_turn("hey", engine.config.persona)
    assert engine.kurisu_state.mood == "affectionate"


def test_prepare_turn_consumes_stash_exactly_once(tmp_path):
    engine = _engine(tmp_path)
    flat = _pcm16(_tone(150.0, 1.5, 0.08))
    analyze_and_stash(engine, flat, SAMPLE_RATE)
    engine.prepare_turn("first turn", engine.config.persona)
    assert engine._vocal_tone is None
    # A second turn with nothing fresh stashed renders no vocal-tone line.
    setup = engine.prepare_turn("second turn", engine.config.persona)
    assert "sounded" not in setup.system_prompt


def test_prepare_turn_with_no_stash_is_byte_identical_to_before_feature(tmp_path):
    engine = _engine(tmp_path)
    setup = engine.prepare_turn("hello", engine.config.persona)
    assert "sounded" not in setup.system_prompt


# ---------------------------------------------------------------------------
# STT-endpoint wiring (SessionStore + _SpeculativeSTT), mocked provider
# ---------------------------------------------------------------------------


class FakeSTT:
    def __init__(self, text: str) -> None:
        self.text = text
        self.calls: List[tuple] = []

    def transcribe_audio(self, audio_data: bytes, sample_rate: int) -> str:
        self.calls.append((audio_data, sample_rate))
        return self.text


class CountingEmbedder:
    def embed_text(self, text: str):
        return [1.0, 0.0]

    def similarity(self, a, b):
        return 1.0


def _store_config(tmp_path) -> Config:
    config = Config()
    config.memory.persist_path = str(tmp_path / "memory.json")
    config.memory.embeddings_path = None
    config.memory.profile_path = str(tmp_path / "profile.json")
    config.memory.transcripts_path = str(tmp_path / "transcripts.json")
    config.audio.enable_voice = False
    config.memory.speculative_recall = False  # isolate the vocal-tone path
    return config


def _patched_store(tmp_path, monkeypatch, config, stt=None):
    import amadeus.web.app as app_module
    from amadeus.web.sessions import SessionStore

    monkeypatch.setattr(app_module, "load_llm", lambda cfg: DummyProvider())
    monkeypatch.setattr(app_module, "load_tts", lambda cfg: None)
    monkeypatch.setattr(app_module, "load_embedder", lambda cfg: CountingEmbedder())
    monkeypatch.setattr(app_module, "load_stt", lambda cfg: stt)
    return SessionStore(config)


def test_stt_wrapper_stashes_vocal_tone_for_most_recent_session(tmp_path, monkeypatch):
    fake = FakeSTT("hello there")
    store = _patched_store(tmp_path, monkeypatch, _store_config(tmp_path), stt=fake)
    _, session = store.get(None, None)

    stt = store.get_stt()
    flat = _pcm16(_tone(150.0, 1.5, 0.08))
    assert stt.transcribe_audio(flat, SAMPLE_RATE) == "hello there"
    assert fake.calls == [(flat, SAMPLE_RATE)]

    assert session._vocal_tone is not None
    result = consume_vocal_tone(session)
    assert result is not None
    assert result.label == "flat"


def test_stt_wrapper_flag_off_leaves_no_stash(tmp_path, monkeypatch):
    config = _store_config(tmp_path)
    config.vocal_emotion.enabled = False
    fake = FakeSTT("hello there")
    store = _patched_store(tmp_path, monkeypatch, config, stt=fake)
    _, session = store.get(None, None)

    stt = store.get_stt()
    flat = _pcm16(_tone(150.0, 1.5, 0.08))
    stt.transcribe_audio(flat, SAMPLE_RATE)
    assert session._vocal_tone is None


def test_stt_wrapper_no_live_session_is_a_noop(tmp_path, monkeypatch):
    fake = FakeSTT("talking before any chat exists")
    store = _patched_store(tmp_path, monkeypatch, _store_config(tmp_path), stt=fake)

    stt = store.get_stt()
    flat = _pcm16(_tone(150.0, 1.5, 0.08))
    # Must not raise even with no live session to target.
    assert stt.transcribe_audio(flat, SAMPLE_RATE) == "talking before any chat exists"


def test_stt_wrapper_analysis_failure_does_not_break_transcription(
    tmp_path, monkeypatch
):
    fake = FakeSTT("hello there")
    store = _patched_store(tmp_path, monkeypatch, _store_config(tmp_path), stt=fake)
    store.get(None, None)

    def _boom(*_args, **_kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(store, "analyze_vocal_tone", _boom)
    stt = store.get_stt()
    before = telemetry.snapshot().get("vocal_emotion.analyze_failures", 0)
    text = stt.transcribe_audio(b"\x00\x00", SAMPLE_RATE)
    assert text == "hello there"
    assert telemetry.snapshot()["vocal_emotion.analyze_failures"] == before + 1


# ---------------------------------------------------------------------------
# Model-upgrade path (fake ONNX-shaped model, no onnxruntime/model download)
# ---------------------------------------------------------------------------


def test_model_unavailable_counted_once_and_falls_back_to_heuristic(
    tmp_path, monkeypatch
):
    from amadeus import vocal_emotion as ve

    monkeypatch.setattr(ve, "_CACHE", {"model": None, "resolved": False})
    monkeypatch.setenv(ve.MODEL_DIR_ENV, str(tmp_path / "missing"))
    config = Config()
    before = telemetry.snapshot().get("vocal_emotion.model_unavailable", 0)
    assert ve.load_vocal_emotion_model(config) is None
    assert telemetry.snapshot()["vocal_emotion.model_unavailable"] == before + 1
    # Cached miss: a second call re-probes nothing and counts nothing more.
    assert ve.load_vocal_emotion_model(config) is None
    assert telemetry.snapshot()["vocal_emotion.model_unavailable"] == before + 1

    # analyze() still works end to end via the heuristic fallback.
    flat = _pcm16(_tone(150.0, 1.5, 0.08))
    result = ve.analyze(flat, SAMPLE_RATE, config)
    assert result is not None
    assert result.source == "heuristic"


def test_heuristic_mode_never_probes_model_directory(tmp_path, monkeypatch):
    from amadeus import vocal_emotion as ve

    monkeypatch.setattr(ve, "_CACHE", {"model": None, "resolved": False})
    config = Config()
    config.vocal_emotion.mode = "heuristic"
    assert ve.load_vocal_emotion_model(config) is None
    assert ve._CACHE["resolved"] is False  # never even attempted discovery


class FakeOnnxModel:
    """Stands in for VocalEmotionModel without importing onnxruntime."""

    def __init__(self, label: str, confidence: float) -> None:
        self.label = label
        self.confidence = confidence

    def __call__(self, pcm16: bytes, sample_rate: int):
        return VocalEmotionResult(
            label=self.label, confidence=self.confidence, source="model"
        )


def test_analyze_auto_mode_does_not_use_the_model(monkeypatch):
    """ "auto" is heuristic-only — the model is opt-in via mode: "model".

    Pins the 2026-08-05 calibration finding: emotion2vec+ base returned six
    different emotions and zero "neutral" across 17 ordinary spoken lines at
    a median confidence of 0.997, so min_confidence cannot gate it. An
    "auto" default that silently preferred it would put confidently-wrong
    tone hints in the prompt on most voice turns.
    """
    from amadeus import vocal_emotion as ve

    called = []

    def _tracking_loader(cfg):
        called.append(cfg)
        return FakeOnnxModel("bright", 0.99)

    monkeypatch.setattr(ve, "load_vocal_emotion_model", _tracking_loader)
    config = Config()
    assert config.vocal_emotion.mode == "auto"
    result = ve.analyze(_pcm16(_speechlike(180.0, 0.06, 0.5)), SAMPLE_RATE, config)
    assert called == []  # never even loaded
    assert result is None or result.source == "heuristic"


def test_analyze_model_mode_uses_the_model(monkeypatch):
    from amadeus import vocal_emotion as ve

    fake_model = FakeOnnxModel("bright", 0.9)
    monkeypatch.setattr(ve, "load_vocal_emotion_model", lambda cfg: fake_model)
    config = Config()
    config.vocal_emotion.mode = "model"
    result = ve.analyze(b"\x00\x00" * 1000, SAMPLE_RATE, config)
    assert result is not None
    assert result.source == "model"
    assert result.label == "bright"


def test_analyze_model_mode_does_not_fall_back_on_low_confidence(monkeypatch):
    from amadeus import vocal_emotion as ve

    fake_model = FakeOnnxModel("bright", 0.1)
    monkeypatch.setattr(ve, "load_vocal_emotion_model", lambda cfg: fake_model)
    config = Config()
    config.vocal_emotion.mode = "model"
    result = ve.analyze(b"\x00\x00" * 1000, SAMPLE_RATE, config)
    assert result is None


def test_analyze_auto_mode_falls_back_to_heuristic_on_low_model_confidence(
    monkeypatch,
):
    from amadeus import vocal_emotion as ve

    fake_model = FakeOnnxModel("bright", 0.1)
    monkeypatch.setattr(ve, "load_vocal_emotion_model", lambda cfg: fake_model)
    config = Config()
    config.vocal_emotion.mode = "auto"
    flat = _pcm16(_tone(150.0, 1.5, 0.08))
    result = ve.analyze(flat, SAMPLE_RATE, config)
    assert result is not None
    assert result.source == "heuristic"
