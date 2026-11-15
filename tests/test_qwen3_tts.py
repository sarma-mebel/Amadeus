"""Tests for the Qwen3-TTS zero-shot provider (model mocked, no downloads)."""

from __future__ import annotations

import io
import sys
import types
import wave
from unittest import mock

import numpy as np
import pytest


def _recording_base_model():
    model = mock.MagicMock()
    model.create_voice_clone_prompt.return_value = ["cached-prompt-item"]
    model.generate_voice_clone.return_value = (
        [np.zeros(2400, dtype=np.float32)],
        24000,
    )
    return model


@pytest.fixture
def fake_qwen(tmp_path, monkeypatch):
    """Install fake faster_qwen3_tts + qwen_tts modules; return the recording mocks."""

    ref = tmp_path / "ref.wav"
    ref.write_bytes(b"RIFFfake")

    base_model = _recording_base_model()

    # faster_qwen3_tts wrapper: exposes the upstream model as `.model` and
    # mirrors generate_voice_clone on itself.
    wrapper = mock.MagicMock()
    wrapper.model = base_model
    wrapper.generate_voice_clone = base_model.generate_voice_clone

    faster_module = types.ModuleType("faster_qwen3_tts")
    faster_module.FasterQwen3TTS = mock.MagicMock()
    faster_module.FasterQwen3TTS.from_pretrained.return_value = wrapper
    monkeypatch.setitem(sys.modules, "faster_qwen3_tts", faster_module)

    qwen_module = types.ModuleType("qwen_tts")
    qwen_module.Qwen3TTSModel = mock.MagicMock()
    qwen_module.Qwen3TTSModel.from_pretrained.return_value = base_model
    monkeypatch.setitem(sys.modules, "qwen_tts", qwen_module)

    torch_module = types.ModuleType("torch")
    torch_module.bfloat16 = "bfloat16-sentinel"
    monkeypatch.setitem(sys.modules, "torch", torch_module)

    return types.SimpleNamespace(
        model=base_model,
        faster=faster_module,
        module=qwen_module,
        ref=str(ref),
    )


def _make_provider(fake_qwen, **kwargs):
    from amadeus.providers.tts.qwen3_tts_provider import Qwen3TTSProvider

    kwargs.setdefault("warmup", False)
    return Qwen3TTSProvider(ref_audio=fake_qwen.ref, ref_text="リファレンス", **kwargs)


def test_prompt_cached_once_and_reused(fake_qwen):
    provider = _make_provider(fake_qwen)
    provider.synthesize("こんにちは")
    provider.synthesize("元気？")

    fake_qwen.model.create_voice_clone_prompt.assert_called_once_with(
        ref_audio=fake_qwen.ref, ref_text="リファレンス"
    )
    assert fake_qwen.model.generate_voice_clone.call_count == 2
    _, kwargs = fake_qwen.model.generate_voice_clone.call_args
    assert kwargs["voice_clone_prompt"] == ["cached-prompt-item"]
    assert kwargs["language"] == "Japanese"
    assert kwargs["text"] == "元気？"


def test_prefers_faster_engine_and_warms_up(fake_qwen):
    provider = _make_provider(fake_qwen, warmup=True)

    args, kwargs = fake_qwen.faster.FasterQwen3TTS.from_pretrained.call_args
    assert "Qwen3-TTS-12Hz-1.7B-Base" in args[0]
    assert kwargs == {
        "device": "cuda:0",
        "dtype": "bfloat16-sentinel",
        "attn_implementation": "sdpa",
    }
    # The warmup generation captures the CUDA graphs at init.
    assert fake_qwen.model.generate_voice_clone.call_count == 1
    assert "faster" in provider.cache_signature
    # Upstream qwen_tts is not used when the faster engine is available.
    fake_qwen.module.Qwen3TTSModel.from_pretrained.assert_not_called()


def test_falls_back_to_upstream_qwen_tts(fake_qwen, monkeypatch):
    monkeypatch.setitem(sys.modules, "faster_qwen3_tts", None)

    provider = _make_provider(fake_qwen, warmup=True)

    args, kwargs = fake_qwen.module.Qwen3TTSModel.from_pretrained.call_args
    assert "Qwen3-TTS-12Hz-1.7B-Base" in args[0]
    assert kwargs == {"device_map": "cuda:0", "dtype": "bfloat16-sentinel"}
    # No CUDA graphs to capture upstream, so no warmup generation either.
    assert fake_qwen.model.generate_voice_clone.call_count == 0
    assert "upstream" in provider.cache_signature


def test_synthesize_returns_valid_wav(fake_qwen):
    provider = _make_provider(fake_qwen)
    audio = provider.synthesize("テスト")

    with wave.open(io.BytesIO(audio), "rb") as wav_file:
        assert wav_file.getframerate() == 24000
        assert wav_file.getnchannels() == 1
        assert wav_file.getsampwidth() == 2
        assert wav_file.getnframes() == 2400


def test_missing_reference_audio_raises(fake_qwen):
    from amadeus.providers.tts.qwen3_tts_provider import Qwen3TTSProvider

    with pytest.raises(FileNotFoundError):
        Qwen3TTSProvider(ref_audio="does/not/exist.wav", ref_text="x")


def test_load_tts_builds_qwen3_provider(fake_qwen):
    from amadeus.config import Config
    from amadeus.provider_loader import load_tts

    config = Config.from_dict(
        {
            "providers": {
                "tts": "qwen3_tts",
                "qwen3_tts_ref_audio": fake_qwen.ref,
                "qwen3_tts_ref_text": "リファレンス",
            }
        }
    )
    provider = load_tts(config)
    # load_tts wraps concrete providers in the disk TTS cache by default.
    inner = getattr(provider, "provider", provider)

    assert type(inner).__name__ == "Qwen3TTSProvider"
    args, kwargs = fake_qwen.faster.FasterQwen3TTS.from_pretrained.call_args
    # Resolved to a local snapshot dir when cached, else the raw model id.
    assert "Qwen3-TTS-12Hz-1.7B-Base" in args[0]
    assert kwargs["device"] == "cuda:0"


# ---- Emotion-conditioned synthesis (per-emotion reference clips) ----------


def _write_emotion_ref(fake_qwen, label: str, transcript: str | None = "感情の台詞"):
    from pathlib import Path

    ref = Path(fake_qwen.ref)
    wav = ref.with_name(f"{ref.stem}_{label}{ref.suffix}")
    wav.write_bytes(b"RIFFfake")
    if transcript is not None:
        wav.with_suffix(".txt").write_text(transcript, encoding="utf-8")
    return wav


def test_emotion_refs_discovered_with_transcripts(fake_qwen):
    _write_emotion_ref(fake_qwen, "happy", "嬉しい台詞")
    _write_emotion_ref(fake_qwen, "sad", "悲しい台詞")
    # No transcript -> skipped loudly rather than cloned against wrong text.
    _write_emotion_ref(fake_qwen, "angry", transcript=None)

    provider = _make_provider(fake_qwen)

    assert provider.available_emotions == frozenset({"happy", "sad"})
    # Base prompt + one per usable emotion ref.
    assert fake_qwen.model.create_voice_clone_prompt.call_count == 3
    ref_texts = {
        kwargs["ref_text"]
        for _, kwargs in fake_qwen.model.create_voice_clone_prompt.call_args_list
    }
    assert ref_texts == {"リファレンス", "嬉しい台詞", "悲しい台詞"}


def test_synthesize_emotion_selects_matching_prompt(fake_qwen):
    from pathlib import Path

    _write_emotion_ref(fake_qwen, "happy", "嬉しい台詞")
    fake_qwen.model.create_voice_clone_prompt.side_effect = (
        lambda ref_audio, ref_text: f"prompt:{Path(ref_audio).name}"
    )

    provider = _make_provider(fake_qwen)

    provider.synthesize("テスト", emotion="happy")
    _, kwargs = fake_qwen.model.generate_voice_clone.call_args
    assert kwargs["voice_clone_prompt"] == "prompt:ref_happy.wav"

    # Unknown label and no label both fall back to the base prompt.
    provider.synthesize("テスト", emotion="confused")
    _, kwargs = fake_qwen.model.generate_voice_clone.call_args
    assert kwargs["voice_clone_prompt"] == "prompt:ref.wav"
    provider.synthesize("テスト")
    _, kwargs = fake_qwen.model.generate_voice_clone.call_args
    assert kwargs["voice_clone_prompt"] == "prompt:ref.wav"


def test_no_emotion_refs_means_empty_available_set(fake_qwen):
    provider = _make_provider(fake_qwen)
    assert provider.available_emotions == frozenset()
    # Base behavior is untouched: one prompt, plain synthesis works.
    assert fake_qwen.model.create_voice_clone_prompt.call_count == 1
    provider.synthesize("テスト")


def test_upstream_fallback_warns_loudly_and_sets_telemetry(
    fake_qwen, monkeypatch, caplog
):
    """The silent ~10x slowdown must not be silent: prominent WARNING with
    the fix, plus a telemetry marker /api/status can surface."""
    import logging

    from amadeus import telemetry

    telemetry.reset()
    try:
        monkeypatch.setitem(sys.modules, "faster_qwen3_tts", None)
        with caplog.at_level(logging.WARNING):
            _make_provider(fake_qwen)

        warnings = [
            record.getMessage()
            for record in caplog.records
            if record.levelno == logging.WARNING
        ]
        assert any(
            "10x" in message and "pip install faster-qwen3-tts" in message
            for message in warnings
        ), warnings
        assert telemetry.snapshot().get("tts.qwen3_slow_fallback") == 1
    finally:
        telemetry.reset()


def test_fast_engine_clears_slow_fallback_marker(fake_qwen):
    """A provider rebuild that finds the fast engine resets a stale marker
    left by an earlier slow-engine load in the same process."""
    from amadeus import telemetry

    telemetry.reset()
    try:
        telemetry.set_gauge("tts.qwen3_slow_fallback", 1)
        _make_provider(fake_qwen)
        assert telemetry.snapshot().get("tts.qwen3_slow_fallback") == 0
    finally:
        telemetry.reset()
