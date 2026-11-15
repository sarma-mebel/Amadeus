"""faster-whisper speech recognition provider."""

from __future__ import annotations


from .base import STTProvider


class FasterWhisperProvider(STTProvider):
    """Wrapper around ``faster-whisper`` for local transcription."""

    # "small" (~460MB first-use download): "base" transcribed wrongly often
    # enough to break trust in voice input (live finding, 2026-08-03
    # shakedown) — for a voice-first daily driver, accuracy beats the
    # smaller download. Configurable via providers.stt_model.
    def __init__(self, model_size: str = "small", device: str = "cpu") -> None:
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("faster-whisper package is required for STT") from exc

        self.model = WhisperModel(model_size, device=device)

    def transcribe_file(self, path: str) -> str:
        segments, _ = self.model.transcribe(path)
        return " ".join(segment.text for segment in segments)

    def transcribe_audio(self, audio_data: bytes, sample_rate: int) -> str:
        import numpy as np

        audio = np.frombuffer(audio_data, dtype=np.int16).astype("float32") / 32768.0
        # beam_size 5 (whisper's own default) over greedy: decoding is a
        # small share of utterance latency, and greedy search was part of
        # the same mis-transcription complaint as the base model.
        segments, _ = self.model.transcribe(audio, language="en", beam_size=5)
        return " ".join(segment.text for segment in segments)
