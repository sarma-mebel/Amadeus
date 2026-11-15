"""Offline pyttsx3 text-to-speech."""

from __future__ import annotations

import tempfile

from .base import TTSProvider


class Pyttsx3Provider(TTSProvider):
    """Use ``pyttsx3`` to synthesize speech offline."""

    def __init__(self, voice: str | None = None) -> None:
        try:
            import pyttsx3
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("pyttsx3 package is required for offline TTS") from exc

        self.engine = pyttsx3.init()
        if voice:
            self.engine.setProperty("voice", voice)

    def synthesize(
        self, text: str, *, speaking_rate: float = 1.0, pitch_shift: float = 0.0
    ) -> bytes:
        self.engine.setProperty("rate", int(200 * speaking_rate))
        self.engine.setProperty("pitch", pitch_shift)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
        self.engine.save_to_file(text, tmp_path)
        self.engine.runAndWait()
        with open(tmp_path, "rb") as handle:
            data = handle.read()
        try:
            import os

            os.remove(tmp_path)
        except OSError:  # pragma: no cover - best effort cleanup
            pass
        return data
