"""OpenAI text-to-speech provider."""

from __future__ import annotations

from .base import TTSProvider


class OpenAITTSProvider(TTSProvider):
    """Use OpenAI's TTS endpoint."""

    def __init__(
        self,
        voice: str = "alloy",
        model: str = "gpt-4o-mini-tts",
        api_key: str | None = None,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "openai package is required for the OpenAI TTS provider"
            ) from exc

        self.client = OpenAI(api_key=api_key)
        self.voice = voice
        self.model = model

    def synthesize(
        self, text: str, *, speaking_rate: float = 1.0, pitch_shift: float = 0.0
    ) -> bytes:
        response = self.client.audio.speech.create(
            model=self.model,
            voice=self.voice,
            input=text,
            response_format="wav",
            speed=speaking_rate,
        )
        return response.read()
