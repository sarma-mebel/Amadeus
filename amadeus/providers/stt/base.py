"""Speech-to-text provider abstractions."""

from __future__ import annotations

from typing import Protocol


class STTProvider(Protocol):
    """Convert audio chunks to text."""

    def transcribe_file(self, path: str) -> str:  # pragma: no cover - interface
        ...

    def transcribe_audio(
        self, audio_data: bytes, sample_rate: int
    ) -> str:  # pragma: no cover - interface
        ...


class DummySTT:
    """Fallback STT provider for text-only sessions."""

    def transcribe_file(self, path: str) -> str:
        raise RuntimeError("Dummy STT cannot transcribe files")

    def transcribe_audio(self, audio_data: bytes, sample_rate: int) -> str:
        raise RuntimeError("Dummy STT cannot transcribe audio")
