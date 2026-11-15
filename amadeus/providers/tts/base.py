"""Text-to-speech provider abstractions."""

from __future__ import annotations

from typing import Protocol


class TTSProvider(Protocol):
    """Synthesize speech from text.

    Optional extensions, feature-detected by callers (same idiom as
    ``synthesize_async``/``synthesize_stream`` — the attribute must be
    absent, not None, when unsupported):

    - ``available_emotions: frozenset[str]`` — emotion labels the provider
      can voice distinctly. Callers pass ``emotion=<label>`` to
      ``synthesize``/``synthesize_stream`` ONLY for labels in this set, so
      providers without the kwarg never receive it.
    """

    def synthesize(
        self, text: str, *, speaking_rate: float = 1.0, pitch_shift: float = 0.0
    ) -> bytes:  # pragma: no cover
        ...
