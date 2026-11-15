"""Speaker-consistency guard for zero-shot voice cloning.

Every zero-shot generation is an independent sample, and some rolls land on
a different speaker entirely (heard as the voice flipping to a stranger
mid-conversation). Sampling temperature reduces but cannot eliminate it.
This guard makes drift detectable and correctable: embed the reference clip
once with a speaker encoder (resemblyzer's d-vector model, CPU, ~50ms per
clip), score every generated clip's cosine similarity against it, and let
the caller regenerate when a clip scores below threshold.

The guard is strictly best-effort: any failure inside it must degrade to
"no guard" rather than break synthesis.
"""

from __future__ import annotations

import logging
from typing import Callable, Optional, Tuple

import numpy as np

from ... import telemetry

logger = logging.getLogger(__name__)

_ENCODER_RATE = 16000
# resemblyzer embeds 1.6s windows; tile shorter clips so short interjections
# ("そうね。") still embed instead of raising.
_MIN_SAMPLES = int(1.7 * _ENCODER_RATE)

GenerateFn = Callable[[], Tuple[np.ndarray, int]]


class SpeakerGuard:
    """Reference-anchored speaker similarity scoring with retry policy."""

    def __init__(
        self,
        ref_audio_path: str,
        *,
        threshold: float = 0.84,
        short_clip_threshold: float = 0.70,
        short_clip_seconds: float = 2.0,
        max_retries: int = 2,
        encoder=None,
    ) -> None:
        # Calibrated on the Kurisu clone (2026-07): same-voice sentences
        # score 0.88-0.94, but clips under ~2s embed noisily (tiled to the
        # encoder's window) and legitimately land around 0.62-0.68 — a flat
        # threshold would retry every short interjection for nothing.
        #
        # Re-calibrated 2026-07-28 against the 16 other character reference
        # voices (all similar female anime voices — the realistic drift
        # targets): the closest impostors score 0.798-0.820 against the
        # Kurisu ref at FULL length, so the original 0.78 threshold passed
        # near-voice drift entirely; 0.84 sits between impostor-max (0.82)
        # and the true-voice band (0.88-0.94). Short clips separate less:
        # at 0.5-1.0s true/impostor ranges overlap, at 1.5s+ true-p10
        # (0.74) clears impostor-max (0.72). 0.70 for shorts trades a
        # ~10-15% needless-retry rate on real short lines for catching the
        # impostor median (0.60) and most of its upper tail.
        self.threshold = threshold
        self.short_clip_threshold = short_clip_threshold
        self.short_clip_seconds = short_clip_seconds
        self.max_retries = max_retries
        if encoder is None:
            from resemblyzer import VoiceEncoder

            encoder = VoiceEncoder("cpu", verbose=False)
        self._encoder = encoder
        self._ref_embedding = self._embed_file(ref_audio_path)

    # ---- embedding ------------------------------------------------------

    @staticmethod
    def _to_encoder_audio(samples: np.ndarray, sample_rate: int) -> np.ndarray:
        wav = np.asarray(samples, dtype=np.float32).reshape(-1)
        if sample_rate != _ENCODER_RATE:
            import soxr

            wav = soxr.resample(wav, sample_rate, _ENCODER_RATE)
        peak = float(np.max(np.abs(wav))) if wav.size else 0.0
        if peak > 0:
            wav = wav / max(peak, 1e-6) * 0.95
        if wav.size and wav.size < _MIN_SAMPLES:
            wav = np.tile(wav, int(np.ceil(_MIN_SAMPLES / wav.size)))[:_MIN_SAMPLES]
        return wav

    def _embed(self, samples: np.ndarray, sample_rate: int) -> np.ndarray:
        wav = self._to_encoder_audio(samples, sample_rate)
        return self._encoder.embed_utterance(wav)

    def _embed_file(self, path: str) -> np.ndarray:
        import soundfile as sf

        data, sample_rate = sf.read(path, dtype="float32")
        if data.ndim > 1:
            data = data.mean(axis=1)
        return self._embed(data, sample_rate)

    # ---- scoring --------------------------------------------------------

    def similarity(self, samples: np.ndarray, sample_rate: int) -> float:
        """Cosine similarity vs the reference (embeddings are L2-normalized)."""
        embedding = self._embed(samples, sample_rate)
        return float(np.dot(embedding, self._ref_embedding))

    def threshold_for(self, duration_seconds: float) -> float:
        """Effective pass mark for a clip of this length."""
        if duration_seconds < self.short_clip_seconds:
            return self.short_clip_threshold
        return self.threshold

    def best_of(
        self, generate: GenerateFn, *, label: str = ""
    ) -> Tuple[np.ndarray, int, Optional[float]]:
        """Generate until a clip clears the threshold, else keep the best.

        Runs ``generate`` up to 1 + max_retries times. Returns
        (samples, sample_rate, score); score is None when scoring itself
        failed (in which case the first clip is returned unjudged).
        """
        best: Optional[Tuple[np.ndarray, int, float]] = None
        for attempt in range(1 + self.max_retries):
            samples, sample_rate = generate()
            try:
                score = self.similarity(samples, sample_rate)
            except Exception:  # scoring must never break synthesis
                logger.exception("speaker guard scoring failed; passing clip through")
                return samples, sample_rate, None
            if best is None or score > best[2]:
                best = (samples, sample_rate, score)
            mark = self.threshold_for(
                np.asarray(samples).reshape(-1).size / sample_rate
            )
            if score >= mark:
                return best
            telemetry.incr("tts.speaker_drift_retries")
            logger.warning(
                "speaker drift%s: similarity %.3f < %.2f (attempt %d/%d)",
                f" [{label}]" if label else "",
                score,
                mark,
                attempt + 1,
                1 + self.max_retries,
            )
        assert best is not None
        # Degrade loudly: an exhausted guard means the user may HEAR a
        # wrong-voice clip — the cache layer refuses to store it (see
        # CachedTTSProvider), so it at least can't become permanent.
        telemetry.incr("tts.speaker_guard_exhausted")
        logger.warning(
            "speaker guard exhausted retries%s; keeping best clip (%.3f)",
            f" [{label}]" if label else "",
            best[2],
        )
        return best


def build_speaker_guard(
    ref_audio_path: str, *, threshold: float, max_retries: int = 2
) -> Optional[SpeakerGuard]:
    """Best-effort factory: returns None (with a log line) when the optional
    resemblyzer dependency or the reference embed fails."""
    try:
        return SpeakerGuard(
            ref_audio_path, threshold=threshold, max_retries=max_retries
        )
    except Exception as exc:
        logger.warning(
            "Speaker guard unavailable (%s); voice-drift detection disabled. "
            "pip install --no-deps resemblyzer webrtcvad to enable.",
            exc,
        )
        return None
