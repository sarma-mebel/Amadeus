"""Semantic end-of-turn prediction for the duplex loop.

The energy VAD's silence hangover can't tell "finished a thought" from
"paused mid-thought" — it will step on "so I was thinking... [pause]
...maybe". smart-turn-v3 (pipecat-ai, BSD-2) is an ~8M-param int8 ONNX
model that answers "is this utterance finished?" from the audio itself
(intonation, trailing-off — fed as Whisper log-mel features computed by
the numpy frontend below) in ~100ms on CPU, so it can sit directly on the
endpoint decision: an "incomplete" verdict holds the endpoint open a
while longer (``duplex_smart_turn_hold_ms``) instead of cutting the user
off.

Everything here is optional and degrade-loudly: no onnxruntime, no model
file, or a broken model all fall back to the plain energy-VAD endpoint and
count ``web.duplex_smart_turn_unavailable`` / ``_errors`` — never a dead
call loop. Fetch the model with ``python scripts/fetch_smart_turn.py``.
"""

from __future__ import annotations

import os
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any, Optional

from ... import telemetry
from ...logging_utils import get_logger

LOGGER = get_logger(__name__)

# Auto-discovery directory (any *.onnx inside is taken as the model, so a
# future v3.1 file drop needs no config change); overridable via config
# (web.duplex_smart_turn_model) or env.
DEFAULT_MODEL_DIR = Path("models") / "smart_turn"
MODEL_DIR_ENV = "AMADEUS_SMART_TURN_DIR"

# smart-turn models are trained on 16 kHz mono windows of ~8s; the tail is
# what carries the endpoint signal, so overlong audio keeps its END.
TARGET_SAMPLE_RATE = 16000
WINDOW_SECONDS = 8
WINDOW_SAMPLES = TARGET_SAMPLE_RATE * WINDOW_SECONDS

# Whisper log-mel parameters (smart-turn v3's input is the standard
# WhisperFeatureExtractor output with chunk_length=8 and do_normalize=True:
# [80 mels x 800 frames], verified against the shipped ONNX input spec
# ['batch_size', 80, 800] on 2026-08-03 — the model does NOT take raw
# audio, which the first blind implementation assumed).
N_FFT = 400
HOP_LENGTH = 160
N_MELS = 80
N_FRAMES = WINDOW_SAMPLES // HOP_LENGTH  # 800

# After this many consecutive inference failures the predictor disables
# itself for the process — one bad model must not tax every endpoint.
MAX_CONSECUTIVE_ERRORS = 3


def _mel_filterbank(np: Any) -> Any:
    """Slaney-scale mel filterbank matching ``librosa.filters.mel(sr=16000,
    n_fft=400, n_mels=80, htk=False, norm="slaney")`` — the filters Whisper
    (and therefore WhisperFeatureExtractor) bakes into its assets. Pure
    numpy so the mel frontend has no librosa/transformers dependency; a
    parity test (tests/test_duplex.py) pins it against the real
    WhisperFeatureExtractor whenever transformers is installed."""

    def hz_to_mel(freq: Any) -> Any:
        freq = np.asarray(freq, dtype=np.float64)
        mel = freq / (200.0 / 3)
        log_region = freq >= 1000.0
        mel = np.where(
            log_region,
            15.0 + np.log(np.maximum(freq, 1e-10) / 1000.0) / np.log(6.4) * 27.0,
            mel,
        )
        return mel

    def mel_to_hz(mel: Any) -> Any:
        mel = np.asarray(mel, dtype=np.float64)
        freq = mel * (200.0 / 3)
        log_region = mel >= 15.0
        return np.where(
            log_region, 1000.0 * np.exp(np.log(6.4) / 27.0 * (mel - 15.0)), freq
        )

    mels = np.linspace(hz_to_mel(0.0), hz_to_mel(TARGET_SAMPLE_RATE / 2), N_MELS + 2)
    freqs = mel_to_hz(mels)
    fft_freqs = np.fft.rfftfreq(N_FFT, 1.0 / TARGET_SAMPLE_RATE)
    fdiff = np.diff(freqs)
    ramps = freqs.reshape(-1, 1) - fft_freqs.reshape(1, -1)
    weights = np.zeros((N_MELS, fft_freqs.size))
    for i in range(N_MELS):
        lower = -ramps[i] / fdiff[i]
        upper = ramps[i + 2] / fdiff[i + 1]
        weights[i] = np.maximum(0.0, np.minimum(lower, upper))
    enorm = 2.0 / (freqs[2 : N_MELS + 2] - freqs[:N_MELS])
    return (weights * enorm[:, None]).astype(np.float32)


def log_mel_features(audio: Any, real_length: int, np: Any, filters: Any) -> Any:
    """Whisper log-mel spectrogram of an already-padded 8s float32 array.

    Mirrors WhisperFeatureExtractor(chunk_length=8, do_normalize=True):
    zero-mean/unit-variance over the REAL samples only (padding stays 0),
    reflect-padded centered STFT (hann 400, hop 160), drop the trailing
    frame (801 -> 800), power -> mel -> log10 -> max-8 clamp -> (x+4)/4.
    """
    real_length = max(1, min(int(real_length), audio.size))
    real = audio[:real_length]
    normed = np.zeros_like(audio)
    normed[:real_length] = (real - real.mean()) / np.sqrt(real.var() + 1e-7)
    padded = np.pad(normed, (N_FFT // 2, N_FFT // 2), mode="reflect")
    window = np.hanning(N_FFT + 1)[:-1].astype(np.float64)
    frame_starts = np.arange(N_FRAMES + 1) * HOP_LENGTH
    frames = np.lib.stride_tricks.sliding_window_view(padded, N_FFT)[frame_starts]
    stft = np.fft.rfft(frames * window, axis=-1)
    # Whisper drops the final frame: [801, 201] -> [800, 201].
    power = (np.abs(stft[:-1]) ** 2).T  # [201, 800]
    mel = filters @ power
    log_spec = np.log10(np.maximum(mel, 1e-10))
    log_spec = np.maximum(log_spec, log_spec.max() - 8.0)
    return ((log_spec + 4.0) / 4.0).astype(np.float32)


class SmartTurnPredictor:
    """Callable wrapper: ``predictor(pcm16, sample_rate) -> Optional[bool]``.

    True = the utterance sounds finished, False = mid-thought, None =
    unavailable/failed (caller endpoints as if the predictor didn't exist).
    """

    def __init__(self, model_path: str) -> None:
        import numpy as np
        import onnxruntime

        self._np = np
        self._session = onnxruntime.InferenceSession(
            str(model_path), providers=["CPUExecutionProvider"]
        )
        spec = self._session.get_inputs()[0]
        self._input_name = spec.name
        # The v3 contract: [batch, 80 mels, 800 frames]. Refuse anything
        # else up front — a shape mismatch must read as "unavailable" at
        # load time, not as a per-call inference error stream.
        dims = [d for d in spec.shape if isinstance(d, int)]
        if dims and dims != [N_MELS, N_FRAMES]:
            raise ValueError(
                f"unexpected smart-turn input shape {spec.shape}; "
                f"expected [batch, {N_MELS}, {N_FRAMES}]"
            )
        self._filters = _mel_filterbank(np)
        self._consecutive_errors = 0

    def __call__(
        self, pcm16: bytes, sample_rate: int = TARGET_SAMPLE_RATE
    ) -> Optional[bool]:
        if self._consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
            return None
        try:
            np = self._np
            audio = np.frombuffer(pcm16, dtype="<i2").astype(np.float32) / 32768.0
            if audio.size == 0:
                return None
            if int(sample_rate) != TARGET_SAMPLE_RATE and sample_rate > 0:
                duration = audio.size / float(sample_rate)
                target_n = max(1, int(duration * TARGET_SAMPLE_RATE))
                audio = np.interp(
                    np.linspace(0.0, audio.size - 1, target_n),
                    np.arange(audio.size),
                    audio,
                ).astype(np.float32)
            if audio.size > WINDOW_SAMPLES:
                audio = audio[-WINDOW_SAMPLES:]  # endpointing lives in the tail
            real_length = audio.size
            if audio.size < WINDOW_SAMPLES:
                audio = np.pad(audio, (0, WINDOW_SAMPLES - audio.size))
            features = log_mel_features(audio, real_length, np, self._filters)
            value = self._session.run(
                None, {self._input_name: features[np.newaxis, :, :]}
            )[0]
            score = float(np.asarray(value).reshape(-1)[0])
            if not 0.0 <= score <= 1.0:  # raw logit → probability
                score = 1.0 / (1.0 + np.exp(-score))
            self._consecutive_errors = 0
            return bool(score > 0.5)
        except Exception as exc:
            self._consecutive_errors += 1
            telemetry.incr("web.duplex_smart_turn_errors")
            LOGGER.warning("smart-turn inference failed: %s", exc)
            if self._consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                LOGGER.warning(
                    "smart-turn disabled for this process after %d failures",
                    self._consecutive_errors,
                )
            return None


# One process-wide worker thread for every ThreadedTurnPredictor: the
# underlying predictor is itself process-wide cached (one ONNX session), the
# consult is checked-once-per-pause, and serializing concurrent duplex
# sessions on a single thread beats oversubscribing CPU inference. Lazy so
# processes that never load a model never spawn the thread.
_INFERENCE_EXECUTOR: Optional[ThreadPoolExecutor] = None


def _inference_executor() -> ThreadPoolExecutor:
    global _INFERENCE_EXECUTOR
    if _INFERENCE_EXECUTOR is None:
        _INFERENCE_EXECUTOR = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="amadeus-smart-turn"
        )
    return _INFERENCE_EXECUTOR


class ThreadedTurnPredictor:
    """Non-blocking adapter around :class:`SmartTurnPredictor`.

    The raw predictor blocks for ~100ms of CPU inference (plus the numpy
    log-mel frontend over up to 8s of audio); called synchronously inside
    ``DuplexController.feed()`` that froze the whole event loop at every
    end-of-utterance pause. This wrapper submits the consult to a shared
    worker thread and returns the ``concurrent.futures.Future`` immediately;
    the controller recognizes the future shape (``.done()``/``.result()``)
    and holds the endpoint "pending" until it resolves, degrading to the
    energy-only endpoint if it never does by the hold deadline. The wrapped
    predictor's own error handling (counters, self-disable) is untouched —
    it runs unchanged, just on the worker thread.
    """

    def __init__(self, predictor: Any) -> None:
        self._predictor = predictor

    def __call__(
        self, pcm16: bytes, sample_rate: int = TARGET_SAMPLE_RATE
    ) -> "Future[Optional[bool]]":
        return _inference_executor().submit(self._predictor, pcm16, sample_rate)


def _discover_model_path(config: Any) -> Optional[Path]:
    configured = getattr(getattr(config, "web", None), "duplex_smart_turn_model", None)
    if configured:
        path = Path(str(configured))
        return path if path.is_file() else None
    directory = Path(os.environ.get(MODEL_DIR_ENV) or DEFAULT_MODEL_DIR)
    if not directory.is_dir():
        return None
    candidates = sorted(directory.glob("*.onnx"))
    return candidates[0] if candidates else None


# Process-wide cache: the model loads once; a definitive miss is also
# remembered so every duplex handshake doesn't re-probe the filesystem.
_CACHE: dict = {"predictor": None, "resolved": False}


def load_smart_turn_predictor(config: Any) -> Optional[SmartTurnPredictor]:
    """The configured predictor, or None (disabled/missing — counted once)."""
    if not getattr(getattr(config, "web", None), "duplex_smart_turn", True):
        return None
    if _CACHE["resolved"]:
        return _CACHE["predictor"]
    _CACHE["resolved"] = True
    path = _discover_model_path(config)
    if path is None:
        telemetry.incr("web.duplex_smart_turn_unavailable")
        LOGGER.info(
            "smart-turn model not found (looked in %s) — duplex endpointing "
            "stays energy-only; run scripts/fetch_smart_turn.py to enable",
            os.environ.get(MODEL_DIR_ENV) or DEFAULT_MODEL_DIR,
        )
        return None
    try:
        _CACHE["predictor"] = SmartTurnPredictor(str(path))
        LOGGER.info("smart-turn endpoint predictor loaded from %s", path)
    except Exception as exc:
        telemetry.incr("web.duplex_smart_turn_unavailable")
        LOGGER.warning("smart-turn model failed to load (%s): %s", path, exc)
        _CACHE["predictor"] = None
    return _CACHE["predictor"]
