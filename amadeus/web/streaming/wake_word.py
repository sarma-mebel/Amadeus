"""Local wake-word detection for WS /api/wake.

The client streams the same 16 kHz PCM16 mono frames the duplex socket uses
(audio/duplexMic.ts is literally reused), the server runs an openwakeword
ONNX model per 80 ms hop, and a detection emits one ``{"type": "wake",
"score": ...}`` frame — the client then closes this socket and starts the
normal duplex call. Detection is entirely local (three small ONNX graphs);
audio is never stored, transcribed, or forwarded anywhere.

Same degrade pattern as smart_turn.py next door: heavy imports live inside
``WakeWordDetector.__init__``, a missing package/model resolves to None with
a counter (``web.wake_word_unavailable``) and a log line pointing at
scripts/fetch_wake_word.py, and repeated inference errors self-disable the
detector rather than spamming per-chunk warnings. Unlike smart-turn the
DETECTOR is per-connection (openwakeword models keep internal streaming
buffers), so only the model-path resolution is cached process-wide.

All timing (the refractory window) is derived from AUDIO time — samples
fed — never the wall clock, so tests are deterministic (same convention as
DuplexController).

Wire protocol::

    client: {"type": "start"}                      # one JSON text frame
    server: {"type": "ready", "sample_rate": 16000}
    client: <binary PCM16 mono frames> ...
    server: {"type": "wake", "score": 0.87}        # on each detection
    client: {"type": "stop"}                       # or just closes

Gate/token/error-frame handling mirrors the duplex endpoint in
routes_chat.py; ``WakeSocketRunner`` below holds everything after the
route-level gates so tests can drive it with a fake websocket.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, List, Optional

from ... import telemetry
from ...logging_utils import get_logger

LOGGER = get_logger(__name__)

DEFAULT_MODEL_DIR = Path("models") / "wake_word"
MODEL_DIR_ENV = "AMADEUS_WAKE_WORD_DIR"
TARGET_SAMPLE_RATE = 16000
# openwakeword's native hop: 80 ms of 16 kHz audio per prediction.
CHUNK_SAMPLES = 1280
MAX_CONSECUTIVE_ERRORS = 3

# openwakeword's shared feature-extractor graphs — never wake models
# themselves, so the model discovery glob must skip them (they live in the
# same directory when fetched by scripts/fetch_wake_word.py).
_FEATURE_MODEL_NAMES = {"melspectrogram.onnx", "embedding_model.onnx"}


class WakeWordDetector:
    """Streaming detector over one openwakeword model.

    ``feed(pcm16)`` buffers arbitrary-sized PCM16 frames, runs the model per
    80 ms hop, and returns the score for a NEW detection (>= threshold,
    outside the refractory window) — else None. Raises ImportError/OSError
    out of ``__init__`` when the package or model is unusable; per-chunk
    inference errors are counted and self-disable the detector after
    MAX_CONSECUTIVE_ERRORS (returning None forever) so a broken runtime
    can't log-spam at 12.5 Hz.
    """

    def __init__(
        self,
        model_path: str,
        threshold: float = 0.5,
        refractory_seconds: float = 2.0,
    ) -> None:
        import numpy as np  # base dep; here for symmetry with onnx imports
        from openwakeword.model import Model

        self._np = np
        path = Path(model_path)
        kwargs: dict = {
            "wakeword_models": [str(path)],
            "inference_framework": "onnx",
        }
        # Prefer feature models fetched next to the wake model; without them
        # openwakeword falls back to its own package resources (which
        # scripts/fetch_wake_word.py also populates).
        melspec = path.parent / "melspectrogram.onnx"
        embedding = path.parent / "embedding_model.onnx"
        if melspec.is_file() and embedding.is_file():
            kwargs["melspec_model_path"] = str(melspec)
            kwargs["embedding_model_path"] = str(embedding)
        self._model = Model(**kwargs)
        self.threshold = float(threshold)
        self._refractory_samples = max(0.0, float(refractory_seconds)) * float(
            TARGET_SAMPLE_RATE
        )
        self._buffer = np.zeros(0, dtype=np.int16)
        self._samples_seen = 0
        self._last_detection_at: Optional[int] = None
        self._consecutive_errors = 0
        self._disabled = False

    def feed(self, pcm16: bytes) -> Optional[float]:
        if self._disabled or not pcm16:
            return None
        np = self._np
        chunk = np.frombuffer(pcm16, dtype=np.int16)
        self._buffer = np.concatenate([self._buffer, chunk])
        detected: Optional[float] = None
        while self._buffer.size >= CHUNK_SAMPLES:
            hop = self._buffer[:CHUNK_SAMPLES]
            self._buffer = self._buffer[CHUNK_SAMPLES:]
            self._samples_seen += CHUNK_SAMPLES
            try:
                prediction = self._model.predict(hop)
                self._consecutive_errors = 0
            except Exception as exc:
                self._consecutive_errors += 1
                telemetry.incr("web.wake_word_errors")
                if self._consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                    self._disabled = True
                    LOGGER.warning(
                        "wake-word model failed %d times in a row (%s) — "
                        "disabling for this connection",
                        self._consecutive_errors,
                        exc,
                    )
                continue
            score = max(float(v) for v in prediction.values()) if prediction else 0.0
            if score < self.threshold:
                continue
            in_refractory = (
                self._last_detection_at is not None
                and (self._samples_seen - self._last_detection_at)
                < self._refractory_samples
            )
            if in_refractory:
                # Still refresh the anchor: a continuing phrase keeps the
                # window open rather than firing on its own tail.
                self._last_detection_at = self._samples_seen
                continue
            self._last_detection_at = self._samples_seen
            detected = max(score, detected or 0.0)
        return detected


# Model-path resolution is cached process-wide (a definitive miss included)
# so per-connection detector construction never re-walks the filesystem —
# same _CACHE convention as smart_turn.py. Detectors themselves are
# per-connection: openwakeword models carry internal streaming buffers.
_CACHE: dict = {"path": None, "resolved": False}


def _discover_model_path(config: Any) -> Optional[Path]:
    explicit = getattr(getattr(config, "wake_word", None), "model", None)
    if explicit:
        path = Path(explicit)
        return path if path.is_file() else None
    directory = Path(os.environ.get(MODEL_DIR_ENV) or DEFAULT_MODEL_DIR)
    if not directory.is_dir():
        return None
    candidates: List[Path] = sorted(
        p
        for p in directory.glob("*.onnx")
        if p.name.lower() not in _FEATURE_MODEL_NAMES
    )
    return candidates[0] if candidates else None


def create_wake_word_detector(config: Any) -> Optional[WakeWordDetector]:
    """Fresh per-connection detector, or None (counted + logged) when the
    package or model is unavailable. Never raises."""
    settings = getattr(config, "wake_word", None)
    if settings is None or not getattr(settings, "enabled", False):
        return None
    if not _CACHE["resolved"]:
        _CACHE["resolved"] = True
        _CACHE["path"] = _discover_model_path(config)
        if _CACHE["path"] is None:
            LOGGER.info(
                "wake-word model not found (looked in %s) — wake word stays "
                "off; run scripts/fetch_wake_word.py to enable",
                getattr(settings, "model", None)
                or os.environ.get(MODEL_DIR_ENV)
                or DEFAULT_MODEL_DIR,
            )
    path = _CACHE["path"]
    if path is None:
        telemetry.incr("web.wake_word_unavailable")
        return None
    try:
        return WakeWordDetector(
            str(path),
            threshold=float(getattr(settings, "threshold", 0.5) or 0.5),
            refractory_seconds=float(
                getattr(settings, "refractory_seconds", 2.0) or 2.0
            ),
        )
    except Exception as exc:
        telemetry.incr("web.wake_word_unavailable")
        LOGGER.warning("wake-word detector failed to load (%s): %s", path, exc)
        return None


class WakeSocketRunner:
    """Everything after the route-level gates (accept/enabled/token), shaped
    like DuplexSocketRunner so tests drive it with the same fake-websocket
    pattern. ``detector_factory`` is the injection seam."""

    def __init__(self, websocket: Any, config: Any, *, detector_factory=None) -> None:
        self.websocket = websocket
        self.config = config
        self._detector_factory = detector_factory or create_wake_word_detector

    async def run(self) -> None:
        from ..blocking import run_blocking

        # Handshake: one JSON text frame ({"type": "start"}) before audio.
        message = await self.websocket.receive()
        if message.get("type") == "websocket.disconnect":
            return
        # Model load opens ONNX sessions — keep it off the event loop.
        detector = await run_blocking(self._detector_factory, self.config)
        if detector is None:
            await self.websocket.send_json(
                {
                    "type": "error",
                    "kind": "wake_unavailable",
                    "message": "Wake word is enabled but no model is usable "
                    "on this server (see scripts/fetch_wake_word.py).",
                }
            )
            return
        telemetry.incr("web.wake_sessions")
        await self.websocket.send_json(
            {"type": "ready", "sample_rate": TARGET_SAMPLE_RATE}
        )
        while True:
            message = await self.websocket.receive()
            if message.get("type") == "websocket.disconnect":
                return
            data = message.get("bytes")
            if data:
                score = await run_blocking(detector.feed, data)
                if score is not None:
                    telemetry.incr("web.wake_detections")
                    await self.websocket.send_json(
                        {"type": "wake", "score": round(float(score), 4)}
                    )
                continue
            text = message.get("text")
            if text and '"stop"' in text:
                return
