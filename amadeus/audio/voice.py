"""Audio capture and playback helpers."""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import io
import numpy as np

try:  # pragma: no cover - optional dependency
    import sounddevice as sd
except Exception:  # pragma: no cover - gracefully degrade
    sd = None

try:  # pragma: no cover - optional dependency
    import webrtcvad
except Exception:  # pragma: no cover
    webrtcvad = None

from ..config import AudioSettings
from ..logging_utils import get_logger

LOGGER = get_logger(__name__)


@dataclass
class AudioChunk:
    data: bytes
    timestamp: float


class VoiceRecorder:
    """Microphone capture with push-to-talk and simple VAD."""

    def __init__(self, settings: AudioSettings) -> None:
        self.settings = settings
        self.sample_rate = settings.sample_rate
        self.channels = 1
        self.vad = webrtcvad.Vad(settings.vad_aggressiveness) if webrtcvad else None
        self.frames: queue.Queue[AudioChunk] = queue.Queue()
        self.recording = False
        self.thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if not sd:
            raise RuntimeError("sounddevice is required for microphone recording")
        if self.recording:
            return
        self.recording = True
        self.thread = threading.Thread(target=self._record_loop, daemon=True)
        self.thread.start()
        LOGGER.info("Voice recorder started")

    def stop(self) -> None:
        self.recording = False
        if self.thread:
            self.thread.join(timeout=1.0)
            self.thread = None
        LOGGER.info("Voice recorder stopped")

    def _record_loop(self) -> None:
        frame_duration = 30  # ms
        frame_samples = int(self.sample_rate * frame_duration / 1000)
        frame_bytes = frame_samples * self.channels * 2  # int16 samples
        pending = bytearray()

        def callback(
            indata, frames, time_info, status
        ):  # pragma: no cover - real-time callback
            if not self.recording:
                raise sd.CallbackStop()
            audio16 = (indata[:, 0] * 32767).astype(np.int16).tobytes()
            timestamp = time.time()
            if not self.vad:
                self.frames.put(AudioChunk(data=audio16, timestamp=timestamp))
                return
            # webrtcvad only accepts exact 10/20/30ms frames, so buffer the
            # incoming audio and feed it whole frames.
            pending.extend(audio16)
            while len(pending) >= frame_bytes:
                frame = bytes(pending[:frame_bytes])
                del pending[:frame_bytes]
                if self.vad.is_speech(frame, self.sample_rate):
                    self.frames.put(AudioChunk(data=frame, timestamp=timestamp))

        with sd.InputStream(
            channels=self.channels,
            samplerate=self.sample_rate,
            dtype="float32",
            blocksize=frame_samples,
            callback=callback,
            device=self.settings.input_device,
        ):
            while self.recording:
                time.sleep(0.1)

    def collect(self, timeout: float = 0.5) -> bytes:
        chunks: list[bytes] = []
        try:
            while True:
                chunk = self.frames.get(timeout=timeout)
                chunks.append(chunk.data)
                if self.frames.empty():
                    break
        except queue.Empty:
            pass
        return b"".join(chunks)


def play_audio(data: bytes, sample_rate: int | None = None) -> None:
    """Play a WAV payload using sounddevice."""

    if not sd:
        LOGGER.warning("sounddevice not available; skipping audio playback")
        return
    with io.BytesIO(data) as buffer:
        import wave

        with wave.open(buffer, "rb") as wav:
            frames = wav.readframes(wav.getnframes())
            rate = wav.getframerate()
            channels = wav.getnchannels()
    audio = np.frombuffer(frames, dtype=np.int16)
    if channels > 1:
        audio = audio.reshape(-1, channels)
    sd.play(audio, samplerate=sample_rate or rate)
    sd.wait()


def save_wave(data: bytes, path: Path, sample_rate: int = 22050) -> None:
    """Persist audio to disk."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.write(data)
