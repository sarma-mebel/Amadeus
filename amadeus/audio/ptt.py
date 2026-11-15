"""Push-to-talk controller built on keyboard + VAD."""

from __future__ import annotations

import queue
import threading
import time
from typing import Optional

try:  # pragma: no cover - optional dependency
    import keyboard
except Exception:  # pragma: no cover - degrade gracefully
    keyboard = None

from ..config import AudioSettings
from ..logging_utils import get_logger
from ..providers.stt.base import STTProvider
from .voice import VoiceRecorder

LOGGER = get_logger(__name__)


class PushToTalkController:
    """Listen for a keypress and trigger speech recognition."""

    def __init__(self, settings: AudioSettings, stt: STTProvider) -> None:
        self.settings = settings
        self.stt = stt
        self.recorder = VoiceRecorder(settings)
        self.transcripts: queue.Queue[str] = queue.Queue()
        self._active = False
        self._listeners_attached = False

    def start(self) -> None:
        if not keyboard:
            LOGGER.warning("keyboard module not available; push-to-talk disabled")
            return
        if self._listeners_attached:
            return
        key = self.settings.push_to_talk_key
        keyboard.on_press_key(key, self._on_press)
        keyboard.on_release_key(key, self._on_release)
        self._listeners_attached = True
        LOGGER.info("Push-to-talk listening on key '%s'", key)

    def stop(self) -> None:
        if not keyboard or not self._listeners_attached:
            return
        keyboard.unhook_all()
        self._listeners_attached = False
        LOGGER.info("Push-to-talk stopped")

    def _on_press(self, event) -> None:  # pragma: no cover - hardware interaction
        if self._active:
            return
        self._active = True
        threading.Thread(target=self._record_session, daemon=True).start()

    def _on_release(self, event) -> None:  # pragma: no cover
        self._active = False

    def _record_session(self) -> None:  # pragma: no cover
        try:
            self.recorder.start()
        except Exception as exc:
            LOGGER.error("Unable to start microphone: %s", exc)
            self._active = False
            return
        while self._active:
            time.sleep(0.01)
        self.recorder.stop()
        audio = self.recorder.collect()
        if not audio:
            return
        try:
            text = self.stt.transcribe_audio(audio, self.settings.sample_rate)
        except Exception as exc:
            LOGGER.error("STT failed: %s", exc)
            return
        self.transcripts.put(text)

    def poll(self) -> Optional[str]:
        try:
            return self.transcripts.get_nowait()
        except queue.Empty:
            return None
