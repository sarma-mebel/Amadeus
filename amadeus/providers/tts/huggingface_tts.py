"""Local text-to-speech provider for the Kurisu Hugging Face snapshot."""

from __future__ import annotations

import inspect
import io
import wave
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from .base import TTSProvider


class HuggingFaceTTSProvider(TTSProvider):
    """Load the Kurisu voice model from a locally downloaded Hugging Face snapshot."""

    def __init__(
        self,
        *,
        model_path: str = "mio/amadeus",
        model_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        try:
            from TTS.api import TTS
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "The Coqui TTS package is required for the Hugging Face TTS provider"
            ) from exc

        path = Path(model_path).expanduser()
        if path.is_dir():
            model_file = self._find_first(
                path,
                (
                    "model.pth",
                    "model.onnx",
                    "model.safetensors",
                ),
                kind="model",
            )
            config_file = self._find_first(
                path,
                ("config.json", "model_config.json", "config.yaml"),
                kind="config",
            )
            vocoder_file = self._find_optional(
                path, ("vocoder.pth", "vocoder.onnx", "vocoder.safetensors")
            )
            vocoder_config_file = self._find_optional(
                path, ("vocoder_config.json", "vocoder.yaml")
            )
        else:
            if not path.exists():
                raise FileNotFoundError(f"Model path '{model_path}' does not exist")
            model_file = path
            config_file = None
            vocoder_file = None
            vocoder_config_file = None

        kwargs: Dict[str, Any] = {
            "model_path": str(model_file),
        }
        if config_file is not None:
            kwargs["config_path"] = str(config_file)
        if vocoder_file is not None:
            kwargs["vocoder_path"] = str(vocoder_file)
        if vocoder_config_file is not None:
            kwargs["vocoder_config_path"] = str(vocoder_config_file)

        init_kwargs: Dict[str, Any] = {}
        synthesis_kwargs: Dict[str, Any] = {}
        if model_kwargs:
            if "init" in model_kwargs or "synthesis" in model_kwargs:
                init_kwargs = dict(model_kwargs.get("init", {}))
                synthesis_kwargs = dict(model_kwargs.get("synthesis", {}))
            else:
                synthesis_kwargs = dict(model_kwargs)
        kwargs.update(init_kwargs)

        self._tts = TTS(**kwargs)
        self._tts_signature = inspect.signature(self._tts.tts)
        self._sample_rate = getattr(
            getattr(self._tts, "synthesizer", None),
            "output_sample_rate",
            22050,
        )
        self._synthesis_kwargs = synthesis_kwargs

    @staticmethod
    def _find_first(directory: Path, candidates: tuple[str, ...], *, kind: str) -> Path:
        for filename in candidates:
            candidate = directory / filename
            if candidate.exists():
                return candidate
            match = next(directory.rglob(filename), None)
            if match is not None:
                return match
        raise FileNotFoundError(
            f"Unable to locate {kind} file in '{directory}'. Checked: {', '.join(candidates)}"
        )

    @staticmethod
    def _find_optional(directory: Path, candidates: tuple[str, ...]) -> Optional[Path]:
        for filename in candidates:
            candidate = directory / filename
            if candidate.exists():
                return candidate
            match = next(directory.rglob(filename), None)
            if match is not None:
                return match
        return None

    def synthesize(
        self,
        text: str,
        *,
        speaking_rate: float = 1.0,
        pitch_shift: float = 0.0,
    ) -> bytes:
        """Generate a WAV payload for ``text`` using the local Kurisu model."""

        kwargs: Dict[str, Any] = dict(self._synthesis_kwargs)
        if "speed" in self._tts_signature.parameters and speaking_rate != 1.0:
            kwargs["speed"] = speaking_rate
        if "pitch" in self._tts_signature.parameters and pitch_shift != 0.0:
            kwargs["pitch"] = pitch_shift

        audio = self._tts.tts(text, **kwargs)

        sample_rate = self._sample_rate
        if isinstance(audio, tuple) and len(audio) == 2:
            audio, sample_rate = audio

        array = np.asarray(audio, dtype=np.float32)
        if array.ndim > 1:
            array = np.squeeze(array)

        array = np.clip(array, -1.0, 1.0)
        pcm = (array * 32767).astype("<i2")

        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(int(sample_rate))
            wav_file.writeframes(pcm.tobytes())

        return buffer.getvalue()
