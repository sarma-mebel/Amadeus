"""MeloTTS provider for custom trained voice models."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Optional

from .base import TTSProvider


def _get_physical_cores() -> int:
    """Get number of physical CPU cores (not hyperthreaded)."""
    try:
        import psutil

        return psutil.cpu_count(logical=False) or os.cpu_count() or 1
    except ImportError:
        # Fallback: assume half of logical cores are physical
        logical_cores = os.cpu_count() or 1
        return max(1, logical_cores // 2)


class MeloTTSProvider(TTSProvider):
    """Use MeloTTS with custom trained checkpoint for Kurisu voice."""

    def __init__(
        self,
        config_path: str,
        checkpoint_path: str,
        language: str = "JP",
        device: str = "cpu",
        speaker_id: Optional[int] = None,
    ) -> None:
        """Initialize MeloTTS with custom trained model.

        Args:
            config_path: Path to config.json from training
            checkpoint_path: Path to G_*.pth checkpoint file
            language: Language code (JP for Japanese)
            device: Device to run on ('cpu' or 'cuda')
            speaker_id: Speaker ID (None = auto-detect first speaker)
        """
        try:
            from melo.api import TTS
            import torch
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "MeloTTS not installed. Install with: "
                "pip install git+https://github.com/myshell-ai/MeloTTS.git"
            ) from exc

        # Optimize PyTorch threading for CPU inference
        num_cores = _get_physical_cores()
        torch.set_num_threads(num_cores)
        torch.set_num_interop_threads(1)

        self.config_path = Path(config_path)
        self.checkpoint_path = Path(checkpoint_path)

        if not self.config_path.exists():
            raise FileNotFoundError(f"MeloTTS config not found: {config_path}")
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"MeloTTS checkpoint not found: {checkpoint_path}")

        # Load custom model (use_hf=False to load local checkpoint)
        self.model = TTS(
            language=language,
            device=device,
            use_hf=False,  # Use custom checkpoint, not HuggingFace
            config_path=str(self.config_path),
            ckpt_path=str(self.checkpoint_path),
        )

        # Get speaker ID (use first speaker if not specified)
        speaker_ids = self.model.hps.data.spk2id
        if speaker_id is None:
            # Use the first available speaker
            self.speaker_id = speaker_ids[list(speaker_ids.keys())[0]]
        else:
            self.speaker_id = speaker_id

    def synthesize(
        self, text: str, *, speaking_rate: float = 1.0, pitch_shift: float = 0.0
    ) -> bytes:
        """Synthesize Japanese text using trained Kurisu voice.

        Args:
            text: Japanese text to synthesize
            speaking_rate: Speed multiplier (1.0 = normal, >1.0 = faster, <1.0 = slower)
            pitch_shift: Pitch adjustment (currently unused by MeloTTS)

        Returns:
            WAV audio data as bytes
        """
        # MeloTTS uses speed parameter (1.0 = normal)
        speed = speaking_rate

        # Create temporary file for output
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            # Generate speech to file
            self.model.tts_to_file(
                text=text,
                speaker_id=self.speaker_id,
                output_path=tmp_path,
                speed=speed,
            )

            # Read WAV file as bytes
            with open(tmp_path, "rb") as f:
                audio_bytes = f.read()

            return audio_bytes

        finally:
            # Cleanup temp file
            try:
                os.remove(tmp_path)
            except OSError:  # pragma: no cover - best effort cleanup
                pass
