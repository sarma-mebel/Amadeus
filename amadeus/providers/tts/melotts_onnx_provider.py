"""MeloTTS ONNX provider using direct ONNX Runtime inference."""

from __future__ import annotations

import os
import tempfile
import types
import wave
from pathlib import Path
from typing import Dict, List

import numpy as np

from .base import TTSProvider


def _get_physical_cores() -> int:
    """Get number of physical CPU cores (not hyperthreaded)."""
    try:
        import psutil

        cores = psutil.cpu_count(logical=False) or os.cpu_count() or 1
    except ImportError:
        logical_cores = os.cpu_count() or 1
        cores = max(1, logical_cores // 2)
    # ONNX Runtime can allocate substantial per-thread workspace for this
    # model. A small cap keeps desktop memory reasonable on high-core CPUs
    # while preserving enough parallelism for sub-second local synthesis.
    return min(4, max(1, cores))


class _OfflineTokenizer:
    """Avoid Melo's import-time Japanese BERT download; ONNX export uses zeros."""

    @classmethod
    def from_pretrained(cls, *_args, **_kwargs):
        return cls()

    def tokenize(self, text: str) -> List[str]:
        return list(text)


def _prepare_melo_text_environment() -> None:
    """Make Melo's text frontend usable offline in this conda environment."""

    os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba-cache")

    try:
        import unidic
        import unidic_lite

        if not Path(getattr(unidic, "DICDIR", "")).exists():
            unidic.DICDIR = unidic_lite.DICDIR
    except ImportError:
        try:
            import sys
            import unidic_lite

            stub = types.ModuleType("unidic")
            stub.DICDIR = unidic_lite.DICDIR
            sys.modules["unidic"] = stub
        except ImportError:
            pass

    try:
        import transformers

        transformers.AutoTokenizer = _OfflineTokenizer
    except ImportError:
        import sys

        transformers_stub = types.ModuleType("transformers")
        transformers_stub.AutoTokenizer = _OfflineTokenizer
        sys.modules["transformers"] = transformers_stub

    try:
        import sys

        if "g2p_en" not in sys.modules:
            g2p_stub = types.ModuleType("g2p_en")

            class _OfflineG2p:
                def __call__(self, text: str) -> List[str]:
                    return list(text)

            g2p_stub.G2p = _OfflineG2p
            sys.modules["g2p_en"] = g2p_stub
    except Exception:
        pass


def _load_token_map(tokens_path: Path) -> Dict[str, int]:
    token_map: Dict[str, int] = {}
    with tokens_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.rstrip("\n")
            if not line:
                continue
            token, _, idx = line.rpartition(" ")
            if not token or not idx:
                continue
            token_map[token] = int(idx)
    return token_map


def _intersperse(values: List[int], blank: int = 0) -> List[int]:
    result = [blank] * (len(values) * 2 + 1)
    result[1::2] = values
    return result


class MeloTTSONNXProvider(TTSProvider):
    """Use a custom MeloTTS ONNX model for CPU inference."""

    def __init__(
        self,
        model_path: str,
        tokens_path: str,
        lexicon_path: str = "",
        speaker_id: int = 0,
        noise_scale: float = 0.667,
        noise_scale_w: float = 0.8,
        length_scale: float = 1.0,
    ) -> None:
        """Initialize direct ONNX Runtime inference.

        Args:
            model_path: Path to model.onnx file.
            tokens_path: Path to tokens.txt file.
            lexicon_path: Kept for config compatibility; not used by direct inference.
            speaker_id: Speaker ID (0 for Kurisu).
            noise_scale: Voice variation scale.
            noise_scale_w: Duration predictor noise scale.
            length_scale: Length scale (1.0 = normal speed).
        """
        _prepare_melo_text_environment()
        try:
            import onnx
            import onnxruntime as ort
            from melo.text.japanese import g2p, text_normalize
            from melo.text.symbols import language_id_map, language_tone_start_map
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "MeloTTS ONNX direct inference requires onnxruntime, onnx, "
                "MeloTTS, and Japanese tokenizer dependencies."
            ) from exc

        self.model_path = Path(model_path)
        self.tokens_path = Path(tokens_path)
        self.lexicon_path = Path(lexicon_path) if lexicon_path else None
        if not self.model_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {model_path}")
        if not self.tokens_path.exists():
            raise FileNotFoundError(f"Tokens file not found: {tokens_path}")

        metadata = {
            item.key: item.value
            for item in onnx.load(str(self.model_path)).metadata_props
        }
        self.sample_rate = int(metadata.get("sample_rate", 44100))
        self.add_blank = bool(int(metadata.get("add_blank", "1")))
        self.language = "JP"
        self.lang_id = int(metadata.get("lang_id", language_id_map[self.language]))
        self.tone_start = int(
            metadata.get("tone_start", language_tone_start_map[self.language])
        )
        self.speaker_id = speaker_id
        self.noise_scale = float(noise_scale)
        self.noise_scale_w = float(noise_scale_w)
        self.length_scale = float(length_scale)
        self.g2p = g2p
        self.text_normalize = text_normalize
        self.token_map = _load_token_map(self.tokens_path)

        options = ort.SessionOptions()
        options.intra_op_num_threads = _get_physical_cores()
        self.session = ort.InferenceSession(
            str(self.model_path),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )

    def _encode(self, text: str) -> tuple[np.ndarray, np.ndarray]:
        normalized = self.text_normalize(text)
        phones, tones, _word2ph = self.g2p(normalized)
        phone_ids = [self.token_map[phone] for phone in phones]
        tone_ids = [int(tone) + self.tone_start for tone in tones]
        if self.add_blank:
            phone_ids = _intersperse(phone_ids, 0)
            tone_ids = _intersperse(tone_ids, 0)
        return (
            np.asarray([phone_ids], dtype=np.int64),
            np.asarray([tone_ids], dtype=np.int64),
        )

    def synthesize(
        self, text: str, *, speaking_rate: float = 1.0, pitch_shift: float = 0.0
    ) -> bytes:
        """Synthesize Japanese text using the ONNX-optimized Kurisu voice."""

        return self._synthesize_direct(text, speaking_rate)

    async def synthesize_async(
        self, text: str, *, speaking_rate: float = 1.0, pitch_shift: float = 0.0
    ) -> bytes:
        """Async-compatible wrapper.

        ONNX Runtime/MeloTTS hangs in this environment when called from
        ``asyncio.to_thread``. Direct inference is sub-second, so the web layer
        calls this wrapper to avoid the generic thread pool.
        """

        return self._synthesize_direct(text, speaking_rate)

    def _synthesize_direct(self, text: str, speaking_rate: float) -> bytes:
        x, tones = self._encode(text)
        length_scale = self.length_scale / max(float(speaking_rate), 0.05)
        inputs = {
            "x": x,
            "x_lengths": np.asarray([x.shape[1]], dtype=np.int64),
            "tones": tones,
            "sid": np.asarray([self.speaker_id], dtype=np.int64),
            "noise_scale": np.asarray([self.noise_scale], dtype=np.float32),
            "length_scale": np.asarray([length_scale], dtype=np.float32),
            "noise_scale_w": np.asarray([self.noise_scale_w], dtype=np.float32),
        }
        output = self._run_session(inputs)[0]
        samples = np.asarray(output).reshape(-1)
        return _wav_bytes(samples, self.sample_rate)

    def _run_session(self, inputs: Dict[str, np.ndarray]) -> List[np.ndarray]:
        return self.session.run(["y"], inputs)

    def close(self) -> None:
        pass

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


def _wav_bytes(samples: np.ndarray, sample_rate: int) -> bytes:
    if samples.dtype != np.int16:
        clipped = np.clip(samples, -1.0, 1.0)
        samples = (clipped * 32767).astype(np.int16)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        with wave.open(tmp_path, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(samples.tobytes())

        with open(tmp_path, "rb") as handle:
            return handle.read()
    finally:
        try:
            os.remove(tmp_path)
        except OSError:  # pragma: no cover - best effort cleanup
            pass
