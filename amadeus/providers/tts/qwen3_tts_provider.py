"""Qwen3-TTS zero-shot voice clone provider (GPU).

Loads Qwen/Qwen3-TTS-12Hz-1.7B-Base once, encodes the Kurisu reference clip
into a voice-clone prompt at startup, then reuses that cached prompt for every
utterance so the reference audio is never re-embedded per request.

Prefers the ``faster-qwen3-tts`` engine (CUDA-graph capture, ~10x faster than
upstream ``qwen-tts`` on consumer GPUs, RTF < 1) and falls back to upstream
``qwen-tts`` when it isn't installed. CUDA graphs are captured by a warmup
generation at init so the first user request doesn't pay the ~10s capture cost.

Emotion-conditioned synthesis (optional): sibling reference clips named
``<ref_stem>_<emotion>.wav`` with a matching ``<ref_stem>_<emotion>.txt``
transcript (e.g. ``kurisu_ref_happy.wav`` + ``kurisu_ref_happy.txt``) are
discovered at init and each becomes its own voice-clone prompt. Callers that
pass ``emotion=`` to ``synthesize``/``synthesize_stream`` get that clip's
delivery; unknown or absent emotions fall back to the base prompt, so the
feature is inert until someone records the clips. ``available_emotions``
advertises the discovered labels (the web layer only threads emotions the
provider actually has). Transcripts must transcribe THEIR clip — reusing the
base ref text against different audio corrupts the clone.
"""

from __future__ import annotations

import io
import logging
import threading
import wave
from pathlib import Path
from typing import Iterator

import numpy as np

from ... import telemetry
from .base import TTSProvider

logger = logging.getLogger(__name__)

DEFAULT_MODEL_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
DEFAULT_REF_AUDIO = "models/kurisu_voice_qwen/kurisu_ref.wav"
DEFAULT_REF_TEXT = "ビクトルコンドリア大学の精神生理学研究所がビジュアルリビルディング"
WARMUP_TEXT = "そうね。"


class Qwen3TTSProvider(TTSProvider):
    """Zero-shot Kurisu voice via Qwen3-TTS with a cached clone prompt."""

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        ref_audio: str = DEFAULT_REF_AUDIO,
        ref_text: str = DEFAULT_REF_TEXT,
        language: str = "Japanese",
        device: str = "cuda:0",
        dtype: str = "bfloat16",
        temperature: float = 0.7,
        top_k: int = 20,
        warmup: bool = True,
        speaker_guard: bool = True,
        speaker_guard_threshold: float = 0.84,
        stream_chunk_size: int = 6,
        guard_prefix_seconds: float = 1.5,
    ) -> None:
        self._stream_chunk_size = max(2, int(stream_chunk_size))
        self._guard_prefix_seconds = max(0.3, float(guard_prefix_seconds))
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "Qwen3-TTS requires the 'torch' and 'qwen-tts' packages "
                "(and the sox binary). Install them in this environment."
            ) from exc

        ref_path = Path(ref_audio)
        if not ref_path.exists():
            raise FileNotFoundError(f"Qwen3-TTS reference audio not found: {ref_audio}")

        self.language = language
        self._model_id = model_id
        self._ref_audio = str(ref_path)
        self._ref_text = ref_text
        self._dtype = dtype
        # Zero-shot cloning drifts to a different speaker on some rolls —
        # most often on short sentences — at the engine defaults
        # (temperature 0.9, top_k 50). Sample more conservatively by
        # default; raise via config if prosody gets too flat.
        self._temperature = temperature
        self._top_k = top_k

        try:
            from faster_qwen3_tts import FasterQwen3TTS
        except ImportError:
            FasterQwen3TTS = None

        if FasterQwen3TTS is not None:
            # Clear any stale marker from a previous slow-engine load in this
            # process (settings reload / character swap can rebuild providers).
            telemetry.set_gauge("tts.qwen3_slow_fallback", 0)
            self._engine = "faster"
            self.model = FasterQwen3TTS.from_pretrained(
                _resolve_local_snapshot(model_id),
                device=device,
                dtype=getattr(torch, dtype),
                attn_implementation="sdpa",
            )
            # FasterQwen3TTS wraps the upstream Qwen3TTSModel as `.model`;
            # the clone prompt is built by the upstream API either way.
            base_model = self.model.model
        else:
            from qwen_tts import Qwen3TTSModel

            # This fallback tanks voice-to-voice latency (~10x slower than
            # the CUDA-graph engine, well above realtime) and is otherwise
            # invisible — make it loud in the logs and in /api/status
            # (tts.engine_warning + the tts.qwen3_slow_fallback counter).
            logger.warning(
                "SLOW TTS FALLBACK: faster-qwen3-tts is not installed, so "
                "speech is synthesized with upstream qwen-tts — roughly 10x "
                "slower (RTF >> 1; every reply will sound noticeably "
                "delayed). Fix: pip install faster-qwen3-tts"
            )
            telemetry.set_gauge("tts.qwen3_slow_fallback", 1)
            self._engine = "upstream"
            self.model = Qwen3TTSModel.from_pretrained(
                _resolve_local_snapshot(model_id),
                device_map=device,
                dtype=getattr(torch, dtype),
            )
            base_model = self.model

        self._clone_prompt = base_model.create_voice_clone_prompt(
            ref_audio=str(ref_path),
            ref_text=ref_text,
        )
        # Per-emotion clone prompts from convention-named sibling clips.
        # Built at init like the base prompt (one embed each, never per
        # request); an empty dict means every call uses the base prompt.
        self._emotion_prompts: dict[str, object] = {}
        for label, (wav_path, transcript) in _discover_emotion_refs(ref_path).items():
            self._emotion_prompts[label] = base_model.create_voice_clone_prompt(
                ref_audio=wav_path,
                ref_text=transcript,
            )
            logger.info(
                "Loaded '%s' emotion voice reference (%s)",
                label,
                Path(wav_path).name,
            )
        self.available_emotions = frozenset(self._emotion_prompts)
        # Autoregressive generation is not reentrant; the web layer may fire
        # concurrent sentence synthesis during streaming.
        self._lock = threading.Lock()

        # Drift guard: every zero-shot generation can land on a different
        # speaker; score each clip against the reference and regenerate on
        # a bad roll. Optional dependency — None disables silently.
        self._guard = None
        self._guard_threshold = speaker_guard_threshold
        if speaker_guard:
            from .speaker_guard import build_speaker_guard

            self._guard = build_speaker_guard(
                str(ref_path), threshold=speaker_guard_threshold
            )

        if warmup and self._engine == "faster":
            self._generate(WARMUP_TEXT)

        # Only expose synthesize_stream when the underlying engine actually
        # supports chunked streaming generation (see _synthesize_stream_impl
        # docstring for the feature-detection rationale).
        if self._engine == "faster":
            self.synthesize_stream = self._synthesize_stream_impl

    def _prompt_for(self, emotion: str | None) -> object:
        """The clone prompt for ``emotion``, or the base prompt when the
        label is unknown/None — callers never need to pre-validate."""
        if emotion:
            return self._emotion_prompts.get(emotion.lower(), self._clone_prompt)
        return self._clone_prompt

    def rebind_voice(
        self, ref_audio: str, ref_text: str, language: str | None = None
    ) -> None:
        """Re-clone onto a new reference WITHOUT reloading the model.

        Character switching swaps only the voice reference; rebuilding the
        multi-GB engine for that would mean minutes of load and a VRAM
        double-residency spike, while a clone prompt is one cheap ref-audio
        embed. Emotion sibling clips and the speaker-drift guard follow the
        new reference; the cache_signature changes with it (a CachedTTSProvider
        wrapper must refresh_signature() afterwards)."""
        ref_path = Path(ref_audio)
        if not ref_path.is_file():
            raise FileNotFoundError(f"Qwen3-TTS reference audio not found: {ref_audio}")
        base_model = self.model.model if self._engine == "faster" else self.model
        clone_prompt = base_model.create_voice_clone_prompt(
            ref_audio=str(ref_path), ref_text=ref_text
        )
        emotion_prompts: dict[str, object] = {}
        for label, (wav_path, transcript) in _discover_emotion_refs(ref_path).items():
            emotion_prompts[label] = base_model.create_voice_clone_prompt(
                ref_audio=wav_path, ref_text=transcript
            )
        guard = None
        if self._guard is not None:
            from .speaker_guard import build_speaker_guard

            guard = build_speaker_guard(str(ref_path), threshold=self._guard_threshold)
        with self._lock:
            self._ref_audio = str(ref_path)
            self._ref_text = ref_text
            if language:
                self.language = language
            self._clone_prompt = clone_prompt
            self._emotion_prompts = emotion_prompts
            self.available_emotions = frozenset(emotion_prompts)
            self._guard = guard
        # The replaced clone prompts / guard tensors are unreferenced now,
        # but torch keeps their VRAM in its allocator cache. On an 8GB card
        # already ~70% full, that cached-but-idle memory is the difference
        # between the next synthesis fitting or CUDA-OOMing, so hand it back.
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # pragma: no cover - torch always present on GPU builds
            pass
        logger.info(
            "Voice rebound to %s (%d emotion refs)", ref_path.name, len(emotion_prompts)
        )

    def _generate(
        self, text: str, emotion: str | None = None
    ) -> tuple[np.ndarray, int]:
        with self._lock:
            try:
                wavs, sample_rate = self.model.generate_voice_clone(
                    text=text,
                    language=self.language,
                    voice_clone_prompt=self._prompt_for(emotion),
                    temperature=self._temperature,
                    top_k=self._top_k,
                )
            except RuntimeError as exc:
                # A transient CUDA OOM (VRAM pressure from other apps on a
                # shared 8GB card) is often recoverable once torch's cached
                # blocks are returned; retry exactly once. A second failure
                # propagates — the streaming pipeline degrades that sentence
                # and the deferred TTS path still covers the reply.
                if "out of memory" not in str(exc).lower():
                    raise
                logger.warning("CUDA OOM during synthesis; retrying once")
                import torch

                torch.cuda.empty_cache()
                wavs, sample_rate = self.model.generate_voice_clone(
                    text=text,
                    language=self.language,
                    voice_clone_prompt=self._prompt_for(emotion),
                    temperature=self._temperature,
                    top_k=self._top_k,
                )
        return np.asarray(wavs[0]), sample_rate

    def synthesize(
        self,
        text: str,
        *,
        speaking_rate: float = 1.0,
        pitch_shift: float = 0.0,
        emotion: str | None = None,
    ) -> bytes:
        """Synthesize speech with the cached Kurisu voice-clone prompt.

        ``speaking_rate``/``pitch_shift`` are accepted for interface
        compatibility; Qwen3-TTS has no equivalent generation controls.
        ``emotion`` picks a per-emotion clone prompt when a matching
        reference clip was discovered at init (base prompt otherwise).
        The drift guard stays anchored to the BASE reference for every
        emotion — the refs are the same speaker, and same-voice clips score
        0.88-0.94 against a 0.78 threshold, so emotional delivery has
        headroom without a per-emotion encoder embed.
        """
        if self._guard is not None:
            samples, sample_rate, _score = self._guard.best_of(
                lambda: self._generate(text, emotion), label=text[:24]
            )
        else:
            samples, sample_rate = self._generate(text, emotion)
        return _wav_bytes(samples, sample_rate)

    def judge_speaker(self, wav_data: bytes) -> bool | None:
        """Score finished WAV bytes against the reference voice.

        Feature-detected by CachedTTSProvider before every cache write: a
        clip that survived best_of only as "best of three bad rolls" (or a
        streamed short line that never hit the prefix-check length) must
        not become a permanently cached wrong-voice line — the 2026-07-28
        cache audit found drifted renders (0.60 similarity at 6.7s) replaying
        on every hit. Returns None when unjudgeable (guard off, decode
        failure): the caller treats None as "no objection".
        """
        if self._guard is None:
            return None
        try:
            with wave.open(io.BytesIO(wav_data), "rb") as wav_file:
                rate = wav_file.getframerate()
                frames = wav_file.readframes(wav_file.getnframes())
            samples = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
            score = self._guard.similarity(samples, rate)
            mark = self._guard.threshold_for(samples.size / rate)
            return score >= mark
        except Exception:  # judging must never break synthesis/caching
            logger.exception("speaker judge failed; treating clip as unjudged")
            return None

    def _synthesize_stream_impl(
        self,
        text: str,
        *,
        speaking_rate: float = 1.0,
        pitch_shift: float = 0.0,
        emotion: str | None = None,
        chunk_size: int | None = None,
    ) -> Iterator[bytes]:
        """Yield WAV-encoded audio chunks as they're generated.

        Bound onto ``self.synthesize_stream`` only when the "faster" engine
        (which exposes ``generate_voice_clone_streaming``) is active — see
        ``__init__``. Callers feature-detect via
        ``getattr(provider, "synthesize_stream", None)`` rather than
        isinstance, mirroring ``CachedTTSProvider``'s ``synthesize_async``
        pattern; the upstream ``qwen-tts`` engine has no streaming API and
        never gets this attribute.

        ``speaking_rate``/``pitch_shift`` are accepted for interface parity
        with ``synthesize`` but unused (Qwen3-TTS has no such controls).

        Not reentrant: holds the same lock as ``synthesize`` for the whole
        generator lifetime, so callers must fully drain (or close) one
        stream before starting another.

        Drift guard on the streaming path: the first ~1.2s of audio is
        buffered and speaker-scored before anything is released; a bad roll
        aborts that generation and rolls a fresh one (once), costing only
        the discarded prefix instead of a whole clip.
        """
        guard = self._guard
        attempts = 2 if guard is not None else 1
        clone_prompt = self._prompt_for(emotion)
        if chunk_size is None:
            chunk_size = self._stream_chunk_size
        with self._lock:
            for attempt in range(attempts):
                prefix: list[tuple[np.ndarray, int]] = []
                prefix_samples = 0
                prefix_checked = guard is None
                aborted = False
                for (
                    audio_chunk,
                    sample_rate,
                    _timing,
                ) in self.model.generate_voice_clone_streaming(
                    text=text,
                    language=self.language,
                    voice_clone_prompt=clone_prompt,
                    temperature=self._temperature,
                    top_k=self._top_k,
                    chunk_size=chunk_size,
                ):
                    if audio_chunk is None or len(audio_chunk) == 0:
                        continue
                    chunk = np.asarray(audio_chunk)
                    if prefix_checked:
                        yield _wav_bytes(chunk, sample_rate)
                        continue
                    prefix.append((chunk, sample_rate))
                    prefix_samples += chunk.reshape(-1).size
                    if prefix_samples < int(self._guard_prefix_seconds * sample_rate):
                        continue
                    joined = np.concatenate([c.reshape(-1) for c, _ in prefix])
                    mark = guard.threshold_for(joined.size / sample_rate)
                    try:
                        score = guard.similarity(joined, sample_rate)
                    except Exception:
                        logger.exception("stream guard scoring failed; passing through")
                        score = None
                    if score is not None and score < mark and attempt + 1 < attempts:
                        telemetry.incr("tts.speaker_drift_stream_retries")
                        logger.warning(
                            "speaker drift in stream (%.3f < %.2f); regenerating",
                            score,
                            mark,
                        )
                        aborted = True
                        break
                    prefix_checked = True
                    for buffered, buffered_rate in prefix:
                        yield _wav_bytes(buffered, buffered_rate)
                    prefix = []
                if aborted:
                    continue
                # Short utterance that never hit the check length: score it
                # once whole; regenerate on a bad roll if we still can.
                if not prefix_checked and prefix:
                    joined = np.concatenate([c.reshape(-1) for c, _ in prefix])
                    rate = prefix[0][1]
                    mark = guard.threshold_for(joined.size / rate)
                    try:
                        score = guard.similarity(joined, rate)
                    except Exception:
                        score = None
                    if score is not None and score < mark and attempt + 1 < attempts:
                        telemetry.incr("tts.speaker_drift_stream_retries")
                        logger.warning(
                            "speaker drift in short stream (%.3f < %.2f); regenerating",
                            score,
                            mark,
                        )
                        continue
                for buffered, buffered_rate in prefix:
                    yield _wav_bytes(buffered, buffered_rate)
                return

    @property
    def cache_signature(self) -> str:
        """Identity string for the disk TTS cache (see providers/tts/cache.py).

        Changing the engine, model, reference clip/text, language, dtype, or
        sampling parameters changes the resulting audio, so any of those must
        invalidate cached entries. (Including sampling also flushes every
        entry cached before the conservative defaults landed — one drifted
        wrong-voice render of a canned line would otherwise replay forever.)
        """
        return "|".join(
            [
                "qwen3_tts",
                self._engine,
                self._model_id,
                self._ref_audio,
                self._ref_text,
                self.language,
                self._dtype,
                repr(self._temperature),
                repr(self._top_k),
            ]
        )


def _discover_emotion_refs(ref_path: Path) -> dict[str, tuple[str, str]]:
    """Map emotion label -> (wav path, transcript) for convention-named
    siblings of the base reference clip.

    ``kurisu_ref.wav`` picks up ``kurisu_ref_happy.wav`` etc.; each clip
    needs a same-stem ``.txt`` holding ITS transcript (a clone prompt built
    against the wrong transcript degrades the voice, so a missing/empty
    transcript skips the clip loudly rather than guessing).
    """
    refs: dict[str, tuple[str, str]] = {}
    prefix = f"{ref_path.stem}_"
    for wav in sorted(ref_path.parent.glob(f"{prefix}*{ref_path.suffix}")):
        label = wav.stem[len(prefix) :].strip().lower()
        if not label:
            continue
        transcript_path = wav.with_suffix(".txt")
        try:
            transcript = transcript_path.read_text(encoding="utf-8").strip()
        except FileNotFoundError:
            logger.warning(
                "Emotion reference %s has no transcript %s; skipping it",
                wav.name,
                transcript_path.name,
            )
            continue
        if not transcript:
            logger.warning(
                "Emotion reference transcript %s is empty; skipping %s",
                transcript_path.name,
                wav.name,
            )
            continue
        refs[label] = (str(wav), transcript)
    return refs


def _resolve_local_snapshot(model_id: str) -> str:
    """Return the cached HF snapshot dir for ``model_id`` if fully downloaded.

    Loading from a local path keeps startup fully offline; transformers
    4.57.x otherwise makes an unconditional hub API call while loading the
    tokenizer, which fails without internet even when the model is cached.
    """
    if Path(model_id).exists():
        return model_id
    try:
        from huggingface_hub import snapshot_download

        return snapshot_download(model_id, local_files_only=True)
    except Exception:
        return model_id  # not cached yet; let from_pretrained download it


def _wav_bytes(samples: np.ndarray, sample_rate: int) -> bytes:
    samples = samples.reshape(-1)
    if samples.dtype != np.int16:
        clipped = np.clip(samples, -1.0, 1.0)
        samples = (clipped * 32767).astype(np.int16)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(samples.tobytes())
    return buffer.getvalue()
