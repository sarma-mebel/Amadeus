"""Vocal emotion perception: a one-line hint about HOW the user's voice
sounded, derived from the same audio clip the STT endpoint transcribes.

Today Kurisu infers mood only from what the user typed/said (text tone via
``memory.last_tone``, ``mood_from_tone`` in engine.py). This module adds a
second, independent signal — the audio itself — analyzed cheaply and never
blocking STT (analysis runs after transcription completes, on the same
bytes, and any failure degrades to "no hint" rather than a broken turn).

Two analysis paths, exactly mirroring ``web/streaming/smart_turn.py``'s
lazy-load / missing-model / degrade-loudly shape:

* The prosody heuristic (the DEFAULT, dependency-free beyond numpy which
  is already a hard dependency): energy level + variance and a rough
  autocorrelation pitch estimate, judged against a rolling per-speaker
  norm (``SpeakerBaseline``) and mapped to a small label set. Below
  ``min_confidence``, or before the baseline has calibrated, the analyzer
  ABSTAINS rather than guessing — a wrong vocal-tone hint would color a
  reply on a coin flip.
* An optional ONNX speech-emotion classifier, auto-discovered from
  ``models/vocal_emotion/`` (see ``scripts/fetch_vocal_emotion.py``), used
  ONLY in ``mode: "model"``; missing model/runtime counts
  ``vocal_emotion.model_unavailable`` once per process.

Why speaker-relative, and why the model is opt-in — both settled by
measurement on 2026-08-05 against 17 voice-acted reference clips plus the
TTS reference lines:

* The heuristic's original absolute thresholds (``mean_pitch > 200`` for
  "bright") fired on 16 of 17 clips at an identical 0.65 confidence.
  Vocal range differs about an octave between people and mic gain moves
  absolute energy more than affect does, so absolute cutoffs measure WHO
  is speaking, not HOW they sound. Hence ``SpeakerBaseline``.
* emotion2vec+ base (the best-available permissively-licensed SER export)
  returned six different emotions and not one "neutral" for those same
  ordinary spoken lines, at a MEDIAN confidence of 0.997 — three read
  "disgusted" at 1.000. A model that is most certain when it is wrong
  cannot be screened by ``min_confidence``, so it does not run by default.
  The wrapper stays implemented and tested for a better-calibrated model.

The result is stashed per-session with a timestamp (see
``amadeus/vocal_tone_stash.py``) and consumed by exactly the next chat
turn within ``vocal_emotion.ttl_seconds`` — never stored to memory, never
in transcripts, never re-used across turns.
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from . import telemetry
from .logging_utils import get_logger

LOGGER = get_logger(__name__)

# Auto-discovery directory (any *.onnx inside is taken as the model),
# overridable via config (vocal_emotion.model_path) or env — same
# convention as smart_turn.py's DEFAULT_MODEL_DIR/MODEL_DIR_ENV.
DEFAULT_MODEL_DIR = Path("models") / "vocal_emotion"
MODEL_DIR_ENV = "AMADEUS_VOCAL_EMOTION_DIR"

TARGET_SAMPLE_RATE = 16000

# Heuristic label set. "neutral" is a real (non-abstaining) verdict distinct
# from "None" (abstained, too little signal to say anything).
HEURISTIC_LABELS = ("bright", "flat", "tense", "low", "neutral")

# Model labels that are catch-all classes, not perceptions: a verdict of
# "other"/"unknown" (emotion2vec+'s two absorber classes) means the model
# itself couldn't place the clip — abstain rather than color a reply with it.
MODEL_ABSTAIN_LABELS = frozenset({"other", "unknown"})

# Human-readable phrasing for the prompt line (persona.py renders these
# verbatim, so they read naturally after "sounded").
LABEL_PHRASES = {
    # Heuristic (prosody) labels.
    "bright": "bright and energetic",
    "flat": "flat, maybe a little tired",
    "tense": "tense",
    "low": "low and quiet",
    "neutral": "fairly neutral",
    # Model (speech-emotion) labels — the emotion2vec+ class list. Kept
    # deliberately hedged ("maybe", "a little"): a classifier verdict is a
    # read on one short clip, not a diagnosis.
    "happy": "upbeat, genuinely pleased",
    "sad": "down, a little deflated",
    "angry": "irritated, maybe frustrated",
    "fearful": "anxious or on edge",
    "surprised": "surprised by something",
    "disgusted": "put off by something",
}


class SpeakerBaseline:
    """Rolling per-speaker norm for pitch and loudness.

    Prosody only means something against a reference: 300 Hz is animated
    for one voice and resting for another, and mic gain shifts absolute
    energy by more than emotion does. This keeps a bounded window of recent
    clips and reports how far the current one deviates, in sigmas.

    ``MIN_SAMPLES`` clips must land before any deviation verdict is
    offered — an average of two clips is not a baseline, and a confidently
    wrong tone hint is worse than none (the analyzer abstains meanwhile).
    Deliberately in-RAM and per-process: this is a running impression of
    the current speaker, not a stored biometric profile, and it costs
    nothing to rebuild after a restart.
    """

    MIN_SAMPLES = 5
    WINDOW = 40

    def __init__(self) -> None:
        self._pitches: list = []
        self._energies: list = []
        # The process-wide instance is mutated from STT thread-executor
        # calls: two clips landing together must not interleave the
        # append/trim pair or read a half-updated window. Held only around
        # list access — never across feature extraction.
        self._lock = threading.Lock()

    @property
    def ready(self) -> bool:
        with self._lock:
            return len(self._pitches) >= self.MIN_SAMPLES

    def observe(self, mean_pitch: float, mean_energy: float) -> None:
        """Fold one clip's features into the norm (voiced clips only)."""
        with self._lock:
            if mean_pitch > 0:
                self._pitches.append(float(mean_pitch))
                del self._pitches[: -self.WINDOW]
            if mean_energy > 0:
                self._energies.append(float(mean_energy))
                del self._energies[: -self.WINDOW]

    @staticmethod
    def _z(values: list, current: float) -> float:
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        var = sum((v - mean) ** 2 for v in values) / len(values)
        std = var**0.5
        # A speaker whose clips are all alike has a near-zero spread; a
        # floor keeps one slightly-off clip from reading as many sigmas.
        return (current - mean) / max(std, mean * 0.08 + 1e-6)

    def pitch_z(self, mean_pitch: float) -> float:
        with self._lock:
            values = list(self._pitches)
        return self._z(values, mean_pitch)

    def energy_z(self, mean_energy: float) -> float:
        with self._lock:
            values = list(self._energies)
        return self._z(values, mean_energy)


@dataclass(frozen=True)
class VocalEmotionResult:
    """One analysis verdict: a label plus a 0-1 confidence and the source
    that produced it ("heuristic" or "model"), for telemetry/debugging."""

    label: str
    confidence: float
    source: str

    def phrase(self) -> str:
        return LABEL_PHRASES.get(self.label, self.label)


def _pcm16_to_float(pcm16: bytes, np: Any) -> Any:
    audio = np.frombuffer(pcm16, dtype="<i2").astype(np.float32) / 32768.0
    return audio


def _autocorrelation_pitch(frame: Any, np: Any, sample_rate: int) -> float:
    """Rough fundamental-frequency estimate via autocorrelation.

    Not a real pitch tracker (no cepstral/YIN precision) — just enough
    signal to separate "high and bright" from "low and flat" speech, which
    is all the heuristic label set needs. Returns 0.0 when no clear peak is
    found (silence, noise).
    """
    if frame.size < 32:
        return 0.0
    windowed = frame * np.hanning(frame.size)
    corr = np.correlate(windowed, windowed, mode="full")
    corr = corr[corr.size // 2 :]
    if corr[0] <= 1e-9:
        return 0.0
    corr = corr / corr[0]
    # Human voice fundamental: roughly 60-400 Hz.
    min_lag = max(1, int(sample_rate / 400))
    max_lag = min(corr.size - 1, int(sample_rate / 60))
    if max_lag <= min_lag:
        return 0.0
    segment = corr[min_lag:max_lag]
    peak_idx = int(np.argmax(segment))
    peak_val = float(segment[peak_idx])
    if peak_val < 0.3:  # no confident periodicity
        return 0.0
    lag = min_lag + peak_idx
    return float(sample_rate) / float(lag)


def analyze_heuristic(
    pcm16: bytes,
    sample_rate: int = TARGET_SAMPLE_RATE,
    *,
    min_confidence: float = 0.5,
    baseline: Optional[SpeakerBaseline] = None,
) -> Optional[VocalEmotionResult]:
    """Dependency-free (beyond numpy) prosody heuristic over PCM16 audio.

    Computes energy level/variance, a rough pitch estimate, and a
    voiced-frame-ratio speaking-rate proxy over short frames, then maps the
    combination onto ``HEURISTIC_LABELS``. Returns None (abstains) when the
    clip is too short/silent to say anything, or when the mapped label's
    confidence falls below ``min_confidence`` — a low-confidence guess is
    worse than no hint at all.
    """
    try:
        import numpy as np
    except Exception:  # pragma: no cover - numpy is a hard dependency
        return None
    if not pcm16 or sample_rate <= 0:
        return None
    audio = _pcm16_to_float(pcm16, np)
    if audio.size < sample_rate * 0.3:  # under ~300ms: not enough to judge
        return None

    frame_len = max(1, int(sample_rate * 0.03))  # 30ms frames
    hop = max(1, frame_len // 2)
    n_frames = max(1, (audio.size - frame_len) // hop + 1)
    energies = np.empty(n_frames, dtype=np.float64)
    pitches = np.empty(n_frames, dtype=np.float64)
    for i in range(n_frames):
        start = i * hop
        frame = audio[start : start + frame_len]
        energies[i] = float(np.sqrt(np.mean(frame.astype(np.float64) ** 2)))
        pitches[i] = _autocorrelation_pitch(frame, np, sample_rate)

    peak_energy = float(energies.max()) if energies.size else 0.0
    if peak_energy < 1e-4:  # effectively silent clip
        return None

    # Voiced-frame threshold relative to this clip's own dynamic range —
    # absolute PCM levels vary wildly with mic gain, so a fixed threshold
    # would misclassify quiet-mic setups as "low" every time.
    voiced_threshold = peak_energy * 0.15
    voiced_mask = energies >= voiced_threshold
    voiced_ratio = float(voiced_mask.mean()) if energies.size else 0.0
    mean_energy = float(energies.mean())
    energy_variance = float(energies.var())
    # Normalized variance (coefficient of variation) — comparable across
    # different absolute gain levels, unlike raw variance.
    energy_cv = energy_variance / (mean_energy**2 + 1e-9)

    voiced_pitches = pitches[voiced_mask & (pitches > 0)]
    mean_pitch = float(voiced_pitches.mean()) if voiced_pitches.size else 0.0

    if voiced_ratio < 0.15:
        # Almost entirely silence/noise: nothing usable to say.
        return None

    label, confidence = _classify(
        mean_energy, energy_cv, mean_pitch, voiced_ratio, baseline
    )
    # Observe AFTER classifying, so a clip is always judged against the
    # speaker's prior norm rather than one that already includes itself.
    if baseline is not None:
        baseline.observe(mean_pitch, mean_energy)
    if confidence < min_confidence:
        return None
    return VocalEmotionResult(label=label, confidence=confidence, source="heuristic")


def _classify(
    mean_energy: float,
    energy_cv: float,
    mean_pitch: float,
    voiced_ratio: float,
    baseline: Optional["SpeakerBaseline"] = None,
) -> tuple:
    """Map prosody features onto (label, confidence).

    Pitch and loudness are judged RELATIVE TO THIS SPEAKER, never against
    absolute thresholds. That distinction is the whole correctness story
    here: measured 2026-08-05 across 17 voice-acted reference clips, mean
    pitch ranged 208-325 Hz, so the old absolute ``mean_pitch > 200`` test
    fired for 16 of 17 and the analyzer returned "bright" at an identical
    0.65 confidence for nearly every clip — it was reporting WHO was
    speaking (vocal range differs ~an octave between people), not HOW they
    sounded. Mic gain makes absolute energy just as speaker-specific.

    So: until ``baseline`` has seen enough clips from this speaker to know
    their norm, the only verdicts available are the ones that need no
    reference point (a clip's own internal steadiness), and everything else
    abstains. Once calibrated, "bright"/"low" mean measurably above/below
    THEIR usual pitch and energy.
    """
    variable = energy_cv > 0.35
    steady = energy_cv < 0.12

    if baseline is None or not baseline.ready:
        # Uncalibrated: only within-clip shape is speaker-independent.
        # Steady + low dynamic range reads flat regardless of who it is;
        # anything else would be a guess dressed up as a perception.
        if steady:
            return "flat", 0.55
        return "neutral", 0.5

    pitch_z = baseline.pitch_z(mean_pitch)
    energy_z = baseline.energy_z(mean_energy)

    # Thresholds are in standard deviations of this speaker's own history.
    # 1.0 sigma is deliberately conservative: a hint only earns a line in
    # the prompt when the clip is clearly off their baseline.
    if energy_z < -1.0 and steady:
        return "low", 0.65
    if energy_z > 1.0 and variable and pitch_z < 0.5:
        return "tense", 0.6
    if pitch_z > 1.0 and (variable or energy_z > 0.5):
        return "bright", 0.65
    if steady and pitch_z < 0.5:
        return "flat", 0.55
    if pitch_z < -1.0 and energy_z < -0.5:
        return "low", 0.55
    return "neutral", 0.5


def _load_head(head_path: Path, np: Any) -> Optional[dict]:
    """Parse and validate a ``*head.json`` classifier sidecar.

    The pinned emotion2vec+ export ships its tiny classifier head as JSON
    (``{"labels": [...], "weight": [[...]], "bias": [...]}``) beside the
    ONNX, because the ONNX itself emits backbone features, not logits.
    Returns ``{"labels", "weight", "bias"}`` with numpy arrays, or None on
    any shape/content mismatch (counted by the caller as a load failure).
    """
    import json

    with open(head_path, "r", encoding="utf-8") as fh:
        raw = json.load(fh)
    labels = [str(x).lower() for x in raw.get("labels") or []]
    weight = np.asarray(raw.get("weight"), dtype=np.float32)
    bias = np.asarray(raw.get("bias"), dtype=np.float32)
    if not labels or weight.ndim != 2 or bias.ndim != 1:
        return None
    if weight.shape[0] != len(labels) or bias.shape[0] != len(labels):
        return None
    return {"labels": labels, "weight": weight, "bias": bias}


def _model_verdict(
    values: Any, np: Any, head: Optional[dict], labels: Optional[list]
) -> Optional[tuple]:
    """Pure postprocessing: raw ONNX output -> (label, confidence) or None.

    Two contracts, decided by which sidecar was found at load time:
    * ``head`` set (emotion2vec+): output is backbone features
      ``[1, T, D]``/``[T, D]`` — mean-pool over time, then the head's
      ``logits = weight @ pooled + bias``.
    * ``labels`` set: output IS the logits, index-ordered by ``labels``.
    Softmax-argmax either way; MODEL_ABSTAIN_LABELS return None.
    """
    arr = np.asarray(values, dtype=np.float32)
    if head is not None:
        if arr.ndim == 3:
            arr = arr[0]
        if arr.ndim != 2 or arr.shape[0] == 0:
            return None
        pooled = arr.mean(axis=0)
        if pooled.shape[0] != head["weight"].shape[1]:
            return None
        logits = head["weight"] @ pooled + head["bias"]
        label_list = head["labels"]
    else:
        logits = arr.reshape(-1)
        label_list = labels or []
    if logits.size == 0 or logits.size != len(label_list):
        return None
    exp = np.exp(logits - logits.max())
    probs = exp / exp.sum()
    idx = int(np.argmax(probs))
    label = str(label_list[idx]).lower()
    if label in MODEL_ABSTAIN_LABELS:
        telemetry.incr("vocal_emotion.model_abstained")
        return None
    return label, float(probs[idx])


class VocalEmotionModel:
    """Callable wrapper around the ONNX speech-emotion classifier.

    ``model(pcm16, sample_rate) -> Optional[VocalEmotionResult]``. Input
    contract: mono float32 waveform ``[1, N]`` at 16 kHz, dynamic length —
    the wrapper resamples, truncates to ``MAX_WINDOW_SECONDS``, and applies
    zero-mean/unit-variance normalization (idempotent for the pinned
    emotion2vec+ export, which folds the same normalization into its graph;
    required by wav2vec2-style models whose preprocessor declares
    ``do_normalize``). Output handling is sidecar-driven — see
    ``_model_verdict``. A model directory with NEITHER a ``*head.json`` nor
    a ``labels.json`` sidecar refuses to load: guessing label order by
    position shipped a fictional contract once, never again
    (scripts/fetch_vocal_emotion.py fetches the sidecar alongside the
    model).
    """

    MAX_WINDOW_SECONDS = 8
    MIN_WINDOW_SECONDS = 0.3
    MAX_CONSECUTIVE_ERRORS = 3

    def __init__(
        self,
        model_path: str,
        head: Optional[dict] = None,
        labels: Optional[list] = None,
        session: Optional[Any] = None,
    ) -> None:
        import numpy as np

        if head is None and not labels:
            raise ValueError(
                "vocal-emotion model needs a *head.json or labels.json "
                "sidecar to know its label order"
            )
        self._np = np
        if session is None:  # test seam: inject a fake session
            import onnxruntime

            session = onnxruntime.InferenceSession(
                str(model_path), providers=["CPUExecutionProvider"]
            )
        self._session = session
        self._input_name = self._session.get_inputs()[0].name
        self._head = head
        self._labels = [str(x).lower() for x in labels] if labels else None
        self._consecutive_errors = 0

    def __call__(
        self, pcm16: bytes, sample_rate: int = TARGET_SAMPLE_RATE
    ) -> Optional[VocalEmotionResult]:
        if self._consecutive_errors >= self.MAX_CONSECUTIVE_ERRORS:
            return None
        try:
            np = self._np
            audio = _pcm16_to_float(pcm16, np)
            if int(sample_rate) != TARGET_SAMPLE_RATE and sample_rate > 0:
                duration = audio.size / float(sample_rate)
                target_n = max(1, int(duration * TARGET_SAMPLE_RATE))
                audio = np.interp(
                    np.linspace(0.0, max(0, audio.size - 1), target_n),
                    np.arange(audio.size),
                    audio,
                ).astype(np.float32)
            if audio.size < TARGET_SAMPLE_RATE * self.MIN_WINDOW_SECONDS:
                return None  # under ~300ms: not enough to judge
            max_samples = TARGET_SAMPLE_RATE * self.MAX_WINDOW_SECONDS
            if audio.size > max_samples:
                # Keep the END of the utterance — that's where the tone of
                # what they just said lives (and what the reply follows).
                audio = audio[-max_samples:]
            std = float(audio.std())
            if std < 1e-6:
                return None  # effectively silent clip
            audio = (audio - float(audio.mean())) / std
            outputs = self._session.run(
                None, {self._input_name: audio[np.newaxis, :].astype(np.float32)}
            )
            verdict = _model_verdict(outputs[0], np, self._head, self._labels)
            self._consecutive_errors = 0
            if verdict is None:
                return None
            label, confidence = verdict
            return VocalEmotionResult(
                label=label, confidence=confidence, source="model"
            )
        except Exception as exc:
            self._consecutive_errors += 1
            telemetry.incr("vocal_emotion.model_errors")
            LOGGER.warning("vocal-emotion model inference failed: %s", exc)
            return None


def _discover_model_path(config: Any) -> Optional[Path]:
    configured = getattr(getattr(config, "vocal_emotion", None), "model_path", None)
    if configured:
        path = Path(str(configured))
        return path if path.is_file() else None
    directory = Path(os.environ.get(MODEL_DIR_ENV) or DEFAULT_MODEL_DIR)
    if not directory.is_dir():
        return None
    candidates = sorted(directory.glob("*.onnx"))
    return candidates[0] if candidates else None


def _discover_sidecars(model_path: Path, np: Any) -> tuple:
    """(head, labels) sidecars from the model's directory.

    ``*head.json`` (feature-backbone contract) wins over ``labels.json``
    (logits contract) when both exist. Malformed sidecars read as absent —
    the caller then refuses the load and says why.
    """
    directory = model_path.parent
    head = None
    for head_path in sorted(directory.glob("*head.json")):
        try:
            head = _load_head(head_path, np)
        except Exception as exc:
            LOGGER.warning(
                "vocal-emotion head sidecar unreadable (%s): %s", head_path, exc
            )
            head = None
        if head is not None:
            break
    labels = None
    labels_path = directory / "labels.json"
    if labels_path.is_file():
        try:
            import json

            with open(labels_path, "r", encoding="utf-8") as fh:
                raw = json.load(fh)
            if isinstance(raw, list) and raw:
                labels = [str(x) for x in raw]
        except Exception as exc:
            LOGGER.warning(
                "vocal-emotion labels.json unreadable (%s): %s", labels_path, exc
            )
    return head, labels


# Process-wide cache, same shape as smart_turn.py's _CACHE: the model loads
# once, and a definitive miss is remembered so every analysis call doesn't
# re-probe the filesystem or re-count the same unavailability.
_CACHE: dict = {"model": None, "resolved": False}


def load_vocal_emotion_model(config: Any) -> Optional[VocalEmotionModel]:
    """The configured ONNX model, or None (disabled/missing — counted once).

    Only probed at all when ``vocal_emotion.mode`` is "auto" or "model" —
    "heuristic" mode never touches onnxruntime or the filesystem.
    """
    mode = str(
        getattr(getattr(config, "vocal_emotion", None), "mode", "auto") or "auto"
    )
    if mode == "heuristic":
        return None
    if _CACHE["resolved"]:
        return _CACHE["model"]
    _CACHE["resolved"] = True
    path = _discover_model_path(config)
    if path is None:
        telemetry.incr("vocal_emotion.model_unavailable")
        LOGGER.info(
            "vocal-emotion model not found (looked in %s) — vocal tone "
            "analysis stays heuristic-only; run "
            "scripts/fetch_vocal_emotion.py to enable the model upgrade",
            os.environ.get(MODEL_DIR_ENV) or DEFAULT_MODEL_DIR,
        )
        return None
    try:
        import numpy as np

        head, labels = _discover_sidecars(path, np)
        if head is None and not labels:
            telemetry.incr("vocal_emotion.model_unavailable")
            LOGGER.warning(
                "vocal-emotion model at %s has no *head.json or labels.json "
                "sidecar — label order would be a guess, refusing to load "
                "(re-run scripts/fetch_vocal_emotion.py, which fetches the "
                "sidecar alongside the model)",
                path,
            )
            return None
        _CACHE["model"] = VocalEmotionModel(str(path), head=head, labels=labels)
        LOGGER.info("vocal-emotion model loaded from %s", path)
    except Exception as exc:
        telemetry.incr("vocal_emotion.model_unavailable")
        LOGGER.warning("vocal-emotion model failed to load (%s): %s", path, exc)
        _CACHE["model"] = None
    return _CACHE["model"]


# Process-wide speaker baseline. One instance because one person talks to
# this assistant — the same reason the TTS speaker guard keeps a single
# reference. Reset between tests via reset_speaker_baseline().
_BASELINE = SpeakerBaseline()


def reset_speaker_baseline() -> None:
    """Drop the accumulated speaker norm (test seam / speaker change)."""
    global _BASELINE
    _BASELINE = SpeakerBaseline()


def analyze(
    pcm16: bytes,
    sample_rate: int,
    config: Any,
) -> Optional[VocalEmotionResult]:
    """Top-level entry point: the speaker-relative prosody heuristic, or an
    explicitly-configured ONNX model. Never raises — a failure here must
    never break STT, so callers that need a hard guarantee should still wrap
    this in their own try/except and count
    ``vocal_emotion.analyze_failures`` (see
    ``vocal_tone_stash.analyze_and_stash``), but this function itself
    already catches everything it can anticipate.

    ``mode: "auto"`` deliberately does NOT prefer the model. Measured
    2026-08-05 against the 17 voice-acted reference clips: emotion2vec+
    base returned SIX different emotions and not one "neutral" for ordinary
    spoken lines, at a median confidence of 0.997 (65% of verdicts above
    0.99) — three read "disgusted" at 1.000. Being most certain when it is
    wrong makes ``min_confidence`` useless as a guard, and a confidently
    false "the user sounded put off" line is worse than saying nothing. The
    model path stays implemented and tested so a better-calibrated model
    can be dropped in, but it is opt-in via ``mode: "model"`` only.
    """
    cfg = getattr(config, "vocal_emotion", None)
    min_confidence = float(getattr(cfg, "min_confidence", 0.5) or 0.5)
    mode = str(getattr(cfg, "mode", "auto") or "auto")
    if mode == "model":
        model = load_vocal_emotion_model(config)
        if model is None:
            return None
        result = model(pcm16, sample_rate)
        if result is not None and result.confidence >= min_confidence:
            return result
        return None  # model mode never falls back to the heuristic
    return analyze_heuristic(
        pcm16, sample_rate, min_confidence=min_confidence, baseline=_BASELINE
    )
