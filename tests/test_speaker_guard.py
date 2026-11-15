"""Speaker-consistency guard: retry policy and scoring plumbing."""

import numpy as np
import pytest

# speaker_guard imports soundfile (voice extra) at module level; CI installs
# only [web], so skip there like the other heavy-dep test files.
pytest.importorskip("soundfile")

from amadeus.providers.tts.speaker_guard import SpeakerGuard


class FakeEncoder:
    """Maps a marker value baked into the audio to a fixed embedding."""

    def embed_utterance(self, wav):
        # The first sample encodes which "speaker" produced the clip.
        marker = float(np.round(wav[0] / max(np.max(np.abs(wav)), 1e-6), 1))
        if marker >= 0.9:
            vec = np.array([1.0, 0.0])  # reference speaker
        elif marker <= -0.9:
            vec = np.array([0.0, 1.0])  # drifted speaker
        else:
            vec = np.array([0.7071, 0.7071])  # in between
        return vec


def _clip(marker: float, seconds: float = 2.0, rate: int = 16000):
    wav = np.full(int(seconds * rate), 0.5, dtype=np.float32)
    wav[0] = marker
    return wav, rate


@pytest.fixture()
def guard(tmp_path):
    import soundfile as sf

    ref, rate = _clip(1.0)
    ref_path = tmp_path / "ref.wav"
    sf.write(ref_path, ref, rate)
    return SpeakerGuard(
        str(ref_path), threshold=0.9, max_retries=2, encoder=FakeEncoder()
    )


def test_on_voice_clip_passes_first_try(guard):
    calls = []

    def generate():
        calls.append(1)
        return _clip(1.0)

    _, _, score = guard.best_of(generate)
    assert len(calls) == 1
    assert score is not None and score >= 0.9


def test_drifted_clip_triggers_regeneration(guard):
    rolls = [_clip(-1.0), _clip(1.0)]

    def generate():
        return rolls.pop(0)

    samples, _, score = guard.best_of(generate)
    assert not rolls  # both rolls consumed
    assert score is not None and score >= 0.9
    assert samples[0] == 1.0  # kept the on-voice clip


def test_exhausted_retries_keeps_best_scoring(guard):
    rolls = [_clip(-1.0), _clip(0.0), _clip(-1.0)]

    def generate():
        return rolls.pop(0)

    samples, _, score = guard.best_of(generate)
    assert not rolls  # 1 + max_retries(2) attempts
    assert samples[0] == 0.0  # the in-between clip scored highest
    assert score == pytest.approx(0.7071, abs=1e-3)


def test_scoring_failure_passes_clip_through(tmp_path):
    import soundfile as sf

    class ExplodingEncoder(FakeEncoder):
        def __init__(self):
            self.calls = 0

        def embed_utterance(self, wav):
            self.calls += 1
            if self.calls > 1:  # ref embed works, clip scoring explodes
                raise RuntimeError("encoder died")
            return np.array([1.0, 0.0])

    ref, rate = _clip(1.0)
    ref_path = tmp_path / "ref.wav"
    sf.write(ref_path, ref, rate)
    guard = SpeakerGuard(
        str(ref_path), threshold=0.9, max_retries=2, encoder=ExplodingEncoder()
    )

    samples, _, score = guard.best_of(lambda: _clip(-1.0))
    assert score is None  # unjudged, but synthesis still returned audio
    assert samples[0] == -1.0


def test_short_clips_are_tiled_to_embeddable_length(guard):
    wav = guard._to_encoder_audio(np.full(1600, 0.5, dtype=np.float32), 16000)
    assert wav.size >= int(1.7 * 16000)


def test_default_thresholds_pin_the_2026_07_28_calibration(tmp_path):
    """Pin the measured thresholds: the closest same-gender impostor voices
    score 0.80-0.82 against the Kurisu ref (16-voice sweep), so the old
    0.78 long / 0.60 short defaults passed near-voice drift outright. A
    refactor silently reverting them would reopen that hole."""
    import inspect

    sig = inspect.signature(SpeakerGuard.__init__)
    assert sig.parameters["threshold"].default == 0.84
    assert sig.parameters["short_clip_threshold"].default == 0.70
