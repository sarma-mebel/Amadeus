"""Variable "think delay" before the first streamed frame.

Two layers:
- pure ``compute_think_delay`` / ``signals_for_turn`` (deterministic, seeded rng)
- integration against the WS streaming path: the beat is bounded, cancel-safe,
  and inert when disabled — with mocked providers (no live API calls).
"""

from __future__ import annotations

import random
import threading
import time

import pytest

from amadeus.config import Config
from amadeus.providers.llm.base import DummyProvider
from amadeus.providers.tts.base import TTSProvider
from amadeus.web.app import ChatRequest, create_app
from amadeus.web.streaming.think_delay import (
    ThinkDelaySignals,
    compute_think_delay,
    signals_for_turn,
)

# --------------------------------------------------------------------------
# Pure delay-computation function
# --------------------------------------------------------------------------


def _fixed_rng(value: float) -> random.Random:
    """An rng whose .random() always returns ``value`` — pins the jitter."""

    class _Fixed(random.Random):
        def random(self):  # noqa: D401 - trivial
            return value

    return _Fixed()


def test_signals_for_turn_reads_length_and_hint():
    # Long, causal question -> long bucket + heavy hint (>300 chars).
    sig = signals_for_turn(
        "Why does dopamine reinforce reward learning, and how does that "
        "interact with memory consolidation over long timescales in the "
        "hippocampus and cortex? Explain the mechanism carefully, and please "
        "walk through the receptor-level dynamics, the phasic versus tonic "
        "firing distinction, and how prediction error signals are thought to "
        "propagate through the basal ganglia over the course of learning."
    )
    assert sig.len_bucket == "long"
    assert sig.heavy_hint is True

    # Bare quip -> short bucket, no heavy hint.
    quip = signals_for_turn("lol nice")
    assert quip.len_bucket == "short"
    assert quip.heavy_hint is False

    # Emotion is threaded through untouched when provided.
    assert signals_for_turn("hi", emotion="sad").emotion == "sad"
    assert signals_for_turn("hi").emotion is None


def test_compute_delay_disabled_range_is_zero():
    sig = signals_for_turn("Why does this happen?")
    # max <= 0 -> feature effectively off regardless of signals.
    assert compute_think_delay(sig, 0.0, 0.0) == 0.0


def test_compute_delay_is_bounded():
    """No signal combination may exceed the configured ceiling or dip below
    the floor, across the whole jitter range."""
    heavy = ThinkDelaySignals(len_bucket="long", heavy_hint=True, emotion="sad")
    trivial = ThinkDelaySignals(len_bucket="empty", heavy_hint=False, emotion=None)
    for r in (0.0, 0.25, 0.5, 0.75, 1.0):
        rng = _fixed_rng(r)
        d_heavy = compute_think_delay(heavy, 0.1, 1.2, rng=_fixed_rng(r))
        d_trivial = compute_think_delay(trivial, 0.1, 1.2, rng=rng)
        assert 0.1 <= d_heavy <= 1.2
        assert 0.1 <= d_trivial <= 1.2


def test_compute_delay_heavy_beats_quip():
    """A heavy/emotional/long turn earns a longer beat than a bare quip, at the
    same jitter draw."""
    heavy = ThinkDelaySignals(len_bucket="long", heavy_hint=True, emotion="sad")
    quip = ThinkDelaySignals(len_bucket="short", heavy_hint=False, emotion=None)
    d_heavy = compute_think_delay(heavy, 0.0, 1.2, rng=_fixed_rng(0.5))
    d_quip = compute_think_delay(quip, 0.0, 1.2, rng=_fixed_rng(0.5))
    assert d_heavy > d_quip
    # The quip is near the floor; the heavy turn is a real pause.
    assert d_quip < 0.2
    assert d_heavy > 0.5


def test_compute_delay_min_ge_max_collapses_to_fixed():
    sig = signals_for_turn("Why?")
    assert compute_think_delay(sig, 0.8, 0.5) == 0.5  # min>=max -> fixed at max


def test_compute_delay_bad_bounds_degrade_to_zero():
    sig = signals_for_turn("Why?")
    assert compute_think_delay(sig, float("nan"), 1.0) == 0.0
    assert compute_think_delay(sig, "x", 1.0) == 0.0  # type: ignore[arg-type]
    # Negative ceiling is treated as off.
    assert compute_think_delay(sig, 0.0, -1.0) == 0.0


def test_compute_delay_emotion_adds_beat():
    """A sad first sentence lengthens the beat versus neutral, same jitter."""
    base = ThinkDelaySignals(len_bucket="short", heavy_hint=False, emotion="neutral")
    sad = ThinkDelaySignals(len_bucket="short", heavy_hint=False, emotion="sad")
    d_neutral = compute_think_delay(base, 0.0, 1.2, rng=_fixed_rng(1.0))
    d_sad = compute_think_delay(sad, 0.0, 1.2, rng=_fixed_rng(1.0))
    assert d_sad > d_neutral


# --------------------------------------------------------------------------
# Integration against the WS streaming path (mocked providers)
# --------------------------------------------------------------------------


class ScriptedProvider(DummyProvider):
    """A DummyProvider whose scripted_reply streams like a real model."""


class FakeTTS(TTSProvider):
    def __init__(self) -> None:
        self.calls: list[str] = []

    def synthesize(
        self, text: str, *, speaking_rate: float = 1.0, pitch_shift: float = 0.0
    ) -> bytes:
        self.calls.append(text)
        return f"WAV:{text}".encode()


def _config(tmp_path) -> Config:
    config = Config()
    config.web.streaming_enabled = True
    config.web.streaming_tagged_output = True
    config.audio.enable_voice = True
    config.memory.persist_path = str(tmp_path / "memory.json")
    config.memory.embeddings_path = str(tmp_path / "embeddings.json")
    config.memory.enable_long_term = False
    return config


def _app(monkeypatch, config, reply):
    scripted = ScriptedProvider(scripted_reply=reply)
    fake_tts = FakeTTS()
    import amadeus.web.app as app_module

    monkeypatch.setattr(app_module, "load_llm", lambda cfg: scripted)
    monkeypatch.setattr(app_module, "load_tts", lambda cfg: fake_tts)
    return create_app(config), scripted, fake_tts


async def _run_stream(app, payload, cancel_after=None):
    store = app.state.session_store
    session_id, session = store.get(payload.get("session_id"), payload.get("user_id"))
    frames = [{"type": "session", "session_id": session_id}]
    cancel_event = threading.Event()

    async def emit(frame):
        frames.append(frame)
        if cancel_after is not None and len(frames) >= cancel_after:
            cancel_event.set()

    request = ChatRequest.model_validate(payload)
    await session.handle_stream(request, emit, cancel_event)
    return session_id, frames


_REPLY = (
    "[sad] I have thought about this a lot. || たくさん考えたの。\n"
    "[relaxed] But there is no simple answer. || でも簡単な答えはないわ。"
)


@pytest.mark.asyncio
async def test_delay_disabled_matches_baseline(monkeypatch, tmp_path):
    """think_delay_enabled=False -> no added latency, full frame set intact."""
    config = _config(tmp_path)
    config.web.think_delay_enabled = False
    app, _s, _t = _app(monkeypatch, config, _REPLY)

    t0 = time.monotonic()
    _sid, frames = await _run_stream(
        app, {"message": "Why is love so hard to reason about?", "user_id": "nd-off"}
    )
    elapsed = time.monotonic() - t0

    assert frames[-1]["type"] == "done"
    assert frames[-1]["cancelled"] is False
    # Two sentences still arrive in order.
    sentences = [f for f in frames if f["type"] == "sentence"]
    assert [s["emotion"] for s in sentences] == ["sad", "relaxed"]
    # No think delay was applied (generous ceiling to stay robust on CI).
    assert elapsed < 0.5


@pytest.mark.asyncio
async def test_delay_enabled_preserves_frame_contract(monkeypatch, tmp_path):
    """With the beat on, the frame order/contract is unchanged and memory is
    still committed exactly once — the delay is purely a pause."""
    config = _config(tmp_path)
    config.web.think_delay_enabled = True
    config.web.think_delay_min_seconds = 0.0
    # Tiny ceiling so the test stays fast while still exercising a real sleep.
    config.web.think_delay_max_seconds = 0.2
    app, _s, _t = _app(monkeypatch, config, _REPLY)

    session_id, frames = await _run_stream(
        app,
        {"message": "Why does grief linger for so long, really?", "user_id": "nd-on"},
    )

    types_seen = [f["type"] for f in frames]
    assert types_seen[0] == "session"
    assert types_seen[-1] == "done"
    # token precedes audio; audio seqs ascend — same invariants as the baseline.
    first_token = types_seen.index("token")
    audio_idx = [i for i, t in enumerate(types_seen) if t == "audio"]
    assert audio_idx and first_token < audio_idx[0]
    audio_seqs = [frames[i]["seq"] for i in audio_idx]
    assert audio_seqs == sorted(audio_seqs)

    # Exactly-once commit.
    store = app.state.session_store
    session = store.sessions[session_id]
    assistant = [m for m in session.memory.short_term if m.role == "assistant"]
    assert len(assistant) == 1
    assert "[sad]" not in assistant[0].content


@pytest.mark.asyncio
async def test_cancel_during_beat_aborts_cleanly(monkeypatch, tmp_path):
    """A barge-in that lands while the think-delay is sleeping must abort the
    turn cleanly: a done frame arrives, it's marked cancelled, and the user
    message is not left dangling (commit-exactly-once still holds)."""
    config = _config(tmp_path)
    config.web.think_delay_enabled = True
    config.web.think_delay_min_seconds = 0.9
    config.web.think_delay_max_seconds = 1.0  # long, guaranteed beat
    app, _s, _t = _app(monkeypatch, config, _REPLY)

    store = app.state.session_store
    session_id, session = store.get(None, "nd-cancel")
    frames = [{"type": "session", "session_id": session_id}]
    cancel_event = threading.Event()

    async def emit(frame):
        frames.append(frame)
        # First real frame after 'session' is the first sentence — but the beat
        # fires BEFORE it, so instead trip the cancel from a background timer.

    # Fire the barge-in shortly after the turn starts, i.e. mid-beat.
    def _late_cancel():
        time.sleep(0.1)
        cancel_event.set()

    canceller = threading.Thread(target=_late_cancel)
    canceller.start()

    t0 = time.monotonic()
    request = ChatRequest.model_validate(
        {"message": "Why did it end this way?", "user_id": "nd-cancel"}
    )
    await session.handle_stream(request, emit, cancel_event)
    elapsed = time.monotonic() - t0
    canceller.join()

    done = frames[-1]
    assert done["type"] == "done"
    assert done["cancelled"] is True
    # The wait aborted well before the ~0.9s beat would have completed: the
    # 0.1s cancel + a couple poll slices should land far under 0.6s, proving
    # the sleep did not run to completion.
    assert elapsed < 0.6
    # No audio should have been emitted — cancellation preempted the first frame.
    assert not [f for f in frames if f["type"] == "audio"]
    # Commit-exactly-once: one user + one assistant message, no dangling user.
    session = store.sessions[session_id]
    users = [m for m in session.memory.short_term if m.role == "user"]
    assistants = [m for m in session.memory.short_term if m.role == "assistant"]
    assert len(users) == 1
    assert len(assistants) == 1
