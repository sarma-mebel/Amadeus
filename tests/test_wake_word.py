"""Wake-word listener: detector state machine, loader degradation, socket
runner protocol, and the route-level gate.

Pure-offline: openwakeword is faked via sys.modules (the [wake] extra is
not a test dependency), PCM is synthesized with array (same helpers as
test_duplex), and the socket runner is driven with the FakeWebSocket
pattern from test_duplex.py.
"""

import array
import asyncio
import json
import sys
import types
from pathlib import Path

import pytest

from amadeus import telemetry
from amadeus.config import Config
from amadeus.web.streaming import wake_word as wake_module
from amadeus.web.streaming.wake_word import (
    CHUNK_SAMPLES,
    MAX_CONSECUTIVE_ERRORS,
    WakeSocketRunner,
    create_wake_word_detector,
)

pytest.importorskip("numpy")


def _pcm(samples: int, value: int = 1000) -> bytes:
    return array.array("h", [value] * samples).tobytes()


class FakeOwwModel:
    """Scripted stand-in for openwakeword.model.Model: pops one score (or
    raises one exception) per predict() call; repeats the last entry."""

    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.scripted = list(self.__class__.script)
        FakeOwwModel.instances.append(self)

    def predict(self, chunk):
        item = self.scripted.pop(0) if self.scripted else 0.0
        if isinstance(item, Exception):
            raise item
        return {"hey_jarvis": item}


@pytest.fixture()
def fake_openwakeword(monkeypatch):
    """Install a fake openwakeword package; yield the fake Model class.
    Tests set FakeOwwModel.script before constructing a detector."""
    FakeOwwModel.instances = []
    FakeOwwModel.script = []
    pkg = types.ModuleType("openwakeword")
    model_mod = types.ModuleType("openwakeword.model")
    model_mod.Model = FakeOwwModel
    pkg.model = model_mod
    monkeypatch.setitem(sys.modules, "openwakeword", pkg)
    monkeypatch.setitem(sys.modules, "openwakeword.model", model_mod)
    yield FakeOwwModel


@pytest.fixture(autouse=True)
def _reset_cache(monkeypatch):
    monkeypatch.setattr(wake_module, "_CACHE", {"path": None, "resolved": False})


def _config(tmp_path: Path, enabled: bool = True, **overrides) -> Config:
    config = Config()
    config.wake_word.enabled = enabled
    model = tmp_path / "hey_jarvis_v0.1.onnx"
    model.write_bytes(b"onnx")
    config.wake_word.model = str(model)
    for key, value in overrides.items():
        setattr(config.wake_word, key, value)
    return config


# --- detector --------------------------------------------------------------


def test_detector_fires_on_threshold_and_buffers_partial_chunks(
    tmp_path, fake_openwakeword
):
    fake_openwakeword.script = [0.1, 0.2, 0.9]
    detector = create_wake_word_detector(_config(tmp_path))
    assert detector is not None
    # Two half-chunks buffer into one hop (score 0.1) — no detection yet.
    assert detector.feed(_pcm(CHUNK_SAMPLES // 2)) is None
    assert detector.feed(_pcm(CHUNK_SAMPLES // 2)) is None
    assert detector.feed(_pcm(CHUNK_SAMPLES)) is None  # 0.2
    assert detector.feed(_pcm(CHUNK_SAMPLES)) == pytest.approx(0.9)  # fires


def test_detector_refractory_suppresses_echo_then_rearms(tmp_path, fake_openwakeword):
    # Fire, then keep scoring high: the phrase's own tail must not re-fire
    # inside the refractory window, but a later detection (after enough
    # AUDIO time) must.
    fake_openwakeword.script = [0.9, 0.9, 0.0, 0.0, 0.0, 0.9]
    config = _config(tmp_path, refractory_seconds=0.2)  # 3200 samples
    detector = create_wake_word_detector(config)
    assert detector.feed(_pcm(CHUNK_SAMPLES)) == pytest.approx(0.9)
    assert detector.feed(_pcm(CHUNK_SAMPLES)) is None  # echo, in refractory
    for _ in range(3):  # 0.0 scores let the window lapse (0.24s of audio)
        assert detector.feed(_pcm(CHUNK_SAMPLES)) is None
    assert detector.feed(_pcm(CHUNK_SAMPLES)) == pytest.approx(0.9)


def test_detector_self_disables_after_consecutive_errors(tmp_path, fake_openwakeword):
    fake_openwakeword.script = [
        RuntimeError("boom") for _ in range(MAX_CONSECUTIVE_ERRORS)
    ] + [0.9]
    detector = create_wake_word_detector(_config(tmp_path))
    before = telemetry.snapshot().get("web.wake_word_errors", 0)
    for _ in range(MAX_CONSECUTIVE_ERRORS):
        assert detector.feed(_pcm(CHUNK_SAMPLES)) is None
    # Disabled: the 0.9 that would have fired is never even evaluated.
    assert detector.feed(_pcm(CHUNK_SAMPLES)) is None
    assert (
        telemetry.snapshot()["web.wake_word_errors"] == before + MAX_CONSECUTIVE_ERRORS
    )


# --- loader ----------------------------------------------------------------


def test_loader_missing_model_counts_and_caches(tmp_path, monkeypatch):
    monkeypatch.setenv(wake_module.MODEL_DIR_ENV, str(tmp_path / "missing"))
    config = Config()
    config.wake_word.enabled = True
    before = telemetry.snapshot().get("web.wake_word_unavailable", 0)
    assert create_wake_word_detector(config) is None
    assert create_wake_word_detector(config) is None
    # Both misses count (visible), but discovery ran once (cached).
    assert telemetry.snapshot()["web.wake_word_unavailable"] == before + 2


def test_loader_missing_package_degrades(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "openwakeword", None)
    monkeypatch.setitem(sys.modules, "openwakeword.model", None)
    before = telemetry.snapshot().get("web.wake_word_unavailable", 0)
    assert create_wake_word_detector(_config(tmp_path)) is None
    assert telemetry.snapshot()["web.wake_word_unavailable"] == before + 1


def test_loader_discovers_model_dir_skipping_feature_models(
    tmp_path, monkeypatch, fake_openwakeword
):
    model_dir = tmp_path / "wake"
    model_dir.mkdir()
    (model_dir / "melspectrogram.onnx").write_bytes(b"feat")
    (model_dir / "embedding_model.onnx").write_bytes(b"feat")
    (model_dir / "hey_jarvis_v0.1.onnx").write_bytes(b"model")
    monkeypatch.setenv(wake_module.MODEL_DIR_ENV, str(model_dir))
    config = Config()
    config.wake_word.enabled = True
    detector = create_wake_word_detector(config)
    assert detector is not None
    kwargs = fake_openwakeword.instances[-1].kwargs
    assert kwargs["wakeword_models"] == [str(model_dir / "hey_jarvis_v0.1.onnx")]
    # Sibling feature models get passed through explicitly.
    assert kwargs["melspec_model_path"] == str(model_dir / "melspectrogram.onnx")
    assert kwargs["embedding_model_path"] == str(model_dir / "embedding_model.onnx")


def test_loader_disabled_returns_none_without_counting(tmp_path):
    before = telemetry.snapshot().get("web.wake_word_unavailable", 0)
    assert create_wake_word_detector(_config(tmp_path, enabled=False)) is None
    assert telemetry.snapshot().get("web.wake_word_unavailable", 0) == before


# --- socket runner ---------------------------------------------------------


class FakeWebSocket:
    def __init__(self, inbound):
        self._inbound = list(inbound)
        self.sent = []

    async def receive(self):
        if not self._inbound:
            return {"type": "websocket.disconnect"}
        item = self._inbound.pop(0)
        if isinstance(item, (bytes, bytearray)):
            return {"type": "websocket.receive", "bytes": bytes(item)}
        return {"type": "websocket.receive", "text": json.dumps(item)}

    async def send_json(self, frame):
        self.sent.append(frame)


class ScriptedDetector:
    def __init__(self, scores):
        self.scores = list(scores)
        self.fed = []

    def feed(self, pcm):
        self.fed.append(pcm)
        return self.scores.pop(0) if self.scores else None


def test_runner_emits_ready_then_wake_frames(tmp_path):
    detector = ScriptedDetector([None, 0.91])
    socket = FakeWebSocket(
        [
            {"type": "start"},
            _pcm(CHUNK_SAMPLES),
            _pcm(CHUNK_SAMPLES),
            {"type": "stop"},
        ]
    )
    runner = WakeSocketRunner(
        socket, _config(tmp_path), detector_factory=lambda config: detector
    )
    before = telemetry.snapshot().get("web.wake_detections", 0)
    asyncio.run(runner.run())
    assert socket.sent[0]["type"] == "ready"
    wakes = [f for f in socket.sent if f["type"] == "wake"]
    assert len(wakes) == 1 and wakes[0]["score"] == pytest.approx(0.91)
    assert len(detector.fed) == 2
    assert telemetry.snapshot()["web.wake_detections"] == before + 1


def test_runner_unavailable_detector_sends_error_frame(tmp_path):
    socket = FakeWebSocket([{"type": "start"}])
    runner = WakeSocketRunner(
        socket, _config(tmp_path), detector_factory=lambda config: None
    )
    asyncio.run(runner.run())
    assert socket.sent == [
        {
            "type": "error",
            "kind": "wake_unavailable",
            "message": socket.sent[0]["message"],
        }
    ]
    assert "fetch_wake_word" in socket.sent[0]["message"]


def test_runner_client_disconnect_before_start_is_clean(tmp_path):
    socket = FakeWebSocket([])
    runner = WakeSocketRunner(
        socket, _config(tmp_path), detector_factory=lambda config: ScriptedDetector([])
    )
    asyncio.run(runner.run())
    assert socket.sent == []


# --- route gate ------------------------------------------------------------


def test_wake_route_disabled_sends_error_frame(tmp_path, monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from amadeus.web import app as app_module

    config = Config()
    config.wake_word.enabled = False
    config.memory.persist_path = str(tmp_path / "memory.json")
    config.memory.embeddings_path = None
    config.memory.profile_path = str(tmp_path / "profile.json")
    app = app_module.create_app(config)
    client = TestClient(app)
    with client.websocket_connect("/api/wake") as ws:
        frame = ws.receive_json()
    assert frame["kind"] == "wake_disabled"
