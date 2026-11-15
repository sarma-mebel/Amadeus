"""Tests for the /api/stt and /api/stt/status endpoints.

Uses a mocked STT provider (monkeypatched load_stt) and a tiny in-memory WAV
clip — no live API calls, no model downloads.
"""

import io
import wave

import pytest

pytest.importorskip("fastapi")

import httpx

from amadeus.config import Config
from amadeus.web.app import create_app


def _isolated_config(tmp_path) -> Config:
    config = Config()
    config.memory.persist_path = str(tmp_path / "memory.json")
    config.memory.embeddings_path = str(tmp_path / "embeddings.json")
    config.memory.profile_path = str(tmp_path / "profile.json")
    return config


def _client(app):
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://testserver")


def _make_wav_bytes(sample_rate: int = 16000, num_frames: int = 1600) -> bytes:
    """Tiny mono 16-bit PCM WAV clip of silence."""
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(b"\x00\x00" * num_frames)
    return buffer.getvalue()


class FakeSTT:
    def __init__(self, text: str = "hello kurisu"):
        self.text = text
        self.calls = []

    def transcribe_file(self, path: str) -> str:  # pragma: no cover - unused here
        return self.text

    def transcribe_audio(self, audio_data: bytes, sample_rate: int) -> str:
        self.calls.append((audio_data, sample_rate))
        return self.text


@pytest.mark.asyncio
async def test_stt_endpoint_returns_transcript(monkeypatch, tmp_path):
    import amadeus.web.app as app_module

    fake_stt = FakeSTT("this is a test transcript")
    monkeypatch.setattr(app_module, "load_stt", lambda cfg: fake_stt)

    app = create_app(_isolated_config(tmp_path))
    wav_bytes = _make_wav_bytes()

    async with _client(app) as client:
        response = await client.post(
            "/api/stt",
            files={"audio": ("clip.wav", wav_bytes, "audio/wav")},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["text"] == "this is a test transcript"
    assert len(fake_stt.calls) == 1
    _, sample_rate = fake_stt.calls[0]
    assert sample_rate == 16000


@pytest.mark.asyncio
async def test_stt_endpoint_records_latency_telemetry(monkeypatch, tmp_path):
    """A successful transcription accumulates the latency.stt_* values that
    /api/status consumers average (sum/count) plus the last-value gauge."""
    from amadeus import telemetry

    import amadeus.web.app as app_module

    telemetry.reset()
    try:
        fake_stt = FakeSTT("timed transcript")
        monkeypatch.setattr(app_module, "load_stt", lambda cfg: fake_stt)
        app = create_app(_isolated_config(tmp_path))

        async with _client(app) as client:
            response = await client.post(
                "/api/stt",
                files={"audio": ("clip.wav", _make_wav_bytes(), "audio/wav")},
            )

        assert response.status_code == 200
        snap = telemetry.snapshot()
        assert snap.get("latency.stt_count") == 1
        assert snap.get("latency.stt_ms_sum", -1) >= 0
        assert snap.get("latency.last_stt_ms", -1) >= 0
    finally:
        telemetry.reset()


@pytest.mark.asyncio
async def test_stt_endpoint_returns_501_when_unavailable(monkeypatch, tmp_path):
    import amadeus.web.app as app_module

    monkeypatch.setattr(app_module, "load_stt", lambda cfg: None)

    app = create_app(_isolated_config(tmp_path))
    wav_bytes = _make_wav_bytes()

    async with _client(app) as client:
        response = await client.post(
            "/api/stt",
            files={"audio": ("clip.wav", wav_bytes, "audio/wav")},
        )

    assert response.status_code == 501
    detail = response.json()["detail"]
    assert detail["available"] is False
    assert "reason" in detail


@pytest.mark.asyncio
async def test_stt_status_reports_unavailable_for_dummy_provider(tmp_path):
    config = _isolated_config(tmp_path)
    config.providers.stt = "dummy"
    app = create_app(config)

    async with _client(app) as client:
        response = await client.get("/api/stt/status")

    assert response.status_code == 200
    payload = response.json()
    assert payload["available"] is False


@pytest.mark.asyncio
async def test_stt_status_does_not_load_model(monkeypatch, tmp_path):
    """The status probe must be cheap — it should never call load_stt (which
    would construct/load the actual model)."""
    import amadeus.web.app as app_module

    def _fail_if_called(cfg):
        raise AssertionError("load_stt should not be called by /api/stt/status")

    monkeypatch.setattr(app_module, "load_stt", _fail_if_called)

    config = _isolated_config(tmp_path)
    config.providers.stt = "faster_whisper"
    app = create_app(config)

    async with _client(app) as client:
        response = await client.get("/api/stt/status")

    assert response.status_code == 200
    payload = response.json()
    # Availability reflects whether faster-whisper is importable in this
    # environment — the point of the test is that either way the probe
    # answered WITHOUT raising via the patched load_stt above.
    import importlib.util

    expected = importlib.util.find_spec("faster_whisper") is not None
    assert payload["available"] is expected
