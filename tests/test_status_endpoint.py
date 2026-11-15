"""Tests for GET /api/status — runtime provider truth (not config-shape)."""

import pytest

pytest.importorskip("fastapi")

import httpx

from amadeus.config import Config
from amadeus.web.app import create_app


def _client(app):
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://testserver")


def _isolated_config(tmp_path) -> Config:
    config = Config()
    config.memory.persist_path = str(tmp_path / "memory.json")
    config.memory.embeddings_path = str(tmp_path / "embeddings.json")
    config.memory.profile_path = str(tmp_path / "profile.json")
    return config


class _FakeTTS:
    _engine = "faster"
    cache_signature = "fake|sig"

    def synthesize(self, text, *, speaking_rate=1.0, pitch_shift=0.0):
        return b"wav"


async def test_status_reports_dummy_llm(tmp_path):
    """No API key configured -> the runtime LLM is the dummy, and status says so."""
    app = create_app(_isolated_config(tmp_path))
    async with _client(app) as client:
        response = await client.get("/api/status")
    assert response.status_code == 200
    payload = response.json()
    assert payload["llm"]["is_dummy"] is True
    assert payload["llm"]["class"] == "DummyProvider"
    assert payload["memory"]["summary_provider_active"] is False
    assert payload["version"]


async def test_status_tts_shape_when_voice_disabled(tmp_path):
    config = _isolated_config(tmp_path)
    config.audio.enable_voice = False
    app = create_app(config)
    async with _client(app) as client:
        payload = (await client.get("/api/status")).json()
    assert payload["tts"]["requested"] is None
    assert payload["tts"]["loaded"] is None
    assert payload["tts"]["warming"] is False
    assert payload["tts"]["fallback_from"] is None


async def test_status_reports_tts_fallback(monkeypatch, tmp_path):
    """qwen3 load failure -> ONNX fallback serves, and status records the swap."""
    import amadeus.web.app as app_module

    fake = _FakeTTS()

    def fake_load_tts(cfg):
        if cfg.providers.tts == "qwen3_tts":
            raise RuntimeError("no CUDA for you")
        return fake

    monkeypatch.setattr(app_module, "load_tts", fake_load_tts)
    config = _isolated_config(tmp_path)
    config.audio.enable_voice = True
    config.providers.tts = "qwen3_tts"
    app = create_app(config)

    # The qwen3 path warms on a background thread; make the result
    # deterministic by loading synchronously on the store.
    store = app.state.session_store
    store.tts = store._load_tts_blocking(config)
    store.tts_warming = False

    async with _client(app) as client:
        payload = (await client.get("/api/status")).json()
    assert payload["tts"]["requested"] == "qwen3_tts"
    assert payload["tts"]["fallback_from"] == "qwen3_tts"
    assert payload["tts"]["loaded"] == "melotts_onnx"
    assert payload["tts"]["error"] == "no CUDA for you"
    assert payload["tts"]["engine"] == "faster"
    # Healthy fast engine -> no slow-engine warning.
    assert payload["tts"]["engine_warning"] is None


async def test_status_flags_slow_qwen3_engine_fallback(monkeypatch, tmp_path):
    """qwen3 loaded on the upstream engine (faster-qwen3-tts missing) is a
    silent ~10x synthesis slowdown — status must call it out (additively)."""
    import amadeus.web.app as app_module

    class _SlowEngineTTS(_FakeTTS):
        _engine = "upstream"

    monkeypatch.setattr(app_module, "load_tts", lambda cfg: _SlowEngineTTS())
    config = _isolated_config(tmp_path)
    config.audio.enable_voice = True
    config.providers.tts = "qwen3_tts"
    app = create_app(config)
    store = app.state.session_store
    store.tts = store._load_tts_blocking(config)
    store.tts_warming = False

    async with _client(app) as client:
        payload = (await client.get("/api/status")).json()
    assert payload["tts"]["loaded"] == "qwen3_tts"
    assert payload["tts"]["engine"] == "upstream"
    assert "faster-qwen3-tts" in payload["tts"]["engine_warning"]
    assert "slow" in payload["tts"]["engine_warning"]


async def test_status_clears_fallback_after_reload(monkeypatch, tmp_path):
    import amadeus.web.app as app_module

    fake = _FakeTTS()

    def fake_load_tts(cfg):
        if cfg.providers.tts == "qwen3_tts":
            raise RuntimeError("boom")
        return fake

    monkeypatch.setattr(app_module, "load_tts", fake_load_tts)
    config = _isolated_config(tmp_path)
    config.audio.enable_voice = True
    config.providers.tts = "qwen3_tts"
    app = create_app(config)
    store = app.state.session_store
    store.tts = store._load_tts_blocking(config)
    store.tts_warming = False
    assert store.tts_fallback_from == "qwen3_tts"

    new_config = _isolated_config(tmp_path)
    new_config.audio.enable_voice = True
    new_config.providers.tts = "melotts_onnx"
    store.reload_providers(new_config)

    async with _client(app) as client:
        payload = (await client.get("/api/status")).json()
    assert payload["tts"]["fallback_from"] is None
    assert payload["tts"]["error"] is None
    assert payload["tts"]["loaded"] == "melotts_onnx"


async def test_settings_llm_is_dummy_uses_runtime_truth(monkeypatch, tmp_path):
    """A config that LOOKS configured but whose runtime LLM is the dummy must
    report llm_is_dummy from the live instance, not the config shape."""
    import amadeus.web.app as app_module

    from amadeus.providers.llm.base import DummyProvider

    # Simulate a runtime construction failure: the loader falls back to the
    # dummy even though the config looks fully configured. (Stubbing the
    # loader also keeps CI off the google-generativeai optional dep.)
    monkeypatch.setattr(app_module, "load_llm", lambda cfg: DummyProvider())
    config = _isolated_config(tmp_path)
    config.providers.llm = "gemini"
    config.gemini_api_key = "test-key-not-real"
    app = create_app(config)

    async with _client(app) as client:
        payload = (await client.get("/api/settings")).json()
    assert payload["providers"]["llm_is_dummy"] is True
    # First-run remains a config-shape question: a key is set, so not first run.
    assert payload["is_first_run"] is False


async def test_status_tools_shape_and_unavailable_reason(tmp_path, monkeypatch):
    """/api/status tools block: enabled flag, registered names, and
    requested-but-unavailable entries with human-readable reasons."""
    import sys

    monkeypatch.setitem(sys.modules, "ddgs", None)  # no ddgs installed
    config = _isolated_config(tmp_path)
    config.tools.enabled = True
    config.web_search.enabled = True
    config.web_search.provider = "auto"
    config.web_search.api_key = None  # no Tavily key -> duckduckgo -> no ddgs
    app = create_app(config)
    async with _client(app) as client:
        payload = (await client.get("/api/status")).json()

    tools = payload["tools"]
    assert set(tools.keys()) == {"enabled", "registered", "unavailable"}
    assert tools["enabled"] is True
    assert "calculator" in tools["registered"]
    assert "code_interpreter" in tools["registered"]
    assert "web_search" not in tools["registered"]
    assert [entry["name"] for entry in tools["unavailable"]] == ["web_search"]
    assert "ddgs" in tools["unavailable"][0]["reason"]


async def test_status_tools_block_when_tools_disabled(tmp_path):
    config = _isolated_config(tmp_path)
    app = create_app(config)
    async with _client(app) as client:
        payload = (await client.get("/api/status")).json()
    tools = payload["tools"]
    assert tools["enabled"] is False
    # Registry always carries the built-ins; nothing was requested and missing.
    assert "calculator" in tools["registered"]
    assert tools["unavailable"] == []
