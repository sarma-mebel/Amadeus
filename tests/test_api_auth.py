"""Optional bearer-token auth (web.api_token) + configurable CORS."""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")

import httpx
from starlette.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from amadeus.config import Config
from amadeus.web.app import create_app

TOKEN = "sekrit-token"


def _config(tmp_path, token=TOKEN) -> Config:
    config = Config()
    config.audio.enable_voice = False
    config.web.api_token = token
    config.memory.persist_path = str(tmp_path / "memory.json")
    config.memory.embeddings_path = str(tmp_path / "embeddings.json")
    config.memory.profile_path = str(tmp_path / "profile.json")
    config.memory.transcripts_path = str(tmp_path / "transcripts.json")
    config.memory.enable_long_term = False
    return config


def _client(app):
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://testserver")


@pytest.mark.asyncio
async def test_api_routes_require_bearer_token(tmp_path, monkeypatch):
    monkeypatch.setenv("KURISU_FRONTEND_DIR", str(tmp_path / "no-bundle"))
    app = create_app(_config(tmp_path))
    async with _client(app) as client:
        # No header / wrong scheme / wrong token → 401.
        assert (await client.get("/api/status")).status_code == 401
        assert (
            await client.get("/api/status", headers={"Authorization": TOKEN})
        ).status_code == 401
        assert (
            await client.get("/api/status", headers={"Authorization": "Bearer wrong"})
        ).status_code == 401
        assert (
            await client.post("/api/chat", json={"message": "hi"})
        ).status_code == 401

        # Correct token → normal responses.
        ok = await client.get(
            "/api/status", headers={"Authorization": f"Bearer {TOKEN}"}
        )
        assert ok.status_code == 200
        chat = await client.post(
            "/api/chat",
            json={"message": "hi"},
            headers={"Authorization": f"Bearer {TOKEN}"},
        )
        assert chat.status_code == 200

        # /healthz and the SPA/static surface stay open.
        assert (await client.get("/healthz")).status_code == 200
        assert (await client.get("/")).status_code == 200


@pytest.mark.asyncio
async def test_no_token_configured_keeps_api_open(tmp_path):
    app = create_app(_config(tmp_path, token=None))
    async with _client(app) as client:
        assert (await client.get("/api/status")).status_code == 200
        assert (
            await client.post("/api/chat", json={"message": "hi"})
        ).status_code == 200


def test_websocket_rejects_bad_token_with_policy_violation(tmp_path):
    app = create_app(_config(tmp_path))
    client = TestClient(app)

    for url in ("/api/chat/stream", "/api/chat/stream?token=wrong"):
        with client.websocket_connect(url) as ws:
            with pytest.raises(WebSocketDisconnect) as excinfo:
                ws.receive_json()
        assert excinfo.value.code == 1008


def test_websocket_accepts_query_token(tmp_path):
    config = _config(tmp_path)
    config.web.streaming_enabled = False  # deterministic first frame
    app = create_app(config)
    client = TestClient(app)

    with client.websocket_connect(f"/api/chat/stream?token={TOKEN}") as ws:
        frame = ws.receive_json()
    # Auth passed: we got the streaming-disabled error frame, not a 1008 close.
    assert frame["type"] == "error"
    assert frame["fallback_to_post"] is True


def test_api_token_env_fallback(monkeypatch):
    monkeypatch.setenv("AMADEUS_API_TOKEN", "from-env")
    config = Config.from_dict({})
    assert config.web.api_token == "from-env"

    # Explicit config value wins over the env var.
    config = Config.from_dict({"web": {"api_token": "from-config"}})
    assert config.web.api_token == "from-config"

    monkeypatch.delenv("AMADEUS_API_TOKEN")
    config = Config.from_dict({})
    assert config.web.api_token is None


def test_cors_origins_config(tmp_path):
    from starlette.middleware.cors import CORSMiddleware

    config = _config(tmp_path, token=None)
    config.web.cors_origins = ["https://kurisu.example.com"]
    app = create_app(config)

    cors = next(m for m in app.user_middleware if m.cls is CORSMiddleware)
    assert cors.kwargs["allow_origins"] == ["https://kurisu.example.com"]
