"""web.demo_mode read-only guard + per-IP turn rate limiting."""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")

import httpx

from amadeus.config import Config
from amadeus.web.app import create_app
from amadeus.web.ratelimit import TurnRateLimiter, client_key


def _config(tmp_path, **web_overrides) -> Config:
    config = Config()
    config.audio.enable_voice = False
    config.memory.persist_path = str(tmp_path / "memory.json")
    config.memory.embeddings_path = str(tmp_path / "embeddings.json")
    config.memory.profile_path = str(tmp_path / "profile.json")
    config.memory.transcripts_path = str(tmp_path / "transcripts.json")
    config.memory.enable_long_term = False
    for key, value in web_overrides.items():
        setattr(config.web, key, value)
    return config


def _client(app):
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://testserver")


@pytest.mark.asyncio
async def test_demo_mode_locks_settings_and_persona_mutations(tmp_path, monkeypatch):
    monkeypatch.setenv("KURISU_FRONTEND_DIR", str(tmp_path / "no-bundle"))
    app = create_app(_config(tmp_path, demo_mode=True))
    async with _client(app) as client:
        assert (
            await client.put("/api/settings", json={"tools_enabled": True})
        ).status_code == 403
        assert (
            await client.post(
                "/api/settings/test",
                json={
                    "kind": "llm",
                    "provider": "gemini",
                    "api_key": "x",
                    "model": None,
                },
            )
        ).status_code == 403
        assert (
            await client.post("/api/persona/versions/snapshot", json={})
        ).status_code == 403

        # Read surface stays open, and status reports the mode.
        assert (await client.get("/api/settings")).status_code == 200
        status = await client.get("/api/status")
        assert status.status_code == 200
        assert status.json()["demo_mode"] is True

        # Chat itself still works (dummy provider in tests).
        assert (
            await client.post("/api/chat", json={"message": "hi"})
        ).status_code == 200


@pytest.mark.asyncio
async def test_demo_mode_off_reports_false_and_allows_settings(tmp_path, monkeypatch):
    monkeypatch.setenv("KURISU_FRONTEND_DIR", str(tmp_path / "no-bundle"))
    monkeypatch.setenv("AMADEUS_USER_CONFIG", str(tmp_path / "user-config.yaml"))
    app = create_app(_config(tmp_path))
    async with _client(app) as client:
        status = await client.get("/api/status")
        assert status.json()["demo_mode"] is False
        assert (
            await client.put("/api/settings", json={"tools_enabled": False})
        ).status_code == 200


@pytest.mark.asyncio
async def test_rate_limit_caps_chat_turns_per_ip(tmp_path, monkeypatch):
    monkeypatch.setenv("KURISU_FRONTEND_DIR", str(tmp_path / "no-bundle"))
    app = create_app(_config(tmp_path, rate_limit_per_minute=2))
    async with _client(app) as client:
        assert (
            await client.post("/api/chat", json={"message": "1"})
        ).status_code == 200
        assert (
            await client.post("/api/chat", json={"message": "2"})
        ).status_code == 200
        third = await client.post("/api/chat", json={"message": "3"})
        assert third.status_code == 429
        # Non-turn endpoints are never limited.
        assert (await client.get("/api/status")).status_code == 200


@pytest.mark.asyncio
async def test_rate_limit_keys_on_forwarded_client_ip(tmp_path, monkeypatch):
    monkeypatch.setenv("KURISU_FRONTEND_DIR", str(tmp_path / "no-bundle"))
    app = create_app(_config(tmp_path, rate_limit_per_minute=1))
    async with _client(app) as client:
        first = await client.post(
            "/api/chat", json={"message": "1"}, headers={"Fly-Client-IP": "1.2.3.4"}
        )
        assert first.status_code == 200
        blocked = await client.post(
            "/api/chat", json={"message": "2"}, headers={"Fly-Client-IP": "1.2.3.4"}
        )
        assert blocked.status_code == 429
        # A different client IP has its own budget.
        other = await client.post(
            "/api/chat", json={"message": "3"}, headers={"Fly-Client-IP": "5.6.7.8"}
        )
        assert other.status_code == 200


@pytest.mark.asyncio
async def test_demo_mode_defaults_a_rate_limit(tmp_path, monkeypatch):
    monkeypatch.setenv("KURISU_FRONTEND_DIR", str(tmp_path / "no-bundle"))
    app = create_app(_config(tmp_path, demo_mode=True))
    limiter = app.state.turn_limiter
    assert limiter is not None
    assert limiter.per_minute == 10


def test_turn_rate_limiter_window_slides(monkeypatch):
    clock = {"now": 1000.0}
    monkeypatch.setattr("amadeus.web.ratelimit.time.monotonic", lambda: clock["now"])
    limiter = TurnRateLimiter(2)

    assert limiter.allow("ip")
    assert limiter.allow("ip")
    assert not limiter.allow("ip")

    clock["now"] += 61.0
    assert limiter.allow("ip")


def test_client_key_precedence():
    assert (
        client_key(
            {"fly-client-ip": "9.9.9.9", "x-forwarded-for": "1.1.1.1"}, "2.2.2.2"
        )
        == "9.9.9.9"
    )
    assert client_key({"x-forwarded-for": "1.1.1.1, 10.0.0.1"}, "2.2.2.2") == "1.1.1.1"
    assert client_key({}, "2.2.2.2") == "2.2.2.2"
    assert client_key({}, None) == "unknown"
