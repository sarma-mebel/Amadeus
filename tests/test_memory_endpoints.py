"""GET /api/memory/export and POST /api/memory/import."""

from __future__ import annotations

import json

import pytest

pytest.importorskip("fastapi")

import httpx

from amadeus.config import Config
from amadeus.memory import backup
from amadeus.web.app import create_app


def _config(tmp_path, **web_overrides) -> Config:
    config = Config()
    config.audio.enable_voice = False
    config.memory.persist_path = str(tmp_path / "memory_store.json")
    config.memory.embeddings_path = str(tmp_path / "memory_embeddings.json")
    config.memory.profile_path = str(tmp_path / "memory_profile.json")
    config.memory.transcripts_path = str(tmp_path / "memory_transcripts.json")
    config.memory.enable_long_term = False
    for key, value in web_overrides.items():
        setattr(config.web, key, value)
    return config


def _client(app):
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://testserver")


def _seed(config, user_id=None):
    for kind, path in backup.memory_paths(config, user_id).items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"kind": kind}), encoding="utf-8")


@pytest.mark.asyncio
async def test_export_404_when_no_memory_yet(tmp_path, monkeypatch):
    monkeypatch.setenv("KURISU_FRONTEND_DIR", str(tmp_path / "no-bundle"))
    app = create_app(_config(tmp_path))
    async with _client(app) as client:
        assert (await client.get("/api/memory/export?user_id=ghost")).status_code == 404


@pytest.mark.asyncio
async def test_export_then_import_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setenv("KURISU_FRONTEND_DIR", str(tmp_path / "no-bundle"))
    config = _config(tmp_path)
    _seed(config, "alice")
    app = create_app(config)
    async with _client(app) as client:
        exported = await client.get("/api/memory/export?user_id=alice")
        assert exported.status_code == 200
        assert exported.headers["content-type"] == "application/zip"
        assert "amadeus-memory-" in exported.headers["content-disposition"]

        imported = await client.post(
            "/api/memory/import?user_id=bob",
            files={"file": ("memory.zip", exported.content, "application/zip")},
        )
        assert imported.status_code == 200
        body = imported.json()
        assert body["status"] == "ok"
        assert "profile" in body["restored"]
        bob_profile = backup.memory_paths(config, "bob")["profile"]
        assert bob_profile.is_file()


@pytest.mark.asyncio
async def test_import_rejects_bad_archive(tmp_path, monkeypatch):
    monkeypatch.setenv("KURISU_FRONTEND_DIR", str(tmp_path / "no-bundle"))
    app = create_app(_config(tmp_path))
    async with _client(app) as client:
        response = await client.post(
            "/api/memory/import",
            files={"file": ("memory.zip", b"garbage", "application/zip")},
        )
        assert response.status_code == 400


@pytest.mark.asyncio
async def test_demo_mode_blocks_import_but_not_export(tmp_path, monkeypatch):
    monkeypatch.setenv("KURISU_FRONTEND_DIR", str(tmp_path / "no-bundle"))
    config = _config(tmp_path, demo_mode=True)
    _seed(config)
    app = create_app(config)
    async with _client(app) as client:
        assert (
            await client.post(
                "/api/memory/import",
                files={"file": ("memory.zip", b"x", "application/zip")},
            )
        ).status_code == 403
        assert (await client.get("/api/memory/export")).status_code == 200


@pytest.mark.asyncio
async def test_import_drops_live_sessions_so_state_reloads(tmp_path, monkeypatch):
    monkeypatch.setenv("KURISU_FRONTEND_DIR", str(tmp_path / "no-bundle"))
    config = _config(tmp_path)
    _seed(config, "alice")
    app = create_app(config)
    async with _client(app) as client:
        # Materialize a live session for alice.
        chat = await client.post(
            "/api/chat", json={"message": "hi", "user_id": "alice"}
        )
        assert chat.status_code == 200
        store = app.state.session_store
        assert any(s.user_id == "alice" for s in store.sessions.values())

        exported = await client.get("/api/memory/export?user_id=alice")
        imported = await client.post(
            "/api/memory/import?user_id=alice",
            files={"file": ("memory.zip", exported.content, "application/zip")},
        )
        assert imported.status_code == 200
        assert imported.json()["sessions_reloaded"] >= 1
        assert not any(s.user_id == "alice" for s in store.sessions.values())
