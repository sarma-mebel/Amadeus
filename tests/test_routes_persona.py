"""amadeus/web/routes_persona.py — the persona-version API surface itself.

test_persona_versions.py covers the store/greeting semantics; this file
exercises each route (list/snapshot/pin/delete/rollback) through the app
with dummy providers, plus the unknown-id and versioning-disabled paths.
"""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from amadeus.config import Config
from amadeus.web.app import create_app


def _config(tmp_path) -> Config:
    config = Config()
    config.audio.enable_voice = False
    config.memory.persist_path = str(tmp_path / "memory.json")
    config.memory.embeddings_path = str(tmp_path / "embeddings.json")
    config.memory.profile_path = str(tmp_path / "profile.json")
    config.memory.transcripts_path = str(tmp_path / "transcripts.json")
    config.memory.persona_versions_path = str(tmp_path / "persona_versions.json")
    config.memory.enable_long_term = False
    return config


def _client(tmp_path, monkeypatch) -> TestClient:
    monkeypatch.setenv("AMADEUS_USER_CONFIG", str(tmp_path / "user_config.yaml"))
    return TestClient(create_app(_config(tmp_path)))


def test_list_starts_empty_with_current_fingerprint(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    body = client.get("/api/persona/versions").json()
    assert body["versions"] == []
    assert body["current_fingerprint"]


def test_snapshot_unnamed_is_unpinned_and_current(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    created = client.post("/api/persona/versions", json={})
    assert created.status_code == 200
    info = created.json()
    assert info["pinned"] is False  # only NAMED saves auto-pin
    assert info["is_current"] is True
    assert info["diff_vs_current"] is None

    listing = client.get("/api/persona/versions").json()
    assert [v["id"] for v in listing["versions"]] == [info["id"]]
    assert listing["current_fingerprint"] == info["fingerprint"]


def test_snapshot_named_pins_and_pin_route_toggles(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    info = client.post("/api/persona/versions", json={"name": "lab look"}).json()
    assert info["name"] == "lab look"
    assert info["pinned"] is True

    unpinned = client.put(
        f"/api/persona/versions/{info['id']}/pin", json={"pinned": False}
    )
    assert unpinned.status_code == 200
    assert unpinned.json()["pinned"] is False

    repinned = client.put(
        f"/api/persona/versions/{info['id']}/pin", json={"pinned": True}
    )
    assert repinned.json()["pinned"] is True


def test_delete_happy_path_and_pinned_conflict(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    info = client.post("/api/persona/versions", json={"name": "keeper"}).json()

    assert client.delete(f"/api/persona/versions/{info['id']}").status_code == 409

    client.put(f"/api/persona/versions/{info['id']}/pin", json={"pinned": False})
    deleted = client.delete(f"/api/persona/versions/{info['id']}")
    assert deleted.status_code == 200
    assert deleted.json() == {"deleted": info["id"]}
    assert client.get("/api/persona/versions").json()["versions"] == []


def test_unknown_version_id_is_404_on_every_id_route(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    assert (
        client.put("/api/persona/versions/nope/pin", json={"pinned": True}).status_code
        == 404
    )
    assert client.delete("/api/persona/versions/nope").status_code == 404
    assert client.post("/api/persona/versions/nope/rollback").status_code == 404


def test_rollback_restores_identity_and_snapshots_pre_rollback(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    baseline = client.post("/api/persona/versions", json={}).json()

    # Change a persona dial through the settings path (as the UI would).
    changed = client.put("/api/settings", json={"tsundere_level": 0.9})
    assert changed.status_code == 200

    listing = client.get("/api/persona/versions").json()
    assert listing["current_fingerprint"] != baseline["fingerprint"]

    rolled = client.post(f"/api/persona/versions/{baseline['id']}/rollback")
    assert rolled.status_code == 200
    body = rolled.json()
    assert body["needs_restart"] is False  # TTS provider unchanged
    assert body["settings"]["persona"]["tsundere_level"] == pytest.approx(0.5)

    # Post-rollback the baseline is current again AND the pre-rollback
    # identity (0.9) was itself snapshotted, so rollback is undoable.
    listing = client.get("/api/persona/versions").json()
    assert listing["current_fingerprint"] == baseline["fingerprint"]
    assert len(listing["versions"]) >= 2
    non_current = [v for v in listing["versions"] if not v["is_current"]]
    assert non_current  # the 0.9 identity survives as a version


def test_disabled_versioning_returns_503(tmp_path, monkeypatch):
    monkeypatch.setenv("AMADEUS_USER_CONFIG", str(tmp_path / "user_config.yaml"))
    config = _config(tmp_path)
    config.memory.persona_versions_path = None
    client = TestClient(create_app(config))
    assert client.get("/api/persona/versions").status_code == 503
    assert client.post("/api/persona/versions", json={}).status_code == 503
