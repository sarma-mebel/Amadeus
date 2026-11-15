"""Character switching API: list, switch to a pack, restore the built-in."""

import json
import types
import wave

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

import yaml

from amadeus.config import Config
from amadeus.web import routes_characters
from amadeus.web.app import create_app
from amadeus.web.sessions import SessionStore


def _isolated_config(tmp_path) -> Config:
    config = Config()
    config.memory.persist_path = str(tmp_path / "memory.json")
    config.memory.embeddings_path = str(tmp_path / "embeddings.json")
    config.memory.profile_path = str(tmp_path / "profile.json")
    return config


def _write_wav(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b"\x00\x00" * 160)


def _make_pack(root, pack_id="testchar", with_voice=True):
    pack_dir = root / "characters" / "packs" / pack_id
    pack_dir.mkdir(parents=True, exist_ok=True)
    display = pack_id.capitalize()
    manifest = {
        "name": display,
        "full_name": f"{display} Character",
        "prompt_file": "prompt.txt",
        "model_id": pack_id,
    }
    if with_voice:
        manifest["voice"] = {
            "ref_audio": "voice_ref.wav",
            "ref_text": "テストです。",
            "language": "Japanese",
        }
        _write_wav(pack_dir / "voice_ref.wav")
    (pack_dir / "character.yaml").write_text(
        yaml.safe_dump(manifest, allow_unicode=True), encoding="utf-8"
    )
    (pack_dir / "prompt.txt").write_text("You are Testchar.", encoding="utf-8")
    return pack_dir


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("AMADEUS_USER_CONFIG", str(tmp_path / "user_config.yaml"))
    # Confine both the pack registry and the installer's destination roots
    # to the temp dir so tests never touch the real repo layout.
    monkeypatch.setattr(routes_characters, "_repo_root", lambda: tmp_path)
    import amadeus.characters as characters_module

    monkeypatch.setattr(characters_module, "_repo_root", lambda: tmp_path)
    _make_pack(tmp_path)
    app = create_app(_isolated_config(tmp_path))
    return TestClient(app)


def test_list_reports_builtin_and_packs(client):
    body = client.get("/api/characters").json()
    ids = [c["id"] for c in body["characters"]]
    assert ids[0] == "kurisu"
    assert "testchar" in ids
    assert body["active"] == "kurisu"
    entry = next(c for c in body["characters"] if c["id"] == "testchar")
    assert entry["has_voice"] is True
    assert entry["model_id"] == "testchar"


def test_switch_to_pack_and_back(client, tmp_path):
    response = client.post("/api/characters/switch", json={"id": "testchar"})
    assert response.status_code == 200
    body = response.json()
    assert body["active"] == "testchar"
    assert body["character"]["name"] == "Testchar"
    assert body["character"]["model_id"] == "testchar"

    # The user config carries the pack identity and the built-in snapshot
    # exists for the way back.
    raw = yaml.safe_load((tmp_path / "user_config.yaml").read_text(encoding="utf-8"))
    assert raw["character"]["name"] == "Testchar"
    assert raw["providers"]["qwen3_tts_ref_text"] == "テストです。"
    snapshot = tmp_path / "characters" / "installed" / "_builtin_identity.json"
    assert snapshot.exists()

    assert client.get("/api/characters").json()["active"] == "testchar"

    response = client.post("/api/characters/switch", json={"id": "kurisu"})
    assert response.status_code == 200
    raw = yaml.safe_load((tmp_path / "user_config.yaml").read_text(encoding="utf-8"))
    assert "character" not in raw
    assert "qwen3_tts_ref_text" not in (raw.get("providers") or {})
    assert client.get("/api/characters").json()["active"] == "kurisu"


def test_switch_between_packs_resets_identity_first(client, tmp_path):
    _make_pack(tmp_path, pack_id="secondchar", with_voice=False)
    client.post("/api/characters/switch", json={"id": "testchar"})
    response = client.post("/api/characters/switch", json={"id": "secondchar"})
    assert response.status_code == 200
    raw = yaml.safe_load((tmp_path / "user_config.yaml").read_text(encoding="utf-8"))
    assert raw["character"]["name"] == "Secondchar"
    # secondchar has no voice: testchar's ref fields must not leak through.
    assert "qwen3_tts_ref_text" not in (raw.get("providers") or {})


def test_switch_rejects_bad_ids(client):
    assert (
        client.post("/api/characters/switch", json={"id": "../evil"}).status_code == 400
    )
    assert (
        client.post("/api/characters/switch", json={"id": "missing_pack"}).status_code
        == 404
    )


def test_status_reflects_switch(client):
    client.post("/api/characters/switch", json={"id": "testchar"})
    character = client.get("/api/status").json()["character"]
    assert character["name"] == "Testchar"
    assert character["custom"] is True
    assert character["model_id"] == "testchar"


class _RebindableTTS:
    cache_signature = "fake|sig"

    def __init__(self):
        self.rebound = None

    def rebind_voice(self, ref_audio, ref_text, language=None):
        self.rebound = (ref_audio, ref_text, language)


def _stub_store(tts):
    return types.SimpleNamespace(
        tts=tts,
        _tts_fingerprint=SessionStore._tts_fingerprint,
        _VOICE_REBIND_KEYS=SessionStore._VOICE_REBIND_KEYS,
    )


def test_voice_rebind_eligibility():
    old = Config()
    new = Config()
    new.providers.qwen3_tts_ref_audio = "other/ref.wav"
    new.providers.qwen3_tts_ref_text = "違うテキスト"
    stub = _stub_store(_RebindableTTS())
    assert SessionStore._voice_rebind_eligible(stub, old, new) is True

    # Any non-ref TTS change disqualifies the in-place path.
    new_engine = Config()
    new_engine.providers.qwen3_tts_ref_audio = "other/ref.wav"
    new_engine.providers.tts = "melotts"
    assert SessionStore._voice_rebind_eligible(stub, old, new_engine) is False

    # Providers without rebind_voice never take the fast path.
    plain = _stub_store(types.SimpleNamespace())
    assert SessionStore._voice_rebind_eligible(plain, old, new) is False
