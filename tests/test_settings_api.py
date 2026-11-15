import sys
import types

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from amadeus.config import Config
from amadeus.web import settings_api
from amadeus.web.app import create_app


def _install_dummy_openai(monkeypatch):
    """SessionStore constructs the configured LLM eagerly, so any test that
    sets providers.llm = "openai" needs a stand-in for the optional `openai`
    package (not installed in this environment) rather than mocking at the
    provider-class level."""

    class DummyOpenAI:
        def __init__(self, *args, **kwargs):
            pass

    dummy_module = types.SimpleNamespace(OpenAI=DummyOpenAI)
    monkeypatch.setitem(sys.modules, "openai", dummy_module)


def _isolated_config(tmp_path) -> Config:
    config = Config()
    config.memory.persist_path = str(tmp_path / "memory.json")
    config.memory.embeddings_path = str(tmp_path / "embeddings.json")
    config.memory.profile_path = str(tmp_path / "profile.json")
    return config


def _client(tmp_path, monkeypatch, config=None):
    monkeypatch.setenv("AMADEUS_USER_CONFIG", str(tmp_path / "user_config.yaml"))
    app = create_app(config or _isolated_config(tmp_path))
    return TestClient(app)


def test_get_settings_reports_dummy_on_fresh_install(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    response = client.get("/api/settings")
    assert response.status_code == 200
    body = response.json()
    assert body["is_first_run"] is True
    assert body["providers"]["llm_is_dummy"] is True
    assert body["openai_api_key_set"] is False


def test_get_settings_reports_configured_provider(tmp_path, monkeypatch):
    _install_dummy_openai(monkeypatch)
    config = _isolated_config(tmp_path)
    config.providers.llm = "openai"
    config.openai_api_key = "sk-test"
    client = _client(tmp_path, monkeypatch, config)

    body = client.get("/api/settings").json()
    assert body["is_first_run"] is False
    assert body["providers"]["llm_is_dummy"] is False
    assert body["openai_api_key_set"] is True


def test_put_settings_persists_to_user_config_not_example(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    user_config_path = tmp_path / "user_config.yaml"
    assert not user_config_path.exists()

    response = client.put("/api/settings", json={"tsundere_level": 0.8})
    assert response.status_code == 200
    assert user_config_path.exists()

    saved = settings_api.load_raw(user_config_path)
    assert saved["persona"]["tsundere_level"] == 0.8


def test_put_settings_persona_dial_does_not_require_restart(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    response = client.put(
        "/api/settings", json={"tsundere_level": 0.9, "safe_mode": True}
    )
    body = response.json()
    assert body["needs_restart"] is False
    assert body["settings"]["persona"]["tsundere_level"] == 0.9
    assert body["settings"]["persona"]["safe_mode"] is True


def test_put_settings_voice_language_roundtrip(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    response = client.put("/api/settings", json={"voice_language": "english"})
    body = response.json()
    assert body["settings"]["voice_language"] == "english"

    saved = settings_api.load_raw(tmp_path / "user_config.yaml")
    assert saved["audio"]["voice_language"] == "english"

    # And back — plus reject junk values at the model layer.
    assert (
        client.put("/api/settings", json={"voice_language": "japanese"}).json()[
            "settings"
        ]["voice_language"]
        == "japanese"
    )
    assert (
        client.put("/api/settings", json={"voice_language": "klingon"}).status_code
        == 422
    )


def test_put_settings_provider_change_requires_restart(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    response = client.put("/api/settings", json={"llm_provider": "gemini"})
    body = response.json()
    assert body["needs_restart"] is True
    assert "provider" in body["restart_reason"].lower()
    assert body["settings"]["providers"]["llm"] == "gemini"


def test_put_settings_key_roundtrip_and_redaction(tmp_path, monkeypatch):
    _install_dummy_openai(monkeypatch)  # default providers.llm is "openai"
    client = _client(tmp_path, monkeypatch)
    response = client.put(
        "/api/settings", json={"openai_api_key": "sk-real-secret-value-0123456789"}
    )
    body = response.json()
    assert body["settings"]["openai_api_key_set"] is True
    # The raw key must never be echoed back to the client.
    assert "sk-real-secret-value-0123456789" not in response.text

    saved = settings_api.load_raw(settings_api.user_config_path())
    assert saved["openai_api_key"] == "sk-real-secret-value-0123456789"


def test_put_settings_ignores_blank_key(tmp_path, monkeypatch):
    """An empty-string key must not overwrite an already-configured one —
    guards against a UI bug or accidental blank submit wiping a real key."""
    _install_dummy_openai(monkeypatch)  # default providers.llm is "openai"
    config = _isolated_config(tmp_path)
    config.openai_api_key = "sk-existing"
    client = _client(tmp_path, monkeypatch, config)

    response = client.put("/api/settings", json={"openai_api_key": ""})
    body = response.json()
    assert body["settings"]["openai_api_key_set"] is True
    assert body["needs_restart"] is False


def test_put_settings_persists_across_requests(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    client.put("/api/settings", json={"tools_enabled": True})

    body = client.get("/api/settings").json()
    assert body["tools_enabled"] is True


def test_reload_providers_applies_to_new_sessions(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    app_state = client.app.state.session_store  # type: ignore[attr-defined]
    assert app_state.config.tools.enabled is False

    client.put("/api/settings", json={"tools_enabled": True})
    assert app_state.config.tools.enabled is True


def test_test_endpoint_dummy_provider_ok(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    response = client.post(
        "/api/settings/test", json={"kind": "llm", "provider": "dummy"}
    )
    assert response.status_code == 200
    assert response.json()["ok"] is True


def test_test_endpoint_reports_missing_key(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    response = client.post(
        "/api/settings/test", json={"kind": "llm", "provider": "openai"}
    )
    body = response.json()
    assert body["ok"] is False
    assert "key" in body["message"].lower()


def test_test_endpoint_invalid_kind_rejected(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    response = client.post(
        "/api/settings/test", json={"kind": "bogus", "provider": "openai"}
    )
    assert response.status_code == 422


def test_apply_update_is_pure_and_merges(tmp_path):
    raw = {"persona": {"tsundere_level": 0.3}, "providers": {"llm": "openai"}}
    update = settings_api.SettingsUpdate(scientific_formality=0.7)
    new_raw, needs_restart, reason = settings_api.apply_update(raw, update)

    assert new_raw["persona"]["tsundere_level"] == 0.3  # preserved
    assert new_raw["persona"]["scientific_formality"] == 0.7  # merged
    assert new_raw["providers"]["llm"] == "openai"  # untouched section preserved
    assert needs_restart is False
    assert reason is None
    assert raw["persona"] == {"tsundere_level": 0.3}  # original untouched


def test_user_config_path_respects_env_override(tmp_path, monkeypatch):
    override = tmp_path / "custom" / "config.yaml"
    monkeypatch.setenv("AMADEUS_USER_CONFIG", str(override))
    assert settings_api.user_config_path() == override


def test_write_raw_is_atomic_and_leaves_no_tmp_file(tmp_path):
    path = tmp_path / "nested" / "config.yaml"
    settings_api.write_raw(path, {"a": 1})
    assert path.exists()
    assert not path.with_suffix(".yaml.tmp").exists()
    assert settings_api.load_raw(path) == {"a": 1}


def test_get_settings_reports_feature_toggles(tmp_path, monkeypatch):
    config = _isolated_config(tmp_path)
    config.web_search.enabled = True
    config.tools.enabled = True
    client = _client(tmp_path, monkeypatch, config)

    payload = client.get("/api/settings").json()
    assert payload["web_search_enabled"] is True
    assert payload["tools_enabled"] is True
    assert payload["reminders_enabled"] is False
    assert payload["memory_enable_long_term"] is True


def test_put_settings_feature_toggles_persist_and_flag_restart(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)

    response = client.put(
        "/api/settings",
        json={
            "tools_enabled": True,
            "reminders_enabled": True,
            "memory_enable_long_term": False,
            "stt_provider": "faster_whisper",
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["needs_restart"] is True
    assert body["settings"]["tools_enabled"] is True
    assert body["settings"]["reminders_enabled"] is True
    assert body["settings"]["memory_enable_long_term"] is False
    assert body["settings"]["providers"]["stt"] == "faster_whisper"

    raw = settings_api.load_raw(settings_api.user_config_path())
    assert raw["tools"]["enabled"] is True
    assert raw["reminders"]["enabled"] is True
    assert raw["memory"]["enable_long_term"] is False
    assert raw["providers"]["stt"] == "faster_whisper"


def test_put_settings_rejects_garbage_api_key(tmp_path, monkeypatch):
    """A pasted chat fragment must not persist as a key (422, key kept)."""
    client = _client(tmp_path, monkeypatch)
    response = client.put(
        "/api/settings", json={"gemini_api_key": "What are you working on?"}
    )
    assert response.status_code == 422
    assert "look like an API key" in response.text


def test_put_settings_rejects_short_api_key(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    response = client.put("/api/settings", json={"gemini_api_key": "abc123"})
    assert response.status_code == 422


def test_put_settings_accepts_plausible_api_key(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    response = client.put(
        "/api/settings",
        json={"gemini_api_key": "AIzaSy" + "A" * 33},
    )
    assert response.status_code == 200
    assert response.json()["settings"]["gemini_api_key_set"] is True


def test_put_settings_whitespace_key_means_leave_unchanged(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    response = client.put("/api/settings", json={"gemini_api_key": "   "})
    assert response.status_code == 200
    assert response.json()["settings"]["gemini_api_key_set"] is False
