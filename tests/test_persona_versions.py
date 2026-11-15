"""Persona/voice versioning: store semantics (tolerant load, idempotent
snapshots, pinned-survives-prune), the identity payload's secret-exclusion
guarantee, the versions API incl. rollback-through-the-settings-path, and
the greeting's announce-once behavior."""

import datetime as dt
import json

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from amadeus.config import Config
from amadeus.persona_versions import (
    MAX_UNPINNED,
    PersonaVersionStore,
    describe_diff,
    identity_fingerprint,
    identity_payload,
)
from amadeus.providers.llm.base import ChatMessage
from amadeus.web.app import create_app


class RecordingProvider:
    name = "recording"

    def __init__(self, reply: str = "Hmph. Took you long enough.") -> None:
        self.reply = reply
        self.calls: list[list[ChatMessage]] = []

    def generate(self, messages, **kwargs):
        self.calls.append(list(messages))
        return self.reply


def _isolated_config(tmp_path) -> Config:
    config = Config()
    config.memory.persist_path = str(tmp_path / "memory.json")
    config.memory.embeddings_path = str(tmp_path / "embeddings.json")
    config.memory.profile_path = str(tmp_path / "profile.json")
    config.memory.persona_versions_path = str(tmp_path / "persona_versions.json")
    config.memory.enable_greeting = True
    return config


# ---------------------------------------------------------------------------
# Identity payload / fingerprint / diff
# ---------------------------------------------------------------------------


def test_identity_payload_excludes_secrets_and_brain():
    config = Config()
    config.openai_api_key = "sk-secret"
    config.gemini_api_key = "gm-secret"
    config.huggingface_api_token = "hf-secret"
    payload = identity_payload(config)
    dumped = json.dumps(payload)
    assert "secret" not in dumped
    assert "llm" not in payload["voice"]
    assert "stt" not in payload["voice"]
    assert "tts_cache_enabled" not in payload["voice"]
    assert payload["voice"]["tts"] == config.providers.tts
    assert payload["persona"]["tsundere_level"] == config.persona.tsundere_level


def test_fingerprint_stable_and_sensitive():
    config = Config()
    fp1 = identity_fingerprint(identity_payload(config))
    fp2 = identity_fingerprint(identity_payload(config))
    assert fp1 == fp2
    config.persona.tsundere_level = 0.9
    assert identity_fingerprint(identity_payload(config)) != fp1
    # Brain/model changes are NOT identity changes.
    config2 = Config()
    config2.llm_model = "some-other-model"
    assert identity_fingerprint(identity_payload(config2)) == fp1


def test_describe_diff_humanizes_known_dials():
    config_a, config_b = Config(), Config()
    config_b.persona.tsundere_level = 0.9
    config_b.providers.tts = "qwen3_tts"
    note = describe_diff(identity_payload(config_a), identity_payload(config_b))
    assert "tsundere level: 0.5 -> 0.9" in note
    assert "voice" in note


# ---------------------------------------------------------------------------
# Store semantics
# ---------------------------------------------------------------------------


def test_store_tolerant_load(tmp_path):
    path = tmp_path / "versions.json"
    path.write_text("{not json", encoding="utf-8")
    store = PersonaVersionStore(path)
    assert store.versions == []
    path.write_text(json.dumps({"versions": "nope"}), encoding="utf-8")
    assert PersonaVersionStore(path).versions == []


def test_store_snapshot_idempotent_and_roundtrips(tmp_path):
    path = tmp_path / "versions.json"
    store = PersonaVersionStore(path)
    payload = identity_payload(Config())
    v1 = store.snapshot(payload)
    v2 = store.snapshot(payload)
    assert v1.id == v2.id
    assert len(store.versions) == 1

    reloaded = PersonaVersionStore(path)
    assert len(reloaded.versions) == 1
    assert reloaded.versions[0].fingerprint == v1.fingerprint
    assert reloaded.versions[0].payload == payload


def test_store_prune_keeps_pinned(tmp_path):
    store = PersonaVersionStore(tmp_path / "versions.json")
    config = Config()
    config.persona.tsundere_level = 0.0
    pinned = store.snapshot(identity_payload(config), name="keeper", pinned=True)
    for i in range(MAX_UNPINNED + 5):
        config.persona.tsundere_level = round(0.01 * (i + 1), 3)
        store.snapshot(identity_payload(config))
    unpinned = [v for v in store.versions if not v.pinned]
    assert len(unpinned) == MAX_UNPINNED
    assert any(v.id == pinned.id for v in store.versions)


def test_store_delete_refuses_pinned(tmp_path):
    store = PersonaVersionStore(tmp_path / "versions.json")
    version = store.snapshot(identity_payload(Config()), pinned=True)
    assert store.delete(version.id) == "pinned"
    store.set_pinned(version.id, False)
    assert store.delete(version.id) == "deleted"
    assert store.delete(version.id) == "missing"


def test_pending_change_note_announce_cycle(tmp_path):
    store = PersonaVersionStore(tmp_path / "versions.json")
    config = Config()
    # First run: nothing announced yet -> no phantom "change from nothing".
    assert store.pending_change_note(config) is None
    store.mark_announced(config)
    assert store.pending_change_note(config) is None
    # A real change is reported until re-announced.
    store.snapshot(identity_payload(config))
    config.persona.tsundere_level = 0.9
    note = store.pending_change_note(config)
    assert note and "tsundere level" in note
    store.mark_announced(config)
    assert store.pending_change_note(config) is None


def test_mark_announced_reloads_before_saving(tmp_path):
    """Reload-before-mutate: mark_announced saving its stale in-memory view
    used to clobber snapshots another process (web + CLI share the file)
    appended after this instance loaded."""
    path = tmp_path / "versions.json"
    config = Config()
    writer = PersonaVersionStore(path)
    writer.snapshot(identity_payload(config), name="base")

    stale = PersonaVersionStore(path)  # loads with only "base" present

    changed = Config()
    changed.persona.tsundere_level = 0.9
    writer.snapshot(identity_payload(changed), name="second")

    stale.mark_announced(config)  # must not erase "second"

    reloaded = PersonaVersionStore(path)
    assert {v.name for v in reloaded.versions} == {"base", "second"}
    assert reloaded.last_announced_fingerprint == identity_fingerprint(
        identity_payload(config)
    )


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------


def _client(tmp_path, monkeypatch):
    config = _isolated_config(tmp_path)
    monkeypatch.setenv("AMADEUS_USER_CONFIG", str(tmp_path / "user_config.yaml"))
    app = create_app(config)
    return TestClient(app), app


def test_versions_api_snapshot_pin_delete(tmp_path, monkeypatch):
    client, app = _client(tmp_path, monkeypatch)

    created = client.post("/api/persona/versions", json={"name": "lab defaults"})
    assert created.status_code == 200
    info = created.json()
    assert info["name"] == "lab defaults"
    assert info["pinned"] is True  # named saves are pinned
    assert info["is_current"] is True

    listing = client.get("/api/persona/versions").json()
    assert listing["current_fingerprint"] == info["fingerprint"]
    assert [v["id"] for v in listing["versions"]] == [info["id"]]

    assert (
        client.delete(f"/api/persona/versions/{info['id']}").status_code == 409
    )  # pinned
    client.put(f"/api/persona/versions/{info['id']}/pin", json={"pinned": False})
    assert client.delete(f"/api/persona/versions/{info['id']}").status_code == 200
    assert client.delete("/api/persona/versions/nope").status_code == 404


def test_settings_change_autosnapshots_previous_identity(tmp_path, monkeypatch):
    client, app = _client(tmp_path, monkeypatch)
    store = app.state.session_store
    before_fp = identity_fingerprint(identity_payload(store.config))

    response = client.put("/api/settings", json={"tsundere_level": 0.9})
    assert response.status_code == 200
    assert store.config.persona.tsundere_level == 0.9

    listing = client.get("/api/persona/versions").json()
    fingerprints = [v["fingerprint"] for v in listing["versions"]]
    assert before_fp in fingerprints  # the PRE-change identity is restorable
    assert listing["current_fingerprint"] != before_fp


def test_rollback_restores_yaml_and_hot_applies(tmp_path, monkeypatch):
    client, app = _client(tmp_path, monkeypatch)
    store = app.state.session_store
    original_fp = identity_fingerprint(identity_payload(store.config))

    client.put("/api/settings", json={"tsundere_level": 0.9})
    listing = client.get("/api/persona/versions").json()
    original = next(v for v in listing["versions"] if v["fingerprint"] == original_fp)

    response = client.post(f"/api/persona/versions/{original['id']}/rollback")
    assert response.status_code == 200
    data = response.json()
    assert data["needs_restart"] is False  # same TTS provider
    assert store.config.persona.tsundere_level == 0.5

    import yaml

    written = yaml.safe_load((tmp_path / "user_config.yaml").read_text())
    assert written["persona"]["tsundere_level"] == 0.5


# ---------------------------------------------------------------------------
# Greeting announcement
# ---------------------------------------------------------------------------


def test_greeting_announces_identity_change_once(tmp_path, monkeypatch):
    config = _isolated_config(tmp_path)
    app = create_app(config)
    store = app.state.session_store
    recorder = RecordingProvider()
    store.llm = recorder
    _, session = store.get(None, "user-1")
    session.llm = recorder
    session.summary_provider = recorder
    # A persona/voice-change announcement presupposes an established
    # relationship, not a first meeting — pin session_count so it isn't
    # read as one.
    session.memory.profile.session_count = 5

    # Baseline: first greeting records the fingerprint, no announcement.
    version_store = session._persona_version_store()
    assert session.open_session_greeting() is not None
    baseline_prompt = recorder.calls[-1][1].content
    assert "settings/voice were changed" not in baseline_prompt

    # The user changes a dial out-of-band (as the settings API would).
    version_store.snapshot(identity_payload(session.config))
    session.config.persona.tsundere_level = 0.95

    # Force a fresh-session greeting (short_term empty already? greeting adds
    # a message) — clear short-term so should_greet fires again.
    session.memory.short_term.clear()
    session._persona_versions = None  # re-read the store like a new process
    assert session.open_session_greeting() is not None
    changed_prompt = recorder.calls[-1][1].content
    assert "settings/voice were changed" in changed_prompt
    assert "tsundere level" in changed_prompt

    # Announced once: the next fresh greeting must not repeat it.
    session.memory.short_term.clear()
    assert session.open_session_greeting() is not None
    third_prompt = recorder.calls[-1][1].content
    assert "settings/voice were changed" not in third_prompt


def test_first_meeting_greeting_records_baseline_then_announces_at_session_2(
    tmp_path,
):
    """G6 regression: the OLD code set version_store = None on the
    first-meeting branch, which skipped mark_announced entirely — so the
    baseline persona fingerprint was never recorded, and the very first
    settings change afterward could never be announced (nothing to diff
    against). The fix keeps version_store alive through the first-meeting
    branch (only the change-note FACT is suppressed, since there's nothing
    to announce on a genuine first meeting) so mark_announced still runs
    and captures the baseline fingerprint immediately."""
    config = _isolated_config(tmp_path)
    app = create_app(config)
    store = app.state.session_store
    recorder = RecordingProvider()
    store.llm = recorder
    _, session = store.get(None, "first-meeting-baseline-user")
    session.llm = recorder
    session.summary_provider = recorder

    # A genuinely fresh profile/session: session_count<=1, no facts, no
    # threads, empty short_term, first_met freshly stamped to today by the
    # session-boundary save in MemoryManager.__init__.
    assert session.memory.profile.session_count <= 1
    assert not session.memory.profile.facts
    assert not session.memory.profile.open_threads
    assert not session.memory.short_term

    version_store = session._persona_version_store()
    assert version_store.last_announced_fingerprint is None

    greeting = session.open_session_greeting()
    assert greeting is not None
    first_system = recorder.calls[-1][0].content
    first_user = recorder.calls[-1][1].content
    assert "FIRST conversation" in first_system
    # Nothing to announce yet on a first meeting.
    assert "settings/voice were changed" not in first_user
    assert "settings/voice were changed" not in first_system

    # The baseline fingerprint MUST now be recorded (this is the crux of the
    # bug: previously version_store was None here, so mark_announced never
    # ran and last_announced_fingerprint stayed None forever).
    version_store_reloaded = session._persona_version_store()
    assert version_store_reloaded.last_announced_fingerprint is not None
    baseline_fp = version_store_reloaded.last_announced_fingerprint
    assert baseline_fp == identity_fingerprint(identity_payload(session.config))

    # Advance to a normal (non-first-meeting) second session: bump
    # session_count and age first_met past 24h so the new predicate reads
    # this as a returning user, then change a persona dial out-of-band (as
    # the settings API would — snapshotting the pre-change identity first,
    # mirroring test_greeting_announces_identity_change_once above).
    session.memory.profile.session_count = 2
    session.memory.profile.first_met = (
        dt.datetime.utcnow() - dt.timedelta(days=2)
    ).isoformat()
    version_store_reloaded.snapshot(identity_payload(session.config))
    session.config.persona.tsundere_level = 0.95
    session.memory.short_term.clear()
    session._persona_versions = None  # re-read the store like a new process

    greeting_2 = session.open_session_greeting()
    assert greeting_2 is not None
    second_prompt = recorder.calls[-1][1].content
    assert "settings/voice were changed" in second_prompt
    assert "tsundere level" in second_prompt
