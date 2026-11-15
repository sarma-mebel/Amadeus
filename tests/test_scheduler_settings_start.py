"""Scheduler start gate vs settings reloads.

The lifespan gate (web/app.py) is evaluated exactly once at boot; enabling a
scheduler-driven feature later through the settings API used to leave the
scheduler silently not running until a full restart. These tests pin the
repair: the lifespan stashes the loop + scheduler handle on the SessionStore,
``scheduler_features_enabled`` mirrors the gate, and
``settings_api._maybe_start_scheduler`` starts a scheduler when a reload
newly needs one (or counts loudly when it can't).
"""

from __future__ import annotations

import datetime as dt  # noqa: F401 - parity with sibling scheduler tests

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from amadeus import telemetry
from amadeus.config import Config
from amadeus.web import settings_api
from amadeus.web.app import create_app
from amadeus.web.scheduler import (
    _SCHEDULER_FEATURE_SECTIONS,
    scheduler_features_enabled,
)


@pytest.fixture(autouse=True)
def _clean_telemetry():
    telemetry.reset()
    yield
    telemetry.reset()


def _all_off_config(tmp_path) -> Config:
    config = Config()
    for section in _SCHEDULER_FEATURE_SECTIONS:
        getattr(config, section).enabled = False
    config.memory.persist_path = str(tmp_path / "memory.json")
    config.memory.embeddings_path = str(tmp_path / "embeddings.json")
    config.memory.profile_path = str(tmp_path / "profile.json")
    config.memory.transcripts_path = str(tmp_path / "transcripts.json")
    return config


# ---------------------------------------------------------------------------
# The gate helper
# ---------------------------------------------------------------------------


def test_scheduler_features_enabled_truth_table(tmp_path):
    config = _all_off_config(tmp_path)
    assert scheduler_features_enabled(config) is False
    # Each gate section alone must satisfy the predicate — a section the
    # helper forgets would reintroduce the silent-never-runs bug for that
    # feature on the settings-reload path.
    for section in _SCHEDULER_FEATURE_SECTIONS:
        scoped = _all_off_config(tmp_path)
        getattr(scoped, section).enabled = True
        assert scheduler_features_enabled(scoped) is True, section


def test_gate_sections_exist_on_config(tmp_path):
    """A typo'd section name would read as disabled forever via getattr."""
    config = Config()
    for section in _SCHEDULER_FEATURE_SECTIONS:
        assert hasattr(config, section), section
        assert hasattr(getattr(config, section), "enabled"), section


# ---------------------------------------------------------------------------
# Lifespan stashes the handles; settings reload starts the scheduler
# ---------------------------------------------------------------------------


def test_lifespan_stashes_loop_and_none_scheduler_when_gate_off(tmp_path):
    config = _all_off_config(tmp_path)
    app = create_app(config)
    store = app.state.session_store
    with TestClient(app):
        assert getattr(store, "scheduler_loop", None) is not None
        assert store.scheduler is None


def test_settings_reload_starts_scheduler_when_newly_needed(tmp_path):
    config = _all_off_config(tmp_path)
    app = create_app(config)
    store = app.state.session_store
    with TestClient(app):
        assert store.scheduler is None
        enabled = _all_off_config(tmp_path)
        enabled.diary.enabled = True
        settings_api._maybe_start_scheduler(store, enabled)
        assert store.scheduler is not None
        assert telemetry.snapshot().get("scheduler.started_on_reload") == 1
    # Lifespan shutdown stops whichever scheduler ended up running —
    # including this settings-started one (no orphaned task).
    assert store.scheduler._task is None


def test_settings_reload_never_starts_when_gate_still_off(tmp_path):
    config = _all_off_config(tmp_path)
    app = create_app(config)
    store = app.state.session_store
    with TestClient(app):
        settings_api._maybe_start_scheduler(store, _all_off_config(tmp_path))
        assert store.scheduler is None
        assert telemetry.snapshot().get("scheduler.started_on_reload") is None


def test_settings_reload_keeps_existing_scheduler(tmp_path):
    """A boot-started scheduler is never replaced (its live `config`
    property already follows store.config, so no restart is needed)."""
    config = _all_off_config(tmp_path)
    config.diary.enabled = True
    app = create_app(config)
    store = app.state.session_store
    with TestClient(app):
        running = store.scheduler
        assert running is not None
        enabled = _all_off_config(tmp_path)
        enabled.daily_brief.enabled = True
        settings_api._maybe_start_scheduler(store, enabled)
        assert store.scheduler is running


def test_no_event_loop_counts_loudly_instead_of_silence(tmp_path):
    """Without a lifespan (no stored loop) the scheduler can't be started;
    the degrade-loudly counter must fire so Diagnostics shows the gap."""

    class BareStore:
        pass

    enabled = _all_off_config(tmp_path)
    enabled.diary.enabled = True
    settings_api._maybe_start_scheduler(BareStore(), enabled)
    assert telemetry.snapshot().get("scheduler.not_running_but_needed") == 1
