"""Web-side reminder announcements: engine lines + /api/reminders/due."""

import datetime as dt

import pytest

pytest.importorskip("fastapi")

import httpx

from amadeus.config import Config
from amadeus.engine import ChatEngine
from amadeus.providers.llm.base import DummyProvider
from amadeus.reminders import ReminderStore
from amadeus.web.app import create_app


def _client(app):
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://testserver")


def _config(tmp_path) -> Config:
    config = Config()
    config.memory.persist_path = str(tmp_path / "memory.json")
    config.memory.embeddings_path = str(tmp_path / "embeddings.json")
    config.memory.profile_path = str(tmp_path / "profile.json")
    config.reminders.enabled = True
    config.reminders.store_path = str(tmp_path / "reminders.json")
    return config


def _seed_due_reminder(config: Config, text: str, minutes_ago: float = 0.5):
    store = ReminderStore(config.reminders.store_path)
    store.add(text, dt.datetime.now() - dt.timedelta(minutes=minutes_ago))


def test_engine_due_reminder_lines_phrasing(tmp_path):
    config = _config(tmp_path)
    _seed_due_reminder(config, "check the scanner booking", minutes_ago=0.5)
    _seed_due_reminder(config, "email the coauthors", minutes_ago=30)

    engine = ChatEngine(config, DummyProvider())
    lines = engine.due_reminder_lines()
    assert len(lines) == 2
    fresh = next(line for line in lines if "scanner booking" in line)
    stale = next(line for line in lines if "coauthors" in line)
    assert fresh.startswith("Reminder:")
    assert stale.startswith("While you were away:")

    # pop_due marked them announced: a second poll is silent.
    assert engine.due_reminder_lines() == []


def test_engine_reminders_disabled_is_noop(tmp_path):
    config = _config(tmp_path)
    config.reminders.enabled = False
    _seed_due_reminder(config, "should never announce")
    engine = ChatEngine(config, DummyProvider())
    assert engine.due_reminder_lines() == []


async def test_due_reminders_endpoint_announces_once(tmp_path):
    config = _config(tmp_path)
    _seed_due_reminder(config, "stand up and stretch")
    app = create_app(config)

    async with _client(app) as client:
        first = (await client.post("/api/reminders/due", json={})).json()
        assert first["reminder"] is not None
        assert "stand up and stretch" in first["reminder"]
        assert "emotion" in first

        # Same session polls again: nothing new to announce.
        again = (
            await client.post(
                "/api/reminders/due", json={"session_id": first["session_id"]}
            )
        ).json()
        assert again["reminder"] is None


async def test_status_reports_tools_block(tmp_path):
    config = _config(tmp_path)
    config.tools.enabled = True
    config.literature.enabled = True
    app = create_app(config)
    async with _client(app) as client:
        payload = (await client.get("/api/status")).json()
    tools = payload["tools"]
    assert tools["enabled"] is True
    for name in (
        "calculator",
        "code_interpreter",
        "pubmed_search",
        "arxiv_search",
        "reminders",
    ):
        assert name in tools["registered"]
    # web_search and desktop_files were not enabled: neither registered nor
    # reported as unavailable.
    assert "web_search" not in tools["registered"]
    assert "desktop_files" not in tools["registered"]
    assert tools["unavailable"] == []
