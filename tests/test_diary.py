"""Kurisu's diary: store semantics, scheduler due-predicate/idempotence, and
the read-only /api/diary endpoint. Structurally cloned from
tests/test_daily_brief.py."""

from __future__ import annotations

import datetime as dt

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from amadeus.config import Config
from amadeus.diary import (
    DiaryEntry,
    DiaryStore,
    collect_diary_seeds,
    generate_diary_entry,
)
from amadeus.memory.profile import UserProfile
from amadeus.providers.llm.base import ChatMessage
from amadeus.web.app import create_app
from amadeus.web.scheduler import BackgroundScheduler


class RecordingProvider:
    name = "recording"

    def __init__(
        self,
        reply: str = "Read a mediocre preprint on quantum teleportation today. Grumbled about it.",
    ) -> None:
        self.reply = reply
        self.calls: list[list[ChatMessage]] = []
        self.fail = False

    def generate(self, messages, **kwargs):
        self.calls.append(list(messages))
        if self.fail:
            raise RuntimeError("provider down")
        return self.reply


def _entry(date: dt.date, text: str = "diary text") -> DiaryEntry:
    return DiaryEntry(
        date=date.isoformat(), text=text, generated_at=f"{date.isoformat()}T07:00:00"
    )


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------


def test_store_roundtrip_and_per_user_keying(tmp_path):
    store = DiaryStore(str(tmp_path / "diary.json"))
    today = dt.date(2026, 7, 10)
    assert store.put(None, _entry(today, "default user")) is True
    assert store.put("alice", _entry(today, "alice's diary")) is True

    reloaded = DiaryStore(str(tmp_path / "diary.json"))
    assert reloaded.get(None).text == "default user"
    assert reloaded.get("alice").text == "alice's diary"


def test_store_put_refuses_same_date(tmp_path):
    store = DiaryStore(str(tmp_path / "diary.json"))
    today = dt.date(2026, 7, 10)
    assert store.put(None, _entry(today)) is True
    assert store.put(None, _entry(today, "duplicate")) is False
    assert store.get(None).text == "diary text"
    # Next day's entry overwrites.
    assert store.put(None, _entry(today + dt.timedelta(days=1), "tomorrow")) is True
    assert store.get(None).text == "tomorrow"


def test_store_peek_and_consume(tmp_path):
    store = DiaryStore(str(tmp_path / "diary.json"))
    today = dt.date(2026, 7, 10)
    store.put(None, _entry(today))
    assert store.peek_unconsumed(None, today).text == "diary text"
    # A stale (yesterday's) entry is never delivered late.
    assert store.peek_unconsumed(None, today + dt.timedelta(days=1)) is None
    store.mark_consumed(None, today.isoformat())
    assert store.peek_unconsumed(None, today) is None
    assert store.get(None).consumed is True


def test_store_corrupt_file_starts_clean(tmp_path):
    path = tmp_path / "diary.json"
    path.write_text("{corrupt", encoding="utf-8")
    store = DiaryStore(str(path))
    assert store.get(None) is None
    assert store.put(None, _entry(dt.date(2026, 7, 10))) is True


# ---------------------------------------------------------------------------
# Seeds / generation
# ---------------------------------------------------------------------------


def test_collect_diary_seeds_without_profile():
    now = dt.datetime(2026, 7, 10, 7, 0)
    seeds = collect_diary_seeds(profile=None, now=now)
    assert seeds == ["Now: Friday 07:00."]


def test_collect_diary_seeds_with_profile():
    now = dt.datetime(2026, 7, 10, 7, 0)
    profile = UserProfile(
        facts={"job": "researcher", "hobby": "climbing"},
        open_threads=["the ICU foundation model deadline"],
        last_episode="we argued about preprint servers",
    )
    seeds = collect_diary_seeds(profile=profile, now=now)
    joined = "\n".join(seeds)
    assert "Now: Friday 07:00." in joined
    assert "researcher" in joined or "climbing" in joined
    assert "ICU foundation model" in joined
    assert "we argued about preprint servers" in joined


def test_generate_diary_entry_returns_none_on_provider_failure():
    provider = RecordingProvider()
    provider.fail = True
    assert generate_diary_entry(provider=provider, facts=["Now: Friday 07:00."]) is None


def test_generate_diary_entry_with_timestamp_only_facts():
    """Unlike the daily brief, an empty profile still produces an entry —
    Kurisu has a day even when she knows nothing new about the user."""
    provider = RecordingProvider()
    text = generate_diary_entry(provider=provider, facts=["Now: Friday 07:00."])
    assert text == provider.reply
    assert len(provider.calls) == 1


def test_generate_diary_entry_none_provider():
    assert generate_diary_entry(provider=None, facts=["Now: Friday 07:00."]) is None


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------


def _scheduler_config(tmp_path) -> Config:
    config = Config()
    config.diary.enabled = True
    config.diary.hour = 7
    config.diary.store_path = str(tmp_path / "diary.json")
    config.memory.persist_path = str(tmp_path / "memory.json")
    config.memory.embeddings_path = str(tmp_path / "embeddings.json")
    config.memory.profile_path = str(tmp_path / "profile.json")
    # Idle-chatter diary peeks go dark inside the sleep window (default 1-8
    # local; CI runners are UTC) — tests must pass at any wall-clock hour.
    config.sleep_schedule.enabled = False
    return config


def _seed_profile(config, *, last_seen: dt.datetime, threads=("finish the draft",)):
    profile = UserProfile(
        open_threads=list(threads),
        last_episode="talked about the draft",
        last_seen=last_seen.isoformat(),
        session_count=3,
    )
    from pathlib import Path

    profile.save_atomic(Path(config.memory.profile_path))


def _make_scheduler(monkeypatch, tmp_path, now: dt.datetime, config=None):
    config = config or _scheduler_config(tmp_path)
    _seed_profile(config, last_seen=now - dt.timedelta(days=1))
    recorder = RecordingProvider()

    import amadeus.web.app as app_module

    monkeypatch.setattr(app_module, "load_llm", lambda cfg: recorder)
    app = create_app(config)
    store = app.state.session_store
    store.llm = recorder
    monkeypatch.setattr(
        "amadeus.engine.build_summary_provider", lambda cfg, llm: recorder
    )
    scheduler = BackgroundScheduler(store, config, clock=lambda: now)
    return scheduler, store, recorder, config


@pytest.mark.asyncio
async def test_scheduler_before_hour_noop(monkeypatch, tmp_path):
    now = dt.datetime(2026, 7, 10, 6, 30)
    scheduler, store, recorder, _ = _make_scheduler(monkeypatch, tmp_path, now)
    await scheduler._diary_job(now)
    assert store.diary_store.get(None) is None
    assert recorder.calls == []


@pytest.mark.asyncio
async def test_scheduler_generates_once_at_hour(monkeypatch, tmp_path):
    now = dt.datetime(2026, 7, 10, 7, 1)
    scheduler, store, recorder, config = _make_scheduler(monkeypatch, tmp_path, now)
    await scheduler._diary_job(now)
    entry = store.diary_store.get(None)
    assert entry is not None and entry.date == "2026-07-10"
    assert len(recorder.calls) == 1

    # Same-day second tick: no regeneration.
    await scheduler._diary_job(now + dt.timedelta(minutes=5))
    assert len(recorder.calls) == 1

    # Restart (fresh scheduler over the same store): still no regeneration.
    fresh = BackgroundScheduler(store, config, clock=lambda: now)
    await fresh._diary_job(now + dt.timedelta(hours=1))
    assert len(recorder.calls) == 1

    # Next day regenerates.
    tomorrow = now + dt.timedelta(days=1)
    await scheduler._diary_job(tomorrow)
    assert len(recorder.calls) == 2
    assert store.diary_store.get(None).date == "2026-07-11"


@pytest.mark.asyncio
async def test_scheduler_disabled_noop(monkeypatch, tmp_path):
    now = dt.datetime(2026, 7, 10, 7, 0)
    scheduler, store, recorder, config = _make_scheduler(monkeypatch, tmp_path, now)
    config.diary.enabled = False
    await scheduler._diary_job(now)
    assert recorder.calls == []


@pytest.mark.asyncio
async def test_brief_and_diary_coexist_no_in_flight_collision(monkeypatch, tmp_path):
    now = dt.datetime(2026, 7, 10, 8, 0)
    config = _scheduler_config(tmp_path)
    config.daily_brief.enabled = True
    config.daily_brief.hour = 6
    config.daily_brief.store_path = str(tmp_path / "brief.json")
    # Isolate brief+diary: the other scheduler jobs (default ON since
    # 2026-07-28, insights since 2026-07-30) would add provider calls and
    # break the exactly-two count.
    config.follow_through.enabled = False
    config.life_threads.enabled = False
    config.insights.enabled = False
    scheduler, store, recorder, config = _make_scheduler(
        monkeypatch, tmp_path, now, config=config
    )

    await scheduler._tick()

    assert store.brief_store is not None and store.brief_store.get(None) is not None
    assert store.diary_store is not None and store.diary_store.get(None) is not None
    # Both jobs ran to completion for the single candidate user: exactly two
    # provider calls (one per job), and no dangling in-flight keys.
    assert len(recorder.calls) == 2
    assert scheduler._in_flight == set()

    # A second tick the same day must not regenerate either.
    await scheduler._tick()
    assert len(recorder.calls) == 2


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------


def test_diary_endpoint_disabled_returns_null(tmp_path):
    config = Config()
    config.memory.persist_path = str(tmp_path / "memory.json")
    config.memory.embeddings_path = str(tmp_path / "embeddings.json")
    config.memory.profile_path = str(tmp_path / "profile.json")
    app = create_app(config)
    client = TestClient(app)
    assert client.get("/api/diary").json() == {"diary": None}


def test_diary_endpoint_returns_entry_without_consuming(monkeypatch, tmp_path):
    config = _scheduler_config(tmp_path)
    recorder = RecordingProvider()
    import amadeus.web.app as app_module

    monkeypatch.setattr(app_module, "load_llm", lambda cfg: recorder)
    app = create_app(config)
    store = app.state.session_store
    store.llm = recorder

    today = dt.date.today()
    store.diary_store.put(None, _entry(today, "endpoint copy"))
    client = TestClient(app)
    data = client.get("/api/diary").json()
    assert data["diary"]["text"] == "endpoint copy"
    assert data["diary"]["consumed"] is False
    assert store.diary_store.get(None).consumed is False


# P4 adds greeting/idle coverage below


def _diary_greeting_app(monkeypatch, tmp_path):
    config = _scheduler_config(tmp_path)
    config.memory.enable_greeting = True
    recorder = RecordingProvider(
        reply="Spent the afternoon arguing with a preprint. You're up next."
    )
    import amadeus.web.app as app_module

    monkeypatch.setattr(app_module, "load_llm", lambda cfg: recorder)
    app = create_app(config)
    store = app.state.session_store
    store.llm = recorder
    _, session = store.get(None, "diary-user")
    session.llm = recorder
    session.summary_provider = recorder
    return app, store, session, recorder


def test_greeting_folds_in_diary_and_consumes(monkeypatch, tmp_path):
    app, store, session, recorder = _diary_greeting_app(monkeypatch, tmp_path)
    # A diary-note fold-in presupposes an established relationship, not a
    # first meeting — pin session_count so it isn't read as one.
    session.memory.profile.session_count = 5
    today = dt.date.today()
    store.diary_store.put(
        "diary-user", _entry(today, "read a mediocre preprint and grumbled about it")
    )

    text = session.open_session_greeting()
    assert text is not None
    prompt_user = recorder.calls[-1][1].content
    assert "read a mediocre preprint and grumbled about it" in prompt_user
    assert "Your own diary note from today" in prompt_user
    assert store.diary_store.get("diary-user").consumed is True


def test_greeting_failure_leaves_diary_unconsumed(monkeypatch, tmp_path):
    app, store, session, recorder = _diary_greeting_app(monkeypatch, tmp_path)
    today = dt.date.today()
    store.diary_store.put("diary-user", _entry(today, "unconsumed diary entry"))
    recorder.fail = True
    assert session.open_session_greeting() is None
    assert store.diary_store.get("diary-user").consumed is False


def test_greeting_folds_in_both_brief_and_diary_and_consumes_both(
    monkeypatch, tmp_path
):
    config = _scheduler_config(tmp_path)
    config.memory.enable_greeting = True
    config.daily_brief.enabled = True
    config.daily_brief.store_path = str(tmp_path / "brief.json")
    recorder = RecordingProvider(reply="Morning briefing, then my own two cents.")
    import amadeus.web.app as app_module

    monkeypatch.setattr(app_module, "load_llm", lambda cfg: recorder)
    app = create_app(config)
    store = app.state.session_store
    store.llm = recorder
    _, session = store.get(None, "both-user")
    session.llm = recorder
    session.summary_provider = recorder
    # A brief/diary fold-in presupposes an established relationship, not a
    # first meeting — pin session_count so it isn't read as one.
    session.memory.profile.session_count = 5

    today = dt.date.today()
    store.brief_store.put("both-user", _brief_entry(today, "the revision is overdue"))
    store.diary_store.put("both-user", _entry(today, "read a paper, grumbled"))

    text = session.open_session_greeting()
    assert text is not None
    prompt_user = recorder.calls[-1][1].content
    assert "the revision is overdue" in prompt_user
    assert "read a paper, grumbled" in prompt_user
    assert store.brief_store.get("both-user").consumed is True
    assert store.diary_store.get("both-user").consumed is True


def _brief_entry(date: dt.date, text: str):
    from amadeus.brief import DailyBrief

    return DailyBrief(
        date=date.isoformat(), text=text, generated_at=f"{date.isoformat()}T06:00:00"
    )


def test_greeting_survives_corrupt_diary_store(monkeypatch, tmp_path):
    app, store, session, recorder = _diary_greeting_app(monkeypatch, tmp_path)

    class ExplodingStore:
        def peek_unconsumed(self, *a, **k):
            raise RuntimeError("disk on fire")

    session.diary_store = ExplodingStore()
    assert session.open_session_greeting() is not None


def test_idle_chatter_includes_todays_diary_without_consuming(monkeypatch, tmp_path):
    app, store, session, recorder = _diary_greeting_app(monkeypatch, tmp_path)
    session.config.memory.enable_idle_chatter = True
    session.memory.add_message("user", "hey")
    session.memory.add_message("assistant", "hey yourself")

    today = dt.date.today()
    store.diary_store.put("diary-user", _entry(today, "argued with a preprint today"))

    text = session.idle_chatter()
    assert text is not None
    prompt_user = recorder.calls[-1][1].content
    assert "argued with a preprint today" in prompt_user
    assert "What you did earlier today (your own note)" in prompt_user
    # Idle chatter peeks only — never consumes.
    assert store.diary_store.get("diary-user").consumed is False


def test_greeting_excludes_yesterdays_diary_entry(monkeypatch, tmp_path):
    app, store, session, recorder = _diary_greeting_app(monkeypatch, tmp_path)
    yesterday = dt.date.today() - dt.timedelta(days=1)
    store.diary_store.put("diary-user", _entry(yesterday, "yesterday's grumbling"))

    text = session.open_session_greeting()
    assert text is not None
    prompt_user = recorder.calls[-1][1].content
    assert "yesterday's grumbling" not in prompt_user
    # Stale entry is never consumed by the greeting either.
    assert store.diary_store.get("diary-user").consumed is False


def test_idle_chatter_excludes_yesterdays_diary_entry(monkeypatch, tmp_path):
    app, store, session, recorder = _diary_greeting_app(monkeypatch, tmp_path)
    session.config.memory.enable_idle_chatter = True
    session.memory.add_message("user", "hey")
    session.memory.add_message("assistant", "hey yourself")

    yesterday = dt.date.today() - dt.timedelta(days=1)
    store.diary_store.put("diary-user", _entry(yesterday, "yesterday's grumbling"))

    text = session.idle_chatter()
    assert text is not None
    prompt_user = recorder.calls[-1][1].content
    assert "yesterday's grumbling" not in prompt_user
