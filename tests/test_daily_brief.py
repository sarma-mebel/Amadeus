"""Daily brief: store semantics, scheduler due-predicate/idempotence, the
greeting fold-in with consume-after-success ordering, and the read-only
/api/brief endpoint + lifespan wiring."""

from __future__ import annotations

import datetime as dt

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from amadeus.brief import (
    DailyBrief,
    DailyBriefStore,
    collect_brief_facts,
    generate_daily_brief,
)
from amadeus.config import Config
from amadeus.memory.profile import UserProfile
from amadeus.providers.llm.base import ChatMessage
from amadeus.web.app import create_app
from amadeus.web.scheduler import BackgroundScheduler


class RecordingProvider:
    name = "recording"

    def __init__(
        self,
        reply: str = "Overdue: the CMB revision. Also, your open thread about the ICU model.",
    ) -> None:
        self.reply = reply
        self.calls: list[list[ChatMessage]] = []
        self.fail = False

    def generate(self, messages, **kwargs):
        self.calls.append(list(messages))
        if self.fail:
            raise RuntimeError("provider down")
        return self.reply


def _brief(date: dt.date, text: str = "brief text") -> DailyBrief:
    return DailyBrief(
        date=date.isoformat(), text=text, generated_at=f"{date.isoformat()}T06:00:00"
    )


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------


def test_store_roundtrip_and_per_user_keying(tmp_path):
    store = DailyBriefStore(str(tmp_path / "brief.json"))
    today = dt.date(2026, 7, 10)
    assert store.put(None, _brief(today, "default user")) is True
    assert store.put("alice", _brief(today, "alice's brief")) is True

    reloaded = DailyBriefStore(str(tmp_path / "brief.json"))
    assert reloaded.get(None).text == "default user"
    assert reloaded.get("alice").text == "alice's brief"


def test_store_put_refuses_same_date(tmp_path):
    store = DailyBriefStore(str(tmp_path / "brief.json"))
    today = dt.date(2026, 7, 10)
    assert store.put(None, _brief(today)) is True
    assert store.put(None, _brief(today, "duplicate")) is False
    assert store.get(None).text == "brief text"
    # Next day's brief overwrites.
    assert store.put(None, _brief(today + dt.timedelta(days=1), "tomorrow")) is True
    assert store.get(None).text == "tomorrow"


def test_store_peek_and_consume(tmp_path):
    store = DailyBriefStore(str(tmp_path / "brief.json"))
    today = dt.date(2026, 7, 10)
    store.put(None, _brief(today))
    assert store.peek_unconsumed(None, today).text == "brief text"
    # A stale (yesterday's) brief is never delivered late.
    assert store.peek_unconsumed(None, today + dt.timedelta(days=1)) is None
    store.mark_consumed(None, today.isoformat())
    assert store.peek_unconsumed(None, today) is None
    assert store.get(None).consumed is True


def test_store_corrupt_file_starts_clean(tmp_path):
    path = tmp_path / "brief.json"
    path.write_text("{corrupt", encoding="utf-8")
    store = DailyBriefStore(str(path))
    assert store.get(None) is None
    assert store.put(None, _brief(dt.date(2026, 7, 10))) is True


# ---------------------------------------------------------------------------
# Facts / generation
# ---------------------------------------------------------------------------


def test_collect_brief_facts_prioritized_content(tmp_path):
    from amadeus.reminders import Reminder

    now = dt.datetime(2026, 7, 10, 6, 0)
    profile = UserProfile(
        open_threads=["the ICU foundation model deadline"],
        last_episode="we argued about preprint servers",
    )
    overdue = Reminder(
        id="r1",
        text="submit the revision",
        due_at=now - dt.timedelta(hours=10),
        created_at=now - dt.timedelta(days=1),
    )
    facts = collect_brief_facts(
        profile=profile,
        pending_reminders=[overdue],
        arxiv_hits=[{"title": "A Paper", "date": "2026-07-09"}],
        now=now,
    )
    joined = "\n".join(facts)
    assert "Overdue reminders: submit the revision" in joined
    assert "ICU foundation model" in joined
    assert "we argued about preprint servers" in joined
    assert "A Paper" in joined


def test_generate_daily_brief_requires_content_and_survives_failure():
    provider = RecordingProvider()
    # Timestamp-only facts -> nothing worth briefing.
    assert generate_daily_brief(provider=provider, facts=["Now: Friday 06:00."]) is None
    assert provider.calls == []

    text = generate_daily_brief(
        provider=provider, facts=["Now: Friday 06:00.", "Open threads: x"]
    )
    assert text == provider.reply

    provider.fail = True
    assert (
        generate_daily_brief(
            provider=provider, facts=["Now: Friday 06:00.", "Open threads: x"]
        )
        is None
    )


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------


def _scheduler_config(tmp_path) -> Config:
    config = Config()
    config.daily_brief.enabled = True
    config.daily_brief.hour = 6
    config.daily_brief.store_path = str(tmp_path / "daily_brief.json")
    config.memory.persist_path = str(tmp_path / "memory.json")
    config.memory.embeddings_path = str(tmp_path / "embeddings.json")
    config.memory.profile_path = str(tmp_path / "profile.json")
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


def _make_scheduler(monkeypatch, tmp_path, now: dt.datetime):
    config = _scheduler_config(tmp_path)
    _seed_profile(config, last_seen=now - dt.timedelta(days=1))
    recorder = RecordingProvider()

    import amadeus.web.app as app_module

    monkeypatch.setattr(app_module, "load_llm", lambda cfg: recorder)
    app = create_app(config)
    store = app.state.session_store
    store.llm = recorder
    monkeypatch.setattr(
        "amadeus.web.scheduler.build_summary_provider",
        lambda cfg, llm: recorder,
        raising=False,
    )
    # build_summary_provider is imported inside _generate_for_user from
    # amadeus.engine — patch it there.
    monkeypatch.setattr(
        "amadeus.engine.build_summary_provider", lambda cfg, llm: recorder
    )
    scheduler = BackgroundScheduler(store, config, clock=lambda: now)
    return scheduler, store, recorder, config


@pytest.mark.asyncio
async def test_scheduler_before_hour_noop(monkeypatch, tmp_path):
    now = dt.datetime(2026, 7, 10, 5, 30)
    scheduler, store, recorder, _ = _make_scheduler(monkeypatch, tmp_path, now)
    await scheduler._daily_brief_job(now)
    assert store.brief_store.get(None) is None
    assert recorder.calls == []


@pytest.mark.asyncio
async def test_scheduler_generates_once_at_hour(monkeypatch, tmp_path):
    now = dt.datetime(2026, 7, 10, 6, 1)
    scheduler, store, recorder, config = _make_scheduler(monkeypatch, tmp_path, now)
    await scheduler._daily_brief_job(now)
    brief = store.brief_store.get(None)
    assert brief is not None and brief.date == "2026-07-10"
    assert len(recorder.calls) == 1

    # Same-day second tick: no regeneration.
    await scheduler._daily_brief_job(now + dt.timedelta(minutes=5))
    assert len(recorder.calls) == 1

    # Restart (fresh scheduler over the same store): still no regeneration.
    fresh = BackgroundScheduler(store, config, clock=lambda: now)
    await fresh._daily_brief_job(now + dt.timedelta(hours=1))
    assert len(recorder.calls) == 1

    # Next day regenerates.
    tomorrow = now + dt.timedelta(days=1)
    await scheduler._daily_brief_job(tomorrow)
    assert len(recorder.calls) == 2
    assert store.brief_store.get(None).date == "2026-07-11"


@pytest.mark.asyncio
async def test_scheduler_survives_provider_failure(monkeypatch, tmp_path):
    now = dt.datetime(2026, 7, 10, 7, 0)
    scheduler, store, recorder, _ = _make_scheduler(monkeypatch, tmp_path, now)
    recorder.fail = True
    await scheduler._tick()  # must not raise
    assert store.brief_store.get(None) is None
    recorder.fail = False
    await scheduler._daily_brief_job(now)  # retried next tick
    assert store.brief_store.get(None) is not None


@pytest.mark.asyncio
async def test_scheduler_failed_brief_holds_after_daily_retry_cap(
    monkeypatch, tmp_path
):
    """Bounded retry: a persistently failing generation used to be re-paid
    every 60s tick until midnight. Three failures now hold the job until
    tomorrow, when the budget resets."""
    now = dt.datetime(2026, 7, 10, 7, 0)
    scheduler, store, recorder, _ = _make_scheduler(monkeypatch, tmp_path, now)
    recorder.fail = True
    for i in range(6):
        await scheduler._daily_brief_job(now + dt.timedelta(minutes=i))
    assert len(recorder.calls) == 3  # capped at 3 paid attempts, not 6
    assert store.brief_store.get(None) is None

    recorder.fail = False
    await scheduler._daily_brief_job(now + dt.timedelta(days=1))
    assert store.brief_store.get(None) is not None


@pytest.mark.asyncio
async def test_scheduler_skips_idle_users(monkeypatch, tmp_path):
    now = dt.datetime(2026, 7, 10, 7, 0)
    scheduler, store, recorder, config = _make_scheduler(monkeypatch, tmp_path, now)
    _seed_profile(config, last_seen=now - dt.timedelta(days=30))
    await scheduler._daily_brief_job(now)
    assert recorder.calls == []
    assert store.brief_store.get(None) is None


@pytest.mark.asyncio
async def test_scheduler_disabled_noop(monkeypatch, tmp_path):
    now = dt.datetime(2026, 7, 10, 7, 0)
    scheduler, store, recorder, config = _make_scheduler(monkeypatch, tmp_path, now)
    config.daily_brief.enabled = False
    await scheduler._daily_brief_job(now)
    assert recorder.calls == []


# ---------------------------------------------------------------------------
# Per-character isolation (the scheduler follows the ACTIVE character)
# ---------------------------------------------------------------------------


def _activate_guest(monkeypatch, store, config):
    """Switch the running store onto a guest character the way the app does
    (routes_characters -> reload_providers), which swaps store.config — the
    live object the scheduler re-reads every tick."""
    import dataclasses

    import amadeus.web.app as app_module

    monkeypatch.setattr(app_module, "load_embedder", lambda cfg: None)
    monkeypatch.setattr(app_module, "load_tts", lambda cfg: None)
    guest_config = dataclasses.replace(config)
    guest_config.character = dataclasses.replace(
        config.character, name="Rem", full_name="Rem", system_prompt="You are Rem."
    )
    store.reload_providers(guest_config)
    return guest_config


@pytest.mark.asyncio
async def test_scheduler_follows_character_switch_between_ticks(monkeypatch, tmp_path):
    """Tick 1 (Kurisu active) reads her profile and stores her brief; after a
    mid-day character switch, tick 2 reads the GUEST's scoped profile and
    stores into the guest's scoped brief file — Kurisu's data is neither
    read nor written while the guest is active."""
    from pathlib import Path

    now = dt.datetime(2026, 7, 10, 6, 30)
    scheduler, store, recorder, config = _make_scheduler(monkeypatch, tmp_path, now)

    # Tick under Kurisu: brief generated from her (unscoped) profile.
    await scheduler._daily_brief_job(now)
    assert len(recorder.calls) == 1
    assert "finish the draft" in recorder.calls[0][-1].content
    assert (tmp_path / "daily_brief.json").exists()

    # Seed the guest's isolated profile with its own open thread.
    guest_profile = UserProfile(
        open_threads=["debug the guest pack"],
        last_episode="talked to Rem",
        last_seen=(now - dt.timedelta(hours=2)).isoformat(),
        session_count=1,
    )
    guest_profile.save_atomic(tmp_path / "characters" / "rem" / "profile.json")

    _activate_guest(monkeypatch, store, config)

    # Next tick: the guest's paths are re-derived from the live config.
    later = now + dt.timedelta(minutes=1)
    await scheduler._daily_brief_job(later)
    assert len(recorder.calls) == 2
    facts = recorder.calls[1][-1].content
    assert "debug the guest pack" in facts
    assert "finish the draft" not in facts
    # Stored in the guest's scoped store, not Kurisu's.
    guest_brief_path = tmp_path / "characters" / "rem" / "daily_brief.json"
    assert store.brief_store.path == str(guest_brief_path)
    assert guest_brief_path.exists()
    guest_brief = store.brief_store.get(None)
    assert guest_brief is not None and guest_brief.date == "2026-07-10"
    # Kurisu's brief file was not touched by the guest's generation.
    kurisu_store = DailyBriefStore(str(tmp_path / "daily_brief.json"))
    assert kurisu_store.get(None).generated_at.startswith("2026-07-10T06:30")
    assert Path(tmp_path / "daily_brief.json").exists()


@pytest.mark.asyncio
async def test_scheduler_guest_without_profile_generates_nothing(monkeypatch, tmp_path):
    """With a guest active and no guest profile on disk yet, there are no
    candidate users — Kurisu's profile must not leak into a guest brief."""
    now = dt.datetime(2026, 7, 10, 6, 30)
    scheduler, store, recorder, config = _make_scheduler(monkeypatch, tmp_path, now)
    _activate_guest(monkeypatch, store, config)

    await scheduler._daily_brief_job(now)
    assert recorder.calls == []
    assert store.brief_store.get(None) is None


@pytest.mark.asyncio
async def test_scheduler_flag_off_keeps_legacy_paths_with_guest_active(
    monkeypatch, tmp_path
):
    """memory.per_character_memory off = fully legacy: even with a guest
    character active the scheduler reads the shared profile and writes the
    shared brief store."""
    now = dt.datetime(2026, 7, 10, 6, 30)
    scheduler, store, recorder, config = _make_scheduler(monkeypatch, tmp_path, now)
    config.memory.per_character_memory = False
    _activate_guest(monkeypatch, store, config)

    await scheduler._daily_brief_job(now)
    assert len(recorder.calls) == 1
    assert "finish the draft" in recorder.calls[0][-1].content
    assert store.brief_store.path == str(tmp_path / "daily_brief.json")
    assert store.brief_store.get(None) is not None


# ---------------------------------------------------------------------------
# Greeting fold-in
# ---------------------------------------------------------------------------


def _greeting_app(monkeypatch, tmp_path):
    config = _scheduler_config(tmp_path)
    config.memory.enable_greeting = True
    recorder = RecordingProvider(
        reply="Morning. Your revision is overdue — deal with it before coffee."
    )
    import amadeus.web.app as app_module

    monkeypatch.setattr(app_module, "load_llm", lambda cfg: recorder)
    app = create_app(config)
    store = app.state.session_store
    store.llm = recorder
    _, session = store.get(None, "brief-user")
    session.llm = recorder
    session.summary_provider = recorder
    return app, store, session, recorder


def test_greeting_folds_in_brief_and_consumes(monkeypatch, tmp_path):
    app, store, session, recorder = _greeting_app(monkeypatch, tmp_path)
    # A morning-brief delivery presupposes an established relationship, not
    # a first meeting — pin session_count so it isn't read as one.
    session.memory.profile.session_count = 5
    today = dt.date.today()
    store.brief_store.put("brief-user", _brief(today, "the revision is overdue"))

    text = session.open_session_greeting()
    assert text is not None
    prompt_user = recorder.calls[-1][1].content
    prompt_system = recorder.calls[-1][0].content
    assert "the revision is overdue" in prompt_user
    assert "2-4 short sentences" in prompt_system
    assert store.brief_store.get("brief-user").consumed is True

    # Second greeting: brief already consumed, back to the one-liner.
    session.memory.short_term.clear()
    assert session.open_session_greeting() is not None
    assert "Morning brief to deliver" not in recorder.calls[-1][1].content
    assert "ONE short opening line" in recorder.calls[-1][0].content


def test_greeting_failure_leaves_brief_unconsumed(monkeypatch, tmp_path):
    app, store, session, recorder = _greeting_app(monkeypatch, tmp_path)
    today = dt.date.today()
    store.brief_store.put("brief-user", _brief(today, "unconsumed brief"))
    recorder.fail = True
    assert session.open_session_greeting() is None
    assert store.brief_store.get("brief-user").consumed is False


def test_greeting_survives_corrupt_brief_store(monkeypatch, tmp_path):
    app, store, session, recorder = _greeting_app(monkeypatch, tmp_path)

    class ExplodingStore:
        def peek_unconsumed(self, *a, **k):
            raise RuntimeError("disk on fire")

    session.brief_store = ExplodingStore()
    assert session.open_session_greeting() is not None


# ---------------------------------------------------------------------------
# Endpoint + lifespan
# ---------------------------------------------------------------------------


def test_brief_endpoint_disabled_returns_null(tmp_path):
    config = Config()
    config.memory.persist_path = str(tmp_path / "memory.json")
    config.memory.embeddings_path = str(tmp_path / "embeddings.json")
    config.memory.profile_path = str(tmp_path / "profile.json")
    app = create_app(config)
    client = TestClient(app)
    assert client.get("/api/brief").json() == {"brief": None}


def test_brief_endpoint_returns_brief_without_consuming(monkeypatch, tmp_path):
    app, store, _session, _recorder = _greeting_app(monkeypatch, tmp_path)
    today = dt.date.today()
    store.brief_store.put(None, _brief(today, "endpoint copy"))
    client = TestClient(app)
    data = client.get("/api/brief").json()
    assert data["brief"]["text"] == "endpoint copy"
    assert data["brief"]["consumed"] is False
    assert store.brief_store.get(None).consumed is False


@pytest.mark.asyncio
async def test_lifespan_starts_scheduler_iff_enabled(monkeypatch, tmp_path):
    config = _scheduler_config(tmp_path)
    recorder = RecordingProvider()
    import amadeus.web.app as app_module

    monkeypatch.setattr(app_module, "load_llm", lambda cfg: recorder)
    app = create_app(config)
    async with app.router.lifespan_context(app):
        assert app.state.scheduler is not None
        assert app.state.scheduler._task is not None
    assert app.state.scheduler._task is None  # stopped on shutdown

    config2 = _scheduler_config(tmp_path)
    # Every scheduler-driving feature must be off for the negative case —
    # diary/follow-through/life-threads default ON since 2026-07-28.
    config2.daily_brief.enabled = False
    config2.diary.enabled = False
    config2.reach_out.enabled = False
    config2.follow_through.enabled = False
    config2.life_threads.enabled = False
    config2.insights.enabled = False
    config2.self_eval.enabled = False
    app2 = create_app(config2)
    async with app2.router.lifespan_context(app2):
        assert app2.state.scheduler is None
