"""Life threads (her own serialized pursuits): store semantics, the
seed/progress lifecycle with the one-call-per-day pace guard, diary and
idle-chatter surfacing, and the read-only /api/life_threads endpoint."""

from __future__ import annotations

import datetime as dt

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from amadeus.config import Config
from amadeus.life_threads import (
    LifeThread,
    LifeThreadStore,
    life_thread_seeds,
    progress_life_thread,
    seed_due,
    seed_life_thread,
    thread_due_for_progress,
)
from amadeus.memory.profile import UserProfile
from amadeus.providers.llm.base import ChatMessage
from amadeus.web.app import create_app
from amadeus.web.scheduler import BackgroundScheduler


class SequenceProvider:
    name = "sequence"

    def __init__(self, replies) -> None:
        self.replies = list(replies)
        self.calls: list[list[ChatMessage]] = []
        self.fail = False

    def generate(self, messages, **kwargs):
        self.calls.append(list(messages))
        if self.fail:
            raise RuntimeError("provider down")
        if not self.replies:
            return ""
        if len(self.replies) == 1:
            return self.replies[0]
        return self.replies.pop(0)


def _thread(
    topic: str = "re-running the retraction-bait stats",
    started: str = "2026-07-01",
    last: str = "2026-07-20",
    status: str = "active",
    progressions: int = 2,
) -> LifeThread:
    return LifeThread(
        topic=topic,
        summary="halfway through the reanalysis",
        started_at=started,
        last_progressed_at=last,
        status=status,
        progressions=progressions,
    )


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------


def test_store_roundtrip_and_duplicate_topic_refused(tmp_path):
    store = LifeThreadStore(str(tmp_path / "life.json"))
    today = dt.date(2026, 7, 27)
    assert store.add_thread(_thread(), today) is True
    assert (
        store.add_thread(_thread(topic="RE-RUNNING the retraction-bait stats"), today)
        is False
    )
    assert [t.topic for t in store.active_threads()] == [_thread().topic]
    assert store.acted_today(today) is True
    assert store.acted_today(today + dt.timedelta(days=1)) is False


def test_store_progress_and_conclude(tmp_path):
    store = LifeThreadStore(str(tmp_path / "life.json"))
    today = dt.date(2026, 7, 27)
    store.add_thread(_thread(), dt.date(2026, 7, 20))
    assert (
        store.record_progress(
            _thread().topic,
            summary="the reanalysis found an off-by-one in their exclusion",
            step_text="found an off-by-one",
            today=today,
        )
        is True
    )
    t = store.active_threads()[0]
    assert t.progressions == 3 and t.last_progressed_at == today.isoformat()
    assert t.history[-1]["text"] == "found an off-by-one"
    # Conclude removes it from active but keeps it for reference.
    store.record_progress(
        _thread().topic,
        summary="done — wrote the whole thing up",
        step_text="wrapped it up",
        today=today,
        concluded=True,
    )
    assert store.active_threads() == []
    assert [t.status for t in store.all_threads()] == ["concluded"]
    # Progressing a concluded/unknown thread is refused.
    assert (
        store.record_progress(_thread().topic, summary="x", step_text="x", today=today)
        is False
    )


def test_store_corrupt_file_starts_clean(tmp_path):
    path = tmp_path / "life.json"
    path.write_text("{broken", encoding="utf-8")
    store = LifeThreadStore(str(path))
    assert store.all_threads() == []
    assert store.add_thread(_thread(), dt.date(2026, 7, 27)) is True


# ---------------------------------------------------------------------------
# Lifecycle predicates + generation
# ---------------------------------------------------------------------------


def test_thread_due_for_progress_interval():
    today = dt.date(2026, 7, 27)
    assert thread_due_for_progress(_thread(last="2026-07-23"), today, 3) is True
    assert thread_due_for_progress(_thread(last="2026-07-26"), today, 3) is False
    assert thread_due_for_progress(_thread(status="concluded"), today, 3) is False
    # Unparseable stamps progress rather than stall forever.
    assert thread_due_for_progress(_thread(last="???", started="???"), today, 3) is True


def test_seed_due_slots_and_stagger(tmp_path):
    store = LifeThreadStore(str(tmp_path / "life.json"))
    today = dt.date(2026, 7, 27)
    assert seed_due(store, today, max_active=2) is True
    # A recently started thread defers the next seed (stagger).
    store.add_thread(_thread(started="2026-07-25"), dt.date(2026, 7, 25))
    assert seed_due(store, today, max_active=2) is False
    # Old-enough start frees the stagger; full roster blocks regardless.
    store2 = LifeThreadStore(str(tmp_path / "life2.json"))
    store2.add_thread(_thread(started="2026-07-01"), dt.date(2026, 7, 1))
    assert seed_due(store2, today, max_active=2) is True
    store2.add_thread(
        _thread(topic="second pursuit", started="2026-07-02"), dt.date(2026, 7, 2)
    )
    assert seed_due(store2, today, max_active=2) is False


def test_seed_life_thread_parse_and_failure():
    provider = SequenceProvider(
        [
            "TOPIC: re-checking a sloppy meta-analysis\nSTATE: Pulled the data tables tonight."
        ]
    )
    seeded = seed_life_thread(provider=provider, existing_topics=["old pursuit"])
    assert seeded is not None
    assert seeded.topic == "re-checking a sloppy meta-analysis"
    assert seeded.summary == "Pulled the data tables tonight."
    # The prompt forbids user-involving pursuits and lists existing topics.
    system = provider.calls[0][0].content
    assert "never about" in system.lower() and "old pursuit" in system

    assert (
        seed_life_thread(
            provider=SequenceProvider(["no labels here"]), existing_topics=[]
        )
        is None
    )
    failing = SequenceProvider(["x"])
    failing.fail = True
    assert seed_life_thread(provider=failing, existing_topics=[]) is None
    assert seed_life_thread(provider=None, existing_topics=[]) is None


def test_progress_life_thread_step_done_and_wrapup():
    provider = SequenceProvider(
        ["The off-by-one was real. Re-ran everything; effect halves."]
    )
    result = progress_life_thread(provider=provider, thread=_thread(), wrap_up=False)
    assert result is not None and result["concluded"] is False
    assert "off-by-one" in result["step"]

    done = progress_life_thread(
        provider=SequenceProvider(["DONE: Wrote it up; the effect doesn't replicate."]),
        thread=_thread(),
        wrap_up=True,
    )
    assert done is not None and done["concluded"] is True
    assert done["summary"].startswith("Wrote it up")

    wrap_provider = SequenceProvider(["step"])
    progress_life_thread(provider=wrap_provider, thread=_thread(), wrap_up=True)
    assert "conclusion" in wrap_provider.calls[0][0].content

    failing = SequenceProvider(["x"])
    failing.fail = True
    assert (
        progress_life_thread(provider=failing, thread=_thread(), wrap_up=False) is None
    )


def test_life_thread_seeds_best_effort(tmp_path):
    assert life_thread_seeds(None) == []
    store = LifeThreadStore(str(tmp_path / "life.json"))
    store.add_thread(_thread(), dt.date(2026, 7, 27))
    seeds = life_thread_seeds(store)
    assert len(seeds) == 1 and _thread().topic in seeds[0]


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------


def _config(tmp_path) -> Config:
    config = Config()
    config.life_threads.enabled = True
    config.life_threads.hour = 7
    config.life_threads.store_path = str(tmp_path / "life.json")
    config.memory.persist_path = str(tmp_path / "memory.json")
    config.memory.embeddings_path = str(tmp_path / "embeddings.json")
    config.memory.profile_path = str(tmp_path / "profile.json")
    # Idle-chatter peeks go dark inside the sleep window (default 1-8 local;
    # CI runners are UTC) — tests must pass at any wall-clock hour.
    config.sleep_schedule.enabled = False
    return config


def _make_scheduler(monkeypatch, tmp_path, now, replies):
    config = _config(tmp_path)
    provider = SequenceProvider(replies)

    import amadeus.web.app as app_module

    monkeypatch.setattr(app_module, "load_llm", lambda cfg: provider)
    app = create_app(config)
    store = app.state.session_store
    store.llm = provider
    monkeypatch.setattr(
        "amadeus.engine.build_summary_provider", lambda cfg, llm: provider
    )
    scheduler = BackgroundScheduler(store, config, clock=lambda: now)
    return scheduler, store, provider, config


@pytest.mark.asyncio
async def test_scheduler_seeds_then_paces_one_call_per_day(monkeypatch, tmp_path):
    now = dt.datetime(2026, 7, 27, 8, 0)
    scheduler, store, provider, _ = _make_scheduler(
        monkeypatch,
        tmp_path,
        now,
        ["TOPIC: a stubborn simulation\nSTATE: It diverges after step 40."],
    )
    await scheduler._life_threads_job(now)
    assert [t.topic for t in store.life_store.active_threads()] == [
        "a stubborn simulation"
    ]
    # Same day, later tick: pace guard blocks a second call.
    await scheduler._life_threads_job(now + dt.timedelta(hours=2))
    assert len(provider.calls) == 1


@pytest.mark.asyncio
async def test_scheduler_progresses_due_thread_before_seeding(monkeypatch, tmp_path):
    now = dt.datetime(2026, 7, 27, 8, 0)
    scheduler, store, provider, _ = _make_scheduler(
        monkeypatch, tmp_path, now, ["Found the divergence: a sign error."]
    )
    store.life_store.add_thread(_thread(last="2026-07-20"), dt.date(2026, 7, 20))
    await scheduler._life_threads_job(now)
    t = store.life_store.active_threads()[0]
    assert "sign error" in t.summary
    # Progression (not a seed) consumed today's single call.
    assert len(provider.calls) == 1
    assert "next small step" in provider.calls[0][0].content


@pytest.mark.asyncio
async def test_scheduler_before_hour_disabled_and_failure(monkeypatch, tmp_path):
    early = dt.datetime(2026, 7, 27, 6, 0)
    scheduler, store, provider, config = _make_scheduler(
        monkeypatch, tmp_path, early, ["TOPIC: t\nSTATE: s"]
    )
    await scheduler._life_threads_job(early)
    assert provider.calls == []

    now = dt.datetime(2026, 7, 27, 8, 0)
    provider.fail = True
    scheduler.clock = lambda: now
    await scheduler._tick()  # must not raise
    assert store.life_store.all_threads() == []
    # Failure left no activity stamp: the next tick retries.
    provider.fail = False
    await scheduler._life_threads_job(now)
    assert len(store.life_store.active_threads()) == 1

    config.life_threads.enabled = False
    await scheduler._life_threads_job(now + dt.timedelta(days=1))
    assert len(provider.calls) == 2  # failed seed + retry seed, nothing after


@pytest.mark.asyncio
async def test_scheduler_failed_generation_holds_after_daily_retry_cap(
    monkeypatch, tmp_path
):
    """Bounded retry: a persistently failing generation used to be re-paid
    every 60s tick for the rest of the day (~1000 calls against the
    documented pace of one). It now stops after 3 failures and resumes
    tomorrow."""
    now = dt.datetime(2026, 7, 27, 8, 0)
    scheduler, store, provider, _ = _make_scheduler(
        monkeypatch, tmp_path, now, ["TOPIC: t\nSTATE: s"]
    )
    provider.fail = True
    for i in range(6):
        await scheduler._life_threads_job(now + dt.timedelta(minutes=i))
    assert len(provider.calls) == 3  # capped at 3 paid attempts, not 6

    # Tomorrow the budget resets and a healthy provider seeds normally.
    provider.fail = False
    await scheduler._life_threads_job(now + dt.timedelta(days=1))
    assert len(store.life_store.active_threads()) == 1


@pytest.mark.asyncio
async def test_scheduler_leak_rejection_clears_pending_input(monkeypatch, tmp_path):
    """A leak-rejected progression consumes the stashed opinion-ask answer:
    left in place, the same input re-provoked the same second-person leak on
    every retry, a deterministic all-day reject loop."""
    now = dt.datetime(2026, 7, 27, 8, 0)
    scheduler, store, provider, _ = _make_scheduler(
        monkeypatch,
        tmp_path,
        now,
        ["You were right — I dropped the outliers like you said."],
    )
    topic = _thread().topic
    store.life_store.add_thread(_thread(last="2026-07-20"), dt.date(2026, 7, 20))
    assert store.life_store.record_user_input(topic, "drop the outliers") is True

    await scheduler._life_threads_job(now)

    (thread,) = store.life_store.active_threads()
    assert thread.progressions == 2  # the leaking step was NOT stored
    assert "outliers like you said" not in thread.summary
    assert thread.pending_input == ""  # the poisoning input is consumed


# ---------------------------------------------------------------------------
# Surfacing
# ---------------------------------------------------------------------------


def test_diary_seeds_include_life_threads():
    from amadeus.diary import collect_diary_seeds

    seeds = collect_diary_seeds(
        profile=UserProfile(),
        now=dt.datetime(2026, 7, 27, 7, 0),
        life_seeds=["Your ongoing pursuit: the simulation — currently: diverging"],
    )
    assert any("ongoing pursuit" in s for s in seeds)


def test_idle_chatter_peeks_life_threads(monkeypatch, tmp_path):
    config = _config(tmp_path)
    config.memory.enable_idle_chatter = True
    provider = SequenceProvider(
        ["That simulation of mine is still diverging, by the way."]
    )
    import amadeus.web.app as app_module

    monkeypatch.setattr(app_module, "load_llm", lambda cfg: provider)
    app = create_app(config)
    store = app.state.session_store
    store.llm = provider
    store.life_store.add_thread(
        _thread(topic="the diverging simulation"), dt.date(2026, 7, 26)
    )
    _, session = store.get(None, "life-user")
    session.llm = provider
    session.summary_provider = provider
    session.memory.add_message("user", "hello there")
    session.memory.add_message("assistant", "hm.")

    line = session.idle_chatter()
    assert line is not None
    assert "the diverging simulation" in provider.calls[-1][1].content


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------


def test_life_threads_endpoint(monkeypatch, tmp_path):
    config = _config(tmp_path)
    import amadeus.web.app as app_module

    monkeypatch.setattr(app_module, "load_llm", lambda cfg: SequenceProvider(["hi"]))
    app = create_app(config)
    store = app.state.session_store
    store.life_store.add_thread(_thread(), dt.date(2026, 7, 27))
    client = TestClient(app)
    body = client.get("/api/life_threads").json()
    assert body["threads"][0]["topic"] == _thread().topic
    assert body["threads"][0]["status"] == "active"


def test_life_threads_endpoint_disabled(monkeypatch, tmp_path):
    config = _config(tmp_path)
    config.life_threads.enabled = False
    import amadeus.web.app as app_module

    monkeypatch.setattr(app_module, "load_llm", lambda cfg: SequenceProvider(["hi"]))
    app = create_app(config)
    client = TestClient(app)
    assert client.get("/api/life_threads").json() == {"threads": []}
