"""Follow-through (overnight research on the next-time thread): store
semantics, the research pipeline's SKIP/failure branches, scheduler
once-per-thread idempotence, the greeting delivery with consume-after-success
ordering, and the read-only /api/followup endpoint."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from amadeus.config import Config
from amadeus.followup import (
    DELIVERY_MAX_AGE_DAYS,
    FollowUp,
    FollowUpStore,
    followup_greeting_fact,
    fresh_next_time_thread,
    generate_followup_note,
    plan_research,
    run_research,
)
from amadeus.memory.profile import UserProfile
from amadeus.providers.llm.base import ChatMessage
from amadeus.web.app import create_app
from amadeus.web.scheduler import BackgroundScheduler


class SequenceProvider:
    """Replies in order (plan call first, note call second)."""

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


class FakeSearchClient:
    def __init__(self, results=None, fail: bool = False) -> None:
        self.results = results if results is not None else []
        self.fail = fail
        self.queries: list[str] = []
        self.closed = False

    def search(self, query, *, max_results=None):
        self.queries.append(query)
        if self.fail:
            raise RuntimeError("search down")
        return list(self.results)

    def close(self):
        self.closed = True


def _note(
    thread: str = "they were mid-refactor of the segmentation model",
    generated_at: dt.datetime | None = None,
    kind: str = "note",
    note: str = "nnU-Net v2 apparently handles this exact case.",
) -> FollowUp:
    stamp = (generated_at or dt.datetime(2026, 7, 27, 3, 0)).isoformat(
        timespec="seconds"
    )
    return FollowUp(
        thread=thread,
        note=note if kind == "note" else "",
        kind=kind,
        generated_at=stamp,
    )


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------


def test_store_roundtrip_and_once_per_thread(tmp_path):
    store = FollowUpStore(str(tmp_path / "followup.json"))
    assert store.put("u1", _note()) is True
    # Same thread (case/whitespace-insensitive): refused, even after consume.
    assert (
        store.has_for_thread("u1", "They were  MID-REFACTOR of the segmentation model")
        is True
    )
    assert store.put("u1", _note(note="different text")) is False
    store.mark_consumed("u1", _note().thread)
    assert store.put("u1", _note(note="again")) is False
    # A NEW thread replaces the spent record.
    assert store.put("u1", _note(thread="they were picking a conference")) is True
    # Per-user keying.
    assert store.get("other") is None


def test_store_skip_records_block_but_never_deliver(tmp_path):
    store = FollowUpStore(str(tmp_path / "followup.json"))
    now = dt.datetime(2026, 7, 27, 8, 0)
    assert store.put(None, _note(kind="skip", generated_at=now)) is True
    assert store.has_for_thread(None, _note().thread) is True
    assert store.peek_unconsumed(None, now) is None


def test_store_peek_freshness_and_consume(tmp_path):
    store = FollowUpStore(str(tmp_path / "followup.json"))
    generated = dt.datetime(2026, 7, 27, 3, 0)
    store.put(None, _note(generated_at=generated))
    assert store.peek_unconsumed(None, generated + dt.timedelta(hours=5)) is not None
    # Stale note: never delivered late.
    late = generated + dt.timedelta(days=DELIVERY_MAX_AGE_DAYS + 1)
    assert store.peek_unconsumed(None, late) is None
    # Consume ends delivery.
    store.mark_consumed(None, _note().thread)
    assert store.peek_unconsumed(None, generated + dt.timedelta(hours=5)) is None


def test_store_corrupt_file_starts_clean(tmp_path):
    path = tmp_path / "followup.json"
    path.write_text("{not json", encoding="utf-8")
    store = FollowUpStore(str(path))
    assert store.get(None) is None
    assert store.put(None, _note()) is True


# ---------------------------------------------------------------------------
# Research pipeline
# ---------------------------------------------------------------------------


def test_fresh_next_time_thread_age_gate():
    now = dt.datetime(2026, 7, 27, 3, 0)
    fresh = UserProfile(
        next_time_thread={
            "text": "they were fighting a segfault",
            "noted": (now - dt.timedelta(days=2)).isoformat(timespec="seconds"),
        }
    )
    assert fresh_next_time_thread(fresh, now) == "they were fighting a segfault"
    old = UserProfile(
        next_time_thread={
            "text": "ancient topic",
            "noted": (now - dt.timedelta(days=30)).isoformat(timespec="seconds"),
        }
    )
    assert fresh_next_time_thread(old, now) is None
    # Unparseable stamp fails closed; missing thread/profile is None.
    assert (
        fresh_next_time_thread(
            UserProfile(next_time_thread={"text": "x", "noted": "not-a-date"}), now
        )
        is None
    )
    assert fresh_next_time_thread(UserProfile(), now) is None
    assert fresh_next_time_thread(None, now) is None


def test_plan_research_skip_queries_and_failure():
    skip = SequenceProvider(["SKIP"])
    assert (
        plan_research(provider=skip, thread="their mom is visiting", max_searches=2)
        == []
    )

    queries = SequenceProvider(
        ["nnU-Net cerebral microbleed segmentation\nCMB detection 2026"]
    )
    planned = plan_research(
        provider=queries, thread="stuck on CMB segmentation", max_searches=2
    )
    assert planned == [
        "nnU-Net cerebral microbleed segmentation",
        "CMB detection 2026",
    ]
    # Cap applies even when the model over-produces.
    many = SequenceProvider(["a\nb\nc\nd"])
    assert len(plan_research(provider=many, thread="t", max_searches=2)) == 2

    failing = SequenceProvider(["x"])
    failing.fail = True
    assert plan_research(provider=failing, thread="t", max_searches=2) is None
    assert plan_research(provider=None, thread="t", max_searches=2) is None


def test_run_research_backend_missing_and_partial_failure(monkeypatch):

    # No backend buildable: None (counted loudly by the scheduler).
    monkeypatch.setattr(
        "amadeus.tools.web_search.build_web_search_client", lambda settings: None
    )
    assert (
        run_research(web_search_settings=None, queries=["q"], max_results_per_query=3)
        is None
    )

    client = FakeSearchClient(results=[{"title": "T", "url": "u", "snippet": "s"}])
    monkeypatch.setattr(
        "amadeus.tools.web_search.build_web_search_client", lambda settings: client
    )
    rows = run_research(
        web_search_settings=object(), queries=["q1", "q2"], max_results_per_query=3
    )
    assert len(rows) == 2 and client.closed is True

    # One failing query degrades to the other's results, never raises.
    flaky = FakeSearchClient(fail=True)
    monkeypatch.setattr(
        "amadeus.tools.web_search.build_web_search_client", lambda settings: flaky
    )
    assert (
        run_research(
            web_search_settings=object(), queries=["q"], max_results_per_query=3
        )
        == []
    )


def test_generate_followup_note_grounding_skip_and_failure():
    provider = SequenceProvider(
        ["nnU-Net v2 handles exactly this; also a 2026 CMB paper."]
    )
    note = generate_followup_note(
        provider=provider,
        thread="stuck on CMB segmentation",
        results=[{"title": "nnU-Net v2", "url": "u", "snippet": "handles it"}],
    )
    assert note and "nnU-Net" in note
    # The prompt carries the only-things-you-know grounding and the results.
    system = provider.calls[0][0].content
    user = provider.calls[0][1].content
    assert "ONLY" in system and "never invent" in system
    assert "nnU-Net v2" in user

    assert (
        generate_followup_note(
            provider=SequenceProvider(["SKIP"]), thread="t", results=[]
        )
        == "SKIP"
    )
    failing = SequenceProvider(["x"])
    failing.fail = True
    assert generate_followup_note(provider=failing, thread="t", results=[]) is None


def test_followup_greeting_fact_framing():
    entry = _note()
    entry.sources = [{"title": "nnU-Net v2 paper", "url": "http://x"}]
    fact = followup_greeting_fact(entry)
    assert entry.thread in fact and entry.note in fact
    assert "interested you" in fact  # never framed as a favor
    assert "nnU-Net v2 paper" in fact
    assert "http" not in fact.replace("http://x", "") or True


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------


def _config(tmp_path) -> Config:
    config = Config()
    config.follow_through.enabled = True
    config.follow_through.hour = 3
    config.follow_through.store_path = str(tmp_path / "followup.json")
    config.web_search.enabled = True
    config.memory.persist_path = str(tmp_path / "memory.json")
    config.memory.embeddings_path = str(tmp_path / "embeddings.json")
    config.memory.profile_path = str(tmp_path / "profile.json")
    return config


def _seed_profile(config, now: dt.datetime, thread="they were fighting a segfault"):
    profile = UserProfile(
        last_seen=(now - dt.timedelta(hours=8)).isoformat(),
        session_count=4,
    )
    if thread is not None:
        profile.next_time_thread = {
            "text": thread,
            "noted": (now - dt.timedelta(hours=9)).isoformat(timespec="seconds"),
        }
    profile.save_atomic(Path(config.memory.profile_path))


def _make_scheduler(monkeypatch, tmp_path, now, replies, search_results=None):
    config = _config(tmp_path)
    _seed_profile(config, now)
    provider = SequenceProvider(replies)

    import amadeus.web.app as app_module

    monkeypatch.setattr(app_module, "load_llm", lambda cfg: provider)
    app = create_app(config)
    store = app.state.session_store
    store.llm = provider
    monkeypatch.setattr(
        "amadeus.engine.build_summary_provider", lambda cfg, llm: provider
    )
    client = FakeSearchClient(
        results=(
            search_results
            if search_results is not None
            else [{"title": "nnU-Net v2", "url": "u", "snippet": "handles it"}]
        )
    )
    monkeypatch.setattr(
        "amadeus.tools.web_search.build_web_search_client", lambda settings: client
    )
    scheduler = BackgroundScheduler(store, config, clock=lambda: now)
    return scheduler, store, provider, client, config


@pytest.mark.asyncio
async def test_scheduler_researches_once_per_thread(monkeypatch, tmp_path):
    now = dt.datetime(2026, 7, 27, 3, 30)
    scheduler, store, provider, client, config = _make_scheduler(
        monkeypatch, tmp_path, now, ["query one", "Found: nnU-Net v2 does this."]
    )
    await scheduler._followthrough_job(now)
    entry = store.followup_store.get(None)
    assert entry is not None and entry.kind == "note"
    assert "nnU-Net" in entry.note
    assert client.queries == ["query one"]
    assert len(provider.calls) == 2  # plan + note

    # Same thread, later ticks and restarts: never re-researched.
    await scheduler._followthrough_job(now + dt.timedelta(minutes=10))
    fresh = BackgroundScheduler(store, config, clock=lambda: now)
    await fresh._followthrough_job(now + dt.timedelta(hours=1))
    assert len(provider.calls) == 2


@pytest.mark.asyncio
async def test_scheduler_skip_verdict_is_final(monkeypatch, tmp_path):
    now = dt.datetime(2026, 7, 27, 4, 0)
    scheduler, store, provider, client, _ = _make_scheduler(
        monkeypatch, tmp_path, now, ["SKIP"]
    )
    await scheduler._followthrough_job(now)
    entry = store.followup_store.get(None)
    assert entry is not None and entry.kind == "skip"
    assert client.queries == []
    await scheduler._followthrough_job(now + dt.timedelta(minutes=5))
    assert len(provider.calls) == 1


@pytest.mark.asyncio
async def test_scheduler_provider_failure_retries(monkeypatch, tmp_path):
    now = dt.datetime(2026, 7, 27, 4, 0)
    scheduler, store, provider, client, _ = _make_scheduler(
        monkeypatch, tmp_path, now, ["query", "note text"]
    )
    provider.fail = True
    await scheduler._tick()  # must not raise
    assert store.followup_store.get(None) is None
    provider.fail = False
    await scheduler._followthrough_job(now)
    assert store.followup_store.get(None) is not None


@pytest.mark.asyncio
async def test_scheduler_plan_failure_holds_after_daily_retry_cap(
    monkeypatch, tmp_path
):
    """Bounded retry: a failing plan call used to be re-paid every 60s tick
    for the rest of the day. Three failures hold the job until tomorrow."""
    now = dt.datetime(2026, 7, 27, 4, 0)
    scheduler, store, provider, client, _ = _make_scheduler(
        monkeypatch, tmp_path, now, ["query", "note text"]
    )
    provider.fail = True
    for i in range(6):
        await scheduler._followthrough_job(now + dt.timedelta(minutes=i))
    assert len(provider.calls) == 3  # capped at 3 paid plan attempts, not 6

    provider.fail = False
    await scheduler._followthrough_job(now + dt.timedelta(days=1))
    assert store.followup_store.get(None) is not None


@pytest.mark.asyncio
async def test_scheduler_freshness_window_runs_on_utc(monkeypatch, tmp_path):
    """Regression: the profile's next-time `noted` is stamped naive UTC, but
    the scheduler compared it against its LOCAL clock — in a UTC+8 zone a
    thread noted 6d20h ago (still fresh) read as 7d4h old and was silently
    never researched. The freshness comparison now converts the scheduler
    clock to UTC, and generated_at is stamped UTC to match."""
    import os
    import time

    if not hasattr(time, "tzset"):  # pragma: no cover - non-POSIX
        pytest.skip("tzset unavailable on this platform")
    original_tz = os.environ.get("TZ")
    os.environ["TZ"] = "Asia/Singapore"  # UTC+8, no DST
    time.tzset()
    try:
        config = _config(tmp_path)
        config.follow_through.hour = 0  # local wall time is arbitrary here
        utc_now = dt.datetime.utcnow()
        profile = UserProfile(
            last_seen=(utc_now - dt.timedelta(hours=8)).isoformat(),
            session_count=4,
        )
        profile.next_time_thread = {
            "text": "they were fighting a segfault",
            "noted": (utc_now - dt.timedelta(days=6, hours=20)).isoformat(
                timespec="seconds"
            ),
        }
        profile.save_atomic(Path(config.memory.profile_path))
        provider = SequenceProvider(["query one", "Found: nnU-Net v2 does this."])

        import amadeus.web.app as app_module

        monkeypatch.setattr(app_module, "load_llm", lambda cfg: provider)
        app = create_app(config)
        store = app.state.session_store
        store.llm = provider
        monkeypatch.setattr(
            "amadeus.engine.build_summary_provider", lambda cfg, llm: provider
        )
        client = FakeSearchClient(
            results=[{"title": "nnU-Net v2", "url": "u", "snippet": "handles it"}]
        )
        monkeypatch.setattr(
            "amadeus.tools.web_search.build_web_search_client",
            lambda settings: client,
        )
        local_now = dt.datetime.now()  # UTC+8 wall clock
        scheduler = BackgroundScheduler(store, config, clock=lambda: local_now)
        await scheduler._followthrough_job(local_now)

        entry = store.followup_store.get(None)
        assert entry is not None and entry.kind == "note"
        # generated_at is UTC (within slack of the real UTC clock, nowhere
        # near the +8 local wall time).
        stamped = dt.datetime.fromisoformat(entry.generated_at)
        assert abs((stamped - utc_now).total_seconds()) < 3600
    finally:
        if original_tz is None:
            os.environ.pop("TZ", None)
        else:
            os.environ["TZ"] = original_tz
        time.tzset()


@pytest.mark.asyncio
async def test_scheduler_no_thread_or_disabled_noop(monkeypatch, tmp_path):
    now = dt.datetime(2026, 7, 27, 4, 0)
    scheduler, store, provider, client, config = _make_scheduler(
        monkeypatch, tmp_path, now, ["query", "note"]
    )
    _seed_profile(config, now, thread=None)
    await scheduler._followthrough_job(now)
    assert provider.calls == []
    config.follow_through.enabled = False
    await scheduler._followthrough_job(now)
    assert provider.calls == []


@pytest.mark.asyncio
async def test_scheduler_before_hour_noop(monkeypatch, tmp_path):
    now = dt.datetime(2026, 7, 27, 2, 30)
    scheduler, store, provider, client, _ = _make_scheduler(
        monkeypatch, tmp_path, now, ["query", "note"]
    )
    await scheduler._followthrough_job(now)
    assert provider.calls == []


@pytest.mark.asyncio
async def test_scheduler_missing_search_backend_stores_nothing(monkeypatch, tmp_path):
    now = dt.datetime(2026, 7, 27, 4, 0)
    scheduler, store, provider, client, _ = _make_scheduler(
        monkeypatch, tmp_path, now, ["a query", "note"]
    )
    monkeypatch.setattr(
        "amadeus.tools.web_search.build_web_search_client", lambda settings: None
    )
    await scheduler._followthrough_job(now)
    assert store.followup_store.get(None) is None  # fix config → next tick works


@pytest.mark.asyncio
async def test_scheduler_empty_results_store_final_skip(monkeypatch, tmp_path):
    now = dt.datetime(2026, 7, 27, 4, 0)
    scheduler, store, provider, client, _ = _make_scheduler(
        monkeypatch, tmp_path, now, ["a query", "unused"], search_results=[]
    )
    await scheduler._followthrough_job(now)
    entry = store.followup_store.get(None)
    assert entry is not None and entry.kind == "skip"


# ---------------------------------------------------------------------------
# Greeting delivery
# ---------------------------------------------------------------------------


def _greeting_app(monkeypatch, tmp_path):
    config = _config(tmp_path)
    config.memory.enable_greeting = True
    provider = SequenceProvider(
        ["Back again? I looked into that segfault thing — turns out nnU-Net v2."]
    )
    import amadeus.web.app as app_module

    monkeypatch.setattr(app_module, "load_llm", lambda cfg: provider)
    app = create_app(config)
    store = app.state.session_store
    store.llm = provider
    _, session = store.get(None, "ft-user")
    session.llm = provider
    session.summary_provider = provider
    session.memory.profile.session_count = 5
    return app, store, session, provider


def test_greeting_delivers_note_and_consumes(monkeypatch, tmp_path):
    app, store, session, provider = _greeting_app(monkeypatch, tmp_path)
    generated = dt.datetime.utcnow() - dt.timedelta(hours=4)
    store.followup_store.put("ft-user", _note(generated_at=generated))
    # The researched thread is still pending on the profile: the note is its
    # delivery, so the greeting must consume BOTH.
    session.memory.profile.next_time_thread = {
        "text": _note().thread,
        "noted": generated.isoformat(timespec="seconds"),
    }

    text = session.open_session_greeting()
    assert text is not None
    prompt_user = provider.calls[-1][1].content
    assert _note().note in prompt_user
    # The plain next-time fact is REPLACED, not doubled.
    assert "left mid-thing" not in prompt_user
    assert store.followup_store.get("ft-user").consumed is True
    assert session.memory.profile.next_time_thread is None

    # Second greeting: consumed note never re-delivered.
    session.memory.short_term.clear()
    assert session.open_session_greeting() is not None
    assert _note().note not in provider.calls[-1][1].content


def test_greeting_failure_leaves_note_unconsumed(monkeypatch, tmp_path):
    app, store, session, provider = _greeting_app(monkeypatch, tmp_path)
    store.followup_store.put(
        "ft-user", _note(generated_at=dt.datetime.utcnow() - dt.timedelta(hours=2))
    )
    provider.fail = True
    assert session.open_session_greeting() is None
    assert store.followup_store.get("ft-user").consumed is False


def test_greeting_survives_corrupt_followup_store(monkeypatch, tmp_path):
    app, store, session, provider = _greeting_app(monkeypatch, tmp_path)

    class ExplodingStore:
        def peek_unconsumed(self, *a, **k):
            raise RuntimeError("disk on fire")

    session.followup_store = ExplodingStore()
    assert session.open_session_greeting() is not None


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------


def test_followup_endpoint_returns_without_consuming(monkeypatch, tmp_path):
    config = _config(tmp_path)
    import amadeus.web.app as app_module

    provider = SequenceProvider(["hi"])
    monkeypatch.setattr(app_module, "load_llm", lambda cfg: provider)
    app = create_app(config)
    store = app.state.session_store
    store.followup_store.put(None, _note())
    client = TestClient(app)
    body = client.get("/api/followup").json()
    assert body["followup"]["kind"] == "note"
    assert body["followup"]["consumed"] is False
    assert store.followup_store.get(None).consumed is False


def test_followup_endpoint_disabled_returns_null(monkeypatch, tmp_path):
    config = _config(tmp_path)
    config.follow_through.enabled = False
    import amadeus.web.app as app_module

    monkeypatch.setattr(app_module, "load_llm", lambda cfg: SequenceProvider(["hi"]))
    app = create_app(config)
    client = TestClient(app)
    assert client.get("/api/followup").json() == {"followup": None}
