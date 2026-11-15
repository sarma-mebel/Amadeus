"""Calendar feed: ICS parsing (unfolding, escaping, DATE/DATE-TIME, RRULE
expansion), cache TTL/staleness, and the daily-brief fact integration."""

from __future__ import annotations

import datetime as dt

import pytest

from amadeus import telemetry
from amadeus.calendar_feed import (
    CalendarCache,
    CalendarEvent,
    expand_events,
    is_configured,
    parse_ics,
    refresh_cache,
    upcoming_events,
)
from amadeus.config import CalendarSettings
from amadeus.brief import collect_brief_facts


@pytest.fixture(autouse=True)
def _reset_telemetry():
    telemetry.reset()
    yield
    telemetry.reset()


# ---------------------------------------------------------------------------
# ICS parsing
# ---------------------------------------------------------------------------


def test_parse_simple_vevent_date_time():
    ics = (
        "BEGIN:VCALENDAR\r\n"
        "BEGIN:VEVENT\r\n"
        "SUMMARY:Lab meeting\r\n"
        "DTSTART:20260810T100000\r\n"
        "DTEND:20260810T110000\r\n"
        "END:VEVENT\r\n"
        "END:VCALENDAR\r\n"
    )
    events = parse_ics(ics)
    assert len(events) == 1
    e = events[0]
    assert e.summary == "Lab meeting"
    assert e.start == dt.datetime(2026, 8, 10, 10, 0, 0)
    assert e.end == dt.datetime(2026, 8, 10, 11, 0, 0)
    assert e.all_day is False


def test_parse_folded_lines_rfc5545():
    # A folded SUMMARY line: continuation starts with a single space.
    ics = (
        "BEGIN:VEVENT\r\n"
        "SUMMARY:Lab meeting about the very long\r\n"
        " project title that keeps going\r\n"
        "DTSTART:20260810T100000\r\n"
        "END:VEVENT\r\n"
    )
    events = parse_ics(ics)
    assert len(events) == 1
    assert (
        events[0].summary
        == "Lab meeting about the very longproject title that keeps going"
    )


def test_parse_escaped_summary():
    ics = (
        "BEGIN:VEVENT\r\n"
        "SUMMARY:Meeting\\, part 1\\; agenda\\nfollow-up\r\n"
        "DTSTART:20260810T100000\r\n"
        "END:VEVENT\r\n"
    )
    events = parse_ics(ics)
    assert events[0].summary == "Meeting, part 1; agenda\nfollow-up"


def test_parse_all_day_event():
    ics = (
        "BEGIN:VEVENT\r\n"
        "SUMMARY:Conference\r\n"
        "DTSTART;VALUE=DATE:20260812\r\n"
        "DTEND;VALUE=DATE:20260813\r\n"
        "END:VEVENT\r\n"
    )
    events = parse_ics(ics)
    assert len(events) == 1
    assert events[0].all_day is True
    assert events[0].start == dt.datetime(2026, 8, 12)


def test_parse_bare_date_without_value_param_is_all_day():
    ics = "BEGIN:VEVENT\r\nSUMMARY:Bday\r\nDTSTART:20260825\r\nEND:VEVENT\r\n"
    events = parse_ics(ics)
    assert events[0].all_day is True
    assert events[0].start.date() == dt.date(2026, 8, 25)


def test_parse_utc_datetime_converted_to_local():
    ics = "BEGIN:VEVENT\r\nSUMMARY:UTC event\r\nDTSTART:20260810T120000Z\r\nEND:VEVENT\r\n"
    events = parse_ics(ics)
    assert len(events) == 1
    # Converted via astimezone(): must equal what the local system computes
    # from the same UTC instant, whatever the test machine's zone is.
    expected = (
        dt.datetime(2026, 8, 10, 12, 0, 0, tzinfo=dt.timezone.utc)
        .astimezone()
        .replace(tzinfo=None)
    )
    assert events[0].start == expected


def test_parse_tzid_treated_as_local_and_counted():
    ics = (
        "BEGIN:VEVENT\r\n"
        "SUMMARY:TZID event\r\n"
        "DTSTART;TZID=America/New_York:20260810T090000\r\n"
        "END:VEVENT\r\n"
    )
    events = parse_ics(ics)
    assert len(events) == 1
    assert events[0].start == dt.datetime(2026, 8, 10, 9, 0, 0)
    assert telemetry.snapshot().get("calendar.tzid_assumed_local") == 1


def test_parse_ignores_vtodo_and_vjournal():
    ics = (
        "BEGIN:VTODO\r\n"
        "SUMMARY:Buy milk\r\n"
        "DTSTART:20260810T100000\r\n"
        "END:VTODO\r\n"
        "BEGIN:VJOURNAL\r\n"
        "SUMMARY:Diary note\r\n"
        "DTSTART:20260810T100000\r\n"
        "END:VJOURNAL\r\n"
        "BEGIN:VEVENT\r\n"
        "SUMMARY:Real event\r\n"
        "DTSTART:20260810T100000\r\n"
        "END:VEVENT\r\n"
    )
    events = parse_ics(ics)
    assert len(events) == 1
    assert events[0].summary == "Real event"


def test_parse_missing_summary_or_dtstart_skips_event_and_counts_failure():
    ics = "BEGIN:VEVENT\r\nDTSTART:20260810T100000\r\nEND:VEVENT\r\n"
    events = parse_ics(ics)
    assert events == []
    assert telemetry.snapshot().get("calendar.parse_failures") == 1


def test_parse_garbage_feed_returns_empty_and_counts_failure(monkeypatch):
    import amadeus.calendar_feed as cf

    def boom(_raw):
        raise ValueError("kaboom")

    monkeypatch.setattr(cf, "_unfold_lines", boom)
    assert cf.parse_ics("anything") == []
    assert telemetry.snapshot().get("calendar.parse_failures") == 1


# ---------------------------------------------------------------------------
# RRULE expansion
# ---------------------------------------------------------------------------


def test_daily_rrule_expansion_within_window():
    ics = (
        "BEGIN:VEVENT\r\n"
        "SUMMARY:Standup\r\n"
        "DTSTART:20260810T090000\r\n"
        "RRULE:FREQ=DAILY\r\n"
        "END:VEVENT\r\n"
    )
    raw = parse_ics(ics)
    window_start = dt.datetime(2026, 8, 10, 0, 0)
    window_end = dt.datetime(2026, 8, 13, 0, 0)
    events = expand_events(raw, window_start, window_end)
    starts = sorted(e.start for e in events)
    assert starts == [
        dt.datetime(2026, 8, 10, 9, 0),
        dt.datetime(2026, 8, 11, 9, 0),
        dt.datetime(2026, 8, 12, 9, 0),
    ]


def test_weekly_rrule_with_byday_expansion():
    ics = (
        "BEGIN:VEVENT\r\n"
        "SUMMARY:Lab meeting\r\n"
        "DTSTART:20260803T100000\r\n"  # a Monday
        "RRULE:FREQ=WEEKLY;BYDAY=MO,WE\r\n"
        "END:VEVENT\r\n"
    )
    raw = parse_ics(ics)
    window_start = dt.datetime(2026, 8, 3, 0, 0)
    window_end = dt.datetime(2026, 8, 16, 0, 0)
    events = expand_events(raw, window_start, window_end)
    starts = sorted(e.start for e in events)
    assert starts == [
        dt.datetime(2026, 8, 3, 10, 0),  # Mon
        dt.datetime(2026, 8, 5, 10, 0),  # Wed
        dt.datetime(2026, 8, 10, 10, 0),  # Mon
        dt.datetime(2026, 8, 12, 10, 0),  # Wed
    ]


def test_rrule_until_bounds_expansion():
    ics = (
        "BEGIN:VEVENT\r\n"
        "SUMMARY:Standup\r\n"
        "DTSTART:20260810T090000\r\n"
        "RRULE:FREQ=DAILY;UNTIL=20260811T090000\r\n"
        "END:VEVENT\r\n"
    )
    raw = parse_ics(ics)
    events = expand_events(raw, dt.datetime(2026, 8, 10), dt.datetime(2026, 8, 20))
    starts = sorted(e.start for e in events)
    assert starts == [dt.datetime(2026, 8, 10, 9, 0), dt.datetime(2026, 8, 11, 9, 0)]


def test_rrule_count_bounds_expansion():
    ics = (
        "BEGIN:VEVENT\r\n"
        "SUMMARY:Standup\r\n"
        "DTSTART:20260810T090000\r\n"
        "RRULE:FREQ=DAILY;COUNT=3\r\n"
        "END:VEVENT\r\n"
    )
    raw = parse_ics(ics)
    events = expand_events(raw, dt.datetime(2026, 8, 10), dt.datetime(2026, 8, 30))
    assert len(events) == 3
    assert events[-1].start == dt.datetime(2026, 8, 12, 9, 0)


def test_unsupported_rrule_freq_skipped_and_counted():
    ics = (
        "BEGIN:VEVENT\r\n"
        "SUMMARY:Monthly thing\r\n"
        "DTSTART:20260810T090000\r\n"
        "RRULE:FREQ=MONTHLY\r\n"
        "END:VEVENT\r\n"
    )
    raw = parse_ics(ics)
    events = expand_events(raw, dt.datetime(2026, 8, 1), dt.datetime(2026, 9, 30))
    assert events == []
    assert telemetry.snapshot().get("calendar.rrule_skipped") == 1


def test_unsupported_rrule_extra_field_skipped_and_counted():
    ics = (
        "BEGIN:VEVENT\r\n"
        "SUMMARY:Weekday thing\r\n"
        "DTSTART:20260810T090000\r\n"
        "RRULE:FREQ=WEEKLY;BYSETPOS=1\r\n"
        "END:VEVENT\r\n"
    )
    raw = parse_ics(ics)
    events = expand_events(raw, dt.datetime(2026, 8, 1), dt.datetime(2026, 9, 30))
    assert events == []
    assert telemetry.snapshot().get("calendar.rrule_skipped") == 1


# ---------------------------------------------------------------------------
# Lookahead windowing / sorting / cap
# ---------------------------------------------------------------------------


def test_upcoming_events_sorted_and_capped(tmp_path):
    cache = CalendarCache(str(tmp_path / "cal.json"))
    now = dt.datetime(2026, 8, 10, 8, 0)
    events = [
        CalendarEvent(
            summary=f"Event {i}",
            start=now + dt.timedelta(hours=i),
            end=None,
            all_day=False,
        )
        for i in reversed(range(12))  # out of order on purpose
    ]
    cache.save(now, events)
    settings = CalendarSettings(ics_url="http://example.com/cal.ics", max_events=5)
    result = upcoming_events(settings, now, cache_path=str(tmp_path / "cal.json"))
    assert len(result) == 5
    assert [e.summary for e in result] == [f"Event {i}" for i in range(5)]
    assert telemetry.snapshot().get("calendar.events_served") == 5


def test_upcoming_events_excludes_past_and_outside_lookahead(tmp_path):
    cache_path = str(tmp_path / "cal.json")
    cache = CalendarCache(cache_path)
    now = dt.datetime(2026, 8, 10, 12, 0)
    events = [
        CalendarEvent(
            summary="Already ended",
            start=now - dt.timedelta(hours=2),
            end=now - dt.timedelta(hours=1),
            all_day=False,
        ),
        CalendarEvent(
            summary="Still running",
            start=now - dt.timedelta(minutes=30),
            end=now + dt.timedelta(minutes=30),
            all_day=False,
        ),
        CalendarEvent(
            summary="Later today",
            start=now + dt.timedelta(hours=2),
            end=None,
            all_day=False,
        ),
        CalendarEvent(
            summary="Way out",
            start=now + dt.timedelta(days=5),
            end=None,
            all_day=False,
        ),
    ]
    cache.save(now, events)
    settings = CalendarSettings(ics_url="http://x", lookahead_hours=36)
    result = upcoming_events(settings, now, cache_path=cache_path)
    summaries = {e.summary for e in result}
    assert summaries == {"Still running", "Later today"}


def test_upcoming_events_keeps_all_day_events_for_today(tmp_path):
    cache_path = str(tmp_path / "cal.json")
    cache = CalendarCache(cache_path)
    now = dt.datetime(2026, 8, 10, 15, 0)  # afternoon
    events = [
        CalendarEvent(
            summary="Her birthday",
            start=dt.datetime(2026, 8, 10),
            end=None,
            all_day=True,
        )
    ]
    cache.save(now, events)
    settings = CalendarSettings(ics_url="http://x")
    result = upcoming_events(settings, now, cache_path=cache_path)
    assert len(result) == 1
    assert result[0].human_line() == "all day: Her birthday"


def test_upcoming_events_not_configured_is_empty(tmp_path):
    settings = CalendarSettings()  # no url, no path
    assert upcoming_events(settings, dt.datetime.now()) == []


def test_is_configured_true_with_either_source():
    assert is_configured(CalendarSettings(ics_url="http://x")) is True
    assert is_configured(CalendarSettings(ics_path="/tmp/x.ics")) is True
    assert is_configured(CalendarSettings()) is False


def test_human_line_formats():
    timed = CalendarEvent(
        summary="lab meeting",
        start=dt.datetime(2026, 8, 10, 10, 0),
        end=None,
        all_day=False,
    )
    assert timed.human_line() == "10:00 lab meeting"
    allday = CalendarEvent(
        summary="conference", start=dt.datetime(2026, 8, 10), end=None, all_day=True
    )
    assert allday.human_line() == "all day: conference"


# ---------------------------------------------------------------------------
# Cache TTL + stale-on-failure (urllib mocked)
# ---------------------------------------------------------------------------


_SIMPLE_ICS = (
    "BEGIN:VEVENT\r\nSUMMARY:Lab meeting\r\nDTSTART:20260810T100000\r\nEND:VEVENT\r\n"
)


class _FakeResponse:
    def __init__(self, body: bytes) -> None:
        self._body = body

    def read(self, _n=-1):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_refresh_cache_fetches_when_cache_missing(monkeypatch, tmp_path):
    calls = []

    def fake_urlopen(req, timeout=None):
        calls.append(req)
        return _FakeResponse(_SIMPLE_ICS.encode("utf-8"))

    monkeypatch.setattr("amadeus.calendar_feed.urllib.request.urlopen", fake_urlopen)
    cache_path = str(tmp_path / "cal.json")
    cache = CalendarCache(cache_path)
    settings = CalendarSettings(ics_url="http://example.com/cal.ics")
    now = dt.datetime(2026, 8, 10, 8, 0)
    refresh_cache(settings, cache, now)
    assert len(calls) == 1
    fetched_at, events = cache.load()
    assert fetched_at == now
    assert len(events) == 1
    assert events[0].summary == "Lab meeting"


def test_refresh_cache_skips_fetch_within_ttl(monkeypatch, tmp_path):
    calls = []

    def fake_urlopen(req, timeout=None):
        calls.append(req)
        return _FakeResponse(_SIMPLE_ICS.encode("utf-8"))

    monkeypatch.setattr("amadeus.calendar_feed.urllib.request.urlopen", fake_urlopen)
    cache_path = str(tmp_path / "cal.json")
    cache = CalendarCache(cache_path)
    settings = CalendarSettings(
        ics_url="http://example.com/cal.ics", refresh_ttl_hours=6.0
    )
    now = dt.datetime(2026, 8, 10, 8, 0)
    refresh_cache(settings, cache, now)
    assert len(calls) == 1
    # Within TTL: no refetch.
    refresh_cache(settings, cache, now + dt.timedelta(hours=1))
    assert len(calls) == 1
    # Past TTL: refetches.
    refresh_cache(settings, cache, now + dt.timedelta(hours=7))
    assert len(calls) == 2


def test_refresh_cache_serves_stale_on_fetch_failure(monkeypatch, tmp_path):
    def fake_urlopen(req, timeout=None):
        return _FakeResponse(_SIMPLE_ICS.encode("utf-8"))

    monkeypatch.setattr("amadeus.calendar_feed.urllib.request.urlopen", fake_urlopen)
    cache_path = str(tmp_path / "cal.json")
    cache = CalendarCache(cache_path)
    settings = CalendarSettings(
        ics_url="http://example.com/cal.ics", refresh_ttl_hours=1.0
    )
    now = dt.datetime(2026, 8, 10, 8, 0)
    refresh_cache(settings, cache, now)
    fetched_at_before, events_before = cache.load()
    assert len(events_before) == 1

    def failing_urlopen(req, timeout=None):
        raise OSError("network down")

    monkeypatch.setattr("amadeus.calendar_feed.urllib.request.urlopen", failing_urlopen)
    later = now + dt.timedelta(hours=2)  # past TTL, forces a refetch attempt
    refresh_cache(settings, cache, later)
    fetched_at_after, events_after = cache.load()
    # Stale cache is left untouched (not cleared, not "fetched_at" bumped).
    assert fetched_at_after == fetched_at_before
    assert len(events_after) == 1
    assert telemetry.snapshot().get("calendar.fetch_failures") == 1


def test_refresh_cache_no_cache_and_fetch_fails_serves_empty(monkeypatch, tmp_path):
    def failing_urlopen(req, timeout=None):
        raise OSError("network down")

    monkeypatch.setattr("amadeus.calendar_feed.urllib.request.urlopen", failing_urlopen)
    cache_path = str(tmp_path / "cal.json")
    cache = CalendarCache(cache_path)
    settings = CalendarSettings(ics_url="http://example.com/cal.ics")
    now = dt.datetime(2026, 8, 10, 8, 0)
    refresh_cache(settings, cache, now)
    assert cache.load() == (None, [])
    result = upcoming_events(settings, now, cache_path=cache_path)
    assert result == []
    assert telemetry.snapshot().get("calendar.fetch_failures") == 1


def test_refresh_cache_reads_local_ics_path(tmp_path):
    ics_file = tmp_path / "personal.ics"
    ics_file.write_text(_SIMPLE_ICS, encoding="utf-8")
    cache_path = str(tmp_path / "cal.json")
    cache = CalendarCache(cache_path)
    settings = CalendarSettings(ics_path=str(ics_file))
    now = dt.datetime(2026, 8, 10, 8, 0)
    refresh_cache(settings, cache, now)
    _fetched_at, events = cache.load()
    assert len(events) == 1
    assert events[0].summary == "Lab meeting"


def test_refresh_cache_missing_local_path_counts_failure(tmp_path):
    cache_path = str(tmp_path / "cal.json")
    cache = CalendarCache(cache_path)
    settings = CalendarSettings(ics_path=str(tmp_path / "does_not_exist.ics"))
    now = dt.datetime(2026, 8, 10, 8, 0)
    refresh_cache(settings, cache, now)
    assert cache.load() == (None, [])
    assert telemetry.snapshot().get("calendar.fetch_failures") == 1


def test_refresh_cache_noop_when_not_configured(monkeypatch, tmp_path):
    def fake_urlopen(req, timeout=None):  # pragma: no cover - must not be called
        raise AssertionError("should not fetch when unconfigured")

    monkeypatch.setattr("amadeus.calendar_feed.urllib.request.urlopen", fake_urlopen)
    cache = CalendarCache(str(tmp_path / "cal.json"))
    settings = CalendarSettings()
    refresh_cache(settings, cache, dt.datetime(2026, 8, 10, 8, 0))
    assert cache.load() == (None, [])


# ---------------------------------------------------------------------------
# Brief integration
# ---------------------------------------------------------------------------


def test_collect_brief_facts_includes_calendar_events():
    now = dt.datetime(2026, 8, 10, 6, 0)
    events = [
        CalendarEvent(
            summary="lab meeting",
            start=dt.datetime(2026, 8, 10, 10, 0),
            end=None,
            all_day=False,
        ),
        CalendarEvent(
            summary="dentist",
            start=dt.datetime(2026, 8, 10, 14, 30),
            end=None,
            all_day=False,
        ),
    ]
    facts = collect_brief_facts(
        profile=None,
        pending_reminders=[],
        arxiv_hits=[],
        now=now,
        calendar_events=events,
    )
    joined = "\n".join(facts)
    assert "Today's calendar:" in joined
    assert "10:00 lab meeting" in joined
    assert "14:30 dentist" in joined


def test_collect_brief_facts_omits_calendar_block_when_absent():
    now = dt.datetime(2026, 8, 10, 6, 0)
    facts_no_arg = collect_brief_facts(
        profile=None, pending_reminders=[], arxiv_hits=[], now=now
    )
    facts_empty_list = collect_brief_facts(
        profile=None,
        pending_reminders=[],
        arxiv_hits=[],
        now=now,
        calendar_events=[],
    )
    assert facts_no_arg == facts_empty_list
    assert not any("Today's calendar" in f for f in facts_no_arg)


# ---------------------------------------------------------------------------
# Scheduler integration (calendar folded into the daily brief job)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_scheduler_brief_job_folds_in_calendar_events(monkeypatch, tmp_path):
    pytest.importorskip("fastapi")
    from amadeus.config import Config
    from amadeus.memory.profile import UserProfile
    from amadeus.providers.llm.base import ChatMessage
    from amadeus.web.app import create_app
    from amadeus.web.scheduler import BackgroundScheduler

    class RecordingProvider:
        name = "recording"

        def __init__(self):
            self.calls: list = []

        def generate(self, messages, **kwargs):
            self.calls.append(list(messages))
            return "Lab meeting at 10, don't forget."

    now = dt.datetime(2026, 8, 10, 6, 30)
    config = Config()
    config.daily_brief.enabled = True
    config.daily_brief.hour = 6
    config.daily_brief.store_path = str(tmp_path / "daily_brief.json")
    config.memory.persist_path = str(tmp_path / "memory.json")
    config.memory.embeddings_path = str(tmp_path / "embeddings.json")
    config.memory.profile_path = str(tmp_path / "profile.json")
    ics_file = tmp_path / "cal.ics"
    ics_file.write_text(_SIMPLE_ICS, encoding="utf-8")
    config.calendar.ics_path = str(ics_file)
    config.calendar.cache_path = str(tmp_path / "calendar_cache.json")

    profile = UserProfile(
        open_threads=["finish the draft"],
        last_episode="talked about the draft",
        last_seen=(now - dt.timedelta(days=1)).isoformat(),
        session_count=3,
    )
    from pathlib import Path

    profile.save_atomic(Path(config.memory.profile_path))

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
    await scheduler._daily_brief_job(now)

    assert len(recorder.calls) == 1
    user_prompt = recorder.calls[0][-1].content
    assert "Today's calendar:" in user_prompt
    assert "10:00 Lab meeting" in user_prompt
    brief = store.brief_store.get(None)
    assert brief is not None


@pytest.mark.asyncio
async def test_scheduler_brief_job_unaffected_when_calendar_unconfigured(
    monkeypatch, tmp_path
):
    """No calendar configured -> brief generation behaves exactly as before
    this feature existed (no calendar fact line, no crash)."""
    pytest.importorskip("fastapi")
    from amadeus.config import Config
    from amadeus.memory.profile import UserProfile
    from amadeus.providers.llm.base import ChatMessage
    from amadeus.web.app import create_app
    from amadeus.web.scheduler import BackgroundScheduler

    class RecordingProvider:
        name = "recording"

        def __init__(self):
            self.calls: list = []

        def generate(self, messages, **kwargs):
            self.calls.append(list(messages))
            return "Finish the draft today."

    now = dt.datetime(2026, 8, 10, 6, 30)
    config = Config()
    config.daily_brief.enabled = True
    config.daily_brief.hour = 6
    config.daily_brief.store_path = str(tmp_path / "daily_brief.json")
    config.memory.persist_path = str(tmp_path / "memory.json")
    config.memory.embeddings_path = str(tmp_path / "embeddings.json")
    config.memory.profile_path = str(tmp_path / "profile.json")
    # config.calendar left at defaults: not configured.

    profile = UserProfile(
        open_threads=["finish the draft"],
        last_episode="talked about the draft",
        last_seen=(now - dt.timedelta(days=1)).isoformat(),
        session_count=3,
    )
    from pathlib import Path

    profile.save_atomic(Path(config.memory.profile_path))

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
    await scheduler._daily_brief_job(now)

    assert len(recorder.calls) == 1
    user_prompt = recorder.calls[0][-1].content
    assert "Today's calendar" not in user_prompt
    assert telemetry.snapshot().get("calendar.fetch_failures", 0) == 0


@pytest.mark.asyncio
async def test_scheduler_brief_job_survives_calendar_failure(monkeypatch, tmp_path):
    """A blown-up calendar read must not prevent the brief from generating."""
    pytest.importorskip("fastapi")
    from amadeus.config import Config
    from amadeus.memory.profile import UserProfile
    from amadeus.web.app import create_app
    from amadeus.web.scheduler import BackgroundScheduler

    class RecordingProvider:
        name = "recording"

        def __init__(self):
            self.calls: list = []

        def generate(self, messages, **kwargs):
            self.calls.append(list(messages))
            return "Finish the draft today."

    now = dt.datetime(2026, 8, 10, 6, 30)
    config = Config()
    config.daily_brief.enabled = True
    config.daily_brief.hour = 6
    config.daily_brief.store_path = str(tmp_path / "daily_brief.json")
    config.memory.persist_path = str(tmp_path / "memory.json")
    config.memory.embeddings_path = str(tmp_path / "embeddings.json")
    config.memory.profile_path = str(tmp_path / "profile.json")
    config.calendar.ics_url = "http://example.com/cal.ics"
    config.calendar.cache_path = str(tmp_path / "calendar_cache.json")

    profile = UserProfile(
        open_threads=["finish the draft"],
        last_episode="talked about the draft",
        last_seen=(now - dt.timedelta(days=1)).isoformat(),
        session_count=3,
    )
    from pathlib import Path

    profile.save_atomic(Path(config.memory.profile_path))

    recorder = RecordingProvider()
    import amadeus.web.app as app_module

    monkeypatch.setattr(app_module, "load_llm", lambda cfg: recorder)
    app = create_app(config)
    store = app.state.session_store
    store.llm = recorder
    monkeypatch.setattr(
        "amadeus.engine.build_summary_provider", lambda cfg, llm: recorder
    )

    def boom(*args, **kwargs):
        raise RuntimeError("cache exploded")

    monkeypatch.setattr("amadeus.web.scheduler.refresh_cache", boom)

    scheduler = BackgroundScheduler(store, config, clock=lambda: now)
    await scheduler._daily_brief_job(now)  # must not raise

    assert len(recorder.calls) == 1
    brief = store.brief_store.get(None)
    assert brief is not None
