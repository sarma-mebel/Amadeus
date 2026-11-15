"""Async reach-out/reminder delivery via Discord DM when no client is
around: the pure decision function's truth table, urllib-mocked delivery,
the client-liveness snapshot file, and scheduler job wiring (pop happens
only when delivery will actually be attempted)."""

from __future__ import annotations

import datetime as dt
import json
import urllib.error

import pytest

pytest.importorskip("fastapi")

from amadeus import telemetry
from amadeus.async_notify import (
    deliver,
    last_seen_path,
    read_last_client_poll,
    read_last_delivery_date,
    record_client_poll,
    record_delivery,
    send_discord_dm,
    should_attempt_delivery,
)
from amadeus.config import Config
from amadeus.reachout import ReachOut, ReachOutStore
from amadeus.reminders import ReminderStore
from amadeus.web.app import create_app
from amadeus.web.scheduler import BackgroundScheduler

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def test_async_notify_settings_defaults():
    config = Config()
    assert config.async_notify.enabled is False
    assert config.async_notify.channel == "discord"
    assert config.async_notify.discord_user_id == ""
    assert config.async_notify.min_client_stale_minutes == 30
    assert config.async_notify.quiet_hours_start == 22
    assert config.async_notify.quiet_hours_end == 8
    assert config.async_notify.max_per_day == 1


def test_async_notify_reuses_discord_token_env_fallback(monkeypatch):
    monkeypatch.setenv("DISCORD_BOT_TOKEN", "from-env")
    config = Config.from_dict(
        {"async_notify": {"enabled": True, "discord_user_id": "123"}}
    )
    assert config.async_notify.enabled is True
    assert config.discord.token == "from-env"


# ---------------------------------------------------------------------------
# should_attempt_delivery truth table
# ---------------------------------------------------------------------------


def _config(**overrides) -> Config:
    config = Config()
    config.async_notify.enabled = True
    config.async_notify.discord_user_id = "999"
    # A usable bot token is part of the deliverable baseline now that
    # should_attempt_delivery refuses without one (pop-before-token-check
    # used to destroy the day's line when the token was missing).
    config.discord.token = "tok"
    for key, value in overrides.items():
        setattr(config.async_notify, key, value)
    return config


def test_disabled_never_attempts():
    config = Config()
    config.async_notify.enabled = False
    config.async_notify.discord_user_id = "999"
    now = dt.datetime(2026, 8, 3, 12, 0)
    assert (
        should_attempt_delivery(
            config=config,
            last_client_poll=None,
            now=now,
            asleep=False,
            last_delivery_date=None,
        )
        is False
    )


def test_no_recipient_never_attempts():
    config = _config(discord_user_id="")
    now = dt.datetime(2026, 8, 3, 12, 0)
    assert (
        should_attempt_delivery(
            config=config,
            last_client_poll=None,
            now=now,
            asleep=False,
            last_delivery_date=None,
        )
        is False
    )


def test_no_token_never_attempts():
    """Enabled-with-no-token must refuse BEFORE the caller pops anything:
    deliver() can never succeed without a token, and the old pop-then-fail
    order permanently destroyed the day's reach-out/reminders."""
    config = _config()
    config.discord.token = None
    now = dt.datetime(2026, 8, 3, 12, 0)
    assert (
        should_attempt_delivery(
            config=config,
            last_client_poll=None,
            now=now,
            asleep=False,
            last_delivery_date=None,
        )
        is False
    )


def test_fresh_client_poll_blocks_delivery():
    config = _config(min_client_stale_minutes=30)
    now = dt.datetime(2026, 8, 3, 12, 0)
    fresh_poll = now - dt.timedelta(minutes=5)
    assert (
        should_attempt_delivery(
            config=config,
            last_client_poll=fresh_poll,
            now=now,
            asleep=False,
            last_delivery_date=None,
        )
        is False
    )


def test_stale_client_poll_allows_delivery():
    config = _config(min_client_stale_minutes=30)
    now = dt.datetime(2026, 8, 3, 12, 0)
    stale_poll = now - dt.timedelta(minutes=45)
    assert (
        should_attempt_delivery(
            config=config,
            last_client_poll=stale_poll,
            now=now,
            asleep=False,
            last_delivery_date=None,
        )
        is True
    )


def test_no_poll_ever_allows_delivery():
    """No client_last_seen file at all (never polled, or a brand-new
    install) must not block delivery forever."""
    config = _config()
    now = dt.datetime(2026, 8, 3, 12, 0)
    assert (
        should_attempt_delivery(
            config=config,
            last_client_poll=None,
            now=now,
            asleep=False,
            last_delivery_date=None,
        )
        is True
    )


def test_asleep_blocks_delivery():
    config = _config()
    now = dt.datetime(2026, 8, 3, 3, 0)
    assert (
        should_attempt_delivery(
            config=config,
            last_client_poll=None,
            now=now,
            asleep=True,
            last_delivery_date=None,
        )
        is False
    )


def test_quiet_hours_wraparound_blocks_delivery():
    config = _config(quiet_hours_start=22, quiet_hours_end=8)
    # Inside the wrap (23:00) and inside the low end (07:00).
    for hour in (23, 7, 0):
        now = dt.datetime(2026, 8, 3, hour, 0)
        assert (
            should_attempt_delivery(
                config=config,
                last_client_poll=None,
                now=now,
                asleep=False,
                last_delivery_date=None,
            )
            is False
        ), hour
    # Boundary: start inclusive, end exclusive (mirrors sleep_schedule).
    assert (
        should_attempt_delivery(
            config=config,
            last_client_poll=None,
            now=dt.datetime(2026, 8, 3, 22, 0),
            asleep=False,
            last_delivery_date=None,
        )
        is False
    )
    assert (
        should_attempt_delivery(
            config=config,
            last_client_poll=None,
            now=dt.datetime(2026, 8, 3, 8, 0),
            asleep=False,
            last_delivery_date=None,
        )
        is True
    )


def test_quiet_hours_non_wrapping_window():
    config = _config(quiet_hours_start=1, quiet_hours_end=6)
    assert (
        should_attempt_delivery(
            config=config,
            last_client_poll=None,
            now=dt.datetime(2026, 8, 3, 3, 0),
            asleep=False,
            last_delivery_date=None,
        )
        is False
    )
    assert (
        should_attempt_delivery(
            config=config,
            last_client_poll=None,
            now=dt.datetime(2026, 8, 3, 12, 0),
            asleep=False,
            last_delivery_date=None,
        )
        is True
    )


def test_quiet_hours_disabled_when_start_equals_end():
    config = _config(quiet_hours_start=5, quiet_hours_end=5)
    now = dt.datetime(2026, 8, 3, 5, 0)
    assert (
        should_attempt_delivery(
            config=config,
            last_client_poll=None,
            now=now,
            asleep=False,
            last_delivery_date=None,
        )
        is True
    )


def test_should_attempt_delivery_daily_cap_blocks_same_day():
    config = _config(max_per_day=1)
    now = dt.datetime(2026, 8, 3, 12, 0)
    assert (
        should_attempt_delivery(
            config=config,
            last_client_poll=None,
            now=now,
            asleep=False,
            last_delivery_date=now.date(),
        )
        is False
    )
    # A prior day's delivery does not carry the cap forward.
    assert (
        should_attempt_delivery(
            config=config,
            last_client_poll=None,
            now=now,
            asleep=False,
            last_delivery_date=now.date() - dt.timedelta(days=1),
        )
        is True
    )


def test_max_per_day_zero_disables():
    config = _config(max_per_day=0)
    now = dt.datetime(2026, 8, 3, 12, 0)
    assert (
        should_attempt_delivery(
            config=config,
            last_client_poll=None,
            now=now,
            asleep=False,
            last_delivery_date=None,
        )
        is False
    )


# ---------------------------------------------------------------------------
# Delivery — urllib mocked, never a live network call
# ---------------------------------------------------------------------------


class _FakeResponse:
    def __init__(self, payload: dict) -> None:
        self._body = json.dumps(payload).encode("utf-8")

    def read(self) -> bytes:
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_send_discord_dm_success(monkeypatch):
    calls = []

    def fake_urlopen(request, timeout=None):
        calls.append(request.full_url)
        if request.full_url.endswith("/channels"):
            return _FakeResponse({"id": "chan1"})
        return _FakeResponse({"id": "msg1"})

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    ok = send_discord_dm(token="tok", recipient_id="123", content="hello")
    assert ok is True
    assert len(calls) == 2
    assert telemetry.snapshot().get("notify.delivery_failures", 0) == 0


def test_send_discord_dm_4xx_counts_failure_no_raise(monkeypatch):
    telemetry.snapshot()  # ensure module imported/initialized

    def fake_urlopen(request, timeout=None):
        raise urllib.error.HTTPError(request.full_url, 401, "Unauthorized", {}, None)

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    before = telemetry.snapshot().get("notify.delivery_failures", 0)
    ok = send_discord_dm(token="bad-token", recipient_id="123", content="hello")
    assert ok is False
    after = telemetry.snapshot().get("notify.delivery_failures", 0)
    assert after == before + 1


def test_send_discord_dm_timeout_counts_failure_no_raise(monkeypatch):
    def fake_urlopen(request, timeout=None):
        raise TimeoutError("timed out")

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    before = telemetry.snapshot().get("notify.delivery_failures", 0)
    ok = send_discord_dm(token="tok", recipient_id="123", content="hello")
    assert ok is False
    assert telemetry.snapshot().get("notify.delivery_failures", 0) == before + 1


def test_send_discord_dm_retries_once_then_succeeds(monkeypatch):
    attempts = {"n": 0}

    def fake_urlopen(request, timeout=None):
        if request.full_url.endswith("/channels"):
            attempts["n"] += 1
            if attempts["n"] == 1:
                raise urllib.error.URLError("connection reset")
            return _FakeResponse({"id": "chan1"})
        return _FakeResponse({"id": "msg1"})

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    ok = send_discord_dm(token="tok", recipient_id="123", content="hello")
    assert ok is True
    assert attempts["n"] == 2  # one failure + one retry


def test_send_discord_dm_missing_args_counts_failure():
    before = telemetry.snapshot().get("notify.delivery_failures", 0)
    assert send_discord_dm(token=None, recipient_id="123", content="hi") is False
    assert send_discord_dm(token="tok", recipient_id="", content="hi") is False
    assert send_discord_dm(token="tok", recipient_id="123", content="") is False
    assert telemetry.snapshot().get("notify.delivery_failures", 0) == before + 3


def test_deliver_unsupported_channel_counts_failure():
    config = _config(channel="email")
    config.discord.token = "tok"
    before = telemetry.snapshot().get("notify.delivery_failures", 0)
    assert deliver(config=config, content="hi") is False
    assert telemetry.snapshot().get("notify.delivery_failures", 0) == before + 1


def test_deliver_no_token_counts_failure():
    config = _config()
    config.discord.token = None
    before = telemetry.snapshot().get("notify.delivery_failures", 0)
    assert deliver(config=config, content="hi") is False
    assert telemetry.snapshot().get("notify.delivery_failures", 0) == before + 1


def test_deliver_routes_to_discord(monkeypatch):
    calls = []

    def fake_send(*, token, recipient_id, content):
        calls.append((token, recipient_id, content))
        return True

    monkeypatch.setattr("amadeus.async_notify.send_discord_dm", fake_send)
    config = _config(discord_user_id="42")
    config.discord.token = "tok"
    assert deliver(config=config, content="hi there") is True
    assert calls == [("tok", "42", "hi there")]


# ---------------------------------------------------------------------------
# Client-liveness snapshot file
# ---------------------------------------------------------------------------


def _snapshot_config(tmp_path) -> Config:
    config = Config()
    config.reminders.store_path = str(tmp_path / "reminders.json")
    return config


def test_last_seen_path_sits_alongside_reminders(tmp_path):
    config = _snapshot_config(tmp_path)
    assert last_seen_path(config) == tmp_path / "client_last_seen.json"


def test_record_and_read_last_client_poll(tmp_path):
    config = _snapshot_config(tmp_path)
    assert read_last_client_poll(config) is None
    now = dt.datetime(2026, 8, 3, 12, 0, 0)
    record_client_poll(config, now)
    assert read_last_client_poll(config) == now


def test_record_client_poll_throttled_to_once_per_minute(tmp_path):
    config = _snapshot_config(tmp_path)
    t0 = dt.datetime(2026, 8, 3, 12, 0, 0)
    record_client_poll(config, t0)
    path = last_seen_path(config)
    mtime_after_first = path.stat().st_mtime

    # A poll 10 seconds later must NOT rewrite the file.
    record_client_poll(config, t0 + dt.timedelta(seconds=10))
    assert read_last_client_poll(config) == t0
    assert path.stat().st_mtime == mtime_after_first

    # A poll a minute later DOES rewrite it.
    later = t0 + dt.timedelta(seconds=61)
    record_client_poll(config, later)
    assert read_last_client_poll(config) == later


def test_record_delivery_persists_across_reads(tmp_path):
    config = _snapshot_config(tmp_path)
    assert read_last_delivery_date(config) is None
    today = dt.date(2026, 8, 3)
    record_delivery(config, today)
    assert read_last_delivery_date(config) == today
    # Recording a delivery must not clobber an existing last_seen stamp.
    now = dt.datetime(2026, 8, 3, 9, 0, 0)
    record_client_poll(config, now)
    record_delivery(config, today)
    assert read_last_client_poll(config) == now
    assert read_last_delivery_date(config) == today


def test_corrupt_snapshot_file_degrades_to_none(tmp_path):
    config = _snapshot_config(tmp_path)
    path = last_seen_path(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{ not json")
    assert read_last_client_poll(config) is None
    assert read_last_delivery_date(config) is None


# ---------------------------------------------------------------------------
# Scheduler job wiring — pop happens only when delivery will be attempted
# ---------------------------------------------------------------------------


def _scheduler_config(tmp_path) -> Config:
    config = Config()
    config.async_notify.enabled = True
    config.async_notify.discord_user_id = "999"
    config.discord.token = "tok"
    config.async_notify.min_client_stale_minutes = 30
    config.async_notify.quiet_hours_start = 22
    config.async_notify.quiet_hours_end = 8
    config.async_notify.max_per_day = 1
    config.reach_out.enabled = True
    config.reach_out.store_path = str(tmp_path / "reachout.json")
    config.reminders.enabled = True
    config.reminders.store_path = str(tmp_path / "reminders.json")
    config.memory.persist_path = str(tmp_path / "memory.json")
    config.memory.embeddings_path = str(tmp_path / "embeddings.json")
    config.memory.profile_path = str(tmp_path / "profile.json")
    config.sleep_schedule.enabled = False
    return config


def _make_scheduler(tmp_path, now: dt.datetime, config: Config = None):
    config = config or _scheduler_config(tmp_path)
    app = create_app(config)
    store = app.state.session_store
    scheduler = BackgroundScheduler(store, config, clock=lambda: now)
    return scheduler, store, config


@pytest.mark.asyncio
async def test_disabled_job_is_noop(tmp_path):
    config = _scheduler_config(tmp_path)
    config.async_notify.enabled = False
    now = dt.datetime(2026, 8, 3, 12, 0)
    scheduler, store, config = _make_scheduler(tmp_path, now, config)
    store.reachout_store.put(
        None, ReachOut(date=now.date().isoformat(), text="hi", generated_at="x")
    )
    await scheduler._async_notify_job(now)
    # Untouched: still there to be delivered by whatever path is enabled.
    assert store.reachout_store.get(None).consumed is False


@pytest.mark.asyncio
async def test_client_active_skips_without_popping(tmp_path, monkeypatch):
    from amadeus import async_notify

    now = dt.datetime(2026, 8, 3, 12, 0)
    scheduler, store, config = _make_scheduler(tmp_path, now)
    store.reachout_store.put(
        None, ReachOut(date=now.date().isoformat(), text="hi", generated_at="x")
    )
    async_notify.record_client_poll(config, now - dt.timedelta(minutes=2))

    sent = []
    monkeypatch.setattr(
        "amadeus.web.scheduler.deliver", lambda **kw: sent.append(kw) or True
    )
    before = telemetry.snapshot().get("notify.skipped_client_active", 0)
    await scheduler._async_notify_job(now)
    assert sent == []
    assert store.reachout_store.get(None).consumed is False
    assert telemetry.snapshot().get("notify.skipped_client_active", 0) == before + 1


@pytest.mark.asyncio
async def test_quiet_hours_skips_without_popping(tmp_path, monkeypatch):
    from amadeus import async_notify

    now = dt.datetime(2026, 8, 3, 23, 0)  # inside default 22->8 quiet window
    scheduler, store, config = _make_scheduler(tmp_path, now)
    store.reachout_store.put(
        None, ReachOut(date=now.date().isoformat(), text="hi", generated_at="x")
    )

    sent = []
    monkeypatch.setattr(
        "amadeus.web.scheduler.deliver", lambda **kw: sent.append(kw) or True
    )
    await scheduler._async_notify_job(now)
    assert sent == []
    assert store.reachout_store.get(None).consumed is False


@pytest.mark.asyncio
async def test_delivers_and_pops_reachout_when_due(tmp_path, monkeypatch):
    now = dt.datetime(2026, 8, 3, 12, 0)
    scheduler, store, config = _make_scheduler(tmp_path, now)
    store.reachout_store.put(
        None,
        ReachOut(
            date=now.date().isoformat(), text="Did the demo go out?", generated_at="x"
        ),
    )

    sent = []

    def fake_deliver(*, config, content):
        sent.append(content)
        return True

    monkeypatch.setattr("amadeus.web.scheduler.deliver", fake_deliver)
    before_delivered = telemetry.snapshot().get("notify.delivered", 0)
    await scheduler._async_notify_job(now)

    assert sent == ["Did the demo go out?"]
    # Exactly-once pop: the reach-out is now consumed.
    assert store.reachout_store.get(None).consumed is True
    assert telemetry.snapshot().get("notify.delivered", 0) == before_delivered + 1
    from amadeus.async_notify import read_last_delivery_date

    assert read_last_delivery_date(config) == now.date()


@pytest.mark.asyncio
async def test_delivers_and_pops_reminders_when_no_reachout(tmp_path, monkeypatch):
    now = dt.datetime(2026, 8, 3, 12, 0)
    scheduler, store, config = _make_scheduler(tmp_path, now)
    reminder_store = ReminderStore(config.reminders.store_path)
    reminder_store.add("take the bread out of the oven", now - dt.timedelta(minutes=5))

    sent = []

    def fake_deliver(*, config, content):
        sent.append(content)
        return True

    monkeypatch.setattr("amadeus.web.scheduler.deliver", fake_deliver)
    await scheduler._async_notify_job(now)

    assert len(sent) == 1
    assert "take the bread out of the oven" in sent[0]
    assert sent[0].startswith("Reminder:")
    fresh = ReminderStore(config.reminders.store_path)
    assert fresh.list_pending() == []  # popped (announced), not left pending


@pytest.mark.asyncio
async def test_daily_cap_blocks_second_delivery_same_day(tmp_path, monkeypatch):
    now = dt.datetime(2026, 8, 3, 12, 0)
    scheduler, store, config = _make_scheduler(tmp_path, now)
    store.reachout_store.put(
        None, ReachOut(date=now.date().isoformat(), text="first", generated_at="x")
    )

    monkeypatch.setattr("amadeus.web.scheduler.deliver", lambda **kw: True)
    await scheduler._async_notify_job(now)
    assert store.reachout_store.get(None).consumed is True

    # A second, independent due item (a reminder) shows up later the same
    # day — the reach-out store's own one-per-day shape already prevents a
    # second reach-out, so a reminder is what actually exercises the cap.
    reminder_store = ReminderStore(config.reminders.store_path)
    reminder_store.add("second thing", now - dt.timedelta(minutes=5))
    later = now + dt.timedelta(hours=2)
    sent = []
    monkeypatch.setattr(
        "amadeus.web.scheduler.deliver", lambda **kw: sent.append(kw) or True
    )
    await scheduler._async_notify_job(later)
    assert sent == []
    # Capped: the reminder is left untouched (still pending) for the normal
    # poll or tomorrow's DM.
    fresh = ReminderStore(config.reminders.store_path)
    assert len(fresh.list_pending()) == 1


@pytest.mark.asyncio
async def test_delivery_failure_does_not_advance_daily_cap(tmp_path, monkeypatch):
    now = dt.datetime(2026, 8, 3, 12, 0)
    scheduler, store, config = _make_scheduler(tmp_path, now)
    store.reachout_store.put(
        None, ReachOut(date=now.date().isoformat(), text="hi", generated_at="x")
    )
    monkeypatch.setattr("amadeus.web.scheduler.deliver", lambda **kw: False)
    await scheduler._async_notify_job(now)
    from amadeus.async_notify import read_last_delivery_date

    # Delivery was attempted and failed: the reach-out is RESTORED (a real
    # outage must not destroy the day's line) and the cap is not consumed.
    assert store.reachout_store.get(None).consumed is False
    assert read_last_delivery_date(config) is None
    assert telemetry.snapshot().get("notify.undelivered_restored", 0) >= 1


@pytest.mark.asyncio
async def test_no_token_skips_without_popping(tmp_path, monkeypatch):
    """Enabled with a recipient but no bot token: nothing pops, nothing is
    destroyed, and the dedicated counter fires (the old order popped first
    and only then discovered delivery was impossible)."""
    config = _scheduler_config(tmp_path)
    config.discord.token = None
    now = dt.datetime(2026, 8, 3, 12, 0)
    scheduler, store, config = _make_scheduler(tmp_path, now, config)
    store.reachout_store.put(
        None, ReachOut(date=now.date().isoformat(), text="hi", generated_at="x")
    )
    reminder_store = ReminderStore(config.reminders.store_path)
    reminder_store.add("due thing", now - dt.timedelta(minutes=5))

    sent = []
    monkeypatch.setattr(
        "amadeus.web.scheduler.deliver", lambda **kw: sent.append(kw) or True
    )
    before = telemetry.snapshot().get("notify.skipped_no_token", 0)
    await scheduler._async_notify_job(now)
    assert sent == []
    assert store.reachout_store.get(None).consumed is False
    assert len(ReminderStore(config.reminders.store_path).list_pending()) == 1
    assert telemetry.snapshot().get("notify.skipped_no_token", 0) == before + 1


@pytest.mark.asyncio
async def test_reminder_delivery_failure_restores_pending(tmp_path, monkeypatch):
    now = dt.datetime(2026, 8, 3, 12, 0)
    scheduler, store, config = _make_scheduler(tmp_path, now)
    reminder_store = ReminderStore(config.reminders.store_path)
    reminder_store.add("take the bread out", now - dt.timedelta(minutes=5))

    monkeypatch.setattr("amadeus.web.scheduler.deliver", lambda **kw: False)
    await scheduler._async_notify_job(now)
    # Restored to pending: the poll path (or tomorrow's DM) can still
    # announce it — a failed DM must not eat the announcement.
    fresh = ReminderStore(config.reminders.store_path)
    assert [r.text for r in fresh.list_pending()] == ["take the bread out"]

    # And a later successful attempt delivers it normally.
    sent = []
    monkeypatch.setattr(
        "amadeus.web.scheduler.deliver", lambda **kw: sent.append(kw["content"]) or True
    )
    await scheduler._async_notify_job(now + dt.timedelta(minutes=2))
    assert len(sent) == 1 and "take the bread out" in sent[0]


@pytest.mark.asyncio
async def test_pops_reachout_for_mapped_discord_user(tmp_path, monkeypatch):
    """A reach-out keyed under the Discord bridge's per-user id
    ("discord-<recipient>") DMs to that recipient too — the None-only pop
    used to strand per-user reach-outs forever."""
    now = dt.datetime(2026, 8, 3, 12, 0)
    scheduler, store, config = _make_scheduler(tmp_path, now)
    store.reachout_store.put(
        "discord-999",
        ReachOut(date=now.date().isoformat(), text="bridge line", generated_at="x"),
    )

    sent = []
    monkeypatch.setattr(
        "amadeus.web.scheduler.deliver", lambda **kw: sent.append(kw["content"]) or True
    )
    await scheduler._async_notify_job(now)
    assert sent == ["bridge line"]
    assert store.reachout_store.get("discord-999").consumed is True


@pytest.mark.asyncio
async def test_asleep_skips_without_popping(tmp_path, monkeypatch):
    now = dt.datetime(2026, 8, 3, 3, 0)
    config = _scheduler_config(tmp_path)
    config.sleep_schedule.enabled = True
    config.sleep_schedule.start_hour = 1
    config.sleep_schedule.end_hour = 8
    # Push quiet hours out of the way so sleep is the only reason to skip.
    config.async_notify.quiet_hours_start = 12
    config.async_notify.quiet_hours_end = 13
    scheduler, store, config = _make_scheduler(tmp_path, now, config)
    store.reachout_store.put(
        None, ReachOut(date=now.date().isoformat(), text="hi", generated_at="x")
    )
    sent = []
    monkeypatch.setattr(
        "amadeus.web.scheduler.deliver", lambda **kw: sent.append(kw) or True
    )
    await scheduler._async_notify_job(now)
    assert sent == []
    assert store.reachout_store.get(None).consumed is False
