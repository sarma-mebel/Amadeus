"""Tests for the local reminders/timers subsystem."""

import datetime as dt
import json
import os

import pytest

from amadeus.reminders import ReminderStore
from amadeus.agents.tools import ReminderTool


def _store(tmp_path):
    return ReminderStore(str(tmp_path / "reminders.json"))


def test_add_and_roundtrip(tmp_path):
    s = _store(tmp_path)
    due = dt.datetime.now() + dt.timedelta(hours=1)
    r = s.add("submit revision", due)
    assert r.id and r.status == "pending"
    # New instance on same path sees it (CLI + tool share one file).
    s2 = _store(tmp_path)
    pend = s2.list_pending()
    assert len(pend) == 1 and pend[0].text == "submit revision"


def test_concurrent_instances_do_not_lose_adds(tmp_path):
    """Two live instances on one file (engine store + tool + jobs'
    throwaways) used to last-writer-win: an add through instance A after
    instance B had loaded silently vanished when B next saved. Reload-
    before-mutate keeps both."""
    a = _store(tmp_path)
    b = _store(tmp_path)  # loads while the file is empty
    a.add("first", dt.datetime.now() + dt.timedelta(hours=1))
    b.add("second", dt.datetime.now() + dt.timedelta(hours=2))
    fresh = _store(tmp_path)
    assert sorted(r.text for r in fresh.list_pending()) == ["first", "second"]


def test_concurrent_pop_due_announces_exactly_once(tmp_path):
    a = _store(tmp_path)
    b = _store(tmp_path)
    a.add("due thing", dt.datetime.now() - dt.timedelta(minutes=1))
    first = a.pop_due()
    second = b.pop_due()  # b must see a's announcement, not re-announce
    assert [r.text for r in first] == ["due thing"]
    assert second == []


def test_instances_share_one_lock_per_path(tmp_path):
    a = _store(tmp_path)
    b = _store(tmp_path)
    assert a._lock is b._lock
    other = ReminderStore(str(tmp_path / "elsewhere.json"))
    assert other._lock is not a._lock


def test_restore_pending_undoes_pop_due(tmp_path):
    """Async-notify's failed-DM path: popped reminders go back to pending so
    the announcement isn't destroyed; already-cancelled rows are untouched."""
    s = _store(tmp_path)
    s.add("was due", dt.datetime.now() - dt.timedelta(minutes=5))
    due = s.pop_due()
    assert [r.text for r in due] == ["was due"]
    assert s.list_pending() == []
    s.restore_pending(due)
    assert [r.text for r in s.list_pending()] == ["was due"]
    # And it can be popped again by the next delivery attempt.
    assert [r.text for r in s.pop_due()] == ["was due"]
    # Restoring again is a no-op (status is announced -> pending only once
    # per pop; a second restore after the re-pop is fine though).
    s.restore_pending([])  # empty input never touches the file


def test_pop_due_marks_announced(tmp_path):
    s = _store(tmp_path)
    past = dt.datetime.now() - dt.timedelta(minutes=5)
    future = dt.datetime.now() + dt.timedelta(hours=2)
    s.add("was due", past)
    s.add("later", future)
    due = s.pop_due()
    assert [r.text for r in due] == ["was due"]
    # Second call returns nothing (already announced).
    assert s.pop_due() == []
    # The future one is still pending.
    assert [r.text for r in s.list_pending()] == ["later"]


def test_pop_due_catches_missed_while_closed(tmp_path):
    s = _store(tmp_path)
    s.add("yesterday task", dt.datetime.now() - dt.timedelta(days=1))
    assert len(s.pop_due()) == 1  # overdue-while-closed still surfaces


def test_corrupt_file_is_tolerated(tmp_path):
    p = tmp_path / "reminders.json"
    p.write_text("{not valid json")
    s = ReminderStore(str(p))
    assert s.list_pending() == []  # no crash
    s.add("x", dt.datetime.now() + dt.timedelta(minutes=1))  # still usable
    assert json.loads(p.read_text())["reminders"]  # atomic save wrote valid JSON


def test_cancel_by_id_and_unknown(tmp_path):
    s = _store(tmp_path)
    r = s.add("cancel me", dt.datetime.now() + dt.timedelta(hours=1))
    assert s.cancel(r.id) is True
    assert s.cancel("nonexistent") is False
    assert s.list_pending() == []


def test_tool_in_minutes(tmp_path):
    tool = ReminderTool(_store(tmp_path))
    res = tool.execute(action="add", text="tea", in_minutes=20)
    assert res.success and "id" in res.metadata


def test_tool_iso_time(tmp_path):
    tool = ReminderTool(_store(tmp_path))
    future = (dt.datetime.now() + dt.timedelta(hours=3)).replace(microsecond=0)
    res = tool.execute(action="add", text="call", at=future.isoformat())
    assert res.success


def test_tool_clock_time_rolls_to_tomorrow(tmp_path):
    store = _store(tmp_path)
    tool = ReminderTool(store)
    # A clock time that already passed today should schedule for tomorrow.
    past_hour = (dt.datetime.now() - dt.timedelta(hours=1)).strftime("%I%p").lstrip("0")
    res = tool.execute(action="add", text="early", at=past_hour)
    assert res.success
    r = store.list_pending()[0]
    assert r.due_at > dt.datetime.now()


def test_tool_garbage_time_is_retryable(tmp_path):
    tool = ReminderTool(_store(tmp_path))
    res = tool.execute(action="add", text="x", at="whenever-ish")
    assert res.success is False and "parse" in (res.error or "").lower()


def test_tool_list_and_cancel(tmp_path):
    store = _store(tmp_path)
    tool = ReminderTool(store)
    add = tool.execute(action="add", text="thing", in_minutes=5)
    rid = add.metadata["id"]
    listed = tool.execute(action="list")
    assert rid in listed.result
    cancelled = tool.execute(action="cancel", reminder_id=rid)
    assert cancelled.success


def test_factory_gates_on_config(tmp_path):
    from amadeus.config import Config
    from amadeus.agents.factory import create_tool_registry

    off = Config.from_dict({"reminders": {"enabled": False}})
    assert "reminders" not in create_tool_registry(off).tools
    on = Config.from_dict(
        {"reminders": {"enabled": True, "store_path": str(tmp_path / "r.json")}}
    )
    assert "reminders" in create_tool_registry(on).tools
