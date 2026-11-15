"""Proactive follow-ups: ripest-open-thread selection, asked exactly once."""

from __future__ import annotations

import datetime as dt

from amadeus.memory.profile import UserProfile


def _profile_with_threads(entries):
    profile = UserProfile()
    profile.open_threads = [text for text, _ in entries]
    profile.open_thread_dates = {text: date for text, date in entries}
    return profile


def _days_ago(n: int) -> str:
    return (dt.date.today() - dt.timedelta(days=n)).isoformat()


def test_ripest_thread_wins():
    profile = _profile_with_threads(
        [
            ("new idea", _days_ago(0)),
            ("interview", _days_ago(3)),
            ("dentist", _days_ago(1)),
        ]
    )
    assert profile.followup_thread() == "interview"


def test_same_day_threads_are_not_followups():
    profile = _profile_with_threads([("today thing", _days_ago(0))])
    assert profile.followup_thread() is None


def test_each_thread_is_asked_once():
    profile = _profile_with_threads([("interview", _days_ago(2))])
    picked = profile.followup_thread()
    assert picked == "interview"
    profile.mark_thread_followed_up(picked)
    assert profile.followup_thread() is None
    # Mark is idempotent and case-insensitive.
    profile.mark_thread_followed_up("Interview")
    assert profile.followed_up_threads == ["interview"]


def test_resolved_thread_frees_its_slot(tmp_path):
    profile = _profile_with_threads([("interview", _days_ago(2))])
    profile.mark_thread_followed_up("interview")
    # Extraction resolves it; later the same text reappears as a new thread.
    profile.set_open_threads([], max_threads=3)
    assert profile.followed_up_threads == []
    profile.set_open_threads(["interview"], max_threads=3)
    # Fresh thread, noted today: not a follow-up candidate yet.
    assert profile.followup_thread() is None


def test_followed_up_survives_save_load(tmp_path):
    profile = _profile_with_threads([("interview", _days_ago(2))])
    profile.mark_thread_followed_up("interview")
    path = tmp_path / "profile.json"
    profile.save_atomic(path)
    loaded = UserProfile.load(path)
    assert loaded.followed_up_threads == ["interview"]
    assert loaded.followup_thread() is None
