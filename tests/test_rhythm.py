"""Rhythm memory (amadeus/rhythm.py): the store's bounds, the LLM-free
notable predicate, and the endpoints.

The tests that matter most are the ones pinning what CANNOT be stored. This
is the only feature in the codebase that keeps a durable record of the
user's computer use, and the whole ethical argument for shipping it rests
on the artifact being coarse, readable in one screen, and erasable — so
"an unknown app bucket is dropped" and "clear actually deletes" are
contract tests, not nice-to-haves.
"""

from __future__ import annotations

import datetime as dt

import pytest

from amadeus import telemetry
from amadeus.rhythm import (
    APP_BUCKETS,
    RETENTION_DAYS,
    RhythmDay,
    RhythmStore,
    notable_rhythm,
    rhythm_greeting_fact,
)


@pytest.fixture(autouse=True)
def _clean_telemetry():
    telemetry.reset()
    yield
    telemetry.reset()


def _day(date: str, **overrides) -> RhythmDay:
    return RhythmDay(date=date, **overrides)


# ---------------------------------------------------------------------------
# What the store will and will not hold
# ---------------------------------------------------------------------------


def test_unknown_app_buckets_are_dropped() -> None:
    """THE CONTRACT. The client maps to a fixed vocabulary before posting,
    but a modified client or a hand-edited file must not be able to widen
    what the store holds into a list of the user's software."""
    day = RhythmDay.from_dict(
        {
            "date": "2026-08-01",
            "app_buckets": {
                "editor": 4,
                "Obsidian": 9,
                "salary-negotiation-notes.md": 2,
                "browser": 1,
            },
        }
    )

    assert day is not None
    assert set(day.app_buckets) <= set(APP_BUCKETS)
    assert day.app_buckets == {"editor": 4, "browser": 1}


def test_from_dict_clamps_and_sanitizes() -> None:
    day = RhythmDay.from_dict(
        {
            "date": "2026-08-01",
            "late_night_blocks": -5,
            "long_focus_stretches": "junk",
            "longest_focus_hours": "nope",
            "active_hour_buckets": [3, 3, 99, -1, "x", 22],
            "music_events": 7,
            "app_buckets": {"editor": -2},
        }
    )

    assert day is not None
    assert day.late_night_blocks == 0
    assert day.long_focus_stretches == 0
    assert day.longest_focus_hours == 0.0
    assert day.active_hour_buckets == [3, 22]
    assert day.music_events == 7
    # A zero/negative count carries no information; it is not stored.
    assert day.app_buckets == {}


def test_from_dict_rejects_undated_entries() -> None:
    assert RhythmDay.from_dict({"late_night_blocks": 3}) is None
    assert RhythmDay.from_dict("not a dict") is None
    assert RhythmDay.from_dict(None) is None


def test_stored_day_has_no_field_for_titles_or_timestamps() -> None:
    """If a field is not in this set, it is not stored — and adding one is
    an ethics-doc change, not a refactor."""
    payload = _day("2026-08-01").to_dict()
    assert set(payload) == {
        "date",
        "late_night_blocks",
        "long_focus_stretches",
        "longest_focus_hours",
        "active_hour_buckets",
        "music_events",
        "battery_low_events",
        "app_buckets",
    }


# ---------------------------------------------------------------------------
# Store mechanics
# ---------------------------------------------------------------------------


def test_record_and_read_roundtrip(tmp_path) -> None:
    store = RhythmStore(str(tmp_path / "rhythm.json"))
    today = dt.date(2026, 8, 7)
    store.record(None, _day("2026-08-07", late_night_blocks=2), today)

    assert [d.date for d in store.days(None)] == ["2026-08-07"]
    # Survives a reload.
    assert [d.date for d in RhythmStore(str(tmp_path / "rhythm.json")).days(None)] == [
        "2026-08-07"
    ]


def test_same_day_post_replaces_rather_than_appends(tmp_path) -> None:
    """The client posts the running total for the current day."""
    store = RhythmStore(str(tmp_path / "rhythm.json"))
    today = dt.date(2026, 8, 7)
    store.record(None, _day("2026-08-07", music_events=1), today)
    store.record(None, _day("2026-08-07", music_events=4), today)

    (day,) = store.days(None)
    assert day.music_events == 4


def test_retention_trims_old_days(tmp_path) -> None:
    store = RhythmStore(str(tmp_path / "rhythm.json"))
    today = dt.date(2026, 8, 20)
    old = (today - dt.timedelta(days=RETENTION_DAYS + 3)).isoformat()
    store.record(None, _day(old), today)
    store.record(None, _day(today.isoformat()), today)

    dates = [d.date for d in store.days(None)]
    assert old not in dates
    assert today.isoformat() in dates


def test_days_are_keyed_per_user(tmp_path) -> None:
    store = RhythmStore(str(tmp_path / "rhythm.json"))
    today = dt.date(2026, 8, 7)
    store.record("a", _day("2026-08-07", music_events=1), today)
    store.record("b", _day("2026-08-07", music_events=9), today)

    assert store.days("a")[0].music_events == 1
    assert store.days("b")[0].music_events == 9


def test_clear_erases_the_tally(tmp_path) -> None:
    """Turning the feature off DELETES the record — that is the promise made
    in the settings copy, so it is pinned here."""
    path = tmp_path / "rhythm.json"
    store = RhythmStore(str(path))
    today = dt.date(2026, 8, 7)
    store.record(None, _day("2026-08-07", late_night_blocks=3), today)

    store.clear(None)

    assert store.days(None) == []
    assert RhythmStore(str(path)).days(None) == []
    assert telemetry.snapshot().get("rhythm.cleared") == 1


def test_corrupt_file_starts_clean_loudly(tmp_path) -> None:
    path = tmp_path / "rhythm.json"
    path.write_text("{ not json", encoding="utf-8")

    store = RhythmStore(str(path))

    assert store.days(None) == []
    # Counts once per read attempt (reload-before-mutate is the store
    # convention), so the assertion is "loudly", not an exact tally.
    assert telemetry.snapshot().get("rhythm.store_load_failed", 0) >= 1


def test_surfaced_floor(tmp_path) -> None:
    store = RhythmStore(str(tmp_path / "rhythm.json"))
    today = dt.date(2026, 8, 7)
    assert store.surfaced_recently(None, today, 7) is False

    store.mark_surfaced(None, today)

    assert store.surfaced_recently(None, today, 7) is True
    assert store.surfaced_recently(None, today + dt.timedelta(days=3), 7) is True
    assert store.surfaced_recently(None, today + dt.timedelta(days=8), 7) is False
    assert telemetry.snapshot().get("rhythm.surfaced") == 1


# ---------------------------------------------------------------------------
# notable_rhythm — deterministic, and usually None
# ---------------------------------------------------------------------------


def test_nothing_notable_is_the_common_answer() -> None:
    today = dt.date(2026, 8, 7)
    ordinary = [
        _day((today - dt.timedelta(days=i)).isoformat(), long_focus_stretches=0)
        for i in range(5)
    ]
    assert notable_rhythm(ordinary, today) is None
    assert notable_rhythm([], today) is None


def test_too_little_history_is_never_a_pattern() -> None:
    today = dt.date(2026, 8, 7)
    two_days = [
        _day((today - dt.timedelta(days=i)).isoformat(), late_night_blocks=4)
        for i in range(2)
    ]
    assert notable_rhythm(two_days, today) is None


def test_repeated_late_nights_are_notable() -> None:
    today = dt.date(2026, 8, 7)
    days = [
        _day((today - dt.timedelta(days=i)).isoformat(), late_night_blocks=2)
        for i in range(4)
    ]
    hint = notable_rhythm(days, today)
    assert hint is not None and "past midnight" in hint


def test_many_long_stretches_name_the_dominant_app_category(tmp_path) -> None:
    today = dt.date(2026, 8, 7)
    days = [
        _day(
            (today - dt.timedelta(days=i)).isoformat(),
            long_focus_stretches=2,
            app_buckets={"editor": 10, "browser": 1},
        )
        for i in range(4)
    ]
    hint = notable_rhythm(days, today)
    assert hint is not None
    assert "uninterrupted stretches" in hint
    assert "editor" in hint


def test_dominant_other_bucket_is_not_named() -> None:
    """ "Mostly in an other" is nonsense; the generic phrasing wins."""
    today = dt.date(2026, 8, 7)
    days = [
        _day(
            (today - dt.timedelta(days=i)).isoformat(),
            long_focus_stretches=2,
            app_buckets={"other": 10},
        )
        for i in range(4)
    ]
    hint = notable_rhythm(days, today)
    assert hint is not None and "mostly in" not in hint


# ---------------------------------------------------------------------------
# Greeting fact framing
# ---------------------------------------------------------------------------


def test_greeting_fact_is_color_never_an_ask() -> None:
    fact = rhythm_greeting_fact("They have been up past midnight most nights.")
    assert fact is not None
    assert "never as a statistic" in fact
    assert "never as concern" in fact
    assert "never as advice" in fact
    assert "at most one short clause" in fact
    # No question is ever asked — the greeting's ask slot belongs to the
    # user's own material (tests/test_greeting_arbitration.py).
    assert "?" not in fact


def test_no_hint_means_no_fact() -> None:
    assert rhythm_greeting_fact(None) is None
    assert rhythm_greeting_fact("") is None


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from amadeus.config import Config  # noqa: E402
from amadeus.web.app import create_app  # noqa: E402


class _Dummy:
    name = "dummy"

    def generate(self, messages, **kwargs):
        return "ok"


def _client(tmp_path, monkeypatch, *, enabled: bool = True):
    import amadeus.web.app as app_module

    config = Config()
    config.memory.persist_path = str(tmp_path / "memory.json")
    config.memory.embeddings_path = str(tmp_path / "embeddings.json")
    config.memory.profile_path = str(tmp_path / "profile.json")
    config.rhythm.enabled = enabled
    config.rhythm.store_path = str(tmp_path / "rhythm.json")
    monkeypatch.setattr(app_module, "load_llm", lambda cfg: _Dummy())
    return TestClient(create_app(config))


def test_post_get_delete_roundtrip(tmp_path, monkeypatch) -> None:
    client = _client(tmp_path, monkeypatch)
    today = dt.date.today().isoformat()

    posted = client.post(
        "/api/rhythm",
        json={
            "date": today,
            "late_night_blocks": 2,
            "long_focus_stretches": 3,
            "longest_focus_hours": 4.5,
            "active_hour_buckets": [9, 10, 23],
            "music_events": 5,
            "app_buckets": {"editor": 8, "Obsidian": 4},
        },
    )
    assert posted.json() == {"recorded": True}

    payload = client.get("/api/rhythm").json()
    (day,) = payload["rhythm"]["days"]
    assert day["date"] == today
    assert day["long_focus_stretches"] == 3
    # The free-text app name never lands, even posted directly at the API.
    assert day["app_buckets"] == {"editor": 8}

    assert client.delete("/api/rhythm").json() == {"cleared": True}
    assert client.get("/api/rhythm").json()["rhythm"]["days"] == []


def test_endpoints_are_inert_when_the_feature_is_off(tmp_path, monkeypatch) -> None:
    """Half the double gate: with rhythm.enabled false there is no store, so
    nothing can be written however the client behaves."""
    client = _client(tmp_path, monkeypatch, enabled=False)

    assert client.post(
        "/api/rhythm", json={"date": dt.date.today().isoformat()}
    ).json() == {"recorded": False}
    assert client.get("/api/rhythm").json() == {"rhythm": None}
    assert client.delete("/api/rhythm").json() == {"cleared": False}


def test_post_rejects_an_out_of_range_payload(tmp_path, monkeypatch) -> None:
    client = _client(tmp_path, monkeypatch)
    response = client.post(
        "/api/rhythm",
        json={"date": dt.date.today().isoformat(), "late_night_blocks": 9999},
    )
    assert response.status_code == 422
