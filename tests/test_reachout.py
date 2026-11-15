"""Proactive reach-out: store semantics, seed gating (the ethics predicate),
scheduler due-logic, and the exactly-once delivery endpoint. Structurally
cloned from tests/test_diary.py."""

from __future__ import annotations

import datetime as dt

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from amadeus.config import Config
from amadeus.memory.profile import UserProfile
from amadeus.providers.llm.base import ChatMessage
from amadeus.reachout import (
    ReachOut,
    ReachOutStore,
    collect_reachout_seeds,
    generate_reach_out,
)
from amadeus.web.app import create_app
from amadeus.web.scheduler import BackgroundScheduler


class RecordingProvider:
    name = "recording"

    def __init__(
        self,
        reply: str = "Did the reviewer response go out in the end? That second figure kept bugging me.",
    ) -> None:
        self.reply = reply
        self.calls: list[list[ChatMessage]] = []
        self.fail = False

    def generate(self, messages, **kwargs):
        self.calls.append(list(messages))
        if self.fail:
            raise RuntimeError("provider down")
        return self.reply


def _entry(date: dt.date, text: str = "reach-out text") -> ReachOut:
    return ReachOut(
        date=date.isoformat(), text=text, generated_at=f"{date.isoformat()}T17:00:00"
    )


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------


def test_store_roundtrip_and_same_date_refusal(tmp_path):
    store = ReachOutStore(str(tmp_path / "reachout.json"))
    today = dt.date(2026, 7, 17)
    assert store.put(None, _entry(today, "first")) is True
    assert store.put(None, _entry(today, "second")) is False  # double-gen guard
    assert store.get(None).text == "first"
    # Per-user keying.
    assert store.put("alice", _entry(today, "alice's line")) is True
    assert store.get("alice").text == "alice's line"


def test_store_pop_unconsumed_is_exactly_once(tmp_path):
    store = ReachOutStore(str(tmp_path / "reachout.json"))
    today = dt.date(2026, 7, 17)
    store.put(None, _entry(today, "the line"))

    popped = store.pop_unconsumed(None, today)
    assert popped is not None and popped.text == "the line"
    # Second pop (another tab, another tick): nothing.
    assert store.pop_unconsumed(None, today) is None
    # The consumed flag survives a reload from disk.
    fresh = ReachOutStore(str(tmp_path / "reachout.json"))
    assert fresh.pop_unconsumed(None, today) is None


def test_store_stale_entry_never_delivered_late(tmp_path):
    store = ReachOutStore(str(tmp_path / "reachout.json"))
    yesterday = dt.date(2026, 7, 16)
    store.put(None, _entry(yesterday, "stale"))
    assert store.pop_unconsumed(None, dt.date(2026, 7, 17)) is None


def test_store_corrupt_file_starts_clean(tmp_path):
    path = tmp_path / "reachout.json"
    path.write_text("{ not json")
    store = ReachOutStore(str(path))
    assert store.get(None) is None


# ---------------------------------------------------------------------------
# Seed gating — the "no manufactured excuse to ping" predicate
# ---------------------------------------------------------------------------

_NOW = dt.datetime(2026, 7, 17, 18, 0)


def _away_profile(hours_away: float, threads=("the reviewer response",)):
    return UserProfile(
        facts={"name": "Alex"},
        open_threads=list(threads),
        last_episode="talked about the resubmission",
        last_seen=(dt.datetime.utcnow() - dt.timedelta(hours=hours_away)).isoformat(),
    )


def test_seeds_none_without_profile_or_threads():
    assert collect_reachout_seeds(profile=None, now=_NOW, min_idle_hours=8) is None
    assert (
        collect_reachout_seeds(
            profile=_away_profile(24, threads=()), now=_NOW, min_idle_hours=8
        )
        is None
    )


def test_seeds_none_when_user_not_away_long_enough():
    assert (
        collect_reachout_seeds(profile=_away_profile(2), now=_NOW, min_idle_hours=8)
        is None
    )


def test_seeds_present_when_away_with_open_thread():
    seeds = collect_reachout_seeds(
        profile=_away_profile(24), now=_NOW, min_idle_hours=8
    )
    joined = "\n".join(seeds)
    assert "reviewer response" in joined
    assert "Alex" in joined
    assert "resubmission" in joined


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


def test_generate_reach_out_prompt_bans_neediness():
    provider = RecordingProvider()
    text = generate_reach_out(provider=provider, facts=["Open thread: the draft."])
    assert text == provider.reply
    system = provider.calls[0][0].content
    assert "never say you miss them" in system
    assert "never ask them to come back" in system
    assert "never mention how long they've been away" in system
    assert "SKIP" in system


def test_generate_reach_out_skip_and_failure_and_gates():
    assert generate_reach_out(provider=None, facts=["x"]) is None
    assert generate_reach_out(provider=RecordingProvider(), facts=None) is None

    skipper = RecordingProvider(reply="SKIP")
    assert generate_reach_out(provider=skipper, facts=["thin material"]) is None
    skipper_dot = RecordingProvider(reply="skip.")
    assert generate_reach_out(provider=skipper_dot, facts=["thin material"]) is None

    failing = RecordingProvider()
    failing.fail = True
    assert generate_reach_out(provider=failing, facts=["x"]) is None


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------


def _scheduler_config(tmp_path) -> Config:
    config = Config()
    config.reach_out.enabled = True
    config.reach_out.hour = 17
    config.reach_out.min_idle_hours = 8
    config.reach_out.store_path = str(tmp_path / "reachout.json")
    config.memory.persist_path = str(tmp_path / "memory.json")
    config.memory.embeddings_path = str(tmp_path / "embeddings.json")
    config.memory.profile_path = str(tmp_path / "profile.json")
    # The delivery pop is HELD inside the sleep window (default 1-8 local;
    # CI runners are UTC) — tests must pass at any wall-clock hour. The
    # sleep-hold test re-enables this with an explicit window covering now.
    config.sleep_schedule.enabled = False
    return config


def _seed_profile(config, *, hours_away: float, threads=("the reviewer response",)):
    from pathlib import Path

    profile = UserProfile(
        open_threads=list(threads),
        last_episode="talked about the resubmission",
        last_seen=(dt.datetime.utcnow() - dt.timedelta(hours=hours_away)).isoformat(),
        session_count=3,
    )
    profile.save_atomic(Path(config.memory.profile_path))


def _make_scheduler(monkeypatch, tmp_path, now: dt.datetime, *, hours_away=24.0):
    config = _scheduler_config(tmp_path)
    _seed_profile(config, hours_away=hours_away)
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
    now = dt.datetime(2026, 7, 17, 16, 30)
    scheduler, store, recorder, _ = _make_scheduler(monkeypatch, tmp_path, now)
    await scheduler._reachout_job(now)
    assert store.reachout_store.get(None) is None
    assert recorder.calls == []


@pytest.mark.asyncio
async def test_scheduler_generates_once_when_user_is_away(monkeypatch, tmp_path):
    now = dt.datetime(2026, 7, 17, 17, 5)
    scheduler, store, recorder, _ = _make_scheduler(monkeypatch, tmp_path, now)
    await scheduler._reachout_job(now)
    entry = store.reachout_store.get(None)
    assert entry is not None and entry.date == "2026-07-17"
    assert len(recorder.calls) == 1

    # Same-day second tick: no regeneration.
    await scheduler._reachout_job(now + dt.timedelta(minutes=5))
    assert len(recorder.calls) == 1


@pytest.mark.asyncio
async def test_scheduler_stays_quiet_for_active_user(monkeypatch, tmp_path):
    """A user who talked to her an hour ago gets NO reach-out — and nothing
    is stored, so the predicate re-evaluates when they do go idle."""
    now = dt.datetime(2026, 7, 17, 17, 5)
    scheduler, store, recorder, config = _make_scheduler(
        monkeypatch, tmp_path, now, hours_away=1.0
    )
    await scheduler._reachout_job(now)
    assert store.reachout_store.get(None) is None
    assert recorder.calls == []

    # They go idle later the same day: the next tick generates.
    _seed_profile(config, hours_away=9.0)
    await scheduler._reachout_job(now + dt.timedelta(hours=2))
    assert store.reachout_store.get(None) is not None
    assert len(recorder.calls) == 1


@pytest.mark.asyncio
async def test_scheduler_disabled_noop(monkeypatch, tmp_path):
    now = dt.datetime(2026, 7, 17, 17, 5)
    scheduler, store, recorder, config = _make_scheduler(monkeypatch, tmp_path, now)
    config.reach_out.enabled = False
    await scheduler._reachout_job(now)
    assert recorder.calls == []


# ---------------------------------------------------------------------------
# Delivery endpoint
# ---------------------------------------------------------------------------


def test_endpoint_disabled_returns_null(tmp_path):
    config = Config()
    config.memory.persist_path = str(tmp_path / "memory.json")
    config.memory.embeddings_path = str(tmp_path / "embeddings.json")
    config.memory.profile_path = str(tmp_path / "profile.json")
    app = create_app(config)
    client = TestClient(app)
    data = client.post("/api/reachout/due", json={}).json()
    assert data["reach_out"] is None


def test_endpoint_delivers_exactly_once_and_commits_to_memory(monkeypatch, tmp_path):
    config = _scheduler_config(tmp_path)
    recorder = RecordingProvider()
    import amadeus.web.app as app_module

    monkeypatch.setattr(app_module, "load_llm", lambda cfg: recorder)
    app = create_app(config)
    store = app.state.session_store
    store.llm = recorder

    today = dt.date.today()
    store.reachout_store.put(None, _entry(today, "Did the draft go out?"))

    client = TestClient(app)
    first = client.post("/api/reachout/due", json={}).json()
    assert first["reach_out"] == "Did the draft go out?"

    # She remembers having said it — the user's reply must make sense in
    # the next turn's context.
    session = store.sessions[first["session_id"]]
    assistant = [m for m in session.memory.short_term if m.role == "assistant"]
    assert any("Did the draft go out?" in m.content for m in assistant)

    # Exactly once: the same session AND a fresh session both get nothing.
    again = client.post(
        "/api/reachout/due", json={"session_id": first["session_id"]}
    ).json()
    assert again["reach_out"] is None


# ---------------------------------------------------------------------------
# Per-character scoping: paths, not prompts (state-side isolation)
# ---------------------------------------------------------------------------


def test_reach_out_store_path_anchors_to_data_dir(monkeypatch, tmp_path):
    """A relative reachout.json must resolve against the platform data dir
    like every other user-data file — previously it was the one side-store
    path missing from resolve_user_data_paths and split by launch CWD."""
    monkeypatch.setenv("AMADEUS_DATA_DIR", str(tmp_path / "data"))
    config = Config()
    config.resolve_user_data_paths()
    assert config.reach_out.store_path == str(tmp_path / "data" / "reachout.json")
    # Explicit absolute paths pass through untouched.
    config.reach_out.store_path = str(tmp_path / "elsewhere" / "r.json")
    config.resolve_user_data_paths()
    assert config.reach_out.store_path == str(tmp_path / "elsewhere" / "r.json")


def _side_store_config(tmp_path) -> Config:
    config = Config()
    config.daily_brief.enabled = True
    config.daily_brief.store_path = str(tmp_path / "daily_brief.json")
    config.diary.enabled = True
    config.diary.store_path = str(tmp_path / "kurisu_diary.json")
    config.reach_out.enabled = True
    config.reach_out.store_path = str(tmp_path / "reachout.json")
    config.follow_through.enabled = True
    config.follow_through.store_path = str(tmp_path / "followup.json")
    config.life_threads.enabled = True
    config.life_threads.store_path = str(tmp_path / "life_threads.json")
    config.insights.enabled = True
    config.insights.store_path = str(tmp_path / "insights.json")
    return config


def _guest(config) -> Config:
    import dataclasses

    guest = dataclasses.replace(config)
    guest.character = dataclasses.replace(
        config.character, name="Rem", full_name="Rem", system_prompt="You are Rem."
    )
    return guest


def test_side_store_paths_builtin_keeps_legacy_locations(tmp_path):
    from amadeus.web.sessions import SessionStore

    config = _side_store_config(tmp_path)
    (
        brief,
        diary,
        reachout,
        followup,
        life,
        insights,
        rhythm,
    ) = SessionStore._side_store_paths(config)
    assert brief == str(tmp_path / "daily_brief.json")
    assert diary == str(tmp_path / "kurisu_diary.json")
    assert reachout == str(tmp_path / "reachout.json")
    assert followup == str(tmp_path / "followup.json")
    assert life == str(tmp_path / "life_threads.json")
    assert insights == str(tmp_path / "insights.json")


def test_side_store_paths_scope_all_four_to_guest_character(tmp_path):
    """Guest character active: brief, diary, reach-out AND follow-up state
    all live under characters/<slug>/ — a guest must never pop and speak a
    line Kurisu generated (or vice versa)."""
    from amadeus.web.sessions import SessionStore

    guest = _guest(_side_store_config(tmp_path))
    (
        brief,
        diary,
        reachout,
        followup,
        life,
        insights,
        rhythm,
    ) = SessionStore._side_store_paths(guest)
    scoped = tmp_path / "characters" / "rem"
    assert brief == str(scoped / "daily_brief.json")
    assert diary == str(scoped / "kurisu_diary.json")
    assert reachout == str(scoped / "reachout.json")
    assert followup == str(scoped / "followup.json")
    assert life == str(scoped / "life_threads.json")
    assert insights == str(scoped / "insights.json")


def test_side_store_paths_flag_off_keeps_legacy_with_guest(tmp_path):
    from amadeus.web.sessions import SessionStore

    guest = _guest(_side_store_config(tmp_path))
    guest.memory.per_character_memory = False
    (
        brief,
        diary,
        reachout,
        followup,
        life,
        insights,
        rhythm,
    ) = SessionStore._side_store_paths(guest)
    assert brief == str(tmp_path / "daily_brief.json")
    assert diary == str(tmp_path / "kurisu_diary.json")
    assert reachout == str(tmp_path / "reachout.json")
    assert followup == str(tmp_path / "followup.json")
    assert life == str(tmp_path / "life_threads.json")
    assert insights == str(tmp_path / "insights.json")


def test_side_store_paths_disabled_sections_are_none(tmp_path):
    from amadeus.web.sessions import SessionStore

    config = _side_store_config(tmp_path)
    config.diary.enabled = False
    config.reach_out.enabled = False
    config.follow_through.enabled = False
    config.life_threads.enabled = False
    config.insights.enabled = False
    (
        brief,
        diary,
        reachout,
        followup,
        life,
        insights,
        rhythm,
    ) = SessionStore._side_store_paths(config)
    assert brief == str(tmp_path / "daily_brief.json")
    assert diary is None
    assert reachout is None
    assert followup is None
    assert life is None
    assert insights is None


def test_seeds_include_own_life_facts_when_supplied():
    """Away-time continuity: her diary/pursuit lines ride along (peek-only,
    caller-supplied) so the line can open from HER elapsed life; without
    them the seed set is unchanged."""
    seeds = collect_reachout_seeds(
        profile=_away_profile(24),
        now=_NOW,
        min_idle_hours=8,
        own_life_facts=["Your diary (2026-07-17): spent the day on the paper."],
    )
    joined = "\n".join(seeds)
    assert "Your own recent days" in joined
    assert "spent the day on the paper" in joined

    plain = collect_reachout_seeds(
        profile=_away_profile(24), now=_NOW, min_idle_hours=8, own_life_facts=[]
    )
    assert not any("Your own recent days" in line for line in plain)


def test_generate_reach_out_prompt_allows_own_life_opener():
    provider = RecordingProvider()
    generate_reach_out(provider=provider, facts=["Open thread: the draft."])
    system = provider.calls[0][0].content
    assert "Your own recent days" in system
    assert "never invented" in system


def test_endpoint_holds_line_during_her_sleep_window(monkeypatch, tmp_path):
    """During the sleep window the line is HELD (not consumed) — the same
    poll delivers it after she wakes, instead of her reaching out at 3am."""
    config = _scheduler_config(tmp_path)
    now_hour = dt.datetime.now().hour
    config.sleep_schedule.enabled = True
    config.sleep_schedule.start_hour = now_hour
    config.sleep_schedule.end_hour = (now_hour + 2) % 24
    recorder = RecordingProvider()
    import amadeus.web.app as app_module

    monkeypatch.setattr(app_module, "load_llm", lambda cfg: recorder)
    app = create_app(config)
    store = app.state.session_store
    store.llm = recorder
    store.reachout_store.put(None, _entry(dt.date.today(), "Did it work out?"))

    client = TestClient(app)
    asleep = client.post("/api/reachout/due", json={}).json()
    assert asleep["reach_out"] is None
    # Not consumed: waking flips only the clock, and delivery proceeds.
    store.config.sleep_schedule.enabled = False
    awake = client.post("/api/reachout/due", json={}).json()
    assert awake["reach_out"] == "Did it work out?"
