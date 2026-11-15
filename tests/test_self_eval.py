"""Nightly self-eval (amadeus/self_eval.py, config.self_eval, the
scheduler's _self_eval_job, and its GET /api/status surfacing).

NO live LLM calls anywhere here: run_nightly_self_eval's live run is
exercised only through a stubbed `runner` callable — the tests below cover
the plumbing around it (gating, once/day, trend append, regression math,
status/telemetry surfacing, crash isolation), not the harness itself
(that's tests/test_eval_harness.py).
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from amadeus import self_eval, telemetry
from amadeus.config import Config
from amadeus.web.app import create_app
from amadeus.web.scheduler import BackgroundScheduler


@pytest.fixture(autouse=True)
def _reset_state():
    self_eval.reset()
    telemetry.reset()
    yield
    self_eval.reset()
    telemetry.reset()


# ---------------------------------------------------------------------------
# Config defaults
# ---------------------------------------------------------------------------


def test_self_eval_config_defaults_off():
    config = Config()
    assert config.self_eval.enabled is False
    assert config.self_eval.hour == 4
    assert config.self_eval.scenarios  # a small non-empty default subset
    assert len(config.self_eval.scenarios) <= 5
    assert config.self_eval.max_calls_per_day == 10
    assert config.self_eval.regression_tolerance == 2.0


def test_self_eval_trend_path_resolved_to_data_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("AMADEUS_DATA_DIR", str(tmp_path))
    config = Config()
    config.resolve_user_data_paths()
    assert config.self_eval.trend_path == str(tmp_path / "self_eval_trend.jsonl")


def test_self_eval_absolute_trend_path_untouched(tmp_path):
    config = Config()
    abs_path = str(tmp_path / "custom_trend.jsonl")
    config.self_eval.trend_path = abs_path
    config.resolve_user_data_paths()
    assert config.self_eval.trend_path == abs_path


# ---------------------------------------------------------------------------
# Pure regression math
# ---------------------------------------------------------------------------


def test_rolling_median_empty_is_none():
    assert self_eval.rolling_median([]) is None


def test_rolling_median_basic():
    assert self_eval.rolling_median([8, 9, 10]) == 9


def test_rolling_median_windowed_to_last_n():
    # Only the last 3 entries count with window=3, despite 5 given.
    assert self_eval.rolling_median([2, 2, 2, 9, 9, 9], window=3) == 9


def test_check_regression_no_prior_entries_never_regresses():
    regressions = self_eval.check_regression([], {"alpha": 2.0})
    assert regressions == {}


def test_check_regression_detects_drop_past_tolerance():
    entries = [
        {"scores": {"alpha": 9}},
        {"scores": {"alpha": 9}},
        {"scores": {"alpha": 8}},
    ]
    # median of [9, 9, 8] = 9; latest 5 is a 4-point drop, over the 2.0 tolerance.
    regressions = self_eval.check_regression(entries, {"alpha": 5}, tolerance=2.0)
    assert "alpha" in regressions
    assert regressions["alpha"]["median"] == 9
    assert regressions["alpha"]["latest"] == 5
    assert regressions["alpha"]["drop"] == 4


def test_check_regression_within_tolerance_is_not_flagged():
    entries = [{"scores": {"alpha": 9}}, {"scores": {"alpha": 8}}]
    # median 8.5, latest 7 -> drop 1.5, under the 2.0 default tolerance.
    regressions = self_eval.check_regression(entries, {"alpha": 7})
    assert regressions == {}


def test_check_regression_only_flags_scenarios_present_in_history():
    entries = [{"scores": {"alpha": 9}}]
    regressions = self_eval.check_regression(
        entries, {"alpha": 9, "beta": 0}, tolerance=2.0
    )
    # beta has no prior history -> nothing to compare against -> never flagged.
    assert "beta" not in regressions
    assert "alpha" not in regressions  # no drop


def test_check_regression_ignores_entries_missing_the_scenario():
    entries = [{"scores": {"alpha": 9}}, {"scores": {"beta": 1}}]  # beta-only entry
    regressions = self_eval.check_regression(entries, {"alpha": 3}, tolerance=2.0)
    assert regressions["alpha"]["median"] == 9  # only alpha's one data point counted


# ---------------------------------------------------------------------------
# Trend file I/O
# ---------------------------------------------------------------------------


def test_read_trend_entries_missing_file_returns_empty(tmp_path):
    assert self_eval.read_trend_entries(tmp_path / "nope.jsonl") == []


def test_append_and_read_trend_roundtrip(tmp_path):
    path = tmp_path / "trend.jsonl"
    self_eval.append_trend_entry(
        path, {"timestamp": "2026-07-01T04:00:00Z", "scores": {"a": 8}}
    )
    self_eval.append_trend_entry(
        path, {"timestamp": "2026-07-02T04:00:00Z", "scores": {"a": 7}}
    )
    entries = self_eval.read_trend_entries(path)
    assert len(entries) == 2
    assert entries[0]["scores"] == {"a": 8}
    assert entries[1]["scores"] == {"a": 7}


def test_read_trend_entries_skips_corrupt_lines(tmp_path):
    path = tmp_path / "trend.jsonl"
    path.write_text(
        '{"timestamp": "2026-07-01T00:00:00Z", "scores": {"a": 8}}\n'
        "not json at all\n"
        '{"timestamp": "2026-07-02T00:00:00Z", "scores": {"a": 7}}\n'
    )
    entries = self_eval.read_trend_entries(path)
    assert len(entries) == 2


def test_last_run_date():
    assert self_eval.last_run_date([]) is None
    # A legacy UTC 'Z' stamp converts to the LOCAL date (the scheduler's
    # due-guard compares against its local clock), so compute the
    # expectation through the same conversion instead of hardcoding — this
    # test must pass in any timezone.
    entries = [{"timestamp": "2026-07-10T12:00:00Z"}]
    expected = (
        dt.datetime(2026, 7, 10, 12, 0, tzinfo=dt.timezone.utc)
        .astimezone()
        .date()
        .isoformat()
    )
    assert self_eval.last_run_date(entries) == expected
    # A naive stamp is assumed local and read verbatim.
    assert self_eval.last_run_date([{"timestamp": "2026-07-10T04:05:00"}]) == (
        "2026-07-10"
    )


def test_last_run_date_utc_offset_regression():
    """THE UTC+8 BUG. In a UTC+8 zone, the 04:00 local run used to stamp
    utcnow() (= 20:00 the PREVIOUS day) and last_run_date read that UTC date
    verbatim, so the scheduler's `== now.date()` guard never matched and
    every 60s tick launched ANOTHER full live-LLM eval run until 08:00
    local. With local-date conversion, a stamp written at 04:00 local
    (however encoded) must map back to that same local date."""
    import os
    import time

    if not hasattr(time, "tzset"):  # pragma: no cover - non-POSIX
        pytest.skip("tzset unavailable on this platform")
    original_tz = os.environ.get("TZ")
    os.environ["TZ"] = "Asia/Singapore"  # UTC+8, no DST
    time.tzset()
    try:
        # 2026-08-08 04:05 local (+08:00) == 2026-08-07 20:05 UTC. The old
        # code returned "2026-08-07" here — yesterday, forever due.
        for stamp in (
            "2026-08-07T20:05:00Z",  # legacy UTC-Z encoding
            "2026-08-08T04:05:00+08:00",  # new local-aware encoding
            "2026-08-08T04:05:00",  # naive local
        ):
            assert (
                self_eval.last_run_date([{"timestamp": stamp}]) == "2026-08-08"
            ), stamp
    finally:
        if original_tz is None:
            os.environ.pop("TZ", None)
        else:
            os.environ["TZ"] = original_tz
        time.tzset()


def test_run_stamp_matches_local_date(tmp_path, monkeypatch):
    """The trend entry written by a run must stamp the same LOCAL date the
    scheduler guard compares against — one clock on both sides."""
    config = _self_eval_config(tmp_path)
    monkeypatch.setattr("amadeus.provider_loader.load_llm", lambda cfg: _FakeLLM())
    monkeypatch.setattr("amadeus.provider_loader.load_embedder", lambda cfg: None)
    self_eval.run_nightly_self_eval(config, runner=lambda *a, **k: {"alpha": 9.0})
    entries = self_eval.read_trend_entries(config.self_eval.trend_path)
    assert self_eval.last_run_date(entries) == dt.date.today().isoformat()


def test_append_trend_entry_creates_parent_dirs(tmp_path):
    path = tmp_path / "nested" / "dir" / "trend.jsonl"
    self_eval.append_trend_entry(path, {"a": 1})
    assert path.exists()


# ---------------------------------------------------------------------------
# run_nightly_self_eval: gating + stubbed runner (no live LLM calls)
# ---------------------------------------------------------------------------


class _FakeLLM:
    name = "fake"
    model = "fake-model"


def _self_eval_config(tmp_path) -> Config:
    config = Config()
    config.self_eval.enabled = True
    config.self_eval.trend_path = str(tmp_path / "trend.jsonl")
    config.self_eval.scenarios = ["alpha", "beta"]
    config.memory.persist_path = str(tmp_path / "memory.json")
    config.memory.embeddings_path = str(tmp_path / "embeddings.json")
    config.memory.profile_path = str(tmp_path / "profile.json")
    return config


def test_run_nightly_self_eval_disabled_noop(tmp_path, monkeypatch):
    config = _self_eval_config(tmp_path)
    config.self_eval.enabled = False
    calls = []

    def fake_runner(*args, **kwargs):
        calls.append(args)
        return {"alpha": 9}

    self_eval.run_nightly_self_eval(config, runner=fake_runner)
    assert calls == []
    assert not (tmp_path / "trend.jsonl").exists()


def test_run_nightly_self_eval_appends_trend_and_status(tmp_path, monkeypatch):
    config = _self_eval_config(tmp_path)
    monkeypatch.setattr("amadeus.provider_loader.load_llm", lambda cfg: _FakeLLM())
    monkeypatch.setattr("amadeus.provider_loader.load_embedder", lambda cfg: None)

    def fake_runner(cfg, llm, embedder, judge_llm, names, scenarios_path):
        assert names == ["alpha", "beta"]
        # The config passed to the runner must be scoped to an isolated tmp
        # store, never the live memory.persist_path the outer config carries.
        assert cfg.memory.persist_path != config.memory.persist_path
        return {"alpha": 9.0, "beta": 8.0}

    self_eval.run_nightly_self_eval(config, runner=fake_runner)

    entries = self_eval.read_trend_entries(config.self_eval.trend_path)
    assert len(entries) == 1
    assert entries[0]["scores"] == {"alpha": 9.0, "beta": 8.0}
    assert "judge_model" in entries[0]
    assert set(entries[0]["config_fingerprint"]) == {
        "llm_provider",
        "llm_model",
        "embedding_provider",
    }
    # Never a secret in the fingerprint.
    assert "key" not in json.dumps(entries[0]["config_fingerprint"]).lower()

    status = self_eval.status_snapshot()
    assert status["last_run_at"] is not None
    assert status["scores"] == {"alpha": 9.0, "beta": 8.0}
    assert status["regression"] is False
    assert status["regressions"] is None


def test_run_nightly_self_eval_flags_regression(tmp_path, monkeypatch):
    config = _self_eval_config(tmp_path)
    monkeypatch.setattr("amadeus.provider_loader.load_llm", lambda cfg: _FakeLLM())
    monkeypatch.setattr("amadeus.provider_loader.load_embedder", lambda cfg: None)
    # Seed prior history so the regression check has a median to compare
    # against (check_regression needs >=1 prior entry).
    self_eval.append_trend_entry(
        Path(config.self_eval.trend_path),
        {"scores": {"alpha": 9.0, "beta": 9.0}},
    )

    def fake_runner(cfg, llm, embedder, judge_llm, names, scenarios_path):
        return {"alpha": 3.0, "beta": 9.0}  # alpha regresses, beta doesn't

    self_eval.run_nightly_self_eval(config, runner=fake_runner)

    status = self_eval.status_snapshot()
    assert status["regression"] is True
    assert "alpha" in status["regressions"]
    assert "beta" not in status["regressions"]
    assert telemetry.snapshot().get("self_eval.regressions") == 1


def test_run_nightly_self_eval_dummy_llm_skips_and_counts_failure(
    tmp_path, monkeypatch
):
    config = _self_eval_config(tmp_path)
    from amadeus.providers.llm.base import DummyProvider

    monkeypatch.setattr("amadeus.provider_loader.load_llm", lambda cfg: DummyProvider())
    calls = []

    def fake_runner(*args, **kwargs):
        calls.append(args)
        return {"alpha": 9}

    self_eval.run_nightly_self_eval(config, runner=fake_runner)
    assert calls == []  # never reached the runner
    assert telemetry.snapshot().get("self_eval.run_failed") == 1
    assert not Path(config.self_eval.trend_path).exists()


def test_run_nightly_self_eval_runner_exception_never_raises(tmp_path, monkeypatch):
    config = _self_eval_config(tmp_path)
    monkeypatch.setattr("amadeus.provider_loader.load_llm", lambda cfg: _FakeLLM())
    monkeypatch.setattr("amadeus.provider_loader.load_embedder", lambda cfg: None)

    def broken_runner(*args, **kwargs):
        raise RuntimeError("boom")

    self_eval.run_nightly_self_eval(config, runner=broken_runner)  # must not raise
    assert telemetry.snapshot().get("self_eval.run_failed") == 1


def test_run_nightly_self_eval_empty_scores_counts_failure(tmp_path, monkeypatch):
    config = _self_eval_config(tmp_path)
    monkeypatch.setattr("amadeus.provider_loader.load_llm", lambda cfg: _FakeLLM())
    monkeypatch.setattr("amadeus.provider_loader.load_embedder", lambda cfg: None)

    self_eval.run_nightly_self_eval(config, runner=lambda *a, **k: {})
    assert telemetry.snapshot().get("self_eval.run_failed") == 1


def test_run_nightly_self_eval_respects_max_calls_per_day(tmp_path, monkeypatch):
    config = _self_eval_config(tmp_path)
    config.self_eval.scenarios = ["alpha", "beta", "gamma", "delta"]
    config.self_eval.max_calls_per_day = 2
    monkeypatch.setattr("amadeus.provider_loader.load_llm", lambda cfg: _FakeLLM())
    monkeypatch.setattr("amadeus.provider_loader.load_embedder", lambda cfg: None)

    seen_names = []

    def fake_runner(cfg, llm, embedder, judge_llm, names, scenarios_path):
        seen_names.extend(names)
        return {n: 9.0 for n in names}

    self_eval.run_nightly_self_eval(config, runner=fake_runner)
    assert seen_names == ["alpha", "beta"]


# ---------------------------------------------------------------------------
# Scheduler job: gating (disabled / hour / once-per-day), crash isolation
# ---------------------------------------------------------------------------


def _scheduler_self_eval_config(tmp_path) -> Config:
    config = Config()
    config.self_eval.enabled = True
    config.self_eval.hour = 4
    config.self_eval.trend_path = str(tmp_path / "trend.jsonl")
    config.memory.persist_path = str(tmp_path / "memory.json")
    config.memory.embeddings_path = str(tmp_path / "embeddings.json")
    config.memory.profile_path = str(tmp_path / "profile.json")
    return config


class _FakeStore:
    def __init__(self, config):
        self.config = config


@pytest.mark.asyncio
async def test_scheduler_self_eval_before_hour_noop(tmp_path, monkeypatch):
    config = _scheduler_self_eval_config(tmp_path)
    store = _FakeStore(config)
    called = []
    monkeypatch.setattr(
        "amadeus.web.scheduler.self_eval_module.run_nightly_self_eval",
        lambda cfg, **kw: called.append(cfg),
    )
    scheduler = BackgroundScheduler(store, config)
    now = dt.datetime(2026, 7, 10, 3, 30)
    await scheduler._self_eval_job(now)
    assert called == []


@pytest.mark.asyncio
async def test_scheduler_self_eval_disabled_noop(tmp_path, monkeypatch):
    config = _scheduler_self_eval_config(tmp_path)
    config.self_eval.enabled = False
    store = _FakeStore(config)
    called = []
    monkeypatch.setattr(
        "amadeus.web.scheduler.self_eval_module.run_nightly_self_eval",
        lambda cfg, **kw: called.append(cfg),
    )
    scheduler = BackgroundScheduler(store, config)
    now = dt.datetime(2026, 7, 10, 5, 0)
    await scheduler._self_eval_job(now)
    assert called == []


@pytest.mark.asyncio
async def test_scheduler_self_eval_runs_once_per_day(tmp_path, monkeypatch):
    config = _scheduler_self_eval_config(tmp_path)
    store = _FakeStore(config)
    calls = []

    def fake_run(cfg, **kw):
        # Simulate the real run appending a trend entry (what a caller of
        # run_nightly_self_eval would actually do), so the job's own
        # last_run_date gate has something to read on the second tick.
        calls.append(1)
        self_eval.append_trend_entry(
            Path(cfg.self_eval.trend_path),
            {"timestamp": dt.datetime(2026, 7, 10, 4, 5).isoformat(), "scores": {}},
        )

    monkeypatch.setattr(
        "amadeus.web.scheduler.self_eval_module.run_nightly_self_eval", fake_run
    )
    scheduler = BackgroundScheduler(store, config)
    now = dt.datetime(2026, 7, 10, 4, 5)
    await scheduler._self_eval_job(now)
    assert len(calls) == 1

    # Same-day second tick: no rerun.
    await scheduler._self_eval_job(now + dt.timedelta(minutes=10))
    assert len(calls) == 1

    # A fresh scheduler instance (simulating a restart) also respects the
    # trend file's last-run-date gate.
    fresh = BackgroundScheduler(store, config)
    await fresh._self_eval_job(now + dt.timedelta(hours=1))
    assert len(calls) == 1

    # Next day reruns.
    await scheduler._self_eval_job(now + dt.timedelta(days=1))
    assert len(calls) == 2


@pytest.mark.asyncio
async def test_scheduler_self_eval_survives_job_exception(tmp_path, monkeypatch):
    """_tick() must not raise even if the self-eval job explodes — same
    exception-isolation contract as the brief/diary/reach-out jobs."""
    config = _scheduler_self_eval_config(tmp_path)
    store = _FakeStore(config)

    def boom(cfg, **kw):
        raise RuntimeError("scheduler explosion")

    monkeypatch.setattr(
        "amadeus.web.scheduler.self_eval_module.run_nightly_self_eval", boom
    )
    scheduler = BackgroundScheduler(
        store, config, clock=lambda: dt.datetime(2026, 7, 10, 5, 0)
    )
    await scheduler._tick()  # must not raise


@pytest.mark.asyncio
async def test_scheduler_self_eval_independent_of_brief_diary_reachout():
    """self_eval alone (brief/diary/reach_out all off) must still start the
    scheduler — the lifespan gate must OR them all together."""
    config = Config()
    config.self_eval.enabled = True
    # Explicitly off (diary/follow-through/life-threads default ON since
    # 2026-07-28): the point is self_eval ALONE starting the scheduler.
    config.daily_brief.enabled = False
    config.diary.enabled = False
    config.reach_out.enabled = False
    config.follow_through.enabled = False
    config.life_threads.enabled = False


# ---------------------------------------------------------------------------
# GET /api/status surfacing
# ---------------------------------------------------------------------------


def _isolated_status_config(tmp_path) -> Config:
    config = Config()
    config.memory.persist_path = str(tmp_path / "memory.json")
    config.memory.embeddings_path = str(tmp_path / "embeddings.json")
    config.memory.profile_path = str(tmp_path / "profile.json")
    return config


def test_status_self_eval_default_not_yet_run(tmp_path):
    app = create_app(_isolated_status_config(tmp_path))
    client = TestClient(app)
    payload = client.get("/api/status").json()
    assert payload["self_eval"] == {
        "last_run_at": None,
        "scores": None,
        "regression": False,
        "regressions": None,
    }


def test_status_self_eval_reflects_last_run(tmp_path):
    self_eval.record_run(scores={"alpha": 9.0}, regressions={})
    app = create_app(_isolated_status_config(tmp_path))
    client = TestClient(app)
    payload = client.get("/api/status").json()
    assert payload["self_eval"]["scores"] == {"alpha": 9.0}
    assert payload["self_eval"]["regression"] is False


def test_status_self_eval_reflects_regression(tmp_path):
    self_eval.record_run(
        scores={"alpha": 3.0},
        regressions={"alpha": {"latest": 3.0, "median": 9.0, "drop": 6.0}},
    )
    app = create_app(_isolated_status_config(tmp_path))
    client = TestClient(app)
    payload = client.get("/api/status").json()
    assert payload["self_eval"]["regression"] is True
    assert "alpha" in payload["self_eval"]["regressions"]


# ---------------------------------------------------------------------------
# Lifespan gate: self_eval alone starts the scheduler
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_lifespan_starts_scheduler_for_self_eval_alone(tmp_path):
    config = _isolated_status_config(tmp_path)
    config.self_eval.enabled = True
    app = create_app(config)
    async with app.router.lifespan_context(app):
        assert app.state.scheduler is not None
        assert app.state.scheduler._task is not None
    assert app.state.scheduler._task is None


@pytest.mark.asyncio
async def test_lifespan_no_scheduler_when_everything_off(tmp_path):
    config = _isolated_status_config(tmp_path)
    # diary/brief/follow-through/life-threads default ON since 2026-07-28 —
    # "everything off" now requires saying so.
    config.daily_brief.enabled = False
    config.diary.enabled = False
    config.reach_out.enabled = False
    config.follow_through.enabled = False
    config.life_threads.enabled = False
    config.insights.enabled = False
    config.self_eval.enabled = False
    app = create_app(config)
    async with app.router.lifespan_context(app):
        assert app.state.scheduler is None
