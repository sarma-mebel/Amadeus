"""Longitudinal simulation (amadeus/timeshift.py +
scripts/longitudinal_sim.py): weeks of time-dependent behavior observed in
one offline run.

Two layers. The timeshift unit tests pin the aging rules — which strings
shift, which never may (fact TEXT that happens to contain a date), and the
date-only key/container rules. The end-to-end test runs one 30-simulated-day
scenario plus a 200-day gap and asserts the whole family of time-dependent
features actually fired: session-count/trust drift, fact aging, store-time
supersede vs consolidation, friction's 21-day heal, next-time-thread
consumption, daily scheduler cadence, and organic forgetting. It is the
closest thing the suite has to living with her for a month.

NO live LLM calls: the sim's RoutedLLM is content-routed on system-prompt
markers (the tests/test_eval_harness.py pattern) and the embedder is the
dependency-free hashing one.
"""

from __future__ import annotations

import datetime as dt
import importlib.util
import json
import sys
from pathlib import Path

import pytest

from amadeus.timeshift import shift_file, shift_stamp, shift_structure, shift_tree

REPO_ROOT = Path(__file__).resolve().parents[1]

_DAY = dt.timedelta(days=1)


def _load_sim_module():
    spec = importlib.util.spec_from_file_location(
        "longitudinal_sim_test", REPO_ROOT / "scripts" / "longitudinal_sim.py"
    )
    module = importlib.util.module_from_spec(spec)
    # Register before exec: the sim defines dataclasses, and dataclass
    # creation resolves the defining module through sys.modules (py3.13).
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# timeshift unit tests
# ---------------------------------------------------------------------------


def test_shift_stamp_datetime_always_shifts() -> None:
    assert shift_stamp("2026-08-07T09:15:00", _DAY, allow_date=False) == (
        "2026-08-06T09:15:00"
    )
    # Microseconds and space separators survive round-trip.
    assert shift_stamp("2026-08-07 09:15:00.123456", _DAY, allow_date=False) == (
        "2026-08-06T09:15:00.123456"
    )


def test_shift_stamp_date_only_needs_permission() -> None:
    assert shift_stamp("2026-08-07", _DAY, allow_date=False) is None
    assert shift_stamp("2026-08-07", _DAY, allow_date=True) == "2026-08-06"
    assert shift_stamp("not a date", _DAY, allow_date=True) is None


def test_content_keys_are_never_shifted() -> None:
    # A fact whose TEXT is exactly a date must never be aged into a
    # different fact — that would silently corrupt the store's content.
    data = {
        "text": "2026-08-07T09:15:00",
        "timestamp": "2026-08-07T09:15:00",
    }
    shifted, count = shift_structure(data, _DAY)
    assert shifted["text"] == "2026-08-07T09:15:00"
    assert shifted["timestamp"] == "2026-08-06T09:15:00"
    assert count == 1


def test_date_keys_and_containers_shift() -> None:
    data = {
        "tone_history": [{"date": "2026-08-07", "tone": 0.4}],
        "friction_topics": [{"topic": "the review", "noted": "2026-08-07"}],
        "open_thread_dates": {"the deadline": "2026-08-07"},
        "facts": {"job": "researcher"},
    }
    shifted, count = shift_structure(data, _DAY)
    assert shifted["tone_history"][0]["date"] == "2026-08-06"
    assert shifted["friction_topics"][0]["noted"] == "2026-08-06"
    assert shifted["open_thread_dates"]["the deadline"] == "2026-08-06"
    assert shifted["friction_topics"][0]["topic"] == "the review"
    assert shifted["facts"]["job"] == "researcher"
    assert count == 3


def test_life_thread_pace_stamps_shift() -> None:
    # Regression: these date-only keys were missing from DATE_KEYS, so
    # after the sim's day 0 acted_today() stayed true forever and life
    # threads froze (a 14-day run showed 1 LLM call, 0 progressions).
    data = {
        "last_activity_date": "2026-08-07",
        "last_ask_date": "2026-08-06",
        "threads": [
            {
                "topic": "re-derive the timing analysis",
                "summary": "the notation needs untangling",
                "started_at": "2026-08-01",
                "last_progressed_at": "2026-08-07",
                "asked_at": "",
            }
        ],
    }
    shifted, count = shift_structure(data, _DAY)
    assert shifted["last_activity_date"] == "2026-08-06"
    assert shifted["last_ask_date"] == "2026-08-05"
    thread = shifted["threads"][0]
    assert thread["started_at"] == "2026-07-31"
    assert thread["last_progressed_at"] == "2026-08-06"
    assert thread["asked_at"] == ""  # never-asked stays never-asked
    assert thread["topic"] == "re-derive the timing analysis"
    assert count == 4


def test_first_met_date_shifts() -> None:
    shifted, count = shift_structure({"first_met": "2026-08-07"}, _DAY)
    assert shifted["first_met"] == "2026-08-06"
    assert count == 1


def test_date_shaped_dict_keys_shift() -> None:
    data = {"days": {"2026-08-07": {"total": 3}}}
    shifted, _ = shift_structure(data, _DAY)
    assert "2026-08-06" in shifted["days"]


def test_shift_file_handles_jsonl_and_skips_corrupt(tmp_path: Path) -> None:
    jsonl = tmp_path / "archive.jsonl"
    jsonl.write_text(
        json.dumps({"archived_at": "2026-08-07T09:00:00", "text": "old"}) + "\n",
        encoding="utf-8",
    )
    assert shift_file(jsonl, _DAY) == 1
    line = json.loads(jsonl.read_text(encoding="utf-8"))
    assert line["archived_at"] == "2026-08-06T09:00:00"

    corrupt = tmp_path / "broken.json"
    corrupt.write_text("{not json", encoding="utf-8")
    assert shift_file(corrupt, _DAY) == 0
    assert corrupt.read_text(encoding="utf-8") == "{not json"


def test_shift_tree_skips_backups(tmp_path: Path) -> None:
    live = tmp_path / "profile.json"
    live.write_text(json.dumps({"last_seen": "2026-08-07T09:00:00"}))
    backup = tmp_path / "profile.json.bak-corrupt"
    backup.write_text(json.dumps({"last_seen": "2026-08-07T09:00:00"}))
    assert shift_tree(tmp_path, 1) == 1
    assert "2026-08-06" in live.read_text()
    assert "2026-08-07" in backup.read_text()


# ---------------------------------------------------------------------------
# End-to-end: one simulated month + a 200-day gap
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def month_report():
    sim = _load_sim_module()
    return sim.run_simulation(days=30, final_gap_days=200)


def test_sessions_and_trust_drift_across_weeks(month_report) -> None:
    final = month_report["final"]
    assert final["session_count"] >= 30
    assert final["trust_level"] > 0.7
    assert final["tone_history_len"] > 5


def test_facts_age_with_the_store(month_report) -> None:
    last_day = month_report["days"][-1]["state"]
    assert last_day["fact_records"] >= 3
    assert last_day["oldest_fact_age_days"] >= 25


def test_consolidation_merged_a_paraphrase_cluster(month_report) -> None:
    merged = sum(
        entry["counters"].get("memory.consolidation_merged", 0)
        for entry in month_report["days"]
    )
    assert merged >= 1
    # The surviving coffee fact is the consolidated one, and it is what
    # recall surfaces before the gap.
    assert month_report["final"]["recall_coffee_before_gap"]


def test_friction_notes_then_heals_at_21_days(month_report) -> None:
    days = month_report["days"]
    had_friction = [entry["day"] for entry in days if entry["state"]["friction_topics"]]
    assert had_friction, "friction was never noted"
    assert min(had_friction) == 5
    # Healed before the run ends: 21 days after the day-5 note.
    assert not days[-1]["state"]["friction_topics"]
    assert max(had_friction) < len(days) - 1


def test_next_time_thread_is_noted_then_consumed(month_report) -> None:
    days = month_report["days"]
    assert days[2]["state"]["next_time_thread"] == "their conference deadline"
    later = [entry["state"]["next_time_thread"] for entry in days[4:]]
    assert all(thread is None for thread in later)


def test_daily_jobs_fire_each_day(month_report) -> None:
    # Session days after warm-up should carry that day's brief and diary.
    for entry in month_report["days"][3:6]:
        if entry["session"]:
            assert entry["state"]["brief_today"] is True
            assert entry["state"]["diary_today"] is True
    generated = sum(
        entry["counters"].get("insights.generated", 0) for entry in month_report["days"]
    )
    assert generated >= 10


def test_life_threads_progress_across_the_month(month_report) -> None:
    # Regression: with the life-thread pace stamps unshifted, the store
    # froze after day 0 — exactly one LLM call ever, zero progressions.
    calls = month_report["final"]["llm_calls"]
    assert calls.get("life_invent_reply", 0) >= 1
    assert calls.get("life_step_reply", 0) >= 2
    threads = month_report["final"]["life_threads"]
    assert any(t["progressions"] >= 2 for t in threads)


def test_first_met_ages_with_the_store(month_report) -> None:
    # Regression: first_met is a date-only stamp; unshifted it stayed
    # "today" forever and anniversary milestones could never fire in a sim.
    first_met = month_report["final"]["first_met"]
    assert first_met
    met = dt.date.fromisoformat(str(first_met)[:10])
    assert (dt.date.today() - met).days >= 200


def test_sim_writes_nothing_outside_its_data_dir(tmp_path, monkeypatch) -> None:
    # Regression: persona_versions_path kept its CWD-relative default, so
    # every sim run rewrote the repo root's persona_versions.json.
    sim = _load_sim_module()
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    data_dir = tmp_path / "data"
    monkeypatch.chdir(cwd)
    sim.run_simulation(days=2, data_dir=data_dir)
    assert list(cwd.iterdir()) == []
    assert (data_dir / "persona_versions.json").exists()


def test_reachout_generated_during_absence(month_report) -> None:
    assert month_report["final"]["reachout_line"]


def test_relational_memory_skips_ordinary_sessions(month_report) -> None:
    skipped = sum(
        entry["counters"].get("memory.relational_skipped", 0)
        for entry in month_report["days"]
    )
    assert skipped >= 10


def test_forgetting_drops_unreinforced_facts_after_the_gap(month_report) -> None:
    assert month_report["final"]["recall_coffee_before_gap"]
    assert month_report["final"]["recall_coffee"] == []
    # Nothing was deleted — the records still exist, decayed below the
    # recall floor (archive-not-delete applies to forgetting too).
    assert month_report["final"]["fact_records"] >= 3
