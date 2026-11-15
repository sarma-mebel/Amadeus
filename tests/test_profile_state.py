"""Persisted relationship state: UserProfile v2 (mood/trust) + trust dynamics."""

import datetime as dt
import json
from pathlib import Path

from amadeus.config import Config
from amadeus.engine import ChatEngine
from amadeus.memory.manager import MemoryManager
from amadeus.memory.profile import (
    DEFAULT_TRUST,
    PROFILE_VERSION,
    TONE_HISTORY_CAP,
    UserProfile,
    greeting_register_facts,
    milestone_events,
    milestone_facts,
    parse_birthday,
    quantize_trust,
    tone_trend,
    updated_arc,
    updated_trust,
    user_birthday_from_facts,
)
from amadeus.providers.llm.base import DummyProvider


def _v1_profile(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "first_met": "2026-01-01",
                "session_count": 4,
                "last_seen": "2026-01-02T00:00:00",
                "facts": {"name": "Alex"},
                "open_threads": [],
                "last_episode": None,
            }
        )
    )


def test_v1_profile_loads_with_state_defaults(tmp_path: Path) -> None:
    path = tmp_path / "profile.json"
    _v1_profile(path)
    profile = UserProfile.load(path)
    assert profile.session_count == 4
    assert profile.mood is None
    assert profile.trust_level == DEFAULT_TRUST


def test_v2_profile_roundtrips_mood_and_trust(tmp_path: Path) -> None:
    path = tmp_path / "profile.json"
    profile = UserProfile(mood="curious", trust_level=0.63)
    profile.save_atomic(path)
    restored = UserProfile.load(path)
    assert restored.mood == "curious"
    assert restored.trust_level == 0.63


def test_profile_load_clamps_bad_trust(tmp_path: Path) -> None:
    path = tmp_path / "profile.json"
    path.write_text(json.dumps({"version": 2, "trust_level": 7.5}))
    assert UserProfile.load(path).trust_level == 1.0
    path.write_text(json.dumps({"version": 2, "trust_level": "garbage"}))
    assert UserProfile.load(path).trust_level == DEFAULT_TRUST


def test_updated_trust_session_ramp_and_cap() -> None:
    # Fresh relationship: floor barely above the default.
    assert updated_trust(DEFAULT_TRUST, 1, None) == 0.5 + 0.015
    # Long relationship: floor caps at +0.25.
    assert updated_trust(DEFAULT_TRUST, 100, None) == 0.75
    # Trust never exceeds 0.9 even with warm tone forever.
    assert updated_trust(0.9, 100, 1.0) == 0.9


def test_updated_trust_tone_bonus_and_floor() -> None:
    # Warm tone nudges up; the nudges accumulate at full precision.
    warm = updated_trust(0.70, 5, 0.5)
    assert warm == 0.71
    # Hostile tone nudges down, but never below the familiarity floor.
    hostile = updated_trust(0.70, 5, -0.9)
    assert hostile == 0.70 - 0.005
    floor = 0.5 + 0.015 * 5
    assert updated_trust(floor, 5, -0.9) == floor  # clamped at floor


def test_quantize_trust() -> None:
    assert quantize_trust(0.63) == 0.65
    assert quantize_trust(0.71) == 0.70
    assert quantize_trust("junk") == DEFAULT_TRUST  # type: ignore[arg-type]


def _profile_config(tmp_path: Path) -> Config:
    config = Config()
    config.memory.persist_path = None
    config.memory.embeddings_path = None
    config.memory.enable_long_term = False
    config.memory.enable_fact_extraction = True
    config.memory.enable_episodes = False
    config.memory.profile_path = str(tmp_path / "profile.json")
    return config


def test_session_boundary_updates_trust_on_disk(tmp_path: Path) -> None:
    config = _profile_config(tmp_path)
    seeded = UserProfile(session_count=3, last_seen="2020-01-01T00:00:00")
    seeded.save_atomic(Path(config.memory.profile_path))

    manager = MemoryManager(config)  # stale last_seen -> new session counted
    assert manager.profile.session_count == 4
    assert manager.profile.trust_level == updated_trust(DEFAULT_TRUST, 4, None)

    on_disk = UserProfile.load(Path(config.memory.profile_path))
    assert on_disk.trust_level == manager.profile.trust_level


def test_engine_hydrates_and_records_state(tmp_path: Path) -> None:
    config = _profile_config(tmp_path)
    profile_path = Path(config.memory.profile_path)
    seeded = UserProfile(mood="affectionate", trust_level=0.72)
    # Fresh last_seen so hydration isn't perturbed by a session bump.
    import datetime as dt

    seeded.last_seen = dt.datetime.utcnow().isoformat()
    seeded.session_count = 9
    seeded.save_atomic(profile_path)

    engine = ChatEngine(config, DummyProvider())
    assert engine.kurisu_state.mood == "affectionate"
    assert engine.kurisu_state.trust_level == quantize_trust(0.72)

    # complete_turn stashes the (normalized) mood back onto the profile.
    engine.kurisu_state.mood = "curious"
    engine.complete_turn("tell me about the lab", "It's a mess, as usual.")
    assert engine.memory.profile.mood == "curious"


def test_engine_without_profile_keeps_defaults(tmp_path: Path) -> None:
    config = Config()
    config.memory.persist_path = None
    config.memory.embeddings_path = None
    config.memory.enable_long_term = False
    config.memory.enable_fact_extraction = False
    config.memory.enable_episodes = False
    config.memory.profile_path = None

    engine = ChatEngine(config, DummyProvider())
    assert engine.kurisu_state.trust_level == DEFAULT_TRUST
    assert engine.kurisu_state.mood == "focused"
    # record_mood is a no-op without a profile.
    engine.complete_turn("hi", "Hello.")


# --- P1: v3 arc floats + tone_history -----------------------------------


def test_v2_profile_loads_with_v3_defaults(tmp_path: Path) -> None:
    path = tmp_path / "profile.json"
    path.write_text(
        json.dumps(
            {
                "version": 2,
                "first_met": "2026-01-01",
                "session_count": 4,
                "last_seen": "2026-01-02T00:00:00",
                "facts": {"name": "Alex"},
                "open_threads": [],
                "last_episode": None,
                "mood": "curious",
                "trust_level": 0.6,
            }
        )
    )
    profile = UserProfile.load(path)
    assert profile.arc_curiosity_vs_annoyance == 0.0
    assert profile.arc_logic_vs_emotion == 0.0
    assert profile.arc_pride_vs_vulnerability == 0.0
    assert profile.tone_history == []


def test_v3_profile_roundtrips_arc_and_tone_history(tmp_path: Path) -> None:
    path = tmp_path / "profile.json"
    profile = UserProfile(
        arc_curiosity_vs_annoyance=0.4,
        arc_logic_vs_emotion=-0.2,
        arc_pride_vs_vulnerability=0.1,
        tone_history=[{"date": "2026-07-01", "tone": 0.5}],
    )
    profile.save_atomic(path)
    restored = UserProfile.load(path)
    assert restored.arc_curiosity_vs_annoyance == 0.4
    assert restored.arc_logic_vs_emotion == -0.2
    assert restored.arc_pride_vs_vulnerability == 0.1
    assert restored.tone_history == [{"date": "2026-07-01", "tone": 0.5}]


def test_profile_load_clamps_arc_values(tmp_path: Path) -> None:
    path = tmp_path / "profile.json"
    path.write_text(
        json.dumps(
            {
                "version": 3,
                "arc_curiosity_vs_annoyance": 5.0,
                "arc_logic_vs_emotion": -5.0,
                "arc_pride_vs_vulnerability": "garbage",
            }
        )
    )
    profile = UserProfile.load(path)
    assert profile.arc_curiosity_vs_annoyance == 1.0
    assert profile.arc_logic_vs_emotion == -1.0
    assert profile.arc_pride_vs_vulnerability == 0.0


def test_profile_load_drops_malformed_tone_history_entries(tmp_path: Path) -> None:
    path = tmp_path / "profile.json"
    path.write_text(
        json.dumps(
            {
                "version": 3,
                "tone_history": [
                    {"date": "2026-07-01", "tone": 0.5},
                    {"date": "2026-07-02", "tone": "garbage"},
                    {"date": "2026-07-03"},
                    {"tone": 0.2},
                    "not-a-dict",
                    {"date": "2026-07-04", "tone": 5.0},
                ],
            }
        )
    )
    profile = UserProfile.load(path)
    assert profile.tone_history == [
        {"date": "2026-07-01", "tone": 0.5},
        {"date": "2026-07-04", "tone": 1.0},
    ]


def test_record_tone_caps_at_history_limit() -> None:
    profile = UserProfile()
    base = dt.date(2026, 1, 1)
    for i in range(25):
        profile.record_tone(0.1, when=base + dt.timedelta(days=i))
    assert len(profile.tone_history) == TONE_HISTORY_CAP
    # Oldest entries (days 0-4) were dropped; the earliest remaining is day 5.
    assert profile.tone_history[0]["date"] == (base + dt.timedelta(days=5)).isoformat()
    assert (
        profile.tone_history[-1]["date"] == (base + dt.timedelta(days=24)).isoformat()
    )


def test_record_tone_clamps_and_defaults_on_bad_input() -> None:
    profile = UserProfile()
    profile.record_tone(3.0, when=dt.date(2026, 1, 1))
    profile.record_tone("garbage", when=dt.date(2026, 1, 2))  # type: ignore[arg-type]
    assert profile.tone_history[0]["tone"] == 1.0
    assert profile.tone_history[1]["tone"] == 0.0


def test_updated_arc_warm_tone_direction() -> None:
    curiosity, logic, pride = updated_arc(
        0.0, 0.0, 0.0, session_count=3, trust_level=0.5, last_tone=0.5
    )
    assert curiosity == 0.02
    assert logic == -0.01
    assert pride == -0.01


def test_updated_arc_hostile_tone_direction() -> None:
    curiosity, logic, pride = updated_arc(
        0.0, 0.0, 0.0, session_count=3, trust_level=0.5, last_tone=-0.9
    )
    assert curiosity == -0.02
    assert logic == 0.0
    assert pride == 0.01


def test_updated_arc_high_trust_drift_with_no_tone() -> None:
    curiosity, logic, pride = updated_arc(
        0.0, 0.0, 0.0, session_count=3, trust_level=0.7, last_tone=None
    )
    assert curiosity == 0.0
    assert logic == -0.005
    assert pride == -0.005


def test_updated_arc_no_movement_without_signal() -> None:
    curiosity, logic, pride = updated_arc(
        0.1, 0.1, 0.1, session_count=3, trust_level=0.5, last_tone=None
    )
    assert (curiosity, logic, pride) == (0.1, 0.1, 0.1)


def test_updated_arc_clamps_at_bounds() -> None:
    curiosity, logic, pride = updated_arc(
        0.99, -0.99, -0.99, session_count=3, trust_level=0.9, last_tone=0.9
    )
    assert curiosity == 1.0
    assert logic == -1.0
    assert pride == -1.0

    curiosity, logic, pride = updated_arc(
        -0.99, 0.99, 0.99, session_count=3, trust_level=0.5, last_tone=-0.9
    )
    assert curiosity == -1.0
    assert pride == 1.0


def test_updated_arc_garbage_inputs_do_not_raise() -> None:
    curiosity, logic, pride = updated_arc(
        "junk",  # type: ignore[arg-type]
        None,  # type: ignore[arg-type]
        object(),  # type: ignore[arg-type]
        session_count=3,
        trust_level="also junk",  # type: ignore[arg-type]
        last_tone=0.5,
    )
    # Coerced to 0.0 then the warm-tone deltas apply.
    assert curiosity == 0.02
    assert logic == -0.01
    assert pride == -0.01


def test_tone_trend_needs_minimum_entries() -> None:
    today = dt.date(2026, 7, 12)
    history = [
        {"date": "2026-07-11", "tone": -0.5},
        {"date": "2026-07-12", "tone": -0.5},
    ]
    assert tone_trend(history, today=today) is None


def test_tone_trend_low() -> None:
    today = dt.date(2026, 7, 12)
    history = [
        {"date": "2026-07-10", "tone": -0.4},
        {"date": "2026-07-11", "tone": -0.3},
        {"date": "2026-07-12", "tone": -0.2},
    ]
    assert tone_trend(history, today=today) == "low"


def test_tone_trend_excludes_entries_outside_window() -> None:
    today = dt.date(2026, 7, 12)
    history = [
        {"date": "2026-06-01", "tone": -0.9},  # outside the 7-day window
        {"date": "2026-06-02", "tone": -0.9},
        {"date": "2026-07-11", "tone": -0.4},
        {"date": "2026-07-12", "tone": -0.4},
    ]
    # Only 2 entries survive the window filter -> below min_entries.
    assert tone_trend(history, today=today) is None


def test_tone_trend_positive() -> None:
    today = dt.date(2026, 7, 12)
    history = [
        {"date": "2026-07-10", "tone": 0.5},
        {"date": "2026-07-11", "tone": 0.4},
        {"date": "2026-07-12", "tone": 0.3},
    ]
    assert tone_trend(history, today=today) == "positive"


def test_tone_trend_mixed_returns_none() -> None:
    today = dt.date(2026, 7, 12)
    history = [
        {"date": "2026-07-10", "tone": 0.5},
        {"date": "2026-07-11", "tone": -0.5},
        {"date": "2026-07-12", "tone": 0.0},
    ]
    assert tone_trend(history, today=today) is None


def test_tone_trend_never_raises_on_garbage() -> None:
    today = dt.date(2026, 7, 12)
    history = [
        {"date": "not-a-date", "tone": 0.5},
        {"date": "2026-07-11", "tone": "garbage"},
        "not-a-dict",
        {"tone": 0.5},
        None,
    ]
    assert tone_trend(history, today=today) is None  # type: ignore[arg-type]


def test_milestone_facts_session_count_exact_match() -> None:
    assert milestone_facts(10, None) == [
        "Milestone: this is conversation number 10 together."
    ]
    assert milestone_facts(11, None) == []


def test_milestone_facts_one_month_anniversary() -> None:
    facts = milestone_facts(3, "2026-06-12", today=dt.date(2026, 7, 12))
    assert "Milestone: today marks 1 month since you first met." in facts


def test_milestone_facts_six_month_anniversary() -> None:
    facts = milestone_facts(3, "2026-01-12", today=dt.date(2026, 7, 12))
    assert "Milestone: today marks 6 months since you first met." in facts


def test_milestone_facts_one_year_anniversary() -> None:
    facts = milestone_facts(3, "2025-07-12", today=dt.date(2026, 7, 12))
    assert "Milestone: today marks 1 year since you first met." in facts


def test_milestone_facts_two_year_anniversary() -> None:
    facts = milestone_facts(3, "2024-07-12", today=dt.date(2026, 7, 12))
    assert "Milestone: today marks 2 years since you first met." in facts


def test_milestone_facts_jan31_first_met_clamps_to_end_of_february() -> None:
    facts = milestone_facts(3, "2026-01-31", today=dt.date(2026, 2, 28))
    assert "Milestone: today marks 1 month since you first met." in facts


def test_milestone_facts_no_first_met_returns_empty() -> None:
    assert milestone_facts(3, None) == []


# --- milestone_events (structured form underlying milestone_facts) --------


def test_milestone_events_session_count_fifty_yields_one_event() -> None:
    events = milestone_events(50, None)
    assert events == [
        {
            "kind": "sessions",
            "value": 50,
            "label": "this is conversation number 50 together",
        }
    ]


def test_milestone_events_session_count_fifty_one_yields_none() -> None:
    assert milestone_events(51, None) == []


def test_milestone_events_first_met_exactly_one_year_ago() -> None:
    today = dt.date(2026, 7, 12)
    events = milestone_events(3, "2025-07-12", today=today)
    assert events == [
        {
            "kind": "anniversary",
            "months": 12,
            "label": "today marks 1 year since you first met",
            "date": "2026-07-12",
        }
    ]


def test_milestone_events_both_session_and_anniversary() -> None:
    today = dt.date(2026, 7, 12)
    events = milestone_events(50, "2025-07-12", today=today)
    assert len(events) == 2
    kinds = {event["kind"] for event in events}
    assert kinds == {"sessions", "anniversary"}


def test_milestone_events_garbage_first_met_yields_sessions_only() -> None:
    events = milestone_events(50, "not-a-date")
    assert events == [
        {
            "kind": "sessions",
            "value": 50,
            "label": "this is conversation number 50 together",
        }
    ]


def test_milestone_facts_matches_pre_refactor_phrasing() -> None:
    """Pinned literals from before the milestone_events extraction — guards
    against milestone_facts drifting from its historical output shape."""
    assert milestone_facts(10, None) == [
        "Milestone: this is conversation number 10 together."
    ]
    assert milestone_facts(25, None) == [
        "Milestone: this is conversation number 25 together."
    ]
    assert milestone_facts(3, "2026-06-12", today=dt.date(2026, 7, 12)) == [
        "Milestone: today marks 1 month since you first met."
    ]
    assert milestone_facts(3, "2026-01-12", today=dt.date(2026, 7, 12)) == [
        "Milestone: today marks 6 months since you first met."
    ]
    assert milestone_facts(3, "2025-07-12", today=dt.date(2026, 7, 12)) == [
        "Milestone: today marks 1 year since you first met."
    ]
    assert milestone_facts(3, "2024-07-12", today=dt.date(2026, 7, 12)) == [
        "Milestone: today marks 2 years since you first met."
    ]


# --- P3: session-boundary arc drift + tone history + overview/reset ------


def test_session_boundary_with_tone_moves_arc_and_appends_tone_on_disk(
    tmp_path: Path,
) -> None:
    config = _profile_config(tmp_path)
    seeded = UserProfile(session_count=3, last_seen="2020-01-01T00:00:00")
    seeded.save_atomic(Path(config.memory.profile_path))

    manager = MemoryManager(config)
    manager._last_tone = 0.5  # simulate a warm tone available at the boundary
    # Force a fresh boundary evaluation as if this were a new process launch.
    manager.profile.last_seen = "2020-01-01T00:00:00"
    manager._maybe_begin_session()

    assert manager.profile.session_count == 5
    curiosity, logic, pride = updated_arc(
        0.0,
        0.0,
        0.0,
        session_count=5,
        trust_level=manager.profile.trust_level,
        last_tone=0.5,
    )
    assert manager.profile.arc_curiosity_vs_annoyance == curiosity
    assert manager.profile.arc_logic_vs_emotion == logic
    assert manager.profile.arc_pride_vs_vulnerability == pride
    assert len(manager.profile.tone_history) == 1
    assert manager.profile.tone_history[0]["tone"] == 0.5

    on_disk = UserProfile.load(Path(config.memory.profile_path))
    assert on_disk.arc_curiosity_vs_annoyance == curiosity
    assert on_disk.arc_logic_vs_emotion == logic
    assert on_disk.arc_pride_vs_vulnerability == pride
    assert on_disk.tone_history == manager.profile.tone_history


def test_session_boundary_without_tone_and_low_trust_leaves_arc_and_tone_untouched(
    tmp_path: Path,
) -> None:
    config = _profile_config(tmp_path)
    seeded = UserProfile(
        session_count=3, last_seen="2020-01-01T00:00:00", trust_level=0.5
    )
    seeded.save_atomic(Path(config.memory.profile_path))

    manager = MemoryManager(config)  # no tone recorded yet -> self._last_tone is None
    assert manager._last_tone is None

    assert manager.profile.arc_curiosity_vs_annoyance == 0.0
    assert manager.profile.arc_logic_vs_emotion == 0.0
    assert manager.profile.arc_pride_vs_vulnerability == 0.0
    assert manager.profile.tone_history == []

    on_disk = UserProfile.load(Path(config.memory.profile_path))
    assert on_disk.arc_curiosity_vs_annoyance == 0.0
    assert on_disk.arc_logic_vs_emotion == 0.0
    assert on_disk.arc_pride_vs_vulnerability == 0.0
    assert on_disk.tone_history == []


def _iso_days_ago(days: int) -> str:
    return (dt.datetime.utcnow() - dt.timedelta(days=days)).isoformat()


def _gap_config(tmp_path: Path) -> Config:
    config = Config()
    config.memory.persist_path = None
    config.memory.embeddings_path = None
    config.memory.enable_long_term = False
    config.memory.enable_fact_extraction = False
    config.memory.enable_episodes = True
    config.memory.session_gap_minutes = 30
    config.memory.summary_trigger_messages = 50
    config.memory.profile_path = str(tmp_path / "profile.json")
    return config


def test_gap_boundary_applies_drift_and_tone_append_exactly_once(
    tmp_path: Path,
) -> None:
    config = _gap_config(tmp_path)
    manager = MemoryManager(config)
    sessions_before = manager.profile.session_count

    manager.add_message("user", "hello")
    manager.flush()
    manager._last_tone = 0.6
    manager.short_term[-1].timestamp = _iso_days_ago(1)  # simulate a long silence
    manager.add_message("user", "back again")
    manager.flush()

    assert manager.profile.session_count == sessions_before + 1
    assert len(manager.profile.tone_history) == 1
    assert manager.profile.tone_history[0]["tone"] == 0.6
    assert manager.profile.arc_curiosity_vs_annoyance > 0.0

    # A second add_message without another gap must not re-apply drift/tone.
    manager.add_message("assistant", "hi again")
    manager.flush()
    assert len(manager.profile.tone_history) == 1


def test_memory_overview_relationship_shape(tmp_path: Path) -> None:
    config = _profile_config(tmp_path)
    manager = MemoryManager(config)
    manager.profile.arc_curiosity_vs_annoyance = 0.123456
    manager.profile.arc_logic_vs_emotion = -0.2
    manager.profile.arc_pride_vs_vulnerability = 0.05
    # Relative dates: tone_trend only counts entries inside its 7-day window,
    # so hardcoded dates rot the test as the calendar advances.
    today = dt.date.today()
    manager.profile.tone_history = [
        {"date": (today - dt.timedelta(days=2)).isoformat(), "tone": 0.5},
        {"date": (today - dt.timedelta(days=1)).isoformat(), "tone": 0.4},
        {"date": today.isoformat(), "tone": 0.3},
    ]

    overview = manager.memory_overview()
    relationship = overview["relationship"]
    assert relationship["trust_level"] == round(manager.profile.trust_level, 3)
    assert relationship["arc"] == {
        "curiosity_vs_annoyance": 0.123,
        "logic_vs_emotion": -0.2,
        "pride_vs_vulnerability": 0.05,
    }
    assert relationship["tone_history"] == manager.profile.tone_history
    assert relationship["tone_trend"] == "positive"


def test_memory_overview_relationship_default_shape_without_profile() -> None:
    config = Config()
    config.memory.persist_path = None
    config.memory.embeddings_path = None
    config.memory.enable_fact_extraction = False
    config.memory.enable_episodes = False
    config.memory.profile_path = None

    manager = MemoryManager(config)
    assert manager.profile is None
    overview = manager.memory_overview()
    assert overview["relationship"] == {}


def test_reset_relationship_zeros_arc_and_tone_preserves_trust(
    tmp_path: Path,
) -> None:
    config = _profile_config(tmp_path)
    manager = MemoryManager(config)
    manager.profile.arc_curiosity_vs_annoyance = 0.4
    manager.profile.arc_logic_vs_emotion = -0.3
    manager.profile.arc_pride_vs_vulnerability = 0.2
    manager.profile.trust_level = 0.77
    manager.profile.tone_history = [{"date": "2026-07-01", "tone": 0.5}]
    manager._save_profile()

    result = manager.reset_relationship()
    assert result is True
    assert manager.profile.arc_curiosity_vs_annoyance == 0.0
    assert manager.profile.arc_logic_vs_emotion == 0.0
    assert manager.profile.arc_pride_vs_vulnerability == 0.0
    assert manager.profile.tone_history == []
    assert manager.profile.trust_level == 0.77

    on_disk = UserProfile.load(Path(config.memory.profile_path))
    assert on_disk.arc_curiosity_vs_annoyance == 0.0
    assert on_disk.arc_logic_vs_emotion == 0.0
    assert on_disk.arc_pride_vs_vulnerability == 0.0
    assert on_disk.tone_history == []
    assert on_disk.trust_level == 0.77


def test_reset_relationship_without_profile_returns_false() -> None:
    config = Config()
    config.memory.persist_path = None
    config.memory.embeddings_path = None
    config.memory.enable_fact_extraction = False
    config.memory.enable_episodes = False
    config.memory.profile_path = None

    manager = MemoryManager(config)
    assert manager.profile is None
    assert manager.reset_relationship() is False


def test_greeting_register_time_of_day_boundaries() -> None:
    # 04:59 is still late night; 05:00 flips to early morning.
    late = dt.datetime(2026, 1, 1, 4, 59)
    early = dt.datetime(2026, 1, 1, 5, 0)
    evening = dt.datetime(2026, 1, 1, 22, 0)
    late_again = dt.datetime(2026, 1, 1, 23, 0)
    utc_now = dt.datetime(2026, 1, 1, 12, 0)

    facts = greeting_register_facts(late, utc_now, None)
    assert len(facts) == 1
    assert "late night" in facts[0]
    assert "never lecture them about sleep" in facts[0]

    facts = greeting_register_facts(early, utc_now, None)
    assert len(facts) == 1
    assert "early morning" in facts[0]

    facts = greeting_register_facts(evening, utc_now, None)
    assert len(facts) == 1
    assert "evening" in facts[0]

    facts = greeting_register_facts(late_again, utc_now, None)
    assert len(facts) == 1
    assert "late night" in facts[0]


def test_greeting_register_no_absence_line_under_48_hours() -> None:
    utc_now = dt.datetime(2026, 1, 3, 12, 0)
    last_seen = utc_now - dt.timedelta(hours=47)
    facts = greeting_register_facts(utc_now, utc_now, last_seen)
    assert len(facts) == 1  # time-of-day only, no "returning after" line


def test_greeting_register_absence_in_days() -> None:
    utc_now = dt.datetime(2026, 1, 3, 12, 0)
    last_seen = utc_now - dt.timedelta(hours=49)
    facts = greeting_register_facts(utc_now, utc_now, last_seen)
    assert len(facts) == 2
    assert "2 days" in facts[1]
    assert "NEVER guilt" in facts[1]


def test_greeting_register_absence_in_weeks() -> None:
    utc_now = dt.datetime(2026, 1, 21, 12, 0)
    last_seen = utc_now - dt.timedelta(days=20)
    facts = greeting_register_facts(utc_now, utc_now, last_seen)
    assert len(facts) == 2
    assert "about 3 weeks" in facts[1]
    assert "NEVER guilt" in facts[1]


def test_greeting_register_none_last_seen_is_time_of_day_only() -> None:
    utc_now = dt.datetime(2026, 1, 1, 15, 0)
    facts = greeting_register_facts(utc_now, utc_now, None)
    assert len(facts) == 1
    assert "afternoon" in facts[0]


# --- v4: last_felt (emotional continuity) ------------------------------------


def test_v3_profile_loads_with_last_felt_none(tmp_path: Path) -> None:
    # An older (v3) profile has no last_felt key; it must load as None, not
    # raise, and keep its v3 state.
    path = tmp_path / "profile.json"
    path.write_text(
        json.dumps(
            {
                "version": 3,
                "first_met": "2026-01-01",
                "session_count": 6,
                "tone_history": [{"date": "2026-01-02", "tone": 0.4}],
            }
        )
    )
    profile = UserProfile.load(path)
    assert profile.last_felt is None
    assert profile.session_count == 6


def test_record_felt_and_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "profile.json"
    profile = UserProfile()
    profile.record_felt("Excited", "the paper deadline", when=dt.date(2026, 1, 5))
    profile.save_atomic(path)
    restored = UserProfile.load(path)
    assert restored.last_felt == {
        "label": "excited",  # lowercased
        "subject": "the paper deadline",
        "date": "2026-01-05",
    }


def test_record_felt_blank_label_clears_slot() -> None:
    profile = UserProfile(last_felt={"label": "anxious"})
    profile.record_felt("   ")
    assert profile.last_felt is None


def test_record_felt_only_keeps_newest() -> None:
    profile = UserProfile()
    profile.record_felt("excited", "the paper")
    profile.record_felt("reflective", "the review")
    assert profile.last_felt["label"] == "reflective"
    assert profile.last_felt["subject"] == "the review"


def test_sanitize_felt_caps_lengths_and_drops_junk_subject() -> None:
    profile = UserProfile()
    profile.record_felt("x" * 50, "none")  # junk subject sentinel dropped
    assert len(profile.last_felt["label"]) == 24
    assert "subject" not in profile.last_felt


def test_felt_fact_renders_with_subject() -> None:
    profile = UserProfile(
        last_felt={
            "label": "excited",
            "subject": "the paper deadline",
            "date": dt.date.today().isoformat(),
        }
    )
    fact = profile.felt_fact()
    assert fact is not None
    assert "excited about the paper deadline" in fact
    assert "Never guilt" in fact
    assert "never recite" in fact


def test_felt_fact_renders_without_subject() -> None:
    profile = UserProfile(
        last_felt={"label": "reflective", "date": dt.date.today().isoformat()}
    )
    fact = profile.felt_fact()
    assert fact is not None
    assert "the user seemed reflective." in fact


def test_felt_fact_none_when_empty() -> None:
    assert UserProfile().felt_fact() is None


def test_felt_fact_suppressed_when_stale() -> None:
    old = (dt.date.today() - dt.timedelta(days=45)).isoformat()
    profile = UserProfile(last_felt={"label": "excited", "date": old})
    assert profile.felt_fact() is None


def test_felt_fact_missing_date_treated_as_current() -> None:
    profile = UserProfile(last_felt={"label": "content"})
    assert profile.felt_fact() is not None


# --- birthdays (parse + fact scan + milestone wiring) ----------------------


def test_parse_birthday_formats() -> None:
    # Every format that character packs or fact extraction plausibly write.
    assert parse_birthday("07-25") == (7, 25)
    assert parse_birthday("7/25") == (7, 25)
    assert parse_birthday("1992-07-25") == (7, 25)  # ISO: year ignored
    assert parse_birthday("July 25") == (7, 25)
    assert parse_birthday("july 25th, 1992") == (7, 25)
    assert parse_birthday("25 July") == (7, 25)
    assert parse_birthday("Jul 25") == (7, 25)


def test_parse_birthday_rejects_garbage() -> None:
    assert parse_birthday(None) is None
    assert parse_birthday("") is None
    assert parse_birthday("sometime in summer") is None
    assert parse_birthday("13-40") is None  # not a real calendar date
    assert parse_birthday("February 30") is None
    # Feb 29 is a real birthday even in non-leap years (validated on 2000).
    assert parse_birthday("February 29") == (2, 29)


def test_user_birthday_from_facts_scans_keys() -> None:
    # Fact keys are free-form LLM output — substring match, not equality.
    assert user_birthday_from_facts({"user birthday": "March 3"}) == (3, 3)
    assert user_birthday_from_facts({"date of birth": "1990-03-03"}) == (3, 3)
    assert user_birthday_from_facts({"favorite drink": "coffee"}) is None
    assert user_birthday_from_facts({"birthday": "unknown-ish"}) is None
    assert user_birthday_from_facts(None) is None


def test_milestone_events_user_birthday_fires_on_the_day() -> None:
    events = milestone_events(3, None, today=dt.date(2026, 3, 3), user_birthday=(3, 3))
    assert [e["kind"] for e in events] == ["user_birthday"]
    assert "BIRTHDAY" in events[0]["label"]


def test_milestone_events_character_birthday_fires_on_the_day() -> None:
    events = milestone_events(
        3, None, today=dt.date(2026, 7, 25), character_birthday=(7, 25)
    )
    assert [e["kind"] for e in events] == ["character_birthday"]
    assert "your OWN birthday" in events[0]["label"]


def test_milestone_events_birthdays_quiet_on_other_days() -> None:
    events = milestone_events(
        3,
        None,
        today=dt.date(2026, 7, 24),
        character_birthday=(7, 25),
        user_birthday=(3, 3),
    )
    assert events == []


def test_milestone_events_birthday_composes_with_anniversary() -> None:
    # A first-met anniversary landing on her birthday reports both.
    events = milestone_events(
        3,
        "2025-07-25",
        today=dt.date(2026, 7, 25),
        character_birthday=(7, 25),
    )
    kinds = [e["kind"] for e in events]
    assert "character_birthday" in kinds
    assert "anniversary" in kinds


def test_engine_character_birthday_resolution(tmp_path: Path) -> None:
    """Built-in Kurisu gets July 25 with zero config; custom packs only get
    a birthday their manifest set — never Kurisu's by inheritance."""
    config = _profile_config(tmp_path)
    assert ChatEngine(config, DummyProvider())._character_birthday == (7, 25)

    custom = _profile_config(tmp_path)
    custom.character.system_prompt = "You are Rin."
    assert ChatEngine(custom, DummyProvider())._character_birthday is None

    dated = _profile_config(tmp_path)
    dated.character.system_prompt = "You are Rin."
    dated.character.birthday = "March 14"
    assert ChatEngine(dated, DummyProvider())._character_birthday == (3, 14)


# ---------------------------------------------------------------------------
# Persisted session tone (v5): the ending tone of a session survives the
# process so the NEXT session boundary can apply tone-driven drift.
# ---------------------------------------------------------------------------


def test_session_tone_roundtrips(tmp_path: Path) -> None:
    path = tmp_path / "profile.json"
    profile = UserProfile()
    profile.set_session_tone(0.5, when=dt.date(2026, 7, 20))
    profile.save_atomic(path)
    restored = UserProfile.load(path)
    assert restored.last_session_tone == {"tone": 0.5, "date": "2026-07-20"}


def test_session_tone_load_rejects_garbage(tmp_path: Path) -> None:
    path = tmp_path / "profile.json"
    path.write_text(json.dumps({"version": 5, "last_session_tone": {"tone": "junk"}}))
    assert UserProfile.load(path).last_session_tone is None
    path.write_text(json.dumps({"version": 5, "last_session_tone": "0.5"}))
    assert UserProfile.load(path).last_session_tone is None


def test_pop_session_tone_consumes_exactly_once() -> None:
    profile = UserProfile()
    profile.set_session_tone(-0.7, when=dt.date(2026, 7, 1))
    tone, when = profile.pop_session_tone()
    assert tone == -0.7
    assert when == dt.date(2026, 7, 1)
    assert profile.last_session_tone is None
    assert profile.pop_session_tone() == (None, None)


def test_boundary_consumes_persisted_session_tone(tmp_path: Path) -> None:
    """A fresh-process session boundary applies the previous session's
    persisted ending tone — previously the restart path always saw None,
    leaving trust/arc tone drift and tone_history permanently dormant."""
    config = _profile_config(tmp_path)
    seeded = UserProfile(session_count=3, last_seen="2020-01-01T00:00:00")
    seeded.set_session_tone(0.5, when=dt.date(2020, 1, 1))
    seeded.save_atomic(Path(config.memory.profile_path))

    manager = MemoryManager(config)  # stale last_seen -> boundary fires
    assert manager.profile.session_count == 4
    assert manager.profile.trust_level == updated_trust(DEFAULT_TRUST, 4, 0.5)
    assert manager.profile.tone_history[-1] == {"date": "2020-01-01", "tone": 0.5}
    # Consumed exactly once — in memory and on disk.
    assert manager.profile.last_session_tone is None
    assert UserProfile.load(Path(config.memory.profile_path)).last_session_tone is None


def test_gap_boundary_fires_with_episodes_off(tmp_path: Path) -> None:
    """The in-process gap boundary (trust/arc/tone drift) must not be gated
    on enable_episodes — only the episode job itself is."""
    config = _profile_config(tmp_path)
    assert config.memory.enable_episodes is False
    manager = MemoryManager(config)
    manager.add_message("user", "hello")
    manager.add_message("assistant", "hi")
    manager._last_tone = 0.6
    before_sessions = manager.profile.session_count
    before_trust = manager.profile.trust_level
    stale = (dt.datetime.utcnow() - dt.timedelta(hours=2)).isoformat()
    for msg in manager.short_term:
        msg.timestamp = stale
    manager.add_message("user", "back again")
    assert manager.profile.session_count == before_sessions + 1
    assert manager.profile.trust_level >= before_trust
    assert manager.profile.tone_history[-1]["tone"] == 0.6


def test_corrupt_profile_is_quarantined_not_silently_wiped(tmp_path: Path) -> None:
    """A corrupt profile file must be kept (renamed .bak-corrupt) and counted —
    previously it silently became a blank profile and the next save
    overwrote the evidence and re-triggered the first-meeting greeting."""
    from amadeus import telemetry

    path = tmp_path / "profile.json"
    path.write_text("{ not valid json", encoding="utf-8")
    before = telemetry.snapshot().get("memory.profile_load_corrupt", 0)
    profile = UserProfile.load(path)
    assert profile.session_count == 0  # blank fallback still works
    assert not path.exists()
    assert (tmp_path / "profile.json.bak-corrupt").exists()
    assert telemetry.snapshot().get("memory.profile_load_corrupt", 0) == before + 1


# ---------------------------------------------------------------------------
# Next-time thread (v6): the one open topic stashed at farewell
# ---------------------------------------------------------------------------


def test_v5_profile_loads_with_next_time_thread_none(tmp_path: Path) -> None:
    # An older (v5) profile has no next_time_thread key; it must load as
    # None, not raise, and keep its v5 state.
    path = tmp_path / "profile.json"
    path.write_text(
        json.dumps(
            {
                "version": 5,
                "first_met": "2026-01-01",
                "session_count": 6,
                "last_session_tone": {"tone": 0.4, "date": "2026-07-01"},
            }
        )
    )
    profile = UserProfile.load(path)
    assert profile.next_time_thread is None
    assert profile.session_count == 6
    assert profile.last_session_tone == {"tone": 0.4, "date": "2026-07-01"}


def test_next_time_thread_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "profile.json"
    profile = UserProfile()
    profile.set_next_time_thread(
        "they were about to submit the manuscript",
        when=dt.datetime(2026, 7, 25, 23, 45, 12),
    )
    profile.save_atomic(path)
    restored = UserProfile.load(path)
    assert restored.next_time_thread == {
        "text": "they were about to submit the manuscript",
        "noted": "2026-07-25T23:45:12",
    }
    assert json.loads(path.read_text())["version"] == PROFILE_VERSION


def test_next_time_thread_load_drops_garbage_shapes(tmp_path: Path) -> None:
    path = tmp_path / "profile.json"
    for garbage in ("not a dict", {"text": ""}, {"noted": "2026-07-01T00:00:00"}, 7):
        path.write_text(json.dumps({"version": 6, "next_time_thread": garbage}))
        assert UserProfile.load(path).next_time_thread is None


def test_next_time_thread_load_caps_text_length(tmp_path: Path) -> None:
    from amadeus.memory.profile import NEXT_TIME_THREAD_MAX_CHARS

    path = tmp_path / "profile.json"
    path.write_text(
        json.dumps(
            {"version": 6, "next_time_thread": {"text": "x" * 500, "noted": "y" * 99}}
        )
    )
    loaded = UserProfile.load(path).next_time_thread
    assert len(loaded["text"]) == NEXT_TIME_THREAD_MAX_CHARS
    assert len(loaded["noted"]) == 19


def test_set_next_time_thread_blank_clears() -> None:
    profile = UserProfile()
    profile.set_next_time_thread("they were mid-experiment")
    profile.set_next_time_thread("   ")
    assert profile.next_time_thread is None


def test_take_next_time_thread_fresh_is_not_stale_and_not_cleared() -> None:
    now = dt.datetime(2026, 7, 26, 12, 0)
    profile = UserProfile()
    profile.set_next_time_thread(
        "finishing the figure", when=now - dt.timedelta(hours=14)
    )
    taken = profile.take_next_time_thread(now)
    assert taken == {"text": "finishing the figure", "stale": False}
    # Peek does NOT consume — clear_next_time_thread() is the consume step.
    assert profile.next_time_thread is not None
    profile.clear_next_time_thread()
    assert profile.next_time_thread is None


def test_take_next_time_thread_stale_after_a_week() -> None:
    now = dt.datetime(2026, 7, 26, 12, 0)
    profile = UserProfile()
    profile.set_next_time_thread(
        "the conference talk", when=now - dt.timedelta(days=10)
    )
    taken = profile.take_next_time_thread(now)
    assert taken == {"text": "the conference talk", "stale": True}
    assert profile.next_time_thread is not None


def test_take_next_time_thread_expired_is_dropped_unconsumed() -> None:
    now = dt.datetime(2026, 7, 26, 12, 0)
    profile = UserProfile()
    profile.set_next_time_thread("an old errand", when=now - dt.timedelta(days=40))
    assert profile.take_next_time_thread(now) is None
    assert profile.next_time_thread is None  # slot cleared by the drop


def test_take_next_time_thread_unparseable_timestamp_is_tentative() -> None:
    profile = UserProfile(next_time_thread={"text": "the move", "noted": "garbage"})
    taken = profile.take_next_time_thread(dt.datetime(2026, 7, 26))
    assert taken == {"text": "the move", "stale": True}


def test_early_acquaintance_active_gates_on_both_caps() -> None:
    from amadeus.memory.profile import early_acquaintance_active

    kwargs = {"max_sessions": 8, "max_facts": 5}
    assert early_acquaintance_active(0, 0, **kwargs) is True
    assert early_acquaintance_active(7, 4, **kwargs) is True
    # Fact cap: she knows enough — phase over even in session two.
    assert early_acquaintance_active(2, 5, **kwargs) is False
    # Session cap: the consent valve — a user who never shares stops
    # being asked even with an empty profile.
    assert early_acquaintance_active(8, 0, **kwargs) is False
    # Either cap at 0 disables outright; junk input is inactive, never a raise.
    assert early_acquaintance_active(0, 0, max_sessions=0, max_facts=5) is False
    assert early_acquaintance_active(0, 0, max_sessions=8, max_facts=0) is False
    assert early_acquaintance_active("x", 0, **kwargs) is False
