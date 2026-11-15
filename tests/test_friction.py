"""Friction memory (UserProfile v7): state, bounds, and the repair contract.

The anti-guilt CONTRACT tests live in tests/test_friction_ethics.py — this
file covers the mechanical state layer only.
"""

import datetime as dt
import json
import time
from pathlib import Path
from typing import List, Optional

from amadeus import telemetry
from amadeus.config import Config
from amadeus.memory.manager import EmbeddingProvider, MemoryManager
from amadeus.memory.profile import (
    FRICTION_MAX_AGE_DAYS,
    FRICTION_TOPIC_MAX_CHARS,
    FRICTION_TOPICS_CAP,
    PROFILE_VERSION,
    UserProfile,
)
from amadeus.providers.llm.base import LLMProvider


def test_record_friction_stores_topic_with_date_and_count() -> None:
    profile = UserProfile()
    profile.record_friction("the funding thing", when=dt.date(2026, 8, 1))

    assert profile.friction_topics == [
        {"topic": "the funding thing", "noted": "2026-08-01", "count": 1}
    ]


def test_record_friction_blank_topic_is_a_noop() -> None:
    profile = UserProfile()
    profile.record_friction("   ")
    assert profile.friction_topics == []


def test_record_friction_dedupes_same_topic_case_insensitively() -> None:
    """A recurring sore spot is ONE topic with a count, not three entries —
    otherwise the tiny cap fills with restatements of the same thing."""
    profile = UserProfile()
    profile.record_friction("The Funding Thing", when=dt.date(2026, 8, 1))
    profile.record_friction("the funding thing", when=dt.date(2026, 8, 3))

    assert len(profile.friction_topics) == 1
    assert profile.friction_topics[0]["count"] == 2
    assert profile.friction_topics[0]["noted"] == "2026-08-03"
    # The original casing is kept; the match is what is case-insensitive.
    assert profile.friction_topics[0]["topic"] == "The Funding Thing"


def test_record_friction_truncates_long_topics() -> None:
    profile = UserProfile()
    profile.record_friction("x" * 200)
    assert len(profile.friction_topics[0]["topic"]) == FRICTION_TOPIC_MAX_CHARS


def test_friction_topics_capped_evicting_oldest() -> None:
    profile = UserProfile()
    for i in range(FRICTION_TOPICS_CAP + 2):
        profile.record_friction(f"topic {i}", when=dt.date(2026, 8, 1))

    assert len(profile.friction_topics) == FRICTION_TOPICS_CAP
    topics = [e["topic"] for e in profile.friction_topics]
    assert "topic 0" not in topics
    assert f"topic {FRICTION_TOPICS_CAP + 1}" in topics


def test_clear_friction_removes_the_topic_and_reports_it() -> None:
    profile = UserProfile()
    profile.record_friction("the funding thing")

    assert profile.clear_friction("THE FUNDING THING") is True
    assert profile.friction_topics == []
    # Clearing something that isn't there is a truthful False, not a crash.
    assert profile.clear_friction("never happened") is False
    assert profile.clear_friction("") is False


def test_friction_expires_after_max_age_with_no_user_action() -> None:
    """THE NO-DEBT CONTRACT. Time alone repairs. A sore topic that only the
    user can clear is a debt she is holding them to — the exact retention
    pattern docs/companion-ethics.md refuses ("no penalty, visible or
    hidden, for being away a month")."""
    profile = UserProfile()
    today = dt.date(2026, 9, 1)
    profile.record_friction(
        "the funding thing",
        when=today - dt.timedelta(days=FRICTION_MAX_AGE_DAYS + 1),
    )

    dropped = profile.prune_stale_friction(today=today)

    assert dropped == 1
    assert profile.friction_topics == []
    assert profile.friction_fact(today=today) is None


def test_friction_within_the_window_survives_pruning() -> None:
    profile = UserProfile()
    today = dt.date(2026, 9, 1)
    profile.record_friction(
        "the funding thing",
        when=today - dt.timedelta(days=FRICTION_MAX_AGE_DAYS - 1),
    )

    assert profile.prune_stale_friction(today=today) == 0
    assert len(profile.friction_topics) == 1


def test_friction_with_unparseable_date_is_treated_as_current() -> None:
    """Best-effort, matching felt_fact: a bad date must not silently delete
    state, and must not silently make it immortal either — it stays live and
    is subject to the cap like anything else."""
    profile = UserProfile(friction_topics=[{"topic": "x", "noted": "garbage"}])
    assert profile.prune_stale_friction(today=dt.date(2026, 9, 1)) == 0
    assert profile.friction_fact(today=dt.date(2026, 9, 1)) is not None


def test_friction_fact_returns_none_when_there_is_no_friction() -> None:
    """The common case must add ZERO prompt characters."""
    assert UserProfile().friction_fact() is None


def test_friction_fact_names_the_topic_and_picks_the_newest() -> None:
    profile = UserProfile()
    profile.record_friction("the old thing", when=dt.date(2026, 8, 1))
    profile.record_friction("the recent thing", when=dt.date(2026, 8, 20))

    fact = profile.friction_fact(today=dt.date(2026, 8, 21))

    assert fact is not None
    assert "the recent thing" in fact
    assert "the old thing" not in fact


def test_profile_v7_roundtrips_friction(tmp_path: Path) -> None:
    path = tmp_path / "profile.json"
    profile = UserProfile()
    profile.record_friction("the funding thing", when=dt.date(2026, 8, 1))
    profile.save_atomic(path)

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["version"] == PROFILE_VERSION == 7

    loaded = UserProfile.load(path)
    assert loaded.friction_topics == profile.friction_topics


def test_v6_profile_loads_with_empty_friction(tmp_path: Path) -> None:
    """Older files load with defaults — no migration, additive only."""
    path = tmp_path / "profile.json"
    path.write_text(
        json.dumps({"version": 6, "facts": {"name": "Alex"}}), encoding="utf-8"
    )

    loaded = UserProfile.load(path)

    assert loaded.friction_topics == []
    assert loaded.facts["name"] == "Alex"


def test_corrupt_friction_entries_are_skipped_not_fatal(tmp_path: Path) -> None:
    """One bad write must never make a profile unloadable."""
    path = tmp_path / "profile.json"
    path.write_text(
        json.dumps(
            {
                "version": 7,
                "friction_topics": [
                    "not a dict",
                    {"no_topic_key": 1},
                    {"topic": "   "},
                    {"topic": "the real one", "noted": "2026-08-01", "count": "junk"},
                ],
            }
        ),
        encoding="utf-8",
    )

    loaded = UserProfile.load(path)

    assert len(loaded.friction_topics) == 1
    assert loaded.friction_topics[0]["topic"] == "the real one"
    # A junk count falls back to 1 rather than propagating a string.
    assert loaded.friction_topics[0]["count"] == 1


def test_load_trims_an_over_cap_stored_list(tmp_path: Path) -> None:
    path = tmp_path / "profile.json"
    path.write_text(
        json.dumps(
            {
                "version": 7,
                "friction_topics": [
                    {"topic": f"t{i}", "noted": "2026-08-01"} for i in range(10)
                ],
            }
        ),
        encoding="utf-8",
    )

    assert len(UserProfile.load(path).friction_topics) == FRICTION_TOPICS_CAP


# ---------------------------------------------------------------------------
# Extraction wiring (rides the existing call — no extra LLM request)
# ---------------------------------------------------------------------------


class _Embedder(EmbeddingProvider):
    def embed_text(self, text: str) -> List[float]:
        return [1.0, 0.0]

    def similarity(self, a: List[float], b: List[float]) -> float:
        return sum(x * y for x, y in zip(a, b))


class _ScriptedSummary(LLMProvider):
    name = "scripted-summary"

    def __init__(self, text: str) -> None:
        self.text = text
        self.prompts: List[str] = []
        # The system prompt carries the schema; the USER message carries the
        # profile snapshot. Both matter here, so keep them separately.
        self.user_prompts: List[str] = []

    def generate(self, messages, **kwargs):  # type: ignore[override]
        self.prompts.append(messages[0].content)
        self.user_prompts.append("\n".join(m.content for m in messages[1:]))
        return self.text

    def stream(self, messages, **kwargs):
        yield self.generate(messages, **kwargs)


def _extraction_manager(
    tmp_path: Path, reply: str, *, friction: bool = True
) -> tuple[MemoryManager, _ScriptedSummary]:
    config = Config()
    config.memory.persist_path = str(tmp_path / "memory.json")
    config.memory.embeddings_path = str(tmp_path / "embeddings.json")
    config.memory.profile_path = str(tmp_path / "profile.json")
    config.memory.enable_long_term = True
    config.memory.enable_fact_extraction = True
    config.memory.enable_episodes = False
    config.memory.summary_trigger_messages = 1
    config.memory.enable_friction_memory = friction
    provider = _ScriptedSummary(reply)
    manager = MemoryManager(config, embedder=_Embedder(), summary_provider=provider)
    return manager, provider


def _wait(predicate, attempts: int = 40) -> None:
    for _ in range(attempts):
        if predicate():
            return
        time.sleep(0.05)


_BASE = '{"summary": "s", "new_facts": [], "profile_updates": {}, "open_threads": [], "tone": 0.0'


def test_disabled_keeps_the_friction_key_out_of_the_prompt(tmp_path: Path) -> None:
    """A disabled install must pay ZERO prompt characters for this."""
    manager, provider = _extraction_manager(tmp_path, _BASE + "}", friction=False)
    manager.add_message("user", "hello")
    _wait(lambda: bool(provider.prompts))

    assert provider.prompts
    assert "friction" not in provider.prompts[0]


def test_enabled_asks_for_friction_and_records_it(tmp_path: Path) -> None:
    telemetry.reset()
    reply = _BASE + ', "friction": {"topic": "the funding thing", "state": "new"}}'
    manager, provider = _extraction_manager(tmp_path, reply)
    manager.add_message("user", "hello")
    _wait(lambda: bool(manager.profile.friction_topics))

    assert "friction" in provider.prompts[0]
    assert [e["topic"] for e in manager.profile.friction_topics] == [
        "the funding thing"
    ]
    assert telemetry.snapshot().get("memory.friction_noted") == 1
    telemetry.reset()


def test_repaired_state_clears_the_topic(tmp_path: Path) -> None:
    telemetry.reset()
    reply = _BASE + ', "friction": {"topic": "the funding thing", "state": "repaired"}}'
    manager, _ = _extraction_manager(tmp_path, reply)
    manager.profile.record_friction("the funding thing")
    manager.add_message("user", "hello")
    _wait(lambda: not manager.profile.friction_topics)

    assert manager.profile.friction_topics == []
    assert telemetry.snapshot().get("memory.friction_repaired") == 1
    telemetry.reset()


def test_malformed_friction_is_dropped_not_fatal(tmp_path: Path) -> None:
    reply = _BASE + ', "friction": "not a dict"}'
    manager, _ = _extraction_manager(tmp_path, reply)
    manager.add_message("user", "hello")
    _wait(lambda: manager.summary_info.get("strategy") == "llm")

    assert manager.profile.friction_topics == []
    # The rest of the extraction still landed — one bad field must not sink
    # the whole call.
    assert manager.summary == "s"


def test_sore_topics_reach_the_extraction_snapshot_for_repair(tmp_path: Path) -> None:
    """The model can only report state="repaired" for a topic it can see."""
    manager, provider = _extraction_manager(tmp_path, _BASE + "}")
    manager.profile.record_friction("the funding thing")
    manager.add_message("user", "hello")
    _wait(lambda: bool(provider.prompts))

    joined = "\n".join(provider.user_prompts)
    assert "sore_topics" in joined
    assert "the funding thing" in joined
