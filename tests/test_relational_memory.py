"""Relational memory (amadeus/memory/relational.py): durable records of
moments that changed something BETWEEN her and the user.

The tests that matter most here are the anti-fixation ones. This is the
first first-person session-end record in the system, and the last time the
codebase had one (episodes, before they were made neutral third-person) her
persona compounded across sessions until she fixated. The weekly cap and
the prompt's hard bans are what keep that from recurring, so both are
pinned rather than assumed.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import List, Optional

import pytest

from amadeus import telemetry
from amadeus.config import Config
from amadeus.memory.manager import EmbeddingProvider, MemoryManager
from amadeus.memory.relational import (
    _clean_relational_line,
    _relational_prompt,
    maybe_store_relational_moment,
)
from amadeus.providers.llm.base import ChatMessage, LLMProvider


@pytest.fixture(autouse=True)
def _clean_telemetry():
    telemetry.reset()
    yield
    telemetry.reset()


class VectorEmbedder(EmbeddingProvider):
    def __init__(self, mapping: Optional[dict] = None, default=(1.0, 0.0)) -> None:
        self.mapping = mapping or {}
        self.default = default

    def embed_text(self, text: str) -> List[float]:
        for key, vec in self.mapping.items():
            if key in text:
                return list(vec)
        return list(self.default)

    def similarity(self, a: List[float], b: List[float]) -> float:
        return sum(x * y for x, y in zip(a, b))


class ScriptedLLM(LLMProvider):
    name = "scripted"

    def __init__(self, replies: Optional[List[str]] = None):
        self.replies = replies or ["we finally talked about the lab"]
        self.calls: List[List[ChatMessage]] = []
        self.fail = False

    def generate(self, messages, max_tokens: int = 256, **kwargs) -> str:
        self.calls.append(list(messages))
        if self.fail:
            raise RuntimeError("provider down")
        idx = min(len(self.calls) - 1, len(self.replies) - 1)
        return self.replies[idx]

    def stream(self, messages, max_tokens: int = 256, **kwargs):
        yield self.generate(messages, max_tokens=max_tokens)


def _manager(
    tmp_path: Path,
    llm: Optional[ScriptedLLM] = None,
    *,
    enabled: bool = True,
    **overrides,
) -> MemoryManager:
    config = Config()
    config.memory.persist_path = str(tmp_path / "memory.json")
    config.memory.embeddings_path = str(tmp_path / "embeddings.json")
    config.memory.profile_path = str(tmp_path / "profile.json")
    config.memory.enable_long_term = True
    config.memory.enable_fact_extraction = False
    config.memory.enable_episodes = False
    # Keep the summarizer off this suite's provider: it shares
    # summary_provider with the relational job, so a triggered summarize()
    # would show up as a phantom relational call.
    config.memory.summary_trigger_messages = 10_000
    config.relational_memory.enabled = enabled
    for key, value in overrides.items():
        setattr(config.relational_memory, key, value)
    return MemoryManager(
        config,
        embedder=VectorEmbedder(),
        summary_provider=llm or ScriptedLLM(),
    )


def _messages(manager: MemoryManager, count: int = 8) -> None:
    for i in range(count):
        manager.add_message("user" if i % 2 == 0 else "assistant", f"line {i}")


def _moments(manager: MemoryManager, source: Optional[str] = None) -> list:
    """Stored moment records, ignoring the untyped raw-message captures the
    manager writes alongside them when fact extraction is off."""
    out = []
    for rec in manager.vector_store.dump():
        metadata = rec.metadata or {}
        if metadata.get("type") != "moment":
            continue
        if source is not None and metadata.get("source") != source:
            continue
        out.append(rec)
    return out


# ---------------------------------------------------------------------------
# Gates
# ---------------------------------------------------------------------------


def test_disabled_by_default_makes_no_call(tmp_path: Path) -> None:
    assert Config().relational_memory.enabled is False
    llm = ScriptedLLM()
    manager = _manager(tmp_path, llm, enabled=False)
    _messages(manager)

    maybe_store_relational_moment(manager, list(manager.short_term))

    assert llm.calls == []


def test_short_session_makes_no_call(tmp_path: Path) -> None:
    llm = ScriptedLLM()
    manager = _manager(tmp_path, llm, min_messages=6)
    _messages(manager, count=4)

    maybe_store_relational_moment(manager, list(manager.short_term))

    assert llm.calls == []
    assert telemetry.snapshot().get("memory.relational_too_short") == 1


# ---------------------------------------------------------------------------
# Happy path and SKIP
# ---------------------------------------------------------------------------


def test_stores_one_record_tagged_source_relational(tmp_path: Path) -> None:
    llm = ScriptedLLM(["we finally talked about why you left the lab"])
    manager = _manager(tmp_path, llm)
    _messages(manager)

    maybe_store_relational_moment(manager, list(manager.short_term))

    records = _moments(manager)
    assert len(records) == 1
    assert records[0].text == "we finally talked about why you left the lab"
    assert records[0].metadata["source"] == "relational"
    assert telemetry.snapshot().get("memory.relational_stored") == 1


def test_skip_verdict_stores_nothing(tmp_path: Path) -> None:
    """SKIP is the EXPECTED answer for a normal session, not a failure."""
    manager = _manager(tmp_path, ScriptedLLM(["SKIP"]))
    _messages(manager)

    maybe_store_relational_moment(manager, list(manager.short_term))

    assert _moments(manager) == []
    assert telemetry.snapshot().get("memory.relational_skipped") == 1


def test_line_is_truncated_to_max_chars(tmp_path: Path) -> None:
    manager = _manager(tmp_path, ScriptedLLM(["x" * 500]), max_chars=80)
    _messages(manager)

    maybe_store_relational_moment(manager, list(manager.short_term))

    (record,) = _moments(manager)
    assert len(record.text) == 80


# ---------------------------------------------------------------------------
# The cap — the real anti-fixation guard
# ---------------------------------------------------------------------------


def test_period_cap_blocks_a_second_record_and_costs_no_tokens(
    tmp_path: Path,
) -> None:
    """The cap must be checked BEFORE the provider call: generating a line
    only to discard it burns tokens on every session end forever."""
    llm = ScriptedLLM(["we talked about the lab", "we talked about something else"])
    manager = _manager(tmp_path, llm, max_per_days=7, max_per_period=1)
    _messages(manager)

    maybe_store_relational_moment(manager, list(manager.short_term))
    assert len(llm.calls) == 1

    maybe_store_relational_moment(manager, list(manager.short_term))

    assert len(llm.calls) == 1, "capped period must not call the provider"
    assert len(_moments(manager)) == 1
    assert telemetry.snapshot().get("memory.relational_cap_skipped") == 1


def test_cap_is_independent_of_co_activity_moments(tmp_path: Path) -> None:
    """Co-activity moments and relational records share a record type but
    must NOT share a cap — a busy reading week must not silence a genuine
    relational record, and vice versa."""
    llm = ScriptedLLM(["we finally talked about the lab"])
    manager = _manager(tmp_path, llm, max_per_period=1)
    for i in range(3):
        assert manager.store_moment(f"we watched episode {i}", source="co_activity")
    _messages(manager)

    maybe_store_relational_moment(manager, list(manager.short_term))

    assert telemetry.snapshot().get("memory.relational_stored") == 1
    assert (
        manager.count_moments_since(
            dt.datetime.utcnow() - dt.timedelta(days=7), source="relational"
        )
        == 1
    )
    assert (
        manager.count_moments_since(
            dt.datetime.utcnow() - dt.timedelta(days=7), source="co_activity"
        )
        == 3
    )
    # The unscoped count still sees everything, as the original caller expects.
    assert manager.count_moments_since(dt.datetime.utcnow() - dt.timedelta(days=7)) == 4


def test_cap_window_expires(tmp_path: Path) -> None:
    llm = ScriptedLLM(["we talked about the lab", "we talked again"])
    manager = _manager(tmp_path, llm, max_per_days=7, max_per_period=1)
    _messages(manager)
    maybe_store_relational_moment(manager, list(manager.short_term))

    # Backdate the stored record past the window.
    for record in _moments(manager):
        record.metadata["timestamp"] = (
            dt.datetime.utcnow() - dt.timedelta(days=9)
        ).isoformat()

    maybe_store_relational_moment(manager, list(manager.short_term))

    assert len(llm.calls) == 2
    assert telemetry.snapshot().get("memory.relational_stored") == 2


# ---------------------------------------------------------------------------
# The prompt contract (anti-fixation)
# ---------------------------------------------------------------------------


def test_prompt_bans_judgment_language_about_the_user(tmp_path: Path) -> None:
    """THE ANTI-FIXATION CONTRACT. Episodes were made neutral third-person
    because a first-person record in her voice compounded her attitude
    across sessions. This record is first-person by design, so the ban on
    recording an opinion of the user is what keeps it from recurring."""
    manager = _manager(tmp_path)
    system = _relational_prompt(manager, "transcript", 200)[0].content

    assert "never your opinion of them" in system
    assert "No adjectives judging the user" in system
    assert "No feelings about them at all" in system
    assert "never anything they owe" in system


def test_prompt_makes_skip_the_default(tmp_path: Path) -> None:
    manager = _manager(tmp_path)
    system = _relational_prompt(manager, "transcript", 200)[0].content

    assert "SKIP is the normal answer" in system
    assert "change nothing between you" in system
    assert "invent nothing" in system


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("we talked about the lab", "we talked about the lab"),
        ('  "we talked about the lab"  ', "we talked about the lab"),
        ("we talked\nextra line", "we talked"),
        ("SKIP", None),
        ("skip.", None),
        ("", None),
        ("   ", None),
    ],
)
def test_clean_relational_line(raw: str, expected: Optional[str]) -> None:
    assert _clean_relational_line(raw, 200) == expected


# ---------------------------------------------------------------------------
# Failure modes — best-effort, never raises
# ---------------------------------------------------------------------------


def test_provider_failure_counts_and_does_not_raise(tmp_path: Path) -> None:
    llm = ScriptedLLM()
    llm.fail = True
    manager = _manager(tmp_path, llm)
    _messages(manager)

    maybe_store_relational_moment(manager, list(manager.short_term))

    assert telemetry.snapshot().get("memory.relational_failures") == 1
    assert _moments(manager) == []


def test_no_summary_provider_is_a_silent_noop(tmp_path: Path) -> None:
    manager = _manager(tmp_path)
    manager.summary_provider = None
    _messages(manager)

    maybe_store_relational_moment(manager, list(manager.short_term))

    assert _moments(manager) == []


# ---------------------------------------------------------------------------
# Integration with the rest of the memory machinery
# ---------------------------------------------------------------------------


def test_relational_record_is_visible_in_the_memory_panel(tmp_path: Path) -> None:
    manager = _manager(tmp_path, ScriptedLLM(["we finally talked about the lab"]))
    _messages(manager)
    maybe_store_relational_moment(manager, list(manager.short_term))

    overview = manager.memory_overview()

    assert "we finally talked about the lab" in [
        rec["text"] for rec in overview["fact_records"]
    ]
    assert overview["counts"]["moments"] == 1


def test_recall_renders_relational_records_with_the_between_you_two_prefix(
    tmp_path: Path,
) -> None:
    from amadeus.memory.context import _recall_line

    class _Match:
        text = "we finally talked about the lab"
        metadata = {"type": "moment", "source": "relational"}
        confidence = None

    line = _recall_line(_Match(), dt.datetime.utcnow())
    assert line.startswith("- between you two: we finally talked about the lab")


def test_non_relational_records_render_byte_identically(tmp_path: Path) -> None:
    """The prefix must cost ZERO characters on every other record type."""
    from amadeus.memory.context import _recall_line

    now = dt.datetime.utcnow()

    class _Fact:
        text = "User studies neuroscience"
        metadata = {"type": "fact"}
        confidence = None

    class _CoActivityMoment:
        text = "we watched the finale together"
        metadata = {"type": "moment", "source": "co_activity"}
        confidence = None

    assert _recall_line(_Fact(), now) == "- User studies neuroscience"
    assert _recall_line(_CoActivityMoment(), now) == "- we watched the finale together"


def test_finalize_session_runs_the_job(tmp_path: Path) -> None:
    """Wiring test: the trigger is finalize_session, which every session-end
    path goes through (farewell, window close, gap, eviction)."""
    llm = ScriptedLLM(["we finally talked about the lab"])
    manager = _manager(tmp_path, llm)
    _messages(manager)

    manager.finalize_session(wait=True)

    assert telemetry.snapshot().get("memory.relational_stored") == 1


def test_finalize_session_skips_the_job_when_disabled(tmp_path: Path) -> None:
    llm = ScriptedLLM()
    manager = _manager(tmp_path, llm, enabled=False)
    _messages(manager)

    manager.finalize_session(wait=True)

    assert llm.calls == []
