"""Tests for the model-facing memory tool (remember/correct/forget) and the
central parameter-validation gap it exposed in ToolRegistry.execute_tool.

Mostly uses a fake MemoryManager matching the agreed contract
(find_facts/archive_fact/store_user_fact). The fake's archive_fact MUST
mirror the real pinned guard (refuse pinned unless allow_pinned=True) — an
earlier version archived unconditionally, which is exactly how the
"forget can't touch pinned facts but claims it did" bug shipped. A couple
of end-to-end tests below run the REAL MemoryManager to pin the
explicit-forget-beats-pins contract against the actual archive path.
"""

import json

from amadeus import telemetry
from amadeus.agents.factory import create_tool_registry
from amadeus.agents.memory_tool import MemoryTool
from amadeus.agents.tools import AnimationTool, ToolCall, ToolRegistry
from amadeus.config import Config
from amadeus.memory.manager import MemoryManager

from test_memory import VectorEmbedder, _extraction_config


class FakeMemoryManager:
    """Duck-types the MemoryManager contract this tool depends on."""

    def __init__(self, facts=None, raise_on=None):
        # facts: list of {"id", "text", "score", "pinned"}
        self.facts = facts if facts is not None else []
        self.raise_on = raise_on or set()
        self.stored = []  # (text, salience, previous_fact)
        self.archived_ids = []

    def find_facts(self, text, k=3):
        if "find_facts" in self.raise_on:
            raise RuntimeError("find_facts boom")
        return list(self.facts[:k])

    def archive_fact(self, record_id, reason, allow_pinned=False):
        # Faithful to fact_archive.archive_fact: pinned records are refused
        # unless the caller explicitly allows them (user-explicit requests).
        if "archive_fact" in self.raise_on:
            raise RuntimeError("archive_fact boom")
        for f in self.facts:
            if f["id"] == record_id:
                if f.get("pinned") and not allow_pinned:
                    return None
                self.archived_ids.append((record_id, reason))
                return f["text"]
        return None

    def store_user_fact(self, text, salience=0.9, previous_fact=None, valid_at=None):
        if "store_user_fact" in self.raise_on:
            raise RuntimeError("store_user_fact boom")
        self.stored.append((text, salience, previous_fact))
        self.stored_valid_at = valid_at
        return True


def _reset_counters():
    # telemetry is a bare module-level dict; tests only assert deltas so no
    # reset helper is required, but grab a baseline for clarity.
    return dict(telemetry._VALUES)


# -- remember --------------------------------------------------------------


def test_remember_stores_fact_and_mentions_it():
    mem = FakeMemoryManager()
    tool = MemoryTool(mem)
    before = telemetry._VALUES.get("tools.memory_remember", 0)

    result = tool.execute(action="remember", text="The user's dog is named Pixel.")

    assert result.success
    assert mem.stored == [("The user's dog is named Pixel.", 0.9, None)]
    assert "Pixel" in result.result
    assert telemetry._VALUES.get("tools.memory_remember", 0) == before + 1


def test_remember_failure_is_actionable():
    mem = FakeMemoryManager()
    mem.store_user_fact = lambda *a, **k: False
    tool = MemoryTool(mem)

    result = tool.execute(action="remember", text="anything")

    assert not result.success
    assert result.error


# -- forget ------------------------------------------------------------


def test_forget_archives_confident_match_and_returns_archived_text():
    mem = FakeMemoryManager(
        facts=[
            {
                "id": "f1",
                "text": "The user's cat is named Mochi.",
                "score": 0.91,
                "pinned": False,
            }
        ]
    )
    tool = MemoryTool(mem)
    before = telemetry._VALUES.get("tools.memory_forget", 0)

    result = tool.execute(action="forget", text="what I said about my cat")

    assert result.success
    assert mem.archived_ids == [("f1", "user_forget")]
    assert "Mochi" in result.result
    assert telemetry._VALUES.get("tools.memory_forget", 0) == before + 1


def test_forget_declines_below_threshold_and_returns_candidates():
    mem = FakeMemoryManager(
        facts=[
            {"id": "f1", "text": "The user likes tea.", "score": 0.4, "pinned": False},
            {
                "id": "f2",
                "text": "The user likes coffee.",
                "score": 0.3,
                "pinned": False,
            },
        ]
    )
    tool = MemoryTool(mem)
    before = telemetry._VALUES.get("tools.memory_forget_no_match", 0)

    result = tool.execute(action="forget", text="something about drinks")

    assert result.success  # not an error — an honest abstention
    assert mem.archived_ids == []
    assert "tea" in result.result and "coffee" in result.result
    assert telemetry._VALUES.get("tools.memory_forget_no_match", 0) == before + 1


def test_forget_on_empty_store_is_honest_no_match():
    mem = FakeMemoryManager(facts=[])
    tool = MemoryTool(mem)

    result = tool.execute(action="forget", text="the time I mentioned Berlin")

    assert result.success
    assert mem.archived_ids == []
    assert "No stored facts" in result.result


def test_forget_pinned_top_match_is_archivable_and_flagged():
    """Explicit forget beats pins: with the fake now honoring the real
    pinned guard, this only passes because the tool asks for
    allow_pinned=True."""
    mem = FakeMemoryManager(
        facts=[
            {
                "id": "f1",
                "text": "The user's birthday is March 3.",
                "score": 0.95,
                "pinned": True,
            }
        ]
    )
    tool = MemoryTool(mem)

    result = tool.execute(action="forget", text="my birthday")

    assert result.success
    assert mem.archived_ids == [("f1", "user_forget")]
    assert result.metadata.get("pinned") is True
    assert "pinned" in result.result.lower()


class LegacyFakeMemoryManager(FakeMemoryManager):
    """A manager predating the allow_pinned parameter, with no real
    vector-store internals to shim through — the worst case, where a pinned
    record genuinely cannot be archived from the tool."""

    def archive_fact(self, record_id, reason):
        for f in self.facts:
            if f["id"] == record_id:
                if f.get("pinned"):
                    return None
                self.archived_ids.append((record_id, reason))
                return f["text"]
        return None


def test_forget_pinned_on_legacy_manager_is_honest_not_already_gone():
    """When the pin truly can't be overridden, the tool must say the memory
    is pinned — never claim "no longer exists" while the fact survives."""
    mem = LegacyFakeMemoryManager(
        facts=[
            {
                "id": "f1",
                "text": "The user's birthday is March 3.",
                "score": 0.95,
                "pinned": True,
            }
        ]
    )
    tool = MemoryTool(mem)
    before = telemetry._VALUES.get("tools.memory_forget_pinned_refused", 0)

    result = tool.execute(action="forget", text="my birthday")

    assert result.success
    assert mem.archived_ids == []
    assert result.metadata.get("pinned") is True
    assert result.metadata.get("matched") is True
    assert "pinned" in result.result.lower()
    assert "no longer exists" not in result.result
    assert telemetry._VALUES.get("tools.memory_forget_pinned_refused", 0) == before + 1


# -- correct -----------------------------------------------------------


def test_correct_archives_and_stores_with_previous_fact():
    mem = FakeMemoryManager(
        facts=[
            {
                "id": "f1",
                "text": "The user lives in Munich.",
                "score": 0.9,
                "pinned": False,
            }
        ]
    )
    tool = MemoryTool(mem)
    before = telemetry._VALUES.get("tools.memory_correct", 0)

    result = tool.execute(
        action="correct",
        text="the user lives in Munich",
        replacement="The user lives in Berlin now, not Munich.",
    )

    assert result.success
    assert mem.archived_ids == [("f1", "user_correct")]
    assert mem.stored == [
        ("The user lives in Berlin now, not Munich.", 0.9, "The user lives in Munich.")
    ]
    assert telemetry._VALUES.get("tools.memory_correct", 0) == before + 1


def test_correct_without_replacement_is_validation_error():
    mem = FakeMemoryManager()
    tool = MemoryTool(mem)

    result = tool.execute(action="correct", text="the user lives in Munich")

    assert not result.success
    assert "replacement" in result.error.lower()
    assert mem.stored == []


def test_correct_with_no_match_still_stores_replacement():
    mem = FakeMemoryManager(facts=[])
    tool = MemoryTool(mem)

    result = tool.execute(
        action="correct", text="some stale fact", replacement="The corrected fact."
    )

    assert result.success
    assert mem.archived_ids == []
    assert mem.stored == [("The corrected fact.", 0.9, None)]
    assert "regardless" in result.result or "new" in result.result


def test_correct_archives_pinned_old_fact_and_supersedes():
    """A pinned outdated fact must not silently survive a user correction —
    that would keep the wrong version stored (and recalled) beside the
    right one."""
    mem = FakeMemoryManager(
        facts=[
            {
                "id": "f1",
                "text": "The user lives in Munich.",
                "score": 0.9,
                "pinned": True,
            }
        ]
    )
    tool = MemoryTool(mem)

    result = tool.execute(
        action="correct",
        text="the user lives in Munich",
        replacement="The user lives in Berlin now, not Munich.",
    )

    assert result.success
    assert mem.archived_ids == [("f1", "user_correct")]
    assert mem.stored == [
        ("The user lives in Berlin now, not Munich.", 0.9, "The user lives in Munich.")
    ]
    assert result.metadata.get("matched") is True


# -- real MemoryManager end-to-end (explicit forget beats pins) ------------


def _archive_lines(manager):
    from amadeus.memory.fact_archive import archive_path

    path = archive_path(manager)
    assert path is not None
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_forget_pinned_fact_against_real_manager_archives_it(tmp_path):
    """End-to-end against the real archive path: the manager's thin
    archive_fact delegate predates allow_pinned, so the tool's shim must
    reach fact_archive.archive_fact directly and still beat the pin."""
    embedder = VectorEmbedder({"birthday": (1.0, 0.0)})
    manager = MemoryManager(_extraction_config(tmp_path), embedder=embedder)
    assert manager.remember("The user's birthday is March 3.", pinned=True)
    tool = MemoryTool(manager)
    before = telemetry._VALUES.get("tools.memory_forget", 0)

    result = tool.execute(action="forget", text="my birthday")

    assert result.success
    assert result.metadata.get("pinned") is True
    assert "pinned" in result.result.lower()
    # The pinned record actually left the hot store...
    assert list(manager.vector_store.dump()) == []
    # ...and survives in the archive with the explicit-forget reason.
    lines = _archive_lines(manager)
    assert len(lines) == 1
    assert lines[0]["text"] == "The user's birthday is March 3."
    assert lines[0]["reason"] == "user_forget"
    assert telemetry._VALUES.get("tools.memory_forget", 0) == before + 1


def test_correct_pinned_fact_against_real_manager_supersedes_it(tmp_path):
    embedder = VectorEmbedder({"Munich": (1.0, 0.0)})
    manager = MemoryManager(_extraction_config(tmp_path), embedder=embedder)
    assert manager.remember("The user lives in Munich.", pinned=True)
    tool = MemoryTool(manager)

    result = tool.execute(
        action="correct",
        text="the user lives in Munich",
        replacement="The user lives in Berlin now, not Munich.",
    )

    assert result.success
    texts = [record.text for record in manager.vector_store.dump()]
    assert "The user lives in Munich." not in texts
    assert "The user lives in Berlin now, not Munich." in texts
    lines = _archive_lines(manager)
    assert any(
        line["text"] == "The user lives in Munich." and line["reason"] == "user_correct"
        for line in lines
    )


# -- failure handling ----------------------------------------------------


def test_manager_raising_returns_failed_tool_result_and_counts():
    mem = FakeMemoryManager(raise_on={"store_user_fact"})
    tool = MemoryTool(mem)
    before = telemetry._VALUES.get("tools.memory_failed", 0)

    result = tool.execute(action="remember", text="anything")

    assert not result.success
    assert result.error
    assert telemetry._VALUES.get("tools.memory_failed", 0) == before + 1


def test_manager_raising_on_find_facts_returns_failed_result():
    mem = FakeMemoryManager(raise_on={"find_facts"})
    tool = MemoryTool(mem)
    before = telemetry._VALUES.get("tools.memory_failed", 0)

    result = tool.execute(action="forget", text="anything")

    assert not result.success
    assert telemetry._VALUES.get("tools.memory_failed", 0) == before + 1


def test_tool_never_raises_on_unknown_action():
    mem = FakeMemoryManager()
    tool = MemoryTool(mem)
    # Unknown action bypasses the enum in validate_parameters only if called
    # directly against execute(); ensure execute() itself is still defensive.
    result = tool.execute(action="teleport", text="anything")
    assert not result.success


# -- registration ---------------------------------------------------------


def test_registry_registers_memory_tool_when_memory_and_config_enabled():
    config = Config()
    assert config.memory.enable_long_term
    assert config.memory.enable_fact_extraction
    mem = FakeMemoryManager()

    registry = create_tool_registry(config, memory=mem)

    assert "memory" in registry.tools
    assert isinstance(registry.tools["memory"], MemoryTool)


def test_registry_records_unavailable_when_memory_none_but_config_wants_it():
    config = Config()
    assert config.memory.enable_long_term
    assert config.memory.enable_fact_extraction

    registry = create_tool_registry(config, memory=None)

    assert "memory" not in registry.tools
    names = [u["name"] for u in registry.unavailable]
    assert "memory" in names


def test_registry_skips_memory_tool_when_long_term_disabled():
    config = Config()
    config.memory.enable_long_term = False
    mem = FakeMemoryManager()

    registry = create_tool_registry(config, memory=mem)

    assert "memory" not in registry.tools
    # Config doesn't want it, so no memory manager -> no unavailable entry either.
    names = [u["name"] for u in registry.unavailable]
    assert "memory" not in names


def test_registry_skips_memory_tool_when_fact_extraction_disabled():
    config = Config()
    config.memory.enable_fact_extraction = False
    mem = FakeMemoryManager()

    registry = create_tool_registry(config, memory=mem)

    assert "memory" not in registry.tools


def test_create_tool_registry_default_memory_none_keeps_working():
    # Existing call sites that don't pass memory= must keep working.
    config = Config()
    registry = create_tool_registry(config)
    assert "memory" not in registry.tools
    names = [u["name"] for u in registry.unavailable]
    assert "memory" in names


# -- central validation in ToolRegistry.execute_tool -----------------------


def test_central_validation_catches_bad_param_for_self_validation_skipping_tool():
    """AnimationTool.execute() does not call validate_parameters() itself —
    ToolRegistry.execute_tool must catch a bad/missing parameter centrally
    so this doesn't raise or silently misbehave."""
    registry = ToolRegistry()
    registry.register(AnimationTool(["Spin", "Jump"]))

    result = registry.execute_tool(ToolCall(tool_name="play_animation", parameters={}))

    assert not result.success
    assert "name" in (result.error or "").lower()


def test_central_validation_rejects_bad_enum_value():
    registry = ToolRegistry()
    registry.register(AnimationTool(["Spin", "Jump"]))

    result = registry.execute_tool(
        ToolCall(tool_name="play_animation", parameters={"name": "Backflip"})
    )

    assert not result.success


def test_central_validation_passes_through_valid_call():
    registry = ToolRegistry()
    registry.register(AnimationTool(["Spin", "Jump"]))

    result = registry.execute_tool(
        ToolCall(tool_name="play_animation", parameters={"name": "Spin"})
    )

    assert result.success


def test_memory_tool_registered_in_registry_survives_central_validation():
    mem = FakeMemoryManager(
        facts=[
            {
                "id": "f1",
                "text": "The user's cat is Mochi.",
                "score": 0.9,
                "pinned": False,
            }
        ]
    )
    registry = ToolRegistry()
    registry.register(MemoryTool(mem))

    result = registry.execute_tool(
        ToolCall(tool_name="memory", parameters={"action": "remember", "text": "hi"})
    )
    assert result.success

    missing_text = registry.execute_tool(
        ToolCall(tool_name="memory", parameters={"action": "remember"})
    )
    assert not missing_text.success


# -- keyword gate ------------------------------------------------------


def test_engine_tool_hint_words_include_memory_keywords():
    from amadeus.engine import ChatEngine

    hints = ChatEngine._TOOL_HINT_WORDS
    for word in ("remember", "forget", "memorize", "memorise"):
        assert word in hints
