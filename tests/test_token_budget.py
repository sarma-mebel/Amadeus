"""Token accounting: the estimator, telemetry value rings, and
assemble_turn_context's budget-triggered trimming.

Everything else in the config was bounded by counts/characters; nothing
measured actual prompt size. This covers: estimate_tokens/estimate_context_
tokens basics, telemetry.record_value/value_snapshot p50/p95 parity with
the duration rings, and the context_token_budget trim path in
memory/context.py (recall dropped tail-first, then oldest messages, floor
of 4, current user message and system prompt always kept, counters/warning
on trim, and a same-output regression guard below budget).
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import List

import pytest

from amadeus import telemetry
from amadeus.config import Config
from amadeus.memory.manager import EmbeddingProvider, MemoryManager
from amadeus.token_budget import (
    estimate_context_tokens,
    estimate_texts_tokens,
    estimate_tokens,
)


@pytest.fixture(autouse=True)
def _clean_counters():
    telemetry.reset()
    yield
    telemetry.reset()


# ---------------------------------------------------------------------------
# estimate_tokens / estimate_context_tokens
# ---------------------------------------------------------------------------


def test_estimate_tokens_empty_and_none():
    assert estimate_tokens("") == 0
    assert estimate_tokens(None) == 0


def test_estimate_tokens_four_chars_is_one_token():
    assert estimate_tokens("abcd") == 1


def test_estimate_tokens_ceil_behavior():
    # 5 chars -> ceil(5/4) = 2, not truncated to 1.
    assert estimate_tokens("abcde") == 2
    assert estimate_tokens("a") == 1
    assert estimate_tokens("abc") == 1
    assert estimate_tokens("abcdefgh") == 2


def test_estimate_tokens_matches_manual_ceil():
    text = "x" * 137
    assert estimate_tokens(text) == math.ceil(137 / 4)


class _Msg:
    def __init__(self, content):
        self.content = content


def test_estimate_context_tokens_sums_parts():
    system_prompt = "a" * 8  # 2 tokens
    messages = [_Msg("b" * 4), _Msg("c" * 8)]  # 1 + 2 tokens
    recall_text = "d" * 4  # 1 token
    total = estimate_context_tokens(system_prompt, messages, recall_text)
    assert total == 2 + 1 + 2 + 1


def test_estimate_context_tokens_recall_optional():
    assert estimate_context_tokens("abcd", []) == 1
    assert estimate_context_tokens("abcd", [], None) == 1


def test_estimate_context_tokens_skips_unusable_content():
    class NoContent:
        pass

    class NonStringContent:
        content = 5

    total = estimate_context_tokens("abcd", [NoContent(), NonStringContent()])
    assert total == 1  # only the system prompt counted


def test_estimate_texts_tokens_sums():
    assert estimate_texts_tokens(["abcd", "abcdefgh"]) == 1 + 2
    assert estimate_texts_tokens([]) == 0


# ---------------------------------------------------------------------------
# telemetry.record_value / value_snapshot
# ---------------------------------------------------------------------------


def test_record_value_basic_stats():
    telemetry.record_value("llm.prompt_est_tokens", 100.0)
    telemetry.record_value("llm.prompt_est_tokens", 300.0)
    entry = telemetry.value_snapshot()["llm.prompt_est_tokens"]
    assert entry["count"] == 2
    assert entry["total"] == pytest.approx(400.0)
    assert entry["max"] == pytest.approx(300.0)
    assert entry["last"] == pytest.approx(300.0)


def test_record_value_percentiles_computed_on_snapshot():
    for v in range(1, 101):  # window keeps last 64: 37..100
        telemetry.record_value("k", float(v))
    entry = telemetry.value_snapshot()["k"]
    assert entry["count"] == 100
    assert entry["total"] == pytest.approx(5050.0)
    assert entry["max"] == pytest.approx(100.0)
    assert entry["last"] == pytest.approx(100.0)
    assert entry["p50"] == pytest.approx(68.0)
    assert entry["p95"] == pytest.approx(97.0)


def test_record_value_single_sample():
    telemetry.record_value("solo", 42.0)
    entry = telemetry.value_snapshot()["solo"]
    assert entry["p50"] == pytest.approx(42.0)
    assert entry["p95"] == pytest.approx(42.0)


def test_record_value_window_is_bounded():
    telemetry.record_value("k", 10_000.0)
    for _ in range(64):
        telemetry.record_value("k", 5.0)
    entry = telemetry.value_snapshot()["k"]
    assert entry["p50"] == pytest.approx(5.0)
    assert entry["p95"] == pytest.approx(5.0)
    assert entry["max"] == pytest.approx(10_000.0)
    assert entry["count"] == 65
    assert len(telemetry._VALUE_SAMPLES["k"]) == 64


def test_value_snapshot_empty_and_reset():
    assert telemetry.value_snapshot() == {}
    telemetry.record_value("k", 1.0)
    assert "p50" in telemetry.value_snapshot()["k"]
    telemetry.reset()
    assert telemetry.value_snapshot() == {}
    assert telemetry._VALUE_SAMPLES == {}


def test_value_ring_independent_of_duration_ring():
    """The two facilities must never collide on the same key."""
    telemetry.record_duration("shared_key", 10.0)
    telemetry.record_value("shared_key", 999.0)
    d = telemetry.duration_snapshot()["shared_key"]
    v = telemetry.value_snapshot()["shared_key"]
    assert d["last_ms"] == pytest.approx(10.0)
    assert v["last"] == pytest.approx(999.0)


# ---------------------------------------------------------------------------
# assemble_turn_context: budget trimming
# ---------------------------------------------------------------------------


class VectorEmbedder(EmbeddingProvider):
    """Every text embeds identically -> every candidate scores as an
    equally strong relevance match; ranking among matches is then driven by
    recall's recency/salience ordering, which is enough for these tests
    (we only need "some deterministic best-first order", not a specific
    one, since assertions key off the actual returned order)."""

    def embed_text(self, text: str) -> List[float]:
        return [1.0, 0.0]

    def similarity(self, a: List[float], b: List[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        return dot


def _budget_config(tmp_path: Path, **overrides) -> Config:
    config = Config()
    config.memory.persist_path = None
    config.memory.enable_long_term = True
    config.memory.embeddings_path = str(tmp_path / "vectors.json")
    config.memory.dedupe_threshold = 0.0
    config.memory.recall_min_query_chars = 0
    config.memory.enable_fact_extraction = False
    config.memory.enable_episodes = False
    config.memory.profile_path = None
    # facts_only=True keeps recall to only the explicitly-seeded fact
    # records below — otherwise add_message's legacy raw-message storage
    # (async, on a background executor) races with the synchronous
    # recall_detailed() call right after it in assemble_turn_context, and
    # whichever short-term message happened to finish embedding first
    # nondeterministically fills the 3rd recall slot.
    config.memory.recall_facts_only = True
    config.memory.recall_summary_dedupe = 0.0
    config.memory.recall_recent_dedupe = 0.0
    config.memory.recall_min_score = 0.0
    for key, value in overrides.items():
        setattr(config.memory, key, value)
    return config


def _seed_facts(manager: MemoryManager, n: int, pad: int = 400) -> None:
    from amadeus.memory.vector_store import VectorRecord

    for i in range(n):
        manager.vector_store.add(
            VectorRecord(
                vector=[1.0, 0.0],
                text=f"fact number {i} " + ("z" * pad),
                metadata={"type": "fact", "timestamp": None},
            )
        )


def test_under_budget_identical_to_budget_disabled(tmp_path: Path) -> None:
    """Below-budget turns must be byte-for-byte identical to budget=0 —
    zero behavior change for the common case."""
    from amadeus.memory.context import assemble_turn_context

    config_on = _budget_config(tmp_path / "on")
    config_on.memory.context_token_budget = 12000
    manager_on = MemoryManager(config_on, embedder=VectorEmbedder())
    _seed_facts(manager_on, 2, pad=20)
    manager_on.add_message("user", "earlier turn")
    manager_on.add_message("assistant", "earlier reply")

    config_off = _budget_config(tmp_path / "off")
    config_off.memory.context_token_budget = 0
    manager_off = MemoryManager(config_off, embedder=VectorEmbedder())
    _seed_facts(manager_off, 2, pad=20)
    manager_off.add_message("user", "earlier turn")
    manager_off.add_message("assistant", "earlier reply")

    ctx_on, items_on = assemble_turn_context(manager_on, "system prompt", "hello?")
    ctx_off, items_off = assemble_turn_context(manager_off, "system prompt", "hello?")

    assert [(m.role, m.content) for m in ctx_on] == [
        (m.role, m.content) for m in ctx_off
    ]
    assert items_on == items_off
    snap = telemetry.snapshot()
    assert "memory.context_trimmed_recall" not in snap
    assert "memory.context_trimmed_messages" not in snap


def test_budget_zero_records_estimate_without_trimming(tmp_path: Path) -> None:
    from amadeus.memory.context import assemble_turn_context

    config = _budget_config(tmp_path, context_token_budget=0)
    manager = MemoryManager(config, embedder=VectorEmbedder())
    _seed_facts(manager, 5, pad=2000)  # would blow any real budget

    _ctx, items = assemble_turn_context(manager, "system " * 50, "tell me about fact 0")

    assert len(items) > 0  # nothing was trimmed
    value_entry = telemetry.value_snapshot()["llm.prompt_est_tokens"]
    assert value_entry["count"] == 1
    assert value_entry["last"] > 0
    snap = telemetry.snapshot()
    assert "memory.context_trimmed_recall" not in snap
    assert "memory.context_trimmed_messages" not in snap


def test_estimate_counts_system_prompt_exactly_once(tmp_path: Path) -> None:
    """Regression: context[0] IS the system prompt, and the old call passed
    the prompt to estimate_context_tokens as well — double-counting it
    (~3.4k phantom tokens for a real persona prompt), inflating
    llm.prompt_est_tokens and starting trims ~28% early."""
    from amadeus.memory.context import assemble_turn_context

    config = _budget_config(tmp_path, context_token_budget=0)
    manager = MemoryManager(config, embedder=VectorEmbedder())
    _seed_facts(manager, 2, pad=20)

    system_prompt = "s" * 4000  # 1000 est tokens — double-counting is obvious
    ctx, _items = assemble_turn_context(manager, system_prompt, "tell me about fact 0")

    assert ctx[0].content == system_prompt
    recorded = telemetry.value_snapshot()["llm.prompt_est_tokens"]["last"]
    assert recorded == estimate_context_tokens(None, ctx)
    # Well under 2x the prompt alone: the prompt was counted exactly once.
    assert recorded < 2 * estimate_tokens(system_prompt)


def test_over_budget_drops_recall_tail_first(tmp_path: Path, caplog) -> None:
    """A budget tight enough to require SOME trimming, but loose enough
    that dropping recall alone (never messages) clears it."""
    from amadeus.memory.context import assemble_turn_context

    config = _budget_config(tmp_path, context_token_budget=0)  # set below
    manager = MemoryManager(config, embedder=VectorEmbedder())
    _seed_facts(manager, 4, pad=200)
    manager.add_message("user", "a short earlier turn")
    manager.add_message("assistant", "a short earlier reply")

    system_prompt = "sys " * 10
    # Measure the untrimmed size, then pick a budget between "0 recall
    # items" and "all recall items" so only *some* recall gets dropped.
    ctx_full, items_full = assemble_turn_context(
        manager, system_prompt, "tell me about fact 0"
    )
    assert len(items_full) == 3  # default recall_limit
    # Undo the add_message side effect of the probe call above so the real
    # test call starts from the same short-term state.
    manager.short_term.pop()

    from amadeus.token_budget import estimate_context_tokens

    # The system prompt is already ctx_full[0]; passing it again would
    # double-count it (the bug assemble_turn_context used to have).
    est_full = estimate_context_tokens(None, ctx_full)
    # Budget that only the recall-less context (drop all 3 items) clears:
    # each recall item text is long (pad=200) so dropping even one or two
    # shrinks the estimate a lot; pick a budget just under the full size.
    budget = est_full - 1
    config.memory.context_token_budget = budget

    caplog.set_level("WARNING", logger="amadeus.memory.context")
    ctx, items = assemble_turn_context(manager, system_prompt, "tell me about fact 0")

    assert len(items) < 3  # at least one recall item was dropped
    assert items == items_full[: len(items)]  # dropped from the TAIL, kept prefix
    est_after = estimate_context_tokens(None, ctx)
    assert est_after <= budget or len(items) == 0

    snap = telemetry.snapshot()
    assert snap.get("memory.context_trimmed_recall", 0) >= 1
    assert "memory.context_trimmed_messages" not in snap  # messages untouched
    assert any("context_token_budget exceeded" in rec.message for rec in caplog.records)

    # Conversation messages are completely untouched.
    conv_contents = [m.content for m in ctx if m.role != "system"]
    assert "a short earlier turn" in conv_contents
    assert "a short earlier reply" in conv_contents
    assert conv_contents[-1] == "tell me about fact 0"


def test_over_budget_drops_messages_after_recall_exhausted(tmp_path: Path) -> None:
    """A budget tight enough that dropping ALL recall isn't enough — oldest
    messages must go next, but never below the floor, and the current user
    message must survive."""
    from amadeus.memory.context import assemble_turn_context

    config = _budget_config(tmp_path, context_token_budget=0)
    manager = MemoryManager(config, embedder=VectorEmbedder())
    _seed_facts(manager, 3, pad=50)

    padding = "m" * 300
    for i in range(6):
        manager.add_message("user" if i % 2 == 0 else "assistant", f"{padding} {i}")

    system_prompt = "sys"
    # A near-zero budget forces both stages: all recall dropped, then as
    # many old messages as possible dropped down to the floor.
    config.memory.context_token_budget = 5

    ctx, items = assemble_turn_context(manager, system_prompt, "final question")

    assert items == []  # all recall dropped
    # Exclude the system prompt (ctx[0]); no recall message survives here
    # since budget=5 forces recall to zero first.
    non_system_prefix_msgs = [m for m in ctx[1:]]
    assert len(non_system_prefix_msgs) >= 4  # floor respected
    assert non_system_prefix_msgs[-1].content == "final question"
    assert ctx[0].content == system_prompt  # system prompt never touched

    snap = telemetry.snapshot()
    assert snap.get("memory.context_trimmed_recall", 0) >= 1
    assert snap.get("memory.context_trimmed_messages", 0) >= 1


def test_recall_items_return_matches_surviving_context(tmp_path: Path) -> None:
    """recall_items (the return value) must always match what actually
    rides in the recall system message, even after trimming."""
    from amadeus.memory.context import assemble_turn_context

    config = _budget_config(tmp_path, context_token_budget=0)
    manager = MemoryManager(config, embedder=VectorEmbedder())
    _seed_facts(manager, 3, pad=200)

    system_prompt = "sys " * 5
    ctx_full, items_full = assemble_turn_context(
        manager, system_prompt, "tell me about fact 0"
    )
    manager.short_term.pop()

    from amadeus.token_budget import estimate_context_tokens

    est_full = estimate_context_tokens(None, ctx_full)
    config.memory.context_token_budget = est_full - 1

    ctx, items = assemble_turn_context(manager, system_prompt, "tell me about fact 0")

    recall_messages = [m for m in ctx if m.content.startswith("Long-term memory")]
    if items:
        assert len(recall_messages) == 1
        for item in items:
            assert item in recall_messages[0].content
        # No items beyond the surviving ones leaked into the message body.
        dropped = [i for i in items_full if i not in items]
        for gone in dropped:
            assert gone not in recall_messages[0].content
    else:
        assert recall_messages == []


def test_counters_increment_across_multiple_turns(tmp_path: Path) -> None:
    from amadeus.memory.context import assemble_turn_context

    config = _budget_config(tmp_path, context_token_budget=10)
    manager = MemoryManager(config, embedder=VectorEmbedder())
    _seed_facts(manager, 3, pad=200)

    assemble_turn_context(manager, "sys " * 5, "tell me about fact 0")
    first = telemetry.snapshot().get("memory.context_trimmed_recall", 0)
    assemble_turn_context(manager, "sys " * 5, "tell me about fact 1 now")
    second = telemetry.snapshot().get("memory.context_trimmed_recall", 0)

    assert first > 0
    assert second >= first  # monotonic across turns
