"""Tests for the summary token-economics fixes (debounce + incremental).

The council-specific token-economics tests that used to live here left with
the council itself (2026-07-17).
"""

from __future__ import annotations

import time
from typing import Iterable, List

from amadeus.config import Config
from amadeus.memory.manager import MemoryManager
from amadeus.providers.llm.base import ChatMessage, LLMProvider


class SpyProvider(LLMProvider):
    """Records generate() calls and their kwargs."""

    name = "spy"

    def __init__(self, reply: str = "A fine answer.", delay: float = 0.0) -> None:
        self.calls: List[dict] = []
        self.reply = reply
        self.delay = delay

    def generate(self, messages: Iterable[ChatMessage], **kwargs: object) -> str:
        messages = list(messages)
        start = time.monotonic()
        if self.delay:
            time.sleep(self.delay)
        self.calls.append(
            {
                "messages": messages,
                "kwargs": dict(kwargs),
                "start": start,
                "end": time.monotonic(),
            }
        )
        return self.reply


def test_summary_debounce(tmp_path):
    config = Config()
    config.memory.summary_trigger_messages = 8
    config.memory.max_context_messages = 12
    config.memory.persist_path = str(tmp_path / "memory.json")
    config.memory.enable_long_term = False
    # This test isolates the debounce math for the main summary trigger;
    # the separate first-meeting fast-path trigger (empty profile facts)
    # is out of scope here, so keep it off.
    config.memory.enable_fact_extraction = False

    manager = MemoryManager(config)
    calls = {"count": 0}
    original = manager.summarize

    def counting_summarize():
        calls["count"] += 1
        original()

    manager.summarize = counting_summarize  # type: ignore[method-assign]

    for i in range(12):
        manager.add_message("user" if i % 2 == 0 else "assistant", f"message {i}")

    # Old behavior: summarize on messages 8..12 = 5 calls.
    # Debounced (every max(2, 8//2)=4 messages): messages 8 and 12 = 2 calls.
    assert calls["count"] == 2


def test_incremental_summary_prompt_includes_previous():
    config = Config()
    spy = SpyProvider(reply="An updated summary.")
    manager = MemoryManager(config, summary_provider=spy)

    result = manager._run_summary_job(
        list(manager.short_term) or [],
        revision=2,
        previous_summary="They discussed time travel.",
    )
    assert result["strategy"] == "llm"
    prompt_user = spy.calls[-1]["messages"][1].content
    assert "Previous summary:" in prompt_user
    assert "They discussed time travel." in prompt_user
