"""Cheap, dependency-free token estimation for prompt-size bookkeeping.

Nothing in this repo previously measured actual prompt size — every bound
was on message COUNTS or CHARACTER lengths, so a window of long pasted
messages plus a large persona block could blow a model's context with no
warning (see ``MemorySettings.context_token_budget``). This module supplies
a cheap heuristic estimate good enough to gate that, not a real tokenizer.

Deliberately has zero imports from the rest of the package (no manager, no
engine, no providers) so any call site — memory, tools, providers, tests —
can use it without pulling in the world or risking an import cycle.
"""

from __future__ import annotations

import math
from typing import Iterable, Optional, Sequence


def estimate_tokens(text: Optional[str]) -> int:
    """Rough token count for *text*: ``ceil(len(text) / 4)``.

    A deliberately cheap heuristic, not a real tokenizer: English prose
    averages roughly 4 characters per token, so this is a reasonable proxy
    for English text. Japanese (and other dense scripts) run far fewer
    characters per token, so this UNDER-estimates JA-heavy text — the
    configured budget is generous specifically to absorb that slack rather
    than pretend at precision this heuristic can't deliver.
    """
    if not text:
        return 0
    return math.ceil(len(text) / 4)


def estimate_context_tokens(
    system_prompt: Optional[str],
    messages: Sequence[object],
    recall_text: Optional[str] = None,
) -> int:
    """Sum the estimated tokens of a system prompt, a list of messages, and
    an optional recall block.

    ``messages`` items only need a ``.content`` string attribute (this
    matches ``ChatMessage`` from ``providers.llm.base`` without importing
    it, keeping this module dependency-free). Anything without a usable
    ``.content`` is skipped rather than raising — this is bookkeeping, not
    a contract that should ever break a turn.
    """
    total = estimate_tokens(system_prompt)
    for msg in messages:
        content = getattr(msg, "content", None)
        if isinstance(content, str):
            total += estimate_tokens(content)
    if recall_text:
        total += estimate_tokens(recall_text)
    return total


def estimate_texts_tokens(texts: Iterable[str]) -> int:
    """Sum of ``estimate_tokens`` over an iterable of strings.

    Small helper for summing e.g. a list of recall item strings without
    building an intermediate ChatMessage-like wrapper.
    """
    return sum(estimate_tokens(t) for t in texts)
