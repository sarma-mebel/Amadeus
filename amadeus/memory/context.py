"""Shared per-turn context assembly for the CLI and web chat loops."""

from __future__ import annotations

import datetime as dt
import logging
import re
from typing import List, Optional, Tuple

from .. import telemetry
from ..providers.llm.base import ChatMessage
from ..token_budget import estimate_context_tokens
from .manager import MemoryManager
from .retrospective import is_retrospective
from .text_utils import (  # noqa: F401 (re-export)
    _age_phrase,
    _already_discussed,
    _confidence_clause,
    _event_clause,
    _raw_query_too_short,
)

LOGGER = logging.getLogger(__name__)

# Never trim short-term conversation messages below this many entries — the
# current user message (always last) counts toward this floor and is never
# itself a candidate for removal.
_MIN_CONTEXT_MESSAGES = 4

RECALL_FRAMING = (
    "Long-term memory surfaced for this turn (things you genuinely remember "
    "about the user). These are background — the user's CURRENT message "
    "always comes first. Use a memory only when it directly serves what they "
    "are saying right now; weave it in naturally, never recite it. Mention "
    "any one remembered topic at most once, and if the user moves on or "
    "doesn't pick it up, let it go — do not steer the conversation back to "
    "old topics just because you remember them. Each carries roughly when "
    "you learned it — and, for dated events, roughly when the event itself "
    'happened or will happen; reference timing conversationally ("you '
    'mentioned a while back..."), never as an exact date. Some also note how '
    'sure you are: when a memory says you are unsure, say so naturally ("if '
    "I'm remembering right...\") instead of stating it flatly, and if the "
    "user corrects you, take the correction — never defend a shaky memory. "
    "These are the ONLY long-term memories "
    "surfaced this "
    "turn — do not claim recollections beyond them, the summary, and this "
    "conversation:"
)


# _age_phrase moved to text_utils.py (graph.py needs it too and importing
# from here would cycle through manager.py); re-exported above unchanged.


def _relational_prefix(metadata: dict) -> str:
    """ "Between you two" marker for relational moments, else "".

    RECALL_FRAMING introduces these items as things she knows ABOUT the
    user, which is right for facts and wrong for a shared event: rendered
    bare, "we finally talked about why you left the lab" invites her to
    assert it AT them rather than reference it as history. Costs 15 chars,
    and only on relational records — every other type renders
    byte-identically to before (pinned by test).
    """
    if metadata.get("source") == "relational":
        return "between you two: "
    return ""


def _recall_line(match, now: dt.datetime) -> str:
    """Render one recalled item, annotated with learn time and — when the
    fact carries a bitemporal ``valid_at`` stamp — its event time. Shared by
    the initial render and the token-budget re-render below so the two can't
    drift (they used to be duplicated verbatim)."""
    metadata = match.metadata or {}
    age = _age_phrase(metadata.get("timestamp"), now)
    event = _event_clause(metadata.get("valid_at"), now)
    # Calibrated recall: how sure she should sound. None for confident facts
    # (the common case), so a solid memory renders exactly as it always did
    # and costs no extra prompt tokens.
    hedge = _confidence_clause(getattr(match, "confidence", None))
    parens = "; ".join(part for part in (age, event, hedge) if part)
    text = f"{_relational_prefix(metadata)}{match.text}"
    return f"- {text} ({parens})" if parens else f"- {text}"


# Query augmentation for pronoun-heavy follow-ups ("did it work out?"):
# such a message embeds to near-noise on its own, so the recall QUERY (never
# recall_items, never what's stored/displayed) gets the previous user turn
# prepended for embedding context. Content-gated, NOT length-gated: the old
# <40-chars shortcut augmented every short message, so a self-contained
# question ("what's a tensor?") got the unrelated previous turn concatenated
# in and — measured on the gemini embedder — the shifted embedding polluted
# recall for exactly the short, focused questions recall serves best. Kept
# deliberately simple: a "no standalone content word" check rather than real
# coreference resolution.
_STOPWORDS = {
    "a",
    "about",
    "an",
    "and",
    "are",
    "at",
    "be",
    "but",
    "by",
    "did",
    "do",
    "does",
    "for",
    "from",
    "had",
    "has",
    "have",
    "he",
    "her",
    "here",
    "him",
    "his",
    "how",
    "i",
    "if",
    "in",
    "is",
    "it",
    "its",
    "me",
    "mine",
    "my",
    "of",
    "on",
    "or",
    "our",
    "out",
    "she",
    "so",
    "than",
    "that",
    "the",
    "them",
    "then",
    "there",
    "they",
    "this",
    "to",
    "up",
    "us",
    "was",
    "we",
    "were",
    "what",
    "when",
    "where",
    "who",
    "why",
    "will",
    "with",
    "would",
    "you",
    "your",
    "yours",
}


def _needs_query_augmentation(text: str) -> bool:
    """True when *text* carries no standalone content word.

    A token only counts as a content word by its pre-apostrophe base, so
    contractions of function words ("what's", "how's", "didn't") never
    qualify — otherwise "what's it about?" would read as self-contained.
    """
    stripped = (text or "").strip()
    if not stripped:
        return False
    tokens = re.findall(r"[a-zA-Z']+", stripped.lower())
    if not tokens:
        return True
    for tok in tokens:
        base = tok.split("'", 1)[0]
        if len(base) > 4 and base not in _STOPWORDS:
            return False
    return True


_TRAILING_PAREN_RE = re.compile(r"\s*\([^()]*\)\s*$")


def _strip_age_suffix(line: str) -> str:
    """Drop a graph line's leading "- " and trailing " (...)" age/provenance
    parenthetical(s) before comparing against vector-recall text — otherwise
    "(earlier today)" / "(previously: ...)" tokens lower the lexical overlap
    ratio below the dedupe threshold even when the underlying fact is the
    exact same one vector recall already surfaced."""
    text = line.lstrip("- ").strip()
    # Graph lines can carry two trailing parens (age, then supersede
    # provenance) — strip repeatedly rather than just once.
    while True:
        stripped = _TRAILING_PAREN_RE.sub("", text)
        if stripped == text:
            return text
        text = stripped


def _last_user_text(memory: MemoryManager) -> Optional[str]:
    """Previous user turn, i.e. the last role=="user" entry in short-term
    context excluding the just-appended current message."""
    window = list(memory.short_term)
    if window and window[-1].role == "user":
        window = window[:-1]
    for msg in reversed(window):
        if msg.role == "user":
            return msg.content
    return None


def assemble_turn_context(
    memory: MemoryManager,
    system_prompt: str,
    user_text: str,
    *,
    recall_limit: int = 3,
) -> Tuple[List[ChatMessage], List[str]]:
    """Commit the user message and build the LLM request context.

    Returns ``(context, recall_items)``. The recalled memories ride as a
    trailing system message so the cached prompt prefix stays untouched.
    """
    memory.add_message("user", user_text)
    context = [
        ChatMessage(role=msg["role"], content=msg["content"])
        for msg in memory.build_context()
    ]
    context.insert(0, ChatMessage(role="system", content=system_prompt))
    min_chars = int(getattr(memory.config.memory, "recall_min_query_chars", 0) or 0)
    matches = []
    # Gate on the RAW user turn before augmentation: a short turn ("did it
    # work?") that gets the previous turn prepended would otherwise sail
    # past the trivial-turn latency gate it was meant for (see
    # _raw_query_too_short's docstring). recall_detailed's own internal gate
    # only ever sees the (possibly augmented) query, so this pre-check is
    # what actually enforces "short raw turns skip recall".
    if not _raw_query_too_short(user_text, min_chars):
        recall_query = user_text
        if _needs_query_augmentation(user_text):
            previous_user_text = _last_user_text(memory)
            if previous_user_text:
                recall_query = f"{previous_user_text}\n{user_text}"
        matches = memory.recall_detailed(
            recall_query,
            limit=recall_limit,
            include_episodes=is_retrospective(user_text),
        )
    recall_items = [match.text for match in matches]
    telemetry.incr("memory.turns")
    telemetry.set_gauge("memory.last_recall_items", len(recall_items))
    if not recall_items:
        telemetry.incr("memory.recall_empty_turns")
    # Entity-graph recall (memory.graph_enabled, default off): name/alias
    # matches against the current message pull that entity's current facts
    # + one-hop relations, merged AHEAD of vector recall and deduped against
    # it — the same "subordinate to the current message, mention once"
    # framing below governs both. A no-op call when the flag is off.
    graph_lines = memory.recall_from_graph(user_text)
    if graph_lines:
        recall_items_lower = [item.lower() for item in recall_items]
        graph_lines = [
            line
            for line in graph_lines
            if not any(
                _already_discussed(_strip_age_suffix(line), item, 0.8)
                for item in recall_items_lower
            )
        ]
    recall_message_index: Optional[int] = None
    if recall_items or graph_lines:
        now = dt.datetime.utcnow()
        lines = list(graph_lines)
        for match in matches:
            lines.append(_recall_line(match, now))
        recall_text = RECALL_FRAMING + "\n" + "\n".join(lines)
        recall_message_index = len(context)
        context.append(ChatMessage(role="system", content=recall_text))

    # --- Token budget: measure every turn, trim only over budget. ---------
    # This is a pre-tool-prompt estimate: the tool loop (agents/tool_loop.py)
    # appends its own tool-instructions prompt AFTER assemble_turn_context
    # returns, on tool turns only, so the real request can run somewhat
    # higher than this number on those turns.
    # The system prompt is already context[0] (inserted above), so it is
    # passed as None here — passing it again double-counted it (~3.4k
    # phantom tokens against the 12k budget, starting trims ~28% early and
    # inflating llm.prompt_est_tokens).
    est_before = estimate_context_tokens(None, context)
    telemetry.record_value("llm.prompt_est_tokens", est_before)

    budget = int(getattr(memory.config.memory, "context_token_budget", 0) or 0)
    if budget > 0 and est_before > budget:
        trimmed_recall = 0
        trimmed_messages = 0

        # Stage 1: drop recall items lowest-ranked-first (tail of the
        # already best-first-ranked `matches`/`recall_items` list). Graph
        # lines are left alone here — they aren't part of the ranked
        # vector-recall list `recall_items` describes, and there's no
        # meaningful "lowest ranked" ordering for them.
        while recall_items and estimate_context_tokens(None, context) > budget:
            recall_items.pop()
            matches.pop()
            trimmed_recall += 1
            telemetry.incr("memory.context_trimmed_recall")
            if recall_items or graph_lines:
                lines = list(graph_lines)
                for match in matches:
                    lines.append(_recall_line(match, now))
                recall_text = RECALL_FRAMING + "\n" + "\n".join(lines)
                context[recall_message_index] = ChatMessage(
                    role="system", content=recall_text
                )
            else:
                # Nothing left to say — drop the recall message entirely.
                del context[recall_message_index]
                recall_message_index = None

        # Stage 2: drop oldest short-term messages. context[0] is always
        # the system prompt; if a recall message survived stage 1 it's the
        # last element. Conversation messages are whatever's between those
        # two — trim from the front (oldest), never past the floor, and
        # never the current user message (always the last conversation
        # message, so the floor already protects it).
        conv_end = (
            recall_message_index if recall_message_index is not None else len(context)
        )
        conv_start = 1  # after the system prompt
        while (
            conv_end - conv_start
        ) > _MIN_CONTEXT_MESSAGES and estimate_context_tokens(None, context) > budget:
            del context[conv_start]
            conv_end -= 1
            if recall_message_index is not None:
                recall_message_index -= 1
            trimmed_messages += 1
            telemetry.incr("memory.context_trimmed_messages")

        if trimmed_recall or trimmed_messages:
            est_after = estimate_context_tokens(None, context)
            LOGGER.warning(
                "context_token_budget exceeded: %d est tokens > budget %d; "
                "trimmed %d recall item(s) and %d message(s); now ~%d est "
                "tokens",
                est_before,
                budget,
                trimmed_recall,
                trimmed_messages,
                est_after,
            )
        # If still over budget after both stages (floor reached, nothing
        # left to trim), we proceed anyway — degrade loudly, don't fail the
        # turn. The warning above already fired if anything was trimmed; if
        # NOTHING could be trimmed (e.g. the floor was already binding) the
        # estimate recorded above still reflects the true, untrimmed size.

    return context, recall_items
