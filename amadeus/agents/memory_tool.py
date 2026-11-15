"""
Model-facing memory tool: lets the LLM act on explicit user memory
instructions ("remember this", "no, I moved to Berlin", "forget that")
deterministically, instead of hoping the background fact extractor
volunteers a replaces-hint that happens to embed close enough.

There is deliberately NO read/search action here — recall (per-turn context
assembly) already surfaces relevant facts automatically; adding a redundant
lookup action would just tempt the model into narrating its own memory
instead of talking to the user.
"""

import inspect
import logging
from typing import Any, List, Optional

from amadeus import telemetry
from amadeus.agents.tools import (
    BaseTool,
    ToolCategory,
    ToolDefinition,
    ToolParameter,
    ToolResult,
)

LOGGER = logging.getLogger(__name__)

# A find_facts() match at or above this cosine score is treated as "this is
# clearly the fact the user means" and is safe to archive/supersede without
# asking again. Below it we abstain rather than guess wrong (see the
# module docstring on forget/correct below).
_MATCH_THRESHOLD = 0.75


class MemoryTool(BaseTool):
    """Store, correct, or archive a durable fact about the user, on their
    explicit instruction.

    Backed by ``MemoryManager.store_user_fact`` / ``find_facts`` /
    ``archive_fact``. Never hard-deletes — "forget" archives (recoverable,
    excluded from recall), it never destroys data.
    """

    def __init__(self, memory: Any):
        self.memory = memory
        super().__init__()

    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="memory",
            description=(
                "Remember, correct, or forget a durable fact about the user, "
                "on their explicit instruction. Use this ONLY when the user "
                "directly asks you to remember/forget/correct something "
                "about themselves or their life ('remember that I...', "
                "'no, I moved to Berlin, not Munich', 'forget what I said "
                "about my ex') — never call it spontaneously to log details "
                "the user just happened to mention in passing, and never to "
                "store your own thoughts or opinions; ordinary facts are "
                "already captured automatically by background memory "
                "extraction. There is no read/search action — recall "
                "surfaces relevant memories on its own; you don't need to "
                "look anything up here. After calling this tool, confirm to "
                "the user in character what was stored or dropped, briefly, "
                "in your own words — never recite internal ids or "
                "similarity scores back to them.\n\n"
                "action=remember: stores `text` verbatim as a new fact.\n"
                "action=correct: finds the existing fact matching `text` "
                "(the outdated version, as the user described it) and "
                "replaces it with `replacement` (required); if no confident "
                "match is found, stores `replacement` as a new fact anyway "
                "since the correction is true either way.\n"
                "action=forget: finds the existing fact matching `text` and "
                "archives it (recoverable, never surfaced again). If no "
                "candidate is a confident match, nothing is archived — you "
                "will get back the closest candidates instead so you can "
                "retry quoting their stored wording, or tell the user "
                "honestly that you don't have a memory matching that "
                "description rather than guessing."
            ),
            category=ToolCategory.DATA,
            parameters=[
                ToolParameter(
                    name="action",
                    type="string",
                    description="remember | correct | forget",
                    required=True,
                    enum=["remember", "correct", "forget"],
                ),
                ToolParameter(
                    name="text",
                    type="string",
                    description=(
                        "remember: the fact to store, in your own clear "
                        "words. forget: the user's description of what to "
                        "forget. correct: the outdated fact, as the user "
                        "described it."
                    ),
                    required=True,
                ),
                ToolParameter(
                    name="replacement",
                    type="string",
                    description="correct only: the corrected fact to store.",
                    required=False,
                ),
                ToolParameter(
                    name="when",
                    type="string",
                    description=(
                        "remember/correct only, optional: ISO date the "
                        "fact's event happened or will happen (e.g. "
                        "'2026-08-15'), ONLY when the user stated it — "
                        "never guess."
                    ),
                    required=False,
                ),
            ],
            returns=(
                "A short instruction describing what was stored/archived (or, "
                "for a failed forget, the nearest candidates and how to "
                "proceed) — not meant to be read aloud verbatim."
            ),
            examples=[
                {"action": "remember", "text": "The user's dog is named Pixel."},
                {
                    "action": "correct",
                    "text": "the user lives in Munich",
                    "replacement": "The user lives in Berlin now, not Munich.",
                },
                {"action": "forget", "text": "what I said about my ex"},
            ],
            cost_tier=1,
        )

    def execute(self, **kwargs) -> ToolResult:
        validation_error = self.validate_parameters(kwargs)
        if validation_error:
            return ToolResult(
                tool_name="memory",
                success=False,
                result=None,
                error=validation_error,
            )

        action = (kwargs.get("action") or "").strip().lower()
        text = kwargs.get("text")
        if not text or not str(text).strip():
            return ToolResult(
                tool_name="memory",
                success=False,
                result=None,
                error=f"action={action} requires non-empty 'text'.",
            )
        text = str(text).strip()

        if action == "correct":
            replacement = kwargs.get("replacement")
            if not replacement or not str(replacement).strip():
                return ToolResult(
                    tool_name="memory",
                    success=False,
                    result=None,
                    error="action=correct requires non-empty 'replacement'.",
                )

        when = str(kwargs.get("when") or "").strip() or None

        try:
            if action == "remember":
                return self._remember(text, when)
            if action == "forget":
                return self._forget(text)
            if action == "correct":
                return self._correct(text, str(kwargs.get("replacement")).strip(), when)
            return ToolResult(
                tool_name="memory",
                success=False,
                result=None,
                error=f"Unknown action '{action}' (remember|correct|forget).",
            )
        except Exception as e:
            LOGGER.warning("memory tool failed (action=%s): %s", action, e)
            telemetry.incr("tools.memory_failed")
            return ToolResult(
                tool_name="memory",
                success=False,
                result=None,
                error=f"Memory operation failed: {e}",
            )

    # -- actions ----------------------------------------------------------

    def _archive_fact(self, record_id: str, reason: str) -> Optional[str]:
        """Archive via the manager, letting the explicit user instruction
        beat a pin (the the project notes contract: "explicit forget beats pins" —
        every call here follows a direct user request, so
        ``allow_pinned=True`` always applies).

        The manager's bound ``archive_fact`` may predate the
        ``allow_pinned`` parameter (manager.py's thin delegate is owned by
        another workstream): when it does, reach the underlying
        ``fact_archive.archive_fact`` helper directly — counted, so the
        shim's survival stays visible — and only a manager exposing
        neither surface falls back to the old refuse-pinned call.
        """
        archive = self.memory.archive_fact
        try:
            accepts = "allow_pinned" in inspect.signature(archive).parameters
        except (TypeError, ValueError):  # builtins / exotic callables
            accepts = False
        if accepts:
            return archive(record_id, reason, allow_pinned=True)
        if getattr(self.memory, "vector_store", None) is not None:
            from amadeus.memory import fact_archive

            telemetry.incr("tools.memory_archive_pinned_shim")
            return fact_archive.archive_fact(
                self.memory, record_id, reason, allow_pinned=True
            )
        return archive(record_id, reason)

    def _remember(self, text: str, when: Optional[str] = None) -> ToolResult:
        ok = self.memory.store_user_fact(text, valid_at=when)
        if not ok:
            return ToolResult(
                tool_name="memory",
                success=False,
                result=None,
                error="Could not store that fact — the memory store rejected it.",
            )
        telemetry.incr("tools.memory_remember")
        return ToolResult(
            tool_name="memory",
            success=True,
            result=(
                f'Stored verbatim: "{text}". Confirm naturally to the user '
                "that you'll remember it — don't recite this exact wording "
                "back like a receipt."
            ),
            metadata={"action": "remember"},
        )

    def _forget(self, text: str) -> ToolResult:
        candidates: List[dict] = self.memory.find_facts(text, k=3) or []
        top = candidates[0] if candidates else None

        if not top or top.get("score", 0.0) < _MATCH_THRESHOLD:
            telemetry.incr("tools.memory_forget_no_match")
            if not candidates:
                hint = "No stored facts resemble that description at all."
            else:
                lines = "; ".join(
                    f"\"{c.get('text', '')}\" (score {c.get('score', 0):.2f})"
                    for c in candidates
                )
                hint = f"Closest candidates, none confident enough: {lines}."
            return ToolResult(
                tool_name="memory",
                success=True,
                result=(
                    f"{hint} Nothing was archived. Either retry action=forget "
                    "quoting the stored wording more closely, or tell the "
                    "user honestly you don't have a memory matching that — "
                    "don't guess or pretend to forget something you can't "
                    "find."
                ),
                metadata={"action": "forget", "matched": False},
            )

        archived_text = self._archive_fact(top["id"], "user_forget")
        if archived_text is None:
            if top.get("pinned"):
                # The archive path refused a pinned record even for an
                # explicit request (a manager without allow_pinned support).
                # Say so honestly instead of claiming the memory is gone.
                telemetry.incr("tools.memory_forget_pinned_refused")
                return ToolResult(
                    tool_name="memory",
                    success=True,
                    result=(
                        f'Found it — "{top.get("text", "")}" — but it is a '
                        "pinned memory and could not be removed from here. "
                        "Tell the user it's pinned, and that they can unpin "
                        "or delete it in the memory panel."
                    ),
                    metadata={"action": "forget", "matched": True, "pinned": True},
                )
            telemetry.incr("tools.memory_forget_no_match")
            return ToolResult(
                tool_name="memory",
                success=True,
                result=(
                    "That memory no longer exists (it may already have been "
                    "removed). Nothing to forget."
                ),
                metadata={"action": "forget", "matched": False},
            )

        telemetry.incr("tools.memory_forget")
        pin_note = (
            " Note: this was a pinned memory — archived anyway per the user's explicit request."
            if top.get("pinned")
            else ""
        )
        return ToolResult(
            tool_name="memory",
            success=True,
            result=(
                f'Archived: "{archived_text}".{pin_note} Confirm to the '
                "user, in your own words, exactly what you're letting go of."
            ),
            metadata={
                "action": "forget",
                "matched": True,
                "pinned": bool(top.get("pinned")),
            },
        )

    def _correct(
        self, text: str, replacement: str, when: Optional[str] = None
    ) -> ToolResult:
        candidates: List[dict] = self.memory.find_facts(text, k=3) or []
        top = candidates[0] if candidates else None

        previous_fact: Optional[str] = None
        if top and top.get("score", 0.0) >= _MATCH_THRESHOLD:
            # Explicit correction archives even a pinned old fact — leaving
            # it in place would keep the outdated version alongside (and
            # recalled beside) the correction.
            archived_text = self._archive_fact(top["id"], "user_correct")
            if archived_text is not None:
                previous_fact = archived_text

        ok = self.memory.store_user_fact(
            replacement, previous_fact=previous_fact, valid_at=when
        )
        if not ok:
            return ToolResult(
                tool_name="memory",
                success=False,
                result=None,
                error="Could not store the corrected fact — the memory store rejected it.",
            )

        telemetry.incr("tools.memory_correct")
        if previous_fact is not None:
            result = (
                f'Replaced "{previous_fact}" with "{replacement}". '
                "Confirm the correction to the user naturally."
            )
        else:
            result = (
                f"No confident match for the old fact was found, so stored "
                f'the correction as new: "{replacement}" (it\'s true '
                "regardless of whether the outdated version was on record). "
                "Confirm to the user naturally."
            )
        return ToolResult(
            tool_name="memory",
            success=True,
            result=result,
            metadata={"action": "correct", "matched": previous_fact is not None},
        )
