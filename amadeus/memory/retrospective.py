"""Retrospective-intent detection for query-aware recall pool selection.

Mirrors ``amadeus/farewell.py::is_goodbye``: a module-level compiled regex,
no LLM call. ``recall_detailed``'s facts-only filter normally excludes
episodes from per-turn recall (they re-narrate whole sessions and duplicate
the rolling summary — see the recall-repetition fix), but a turn that
explicitly asks about the past ("remember when...", "what did we talk about
last week?") can only be answered from an episode. Precision matters here:
a false positive re-admits episodes into an ordinary turn and reopens the
repetition bug this filter fixed, so when in doubt, don't match.
"""

from __future__ import annotations

import re

RETROSPECTIVE_RE = re.compile(
    r"("
    r"remember\s+when|"
    r"do\s+you\s+remember|"
    r"what\s+did\s+(we|you|i)\s+(talk|chat)\s+about|"
    r"what\s+did\s+i\s+(say|tell\s+you)|"
    r"last\s+time\s+we|"
    r"we\s+(discussed|talked\s+about)|"
    r"you\s+mentioned\b.*\b(before|earlier)|"
    r"a\s+while\s+(back|ago)\s+we|"
    r"we\s+.*\ba\s+while\s+(back|ago)"
    r")",
    re.IGNORECASE,
)


def is_retrospective(text: str) -> bool:
    """True when *text* is asking to recall a past conversation/topic."""
    stripped = (text or "").strip()
    if not stripped:
        return False
    return bool(RETROSPECTIVE_RE.search(stripped))
