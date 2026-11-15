"""Goodbye-intent detection for the anti-dark-pattern farewell.

Detection is deliberately regex-based (no LLM call): the farewell itself is
the only generation cost, and the server re-validates whatever the client
pre-gated so the two never disagree. The frontend mirrors this pattern in
``frontend/src/farewell.ts`` — keep the two in sync.
"""

from __future__ import annotations

import re

# Long messages that merely contain "bye" (e.g. recounting a story about a
# goodbye) are not farewells; a real sign-off is short. The word gate kills
# sentences *about* goodbyes; the question gate kills "is goodbye a noun?" —
# a false negative here just becomes a normal chat turn, so precision wins.
MAX_GOODBYE_LEN = 80
MAX_GOODBYE_WORDS = 8

GOODBYE_RE = re.compile(
    r"\b("
    r"good\s?bye|bye+|bye\s?now|good\s?night|g'?night|"
    r"gotta\s+(go|run|sleep|head\s+(out|off))|"
    r"(i'?m|i\s+am)\s+(off|heading\s+(out|off|to\s+bed)|going\s+to\s+(bed|sleep))|"
    r"see\s+(you|ya)(\s+(later|tomorrow|soon|tonight))?|"
    r"talk\s+(to\s+you\s+)?(later|tomorrow|soon)|"
    r"ttyl|signing\s+off|logging\s+off|heading\s+(out|off)|"
    r"oyasumi"
    r")\b",
    re.IGNORECASE,
)


def is_goodbye(text: str) -> bool:
    """True when *text* reads as the user ending the conversation."""
    stripped = (text or "").strip()
    if not stripped or len(stripped) > MAX_GOODBYE_LEN:
        return False
    if len(stripped.split()) > MAX_GOODBYE_WORDS:
        return False
    if stripped.endswith("?"):
        return False
    return bool(GOODBYE_RE.search(stripped))
