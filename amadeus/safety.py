"""Very small content guardrails."""

from __future__ import annotations

from typing import Optional

PII_KEYWORDS = {
    "social security",
    "credit card",
    "password",
}


def guardrail_response(user_input: str) -> Optional[str]:
    lowered = user_input.lower()
    if any(keyword in lowered for keyword in PII_KEYWORDS):
        return "Keep personal data private. I won't help collect or share sensitive information."
    return None
