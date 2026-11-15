"""Rate-limit failover composite LLM provider.

Wraps a primary provider (e.g. Gemini free tier) and a fallback (e.g. OpenRouter
:free via the OpenAI-compatible adapter). On a rate-limit/quota error — and ONLY
that — the same request is retried on the fallback. Auth errors, bad requests,
timeouts, and safety blocks re-raise unchanged (they'd fail on the fallback too,
and masking them hides misconfiguration).
"""

from __future__ import annotations

import logging
from typing import Iterable

from .base import ChatMessage, LLMProvider

LOGGER = logging.getLogger(__name__)


def _is_rate_limit(exc: BaseException) -> bool:
    """True if `exc` looks like a rate-limit / quota-exceeded error from any
    provider. Duck-typed + guarded imports so neither SDK is a hard dependency."""
    # Gemini
    try:
        from google.api_core.exceptions import ResourceExhausted  # type: ignore

        if isinstance(exc, ResourceExhausted):
            return True
    except Exception:  # pragma: no cover - google SDK absent
        pass
    if type(exc).__name__ == "ResourceExhausted":
        return True
    # OpenAI / OpenRouter
    try:
        from openai import RateLimitError  # type: ignore

        if isinstance(exc, RateLimitError):
            return True
    except Exception:  # pragma: no cover - openai SDK absent
        pass
    if type(exc).__name__ == "RateLimitError":
        return True
    # Generic HTTP 429
    for attr in ("status_code", "code", "http_status"):
        if getattr(exc, attr, None) == 429:
            return True
    return False


class FallbackLLMProvider(LLMProvider):
    """Primary provider with rate-limit failover to a secondary."""

    def __init__(self, primary: LLMProvider, fallback: LLMProvider) -> None:
        self.primary = primary
        self.fallback = fallback
        self.name = f"{getattr(primary, 'name', '?')}+{getattr(fallback, 'name', '?')}"
        # Whichever delegate actually served the most recent call; last_usage
        # (below) reads through it so token telemetry attributes usage to the
        # provider that really answered, not blindly to the primary.
        self._active: LLMProvider = primary

    @property
    def last_usage(self) -> dict | None:
        return getattr(self._active, "last_usage", None)

    def generate(self, messages: Iterable[ChatMessage], **kwargs: object) -> str:
        msgs = list(messages)  # may be consumed twice
        self._active = self.primary
        try:
            return self.primary.generate(msgs, **kwargs)
        except Exception as exc:
            if not _is_rate_limit(exc):
                raise
            LOGGER.warning(
                "Primary LLM (%s) rate-limited; failing over to %s",
                getattr(self.primary, "name", "?"),
                getattr(self.fallback, "name", "?"),
            )
            self._active = self.fallback
            return self.fallback.generate(msgs, **kwargs)

    def stream(self, messages: Iterable[ChatMessage], **kwargs: object):
        msgs = list(messages)
        self._active = self.primary
        try:
            it = self.primary.stream(msgs, **kwargs)
            first = next(it)  # a lazily-raised 429 surfaces here
        except StopIteration:
            return  # empty stream, no failover needed
        except Exception as exc:
            if not _is_rate_limit(exc):
                raise
            LOGGER.warning(
                "Primary LLM (%s) rate-limited mid-stream-open; " "failing over to %s",
                getattr(self.primary, "name", "?"),
                getattr(self.fallback, "name", "?"),
            )
            self._active = self.fallback
            if self.fallback.supports_stream():
                yield from self.fallback.stream(msgs, **kwargs)
            else:
                yield self.fallback.generate(msgs, **kwargs)
            return
        # Primary is producing output — yield it. A mid-stream failure after
        # chunks were emitted re-raises (restarting would duplicate output).
        yield first
        yield from it

    def supports_stream(self) -> bool:
        return self.primary.supports_stream()

    def supports_tools(self) -> bool:
        return self.primary.supports_tools()

    def supports_vision(self) -> bool:
        # Forward like the other supports_* capabilities. Omitting this left
        # the base class's False in place, which silently hard-disabled every
        # vision feature (screen peek, read-along, watch-together, image
        # turns) whenever the session LLM was wrapped in this fallback.
        return self.primary.supports_vision()

    def generate_with_tools(self, messages, tools, **kwargs):
        # Native function calling goes to the primary. On a rate-limit we can't
        # cleanly resume a native tool turn on a different provider mid-loop, so
        # let the caller's own fallback (text loop) handle that path.
        self._active = self.primary
        return self.primary.generate_with_tools(messages, tools, **kwargs)
