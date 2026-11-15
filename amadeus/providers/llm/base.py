"""Abstractions for LLM providers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass
class ImageContent:
    """One inline image attachment (vision turns): raw base64, no data: URI."""

    mime: str
    data: str


@dataclass
class ChatMessage:
    role: str
    content: str
    # Vision: images attached to this message (normally at most one, on the
    # current user turn only). None for the overwhelmingly common text-only
    # case, so every existing constructor call stays valid and non-vision
    # providers can ignore the field entirely.
    images: "list[ImageContent] | None" = None


@dataclass
class ToolTurn:
    """One step of a native function-calling turn.

    Exactly one of (tool_name, text) is meaningful:
      * tool_name set  -> the model wants a tool run; args is its input; `raw`
        is the provider's native call object to echo back on the next request.
      * text set       -> the model produced a final answer; no tool needed.
    `history` carries the provider-native conversation so far so the loop can
    append the tool result and continue without rebuilding it."""

    tool_name: str | None = None
    args: dict | None = None
    text: str | None = None
    raw: object = None
    history: object = None


class LLMProvider:
    """Interface for chat-oriented language models."""

    name: str = "base"

    # Token usage of the most recent completed call, when the backend reported
    # it: {"prompt": int, "completion": int}. Purely additive observability —
    # return types of generate()/stream()/generate_with_tools() are unchanged.
    # Providers that don't surface usage (dummy, local) leave it None, and
    # every consumer must tolerate None (see telemetry.record_llm_usage).
    last_usage: dict | None = None

    def generate(
        self, messages: Iterable[ChatMessage], **kwargs: object
    ) -> str:  # pragma: no cover - interface
        raise NotImplementedError

    def stream(
        self, messages: Iterable[ChatMessage], **kwargs: object
    ):  # pragma: no cover - optional
        yield self.generate(messages, **kwargs)

    def supports_stream(self) -> bool:
        return False

    # --- Native tool/function calling (optional capability) ------------------
    # Providers that support structured function calling override these. The
    # tool loop prefers this path (no JSON-in-text to malform) and falls back
    # to text ReAct when supports_tools() is False.

    def supports_tools(self) -> bool:
        return False

    def supports_vision(self) -> bool:
        """Whether this provider can consume ChatMessage.images. Providers
        that return False silently receive text-only messages (the web layer
        drops attachments before they get here)."""
        return False

    def generate_with_tools(
        self, messages: Iterable[ChatMessage], tools: list, **kwargs: object
    ):  # pragma: no cover - interface
        """Return a ToolTurn: either a tool call to execute, or final text.

        `tools` is a list of provider-agnostic tool specs
        ({name, description, parameters(json-schema)}). Implementations translate
        to their native format. Not called unless supports_tools() is True."""
        raise NotImplementedError


class DummyProvider(LLMProvider):
    """Fallback provider that mirrors the prompt for debugging.

    Accepts an optional ``scripted_reply`` so tests can drive deterministic
    (optionally tagged) output through the streaming pipeline without a real
    provider.
    """

    name = "dummy"

    def __init__(self, scripted_reply: str | None = None) -> None:
        self.scripted_reply = scripted_reply

    # Shown (in place of a real reply) when no LLM API key is configured, so a
    # first-run user gets an in-character prompt to set one up instead of a
    # debug echo. Tagged format ([emotion] English || Japanese) so it flows
    # through the normal emotion/voice pipeline and Kurisu actually says it.
    NO_KEY_REPLY = (
        "[relaxed] Tch — I can't reach my mind right now. There's no API key set, "
        "so I'm running on an empty shell. || ちっ、今は頭が回らないの。APIキーが"
        "設定されてないのよ。\n"
        "[neutral] Open Settings and add a Gemini API key — it's free to get one — "
        "then I'll actually be myself. || 設定からGeminiのAPIキーを入れて。無料で"
        "取れるわ。そうすれば本来の私になれる。"
    )

    def generate(self, messages: Iterable[ChatMessage], **kwargs: object) -> str:
        if self.scripted_reply is not None:
            return self.scripted_reply
        return self.NO_KEY_REPLY

    def stream(self, messages: Iterable[ChatMessage], **kwargs: object):
        text = self.generate(messages, **kwargs)
        # Chunk on whitespace boundaries (with the trailing space kept) to
        # exercise multi-delta parsing without being unrealistically fine.
        start = 0
        for idx, char in enumerate(text):
            if char == " ":
                yield text[start : idx + 1]
                start = idx + 1
        if start < len(text):
            yield text[start:]

    def supports_stream(self) -> bool:
        return True
