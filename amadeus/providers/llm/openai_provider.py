"""OpenAI Chat Completions backend."""

from __future__ import annotations

from typing import Iterable

from .base import ChatMessage, LLMProvider


class OpenAIProvider(LLMProvider):
    """Adapter around OpenAI's chat completion API."""

    name = "openai"

    def __init__(
        self, model: str, api_key: str | None = None, base_url: str | None = None
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "openai package is required for the OpenAI provider"
            ) from exc

        # base_url lets this same adapter target any OpenAI-compatible endpoint
        # (e.g. OpenRouter at https://openrouter.ai/api/v1).
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    @staticmethod
    def _request_kwargs(kwargs: dict) -> dict:
        request: dict = {"temperature": kwargs.get("temperature", 0.8)}
        max_tokens = kwargs.get("max_tokens")
        if max_tokens:
            request["max_tokens"] = int(max_tokens)
        return request

    @staticmethod
    def _serialize_message(msg: ChatMessage) -> dict:
        """String content for text-only messages (byte-identical to the
        historical requests); content-parts form only when images ride along
        (vision turns)."""
        images = getattr(msg, "images", None)
        if not images:
            return {"role": msg.role, "content": msg.content}
        parts: list = []
        if msg.content:
            parts.append({"type": "text", "text": msg.content})
        for image in images:
            parts.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{image.mime};base64,{image.data}"},
                }
            )
        return {"role": msg.role, "content": parts}

    def _capture_usage(self, response: object) -> None:
        """Record token usage from a response/chunk onto ``last_usage``.

        Best-effort: ``usage`` may be absent (some OpenAI-compatible
        endpoints omit it) — then ``last_usage`` stays as-is (callers reset
        it to None at call start), and this never raises.
        """
        try:
            usage = getattr(response, "usage", None)
            if usage is None:
                return
            self.last_usage = {
                "prompt": int(getattr(usage, "prompt_tokens", 0) or 0),
                "completion": int(getattr(usage, "completion_tokens", 0) or 0),
            }
        except Exception:  # pragma: no cover - observability must never raise
            pass

    def generate(self, messages: Iterable[ChatMessage], **kwargs: object) -> str:
        self.last_usage = None
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[self._serialize_message(msg) for msg in messages],
            **self._request_kwargs(kwargs),
        )
        self._capture_usage(response)
        return response.choices[0].message.content or ""

    def stream(self, messages: Iterable[ChatMessage], **kwargs: object):
        self.last_usage = None
        stream = self.client.chat.completions.create(
            model=self.model,
            messages=[self._serialize_message(msg) for msg in messages],
            stream=True,
            # Ask for the terminal usage-bearing chunk (its choices list is
            # empty). Endpoints that ignore the option just never send it.
            stream_options={"include_usage": True},
            **self._request_kwargs(kwargs),
        )
        for event in stream:
            self._capture_usage(event)
            choices = getattr(event, "choices", None)
            if not choices:
                continue  # the final usage chunk carries no delta
            delta = choices[0].delta.content
            if delta:
                yield delta

    def supports_stream(self) -> bool:
        return True

    def supports_vision(self) -> bool:
        return True
