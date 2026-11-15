"""Local transformers-based provider."""

from __future__ import annotations

from typing import Iterable

from .base import ChatMessage, LLMProvider


class LocalTransformersProvider(LLMProvider):
    """Use a local causal language model via ``transformers``."""

    name = "local_transformers"

    def __init__(
        self,
        model_name: str = "mistralai/Mistral-7B-Instruct-v0.2",
        device: str | None = None,
    ) -> None:
        try:
            from transformers import (  # noqa: F401 - availability check
                AutoModelForCausalLM,  # noqa: F401
                AutoTokenizer,  # noqa: F401
                pipeline,  # noqa: F401
            )
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "transformers is required for the local provider"
            ) from exc

        self.pipeline = pipeline(
            "text-generation",
            model=model_name,
            tokenizer=model_name,
            device=device,
            model_kwargs={"torch_dtype": "auto"},
        )

    def generate(self, messages: Iterable[ChatMessage], **kwargs: object) -> str:
        prompt = "\n".join(f"{msg.role}: {msg.content}" for msg in messages)
        outputs = self.pipeline(
            prompt,
            max_new_tokens=256,
            do_sample=True,
            temperature=kwargs.get("temperature", 0.7),
        )
        return outputs[0]["generated_text"][len(prompt) :].strip()
