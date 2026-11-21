"""OpenAIProvider token-usage capture (no live API, no openai SDK needed).

Built via ``__new__`` with a fake client (same pattern as the Gemini provider
tests) so nothing here imports or calls the real openai package.
"""

from __future__ import annotations

import types

from amadeus.providers.llm.base import ChatMessage
from amadeus.providers.llm.openai_provider import OpenAIProvider


class _FakeCompletions:
    def __init__(self, response=None, events=None):
        self._response = response
        self._events = events or []
        self.kwargs: list[dict] = []

    def create(self, **kwargs):
        self.kwargs.append(kwargs)
        if kwargs.get("stream"):
            return iter(self._events)
        return self._response


def _provider(response=None, events=None):
    provider = OpenAIProvider.__new__(OpenAIProvider)
    completions = _FakeCompletions(response=response, events=events)
    provider.client = types.SimpleNamespace(
        chat=types.SimpleNamespace(completions=completions)
    )
    provider.model = "gpt-test"
    return provider, completions


def _usage(prompt=10, completion=3):
    return types.SimpleNamespace(prompt_tokens=prompt, completion_tokens=completion)


def _response(text="hello", usage=None):
    return types.SimpleNamespace(
        choices=[types.SimpleNamespace(message=types.SimpleNamespace(content=text))],
        usage=usage,
    )


def _delta_event(text):
    return types.SimpleNamespace(
        choices=[types.SimpleNamespace(delta=types.SimpleNamespace(content=text))],
        usage=None,
    )


def _msgs(text="hi"):
    return [ChatMessage(role="user", content=text)]


def test_generate_populates_last_usage():
    provider, _ = _provider(response=_response("hello", usage=_usage(10, 3)))
    assert provider.generate(_msgs()) == "hello"
    assert provider.last_usage == {"prompt": 10, "completion": 3}


def test_generate_without_usage_resets_last_usage():
    provider, _ = _provider(response=_response("hello", usage=None))
    provider.last_usage = {"prompt": 1, "completion": 1}  # stale from a prior call
    provider.generate(_msgs())
    assert provider.last_usage is None


def test_stream_requests_usage_and_reads_final_chunk():
    # The terminal usage-bearing chunk has an EMPTY choices list — the stream
    # must skip it as a delta but still capture its usage.
    final = types.SimpleNamespace(choices=[], usage=_usage(42, 17))
    provider, completions = _provider(
        events=[_delta_event("Hel"), _delta_event("lo"), final]
    )
    assert "".join(provider.stream(_msgs())) == "Hello"
    assert provider.last_usage == {"prompt": 42, "completion": 17}
    assert completions.kwargs[0]["stream_options"] == {"include_usage": True}


def test_stream_without_usage_leaves_none():
    provider, _ = _provider(events=[_delta_event("Hi")])
    provider.last_usage = {"prompt": 1, "completion": 1}  # stale from a prior call
    "".join(provider.stream(_msgs()))
    assert provider.last_usage is None


def test_new_gpt_models_use_supported_sampling_and_output_limit():
    provider, calls = _provider(response=_response("да"))
    for model in ("gpt-5.4-mini", "gpt-6.2-sol", "o3"):
        provider.model = model
        provider.generate(_msgs(), temperature=0.8, max_tokens=240)
        assert calls.kwargs[-1]["max_completion_tokens"] == 240
        assert "temperature" not in calls.kwargs[-1]
        assert "max_tokens" not in calls.kwargs[-1]
    provider.model = "gpt-4o-mini"
    provider.generate(_msgs(), temperature=0.7, max_tokens=240)
    assert calls.kwargs[-1]["max_tokens"] == 240
    assert calls.kwargs[-1]["temperature"] == 0.7


def test_mobile_astra_uses_medium_reasoning_without_sampling_parameters():
    provider, completions = _provider(response=_response("Привет"))
    provider.model = OpenAIProvider.ASTRA_MODEL
    assert provider.generate(_msgs(), temperature=0.4, max_tokens=32) == "Привет"
    sent = completions.kwargs[0]
    assert sent["model"] == "gpt-6.2-astra"
    assert sent["reasoning_effort"] == "medium"
    assert sent["max_completion_tokens"] >= 32
    assert "temperature" not in sent and "max_tokens" not in sent
