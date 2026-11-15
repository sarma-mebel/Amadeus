"""Tests for FallbackLLMProvider rate-limit failover."""

import pytest

from amadeus.providers.llm.base import ChatMessage, LLMProvider, DummyProvider
from amadeus.providers.llm.fallback import FallbackLLMProvider, _is_rate_limit


class ResourceExhausted(Exception):
    """Mimics google.api_core.exceptions.ResourceExhausted by class name
    (the failover detects Gemini rate limits by this exact class name)."""


class _Rate429(Exception):
    status_code = 429


class StubProvider(LLMProvider):
    def __init__(self, name, reply=None, raise_exc=None, stream_chunks=None):
        self.name = name
        self.reply = reply
        self.raise_exc = raise_exc
        self.stream_chunks = stream_chunks or []
        self.calls = 0

    def generate(self, messages, **kwargs):
        self.calls += 1
        if self.raise_exc:
            raise self.raise_exc
        return self.reply

    def stream(self, messages, **kwargs):
        self.calls += 1
        if self.raise_exc:
            raise self.raise_exc
        yield from self.stream_chunks

    def supports_stream(self):
        return True


MSGS = [ChatMessage(role="user", content="hi")]


def test_rate_limit_detection():
    assert _is_rate_limit(ResourceExhausted())
    assert _is_rate_limit(_Rate429())
    assert not _is_rate_limit(ValueError("bad request"))


def test_primary_success_no_fallback():
    primary = StubProvider("gemini", reply="from primary")
    fallback = StubProvider("openrouter", reply="from fallback")
    fp = FallbackLLMProvider(primary, fallback)
    assert fp.generate(MSGS) == "from primary"
    assert fallback.calls == 0


def test_rate_limit_triggers_fallback():
    primary = StubProvider("gemini", raise_exc=_Rate429())
    fallback = StubProvider("openrouter", reply="rescued")
    fp = FallbackLLMProvider(primary, fallback)
    assert fp.generate(MSGS) == "rescued"
    assert fallback.calls == 1


def test_non_rate_limit_reraises():
    primary = StubProvider("gemini", raise_exc=ValueError("bad request"))
    fallback = StubProvider("openrouter", reply="should not be used")
    fp = FallbackLLMProvider(primary, fallback)
    with pytest.raises(ValueError):
        fp.generate(MSGS)
    assert fallback.calls == 0


def test_stream_failover_before_first_chunk():
    primary = StubProvider("gemini", raise_exc=_Rate429())
    fallback = StubProvider("openrouter", stream_chunks=["a", "b", "c"])
    fp = FallbackLLMProvider(primary, fallback)
    assert "".join(fp.stream(MSGS)) == "abc"


def test_stream_primary_success():
    primary = StubProvider("gemini", stream_chunks=["x", "y"])
    fallback = StubProvider("openrouter", stream_chunks=["nope"])
    fp = FallbackLLMProvider(primary, fallback)
    assert "".join(fp.stream(MSGS)) == "xy"
    assert fallback.calls == 0


def test_name_is_composite():
    fp = FallbackLLMProvider(StubProvider("gemini"), StubProvider("openrouter"))
    assert fp.name == "gemini+openrouter"


def test_load_llm_wrapping_conditions():
    from amadeus.config import Config
    from amadeus.provider_loader import load_llm

    # Flag off -> bare primary (dummy here, since no key)
    off = Config.from_dict({"providers": {"llm": "gemini"}})
    assert not isinstance(load_llm(off), FallbackLLMProvider)

    # Flag on but dummy primary (no gemini key) -> not wrapped
    dummy_on = Config.from_dict(
        {
            "providers": {"llm": "gemini"},
            "llm_fallback_enabled": True,
            "openrouter_api_key": "sk-or-x",
        }
    )
    assert isinstance(load_llm(dummy_on), DummyProvider)


def test_every_base_capability_probe_is_forwarded():
    """Every supports_* on LLMProvider must be overridden by the fallback
    wrapper. Omitting one silently inherits the base class's conservative
    default — supports_vision was missed and hard-disabled every vision
    feature (screen peek, read-along, watch-together, image turns) whenever
    the session LLM was wrapped."""

    probes = [name for name in dir(LLMProvider) if name.startswith("supports_")]
    assert probes  # the contract below is vacuous if the base drops them
    for name in probes:
        assert getattr(FallbackLLMProvider, name) is not getattr(
            LLMProvider, name
        ), f"FallbackLLMProvider must forward {name} to the primary provider"


def test_supports_vision_follows_primary():
    class VisionStub(StubProvider):
        def supports_vision(self) -> bool:
            return True

    assert FallbackLLMProvider(VisionStub("p"), StubProvider("f")).supports_vision()
    assert not FallbackLLMProvider(
        StubProvider("p"), StubProvider("f")
    ).supports_vision()
