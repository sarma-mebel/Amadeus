"""load_llm wiring for OpenAI-compatible endpoints (no openai SDK required).

The openai package is optional and absent from the test env, so a stub module
stands in for it — these tests pin the loader's construction kwargs, not the
SDK behaviour.
"""

import sys
import types

from amadeus.config import Config
from amadeus.provider_loader import load_llm
from amadeus.providers.llm.base import DummyProvider


class _RecordingOpenAI:
    last_kwargs: dict | None = None

    def __init__(self, **kwargs):
        _RecordingOpenAI.last_kwargs = kwargs


def _stub_openai_sdk(monkeypatch) -> None:
    module = types.ModuleType("openai")
    module.OpenAI = _RecordingOpenAI
    monkeypatch.setitem(sys.modules, "openai", module)
    _RecordingOpenAI.last_kwargs = None


def test_openai_base_url_parses_from_config() -> None:
    config = Config.from_dict(
        {"providers": {"llm": "openai", "openai_base_url": "http://localhost:11434/v1"}}
    )
    assert config.providers.openai_base_url == "http://localhost:11434/v1"
    assert Config.from_dict({}).providers.openai_base_url is None


def test_keyless_base_url_loads_openai_provider(monkeypatch) -> None:
    """A local OpenAI-compatible server needs no API key — placeholder is sent."""
    _stub_openai_sdk(monkeypatch)
    config = Config.from_dict(
        {
            "llm_model": "llama3",
            "providers": {
                "llm": "openai",
                "openai_base_url": "http://localhost:11434/v1",
            },
        }
    )
    config.openai_api_key = None
    provider = load_llm(config)
    assert not isinstance(provider, DummyProvider)
    assert provider.model == "llama3"
    assert _RecordingOpenAI.last_kwargs == {
        "api_key": "not-needed",
        "base_url": "http://localhost:11434/v1",
    }


def test_real_key_rides_along_with_base_url(monkeypatch) -> None:
    """Hosted OpenAI-compatible services (OpenRouter, Groq) keep the real key."""
    _stub_openai_sdk(monkeypatch)
    config = Config.from_dict(
        {
            "providers": {
                "llm": "openai",
                "openai_base_url": "https://openrouter.ai/api/v1",
            }
        }
    )
    config.openai_api_key = "sk-or-abc"
    load_llm(config)
    assert _RecordingOpenAI.last_kwargs == {
        "api_key": "sk-or-abc",
        "base_url": "https://openrouter.ai/api/v1",
    }


def test_no_key_no_base_url_still_falls_back_to_dummy(monkeypatch) -> None:
    """Pre-existing fallback behaviour is unchanged when nothing is configured."""
    _stub_openai_sdk(monkeypatch)
    config = Config.from_dict({"providers": {"llm": "openai"}})
    config.openai_api_key = None
    assert isinstance(load_llm(config), DummyProvider)
