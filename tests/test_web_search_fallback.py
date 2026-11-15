"""Web search provider auto-selection and the keyless DuckDuckGo fallback."""

import sys
import types

import pytest

from amadeus.config import Config, WebSearchSettings
from amadeus.tools.web_search import (
    WebSearchClient,
    _resolve_provider,
    build_web_search_client,
)


def _settings(**overrides) -> WebSearchSettings:
    values = {"enabled": True}
    values.update(overrides)
    return WebSearchSettings(**values)


class _FakeDDGS:
    """Stands in for ddgs.DDGS — records queries, returns canned results."""

    last_query = None

    def __init__(self, timeout=None):
        self.timeout = timeout

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def text(self, query, max_results=None):
        _FakeDDGS.last_query = (query, max_results)
        return [
            {"title": "Result A", "href": "https://a.example", "body": "Snippet A"},
            {"title": "Result B", "href": "https://b.example", "body": "Snippet B"},
        ]


@pytest.fixture
def fake_ddgs(monkeypatch):
    module = types.ModuleType("ddgs")
    module.DDGS = _FakeDDGS
    monkeypatch.setitem(sys.modules, "ddgs", module)
    return module


# --- provider resolution -------------------------------------------------


def test_auto_prefers_tavily_when_key_present():
    assert _resolve_provider(_settings(provider="auto", api_key="tv-key")) == "tavily"


def test_auto_falls_back_to_duckduckgo_without_key():
    assert _resolve_provider(_settings(provider="auto", api_key=None)) == "duckduckgo"
    assert _resolve_provider(_settings(provider="", api_key=None)) == "duckduckgo"


def test_default_provider_is_auto():
    assert Config().web_search.provider == "auto"


def test_unknown_provider_rejected():
    with pytest.raises(ValueError):
        _resolve_provider(_settings(provider="bing"))


def test_explicit_tavily_without_key_raises():
    with pytest.raises(RuntimeError, match="api_key"):
        WebSearchClient(_settings(provider="tavily", api_key=None))


def test_duckduckgo_without_ddgs_package_raises_clear_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "ddgs", None)  # force ImportError
    with pytest.raises(RuntimeError, match="ddgs"):
        WebSearchClient(_settings(provider="duckduckgo"))


# --- DuckDuckGo search (ddgs mocked) --------------------------------------


def test_duckduckgo_search_result_shape(fake_ddgs):
    client = WebSearchClient(_settings(provider="auto", api_key=None, max_results=2))
    assert client.provider == "duckduckgo"
    results = client.search("time machine")
    assert _FakeDDGS.last_query == ("time machine", 2)
    assert results == [
        {"title": "Result A", "snippet": "Snippet A", "url": "https://a.example"},
        {"title": "Result B", "snippet": "Snippet B", "url": "https://b.example"},
    ]


def test_duckduckgo_search_errors_return_empty(fake_ddgs, monkeypatch):
    client = WebSearchClient(_settings(provider="duckduckgo"))

    def _boom(self, query, max_results=None):
        raise RuntimeError("network down")

    monkeypatch.setattr(_FakeDDGS, "text", _boom)
    assert client.search("anything") == []


def test_build_web_search_client_swallow_behavior_for_council():
    # The council path still expects None (not an exception) on failure.
    assert build_web_search_client(_settings(provider="tavily", api_key=None)) is None
    assert build_web_search_client(WebSearchSettings(enabled=False)) is None


# --- factory: registration + unavailable reasons --------------------------


def _config(tmp_path) -> Config:
    config = Config()
    config.memory.persist_path = str(tmp_path / "memory.json")
    config.memory.embeddings_path = str(tmp_path / "embeddings.json")
    config.memory.profile_path = str(tmp_path / "profile.json")
    # These tests only care about web_search's own unavailable-tracking;
    # create_tool_registry() here is never given a MemoryManager, and since
    # 2026-07-31 that alone makes the memory tool land in `unavailable` too
    # (memory.enable_long_term/enable_fact_extraction default on). Turn the
    # memory tool's want off so `unavailable` stays scoped to web_search.
    config.memory.enable_fact_extraction = False
    return config


def test_factory_registers_duckduckgo_web_search(tmp_path, fake_ddgs):
    from amadeus.agents.factory import create_tool_registry

    config = _config(tmp_path)
    config.web_search.enabled = True
    registry = create_tool_registry(config)
    assert "web_search" in registry.tools
    assert registry.unavailable == []


def test_factory_records_unavailable_reason(tmp_path, monkeypatch):
    from amadeus.agents.factory import create_tool_registry

    monkeypatch.setitem(sys.modules, "ddgs", None)  # no ddgs, no key
    config = _config(tmp_path)
    config.web_search.enabled = True
    registry = create_tool_registry(config)
    assert "web_search" not in registry.tools
    assert len(registry.unavailable) == 1
    entry = registry.unavailable[0]
    assert entry["name"] == "web_search"
    assert "ddgs" in entry["reason"]
