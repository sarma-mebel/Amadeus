"""Lightweight web search client for the tool loop.

Providers: Tavily (hosted, needs an API key) and DuckDuckGo (keyless, via the
``ddgs`` package). ``provider: auto`` (the default) picks Tavily when an
api_key is configured and DuckDuckGo otherwise, so web search works with zero
API keys as long as ``ddgs`` is installed.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import httpx

from ..config import WebSearchSettings

LOGGER = logging.getLogger(__name__)

_DDGS_INSTALL_HINT = (
    "DuckDuckGo web search requires the 'ddgs' package "
    '(pip install "amadeus-kurisu[search]" or pip install ddgs)'
)


def _resolve_provider(settings: WebSearchSettings) -> str:
    """Map the configured provider string to a concrete backend name."""
    raw = (settings.provider or "auto").strip().lower()
    if raw in {"", "auto"}:
        return "tavily" if settings.api_key else "duckduckgo"
    if raw in {"tavily", "duckduckgo"}:
        return raw
    raise ValueError(f"Unsupported web search provider: {settings.provider}")


class WebSearchClient:
    """Wrapper around web search backends (Tavily or DuckDuckGo)."""

    def __init__(self, settings: WebSearchSettings) -> None:
        self.settings = settings
        if not settings.enabled:
            raise RuntimeError("Web search is disabled in configuration")
        self.provider = _resolve_provider(settings)
        self._client: Optional[httpx.Client] = None
        self._ddgs_cls = None
        if self.provider == "tavily":
            if not settings.api_key:
                raise RuntimeError(
                    "Tavily web search requires an api_key "
                    "(set web_search.api_key, or use provider: duckduckgo/auto)"
                )
            self._client = httpx.Client(timeout=settings.timeout)
        else:  # duckduckgo — lazy import so the extra stays optional
            try:
                from ddgs import DDGS  # type: ignore
            except ImportError as exc:
                raise RuntimeError(_DDGS_INSTALL_HINT) from exc
            self._ddgs_cls = DDGS

    def search(
        self, query: str, *, max_results: Optional[int] = None
    ) -> List[Dict[str, str]]:
        limit = max_results or self.settings.max_results
        if self.provider == "tavily":
            return self._search_tavily(query, limit)
        if self.provider == "duckduckgo":
            return self._search_duckduckgo(query, limit)
        return []

    def _search_tavily(self, query: str, limit: int) -> List[Dict[str, str]]:
        url = "https://api.tavily.com/search"
        payload = {
            "api_key": self.settings.api_key,
            "query": query,
            "max_results": limit,
            "include_images": False,
            "search_depth": "basic",
        }
        try:
            response = self._client.post(url, json=payload)
            response.raise_for_status()
            data = response.json() or {}
            results = data.get("results", [])
            parsed: List[Dict[str, str]] = []
            for item in results:
                parsed.append(
                    {
                        "title": str(item.get("title", "")),
                        "snippet": str(item.get("content", "")),
                        "url": str(item.get("url", "")),
                    }
                )
            return parsed
        except Exception as exc:  # pragma: no cover - network dependent
            LOGGER.warning("Web search failed: %s", exc)
            return []

    def _search_duckduckgo(self, query: str, limit: int) -> List[Dict[str, str]]:
        try:
            with self._ddgs_cls(timeout=self.settings.timeout) as ddgs:
                results = ddgs.text(query, max_results=limit) or []
            parsed: List[Dict[str, str]] = []
            for item in results:
                parsed.append(
                    {
                        # ddgs returns title/href/body; tolerate url/snippet
                        # variants across package versions.
                        "title": str(item.get("title", "")),
                        "snippet": str(item.get("body") or item.get("snippet") or ""),
                        "url": str(item.get("href") or item.get("url") or ""),
                    }
                )
            return parsed[:limit]
        except Exception as exc:  # pragma: no cover - network dependent
            LOGGER.warning("Web search failed: %s", exc)
            return []

    def close(self) -> None:
        try:
            if self._client is not None:
                self._client.close()
        except Exception:  # pragma: no cover - best effort
            pass


def build_web_search_client(
    settings: Optional[WebSearchSettings],
) -> Optional[WebSearchClient]:
    if not settings or not settings.enabled:
        return None
    try:
        return WebSearchClient(settings)
    except Exception as exc:
        LOGGER.warning("Web search unavailable: %s", exc)
        return None
