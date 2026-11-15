"""
Factory for creating and initializing agent components.
"""

import logging
from typing import Any, Optional

from amadeus.config import Config
from amadeus.agents.tools import (
    ToolRegistry,
    AnimationTool,
    CalculatorTool,
    CodeInterpreterTool,
    WebSearchTool,
    DesktopFilesTool,
    PubMedSearchTool,
    ArxivSearchTool,
    ReminderTool,
)
from amadeus.agents.memory_tool import MemoryTool

LOGGER = logging.getLogger(__name__)


def _record_unavailable(registry: ToolRegistry, name: str, reason: str) -> None:
    """Log a requested-but-unregisterable tool and remember why for /api/status."""
    LOGGER.warning("Tool %s requested but unavailable: %s", name, reason)
    registry.unavailable.append({"name": name, "reason": reason})


def create_tool_registry(config: Config, memory: Optional[Any] = None) -> ToolRegistry:
    """
    Create and initialize the ToolRegistry with enabled tools.

    Tools that are enabled in config but cannot be constructed (missing
    package, missing API key, …) are recorded on ``registry.unavailable``
    with a human-readable reason instead of being silently skipped.

    Args:
        config: Application configuration
        memory: Optional MemoryManager instance (or duck-typed equivalent)
            backing the model-facing "memory" tool. Callers that don't have
            one yet (e.g. a status-only registry build) can omit it — the
            tool is simply left unregistered, same as any other disabled
            tool, never an error.

    Returns:
        Initialized ToolRegistry
    """
    registry = ToolRegistry()

    # Always register basic tools
    registry.register(CalculatorTool())
    code_timeout = float(
        getattr(getattr(config, "tools", None), "code_timeout_seconds", 10.0) or 10.0
    )
    registry.register(CodeInterpreterTool(timeout_seconds=code_timeout))

    # Avatar animation flourishes ("dance for me") — cosmetic, so always on
    # when any clips are allowlisted; an empty list disables it.
    animation_clips = list(
        getattr(getattr(config, "tools", None), "animation_clips", []) or []
    )
    if animation_clips:
        registry.register(AnimationTool(animation_clips))

    # Register web search if enabled
    if config.web_search.enabled:
        try:
            from amadeus.tools.web_search import WebSearchClient

            registry.register(WebSearchTool(WebSearchClient(config.web_search)))
        except ImportError as e:
            _record_unavailable(
                registry, "web_search", f"could not import web search client: {e}"
            )
        except Exception as e:
            _record_unavailable(registry, "web_search", str(e))

    # Register read-only desktop file access if enabled
    if getattr(config, "desktop_files", None) and config.desktop_files.enabled:
        try:
            from amadeus.tools.desktop_files import build_desktop_files_client

            files_client = build_desktop_files_client(config.desktop_files)
            registry.register(DesktopFilesTool(files_client))
        except ImportError as e:
            _record_unavailable(
                registry, "desktop_files", f"could not import desktop files client: {e}"
            )
        except Exception as e:
            _record_unavailable(registry, "desktop_files", str(e))

    # Register scientific-literature search (PubMed + arXiv) if enabled
    if getattr(config, "literature", None) and config.literature.enabled:
        try:
            from amadeus.tools.literature import PubMedClient, ArxivClient

            registry.register(PubMedSearchTool(PubMedClient()))
            registry.register(ArxivSearchTool(ArxivClient()))
        except ImportError as e:
            _record_unavailable(
                registry,
                "pubmed_search/arxiv_search",
                f"could not import literature clients: {e}",
            )
        except Exception as e:
            _record_unavailable(registry, "pubmed_search/arxiv_search", str(e))

    # Register local reminders/timers if enabled
    if getattr(config, "reminders", None) and config.reminders.enabled:
        try:
            from amadeus.reminders import ReminderStore

            registry.register(ReminderTool(ReminderStore(config.reminders.store_path)))
        except Exception as e:
            _record_unavailable(registry, "reminders", str(e))

    # Register the model-facing memory tool (explicit remember/correct/forget)
    # only when long-term memory + fact extraction are both on — without
    # fact extraction there's no durable-fact store for it to act on, and
    # without long-term memory find_facts/archive_fact have nothing to query.
    mem_cfg = getattr(config, "memory", None)
    wants_memory_tool = bool(
        mem_cfg
        and getattr(mem_cfg, "enable_long_term", False)
        and getattr(mem_cfg, "enable_fact_extraction", False)
    )
    if wants_memory_tool:
        if memory is not None:
            registry.register(MemoryTool(memory))
        else:
            _record_unavailable(
                registry, "memory", "no MemoryManager instance was provided"
            )

    # Bridge tools from user-configured MCP servers (config section `mcp`,
    # default off). load_mcp_tools never raises — a missing SDK or a dead
    # server comes back as unavailable entries — and caches connections
    # process-wide, so the /api/status probe registry (rebuilt per poll)
    # never respawns stdio server processes.
    if getattr(config, "mcp", None) and config.mcp.enabled:
        try:
            from amadeus.tools.mcp_bridge import load_mcp_tools

            mcp_tools, mcp_unavailable = load_mcp_tools(config)
            for tool in mcp_tools:
                registry.register(tool)
            for entry in mcp_unavailable:
                _record_unavailable(registry, entry["name"], entry["reason"])
        except Exception as e:
            _record_unavailable(registry, "mcp", str(e))

    return registry
