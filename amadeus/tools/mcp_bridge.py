"""MCP (Model Context Protocol) client bridge.

Surfaces tools from user-configured MCP servers (config section ``mcp``) as
first-class registry tools named ``mcp_<server>_<tool>``, so the existing
tool loop — allowlist firewall, central parameter validation, exfiltration
guard, observation sanitizer — applies to them unchanged. Two hard
constraints shape everything here:

1. **Sync world.** ``BaseTool.execute`` is fully synchronous and runs on a
   plain worker thread under FastAPI (web/blocking.py) and on the bare main
   thread in the CLI — there is never a running event loop to await on. The
   mcp SDK is asyncio-native, so the bridge owns ONE dedicated daemon thread
   running its own loop; every connect/list/call is submitted with
   ``run_coroutine_threadsafe`` and awaited synchronously under a timeout.
   Each server's contexts (stdio pipes / HTTP session) are opened and closed
   inside a single long-lived task on that loop — anyio cancel scopes break
   if entered and exited in different tasks.

2. **Probe-safe construction.** ``GET /api/status`` builds a throwaway probe
   registry on every poll (web/routes_meta.py::_tools_status). Stdio MCP
   servers are real child processes, so the connected bridge (or the recorded
   failure) is cached process-wide keyed on the mcp config fingerprint and
   reused — a status poll must never respawn servers.

Security posture: every bridged tool is marked ``untrusted_source`` (its
output is retrieved third-party content), and ``network_capable`` unless the
server's config says ``network: false`` — which makes the tool loop refuse
it after any desktop_files call the same turn, exactly like web_search
(agents/tool_loop.py::_is_network_tool).

Degrade loudly: connect/list/call failures are recorded on
``registry.unavailable`` (surfaced in /api/status) and counted
(``tools.mcp_connect_failed`` / ``tools.mcp_call_failures``) — never
silently dropped.
"""

from __future__ import annotations

import asyncio
import atexit
import json
import re
import threading
from typing import Any, Dict, List, Optional, Tuple

from .. import telemetry
from ..agents.tools import (
    BaseTool,
    ToolCategory,
    ToolDefinition,
    ToolParameter,
    ToolResult,
)
from ..logging_utils import get_logger

LOGGER = get_logger(__name__)

# ToolParameter.type values understood by BaseTool.validate_parameters.
# JSON-Schema "integer" collapses to "number"; anything exotic ("null",
# unions, $ref) falls back to "string" — the remote server re-validates.
_SCHEMA_TYPES = {
    "string": "string",
    "number": "number",
    "integer": "number",
    "boolean": "boolean",
    "array": "array",
    "object": "object",
}

_NAME_SANITIZE_RE = re.compile(r"[^a-z0-9_]+")


def _sanitize(part: str) -> str:
    return _NAME_SANITIZE_RE.sub("_", str(part).strip().lower()).strip("_")


def bridged_tool_name(server: str, remote_tool: str) -> str:
    """Registry name for a bridged tool: mcp_<server>_<tool>, lowercased and
    sanitized (the "mcp_" prefix is what the allowlist meta-entry and the
    guard attributes key off — see agents/tools.py::tool_allowed)."""
    return f"mcp_{_sanitize(server)}_{_sanitize(remote_tool)}"


def parameters_from_schema(schema: Optional[Dict[str, Any]]) -> List[ToolParameter]:
    """Flatten an MCP tool's JSON-Schema input into the registry's flat
    ToolParameter list. Only top-level properties survive (nested objects
    become bare ``object`` params) — validate_parameters rejects unknown
    top-level keys, and the server re-validates the full schema on its side.
    """
    if not isinstance(schema, dict):
        return []
    props = schema.get("properties")
    if not isinstance(props, dict):
        return []
    required = set(schema.get("required") or [])
    parameters: List[ToolParameter] = []
    for pname, pspec in props.items():
        if not isinstance(pspec, dict):
            pspec = {}
        ptype = pspec.get("type")
        if isinstance(ptype, list):
            # Union type: pick the first mappable member ("['string','null']"
            # is the common optional-param shape).
            ptype = next((t for t in ptype if t in _SCHEMA_TYPES), "string")
        enum = pspec.get("enum")
        if not (isinstance(enum, list) and all(isinstance(v, str) for v in enum)):
            enum = None
        parameters.append(
            ToolParameter(
                name=str(pname),
                type=_SCHEMA_TYPES.get(ptype, "string"),
                description=str(pspec.get("description") or ""),
                required=pname in required,
                enum=list(enum) if enum else None,
            )
        )
    return parameters


def format_call_result(result: Any) -> str:
    """Render an mcp CallToolResult into observation text: prefer
    structuredContent (JSON), else join text parts, noting non-text content
    instead of dropping it silently. Duck-typed so tests can use plain
    stand-ins."""
    # SDK 1.x uses camelCase, 2.0 snake_case.
    structured = getattr(result, "structuredContent", None) or getattr(
        result, "structured_content", None
    )
    if structured:
        try:
            return json.dumps(structured, ensure_ascii=False)
        except (TypeError, ValueError):
            pass
    parts: List[str] = []
    for item in getattr(result, "content", None) or []:
        text = getattr(item, "text", None)
        if isinstance(text, str):
            parts.append(text)
        else:
            kind = getattr(item, "type", None) or type(item).__name__
            parts.append(f"[non-text content: {kind}]")
    return "\n".join(parts) if parts else "[empty result]"


class McpBridge:
    """Owns the loop thread and the per-server sessions. One instance per
    distinct mcp config (see the process-wide cache below)."""

    def __init__(self, settings: Any) -> None:
        self.settings = settings
        self.failures: List[Dict[str, str]] = []
        # server name -> list of (remote_tool_name, description, input_schema)
        self.tools_by_server: Dict[str, List[Tuple[str, str, Dict[str, Any]]]] = {}
        self._sessions: Dict[str, Any] = {}
        self._stop_events: Dict[str, asyncio.Event] = {}
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None

    # -- loop plumbing ------------------------------------------------------

    def _ensure_loop(self) -> asyncio.AbstractEventLoop:
        if self._loop is None:
            loop = asyncio.new_event_loop()
            thread = threading.Thread(
                target=loop.run_forever, name="amadeus-mcp-bridge", daemon=True
            )
            thread.start()
            self._loop, self._thread = loop, thread
        return self._loop

    def _run(self, coro, timeout: float):
        future = asyncio.run_coroutine_threadsafe(coro, self._ensure_loop())
        try:
            return future.result(timeout)
        except Exception:
            future.cancel()
            raise

    # -- lifecycle ----------------------------------------------------------

    def connect(self) -> None:
        """Connect every configured server; failures are recorded per server
        on ``self.failures`` (and counted), never raised — one bad server
        must not take down the rest of the registry build."""
        connect_timeout = float(
            getattr(self.settings, "connect_timeout_seconds", 15.0) or 15.0
        )
        for spec in getattr(self.settings, "servers", None) or []:
            if not isinstance(spec, dict):
                self._record_failure("mcp", f"malformed server entry: {spec!r}")
                continue
            name = _sanitize(spec.get("name") or "")
            if not name:
                self._record_failure("mcp", f"server entry missing name: {spec!r}")
                continue
            if name in self._sessions:
                self._record_failure(f"mcp:{name}", "duplicate server name")
                continue
            if not (spec.get("url") or spec.get("command")):
                self._record_failure(
                    f"mcp:{name}", "server entry needs either 'command' or 'url'"
                )
                continue
            try:
                self._run(
                    self._start_server(name, spec, connect_timeout),
                    # The outer wait needs headroom over the inner one so the
                    # inner timeout (which produces the clearer message) wins.
                    connect_timeout + 5.0,
                )
            except Exception as exc:
                self._record_failure(f"mcp:{name}", str(exc) or type(exc).__name__)

    async def _start_server(self, name: str, spec: Dict[str, Any], timeout: float):
        ready: asyncio.Future = asyncio.get_running_loop().create_future()
        stop = asyncio.Event()
        self._stop_events[name] = stop
        # The whole server lifetime lives in ONE task: anyio-based transports
        # (stdio_client) require their context to enter and exit in the same
        # task, so the task opens everything, parks on the stop event, and
        # unwinds in-task on shutdown.
        asyncio.get_running_loop().create_task(
            self._server_lifetime(name, spec, ready, stop)
        )
        await asyncio.wait_for(ready, timeout)

    async def _server_lifetime(
        self,
        name: str,
        spec: Dict[str, Any],
        ready: asyncio.Future,
        stop: asyncio.Event,
    ) -> None:
        try:
            from mcp import ClientSession

            if spec.get("url"):
                try:  # SDK 1.x name
                    from mcp.client.streamable_http import streamablehttp_client
                except ImportError:  # renamed in SDK 2.0
                    from mcp.client.streamable_http import (
                        streamable_http_client as streamablehttp_client,
                    )

                transport_cm = streamablehttp_client(str(spec["url"]))
            else:
                from mcp import StdioServerParameters
                from mcp.client.stdio import stdio_client

                transport_cm = stdio_client(
                    StdioServerParameters(
                        command=str(spec["command"]),
                        args=[str(a) for a in spec.get("args") or []],
                        env={str(k): str(v) for k, v in (spec.get("env") or {}).items()}
                        or None,
                    )
                )
            async with transport_cm as streams:
                read, write = streams[0], streams[1]
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    listing = await session.list_tools()
                    self.tools_by_server[name] = [
                        (
                            tool.name,
                            tool.description or "",
                            # SDK 1.x uses camelCase, 2.0 snake_case.
                            getattr(tool, "inputSchema", None)
                            or getattr(tool, "input_schema", None)
                            or {},
                        )
                        for tool in listing.tools
                    ]
                    self._sessions[name] = session
                    ready.set_result(True)
                    await stop.wait()
        except Exception as exc:
            if not ready.done():
                ready.set_exception(exc)
            else:
                LOGGER.warning("MCP server %s died: %s", name, exc)
                telemetry.incr("tools.mcp_server_died")
        finally:
            self._sessions.pop(name, None)

    def close(self) -> None:
        """Best-effort shutdown (atexit): release the stop events so each
        server task unwinds its contexts in-task, then stop the loop."""
        loop = self._loop
        if loop is None or not loop.is_running():
            return
        for stop in self._stop_events.values():
            loop.call_soon_threadsafe(stop.set)
        # Give the lifetime tasks a beat to unwind before stopping the loop;
        # daemon thread + OS teardown covers anything slower.
        loop.call_soon_threadsafe(loop.call_later, 1.0, loop.stop)

    # -- calls --------------------------------------------------------------

    def call(self, server: str, remote_tool: str, arguments: Dict[str, Any]) -> str:
        session = self._sessions.get(server)
        if session is None:
            raise RuntimeError(f"MCP server '{server}' is not connected")
        timeout = float(getattr(self.settings, "call_timeout_seconds", 30.0) or 30.0)
        result = self._run(session.call_tool(remote_tool, arguments or {}), timeout)
        text = format_call_result(result)
        if getattr(result, "isError", False) or getattr(result, "is_error", False):
            raise RuntimeError(text or "MCP tool reported an error")
        return text

    def _record_failure(self, name: str, reason: str) -> None:
        LOGGER.warning("MCP bridge: %s unavailable: %s", name, reason)
        telemetry.incr("tools.mcp_connect_failed")
        self.failures.append({"name": name, "reason": reason})


class McpProxyTool(BaseTool):
    """Registry adapter for one remote MCP tool. Attribute contract read by
    the tool loop's guards: ``untrusted_source`` always (external output is
    retrieved content), ``network_capable`` per server config (default True —
    refused after a desktop_files call the same turn, like web_search)."""

    untrusted_source = True

    def __init__(
        self,
        bridge: McpBridge,
        server: str,
        remote_name: str,
        description: str,
        schema: Optional[Dict[str, Any]],
        network: bool = True,
    ) -> None:
        self._bridge = bridge
        self._server = server
        self._remote_name = remote_name
        self._description = description
        self._schema = schema
        self.network_capable = bool(network)
        super().__init__()

    def get_definition(self) -> ToolDefinition:
        name = bridged_tool_name(self._server, self._remote_name)
        description = (
            f"[MCP tool from server '{self._server}'] "
            f"{self._description or self._remote_name}"
        )
        return ToolDefinition(
            name=name,
            description=description[:1000],
            category=ToolCategory.DATA,
            parameters=parameters_from_schema(self._schema),
            returns="The MCP server's response (treated as retrieved data).",
            cost_tier=2,
        )

    def execute(self, **kwargs) -> ToolResult:
        try:
            text = self._bridge.call(self._server, self._remote_name, kwargs)
            telemetry.incr("tools.mcp_calls")
            return ToolResult(tool_name=self.definition.name, success=True, result=text)
        except Exception as exc:
            telemetry.incr("tools.mcp_call_failures")
            LOGGER.warning(
                "MCP call %s.%s failed: %s", self._server, self._remote_name, exc
            )
            return ToolResult(
                tool_name=self.definition.name,
                success=False,
                result=None,
                error=f"MCP call failed: {exc}",
            )


# Process-wide bridge cache. /api/status probes rebuild the tool registry on
# every poll; stdio servers are child processes and must be spawned once per
# process, not per poll. A config change (new fingerprint) closes the old
# bridge and connects a fresh one. Failures are cached the same way — a
# permanently-broken server logs once per process, not once per status poll.
_CACHE: Dict[str, Any] = {"fingerprint": None, "bridge": None}
_CACHE_LOCK = threading.Lock()


def _fingerprint(settings: Any) -> str:
    return json.dumps(
        {
            "servers": getattr(settings, "servers", None) or [],
            "connect_timeout": getattr(settings, "connect_timeout_seconds", None),
            "call_timeout": getattr(settings, "call_timeout_seconds", None),
        },
        sort_keys=True,
        default=str,
    )


def _get_bridge(settings: Any) -> McpBridge:
    fingerprint = _fingerprint(settings)
    with _CACHE_LOCK:
        if _CACHE["fingerprint"] == fingerprint and _CACHE["bridge"] is not None:
            return _CACHE["bridge"]
        old = _CACHE["bridge"]
        if old is not None:
            try:
                old.close()
            except Exception:  # pragma: no cover - best-effort teardown
                pass
        bridge = McpBridge(settings)
        bridge.connect()
        _CACHE["fingerprint"] = fingerprint
        _CACHE["bridge"] = bridge
        return bridge


@atexit.register
def _close_cached_bridge() -> None:  # pragma: no cover - process teardown
    bridge = _CACHE.get("bridge")
    if bridge is not None:
        try:
            bridge.close()
        except Exception:
            pass


def load_mcp_tools(config: Any) -> Tuple[List[BaseTool], List[Dict[str, str]]]:
    """Factory entry point: (proxy tools to register, unavailable entries).

    Never raises: a missing SDK, a dead server, or a malformed entry becomes
    an unavailable record (surfaced in /api/status) instead. The bridge and
    its child processes/connections are cached process-wide.
    """
    settings = getattr(config, "mcp", None)
    if settings is None or not getattr(settings, "enabled", False):
        return [], []
    if not (getattr(settings, "servers", None) or []):
        return [], [
            {"name": "mcp", "reason": "mcp.enabled is on but no servers configured"}
        ]
    try:
        import mcp  # noqa: F401  (probe only; real imports are per-transport)
    except ImportError as exc:
        telemetry.incr("tools.mcp_connect_failed")
        return [], [
            {
                "name": "mcp",
                "reason": (
                    f"could not import the mcp SDK: {exc}; install with "
                    'pip install "amadeus-kurisu[mcp]" (Python 3.10+)'
                ),
            }
        ]
    try:
        bridge = _get_bridge(settings)
    except Exception as exc:  # pragma: no cover - defensive: cache/connect
        telemetry.incr("tools.mcp_connect_failed")
        return [], [{"name": "mcp", "reason": str(exc)}]

    tools: List[BaseTool] = []
    unavailable = list(bridge.failures)
    for spec in getattr(settings, "servers", None) or []:
        if not isinstance(spec, dict):
            continue
        server = _sanitize(spec.get("name") or "")
        listed = bridge.tools_by_server.get(server)
        if not listed:
            continue
        allowed_tools = spec.get("allowed_tools")
        network = spec.get("network", True)
        for remote_name, description, schema in listed:
            if allowed_tools is not None and remote_name not in allowed_tools:
                continue
            try:
                tools.append(
                    McpProxyTool(
                        bridge, server, remote_name, description, schema, network
                    )
                )
            except Exception as exc:
                unavailable.append(
                    {
                        "name": bridged_tool_name(server, remote_name),
                        "reason": f"could not wrap tool: {exc}",
                    }
                )
    return tools, unavailable
