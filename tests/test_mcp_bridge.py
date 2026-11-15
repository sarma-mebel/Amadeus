"""MCP client bridge: schema flattening, proxy tools, guard integration,
factory registration, and the process-wide bridge cache.

Pure-offline by design: the real mcp SDK is not a test dependency — the
bridge's pure helpers are tested directly, load_mcp_tools is tested with a
faked-out ``mcp`` module + bridge, and the tool loop's guard behavior is
tested with fake BaseTool stand-ins carrying the attribute contract
(``network_capable`` / ``untrusted_source``) the loop reads.
"""

import sys
import types
from typing import List

import pytest

from amadeus import telemetry
from amadeus.agents.factory import create_tool_registry
from amadeus.agents.tool_loop import (
    _is_network_tool,
    _is_untrusted_source,
    run_tool_loop,
)
from amadeus.agents.tools import (
    BaseTool,
    ToolCategory,
    ToolDefinition,
    ToolParameter,
    ToolRegistry,
    ToolResult,
    tool_allowed,
)
from amadeus.config import Config
from amadeus.providers.llm.base import ChatMessage
from amadeus.tools import mcp_bridge
from amadeus.tools.mcp_bridge import (
    McpProxyTool,
    bridged_tool_name,
    format_call_result,
    load_mcp_tools,
    parameters_from_schema,
)

# --- fakes -----------------------------------------------------------------


class FakeBridge:
    """Duck-types the two McpBridge members McpProxyTool touches."""

    def __init__(self, response="ok", error: Exception = None):
        self.calls = []
        self.response = response
        self.error = error
        self.failures = []
        self.tools_by_server = {}

    def call(self, server, remote_tool, arguments):
        self.calls.append((server, remote_tool, arguments))
        if self.error is not None:
            raise self.error
        return self.response


class FakeDesktopFilesTool(BaseTool):
    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="desktop_files",
            description="fake",
            category=ToolCategory.FILE,
            parameters=[
                ToolParameter(name="action", type="string", description="read"),
                ToolParameter(
                    name="path", type="string", description="path", required=False
                ),
            ],
            returns="content",
        )

    def execute(self, **kwargs) -> ToolResult:
        return ToolResult(
            tool_name="desktop_files", success=True, result="SECRET CONTENTS"
        )


def _proxy(network=True, bridge=None, schema=None) -> McpProxyTool:
    return McpProxyTool(
        bridge or FakeBridge(),
        "myserver",
        "do_thing",
        "Does a thing.",
        schema or {"type": "object", "properties": {}},
        network=network,
    )


def _context() -> List[ChatMessage]:
    return [
        ChatMessage(role="system", content="You are a helpful assistant."),
        ChatMessage(role="user", content="please do the thing"),
    ]


@pytest.fixture(autouse=True)
def _reset_bridge_cache():
    saved = dict(mcp_bridge._CACHE)
    mcp_bridge._CACHE.update({"fingerprint": None, "bridge": None})
    yield
    mcp_bridge._CACHE.update(saved)


# --- pure helpers ----------------------------------------------------------


def test_bridged_tool_name_sanitization():
    assert bridged_tool_name("files", "read_file") == "mcp_files_read_file"
    assert bridged_tool_name("My Server!", "Do-Thing") == "mcp_my_server_do_thing"


def test_parameters_from_schema_flattens_types_required_enum():
    params = parameters_from_schema(
        {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "what to search"},
                "count": {"type": "integer"},
                "deep": {"type": "boolean"},
                "filters": {"type": "object"},
                "mode": {"type": "string", "enum": ["fast", "slow"]},
                "maybe": {"type": ["string", "null"]},
                "weird": {"type": "no-such-type"},
                "coded_enum": {"type": "number", "enum": [1, 2]},
            },
            "required": ["query"],
        }
    )
    by_name = {p.name: p for p in params}
    assert by_name["query"].required and by_name["query"].type == "string"
    assert by_name["count"].type == "number"  # integer collapses to number
    assert by_name["deep"].type == "boolean"
    assert by_name["filters"].type == "object"
    assert by_name["mode"].enum == ["fast", "slow"]
    assert by_name["maybe"].type == "string"  # union picks a mappable member
    assert by_name["weird"].type == "string"  # unknown falls back to string
    assert by_name["coded_enum"].enum is None  # non-string enums dropped
    assert not by_name["count"].required


def test_parameters_from_schema_tolerates_junk():
    assert parameters_from_schema(None) == []
    assert parameters_from_schema({"type": "object"}) == []
    assert parameters_from_schema({"properties": "nope"}) == []


def test_format_call_result_variants():
    class Item:
        def __init__(self, text=None, type=None):
            if text is not None:
                self.text = text
            if type is not None:
                self.type = type

    class Result:
        def __init__(self, content=None, structured=None):
            self.content = content or []
            if structured is not None:
                self.structuredContent = structured

    assert format_call_result(Result(structured={"a": 1})) == '{"a": 1}'
    assert format_call_result(Result(content=[Item(text="hi")])) == "hi"
    assert (
        format_call_result(Result(content=[Item(text="a"), Item(type="image")]))
        == "a\n[non-text content: image]"
    )
    assert format_call_result(Result()) == "[empty result]"


# --- proxy tool ------------------------------------------------------------


def test_proxy_tool_definition_and_guard_attributes():
    tool = _proxy(network=True)
    assert tool.definition.name == "mcp_myserver_do_thing"
    assert "myserver" in tool.definition.description
    assert tool.untrusted_source is True
    assert tool.network_capable is True
    assert _proxy(network=False).network_capable is False


def test_proxy_tool_executes_via_bridge():
    bridge = FakeBridge(response="the answer")
    tool = _proxy(
        bridge=bridge,
        schema={
            "type": "object",
            "properties": {"q": {"type": "string"}},
            "required": ["q"],
        },
    )
    before = telemetry.snapshot().get("tools.mcp_calls", 0)
    result = tool.execute(q="hello")
    assert result.success and result.result == "the answer"
    assert bridge.calls == [("myserver", "do_thing", {"q": "hello"})]
    assert telemetry.snapshot()["tools.mcp_calls"] == before + 1


def test_proxy_tool_call_failure_counts_and_returns_error():
    before = telemetry.snapshot().get("tools.mcp_call_failures", 0)
    tool = _proxy(bridge=FakeBridge(error=RuntimeError("server exploded")))
    result = tool.execute()
    assert not result.success
    assert "server exploded" in result.error
    assert telemetry.snapshot()["tools.mcp_call_failures"] == before + 1


def test_proxy_tool_survives_central_validation():
    """execute_tool's defensive validate_parameters pass must accept a
    conforming call and reject unknown params, same as built-in tools."""
    from amadeus.agents.tools import ToolCall

    registry = ToolRegistry()
    tool = _proxy(schema={"type": "object", "properties": {"q": {"type": "string"}}})
    registry.register(tool)
    ok = registry.execute_tool(
        ToolCall(tool_name="mcp_myserver_do_thing", parameters={"q": "x"})
    )
    assert ok.success
    bad = registry.execute_tool(
        ToolCall(tool_name="mcp_myserver_do_thing", parameters={"nope": "x"})
    )
    assert not bad.success and "Unknown parameter" in bad.error


# --- allowlist + guards ----------------------------------------------------


def test_tool_allowed_meta_entry():
    assert tool_allowed("mcp_files_read", ["mcp"])
    assert tool_allowed("mcp_files_read", ["web_search", "mcp"])
    assert not tool_allowed("mcp_files_read", ["web_search"])
    assert tool_allowed("mcp_files_read", ["mcp_files_read"])  # explicit name
    assert tool_allowed("web_search", ["web_search"])
    assert not tool_allowed("web_search", ["mcp"])
    assert tool_allowed("anything", None)


def test_is_network_and_untrusted_classification():
    registry = ToolRegistry()
    net = _proxy(network=True)
    local = McpProxyTool(
        FakeBridge(),
        "local",
        "read",
        "",
        {"type": "object", "properties": {}},
        network=False,
    )
    registry.register(net)
    registry.register(local)
    assert _is_network_tool("web_search", registry)  # static set still wins
    assert _is_network_tool(net.definition.name, registry)
    assert not _is_network_tool(local.definition.name, registry)
    # Every MCP proxy is an untrusted source regardless of network flag.
    assert _is_untrusted_source(net.definition.name, registry)
    assert _is_untrusted_source(local.definition.name, registry)
    assert not _is_network_tool("mcp_unregistered_tool", registry)


def test_react_loop_refuses_network_mcp_tool_after_file_read():
    registry = ToolRegistry()
    registry.register(FakeDesktopFilesTool())
    bridge = FakeBridge()
    registry.register(_proxy(bridge=bridge))

    replies = [
        'ACTION: {"tool": "desktop_files", "params": {"action": "read"}}',
        'ACTION: {"tool": "mcp_myserver_do_thing", "params": {}}',
        "Summarized from your notes.",
    ]
    seen_context = []

    def generate(messages):
        seen_context.append(list(messages))
        return replies.pop(0)

    reply = run_tool_loop(generate, registry, _context())
    assert reply == "Summarized from your notes."
    # The bridged call was refused, not executed.
    assert bridge.calls == []
    last_msg = seen_context[-1][-1]
    assert "refused" in last_msg.content
    assert "file contents were read" in last_msg.content


def test_react_loop_allows_local_mcp_tool_after_file_read():
    registry = ToolRegistry()
    registry.register(FakeDesktopFilesTool())
    bridge = FakeBridge(response="local result")
    registry.register(_proxy(bridge=bridge, network=False))

    replies = [
        'ACTION: {"tool": "desktop_files", "params": {"action": "read"}}',
        'ACTION: {"tool": "mcp_myserver_do_thing", "params": {}}',
        "Done.",
    ]

    def generate(messages):
        return replies.pop(0)

    assert run_tool_loop(generate, registry, _context()) == "Done."
    assert bridge.calls == [("myserver", "do_thing", {})]


# --- load_mcp_tools --------------------------------------------------------


def test_load_mcp_tools_disabled_returns_nothing():
    config = Config()
    assert load_mcp_tools(config) == ([], [])


def test_load_mcp_tools_enabled_without_servers_reports():
    config = Config()
    config.mcp.enabled = True
    tools, unavailable = load_mcp_tools(config)
    assert tools == []
    assert unavailable and "no servers configured" in unavailable[0]["reason"]


def test_load_mcp_tools_missing_sdk_degrades_loudly(monkeypatch):
    config = Config()
    config.mcp.enabled = True
    config.mcp.servers = [{"name": "files", "command": "echo"}]
    monkeypatch.setitem(sys.modules, "mcp", None)  # import mcp -> ImportError
    before = telemetry.snapshot().get("tools.mcp_connect_failed", 0)
    tools, unavailable = load_mcp_tools(config)
    assert tools == []
    assert unavailable and "amadeus-kurisu[mcp]" in unavailable[0]["reason"]
    assert telemetry.snapshot()["tools.mcp_connect_failed"] == before + 1


def _fake_sdk(monkeypatch):
    monkeypatch.setitem(sys.modules, "mcp", types.ModuleType("mcp"))


def _fake_connected_bridge(monkeypatch, tools_by_server, failures=None):
    bridge = FakeBridge()
    bridge.tools_by_server = tools_by_server
    bridge.failures = failures or []
    monkeypatch.setattr(mcp_bridge, "_get_bridge", lambda settings: bridge)
    return bridge


def test_load_mcp_tools_wraps_and_filters(monkeypatch):
    _fake_sdk(monkeypatch)
    _fake_connected_bridge(
        monkeypatch,
        {
            "files": [
                ("read_file", "Read a file.", {"type": "object", "properties": {}}),
                ("write_file", "Write a file.", {"type": "object", "properties": {}}),
            ],
            "search": [("query", "Search.", {})],
        },
        failures=[{"name": "mcp:dead", "reason": "spawn failed"}],
    )
    config = Config()
    config.mcp.enabled = True
    config.mcp.servers = [
        {
            "name": "files",
            "command": "x",
            "allowed_tools": ["read_file"],
            "network": False,
        },
        {"name": "search", "url": "http://localhost:1/mcp"},
        {"name": "dead", "command": "y"},
    ]
    tools, unavailable = load_mcp_tools(config)
    names = {t.definition.name for t in tools}
    # write_file filtered by allowed_tools; dead server only in unavailable.
    assert names == {"mcp_files_read_file", "mcp_search_query"}
    by_name = {t.definition.name: t for t in tools}
    assert by_name["mcp_files_read_file"].network_capable is False
    assert by_name["mcp_search_query"].network_capable is True
    assert {"name": "mcp:dead", "reason": "spawn failed"} in unavailable


def test_bridge_cache_reuses_connection(monkeypatch):
    connects = []
    monkeypatch.setattr(
        mcp_bridge.McpBridge, "connect", lambda self: connects.append(1)
    )
    config = Config()
    config.mcp.servers = [{"name": "files", "command": "x"}]
    first = mcp_bridge._get_bridge(config.mcp)
    second = mcp_bridge._get_bridge(config.mcp)
    assert first is second
    assert len(connects) == 1
    # A config change (new fingerprint) builds a fresh bridge.
    config.mcp.servers = [{"name": "other", "command": "y"}]
    third = mcp_bridge._get_bridge(config.mcp)
    assert third is not first
    assert len(connects) == 2


def test_factory_registers_mcp_tools(monkeypatch, tmp_path):
    _fake_sdk(monkeypatch)
    _fake_connected_bridge(
        monkeypatch,
        {"files": [("read_file", "Read.", {"type": "object", "properties": {}})]},
    )
    config = Config()
    config.memory.enable_fact_extraction = False
    config.mcp.enabled = True
    config.mcp.servers = [{"name": "files", "command": "x"}]
    registry = create_tool_registry(config)
    assert "mcp_files_read_file" in registry.tools

    off = Config()
    off.memory.enable_fact_extraction = False
    assert not any(name.startswith("mcp_") for name in create_tool_registry(off).tools)


def test_factory_records_unavailable_from_bridge(monkeypatch):
    _fake_sdk(monkeypatch)
    _fake_connected_bridge(
        monkeypatch, {}, failures=[{"name": "mcp:files", "reason": "boom"}]
    )
    config = Config()
    config.memory.enable_fact_extraction = False
    config.mcp.enabled = True
    config.mcp.servers = [{"name": "files", "command": "x"}]
    registry = create_tool_registry(config)
    assert {"name": "mcp:files", "reason": "boom"} in registry.unavailable
