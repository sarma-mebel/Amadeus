# MCP tool bridge

Amadeus can consume tools from external MCP (Model Context Protocol)
servers. Anything you configure shows up in her tool registry as
`mcp_<server>_<tool>`, next to the built-ins, and runs through the exact
same hardened tool loop: allowlist firewall, central parameter validation,
observation sanitizer, and the exfiltration guard.

## Setup

```bash
pip install "amadeus-kurisu[mcp]"   # needs Python 3.10+
```

```yaml
tools:
  enabled: true

mcp:
  enabled: true
  servers:
    - name: files                   # stdio server, spawned as a child process
      command: npx
      args: ["-y", "@modelcontextprotocol/server-filesystem", "/some/path"]
      network: false                # a filesystem server never touches the net
    - name: search                  # streamable-HTTP server
      url: http://127.0.0.1:8931/mcp
      allowed_tools: [search]       # optional: bridge only these tools
```

The default `tools.allowed` list contains the meta-entry `"mcp"`, which
admits every bridged tool. Remove it and list `mcp_<server>_<tool>` names
individually if you want per-tool control.

Check `GET /api/status` after startup: bridged tools appear under
`tools.registered`, and anything that failed to connect appears under
`tools.unavailable` with a reason.

## A first working setup

The clearest starting point is the official filesystem MCP server, pointed
at a notes folder, so she can read and search your notes when you ask.
It's local-only, so it clears the exfiltration guard cleanly (see below),
and it's useful on day one.

Prerequisites: Node.js installed (`npx` spawns the server), the `[mcp]`
extra, and Python 3.10+.

```yaml
tools:
  enabled: true

mcp:
  enabled: true
  servers:
    - name: notes
      command: npx
      args: ["-y", "@modelcontextprotocol/server-filesystem", "/path/to/your/notes"]
      network: false
```

`network: false` matters here beyond documentation: it exempts these tools
from the exfiltration guard, so they still work in turns that also touched
local files via `desktop_files`. A network-capable server would get
refused in that situation. That makes local-only servers like this one the
sweet spot for bridged tools.

Try asking her something like "what do my notes say about the Amadeus
project?" and she'll call the bridged tool to look.

To verify it's wired up: `GET /api/status` should list the bridged tools
under `tools.registered` as `mcp_notes_<tool>` (e.g. `mcp_notes_read_file`),
and each successful call increments the `tools.mcp_calls` counter.

## Security model

Bridged tools are third-party code returning third-party content, so they
get the strict end of the existing guards:

- Every MCP tool's output is treated as untrusted retrieved data (same
  framing as web search results; instructions inside it are ignored).
- MCP tools are assumed network-capable: after any `desktop_files` call in
  a turn, they are refused for the rest of that turn, exactly like
  `web_search`, so local file contents can't be exfiltrated through a bridged
  tool's arguments. Set `network: false` on a server ONLY when it is
  genuinely local-only; that exempts it from this guard.
- Parameters are validated centrally before the call; the server
  re-validates its own schema on its side.

## Operational notes

- Stdio servers are real child processes. They are spawned once per backend
  process and reused (the `/api/status` probe never respawns them); a
  config change reconnects on the next registry build.
- Per-call failures return an error observation to the model and count
  `tools.mcp_call_failures`; connect failures count
  `tools.mcp_connect_failed`. All-zeros is healthy.
- Timeouts: `mcp.connect_timeout_seconds` (default 15) and
  `mcp.call_timeout_seconds` (default 30).
- Nested/exotic JSON-Schema parameters flatten to plain objects: tools
  with deeply structured inputs work, but the model only sees top-level
  parameter names and types.
