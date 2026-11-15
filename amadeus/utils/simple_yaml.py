"""Simple YAML reader/writer for environments without PyYAML."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

__all__ = ["safe_load", "safe_dump"]


def safe_load(text: str) -> Dict[str, Any]:
    if hasattr(text, "read"):
        text = text.read()
    lines = str(text).splitlines()
    root: Dict[str, Any] = {}
    stack: List[Tuple[Dict[str, Any], int]] = [(root, -1)]
    for raw in lines:
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        indent = len(raw) - len(raw.lstrip(" "))
        key, _, value = raw.partition(":")
        key = key.strip()
        value = _strip_inline_comment(value.strip())
        while stack and indent <= stack[-1][1]:
            stack.pop()
        parent, _ = stack[-1]
        if value == "":
            node: Dict[str, Any] = {}
            parent[key] = node
            stack.append((node, indent))
        else:
            parent[key] = _parse_scalar(value)
    return _convert_block_lists(root)


def _convert_block_lists(node: Dict[str, Any]) -> Dict[str, Any]:
    """Turn ``key:`` sections whose children are all bare ``- item`` lines
    into real lists. The line loop above parses those lines as valueless
    keys, so a block sequence of scalars arrives here as
    ``{"- a": {}, "- b": {}}``. Only that shape converts; sequences of
    maps stay unsupported (and duplicate scalar items would collapse into
    one dict key before this ever runs)."""
    for key, child in list(node.items()):
        if not isinstance(child, dict) or not child:
            continue
        if all(sub.startswith("- ") for sub in child) and all(
            value == {} for value in child.values()
        ):
            node[key] = [
                _parse_scalar(_strip_inline_comment(sub[1:].strip())) for sub in child
            ]
        else:
            _convert_block_lists(child)
    return node


def _split_flow_items(inner: str) -> List[str]:
    """Split a flow sequence body on commas outside quotes."""
    items: List[str] = []
    buffer: List[str] = []
    quote: str | None = None
    for char in inner:
        if quote:
            buffer.append(char)
            if char == quote:
                quote = None
        elif char in "'\"":
            quote = char
            buffer.append(char)
        elif char == ",":
            items.append("".join(buffer).strip())
            buffer = []
        else:
            buffer.append(char)
    tail = "".join(buffer).strip()
    if tail:
        items.append(tail)
    return items


def safe_dump(data: Dict[str, Any], handle, sort_keys: bool = False) -> None:
    keys = sorted(data.keys()) if sort_keys else data.keys()
    for key in keys:
        _write_pair(handle, key, data[key], indent=0)


def _write_pair(handle, key: str, value: Any, indent: int) -> None:
    prefix = " " * indent
    if isinstance(value, dict):
        handle.write(f"{prefix}{key}:\n")
        for sub_key, sub_value in value.items():
            _write_pair(handle, sub_key, sub_value, indent + 2)
    else:
        handle.write(f"{prefix}{key}: {_format_scalar(value)}\n")


def _strip_inline_comment(value: str) -> str:
    """Drop a trailing ` # comment` per YAML rules: a hash opens a comment
    only at the value start or after whitespace, and never inside a quoted
    value. Before this existed, ``count: 5  # note`` loaded as the STRING
    ``"5  # note"`` — the example config uses inline comments, so a
    PyYAML-less install silently misread real settings."""
    if not value:
        return value
    quote = value[0] if value[0] in "'\"" else None
    if quote:
        closing = value.find(quote, 1)
        # Keep the quoted span (hashes inside are data); anything after the
        # closing quote can only be a comment in the grammar we support.
        return value[: closing + 1] if closing != -1 else value
    for index, char in enumerate(value):
        if char == "#" and (index == 0 or value[index - 1] in " \t"):
            return value[:index].rstrip()
    return value


def _parse_scalar(value: str) -> Any:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
        return value[1:-1]
    if value == "{}":
        return {}
    if len(value) >= 2 and value[0] == "[" and value[-1] == "]":
        inner = value[1:-1].strip()
        return (
            [_parse_scalar(item) for item in _split_flow_items(inner)] if inner else []
        )
    lowered = value.lower()
    if lowered in {"null", "none"}:
        return None
    if lowered in {"true", "yes"}:
        return True
    if lowered in {"false", "no"}:
        return False
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value.strip('"')


def _format_scalar(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if value == "":
        # A bare ``key: `` line reads back as an empty SECTION, not an empty
        # string — quote it so dump/load round-trips.
        return '""'
    return str(value)
