"""Read-only desktop file access for the tool loop.

Bridges to the `desktop-ops` server's READ functions only (list / find /
classify). Write operations (move/trash) are intentionally never imported here,
so Amadeus physically cannot modify the filesystem through this tool — the same
guarantee the read-only MCP mode gives, enforced at import time.
"""

from __future__ import annotations

import importlib
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Path to the desktop-ops package (installed in its own venv, but the module is
# pure-stdlib for the read tools, so we import it directly by path).
_DESKTOP_OPS_PKG = os.environ.get(
    "DESKTOP_OPS_PKG",
    "desktop-ops-mcp",
)


class DesktopFilesClient:
    """Read-only client over the desktop tree.

    Exposes exactly three operations — list, find, classify — and nothing that
    can mutate the filesystem.
    """

    def __init__(self, root: Optional[str] = None) -> None:
        if root:
            os.environ["DESKTOP_OPS_ROOT"] = root
        # Force read-only so even the imported module registers no write tools.
        os.environ["DESKTOP_OPS_READONLY"] = "1"
        if _DESKTOP_OPS_PKG not in sys.path:
            sys.path.insert(0, _DESKTOP_OPS_PKG)
        # Import the pure core (no MCP dependency). Bind ONLY the read
        # functions — file_into/trash/restore are never referenced here.
        core = importlib.import_module("desktop_ops.core")
        self._list = core.list_desktop
        self._find = core.find_files
        self._classify = core.classify_clutter
        self._read = core.read_file  # read-only; whitelist-confined in core
        self._readable_root = Path(core.READABLE_CONTENT_ROOT)

    def list_desktop(
        self, subdir: str = "", pattern: str = "*", max_results: int = 200
    ) -> List[Dict[str, Any]]:
        entries = self._list(subdir=subdir, pattern=pattern, max_results=max_results)
        # Office lock files (~$foo.docx) are pure noise in a listing the
        # model reasons over — they look like the real document.
        return [
            entry
            for entry in entries
            if not os.path.basename(str(entry.get("path", ""))).startswith("~$")
        ]

    def find_files(self, query: str, max_results: int = 50) -> List[Dict[str, Any]]:
        return self._find(query=query, max_results=max_results)

    def classify_clutter(self) -> Dict[str, Any]:
        return self._classify()

    def read_file(self, path: str) -> Dict[str, Any]:
        """Read the text contents of a file under _Organized/ (PDF/docx/text).
        Whitelist-confined and capped in the core; still read-only.

        The core only reads under _Organized/, but the model usually asks for
        paths the way they appear in list results ("Manuscripts/x.docx", or a
        root-level file that has an organized sibling). Meet it halfway:
        normalize, try the _Organized/ prefix, then fall back to locating a
        unique readable file with the same basename — otherwise the model
        burns its whole iteration budget guessing path shapes (observed live:
        four failed guesses, then an empty answer).
        """
        norm = path.strip().lstrip("/\\")
        candidates = [norm]
        if not norm.startswith("_Organized/"):
            candidates.append(f"_Organized/{norm}")
        last_error: Dict[str, Any] = {"error": "read failed"}
        for candidate in candidates:
            result = self._read(candidate)
            if not (isinstance(result, dict) and result.get("error")):
                return result
            last_error = result

        base = os.path.basename(norm)
        if base:
            # Deliberately NOT core.find_files here: that walks the entire
            # desktop tree, and a query with no early hits takes ~1 minute
            # over the /mnt/c mount. Readable files can only live under
            # _Organized/, so walk just that (fraction of a second), with a
            # time bound as a belt-and-suspenders.
            matches = self._find_readable_by_basename(base)
            if len(matches) == 1:
                result = self._read(matches[0])
                if not (isinstance(result, dict) and result.get("error")):
                    result["resolved_path"] = matches[0]
                    return result
            elif matches:
                return {
                    "error": f"'{path}' matches several readable files — "
                    "pick one exact path",
                    "candidates": matches[:5],
                }

        return {
            "error": (
                f"{last_error.get('error', 'read failed')}. Only files under "
                "'_Organized/' are readable. Use action=find with part of the "
                "filename to locate a readable copy, then read its exact path."
            )
        }

    def _find_readable_by_basename(
        self, basename: str, limit: int = 6, budget_seconds: float = 3.0
    ) -> List[str]:
        """Exact basename matches under _Organized/, as core-relative paths."""
        deadline = time.monotonic() + budget_seconds
        wanted = basename.lower()
        matches: List[str] = []
        root = self._readable_root
        for dirpath, _dirnames, filenames in os.walk(root):
            if time.monotonic() > deadline:
                break
            for name in filenames:
                if name.lower() == wanted:
                    rel = os.path.relpath(os.path.join(dirpath, name), root.parent)
                    matches.append(rel.replace(os.sep, "/"))
                    if len(matches) >= limit:
                        return matches
        return matches


def build_desktop_files_client(settings) -> DesktopFilesClient:
    """Factory mirroring build_web_search_client. `settings` is the
    DesktopFilesSettings dataclass (root path lives there)."""
    if not getattr(settings, "enabled", False):
        raise RuntimeError("Desktop file access is disabled in configuration")
    return DesktopFilesClient(root=getattr(settings, "root", None))
