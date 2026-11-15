"""
Tool Registry and Base Interfaces for Agent Tool Use

Provides a standardized interface for agents to call external tools,
including web search, code execution, calculators, and file operations.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from enum import Enum
import ast
import json
import re


class ToolCategory(Enum):
    """Categories of tools"""

    SEARCH = "search"
    COMPUTATION = "computation"
    DATA = "data"
    COMMUNICATION = "communication"
    FILE = "file"


@dataclass
class ToolParameter:
    """Definition of a tool parameter"""

    name: str
    type: str  # "string", "number", "boolean", "array", "object"
    description: str
    required: bool = True
    default: Optional[Any] = None
    enum: Optional[List[str]] = None  # For restricted choices


@dataclass
class ToolDefinition:
    """Definition of a tool's interface"""

    name: str
    description: str
    category: ToolCategory
    parameters: List[ToolParameter]
    returns: str  # Description of return value
    examples: Optional[List[Dict[str, Any]]] = None
    cost_tier: int = 1  # 1=free, 2=moderate, 3=expensive


@dataclass
class ToolCall:
    """A request to call a tool"""

    tool_name: str
    parameters: Dict[str, Any]
    call_id: Optional[str] = None  # For tracking async calls


@dataclass
class ToolResult:
    """Result of a tool execution"""

    tool_name: str
    success: bool
    result: Any
    error: Optional[str] = None
    call_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None  # Execution time, cost, etc.


class BaseTool(ABC):
    """Abstract base class for all tools"""

    def __init__(self):
        self.definition = self.get_definition()

    @abstractmethod
    def get_definition(self) -> ToolDefinition:
        """Return the tool's definition"""
        pass

    @abstractmethod
    def execute(self, **kwargs) -> ToolResult:
        """
        Execute the tool with given parameters.

        Args:
            **kwargs: Tool parameters as defined in ToolDefinition

        Returns:
            ToolResult with success status and result/error
        """
        pass

    def validate_parameters(self, parameters: Dict[str, Any]) -> Optional[str]:
        """
        Validate parameters against tool definition.

        Returns:
            Error message if validation fails, None if valid
        """
        # Check required parameters
        for param in self.definition.parameters:
            if param.required and param.name not in parameters:
                return f"Missing required parameter: {param.name}"

        # Check parameter types (basic validation)
        for param_name, param_value in parameters.items():
            param_def = next(
                (p for p in self.definition.parameters if p.name == param_name), None
            )
            if param_def is None:
                return f"Unknown parameter: {param_name}"

            # Type checking
            if param_def.type == "string" and not isinstance(param_value, str):
                return f"Parameter {param_name} must be a string"
            elif param_def.type == "number" and not isinstance(
                param_value, (int, float)
            ):
                return f"Parameter {param_name} must be a number"
            elif param_def.type == "boolean" and not isinstance(param_value, bool):
                return f"Parameter {param_name} must be a boolean"
            elif param_def.type == "array" and not isinstance(param_value, list):
                return f"Parameter {param_name} must be an array"
            elif param_def.type == "object" and not isinstance(param_value, dict):
                return f"Parameter {param_name} must be an object"

            # Enum validation
            if param_def.enum and param_value not in param_def.enum:
                return f"Parameter {param_name} must be one of {param_def.enum}"

        return None


class WebSearchTool(BaseTool):
    """Tool for searching the web"""

    def __init__(self, search_client):
        """
        Args:
            search_client: Instance of WebSearchClient from tools.web_search
        """
        self.search_client = search_client
        super().__init__()

    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="web_search",
            description="Search the web for current information, research papers, and references",
            category=ToolCategory.SEARCH,
            parameters=[
                ToolParameter(
                    name="query",
                    type="string",
                    description="Search query string",
                    required=True,
                ),
                ToolParameter(
                    name="max_results",
                    type="number",
                    description="Maximum number of results to return",
                    required=False,
                    default=3,
                ),
            ],
            returns="List of search results with title, snippet, and URL",
            examples=[
                {"query": "recent advances in psychedelic therapy", "max_results": 5}
            ],
            cost_tier=2,
        )

    def execute(self, **kwargs) -> ToolResult:
        """Execute web search"""
        validation_error = self.validate_parameters(kwargs)
        if validation_error:
            return ToolResult(
                tool_name="web_search",
                success=False,
                result=None,
                error=validation_error,
            )

        try:
            query = kwargs["query"]
            max_results = kwargs.get("max_results", 3)

            results = self.search_client.search(query, max_results=max_results)

            return ToolResult(
                tool_name="web_search",
                success=True,
                result=results,
                metadata={"query": query, "result_count": len(results)},
            )
        except Exception as e:
            return ToolResult(
                tool_name="web_search", success=False, result=None, error=str(e)
            )


class DesktopFilesTool(BaseTool):
    """Read-only access to the user's desktop files (list / find / classify).

    Cannot move, rename, or delete anything — the backing client only exposes
    read operations."""

    def __init__(self, files_client):
        self.files_client = files_client
        super().__init__()

    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="desktop_files",
            description=(
                "Read-only search over the user's desktop: find a file by name, "
                "list a folder, or report clutter. Never modifies files."
            ),
            category=ToolCategory.SEARCH,
            parameters=[
                ToolParameter(
                    name="action",
                    type="string",
                    description="One of: find, list, classify, read",
                    required=True,
                ),
                ToolParameter(
                    name="query",
                    type="string",
                    description="For action=find: substring of the filename",
                    required=False,
                ),
                ToolParameter(
                    name="path",
                    type="string",
                    description="For action=read: relative path of a file under "
                    "_Organized/ (PDF, .docx, or text) to read.",
                    required=False,
                ),
                ToolParameter(
                    name="subdir",
                    type="string",
                    description="For action=list: relative folder (default root)",
                    required=False,
                    default="",
                ),
                ToolParameter(
                    name="max_results",
                    type="number",
                    description="Cap on results",
                    required=False,
                    default=50,
                ),
            ],
            returns="File metadata (relative path, size, modified date).",
            examples=[
                {"action": "find", "query": "invoice"},
                {"action": "list", "subdir": "Documents"},
                {
                    "action": "read",
                    "path": "Documents/report.pdf",
                },
                {"action": "classify"},
            ],
            cost_tier=1,
        )

    def execute(self, **kwargs) -> ToolResult:
        validation_error = self.validate_parameters(kwargs)
        if validation_error:
            return ToolResult(
                tool_name="desktop_files",
                success=False,
                result=None,
                error=validation_error,
            )
        action = (kwargs.get("action") or "").lower()
        try:
            if action == "find":
                q = kwargs.get("query")
                if not q:
                    raise ValueError("action=find requires 'query'")
                result = self.files_client.find_files(
                    q, max_results=int(kwargs.get("max_results", 50))
                )
            elif action == "list":
                result = self.files_client.list_desktop(
                    subdir=kwargs.get("subdir", ""),
                    max_results=int(kwargs.get("max_results", 200)),
                )
            elif action == "classify":
                result = self.files_client.classify_clutter()
            elif action == "read":
                p = kwargs.get("path")
                if not p:
                    raise ValueError("action=read requires 'path'")
                result = self.files_client.read_file(p)
            else:
                raise ValueError(f"unknown action '{action}' (find|list|classify|read)")
            return ToolResult(
                tool_name="desktop_files",
                success=True,
                result=result,
                metadata={"action": action},
            )
        except Exception as e:
            return ToolResult(
                tool_name="desktop_files", success=False, result=None, error=str(e)
            )


class PubMedSearchTool(BaseTool):
    """Search PubMed for biomedical papers (free, no key)."""

    def __init__(self, client):
        self.client = client
        super().__init__()

    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="pubmed_search",
            description=(
                "PubMed biomedical/clinical papers. action=search returns a LIST "
                "(title, authors, venue, date, url) — titles/metadata only, NOT "
                "abstracts. action=fetch returns ONE paper's FULL record incl. the "
                "complete author list and the ABSTRACT — use fetch before answering "
                "questions about a specific paper's authors, methods, or findings."
            ),
            category=ToolCategory.SEARCH,
            parameters=[
                ToolParameter(
                    name="action",
                    type="string",
                    description="search (find papers) or fetch (one paper's details)",
                    required=False,
                    default="search",
                ),
                ToolParameter(
                    name="query",
                    type="string",
                    description="For search: keywords. For fetch: the paper title or PMID.",
                    required=True,
                ),
                ToolParameter(
                    name="max_results",
                    type="number",
                    description="search only: how many (default 5)",
                    required=False,
                    default=5,
                ),
            ],
            returns="search: list of papers (no abstracts). fetch: one paper + abstract.",
            examples=[
                {"action": "search", "query": "cerebral microbleed segmentation"},
                {
                    "action": "fetch",
                    "query": "psilocybin default mode network Carhart-Harris",
                },
            ],
            cost_tier=1,
        )

    def execute(self, **kwargs) -> ToolResult:
        err = self.validate_parameters(kwargs)
        if err:
            return ToolResult(
                tool_name="pubmed_search", success=False, result=None, error=err
            )
        try:
            if (kwargs.get("action") or "search").lower() == "fetch":
                paper = self.client.fetch(kwargs["query"])
                if not paper:
                    return ToolResult(
                        tool_name="pubmed_search",
                        success=True,
                        result="No matching paper found to fetch.",
                    )
                return ToolResult(
                    tool_name="pubmed_search",
                    success=True,
                    result=paper,
                    metadata={"mode": "fetch"},
                )
            results = self.client.search(
                kwargs["query"], max_results=int(kwargs.get("max_results", 5))
            )
            return ToolResult(
                tool_name="pubmed_search",
                success=True,
                result=results,
                metadata={"count": len(results), "mode": "search"},
            )
        except Exception as e:
            return ToolResult(
                tool_name="pubmed_search", success=False, result=None, error=str(e)
            )


class ArxivSearchTool(BaseTool):
    """Search arXiv for preprints (free, no key)."""

    def __init__(self, client):
        self.client = client
        super().__init__()

    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="arxiv_search",
            description=(
                "arXiv preprints (CS/physics/stats/bio). action=search returns a "
                "LIST with truncated summaries. action=fetch returns ONE paper's "
                "FULL record incl. the complete author list and full abstract — use "
                "fetch before answering about a specific paper's authors or methods."
            ),
            category=ToolCategory.SEARCH,
            parameters=[
                ToolParameter(
                    name="action",
                    type="string",
                    description="search (find papers) or fetch (one paper's details)",
                    required=False,
                    default="search",
                ),
                ToolParameter(
                    name="query",
                    type="string",
                    description="For search: keywords. For fetch: the paper title or arXiv id.",
                    required=True,
                ),
                ToolParameter(
                    name="max_results",
                    type="number",
                    description="search only: how many (default 5)",
                    required=False,
                    default=5,
                ),
            ],
            returns="search: list of preprints. fetch: one preprint + full abstract.",
            examples=[
                {"action": "search", "query": "foundation model time series ICU"},
                {
                    "action": "fetch",
                    "query": "curiosity information-seeking neuro-computational",
                },
            ],
            cost_tier=1,
        )

    def execute(self, **kwargs) -> ToolResult:
        err = self.validate_parameters(kwargs)
        if err:
            return ToolResult(
                tool_name="arxiv_search", success=False, result=None, error=err
            )
        try:
            if (kwargs.get("action") or "search").lower() == "fetch":
                paper = self.client.fetch(kwargs["query"])
                if not paper:
                    return ToolResult(
                        tool_name="arxiv_search",
                        success=True,
                        result="No matching paper found to fetch.",
                    )
                return ToolResult(
                    tool_name="arxiv_search",
                    success=True,
                    result=paper,
                    metadata={"mode": "fetch"},
                )
            results = self.client.search(
                kwargs["query"], max_results=int(kwargs.get("max_results", 5))
            )
            return ToolResult(
                tool_name="arxiv_search",
                success=True,
                result=results,
                metadata={"count": len(results), "mode": "search"},
            )
        except Exception as e:
            return ToolResult(
                tool_name="arxiv_search", success=False, result=None, error=str(e)
            )


class ReminderTool(BaseTool):
    """Set / list / cancel spoken reminders and timers (local, no network)."""

    def __init__(self, store):
        self.store = store
        super().__init__()

    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="reminders",
            description=(
                "Set, list, or cancel spoken reminders and timers. "
                "For a timer ('in 20 minutes') use in_minutes. For an absolute "
                "time ('at 4pm') use at (ISO like 2026-07-10T16:00, or '4pm')."
            ),
            category=ToolCategory.COMMUNICATION,
            parameters=[
                ToolParameter(
                    name="action",
                    type="string",
                    description="add | list | cancel",
                    required=True,
                ),
                ToolParameter(
                    name="text",
                    type="string",
                    description="What to say when due (for add)",
                    required=False,
                ),
                ToolParameter(
                    name="in_minutes",
                    type="number",
                    description="Relative delay in minutes (timers)",
                    required=False,
                ),
                ToolParameter(
                    name="at",
                    type="string",
                    description="Absolute time: ISO (2026-07-10T16:00) or '4pm'/'16:30'",
                    required=False,
                ),
                ToolParameter(
                    name="reminder_id",
                    type="string",
                    description="ID from the list output (for cancel)",
                    required=False,
                ),
            ],
            returns="Confirmation with id and resolved time, or the pending list.",
            examples=[
                {
                    "action": "add",
                    "text": "submit the revision",
                    "at": "2026-07-10T16:00",
                },
                {"action": "add", "text": "tea is ready", "in_minutes": 20},
                {"action": "list"},
                {"action": "cancel", "reminder_id": "a3f1c2"},
            ],
            cost_tier=1,
        )

    def _parse_time(self, kwargs):
        import datetime as _dt

        now = _dt.datetime.now()
        if kwargs.get("in_minutes") is not None:
            return now + _dt.timedelta(minutes=float(kwargs["in_minutes"])), None
        at = kwargs.get("at")
        if not at:
            return None, "Provide in_minutes or at."
        at = str(at).strip()
        # ISO first
        try:
            return _dt.datetime.fromisoformat(at), None
        except Exception:
            pass
        # Forgiving fallbacks for weak models: 4pm / 4:30pm / 16:30
        for fmt in ("%I%p", "%I:%M%p", "%H:%M"):
            try:
                t = _dt.datetime.strptime(at.upper().replace(" ", ""), fmt).time()
                cand = now.replace(
                    hour=t.hour, minute=t.minute, second=0, microsecond=0
                )
                if cand <= now:  # already past today -> tomorrow
                    cand += _dt.timedelta(days=1)
                return cand, None
            except Exception:
                continue
        return None, (
            "Could not parse time; give in_minutes, or ISO like "
            "2026-07-10T16:00, or a clock time like '4pm'."
        )

    def execute(self, **kwargs) -> ToolResult:
        err = self.validate_parameters(kwargs)
        if err:
            return ToolResult(
                tool_name="reminders", success=False, result=None, error=err
            )
        action = (kwargs.get("action") or "").lower()
        try:
            if action == "add":
                text = kwargs.get("text")
                if not text:
                    return ToolResult(
                        tool_name="reminders",
                        success=False,
                        result=None,
                        error="add requires 'text'.",
                    )
                due, perr = self._parse_time(kwargs)
                if perr:
                    return ToolResult(
                        tool_name="reminders", success=False, result=None, error=perr
                    )
                r = self.store.add(text, due)
                return ToolResult(
                    tool_name="reminders",
                    success=True,
                    result=f"Set '{r.text}' for {r.due_at.strftime('%I:%M %p on %b %d')} "
                    f"(id {r.id}).",
                    metadata={"id": r.id},
                )
            if action == "list":
                pend = self.store.list_pending()
                if not pend:
                    return ToolResult(
                        tool_name="reminders",
                        success=True,
                        result="No pending reminders.",
                    )
                lines = [
                    f"{r.id}: {r.text} — {r.due_at.strftime('%I:%M %p %b %d')}"
                    for r in pend
                ]
                return ToolResult(
                    tool_name="reminders", success=True, result="\n".join(lines)
                )
            if action == "cancel":
                rid = kwargs.get("reminder_id")
                if not rid:
                    return ToolResult(
                        tool_name="reminders",
                        success=False,
                        result=None,
                        error="cancel requires 'reminder_id'.",
                    )
                ok = self.store.cancel(rid)
                return ToolResult(
                    tool_name="reminders",
                    success=ok,
                    result="Cancelled." if ok else None,
                    error=None if ok else f"No pending reminder matching '{rid}'.",
                )
            return ToolResult(
                tool_name="reminders",
                success=False,
                result=None,
                error=f"Unknown action '{action}' (add|list|cancel).",
            )
        except Exception as e:
            return ToolResult(
                tool_name="reminders", success=False, result=None, error=str(e)
            )


class AnimationTool(BaseTool):
    """Performs a vetted one-shot animation on the browser avatar.

    Server-side this only validates the clip name against the allowlist and
    records the request on the instance; the web session drains it into
    ``ChatResponse.animations`` and the frontend plays it through
    ``window.vrmAnimations``. Purely cosmetic — no filesystem/network reach —
    so it's safe in the default tool allowlist. Clip names must match the
    animation manifest's display names (static/animations/manifest.json)."""

    def __init__(self, clips: List[str]):
        self.clips = [str(clip) for clip in clips if str(clip).strip()]
        self.requested: List[str] = []
        super().__init__()

    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="play_animation",
            description=(
                "Perform a visible gesture, pose, or dance with your on-screen "
                "avatar body. Call this ONLY when the user explicitly asks you "
                "to perform one (dance, wave, jump, pose, clap, and so on). "
                "Your avatar already performs emotional body language "
                "automatically with every reply, so never call this to "
                "accompany an ordinary answer, greeting, thanks, or good news "
                "- that reads as repetitive and robotic. Available animations: "
                + ", ".join(self.clips)
            ),
            category=ToolCategory.COMMUNICATION,
            parameters=[
                ToolParameter(
                    name="name",
                    type="string",
                    description="Which animation to perform",
                    required=True,
                    enum=self.clips,
                )
            ],
            returns="Confirmation that the animation is playing on the avatar",
        )

    def execute(self, name: str = "", **kwargs) -> ToolResult:
        clip = str(name).strip()
        if clip not in self.clips:
            return ToolResult(
                tool_name="play_animation",
                success=False,
                result=None,
                error=(
                    f"Unknown animation '{clip}'. Available: " + ", ".join(self.clips)
                ),
            )
        self.requested.append(clip)
        return ToolResult(
            tool_name="play_animation",
            success=True,
            result=(
                f"'{clip}' is now playing on your avatar. Acknowledge it "
                "briefly in character; don't narrate the animation mechanics."
            ),
        )

    def pop_requested(self) -> List[str]:
        """Drain the animation requests recorded this turn."""
        drained, self.requested = self.requested, []
        return drained


class CalculatorTool(BaseTool):
    """Tool for mathematical calculations"""

    # The full syntax surface a calculator expression may use: literals,
    # arithmetic/comparison/boolean operators, list/tuple arguments, and
    # direct calls to names in the safe namespace. Everything else
    # (attributes, subscripts, comprehensions, lambdas, f-strings, ...) is
    # rejected before eval ever sees the expression.
    _ALLOWED_AST_NODES = (
        ast.Expression,
        ast.BinOp,
        ast.UnaryOp,
        ast.Compare,
        ast.BoolOp,
        ast.Call,
        ast.Constant,
        ast.Name,
        ast.List,
        ast.Tuple,
        ast.Load,
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.FloorDiv,
        ast.Mod,
        ast.Pow,
        ast.UAdd,
        ast.USub,
        ast.Eq,
        ast.NotEq,
        ast.Lt,
        ast.LtE,
        ast.Gt,
        ast.GtE,
        ast.And,
        ast.Or,
        ast.Not,
    )

    @classmethod
    def _validate_expression(cls, expression: str, safe_names) -> None:
        """Reject anything but plain arithmetic BEFORE eval sees it.

        ``eval`` with an empty ``__builtins__`` still allows dunder
        traversal (``().__class__.__mro__...`` reaches every loaded class),
        so the expression must first parse to an AST containing only
        whitelisted node types, names from the safe namespace, and direct
        positional calls to those names. Raises ValueError otherwise.
        """
        if "__" in expression:
            raise ValueError("double underscores are not allowed")
        tree = ast.parse(expression, mode="eval")
        for node in ast.walk(tree):
            if not isinstance(node, cls._ALLOWED_AST_NODES):
                raise ValueError(f"unsupported syntax: {type(node).__name__}")
            if isinstance(node, ast.Name) and node.id not in safe_names:
                raise ValueError(f"unknown name: {node.id}")
            if isinstance(node, ast.Call):
                if (
                    not isinstance(node.func, ast.Name)
                    or node.func.id not in safe_names
                ):
                    raise ValueError(
                        "only direct calls to known math functions are allowed"
                    )
                if node.keywords:
                    raise ValueError("keyword arguments are not allowed")
            if isinstance(node, ast.Constant) and not isinstance(
                node.value, (int, float, complex)
            ):
                raise ValueError(f"unsupported constant: {node.value!r}")

    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="calculator",
            description="Perform mathematical calculations and evaluate expressions",
            category=ToolCategory.COMPUTATION,
            parameters=[
                ToolParameter(
                    name="expression",
                    type="string",
                    description="Mathematical expression to evaluate (e.g., '2 + 2', 'sqrt(16)', 'log(100)')",
                    required=True,
                ),
            ],
            returns="Numerical result of the calculation",
            examples=[
                {"expression": "2 + 2"},
                {"expression": "sqrt(16) * 3.14"},
            ],
            cost_tier=1,
        )

    def execute(self, **kwargs) -> ToolResult:
        """Execute calculation"""
        validation_error = self.validate_parameters(kwargs)
        if validation_error:
            return ToolResult(
                tool_name="calculator",
                success=False,
                result=None,
                error=validation_error,
            )

        try:
            import math
            import numpy as np

            expression = kwargs["expression"]

            # Safe evaluation using restricted namespace
            safe_dict = {
                "abs": abs,
                "round": round,
                "min": min,
                "max": max,
                "sum": sum,
                "pow": pow,
                "sqrt": math.sqrt,
                "exp": math.exp,
                "log": math.log,
                "log10": math.log10,
                "log2": math.log2,
                "sin": math.sin,
                "cos": math.cos,
                "tan": math.tan,
                "asin": math.asin,
                "acos": math.acos,
                "atan": math.atan,
                "sinh": math.sinh,
                "cosh": math.cosh,
                "tanh": math.tanh,
                "pi": math.pi,
                "e": math.e,
                "ceil": math.ceil,
                "floor": math.floor,
                "factorial": math.factorial,
                "mean": np.mean,
                "median": np.median,
                "std": np.std,
            }

            # AST-whitelisted above; the restricted namespaces stay as
            # defense in depth.
            self._validate_expression(expression, safe_dict)
            result = eval(  # noqa: S307 - validated against the AST whitelist
                expression, {"__builtins__": {}}, safe_dict
            )

            return ToolResult(
                tool_name="calculator",
                success=True,
                result=result,
                metadata={"expression": expression},
            )
        except Exception as e:
            return ToolResult(
                tool_name="calculator",
                success=False,
                result=None,
                error=f"Calculation error: {str(e)}",
            )


class CodeInterpreterTool(BaseTool):
    """Executes Python code in a sandboxed SUBPROCESS (not in-process exec).

    Isolation measures:
      * ``sys.executable -I -c <code>`` — isolated mode (no user site dir, no
        cwd on sys.path, PYTHON* env vars ignored).
      * Fresh temporary working directory per call (removed afterwards).
      * Trimmed environment — a minimal whitelist; API keys and other parent
        env vars are never passed to the child.
      * Hard wall-clock timeout (kills the process on expiry).
      * ``resource.setrlimit`` (address space + CPU) via preexec_fn on POSIX.
      * stdout/stderr each capped at a few KB in the returned observation.
    """

    # Env vars the child may inherit; everything else (API keys!) is dropped.
    _ENV_WHITELIST = ("PATH", "SYSTEMROOT", "LANG", "LC_ALL", "TZ")
    _OUTPUT_CAP = 4096  # chars kept per stream
    _ADDRESS_SPACE_LIMIT = 1 << 30  # 1 GiB virtual memory

    def __init__(self, timeout_seconds: float = 10.0):
        self.timeout_seconds = float(timeout_seconds) or 10.0
        super().__init__()

    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="code_interpreter",
            description="Execute Python code in a sandboxed subprocess for data analysis and computation",
            category=ToolCategory.COMPUTATION,
            parameters=[
                ToolParameter(
                    name="code",
                    type="string",
                    description="Python code to execute",
                    required=True,
                ),
                ToolParameter(
                    name="timeout",
                    type="number",
                    description="Maximum execution time in seconds",
                    required=False,
                    default=10,
                ),
            ],
            returns="Execution stdout/stderr and exit status",
            examples=[{"code": "print(sum([1, 2, 3, 4, 5]) / 5)"}],
            cost_tier=2,
        )

    def _make_preexec_fn(self, cpu_seconds: int):
        """POSIX resource limits for the child; None where unavailable."""
        try:
            import resource
        except ImportError:  # pragma: no cover - Windows
            return None

        limit = self._ADDRESS_SPACE_LIMIT

        def _limits():  # pragma: no cover - runs in the forked child
            try:
                resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))
            except (ValueError, OSError):
                pass
            try:
                resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
            except (ValueError, OSError):
                pass

        return _limits

    @classmethod
    def _cap(cls, text: str) -> str:
        if text and len(text) > cls._OUTPUT_CAP:
            return text[: cls._OUTPUT_CAP] + " …[truncated]"
        return text or ""

    def execute(self, **kwargs) -> ToolResult:
        """Execute Python code in an isolated subprocess."""
        validation_error = self.validate_parameters(kwargs)
        if validation_error:
            return ToolResult(
                tool_name="code_interpreter",
                success=False,
                result=None,
                error=validation_error,
            )

        import os
        import shutil
        import subprocess
        import sys
        import tempfile

        code = kwargs["code"]
        requested = kwargs.get("timeout")
        timeout = self.timeout_seconds
        if requested:
            try:
                # The model may ask for less time, never more than configured.
                timeout = min(float(requested), self.timeout_seconds)
            except (TypeError, ValueError):
                pass
        timeout = max(timeout, 0.1)

        workdir = tempfile.mkdtemp(prefix="amadeus_code_")
        env = {key: os.environ[key] for key in self._ENV_WHITELIST if key in os.environ}
        env["PYTHONIOENCODING"] = "utf-8"
        env["TMPDIR"] = workdir

        try:
            proc = subprocess.run(
                [sys.executable, "-I", "-c", code],
                cwd=workdir,
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout,
                preexec_fn=self._make_preexec_fn(int(timeout) + 1),
            )
        except subprocess.TimeoutExpired as exc:
            return ToolResult(
                tool_name="code_interpreter",
                success=False,
                result={
                    "stdout": self._cap(
                        exc.stdout if isinstance(exc.stdout, str) else ""
                    ),
                    "stderr": self._cap(
                        exc.stderr if isinstance(exc.stderr, str) else ""
                    ),
                    "exit_code": None,
                },
                error=f"Execution timed out after {timeout:g}s and was killed",
            )
        except Exception as e:
            return ToolResult(
                tool_name="code_interpreter",
                success=False,
                result=None,
                error=f"Execution error: {str(e)}",
            )
        finally:
            shutil.rmtree(workdir, ignore_errors=True)

        payload = {
            "stdout": self._cap(proc.stdout) or "(no output)",
            "stderr": self._cap(proc.stderr),
            "exit_code": proc.returncode,
        }
        if proc.returncode != 0:
            return ToolResult(
                tool_name="code_interpreter",
                success=False,
                result=payload,
                error=f"Code exited with status {proc.returncode}: "
                f"{self._cap(proc.stderr).strip() or 'no error output'}",
            )
        return ToolResult(
            tool_name="code_interpreter",
            success=True,
            result=payload,
            metadata={"lines_of_code": len(code.splitlines())},
        )


def tool_allowed(name: str, allowed: Optional[List[str]]) -> bool:
    """Allowlist membership shared by the tool loop's firewall and every
    catalog/spec builder (advertising and permitting must never drift).

    ``"mcp"`` in the allowlist is a meta-entry admitting every bridged MCP
    tool: bridged names are config-derived (``mcp_<server>_<tool>``), so
    without it users would have to mirror each MCP server change into
    ``tools.allowed`` by hand — and a stale list would silently refuse
    tools the user explicitly configured a server for.
    """
    if allowed is None:
        return True
    if name in allowed:
        return True
    return name.startswith("mcp_") and "mcp" in allowed


class ToolRegistry:
    """
    Central registry for all available tools.

    Manages tool registration, discovery, and execution.
    """

    def __init__(self):
        self.tools: Dict[str, BaseTool] = {}
        # Tools that were requested (enabled in config) but could not be
        # registered, with a human-readable reason. Filled by the factory;
        # surfaced through /api/status.
        self.unavailable: List[Dict[str, str]] = []

    def register(self, tool: BaseTool) -> None:
        """Register a tool"""
        self.tools[tool.definition.name] = tool

    def unregister(self, tool_name: str) -> None:
        """Unregister a tool"""
        if tool_name in self.tools:
            del self.tools[tool_name]

    def get_tool(self, tool_name: str) -> Optional[BaseTool]:
        """Get a tool by name"""
        return self.tools.get(tool_name)

    def list_tools(
        self, category: Optional[ToolCategory] = None
    ) -> List[ToolDefinition]:
        """
        List all available tools.

        Args:
            category: Optional filter by category

        Returns:
            List of ToolDefinitions
        """
        definitions = [tool.definition for tool in self.tools.values()]

        if category:
            definitions = [d for d in definitions if d.category == category]

        return definitions

    def execute_tool(self, tool_call: ToolCall) -> ToolResult:
        """
        Execute a tool call.

        Args:
            tool_call: ToolCall with tool name and parameters

        Returns:
            ToolResult with execution outcome
        """
        tool = self.get_tool(tool_call.tool_name)

        if tool is None:
            return ToolResult(
                tool_name=tool_call.tool_name,
                success=False,
                result=None,
                error=f"Tool not found: {tool_call.tool_name}",
                call_id=tool_call.call_id,
            )

        # Central validation: individual tools already call
        # validate_parameters() at the top of execute(), but that's
        # per-tool self-discipline and at least one tool (AnimationTool)
        # has skipped it. Validating here too means a forgetful tool still
        # gets a clean ToolResult(success=False) instead of raising deep
        # inside execute() on a missing/mistyped parameter. Harmless
        # double-validation for tools that already do it themselves.
        validation_error = tool.validate_parameters(tool_call.parameters)
        if validation_error:
            return ToolResult(
                tool_name=tool_call.tool_name,
                success=False,
                result=None,
                error=validation_error,
                call_id=tool_call.call_id,
            )

        try:
            result = tool.execute(**tool_call.parameters)
            result.call_id = tool_call.call_id
            return result
        except Exception as e:
            return ToolResult(
                tool_name=tool_call.tool_name,
                success=False,
                result=None,
                error=f"Execution failed: {str(e)}",
                call_id=tool_call.call_id,
            )

    def get_tool_descriptions_for_llm(self, allowed: Optional[List[str]] = None) -> str:
        """
        Generate a formatted description of all tools for LLM context.

        ``allowed`` filters the catalog to those tool names (None = all):
        no point advertising a tool the caller's firewall would refuse.

        Returns:
            Markdown-formatted string describing available tools
        """
        descriptions = ["# Available Tools\n"]

        for tool in self.tools.values():
            if not tool_allowed(tool.definition.name, allowed):
                continue
            defn = tool.definition
            descriptions.append(f"\n## {defn.name}")
            descriptions.append(f"{defn.description}\n")
            descriptions.append("**Parameters:**")

            for param in defn.parameters:
                required = "required" if param.required else "optional"
                desc = (
                    f"- `{param.name}` ({param.type}, {required}): {param.description}"
                )
                if param.default is not None:
                    desc += f" (default: {param.default})"
                descriptions.append(desc)

            descriptions.append(f"\n**Returns:** {defn.returns}")

            if defn.examples:
                descriptions.append("\n**Examples:**")
                for example in defn.examples:
                    descriptions.append(
                        f"```json\n{json.dumps(example, indent=2)}\n```"
                    )

        return "\n".join(descriptions)

    def get_function_schemas(self) -> List[Dict[str, Any]]:
        """
        Get tool definitions in OpenAI function calling format.

        Returns:
            List of function schemas compatible with OpenAI API
        """
        schemas = []

        for tool in self.tools.values():
            defn = tool.definition

            parameters = {"type": "object", "properties": {}, "required": []}

            for param in defn.parameters:
                param_schema = {"type": param.type, "description": param.description}

                if param.enum:
                    param_schema["enum"] = param.enum

                parameters["properties"][param.name] = param_schema

                if param.required:
                    parameters["required"].append(param.name)

            schema = {
                "name": defn.name,
                "description": defn.description,
                "parameters": parameters,
            }

            schemas.append(schema)

        return schemas


def parse_tool_calls_from_text(text: str) -> List[ToolCall]:
    """
    Parse tool calls from LLM-generated text.

    Supports formats:
    1. JSON function calls: {"tool": "web_search", "params": {...}}
    2. Markdown code blocks with tool calls
    3. Natural language directives (basic parsing)

    Args:
        text: Text potentially containing tool call requests

    Returns:
        List of ToolCall objects
    """
    tool_calls = []

    # Pattern 1: JSON format
    # Look for JSON objects with "tool" and "params" keys
    json_pattern = (
        r'\{[^{}]*"tool"\s*:\s*"([^"]+)"[^{}]*"params"\s*:\s*\{[^}]*\}[^}]*\}'
    )
    json_matches = re.finditer(json_pattern, text, re.DOTALL)

    for match in json_matches:
        try:
            call_dict = json.loads(match.group())
            tool_calls.append(
                ToolCall(tool_name=call_dict["tool"], parameters=call_dict["params"])
            )
        except Exception:
            pass

    # Pattern 2: Markdown code blocks
    code_block_pattern = r'```(?:json)?\s*\n(\{[^`]*"tool"[^`]*\})\s*\n```'
    code_matches = re.finditer(code_block_pattern, text, re.DOTALL)

    for match in code_matches:
        try:
            call_dict = json.loads(match.group(1))
            tool_calls.append(
                ToolCall(tool_name=call_dict["tool"], parameters=call_dict["params"])
            )
        except Exception:
            pass

    # Pattern 3: Function call syntax
    # TOOL_NAME(param1=value1, param2=value2)
    func_pattern = r"([a-z_]+)\(([^)]+)\)"
    func_matches = re.finditer(func_pattern, text.lower())

    for match in func_matches:
        tool_name = match.group(1)
        params_str = match.group(2)

        # Parse parameters
        params = {}
        for param_pair in params_str.split(","):
            if "=" in param_pair:
                key, value = param_pair.split("=", 1)
                key = key.strip()
                value = value.strip().strip("\"'")
                params[key] = value

        if params:  # Only add if we successfully parsed parameters
            tool_calls.append(ToolCall(tool_name=tool_name, parameters=params))

    return tool_calls
