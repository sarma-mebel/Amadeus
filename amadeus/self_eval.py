"""Nightly conversation-quality self-check, default OFF (config.self_eval).

Runs a small subset of scripts/eval_conversation.py's scenarios against a
LIVE ChatEngine, exactly like the harness itself, but from the scheduler
(amadeus/web/scheduler.py) rather than by hand. The point is an unattended
early-warning: if a prompt/memory/persona change quietly degrades
conversation quality, the trend file shows a scenario dropping below its own
recent normal instead of the regression only surfacing when someone happens
to run `make eval` by hand.

Three layers, deliberately separated so each is independently testable
without a live LLM:

- ``rolling_median`` / ``check_regression`` — pure math over trend entries.
- ``read_trend_entries`` / ``append_trend_entry`` — trend-file I/O (JSON
  lines, same shape scripts/eval_conversation.py's --trend-file writes:
  timestamp, per-scenario overall scores, judge model, config fingerprint —
  provider/model NAMES only, never secrets).
- ``run_nightly_self_eval`` — the actual live-LLM run. Isolated tmp memory
  store per call (never the user's live store — same isolation
  scripts/eval_conversation.py uses for every scenario), built through
  ``_default_runner`` which imports scripts/eval_conversation.py lazily (it
  is not an installed package). The scheduler calls this from a worker
  thread (``asyncio.to_thread``), same pattern as the brief/diary/reach-out
  jobs' LLM calls.

Self-eval health (state/last-run summary/regression flag) is a flat
module-level record, mirroring ``provider_health.py`` — process-wide, not
per-session, self-healing is not applicable here (a regression is a fact
about the last run, not a transient error), so it's simply overwritten each
run and surfaced as ``self_eval`` on GET /api/status.
"""

from __future__ import annotations

import datetime as dt
import json
import statistics
import threading
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from . import telemetry
from .logging_utils import get_logger

LOGGER = get_logger(__name__)

# How many trailing trend entries feed the rolling median per scenario.
ROLLING_WINDOW = 7


# ---------------------------------------------------------------------------
# Pure regression math
# ---------------------------------------------------------------------------


def rolling_median(
    scores: List[float], window: int = ROLLING_WINDOW
) -> Optional[float]:
    """Median of the last ``window`` scores (most-recent-last order).

    None when there's nothing to compute a median over (empty input) — the
    caller treats that as "no baseline yet", not a regression.
    """
    if not scores:
        return None
    recent = scores[-window:]
    return statistics.median(recent)


def check_regression(
    entries: List[Dict[str, Any]],
    latest: Dict[str, float],
    *,
    tolerance: float = 2.0,
    window: int = ROLLING_WINDOW,
) -> Dict[str, Dict[str, float]]:
    """Per-scenario regression check: latest score vs the rolling median of
    ``entries`` (prior runs, oldest-first, each ``{"scores": {name: score}}``
    shaped like a trend-file line).

    Returns ``{scenario_name: {"latest": x, "median": y, "drop": y - x}}``
    for every scenario whose latest score fell more than ``tolerance`` below
    its rolling median. A scenario with fewer than 1 prior entry (no median
    yet) can never regress — nothing to compare against. Pure function, no
    I/O, no LLM calls.
    """
    regressions: Dict[str, Dict[str, float]] = {}
    for name, latest_score in latest.items():
        history = [
            float(entry["scores"][name])
            for entry in entries
            if isinstance(entry.get("scores"), dict) and name in entry["scores"]
        ]
        median = rolling_median(history, window=window)
        if median is None:
            continue
        drop = median - float(latest_score)
        if drop > tolerance:
            regressions[name] = {
                "latest": float(latest_score),
                "median": median,
                "drop": drop,
            }
    return regressions


# ---------------------------------------------------------------------------
# Trend file I/O
# ---------------------------------------------------------------------------


def read_trend_entries(path: "str | Path") -> List[Dict[str, Any]]:
    """All parseable JSON-line entries, oldest-first. Missing file -> [].
    A malformed line is skipped (never raises) — one corrupt line must not
    blind the regression check to every entry around it. Accepts a str or
    Path (config fields carry the former; callers building paths carry the
    latter) so this never trips a str-has-no-.exists() bug at either
    call site."""
    path = Path(path)
    if not path.exists():
        return []
    entries: List[Dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except OSError as exc:
        LOGGER.warning("Self-eval trend file read failed (%s): %s", path, exc)
        return []
    return entries


def append_trend_entry(path: "str | Path", entry: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry) + "\n")


def _parse_iso(stamp: str) -> Optional[dt.datetime]:
    """dt.datetime.fromisoformat, tolerant of a trailing 'Z' (UTC) —
    Python's stdlib only accepts '+00:00'-style offsets before 3.11.
    Trend entries written before 2026-08 carry 'Z'-suffixed UTC stamps
    (and record_run's status stamp still does); new trend entries are
    local-aware so the scheduler's local-date due-guard compares like
    with like (see last_run_date)."""
    text = stamp[:-1] + "+00:00" if stamp.endswith("Z") else stamp
    try:
        return dt.datetime.fromisoformat(text)
    except ValueError:
        return None


def last_run_date(entries: List[Dict[str, Any]]) -> Optional[str]:
    """LOCAL ISO date (not datetime) of the most recent trend entry, or None.

    The scheduler's due-guard compares this against ``now.date()`` from its
    LOCAL clock, so the date must be local too. Entries are stamped
    local-aware (run_nightly_self_eval); older 'Z'-suffixed UTC entries are
    converted rather than read verbatim — reading the raw UTC date in a
    UTC+8 zone made the 04:00 local run stamp *yesterday*, so every 60s tick
    until 08:00 local launched another full live-LLM eval run.
    ``astimezone()`` handles both: aware stamps convert to local, naive
    stamps are assumed local (matching the sibling daily jobs' convention).
    """
    if not entries:
        return None
    stamp = entries[-1].get("timestamp")
    if not stamp:
        return None
    parsed = _parse_iso(str(stamp))
    if parsed is None:
        return None
    try:
        return parsed.astimezone().date().isoformat()
    except (OSError, OverflowError, ValueError):  # absurd stamp: fail open
        return parsed.date().isoformat()


# ---------------------------------------------------------------------------
# Status record (mirrors provider_health.py's module-level pattern)
# ---------------------------------------------------------------------------

_LOCK = threading.Lock()
_STATUS: Dict[str, Any] = {
    "last_run_at": None,
    "scores": None,
    "regression": False,
    "regressions": None,
}


def record_run(
    *, scores: Dict[str, float], regressions: Dict[str, Dict[str, float]]
) -> None:
    with _LOCK:
        _STATUS["last_run_at"] = dt.datetime.utcnow().isoformat() + "Z"
        _STATUS["scores"] = dict(scores)
        _STATUS["regression"] = bool(regressions)
        _STATUS["regressions"] = dict(regressions) if regressions else None
    if regressions:
        telemetry.incr("self_eval.regressions")


def status_snapshot() -> Dict[str, Any]:
    """Copy of the current self-eval status, for GET /api/status's
    ``self_eval`` key. All-None/False is "hasn't run yet" — same "absent
    field renders nothing" contract the diagnostics UI already follows for
    llm_health/durations."""
    with _LOCK:
        return dict(_STATUS)


def reset() -> None:
    """Reset to the default not-yet-run state (tests only)."""
    with _LOCK:
        _STATUS["last_run_at"] = None
        _STATUS["scores"] = None
        _STATUS["regression"] = False
        _STATUS["regressions"] = None


# ---------------------------------------------------------------------------
# The live-LLM run itself
# ---------------------------------------------------------------------------

# Type of the pluggable runner: (config, llm, embedder, judge_llm,
# scenario_names, tmp_dir) -> {scenario_name: overall_score}. Swappable so
# scheduler-job tests can stub out every live call while still exercising
# the due/gating/append/regression-detection plumbing around it.
ScenarioRunner = Callable[..., Dict[str, float]]


def _default_runner(
    config: Any,
    llm: Any,
    embedder: Any,
    judge_llm: Any,
    scenario_names: List[str],
    scenarios_path: Path,
) -> Dict[str, float]:
    """Runs the requested scenarios via scripts/eval_conversation.py's own
    run_scenario/judge_transcript (not a separate reimplementation) so the
    nightly job and `make eval` score the exact same way. Imported lazily
    and by file path: the script is not an installed package (mirrors how
    scripts/eval_conversation.py itself is loaded in tests/test_eval_checks.py
    and tests/test_eval_harness.py)."""
    import importlib.util

    repo_root = Path(__file__).resolve().parent.parent
    spec = importlib.util.spec_from_file_location(
        "amadeus_self_eval_harness", repo_root / "scripts" / "eval_conversation.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]

    import yaml

    spec_data = yaml.safe_load(scenarios_path.read_text(encoding="utf-8"))
    all_scenarios = {s["name"]: s for s in spec_data.get("scenarios", [])}

    scores: Dict[str, float] = {}
    for name in scenario_names:
        scenario = all_scenarios.get(name)
        if scenario is None:
            LOGGER.warning("Self-eval: unknown scenario %r, skipping", name)
            continue
        result = module.run_scenario(scenario, config, llm, embedder)
        judge = module.judge_transcript(result, judge_llm, samples=1)
        scores[name] = float(judge.get("overall", 0))
    return scores


def run_nightly_self_eval(
    config: Any,
    *,
    runner: Optional[ScenarioRunner] = None,
) -> None:
    """Entry point for the scheduler job. Best-effort by contract: any
    failure counts ``self_eval.run_failed`` and returns — never raises,
    never crashes the scheduler tick.

    Builds an ISOLATED tmp memory store for the run (never the live user
    store) exactly like scripts/eval_conversation.py's run_scenario does,
    loads the real LLM/embedder/judge from the live config, runs the
    configured scenario subset, appends one trend-file line, and updates the
    module-level status record (including the regression flag GET
    /api/status surfaces).
    """
    cfg = getattr(config, "self_eval", None)
    if not cfg or not getattr(cfg, "enabled", False):
        return
    try:
        import tempfile

        from .provider_loader import load_embedder, load_llm

        trend_path = Path(cfg.trend_path)
        entries = read_trend_entries(trend_path)

        llm = load_llm(config)
        from .providers.llm.base import DummyProvider

        if isinstance(llm, DummyProvider):
            # Mirrors eval_conversation.py's own guard: a dummy LLM would
            # score nothing real, and every scenario would silently "pass"
            # by producing dummy text the checks happen not to forbid.
            LOGGER.warning("Self-eval: LLM resolved to dummy, skipping run.")
            telemetry.incr("self_eval.run_failed")
            return
        embedder = load_embedder(config)

        scenario_names = list(getattr(cfg, "scenarios", []) or [])[
            : max(0, int(getattr(cfg, "max_calls_per_day", 10) or 10))
        ]
        if not scenario_names:
            return

        scenarios_path = (
            Path(__file__).resolve().parent.parent / "scripts" / "eval_scenarios.yaml"
        )
        run = runner or _default_runner
        with tempfile.TemporaryDirectory(prefix="amadeus-self-eval-") as tmp:
            import dataclasses as _dc

            # dataclasses.replace() only shallow-copies: config.memory would
            # otherwise stay the SAME nested MemorySettings object shared
            # with the live config, so mutating persist_path here would
            # mutate the live config's memory settings too (config.memory is
            # a dataclass, replace() doesn't recurse into it). Replace the
            # nested dataclass itself first so the live config is never
            # touched — the isolation this function promises.
            scoped_memory = _dc.replace(
                config.memory,
                persist_path=str(Path(tmp) / "store.json"),
                embeddings_path=str(Path(tmp) / "vectors.json"),
                profile_path=str(Path(tmp) / "profile.json"),
            )
            scoped_config = _dc.replace(config, memory=scoped_memory)
            scores = run(
                scoped_config, llm, embedder, llm, scenario_names, scenarios_path
            )
        if not scores:
            telemetry.incr("self_eval.run_failed")
            return

        regressions = check_regression(
            entries,
            scores,
            tolerance=float(getattr(cfg, "regression_tolerance", 2.0) or 2.0),
        )
        entry = {
            # Local-aware (never bare utcnow+"Z"): the scheduler's once-a-day
            # guard compares last_run_date(entries) against its LOCAL clock's
            # date, so stamp and comparison must share a clock — same
            # convention as the sibling daily jobs (brief/diary stamp local).
            "timestamp": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
            "scores": scores,
            "judge_model": getattr(config, "llm_model", None),
            "config_fingerprint": {
                "llm_provider": getattr(config.providers, "llm", None),
                "llm_model": getattr(config, "llm_model", None),
                "embedding_provider": getattr(config.providers, "embedding", None),
            },
        }
        append_trend_entry(trend_path, entry)
        record_run(scores=scores, regressions=regressions)
        if regressions:
            LOGGER.warning("Self-eval regression detected: %s", regressions)
        else:
            LOGGER.info("Self-eval run complete: %s", scores)
    except Exception as exc:  # never break the scheduler
        LOGGER.warning("Self-eval run failed: %s", exc)
        telemetry.incr("self_eval.run_failed")
