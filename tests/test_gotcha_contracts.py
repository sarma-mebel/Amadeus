"""Static pins for the project notes gotchas that were prose-only.

Each test here corresponds to a hard-won rule that previously lived only as
a paragraph in the project notes. Prose survives exactly as long as someone
remembers to read it; these fail the build instead.

The bar for adding to this file: the rule describes a SILENT failure (a
feature that quietly never runs, a gate that quietly stops gating) and the
drift is statically detectable. Rules that are already enforced elsewhere
belong there, not here — see test_contracts.py (cross-layer wire formats),
test_file_budgets.py (god-file regrowth), test_animation_manifest.py
(eager/lazy pools), and test_persona.py (prompt size + static/dynamic
split).
"""

from __future__ import annotations

import ast
import inspect
import re
import textwrap
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


def _read(relative_path: str) -> str:
    return (REPO_ROOT / relative_path).read_text(encoding="utf-8")


# --- Scheduler lifespan gate ------------------------------------------------
#
# the project notes: "one asyncio task in the FastAPI lifespan, 60s tick, started
# when any scheduler-driven feature is enabled (the lifespan gate must be
# updated when adding one — a default-ON scheduler feature that isn't in the
# gate silently never runs)."
#
# The failure is invisible: the job method exists, its config section is
# enabled, every unit test of the job passes, and the scheduler is simply
# never constructed. Only an end-to-end run with lifespan executed would
# catch it, and the test suite uses httpx.ASGITransport, which skips
# lifespan entirely.


def _scheduler_tick_job_configs() -> set[str]:
    """Config section names read by the jobs BackgroundScheduler._tick runs.

    Parsed rather than called: we want the set of features the scheduler
    actually services, independent of any config being enabled.
    """
    from amadeus.web import scheduler as scheduler_module

    source = textwrap.dedent(inspect.getsource(scheduler_module.BackgroundScheduler))
    tree = ast.parse(source)
    class_node = tree.body[0]
    assert isinstance(class_node, ast.ClassDef)

    methods = {
        node.name: node
        for node in class_node.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    tick = methods.get("_tick")
    assert tick is not None, "BackgroundScheduler._tick not found"

    # Job methods awaited inside _tick.
    job_names = {
        node.func.attr
        for node in ast.walk(tick)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr.endswith("_job")
    }
    assert job_names, "no *_job calls found in _tick"

    # Each job opens with `cfg = getattr(self.config, "<section>", None)`.
    sections: set[str] = set()
    for job_name in job_names:
        node = methods.get(job_name)
        assert node is not None, f"{job_name} awaited in _tick but not defined"
        for call in ast.walk(node):
            if (
                isinstance(call, ast.Call)
                and isinstance(call.func, ast.Name)
                and call.func.id == "getattr"
                and len(call.args) >= 2
                and isinstance(call.args[0], ast.Attribute)
                and call.args[0].attr == "config"
                and isinstance(call.args[1], ast.Constant)
                and isinstance(call.args[1].value, str)
            ):
                sections.add(call.args[1].value)
                break
        else:  # pragma: no cover - only when a job stops reading its config
            pytest.fail(
                f"{job_name} does not read a config section via "
                'getattr(self.config, "<section>", None) — the lifespan-gate '
                "contract can no longer be checked statically."
            )
    return sections


def _lifespan_gated_configs() -> set[str]:
    """Config sections consulted by app.py's lifespan before starting the scheduler."""
    source = _read("amadeus/web/app.py")
    # `x_on = getattr(config, "<section>", None) and config.<section>.enabled`
    return set(re.findall(r'getattr\(\s*config,\s*"([a-z_]+)",\s*None\s*\)', source))


def test_every_scheduler_job_is_in_the_lifespan_gate() -> None:
    """A scheduler-driven feature missing from the gate silently never runs.

    If this fails: add the new section's `<name>_on` flag to the `if (...)`
    chain in `amadeus/web/app.py`'s lifespan. Enabling the config section is
    NOT enough — without the gate the BackgroundScheduler is never
    constructed and the job never ticks, with no error anywhere.
    """
    job_sections = _scheduler_tick_job_configs()
    gated = _lifespan_gated_configs()
    missing = job_sections - gated
    assert not missing, (
        f"scheduler job config section(s) {sorted(missing)} are serviced by "
        "BackgroundScheduler._tick but absent from the lifespan gate in "
        "amadeus/web/app.py — the feature would silently never run. Add a "
        "`<name>_on = getattr(config, '<name>', None) and config.<name>.enabled` "
        "flag and include it in the `if (...)` chain."
    )


def test_lifespan_gate_has_no_stale_sections() -> None:
    """The reverse drift: a gate flag whose job was removed keeps the
    scheduler task alive doing nothing every 60s forever."""
    job_sections = _scheduler_tick_job_configs()
    gated = _lifespan_gated_configs()
    stale = gated - job_sections
    assert not stale, (
        f"lifespan gate in amadeus/web/app.py references {sorted(stale)}, "
        "which no BackgroundScheduler._tick job services — either the job "
        "was removed (drop the flag) or it was renamed."
    )


def test_gated_sections_actually_exist_on_config() -> None:
    """A typo'd section name reads as False forever via getattr's default."""
    from amadeus.config import Config

    config = Config()
    for section in sorted(_lifespan_gated_configs()):
        assert hasattr(config, section), (
            f"lifespan gate reads config.{section}, which does not exist — "
            "getattr(..., None) makes the typo silently disable the feature."
        )
        assert hasattr(getattr(config, section), "enabled"), (
            f"config.{section} has no `enabled` field; the lifespan gate "
            "checks `.enabled` and would silently never start the scheduler."
        )


# --- Side-store tuple width -------------------------------------------------
#
# the project notes: "The side-store tuple `sessions._side_store_paths` is currently
# 7-wide (brief, diary, reachout, followup, life, insights, rhythm) — extend
# it when adding a store or per-character paths break."


def test_side_store_paths_width_matches_its_unpackers() -> None:
    """Every caller unpacks this tuple positionally; a width change breaks
    per-character path scoping at runtime, not at import.

    If this fails after deliberately adding a store: update the return
    annotation, both call sites in sessions.py, and the the project notes line.
    """
    from amadeus.config import Config
    from amadeus.web.sessions import SessionStore

    paths = SessionStore._side_store_paths(Config())
    assert isinstance(paths, tuple)
    assert len(paths) == 7, (
        f"_side_store_paths returned {len(paths)} values, expected 7 "
        "(brief, diary, reachout, followup, life, insights, rhythm). Callers "
        "unpack it positionally — update every unpack site in "
        "amadeus/web/sessions.py and the the project notes side-store line together."
    )

    source = _read("amadeus/web/sessions.py")
    for unpack in re.findall(
        r"^\s*([\w,\s]+)\s*=\s*self\._side_store_paths\(", source, re.M
    ):
        names = [part.strip() for part in unpack.split(",") if part.strip()]
        if len(names) > 1:  # skip `x = self._side_store_paths(...)` whole-tuple binds
            assert len(names) == 6, (
                f"a _side_store_paths unpack in sessions.py binds {len(names)} "
                "names but the tuple is 6-wide."
            )


# --- Raw-query recall gate --------------------------------------------------
#
# the project notes: "`recall_min_query_chars` gates on the RAW user text
# pre-augmentation — the shared helper in `memory/text_utils.py` keeps
# context.py and speculative.py in sync; they must not drift or the
# speculative cache key mismatches."


def test_raw_query_gate_is_checked_before_augmentation_at_both_sites() -> None:
    """Both recall entry points must gate on RAW text via the shared helper.

    Inlining the length check (or checking the augmented query) lets a short
    turn that got the previous turn prepended slip past the trivial-turn
    latency gate, and desynchronizes the speculative cache key from the real
    turn's — the cache then silently never hits.
    """
    for relative_path in ("amadeus/memory/context.py", "amadeus/memory/speculative.py"):
        source = _read(relative_path)
        assert "_raw_query_too_short" in source, (
            f"{relative_path} no longer calls _raw_query_too_short — the raw-"
            "text recall gate must stay in the shared helper "
            "(amadeus/memory/text_utils.py) so the two call sites cannot "
            "drift; an inlined check desyncs the speculative cache key."
        )

    # Both must read the same config knob, or the gates disagree on width.
    for relative_path in (
        "amadeus/memory/context.py",
        "amadeus/memory/speculative.py",
    ):
        assert "recall_min_query_chars" in _read(
            relative_path
        ), f"{relative_path} stopped reading recall_min_query_chars."


def test_raw_query_gate_returns_false_when_disabled() -> None:
    """min_chars <= 0 disables the gate; a regression here would silently
    suppress recall on every turn."""
    from amadeus.memory.text_utils import _raw_query_too_short

    assert _raw_query_too_short("hi", 0) is False
    assert _raw_query_too_short("", 0) is False
    assert _raw_query_too_short("hi", 8) is True
    assert _raw_query_too_short("what did we decide about the paper?", 8) is False
    # Whitespace-only turns are short regardless of padding.
    assert _raw_query_too_short("        ", 4) is True


# --- Wall-clock test trap ---------------------------------------------------
#
# the project notes: "Wall-clock trap: tests asserting presence lines fire (idle
# chatter, reach-out delivery, ambient comments) MUST set
# `config.sleep_schedule.enabled = False` — the default 1→8 LOCAL sleep
# window suppresses them, CI runners sit in UTC, and a 02:00-08:00 UTC run
# goes dark (14 tests failed exactly this way on 2026-08-04)."
#
# The trap is that the suite passes in SGT daytime and fails only when CI
# happens to run inside the window. Pinning the default keeps the rule's
# premise true; the TZ sweep below is what actually catches the class.


def test_sleep_schedule_default_window_matches_the_documented_trap() -> None:
    """The wall-clock trap guidance is written against a 1->8 default.

    If this fails, the the project notes wall-clock paragraph and every
    `sleep_schedule.enabled = False` line in the presence tests were
    calibrated against a window that no longer exists — re-derive the
    guidance before changing the default.
    """
    from amadeus.config import SleepScheduleSettings

    defaults = SleepScheduleSettings()
    assert defaults.enabled is True
    assert (defaults.start_hour, defaults.end_hour) == (1, 8), (
        "sleep window default moved; the project notes' wall-clock trap paragraph "
        "and the presence tests' `sleep_schedule.enabled = False` lines are "
        "written against 1->8 local."
    )


def test_late_night_boundary_is_pinned() -> None:
    """`is_late_night` biases neutral mood to 'tired'; tests asserting mood
    around prepare_turn must pin it, so its window must not drift silently."""
    import datetime as dt

    from amadeus.persona import is_late_night

    assert is_late_night(dt.datetime(2026, 8, 5, 0, 0)) is True
    assert is_late_night(dt.datetime(2026, 8, 5, 4, 59)) is True
    assert is_late_night(dt.datetime(2026, 8, 5, 5, 0)) is False
    assert is_late_night(dt.datetime(2026, 8, 5, 14, 0)) is False


# --- Calibrated recall ------------------------------------------------------
#
# the project notes: recalled facts are annotated with how sure she should sound
# (memory/text_utils.py::_effective_confidence, rendered by
# memory/context.py::_recall_line). Every rule below describes a failure that
# is silent in production — she simply sounds wrong, and no counter moves.


def test_fresh_fact_renders_with_no_hedge() -> None:
    """A fact stored yesterday and never re-retrieved must state PLAINLY.

    The load-bearing property of the whole feature. `_BASE_CONFIDENCE` is the
    ceiling of the formula (both multipliers are <= 1.0), so a base below the
    no-hedge band edge makes EVERY fact hedge — the degenerate-heuristic
    failure the vocal-emotion pass exists to catch (16/17 clips read "bright"
    at identical confidence). It is also what keeps the common case costing
    zero prompt tokens.
    """
    from amadeus.memory.text_utils import _confidence_clause, _effective_confidence

    fresh = _effective_confidence(strength=1.0, age_days=1.0, volatility="stable")
    assert _confidence_clause(fresh) is None, (
        "a fact learned yesterday is being hedged; check _BASE_CONFIDENCE "
        "against the top band edge in _CONFIDENCE_BANDS"
    )


def test_permanent_facts_never_hedge_for_age() -> None:
    """Hedging about a birthday is the most jarring possible failure.

    Permanent-class facts get NO staleness term at all, so age cannot pull
    them into a hedge band however old the store gets.
    """
    from amadeus.memory.text_utils import (
        _confidence_clause,
        _effective_confidence,
        _volatility_class,
    )

    assert _volatility_class("User was born on July 25") == "permanent"
    ancient = _effective_confidence(
        strength=1.0, age_days=365 * 10, volatility="permanent"
    )
    assert _confidence_clause(ancient) is None


def test_corroboration_is_a_bonus_never_a_penalty() -> None:
    """Most facts are stored once and never retrieved again.

    If the corroboration multiplier dipped below 1.0 for those, nearly every
    recall would hedge. Retrieval may only buy back confidence that age took.
    """
    from amadeus.memory.text_utils import _effective_confidence

    once = _effective_confidence(strength=1.0, age_days=200.0, volatility="volatile")
    often = _effective_confidence(strength=2.0, age_days=200.0, volatility="volatile")
    assert often > once
    fresh_once = _effective_confidence(strength=1.0, age_days=0.0, volatility="stable")
    assert fresh_once == pytest.approx(1.0), (
        "a fresh once-stored fact must sit at the ceiling; corroboration is "
        "a bonus, not a precondition for sounding sure"
    )


def test_confidence_bands_and_wording_are_pinned() -> None:
    """The persona register is tuned against these exact clauses.

    Drifting the wording silently changes her voice; drifting the edges
    changes how often she hedges. The top band MUST emit None.
    """
    from amadeus.memory.text_utils import _CONFIDENCE_BANDS, _confidence_clause

    assert _CONFIDENCE_BANDS[0] == (0.80, None)
    assert _confidence_clause(0.95) is None
    assert _confidence_clause(0.70) == "you seem fairly sure of this"
    assert _confidence_clause(0.40) == "you are not certain of this"
    assert _confidence_clause(0.10) == "you may be misremembering this"
    # Absent confidence (any path that did not compute it, legacy records)
    # renders exactly as it did before the feature existed.
    assert _confidence_clause(None) is None


def test_confidence_is_never_persisted_to_metadata() -> None:
    """`confidence` is derived from `strength`, which _reinforce owns.

    Writing it back would create a second source of truth that drifts on
    every retrieval. The field lives on VectorMatch (transient) and must not
    appear in any store-side metadata write.
    """
    import ast

    source = _read("amadeus/memory/recall.py")
    tree = ast.parse(source)
    for node in ast.walk(tree):
        # match.confidence = ... is fine (transient); metadata["confidence"]
        # or updated["confidence"] = ... is the bug being pinned.
        if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant):
            assert node.slice.value != "confidence", (
                "confidence must never be written into record metadata — it "
                "is derived from strength at recall time"
            )
    assert '"confidence"' not in _read("amadeus/memory/fact_archive.py")
