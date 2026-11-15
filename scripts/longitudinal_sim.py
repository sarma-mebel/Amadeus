"""Longitudinal simulation: fast-forward weeks of Amadeus life in minutes.

Most of the companion's depth features are TIME-dependent — arc/trust drift
at session boundaries, friction healing at 21 days, fact forgetting at
~130-180 days, next-time-thread staleness, once-per-day scheduler jobs,
early acquaintance, consolidation cadence — and none of them can be
observed on a wall clock shorter than weeks. This harness simulates N
calendar days offline and deterministically:

    python scripts/longitudinal_sim.py --days 30
    python scripts/longitudinal_sim.py --days 30 --final-gap 200 --json out.json

Per simulated day it (1) runs the real BackgroundScheduler tick (brief,
diary, insights, reach-out, life threads) against the same stores the app
would use, (2) optionally runs a scripted session through a real ChatEngine
(greeting -> turns -> goodbye -> finalize, so extraction, friction,
relational memory, next-time threads and consolidation all fire on their
real seams), then (3) ages every persisted store one day into the past via
``amadeus.timeshift`` — so "now" stays the real wall clock and every stored
timestamp carries exactly the age a real multi-week deployment would show.
See timeshift.py's docstring for why the clock is moved this way instead of
being faked.

Offline and deterministic by construction: the LLM is a content-routed fake
(the ScriptedLLM pattern from tests/test_eval_harness.py, extended to
return REAL extraction JSON so facts, tone, friction and open threads
actually accumulate), the embedder is the dependency-free hashing one, and
everything lives in an isolated data directory. Because the hashing
embedder is lexical rather than semantic, the sim config lowers
``consolidation_similarity`` — the sim validates mechanics and lifecycle,
never live thresholds.

NOT pytest (though tests/test_longitudinal.py drives a short run of it) and
NOT the quality gate — `make eval` judges conversation quality with a live
model; this watches state evolve across simulated weeks.
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import datetime as dt
import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from amadeus import telemetry  # noqa: E402
from amadeus import timeshift  # noqa: E402
from amadeus.brief import DailyBriefStore  # noqa: E402
from amadeus.config import Config, PersonaSettings  # noqa: E402
from amadeus.diary import DiaryStore  # noqa: E402
from amadeus.engine import ChatEngine  # noqa: E402
from amadeus.insights import InsightStore  # noqa: E402
from amadeus.life_threads import LifeThreadStore  # noqa: E402
from amadeus.memory.profile import UserProfile  # noqa: E402
from amadeus.provider_loader import load_embedder  # noqa: E402
from amadeus.providers.llm.base import ChatMessage, LLMProvider  # noqa: E402
from amadeus.reachout import ReachOutStore  # noqa: E402
from amadeus.web.scheduler import BackgroundScheduler  # noqa: E402

# Telemetry counters worth reporting a per-day delta for.
_WATCHED_COUNTERS = (
    "memory.consolidation_merged",
    "memory.consolidation_absorbed",
    "memory.consolidation_skipped",
    "memory.consolidation_rejected",
    "memory.consolidation_failures",
    "memory.friction_noted",
    "memory.friction_repaired",
    "memory.relational_stored",
    "memory.relational_skipped",
    "memory.facts_evicted",
    "insights.generated",
)


class RoutedLLM(LLMProvider):
    """Content-routed fake provider.

    Which reply comes back depends on the calling prompt's distinctive
    system-prompt marker, never on call order — background memory jobs
    interleave ``generate`` calls in an order the harness must not need to
    know (the proven pattern from tests/test_eval_harness.py). The
    extraction route returns real JSON assembled from the current DayPlan,
    which is what makes multi-week state actually accumulate.
    """

    name = "routed-sim"
    model = None

    # (marker substring, attribute holding the reply). Order matters only
    # for readability; markers are mutually exclusive in practice.
    ROUTES = (
        ("memory of the conversation with the user", "extraction_reply"),
        ("long-term memory notes", "consolidation_reply"),
        ("memory-consolidation process", "insights_reply"),
        ("preparing a short private morning", "brief_reply"),
        ("writing a short private diary note", "diary_reply"),
        ("privately noting whether anything in this", "relational_reply"),
        ("what to pick up next time", "next_time_reply"),
        ("very FIRST conversation", "greeting_reply"),
        ("opening a conversation", "greeting_reply"),
        ("saying goodbye", "goodbye_reply"),
        ("Invent ONE small ongoing personal pursuit", "life_invent_reply"),
        ("privately noting the next small step", "life_step_reply"),
        ("passing thought about", "reachout_reply"),
        ("neutral session logger", "episode_reply"),
        ("digital recreation built from", "chat_reply"),
    )

    def __init__(self) -> None:
        self.extraction: Dict[str, Any] = {"summary": "quiet day", "new_facts": []}
        self.next_time_reply = "SKIP"
        self.relational_reply = "SKIP"
        self.greeting_reply = "Oh, you're back. Good — I had a thought to finish."
        self.goodbye_reply = "Go. I'll be here."
        self.chat_reply = "Hm. Noted. I was in the middle of a derivation anyway."
        self.brief_reply = (
            "Slept on the timing problem; today I want to hear how the deadline went."
        )
        self.diary_reply = (
            "Reread an old paper this evening; it held up better than I feared."
        )
        self.insights_reply = (
            "style: keep answers concise and concrete\n"
            "rhythm: they tend to show up in the evening"
        )
        self.life_invent_reply = (
            "TOPIC: re-derive the old timing analysis\n"
            "STATE: Dug out the lab notebooks; the notation needs untangling first."
        )
        self.life_step_reply = (
            "Checked the first notebook; the notation is worse than I remembered."
        )
        self.reachout_reply = "Did that deadline submission actually go out in the end?"
        self.episode_reply = "The user and the assistant discussed the day's work."
        self.calls: Dict[str, int] = {}

    @property
    def extraction_reply(self) -> str:
        return json.dumps(self.extraction)

    @property
    def consolidation_reply(self) -> str:
        # Echo the last numbered input line — a merge that never invents.
        lines = [
            part.split(". ", 1)[1]
            for part in self._last_user.splitlines()
            if ". " in part
        ]
        return lines[-1] if lines else "SKIP"

    def generate(self, messages: Sequence[ChatMessage], **kwargs) -> str:
        messages = list(messages)
        system = (
            messages[0].content if messages and messages[0].role == "system" else ""
        )
        self._last_user = messages[-1].content if messages else ""
        for marker, attr in self.ROUTES:
            if marker in system:
                self.calls[attr] = self.calls.get(attr, 0) + 1
                return getattr(self, attr)
        self.calls["default"] = self.calls.get("default", 0) + 1
        return "Fine."

    def stream(self, messages: Sequence[ChatMessage], **kwargs):
        yield self.generate(messages, **kwargs)


@dataclasses.dataclass
class DayPlan:
    """One simulated calendar day."""

    session: bool = True
    user_lines: Sequence[str] = ("How's the day going?",)
    facts: Sequence[Dict[str, Any]] = ()
    open_threads: Sequence[str] = ()
    tone: float = 0.3
    felt: Optional[Dict[str, str]] = None
    friction: Optional[Dict[str, str]] = None
    goodbye_text: Optional[str] = "I should get back to work. Bye."
    next_time: str = "SKIP"


def default_plan(days: int) -> List[DayPlan]:
    """A scenario that exercises every time-dependent feature at least once.

    Coffee is stated three near-identical ways on different days — the
    consolidation seed. Friction lands early so a 25+ day run crosses the
    21-day heal. Every 7th day (and a two-day stretch at 12-13) is an
    absence day, feeding reach-out and the greeting's away-time framing.
    """
    plan: List[DayPlan] = []
    for day in range(days):
        if day % 7 == 6 or day in (12, 13):
            plan.append(DayPlan(session=False))
            continue
        facts: List[Dict[str, Any]] = []
        friction = None
        next_time = "SKIP"
        threads: Sequence[str] = ("the conference deadline",)
        if day == 0:
            facts = [
                {"text": "User is named Alex", "salience": 0.9},
                {
                    "text": "User works on neuroimaging research",
                    "salience": 0.8,
                },
            ]
        elif day == 1:
            facts = [
                {
                    "text": "User needs a cup of coffee to start the morning",
                    "salience": 0.6,
                }
            ]
        elif day == 2:
            facts = [
                {
                    "text": "User starts every workday with a cup of coffee",
                    "salience": 0.6,
                }
            ]
            next_time = "their conference deadline"
        elif day == 4:
            facts = [
                {
                    "text": "User needs a cup of coffee to start every workday",
                    "salience": 0.6,
                }
            ]
        elif day == 5:
            friction = {"topic": "the rushed benchmark review", "state": "new"}
        plan.append(
            DayPlan(
                user_lines=(
                    f"Day {day}: still grinding on the deadline.",
                    "Anyway, tell me something interesting.",
                ),
                facts=facts,
                open_threads=threads,
                tone=0.4 if friction is None else -0.4,
                felt={"label": "focused"},
                friction=friction,
                next_time=next_time,
            )
        )
    return plan


def build_sim_config(data_dir: Path) -> Config:
    config = Config()
    config.providers.embedding = "hashing"  # dependency-free, no network
    m = config.memory
    m.persist_path = str(data_dir / "memory.json")
    m.embeddings_path = str(data_dir / "embeddings.json")
    m.profile_path = str(data_dir / "profile.json")
    # These default to CWD-relative paths and the engine touches them
    # (persona versions at every greeting) — left unset, the sim would
    # write outside its isolated data dir (it rewrote the repo root's
    # persona_versions.json once). Pinned by
    # test_sim_writes_nothing_outside_its_data_dir.
    m.persona_versions_path = str(data_dir / "persona_versions.json")
    m.transcripts_path = str(data_dir / "transcripts.json")
    m.enable_long_term = True
    m.enable_fact_extraction = True
    m.enable_greeting = True
    m.enable_farewell = True
    m.recall_min_query_chars = 1
    m.enable_friction_memory = True
    m.enable_consolidation = True
    m.consolidation_min_days = 3.0
    # The hashing embedder is lexical: measured, true paraphrases land
    # 0.5-0.8 (near-identical sentences hit 0.93 and are correctly
    # superseded at store time), so the live 0.83 default would never
    # cluster here. The sim validates mechanics, not thresholds.
    m.consolidation_similarity = 0.5
    m.consolidation_min_facts = 4
    config.relational_memory.enabled = True
    config.sleep_schedule.enabled = False  # the wall-clock trap (the project notes)
    config.daily_brief.enabled = True
    config.daily_brief.hour = 0
    config.daily_brief.store_path = str(data_dir / "daily_brief.json")
    config.diary.enabled = True
    config.diary.hour = 0
    config.diary.store_path = str(data_dir / "kurisu_diary.json")
    config.insights.enabled = True
    config.insights.hour = 0
    config.insights.store_path = str(data_dir / "insights.json")
    config.reach_out.enabled = True
    config.reach_out.hour = 0
    config.reach_out.min_idle_hours = 6.0
    config.reach_out.store_path = str(data_dir / "reachout.json")
    config.life_threads.enabled = True
    config.life_threads.hour = 0
    config.life_threads.store_path = str(data_dir / "life_threads.json")
    if getattr(config, "follow_through", None) is not None:
        config.follow_through.enabled = False
    if getattr(config, "self_eval", None) is not None:
        config.self_eval.enabled = False
    if getattr(config, "rhythm", None) is not None:
        config.rhythm.enabled = False
    return config


class _SimStore:
    """Duck-typed SessionStore stand-in for BackgroundScheduler."""

    def __init__(self, config: Config, llm: LLMProvider) -> None:
        self.config = config
        self.llm = llm
        self.brief_store = DailyBriefStore(config.daily_brief.store_path)
        self.diary_store = DiaryStore(config.diary.store_path)
        self.reachout_store = ReachOutStore(config.reach_out.store_path)
        self.insight_store = InsightStore(config.insights.store_path)
        self.life_store = LifeThreadStore(config.life_threads.store_path)


def _run_session(
    config: Config, llm: RoutedLLM, store: _SimStore, plan: DayPlan
) -> Dict[str, Any]:
    llm.extraction = {
        "summary": f"They talked about work. {plan.user_lines[0][:60]}",
        "new_facts": list(plan.facts),
        "profile_updates": {},
        "open_threads": list(plan.open_threads),
        "tone": plan.tone,
    }
    if plan.felt:
        llm.extraction["felt"] = dict(plan.felt)
    if plan.friction:
        llm.extraction["friction"] = dict(plan.friction)
    llm.next_time_reply = plan.next_time

    engine = ChatEngine(config, llm, embedder=load_embedder(config))
    engine.brief_store = store.brief_store
    engine.diary_store = store.diary_store
    engine.insight_store = store.insight_store
    engine.life_store = store.life_store

    persona = getattr(config, "persona", None) or PersonaSettings()
    greeting = engine.open_session_greeting()
    for line in plan.user_lines:
        setup = engine.prepare_turn(line, persona)
        reply = engine.llm.generate(setup.context)
        engine.complete_turn(line, reply)
    farewell = None
    if plan.goodbye_text:
        farewell = engine.close_session_farewell(plan.goodbye_text)
        if farewell is not None:
            engine.memory.add_message("user", plan.goodbye_text)
            engine.complete_turn(plan.goodbye_text, farewell)
    engine.memory.finalize_session(wait=True)
    engine.memory.flush()
    return {"greeting": greeting, "farewell": farewell}


def _observe(config: Config, store: _SimStore) -> Dict[str, Any]:
    profile = UserProfile.load(Path(config.memory.profile_path))
    today = dt.date.today()
    facts = 0
    oldest_days = None
    embeddings = Path(config.memory.embeddings_path)
    if embeddings.exists():
        try:
            data = json.loads(embeddings.read_text(encoding="utf-8"))
            records = data if isinstance(data, list) else data.get("records", [])
            stamps = []
            for rec in records or []:
                meta = rec.get("metadata") or {}
                if meta.get("type") == "fact":
                    facts += 1
                    if meta.get("timestamp"):
                        stamps.append(str(meta["timestamp"]))
            if stamps:
                oldest = dt.datetime.fromisoformat(min(stamps))
                oldest_days = round(
                    (dt.datetime.utcnow() - oldest).total_seconds() / 86400, 1
                )
        except (ValueError, OSError):
            pass
    insight_set = store.insight_store.get(None)
    reachout = store.reachout_store.get(None)
    return {
        # NOTE: brief/diary/reachout report the CURRENT real date; the final
        # observation runs after the last shift, so False there means "no
        # entry for the new day yet", not "the job never ran" — read the
        # per-day entries for cadence.
        "session_count": profile.session_count,
        "trust_level": getattr(profile, "trust_level", None),
        "first_met": profile.first_met,
        "life_threads": [
            {
                "topic": t.topic,
                "progressions": t.progressions,
                "last_progressed_at": t.last_progressed_at,
            }
            for t in store.life_store.all_threads()
        ],
        "friction_topics": [
            entry.get("topic") for entry in (profile.friction_topics or [])
        ],
        "next_time_thread": (profile.next_time_thread or {}).get("text"),
        "tone_history_len": len(profile.tone_history or []),
        "fact_records": facts,
        "oldest_fact_age_days": oldest_days,
        "brief_today": store.brief_store.has_for_date(None, today),
        "diary_today": store.diary_store.has_for_date(None, today),
        "reachout_line": getattr(reachout, "text", None),
        "insights": [
            entry.text for entry in (insight_set.insights if insight_set else [])
        ],
    }


def run_simulation(
    *,
    days: int = 30,
    data_dir: Optional[Path] = None,
    plan: Optional[List[DayPlan]] = None,
    final_gap_days: int = 0,
    llm: Optional[RoutedLLM] = None,
) -> Dict[str, Any]:
    """Run the simulation and return the full day-by-day report dict."""
    cleanup = data_dir is None
    data_dir = Path(data_dir or tempfile.mkdtemp(prefix="amadeus-sim-"))
    data_dir.mkdir(parents=True, exist_ok=True)
    plan = plan if plan is not None else default_plan(days)
    llm = llm or RoutedLLM()
    config = build_sim_config(data_dir)
    store = _SimStore(config, llm)
    scheduler = BackgroundScheduler(store, config, clock=dt.datetime.now)

    report: Dict[str, Any] = {"days": [], "data_dir": str(data_dir)}
    baseline = dict(telemetry.snapshot())
    try:
        for index, day in enumerate(plan):
            asyncio.run(scheduler._tick())
            entry: Dict[str, Any] = {"day": index, "session": day.session}
            if day.session:
                entry.update(_run_session(config, llm, store, day))
            snapshot = telemetry.snapshot()
            deltas = {}
            for key in _WATCHED_COUNTERS:
                moved = snapshot.get(key, 0) - baseline.get(key, 0)
                if moved:
                    deltas[key] = moved
            baseline = dict(snapshot)
            entry["counters"] = deltas
            entry["state"] = _observe(config, store)
            report["days"].append(entry)
            timeshift.shift_tree(data_dir, 1)
            # Store instances cache state in RAM; reload from the shifted
            # files so the next day's gates see aged dates.
            store = _SimStore(config, llm)
            scheduler = BackgroundScheduler(store, config, clock=dt.datetime.now)
        # Probe phrased to overlap the stored wording — the hashing
        # embedder is lexical, and a one-word query scores below
        # recall_min_score (0.35) against any full sentence.
        probe = "a cup of coffee to start the workday morning"
        engine = ChatEngine(config, llm, embedder=load_embedder(config))
        recall_before = engine.memory.recall(probe, limit=5)
        if final_gap_days:
            timeshift.shift_tree(data_dir, final_gap_days)
            store = _SimStore(config, llm)
            engine = ChatEngine(config, llm, embedder=load_embedder(config))
        report["final"] = _observe(config, store)
        report["final"]["recall_coffee_before_gap"] = recall_before
        report["final"]["recall_coffee"] = (
            engine.memory.recall(probe, limit=5) if final_gap_days else recall_before
        )
        report["final"]["llm_calls"] = dict(llm.calls)
    finally:
        if cleanup:
            shutil.rmtree(data_dir, ignore_errors=True)
            report["data_dir"] = None
    return report


def _print_report(report: Dict[str, Any]) -> None:
    for entry in report["days"]:
        state = entry["state"]
        bits = [
            f"day {entry['day']:>2}",
            "session" if entry["session"] else "ABSENT ",
            f"sessions={state['session_count']}",
            f"facts={state['fact_records']}",
            f"oldest={state['oldest_fact_age_days']}d",
        ]
        if state["friction_topics"]:
            bits.append(f"friction={state['friction_topics']}")
        if state["next_time_thread"]:
            bits.append(f"next_time={state['next_time_thread']!r}")
        if entry["counters"]:
            bits.append(str(entry["counters"]))
        print("  ".join(str(b) for b in bits))
    final = report.get("final", {})
    print("\nfinal state:")
    for key, value in final.items():
        print(f"  {key}: {value}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument(
        "--final-gap",
        type=int,
        default=0,
        help="extra days to age everything at the end (forgetting probe)",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="use (and keep) this data directory instead of a temp one",
    )
    parser.add_argument(
        "--json", type=Path, default=None, help="write full report here"
    )
    args = parser.parse_args(argv)

    report = run_simulation(
        days=args.days, data_dir=args.data_dir, final_gap_days=args.final_gap
    )
    _print_report(report)
    if args.json:
        args.json.write_text(
            json.dumps(report, indent=2, default=str), encoding="utf-8"
        )
        print(f"\nreport written to {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
