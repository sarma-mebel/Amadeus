"""In-process background scheduler for the web app (daily brief).

A single asyncio task started from the FastAPI lifespan, ticking every
minute. Every job is individually exception-wrapped — a failing tick must
never kill the loop. Reminders deliberately have no job here: the poll path
(`POST /api/reminders/due`) owns delivery, and `pop_due` is destructive (it
marks reminders announced), so a server-side sweep with no connected client
would eat announcements.

The brief job reads DISK SNAPSHOTS (UserProfile.load, ReminderStore.
list_pending) rather than constructing a MemoryManager — a live manager
fires session-boundary bookkeeping (session_count/trust) on construction,
which a daily job must not inflate. Due-predicate: `now.time() >= hour AND
no brief stored for today` — that one check gives on-time firing, startup
catch-up, and midnight-crossing for free, with no back-fill of missed days
(a Tuesday brief delivered Thursday is worse than none).

Per-character isolation: every tick re-derives the ACTIVE character's scoped
paths from the live ``SessionStore.config`` (see the ``config`` property) —
profile reads via ``Config.character_memory_path``, brief/diary/reach-out
stores via the SessionStore attributes that ``reload_providers`` rebuilds on
a character switch. Reminders are the one deliberate exception: they stay
shared across characters (real-world timers, not persona memory).

The self-eval job (``self_eval.*``, default off) is the odd one out: it is
process-global (one conversation-quality check/day, not per user/character)
and it DOES make live LLM calls through a real (isolated tmp store)
ChatEngine, unlike every other job here which only reads disk snapshots.
Scheduling (due-predicate, once/day, exception isolation) lives here;
``amadeus/self_eval.py`` owns the run itself, the trend file, and the
regression math.
"""

from __future__ import annotations

import asyncio
import datetime as dt
from contextlib import suppress
from pathlib import Path
from typing import Any, Callable, List, Optional

from ..async_notify import (
    deliver,
    read_last_client_poll,
    read_last_delivery_date,
    record_delivery,
    should_attempt_delivery,
)
from ..brief import (
    DailyBrief,
    arxiv_topic_from_profile,
    collect_brief_facts,
    generate_daily_brief,
)
from ..calendar_feed import CalendarCache, is_configured, refresh_cache, upcoming_events
from ..diary import DiaryEntry, collect_diary_seeds, generate_diary_entry
from ..followup import (
    FollowUp,
    fresh_next_time_thread,
    generate_followup_note,
    plan_research,
    run_research,
)
from ..life_threads import (
    life_thread_seeds,
    progress_life_thread,
    seed_due,
    seed_life_thread,
    thread_due_for_progress,
)
from ..insights import (
    InsightSet,
    collect_insight_facts,
    generate_insights,
)
from ..logging_utils import get_logger
from ..memory.profile import UserProfile
from ..reachout import ReachOut, collect_reachout_seeds, generate_reach_out
from ..reminders import ReminderStore
from .. import self_eval as self_eval_module

LOGGER = get_logger(__name__)

# Users whose profile hasn't been touched in this long stop costing a daily
# LLM call; they resume the first time they show up again.
_IDLE_USER_CUTOFF_DAYS = 14

# Bounded retry for FAILED daily generations: a job whose LLM call failed
# used to retry every 60s tick for the rest of the day (a deterministic
# failure — e.g. a life-thread leak rejection — meant ~1000 paid calls/day
# against a documented cap of 1). After this many failures in one local day
# the job holds until tomorrow. In-memory only: a process restart resets the
# count, which is fine — the point is stopping the 60s hammer, not exactness.
_MAX_DAILY_JOB_FAILURES = 3

# Config sections whose scheduler job should trigger starting the scheduler.
# Mirrors the lifespan gate in web/app.py (which test_gotcha_contracts.py
# pins against BackgroundScheduler._tick); the settings-reload path uses
# this helper because it has no lifespan to re-run.
_SCHEDULER_FEATURE_SECTIONS = (
    "daily_brief",
    "diary",
    "reach_out",
    "follow_through",
    "life_threads",
    "insights",
    "self_eval",
    "async_notify",
)


def scheduler_features_enabled(config: Any) -> bool:
    """True when any scheduler-driven feature is enabled — the same
    predicate the app lifespan's start gate evaluates at boot. Used by the
    settings-reload path to start a scheduler that boot skipped (enabling a
    feature via settings used to silently never run it until restart)."""
    for section in _SCHEDULER_FEATURE_SECTIONS:
        cfg = getattr(config, section, None)
        if cfg is not None and getattr(cfg, "enabled", False):
            return True
    return False


class BackgroundScheduler:
    def __init__(
        self,
        store: Any,  # SessionStore (Any avoids a circular import)
        config: Any,
        *,
        tick_seconds: float = 60.0,
        clock: Callable[[], dt.datetime] = dt.datetime.now,
    ) -> None:
        self.store = store
        self._config = config
        self.tick_seconds = tick_seconds
        self.clock = clock
        self._task: Optional[asyncio.Task] = None
        self._in_flight: set = set()
        # Per-job-key ("brief:default", "life_threads", ...) failure ledger
        # for the bounded daily retry policy: {key: (iso_date, count)}.
        self._job_failures: dict = {}

    @property
    def config(self) -> Any:
        """The LIVE runtime config, re-read on every access.

        The scheduler is constructed once at app startup, but settings saves
        and character switches replace ``SessionStore.config`` with a brand
        new Config object (``reload_providers``) — the constructor argument
        goes stale immediately after the first such save. Reading through the
        store means every tick sees the ACTIVE character, so a mid-day
        character switch moves the brief/diary/reach-out onto the guest's
        scoped profile/stores by the next tick. The constructor config is
        only the fallback for stores without a ``config`` attribute (tests).
        """
        return getattr(self.store, "config", None) or self._config

    def _profile_base(self) -> Optional[Path]:
        """Character-scoped base profile path for the active character.

        Per-character isolation happens here, at point of use (never written
        back into config fields): with a guest character active the jobs
        must read the GUEST's profile snapshot — otherwise the brief/diary/
        reach-out would be generated from Kurisu's data and folded into the
        guest's greeting. Builtin Kurisu (or per_character_memory off) keeps
        the legacy unscoped path.
        """
        config = self.config
        base = getattr(config.memory, "profile_path", None)
        if not base:
            return None
        scope = getattr(config, "character_memory_path", None)
        if callable(scope):
            base = scope(base)
        return Path(base) if base else None

    def _profile_path(self, user_id: Optional[str]) -> Optional[Path]:
        base = self._profile_base()
        if base is None:
            return None
        if user_id:
            return base.parent / f"{base.stem}_users" / f"{user_id}{base.suffix}"
        return base

    def _load_profile(self, user_id: Optional[str], job: str) -> Optional[UserProfile]:
        """Disk-snapshot read of the active character's view of this user."""
        path = self._profile_path(user_id)
        if path is None:
            return None
        try:
            if path.exists():
                return UserProfile.load(path)
        except Exception as exc:
            LOGGER.warning("%s profile read failed for %s: %s", job, user_id, exc)
        return None

    def _character_name(self) -> str:
        """Full character name for the brief/diary prompts (BYO-character)."""
        char = getattr(self.config, "character", None)
        return (
            getattr(char, "full_name", None) or "Makise Kurisu"
        ).strip() or "Makise Kurisu"

    # --- Bounded daily retry for failed generations ----------------------
    #
    # Success paths self-limit (has_for_date / acted_today / has_for_thread),
    # but a FAILED generation used to leave no stamp at all, so the 60s tick
    # retried it — each retry a paid LLM call — for the rest of the day.
    # These two helpers give every daily job the same policy: up to
    # _MAX_DAILY_JOB_FAILURES attempts per local day, then hold until
    # tomorrow. Legitimate nothing-to-do outcomes (no seeds, no thread,
    # nothing worth briefing, keyless provider) never count as failures.

    def _job_failures_exhausted(self, key: str, today: dt.date) -> bool:
        date_iso, count = self._job_failures.get(key, ("", 0))
        return date_iso == today.isoformat() and count >= _MAX_DAILY_JOB_FAILURES

    def _note_job_failure(self, key: str, today: dt.date) -> None:
        from .. import telemetry

        date_iso, count = self._job_failures.get(key, ("", 0))
        count = count + 1 if date_iso == today.isoformat() else 1
        self._job_failures[key] = (today.isoformat(), count)
        telemetry.incr("scheduler.job_failures")
        if count >= _MAX_DAILY_JOB_FAILURES:
            telemetry.incr("scheduler.job_retries_exhausted")
            LOGGER.warning(
                "Scheduler job %s failed %d times today; holding until tomorrow",
                key,
                count,
            )

    async def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            with suppress(asyncio.CancelledError):
                await self._task
            self._task = None

    async def _run(self) -> None:
        while True:
            await self._tick()
            await asyncio.sleep(self.tick_seconds)

    async def _tick(self) -> None:
        # Each job is exception-wrapped (a failing job never kills the loop)
        # and every catch COUNTS (degrade loudly): a job dying on the same
        # exception every tick used to be one identical warning line in a
        # log nobody reads, invisible on /api/status.
        from .. import telemetry

        now = self.clock()
        try:
            await self._daily_brief_job(now)
        except Exception as exc:
            telemetry.incr("scheduler.brief_job_failed")
            LOGGER.warning("Scheduler brief job failed: %s", exc)
        try:
            await self._diary_job(now)
        except Exception as exc:
            telemetry.incr("scheduler.diary_job_failed")
            LOGGER.warning("Scheduler diary job failed: %s", exc)
        try:
            await self._reachout_job(now)
        except Exception as exc:
            telemetry.incr("scheduler.reachout_job_failed")
            LOGGER.warning("Scheduler reach-out job failed: %s", exc)
        try:
            await self._async_notify_job(now)
        except Exception as exc:
            telemetry.incr("scheduler.async_notify_job_failed")
            LOGGER.warning("Scheduler async-notify job failed: %s", exc)
        try:
            await self._followthrough_job(now)
        except Exception as exc:
            telemetry.incr("scheduler.followthrough_job_failed")
            LOGGER.warning("Scheduler follow-through job failed: %s", exc)
        try:
            await self._life_threads_job(now)
        except Exception as exc:
            telemetry.incr("scheduler.life_threads_job_failed")
            LOGGER.warning("Scheduler life-threads job failed: %s", exc)
        try:
            await self._insights_job(now)
        except Exception as exc:
            telemetry.incr("scheduler.insights_job_failed")
            LOGGER.warning("Scheduler insights job failed: %s", exc)
        try:
            await self._self_eval_job(now)
        except Exception as exc:
            telemetry.incr("scheduler.self_eval_job_failed")
            LOGGER.warning("Scheduler self-eval job failed: %s", exc)

    # --- Daily brief ----------------------------------------------------------

    async def _daily_brief_job(self, now: dt.datetime) -> None:
        cfg = getattr(self.config, "daily_brief", None)
        if not cfg or not getattr(cfg, "enabled", False):
            return
        brief_store = getattr(self.store, "brief_store", None)
        if brief_store is None:
            return
        if now.time() < dt.time(hour=max(0, min(23, int(cfg.hour)))):
            return
        for user_id in self._candidate_users():
            key = user_id or "default"
            if key in self._in_flight:
                continue
            if brief_store.has_for_date(user_id, now.date()):
                continue
            if self._job_failures_exhausted(f"brief:{key}", now.date()):
                continue
            self._in_flight.add(key)
            try:
                # Provider + httpx calls stay off the event loop.
                await asyncio.to_thread(self._generate_for_user, user_id, now)
            finally:
                self._in_flight.discard(key)

    def _candidate_users(self) -> List[Optional[str]]:
        """Users with a fresh-enough profile: the base profile (None) plus
        every per-user profile file, mirroring SessionStore._user_path's
        `<stem>_users/<id>.json` layout. Reads the ACTIVE character's scoped
        profile directory, so a guest character's jobs enumerate the guest's
        users (typically none until they've talked), never Kurisu's."""
        base_path = self._profile_base()
        if base_path is None:
            return []
        candidates: List[tuple[Optional[str], Path]] = []
        if base_path.exists():
            candidates.append((None, base_path))
        users_dir = base_path.parent / f"{base_path.stem}_users"
        if users_dir.is_dir():
            for entry in sorted(users_dir.glob("*.json")):
                candidates.append((entry.stem, entry))
        fresh: List[Optional[str]] = []
        cutoff = self.clock() - dt.timedelta(days=_IDLE_USER_CUTOFF_DAYS)
        for user_id, path in candidates:
            try:
                profile = UserProfile.load(path)
            except Exception:
                continue
            last_seen = None
            if profile.last_seen:
                try:
                    last_seen = dt.datetime.fromisoformat(profile.last_seen)
                except ValueError:
                    last_seen = None
            if last_seen is None or last_seen >= cutoff:
                fresh.append(user_id)
        return fresh

    def _generate_for_user(self, user_id: Optional[str], now: dt.datetime) -> None:
        cfg = self.config.daily_brief
        brief_store = self.store.brief_store

        profile = self._load_profile(user_id, "Brief")

        pending: list = []
        reminders_cfg = getattr(self.config, "reminders", None)
        if reminders_cfg and getattr(reminders_cfg, "enabled", False):
            try:
                # Read-only: list_pending never marks anything announced.
                # Reminders stay UNSCOPED on purpose (no character_memory_
                # path here): they are the user's real-world timers, not
                # persona memory — an appointment set while talking to
                # Kurisu must still make the brief while a guest character
                # is active.
                pending = ReminderStore(reminders_cfg.store_path).list_pending()
            except Exception as exc:
                LOGGER.warning("Brief reminder read failed: %s", exc)

        arxiv_hits: list = []
        if getattr(cfg, "include_literature", False) and getattr(
            getattr(self.config, "literature", None), "enabled", False
        ):
            topic = arxiv_topic_from_profile(profile)
            if topic:
                try:
                    from ..tools.literature import ArxivClient

                    arxiv_hits = ArxivClient().search(
                        topic, max_results=int(cfg.max_arxiv_results)
                    )
                except Exception as exc:
                    LOGGER.warning("Brief arXiv search failed: %s", exc)

        # Calendar stays UNSCOPED like reminders (no character_memory_path
        # here): it's the user's real-world schedule, not persona memory.
        # The refetch (network/disk IO) happens once per tick regardless of
        # how many candidate users there are — refresh_cache no-ops when the
        # TTL hasn't elapsed, so a multi-user install doesn't refetch per
        # user. Failures degrade to the stale (or empty) cache; never raise.
        calendar_events: list = []
        calendar_cfg = getattr(self.config, "calendar", None)
        if calendar_cfg and is_configured(calendar_cfg):
            try:
                cache = CalendarCache(calendar_cfg.cache_path)
                refresh_cache(calendar_cfg, cache, now)
                calendar_events = upcoming_events(calendar_cfg, now)
            except Exception as exc:
                LOGGER.warning("Brief calendar read failed: %s", exc)

        facts = collect_brief_facts(
            profile=profile,
            pending_reminders=pending,
            arxiv_hits=arxiv_hits,
            now=now,
            calendar_events=calendar_events,
        )
        if len(facts) <= 1:
            # Only the timestamp fact: nothing worth briefing (mirrors
            # generate_daily_brief's own bail-out). NOT a failure — a
            # reminder set later today makes the next tick's brief real.
            return

        from ..engine import build_summary_provider

        provider = build_summary_provider(self.config, self.store.llm)
        if provider is None:
            # Keyless/dummy install: no provider is a documented no-op,
            # not a daily failure — the *_failed counters stay reserved
            # for real generation errors (degrade loudly, but don't cry
            # wolf about a feature that can't run at all).
            return
        text = generate_daily_brief(
            provider=provider,
            facts=facts,
            max_tokens=int(cfg.max_tokens),
            character_name=self._character_name(),
            character=getattr(self.config, "character", None),
        )
        if not text:
            # Generation failure (counted as brief.generation_failed at the
            # call): retry next tick, bounded by the daily failure cap.
            self._note_job_failure(f"brief:{user_id or 'default'}", now.date())
            return
        stored = brief_store.put(
            user_id,
            DailyBrief(
                date=now.date().isoformat(),
                text=text,
                generated_at=now.isoformat(timespec="seconds"),
            ),
        )
        if stored:
            LOGGER.info("Daily brief generated for %s", user_id or "default")

    # --- Diary ------------------------------------------------------------

    async def _diary_job(self, now: dt.datetime) -> None:
        cfg = getattr(self.config, "diary", None)
        if not cfg or not getattr(cfg, "enabled", False):
            return
        diary_store = getattr(self.store, "diary_store", None)
        if diary_store is None:
            return
        if now.time() < dt.time(hour=max(0, min(23, int(cfg.hour)))):
            return
        for user_id in self._candidate_users():
            # Prefixed so this job's in-flight keys never collide with the
            # daily brief job's bare `user_id or "default"` keys.
            key = f"diary:{user_id or 'default'}"
            if key in self._in_flight:
                continue
            if diary_store.has_for_date(user_id, now.date()):
                continue
            if self._job_failures_exhausted(key, now.date()):
                continue
            self._in_flight.add(key)
            try:
                # Provider + httpx calls stay off the event loop.
                await asyncio.to_thread(self._generate_diary_for_user, user_id, now)
            finally:
                self._in_flight.discard(key)

    # --- Reach-out ----------------------------------------------------------

    async def _reachout_job(self, now: dt.datetime) -> None:
        cfg = getattr(self.config, "reach_out", None)
        if not cfg or not getattr(cfg, "enabled", False):
            return
        reachout_store = getattr(self.store, "reachout_store", None)
        if reachout_store is None:
            return
        if now.time() < dt.time(hour=max(0, min(23, int(cfg.hour)))):
            return
        for user_id in self._candidate_users():
            key = f"reachout:{user_id or 'default'}"
            if key in self._in_flight:
                continue
            if reachout_store.has_for_date(user_id, now.date()):
                continue
            if self._job_failures_exhausted(key, now.date()):
                continue
            self._in_flight.add(key)
            try:
                # Provider calls stay off the event loop.
                await asyncio.to_thread(self._generate_reachout_for_user, user_id, now)
            finally:
                self._in_flight.discard(key)

    def _own_life_facts(self, user_id: Optional[str]) -> List[str]:
        """Peek-only lines about HER elapsed life (latest diary note — even
        an already-delivered one is still her recent days — plus active
        pursuits) for the reach-out's away-time-continuity framing.
        Best-effort: the reach-out happens with or without them."""
        facts: List[str] = []
        diary_store = getattr(self.store, "diary_store", None)
        if diary_store is not None:
            try:
                entry = diary_store.get(user_id)
                if entry is not None and entry.text:
                    note = " ".join(entry.text.split())
                    if len(note) > 220:
                        note = note[:220].rsplit(" ", 1)[0] + "…"
                    facts.append(f"Your diary ({entry.date}): {note}")
            except Exception as exc:  # pragma: no cover - store read failure
                LOGGER.warning("Reach-out diary peek failed: %s", exc)
        facts.extend(life_thread_seeds(getattr(self.store, "life_store", None)))
        return facts

    def _generate_reachout_for_user(
        self, user_id: Optional[str], now: dt.datetime
    ) -> None:
        cfg = self.config.reach_out
        reachout_store = self.store.reachout_store

        profile = self._load_profile(user_id, "Reach-out")

        # None when a reach-out would be wrong today (user not away long
        # enough, or no open thread to genuinely follow up on) — in that
        # case nothing is stored and the predicate re-evaluates next tick,
        # so a user who goes idle later in the day still gets their line.
        seeds = collect_reachout_seeds(
            profile=profile,
            now=now,
            min_idle_hours=float(getattr(cfg, "min_idle_hours", 8.0)),
            own_life_facts=self._own_life_facts(user_id),
        )
        if not seeds:
            return

        from ..engine import build_summary_provider

        provider = build_summary_provider(self.config, self.store.llm)
        if provider is None:
            # Keyless/dummy install: no provider is a documented no-op,
            # not a daily failure — the *_failed counters stay reserved
            # for real generation errors (degrade loudly, but don't cry
            # wolf about a feature that can't run at all).
            return
        text = generate_reach_out(
            provider=provider,
            facts=seeds,
            max_tokens=int(cfg.max_tokens),
            character_name=self._character_name(),
            character=getattr(self.config, "character", None),
        )
        if not text:
            # Provider failure OR a model SKIP — indistinguishable here
            # (generate_reach_out returns None for both), and both mean the
            # next tick pays another call, so both ride the bounded daily
            # retry. Seeds=None above stays uncounted: no-work is not
            # failure and must keep re-evaluating all day.
            self._note_job_failure(f"reachout:{user_id or 'default'}", now.date())
            return
        stored = reachout_store.put(
            user_id,
            ReachOut(
                date=now.date().isoformat(),
                text=text,
                generated_at=now.isoformat(timespec="seconds"),
            ),
        )
        if stored:
            LOGGER.info("Reach-out generated for %s", user_id or "default")

    # --- Async notify (Discord DM when no client is around) -----------------
    #
    # Fallback delivery for the same two things the frontend poll already
    # delivers (due reach-out, due reminders) — for the case where the app
    # is simply closed. Reuses the SAME exactly-once pop paths those poll
    # endpoints use (ReachOutStore.pop_unconsumed, ReminderStore.pop_due):
    # a line delivered here is popped exactly as if a client had received
    # it, so there is only ever one delivery path per line, never two. The
    # pop only happens once delivery is about to be attempted — never pop
    # first and then decide to skip (that would silently eat the line).

    async def _async_notify_job(self, now: dt.datetime) -> None:
        cfg = getattr(self.config, "async_notify", None)
        if not cfg or not getattr(cfg, "enabled", False):
            return
        key = "async_notify"
        if key in self._in_flight:
            return
        self._in_flight.add(key)
        try:
            # Network (Discord REST) + disk I/O stay off the event loop.
            await asyncio.to_thread(self._run_async_notify, now)
        finally:
            self._in_flight.discard(key)

    def _run_async_notify(self, now: dt.datetime) -> None:
        from .. import telemetry
        from ..sleep_schedule import is_sleep_hours

        config = self.config
        cfg = config.async_notify

        last_poll = read_last_client_poll(config)
        if last_poll is not None:
            stale_minutes = (now - last_poll).total_seconds() / 60.0
            if stale_minutes < float(cfg.min_client_stale_minutes):
                telemetry.incr("notify.skipped_client_active")
                return

        asleep = is_sleep_hours(config, now)
        last_delivery_date = read_last_delivery_date(config)
        if not should_attempt_delivery(
            config=config,
            last_client_poll=last_poll,
            now=now,
            asleep=asleep,
            last_delivery_date=last_delivery_date,
        ):
            # Distinguish the silent-by-design reasons so Diagnostics can
            # tell a quiet night from a config that never fires.
            # Client-active already returned above. Sleep uses the same
            # hold-not-consume counter the poll endpoint already reports
            # (presence.sleep_suppressed) rather than a new one here.
            if last_delivery_date is not None and last_delivery_date == now.date():
                telemetry.incr("notify.skipped_daily_cap")
            elif asleep:
                telemetry.incr("presence.sleep_suppressed")
            elif not getattr(getattr(config, "discord", None), "token", None):
                # Enabled with no bot token: should_attempt_delivery refuses
                # BEFORE anything is popped (popping first destroyed the
                # day's reach-out/reminders with nowhere to send them).
                telemetry.incr("notify.skipped_no_token")
            else:
                telemetry.incr("notify.skipped_quiet_hours")
            return

        popped = self._pop_due_content_for_notify(now)
        if popped is None:
            return
        content, undo = popped
        if deliver(config=config, content=content):
            telemetry.incr("notify.delivered")
            record_delivery(config, now.date())
            return
        # deliver() already counts notify.delivery_failures. Delivery failed
        # AFTER the pop: restore the popped line so a Discord outage doesn't
        # permanently destroy it — it goes back to being deliverable by the
        # normal poll path or tomorrow's DM attempt.
        try:
            undo()
            telemetry.incr("notify.undelivered_restored")
        except Exception as exc:
            telemetry.incr("notify.undelivered_lost")
            LOGGER.warning(
                "Async-notify failed to deliver AND to restore; content lost: "
                "%r (%s)",
                content,
                exc,
            )

    def _pop_due_content_for_notify(
        self, now: dt.datetime
    ) -> Optional[tuple[str, Callable[[], Any]]]:
        """Pop whatever is due (reach-out first, then reminders) and return
        (DM-ready string, undo-callable), or None when nothing is due. The
        undo restores the popped item if delivery then fails — see
        _run_async_notify. Popping and returning None never happens for the
        SAME source — each branch only pops once it has already committed to
        attempting delivery."""
        reachout_store = getattr(self.store, "reachout_store", None)
        if reachout_store is not None:
            # The base install keys everything under None; a user who talks
            # over the Discord bridge gets reach-outs keyed under the
            # bridge's per-user id ("discord-<id>", discord_bridge.py) — DM
            # those to the configured recipient too, since that recipient IS
            # that user.
            user_keys: List[Optional[str]] = [None]
            recipient = str(
                getattr(
                    getattr(self.config, "async_notify", None), "discord_user_id", ""
                )
                or ""
            )
            if recipient:
                user_keys.append(f"discord-{recipient}")
            for user_key in user_keys:
                entry = reachout_store.pop_unconsumed(user_key, now.date())
                if entry is not None:

                    def _undo_reachout(key=user_key):
                        reachout_store.restore_unconsumed(key, now.date())

                    return entry.text, _undo_reachout

        reminders_cfg = getattr(self.config, "reminders", None)
        if reminders_cfg and getattr(reminders_cfg, "enabled", False):
            try:
                from ..reminders import ReminderStore

                reminder_store = ReminderStore(reminders_cfg.store_path)
                due = reminder_store.pop_due(now)
            except Exception as exc:  # pragma: no cover - store I/O failure
                LOGGER.warning("Async-notify reminder read failed: %s", exc)
                due = []
            if due:
                lines = "; ".join(r.text for r in due)

                def _undo_reminders():
                    reminder_store.restore_pending(due)

                return f"Reminder: {lines}", _undo_reminders
        return None

    def _generate_diary_for_user(
        self, user_id: Optional[str], now: dt.datetime
    ) -> None:
        cfg = self.config.diary
        diary_store = self.store.diary_store

        profile = self._load_profile(user_id, "Diary")

        seeds = collect_diary_seeds(
            profile=profile,
            now=now,
            life_seeds=life_thread_seeds(getattr(self.store, "life_store", None)),
        )

        from ..engine import build_summary_provider

        provider = build_summary_provider(self.config, self.store.llm)
        if provider is None:
            # Keyless/dummy install: no provider is a documented no-op,
            # not a daily failure — the *_failed counters stay reserved
            # for real generation errors (degrade loudly, but don't cry
            # wolf about a feature that can't run at all).
            return
        text = generate_diary_entry(
            provider=provider,
            facts=seeds,
            max_tokens=int(cfg.max_tokens),
            character_name=self._character_name(),
            character=getattr(self.config, "character", None),
        )
        if not text:
            # Generation failure (counted as diary.generation_failed at the
            # call): retry next tick, bounded by the daily failure cap.
            self._note_job_failure(f"diary:{user_id or 'default'}", now.date())
            return
        stored = diary_store.put(
            user_id,
            DiaryEntry(
                date=now.date().isoformat(),
                text=text,
                generated_at=now.isoformat(timespec="seconds"),
            ),
        )
        if stored:
            LOGGER.info("Diary entry generated for %s", user_id or "default")

    # --- Follow-through (overnight research on the next-time thread) --------

    async def _followthrough_job(self, now: dt.datetime) -> None:
        cfg = getattr(self.config, "follow_through", None)
        if not cfg or not getattr(cfg, "enabled", False):
            return
        followup_store = getattr(self.store, "followup_store", None)
        if followup_store is None:
            return
        if now.time() < dt.time(hour=max(0, min(23, int(cfg.hour)))):
            return
        for user_id in self._candidate_users():
            key = f"followup:{user_id or 'default'}"
            if key in self._in_flight:
                continue
            if self._job_failures_exhausted(key, now.date()):
                continue
            self._in_flight.add(key)
            try:
                # Provider + search HTTP calls stay off the event loop. The
                # once-per-thread guard lives inside (it needs the profile
                # read), unlike the has_for_date guards above.
                await asyncio.to_thread(self._generate_followup_for_user, user_id, now)
            finally:
                self._in_flight.discard(key)

    def _generate_followup_for_user(
        self, user_id: Optional[str], now: dt.datetime
    ) -> None:
        from .. import telemetry

        cfg = self.config.follow_through
        followup_store = self.store.followup_store
        retry_key = f"followup:{user_id or 'default'}"
        # The next-time thread's `noted` stamp and FollowUp.generated_at are
        # both naive UTC, so freshness math here runs on the UTC clock — the
        # local `now` (kept for the hour gate and the retry ledger) skewed
        # the 7-day research window by the UTC offset. Derived from `now`
        # (naive local -> naive UTC) instead of utcnow() so the injected
        # test clock stays authoritative.
        utc_now = now.astimezone(dt.timezone.utc).replace(tzinfo=None)

        profile = self._load_profile(user_id, "Follow-up")
        # Only a fresh next-time thread is worth researching; no thread (the
        # common case, most nights) means nothing to do — quietly, no
        # counter: absence of work is not a degraded state.
        thread = fresh_next_time_thread(profile, utc_now)
        if not thread:
            return
        if followup_store.has_for_thread(user_id, thread):
            # Once per thread, EVER: a delivered note or a SKIP verdict is
            # final (companion-ethics contract, and the quota guard).
            return

        from ..engine import build_summary_provider

        provider = build_summary_provider(self.config, self.store.llm)
        if provider is None:
            # Keyless/dummy install: no provider is a documented no-op,
            # not a daily failure — the *_failed counters stay reserved
            # for real generation errors (degrade loudly, but don't cry
            # wolf about a feature that can't run at all).
            return
        character = getattr(self.config, "character", None)

        queries = plan_research(
            provider=provider,
            thread=thread,
            max_searches=int(cfg.max_searches),
            character=character,
            character_name=self._character_name(),
        )
        if queries is None:
            # Provider failure: store nothing so the next tick retries
            # (bounded by the daily failure cap).
            telemetry.incr("followup.plan_failed")
            self._note_job_failure(retry_key, now.date())
            return
        stamp = utc_now.isoformat(timespec="seconds")
        if not queries:
            # Legitimate SKIP (personal topic) — recorded as final.
            telemetry.incr("followup.skipped")
            followup_store.put(
                user_id,
                FollowUp(thread=thread, note="", kind="skip", generated_at=stamp),
            )
            return

        results = run_research(
            web_search_settings=getattr(self.config, "web_search", None),
            queries=queries,
            max_results_per_query=int(cfg.max_results_per_search),
        )
        if results is None:
            # follow_through is on but web_search isn't usable: loud, and
            # nothing stored — fixing the config makes a later tick work.
            # Rides the daily failure cap: without it, every retry still
            # paid the plan_research call above, all day.
            telemetry.incr("followup.no_search_backend")
            self._note_job_failure(retry_key, now.date())
            LOGGER.warning(
                "Follow-through enabled but no web-search backend is "
                "available (web_search.enabled?)"
            )
            return
        if not results:
            # Searches ran and found nothing; tomorrow won't differ.
            telemetry.incr("followup.no_results")
            followup_store.put(
                user_id,
                FollowUp(thread=thread, note="", kind="skip", generated_at=stamp),
            )
            return

        note = generate_followup_note(
            provider=provider,
            thread=thread,
            results=results,
            max_tokens=int(cfg.max_tokens),
            character=character,
            character_name=self._character_name(),
        )
        if note is None:
            telemetry.incr("followup.generation_failed")
            self._note_job_failure(retry_key, now.date())
            return
        if note == "SKIP":
            telemetry.incr("followup.skipped")
            followup_store.put(
                user_id,
                FollowUp(thread=thread, note="", kind="skip", generated_at=stamp),
            )
            return
        sources = [
            {
                "title": (r.get("title") or "").strip(),
                "url": (r.get("url") or "").strip(),
            }
            for r in results[:3]
            if (r.get("title") or "").strip()
        ]
        stored = followup_store.put(
            user_id,
            FollowUp(
                thread=thread,
                note=note,
                kind="note",
                generated_at=stamp,
                sources=sources,
            ),
        )
        if stored:
            telemetry.incr("followup.generated")
            LOGGER.info("Follow-up note generated for %s", user_id or "default")

    # --- Life threads (her own pursuits) ------------------------------------
    #
    # Unlike brief/diary/reach-out/follow-through, this job is NOT per-user:
    # her life doesn't fork per person she talks to. One store per character,
    # at most ONE generation call per day total (a seed or one progression) —
    # the pace guard lives in the store's last_activity_date.

    async def _life_threads_job(self, now: dt.datetime) -> None:
        cfg = getattr(self.config, "life_threads", None)
        if not cfg or not getattr(cfg, "enabled", False):
            return
        life_store = getattr(self.store, "life_store", None)
        if life_store is None:
            return
        if now.time() < dt.time(hour=max(0, min(23, int(cfg.hour)))):
            return
        if life_store.acted_today(now.date()):
            return
        key = "life_threads"
        if key in self._in_flight:
            return
        if self._job_failures_exhausted(key, now.date()):
            return
        self._in_flight.add(key)
        try:
            # Provider calls stay off the event loop.
            await asyncio.to_thread(self._advance_life_threads, now)
        finally:
            self._in_flight.discard(key)

    def _advance_life_threads(self, now: dt.datetime) -> None:
        from .. import telemetry

        cfg = self.config.life_threads
        life_store = self.store.life_store
        today = now.date()

        from ..engine import build_summary_provider

        provider = build_summary_provider(self.config, self.store.llm)
        if provider is None:
            # Keyless/dummy install: no provider is a documented no-op,
            # not a daily failure — the *_failed counters stay reserved
            # for real generation errors (degrade loudly, but don't cry
            # wolf about a feature that can't run at all).
            return
        character = getattr(self.config, "character", None)

        # Progression before seeding: developing an existing pursuit beats
        # starting a new one (and both together would break the 1-call/day
        # pace). The most-overdue thread goes first.
        due = [
            t
            for t in life_store.active_threads()
            if thread_due_for_progress(t, today, int(cfg.progress_interval_days))
        ]
        if due:
            thread = sorted(due, key=lambda t: t.last_progressed_at)[0]
            result = progress_life_thread(
                provider=provider,
                thread=thread,
                wrap_up=thread.progressions >= int(cfg.max_progressions),
                max_tokens=int(cfg.max_tokens),
                character=character,
                character_name=self._character_name(),
                # Their answer to an opinion-ask, if one is waiting. Shapes
                # this one step and is cleared by record_progress below —
                # the pursuit stays hers, and the leak guard in
                # progress_life_thread rejects any step that says otherwise.
                user_input=thread.pending_input or None,
            )
            if result is None:
                # Store untouched (no activity stamp): next tick retries,
                # bounded by the daily failure cap — an unbounded retry here
                # was ~1000 paid generations/day against the documented pace
                # of one.
                telemetry.incr("life.progress_failed")
                self._note_job_failure("life_threads", now.date())
                return
            if result.get("leak_rejected"):
                # The step referenced the user and was rejected (counted as
                # life.progress_user_leak_rejected inside the call). The
                # stashed opinion-ask answer likely caused it and would
                # poison every retry identically, so consume it now — the
                # influence-shot is spent on a rejected step, which beats a
                # deterministic reject loop.
                if thread.pending_input:
                    life_store.clear_pending_input(thread.topic)
                self._note_job_failure("life_threads", now.date())
                return
            life_store.record_progress(
                thread.topic,
                summary=result["summary"],
                step_text=result["step"],
                today=today,
                concluded=bool(result["concluded"]),
            )
            if result["concluded"]:
                telemetry.incr("life.concluded")
            telemetry.incr("life.progressed")
            return

        if not seed_due(life_store, today, max_active=int(cfg.max_active)):
            return
        seeded = seed_life_thread(
            provider=provider,
            existing_topics=[t.topic for t in life_store.all_threads()],
            max_tokens=int(cfg.max_tokens),
            character=character,
            character_name=self._character_name(),
        )
        if seeded is None:
            telemetry.incr("life.seed_failed")
            self._note_job_failure("life_threads", now.date())
            return
        if life_store.add_thread(seeded, today):
            telemetry.incr("life.seeded")
            LOGGER.info("Life thread seeded: %s", seeded.topic)

    # --- Insight consolidation (nightly, per user) --------------------------
    #
    # The "sleep" pass: distill the profile snapshot into standing behavioral
    # insights (amadeus/insights.py) — how to treat this person, not what to
    # remember about them. Same disk-snapshot-only stance as the brief; the
    # store's put() same-date refusal is the double-generation guard.

    async def _insights_job(self, now: dt.datetime) -> None:
        cfg = getattr(self.config, "insights", None)
        if not cfg or not getattr(cfg, "enabled", False):
            return
        insight_store = getattr(self.store, "insight_store", None)
        if insight_store is None:
            return
        if now.time() < dt.time(hour=max(0, min(23, int(cfg.hour)))):
            return
        for user_id in self._candidate_users():
            key = f"insights:{user_id or 'default'}"
            if key in self._in_flight:
                continue
            if insight_store.has_for_date(user_id, now.date()):
                continue
            if self._job_failures_exhausted(key, now.date()):
                continue
            self._in_flight.add(key)
            try:
                # Provider calls stay off the event loop.
                await asyncio.to_thread(
                    self._consolidate_insights_for_user, user_id, now
                )
            finally:
                self._in_flight.discard(key)

    def _consolidate_insights_for_user(
        self, user_id: Optional[str], now: dt.datetime
    ) -> None:
        from .. import telemetry

        cfg = self.config.insights
        insight_store = self.store.insight_store

        profile = self._load_profile(user_id, "Insights")
        facts = collect_insight_facts(profile=profile)
        if not facts:
            return

        from ..engine import build_summary_provider

        provider = build_summary_provider(self.config, self.store.llm)
        if provider is None:
            # Keyless/dummy install: no provider is a documented no-op,
            # not a daily failure — the *_failed counters stay reserved
            # for real generation errors (degrade loudly, but don't cry
            # wolf about a feature that can't run at all).
            return
        insights = generate_insights(
            provider=provider,
            facts=facts,
            previous=insight_store.get(user_id),
            max_tokens=int(cfg.max_tokens),
            min_sessions=int(getattr(cfg, "min_sessions", 3)),
            character_name=self._character_name(),
            character=getattr(self.config, "character", None),
        )
        if insights is None:
            # Failure or too-little-signal (counted inside generate_insights):
            # nothing stored, the next tick re-evaluates — bounded by the
            # daily failure cap (too-little-signal costs no LLM call, so
            # burning its three attempts is harmless; a real failure used to
            # re-pay the call every tick). An empty list (a legitimate model
            # SKIP) falls through and stores an empty set — that finalizes
            # today instead of re-asking every tick.
            self._note_job_failure(f"insights:{user_id or 'default'}", now.date())
            return
        stored = insight_store.put(
            user_id,
            InsightSet(
                date=now.date().isoformat(),
                insights=insights,
                generated_at=now.isoformat(timespec="seconds"),
            ),
        )
        if stored:
            telemetry.incr("insights.generated")
            LOGGER.info("Insights consolidated for %s", user_id or "default")

    # --- Nightly self-eval ------------------------------------------------
    #
    # Unlike brief/diary/reach-out (per-user, disk-snapshot-only), this job
    # is process-global (one conversation-quality run/day, not per user) and
    # DOES make live LLM calls through a real (isolated tmp store) engine —
    # amadeus/self_eval.py owns that isolation and the actual run. This
    # method only owns the scheduling contract: at most once/day, after
    # `hour`, gated on self_eval.enabled independent of brief/diary/
    # reach_out (see the app lifespan's scheduler-start gate).

    async def _self_eval_job(self, now: dt.datetime) -> None:
        cfg = getattr(self.config, "self_eval", None)
        if not cfg or not getattr(cfg, "enabled", False):
            return
        if now.time() < dt.time(hour=max(0, min(23, int(cfg.hour)))):
            return
        key = "self_eval"
        if key in self._in_flight:
            return
        from pathlib import Path as _Path

        # Both sides of this comparison are LOCAL dates: `now` comes from the
        # scheduler's local clock and last_run_date converts the trend stamp
        # (local-aware since 2026-08; legacy UTC-Z entries are converted) to
        # the local date. Comparing the raw UTC date against local `now`
        # made a UTC+8 install's 04:00 run stamp "yesterday", so every tick
        # until 08:00 local launched another full live-LLM eval run.
        entries = self_eval_module.read_trend_entries(_Path(cfg.trend_path))
        if self_eval_module.last_run_date(entries) == now.date().isoformat():
            return
        if self._job_failures_exhausted(key, now.date()):
            return
        self._in_flight.add(key)
        try:
            # Live LLM calls stay off the event loop, same as the brief/
            # diary/reach-out generation calls.
            await asyncio.to_thread(self_eval_module.run_nightly_self_eval, self.config)
        finally:
            self._in_flight.discard(key)
        # run_nightly_self_eval never raises; a failed run appends no trend
        # entry, which used to mean an immediate retry on the next tick — the
        # most expensive job here, hammered every 60s. Success is "a trend
        # entry for today exists"; anything else counts against the cap.
        entries = self_eval_module.read_trend_entries(_Path(cfg.trend_path))
        if self_eval_module.last_run_date(entries) != now.date().isoformat():
            self._note_job_failure(key, now.date())
