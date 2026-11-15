"""Greeting ask-slot arbitration contract.

The opening greeting has exactly ONE question to spend, and by 2026-08 five
features compete for it. The priority order lives implicitly in
``amadeus/greeting.py``'s control flow; this file pins it explicitly so the
next feature that wants the slot has to update a contract, not just win a
code-order race:

    follow-up research note            (owns the slot, REPLACES the plain
                                        next-time fact for the same topic)
  > fresh next-time thread             (owns the slot)
  > open-thread selection              (profile.followup_thread(), aged >= 1d)
  > early-acquaintance nudge           (only when nothing above claimed it)

A STALE next-time thread gets a tentative mention but does NOT own the slot —
the open-thread ask may still fire alongside it.

Brief / diary / milestones / persona-change note / register lines are
additive color, never asks: with no thread state at all the system prompt
must carry the no-ask clause, whatever color is folded in.

Consume-after-success applies to every consumable at once: one failed
generation must leave the follow-up note, next-time thread, brief, diary,
open-thread mark, and persona-change announcement ALL untouched for retry.
"""

from __future__ import annotations

import datetime as dt

import pytest

pytest.importorskip("fastapi")

from amadeus.brief import DailyBrief, DailyBriefStore
from amadeus.config import Config
from amadeus.diary import DiaryEntry, DiaryStore
from amadeus.followup import FollowUp, FollowUpStore
from amadeus.providers.llm.base import ChatMessage
from amadeus.web.app import create_app

# Stable prompt markers, asserted instead of re-derived so a wording change
# is a conscious contract update, not silent drift.
ASK_CLAUSE = "open by asking about it"  # system prompt: the slot is spent
NO_ASK_CLAUSE = "You may nod to the time of day"  # system prompt: slot unspent
OPEN_THREAD_ASK = "Unresolved since last time (ask about THIS)"
NEXT_TIME_FRESH = "left mid-thing"
NEXT_TIME_STALE = "phrase it tentatively"
FOLLOWUP_NOTE = "You actually looked into it since"
ACQUAINTANCE = "you still know very little about this user"


class RecordingProvider:
    name = "recording"

    def __init__(self, reply: str = "Oh. You're here.") -> None:
        self.reply = reply
        self.calls: list[list[ChatMessage]] = []

    def generate(self, messages, **kwargs):
        self.calls.append(list(messages))
        return self.reply


class FakeVersionStore:
    """Persona-version store double: a pending change note plus a record of
    whether the greeting marked it announced (it must only after success)."""

    def __init__(self, note: str | None = "voice pitch was raised") -> None:
        self.note = note
        self.announced = 0

    def pending_change_note(self, config):
        return self.note

    def mark_announced(self, config):
        self.announced += 1


def _session(tmp_path, user_id: str):
    config = Config()
    config.memory.persist_path = str(tmp_path / "memory.json")
    config.memory.embeddings_path = str(tmp_path / "embeddings.json")
    config.memory.profile_path = str(tmp_path / "profile.json")
    config.memory.enable_greeting = True
    config.memory.enable_fact_extraction = True
    app = create_app(config)
    store = app.state.session_store
    recorder = RecordingProvider()
    store.llm = recorder
    _, session = store.get(None, user_id)
    session.llm = recorder
    session.summary_provider = recorder
    profile = session.memory.profile
    profile.first_met = "2026-01-01"
    return session, recorder


def _aged_open_thread(profile, text: str, days: int = 3) -> None:
    profile.set_open_threads([text], max_threads=3)
    profile.open_thread_dates[text] = (
        dt.date.today() - dt.timedelta(days=days)
    ).isoformat()


def _seed_followup_note(session, tmp_path, thread: str) -> FollowUpStore:
    store = FollowUpStore(tmp_path / "followup.json")
    store.put(
        session.user_id,
        FollowUp(
            thread=thread,
            note="nnU-Net v2 apparently handles this exact case.",
            kind="note",
            generated_at=(dt.datetime.utcnow() - dt.timedelta(hours=6)).isoformat(
                timespec="seconds"
            ),
        ),
    )
    session.followup_store = store
    return store


def _seed_brief(session, tmp_path) -> DailyBriefStore:
    store = DailyBriefStore(tmp_path / "brief.json")
    today = dt.date.today().isoformat()
    store.put(
        session.user_id,
        DailyBrief(
            date=today,
            text="Two reminders today; rain this afternoon.",
            generated_at=dt.datetime.utcnow().isoformat(timespec="seconds"),
        ),
    )
    session.brief_store = store
    return store


def _seed_diary(session, tmp_path) -> DiaryStore:
    store = DiaryStore(tmp_path / "diary.json")
    today = dt.date.today().isoformat()
    store.put(
        session.user_id,
        DiaryEntry(
            date=today,
            text="Spent the morning on the kinesthetics survey draft.",
            generated_at=dt.datetime.utcnow().isoformat(timespec="seconds"),
        ),
    )
    session.diary_store = store
    return store


def _prompt_text(recorder: RecordingProvider) -> str:
    return "\n".join(m.content for m in recorder.calls[0])


THREAD = "they were mid-refactor of the segmentation model"
OPEN_THREAD = "the reviewer response letter"


def _full_house(tmp_path, user_id: str):
    """Every ask-slot contender and every additive consumable at once."""
    session, recorder = _session(tmp_path, user_id)
    profile = session.memory.profile
    # session_count 5 keeps early acquaintance ELIGIBLE (< max_sessions 8,
    # no facts) so the test proves suppression, not inapplicability.
    profile.session_count = 5
    profile.set_next_time_thread(
        THREAD, when=dt.datetime.utcnow() - dt.timedelta(hours=8)
    )
    _aged_open_thread(profile, OPEN_THREAD)
    session.memory.persist_profile()
    followup_store = _seed_followup_note(session, tmp_path, THREAD)
    brief_store = _seed_brief(session, tmp_path)
    diary_store = _seed_diary(session, tmp_path)
    version_store = FakeVersionStore()
    session._persona_version_store = lambda: version_store
    return session, recorder, followup_store, brief_store, diary_store, version_store


def test_followup_note_wins_full_house_and_everything_consumes(tmp_path):
    (
        session,
        recorder,
        followup_store,
        brief_store,
        diary_store,
        version_store,
    ) = _full_house(tmp_path, "full-house")

    assert session.open_session_greeting() is not None
    prompt = _prompt_text(recorder)

    # The note owns the one ask; every other contender stands down.
    assert FOLLOWUP_NOTE in prompt
    assert prompt.count(ASK_CLAUSE) == 1
    assert NEXT_TIME_FRESH not in prompt  # replaced, not doubled
    assert OPEN_THREAD_ASK not in prompt
    assert ACQUAINTANCE not in prompt

    # Additive color still rides along — losing the ask slot never means
    # losing delivery.
    assert "Morning brief" in prompt
    assert "Your own diary note" in prompt
    assert "your own settings/voice were changed" in prompt

    # Consume-after-success, all at once: note, thread, brief, diary,
    # persona announcement. The open thread never surfaced as an ask, so it
    # must NOT be burned.
    assert followup_store.get(session.user_id).consumed is True
    assert session.memory.profile.next_time_thread is None
    assert brief_store.peek_unconsumed(session.user_id, dt.date.today()) is None
    assert diary_store.peek_unconsumed(session.user_id, dt.date.today()) is None
    assert version_store.announced == 1
    assert session.memory.profile.followed_up_threads == []


def test_failed_generation_consumes_nothing(tmp_path):
    (
        session,
        recorder,
        followup_store,
        brief_store,
        diary_store,
        version_store,
    ) = _full_house(tmp_path, "full-house-fail")
    recorder.reply = ""  # empty generation -> greeting returns None

    assert session.open_session_greeting() is None

    # Every consumable survives for retry next session.
    assert followup_store.get(session.user_id).consumed is False
    assert session.memory.profile.next_time_thread is not None
    assert brief_store.peek_unconsumed(session.user_id, dt.date.today()) is not None
    assert diary_store.peek_unconsumed(session.user_id, dt.date.today()) is not None
    assert version_store.announced == 0
    assert session.memory.profile.followed_up_threads == []


def test_followup_note_alone_suppresses_open_thread_and_acquaintance(tmp_path):
    """The note claims the slot even with NO pending next-time thread — its
    presence alone must stand the open-thread ask and the acquaintance nudge
    down (it sets next_time_ask by itself)."""
    session, recorder = _session(tmp_path, "note-only")
    profile = session.memory.profile
    profile.session_count = 5  # acquaintance eligible: sparse + young
    _aged_open_thread(profile, OPEN_THREAD)
    _seed_followup_note(session, tmp_path, THREAD)

    assert session.open_session_greeting() is not None
    prompt = _prompt_text(recorder)
    assert FOLLOWUP_NOTE in prompt
    assert OPEN_THREAD_ASK not in prompt
    assert ACQUAINTANCE not in prompt
    assert prompt.count(ASK_CLAUSE) == 1
    # The un-asked open thread is not burned.
    assert profile.followed_up_threads == []


def test_stale_next_time_yields_ask_to_open_thread(tmp_path):
    """A stale next-time thread is a tentative mention, not an ask — the
    open-thread selection still runs and spends the slot, and both consume:
    the stale thread clears, the asked thread is marked followed-up."""
    session, recorder = _session(tmp_path, "stale-vs-thread")
    profile = session.memory.profile
    profile.session_count = 9  # acquaintance out of the picture
    profile.set_next_time_thread(
        THREAD, when=dt.datetime.utcnow() - dt.timedelta(days=12)
    )
    _aged_open_thread(profile, OPEN_THREAD)
    session.memory.persist_profile()

    assert session.open_session_greeting() is not None
    prompt = _prompt_text(recorder)
    assert NEXT_TIME_STALE in prompt
    assert OPEN_THREAD_ASK in prompt
    assert ASK_CLAUSE in prompt
    assert profile.next_time_thread is None
    assert OPEN_THREAD in profile.followed_up_threads


def test_open_thread_ask_suppresses_acquaintance(tmp_path):
    """With no next-time state at all, an aged open thread beats the
    acquaintance nudge for the slot."""
    session, recorder = _session(tmp_path, "thread-vs-acquaintance")
    profile = session.memory.profile
    profile.session_count = 5  # acquaintance eligible: sparse + young
    _aged_open_thread(profile, OPEN_THREAD)

    assert session.open_session_greeting() is not None
    prompt = _prompt_text(recorder)
    assert OPEN_THREAD_ASK in prompt
    assert ACQUAINTANCE not in prompt
    assert OPEN_THREAD in profile.followed_up_threads


def test_additive_color_never_claims_the_ask_slot(tmp_path):
    """Brief + diary + milestone + persona-change note, zero thread state:
    all four deliver as color while the system prompt carries the no-ask
    clause — none of them is allowed to masquerade as the question."""
    session, recorder = _session(tmp_path, "color-only")
    profile = session.memory.profile
    profile.session_count = 10  # milestone; also ends early acquaintance
    _seed_brief(session, tmp_path)
    _seed_diary(session, tmp_path)
    version_store = FakeVersionStore()
    session._persona_version_store = lambda: version_store

    assert session.open_session_greeting() is not None
    prompt = _prompt_text(recorder)
    assert "Morning brief" in prompt
    assert "Your own diary note" in prompt
    assert "conversation number 10" in prompt
    assert "your own settings/voice were changed" in prompt
    assert ASK_CLAUSE not in prompt
    assert NO_ASK_CLAUSE in prompt
    assert ACQUAINTANCE not in prompt
