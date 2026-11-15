"""Confirms the greeting prompt actually surfaces last_episode/session_count
from the user profile — the continuity piece of the game-UX pass. This is
existing plumbing (engine.open_session_greeting -> profile_block ->
MemoryProfile.render_block), verified here rather than assumed."""

import datetime as dt

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from amadeus.config import Config
from amadeus.providers.llm.base import ChatMessage
from amadeus.web.app import create_app


class RecordingProvider:
    """Fake LLM that records every prompt it's asked to complete, so tests
    can assert on prompt *content* rather than just the final reply text."""

    name = "recording"

    def __init__(self, reply: str = "Oh. You're here.") -> None:
        self.reply = reply
        self.calls: list[list[ChatMessage]] = []

    def generate(self, messages, **kwargs):
        self.calls.append(list(messages))
        return self.reply


def _isolated_config(tmp_path) -> Config:
    config = Config()
    config.memory.persist_path = str(tmp_path / "memory.json")
    config.memory.embeddings_path = str(tmp_path / "embeddings.json")
    config.memory.profile_path = str(tmp_path / "profile.json")
    config.memory.enable_greeting = True
    config.memory.enable_fact_extraction = True
    return config


def test_greeting_prompt_includes_last_episode_and_session_count(tmp_path):
    config = _isolated_config(tmp_path)
    app = create_app(config)
    store = app.state.session_store
    recorder = RecordingProvider()
    store.llm = recorder

    client = TestClient(app)
    session_id, session = store.get(None, "user-1")
    session.llm = recorder
    session.summary_provider = recorder

    # Seed the profile as if this user has a real history — this is the
    # continuity signal the greeting is supposed to draw on.
    session.memory.profile.session_count = 4
    session.memory.profile.first_met = "2026-01-01"
    session.memory.profile.last_episode = "we debated whether D-mails erase memories"
    session.profile_block = session.memory.profile_block()

    response = client.post(
        "/api/greeting", json={"session_id": session_id, "user_id": "user-1"}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["greeting"] == "Oh. You're here."

    assert recorder.calls, "greeting must have called the LLM"
    prompt_text = "\n".join(m.content for m in recorder.calls[0])
    assert "D-mails" in prompt_text
    assert "talked 4 times" in prompt_text


def test_greeting_prompt_includes_felt_tone(tmp_path):
    config = _isolated_config(tmp_path)
    app = create_app(config)
    store = app.state.session_store
    recorder = RecordingProvider()
    store.llm = recorder

    client = TestClient(app)
    session_id, session = store.get(None, "felt-user")
    session.llm = recorder
    session.summary_provider = recorder

    # A real history plus a persisted felt-tone from last session.
    session.memory.profile.session_count = 4
    session.memory.profile.first_met = "2026-01-01"
    session.memory.profile.record_felt("excited", "the paper deadline")
    session.profile_block = session.memory.profile_block()

    response = client.post(
        "/api/greeting", json={"session_id": session_id, "user_id": "felt-user"}
    )
    assert response.status_code == 200

    assert recorder.calls, "greeting must have called the LLM"
    prompt_text = "\n".join(m.content for m in recorder.calls[0])
    assert "excited about the paper deadline" in prompt_text
    assert "Emotional continuity" in prompt_text
    # Anti-guilt constraint travels with the fact.
    assert "Never guilt" in prompt_text


def test_greeting_omits_stale_felt_tone(tmp_path):
    config = _isolated_config(tmp_path)
    app = create_app(config)
    store = app.state.session_store
    recorder = RecordingProvider()
    store.llm = recorder

    client = TestClient(app)
    session_id, session = store.get(None, "stale-felt-user")
    session.llm = recorder
    session.summary_provider = recorder

    session.memory.profile.session_count = 4
    session.memory.profile.first_met = "2026-01-01"
    old = (dt.date.today() - dt.timedelta(days=45)).isoformat()
    session.memory.profile.last_felt = {"label": "excited", "date": old}
    session.profile_block = session.memory.profile_block()

    response = client.post(
        "/api/greeting", json={"session_id": session_id, "user_id": "stale-felt-user"}
    )
    assert response.status_code == 200
    prompt_text = "\n".join(m.content for m in recorder.calls[0])
    assert "Emotional continuity" not in prompt_text


def test_greeting_skipped_without_enable_greeting_flag(tmp_path):
    config = _isolated_config(tmp_path)
    config.memory.enable_greeting = False
    app = create_app(config)
    client = TestClient(app)

    response = client.post("/api/greeting", json={"user_id": "user-2"})
    assert response.status_code == 200
    assert response.json()["greeting"] is None


def _greeting_session(tmp_path, user_id: str = "milestone-user"):
    config = _isolated_config(tmp_path)
    app = create_app(config)
    store = app.state.session_store
    recorder = RecordingProvider()
    store.llm = recorder
    session_id, session = store.get(None, user_id)
    session.llm = recorder
    session.summary_provider = recorder
    return session, recorder


def test_milestone_fact_appears_at_session_count_ten(tmp_path):
    session, recorder = _greeting_session(tmp_path, "milestone-ten")
    session.memory.profile.session_count = 10
    session.memory.profile.first_met = "2020-01-01"
    session.profile_block = session.memory.profile_block()

    assert session.open_session_greeting() is not None
    prompt_text = "\n".join(m.content for m in recorder.calls[0])
    assert "conversation number 10" in prompt_text


def test_milestone_fact_absent_at_session_count_eleven(tmp_path):
    session, recorder = _greeting_session(tmp_path, "milestone-eleven")
    session.memory.profile.session_count = 11
    session.memory.profile.first_met = "2020-01-01"
    session.profile_block = session.memory.profile_block()

    assert session.open_session_greeting() is not None
    prompt_text = "\n".join(m.content for m in recorder.calls[0])
    assert "Milestone: this is conversation number" not in prompt_text


def test_anniversary_fact_appears_exactly_one_year_after_first_met(tmp_path):
    import datetime as dt

    session, recorder = _greeting_session(tmp_path, "anniversary-user")
    today = dt.date.today()
    one_year_ago = dt.date(today.year - 1, today.month, today.day)
    session.memory.profile.session_count = 4
    session.memory.profile.first_met = one_year_ago.isoformat()
    session.profile_block = session.memory.profile_block()

    assert session.open_session_greeting() is not None
    prompt_text = "\n".join(m.content for m in recorder.calls[0])
    assert "Milestone: today marks 1 year since you first met." in prompt_text


def test_low_tone_trend_fact_appears_with_three_recent_low_entries(tmp_path):
    import datetime as dt

    session, recorder = _greeting_session(tmp_path, "low-tone-user")
    today = dt.date.today()
    # A tone trend over "recent conversations" presupposes there have been
    # some — pin session_count so this isn't read as a first meeting.
    session.memory.profile.session_count = 5
    session.memory.profile.tone_history = [
        {"date": (today - dt.timedelta(days=i)).isoformat(), "tone": -0.6}
        for i in range(3)
    ]
    session.profile_block = session.memory.profile_block()

    assert session.open_session_greeting() is not None
    prompt_text = "\n".join(m.content for m in recorder.calls[0])
    assert "the user has seemed low across recent conversations" in prompt_text


def test_low_tone_trend_fact_absent_with_only_two_entries(tmp_path):
    import datetime as dt

    session, recorder = _greeting_session(tmp_path, "low-tone-insufficient")
    today = dt.date.today()
    session.memory.profile.tone_history = [
        {"date": (today - dt.timedelta(days=i)).isoformat(), "tone": -0.6}
        for i in range(2)
    ]
    session.profile_block = session.memory.profile_block()

    assert session.open_session_greeting() is not None
    prompt_text = "\n".join(m.content for m in recorder.calls[0])
    assert "the user has seemed low across recent conversations" not in prompt_text


def test_greeting_prompt_flags_return_after_long_absence(tmp_path):
    config = _isolated_config(tmp_path)
    app = create_app(config)
    store = app.state.session_store
    recorder = RecordingProvider()
    store.llm = recorder

    client = TestClient(app)
    session_id, session = store.get(None, "user-3")
    session.llm = recorder
    session.summary_provider = recorder

    # A "returning after N days away" beat presupposes a prior conversation
    # — pin session_count so this isn't read as a first meeting.
    session.memory.profile.session_count = 5
    # 3 days away is well past the 48h absence threshold.
    last_seen = dt.datetime.utcnow() - dt.timedelta(days=3)
    session.memory.profile.last_seen = last_seen.isoformat()
    session.profile_block = session.memory.profile_block()

    response = client.post(
        "/api/greeting", json={"session_id": session_id, "user_id": "user-3"}
    )
    assert response.status_code == 200

    assert recorder.calls, "greeting must have called the LLM"
    prompt_text = "\n".join(m.content for m in recorder.calls[0])
    assert "returning after 3 days" in prompt_text
    assert "NEVER guilt" in prompt_text


def test_greeting_prompt_omits_return_line_under_48_hours(tmp_path):
    config = _isolated_config(tmp_path)
    app = create_app(config)
    store = app.state.session_store
    recorder = RecordingProvider()
    store.llm = recorder

    client = TestClient(app)
    session_id, session = store.get(None, "user-4")
    session.llm = recorder
    session.summary_provider = recorder

    # 5 hours away is well under the 48h absence threshold.
    last_seen = dt.datetime.utcnow() - dt.timedelta(hours=5)
    session.memory.profile.last_seen = last_seen.isoformat()
    session.profile_block = session.memory.profile_block()

    response = client.post(
        "/api/greeting", json={"session_id": session_id, "user_id": "user-4"}
    )
    assert response.status_code == 200

    assert recorder.calls, "greeting must have called the LLM"
    prompt_text = "\n".join(m.content for m in recorder.calls[0])
    assert "returning after" not in prompt_text


# --- First-meeting greeting variant (M6) -----------------------------------


def test_first_meeting_prompt_introduces_and_asks_for_name(tmp_path):
    """A genuinely fresh profile (session_count<=1, no facts, no open
    threads) gets the FIRST-conversation branch: introduce herself, ask for
    the user's name, no continuity color."""
    session, recorder = _greeting_session(tmp_path, "first-meeting-user")
    assert session.memory.profile.session_count <= 1
    assert not session.memory.profile.facts
    assert not session.memory.profile.open_threads

    greeting = session.open_session_greeting()
    assert greeting is not None
    prompt_text = "\n".join(m.content for m in recorder.calls[0])
    assert "FIRST conversation" in prompt_text
    assert "ask for their name" in prompt_text


def test_profile_with_facts_is_not_treated_as_first_meeting(tmp_path):
    """A profile that already has facts must never hit the first-meeting
    branch, even if session_count is low."""
    session, recorder = _greeting_session(tmp_path, "returning-user")
    session.memory.profile.session_count = 1
    session.memory.profile.facts["name"] = "Alex"
    session.profile_block = session.memory.profile_block()

    greeting = session.open_session_greeting()
    assert greeting is not None
    prompt_text = "\n".join(m.content for m in recorder.calls[0])
    assert "FIRST conversation" not in prompt_text


def test_greeting_endpoint_includes_milestones_for_milestone_session_count(
    tmp_path,
):
    config = _isolated_config(tmp_path)
    app = create_app(config)
    store = app.state.session_store
    recorder = RecordingProvider()
    store.llm = recorder

    client = TestClient(app)
    session_id, session = store.get(None, "milestone-endpoint-user")
    session.llm = recorder
    session.summary_provider = recorder

    # Force a milestone session_count and non-empty facts/threads so this
    # reads as a returning user, not a first meeting.
    session.memory.profile.session_count = 10
    session.memory.profile.facts["name"] = "Alex"
    session.profile_block = session.memory.profile_block()

    response = client.post(
        "/api/greeting",
        json={"session_id": session_id, "user_id": "milestone-endpoint-user"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["greeting"] is not None
    assert "milestones" in body
    assert body["milestones"], "expected at least one milestone event"
    assert body["milestones"][0]["kind"] == "sessions"
    assert body["milestones"][0]["value"] == 10


# --- First-meeting predicate hardening (G7) --------------------------------


def test_rehydrated_short_term_is_never_first_meeting(tmp_path):
    """G7 regression: a returning user in a live/rehydrated session already
    has conversation context in short_term, so even a profile that LOOKS
    empty (session_count<=1, no facts, no threads — e.g. fact extraction
    never ran) must never be misclassified as a first meeting."""
    session, recorder = _greeting_session(tmp_path, "rehydrated-user")
    assert session.memory.profile.session_count <= 1
    assert not session.memory.profile.facts
    assert not session.memory.profile.open_threads

    # Simulate a rehydrated/live session: prior messages already exist in
    # short_term even though the profile itself never accumulated facts.
    session.memory.add_message("user", "hey, remember me?")
    session.memory.add_message("assistant", "Of course.")
    # should_greet() only fires for a FRESH session (gap since the last
    # message); force that condition without wiping short_term.
    import datetime as _dt

    stale = _dt.datetime.utcnow() - _dt.timedelta(minutes=60)
    session.memory.short_term[-1].timestamp = stale.isoformat()

    greeting = session.open_session_greeting()
    assert greeting is not None
    # calls[0] is the greeting call itself — the stale timestamp also makes
    # the greeting's own trailing add_message("assistant", ...) trip
    # _maybe_finalize_gap's episode-consolidation LLM call, which lands
    # AFTER the greeting call in recorder.calls.
    prompt_text = "\n".join(m.content for m in recorder.calls[0])
    assert "FIRST conversation" not in prompt_text


def test_old_first_met_is_never_first_meeting(tmp_path):
    """G7 regression: session_count<=1 with no facts/threads is not enough
    on its own — an old first_met (or malformed/missing) must NOT count as
    'recent', so the fail-closed predicate correctly reads this as NOT a
    first meeting rather than a stranger encounter."""
    session, recorder = _greeting_session(tmp_path, "old-first-met-user")
    session.memory.profile.session_count = 1
    session.memory.profile.first_met = "2020-01-01"
    session.profile_block = session.memory.profile_block()

    greeting = session.open_session_greeting()
    assert greeting is not None
    prompt_text = "\n".join(m.content for m in recorder.calls[0])
    assert "FIRST conversation" not in prompt_text


def test_missing_first_met_is_never_first_meeting(tmp_path):
    """Tolerate-failure case: first_met is None (never stamped) — the
    predicate must fail closed (NOT a first meeting) rather than risk
    reintroducing herself to an established profile with a missing
    timestamp."""
    session, recorder = _greeting_session(tmp_path, "no-first-met-user")
    session.memory.profile.session_count = 1
    session.memory.profile.first_met = None
    session.profile_block = session.memory.profile_block()

    greeting = session.open_session_greeting()
    assert greeting is not None
    prompt_text = "\n".join(m.content for m in recorder.calls[0])
    assert "FIRST conversation" not in prompt_text


def test_garbage_first_met_is_never_first_meeting(tmp_path):
    """Malformed first_met must not raise and must fail closed (NOT a first
    meeting)."""
    session, recorder = _greeting_session(tmp_path, "garbage-first-met-user")
    session.memory.profile.session_count = 1
    session.memory.profile.first_met = "not-a-date"
    session.profile_block = session.memory.profile_block()

    greeting = session.open_session_greeting()
    assert greeting is not None
    prompt_text = "\n".join(m.content for m in recorder.calls[0])
    assert "FIRST conversation" not in prompt_text


def test_genuine_first_meeting_still_fires_with_fresh_first_met(tmp_path):
    """Sanity check: a truly fresh profile (first_met stamped to today by
    the session-boundary save in MemoryManager.__init__, empty short_term,
    session_count<=1, no facts/threads) still correctly hits the
    first-meeting branch — the hardened predicate must not be so strict it
    breaks the real case."""
    session, recorder = _greeting_session(tmp_path, "genuinely-new-user")
    assert session.memory.profile.session_count <= 1
    assert not session.memory.profile.facts
    assert not session.memory.profile.open_threads
    assert not session.memory.short_term
    assert session.memory.profile.first_met is not None

    greeting = session.open_session_greeting()
    assert greeting is not None
    prompt_text = "\n".join(m.content for m in recorder.calls[0])
    assert "FIRST conversation" in prompt_text


# --- Milestone computed once (G8) ------------------------------------------


def test_greeting_endpoint_milestones_match_what_greeting_used(tmp_path):
    """G8 regression: the endpoint must report EXACTLY the milestone events
    that informed the greeting text (session.last_greeting_milestones), not
    a second independent milestone_events() call that could diverge."""
    config = _isolated_config(tmp_path)
    app = create_app(config)
    store = app.state.session_store
    recorder = RecordingProvider()
    store.llm = recorder

    client = TestClient(app)
    session_id, session = store.get(None, "milestone-match-user")
    session.llm = recorder
    session.summary_provider = recorder

    session.memory.profile.session_count = 25
    session.memory.profile.facts["name"] = "Alex"
    session.profile_block = session.memory.profile_block()

    response = client.post(
        "/api/greeting",
        json={"session_id": session_id, "user_id": "milestone-match-user"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["milestones"] == session.last_greeting_milestones
    assert body["milestones"][0]["value"] == 25


def test_first_meeting_greeting_reports_no_milestones(tmp_path):
    """First-meeting greetings never have milestone color — the endpoint
    must reflect that as an empty list, not a stale/leftover value."""
    session, recorder = _greeting_session(tmp_path, "first-meeting-no-milestones")
    assert session.memory.profile.session_count <= 1

    greeting = session.open_session_greeting()
    assert greeting is not None
    assert session.last_greeting_milestones == []


# ---------------------------------------------------------------------------
# "Next time" thread: the topic stashed at the last farewell resurfaces in
# the next greeting, consumed exactly once AFTER a successful generation
# ---------------------------------------------------------------------------


def _next_time_session(tmp_path, user_id, *, age: dt.timedelta):
    session, recorder = _greeting_session(tmp_path, user_id)
    session.memory.profile.session_count = 5
    session.memory.profile.first_met = "2026-01-01"
    session.memory.profile.set_next_time_thread(
        "they were about to submit the manuscript",
        when=dt.datetime.utcnow() - age,
    )
    session.memory.persist_profile()
    return session, recorder


def test_greeting_folds_fresh_next_time_thread_and_consumes(tmp_path):
    session, recorder = _next_time_session(
        tmp_path, "next-time-fresh", age=dt.timedelta(hours=12)
    )

    assert session.open_session_greeting() is not None
    prompt_text = "\n".join(m.content for m in recorder.calls[0])
    assert "about to submit the manuscript" in prompt_text
    assert "ask about THIS" in prompt_text
    # Fresh thread owns the ask slot: the system prompt gets the ask clause.
    assert "open by asking about it" in prompt_text
    # Consumed exactly once, and the consumption is persisted.
    assert session.memory.profile.next_time_thread is None
    from amadeus.memory.profile import UserProfile

    persisted = [
        UserProfile.load(p)
        for p in tmp_path.rglob("*.json")
        if "next_time_thread" in p.read_text()
    ]
    assert persisted and all(p.next_time_thread is None for p in persisted)


def test_greeting_phrases_stale_next_time_thread_tentatively(tmp_path):
    session, recorder = _next_time_session(
        tmp_path, "next-time-stale", age=dt.timedelta(days=12)
    )

    assert session.open_session_greeting() is not None
    prompt_text = "\n".join(m.content for m in recorder.calls[0])
    assert "about to submit the manuscript" in prompt_text
    assert "phrase it tentatively" in prompt_text
    assert "ask about THIS" not in prompt_text
    # Stale delivery still consumes.
    assert session.memory.profile.next_time_thread is None


def test_greeting_drops_expired_next_time_thread_unconsumed(tmp_path):
    session, recorder = _next_time_session(
        tmp_path, "next-time-expired", age=dt.timedelta(days=40)
    )

    assert session.open_session_greeting() is not None
    prompt_text = "\n".join(m.content for m in recorder.calls[0])
    # Never surfaces, and the slot is cleared (dropped, not delivered late).
    assert "manuscript" not in prompt_text
    assert session.memory.profile.next_time_thread is None


def test_failed_greeting_leaves_next_time_thread_unconsumed(tmp_path):
    session, recorder = _next_time_session(
        tmp_path, "next-time-failed", age=dt.timedelta(hours=6)
    )
    recorder.reply = ""  # generation produces nothing -> greeting fails

    assert session.open_session_greeting() is None
    # Retry next session: the thread survives a failed generation.
    thread = session.memory.profile.next_time_thread
    assert thread is not None
    assert thread["text"] == "they were about to submit the manuscript"


def test_fresh_next_time_thread_suppresses_open_thread_followup(tmp_path):
    session, recorder = _next_time_session(
        tmp_path, "next-time-vs-followup", age=dt.timedelta(hours=3)
    )
    profile = session.memory.profile
    profile.set_open_threads(["the reviewer response"], max_threads=3)

    assert session.open_session_greeting() is not None
    prompt_text = "\n".join(m.content for m in recorder.calls[0])
    # One ask, not two: the farewell-time thread wins the ask slot, and the
    # stored open thread is not ALSO promoted to an "ask about THIS" fact.
    assert prompt_text.count("ask about THIS") == 1
    assert "about to submit the manuscript" in prompt_text


def test_greeting_nudges_getting_acquainted_while_profile_sparse(tmp_path):
    """Early sessions with an empty profile spend the greeting's ask slot on
    learning who the user is (the first-meeting branch already asks for a
    name; this covers sessions 2..N)."""
    session, recorder = _greeting_session(tmp_path, "sparse-user")
    session.memory.profile.session_count = 2
    session.memory.profile.first_met = "2026-01-01"

    assert session.open_session_greeting() is not None
    prompt_text = "\n".join(m.content for m in recorder.calls[0])
    assert "you still know very little about this user" in prompt_text


def test_greeting_acquaintance_nudge_yields_ask_slot_to_next_time(tmp_path):
    """A fresh next-time thread owns the greeting's one ask; the
    getting-acquainted nudge must stand down rather than stack a second
    question onto the same greeting."""
    session, recorder = _greeting_session(tmp_path, "sparse-user-thread")
    session.memory.profile.session_count = 2
    session.memory.profile.first_met = "2026-01-01"
    session.memory.profile.set_next_time_thread(
        "the robotics demo", when=dt.datetime.utcnow()
    )

    assert session.open_session_greeting() is not None
    prompt_text = "\n".join(m.content for m in recorder.calls[0])
    assert "robotics demo" in prompt_text
    assert "you still know very little about this user" not in prompt_text


def test_greeting_acquaintance_nudge_absent_once_profile_matures(tmp_path):
    session, recorder = _greeting_session(tmp_path, "mature-user")
    session.memory.profile.session_count = 9
    session.memory.profile.first_met = "2026-01-01"

    assert session.open_session_greeting() is not None
    prompt_text = "\n".join(m.content for m in recorder.calls[0])
    assert "you still know very little about this user" not in prompt_text


class _FakeLifeThread:
    topic = "a kinesthetics survey paper"
    summary = "halfway through the methods section"


class _FakeLifeStore:
    def active_threads(self):
        return [_FakeLifeThread()]


def test_greeting_long_gap_mentions_her_own_elapsed_life(tmp_path):
    """After a multi-day gap the greeting prompt carries one peek-only
    pursuit line framed as what SHE did while the user was away."""
    session, recorder = _greeting_session(tmp_path, "away-user")
    session.memory.profile.session_count = 9
    session.memory.profile.first_met = "2026-01-01"
    session.memory.profile.last_seen = (
        dt.datetime.utcnow() - dt.timedelta(days=4)
    ).isoformat()
    session.life_store = _FakeLifeStore()

    assert session.open_session_greeting() is not None
    prompt_text = "\n".join(m.content for m in recorder.calls[0])
    assert "kinesthetics survey paper" in prompt_text
    assert "while they were away" in prompt_text


def test_greeting_short_gap_skips_her_own_life_register(tmp_path):
    session, recorder = _greeting_session(tmp_path, "recent-user")
    session.memory.profile.session_count = 9
    session.memory.profile.first_met = "2026-01-01"
    session.memory.profile.last_seen = (
        dt.datetime.utcnow() - dt.timedelta(hours=5)
    ).isoformat()
    session.life_store = _FakeLifeStore()

    assert session.open_session_greeting() is not None
    prompt_text = "\n".join(m.content for m in recorder.calls[0])
    assert "while they were away" not in prompt_text
