"""Life-thread opinion asks: the reciprocity widening and its guards.

This is the one place her own material makes a claim on the user's
attention, and it deliberately relaxes docs/companion-ethics.md's "never
about, for, or involving the person you talk to". The relaxation is only
defensible because of the guards pinned here:

- The user's answer shapes ONE progression as context and is then gone; the
  progression prompt still forbids mentioning them, and a step whose stored
  text leaks the user is REJECTED rather than stored. That check is code,
  not a prompt hope, because a cheap model at max_tokens=140 cannot be
  trusted to hold a line like this on its own.
- A thread is asked about at most ONCE, ever.
- NOTHING records whether the user answered. A system that cannot store the
  slight cannot sulk about it — which is what makes ignoring an ask a
  complete, cost-free answer rather than a debt.
"""

from __future__ import annotations

import datetime as dt

import pytest

from amadeus import telemetry
from amadeus.life_threads import (
    LifeThread,
    LifeThreadStore,
    ask_due,
    ask_thread_fact,
    progress_life_thread,
    progression_leaks_user,
)
from amadeus.providers.llm.base import ChatMessage


@pytest.fixture(autouse=True)
def _clean_telemetry():
    telemetry.reset()
    yield
    telemetry.reset()


class _Provider:
    name = "scripted"

    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.calls: list[list[ChatMessage]] = []

    def generate(self, messages, **kwargs):
        self.calls.append(list(messages))
        return self.reply


def _thread(**over) -> LifeThread:
    base = dict(
        topic="re-running the retraction-bait stats",
        summary="halfway through the reanalysis",
        started_at="2026-07-01",
        last_progressed_at="2026-08-01",
        progressions=2,
    )
    base.update(over)
    return LifeThread(**base)


def _store(tmp_path, *threads) -> LifeThreadStore:
    store = LifeThreadStore(str(tmp_path / "life.json"))
    for thread in threads:
        store.add_thread(thread, dt.date(2026, 8, 1))
    return store


# ---------------------------------------------------------------------------
# The leak guard — code, not prompt
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "You were right about the outliers, so I dropped them.",
        "Your suggestion worked.",
        "We finally cracked it.",
        "I took your advice and re-ran it.",
    ],
)
def test_progression_leaks_user_catches_second_person_and_plural(text: str) -> None:
    assert progression_leaks_user(text) is True


@pytest.mark.parametrize(
    "text",
    [
        "Dropped the outliers and re-ran it; the effect halved.",
        "The simulation stalled again. Reverted to the older solver.",
        "",
    ],
)
def test_progression_leaks_user_passes_her_own_work(text: str) -> None:
    assert progression_leaks_user(text) is False


def test_a_leaking_progression_is_rejected_not_stored() -> None:
    """THE AUDITABLE GUARD. The widening lets their answer influence her
    thinking; it must not let the pursuit become about them. A step that
    says otherwise is dropped and counted, so erosion is visible. The
    sentinel return (distinct from None = provider failure) lets the
    scheduler clear the pending_input that provoked the leak — the same
    stashed answer fed to every retry made rejection deterministic."""
    provider = _Provider("You were right — I dropped the outliers like you said.")

    result = progress_life_thread(
        provider=provider, thread=_thread(), wrap_up=False, user_input="drop them"
    )

    assert result == {"leak_rejected": True}
    assert telemetry.snapshot().get("life.progress_user_leak_rejected") == 1


def test_a_clean_progression_still_stores_after_user_input() -> None:
    provider = _Provider("Dropped the outliers and re-ran it; the effect halved.")

    result = progress_life_thread(
        provider=provider, thread=_thread(), wrap_up=False, user_input="drop them"
    )

    assert result is not None
    assert result["summary"].startswith("Dropped the outliers")
    assert telemetry.snapshot().get("life.progress_user_leak_rejected") is None


def test_user_input_reaches_the_prompt_but_the_ban_survives() -> None:
    provider = _Provider("Re-ran it with the outliers excluded.")

    progress_life_thread(
        provider=provider,
        thread=_thread(),
        wrap_up=False,
        user_input="I'd drop the outliers",
    )

    system = provider.calls[0][0].content
    user = provider.calls[0][1].content
    assert "I'd drop the outliers" in user
    assert "never referencing them" in user
    # The original hard rule is untouched by the widening.
    assert "Never mention the person you talk to." in system


# ---------------------------------------------------------------------------
# Rate: rare by construction
# ---------------------------------------------------------------------------


def test_ask_due_requires_a_developed_thread(tmp_path) -> None:
    store = _store(tmp_path, _thread(progressions=0))
    assert ask_due(store, dt.date(2026, 8, 7), min_days_between_asks=21) is None


def test_ask_due_returns_a_developed_unasked_thread(tmp_path) -> None:
    store = _store(tmp_path, _thread())
    due = ask_due(store, dt.date(2026, 8, 7), min_days_between_asks=21)
    assert due is not None and due.topic == "re-running the retraction-bait stats"


def test_a_thread_is_asked_about_at_most_once_ever(tmp_path) -> None:
    """One-way by design: an ignored ask is terminal, never re-raised."""
    store = _store(tmp_path, _thread())
    today = dt.date(2026, 8, 7)
    assert store.mark_asked("re-running the retraction-bait stats", today) is True

    # Not even after the store-wide floor expires.
    assert (
        ask_due(store, today + dt.timedelta(days=365), min_days_between_asks=21) is None
    )


def test_store_wide_floor_blocks_other_threads(tmp_path) -> None:
    store = _store(tmp_path, _thread(), _thread(topic="the ruined sourdough starter"))
    today = dt.date(2026, 8, 7)
    store.mark_asked("re-running the retraction-bait stats", today)

    assert (
        ask_due(store, today + dt.timedelta(days=5), min_days_between_asks=21) is None
    )
    later = ask_due(store, today + dt.timedelta(days=22), min_days_between_asks=21)
    assert later is not None and later.topic == "the ruined sourdough starter"


def test_ask_due_is_none_without_a_store() -> None:
    assert ask_due(None, dt.date(2026, 8, 7), min_days_between_asks=21) is None


# ---------------------------------------------------------------------------
# Framing: an ask, never a claim
# ---------------------------------------------------------------------------


def test_ask_fact_forbids_neediness_and_re_asking() -> None:
    fact = ask_thread_fact(_thread())
    assert fact is not None

    assert "Ask ONE short, specific question about the pursuit itself" in fact
    assert "do NOT frame it as needing company" in fact
    assert "do NOT apologize for asking" in fact
    # Declining is complete, and there is no second attempt.
    assert "that is a complete answer" in fact
    assert "never ask again, never bring it up later" in fact


def test_no_thread_means_no_fact() -> None:
    assert ask_thread_fact(None) is None


# ---------------------------------------------------------------------------
# The answer is one-shot, and non-answers are unrecorded
# ---------------------------------------------------------------------------


def test_user_input_is_consumed_by_exactly_one_progression(tmp_path) -> None:
    store = _store(tmp_path, _thread())
    topic = "re-running the retraction-bait stats"
    assert store.record_user_input(topic, "I'd drop the outliers") is True
    assert store.active_threads()[0].pending_input == "I'd drop the outliers"

    store.record_progress(
        topic,
        summary="Re-ran it without the outliers.",
        step_text="Re-ran it without the outliers.",
        today=dt.date(2026, 8, 8),
    )

    assert store.active_threads()[0].pending_input == ""


def test_nothing_records_that_an_ask_went_unanswered(tmp_path) -> None:
    """THE ANTI-SULK CONTRACT. There is no field distinguishing "they said
    nothing" from "already used" — so no code can ever act on the
    difference, however a future prompt might be tempted to."""
    store = _store(tmp_path, _thread())
    topic = "re-running the retraction-bait stats"
    store.mark_asked(topic, dt.date(2026, 8, 7))

    (thread,) = store.active_threads()
    assert thread.pending_input == ""
    assert set(thread.to_dict()) == {
        "topic",
        "summary",
        "started_at",
        "last_progressed_at",
        "progressions",
        "status",
        "history",
        "asked_at",
        "pending_input",
    }


def test_blank_user_input_is_not_recorded(tmp_path) -> None:
    store = _store(tmp_path, _thread())
    assert (
        store.record_user_input("re-running the retraction-bait stats", "   ") is False
    )


def test_ask_state_survives_a_reload(tmp_path) -> None:
    path = tmp_path / "life.json"
    store = LifeThreadStore(str(path))
    store.add_thread(_thread(), dt.date(2026, 8, 1))
    store.mark_asked("re-running the retraction-bait stats", dt.date(2026, 8, 7))
    store.record_user_input("re-running the retraction-bait stats", "drop them")

    reloaded = LifeThreadStore(str(path))
    (thread,) = reloaded.active_threads()
    assert thread.asked_at == "2026-08-07"
    assert thread.pending_input == "drop them"
    assert reloaded.last_ask_date() == "2026-08-07"


# ---------------------------------------------------------------------------
# Wiring: idle chatter surfaces it; the next turn captures the reply
# ---------------------------------------------------------------------------

pytest.importorskip("fastapi")

from amadeus.config import Config  # noqa: E402
from amadeus.engine import ChatEngine  # noqa: E402


class _IdleProvider:
    name = "idle"

    def __init__(self) -> None:
        self.calls: list[list[ChatMessage]] = []

    def generate(self, messages, **kwargs):
        self.calls.append(list(messages))
        return "Does dropping the outliers read as defensible to you?"

    def stream(self, messages, **kwargs):
        yield self.generate(messages, **kwargs)


def _engine(tmp_path, *, asks: bool) -> tuple[ChatEngine, _IdleProvider]:
    config = Config()
    config.memory.persist_path = str(tmp_path / "memory.json")
    config.memory.embeddings_path = str(tmp_path / "embeddings.json")
    config.memory.profile_path = str(tmp_path / "profile.json")
    config.memory.enable_idle_chatter = True
    config.memory.idle_cooldown_seconds = 0
    config.sleep_schedule.enabled = False
    config.life_threads.enable_opinion_asks = asks
    provider = _IdleProvider()
    engine = ChatEngine(config, provider)
    engine.summary_provider = provider
    engine.life_store = _store(tmp_path, _thread())
    # idle_chatter only fires mid-conversation (a fresh session gets the
    # greeting instead), so give it something to have been quiet after.
    engine.memory.add_message("user", "anyway, that was the plan")
    engine.memory.add_message("assistant", "Right.")
    return engine, provider


def test_idle_chatter_surfaces_a_due_ask(tmp_path) -> None:
    engine, provider = _engine(tmp_path, asks=True)

    line = engine.idle_chatter()

    assert line is not None
    joined = "\n".join(m.content for call in provider.calls for m in call)
    assert "you want their take" in joined
    assert telemetry.snapshot().get("life.ask_surfaced") == 1
    # Marked only after the line existed, and one-way from here.
    assert engine.life_store.active_threads()[0].asked_at != ""


def test_asks_are_off_by_default(tmp_path) -> None:
    assert Config().life_threads.enable_opinion_asks is False
    engine, provider = _engine(tmp_path, asks=False)

    engine.idle_chatter()

    joined = "\n".join(m.content for call in provider.calls for m in call)
    assert "you want their take" not in joined
    assert engine.life_store.active_threads()[0].asked_at == ""


def test_the_next_user_turn_is_captured_as_the_answer(tmp_path) -> None:
    engine, _ = _engine(tmp_path, asks=True)
    engine.idle_chatter()

    from amadeus.persona import PersonaSettings

    engine.prepare_turn(
        "I'd drop them, the tail is doing all the work", PersonaSettings()
    )

    (thread,) = engine.life_store.active_threads()
    assert thread.pending_input.startswith("I'd drop them")
    assert telemetry.snapshot().get("life.ask_answered") == 1


def test_only_the_immediately_next_turn_counts(tmp_path) -> None:
    """The arm is one-shot: a later unrelated turn must not be swept up as
    an answer to a question two topics ago."""
    engine, _ = _engine(tmp_path, asks=True)
    engine.idle_chatter()

    from amadeus.persona import PersonaSettings

    engine.prepare_turn("sure, drop them", PersonaSettings())
    engine.prepare_turn("anyway, what's the weather", PersonaSettings())

    (thread,) = engine.life_store.active_threads()
    assert thread.pending_input == "sure, drop them"
