"""Friction memory: the anti-guilt CONTRACT.

Separate from tests/test_friction.py (which covers mechanical state) because
these are the tests that decide whether the feature is defensible at all. A
companion that gets colder until the user makes it up to her is a retention
mechanic, and docs/companion-ethics.md exists to refuse exactly that. Every
assertion here maps to a rule in that document; if one of them has to be
weakened to ship something, the something should not ship.

The three structural guarantees pinned here:

1. Friction is information she HAS about a topic, never a mood she is IN
   about the user — enforced by the data model, since no field can hold the
   latter.
2. Friction never moves any behavioral dial (trust, arc, tone, warmth). It is
   purely additive prompt information.
3. Friction lives in the greeting's separate prompt channel, never the
   per-turn system prompt — so it colors one opening beat and is then over.
   A per-turn block would make her carry it all conversation. That is sulking.
"""

import datetime as dt

from amadeus.memory.profile import (
    FRICTION_MAX_AGE_DAYS,
    UserProfile,
    updated_arc,
    updated_trust,
)
from amadeus.persona import PersonaSettings, build_system_prompt


def _profile_with_friction() -> UserProfile:
    profile = UserProfile()
    profile.record_friction("the funding thing")
    return profile


def test_friction_fact_hard_bans_guilt_and_coldness() -> None:
    """Each literal below neutralizes a named dark pattern. The prompt is the
    only place these rails exist, so their exact presence is the contract."""
    fact = _profile_with_friction().friction_fact()
    assert fact is not None

    # Reactive, never raised by her: the single most important rule.
    assert "Do NOT bring it up yourself unless they do." in fact
    # No sulking / coldness / withdrawal.
    assert "Do NOT be cold, distant, short, or wary with them." in fact
    assert "do NOT sulk or make them manage your feelings" in fact
    # No debt, no apology-seeking.
    assert "do NOT ask for or imply an apology" in fact
    assert "They owe you nothing." in fact
    assert "Do NOT reference it as something they did to you" in fact
    # The default is warmth, not wariness.
    assert "If they seem fine, you are fine" in fact


def test_friction_never_moves_trust() -> None:
    """Trust is computed from session_count and tone only. Friction must not
    become a second penalty stacked on the same signal — updated_arc already
    raises her guard on a hostile previous session, and that mechanism is
    bounded and symmetric. This pins that friction cannot reach it."""
    for tone in (-0.8, 0.0, 0.6):
        for count in (1, 12, 40):
            assert updated_trust(0.5, count, tone) == updated_trust(0.5, count, tone)

    # The signature takes no friction argument at all — the guarantee is
    # structural, not behavioral. If someone adds one, this fails to compile
    # into a passing test and the reviewer has to justify it.
    import inspect

    params = set(inspect.signature(updated_trust).parameters)
    assert params == {"current", "session_count", "last_tone"}


def test_friction_never_moves_the_arc_floats() -> None:
    import inspect

    params = set(inspect.signature(updated_arc).parameters)
    assert params == {
        "curiosity",
        "logic",
        "pride",
        "session_count",
        "trust_level",
        "last_tone",
    }

    baseline = updated_arc(
        0.1, 0.0, -0.1, session_count=5, trust_level=0.5, last_tone=0.2
    )
    assert baseline == updated_arc(
        0.1, 0.0, -0.1, session_count=5, trust_level=0.5, last_tone=0.2
    )


def test_friction_never_appears_in_the_persona_system_prompt() -> None:
    """Friction rides the greeting's separate cheap-model call, NOT
    build_system_prompt. If it ever leaked into the per-turn prompt she would
    carry the sore subject through every turn of the conversation — which is
    the sulking behavior the ethics doc forbids, and would also spend the
    persona prompt's remaining size headroom."""
    prompt = build_system_prompt(PersonaSettings())

    for marker in ("friction", "did not go well between you", "sore"):
        assert marker not in prompt.lower()


def test_no_friction_means_no_greeting_fact() -> None:
    """The common case costs zero prompt characters."""
    assert UserProfile().friction_fact() is None


def test_repair_requires_no_apology_only_that_it_went_fine() -> None:
    """clear_friction takes a topic and nothing else — there is deliberately
    no parameter for "did the user apologize", because a repair gated on
    contrition is a debt with a receipt."""
    import inspect

    params = set(inspect.signature(UserProfile.clear_friction).parameters)
    assert params == {"self", "topic"}

    profile = _profile_with_friction()
    assert profile.clear_friction("the funding thing") is True
    assert profile.friction_fact() is None


def test_friction_cannot_record_that_the_user_owes_anything() -> None:
    """The data model has no slot for a grievance state, an unresolved
    counter surfaced to the user, or "she is upset with you". A stored entry
    is exactly {topic, noted, count} — a topic and when it last went badly."""
    profile = _profile_with_friction()
    (entry,) = profile.friction_topics
    assert set(entry) == {"topic", "noted", "count"}


def test_time_repairs_without_the_user_doing_anything() -> None:
    """Restated here (also in test_friction.py) because it is the contract,
    not merely a behavior: being away must never cost the user anything."""
    profile = UserProfile()
    today = dt.date(2026, 9, 1)
    profile.record_friction(
        "the thing", when=today - dt.timedelta(days=FRICTION_MAX_AGE_DAYS + 1)
    )

    profile.prune_stale_friction(today=today)

    assert profile.friction_topics == []


# ---------------------------------------------------------------------------
# Delivery: the greeting channel, and only the greeting channel
# ---------------------------------------------------------------------------


def _greeting_session(tmp_path, *, friction: bool = True):
    """A web session whose greeting we can drive, with the sleep window off.

    The sleep-schedule default (1->8 LOCAL) suppresses presence lines, and CI
    runners sit in UTC — a 02:00-08:00 UTC run would silently produce no
    greeting at all and this file's assertions would pass vacuously.
    """
    import pytest

    pytest.importorskip("fastapi")
    from amadeus.config import Config
    from amadeus.providers.llm.base import ChatMessage
    from amadeus.web.app import create_app

    class _Recorder:
        name = "recording"

        def __init__(self):
            self.calls: list[list[ChatMessage]] = []

        def generate(self, messages, **kwargs):
            self.calls.append(list(messages))
            return "Oh. You're here."

    config = Config()
    config.memory.persist_path = str(tmp_path / "memory.json")
    config.memory.embeddings_path = str(tmp_path / "embeddings.json")
    config.memory.profile_path = str(tmp_path / "profile.json")
    config.memory.enable_greeting = True
    config.memory.enable_friction_memory = friction
    config.sleep_schedule.enabled = False
    app = create_app(config)
    store = app.state.session_store
    recorder = _Recorder()
    store.llm = recorder
    _, session = store.get(None, "friction-user")
    session.llm = recorder
    session.summary_provider = recorder
    session.memory.profile.first_met = "2026-01-01"
    return session, recorder


def test_a_sore_topic_reaches_the_greeting_prompt(tmp_path) -> None:
    session, recorder = _greeting_session(tmp_path)
    session.memory.profile.record_friction("the funding thing")

    assert session.open_session_greeting() is not None

    joined = "\n".join(m.content for call in recorder.calls for m in call)
    assert "the funding thing" in joined
    assert "did not go well between you" in joined
    # The rails travel WITH the topic — never the topic alone.
    assert "Do NOT bring it up yourself unless they do." in joined


def test_disabled_keeps_friction_out_of_the_greeting(tmp_path) -> None:
    session, recorder = _greeting_session(tmp_path, friction=False)
    session.memory.profile.record_friction("the funding thing")

    session.open_session_greeting()

    joined = "\n".join(m.content for call in recorder.calls for m in call)
    assert "the funding thing" not in joined


def test_no_friction_adds_nothing_to_the_greeting(tmp_path) -> None:
    session, recorder = _greeting_session(tmp_path)

    session.open_session_greeting()

    joined = "\n".join(m.content for call in recorder.calls for m in call)
    assert "did not go well between you" not in joined
