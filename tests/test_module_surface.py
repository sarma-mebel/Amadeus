"""Drift guard for the free-function satellite modules of ChatEngine.

co_activity / next_time / greeting / presence / safety follow the
free-function-over-owner pattern: engine.py holds one-line delegates and
tests mostly exercise them THROUGH the facade. If a rename or extraction
ever breaks these modules' own public surface, the facade tests can keep
passing against stale re-exports while direct importers (routes, scheduler,
future callers) break. This file imports each module directly and pins its
key callables.
"""

from __future__ import annotations

import inspect

import amadeus.co_activity as co_activity
import amadeus.greeting as greeting
import amadeus.next_time as next_time
import amadeus.presence as presence
import amadeus.safety as safety


def _assert_functions(module, names):
    for name in names:
        obj = getattr(module, name, None)
        assert obj is not None, f"{module.__name__}.{name} is missing"
        assert callable(obj), f"{module.__name__}.{name} is not callable"


def test_co_activity_surface():
    _assert_functions(
        co_activity,
        ["read_along_page", "read_along_end", "watch_frame", "watch_end"],
    )
    # Both activity pairs take the engine as first argument (the pattern
    # every caller and delegate relies on).
    for name in ("read_along_page", "read_along_end", "watch_frame", "watch_end"):
        first_param = next(
            iter(inspect.signature(getattr(co_activity, name)).parameters)
        )
        assert first_param == "engine"


def test_next_time_surface():
    _assert_functions(next_time, ["note_next_time_thread", "next_time_greeting_fact"])


def test_greeting_surface():
    _assert_functions(greeting, ["open_session_greeting"])
    first_param = next(
        iter(inspect.signature(greeting.open_session_greeting).parameters)
    )
    assert first_param == "engine"


def test_presence_surface():
    _assert_functions(presence, ["poke_reaction", "idle_chatter", "screen_comment"])
    for name in ("poke_reaction", "idle_chatter", "screen_comment"):
        first_param = next(iter(inspect.signature(getattr(presence, name)).parameters))
        assert first_param == "engine"


def test_safety_surface_and_behavior():
    _assert_functions(safety, ["guardrail_response"])
    # safety is pure and engine-free — cheap to verify behavior directly.
    blocked = safety.guardrail_response("what's my credit card number?")
    assert isinstance(blocked, str) and blocked
    assert safety.guardrail_response("tell me about time travel") is None


def test_engine_facade_binds_the_same_module_objects():
    # The facade's delegates must resolve to THESE modules/functions, not
    # private copies — identity, not just importability.
    import amadeus.engine as engine

    assert engine.presence is presence
    assert engine.greeting is greeting
    assert engine.co_activity is co_activity
    assert engine.note_next_time_thread is next_time.note_next_time_thread
    assert engine.guardrail_response is safety.guardrail_response
