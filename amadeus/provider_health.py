"""Process-wide LLM and TTS provider health classification.

Companion to ``telemetry.py`` (counters answer "how often has X happened";
this answers "what state is the provider in right now"). A quota death at
4pm and a bad key look identical to the user — both just make her stop
thinking clearly — but they need different fixes (wait, vs. go to Settings),
so ``/api/status`` needs a machine-readable field the frontend can key a
distinct message off of instead of just "something's wrong". The same
reasoning applies to TTS: "voice sometimes doesn't work" took 5 separate
root causes to run down (PR #13) precisely because nothing stated plainly
that synthesis was failing repeatedly — only scattered counters
(``tts.stream_synth_failures`` et al.) and a ``warming`` flag existed.

Deliberately flat module-level records, not per-session: both the LLM and
TTS provider instances are process-wide (``SessionStore.llm``/``.tts``,
``ChatEngine.llm``), so their health is too. Self-healing by construction —
recording success on every successful call means a degraded/failing state
clears itself the moment a call succeeds again (the free-tier quota window
rolling over at midnight, a key getting fixed in Settings, a transient CUDA
OOM going away) without needing a separate reset path.
"""

from __future__ import annotations

import datetime as dt
import threading
from typing import Dict, Optional

# ok: last call succeeded (or nothing has been tried yet — the default).
# rate_limited: the provider is refusing calls for 429/quota reasons.
# invalid_key: the provider is refusing calls for 401/403 (bad credentials).
# error: a call failed for some other reason (timeout, network, 5xx, ...).
# dummy: no real provider is configured/loaded at all — every "call" is
#        canned text, so rate_limited/invalid_key/error can never apply.
_STATES = frozenset({"ok", "rate_limited", "invalid_key", "error", "dummy"})


def _stamp() -> str:
    """Now, as the naive-UTC ISO string with a ``Z`` suffix these records use.

    ``datetime.utcnow()`` is deprecated and scheduled for removal, but its
    naive-UTC output is what every ``since`` value has always been, so the
    aware clock is read and the tzinfo dropped rather than switching these
    to offset-bearing strings. ``since`` is display-only (nothing parses or
    compares it), so this is a pure deprecation fix with byte-identical
    output.
    """
    return dt.datetime.now(dt.timezone.utc).replace(tzinfo=None).isoformat() + "Z"


_LOCK = threading.Lock()
_STATE: Dict[str, object] = {
    "state": "ok",
    "detail": None,
    "since": None,
}

# ok: last synthesis succeeded (or nothing has been tried yet — the default).
# warming: the provider is still loading (model download/CUDA warmup) — not
#          a failure, just not ready yet; mirrors the pre-existing
#          store.tts_warming flag rather than replacing it.
# degraded: at least one synthesis has failed since the last success, but
#           not (yet) enough in a row to call it "failing".
# failing: _TTS_FAILING_THRESHOLD or more CONSECUTIVE synthesis failures with
#          no intervening success — the voice is effectively dead, not just
#          having a bad moment.
# off: no TTS provider is loaded and it isn't warming either — voice is
#      disabled/unconfigured, which is a normal, non-alarming state (unlike
#      "failing").
_TTS_STATES = frozenset({"ok", "warming", "degraded", "failing", "off"})
_TTS_FAILING_THRESHOLD = 3

_TTS_LOCK = threading.Lock()
_TTS_STATE: Dict[str, object] = {
    "state": "ok",
    "detail": None,
    "since": None,
    "consecutive_failures": 0,
}


def set_llm_health(state: str, detail: Optional[str] = None) -> None:
    """Record the LLM provider's current health state.

    ``state`` must be one of the values in ``_STATES``; an unrecognized
    value is coerced to ``"error"`` with the bad value folded into
    ``detail`` rather than raising — health reporting must never be able to
    crash the call site that's already handling a real provider failure.
    ``since`` is stamped every call (not just on a state change) so
    "how long has it been like this" and "how fresh is this reading" are the
    same timestamp — the simplest thing that answers both.
    """
    if state not in _STATES:
        detail = f"(unrecognized state {state!r}) {detail or ''}".strip()
        state = "error"
    with _LOCK:
        _STATE["state"] = state
        _STATE["detail"] = detail
        _STATE["since"] = _stamp()


def llm_health_snapshot() -> Dict[str, object]:
    """Copy of the current health record, for GET /api/status's ``llm_health``."""
    with _LOCK:
        return dict(_STATE)


def reset() -> None:
    """Reset to the default ok state (tests only)."""
    with _LOCK:
        _STATE["state"] = "ok"
        _STATE["detail"] = None
        _STATE["since"] = None
    reset_tts()


def reset_tts() -> None:
    """Reset TTS health to the default ok state, zero failure streak (tests
    only). Split from ``reset()`` so a test that only cares about one of the
    two health records doesn't have to reason about the other, but ``reset()``
    still clears both for callers that just want a clean slate."""
    with _TTS_LOCK:
        _TTS_STATE["state"] = "ok"
        _TTS_STATE["detail"] = None
        _TTS_STATE["since"] = None
        _TTS_STATE["consecutive_failures"] = 0


def record_llm_success(llm: object) -> None:
    """Self-heal to "ok" after a successful LLM call — unless the "call"
    was a DummyProvider canned reply.

    Every success call site used to stamp ``set_llm_health("ok")``
    directly, which meant one chat turn against the dummy fallback flipped
    the state from "dummy" back to "ok" and defeated the whole "check
    /api/status for silently-dummy assistants" contract. A dummy reply is
    not evidence of provider health; keep the dummy stamp until a real
    provider actually answers.

    Exact-type check on purpose: the production fallback is always a bare
    ``DummyProvider()`` (see provider_loader), while test harnesses
    subclass DummyProvider to build scripted stand-ins for REAL providers
    — those should keep healing to ok like the providers they imitate.
    """
    from .providers.llm.base import DummyProvider

    if type(llm) is DummyProvider:
        return
    set_llm_health("ok")


def classify_and_record(exc: BaseException) -> str:
    """Classify a provider-call exception, record it as the current health
    state, and return the classification.

    Reuses the two existing, already-tested detectors rather than growing a
    third pattern list: ``_is_rate_limit`` (structured SDK exception checks
    for Gemini ResourceExhausted / OpenAI RateLimitError / generic HTTP 429 —
    ``providers/llm/fallback.py``, the same classifier the rate-limit
    failover and the WS/POST error-frame ``kind`` field use) and
    ``_is_auth_error`` (English-marker check for 401/403-shaped messages —
    ``agents/tool_loop.py``, the same classifier ``ToolLoopAuthError`` uses).
    Imported lazily to avoid adding a module-load-order dependency for a
    function that only runs inside an already-caught exception handler.
    """
    from .agents.tool_loop import _is_auth_error
    from .providers.llm.fallback import _is_rate_limit

    try:
        if _is_rate_limit(exc):
            set_llm_health("rate_limited", str(exc))
            return "rate_limited"
        if _is_auth_error(exc):
            set_llm_health("invalid_key", str(exc))
            return "invalid_key"
    except Exception:  # pragma: no cover - classifiers must never break this
        pass
    set_llm_health("error", str(exc))
    return "error"


# ---------------------------------------------------------------------------
# TTS health
# ---------------------------------------------------------------------------
#
# This is an AGGREGATE layer over the pre-existing scattered signals — it
# does not replace store.tts_warming, the tts.synth_failures/
# tts.stream_synth_failures/tts.deferred_synth_failures/
# tts.presence_synth_failures telemetry counters, or /api/status's
# tts.error/tts.fallback_from fields. Those keep answering "how often" and
# "what happened last"; tts_health answers "is voice OK right now", the same
# job llm_health does for the LLM side.


def set_tts_health(state: str, detail: Optional[str] = None) -> None:
    """Record the TTS provider's current health state directly.

    Like ``set_llm_health``, an unrecognized ``state`` is coerced to
    ``"error"``... except TTS has no ``"error"`` state in its vocabulary, so
    it coerces to ``"degraded"`` instead (the closest "something's wrong but
    not necessarily dead" analogue) with the bad value folded into
    ``detail``. Health reporting must never be able to crash the call site
    that's already handling a real synthesis failure.
    """
    if state not in _TTS_STATES:
        detail = f"(unrecognized state {state!r}) {detail or ''}".strip()
        state = "degraded"
    with _TTS_LOCK:
        _TTS_STATE["state"] = state
        _TTS_STATE["detail"] = detail
        _TTS_STATE["since"] = _stamp()
        if state != "failing" and state != "degraded":
            _TTS_STATE["consecutive_failures"] = 0


def tts_health_snapshot(
    *, loaded: bool = True, warming: bool = False
) -> Dict[str, object]:
    """Copy of the current TTS health record, for GET /api/status's
    ``tts_health`` — layering the live loaded/warming signals over the
    recorded synthesis-outcome state rather than requiring every one of
    ``SessionStore``'s several ``tts_warming`` mutation sites (constructor,
    ``reload_providers``, ``_warm_tts_in_background``) to also push into
    this module. The caller already reads ``store.tts``/``store.tts_warming``
    under its own lock for the rest of the /api/status TTS block, so passing
    them here is one extra pair of booleans, not a second source of truth.

    Precedence, most to least urgent:
    1. ``warming=True`` -> "warming" (still loading; not a failure, and a
       consecutive-failure streak from a *previous* provider instance
       shouldn't read as "failing" while a fresh one is coming up).
    2. ``loaded=False`` and not warming -> "off" (no provider at all — a
       normal state, e.g. voice disabled — never "failing").
    3. Otherwise, the recorded state from ``record_tts_success``/
       ``record_tts_failure`` (ok/degraded/failing) or ``set_tts_health``.

    ``consecutive_failures`` is internal bookkeeping but cheap and harmless
    to expose — useful for "how close to failing" without a separate
    endpoint.
    """
    with _TTS_LOCK:
        snap = dict(_TTS_STATE)
    if warming:
        snap["state"] = "warming"
        snap["detail"] = None
    elif not loaded:
        snap["state"] = "off"
        snap["detail"] = None
    return snap


def record_tts_success(provider: object = None) -> None:
    """Self-heal to "ok" after a successful synthesis call.

    ``provider`` is accepted (and ignored) for symmetry with
    ``record_llm_success(llm)`` and in case a future dummy/no-op TTS
    provider needs the same exact-type carve-out the LLM side has — no such
    provider exists today (TTS is simply ``None`` when unconfigured, handled
    by ``tts_health_snapshot(loaded=False)``), so there is nothing to
    special-case yet.
    """
    with _TTS_LOCK:
        _TTS_STATE["state"] = "ok"
        _TTS_STATE["detail"] = None
        _TTS_STATE["since"] = _stamp()
        _TTS_STATE["consecutive_failures"] = 0


def record_tts_failure(exc_or_reason: object) -> str:
    """Record one synthesis failure and classify the resulting state.

    Bumps an internal consecutive-failure counter (reset by
    ``record_tts_success``): the first failure after a healthy streak
    becomes ``"degraded"``, and once ``_TTS_FAILING_THRESHOLD`` (3) failures
    have landed in a row with no success in between, the state escalates to
    ``"failing"`` — a single bad synthesis (network blip, one malformed
    sentence) shouldn't read the same as the voice being persistently dead.
    Accepts either an exception or a plain string reason since call sites
    range from a caught ``Exception`` to a generator that produced zero
    chunks (no exception at all, just an empty result).
    """
    detail = str(exc_or_reason) if exc_or_reason is not None else None
    with _TTS_LOCK:
        count = int(_TTS_STATE.get("consecutive_failures") or 0) + 1
        _TTS_STATE["consecutive_failures"] = count
        state = "failing" if count >= _TTS_FAILING_THRESHOLD else "degraded"
        _TTS_STATE["state"] = state
        _TTS_STATE["detail"] = detail
        _TTS_STATE["since"] = _stamp()
        return state
