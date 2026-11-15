"""First-run self-test: "is she actually working?", answered leg by leg.

The repo already degrades loudly — to *us*. Every silent fallback counts a
telemetry counter and every provider carries a health state, all of it on
``GET /api/status``. Nobody reads counters. The failure mode this closes is
the one the project notes names outright: "an assistant that 'works' may be running
on dummies", where a misconfigured install presents as "she's oddly bland"
rather than "your API key never loaded". Every quality feature in the repo
is invisible behind that.

So this is the last mile of degrade-loudly: run the real path end to end,
report each leg pass/warn/fail, and — the part that matters — say what to
DO about it. A leg that fails without an actionable fix is just a counter
with nicer formatting.

Distinct from ``settings_api.run_provider_test``, which builds a CANDIDATE
config from unsaved form fields to answer "would this key work?" before
saving. This runs the LIVE, wired providers the session is actually using,
which is the only thing that answers "is what I'm running right now
healthy?" — a saved-but-unloaded provider, a TTS that fell back to CPU, or
a voice that died after warmup are all invisible to a candidate test.

Contract for every leg:

- **Never raises.** A leg that blows up is a FAIL with the exception text,
  never a 500 — a broken self-test that cannot report is the worst outcome.
- **Read-only against user state.** No memory writes, no session mutation,
  no store persistence. The memory leg uses an isolated temp store.
- **Wall-clocked.** Each leg reports its duration and a slow PASS is
  downgraded to WARN. This is reporting, not cancellation — see
  ``_SLOW_LEG_SECONDS``.
- **Actionable.** ``fix`` is the specific next step, not "check your
  configuration".
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from .. import telemetry
from ..logging_utils import get_logger

LOGGER = get_logger(__name__)

# Above this, a PASSING leg is downgraded to WARN: it worked, but slowly
# enough that the user would experience it as broken. Generous on purpose —
# a cold TTS load or a first-call model download legitimately takes tens of
# seconds, and calling that a failure would teach users to distrust the test.
#
# NOT a timeout: legs are blocking provider calls with no cancellation
# point, so this cannot interrupt a hang, only report one after the fact.
# A genuinely wedged provider hangs the request; the honest fix would be a
# per-leg worker with a hard deadline, which is more machinery than the
# failure mode has earned so far.
_SLOW_LEG_SECONDS = 90.0

PASS = "pass"
WARN = "warn"
FAIL = "fail"
SKIP = "skip"


@dataclass
class LegResult:
    """One checked capability."""

    key: str
    label: str
    status: str
    detail: str
    fix: Optional[str] = None
    duration_ms: Optional[int] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "key": self.key,
            "label": self.label,
            "status": self.status,
            "detail": self.detail,
        }
        if self.fix:
            payload["fix"] = self.fix
        if self.duration_ms is not None:
            payload["duration_ms"] = self.duration_ms
        if self.extra:
            payload.update(self.extra)
        return payload


def _run_leg(key: str, label: str, fn: Callable[[], LegResult]) -> LegResult:
    """Execute one leg, converting any escape into a FAIL result.

    The broad except is the point, not an oversight: this function's whole
    job is that a self-test never becomes another silent failure.
    """
    started = time.monotonic()
    try:
        result = fn()
    except Exception as exc:  # noqa: BLE001 - reported verbatim to the user
        LOGGER.warning("Self-test leg %s failed: %s", key, exc)
        telemetry.incr("self_test.leg_error")
        result = LegResult(
            key=key,
            label=label,
            status=FAIL,
            detail=str(exc)[:300] or exc.__class__.__name__,
            fix="Unexpected error — check the backend logs for a traceback.",
        )
    elapsed_ms = int((time.monotonic() - started) * 1000)
    if result.duration_ms is None:
        result.duration_ms = elapsed_ms
    if elapsed_ms > _SLOW_LEG_SECONDS * 1000 and result.status == PASS:
        # Completed, but slowly enough that the user would experience it as
        # broken. Downgrade rather than pass silently.
        result.status = WARN
        result.detail = f"{result.detail} (took {elapsed_ms / 1000:.0f}s)"
        result.fix = result.fix or (
            "This leg works but is very slow — the first call often loads a "
            "model; re-run to see if it is warm now."
        )
    return result


# --- Legs -------------------------------------------------------------------


def check_llm(store: Any) -> LegResult:
    """Round-trip the LIVE chat provider.

    The dummy check is the reason this leg exists: a keyless install answers
    every message with canned text and looks like a working assistant.
    """
    label = "Chat model"
    llm = getattr(store, "llm", None)
    if llm is None:
        return LegResult(
            key="llm",
            label=label,
            status=FAIL,
            detail="No chat provider is loaded.",
            fix="Set providers.llm in your config, then restart.",
        )
    name = llm.__class__.__name__
    if name == "DummyProvider":
        return LegResult(
            key="llm",
            label=label,
            status=FAIL,
            detail=(
                "Running on the dummy provider — replies are canned, not "
                "generated. This is what a missing or rejected API key looks "
                "like."
            ),
            fix=(
                "Add your API key (Settings -> Providers, or the "
                "OPENAI_API_KEY / GEMINI_API_KEY env var) and restart. "
                "Settings -> Providers -> Test checks a key before saving."
            ),
        )
    from ..providers.llm.base import ChatMessage

    reply = llm.generate([ChatMessage(role="user", content="Reply with: OK")])
    if not reply or not str(reply).strip():
        return LegResult(
            key="llm",
            label=label,
            status=FAIL,
            detail=f"{name} returned an empty response.",
            fix=(
                "The provider accepted the request but sent nothing back — "
                "check the model name in Settings -> Providers and your "
                "account's quota."
            ),
        )
    return LegResult(
        key="llm",
        label=label,
        status=PASS,
        detail=f"{name} replied.",
        extra={"provider": name},
    )


def check_tts(store: Any) -> LegResult:
    """Synthesize a real clip on the LIVE voice provider.

    Returns the audio so the caller can PLAY it: "the endpoint returned
    bytes" and "she sounds right" are different claims, and only the user's
    ears can settle the second one.
    """
    label = "Voice (text to speech)"
    config = getattr(store, "config", None)
    configured = getattr(getattr(config, "providers", None), "tts", None)
    if configured in (None, "", "dummy"):
        return LegResult(
            key="tts",
            label=label,
            status=SKIP,
            detail="Voice output is turned off in config.",
            fix="Set providers.tts if you want her to speak.",
        )
    # SessionStore exposes TTS as an attribute (loaded/warmed in the
    # background), unlike STT which has a lazy get_stt() loader.
    tts = getattr(store, "tts", None)
    if tts is None:
        return LegResult(
            key="tts",
            label=label,
            status=FAIL,
            detail=f"Voice provider '{configured}' did not load.",
            fix=(
                f"Install the extra for '{configured}' (see docs/dev-setup.md) "
                "and restart. Until then she is silent, not broken."
            ),
        )
    name = tts.__class__.__name__
    audio = tts.synthesize("Test.", speaking_rate=1.0, pitch_shift=0.0)
    if not audio:
        return LegResult(
            key="tts",
            label=label,
            status=FAIL,
            detail=f"{name} returned no audio.",
            fix=(
                "The provider loaded but produced nothing — check the backend "
                "logs for a synthesis error, and for qwen3 that the reference "
                "clip exists."
            ),
        )
    return LegResult(
        key="tts",
        label=label,
        status=PASS,
        detail=f"{name} produced {len(audio)} bytes. Press play to hear her.",
        extra={"provider": name, "audio_bytes": len(audio)},
    )


def check_stt(store: Any) -> LegResult:
    """Confirm the mic transcription path is loadable.

    Deliberately does NOT synthesize speech to transcribe: a TTS clip fed
    back through STT tests the pair, not the leg, and a failure would be
    unattributable. Loading is what distinguishes "configured" from "will
    actually work when you hold the button".
    """
    label = "Microphone (speech to text)"
    config = getattr(store, "config", None)
    configured = getattr(getattr(config, "providers", None), "stt", None)
    if configured in (None, "", "dummy"):
        return LegResult(
            key="stt",
            label=label,
            status=SKIP,
            detail="Speech input is turned off in config.",
            fix="Set providers.stt to talk to her instead of typing.",
        )
    stt = store.get_stt()
    if stt is None:
        return LegResult(
            key="stt",
            label=label,
            status=FAIL,
            detail=f"Speech provider '{configured}' did not load.",
            fix=(
                f"Install the extra for '{configured}' (pip install "
                '".[voice]") and restart. Typing still works meanwhile.'
            ),
        )
    # SessionStore wraps STT in _SpeculativeSTT (speculative recall); report
    # the provider the user configured, not our internal decorator.
    # NOTE: _SpeculativeSTT.__getattr__ proxies unknown attributes to the
    # wrapped provider, so a missing-attribute guess here would silently
    # resolve to something else entirely — the attribute name is `provider`.
    inner = getattr(stt, "provider", None)
    name = (inner if inner is not None else stt).__class__.__name__
    return LegResult(
        key="stt",
        label=label,
        status=PASS,
        detail=f"{name} is loaded.",
        extra={"provider": name},
    )


def check_memory(store: Any) -> LegResult:
    """Write a fact to an ISOLATED store and read it back.

    Never touches the user's real memory: this runs against a temp path, so
    a self-test can never pollute what she remembers. It answers "would a
    fact survive?", which covers the embedder-dimension mismatch that
    silently disables recall after an embedder switch.
    """
    import tempfile
    from pathlib import Path

    label = "Memory"
    config = getattr(store, "config", None)
    if config is None or not getattr(config.memory, "enable_long_term", False):
        return LegResult(
            key="memory",
            label=label,
            status=SKIP,
            detail="Long-term memory is turned off in config.",
            fix="Set memory.enable_long_term to have her remember across sessions.",
        )
    import copy

    from ..memory.manager import MemoryManager

    with tempfile.TemporaryDirectory() as tmp:
        probe_config = copy.deepcopy(config)
        probe_config.memory.persist_path = None
        probe_config.memory.profile_path = str(Path(tmp) / "profile.json")
        probe_config.memory.embeddings_path = str(Path(tmp) / "vectors.json")
        manager = MemoryManager(probe_config, embedder=store.embedder)
        if manager.embedder is None or manager.vector_store is None:
            return LegResult(
                key="memory",
                label=label,
                status=FAIL,
                detail="No embedder is loaded, so nothing can be recalled.",
                fix=(
                    "Check memory.embedder in your config and that its extra "
                    'is installed (pip install ".[memory]").'
                ),
            )
        probe = "The self-test canary fact about aardvarks."
        manager.store_user_fact(probe, salience=0.9)
        # The query must clear recall_min_query_chars (default 10) or recall
        # skips the embed entirely and returns nothing — a short probe made
        # this leg report a FALSE failure on every healthy install, telling
        # users to run a migration they did not need.
        matches = manager.recall_detailed("the aardvarks canary fact", limit=3)
        if not any(probe in (m.text or "") for m in matches):
            return LegResult(
                key="memory",
                label=label,
                status=FAIL,
                detail="A fact was stored but could not be recalled.",
                fix=(
                    "Recall is not returning stored facts. If you recently "
                    "switched embedders, run scripts/migrate_memory.py — a "
                    "vector-dimension mismatch disables recall silently."
                ),
                extra={"embedder": manager.embedder.__class__.__name__},
            )
    return LegResult(
        key="memory",
        label=label,
        status=PASS,
        detail="Stored a fact and recalled it.",
        extra={"embedder": store.embedder.__class__.__name__},
    )


def _summarize(legs: List[LegResult]) -> Dict[str, Any]:
    """Headline verdict plus the one thing to fix first."""
    failures = [leg for leg in legs if leg.status == FAIL]
    warnings = [leg for leg in legs if leg.status == WARN]
    if failures:
        status = FAIL
        # Leg order is capability order (chat before voice before mic), so
        # the first failure is the most fundamental one — fixing it often
        # clears the rest.
        headline = f"{failures[0].label} is not working."
        next_step = failures[0].fix
    elif warnings:
        status = WARN
        headline = f"{warnings[0].label} needs attention."
        next_step = warnings[0].fix
    else:
        status = PASS
        headline = "Everything checked out."
        next_step = None
    return {
        "status": status,
        "headline": headline,
        "next_step": next_step,
        "passed": sum(1 for leg in legs if leg.status == PASS),
        "failed": len(failures),
        "warned": len(warnings),
        "skipped": sum(1 for leg in legs if leg.status == SKIP),
    }


def run_self_test(store: Any, *, include_audio: bool = False) -> Dict[str, Any]:
    """Run every leg and return the report.

    Legs run in capability order — chat, voice, mic, memory — so the summary
    can point at the most fundamental failure first. ``include_audio``
    attaches the TTS sample as base64 so the UI can play it; off by default
    since the clip is only useful to an interactive caller.
    """
    started = time.monotonic()
    legs = [
        _run_leg("llm", "Chat model", lambda: check_llm(store)),
        _run_leg("tts", "Voice (text to speech)", lambda: check_tts(store)),
        _run_leg("stt", "Microphone (speech to text)", lambda: check_stt(store)),
        _run_leg("memory", "Memory", lambda: check_memory(store)),
    ]
    report = {
        "legs": [leg.as_dict() for leg in legs],
        "summary": _summarize(legs),
        "duration_ms": int((time.monotonic() - started) * 1000),
    }
    telemetry.incr("self_test.runs")
    if report["summary"]["status"] == FAIL:
        telemetry.incr("self_test.failed_runs")
    if include_audio:
        report["audio_available"] = any(
            leg.key == "tts" and leg.status == PASS for leg in legs
        )
    return report
