"""Per-session stash for the most recent vocal-emotion analysis.

Same shape as ``memory/speculative.py``'s pre-started recall stash: a
single-slot, timestamped, consume-exactly-once cache — but simpler, since
there's no future to await here (analysis already ran by the time
``analyze_and_stash`` is called). Free functions taking the ``ChatEngine``
(or ``WebChatSession``) as first argument — the same free-function-over-
owner pattern as ``presence.py``/``memory/introspection.py`` — so
``ChatEngine`` itself only needs one state slot (``_vocal_tone``).

Lifecycle:

1. The STT endpoint (plain ``/api/stt`` or the duplex loop) finishes
   transcribing a clip and calls :func:`analyze_and_stash` with the same
   PCM bytes, targeting whichever session is about to receive the chat
   turn. Analysis failure never breaks STT — it's caught here and counted.
2. The NEXT call to :func:`consume_vocal_tone` (from
   ``ChatEngine.prepare_turn``) pops the stash if it is still within
   ``vocal_emotion.ttl_seconds``; a stale entry is dropped and counted
   instead of being served to a turn it was never analyzed for.
3. Consumption is exactly-once: a second call in the same turn (there
   isn't one today, but the contract matters) finds nothing.

Never stored to memory, never added to transcripts — the result lives only
in this in-RAM slot and, for at most one turn, in the dynamic prompt tail.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Optional

from . import telemetry
from .logging_utils import get_logger
from .vocal_emotion import VocalEmotionResult, analyze

LOGGER = get_logger(__name__)


@dataclass
class _VocalToneEntry:
    result: VocalEmotionResult
    expires_at: float


def analyze_and_stash(engine: Any, pcm16: bytes, sample_rate: int) -> None:
    """Analyze ``pcm16`` and stash the result on ``engine`` (best-effort).

    Called right after a successful transcription, on the same audio bytes.
    Cheap (a few ms of numpy work) and never blocks/breaks the STT call
    site: any exception is caught, logged, and counted as
    ``vocal_emotion.analyze_failures`` rather than propagated.
    """
    cfg = getattr(getattr(engine, "config", None), "vocal_emotion", None)
    if not bool(getattr(cfg, "enabled", True)):
        return
    try:
        result = analyze(pcm16, sample_rate, engine.config)
    except Exception:
        telemetry.incr("vocal_emotion.analyze_failures")
        LOGGER.warning("Vocal emotion analysis failed", exc_info=True)
        return
    if result is None:
        return  # abstained — no hint is better than a low-confidence guess
    ttl = float(getattr(cfg, "ttl_seconds", 25.0) or 25.0)
    engine._vocal_tone = _VocalToneEntry(
        result=result, expires_at=time.monotonic() + ttl
    )
    telemetry.incr("vocal_emotion.analyzed")


def consume_vocal_tone(engine: Any) -> Optional[VocalEmotionResult]:
    """Pop and return the stashed result if still fresh, else None.

    Exactly-once: a hit clears the slot so a later call in the same or a
    following turn finds nothing left to consume. An expired entry is
    dropped and counted (``vocal_emotion.stale_dropped``) rather than
    served — a vocal-tone hint from three turns ago no longer describes
    "just now".
    """
    entry: Optional[_VocalToneEntry] = getattr(engine, "_vocal_tone", None)
    if entry is None:
        return None
    engine._vocal_tone = None
    if time.monotonic() > entry.expires_at:
        telemetry.incr("vocal_emotion.stale_dropped")
        return None
    telemetry.incr("vocal_emotion.consumed")
    return entry.result
