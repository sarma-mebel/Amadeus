"""Her sleep window: the one pure question "is she asleep right now?".

Consumed by the presence family (idle chatter suppression, groggy poke
register), the reach-out delivery pop (held, not consumed, until morning),
the greeting (woken-by-your-arrival register), and ``/api/status`` (so the
UI can render the state). Local wall-clock hours, matching the persona
clock line — her bedtime follows the machine's timezone the same way her
"late night" mood bias does.
"""

from __future__ import annotations

import datetime as dt
from typing import Any, Optional


def is_sleep_hours(config: Any, now: Optional[dt.datetime] = None) -> bool:
    """True while the configured sleep window covers the local hour.

    start == end (or malformed hours) disables — never a raise: a broken
    config value must degrade to "always awake", not a dead presence loop.
    """
    cfg = getattr(config, "sleep_schedule", None)
    if cfg is None or not getattr(cfg, "enabled", False):
        return False
    try:
        start = int(getattr(cfg, "start_hour", 1)) % 24
        end = int(getattr(cfg, "end_hour", 8)) % 24
    except (TypeError, ValueError):
        return False
    if start == end:
        return False
    hour = (now or dt.datetime.now()).hour
    if start < end:
        return start <= hour < end
    # Window wraps midnight (e.g. 23 -> 7).
    return hour >= start or hour < end


def sleep_status(config: Any, now: Optional[dt.datetime] = None) -> dict:
    """The ``/api/status`` payload block: schedule + live state."""
    cfg = getattr(config, "sleep_schedule", None)
    return {
        "enabled": bool(getattr(cfg, "enabled", False)) if cfg else False,
        "start_hour": int(getattr(cfg, "start_hour", 1)) if cfg else 1,
        "end_hour": int(getattr(cfg, "end_hour", 8)) if cfg else 8,
        "asleep_now": is_sleep_hours(config, now),
    }
