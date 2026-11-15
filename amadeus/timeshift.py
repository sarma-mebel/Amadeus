"""Age Amadeus's persisted state: shift every stored timestamp into the past.

The clock strategy of the longitudinal simulation (scripts/longitudinal_sim.py
and tests/test_longitudinal.py). Faking ``datetime`` process-wide is the wrong
tool here: the codebase mixes naive-UTC and local clock reads across 30+
modules, background executor threads stamp records after the barrier a test
can await, and the biggest time-dependent behaviors — arc/trust drift, early
acquaintance — read no clock at all (they advance when ``profile.last_seen``
goes stale). So instead of moving "now", we move the DATA: after each
simulated day, every persisted timestamp shifts one day into the past, and
"now" stays the real wall clock. Every comparison in the codebase
(``utcnow() - timestamp``, ``today - noted``…) then sees exactly the ages a
real multi-week deployment would produce. ``scripts/eval_conversation.py``'s
``backdate_store_days`` is the single-store proof of this approach; this
module generalizes it to every store at once.

Shifting rules (walk every ``*.json`` / ``*.jsonl`` under a data directory):

- A string that fully parses as an ISO datetime WITH a time component
  ("2026-08-07T09:15:00", space-separated accepted) shifts wherever it
  appears — datetime-shaped strings are stamps in every store this repo
  writes — EXCEPT under content-bearing keys (fact text, summaries…),
  which are never touched even if a fact happens to be exactly a date.
- A date-only string ("2026-08-07") shifts only under known date keys
  (``DATE_KEYS`` below — fact/thread dates, the life-thread pace stamps,
  ``first_met``…), inside known date-container dicts (``open_thread_dates``,
  ``last_surfaced`` — value dates keyed by topic/user), or when it is
  itself a dict KEY (per-day stores keyed by date).

This is a simulation/testing tool. Pointing it at a live instance's data
directory rewrites real memory — only ever run it against an isolated copy,
with the backend stopped.
"""

from __future__ import annotations

import datetime as dt
import json
import re
from pathlib import Path
from typing import Any, Iterable, Optional, Tuple

from .logging_utils import get_logger

LOGGER = get_logger(__name__)

# Keys whose string values are prose, never stamps. A fact text that IS a
# bare date ("2026-08-15") would otherwise be aged into a different fact.
CONTENT_KEYS = {
    "text",
    "content",
    "en",
    "ja",
    "summary",
    "previous_fact",
    "detail",
    "topic",
    "value",
    "reason",
    "label",
    "subject",
}

# Keys whose values may be date-only strings that must still shift.
# Membership here only grants PERMISSION for date-only values — a
# datetime-shaped value shifts under any non-content key regardless, so a
# key listed here that carries a full timestamp elsewhere is still handled.
DATE_KEYS = {
    "date",
    "noted",
    "when",
    "valid_at",
    # Life threads (life_threads.py) persist date-only stamps for the pace
    # guards and per-thread history; unshifted, acted_today() stays true
    # forever after the sim's day 0 and her pursuits freeze.
    "started_at",
    "last_progressed_at",
    "asked_at",
    "last_activity_date",
    "last_ask_date",
    # Profile first-met (profile.py) is a date-only stamp; anniversary
    # milestones can never fire in a sim unless it ages.
    "first_met",
}

# Dicts whose VALUES are date strings keyed by something else entirely
# (open thread topic -> date noted, user key -> last-surfaced date).
DATE_CONTAINER_KEYS = {"open_thread_dates", "last_surfaced"}

_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_DATETIME_RE = re.compile(r"^\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}(:\d{2}(\.\d+)?)?")


def shift_stamp(value: str, delta: dt.timedelta, *, allow_date: bool) -> Optional[str]:
    """Shift one ISO string by ``-delta``. None when it isn't shiftable."""
    text = value.strip()
    if _DATETIME_RE.match(text):
        try:
            stamp = dt.datetime.fromisoformat(text)
        except ValueError:
            return None
        return (stamp - delta).isoformat()
    if allow_date and _DATE_RE.match(text):
        try:
            day = dt.date.fromisoformat(text)
        except ValueError:
            return None
        return (day - delta).isoformat()
    return None


def shift_structure(
    obj: Any,
    delta: dt.timedelta,
    *,
    key: Optional[str] = None,
    parent_key: Optional[str] = None,
) -> Tuple[Any, int]:
    """Recursively shift stamps inside a JSON structure.

    Returns ``(new_object, shifted_count)``. Pure function — never mutates
    the input.
    """
    if isinstance(obj, str):
        if key in CONTENT_KEYS:
            return obj, 0
        allow_date = key in DATE_KEYS or parent_key in DATE_CONTAINER_KEYS
        shifted = shift_stamp(obj, delta, allow_date=allow_date)
        return (shifted, 1) if shifted is not None else (obj, 0)
    if isinstance(obj, dict):
        out = {}
        count = 0
        for k, v in obj.items():
            new_key = k
            if isinstance(k, str) and _DATE_RE.match(k):
                shifted_key = shift_stamp(k, delta, allow_date=True)
                if shifted_key is not None:
                    new_key = shifted_key
                    count += 1
            new_value, sub = shift_structure(v, delta, key=k, parent_key=key)
            out[new_key] = new_value
            count += sub
        return out, count
    if isinstance(obj, list):
        out_list = []
        count = 0
        for item in obj:
            new_item, sub = shift_structure(item, delta, key=key, parent_key=parent_key)
            out_list.append(new_item)
            count += sub
        return out_list, count
    return obj, 0


def shift_file(path: Path, delta: dt.timedelta) -> int:
    """Shift one ``.json`` or ``.jsonl`` file in place. Returns stamps moved.

    Unparseable files are left untouched (warned, never raised) — the
    simulation should observe a corrupt store the same way the app would,
    not destroy the evidence.
    """
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        LOGGER.warning("timeshift: unreadable %s: %s", path, exc)
        return 0
    if not raw.strip():
        return 0
    total = 0
    try:
        if path.suffix == ".jsonl":
            lines = []
            for line in raw.splitlines():
                if not line.strip():
                    continue
                data = json.loads(line)
                shifted, count = shift_structure(data, delta)
                total += count
                lines.append(json.dumps(shifted, ensure_ascii=False))
            payload = "\n".join(lines) + "\n"
        else:
            data = json.loads(raw)
            shifted, count = shift_structure(data, delta)
            total += count
            payload = json.dumps(shifted, ensure_ascii=False, indent=2)
    except (ValueError, TypeError) as exc:
        LOGGER.warning("timeshift: skipping unparseable %s: %s", path, exc)
        return 0
    if total:
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(payload, encoding="utf-8")
        tmp.replace(path)
    return total


def shift_tree(root: Path, days: float) -> int:
    """Shift every JSON/JSONL store under ``root`` back by ``days``.

    Returns the number of stamps moved. ``.bak*`` quarantine/backup files
    and ``.tmp`` remnants are skipped (they are historical evidence);
    SQLite/Chroma stores are not handled — the simulation runs on the JSON
    backend by design.
    """
    delta = dt.timedelta(days=days)
    total = 0
    for path in sorted(_store_files(root)):
        total += shift_file(path, delta)
    return total


def _store_files(root: Path) -> Iterable[Path]:
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix not in (".json", ".jsonl"):
            continue
        if ".bak" in path.name or path.name.endswith(".tmp"):
            continue
        yield path
