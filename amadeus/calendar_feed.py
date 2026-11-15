"""Read-only ICS calendar awareness for the daily brief.

Stdlib only, by design: a minimal RFC5545 parser (line unfolding, VEVENT
blocks, DTSTART/DTEND in both DATE and DATE-TIME forms, SUMMARY unescaping)
plus a conservative RRULE expander that only handles simple DAILY/WEEKLY
recurrence. Anything it isn't confident about is SKIPPED, never guessed — a
missing event is a minor annoyance, a wrong one erodes trust in the brief.

Fetching (``ics_url`` via ``urllib`` or a local ``ics_path``) happens ONLY
from the scheduler tick context (see ``amadeus/web/scheduler.py``), never on
a chat turn — the same disk-snapshot stance as ``brief.py``. Results are
cached to a small JSON snapshot (fetched_at + parsed upcoming events, never
the raw ICS text) with a refresh TTL; a fetch failure serves the stale cache
instead of going empty, and an empty/missing cache degrades to no events
rather than blocking the brief.

Not scoped per-user or per-character: like reminders, a calendar is the
user's real-world schedule, not persona memory.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import os
import re
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import telemetry

LOGGER = logging.getLogger(__name__)

_FETCH_TIMEOUT_SECONDS = 8.0
_MAX_ICS_BYTES = 5_000_000  # 5MB: a sane ceiling for a personal feed

_WEEKDAY_CODES = {
    "MO": 0,
    "TU": 1,
    "WE": 2,
    "TH": 3,
    "FR": 4,
    "SA": 5,
    "SU": 6,
}


@dataclass
class CalendarEvent:
    """One (already-expanded, if recurring) calendar occurrence."""

    summary: str
    start: dt.datetime  # local, naive
    end: Optional[dt.datetime]
    all_day: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "summary": self.summary,
            "start": self.start.isoformat(),
            "end": self.end.isoformat() if self.end else None,
            "all_day": self.all_day,
        }

    @classmethod
    def from_dict(cls, data: Any) -> Optional["CalendarEvent"]:
        if (
            not isinstance(data, dict)
            or not data.get("summary")
            or not data.get("start")
        ):
            return None
        try:
            start = dt.datetime.fromisoformat(str(data["start"]))
        except ValueError:
            return None
        end_raw = data.get("end")
        end = None
        if end_raw:
            try:
                end = dt.datetime.fromisoformat(str(end_raw))
            except ValueError:
                end = None
        return cls(
            summary=str(data["summary"]),
            start=start,
            end=end,
            all_day=bool(data.get("all_day", False)),
        )

    def human_line(self) -> str:
        """Compact "10:00 lab meeting" / "all day: her birthday" string."""
        summary = self.summary.strip() or "(untitled event)"
        if self.all_day:
            return f"all day: {summary}"
        return f"{self.start.strftime('%H:%M')} {summary}"


# ---------------------------------------------------------------------------
# ICS parsing
# ---------------------------------------------------------------------------


def _unfold_lines(raw: str) -> List[str]:
    """RFC5545 line unfolding: CRLF/LF followed by a leading space or tab
    continues the previous line."""
    # Normalize line endings first so both CRLF and bare LF feeds unfold.
    normalized = raw.replace("\r\n", "\n").replace("\r", "\n")
    lines = normalized.split("\n")
    unfolded: List[str] = []
    for line in lines:
        if line[:1] in (" ", "\t") and unfolded:
            unfolded[-1] += line[1:]
        else:
            unfolded.append(line)
    return unfolded


def _unescape_text(value: str) -> str:
    """RFC5545 TEXT unescaping: \\\\, \\;, \\,, \\N/\\n -> literal chars."""
    out = []
    i = 0
    n = len(value)
    while i < n:
        ch = value[i]
        if ch == "\\" and i + 1 < n:
            nxt = value[i + 1]
            if nxt in ("n", "N"):
                out.append("\n")
            elif nxt in (";", ",", "\\"):
                out.append(nxt)
            else:
                out.append(nxt)
            i += 2
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def _parse_content_line(line: str) -> Optional[tuple]:
    """Split a content line into (name, params_dict, value). None if the
    line doesn't look like NAME;PARAMS:VALUE or NAME:VALUE."""
    if ":" not in line:
        return None
    head, value = line.split(":", 1)
    parts = head.split(";")
    name = parts[0].strip().upper()
    if not name:
        return None
    params: Dict[str, str] = {}
    for part in parts[1:]:
        if "=" in part:
            key, val = part.split("=", 1)
            params[key.strip().upper()] = val.strip()
    return name, params, value


def _parse_datetime(
    value: str, params: Dict[str, str]
) -> "tuple[Optional[dt.datetime], bool]":
    """Parse a DTSTART/DTEND/UNTIL value. Returns (datetime, all_day).

    Handles: DATE form (YYYYMMDD, all-day), DATE-TIME UTC (trailing Z ->
    converted to local naive time), DATE-TIME with TZID param (treated as
    already-local — counted via calendar.tzid_assumed_local, no timezone
    database is consulted), and floating (no Z, no TZID -> treated as
    local as-is).
    """
    value = value.strip()
    if params.get("VALUE") == "DATE" or re.fullmatch(r"\d{8}", value):
        try:
            return dt.datetime.strptime(value, "%Y%m%d"), True
        except ValueError:
            return None, False
    is_utc = value.endswith("Z")
    core = value[:-1] if is_utc else value
    parsed: Optional[dt.datetime] = None
    for fmt in ("%Y%m%dT%H%M%S", "%Y%m%dT%H%M"):
        try:
            parsed = dt.datetime.strptime(core, fmt)
            break
        except ValueError:
            continue
    if parsed is None:
        return None, False
    if is_utc:
        utc_dt = parsed.replace(tzinfo=dt.timezone.utc)
        local = utc_dt.astimezone().replace(tzinfo=None)
        return local, False
    if "TZID" in params:
        telemetry.incr("calendar.tzid_assumed_local")
        return parsed, False
    return parsed, False


@dataclass
class _RawEvent:
    summary: str
    start: dt.datetime
    end: Optional[dt.datetime]
    all_day: bool
    rrule: Optional[Dict[str, str]]


def _parse_rrule(value: str) -> Dict[str, str]:
    parts: Dict[str, str] = {}
    for chunk in value.split(";"):
        if "=" in chunk:
            key, val = chunk.split("=", 1)
            parts[key.strip().upper()] = val.strip()
    return parts


def parse_ics(raw: str) -> List[_RawEvent]:
    """Parse VEVENT blocks (ignoring VTODO/VJOURNAL). Malformed events are
    skipped individually; a fully unparseable feed returns an empty list and
    counts ``calendar.parse_failures``."""
    events: List[_RawEvent] = []
    try:
        lines = _unfold_lines(raw)
        in_event = False
        cur: Dict[str, Any] = {}
        for line in lines:
            if not line:
                continue
            if line.strip().upper() == "BEGIN:VEVENT":
                in_event = True
                cur = {}
                continue
            if line.strip().upper() == "END:VEVENT":
                in_event = False
                event = _finish_event(cur)
                if event is not None:
                    events.append(event)
                cur = {}
                continue
            if not in_event:
                continue
            parsed = _parse_content_line(line)
            if parsed is None:
                continue
            name, params, value = parsed
            if name == "SUMMARY":
                cur["summary"] = _unescape_text(value)
            elif name == "DTSTART":
                cur["dtstart"] = (value, params)
            elif name == "DTEND":
                cur["dtend"] = (value, params)
            elif name == "RRULE":
                cur["rrule"] = _parse_rrule(value)
    except Exception as exc:  # pragma: no cover - defensive, degrade loudly
        LOGGER.warning("ICS parse failed: %s", exc)
        telemetry.incr("calendar.parse_failures")
        return []
    return events


def _finish_event(cur: Dict[str, Any]) -> Optional[_RawEvent]:
    summary = cur.get("summary")
    dtstart = cur.get("dtstart")
    if not summary or not dtstart:
        telemetry.incr("calendar.parse_failures")
        return None
    value, params = dtstart
    start, all_day = _parse_datetime(value, params)
    if start is None:
        telemetry.incr("calendar.parse_failures")
        return None
    end = None
    dtend = cur.get("dtend")
    if dtend:
        end_value, end_params = dtend
        end, _ = _parse_datetime(end_value, end_params)
    return _RawEvent(
        summary=summary, start=start, end=end, all_day=all_day, rrule=cur.get("rrule")
    )


# ---------------------------------------------------------------------------
# RRULE expansion (simple DAILY/WEEKLY only)
# ---------------------------------------------------------------------------


def _expand_rrule(
    event: _RawEvent, window_start: dt.datetime, window_end: dt.datetime
) -> List[dt.datetime]:
    """Conservative RRULE expansion: only FREQ=DAILY and FREQ=WEEKLY (with
    optional BYDAY) are handled, with UNTIL/COUNT applied approximately.
    Anything else is skipped (counter ``calendar.rrule_skipped``); a
    dropped event beats a wrongly-computed one."""
    rrule = event.rrule or {}
    freq = rrule.get("FREQ", "").upper()
    if freq not in ("DAILY", "WEEKLY"):
        telemetry.incr("calendar.rrule_skipped")
        return []
    supported_keys = {"FREQ", "UNTIL", "COUNT", "INTERVAL", "BYDAY", "WKST"}
    if set(rrule.keys()) - supported_keys:
        telemetry.incr("calendar.rrule_skipped")
        return []
    try:
        interval = int(rrule.get("INTERVAL", "1"))
    except ValueError:
        telemetry.incr("calendar.rrule_skipped")
        return []
    if interval <= 0:
        telemetry.incr("calendar.rrule_skipped")
        return []

    until: Optional[dt.datetime] = None
    if "UNTIL" in rrule:
        until, _ = _parse_datetime(rrule["UNTIL"], {})
        if until is None:
            telemetry.incr("calendar.rrule_skipped")
            return []

    count: Optional[int] = None
    if "COUNT" in rrule:
        try:
            count = int(rrule["COUNT"])
        except ValueError:
            telemetry.incr("calendar.rrule_skipped")
            return []
        if count <= 0:
            telemetry.incr("calendar.rrule_skipped")
            return []

    byday: Optional[List[int]] = None
    if "BYDAY" in rrule:
        if freq != "WEEKLY":
            # BYDAY on a DAILY rule is a pattern we don't need to support.
            telemetry.incr("calendar.rrule_skipped")
            return []
        codes = [c.strip().upper() for c in rrule["BYDAY"].split(",") if c.strip()]
        byday = []
        for code in codes:
            if code not in _WEEKDAY_CODES:
                telemetry.incr("calendar.rrule_skipped")
                return []
            byday.append(_WEEKDAY_CODES[code])
        if not byday:
            telemetry.incr("calendar.rrule_skipped")
            return []

    occurrences: List[dt.datetime] = []
    # Hard safety cap: never walk more than a couple of years of occurrences
    # even if COUNT/UNTIL are absent or huge (a conservative bound, not the
    # spec's semantics) — the window clip below does the real limiting.
    max_iterations = 3000
    step = dt.timedelta(days=1) if freq == "DAILY" else dt.timedelta(weeks=interval)
    cursor = event.start
    emitted = 0
    iterations = 0
    hard_stop = window_end + dt.timedelta(days=1)

    if freq == "DAILY":
        while cursor <= hard_stop and iterations < max_iterations:
            iterations += 1
            if until is not None and cursor > until:
                break
            if count is not None and emitted >= count:
                break
            if cursor >= event.start:
                if count is not None:
                    emitted += 1
                if window_start <= cursor <= window_end:
                    occurrences.append(cursor)
            cursor = cursor + dt.timedelta(days=interval)
        return occurrences

    # WEEKLY (with optional BYDAY): walk week-by-week from the event's own
    # week, emitting each requested weekday within that week.
    week_anchor = event.start - dt.timedelta(days=event.start.weekday())
    days_of_week = byday if byday is not None else [event.start.weekday()]
    while week_anchor <= hard_stop and iterations < max_iterations:
        iterations += 1
        for wd in sorted(days_of_week):
            occ = week_anchor + dt.timedelta(days=wd)
            occ = occ.replace(
                hour=event.start.hour,
                minute=event.start.minute,
                second=event.start.second,
            )
            if occ < event.start:
                continue
            if until is not None and occ > until:
                continue
            if count is not None and emitted >= count:
                break
            if count is not None:
                emitted += 1
            if window_start <= occ <= window_end:
                occurrences.append(occ)
        if count is not None and emitted >= count:
            break
        week_anchor = week_anchor + step
    occurrences.sort()
    return occurrences


def expand_events(
    raw_events: List[_RawEvent], window_start: dt.datetime, window_end: dt.datetime
) -> List[CalendarEvent]:
    """Turn parsed VEVENTs (single-shot or recurring) into concrete
    occurrences that fall inside [window_start, window_end]."""
    out: List[CalendarEvent] = []
    for event in raw_events:
        duration = (
            (event.end - event.start)
            if (event.end and event.end > event.start)
            else None
        )
        if event.rrule:
            starts = _expand_rrule(event, window_start, window_end)
        else:
            starts = [event.start] if window_start <= event.start <= window_end else []
        for start in starts:
            end = start + duration if duration else None
            out.append(
                CalendarEvent(
                    summary=event.summary,
                    start=start,
                    end=end,
                    all_day=event.all_day,
                )
            )
    return out


# ---------------------------------------------------------------------------
# Fetch + cache
# ---------------------------------------------------------------------------


def is_configured(settings: Any) -> bool:
    """Active iff a URL or a local path is set — no separate enabled flag."""
    return bool(getattr(settings, "ics_url", "") or getattr(settings, "ics_path", ""))


def _fetch_raw_ics(settings: Any) -> Optional[str]:
    """Fetch the ICS text from ics_url (preferred) or ics_path. None on any
    failure; counts ``calendar.fetch_failures``."""
    url = getattr(settings, "ics_url", "") or ""
    path = getattr(settings, "ics_path", "") or ""
    try:
        if url:
            req = urllib.request.Request(url, headers={"User-Agent": "Amadeus/1.0"})
            with urllib.request.urlopen(req, timeout=_FETCH_TIMEOUT_SECONDS) as resp:
                body = resp.read(_MAX_ICS_BYTES + 1)
            if len(body) > _MAX_ICS_BYTES:
                raise ValueError("ICS feed exceeds size ceiling")
            return body.decode("utf-8", errors="replace")
        if path:
            p = Path(path)
            if not p.exists():
                raise FileNotFoundError(path)
            return p.read_text(encoding="utf-8", errors="replace")
    except Exception as exc:
        LOGGER.warning("Calendar fetch failed: %s", exc)
        telemetry.incr("calendar.fetch_failures")
        return None
    return None


class CalendarCache:
    """JSON snapshot of the parsed upcoming events (never raw ICS), keyed by
    a fetched_at timestamp so ``refresh`` can decide staleness. One shared
    file — the calendar is process-global like reminders, not per-user or
    per-character."""

    def __init__(self, path: str) -> None:
        self.path = str(path)

    def load(self) -> "tuple[Optional[dt.datetime], List[CalendarEvent]]":
        if not os.path.exists(self.path):
            return None, []
        try:
            with open(self.path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            fetched_at = None
            raw_fetched_at = data.get("fetched_at")
            if raw_fetched_at:
                try:
                    fetched_at = dt.datetime.fromisoformat(str(raw_fetched_at))
                except ValueError:
                    fetched_at = None
            events = []
            for entry in data.get("events") or []:
                event = CalendarEvent.from_dict(entry)
                if event is not None:
                    events.append(event)
            return fetched_at, events
        except Exception as exc:
            LOGGER.warning("Could not read calendar cache %s: %s", self.path, exc)
            return None, []

    def save(self, fetched_at: dt.datetime, events: List[CalendarEvent]) -> None:
        payload = {
            "version": 1,
            "fetched_at": fetched_at.isoformat(timespec="seconds"),
            "events": [e.to_dict() for e in events],
        }
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        os.replace(tmp, self.path)


def refresh_cache(settings: Any, cache: CalendarCache, now: dt.datetime) -> None:
    """Refetch + reparse if the cache is missing or past its TTL, storing a
    fresh parsed snapshot. Called from the scheduler tick context ONLY —
    never on a chat turn. On fetch failure the existing cache (if any) is
    left untouched, so callers still serve stale data."""
    if not is_configured(settings):
        return
    fetched_at, _ = cache.load()
    ttl = dt.timedelta(hours=float(getattr(settings, "refresh_ttl_hours", 6.0)))
    if fetched_at is not None and now - fetched_at < ttl:
        return
    raw = _fetch_raw_ics(settings)
    if raw is None:
        return  # stale cache (if any) stands; fetch_failures already counted
    raw_events = parse_ics(raw)
    lookahead_hours = int(getattr(settings, "lookahead_hours", 36))
    window_start = now - dt.timedelta(days=1)  # keep some slack for all-day/local math
    window_end = now + dt.timedelta(hours=lookahead_hours)
    events = expand_events(raw_events, window_start, window_end)
    cache.save(now, events)


def upcoming_events(
    settings: Any, now: dt.datetime, *, cache_path: Optional[str] = None
) -> List[CalendarEvent]:
    """Public API: today's remaining events + the lookahead window's
    events, sorted and capped at ``max_events``. Reads the cache only —
    never fetches (fetching is the scheduler's job). Empty (not an error)
    when the feature isn't configured or nothing is cached yet.

    ``cache_path`` defaults to ``settings.cache_path`` (set by
    ``Config.resolve_user_data_paths``, same convention as every other
    side-store path) so callers normally don't need to pass it explicitly.
    """
    if not is_configured(settings):
        return []
    resolved_path = (
        cache_path or getattr(settings, "cache_path", None) or ("calendar_cache.json")
    )
    cache = CalendarCache(resolved_path)
    _fetched_at, events = cache.load()
    lookahead_hours = int(getattr(settings, "lookahead_hours", 36))
    window_end = now + dt.timedelta(hours=lookahead_hours)
    # All-day events: keep them if "today or later" (by date) even though
    # their naive start time (midnight) is technically <= now; timed events
    # keep only those not yet ended.
    in_window = [
        e
        for e in events
        if (
            (e.all_day and e.start.date() >= now.date() and e.start <= window_end)
            or (not e.all_day and now <= (e.end or e.start) and e.start <= window_end)
        )
    ]
    in_window.sort(key=lambda e: e.start)
    max_events = int(getattr(settings, "max_events", 8))
    capped = in_window[:max_events]
    if capped:
        telemetry.incr("calendar.events_served", len(capped))
    return capped
