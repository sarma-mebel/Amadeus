"""Async reach-out delivery: Discord DM when no client is around.

The 30s frontend poll (``POST /api/reachout/due`` / ``POST /api/reminders/
due``) can only deliver a due line while a browser/pet window is open and
polling. If the app is closed for the day, a reach-out or an overdue
reminder just never reaches the user — the scheduler generates it, nothing
ever pops it, and it goes stale. This module is the fallback: when the
scheduler notices no client has polled recently, it delivers the same due
material over a Discord DM instead of waiting for a poll that may never
come.

Two independent halves, deliberately kept separate:

- :func:`should_attempt_delivery` is a pure decision function (no I/O, no
  network) — same style as ``discord_bridge.py``'s ``should_respond``. It
  answers "should we even try to send a DM right now?" from plain data:
  config, wall-clock timestamps, quiet hours, sleep state, and the last
  delivery date. Fully unit-testable without a bot token or network access.
- :func:`send_discord_dm` is the actual network call: bare REST over
  stdlib ``urllib`` (no discord.py dependency — this path only ever sends
  one-off DMs, it never listens for anything, so the full gateway client
  the bridge needs would be pure overhead here). Best-effort by contract:
  it never raises, and any failure counts ``notify.delivery_failures``
  instead of propagating (this runs from the scheduler tick, which must
  never die because Discord had a bad moment).

Reuses ``discord.token`` (config or ``DISCORD_BOT_TOKEN`` env fallback,
wired in ``Config.from_dict``) — a second bot token field would just be the
same credential typed twice.
"""

from __future__ import annotations

import datetime as dt
import json
import urllib.error
import urllib.request
from typing import Any, Optional

from . import telemetry
from .logging_utils import get_logger

LOGGER = get_logger(__name__)

DISCORD_API_BASE = "https://discord.com/api/v10"
_REQUEST_TIMEOUT_SECONDS = 6.0
_MAX_ATTEMPTS = 2  # one try + one retry


# ---------------------------------------------------------------------------
# Pure decision logic — no network, no disk. Mirrors discord_bridge.py's
# should_respond in spirit: plain data in, a bool out.
# ---------------------------------------------------------------------------


def should_attempt_delivery(
    *,
    config: Any,
    last_client_poll: Optional[dt.datetime],
    now: dt.datetime,
    asleep: bool,
    last_delivery_date: Optional[dt.date],
) -> bool:
    """Whether the scheduler should attempt an async DM delivery this tick.

    Checked in cheapest-first order so the common "a client is open" case
    exits before touching quiet hours or the daily cap. Every negative path
    here has a matching ``notify.skipped_*`` counter recorded by the caller
    (this function stays pure and side-effect-free on purpose — the
    scheduler decides what to count, since only it knows whether a
    reach-out/reminder was even available to skip).
    """
    cfg = getattr(config, "async_notify", None)
    if cfg is None or not getattr(cfg, "enabled", False):
        return False
    if not getattr(cfg, "discord_user_id", None):
        return False
    # Token missing = nothing can ever be delivered. Checked HERE, before the
    # caller pops anything: deliver() also refuses without a token, but by
    # then the reach-out/reminders would already be consumed — enabled-with-
    # no-token used to destroy the day's line without sending it anywhere.
    channel = getattr(cfg, "channel", "discord") or "discord"
    if channel == "discord" and not getattr(
        getattr(config, "discord", None), "token", None
    ):
        return False

    if last_client_poll is not None:
        stale_minutes = (now - last_client_poll).total_seconds() / 60.0
        if stale_minutes < float(getattr(cfg, "min_client_stale_minutes", 30)):
            return False

    if asleep:
        return False

    if _in_quiet_hours(cfg, now):
        return False

    max_per_day = int(getattr(cfg, "max_per_day", 1))
    if max_per_day <= 0:
        return False
    if last_delivery_date is not None and last_delivery_date == now.date():
        return False

    return True


def _in_quiet_hours(cfg: Any, now: dt.datetime) -> bool:
    """User quiet hours (distinct from HER sleep schedule). start == end (or
    malformed hours) disables — a broken config value degrades to "never
    quiet", never a crash."""
    try:
        start = int(getattr(cfg, "quiet_hours_start", 22)) % 24
        end = int(getattr(cfg, "quiet_hours_end", 8)) % 24
    except (TypeError, ValueError):
        return False
    if start == end:
        return False
    hour = now.hour
    if start < end:
        return start <= hour < end
    # Window wraps midnight (e.g. 22 -> 8).
    return hour >= start or hour < end


# ---------------------------------------------------------------------------
# Discord REST delivery — stdlib urllib only, no discord.py dependency.
# ---------------------------------------------------------------------------


def _post_json(url: str, *, token: str, payload: dict) -> Optional[dict]:
    """One JSON POST to the Discord REST API. Returns the parsed response
    body, or None on any failure (never raises)."""
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={
            "Authorization": f"Bot {token}",
            "Content-Type": "application/json",
            "User-Agent": "Amadeus (async_notify, 1.0)",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=_REQUEST_TIMEOUT_SECONDS) as resp:
            raw = resp.read()
            return json.loads(raw) if raw else {}
    except urllib.error.HTTPError as exc:
        LOGGER.warning("Discord async-notify request failed (%s): %s", exc.code, url)
        return None
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
        LOGGER.warning("Discord async-notify request failed: %s", exc)
        return None


def _open_dm_channel(*, token: str, recipient_id: str) -> Optional[str]:
    data = _post_json(
        f"{DISCORD_API_BASE}/users/@me/channels",
        token=token,
        payload={"recipient_id": str(recipient_id)},
    )
    if not data:
        return None
    channel_id = data.get("id")
    return str(channel_id) if channel_id else None


def _send_channel_message(*, token: str, channel_id: str, content: str) -> bool:
    data = _post_json(
        f"{DISCORD_API_BASE}/channels/{channel_id}/messages",
        token=token,
        payload={"content": content},
    )
    return data is not None


def send_discord_dm(*, token: Optional[str], recipient_id: str, content: str) -> bool:
    """Send one Discord DM. Best-effort: never raises out, at most one
    retry of the whole (open-channel, send) sequence, and any failure counts
    ``notify.delivery_failures`` rather than propagating — a scheduler tick
    must survive Discord being unreachable.

    Returns True on a confirmed send, False otherwise.
    """
    if not token or not recipient_id or not content:
        telemetry.incr("notify.delivery_failures")
        return False
    for attempt in range(_MAX_ATTEMPTS):
        channel_id = _open_dm_channel(token=token, recipient_id=recipient_id)
        if channel_id and _send_channel_message(
            token=token, channel_id=channel_id, content=content
        ):
            return True
        if attempt + 1 < _MAX_ATTEMPTS:
            LOGGER.info(
                "Discord async-notify attempt %d failed, retrying once", attempt + 1
            )
    telemetry.incr("notify.delivery_failures")
    return False


def deliver(*, config: Any, content: str) -> bool:
    """Deliver ``content`` over the configured async channel. Only
    ``discord`` exists today; an unrecognized channel is a documented no-op
    (degrades loudly via the failure counter, never raises)."""
    cfg = getattr(config, "async_notify", None)
    channel = getattr(cfg, "channel", "discord") if cfg else "discord"
    if channel != "discord":
        LOGGER.warning("Unsupported async_notify.channel: %s", channel)
        telemetry.incr("notify.delivery_failures")
        return False
    token = getattr(getattr(config, "discord", None), "token", None)
    if not token:
        LOGGER.warning(
            "async_notify.enabled but no Discord bot token configured "
            "(discord.token / DISCORD_BOT_TOKEN)."
        )
        telemetry.incr("notify.delivery_failures")
        return False
    recipient_id = getattr(cfg, "discord_user_id", "")
    return send_discord_dm(token=token, recipient_id=recipient_id, content=content)


# ---------------------------------------------------------------------------
# Client-liveness snapshot — a tiny disk file the scheduler (a different
# process lifetime than the request handlers) reads to know whether a
# client polled recently. Same disk-snapshot convention as the brief/diary/
# reach-out jobs.
# ---------------------------------------------------------------------------

_LAST_SEEN_FILENAME = "client_last_seen.json"
_WRITE_THROTTLE_SECONDS = 60.0


def last_seen_path(config: Any) -> "Any":
    """Where the client-liveness snapshot lives: alongside reminders.json in
    the platform data dir (same anchoring as every other user-data file)."""
    from pathlib import Path

    from .paths import default_data_dir

    reminders_cfg = getattr(config, "reminders", None)
    store_path = getattr(reminders_cfg, "store_path", None) if reminders_cfg else None
    if store_path:
        return Path(store_path).parent / _LAST_SEEN_FILENAME
    return default_data_dir() / _LAST_SEEN_FILENAME


def record_client_poll(config: Any, now: Optional[dt.datetime] = None) -> None:
    """Stamp 'a client polled just now', throttled to at most once/minute.

    Called from the reach-out/reminders due-poll endpoints. Cheap by
    design: reads the existing timestamp first and skips the write
    entirely when the last stamp is under the throttle window, so a client
    polling every 30s doesn't turn into a disk write every 30s.
    """
    now = now or dt.datetime.now()
    path = last_seen_path(config)
    existing = _read_last_seen(path)
    if (
        existing is not None
        and (now - existing).total_seconds() < _WRITE_THROTTLE_SECONDS
    ):
        return
    _write_snapshot(
        path, last_seen=now, last_delivery_date=_read_last_delivery_date(path)
    )


def read_last_client_poll(config: Any) -> Optional[dt.datetime]:
    """The disk-snapshot read side, used by the scheduler job (a different
    process lifetime than the request handlers that write it)."""
    return _read_last_seen(last_seen_path(config))


def read_last_delivery_date(config: Any) -> Optional[dt.date]:
    return _read_last_delivery_date(last_seen_path(config))


def record_delivery(config: Any, today: dt.date) -> None:
    """Persist 'a DM went out today' — the max_per_day guard's memory,
    surviving process restarts the same way every other side-store does."""
    path = last_seen_path(config)
    _write_snapshot(
        path,
        last_seen=_read_last_seen(path),
        last_delivery_date=today,
    )


def _read_snapshot(path: Any) -> dict:
    try:
        if path.exists():
            with open(path, "r", encoding="utf-8") as fh:
                return json.load(fh) or {}
    except Exception as exc:  # corrupt/unreadable — start clean, never raise
        LOGGER.warning("Could not read client-liveness file %s: %s", path, exc)
    return {}


def _read_last_seen(path: Any) -> Optional[dt.datetime]:
    data = _read_snapshot(path)
    raw = data.get("last_seen")
    if not raw:
        return None
    try:
        return dt.datetime.fromisoformat(raw)
    except (TypeError, ValueError):
        return None


def _read_last_delivery_date(path: Any) -> Optional[dt.date]:
    data = _read_snapshot(path)
    raw = data.get("last_delivery_date")
    if not raw:
        return None
    try:
        return dt.date.fromisoformat(raw)
    except (TypeError, ValueError):
        return None


def _write_snapshot(
    path: Any,
    *,
    last_seen: Optional[dt.datetime],
    last_delivery_date: Optional[dt.date],
) -> None:
    import os

    payload = {
        "version": 1,
        "last_seen": last_seen.isoformat() if last_seen else None,
        "last_delivery_date": (
            last_delivery_date.isoformat() if last_delivery_date else None
        ),
    }
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        os.replace(tmp, path)
    except OSError as exc:  # never let a liveness stamp break a request
        LOGGER.warning("Could not write client-liveness file %s: %s", path, exc)
