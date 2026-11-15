"""Per-client sliding-window rate limiter for the expensive turn endpoints.

Deliberately in-process and dependency-free: the hosted demo runs a single
uvicorn worker, so shared-state backends (Redis etc.) would be pure overhead.
Multi-worker deployments get a per-worker limit, which still bounds abuse.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from typing import Deque, Dict, Mapping, Optional


def client_key(headers: Mapping[str, str], fallback_host: Optional[str]) -> str:
    """Best-effort client identity behind proxies.

    Fly.io sets Fly-Client-IP; generic reverse proxies set X-Forwarded-For
    (first hop is the real client). A direct connection falls back to the
    socket peer address.
    """

    fly_ip = headers.get("fly-client-ip")
    if fly_ip:
        return fly_ip.strip()
    forwarded = headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return fallback_host or "unknown"


class TurnRateLimiter:
    """Allow up to ``per_minute`` hits per key in a sliding 60s window."""

    # Above this many tracked keys, expired entries are swept on the next
    # hit so an IP-churning scraper can't grow the dict without bound.
    _PRUNE_THRESHOLD = 1024

    def __init__(self, per_minute: int, window_seconds: float = 60.0) -> None:
        if per_minute < 1:
            raise ValueError("per_minute must be >= 1")
        self.per_minute = per_minute
        self.window = window_seconds
        self._hits: Dict[str, Deque[float]] = {}
        self._lock = threading.Lock()

    def allow(self, key: str) -> bool:
        now = time.monotonic()
        cutoff = now - self.window
        with self._lock:
            hits = self._hits.setdefault(key, deque())
            while hits and hits[0] < cutoff:
                hits.popleft()
            if len(hits) >= self.per_minute:
                return False
            hits.append(now)
            if len(self._hits) > self._PRUNE_THRESHOLD:
                stale = [k for k, v in self._hits.items() if not v or v[-1] < cutoff]
                for k in stale:
                    del self._hits[k]
            return True
