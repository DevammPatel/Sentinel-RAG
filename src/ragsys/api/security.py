"""Request-level protection for a public demo: rate limits and admin auth.

The limiter is deliberately simple (in-process, in-memory). It exists to keep one
visitor, or one script, from burning the free LLM quota for everyone else. It is
correct for a single worker process, which is how this service is deployed; a
multi-instance deployment would move the counters to Redis.
"""
from __future__ import annotations

import hmac
import threading
import time
from collections import defaultdict, deque

from fastapi import Header, HTTPException, Request

from ragsys.config import settings

_DAY = 86_400


class RateLimiter:
    def __init__(self, per_minute: int, per_day: int, global_per_day: int) -> None:
        self.per_minute = per_minute
        self.per_day = per_day
        self.global_per_day = global_per_day
        self._minute: dict[str, deque[float]] = defaultdict(deque)
        self._day: dict[str, deque[float]] = defaultdict(deque)
        self._global: deque[float] = deque()
        self._lock = threading.Lock()

    @staticmethod
    def _trim(q: deque[float], cutoff: float) -> None:
        while q and q[0] <= cutoff:
            q.popleft()

    def check(self, client: str, now: float | None = None) -> tuple[bool, int, str]:
        """Record one request. Returns (allowed, retry_after_seconds, reason)."""
        now = time.time() if now is None else now
        with self._lock:
            minute, day = self._minute[client], self._day[client]
            self._trim(minute, now - 60)
            self._trim(day, now - _DAY)
            self._trim(self._global, now - _DAY)

            if len(minute) >= self.per_minute:
                return False, max(1, int(minute[0] + 60 - now) + 1), "minute"
            if len(day) >= self.per_day:
                return False, max(1, int(day[0] + _DAY - now) + 1), "day"
            if len(self._global) >= self.global_per_day:
                return False, max(1, int(self._global[0] + _DAY - now) + 1), "global"

            minute.append(now)
            day.append(now)
            self._global.append(now)
            # Drop idle clients so the tables can't grow without bound.
            if len(self._day) > 5000:
                for key in [k for k, q in self._day.items() if not q]:
                    self._day.pop(key, None)
                    self._minute.pop(key, None)
            return True, 0, ""


limiter = RateLimiter(
    settings.rate_limit_per_minute, settings.rate_limit_per_day, settings.global_daily_limit
)

_MESSAGES = {
    "minute": "Too many questions in a short time. Please wait a moment and try again.",
    "day": "You have reached today's question limit for this demo. Please come back tomorrow.",
    "global": "This demo has reached its daily limit across all visitors. Please come back tomorrow.",
}


def client_ip(request: Request) -> str:
    """The caller's address.

    Behind a reverse proxy the socket peer is the proxy, so when
    TRUST_PROXY_HEADERS is on we read X-Forwarded-For. Each proxy appends the
    address it received the request from, so with N trusted proxies the client
    is the Nth entry from the right. Entries further left are whatever the
    client chose to send and can be forged to dodge the limit, so they are
    never used.
    """
    if settings.trust_proxy_headers:
        forwarded = request.headers.get("x-forwarded-for", "")
        parts = [p.strip() for p in forwarded.split(",") if p.strip()]
        if parts:
            return parts[-min(max(settings.proxy_hops, 1), len(parts))]
    return request.client.host if request.client else "unknown"


def enforce_rate_limit(request: Request) -> None:
    allowed, retry_after, reason = limiter.check(client_ip(request))
    if not allowed:
        raise HTTPException(
            status_code=429,
            detail=_MESSAGES[reason],
            headers={"Retry-After": str(retry_after)},
        )


def require_admin(authorization: str | None = Header(default=None)) -> None:
    """Guard for ingest endpoints.

    With no ADMIN_TOKEN configured the endpoints answer 404, so a public
    deployment exposes no write path at all.
    """
    if not settings.admin_token:
        raise HTTPException(status_code=404, detail="Not Found")
    supplied = (authorization or "").removeprefix("Bearer ").strip()
    if not supplied or not hmac.compare_digest(supplied, settings.admin_token):
        raise HTTPException(status_code=401, detail="invalid or missing admin token")
