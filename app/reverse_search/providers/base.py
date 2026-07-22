"""
Provider contract for the reverse-image-search chain.

Every adapter subclasses ReverseSearchProvider and implements search(). On
failure it raises one of the typed exceptions below — the orchestrator treats
"no match" (return None) and "provider failed" (raised exception) differently:
a failed provider is logged and skipped, never turned into a 500 for the
caller. The whole point of chaining providers is resilience to any single
one having a bad day.

Retry policy (shared by all adapters via retry_call): retry ONLY on 429 and
5xx. Never retry 401/403 (bad credentials — retrying can't fix that) or 404
(no result — not an error). This mirrors the "RETRIES" section of the spec
exactly and lives in one place so no adapter re-implements backoff.
"""

from __future__ import annotations

import random
import time
from typing import Callable, Optional, TypeVar
from urllib.parse import urlparse

from app.reverse_search.models import ProviderMatch

T = TypeVar("T")

RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})
NEVER_RETRY_STATUS = frozenset({401, 403, 404})


class ProviderError(Exception):
    """Base class for all provider failures."""
    def __init__(self, provider: str, message: str) -> None:
        self.provider = provider
        super().__init__(f"[{provider}] {message}")


class ProviderAuthError(ProviderError):
    """401/403 — bad or missing API key. Never retried; almost always a config issue."""
    def __init__(self, provider: str, status_code: int) -> None:
        self.status_code = status_code
        super().__init__(provider, f"authentication failed (HTTP {status_code})")


class ProviderRateLimitError(ProviderError):
    """429 — provider quota/rate limit hit. Retryable, honors Retry-After if present."""
    def __init__(self, provider: str, retry_after: Optional[float] = None) -> None:
        self.retry_after = retry_after
        super().__init__(provider, "rate limited (HTTP 429)")


class ProviderTransientError(ProviderError):
    """5xx — transient server-side failure, safe to retry a bounded number of times."""
    def __init__(self, provider: str, status_code: int) -> None:
        self.status_code = status_code
        super().__init__(provider, f"transient server error (HTTP {status_code})")


class ProviderTimeoutError(ProviderError):
    """Connect or read timeout — the provider took too long. Not retried here;
    the next provider in the chain is tried instead (retrying a slow provider
    only makes p99 worse)."""
    def __init__(self, provider: str) -> None:
        super().__init__(provider, "request timed out")


class ProviderUnavailableError(ProviderError):
    """Provider not usable right now: missing config, or (Mungfali) an adapter
    whose upstream contract could not be verified — see providers/mungfali.py."""


class ProviderResponseError(ProviderError):
    """Unexpected 4xx (not 401/403/404) or an unparseable response body."""


class ReverseSearchProvider:
    """Base class for provider adapters."""
    name: str = "base"

    def search(
        self, *, image_bytes: bytes, image_url: Optional[str], timeout_s: float,
        deadline: Optional[float] = None,
    ) -> Optional[ProviderMatch]:
        """Return the best match, or None if the provider found nothing.
        Raise a ProviderError subclass if the call itself failed.

        `deadline` is an optional absolute time.perf_counter() value: once
        past it, retry_call() stops retrying even if attempts remain, so one
        slow/flaky provider can't blow through the whole request's time
        budget before the chain even reaches the next provider.
        """
        raise NotImplementedError


def retry_call(fn: Callable[[], T], *, max_retries: int, deadline: Optional[float] = None) -> T:
    """
    Call fn() with retry ONLY for ProviderRateLimitError / ProviderTransientError.
    Backoff: the provider's own Retry-After (429) when given, else capped
    exponential (0.5s, 1s, 2s, 4s) plus a little jitter to avoid thundering
    herds across concurrent requests.

    If `deadline` is set, a retry is skipped (the last exception is
    re-raised immediately) once we're at or past it — better to move on to
    the next provider in the chain than sleep through the rest of the
    request's time budget.
    """
    attempt = 0
    while True:
        try:
            return fn()
        except (ProviderRateLimitError, ProviderTransientError) as exc:
            attempt += 1
            if attempt > max_retries:
                raise
            if deadline is not None and time.perf_counter() >= deadline:
                raise
            retry_after = getattr(exc, "retry_after", None)
            delay = min(float(retry_after), 10.0) if retry_after else min(0.5 * (2 ** (attempt - 1)), 4.0)
            if deadline is not None:
                delay = min(delay, max(0.0, deadline - time.perf_counter()))
            time.sleep(delay + random.uniform(0, 0.25))


def hostname_of(url: str) -> str:
    """Shared by orchestrator (corroboration/domain trust) and provider
    adapters (deriving a display name when a page has no title)."""
    try:
        return (urlparse(url).hostname or url).lower()
    except ValueError:
        return url.lower()
