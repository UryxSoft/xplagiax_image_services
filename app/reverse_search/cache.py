"""
SHA-256 keyed cache for reverse-image-search results.

Same image bytes -> same digest -> same cache key. The digest is computed
EXACTLY ONCE by the orchestrator and passed in here; this module never
re-hashes. Positive ("found") results are cached far longer than negative
ones: a discovered copy of an image rarely un-appears, but "not found today"
can flip to "found" as provider indexes grow — so a miss must not be allowed
to silence re-checks for weeks.

Built on top of the existing CacheClient (Redis, graceful degradation) rather
than a new Redis connection — if Redis is down, get() simply misses and set()
is a no-op, exactly like the rest of the app.
"""

from __future__ import annotations

from typing import Optional

from app.reverse_search.models import ReverseSearchResult

_PREFIX = "reverse_search_v1"  # bump the version if the cached payload shape ever changes


class ReverseSearchCache:
    def __init__(self, cache_client, ttl_found: int, ttl_not_found: int) -> None:
        self._cache = cache_client
        self._ttl_found = ttl_found
        self._ttl_not_found = ttl_not_found

    @property
    def available(self) -> bool:
        return bool(self._cache and self._cache.available)

    @staticmethod
    def _key(digest: str) -> str:
        return f"{_PREFIX}:{digest}"

    def get(self, digest: str) -> Optional[ReverseSearchResult]:
        if not self.available:
            return None
        raw = self._cache.get(self._key(digest))
        if raw is None:
            return None
        return ReverseSearchResult(
            found=raw.get("found", False),
            website=raw.get("website"),
            url=raw.get("url"),
            similarity=raw.get("similarity", 0.0),
            provider=raw.get("provider"),
            elapsed_ms=0,  # caller overwrites with the actual cache-lookup latency
            cache_hit=True,
            stop_reason="cache_hit",
        )

    def set(self, digest: str, result: ReverseSearchResult) -> None:
        if not self.available:
            return
        ttl = self._ttl_found if result.found else self._ttl_not_found
        payload = {
            "found": result.found,
            "website": result.website,
            "url": result.url,
            "similarity": result.similarity,
            "provider": result.provider,
        }
        self._cache.set(self._key(digest), payload, ttl)
