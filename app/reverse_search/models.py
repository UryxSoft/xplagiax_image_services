"""
Data models for the reverse-image-search module.

Kept intentionally tiny: a provider match (raw signal from one provider) and
a search result (what the orchestrator returns). Plain frozen dataclasses are
enough for six fields — no ORM, no schema validation framework.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional


@dataclass(frozen=True)
class ProviderMatch:
    """Best match found by a single provider for one query image."""
    website: str
    url: str
    similarity: float  # 0-100


@dataclass(frozen=True)
class ReverseSearchResult:
    """
    Outcome of the full early-stop search.

    to_response() returns exactly the six fields the public API contract
    promises. cache_hit/stop_reason exist for server-side logging only and
    are never serialized to the client ("nada mas" per spec).
    """
    found: bool
    website: Optional[str]
    url: Optional[str]
    similarity: float
    provider: Optional[str]
    elapsed_ms: int
    cache_hit: bool = False
    # "threshold_met" | "providers_exhausted" | "cache_hit" | "no_providers_available"
    stop_reason: Optional[str] = None

    def to_response(self) -> dict:
        return {
            "found": self.found,
            "website": self.website,
            "url": self.url,
            "similarity": self.similarity,
            "provider": self.provider,
            "elapsed_ms": self.elapsed_ms,
        }

    def with_timing(self, elapsed_ms: int, cache_hit: bool = False) -> "ReverseSearchResult":
        """Return a copy with updated timing/cache_hit (used after a cache read,
        since the cached payload doesn't know how long *this* lookup took)."""
        return replace(self, elapsed_ms=elapsed_ms, cache_hit=cache_hit)


NOT_FOUND = ReverseSearchResult(
    found=False, website=None, url=None, similarity=0.0, provider=None, elapsed_ms=0,
)
