"""
Early-stop orchestrator — the core cost-saving logic of this module.

Providers are tried in priority order (lowest number first). As soon as one
returns a similarity >= its OWN configured stop_threshold, the search stops
immediately: no further providers are called, no further quota is spent.
This is the single most important behavior here — see the "early stop"
requirement this module exists to satisfy.

Design notes:
  - The image is hashed with SHA-256 exactly once, at the top of search().
    That digest is the cache key. Nothing downstream re-hashes or re-reads
    the bytes.
  - Only the first URL-based provider reached in a request needs a temporary
    public URL; it's created lazily and reused for any later URL-based
    provider in the same request. If every enabled provider takes inline
    bytes (e.g. only Google Vision), hosting is never created — zero cost.
  - Every provider attempt is wrapped in its own try/except: one provider's
    failure is logged and the chain moves on, it never fails the request.
  - Nothing here ever logs image bytes.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from typing import Optional

from app.observability.telemetry import get_logger
from app.reverse_search.cache import ReverseSearchCache
from app.reverse_search.models import ReverseSearchResult
from app.reverse_search.providers.base import ProviderError, ReverseSearchProvider
from app.reverse_search.temp_hosting import TempHostingUnavailableError, TempImageHost

logger = get_logger(__name__)


@dataclass(frozen=True)
class ProviderSlot:
    """One entry in the priority-ordered provider chain."""
    name: str
    provider: ReverseSearchProvider
    priority: int
    stop_threshold: float
    timeout_s: float
    requires_public_url: bool


class ReverseSearchOrchestrator:
    def __init__(
        self,
        slots: list[ProviderSlot],
        cache: ReverseSearchCache,
        temp_host: Optional[TempImageHost],
        max_providers: int,
    ) -> None:
        ordered = sorted(slots, key=lambda s: s.priority)
        self._slots = ordered[:max_providers] if max_providers and max_providers > 0 else ordered
        self._cache = cache
        self._temp_host = temp_host

    @property
    def temp_host(self) -> Optional[TempImageHost]:
        """Exposed so routes.py can register the internal image-serving
        endpoint against the same TempImageHost instance used here."""
        return self._temp_host

    def search(self, image_bytes: bytes, content_type: str = "application/octet-stream") -> ReverseSearchResult:
        t0 = time.perf_counter()
        digest = hashlib.sha256(image_bytes).hexdigest()

        cached = self._cache.get(digest)
        if cached is not None:
            result = cached.with_timing(int((time.perf_counter() - t0) * 1000), cache_hit=True)
            logger.info(
                "reverse_search_completed",
                cache_hit=True, found=result.found, provider=result.provider,
                similarity=result.similarity, elapsed_ms=result.elapsed_ms,
                stop_reason="cache_hit",
            )
            return result

        if not self._slots:
            elapsed_ms = int((time.perf_counter() - t0) * 1000)
            logger.warning("reverse_search_no_providers_available", elapsed_ms=elapsed_ms)
            return ReverseSearchResult(
                found=False, website=None, url=None, similarity=0.0, provider=None,
                elapsed_ms=elapsed_ms, stop_reason="no_providers_available",
            )

        best_match = None
        best_provider: Optional[str] = None
        hosted_url: Optional[str] = None

        for slot in self._slots:
            image_url = None
            if slot.requires_public_url:
                if hosted_url is None:
                    if self._temp_host is None:
                        logger.warning(
                            "reverse_search_provider_skipped",
                            provider=slot.name, status="hosting_unavailable",
                        )
                        continue
                    try:
                        hosted_url = self._temp_host.host(image_bytes, content_type=content_type)
                        self._temp_host.warn_if_unreachable(logger)
                    except TempHostingUnavailableError as exc:
                        logger.warning(
                            "reverse_search_provider_skipped",
                            provider=slot.name, status="hosting_unavailable", reason=str(exc),
                        )
                        continue
                image_url = hosted_url

            provider_t0 = time.perf_counter()
            try:
                match = slot.provider.search(
                    image_bytes=image_bytes, image_url=image_url, timeout_s=slot.timeout_s
                )
            except ProviderError as exc:
                logger.warning(
                    "reverse_search_provider_error",
                    provider=slot.name, status=type(exc).__name__, error=str(exc),
                    elapsed_ms=round((time.perf_counter() - provider_t0) * 1000, 1),
                )
                continue
            except Exception as exc:  # never let one adapter's bug fail the whole request
                logger.error(
                    "reverse_search_provider_unexpected_error",
                    provider=slot.name, error=str(exc), exc_info=True,
                )
                continue

            provider_elapsed_ms = round((time.perf_counter() - provider_t0) * 1000, 1)

            if match is None:
                logger.info(
                    "reverse_search_provider_attempt",
                    provider=slot.name, status="no_match", elapsed_ms=provider_elapsed_ms,
                )
                continue

            logger.info(
                "reverse_search_provider_attempt",
                provider=slot.name, status="match", similarity=match.similarity,
                elapsed_ms=provider_elapsed_ms,
            )

            if best_match is None or match.similarity > best_match.similarity:
                best_match, best_provider = match, slot.name

            if match.similarity >= slot.stop_threshold:
                elapsed_ms = int((time.perf_counter() - t0) * 1000)
                result = ReverseSearchResult(
                    found=True, website=match.website, url=match.url,
                    similarity=match.similarity, provider=slot.name,
                    elapsed_ms=elapsed_ms, stop_reason="threshold_met",
                )
                self._cache.set(digest, result)
                logger.info(
                    "reverse_search_completed", cache_hit=False, found=True,
                    provider=slot.name, similarity=match.similarity,
                    elapsed_ms=elapsed_ms, stop_reason="threshold_met",
                )
                return result

        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        if best_match is not None:
            result = ReverseSearchResult(
                found=True, website=best_match.website, url=best_match.url,
                similarity=best_match.similarity, provider=best_provider,
                elapsed_ms=elapsed_ms, stop_reason="providers_exhausted",
            )
        else:
            result = ReverseSearchResult(
                found=False, website=None, url=None, similarity=0.0, provider=None,
                elapsed_ms=elapsed_ms, stop_reason="providers_exhausted",
            )
        self._cache.set(digest, result)
        logger.info(
            "reverse_search_completed", cache_hit=False, found=result.found,
            provider=result.provider, similarity=result.similarity,
            elapsed_ms=elapsed_ms, stop_reason="providers_exhausted",
        )
        return result
