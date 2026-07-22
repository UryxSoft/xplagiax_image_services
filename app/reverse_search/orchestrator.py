"""
Early-stop orchestrator — the core cost-saving logic of this module.

Two modes, chosen by config.mode (default "cost") or a per-request override:

  cost (default) — providers are tried in priority order (lowest number
    first), sequentially. As soon as one returns a similarity >= its OWN
    configured stop_threshold, the search stops immediately: no further
    providers are called, no further quota is spent.

  latency — every configured provider is called CONCURRENTLY via gevent.
    Total wall time is bounded by the slowest provider's timeout instead of
    the sum of all of them, but every provider's quota is always spent since
    there's no "stop before calling" once they're all in flight together.
    This is an explicit cost-for-speed trade, opt-in only.

Design notes:
  - The image is hashed with SHA-256 exactly once, at the top of search().
    That digest is the cache key. Nothing downstream re-hashes or re-reads
    the bytes.
  - request_deadline_s bounds how long the RETRY budget can run in "cost"
    mode: once past it, a provider's own retry_call() stops retrying and
    moves on, and the sequential loop stops trying further providers. It
    does not shrink an individual provider's own per-call timeout — only
    the extra time retries would otherwise add.
  - Cross-provider corroboration: if two DIFFERENT providers report a match
    on the same hostname within one request, that's independent evidence,
    so the similarity gets a configurable bonus (see _adjust_similarity).
    Optional trusted/distrusted domain lists apply the same way and are
    empty by default — they only do anything once explicitly configured.
  - Only the first URL-based provider reached needs a temporary public URL;
    it's created once (lazily in "cost" mode, upfront in "latency" mode
    since all providers may need it at once) and reused for the rest of the
    request. If every enabled provider takes inline bytes (Google Vision),
    hosting is never created — zero cost in the common case.
  - Every provider attempt is wrapped in its own try/except: one provider's
    failure is logged and the chain moves on, it never fails the request.
  - Nothing here ever logs image bytes.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from typing import Optional

import gevent

from app.config import ReverseSearchConfig
from app.observability.telemetry import get_logger, get_metrics
from app.reverse_search.cache import ReverseSearchCache
from app.reverse_search.models import ProviderMatch, ReverseSearchResult
from app.reverse_search.providers.base import ProviderError, ReverseSearchProvider, hostname_of
from app.reverse_search.temp_hosting import TempHostingUnavailableError, TempImageHost
from app.services.circuit_breaker import CircuitBreaker

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
    # Redis-backed, sustained-failure breaker (None = no breaker, always allowed).
    # Distinct from retry_call()'s deadline: that bounds ONE request's retry
    # time, this remembers "this provider has been failing across MANY
    # requests" so a persistently-down provider stops being tried at all
    # until its recovery window passes.
    breaker: Optional[CircuitBreaker] = None


class ReverseSearchOrchestrator:
    def __init__(
        self,
        slots: list[ProviderSlot],
        cache: ReverseSearchCache,
        temp_host: Optional[TempImageHost],
        config: ReverseSearchConfig,
    ) -> None:
        ordered = sorted(slots, key=lambda s: s.priority)
        max_providers = config.max_providers
        self._slots = ordered[:max_providers] if max_providers and max_providers > 0 else ordered
        self._cache = cache
        self._temp_host = temp_host
        self._config = config

    @property
    def temp_host(self) -> Optional[TempImageHost]:
        """Exposed so routes.py can register the internal image-serving
        endpoint against the same TempImageHost instance used here."""
        return self._temp_host

    def search(
        self, image_bytes: bytes, content_type: str = "application/octet-stream",
        mode: Optional[str] = None,
    ) -> ReverseSearchResult:
        t0 = time.perf_counter()
        digest = hashlib.sha256(image_bytes).hexdigest()

        cached = self._cache.get(digest)
        if cached is not None:
            result = cached.with_timing(int((time.perf_counter() - t0) * 1000), cache_hit=True)
            get_metrics().reverse_search_completed_total.labels(
                found=str(result.found).lower(), stop_reason="cache_hit", cache_hit="true",
            ).inc()
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

        effective_mode = mode or self._config.mode
        if effective_mode == "latency":
            result = self._search_parallel(image_bytes, content_type, t0)
        else:
            result = self._search_sequential(image_bytes, content_type, t0)

        self._cache.set(digest, result)
        get_metrics().reverse_search_completed_total.labels(
            found=str(result.found).lower(), stop_reason=result.stop_reason or "unknown", cache_hit="false",
        ).inc()
        logger.info(
            "reverse_search_completed", cache_hit=False, found=result.found,
            provider=result.provider, similarity=result.similarity,
            elapsed_ms=result.elapsed_ms, stop_reason=result.stop_reason, mode=effective_mode,
        )
        return result

    # ------------------------------------------------------------------
    # Shared helpers — HTTP call w/ logging+metrics, similarity adjustment,
    # and lazy hosting. Orchestration (loop vs concurrent dispatch, the stop
    # decision) differs between the two modes below; this part doesn't.
    # ------------------------------------------------------------------

    def _call_provider(
        self, slot: ProviderSlot, image_bytes: bytes, image_url: Optional[str], deadline: float,
    ) -> Optional[ProviderMatch]:
        if slot.breaker is not None and not slot.breaker.is_allowed():
            logger.warning("reverse_search_circuit_open", provider=slot.name)
            get_metrics().reverse_search_provider_duration.labels(provider=slot.name, status="circuit_open").observe(0.0)
            return None

        provider_t0 = time.perf_counter()
        try:
            match = slot.provider.search(
                image_bytes=image_bytes, image_url=image_url, timeout_s=slot.timeout_s, deadline=deadline,
            )
        except ProviderError as exc:
            status = type(exc).__name__
            elapsed_s = time.perf_counter() - provider_t0
            logger.warning(
                "reverse_search_provider_error",
                provider=slot.name, status=status, error=str(exc), elapsed_ms=round(elapsed_s * 1000, 1),
            )
            get_metrics().reverse_search_provider_duration.labels(provider=slot.name, status=status).observe(elapsed_s)
            if slot.breaker is not None:
                slot.breaker.record_failure()
            return None
        except Exception as exc:  # never let one adapter's bug fail the whole request
            elapsed_s = time.perf_counter() - provider_t0
            logger.error(
                "reverse_search_provider_unexpected_error", provider=slot.name, error=str(exc), exc_info=True,
            )
            get_metrics().reverse_search_provider_duration.labels(
                provider=slot.name, status="unexpected_error",
            ).observe(elapsed_s)
            if slot.breaker is not None:
                slot.breaker.record_failure()
            return None

        if slot.breaker is not None:
            slot.breaker.record_success()

        elapsed_s = time.perf_counter() - provider_t0
        if match is None:
            logger.info(
                "reverse_search_provider_attempt", provider=slot.name, status="no_match",
                elapsed_ms=round(elapsed_s * 1000, 1),
            )
            get_metrics().reverse_search_provider_duration.labels(provider=slot.name, status="no_match").observe(elapsed_s)
            return None

        logger.info(
            "reverse_search_provider_attempt", provider=slot.name, status="match",
            similarity=match.similarity, elapsed_ms=round(elapsed_s * 1000, 1),
        )
        get_metrics().reverse_search_provider_duration.labels(provider=slot.name, status="match").observe(elapsed_s)
        get_metrics().reverse_search_similarity.observe(match.similarity)
        return match

    def _adjust_similarity(
        self, match: ProviderMatch, provider_name: str, seen_hosts: dict[str, str],
    ) -> ProviderMatch:
        """
        Boost similarity when a DIFFERENT provider already reported the same
        hostname in this same request (independent corroborating evidence),
        and apply optional trusted/distrusted domain adjustments. Both
        adjustments are additive/subtractive on top of the provider's own
        heuristic score, then clamped to [0, 99.9].
        """
        host = hostname_of(match.url)
        similarity = match.similarity
        corroborated = False

        prior_provider = seen_hosts.get(host)
        if prior_provider is not None and prior_provider != provider_name:
            similarity += self._config.corroboration_bonus
            corroborated = True
        if host in self._config.trusted_domains:
            similarity += self._config.trusted_domain_bonus
        if host in self._config.distrusted_domains:
            similarity -= self._config.distrusted_domain_penalty
        similarity = max(0.0, min(99.9, similarity))

        if host not in seen_hosts:
            seen_hosts[host] = provider_name

        if similarity != match.similarity:
            logger.info(
                "reverse_search_similarity_adjusted", provider=provider_name, host=host,
                original=match.similarity, adjusted=similarity, corroborated=corroborated,
            )
            return ProviderMatch(website=match.website, url=match.url, similarity=similarity)
        return match

    def _ensure_hosted(self, image_bytes: bytes, content_type: str) -> Optional[str]:
        if self._temp_host is None:
            return None
        try:
            url = self._temp_host.host(image_bytes, content_type=content_type)
            self._temp_host.warn_if_unreachable(logger)
            return url
        except TempHostingUnavailableError as exc:
            logger.warning("reverse_search_hosting_unavailable", reason=str(exc))
            return None

    # ------------------------------------------------------------------
    # Mode: cost (default) — sequential Early Stop
    # ------------------------------------------------------------------

    def _search_sequential(self, image_bytes: bytes, content_type: str, t0: float) -> ReverseSearchResult:
        deadline = t0 + self._config.request_deadline_s
        best_match: Optional[ProviderMatch] = None
        best_provider: Optional[str] = None
        hosted_url: Optional[str] = None
        seen_hosts: dict[str, str] = {}

        for slot in self._slots:
            if time.perf_counter() >= deadline:
                logger.warning("reverse_search_deadline_exceeded", provider_skipped=slot.name)
                break

            image_url = None
            if slot.requires_public_url:
                if hosted_url is None:
                    hosted_url = self._ensure_hosted(image_bytes, content_type)
                    if hosted_url is None:
                        logger.warning(
                            "reverse_search_provider_skipped", provider=slot.name, status="hosting_unavailable",
                        )
                        continue
                image_url = hosted_url

            match = self._call_provider(slot, image_bytes, image_url, deadline)
            if match is None:
                continue

            match = self._adjust_similarity(match, slot.name, seen_hosts)

            if best_match is None or match.similarity > best_match.similarity:
                best_match, best_provider = match, slot.name

            if match.similarity >= slot.stop_threshold:
                elapsed_ms = int((time.perf_counter() - t0) * 1000)
                return ReverseSearchResult(
                    found=True, website=match.website, url=match.url,
                    similarity=match.similarity, provider=slot.name,
                    elapsed_ms=elapsed_ms, stop_reason="threshold_met",
                )

        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        if best_match is not None:
            return ReverseSearchResult(
                found=True, website=best_match.website, url=best_match.url,
                similarity=best_match.similarity, provider=best_provider,
                elapsed_ms=elapsed_ms, stop_reason="providers_exhausted",
            )
        return ReverseSearchResult(
            found=False, website=None, url=None, similarity=0.0, provider=None,
            elapsed_ms=elapsed_ms, stop_reason="providers_exhausted",
        )

    # ------------------------------------------------------------------
    # Mode: latency — every provider concurrently, bounded by the slowest
    # one instead of the sum. Opt-in: always spends every provider's quota.
    # ------------------------------------------------------------------

    def _search_parallel(self, image_bytes: bytes, content_type: str, t0: float) -> ReverseSearchResult:
        deadline = t0 + self._config.request_deadline_s
        hosted_url: Optional[str] = None
        if any(s.requires_public_url for s in self._slots):
            hosted_url = self._ensure_hosted(image_bytes, content_type)

        def attempt(slot: ProviderSlot):
            image_url = hosted_url if slot.requires_public_url else None
            if slot.requires_public_url and image_url is None:
                logger.warning(
                    "reverse_search_provider_skipped", provider=slot.name, status="hosting_unavailable",
                )
                return slot, None
            return slot, self._call_provider(slot, image_bytes, image_url, deadline)

        greenlets = [gevent.spawn(attempt, slot) for slot in self._slots]
        gevent.joinall(greenlets)

        seen_hosts: dict[str, str] = {}
        candidates: list[tuple[ProviderSlot, ProviderMatch]] = []
        for g in greenlets:
            slot, match = g.value
            if match is None:
                continue
            candidates.append((slot, self._adjust_similarity(match, slot.name, seen_hosts)))

        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        winners = [(slot, match) for slot, match in candidates if match.similarity >= slot.stop_threshold]
        if winners:
            slot, match = max(winners, key=lambda sm: sm[1].similarity)
            return ReverseSearchResult(
                found=True, website=match.website, url=match.url,
                similarity=match.similarity, provider=slot.name,
                elapsed_ms=elapsed_ms, stop_reason="threshold_met_parallel",
            )
        if candidates:
            slot, match = max(candidates, key=lambda sm: sm[1].similarity)
            return ReverseSearchResult(
                found=True, website=match.website, url=match.url,
                similarity=match.similarity, provider=slot.name,
                elapsed_ms=elapsed_ms, stop_reason="providers_exhausted",
            )
        return ReverseSearchResult(
            found=False, website=None, url=None, similarity=0.0, provider=None,
            elapsed_ms=elapsed_ms, stop_reason="providers_exhausted",
        )
