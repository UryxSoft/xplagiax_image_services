"""
Smart API Rotator for external reverse-image and patent search APIs.

Improvements over the original:
  1. ATOMIC COUNTERS: usage tracked in Redis (incr) → process/thread safe
  2. DYNAMIC SCORING: each provider gets a score based on error rate + latency
  3. CIRCUIT BREAKER: providers are penalised for rate-limit / 5xx errors
  4. EXPLICIT TIMEOUTS: every HTTP call has a timeout — no indefinite blocks
  5. RETRY WITH EXPONENTIAL BACKOFF: tenacity handles transient failures
  6. BACKGROUND HEALTH CHECKS: score recovery happens automatically
  7. FALLBACK TO FILE: if Redis unavailable, uses atomic file locking
"""

from __future__ import annotations

import collections
import datetime
import fcntl
import json
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import requests
from requests.exceptions import ConnectionError, HTTPError, Timeout
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.observability.telemetry import get_logger, get_metrics

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class AllApisExhaustedException(Exception):
    """All providers have hit their quota or are penalised."""


class ProviderRateLimitError(Exception):
    """Provider returned HTTP 429."""
    def __init__(self, provider: str):
        self.provider = provider
        super().__init__(f"Rate limited by {provider}")


# ---------------------------------------------------------------------------
# Provider config
# ---------------------------------------------------------------------------

@dataclass
class ProviderConfig:
    name: str
    api_key: str
    monthly_limit: int
    base_url_search: str
    base_url_patents: str


# ---------------------------------------------------------------------------
# Score tracker (in-memory, per process)
# ---------------------------------------------------------------------------

@dataclass
class ProviderScore:
    name: str
    score: float = 1.0                              # 0.0 (bad) → 1.0 (perfect)
    penalty_until: Optional[datetime.datetime] = None
    # Rolling window of last 20 response times (ms)
    response_times: collections.deque = field(
        default_factory=lambda: collections.deque(maxlen=20)
    )
    error_window: collections.deque = field(
        default_factory=lambda: collections.deque(maxlen=10)  # last 10: True=error
    )
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def record(self, success: bool, response_ms: float) -> None:
        with self._lock:
            self.error_window.append(not success)
            self.response_times.append(response_ms)
            self._recalculate()

    def penalise(self, duration_seconds: int = 3600) -> None:
        self.penalty_until = datetime.datetime.utcnow() + datetime.timedelta(
            seconds=duration_seconds
        )
        self.score = 0.0
        logger.warning(
            "api_provider_penalised",
            provider=self.name,
            until=self.penalty_until.isoformat(),
        )

    @property
    def is_penalised(self) -> bool:
        if self.penalty_until is None:
            return False
        if datetime.datetime.utcnow() > self.penalty_until:
            self.penalty_until = None  # auto-recover
            return False
        return True

    def _recalculate(self) -> None:
        error_rate = (
            sum(self.error_window) / len(self.error_window)
            if self.error_window
            else 0.0
        )
        avg_ms = (
            sum(self.response_times) / len(self.response_times)
            if self.response_times
            else 1000.0
        )
        # Normalise latency: 0ms → 1.0, ≥5000ms → 0.0
        time_score = max(0.0, 1.0 - (avg_ms / 5000.0))
        self.score = (1.0 - error_rate) * 0.7 + time_score * 0.3


# ---------------------------------------------------------------------------
# Usage persistence (Redis-first, file fallback)
# ---------------------------------------------------------------------------

class UsageTracker:
    """
    Tracks monthly API usage with atomic increments.
    Redis-first; falls back to file with fcntl for multi-process safety.
    """

    def __init__(self, cache, usage_file: str) -> None:
        self._cache = cache           # CacheClient instance (may be None)
        self._usage_file = usage_file
        self._lock = threading.Lock()

    def _year_month(self) -> str:
        return datetime.datetime.utcnow().strftime("%Y-%m")

    def increment(self, provider: str) -> int:
        """Atomically increment and return new count."""
        ym = self._year_month()
        if self._cache and self._cache.available:
            result = self._cache.increment_api_usage(provider, ym)
            if result is not None:
                get_metrics().api_calls_total.labels(
                    provider=provider, operation="increment", status="ok"
                ).inc()
                return result
        # Fallback: file-based with process lock
        return self._file_increment(provider, ym)

    def get_count(self, provider: str) -> int:
        ym = self._year_month()
        if self._cache and self._cache.available:
            count = self._cache.get_api_usage(provider, ym)
            if count is not None:
                return count
        return self._file_get(provider, ym)

    def _file_increment(self, provider: str, ym: str) -> int:
        with self._lock:
            try:
                with open(self._usage_file, "a+") as f:
                    fcntl.flock(f, fcntl.LOCK_EX)
                    try:
                        f.seek(0)
                        raw = f.read()
                        data = json.loads(raw) if raw.strip() else {}
                        key = f"{provider}:{ym}"
                        data[key] = data.get(key, 0) + 1
                        f.seek(0)
                        f.truncate()
                        json.dump(data, f)
                        return data[key]
                    finally:
                        fcntl.flock(f, fcntl.LOCK_UN)
            except Exception as exc:
                logger.warning("usage_file_write_failed", error=str(exc))
                return 0

    def _file_get(self, provider: str, ym: str) -> int:
        try:
            with open(self._usage_file, "r") as f:
                data = json.load(f)
            return data.get(f"{provider}:{ym}", 0)
        except Exception:
            return 0


# ---------------------------------------------------------------------------
# Main rotator
# ---------------------------------------------------------------------------

class SmartApiRotator:
    """
    Intelligent API rotator with dynamic scoring, circuit breakers,
    and atomic usage tracking.
    """

    def __init__(
        self,
        providers: list[ProviderConfig],
        usage_tracker: UsageTracker,
        request_timeout_s: float = 15.0,
        max_retries: int = 3,
        health_check_interval: int = 300,
    ) -> None:
        self._providers = {p.name: p for p in providers}
        self._usage = usage_tracker
        self._timeout = request_timeout_s
        self._max_retries = max_retries
        self._scores: dict[str, ProviderScore] = {
            p.name: ProviderScore(name=p.name) for p in providers
        }
        self._start_health_checker(health_check_interval)

    # ------------------------------------------------------------------
    # Provider selection
    # ------------------------------------------------------------------

    def _available_providers(self, monthly_limit_override: dict = None) -> list[str]:
        """Return providers ordered by score, excluding penalised / exhausted."""
        candidates = []
        for name, cfg in self._providers.items():
            score_obj = self._scores[name]
            if score_obj.is_penalised:
                continue
            used = self._usage.get_count(name)
            limit = cfg.monthly_limit
            if used >= limit * 0.9:
                logger.warning(
                    "api_provider_near_limit",
                    provider=name,
                    used=used,
                    limit=limit,
                )
                if used >= limit:
                    continue
            candidates.append((name, score_obj.score))

        if not candidates:
            raise AllApisExhaustedException(
                "All API providers are exhausted or penalised. "
                "Check usage limits or wait for the next month."
            )

        # Sort by score descending
        candidates.sort(key=lambda x: x[1], reverse=True)
        return [name for name, _ in candidates]

    # ------------------------------------------------------------------
    # HTTP execution with retry + scoring
    # ------------------------------------------------------------------

    def _execute(self, provider: str, url: str, params: dict) -> dict:
        """Make a single HTTP request, recording score and usage."""
        start = time.perf_counter()
        try:
            response = requests.get(url, params=params, timeout=self._timeout)
        except (Timeout, ConnectionError) as exc:
            elapsed_ms = (time.perf_counter() - start) * 1000
            self._scores[provider].record(success=False, response_ms=elapsed_ms)
            get_metrics().api_calls_total.labels(
                provider=provider, operation="search", status="network_error"
            ).inc()
            raise  # tenacity will retry

        elapsed_ms = (time.perf_counter() - start) * 1000

        if response.status_code == 429:
            self._scores[provider].penalise(3600)
            get_metrics().api_calls_total.labels(
                provider=provider, operation="search", status="rate_limited"
            ).inc()
            raise ProviderRateLimitError(provider)

        if response.status_code >= 500:
            self._scores[provider].record(success=False, response_ms=elapsed_ms)
            get_metrics().api_calls_total.labels(
                provider=provider, operation="search", status=str(response.status_code)
            ).inc()
            response.raise_for_status()

        if response.status_code != 200:
            # 4xx from us (bad params) — don't retry, but don't penalise provider
            get_metrics().api_calls_total.labels(
                provider=provider, operation="search", status=str(response.status_code)
            ).inc()
            raise HTTPError(
                f"API {provider} returned {response.status_code}: {response.text[:200]}"
            )

        self._scores[provider].record(success=True, response_ms=elapsed_ms)
        self._usage.increment(provider)
        get_metrics().api_calls_total.labels(
            provider=provider, operation="search", status="ok"
        ).inc()
        get_metrics().api_usage_remaining.labels(provider=provider).set(
            self._providers[provider].monthly_limit - self._usage.get_count(provider)
        )

        logger.info(
            "api_call_success",
            provider=provider,
            url=url,
            status=response.status_code,
            elapsed_ms=round(elapsed_ms, 1),
        )
        return response.json()

    def _extract_image_sources(self, response_data: dict) -> list:
        """
        Extract direct image URLs and related sources from a SerpApi or ZenSerp reverse image search response.
        Google Reverse Image Search JSON structure typically hides matches under:
        - `image_results` (ZenSerp / older SerpApi)
        - `visual_matches` (SerpApi new format)
        - `inline_images`
        - `knowledge_graph` (identifications)
        """
        extracted = []

        # 1. Look for 'visual_matches' (SerpApi standard for identical / visually similar)
        if "visual_matches" in response_data:
            for item in response_data["visual_matches"]:
                if "link" in item:
                    extracted.append({
                        "title": item.get("title", ""),
                        "source": item.get("source", ""),
                        "url": item["link"],
                        "thumbnail": item.get("thumbnail", "")
                    })

        # 2. Look for 'image_results' (Fallback for ZenSerp or legacy Google Images)
        if "image_results" in response_data:
            for item in response_data["image_results"]:
                # Often ZenSerp uses 'sourceUrl' or SerpApi uses 'link'
                url = item.get("link") or item.get("sourceUrl")
                if url:
                    extracted.append({
                        "title": item.get("title", ""),
                        "source": item.get("source", ""),
                        "url": url,
                        "thumbnail": item.get("thumbnail", "")
                    })

        # 3. Look for 'inline_images'
        if "inline_images" in response_data:
            for item in response_data["inline_images"]:
                if "link" in item:
                    extracted.append({
                        "title": item.get("title", "Inline Image"),
                        "source": item.get("source", ""),
                        "url": item["link"],
                        "thumbnail": item.get("thumbnail", "")
                    })

        # Remove duplicates by URL
        seen = set()
        unique_extracted = []
        for item in extracted:
            if item["url"] not in seen:
                seen.add(item["url"])
                unique_extracted.append(item)

        return unique_extracted

    def _execute_with_fallback(self, build_params: Callable[[str], tuple[str, dict]]) -> dict:
        """
        Try providers in score order. On failure, try next provider.
        `build_params(provider_name)` returns (url, params) for that provider.
        """
        ordered = self._available_providers()
        last_exc: Optional[Exception] = None

        for provider_name in ordered:

            @retry(
                stop=stop_after_attempt(self._max_retries),
                wait=wait_exponential(multiplier=1, min=1, max=10),
                retry=retry_if_exception_type((Timeout, ConnectionError)),
                reraise=True,
            )
            def attempt():
                url, params = build_params(provider_name)
                return self._execute(provider_name, url, params)

            try:
                return attempt()
            except ProviderRateLimitError:
                last_exc = None  # already penalised, try next
                continue
            except Exception as exc:
                last_exc = exc
                logger.warning(
                    "api_provider_failed_trying_next",
                    provider=provider_name,
                    error=str(exc),
                )
                continue

        raise last_exc or AllApisExhaustedException("All providers failed")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reverse_image_search(self, image_url: str, num_results: int = 10) -> dict:
        def build(provider: str) -> tuple[str, dict]:
            if provider == "serpapi":
                return (
                    "https://serpapi.com/search.json",
                    {
                        "engine": "google_reverse_image",
                        "image_url": image_url,
                        "api_key": self._providers[provider].api_key,
                        "num": num_results,
                    },
                )
            elif provider == "zenserp":
                return (
                    "https://app.zenserp.com/api/v2/search",
                    {
                        "apikey": self._providers[provider].api_key,
                        "q": "*",
                        "image_url": image_url,
                        "num": num_results,
                    },
                )
            raise ValueError(f"Unknown provider: {provider}")

        response_data = self._execute_with_fallback(build)

        # Inject our extracted sources to make the response robust and friendly
        # Since standard JSON from Google is deeply nested
        extracted_sources = self._extract_image_sources(response_data)
        if extracted_sources:
            response_data["extracted_visual_matches"] = extracted_sources[:num_results]

        return response_data

    def patent_text_search(self, query: str, num_results: int = 10) -> dict:
        def build(provider: str) -> tuple[str, dict]:
            if provider == "serpapi":
                return (
                    "https://serpapi.com/search.json",
                    {
                        "engine": "google_patents",
                        "q": query,
                        "api_key": self._providers[provider].api_key,
                        "num": min(max(num_results, 10), 100),
                    },
                )
            elif provider == "zenserp":
                return (
                    "https://app.zenserp.com/api/v2/search",
                    {
                        "apikey": self._providers[provider].api_key,
                        "q": query,
                        "tbm": "patent",
                        "num": num_results,
                    },
                )
            raise ValueError(f"Unknown provider: {provider}")

        return self._execute_with_fallback(build)

    def patent_image_search(self, image_url: str, num_results: int = 10) -> dict:
        """
        Two-step: reverse image → extract keywords → patent search.
        Counts as 2 API calls.
        """
        reverse = self.reverse_image_search(image_url, num_results=3)

        # Try to extract keywords from our robust extraction first
        if "extracted_visual_matches" in reverse and reverse["extracted_visual_matches"]:
            keywords = " ".join(r.get("title", "") for r in reverse["extracted_visual_matches"][:3]).strip()
        else:
            # Fallback to standard
            keywords = " ".join(
                r.get("title", "")
                for r in (reverse.get("image_results") or reverse.get("visual_matches") or [])[:3]
            ).strip()

        if not keywords:
            raise ValueError(
                "Could not extract keywords from reverse image search results."
            )

        return self.patent_text_search(keywords, num_results)

    def get_patent_details(self, patent_id: str) -> dict:
        def build(provider: str) -> tuple[str, dict]:
            if provider == "serpapi":
                return (
                    "https://serpapi.com/search.json",
                    {
                        "engine": "google_patents",
                        "id": patent_id,
                        "api_key": self._providers[provider].api_key,
                    },
                )
            raise ValueError(f"Patent details not supported by provider: {provider}")

        return self._execute_with_fallback(build)

    def get_usage_status(self) -> dict:
        status = {}
        ym = datetime.datetime.utcnow().strftime("%Y-%m")
        for name, cfg in self._providers.items():
            used = self._usage.get_count(name)
            score_obj = self._scores[name]
            status[name] = {
                "used":            used,
                "limit":           cfg.monthly_limit,
                "remaining":       max(0, cfg.monthly_limit - used),
                "percent_used":    round(used / cfg.monthly_limit * 100, 1) if cfg.monthly_limit else 0,
                "score":           round(score_obj.score, 3),
                "is_penalised":    score_obj.is_penalised,
                "penalty_until":   score_obj.penalty_until.isoformat() if score_obj.penalty_until else None,
                "year_month":      ym,
            }
        return status

    # ------------------------------------------------------------------
    # Background health checker
    # ------------------------------------------------------------------

    def _start_health_checker(self, interval: int) -> None:
        def loop():
            while True:
                time.sleep(interval)
                self._run_health_checks()

        t = threading.Thread(target=loop, daemon=True, name="api-health-checker")
        t.start()

    def _run_health_checks(self) -> None:
        """Lightweight availability check — does NOT consume quota."""
        for name in self._providers:
            if self._scores[name].is_penalised:
                # Allow gradual score recovery after penalty expires
                continue
            try:
                # SerpApi has a public status endpoint
                if name == "serpapi":
                    resp = requests.head("https://serpapi.com/", timeout=5)
                    ok = resp.status_code < 500
                elif name == "zenserp":
                    resp = requests.head("https://app.zenserp.com/", timeout=5)
                    ok = resp.status_code < 500
                else:
                    ok = True

                if ok:
                    self._scores[name].record(success=True, response_ms=200)
                else:
                    self._scores[name].record(success=False, response_ms=5000)
            except Exception:
                self._scores[name].record(success=False, response_ms=5000)
