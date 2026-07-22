"""
Tests for the reverse-image-search module: early-stop orchestrator, cache
TTL selection, and the shared retry policy. Uses the same unittest.TestCase
style as tests/test_suite.py so both files run under either
`pytest tests/ -v` or `python -m unittest`.

Run:
    pytest tests/test_reverse_search.py -v
"""

from __future__ import annotations

import unittest
from typing import Optional
from unittest.mock import MagicMock, patch

from app.observability.telemetry import init_metrics
from app.reverse_search.cache import ReverseSearchCache
from app.reverse_search.models import ProviderMatch, ReverseSearchResult
from app.reverse_search.orchestrator import ProviderSlot, ReverseSearchOrchestrator
from app.reverse_search.providers.base import (
    ProviderAuthError,
    ProviderRateLimitError,
    ProviderTransientError,
    ReverseSearchProvider,
    retry_call,
)
from app.reverse_search.providers.mungfali import MungfaliProvider

# The orchestrator records Prometheus metrics on every call; init_metrics()
# just needs to have run once per process (idempotent — see telemetry.py),
# regardless of which test class happens to run first.
init_metrics("test-reverse-search")


class _FakeProvider(ReverseSearchProvider):
    """Test double: returns a fixed match/None, or raises a fixed exception,
    and records every call so tests can assert whether it was invoked."""

    def __init__(self, name: str, match: Optional[ProviderMatch] = None, error: Optional[Exception] = None):
        self.name = name
        self._match = match
        self._error = error
        self.calls = 0

    def search(self, *, image_bytes, image_url, timeout_s, deadline=None):
        self.calls += 1
        if self._error is not None:
            raise self._error
        return self._match


def _slot(provider: ReverseSearchProvider, priority: int, stop_threshold: float,
          requires_public_url: bool = False, breaker=None) -> ProviderSlot:
    return ProviderSlot(
        name=provider.name, provider=provider, priority=priority,
        stop_threshold=stop_threshold, timeout_s=1.0, requires_public_url=requires_public_url,
        breaker=breaker,
    )


class _FakeBreaker:
    """Duck-typed stand-in for app.services.circuit_breaker.CircuitBreaker —
    avoids needing real Redis just to test the skip-when-open behavior."""

    def __init__(self, allowed: bool = True):
        self.allowed = allowed
        self.successes = 0
        self.failures = 0

    def is_allowed(self) -> bool:
        return self.allowed

    def record_success(self) -> None:
        self.successes += 1

    def record_failure(self) -> None:
        self.failures += 1


def _no_cache() -> ReverseSearchCache:
    """A cache that always misses / degrades — the orchestrator must still
    behave correctly with caching effectively disabled."""
    return ReverseSearchCache(cache_client=None, ttl_found=300, ttl_not_found=60)


def _test_rs_config(**overrides):
    """Minimal ReverseSearchConfig for orchestrator-level unit tests (as
    opposed to _build_test_config() below, which builds a full AppConfig for
    Flask-integration tests)."""
    from app.config import ProviderSettings, ReverseSearchConfig
    base = dict(
        enabled=True, max_providers=3, max_retries=1, max_batch_size=20,
        cache_ttl_found=300, cache_ttl_not_found=60,
        public_base_url=None, temp_hosting_ttl=30,
        google_vision_api_key=None,
        google_vision=ProviderSettings(enabled=False, priority=1, stop_threshold=98.0,
                                        timeout_s=1.0, requires_public_url=False),
        serper_api_key=None,
        serper=ProviderSettings(enabled=False, priority=2, stop_threshold=95.0,
                                 timeout_s=1.0, requires_public_url=True),
        mungfali_api_key=None,
        mungfali=ProviderSettings(enabled=False, priority=3, stop_threshold=90.0,
                                   timeout_s=1.0, requires_public_url=True),
        request_deadline_s=30.0,
        mode="cost",
        batch_concurrency=10,
        corroboration_bonus=5.0,
        trusted_domains=frozenset(),
        trusted_domain_bonus=3.0,
        distrusted_domains=frozenset(),
        distrusted_domain_penalty=10.0,
    )
    base.update(overrides)
    return ReverseSearchConfig(**base)


# ===========================================================================
# TestEarlyStopOrchestrator
# ===========================================================================

class TestEarlyStopOrchestrator(unittest.TestCase):

    def test_stops_at_first_provider_when_threshold_met(self):
        p1 = _FakeProvider("p1", match=ProviderMatch(website="Wikipedia", url="https://es.wikipedia.org/x", similarity=99.7))
        p2 = _FakeProvider("p2", match=ProviderMatch(website="Other", url="https://other.example/x", similarity=100.0))

        orch = ReverseSearchOrchestrator(
            slots=[_slot(p1, priority=1, stop_threshold=98.0), _slot(p2, priority=2, stop_threshold=95.0)],
            cache=_no_cache(), temp_host=None, config=_test_rs_config(max_providers=3),
        )
        result = orch.search(b"fake-bytes")

        self.assertTrue(result.found)
        self.assertEqual(result.provider, "p1")
        self.assertEqual(result.similarity, 99.7)
        self.assertEqual(result.stop_reason, "threshold_met")
        self.assertEqual(p1.calls, 1)
        self.assertEqual(p2.calls, 0, "provider 2 must NOT be called once provider 1 met its threshold")

    def test_falls_through_when_below_threshold(self):
        p1 = _FakeProvider("p1", match=ProviderMatch(website="Weak", url="https://weak.example/x", similarity=62.0))
        p2 = _FakeProvider("p2", match=ProviderMatch(website="Strong", url="https://strong.example/x", similarity=99.0))

        orch = ReverseSearchOrchestrator(
            slots=[_slot(p1, priority=1, stop_threshold=98.0), _slot(p2, priority=2, stop_threshold=95.0)],
            cache=_no_cache(), temp_host=None, config=_test_rs_config(max_providers=3),
        )
        result = orch.search(b"fake-bytes")

        self.assertTrue(result.found)
        self.assertEqual(result.provider, "p2")
        self.assertEqual(result.stop_reason, "threshold_met")
        self.assertEqual(p1.calls, 1)
        self.assertEqual(p2.calls, 1)

    def test_no_match_from_any_provider(self):
        p1 = _FakeProvider("p1", match=None)
        p2 = _FakeProvider("p2", match=None)

        orch = ReverseSearchOrchestrator(
            slots=[_slot(p1, priority=1, stop_threshold=98.0), _slot(p2, priority=2, stop_threshold=95.0)],
            cache=_no_cache(), temp_host=None, config=_test_rs_config(max_providers=3),
        )
        result = orch.search(b"fake-bytes")

        self.assertFalse(result.found)
        self.assertIsNone(result.provider)
        self.assertEqual(result.stop_reason, "providers_exhausted")

    def test_best_effort_result_when_no_threshold_is_met(self):
        """Spec: if every provider is tried and none clears its threshold,
        respond with the best result found so far — not a hard failure."""
        p1 = _FakeProvider("p1", match=ProviderMatch(website="Weak", url="https://weak.example/x", similarity=62.0))
        p2 = _FakeProvider("p2", match=ProviderMatch(website="Weaker", url="https://weaker.example/x", similarity=40.0))

        orch = ReverseSearchOrchestrator(
            slots=[_slot(p1, priority=1, stop_threshold=98.0), _slot(p2, priority=2, stop_threshold=98.0)],
            cache=_no_cache(), temp_host=None, config=_test_rs_config(max_providers=3),
        )
        result = orch.search(b"fake-bytes")

        self.assertTrue(result.found)
        self.assertEqual(result.provider, "p1")  # 62.0 > 40.0
        self.assertEqual(result.stop_reason, "providers_exhausted")

    def test_provider_error_does_not_fail_the_request(self):
        """A failing provider must be skipped, never raised to the caller."""
        p1 = _FakeProvider("p1", error=ProviderAuthError("p1", 401))
        p2 = _FakeProvider("p2", match=ProviderMatch(website="Ok", url="https://ok.example/x", similarity=97.0))

        orch = ReverseSearchOrchestrator(
            slots=[_slot(p1, priority=1, stop_threshold=98.0), _slot(p2, priority=2, stop_threshold=95.0)],
            cache=_no_cache(), temp_host=None, config=_test_rs_config(max_providers=3),
        )
        result = orch.search(b"fake-bytes")

        self.assertTrue(result.found)
        self.assertEqual(result.provider, "p2")

    def test_mungfali_stub_is_skipped_cleanly(self):
        """The unverified/experimental Mungfali adapter must behave like any
        other unavailable provider: logged and skipped, chain continues."""
        mungfali = MungfaliProvider(api_key="dummy")
        p2 = _FakeProvider("p2", match=ProviderMatch(website="Ok", url="https://ok.example/x", similarity=97.0))

        orch = ReverseSearchOrchestrator(
            slots=[_slot(mungfali, priority=1, stop_threshold=90.0), _slot(p2, priority=2, stop_threshold=95.0)],
            cache=_no_cache(), temp_host=None, config=_test_rs_config(max_providers=3),
        )
        result = orch.search(b"fake-bytes")

        self.assertTrue(result.found)
        self.assertEqual(result.provider, "p2")

    def test_max_providers_caps_the_chain(self):
        p1 = _FakeProvider("p1", match=ProviderMatch(website="Weak", url="https://weak.example/x", similarity=10.0))
        p2 = _FakeProvider("p2", match=ProviderMatch(website="Weak2", url="https://weak2.example/x", similarity=5.0))
        p3 = _FakeProvider("p3", match=ProviderMatch(website="Strong", url="https://strong.example/x", similarity=99.0))

        orch = ReverseSearchOrchestrator(
            slots=[
                _slot(p1, priority=1, stop_threshold=98.0),
                _slot(p2, priority=2, stop_threshold=98.0),
                _slot(p3, priority=3, stop_threshold=98.0),
            ],
            cache=_no_cache(), temp_host=None, config=_test_rs_config(max_providers=2),
        )
        result = orch.search(b"fake-bytes")

        # p3 (priority 3) is beyond max_providers=2 and must never be tried,
        # even though it's the only one that would have matched confidently.
        self.assertEqual(p1.calls, 1)
        self.assertEqual(p2.calls, 1)
        self.assertEqual(p3.calls, 0)
        self.assertEqual(result.provider, "p1")  # best of the two actually tried (10.0 > 5.0)

    def test_cache_hit_skips_every_provider(self):
        p1 = _FakeProvider("p1", match=ProviderMatch(website="X", url="https://x.example", similarity=99.0))

        fake_cache_client = MagicMock()
        fake_cache_client.available = True
        fake_cache_client.get.return_value = {
            "found": True, "website": "Cached", "url": "https://cached.example",
            "similarity": 99.9, "provider": "p1",
        }
        cache = ReverseSearchCache(fake_cache_client, ttl_found=300, ttl_not_found=60)

        orch = ReverseSearchOrchestrator(
            slots=[_slot(p1, priority=1, stop_threshold=98.0)],
            cache=cache, temp_host=None, config=_test_rs_config(max_providers=3),
        )
        result = orch.search(b"fake-bytes")

        self.assertTrue(result.cache_hit)
        self.assertEqual(result.website, "Cached")
        self.assertEqual(p1.calls, 0, "a cache hit must never call any provider")

    def test_deadline_cuts_off_remaining_providers(self):
        """Speed: a request-level retry/time budget stops the sequential
        chain from trying further providers once it's already spent."""
        import time as _t

        class SlowFakeProvider(_FakeProvider):
            def search(self, *, image_bytes, image_url, timeout_s, deadline=None):
                _t.sleep(0.05)
                return super().search(image_bytes=image_bytes, image_url=image_url,
                                      timeout_s=timeout_s, deadline=deadline)

        p1 = SlowFakeProvider("p1", match=ProviderMatch(website="W1", url="https://w1.example", similarity=50.0))
        p2 = _FakeProvider("p2", match=ProviderMatch(website="W2", url="https://w2.example", similarity=99.0))

        orch = ReverseSearchOrchestrator(
            slots=[_slot(p1, priority=1, stop_threshold=98.0), _slot(p2, priority=2, stop_threshold=95.0)],
            cache=_no_cache(), temp_host=None,
            config=_test_rs_config(request_deadline_s=0.03),  # smaller than p1's simulated latency
        )
        result = orch.search(b"fake-bytes")

        self.assertEqual(p1.calls, 1)
        self.assertEqual(p2.calls, 0, "the deadline was already exceeded before p2 could be tried")
        self.assertEqual(result.provider, "p1")  # best-effort result from whatever was tried

    def test_latency_mode_calls_every_provider_and_picks_the_best(self):
        """Speed: mode=latency races every provider concurrently instead of
        stopping the chain, trading quota for a wall-time bounded by the
        slowest provider instead of the sum of all of them."""
        p1 = _FakeProvider("p1", match=ProviderMatch(website="Weak", url="https://weak.example", similarity=60.0))
        p2 = _FakeProvider("p2", match=ProviderMatch(website="Strong", url="https://strong.example", similarity=99.0))

        orch = ReverseSearchOrchestrator(
            slots=[_slot(p1, priority=1, stop_threshold=98.0), _slot(p2, priority=2, stop_threshold=95.0)],
            cache=_no_cache(), temp_host=None, config=_test_rs_config(mode="latency"),
        )
        result = orch.search(b"fake-bytes")

        self.assertEqual(p1.calls, 1)
        self.assertEqual(p2.calls, 1, "latency mode must call every configured provider, not just p1")
        self.assertTrue(result.found)
        self.assertEqual(result.provider, "p2")
        self.assertEqual(result.stop_reason, "threshold_met_parallel")

    def test_per_request_mode_override_forces_latency(self):
        """A per-request mode= argument overrides the orchestrator's default,
        without needing a second orchestrator instance."""
        p1 = _FakeProvider("p1", match=ProviderMatch(website="Exact", url="https://exact.example", similarity=99.7))
        p2 = _FakeProvider("p2", match=ProviderMatch(website="Other", url="https://other.example", similarity=50.0))

        orch = ReverseSearchOrchestrator(
            slots=[_slot(p1, priority=1, stop_threshold=98.0), _slot(p2, priority=2, stop_threshold=95.0)],
            cache=_no_cache(), temp_host=None, config=_test_rs_config(mode="cost"),
        )

        # Default "cost" mode: p1 alone meets its threshold, p2 is never called.
        orch.search(b"fake-bytes-a")
        self.assertEqual(p2.calls, 0)

        # Same orchestrator, per-request override to "latency": p2 IS called
        # even though p1 alone would have stopped the chain.
        orch.search(b"fake-bytes-b", mode="latency")
        self.assertEqual(p2.calls, 1)

    def test_open_circuit_breaker_skips_the_call_entirely(self):
        p1 = _FakeProvider("p1", match=ProviderMatch(website="W", url="https://w.example", similarity=99.0))

        orch = ReverseSearchOrchestrator(
            slots=[_slot(p1, priority=1, stop_threshold=98.0, breaker=_FakeBreaker(allowed=False))],
            cache=_no_cache(), temp_host=None, config=_test_rs_config(),
        )
        result = orch.search(b"fake-bytes")

        self.assertEqual(p1.calls, 0, "provider must never be called while its circuit is open")
        self.assertFalse(result.found)

    def test_breaker_records_success_and_failure(self):
        p_ok = _FakeProvider("ok", match=ProviderMatch(website="W", url="https://w.example", similarity=10.0))
        breaker_ok = _FakeBreaker(allowed=True)

        p_fail = _FakeProvider("fail", error=ProviderAuthError("fail", 401))
        breaker_fail = _FakeBreaker(allowed=True)

        orch = ReverseSearchOrchestrator(
            slots=[
                _slot(p_ok, priority=1, stop_threshold=98.0, breaker=breaker_ok),
                _slot(p_fail, priority=2, stop_threshold=98.0, breaker=breaker_fail),
            ],
            cache=_no_cache(), temp_host=None, config=_test_rs_config(),
        )
        orch.search(b"fake-bytes")

        self.assertEqual(breaker_ok.successes, 1)
        self.assertEqual(breaker_fail.failures, 1)


# ===========================================================================
# TestSimilarityAdjustment — cross-provider corroboration + domain trust
# ===========================================================================

class TestSimilarityAdjustment(unittest.TestCase):

    def test_cross_provider_corroboration_boosts_similarity(self):
        p1 = _FakeProvider("p1", match=ProviderMatch(website="Site", url="https://same.example/img.jpg", similarity=80.0))
        p2 = _FakeProvider("p2", match=ProviderMatch(website="Site2", url="https://same.example/other.jpg", similarity=85.0))

        orch = ReverseSearchOrchestrator(
            slots=[_slot(p1, priority=1, stop_threshold=98.0), _slot(p2, priority=2, stop_threshold=98.0)],
            cache=_no_cache(), temp_host=None, config=_test_rs_config(corroboration_bonus=5.0),
        )
        result = orch.search(b"fake-bytes")

        # p2 independently reports the same hostname p1 already reported ->
        # +5 corroboration bonus on top of p2's own 85.0.
        self.assertEqual(result.similarity, 90.0)
        self.assertEqual(result.provider, "p2")

    def test_same_provider_reporting_twice_does_not_self_corroborate(self):
        # A single provider can only contribute one match per request in this
        # design, so this mostly documents the guard rather than exercising a
        # realistic path — corroboration requires a DIFFERENT provider name.
        p1 = _FakeProvider("p1", match=ProviderMatch(website="Site", url="https://same.example/x", similarity=80.0))
        orch = ReverseSearchOrchestrator(
            slots=[_slot(p1, priority=1, stop_threshold=98.0)],
            cache=_no_cache(), temp_host=None, config=_test_rs_config(corroboration_bonus=5.0),
        )
        result = orch.search(b"fake-bytes")
        self.assertEqual(result.similarity, 80.0)

    def test_trusted_domain_bonus_applied(self):
        p1 = _FakeProvider("p1", match=ProviderMatch(website="Wiki", url="https://wikipedia.org/wiki/X", similarity=90.0))

        orch = ReverseSearchOrchestrator(
            slots=[_slot(p1, priority=1, stop_threshold=98.0)],
            cache=_no_cache(), temp_host=None,
            config=_test_rs_config(trusted_domains=frozenset({"wikipedia.org"}), trusted_domain_bonus=5.0),
        )
        result = orch.search(b"fake-bytes")
        self.assertEqual(result.similarity, 95.0)

    def test_distrusted_domain_penalty_applied(self):
        p1 = _FakeProvider("p1", match=ProviderMatch(website="Spam", url="https://spamsite.example/x", similarity=90.0))

        orch = ReverseSearchOrchestrator(
            slots=[_slot(p1, priority=1, stop_threshold=98.0)],
            cache=_no_cache(), temp_host=None,
            config=_test_rs_config(distrusted_domains=frozenset({"spamsite.example"}), distrusted_domain_penalty=20.0),
        )
        result = orch.search(b"fake-bytes")
        self.assertEqual(result.similarity, 70.0)

    def test_similarity_never_exceeds_99_9_even_with_bonuses_stacked(self):
        p1 = _FakeProvider("p1", match=ProviderMatch(website="X", url="https://trusted.example/x", similarity=99.0))
        p2 = _FakeProvider("p2", match=ProviderMatch(website="X2", url="https://trusted.example/y", similarity=99.0))

        orch = ReverseSearchOrchestrator(
            slots=[_slot(p1, priority=1, stop_threshold=99.99), _slot(p2, priority=2, stop_threshold=99.99)],
            cache=_no_cache(), temp_host=None,
            config=_test_rs_config(
                corroboration_bonus=5.0,
                trusted_domains=frozenset({"trusted.example"}), trusted_domain_bonus=5.0,
            ),
        )
        result = orch.search(b"fake-bytes")
        self.assertLessEqual(result.similarity, 99.9)


# ===========================================================================
# TestReverseSearchCacheTTL
# ===========================================================================

class TestReverseSearchCacheTTL(unittest.TestCase):

    def test_found_result_uses_the_long_ttl(self):
        fake_cache_client = MagicMock()
        fake_cache_client.available = True
        cache = ReverseSearchCache(fake_cache_client, ttl_found=2_592_000, ttl_not_found=86_400)

        result = ReverseSearchResult(found=True, website="W", url="https://w", similarity=99.0, provider="p1", elapsed_ms=10)
        cache.set("digest123", result)

        args, kwargs = fake_cache_client.set.call_args
        ttl_used = args[2] if len(args) > 2 else kwargs.get("ttl")
        self.assertEqual(ttl_used, 2_592_000)

    def test_not_found_result_uses_the_short_ttl(self):
        fake_cache_client = MagicMock()
        fake_cache_client.available = True
        cache = ReverseSearchCache(fake_cache_client, ttl_found=2_592_000, ttl_not_found=86_400)

        result = ReverseSearchResult(found=False, website=None, url=None, similarity=0.0, provider=None, elapsed_ms=10)
        cache.set("digest123", result)

        args, kwargs = fake_cache_client.set.call_args
        ttl_used = args[2] if len(args) > 2 else kwargs.get("ttl")
        self.assertEqual(ttl_used, 86_400)

    def test_set_is_a_noop_when_cache_unavailable(self):
        cache = ReverseSearchCache(cache_client=None, ttl_found=300, ttl_not_found=60)
        result = ReverseSearchResult(found=True, website="W", url="https://w", similarity=99.0, provider="p1", elapsed_ms=10)
        cache.set("digest123", result)  # must not raise


# ===========================================================================
# TestRetryPolicy
# ===========================================================================

class TestRetryPolicy(unittest.TestCase):

    def setUp(self):
        # Retry backoff sleeps are real work we don't want slowing the suite down.
        patcher = patch("app.reverse_search.providers.base.time.sleep")
        self.addCleanup(patcher.stop)
        patcher.start()

    def test_retries_on_transient_error_then_succeeds(self):
        attempts = {"n": 0}

        def flaky():
            attempts["n"] += 1
            if attempts["n"] < 3:
                raise ProviderTransientError("p1", 503)
            return "ok"

        result = retry_call(flaky, max_retries=3)
        self.assertEqual(result, "ok")
        self.assertEqual(attempts["n"], 3)

    def test_gives_up_after_max_retries(self):
        def always_fails():
            raise ProviderTransientError("p1", 503)

        with self.assertRaises(ProviderTransientError):
            retry_call(always_fails, max_retries=2)

    def test_auth_error_is_never_retried(self):
        calls = {"n": 0}

        def unauthorized():
            calls["n"] += 1
            raise ProviderAuthError("p1", 401)

        with self.assertRaises(ProviderAuthError):
            retry_call(unauthorized, max_retries=5)
        self.assertEqual(calls["n"], 1, "auth errors must never be retried")

    def test_rate_limit_error_is_retried(self):
        attempts = {"n": 0}

        def rate_limited_once():
            attempts["n"] += 1
            if attempts["n"] < 2:
                raise ProviderRateLimitError("p1", retry_after=0.01)
            return "ok"

        result = retry_call(rate_limited_once, max_retries=2)
        self.assertEqual(result, "ok")
        self.assertEqual(attempts["n"], 2)

    def test_stops_retrying_once_past_the_deadline(self):
        import time as _t

        calls = {"n": 0}

        def always_fails():
            calls["n"] += 1
            raise ProviderTransientError("p1", 503)

        past_deadline = _t.perf_counter() - 1.0  # already expired
        with self.assertRaises(ProviderTransientError):
            retry_call(always_fails, max_retries=5, deadline=past_deadline)
        self.assertEqual(calls["n"], 1, "must not retry once already past the deadline")


# ===========================================================================
# TestReverseImageSearchIntegration — Flask route -> orchestrator -> real
# GoogleVisionProvider parsing logic, with only the outbound HTTP call mocked.
# ===========================================================================

def _build_test_config(max_batch_size: int = 20):
    """Shared minimal AppConfig for Flask-integration tests in this file."""
    from app.config import (
        AppConfig, QdrantConfig, RedisConfig, ModelConfig, ApiRotatorConfig,
        SecurityConfig, StorageConfig, ObservabilityConfig,
        ReverseSearchConfig, ProviderSettings,
    )
    return AppConfig(
        qdrant=QdrantConfig(host="x", port=1, collection="x", api_key=None),
        redis=RedisConfig(host="127.0.0.1", port=1, password=None, db=0, socket_timeout=0.1,
                           embedding_ttl=60, result_ttl=60, job_ttl=60),
        model=ModelConfig(siglip_model_id="x", clip_model_id="x", device="cpu",
                           max_batch_size=1, inference_timeout_s=1.0),
        api_rotator=ApiRotatorConfig(serpapi_key=None, zenserp_key=None, usage_backend="file",
                                      usage_file="/tmp/xplagiax_test_usage.json", request_timeout_s=1.0,
                                      max_retries=1, serpapi_limit=1, zenserp_limit=1,
                                      health_check_interval=9999),
        security=SecurityConfig(
            api_key=None, admin_api_key=None, require_auth=False,
            max_image_bytes=5 * 1024 * 1024, max_image_pixels=40_000_000,
            allowed_mime_types=frozenset({"jpeg", "png", "webp"}),
            rate_limit_per_minute=10_000, rate_limit_per_hour=100_000,
            trusted_proxies=(), trusted_proxy_count=0, allow_local_image_path=False,
        ),
        storage=StorageConfig(image_backend="local", local_base_path="/tmp", seaweedfs_replication="000",
                               seaweedfs_collection="", seaweedfs_ttl="", seaweedfs_request_timeout=1.0,
                               seaweedfs_max_retries=1, seaweedfs_filer_url=None, seaweedfs_public_url=None,
                               seaweedfs_master_url=None, seaweedfs_public_volume_url=None),
        observability=ObservabilityConfig(log_level="ERROR", log_format="text", prometheus_enabled=False,
                                           prometheus_port=9, otel_enabled=False, otel_endpoint=None,
                                           service_name="test", environment="test"),
        reverse_search=ReverseSearchConfig(
            enabled=True, max_providers=3, max_retries=1, max_batch_size=max_batch_size,
            cache_ttl_found=300, cache_ttl_not_found=60,
            public_base_url=None, temp_hosting_ttl=30,
            google_vision_api_key="fake-vision-key",
            google_vision=ProviderSettings(enabled=True, priority=1, stop_threshold=98.0,
                                            timeout_s=1.0, requires_public_url=False),
            serper_api_key=None,
            serper=ProviderSettings(enabled=False, priority=2, stop_threshold=95.0,
                                     timeout_s=1.0, requires_public_url=True),
            mungfali_api_key=None,
            mungfali=ProviderSettings(enabled=False, priority=3, stop_threshold=90.0,
                                       timeout_s=1.0, requires_public_url=True),
        ),
        debug=False, workers=1,
    )


def _fake_vision_response(similarity_full_match: bool = True):
    fake_response = MagicMock()
    fake_response.status_code = 200
    fake_response.json.return_value = {
        "responses": [{
            "webDetection": {
                "fullMatchingImages": [{"url": "https://example.com/exact.jpg"}] if similarity_full_match else [],
                "pagesWithMatchingImages": [{
                    "url": "https://es.wikipedia.org/wiki/Example",
                    "pageTitle": "Example - Wikipedia",
                }],
            }
        }]
    }
    return fake_response


class TestReverseImageSearchIntegration(unittest.TestCase):

    def test_end_to_end_match_via_mocked_vision_api(self):
        import io
        from PIL import Image
        from app.reverse_search.app import create_app

        buf = io.BytesIO()
        Image.new("RGB", (64, 64), color=(10, 20, 30)).save(buf, format="JPEG")
        jpeg_bytes = buf.getvalue()

        with patch("requests.Session.post", return_value=_fake_vision_response()):
            app = create_app(config=_build_test_config())
            client = app.test_client()
            resp = client.post(
                "/api/v1/reverse-image-search",
                data={"image": (io.BytesIO(jpeg_bytes), "photo.jpg")},
                content_type="multipart/form-data",
            )

        self.assertEqual(resp.status_code, 200)
        body = resp.get_json()
        self.assertEqual(set(body.keys()), {"found", "website", "url", "similarity", "provider", "elapsed_ms"})
        self.assertTrue(body["found"])
        self.assertEqual(body["provider"], "google_vision")
        self.assertEqual(body["website"], "Example - Wikipedia")
        self.assertEqual(body["url"], "https://es.wikipedia.org/wiki/Example")
        self.assertEqual(body["similarity"], 97.0)  # base score for 1 full match, no extra corroboration

    def test_missing_image_returns_400(self):
        from app.reverse_search.app import create_app
        app = create_app(config=_build_test_config())
        client = app.test_client()
        resp = client.post("/api/v1/reverse-image-search", data={})
        self.assertEqual(resp.status_code, 400)
        self.assertEqual(resp.get_json()["code"], "MISSING_FILE")


# ===========================================================================
# TestReverseImageSearchBatch — POST /api/v1/reverse-image-search/batch
# ===========================================================================

class TestReverseImageSearchBatch(unittest.TestCase):

    @staticmethod
    def _jpeg_bytes(color=(10, 20, 30)):
        import io
        from PIL import Image
        buf = io.BytesIO()
        Image.new("RGB", (64, 64), color=color).save(buf, format="JPEG")
        return buf.getvalue()

    def test_batch_processes_each_image_independently(self):
        import io
        from app.reverse_search.app import create_app

        with patch("requests.Session.post", return_value=_fake_vision_response()):
            app = create_app(config=_build_test_config())
            client = app.test_client()
            resp = client.post(
                "/api/v1/reverse-image-search/batch",
                data={
                    "files": [
                        (io.BytesIO(self._jpeg_bytes()), "a.jpg"),
                        (io.BytesIO(self._jpeg_bytes()), "b.jpg"),
                    ],
                },
                content_type="multipart/form-data",
            )

        self.assertEqual(resp.status_code, 207)
        body = resp.get_json()
        self.assertEqual(body["count"], 2)
        sources = {r["source"] for r in body["results"]}
        self.assertEqual(sources, {"a.jpg", "b.jpg"})
        for r in body["results"]:
            self.assertTrue(r["found"])
            self.assertEqual(r["provider"], "google_vision")

    def test_batch_reports_per_item_validation_errors(self):
        import io
        from app.reverse_search.app import create_app

        with patch("requests.Session.post", return_value=_fake_vision_response()):
            app = create_app(config=_build_test_config())
            client = app.test_client()
            resp = client.post(
                "/api/v1/reverse-image-search/batch",
                data={
                    "files": [
                        (io.BytesIO(self._jpeg_bytes()), "good.jpg"),
                        (io.BytesIO(b"not an image"), "bad.jpg"),
                    ],
                },
                content_type="multipart/form-data",
            )

        self.assertEqual(resp.status_code, 207)
        by_source = {r["source"]: r for r in resp.get_json()["results"]}
        self.assertTrue(by_source["good.jpg"]["found"])
        self.assertEqual(by_source["bad.jpg"]["code"], "INVALID_IMAGE")

    def test_batch_over_the_cap_is_rejected(self):
        import io
        from app.reverse_search.app import create_app

        app = create_app(config=_build_test_config(max_batch_size=2))
        client = app.test_client()
        resp = client.post(
            "/api/v1/reverse-image-search/batch",
            data={
                "files": [
                    (io.BytesIO(self._jpeg_bytes()), "a.jpg"),
                    (io.BytesIO(self._jpeg_bytes()), "b.jpg"),
                    (io.BytesIO(self._jpeg_bytes()), "c.jpg"),
                ],
            },
            content_type="multipart/form-data",
        )

        self.assertEqual(resp.status_code, 400)
        self.assertEqual(resp.get_json()["code"], "BATCH_TOO_LARGE")

    def test_empty_batch_returns_400(self):
        from app.reverse_search.app import create_app
        app = create_app(config=_build_test_config())
        client = app.test_client()
        resp = client.post("/api/v1/reverse-image-search/batch", data={})
        self.assertEqual(resp.status_code, 400)
        self.assertEqual(resp.get_json()["code"], "MISSING_FILES")

    def test_batch_respects_concurrency_cap_but_processes_everything(self):
        """Speed/safety: a concurrency cap smaller than the batch must still
        process every item correctly (just queued through the pool), not
        silently drop any of them."""
        import dataclasses
        import io
        from app.reverse_search.app import create_app

        config = _build_test_config()
        config = dataclasses.replace(
            config, reverse_search=dataclasses.replace(config.reverse_search, batch_concurrency=1),
        )

        with patch("requests.Session.post", return_value=_fake_vision_response()):
            app = create_app(config=config)
            client = app.test_client()
            resp = client.post(
                "/api/v1/reverse-image-search/batch",
                data={"files": [
                    (io.BytesIO(self._jpeg_bytes()), "a.jpg"),
                    (io.BytesIO(self._jpeg_bytes()), "b.jpg"),
                    (io.BytesIO(self._jpeg_bytes()), "c.jpg"),
                ]},
                content_type="multipart/form-data",
            )

        self.assertEqual(resp.status_code, 207)
        body = resp.get_json()
        self.assertEqual(body["count"], 3)
        sources = {r["source"] for r in body["results"]}
        self.assertEqual(sources, {"a.jpg", "b.jpg", "c.jpg"})


# ===========================================================================
# TestTempHostImageStorageAdapter — bridges TempImageHost to the
# save/get_url/delete shape patents_bp expects, standalone-only.
# ===========================================================================

class TestTempHostImageStorageAdapter(unittest.TestCase):

    def test_save_then_get_url_then_delete(self):
        from app.reverse_search.app import _TempHostImageStorage
        from app.reverse_search.temp_hosting import TempImageHost

        fake_cache = MagicMock()
        fake_cache.available = True
        fake_cache.set_bytes.return_value = True
        fake_cache.set.return_value = True

        temp_host = TempImageHost(fake_cache, public_base_url="https://example.com", ttl_s=60)
        adapter = _TempHostImageStorage(temp_host)

        key = adapter.save(b"raw-bytes", "abc123hash", "temp_search", "photo.jpg", "jpeg")
        self.assertEqual(key, "abc123hash")

        url = adapter.get_url(key, expiry_seconds=3600)
        self.assertTrue(url.startswith("https://example.com/api/v1/_tmp-image/"))

        adapter.delete(key)
        self.assertEqual(adapter.get_url(key), "")


# ===========================================================================
# TestStandaloneAppPatents — the standalone entrypoint also serves
# /api/v1/patents/* (api_rotator.py has zero heavy-ML dependencies), but
# never the CLIP/SigLIP/Qdrant-backed routes.
# ===========================================================================

class TestStandaloneAppPatents(unittest.TestCase):

    def test_patent_text_search_works_standalone_with_mocked_serpapi(self):
        import dataclasses
        import os
        import tempfile
        from app.reverse_search.app import create_app

        # Dedicated, fresh usage file: the shared one from _build_test_config()
        # persists real usage counts on disk (usage_backend="file"), and this
        # is the only test that actually increments serpapi's counter. Reusing
        # the shared path would make this test pass once and then fail on
        # every subsequent run (used >= serpapi_limit=1) regardless of test
        # ordering — a stale on-disk counter, not a real app bug.
        fd, usage_file = tempfile.mkstemp(prefix="xplagiax_test_usage_", suffix=".json")
        os.close(fd)
        os.remove(usage_file)
        self.addCleanup(lambda: os.path.exists(usage_file) and os.remove(usage_file))

        config = _build_test_config()
        config = dataclasses.replace(
            config, api_rotator=dataclasses.replace(
                config.api_rotator, serpapi_key="fake-serpapi-key", usage_file=usage_file,
            ),
        )

        fake_response = MagicMock()
        fake_response.status_code = 200
        fake_response.json.return_value = {
            "organic_results": [{"title": "Some Patent", "patent_id": "US123"}],
        }

        with patch("requests.Session.get", return_value=fake_response):
            app = create_app(config=config)
            client = app.test_client()
            resp = client.post("/api/v1/patents/search/text", json={"query": "solar panel coating"})

        self.assertEqual(resp.status_code, 200)
        body = resp.get_json()
        self.assertEqual(body["status"], "success")

    def test_patents_unavailable_without_keys_returns_503(self):
        from app.reverse_search.app import create_app
        app = create_app(config=_build_test_config())
        client = app.test_client()
        resp = client.post("/api/v1/patents/search/text", json={"query": "x"})
        self.assertEqual(resp.status_code, 503)
        self.assertEqual(resp.get_json()["code"], "SERVICE_UNAVAILABLE")

    def test_standalone_does_not_expose_clip_backed_routes(self):
        from app.reverse_search.app import create_app
        app = create_app(config=_build_test_config())
        client = app.test_client()
        self.assertEqual(client.post("/api/v1/images").status_code, 404)
        self.assertEqual(client.post("/api/v1/search/similar").status_code, 404)


if __name__ == "__main__":
    unittest.main()
