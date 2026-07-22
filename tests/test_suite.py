"""
Test suite for xplagiax image service.

Structure:
  TestImageValidation       — unit tests for magic byte detection, size limits, sanitisation
  TestVectorRepository      — unit tests with Qdrant mock
  TestCacheClient           — unit tests for Redis graceful degradation
  TestModelRegistry         — unit tests for model error handling
  TestSimilarityService     — unit tests for scoring / classification logic
  TestApiRoutes             — integration tests against Flask test client
  TestApiRotatorScoring     — unit tests for dynamic scoring logic

Run:
    pytest tests/ -v --tb=short
    pytest tests/ -v -k "TestImageValidation"
"""

from __future__ import annotations

import hashlib
import io
import json
import unittest
from unittest.mock import MagicMock, patch, PropertyMock
from dataclasses import dataclass
from typing import Optional

from PIL import Image


# ===========================================================================
# Helpers
# ===========================================================================

def _make_jpeg_bytes(width: int = 100, height: int = 100) -> bytes:
    """Create minimal valid JPEG bytes."""
    buf = io.BytesIO()
    img = Image.new("RGB", (width, height), color=(128, 64, 200))
    img.save(buf, format="JPEG")
    return buf.getvalue()


def _make_png_bytes(width: int = 100, height: int = 100) -> bytes:
    buf = io.BytesIO()
    img = Image.new("RGB", (width, height), color=(0, 128, 255))
    img.save(buf, format="PNG")
    return buf.getvalue()


# ===========================================================================
# TestImageValidation
# ===========================================================================

class TestImageValidation(unittest.TestCase):

    def setUp(self):
        from app.utils.image_validation import validate_and_load, sanitise_filename, sanitise_group_id
        self.validate = validate_and_load
        self.sanitise_filename = sanitise_filename
        self.sanitise_group_id = sanitise_group_id

    def test_valid_jpeg_accepted(self):
        data = _make_jpeg_bytes()
        img, mime = self.validate(data)
        self.assertEqual(mime, "jpeg")
        self.assertIsInstance(img, Image.Image)

    def test_valid_png_accepted(self):
        data = _make_png_bytes()
        img, mime = self.validate(data)
        self.assertEqual(mime, "png")

    def test_empty_bytes_rejected(self):
        from app.utils.image_validation import ImageValidationError
        with self.assertRaises(ImageValidationError) as ctx:
            self.validate(b"")
        self.assertIn("Empty", str(ctx.exception))

    def test_oversized_rejected(self):
        from app.utils.image_validation import ImageValidationError
        data = _make_jpeg_bytes()
        with self.assertRaises(ImageValidationError) as ctx:
            self.validate(data, max_bytes=10)
        self.assertIn("too large", str(ctx.exception))

    def test_non_image_bytes_rejected(self):
        from app.utils.image_validation import ImageValidationError
        with self.assertRaises(ImageValidationError):
            self.validate(b"This is not an image at all %PDF-1.4")

    def test_disallowed_mime_rejected(self):
        from app.utils.image_validation import ImageValidationError
        data = _make_jpeg_bytes()
        with self.assertRaises(ImageValidationError) as ctx:
            self.validate(data, allowed_mimes=frozenset({"png"}))
        self.assertIn("not allowed", str(ctx.exception))

    def test_too_small_image_rejected(self):
        from app.utils.image_validation import ImageValidationError
        buf = io.BytesIO()
        Image.new("RGB", (10, 10)).save(buf, format="JPEG")
        with self.assertRaises(ImageValidationError) as ctx:
            self.validate(buf.getvalue())
        self.assertIn("too small", str(ctx.exception))

    def test_corrupt_bytes_rejected(self):
        from app.utils.image_validation import ImageValidationError
        # Valid JPEG magic but truncated
        corrupt = b"\xff\xd8\xff" + b"\x00" * 50
        with self.assertRaises(ImageValidationError):
            self.validate(corrupt)

    # -- sanitise_filename --

    def test_sanitise_filename_strips_directory(self):
        result = self.sanitise_filename("../../etc/passwd")
        self.assertEqual(result, "passwd")

    def test_sanitise_filename_normal(self):
        result = self.sanitise_filename("my_image.jpg")
        self.assertEqual(result, "my_image.jpg")

    def test_sanitise_filename_empty(self):
        result = self.sanitise_filename("")
        self.assertEqual(result, "unknown")

    def test_sanitise_filename_null_bytes(self):
        result = self.sanitise_filename("file\x00name.jpg")
        self.assertEqual(result, "filename.jpg")

    # -- sanitise_group_id --

    def test_sanitise_group_id_allowlist(self):
        result = self.sanitise_group_id("project-123_abc")
        self.assertEqual(result, "project-123_abc")

    def test_sanitise_group_id_strips_special_chars(self):
        result = self.sanitise_group_id("../../evil; rm -rf /")
        # Only alphanumeric, hyphens, underscores survive
        self.assertNotIn("/", result)
        self.assertNotIn(".", result)
        self.assertNotIn(";", result)

    def test_sanitise_group_id_empty(self):
        result = self.sanitise_group_id("")
        self.assertEqual(result, "default")


# ===========================================================================
# TestCacheClient
# ===========================================================================

class TestCacheClient(unittest.TestCase):

    def _make_unavailable_cache(self):
        """Cache that simulates Redis being down."""
        from app.cache.redis_client import CacheClient
        from redis.exceptions import RedisError
        with patch("redis.Redis.ping", side_effect=RedisError("connection refused")):
            with patch("app.cache.redis_client.get_metrics") as mock_metrics:
                mock_metrics.return_value = MagicMock()
                cache = CacheClient(
                    host="localhost", port=6379, password=None, db=0,
                    socket_timeout=0.1, embedding_ttl=86400,
                    result_ttl=300, job_ttl=3600,
                )
        return cache

    def test_get_returns_none_when_unavailable(self):
        cache = self._make_unavailable_cache()
        result = cache.get("any_key")
        self.assertIsNone(result)

    def test_set_returns_false_when_unavailable(self):
        cache = self._make_unavailable_cache()
        result = cache.set("any_key", {"data": "value"}, 300)
        self.assertFalse(result)

    def test_get_embedding_returns_none_when_unavailable(self):
        cache = self._make_unavailable_cache()
        result = cache.get_embedding(b"fake image bytes")
        self.assertIsNone(result)

    def test_embedding_key_is_deterministic(self):
        from app.cache.redis_client import CacheClient
        data = b"some image bytes"
        key1 = CacheClient.embedding_key(data)
        key2 = CacheClient.embedding_key(data)
        self.assertEqual(key1, key2)

    def test_embedding_key_differs_for_different_data(self):
        from app.cache.redis_client import CacheClient
        key1 = CacheClient.embedding_key(b"image_a")
        key2 = CacheClient.embedding_key(b"image_b")
        self.assertNotEqual(key1, key2)

    def test_health_check_unavailable(self):
        cache = self._make_unavailable_cache()
        result = cache.health_check()
        self.assertEqual(result["status"], "unavailable")


# ===========================================================================
# TestVectorRepository — deterministic ID + idempotency
# ===========================================================================

class TestVectorRepository(unittest.TestCase):

    def test_deterministic_id_is_stable(self):
        from app.storage.vector_repository import VectorRepository
        hash_val = "abc123" * 10
        id1 = VectorRepository.deterministic_id(hash_val)
        id2 = VectorRepository.deterministic_id(hash_val)
        self.assertEqual(id1, id2)

    def test_different_hashes_produce_different_ids(self):
        from app.storage.vector_repository import VectorRepository
        id1 = VectorRepository.deterministic_id("hash_a" * 10)
        id2 = VectorRepository.deterministic_id("hash_b" * 10)
        self.assertNotEqual(id1, id2)

    def test_id_is_valid_uuid_format(self):
        from app.storage.vector_repository import VectorRepository
        import uuid
        raw_id = VectorRepository.deterministic_id("test_content_hash_value_here")
        # Should not raise
        parsed = uuid.UUID(raw_id)
        self.assertIsNotNone(parsed)


# ===========================================================================
# TestSimilarityService — scoring logic
# ===========================================================================

class TestSimilarityService(unittest.TestCase):

    def test_score_above_097_is_exact_copy(self):
        from app.services.similarity import classify_similarity
        match_type, _ = classify_similarity(0.985)
        self.assertEqual(match_type, "EXACT_COPY")

    def test_score_between_090_097_is_modified_copy(self):
        from app.services.similarity import classify_similarity
        match_type, _ = classify_similarity(0.93)
        self.assertEqual(match_type, "MODIFIED_COPY")

    def test_score_between_080_090_is_high_similarity(self):
        from app.services.similarity import classify_similarity
        match_type, _ = classify_similarity(0.85)
        self.assertEqual(match_type, "HIGH_SIMILARITY")

    def test_score_below_080_is_moderate(self):
        from app.services.similarity import classify_similarity
        match_type, _ = classify_similarity(0.65)
        self.assertEqual(match_type, "MODERATE_SIMILARITY")

    def test_exact_boundary_097(self):
        from app.services.similarity import classify_similarity
        match_type, _ = classify_similarity(0.97)
        self.assertEqual(match_type, "EXACT_COPY")

    def test_exact_boundary_090(self):
        from app.services.similarity import classify_similarity
        match_type, _ = classify_similarity(0.90)
        self.assertEqual(match_type, "MODIFIED_COPY")


# ===========================================================================
# TestApiRotatorScoring
# ===========================================================================

class TestApiRotatorScoring(unittest.TestCase):

    def _make_score(self, name="serpapi"):
        from app.services.api_rotator import ProviderScore
        return ProviderScore(name=name)

    def test_initial_score_is_perfect(self):
        score = self._make_score()
        self.assertEqual(score.score, 1.0)

    def test_errors_degrade_score(self):
        score = self._make_score()
        for _ in range(5):
            score.record(success=False, response_ms=5000)
        self.assertLess(score.score, 0.5)

    def test_successes_maintain_score(self):
        score = self._make_score()
        for _ in range(10):
            score.record(success=True, response_ms=200)
        self.assertGreater(score.score, 0.8)

    def test_penalty_marks_as_penalised(self):
        score = self._make_score()
        self.assertFalse(score.is_penalised)
        score.penalise(3600)
        self.assertTrue(score.is_penalised)

    def test_penalty_expiry_auto_recovers(self):
        import datetime
        score = self._make_score()
        # Set penalty to past
        score.penalty_until = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(seconds=1)
        self.assertFalse(score.is_penalised)  # auto-clears

    def test_mixed_success_failure_scores_reasonably(self):
        score = self._make_score()
        for _ in range(7):
            score.record(success=True, response_ms=300)
        for _ in range(3):
            score.record(success=False, response_ms=3000)
        # 70% success, decent latency → score should be moderate
        self.assertGreater(score.score, 0.4)
        self.assertLess(score.score, 0.9)


# ===========================================================================
# TestApiRoutes — Flask integration tests (no real ML, mocked deps)
# ===========================================================================

class TestApiRoutes(unittest.TestCase):
    """
    Integration tests using Flask test client.
    All ML models, Qdrant, and Redis are mocked.
    """

    def setUp(self):
        """Build a minimal Flask app with all dependencies mocked."""
        # We import here to avoid module-level side effects in other tests
        from app.factory import create_app
        from app.config import (
            AppConfig, QdrantConfig, RedisConfig, ModelConfig,
            ApiRotatorConfig, SecurityConfig, StorageConfig, ObservabilityConfig,
            ReverseSearchConfig, ProviderSettings,
        )

        import dataclasses
        # Build minimal test config
        test_config = AppConfig(
            qdrant=QdrantConfig(
                host="localhost", port=6333, collection="test",
                api_key=None, hnsw_m=16, hnsw_ef_construct=100,
                hnsw_ef_search=64, full_scan_threshold=1000,
            ),
            redis=RedisConfig(
                host="localhost", port=6379, password=None, db=15,
                socket_timeout=0.1, embedding_ttl=300, result_ttl=60, job_ttl=300,
            ),
            model=ModelConfig(
                siglip_model_id="test/model", clip_model_id="test/clip",
                device="cpu", max_batch_size=1, inference_timeout_s=5.0,
            ),
            api_rotator=ApiRotatorConfig(
                serpapi_key=None, zenserp_key=None, usage_backend="file",
                usage_file="/tmp/test_usage.json", request_timeout_s=5.0,
                max_retries=1, serpapi_limit=250, zenserp_limit=50,
                health_check_interval=9999,
            ),
            security=SecurityConfig(
                api_key="test-api-key-1234",
                admin_api_key="test-admin-key-1234",
                require_auth=True,
                max_image_bytes=5 * 1024 * 1024,
                max_image_pixels=40_000_000,
                allowed_mime_types=frozenset({"jpeg", "png", "webp"}),
                rate_limit_per_minute=1000,
                rate_limit_per_hour=10000,
                trusted_proxies=("127.0.0.1",),
                trusted_proxy_count=1,
                allow_local_image_path=False,
            ),
            storage=StorageConfig(
                image_backend="local", local_base_path="/tmp/test_images",
                seaweedfs_replication="000", seaweedfs_collection="", seaweedfs_ttl="",
                seaweedfs_request_timeout=5.0, seaweedfs_max_retries=3,
                seaweedfs_filer_url=None, seaweedfs_public_url=None,
                seaweedfs_master_url=None, seaweedfs_public_volume_url=None
            ),
            observability=ObservabilityConfig(
                log_level="ERROR", log_format="text",
                prometheus_enabled=False, prometheus_port=9999,
                otel_enabled=False, otel_endpoint=None,
                service_name="test-service", environment="test",
            ),
            reverse_search=ReverseSearchConfig(
                enabled=False,
                max_providers=3, max_retries=1, max_batch_size=20,
                cache_ttl_found=300, cache_ttl_not_found=60,
                public_base_url=None, temp_hosting_ttl=30,
                google_vision_api_key=None,
                google_vision=ProviderSettings(
                    enabled=False, priority=1, stop_threshold=98.0,
                    timeout_s=1.0, requires_public_url=False,
                ),
                serper_api_key=None,
                serper=ProviderSettings(
                    enabled=False, priority=2, stop_threshold=95.0,
                    timeout_s=1.0, requires_public_url=True,
                ),
                mungfali_api_key=None,
                mungfali=ProviderSettings(
                    enabled=False, priority=3, stop_threshold=90.0,
                    timeout_s=1.0, requires_public_url=True,
                ),
            ),
            debug=False,
            workers=1,
        )

        # Patch all external deps before create_app
        patches = [
            patch("app.storage.vector_repository.VectorRepository.__init__", return_value=None),
            patch("app.storage.vector_repository.VectorRepository._ensure_collection"),
            patch("app.storage.vector_repository.VectorRepository._ensure_payload_indexes"),
            patch("app.storage.vector_repository.VectorRepository.health_check",
                  return_value={"status": "ok"}),
            patch("app.storage.vector_repository.VectorRepository.stats",
                  return_value=dataclasses.make_dataclass('Stats', [('total_vectors', int), ('indexed_vectors_count', int), ('collection_name', str), ('status', str)])(0, 0, "test", "green")),
            patch("app.cache.redis_client.CacheClient.__init__", return_value=None),
            patch("app.cache.redis_client.CacheClient.available", new_callable=PropertyMock,
                  return_value=False),
            patch("app.cache.redis_client.CacheClient.health_check",
                  return_value={"status": "unavailable"}),
            patch("app.cache.redis_client.CacheClient.get_embedding", return_value=None),
            patch("app.cache.redis_client.CacheClient.set_embedding", return_value=False),
            patch("app.cache.redis_client.CacheClient.get_job", return_value=None),
            patch("app.models.registry.ModelRegistry.load_all"),
            patch("app.models.registry.ModelRegistry.clip_ready", new_callable=PropertyMock,
                  return_value=True),
            patch("app.models.registry.ModelRegistry.siglip_ready", new_callable=PropertyMock,
                  return_value=True),
            patch("app.models.registry.ModelRegistry.get_status", return_value={
                "clip":   {"loaded": True, "device": "cpu", "model_id": "test", "error": None, "load_time_s": 1.0},
                "siglip": {"loaded": True, "device": "cpu", "model_id": "test", "error": None, "load_time_s": 2.0, "labels": ["human", "ai"]},
                "device": "cpu",
            }),
            patch("app.observability.telemetry.get_metrics", return_value=MagicMock()),
            patch("app.observability.telemetry.init_metrics", return_value=MagicMock()),
        ]

        for p in patches:
            p.start()
            self.addCleanup(p.stop)

        self.app = create_app(config=test_config)
        self.client = self.app.test_client()
        self.headers = {"X-API-Key": "test-api-key-1234"}

    # -- Health probes --

    def test_liveness_always_200(self):
        resp = self.client.get("/healthz")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertEqual(data["status"], "alive")

    def test_readiness_returns_json(self):
        resp = self.client.get("/readyz")
        data = resp.get_json()
        self.assertIn("status", data)
        self.assertIn("checks", data)

    def test_health_detailed_authenticated(self):
        resp = self.client.get("/health", headers=self.headers)
        self.assertIn(resp.status_code, (200, 503))

    # -- Authentication --

    def test_missing_api_key_returns_401(self):
        resp = self.client.post("/api/v1/images")
        self.assertEqual(resp.status_code, 401)

    def test_wrong_api_key_returns_401(self):
        resp = self.client.post(
            "/api/v1/images",
            headers={"X-API-Key": "wrong-key"},
        )
        self.assertEqual(resp.status_code, 401)

    def test_bearer_token_accepted(self):
        """X-API-Key can also be sent as Authorization: Bearer."""
        resp = self.client.get(
            "/api/v1/admin/models",
            headers={"Authorization": "Bearer test-api-key-1234"},
        )
        # Will succeed auth, may fail on deps — but not 401
        self.assertNotEqual(resp.status_code, 401)

    # -- Upload validation --

    def test_upload_no_file_returns_400(self):
        resp = self.client.post("/api/v1/images", headers=self.headers)
        self.assertEqual(resp.status_code, 400)
        data = resp.get_json()
        self.assertIn("code", data)

    def test_upload_invalid_image_returns_400(self):
        resp = self.client.post(
            "/api/v1/images",
            data={"file": (io.BytesIO(b"not an image"), "test.jpg")},
            content_type="multipart/form-data",
            headers=self.headers,
        )
        self.assertEqual(resp.status_code, 400)
        data = resp.get_json()
        self.assertEqual(data["code"], "INVALID_IMAGE")

    def test_search_no_file_returns_400(self):
        resp = self.client.post("/api/v1/search/similar", headers=self.headers)
        self.assertEqual(resp.status_code, 400)

    def test_plagiarism_no_file_returns_400(self):
        resp = self.client.post("/api/v1/search/plagiarism", headers=self.headers)
        self.assertEqual(resp.status_code, 400)

    # -- Admin: reset without confirmation --

    def test_reset_without_confirmation_returns_400(self):
        resp = self.client.delete(
            "/api/v1/admin/collection/reset",
            json={},
            headers={"X-Admin-Key": "test-admin-key-1234"},
        )
        self.assertEqual(resp.status_code, 400)
        data = resp.get_json()
        self.assertEqual(data["code"], "CONFIRMATION_REQUIRED")

    def test_reset_with_service_key_only_is_forbidden(self):
        """Destructive admin op must reject the plain service key."""
        resp = self.client.delete(
            "/api/v1/admin/collection/reset",
            json={"confirm": "I_UNDERSTAND_THIS_WILL_DELETE_ALL_DATA"},
            headers=self.headers,  # service key, not admin key
        )
        self.assertEqual(resp.status_code, 403)

    # -- Error format --

    def test_404_returns_json(self):
        resp = self.client.get("/nonexistent/endpoint", headers=self.headers)
        self.assertEqual(resp.status_code, 404)
        data = resp.get_json()
        self.assertIn("error", data)
        self.assertIn("code", data)

    def test_405_returns_json(self):
        resp = self.client.get(
            "/api/v1/images",   # POST-only endpoint
            headers=self.headers,
        )
        self.assertEqual(resp.status_code, 405)
        data = resp.get_json()
        self.assertIn("error", data)

    # -- Request ID propagation --

    def test_response_has_request_id_header(self):
        resp = self.client.get("/healthz")
        # X-Request-ID should be present
        # (may be empty string if g.request_id not set — still present)
        self.assertIn("X-Request-ID", resp.headers)

    def test_custom_request_id_propagated(self):
        custom_id = "my-trace-id-abc123"
        resp = self.client.get(
            "/healthz",
            headers={"X-Request-ID": custom_id},
        )
        self.assertEqual(resp.headers.get("X-Request-ID"), custom_id)


# ===========================================================================
# TestLocalImageStorage
# ===========================================================================

class TestLocalImageStorage(unittest.TestCase):

    def setUp(self):
        import tempfile
        self.tmpdir = tempfile.mkdtemp()
        from app.storage.image_storage import LocalImageStorage
        self.storage = LocalImageStorage(self.tmpdir)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_save_and_load_roundtrip(self):
        data = _make_jpeg_bytes()
        content_hash = hashlib.sha256(data).hexdigest()
        key = self.storage.save(data, content_hash, "grp1", "test.jpg", "jpeg")
        loaded = self.storage.load(key)
        self.assertEqual(data, loaded)

    def test_save_is_idempotent(self):
        data = _make_png_bytes()
        content_hash = hashlib.sha256(data).hexdigest()
        key1 = self.storage.save(data, content_hash, "grp1", "img.png", "png")
        key2 = self.storage.save(data, content_hash, "grp1", "img.png", "png")
        self.assertEqual(key1, key2)

    def test_load_missing_raises(self):
        with self.assertRaises(FileNotFoundError):
            self.storage.load("nonexistent/00/fakehash.jpg")

    def test_exists_true_after_save(self):
        data = _make_jpeg_bytes()
        content_hash = hashlib.sha256(data).hexdigest()
        key = self.storage.save(data, content_hash, "grp2", "img.jpg", "jpeg")
        self.assertTrue(self.storage.exists(key))

    def test_exists_false_before_save(self):
        self.assertFalse(self.storage.exists("grp/00/doesnotexist.jpg"))

    def test_delete_removes_file(self):
        data = _make_jpeg_bytes()
        content_hash = hashlib.sha256(data).hexdigest()
        key = self.storage.save(data, content_hash, "grp3", "img.jpg", "jpeg")
        self.storage.delete(key)
        self.assertFalse(self.storage.exists(key))

    def test_url_is_api_path(self):
        key = "grp1/ab/abcdef.jpg"
        url = self.storage.get_url(key)
        self.assertTrue(url.startswith("/api/v1/images/"))

    def test_path_traversal_cannot_escape_base(self):
        """Saving with a content_hash cannot escape the storage directory."""
        data = _make_jpeg_bytes()
        # Even if group_id had path components (should be sanitised by caller)
        # the storage key stays within base_path
        content_hash = hashlib.sha256(data).hexdigest()
        key = self.storage.save(data, content_hash, "safe_group", "img.jpg", "jpeg")
        import os
        full_path = os.path.join(self.tmpdir, key)
        self.assertTrue(os.path.abspath(full_path).startswith(os.path.abspath(self.tmpdir)))


# ===========================================================================
# TestLabelSemantics — robust AI/human label resolution
# ===========================================================================

class TestLabelSemantics(unittest.TestCase):

    def test_ai_hum_labels(self):
        from app.models.labels import resolve_label_semantics
        m = resolve_label_semantics({0: "ai", 1: "hum"})
        self.assertEqual(m, {0: "ai", 1: "human"})

    def test_human_artificial_labels(self):
        from app.models.labels import resolve_label_semantics
        m = resolve_label_semantics({0: "human", 1: "artificial"})
        self.assertEqual(m[0], "human")
        self.assertEqual(m[1], "ai")

    def test_real_fake_labels(self):
        from app.models.labels import resolve_label_semantics
        m = resolve_label_semantics({0: "REAL", 1: "FAKE"})
        self.assertEqual(m[0], "human")
        self.assertEqual(m[1], "ai")

    def test_unknown_labels_raise(self):
        from app.models.labels import resolve_label_semantics
        with self.assertRaises(ValueError):
            resolve_label_semantics({0: "cat", 1: "dog"})

    def test_must_cover_both_classes(self):
        from app.models.labels import resolve_label_semantics
        with self.assertRaises(ValueError):
            resolve_label_semantics({0: "ai", 1: "generated"})

    def test_override_by_index_for_opaque_labels(self):
        from app.models.labels import resolve_label_semantics
        m = resolve_label_semantics({0: "LABEL_0", 1: "LABEL_1"},
                                    override={"0": "human", "1": "ai"})
        self.assertEqual(m, {0: "human", 1: "ai"})

    def test_override_by_label_string(self):
        from app.models.labels import resolve_label_semantics
        m = resolve_label_semantics({0: "class_a", 1: "class_b"},
                                    override={"class_a": "ai", "class_b": "human"})
        self.assertEqual(m[0], "ai")
        self.assertEqual(m[1], "human")

    def test_override_invalid_value_raises(self):
        from app.models.labels import resolve_label_semantics
        with self.assertRaises(ValueError):
            resolve_label_semantics({0: "x", 1: "y"}, override={"0": "robot", "1": "human"})


# ===========================================================================
# TestReverseNormalization — provider-agnostic reverse-image verdict
# ===========================================================================

class TestReverseNormalization(unittest.TestCase):

    def test_serpapi_google_lens_matches(self):
        from app.services.api_rotator import normalize_reverse_results
        raw = {"visual_matches": [
            {"title": "A", "link": "http://a", "source": "a.com", "thumbnail": "t"},
            {"title": "B", "link": "http://b", "source": "b.com", "thumbnail": "t"},
            {"title": "C", "link": "http://c", "source": "c.com", "thumbnail": "t"},
        ]}
        out = normalize_reverse_results("serpapi", "google_lens", raw, "http://img", 10)
        self.assertTrue(out["found_on_internet"])
        self.assertEqual(out["match_count"], 3)
        self.assertEqual(out["verdict"], "LIKELY_PRESENT_ONLINE")
        self.assertEqual(out["matches"][0]["link"], "http://a")

    def test_no_matches_not_found(self):
        from app.services.api_rotator import normalize_reverse_results
        out = normalize_reverse_results("serpapi", "google_lens", {}, "http://img", 10)
        self.assertFalse(out["found_on_internet"])
        self.assertEqual(out["verdict"], "NOT_FOUND")

    def test_num_results_truncation(self):
        from app.services.api_rotator import normalize_reverse_results
        raw = {"image_results": [{"title": str(i), "link": f"http://{i}"} for i in range(20)]}
        out = normalize_reverse_results("serpapi", "google_reverse_image", raw, "u", 5)
        self.assertEqual(out["match_count"], 5)


# ===========================================================================
# TestSSRFGuards — IP safety classification
# ===========================================================================

class TestSSRFGuards(unittest.TestCase):

    def test_public_ip_is_safe(self):
        from app.security.http_client import is_safe_ip
        self.assertTrue(is_safe_ip("8.8.8.8"))

    def test_private_and_metadata_blocked(self):
        from app.security.http_client import is_safe_ip
        for ip in ("127.0.0.1", "10.0.0.5", "192.168.1.1", "169.254.169.254", "0.0.0.0", "::1"):
            self.assertFalse(is_safe_ip(ip), ip)

    def test_garbage_is_unsafe(self):
        from app.security.http_client import is_safe_ip
        self.assertFalse(is_safe_ip("not-an-ip"))


# ===========================================================================
# TestResultCaches — float32 embeddings + reverse/AI result caches
# ===========================================================================

class _FakeRedis:
    """Minimal in-memory stand-in for redis.Redis (bytes/str values)."""
    def __init__(self):
        self.store = {}
    def ping(self):
        return True
    def setex(self, key, ttl, value):
        self.store[key] = value
    def get(self, key):
        return self.store.get(key)


class TestResultCaches(unittest.TestCase):

    def setUp(self):
        p = patch("app.cache.redis_client.get_metrics", return_value=MagicMock())
        p.start(); self.addCleanup(p.stop)

    def _cache(self):
        from app.cache.redis_client import CacheClient
        fake = _FakeRedis()
        with patch("redis.ConnectionPool"), patch("redis.Redis", return_value=fake):
            return CacheClient(
                host="x", port=1, password=None, db=0, socket_timeout=0.1,
                embedding_ttl=10, result_ttl=10, job_ttl=10,
            )

    def test_embedding_float32_roundtrip(self):
        c = self._cache()
        vec = [0.1, -0.5, 0.3333333, 1.0, 0.0]
        self.assertTrue(c.set_embedding(digest="abc", vector=vec))
        out = c.get_embedding(digest="abc")
        self.assertEqual(len(out), len(vec))
        for a, b in zip(out, vec):
            self.assertAlmostEqual(a, b, places=5)

    def test_embedding_miss_returns_none(self):
        self.assertIsNone(self._cache().get_embedding(digest="missing"))

    def test_embedding_key_is_versioned(self):
        from app.cache.redis_client import CacheClient
        self.assertIn(":v2:", CacheClient.embedding_key(b"x"))

    def test_reverse_cache_roundtrip(self):
        c = self._cache()
        data = {"found_on_internet": True, "matches": [{"link": "x"}]}
        self.assertTrue(c.set_reverse("k", data))
        self.assertEqual(c.get_reverse("k"), data)

    def test_ai_detection_cache_roundtrip(self):
        c = self._cache()
        data = {"is_ai": True, "confidence": 0.9}
        self.assertTrue(c.set_ai_detection("digest", data))
        self.assertEqual(c.get_ai_detection("digest"), data)


# ===========================================================================
# TestConfigDefaults — new configurable knobs
# ===========================================================================

class TestConfigDefaults(unittest.TestCase):

    def test_embedding_dim_default(self):
        from app.config import load_config
        self.assertEqual(load_config().qdrant.embedding_dim, 512)

    def test_embedding_dim_override(self):
        import os
        from app.config import load_config
        with patch.dict(os.environ, {"CLIP_EMBEDDING_DIM": "768"}):
            self.assertEqual(load_config().qdrant.embedding_dim, 768)

    def test_reverse_engine_default_google_lens(self):
        from app.config import load_config
        self.assertEqual(load_config().api_rotator.reverse_engine, "google_lens")


# ===========================================================================
# Entry point
# ===========================================================================

if __name__ == "__main__":
    unittest.main(verbosity=2)
