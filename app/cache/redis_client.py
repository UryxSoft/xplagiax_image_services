"""
Redis cache layer with graceful degradation.

Design principles:
  - Redis is OPTIONAL. Every operation catches RedisError and falls through.
  - The application NEVER fails because Redis is down.
  - Cache misses are silent; cache errors are logged + metered.
  - All keys are namespaced to avoid collisions with other services.
  - Connection uses a connection pool (not a new connection per call).
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Optional

import redis
from redis.exceptions import RedisError

from app.observability.telemetry import get_logger, get_metrics

logger = get_logger(__name__)

_NS = "xplagiax"  # global key namespace


class CacheClient:
    """
    Thin wrapper around Redis with:
      - Automatic fallback (None return) on any RedisError
      - Namespaced keys
      - Prometheus instrumentation
      - Helper methods for specific use cases (embeddings, results, jobs)
    """

    def __init__(
        self,
        host: str,
        port: int,
        password: Optional[str],
        db: int,
        socket_timeout: float,
        embedding_ttl: int,
        result_ttl: int,
        job_ttl: int,
    ) -> None:
        self._embedding_ttl = embedding_ttl
        self._result_ttl = result_ttl
        self._job_ttl = job_ttl
        self._available = False

        try:
            pool = redis.ConnectionPool(
                host=host,
                port=port,
                password=password,
                db=db,
                socket_timeout=socket_timeout,
                socket_connect_timeout=socket_timeout,
                decode_responses=False,  # we handle encoding ourselves
                max_connections=20,
            )
            self._redis = redis.Redis(connection_pool=pool)
            # Validate connection at startup
            self._redis.ping()
            self._available = True
            logger.info("redis_connected", host=host, port=port, db=db)
        except RedisError as exc:
            logger.warning(
                "redis_unavailable_degraded_mode",
                error=str(exc),
                hint="Service will run without caching. Set REDIS_HOST to enable.",
            )
            self._redis = None

    # ------------------------------------------------------------------
    # Low-level get/set with graceful degradation
    # ------------------------------------------------------------------

    def get(self, key: str) -> Optional[Any]:
        """Get a JSON-deserialized value. Returns None on miss or error."""
        if self._redis is None:
            return None
        full_key = f"{_NS}:{key}"
        try:
            raw = self._redis.get(full_key)
            if raw is None:
                get_metrics().cache_misses.labels(cache_type="redis").inc()
                return None
            get_metrics().cache_hits.labels(cache_type="redis").inc()
            return json.loads(raw)
        except RedisError as exc:
            get_metrics().cache_errors.labels(operation="get").inc()
            logger.warning("redis_get_error", key=full_key, error=str(exc))
            return None

    def set(self, key: str, value: Any, ttl: int) -> bool:
        """Set a JSON-serialized value with TTL. Returns True on success."""
        if self._redis is None:
            return False
        full_key = f"{_NS}:{key}"
        try:
            self._redis.setex(full_key, ttl, json.dumps(value))
            return True
        except RedisError as exc:
            get_metrics().cache_errors.labels(operation="set").inc()
            logger.warning("redis_set_error", key=full_key, error=str(exc))
            return False

    def delete(self, key: str) -> bool:
        if self._redis is None:
            return False
        try:
            self._redis.delete(f"{_NS}:{key}")
            return True
        except RedisError:
            return False

    def incr(self, key: str, ttl_if_new: Optional[int] = None) -> Optional[int]:
        """
        Atomic increment. Used for usage counters (API rotator).
        Returns new value, or None if Redis is unavailable.
        """
        if self._redis is None:
            return None
        full_key = f"{_NS}:{key}"
        try:
            pipe = self._redis.pipeline()
            pipe.incr(full_key)
            if ttl_if_new is not None:
                pipe.expire(full_key, ttl_if_new, nx=True)  # set TTL only if new key
            results = pipe.execute()
            return results[0]
        except RedisError as exc:
            get_metrics().cache_errors.labels(operation="incr").inc()
            logger.warning("redis_incr_error", key=full_key, error=str(exc))
            return None

    def get_int(self, key: str) -> Optional[int]:
        if self._redis is None:
            return None
        full_key = f"{_NS}:{key}"
        try:
            raw = self._redis.get(full_key)
            return int(raw) if raw is not None else None
        except RedisError:
            return None

    # ------------------------------------------------------------------
    # Domain-specific helpers
    # ------------------------------------------------------------------

    @staticmethod
    def embedding_key(image_bytes: bytes) -> str:
        """Deterministic cache key from image content hash."""
        digest = hashlib.sha256(image_bytes).hexdigest()
        return f"embed:clip:{digest}"

    def get_embedding(self, image_bytes: bytes) -> Optional[list]:
        return self.get(self.embedding_key(image_bytes))

    def set_embedding(self, image_bytes: bytes, vector: list) -> bool:
        return self.set(self.embedding_key(image_bytes), vector, self._embedding_ttl)

    def job_key(self, job_id: str) -> str:
        return f"job:{job_id}"

    def get_job(self, job_id: str) -> Optional[dict]:
        return self.get(self.job_key(job_id))

    def set_job(self, job_id: str, status: dict) -> bool:
        return self.set(self.job_key(job_id), status, self._job_ttl)

    def update_job(self, job_id: str, updates: dict) -> bool:
        existing = self.get_job(job_id) or {}
        existing.update(updates)
        return self.set_job(job_id, existing)

    # API usage counters — month-scoped, atomic
    @staticmethod
    def api_usage_key(provider: str, year_month: str) -> str:
        """e.g. 'api_usage:serpapi:2025-03'"""
        return f"api_usage:{provider}:{year_month}"

    def increment_api_usage(self, provider: str, year_month: str) -> Optional[int]:
        """
        Atomically increment and return new count.
        Key expires after 35 days (covers full month + buffer).
        """
        key = self.api_usage_key(provider, year_month)
        return self.incr(key, ttl_if_new=35 * 86_400)

    def get_api_usage(self, provider: str, year_month: str) -> int:
        key = self.api_usage_key(provider, year_month)
        return self.get_int(key) or 0

    @property
    def available(self) -> bool:
        return self._available

    def health_check(self) -> dict:
        if self._redis is None:
            return {"status": "unavailable", "mode": "degraded"}
        try:
            self._redis.ping()
            return {"status": "ok"}
        except RedisError as exc:
            return {"status": "error", "error": str(exc)}
