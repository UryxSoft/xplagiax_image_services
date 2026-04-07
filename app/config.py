"""
Configuration module — fail-fast on missing critical env vars,
graceful degradation on optional ones.

All configuration is read once at startup. No os.getenv() scattered
across the codebase — single source of truth.
"""

from __future__ import annotations

import os
import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _require(key: str) -> str:
    """Fail at startup if a required env var is missing."""
    val = os.environ.get(key)
    if not val:
        raise EnvironmentError(
            f"Required environment variable '{key}' is not set. "
            "Check your .env or container env config."
        )
    return val


def _optional(key: str, default: str = "") -> str:
    return os.environ.get(key, default)


def _optional_int(key: str, default: int) -> int:
    raw = os.environ.get(key)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        logger.warning("Invalid integer for %s='%s', using default %d", key, raw, default)
        return default


def _optional_float(key: str, default: float) -> float:
    raw = os.environ.get(key)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        logger.warning("Invalid float for %s='%s', using default %f", key, raw, default)
        return default


def _optional_bool(key: str, default: bool) -> bool:
    raw = os.environ.get(key, "").lower()
    if raw in ("1", "true", "yes"):
        return True
    if raw in ("0", "false", "no"):
        return False
    return default


# ---------------------------------------------------------------------------
# Config dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class QdrantConfig:
    host: str
    port: int
    collection: str
    api_key: Optional[str]
    # HNSW tuning — higher ef_construct / m = better recall, slower indexing
    hnsw_m: int = 16
    hnsw_ef_construct: int = 200
    hnsw_ef_search: int = 128
    full_scan_threshold: int = 10_000


@dataclass(frozen=True)
class RedisConfig:
    host: str
    port: int
    password: Optional[str]
    db: int
    socket_timeout: float
    # Cache TTLs in seconds
    embedding_ttl: int       # 24h — embeddings rarely change for same image bytes
    result_ttl: int          # 5 min — search results can change as collection grows
    job_ttl: int             # 1h — async job results


@dataclass(frozen=True)
class ModelConfig:
    siglip_model_id: str
    clip_model_id: str
    device: str               # "cuda", "cpu", or "auto"
    max_batch_size: int
    inference_timeout_s: float


@dataclass(frozen=True)
class ApiRotatorConfig:
    serpapi_key: Optional[str]
    zenserp_key: Optional[str]
    usage_backend: str        # "redis" | "file"
    usage_file: str
    request_timeout_s: float
    max_retries: int
    # Per-provider monthly limits (free tier)
    serpapi_limit: int
    zenserp_limit: int
    # Health check interval in seconds
    health_check_interval: int


@dataclass(frozen=True)
class SecurityConfig:
    api_key: Optional[str]              # None = auth disabled (dev only)
    require_auth: bool
    max_image_bytes: int                # hard reject above this
    allowed_mime_types: frozenset
    rate_limit_per_minute: int
    rate_limit_per_hour: int


@dataclass(frozen=True)
class StorageConfig:
    image_backend: str                      # "local" | "seaweedfs_filer" | "seaweedfs_native"
    local_base_path: str
    # SeaweedFS — shared by both filer and native modes
    seaweedfs_replication: str              # "000" no replication | "001" rack | "100" DC
    seaweedfs_collection: str              # logical namespace, empty = default
    seaweedfs_ttl: str                     # file TTL: "3m","4h","5d","" = permanent
    seaweedfs_request_timeout: float       # HTTP timeout for storage calls (seconds)
    seaweedfs_max_retries: int             # retry on 5xx / transient network errors
    # SeaweedFS Filer mode (IMAGE_BACKEND=seaweedfs_filer) — recommended
    seaweedfs_filer_url: Optional[str]     # internal: http://seaweedfs-filer:8888
    seaweedfs_public_url: Optional[str]    # public-facing URL for get_url()
    # SeaweedFS Native mode (IMAGE_BACKEND=seaweedfs_native)
    seaweedfs_master_url: Optional[str]    # internal: http://seaweedfs-master:9333
    seaweedfs_public_volume_url: Optional[str]  # override volume URL for public access


@dataclass(frozen=True)
class ObservabilityConfig:
    log_level: str
    log_format: str                     # "json" | "text"
    prometheus_enabled: bool
    prometheus_port: int
    otel_enabled: bool
    otel_endpoint: Optional[str]
    service_name: str
    environment: str


@dataclass(frozen=True)
class AppConfig:
    qdrant: QdrantConfig
    redis: RedisConfig
    model: ModelConfig
    api_rotator: ApiRotatorConfig
    security: SecurityConfig
    storage: StorageConfig
    observability: ObservabilityConfig
    debug: bool
    workers: int


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def load_config() -> AppConfig:
    """
    Build AppConfig from environment variables.

    Call once at app startup. Raises EnvironmentError if any required
    variable is missing.
    """
    qdrant = QdrantConfig(
        host=_optional("QDRANT_HOST", "qdrant"),
        port=_optional_int("QDRANT_PORT", 6333),
        collection=_optional("QDRANT_COLLECTION", "xplagiax_images"),
        api_key=_optional("QDRANT_API_KEY") or None,
        hnsw_m=_optional_int("QDRANT_HNSW_M", 16),
        hnsw_ef_construct=_optional_int("QDRANT_HNSW_EF_CONSTRUCT", 200),
        hnsw_ef_search=_optional_int("QDRANT_HNSW_EF_SEARCH", 128),
        full_scan_threshold=_optional_int("QDRANT_FULL_SCAN_THRESHOLD", 10_000),
    )

    redis = RedisConfig(
        host=_optional("REDIS_HOST", "redis"),
        port=_optional_int("REDIS_PORT", 6379),
        password=_optional("REDIS_PASSWORD") or None,
        db=_optional_int("REDIS_DB", 0),
        socket_timeout=_optional_float("REDIS_SOCKET_TIMEOUT", 1.0),
        embedding_ttl=_optional_int("REDIS_EMBEDDING_TTL", 86_400),
        result_ttl=_optional_int("REDIS_RESULT_TTL", 300),
        job_ttl=_optional_int("REDIS_JOB_TTL", 3_600),
    )

    import torch
    device = _optional("MODEL_DEVICE", "auto")
    
    # En ModelRegistry.__init__:
    if device == "auto":
        import torch
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        self._device = torch.device(device)

    model = ModelConfig(
        siglip_model_id=_optional(
            "SIGLIP_MODEL_ID", "Ateeqq/ai-vs-human-image-detector"
        ),
        clip_model_id=_optional("CLIP_MODEL_ID", "clip-ViT-B-32"),
        device=device,
        max_batch_size=_optional_int("MODEL_MAX_BATCH_SIZE", 32),
        inference_timeout_s=_optional_float("MODEL_INFERENCE_TIMEOUT", 30.0),
    )

    serpapi_key = _optional("SERPAPI_KEY") or None
    zenserp_key = _optional("ZENSERP_KEY") or None
    if not serpapi_key:
        logger.warning(
            "SERPAPI_KEY not set — reverse image search will be unavailable"
        )

    api_rotator = ApiRotatorConfig(
        serpapi_key=serpapi_key,
        zenserp_key=zenserp_key,
        usage_backend=_optional("API_USAGE_BACKEND", "redis"),
        usage_file=_optional("API_USAGE_FILE", "/tmp/api_usage.json"),
        request_timeout_s=_optional_float("API_REQUEST_TIMEOUT", 15.0),
        max_retries=_optional_int("API_MAX_RETRIES", 3),
        serpapi_limit=_optional_int("SERPAPI_MONTHLY_LIMIT", 250),
        zenserp_limit=_optional_int("ZENSERP_MONTHLY_LIMIT", 50),
        health_check_interval=_optional_int("API_HEALTH_CHECK_INTERVAL", 300),
    )

    require_auth = _optional_bool("REQUIRE_AUTH", True)
    api_key = _optional("SERVICE_API_KEY") or None
    if require_auth and not api_key:
        raise EnvironmentError(
            "REQUIRE_AUTH=true but SERVICE_API_KEY is not set. "
            "Either set SERVICE_API_KEY or set REQUIRE_AUTH=false (dev only)."
        )

    security = SecurityConfig(
        api_key=api_key,
        require_auth=require_auth,
        max_image_bytes=_optional_int("MAX_IMAGE_BYTES", 20 * 1024 * 1024),  # 20 MB
        allowed_mime_types=frozenset(
            _optional("ALLOWED_MIME_TYPES", "jpeg,png,webp,bmp,tiff,gif").split(",")
        ),
        rate_limit_per_minute=_optional_int("RATE_LIMIT_PER_MINUTE", 30),
        rate_limit_per_hour=_optional_int("RATE_LIMIT_PER_HOUR", 500),
    )

    storage = StorageConfig(
        image_backend=_optional("IMAGE_BACKEND", "seaweedfs_filer"),
        local_base_path=_optional("IMAGE_BASE_PATH", "/data/images"),
        seaweedfs_replication=_optional("SEAWEEDFS_REPLICATION", "000"),
        seaweedfs_collection=_optional("SEAWEEDFS_COLLECTION", ""),
        seaweedfs_ttl=_optional("SEAWEEDFS_TTL", ""),
        seaweedfs_request_timeout=_optional_float("SEAWEEDFS_REQUEST_TIMEOUT", 30.0),
        seaweedfs_max_retries=_optional_int("SEAWEEDFS_MAX_RETRIES", 3),
        seaweedfs_filer_url=_optional("SEAWEEDFS_FILER_URL") or None,
        seaweedfs_public_url=_optional("SEAWEEDFS_PUBLIC_URL") or None,
        seaweedfs_master_url=_optional("SEAWEEDFS_MASTER_URL") or None,
        seaweedfs_public_volume_url=_optional("SEAWEEDFS_PUBLIC_VOLUME_URL") or None,
    )

    obs = ObservabilityConfig(
        log_level=_optional("LOG_LEVEL", "INFO").upper(),
        log_format=_optional("LOG_FORMAT", "json"),
        prometheus_enabled=_optional_bool("PROMETHEUS_ENABLED", True),
        prometheus_port=_optional_int("PROMETHEUS_PORT", 9090),
        otel_enabled=_optional_bool("OTEL_ENABLED", False),
        otel_endpoint=_optional("OTEL_EXPORTER_OTLP_ENDPOINT") or None,
        service_name=_optional("SERVICE_NAME", "xplagiax-image-service"),
        environment=_optional("ENVIRONMENT", "production"),
    )

    return AppConfig(
        qdrant=qdrant,
        redis=redis,
        model=model,
        api_rotator=api_rotator,
        security=security,
        storage=storage,
        observability=obs,
        debug=_optional_bool("FLASK_DEBUG", False),
        workers=_optional_int("WORKER_PROCESSES", 1),
    )
