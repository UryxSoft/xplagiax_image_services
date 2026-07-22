"""
Configuration module — fail-fast on missing critical env vars,
graceful degradation on optional ones.

All configuration is read once at startup. No os.getenv() scattered
across the codebase — single source of truth.
"""

from __future__ import annotations

import os
import json
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
    # Vector dimension — must match the CLIP model (clip-ViT-B-32 = 512)
    embedding_dim: int = 512
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
    reverse_ttl: int = 7 * 86_400    # 7d — reverse-search results (web changes slowly)
    ai_ttl: int = 30 * 86_400        # 30d — AI detection (deterministic for same bytes)


@dataclass(frozen=True)
class ModelConfig:
    siglip_model_id: str
    clip_model_id: str
    device: str               # "cuda", "cpu", or "auto"
    max_batch_size: int
    inference_timeout_s: float
    # AI-detection thresholds & calibration
    ai_confidence_high: float = 0.85
    ai_confidence_med: float = 0.60
    ai_temperature: float = 1.0          # >1 softens over-confident logits (calibration)
    # Cap concurrent inferences to protect RAM under load
    inference_max_concurrency: int = 2
    # Optional override mapping label/index -> 'ai'|'human' (for swapping models)
    ai_label_map: Optional[dict] = None


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
    # SerpApi reverse-image engine: "google_lens" (recommended) | "google_reverse_image"
    reverse_engine: str = "google_lens"
    # Distributed circuit breaker (Redis-backed)
    circuit_failure_threshold: int = 5
    circuit_recovery_s: int = 120


@dataclass(frozen=True)
class SecurityConfig:
    api_key: Optional[str]              # None = auth disabled (dev only)
    admin_api_key: Optional[str]        # separate key for destructive admin ops
    require_auth: bool
    max_image_bytes: int                # hard reject above this
    max_image_pixels: int              # decompression-bomb guard (W*H)
    allowed_mime_types: frozenset
    rate_limit_per_minute: int
    rate_limit_per_hour: int
    # Trusted reverse-proxy CIDRs/IPs whose X-Forwarded-For we honour
    trusted_proxies: tuple
    # Number of proxies in front of the app (for werkzeug ProxyFix)
    trusted_proxy_count: int
    # Allow reading images from local absolute filesystem paths (LFI risk → off in prod)
    allow_local_image_path: bool


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
class ProviderSettings:
    """Per-provider tuning for the reverse-image-search early-stop chain."""
    enabled: bool
    priority: int              # lower = tried first
    stop_threshold: float      # similarity (0-100) at/above which we stop immediately
    timeout_s: float           # connect+read timeout for this provider's HTTP call
    requires_public_url: bool  # True if the provider needs a fetchable image URL (vs inline bytes)


@dataclass(frozen=True)
class ReverseSearchConfig:
    enabled: bool
    max_providers: int
    max_retries: int
    max_batch_size: int          # cap on items per /reverse-image-search/batch request
    cache_ttl_found: int         # seconds — a discovered copy rarely un-appears
    cache_ttl_not_found: int     # seconds — much shorter: absence can change as the web index grows
    public_base_url: Optional[str]  # this service's own public URL, for URL-based providers
    temp_hosting_ttl: int            # seconds an uploaded image stays servable at a temp public URL

    google_vision_api_key: Optional[str]
    google_vision: ProviderSettings

    serper_api_key: Optional[str]
    serper: ProviderSettings

    mungfali_api_key: Optional[str]
    mungfali: ProviderSettings


@dataclass(frozen=True)
class AppConfig:
    qdrant: QdrantConfig
    redis: RedisConfig
    model: ModelConfig
    api_rotator: ApiRotatorConfig
    security: SecurityConfig
    storage: StorageConfig
    observability: ObservabilityConfig
    reverse_search: ReverseSearchConfig
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
        embedding_dim=_optional_int("CLIP_EMBEDDING_DIM", 512),
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
        reverse_ttl=_optional_int("REDIS_REVERSE_TTL", 7 * 86_400),
        ai_ttl=_optional_int("REDIS_AI_TTL", 30 * 86_400),
    )

    device = _optional("MODEL_DEVICE", "auto")

    ai_label_map_raw = _optional("AI_LABEL_MAP")
    ai_label_map = None
    if ai_label_map_raw:
        try:
            ai_label_map = json.loads(ai_label_map_raw)
            if not isinstance(ai_label_map, dict):
                raise ValueError("must be a JSON object")
        except (ValueError, json.JSONDecodeError) as exc:
            logger.warning("Invalid AI_LABEL_MAP (%s) — ignoring: %s", ai_label_map_raw, exc)
            ai_label_map = None

    model = ModelConfig(
        siglip_model_id=_optional(
            "SIGLIP_MODEL_ID", "Ateeqq/ai-vs-human-image-detector"
        ),
        clip_model_id=_optional("CLIP_MODEL_ID", "clip-ViT-B-32"),
        device=device,
        max_batch_size=_optional_int("MODEL_MAX_BATCH_SIZE", 32),
        inference_timeout_s=_optional_float("MODEL_INFERENCE_TIMEOUT", 30.0),
        ai_confidence_high=_optional_float("AI_CONFIDENCE_HIGH", 0.85),
        ai_confidence_med=_optional_float("AI_CONFIDENCE_MED", 0.60),
        ai_temperature=_optional_float("AI_TEMPERATURE", 1.0),
        inference_max_concurrency=_optional_int("INFERENCE_MAX_CONCURRENCY", 2),
        ai_label_map=ai_label_map,
    )

    # Secrets are ALWAYS read from the environment — never hardcoded.
    serpapi_key = _optional("SERPAPI_KEY") or None
    zenserp_key = _optional("ZENSERP_KEY") or None
    if not serpapi_key and not zenserp_key:
        logger.warning(
            "No reverse-image provider key set (SERPAPI_KEY / ZENSERP_KEY) — "
            "reverse image search will be unavailable"
        )

    api_rotator = ApiRotatorConfig(
        serpapi_key=serpapi_key,
        zenserp_key=zenserp_key,
        usage_backend=_optional("API_USAGE_BACKEND", "redis"),
        usage_file=_optional("API_USAGE_FILE", "/tmp/api_usage.json"),
        request_timeout_s=_optional_float("API_REQUEST_TIMEOUT", 10.0),
        max_retries=_optional_int("API_MAX_RETRIES", 3),
        serpapi_limit=_optional_int("SERPAPI_MONTHLY_LIMIT", 250),
        zenserp_limit=_optional_int("ZENSERP_MONTHLY_LIMIT", 50),
        health_check_interval=_optional_int("API_HEALTH_CHECK_INTERVAL", 300),
        reverse_engine=_optional("REVERSE_IMAGE_ENGINE", "google_lens"),
        circuit_failure_threshold=_optional_int("API_CIRCUIT_FAILURE_THRESHOLD", 5),
        circuit_recovery_s=_optional_int("API_CIRCUIT_RECOVERY_S", 120),
    )

    require_auth = _optional_bool("REQUIRE_AUTH", False)
    api_key = _optional("SERVICE_API_KEY") or None
    admin_api_key = _optional("ADMIN_API_KEY") or None
    if require_auth and not api_key:
        raise EnvironmentError(
            "REQUIRE_AUTH=true but SERVICE_API_KEY is not set. "
            "Either set SERVICE_API_KEY or set REQUIRE_AUTH=false (dev only)."
        )

    trusted_proxies = tuple(
        p.strip()
        for p in _optional("TRUSTED_PROXIES", "127.0.0.1,10.0.0.0/8,172.16.0.0/12,192.168.0.0/16").split(",")
        if p.strip()
    )

    security = SecurityConfig(
        api_key=api_key,
        admin_api_key=admin_api_key,
        require_auth=require_auth,
        max_image_bytes=_optional_int("MAX_IMAGE_BYTES", 20 * 1024 * 1024),  # 20 MB
        max_image_pixels=_optional_int("MAX_IMAGE_PIXELS", 40_000_000),      # 40 MP
        allowed_mime_types=frozenset(
            _optional("ALLOWED_MIME_TYPES", "jpeg,png,webp,bmp,tiff,gif").split(",")
        ),
        rate_limit_per_minute=_optional_int("RATE_LIMIT_PER_MINUTE", 30),
        rate_limit_per_hour=_optional_int("RATE_LIMIT_PER_HOUR", 500),
        trusted_proxies=trusted_proxies,
        trusted_proxy_count=_optional_int("TRUSTED_PROXY_COUNT", 1),
        allow_local_image_path=_optional_bool("ALLOW_LOCAL_IMAGE_PATH", False),
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

    # Reverse-image-search — lightweight external-API orchestrator (no local ML).
    # Secrets read from env ONLY, same convention as serpapi_key/zenserp_key above.
    google_vision_api_key = _optional("GOOGLE_VISION_API_KEY") or None
    serper_api_key = _optional("SERPER_API_KEY") or None
    mungfali_api_key = _optional("MUNGFALI_API_KEY") or None

    reverse_search = ReverseSearchConfig(
        enabled=_optional_bool("REVERSE_SEARCH_ENABLED", True),
        max_providers=_optional_int("REVERSE_SEARCH_MAX_PROVIDERS", 3),
        max_retries=_optional_int("REVERSE_SEARCH_MAX_RETRIES", 2),
        max_batch_size=_optional_int("REVERSE_SEARCH_MAX_BATCH_SIZE", 20),
        cache_ttl_found=_optional_int("REVERSE_SEARCH_CACHE_TTL_FOUND", 30 * 86_400),
        cache_ttl_not_found=_optional_int("REVERSE_SEARCH_CACHE_TTL_NOT_FOUND", 86_400),
        public_base_url=_optional("REVERSE_SEARCH_PUBLIC_BASE_URL") or None,
        temp_hosting_ttl=_optional_int("REVERSE_SEARCH_TEMP_HOSTING_TTL", 120),
        google_vision_api_key=google_vision_api_key,
        google_vision=ProviderSettings(
            enabled=_optional_bool("GOOGLE_VISION_ENABLED", True),
            priority=_optional_int("GOOGLE_VISION_PRIORITY", 1),
            stop_threshold=_optional_float("GOOGLE_VISION_STOP_THRESHOLD", 98.0),
            timeout_s=_optional_float("GOOGLE_VISION_TIMEOUT", 5.0),
            requires_public_url=False,  # sends image bytes inline (base64) — no temp hosting needed
        ),
        serper_api_key=serper_api_key,
        serper=ProviderSettings(
            enabled=_optional_bool("SERPER_ENABLED", True),
            priority=_optional_int("SERPER_PRIORITY", 2),
            stop_threshold=_optional_float("SERPER_STOP_THRESHOLD", 95.0),
            timeout_s=_optional_float("SERPER_TIMEOUT", 6.0),
            requires_public_url=True,  # google.serper.dev/lens fetches the image itself from a URL
        ),
        mungfali_api_key=mungfali_api_key,
        mungfali=ProviderSettings(
            # Disabled by default: Mungfali's public docs describe a keyword/stock
            # image search product. We could not verify an image-upload reverse-
            # search contract, so this adapter is a template — enable only after
            # confirming the real endpoint with the vendor (see providers/mungfali.py).
            enabled=_optional_bool("MUNGFALI_ENABLED", False),
            priority=_optional_int("MUNGFALI_PRIORITY", 3),
            stop_threshold=_optional_float("MUNGFALI_STOP_THRESHOLD", 90.0),
            timeout_s=_optional_float("MUNGFALI_TIMEOUT", 5.0),
            requires_public_url=True,
        ),
    )
    if reverse_search.enabled and not (google_vision_api_key or serper_api_key or mungfali_api_key):
        logger.warning(
            "REVERSE_SEARCH_ENABLED=true but no provider API key is set "
            "(GOOGLE_VISION_API_KEY / SERPER_API_KEY / MUNGFALI_API_KEY) — "
            "reverse image search will be unavailable"
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
        reverse_search=reverse_search,
        debug=_optional_bool("FLASK_DEBUG", False),
        workers=_optional_int("WORKER_PROCESSES", 1),
    )
