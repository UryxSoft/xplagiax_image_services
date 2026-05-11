"""
Flask application factory.

All dependencies are wired here and stored in app.extensions under
namespaced keys so any route can access them via current_app.extensions.

Startup sequence:
  1. Logging (must be first)
  2. Metrics initialisation
  3. Configuration loading (fail-fast on missing required env vars)
  4. Cache (Redis) — optional, graceful degradation
  5. Storage (local / S3)
  6. Qdrant repository
  7. Model registry — models loaded after Flask starts (not at import time)
  8. Services (indexing, similarity)
  9. API rotator
  10. Blueprints + error handlers
  11. Security middleware
  12. Prometheus server (background thread)

Usage:
    # Production (Gunicorn)
    gunicorn "app.factory:create_app()" --workers 1 --worker-class gevent

    # Development
    FLASK_DEBUG=true python -m flask --app "app.factory:create_app" run
"""

from __future__ import annotations

import os
import threading
from typing import Optional

from flask import Flask

from app.config import AppConfig, load_config
from app.observability.telemetry import (
    configure_logging,
    get_logger,
    init_metrics,
    instrument_flask,
)


def create_app(config: Optional[AppConfig] = None) -> Flask:
    """
    Flask application factory.

    Pass a pre-built AppConfig for testing; otherwise loads from env vars.
    """
    app = Flask(__name__)

    # ------------------------------------------------------------------ #
    # 1. Logging — must be first                                          #
    # ------------------------------------------------------------------ #
    # We configure with defaults here; reconfigure once config is loaded
    #configure_logging(log_level="INFO", log_format="json")
    #logger = get_logger(__name__)

    # ------------------------------------------------------------------ #
    # 2. Configuration                                                    #
    # ------------------------------------------------------------------ #
    if config is None:
        config = load_config()
    configure_logging(config.observability.log_level, config.observability.log_format)
    logger = get_logger(__name__)
    #configure_logging(
    #    log_level=config.observability.log_level,
    #    log_format=config.observability.log_format,
    #)
    logger.info(
        "app_starting",
        service=config.observability.service_name,
        environment=config.observability.environment,
        device=config.model.device,
    )

    # ------------------------------------------------------------------ #
    # 3. Metrics                                                          #
    # ------------------------------------------------------------------ #
    metrics = init_metrics(config.observability.service_name)
    if config.observability.prometheus_enabled:
        # Prevent Address already in use error when Flask reloader spawns child process
        if not app.debug or os.environ.get("WERKZEUG_RUN_MAIN") == "true":
            metrics.start_prometheus_server(config.observability.prometheus_port)

    # ------------------------------------------------------------------ #
    # 4. Redis cache                                                      #
    # ------------------------------------------------------------------ #
    from app.cache.redis_client import CacheClient
    cache = CacheClient(
        host=config.redis.host,
        port=config.redis.port,
        password=config.redis.password,
        db=config.redis.db,
        socket_timeout=config.redis.socket_timeout,
        embedding_ttl=config.redis.embedding_ttl,
        result_ttl=config.redis.result_ttl,
        job_ttl=config.redis.job_ttl,
    )
    app.extensions["xplagiax_cache"] = cache

    # ------------------------------------------------------------------ #
    # 5. Image storage                                                    #
    # ------------------------------------------------------------------ #
    from app.storage.image_storage import create_storage
    storage = create_storage(
        backend=config.storage.image_backend,
        local_base_path=config.storage.local_base_path,
        seaweedfs_filer_url=config.storage.seaweedfs_filer_url,
        seaweedfs_public_url=config.storage.seaweedfs_public_url,
        seaweedfs_replication=config.storage.seaweedfs_replication,
        seaweedfs_collection=config.storage.seaweedfs_collection,
        seaweedfs_ttl=config.storage.seaweedfs_ttl,
        seaweedfs_request_timeout=config.storage.seaweedfs_request_timeout,
        seaweedfs_max_retries=config.storage.seaweedfs_max_retries,
        seaweedfs_master_url=config.storage.seaweedfs_master_url,
        seaweedfs_public_volume_url=config.storage.seaweedfs_public_volume_url,
    )
    app.extensions["xplagiax_storage"] = storage
    app.extensions["xplagiax_storage_config"] = config.storage

    # ------------------------------------------------------------------ #
    # 6. Qdrant repository                                               #
    # ------------------------------------------------------------------ #
    from app.storage.vector_repository import VectorRepository
    repo = VectorRepository(
        host=config.qdrant.host,
        port=config.qdrant.port,
        collection=config.qdrant.collection,
        api_key=config.qdrant.api_key,
        hnsw_m=config.qdrant.hnsw_m,
        hnsw_ef_construct=config.qdrant.hnsw_ef_construct,
        hnsw_ef_search=config.qdrant.hnsw_ef_search,
        full_scan_threshold=config.qdrant.full_scan_threshold,
    )
    app.extensions["xplagiax_repo"] = repo

    # ------------------------------------------------------------------ #
    # 7. Model registry (lazy load AFTER Flask starts)                   #
    # ------------------------------------------------------------------ #
    from app.models.registry import ModelRegistry
    models = ModelRegistry(
        siglip_model_id=config.model.siglip_model_id,
        clip_model_id=config.model.clip_model_id,
        device=config.model.device,
        max_batch_size=config.model.max_batch_size,
    )
    app.extensions["xplagiax_models"] = models

    def _load_models_background():
        """Load models in a background thread so Flask starts immediately."""
        with app.app_context():
            try:
                models.load_all()
                logger.info("models_ready_serving_traffic")
            except Exception as e:
                logger.error("background_model_loader_failed", error=str(e), exc_info=True)

    t = threading.Thread(target=_load_models_background, daemon=True, name="model-loader")
    t.start()

    # ------------------------------------------------------------------ #
    # 8. Services                                                         #
    # ------------------------------------------------------------------ #
    from app.services.similarity import SimilarityService
    similarity_svc = SimilarityService(
        model_registry=models,
        vector_repository=repo,
        cache=cache,
    )
    app.extensions["xplagiax_similarity"] = similarity_svc

    # Indexing service: determine if async mode is available
    raw_redis = None
    if cache.available:
        try:
            import redis as redis_lib
            raw_redis = redis_lib.Redis(
                host=config.redis.host,
                port=config.redis.port,
                password=config.redis.password,
                db=config.redis.db,
            )
        except Exception:
            pass

    from app.services.indexing import IndexingService
    indexing_svc = IndexingService(
        model_registry=models,
        vector_repository=repo,
        image_storage=storage,
        cache=cache,
        redis_client=raw_redis,
        async_enabled=True,
    )
    app.extensions["xplagiax_indexing"] = indexing_svc

    # ------------------------------------------------------------------ #
    # 9. API Rotator                                                      #
    # ------------------------------------------------------------------ #
    api_rotator = _build_api_rotator(config, cache)
    app.extensions["xplagiax_api_rotator"] = api_rotator

    # ------------------------------------------------------------------ #
    # 10. Security config (available to decorators via extensions)       #
    # ------------------------------------------------------------------ #
    app.extensions["xplagiax_security_config"] = config.security
    app.config["MAX_CONTENT_LENGTH"] = config.security.max_image_bytes

    # ------------------------------------------------------------------ #
    # 11. Blueprints                                                      #
    # ------------------------------------------------------------------ #
    from app.routes.blueprints import (
        admin_bp,
        health_bp,
        images_bp,
        jobs_bp,
        patents_bp,
        search_bp,
    )
    for bp in (health_bp, images_bp, search_bp, patents_bp, admin_bp, jobs_bp):
        app.register_blueprint(bp)

    # ------------------------------------------------------------------ #
    # 12. Error handlers + observability hooks                           #
    # ------------------------------------------------------------------ #
    from app.security.middleware import SecurityMiddleware, register_error_handlers
    register_error_handlers(app)
    app.wsgi_app = SecurityMiddleware(
        app.wsgi_app,
        max_content_length=config.security.max_image_bytes,
    )
    instrument_flask(app)

    logger.info(
        "app_ready",
        endpoints=[
            "/healthz", "/readyz", "/health",
            "/api/v1/images", "/api/v1/search/similar",
            "/api/v1/search/plagiarism", "/api/v1/search/ai-detection",
            "/api/v1/patents/search/image", "/api/v1/admin/collection/items",
        ],
    )
    return app


# ---------------------------------------------------------------------------
# API Rotator factory helper
# ---------------------------------------------------------------------------

def _build_api_rotator(config: AppConfig, cache):
    from app.services.api_rotator import (
        ProviderConfig,
        SmartApiRotator,
        UsageTracker,
    )

    providers = []
    if config.api_rotator.serpapi_key:
        providers.append(
            ProviderConfig(
                name="serpapi",
                api_key=config.api_rotator.serpapi_key,
                monthly_limit=config.api_rotator.serpapi_limit,
                base_url_search="https://serpapi.com/search.json",
                base_url_patents="https://serpapi.com/search.json",
            )
        )
    if config.api_rotator.zenserp_key:
        providers.append(
            ProviderConfig(
                name="zenserp",
                api_key=config.api_rotator.zenserp_key,
                monthly_limit=config.api_rotator.zenserp_limit,
                base_url_search="https://app.zenserp.com/api/v2/search",
                base_url_patents="https://app.zenserp.com/api/v2/search",
            )
        )

    if not providers:
        get_logger(__name__).warning(
            "api_rotator_disabled",
            reason="No provider API keys configured",
        )
        return None

    tracker = UsageTracker(cache=cache, usage_file=config.api_rotator.usage_file)

    return SmartApiRotator(
        providers=providers,
        usage_tracker=tracker,
        request_timeout_s=config.api_rotator.request_timeout_s,
        max_retries=config.api_rotator.max_retries,
        health_check_interval=config.api_rotator.health_check_interval,
    )
