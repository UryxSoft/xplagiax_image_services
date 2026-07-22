"""
Standalone entrypoint for the reverse-image-search microservice.

This factory wires ONLY what these features need: config, structured
logging + Prometheus metrics, Redis cache, security middleware
(auth/rate-limit), the reverse-search blueprint, and — since neither needs
torch/transformers/qdrant-client either — the existing patents/reverse-web
blueprint (api_rotator.py, SerpApi/Zenserp). It deliberately never imports
app.models.registry, app.storage.vector_repository, app.services.similarity
/indexing, or app.workers.*, so the ML stack is never imported and never
loaded into RAM. That is what makes this an "extremely lightweight"
deployment, as opposed to registering the same blueprints inside the full
xplagiax monolith (see app/factory.py) — also supported, but that process
shares RAM with CLIP/SigLIP/Qdrant regardless of whether either is used.

What you get here that the monolith also has: /api/v1/reverse-image-search(/batch)
and /api/v1/patents/*. What you DON'T get: /api/v1/images, /api/v1/search/*
(CLIP similarity, AI detection) — those need the Qdrant/CLIP/SigLIP stack
and only exist in the monolith.

Usage:
    # Production (Gunicorn)
    gunicorn "app.reverse_search.app:create_app()" \\
        --config docker/gunicorn.reverse_search.conf.py

    # Development
    FLASK_DEBUG=true python -m flask --app "app.reverse_search.app:create_app" run
"""

from __future__ import annotations

import os
from typing import Optional

from flask import Flask, jsonify

from app.config import load_config
from app.observability.telemetry import configure_logging, get_logger, init_metrics, instrument_flask
from app.reverse_search.temp_hosting import TempImageHost


class _TempHostImageStorage:
    """
    Adapts TempImageHost (bytes + content_type -> single-use public URL) to
    the save/get_url/delete shape app.routes.blueprints.patents_bp expects
    from its "storage" service, so patents_bp's existing temp-upload-for-
    reverse-search flow works standalone — without SeaweedFS, local disk,
    or the Qdrant-coupled images_bp retrieval route. Reuses the exact same
    ephemeral hosting this module already built for its own URL-based
    providers (Serper Lens); nothing new to configure.

    save()/get_url()/delete() are always called back-to-back within the
    SAME request in blueprints.py, so per-instance dict state (shared
    across requests only incidentally) is safe here.
    """

    def __init__(self, temp_host: TempImageHost) -> None:
        self._temp_host = temp_host
        self._urls: dict[str, str] = {}

    def save(self, raw_bytes: bytes, content_hash: str, group_id: str, filename: str, fmt: str) -> str:
        content_type = f"image/{fmt}" if fmt else "application/octet-stream"
        self._urls[content_hash] = self._temp_host.host(raw_bytes, content_type=content_type)
        return content_hash

    def get_url(self, key: str, expiry_seconds: int = 3600) -> str:
        return self._urls.get(key, "")

    def delete(self, key: str) -> None:
        # Best-effort bookkeeping only — the underlying entry is single-use
        # (consumed on the provider's first fetch) and always expires via
        # REVERSE_SEARCH_TEMP_HOSTING_TTL regardless of this call.
        self._urls.pop(key, None)


def create_app(config=None) -> Flask:
    app = Flask(__name__)

    if config is None:
        config = load_config()
    configure_logging(config.observability.log_level, config.observability.log_format)
    logger = get_logger(__name__)
    app.debug = config.debug

    logger.info(
        "reverse_search_app_starting",
        service=config.observability.service_name,
        environment=config.observability.environment,
    )

    metrics = init_metrics(config.observability.service_name)
    if config.observability.prometheus_enabled:
        # Avoid "Address already in use" when the Werkzeug reloader forks.
        if not app.debug or os.environ.get("WERKZEUG_RUN_MAIN") == "true":
            metrics.start_prometheus_server(config.observability.prometheus_port)

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
        reverse_ttl=config.redis.reverse_ttl,
        ai_ttl=config.redis.ai_ttl,
    )
    app.extensions["xplagiax_cache"] = cache

    # Shared ephemeral public-hosting instance: reverse_search's own
    # URL-based providers (Serper Lens) AND patents_bp's file-upload path
    # both need to expose raw bytes at a short-lived public URL.
    shared_temp_host = TempImageHost(
        cache, public_base_url=config.reverse_search.public_base_url,
        ttl_s=config.reverse_search.temp_hosting_ttl,
    )
    app.extensions["xplagiax_temp_host"] = shared_temp_host

    from app.reverse_search.factory import build_reverse_search_orchestrator
    orchestrator = build_reverse_search_orchestrator(config.reverse_search, cache, temp_host=shared_temp_host)
    app.extensions["xplagiax_reverse_search_orchestrator"] = orchestrator
    app.extensions["xplagiax_reverse_search_config"] = config.reverse_search

    # Patents + reverse-web-search (SerpApi/Zenserp) — a completely separate
    # feature from app/reverse_search/ above, predating it, but reused here
    # unchanged because api_rotator.py and routes/blueprints.py have zero
    # heavy-ML imports either. The file-upload path (as opposed to
    # image_url) needs a place to host the bytes publicly, which is exactly
    # what shared_temp_host already does.
    from app.factory import build_api_rotator
    api_rotator = build_api_rotator(config, cache, cache.raw)
    app.extensions["xplagiax_api_rotator"] = api_rotator
    app.extensions["xplagiax_storage"] = _TempHostImageStorage(shared_temp_host)

    app.extensions["xplagiax_security_config"] = config.security
    app.config["MAX_CONTENT_LENGTH"] = config.security.max_image_bytes

    from app.reverse_search.routes import reverse_search_bp
    from app.routes.blueprints import patents_bp
    app.register_blueprint(reverse_search_bp)
    app.register_blueprint(patents_bp)

    @app.route("/healthz")
    def liveness():
        return jsonify({"status": "alive"}), 200

    @app.route("/readyz")
    def readiness():
        ready = orchestrator is not None
        return jsonify({
            "status": "ready" if ready else "not_ready",
            "checks": {
                "reverse_search": ready,
                "patents": api_rotator is not None,
                "redis": cache.available,
            },
        }), (200 if ready else 503)

    from app.security.middleware import SecurityMiddleware, register_error_handlers
    register_error_handlers(app)
    app.wsgi_app = SecurityMiddleware(app.wsgi_app, max_content_length=config.security.max_image_bytes)

    # ProxyFix MUST be outermost so REMOTE_ADDR/scheme reflect the real client
    # before SecurityMiddleware and the rate limiter run.
    from werkzeug.middleware.proxy_fix import ProxyFix
    app.wsgi_app = ProxyFix(
        app.wsgi_app,
        x_for=config.security.trusted_proxy_count,
        x_proto=config.security.trusted_proxy_count,
        x_host=config.security.trusted_proxy_count,
    )
    instrument_flask(app)

    logger.info(
        "reverse_search_app_ready",
        endpoints=[
            "/healthz", "/readyz",
            "/api/v1/reverse-image-search", "/api/v1/reverse-image-search/batch",
            "/api/v1/patents/search/image", "/api/v1/patents/search/text",
            "/api/v1/patents/reverse-image", "/api/v1/patents/usage",
        ],
    )
    return app
