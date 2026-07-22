"""
Standalone entrypoint for the reverse-image-search microservice.

This factory wires ONLY what this feature needs: config, structured logging
+ Prometheus metrics, Redis cache, security middleware (auth/rate-limit), and
the reverse-search blueprint. It deliberately never imports
app.models.registry, app.storage.vector_repository, app.storage.image_storage,
app.services.similarity/indexing, or app.workers.* — so torch, transformers,
sentence-transformers and qdrant-client are never imported and never loaded
into RAM. That is what makes this an "extremely lightweight" deployment, as
opposed to registering the same blueprint inside the full xplagiax monolith
(see app/factory.py) — also supported, but that process shares RAM with
CLIP/SigLIP/Qdrant regardless of whether reverse-search is even used.

Usage:
    # Production (Gunicorn)
    gunicorn "app.reverse_search.app:create_app()" \\
        --config docker/gunicorn.reverse_search.conf.py

    # Development
    FLASK_DEBUG=true python -m flask --app "app.reverse_search.app:create_app" run
"""

from __future__ import annotations

import os

from flask import Flask, jsonify

from app.config import load_config
from app.observability.telemetry import configure_logging, get_logger, init_metrics, instrument_flask


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

    from app.reverse_search.factory import build_reverse_search_orchestrator
    orchestrator = build_reverse_search_orchestrator(config.reverse_search, cache)
    app.extensions["xplagiax_reverse_search_orchestrator"] = orchestrator
    app.extensions["xplagiax_reverse_search_config"] = config.reverse_search

    app.extensions["xplagiax_security_config"] = config.security
    app.config["MAX_CONTENT_LENGTH"] = config.security.max_image_bytes

    from app.reverse_search.routes import reverse_search_bp
    app.register_blueprint(reverse_search_bp)

    @app.route("/healthz")
    def liveness():
        return jsonify({"status": "alive"}), 200

    @app.route("/readyz")
    def readiness():
        ready = orchestrator is not None
        return jsonify({
            "status": "ready" if ready else "not_ready",
            "checks": {"reverse_search": ready, "redis": cache.available},
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
        endpoints=["/healthz", "/readyz", "/api/v1/reverse-image-search"],
    )
    return app
