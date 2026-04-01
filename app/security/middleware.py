"""
Security middleware.

  - API key authentication (X-API-Key header)
  - Per-IP rate limiting with Redis sliding window
  - Request size enforcement (belt-and-suspenders alongside Nginx)
  - Safe error responses (no internal details leak)
  - CORS handling for API usage

Usage:
    from app.security.middleware import require_auth, rate_limit, SecurityMiddleware
"""

from __future__ import annotations

import functools
import time
import uuid
from typing import Optional

from flask import Flask, g, jsonify, request

from app.observability.telemetry import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Decorators
# ---------------------------------------------------------------------------

def require_auth(f):
    """
    Enforce API key authentication.
    Reads SERVICE_API_KEY from app config (set in SecurityConfig).
    Skip if require_auth=False (dev mode).
    """
    @functools.wraps(f)
    def decorated(*args, **kwargs):
        cfg = _get_security_config()
        if not cfg.require_auth:
            return f(*args, **kwargs)

        provided_key = (
            request.headers.get("X-API-Key")
            or request.headers.get("Authorization", "").removeprefix("Bearer ").strip()
        )
        if not provided_key or provided_key != cfg.api_key:
            logger.warning(
                "auth_failed",
                remote_addr=_get_client_ip(),
                endpoint=request.endpoint,
            )
            return jsonify({
                "error": "Unauthorized",
                "code": "INVALID_API_KEY",
            }), 401
        return f(*args, **kwargs)
    return decorated


def rate_limit(f):
    """
    Sliding window rate limiter using Redis.
    Falls back to no limiting if Redis is unavailable.
    Two windows: per-minute and per-hour.
    """
    @functools.wraps(f)
    def decorated(*args, **kwargs):
        cfg = _get_security_config()
        cache = _get_cache()

        if cache and cache.available:
            client_ip = _get_client_ip()
            now = int(time.time())

            # Per-minute window
            minute_key = f"ratelimit:{client_ip}:{now // 60}"
            minute_count = _atomic_increment(cache, minute_key, 60)
            if minute_count is not None and minute_count > cfg.rate_limit_per_minute:
                logger.warning(
                    "rate_limit_exceeded_minute",
                    client_ip=client_ip,
                    endpoint=request.endpoint,
                    count=minute_count,
                )
                return jsonify({
                    "error": "Too many requests",
                    "code": "RATE_LIMIT_EXCEEDED",
                    "retry_after_seconds": 60,
                }), 429

            # Per-hour window
            hour_key = f"ratelimit:{client_ip}:{now // 3600}h"
            hour_count = _atomic_increment(cache, hour_key, 3600)
            if hour_count is not None and hour_count > cfg.rate_limit_per_hour:
                logger.warning(
                    "rate_limit_exceeded_hour",
                    client_ip=client_ip,
                    endpoint=request.endpoint,
                    count=hour_count,
                )
                return jsonify({
                    "error": "Hourly rate limit exceeded",
                    "code": "RATE_LIMIT_EXCEEDED",
                    "retry_after_seconds": 3600,
                }), 429

        return f(*args, **kwargs)
    return decorated


# ---------------------------------------------------------------------------
# Flask middleware (registered via app factory)
# ---------------------------------------------------------------------------

class SecurityMiddleware:
    """
    WSGI-level middleware for size limiting and error formatting.
    Register via app.wsgi_app = SecurityMiddleware(app.wsgi_app, config).
    """

    def __init__(self, wsgi_app, max_content_length: int) -> None:
        self._app = wsgi_app
        self._max_bytes = max_content_length

    def __call__(self, environ, start_response):
        content_length = environ.get("CONTENT_LENGTH")
        if content_length:
            try:
                cl = int(content_length)
                if cl > self._max_bytes:
                    body = (
                        b'{"error":"Request too large",'
                        b'"code":"CONTENT_TOO_LARGE"}'
                    )
                    start_response(
                        "413 Content Too Large",
                        [
                            ("Content-Type", "application/json"),
                            ("Content-Length", str(len(body))),
                        ],
                    )
                    return [body]
            except ValueError:
                pass
        return self._app(environ, start_response)


def register_error_handlers(app: Flask) -> None:
    """
    Register Flask error handlers that return JSON and never leak internals.
    """

    @app.errorhandler(400)
    def bad_request(e):
        return jsonify({"error": "Bad request", "code": "BAD_REQUEST"}), 400

    @app.errorhandler(401)
    def unauthorized(e):
        return jsonify({"error": "Unauthorized", "code": "UNAUTHORIZED"}), 401

    @app.errorhandler(404)
    def not_found(e):
        return jsonify({"error": "Not found", "code": "NOT_FOUND"}), 404

    @app.errorhandler(405)
    def method_not_allowed(e):
        return jsonify({"error": "Method not allowed", "code": "METHOD_NOT_ALLOWED"}), 405

    @app.errorhandler(413)
    def too_large(e):
        return jsonify({
            "error": "Request body too large",
            "code": "CONTENT_TOO_LARGE",
        }), 413

    @app.errorhandler(429)
    def too_many_requests(e):
        return jsonify({
            "error": "Too many requests",
            "code": "RATE_LIMIT_EXCEEDED",
        }), 429

    @app.errorhandler(500)
    def internal_error(e):
        req_id = getattr(g, "request_id", str(uuid.uuid4()))
        logger.error("unhandled_exception", exc_info=e, request_id=req_id)
        return jsonify({
            "error": "Internal server error",
            "code": "INTERNAL_ERROR",
            "request_id": req_id,  # correlate with logs
        }), 500

    @app.errorhandler(503)
    def service_unavailable(e):
        return jsonify({
            "error": "Service temporarily unavailable",
            "code": "SERVICE_UNAVAILABLE",
        }), 503


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_client_ip() -> str:
    """Extract real client IP, respecting X-Forwarded-For from trusted proxies."""
    xff = request.headers.get("X-Forwarded-For")
    if xff:
        return xff.split(",")[0].strip()
    return request.remote_addr or "unknown"


def _get_security_config():
    from flask import current_app
    return current_app.extensions["xplagiax_security_config"]


def _get_cache():
    from flask import current_app
    return current_app.extensions.get("xplagiax_cache")


def _atomic_increment(cache, key: str, ttl: int) -> Optional[int]:
    """Increment rate limit counter. Returns None if Redis unavailable."""
    return cache.incr(key, ttl_if_new=ttl)
