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
import hmac
import ipaddress
import threading
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
        if not provided_key or not hmac.compare_digest(provided_key, cfg.api_key):
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


def require_admin(f):
    """
    Enforce a SEPARATE admin key for destructive operations.
    Uses ADMIN_API_KEY if set, otherwise falls back to SERVICE_API_KEY.
    Always enforced when require_auth is on (no anonymous admin).
    """
    @functools.wraps(f)
    def decorated(*args, **kwargs):
        cfg = _get_security_config()
        if not cfg.require_auth:
            return f(*args, **kwargs)

        expected = cfg.admin_api_key or cfg.api_key
        provided_key = (
            request.headers.get("X-Admin-Key")
            or request.headers.get("X-API-Key")
            or request.headers.get("Authorization", "").removeprefix("Bearer ").strip()
        )
        if not expected or not provided_key or not hmac.compare_digest(provided_key, expected):
            logger.warning(
                "admin_auth_failed",
                remote_addr=_get_client_ip(),
                endpoint=request.endpoint,
            )
            return jsonify({"error": "Admin authorization required", "code": "ADMIN_FORBIDDEN"}), 403
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
        client_ip = _get_client_ip()
        now = int(time.time())

        if cache and cache.available:
            # Distributed sliding window via Redis (atomic).
            minute_count = _atomic_increment(cache, f"ratelimit:{client_ip}:{now // 60}", 60)
            if minute_count is not None and minute_count > cfg.rate_limit_per_minute:
                return _rate_limited(client_ip, "minute", 60)
            hour_count = _atomic_increment(cache, f"ratelimit:{client_ip}:{now // 3600}h", 3600)
            if hour_count is not None and hour_count > cfg.rate_limit_per_hour:
                return _rate_limited(client_ip, "hour", 3600)
            return f(*args, **kwargs)

        # Redis down → degrade to a PER-PROCESS limiter (do NOT silently bypass).
        if _LOCAL_LIMITER.hit(f"{client_ip}:m:{now // 60}", cfg.rate_limit_per_minute, 60):
            logger.warning("rate_limit_local_fallback_minute", client_ip=client_ip)
            return _rate_limited(client_ip, "minute(local)", 60)
        if _LOCAL_LIMITER.hit(f"{client_ip}:h:{now // 3600}", cfg.rate_limit_per_hour, 3600):
            return _rate_limited(client_ip, "hour(local)", 3600)
        return f(*args, **kwargs)
    return decorated


def _rate_limited(client_ip: str, window: str, retry_after: int):
    logger.warning("rate_limit_exceeded", client_ip=client_ip, window=window,
                   endpoint=request.endpoint)
    return jsonify({
        "error": "Too many requests",
        "code": "RATE_LIMIT_EXCEEDED",
        "retry_after_seconds": retry_after,
    }), 429


class _LocalRateLimiter:
    """Tiny in-process fixed-window limiter used only when Redis is unavailable."""

    def __init__(self) -> None:
        self._counters: dict[str, tuple[int, float]] = {}
        self._lock = threading.Lock()

    def hit(self, key: str, limit: int, window_s: int) -> bool:
        """Increment; return True if the request EXCEEDS the limit."""
        now = time.time()
        with self._lock:
            if len(self._counters) > 100_000:        # crude memory cap
                self._counters = {k: v for k, v in self._counters.items() if v[1] > now}
            count, expiry = self._counters.get(key, (0, now + window_s))
            if expiry <= now:
                count, expiry = 0, now + window_s
            count += 1
            self._counters[key] = (count, expiry)
            return count > limit


_LOCAL_LIMITER = _LocalRateLimiter()


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
_TRUSTED_NETS_CACHE: dict = {}


def _trusted_networks():
    """Parse and cache the configured trusted-proxy CIDRs as ip_network objects."""
    cfg = _get_security_config()
    raw = getattr(cfg, "trusted_proxies", ()) or ()
    key = tuple(raw)
    cached = _TRUSTED_NETS_CACHE.get("key"), _TRUSTED_NETS_CACHE.get("nets")
    if cached[0] == key:
        return cached[1]
    nets = []
    for entry in raw:
        try:
            nets.append(ipaddress.ip_network(entry, strict=False))
        except ValueError:
            logger.warning("invalid_trusted_proxy_cidr", entry=entry)
    _TRUSTED_NETS_CACHE["key"] = key
    _TRUSTED_NETS_CACHE["nets"] = nets
    return nets


def _is_trusted_proxy(ip_str: str) -> bool:
    try:
        ip = ipaddress.ip_address(ip_str)
    except ValueError:
        return False
    return any(ip in net for net in _trusted_networks())


def _get_client_ip() -> str:
    """
    Real client IP. X-Forwarded-For is honoured ONLY when the direct peer is a
    configured trusted proxy (proper CIDR match). We then take the right-most
    address that is NOT itself a trusted proxy — the true client.
    """
    remote = request.remote_addr or "unknown"
    if not _is_trusted_proxy(remote):
        return remote
    xff = request.headers.get("X-Forwarded-For", "")
    if not xff:
        return remote
    for candidate in reversed([p.strip() for p in xff.split(",") if p.strip()]):
        if not _is_trusted_proxy(candidate):
            return candidate
    return remote


def _get_security_config():
    from flask import current_app
    return current_app.extensions["xplagiax_security_config"]


def _get_cache():
    from flask import current_app
    return current_app.extensions.get("xplagiax_cache")


def _atomic_increment(cache, key: str, ttl: int) -> Optional[int]:
    """Increment rate limit counter. Returns None if Redis unavailable."""
    return cache.incr(key, ttl_if_new=ttl)
