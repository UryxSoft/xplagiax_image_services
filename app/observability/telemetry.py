"""
Observability — structured logging, Prometheus metrics, optional OpenTelemetry.

Import order matters: configure_logging() must be called BEFORE any other
module creates a logger, so that all loggers inherit the correct handler.
"""

from __future__ import annotations

import logging
import sys
import time
import uuid
from contextlib import contextmanager
from typing import Generator, Optional

import structlog
from prometheus_client import (
    Counter,
    Gauge,
    Histogram,
    Info,
    start_http_server,
    REGISTRY,
)

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def configure_logging(log_level: str = "INFO", log_format: str = "json") -> None:
    """
    Configure structlog for structured JSON logging.
    Call once at app startup before any logger.getLogger() calls.
    """
    shared_processors = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.ExceptionRenderer(),
    ]

    if log_format == "json":
        renderer = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=True)

    structlog.configure(
        processors=shared_processors + [renderer],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, log_level, logging.INFO)
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(file=sys.stdout),
        cache_logger_on_first_use=True,
    )

    # Also configure stdlib logging to route through structlog
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, log_level, logging.INFO),
    )


def get_logger(name: str) -> structlog.BoundLogger:
    return structlog.get_logger(name)


# ---------------------------------------------------------------------------
# Request context helpers
# ---------------------------------------------------------------------------

def bind_request_context(
    request_id: Optional[str] = None,
    endpoint: Optional[str] = None,
    group_id: Optional[str] = None,
) -> None:
    """Bind fields to current log context (structlog contextvars)."""
    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(
        request_id=request_id or str(uuid.uuid4()),
        endpoint=endpoint or "unknown",
        **({"group_id": group_id} if group_id else {}),
    )


def clear_request_context() -> None:
    structlog.contextvars.clear_contextvars()


# ---------------------------------------------------------------------------
# Prometheus metrics — define once, reuse everywhere
# ---------------------------------------------------------------------------

class Metrics:
    """
    Singleton metric registry. Instantiate once in app factory,
    import and use anywhere.
    """

    def __init__(self, service_name: str) -> None:
        labels = ["endpoint"]
        api_labels = ["provider", "operation", "status"]
        model_labels = ["model"]

        self.service_info = Info(
            "service",
            "Service metadata",
        )
        self.service_info.info({"service_name": service_name})

        # HTTP
        self.http_requests_total = Counter(
            "http_requests_total",
            "Total HTTP requests",
            ["endpoint", "method", "status_code"],
        )
        self.http_request_duration = Histogram(
            "http_request_duration_seconds",
            "HTTP request latency",
            labels,
            buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0],
        )

        # ML inference
        self.clip_inference_duration = Histogram(
            "clip_inference_duration_seconds",
            "CLIP embedding extraction latency",
            ["batch_size"],
            buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0],
        )
        self.siglip_inference_duration = Histogram(
            "siglip_inference_duration_seconds",
            "SigLIP classification latency",
            ["batch_size"],
            buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0],
        )
        self.model_loaded = Gauge(
            "model_loaded",
            "Whether a model is loaded and ready",
            model_labels,
        )
        self.siglip_confidence = Histogram(
            "siglip_prediction_confidence",
            "Distribution of SigLIP prediction confidence scores",
            buckets=[0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0],
        )

        # Vector search
        self.qdrant_operation_duration = Histogram(
            "qdrant_operation_duration_seconds",
            "Qdrant operation latency",
            ["operation"],
            buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
        )
        self.qdrant_collection_size = Gauge(
            "qdrant_collection_size_total",
            "Number of vectors in the Qdrant collection",
        )
        self.similarity_score = Histogram(
            "similarity_score_distribution",
            "Distribution of cosine similarity scores returned",
            buckets=[0.5, 0.7, 0.8, 0.85, 0.9, 0.95, 0.98, 1.0],
        )

        # Cache
        self.cache_hits = Counter(
            "cache_hits_total", "Cache hits", ["cache_type"]
        )
        self.cache_misses = Counter(
            "cache_misses_total", "Cache misses", ["cache_type"]
        )
        self.cache_errors = Counter(
            "cache_errors_total", "Cache errors (Redis unavailable)", ["operation"]
        )

        # External API rotator
        self.api_calls_total = Counter(
            "api_rotator_calls_total",
            "External API calls",
            api_labels,
        )
        self.api_usage_remaining = Gauge(
            "api_usage_remaining_searches",
            "Remaining searches for external API this month",
            ["provider"],
        )

        # Jobs (async indexing)
        self.jobs_enqueued = Counter("jobs_enqueued_total", "Async jobs enqueued", ["job_type"])
        self.jobs_completed = Counter("jobs_completed_total", "Async jobs completed", ["job_type", "status"])
        self.job_duration = Histogram(
            "job_duration_seconds",
            "Async job processing time",
            ["job_type"],
            buckets=[0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0],
        )

        # Errors
        self.errors_total = Counter(
            "errors_total",
            "Application errors",
            ["error_type", "endpoint"],
        )

    @contextmanager
    def timed(self, histogram: Histogram, **label_values) -> Generator:
        """Context manager to time a block and record to histogram."""
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            histogram.labels(**label_values).observe(elapsed)

    def start_prometheus_server(self, port: int) -> None:
        """Start Prometheus metrics HTTP server on separate port."""
        start_http_server(port)
        get_logger(__name__).info(
            "prometheus_server_started", port=port
        )


# Module-level singleton — replaced in app factory via init_metrics()
_metrics: Optional[Metrics] = None


def init_metrics(service_name: str) -> Metrics:
    global _metrics
    _metrics = Metrics(service_name)
    return _metrics


def get_metrics() -> Metrics:
    if _metrics is None:
        raise RuntimeError(
            "Metrics not initialised. Call init_metrics() in app factory."
        )
    return _metrics


# ---------------------------------------------------------------------------
# Flask middleware — request instrumentation
# ---------------------------------------------------------------------------

def instrument_flask(app) -> None:
    """Attach before/after request hooks to auto-instrument all endpoints."""
    from flask import g, request as flask_request

    @app.before_request
    def before():
        g.start_time = time.perf_counter()
        g.request_id = flask_request.headers.get("X-Request-ID") or str(uuid.uuid4())
        bind_request_context(
            request_id=g.request_id,
            endpoint=flask_request.endpoint,
            group_id=flask_request.form.get("group_id")
            or (flask_request.get_json(silent=True) or {}).get("group_id"),
        )

    @app.after_request
    def after(response):
        if hasattr(g, "start_time"):
            elapsed = time.perf_counter() - g.start_time
            endpoint = flask_request.endpoint or "unknown"
            m = get_metrics()
            m.http_requests_total.labels(
                endpoint=endpoint,
                method=flask_request.method,
                status_code=str(response.status_code),
            ).inc()
            m.http_request_duration.labels(endpoint=endpoint).observe(elapsed)

        response.headers["X-Request-ID"] = getattr(g, "request_id", "")
        clear_request_context()
        return response

    @app.teardown_request
    def teardown(exc):
        if exc is not None:
            m = get_metrics()
            endpoint = "unknown"
            try:
                from flask import request as r
                endpoint = r.endpoint or "unknown"
            except Exception:
                pass
            m.errors_total.labels(
                error_type=type(exc).__name__, endpoint=endpoint
            ).inc()
