"""
ML Worker — runs as a standalone RQ worker process.

This is the dedicated process that:
  1. Dequeues indexing jobs from Redis
  2. Loads images from storage
  3. Runs CLIP + SigLIP inference
  4. Upserts results to Qdrant
  5. Updates job status in Redis

Launch:
    rq worker indexing --url redis://redis:6379

The worker imports the Flask app factory to get the same DI
context as the web process (same config, same Qdrant collection).
Each worker process loads models once at startup.

IMPORTANT: Run only ONE instance of this worker per GPU.
For CPU-only, multiple workers are fine (each loads its own model copy,
which is acceptable since CPU inference is parallelizable).
"""

from __future__ import annotations

import hashlib
import io
import os
import time
from typing import Optional

from PIL import Image

from app.observability.telemetry import get_logger, get_metrics

logger = get_logger(__name__)

# Module-level singletons — initialised once per worker process
_models = None
_repo = None
_storage = None
_cache = None


def _init_worker():
    """
    Initialise all services in the worker process.
    Called lazily on first job — or explicitly by worker startup hook.
    """
    global _models, _repo, _storage, _cache

    if _models is not None:
        return  # Already initialised

    logger.info("worker_initializing")

    from app.config import load_config
    from app.cache.redis_client import CacheClient
    from app.models.registry import ModelRegistry
    from app.storage.image_storage import create_storage
    from app.storage.vector_repository import VectorRepository
    from app.observability.telemetry import init_metrics

    cfg = load_config()
    init_metrics(cfg.observability.service_name + "-worker")
    configure_logging(cfg.observability.log_level, cfg.observability.log_format)

    _cache = CacheClient(
        host=cfg.redis.host,
        port=cfg.redis.port,
        password=cfg.redis.password,
        db=cfg.redis.db,
        socket_timeout=cfg.redis.socket_timeout,
        embedding_ttl=cfg.redis.embedding_ttl,
        result_ttl=cfg.redis.result_ttl,
        job_ttl=cfg.redis.job_ttl,
    )

    _storage = create_storage(
        backend=cfg.storage.image_backend,
        local_base_path=cfg.storage.local_base_path,
        s3_bucket=cfg.storage.s3_bucket,
        s3_region=cfg.storage.s3_region,
        s3_endpoint_url=cfg.storage.s3_endpoint_url,
        s3_access_key=cfg.storage.s3_access_key,
        s3_secret_key=cfg.storage.s3_secret_key,
    )

    _repo = VectorRepository(
        host=cfg.qdrant.host,
        port=cfg.qdrant.port,
        collection=cfg.qdrant.collection,
        api_key=cfg.qdrant.api_key,
        hnsw_m=cfg.qdrant.hnsw_m,
        hnsw_ef_construct=cfg.qdrant.hnsw_ef_construct,
        hnsw_ef_search=cfg.qdrant.hnsw_ef_search,
    )

    _models = ModelRegistry(
        siglip_model_id=cfg.model.siglip_model_id,
        clip_model_id=cfg.model.clip_model_id,
        device=cfg.model.device,
        max_batch_size=cfg.model.max_batch_size,
    )
    _models.load_all()  # Synchronous in worker — no background thread needed

    logger.info("worker_ready")


def configure_logging(level: str, fmt: str):
    from app.observability.telemetry import configure_logging as _cl
    _cl(log_level=level, log_format=fmt)


# ---------------------------------------------------------------------------
# Job entrypoint — called by RQ
# ---------------------------------------------------------------------------

def process_index_job(
    job_id: str,
    storage_key: str,
    filename: str,
    group_id: str,
    content_hash: str,
    size_bytes: int,
    width: int,
    height: int,
    mime_type: str,
    page: Optional[int] = None,
    run_ai_detection: bool = True,
    extra_metadata: Optional[dict] = None,
    **kwargs,
) -> dict:
    """
    RQ job entrypoint for asynchronous image indexing.

    The Flask API saves the image to storage and enqueues this job.
    This function runs in the RQ worker process.
    """
    _init_worker()
    start = time.perf_counter()

    _cache.update_job(job_id, {"status": "processing", "started_at": time.time()})

    try:
        # Load image from storage
        image_bytes = _storage.load(storage_key)
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # CLIP embedding
        embed_result = _models.embed_single(pil_image)

        # Cache embedding for future requests
        _cache.set_embedding(image_bytes, embed_result.vector)

        # AI detection (optional, degraded gracefully)
        ai_result = None
        if run_ai_detection and _models.siglip_ready:
            try:
                cls = _models.classify_single(pil_image)
                ai_result = {
                    "is_ai":       cls.is_ai,
                    "is_human":    cls.is_human,
                    "label":       cls.label,
                    "confidence":  cls.confidence,
                    "ai_score":    cls.ai_score,
                    "human_score": cls.human_score,
                }
            except Exception as exc:
                logger.warning("worker_siglip_failed", job_id=job_id, error=str(exc))

        # Build and upsert point
        from app.storage.vector_repository import ImagePoint
        point = ImagePoint(
            vector=embed_result.vector,
            filename=filename,
            group_id=group_id,
            content_hash=content_hash,
            size_bytes=size_bytes,
            width=width,
            height=height,
            mime_type=mime_type,
            page=page,
            is_ai=ai_result["is_ai"] if ai_result else None,
            is_human=ai_result["is_human"] if ai_result else None,
            ai_confidence=ai_result["confidence"] if ai_result else None,
            ai_label=ai_result["label"] if ai_result else None,
            ai_score=ai_result["ai_score"] if ai_result else None,
            human_score=ai_result["human_score"] if ai_result else None,
            storage_backend=_storage.backend_name(),
            storage_key=storage_key,
            extra=extra_metadata or {},
        )
        point_id = _repo.upsert(point)

        elapsed_ms = (time.perf_counter() - start) * 1000
        result = {
            "job_id":       job_id,
            "status":       "done",
            "point_id":     point_id,
            "content_hash": content_hash,
            "filename":     filename,
            "group_id":     group_id,
            "ai_detection": ai_result,
            "duration_ms":  round(elapsed_ms, 1),
            "completed_at": time.time(),
        }
        _cache.update_job(job_id, result)

        get_metrics().jobs_completed.labels(job_type="index", status="done").inc()
        get_metrics().job_duration.labels(job_type="index").observe(elapsed_ms / 1000)
        logger.info(
            "worker_job_done",
            job_id=job_id,
            point_id=point_id,
            elapsed_ms=round(elapsed_ms, 1),
        )
        return result

    except Exception as exc:
        elapsed_ms = (time.perf_counter() - start) * 1000
        error_info = {
            "job_id":    job_id,
            "status":    "failed",
            "error":     str(exc),
            "duration_ms": round(elapsed_ms, 1),
            "failed_at": time.time(),
        }
        _cache.update_job(job_id, error_info)
        get_metrics().jobs_completed.labels(job_type="index", status="failed").inc()
        logger.error(
            "worker_job_failed",
            job_id=job_id,
            error=str(exc),
            exc_info=True,
        )
        raise  # RQ marks job as failed; retries if configured
