"""
Async indexing service.

Heavy operations (CLIP + SigLIP inference) run in a dedicated RQ worker
process so Flask threads are never blocked by ML inference.

Architecture:
  Flask API       → validates, stores raw bytes, enqueues job, returns 202
  RQ Worker       → dequeues job, runs ML, upserts to Qdrant, updates job status
  Redis           → job queue + result storage

For simple deployments without Redis, we fall back to synchronous processing
(degraded mode: higher latency, but functional).

Job status lifecycle:
  queued → processing → done | failed
"""

from __future__ import annotations

import hashlib
import io
import time
import traceback
import uuid
from dataclasses import asdict, dataclass
from typing import Optional

from PIL import Image

from app.observability.telemetry import get_logger, get_metrics

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Job data structures
# ---------------------------------------------------------------------------

@dataclass
class IndexJobRequest:
    job_id: str
    filename: str
    group_id: str
    content_hash: str           # pre-computed before enqueue
    size_bytes: int
    page: Optional[int]
    run_ai_detection: bool = True
    extra_metadata: dict = None


@dataclass
class IndexJobResult:
    job_id: str
    status: str                 # "done" | "failed"
    point_id: Optional[str]
    content_hash: str
    filename: str
    group_id: str
    ai_detection: Optional[dict]
    duration_ms: float
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Job service
# ---------------------------------------------------------------------------

class IndexingService:
    """
    Orchestrates image indexing — either async (RQ) or sync (fallback).
    Injected with all dependencies to avoid global state.
    """

    def __init__(
        self,
        model_registry,
        vector_repository,
        image_storage,
        cache,
        redis_client=None,   # raw Redis connection for RQ queue
        async_enabled: bool = True,
    ) -> None:
        self._models = model_registry
        self._repo = vector_repository
        self._storage = image_storage
        self._cache = cache
        self._async_enabled = async_enabled and redis_client is not None
        self._queue = None

        if self._async_enabled:
            try:
                from rq import Queue as RQQueue
                self._queue = RQQueue("indexing", connection=redis_client)
                logger.info("async_indexing_enabled")
            except ImportError:
                logger.warning(
                    "rq_not_installed",
                    hint="pip install rq  to enable async indexing",
                )
                self._async_enabled = False

        if not self._async_enabled:
            logger.info(
                "sync_indexing_mode",
                reason="Redis or RQ not available — requests will block on ML inference",
            )

    # ------------------------------------------------------------------
    # Enqueue or process
    # ------------------------------------------------------------------

    def submit(
        self,
        image_bytes: bytes,
        pil_image: Image.Image,
        filename: str,
        group_id: str,
        page: Optional[int] = None,
        run_ai_detection: bool = True,
        extra_metadata: Optional[dict] = None,
    ) -> dict:
        """
        Submit an image for indexing.

        Async mode: saves bytes to storage, enqueues job, returns 202 payload.
        Sync mode: processes immediately, returns 200 payload with full result.
        """
        content_hash = hashlib.sha256(image_bytes).hexdigest()
        job_id = str(uuid.uuid4())

        # Check if this exact image is already indexed (idempotency)
        expected_point_id = self._repo.deterministic_id(content_hash)
        existing = self._repo.get_by_id(expected_point_id)
        if existing:
            logger.info(
                "image_already_indexed",
                content_hash=content_hash[:16],
                point_id=expected_point_id,
            )
            return {
                "job_id": job_id,
                "status": "already_indexed",
                "point_id": expected_point_id,
                "content_hash": content_hash,
                "duplicate": True,
            }

        # Save to persistent storage first (decoupled from ML)
        storage_key = self._storage.save(
            image_bytes=image_bytes,
            content_hash=content_hash,
            group_id=group_id,
            filename=filename,
            mime_type=pil_image.format.lower() if pil_image.format else "jpeg",
        )

        if self._async_enabled:
            return self._enqueue(
                job_id=job_id,
                storage_key=storage_key,
                filename=filename,
                group_id=group_id,
                content_hash=content_hash,
                size_bytes=len(image_bytes),
                width=pil_image.width,
                height=pil_image.height,
                mime_type=pil_image.format.lower() if pil_image.format else "jpeg",
                page=page,
                run_ai_detection=run_ai_detection,
                extra_metadata=extra_metadata or {},
            )
        else:
            return self._process_sync(
                job_id=job_id,
                pil_image=pil_image,
                image_bytes=image_bytes,
                storage_key=storage_key,
                filename=filename,
                group_id=group_id,
                content_hash=content_hash,
                size_bytes=len(image_bytes),
                page=page,
                run_ai_detection=run_ai_detection,
                extra_metadata=extra_metadata or {},
            )

    def _enqueue(self, job_id: str, **job_kwargs) -> dict:
        """Store initial job status and push to RQ."""
        initial_status = {
            "job_id":     job_id,
            "status":     "queued",
            "created_at": time.time(),
            **{k: v for k, v in job_kwargs.items() if k not in ("image_bytes",)},
        }
        self._cache.set_job(job_id, initial_status)
        get_metrics().jobs_enqueued.labels(job_type="index").inc()

        self._queue.enqueue(
            "app.workers.ml_worker.process_index_job",
            kwargs={
                "job_id": job_id,
                **job_kwargs,
            },
            job_id=job_id,   
            job_timeout=120,
            result_ttl=3600,
            failure_ttl=3600,
        )

        return {
            "job_id":     job_id,
            "status":     "queued",
            "poll_url":   f"/api/v1/jobs/{job_id}",
            "duplicate":  False,
        }

    def _process_sync(
        self,
        job_id: str,
        pil_image: Image.Image,
        image_bytes: bytes,
        storage_key: str,
        filename: str,
        group_id: str,
        content_hash: str,
        size_bytes: int,
        page: Optional[int],
        run_ai_detection: bool,
        extra_metadata: dict,
    ) -> dict:
        """Synchronous processing — blocks the Flask thread."""
        start = time.perf_counter()
        from app.storage.vector_repository import ImagePoint

        try:
            # CLIP embedding
            embed_result = self._models.embed_single(pil_image)

            # AI detection (optional)
            ai_result = None
            if run_ai_detection and self._models.siglip_ready:
                try:
                    cls = self._models.classify_single(pil_image)
                    ai_result = {
                        "is_ai":        cls.is_ai,
                        "is_human":     cls.is_human,
                        "label":        cls.label,
                        "confidence":   cls.confidence,
                        "ai_score":     cls.ai_score,
                        "human_score":  cls.human_score,
                    }
                except Exception as exc:
                    logger.warning("siglip_failed_degraded", error=str(exc))

            # Build and upsert point
            point = ImagePoint(
                vector=embed_result.vector,
                filename=filename,
                group_id=group_id,
                content_hash=content_hash,
                size_bytes=size_bytes,
                width=pil_image.width,
                height=pil_image.height,
                mime_type=pil_image.format.lower() if pil_image.format else "jpeg",
                page=page,
                is_ai=ai_result["is_ai"] if ai_result else None,
                is_human=ai_result["is_human"] if ai_result else None,
                ai_confidence=ai_result["confidence"] if ai_result else None,
                ai_label=ai_result["label"] if ai_result else None,
                ai_score=ai_result["ai_score"] if ai_result else None,
                human_score=ai_result["human_score"] if ai_result else None,
                storage_backend=self._storage.backend_name(),
                storage_key=storage_key,
                extra=extra_metadata,
            )
            point_id = self._repo.upsert(point)

            # Cache the embedding for future searches of this same image
            self._cache.set_embedding(image_bytes, embed_result.vector)

            elapsed_ms = (time.perf_counter() - start) * 1000
            get_metrics().jobs_completed.labels(job_type="index", status="done").inc()
            get_metrics().job_duration.labels(job_type="index").observe(elapsed_ms / 1000)

            return {
                "job_id":       job_id,
                "status":       "done",
                "point_id":     point_id,
                "content_hash": content_hash,
                "filename":     filename,
                "group_id":     group_id,
                "ai_detection": ai_result,
                "duration_ms":  round(elapsed_ms, 1),
                "duplicate":    False,
            }

        except Exception as exc:
            elapsed_ms = (time.perf_counter() - start) * 1000
            get_metrics().jobs_completed.labels(job_type="index", status="failed").inc()
            logger.error("indexing_failed", error=str(exc), exc_info=True)
            raise

    def get_job_status(self, job_id: str) -> Optional[dict]:
        """Check job status. Works for both async and sync modes."""
        return self._cache.get_job(job_id)
