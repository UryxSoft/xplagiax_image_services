"""
Similarity and plagiarism analysis service.

Encapsulates:
  - Embedding extraction with cache
  - Vector search against Qdrant
  - Similarity score classification
  - Plagiarism alert logic
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from typing import Optional

from PIL import Image
import imagehash

from app.observability.telemetry import get_logger, get_metrics

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------

SIMILARITY_THRESHOLDS = {
    "exact_copy":       0.97,   # bit-for-bit or near-identical
    "plagiarism":       0.90,   # modified copy (color, crop, watermark)
    "high_similarity":  0.80,   # related content
    "moderate":         0.70,   # loose relation
}


def classify_similarity(score: float) -> tuple[str, str]:
    """
    Returns (match_type, description) for a given similarity score.
    Using named constants avoids magic numbers scattered in routes.
    """
    if score >= SIMILARITY_THRESHOLDS["exact_copy"]:
        return "EXACT_COPY", "Same image or near-identical"
    if score >= SIMILARITY_THRESHOLDS["plagiarism"]:
        return "MODIFIED_COPY", "Edited, cropped, recoloured, or watermarked"
    if score >= SIMILARITY_THRESHOLDS["high_similarity"]:
        return "HIGH_SIMILARITY", "Visually very related, possible derivative"
    return "MODERATE_SIMILARITY", "Some visual relation — manual review recommended"


# ---------------------------------------------------------------------------
# Search result DTO
# ---------------------------------------------------------------------------

@dataclass
class SimilarityMatch:
    point_id: str
    score: float
    similarity_percent: float
    match_type: str
    description: str
    filename: str
    group_id: str
    page: Optional[int]
    is_ai: Optional[bool]
    content_hash: Optional[str]
    image_url: str
    storage_key: Optional[str]
    storage_backend: str
    metadata: dict


@dataclass
class PlagiarismReport:
    analyzed: bool
    total_matches: int
    threshold: float
    matches: list[SimilarityMatch]
    alert: Optional[str]
    content_hash: str
    duration_ms: float


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------

class SimilarityService:

    def __init__(
        self,
        model_registry,
        vector_repository,
        cache,
    ) -> None:
        self._models = model_registry
        self._repo = vector_repository
        self._cache = cache

    # ------------------------------------------------------------------
    # Embedding with cache
    # ------------------------------------------------------------------

    def get_embedding(
        self,
        image_bytes: bytes,
        pil_image: Image.Image,
        content_hash: Optional[str] = None,
    ) -> list[float]:
        """
        Get CLIP embedding for image bytes.
        Checks cache first; computes and caches on miss.

        `content_hash` lets callers reuse an already-computed SHA-256 so the
        bytes are hashed only ONCE per request (cache.get_embedding handles
        the embedding hit/miss metrics).
        """
        digest = content_hash or hashlib.sha256(image_bytes).hexdigest()

        cached = self._cache.get_embedding(digest=digest)
        if cached is not None:
            return cached

        # Secondary near-duplicate cache via perceptual hash.
        phash = None
        try:
            phash = str(imagehash.phash(pil_image))
            cached_by_phash = self._cache.get(f"embed:phash:{phash}")
            if cached_by_phash is not None:
                get_metrics().cache_hits.labels(cache_type="embedding_phash").inc()
                return cached_by_phash
        except Exception as e:
            logger.warning(f"phash calculation failed: {e}")

        result = self._models.embed_single(pil_image)
        self._cache.set_embedding(digest=digest, vector=result.vector)
        if phash:
            self._cache.set(f"embed:phash:{phash}", result.vector, ttl=86400 * 30)

        return result.vector

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search_similar(
        self,
        image_bytes: bytes,
        pil_image: Image.Image,
        limit: int = 10,
        threshold: float = 0.0,
        group_id: Optional[str] = None,
    ) -> list[SimilarityMatch]:
        vector = self.get_embedding(image_bytes, pil_image)
        results = self._repo.search(
            query_vector=vector,
            limit=limit,
            score_threshold=threshold,
            group_id=group_id,
        )
        return [self._to_match(r) for r in results]

    def analyze_plagiarism(
        self,
        image_bytes: bytes,
        pil_image: Image.Image,
        threshold: float = SIMILARITY_THRESHOLDS["plagiarism"],
        limit: int = 5,
        group_id: Optional[str] = None,
        exclude_self: bool = True,
    ) -> PlagiarismReport:
        """
        Full plagiarism analysis — returns structured report with alert level.

        exclude_self=True: excludes results with the same content_hash
        as the query image (prevents self-matches on re-uploads).
        """
        start = time.perf_counter()
        content_hash = hashlib.sha256(image_bytes).hexdigest()

        # Reuse the hash we just computed — do not re-hash inside get_embedding.
        vector = self.get_embedding(image_bytes, pil_image, content_hash=content_hash)

        exclude_hash = content_hash if exclude_self else None

        results = self._repo.search(
            query_vector=vector,
            limit=limit,
            score_threshold=threshold,
            group_id=group_id,
            exclude_content_hash=exclude_hash,
        )

        matches = [self._to_match(r) for r in results]

        # Determine alert level
        alert = None
        if matches:
            top_score = matches[0].score
            if top_score >= SIMILARITY_THRESHOLDS["exact_copy"]:
                alert = "EXACT_COPY_DETECTED"
            elif top_score >= SIMILARITY_THRESHOLDS["plagiarism"]:
                alert = "MODIFIED_COPY_DETECTED"
            elif top_score >= SIMILARITY_THRESHOLDS["high_similarity"]:
                alert = "HIGH_SIMILARITY_DETECTED"

        elapsed_ms = (time.perf_counter() - start) * 1000

        logger.info(
            "plagiarism_analysis_complete",
            content_hash=content_hash[:16],
            matches=len(matches),
            threshold=threshold,
            top_score=matches[0].score if matches else 0.0,
            elapsed_ms=round(elapsed_ms, 1),
        )

        return PlagiarismReport(
            analyzed=True,
            total_matches=len(matches),
            threshold=threshold,
            matches=matches,
            alert=alert,
            content_hash=content_hash,
            duration_ms=round(elapsed_ms, 1),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _to_match(self, result) -> SimilarityMatch:
        match_type, description = classify_similarity(result.score)
        return SimilarityMatch(
            point_id=result.point_id,
            score=round(result.score, 6),
            similarity_percent=round(result.score * 100, 2),
            match_type=match_type,
            description=description,
            filename=result.filename,
            group_id=result.group_id,
            page=result.page,
            is_ai=result.is_ai,
            content_hash=result.content_hash,
            image_url=f"/api/v1/images/{result.point_id}",
            storage_key=result.storage_key,
            storage_backend=result.storage_backend,
            metadata={
                k: v for k, v in result.payload.items()
                if k not in ("vector",)   # never return raw vectors to clients
            },
        )
