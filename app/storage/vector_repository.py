"""
Qdrant storage layer.

Responsibilities:
  - Connection management with retry on startup
  - Idempotent collection creation (with payload indexes)
  - Typed upsert / search / delete operations
  - HNSW parameter management
  - Prometheus instrumentation
  - Deterministic point IDs from image content hash (idempotent indexing)
"""

from __future__ import annotations

import hashlib
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from qdrant_client.http.exceptions import UnexpectedResponse

from app.observability.telemetry import get_logger, get_metrics

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# DTOs
# ---------------------------------------------------------------------------

@dataclass
class ImagePoint:
    """Represents a vector + metadata record in Qdrant."""
    vector: list[float]
    filename: str
    group_id: str
    content_hash: str           # SHA-256 of original image bytes
    size_bytes: int
    width: int
    height: int
    mime_type: str
    page: Optional[int] = None
    # AI detection metadata (optional — set if SigLIP ran)
    is_ai: Optional[bool] = None
    is_human: Optional[bool] = None
    ai_confidence: Optional[float] = None
    ai_label: Optional[str] = None
    ai_score: Optional[float] = None
    human_score: Optional[float] = None
    # Object storage reference (optional — set if using S3/MinIO)
    storage_backend: str = "local"
    storage_key: Optional[str] = None     # S3 key or local relative path
    # Extra application metadata
    extra: dict = field(default_factory=dict)


@dataclass
class SearchResult:
    point_id: str
    score: float
    filename: str
    group_id: str
    page: Optional[int]
    is_ai: Optional[bool]
    content_hash: Optional[str]
    storage_key: Optional[str]
    storage_backend: str
    payload: dict


@dataclass
class CollectionStats:
    total_vectors: int
    indexed_vectors_count: int
    collection_name: str
    status: str


# ---------------------------------------------------------------------------
# Repository
# ---------------------------------------------------------------------------

class VectorRepository:
    """
    All Qdrant interactions go through this class.
    Business logic lives in the service layer, not here.
    """

    EMBEDDING_DIM = 512   # default; overridable per-instance via embedding_dim

    def __init__(
        self,
        host: str,
        port: int,
        collection: str,
        api_key: Optional[str],
        embedding_dim: int = 512,
        hnsw_m: int = 16,
        hnsw_ef_construct: int = 200,
        hnsw_ef_search: int = 128,
        full_scan_threshold: int = 10_000,
    ) -> None:
        self._collection = collection
        self._embedding_dim = embedding_dim
        self._hnsw_ef_search = hnsw_ef_search

        if host == ":memory:":
            self._client = QdrantClient(
                location=":memory:",
            )
        else:
            self._client = QdrantClient(
                host=host,
                port=port,
                api_key=api_key,
                timeout=10.0,
            )

        self._ensure_collection(hnsw_m, hnsw_ef_construct, full_scan_threshold)
        self._ensure_payload_indexes()
        logger.info(
            "qdrant_ready",
            host=host,
            port=port,
            collection=collection,
        )

    # ------------------------------------------------------------------
    # Collection setup
    # ------------------------------------------------------------------

    def _ensure_collection(
        self,
        hnsw_m: int,
        hnsw_ef_construct: int,
        full_scan_threshold: int,
    ) -> None:
        """Create collection with optimal HNSW config if it doesn't exist."""
        if self._client.collection_exists(collection_name=self._collection):
            logger.info("qdrant_collection_exists", collection=self._collection)
            return

        self._client.create_collection(
            collection_name=self._collection,
            vectors_config=qmodels.VectorParams(
                size=self._embedding_dim,
                distance=qmodels.Distance.COSINE,
                hnsw_config=qmodels.HnswConfigDiff(
                    m=hnsw_m,
                    ef_construct=hnsw_ef_construct,
                    full_scan_threshold=full_scan_threshold,
                    on_disk=False,   # keep in RAM for speed
                ),
            ),
        )
        logger.info(
            "qdrant_collection_created",
            collection=self._collection,
            hnsw_m=hnsw_m,
            ef_construct=hnsw_ef_construct,
        )

    def _ensure_payload_indexes(self) -> None:
        """
        Create payload indexes for fields used in filters.
        Safe to call multiple times — Qdrant is idempotent for existing indexes.
        """
        indexes = [
            ("group_id",        qmodels.PayloadSchemaType.KEYWORD),
            ("is_ai",           qmodels.PayloadSchemaType.BOOL),
            ("content_hash",    qmodels.PayloadSchemaType.KEYWORD),
            ("mime_type",       qmodels.PayloadSchemaType.KEYWORD),
            ("storage_backend", qmodels.PayloadSchemaType.KEYWORD),
        ]
        for field_name, schema_type in indexes:
            try:
                self._client.create_payload_index(
                    collection_name=self._collection,
                    field_name=field_name,
                    field_schema=schema_type,
                )
            except UnexpectedResponse:
                pass  # Already exists — expected after first run
        logger.info("qdrant_payload_indexes_ready", collection=self._collection)

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    @staticmethod
    def deterministic_id(content_hash: str, group_id: str = "default") -> str:
        """
        Generate a deterministic UUID from content hash and group_id.
        Same image bytes in the same group always produce the same point ID.
        Qdrant point IDs must be valid UUIDs or unsigned integers.
        """
        # Use UUID5 (SHA-1 namespace-based) for determinism
        return str(uuid.uuid5(uuid.NAMESPACE_OID, f"{content_hash}:{group_id}"))

    def upsert(self, point: ImagePoint) -> str:
        """
        Insert or update a point. Returns the point ID.
        Idempotent: same content_hash + group_id always maps to same ID.
        """
        point_id = self.deterministic_id(point.content_hash, point.group_id)

        payload = {
            "filename":        point.filename,
            "group_id":        point.group_id,
            "content_hash":    point.content_hash,
            "size_bytes":      point.size_bytes,
            "width":           point.width,
            "height":          point.height,
            "mime_type":       point.mime_type,
            "storage_backend": point.storage_backend,
            "storage_key":     point.storage_key,
            "page":            point.page,
        }
        if point.is_ai is not None:
            payload.update({
                "is_ai":          point.is_ai,
                "is_human":       point.is_human,
                "ai_confidence":  point.ai_confidence,
                "ai_label":       point.ai_label,
                "ai_score":       point.ai_score,
                "human_score":    point.human_score,
            })
        if point.extra:
            payload["extra"] = point.extra

        with get_metrics().timed(
            get_metrics().qdrant_operation_duration, operation="upsert"
        ):
            self._client.upsert(
                collection_name=self._collection,
                points=[
                    qmodels.PointStruct(
                        id=point_id,
                        vector=point.vector,
                        payload=payload,
                    )
                ],
                wait=True,   # wait for indexing — ensures immediate searchability
            )

        logger.info(
            "vector_upserted",
            point_id=point_id,
            group_id=point.group_id,
            filename=point.filename,
            content_hash=point.content_hash[:16],
        )
        return point_id

    def upsert_batch(self, points: list[ImagePoint]) -> list[str]:
        """Batch upsert — more efficient than calling upsert() in a loop."""
        if not points:
            return []

        structs = []
        ids = []
        for p in points:
            point_id = self.deterministic_id(p.content_hash, p.group_id)
            ids.append(point_id)
            payload = {
                "filename":        p.filename,
                "group_id":        p.group_id,
                "content_hash":    p.content_hash,
                "size_bytes":      p.size_bytes,
                "width":           p.width,
                "height":          p.height,
                "mime_type":       p.mime_type,
                "storage_backend": p.storage_backend,
                "storage_key":     p.storage_key,
                "page":            p.page,
            }
            if p.is_ai is not None:
                payload.update({
                    "is_ai":         p.is_ai,
                    "is_human":      p.is_human,
                    "ai_confidence": p.ai_confidence,
                    "ai_label":      p.ai_label,
                })
            structs.append(
                qmodels.PointStruct(id=point_id, vector=p.vector, payload=payload)
            )

        with get_metrics().timed(
            get_metrics().qdrant_operation_duration, operation="upsert_batch"
        ):
            self._client.upsert(
                collection_name=self._collection,
                points=structs,
                wait=True,
            )

        logger.info("batch_upserted", count=len(points))
        return ids

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def search(
        self,
        query_vector: list[float],
        limit: int = 10,
        score_threshold: float = 0.0,
        group_id: Optional[str] = None,
        exclude_content_hash: Optional[str] = None,
    ) -> list[SearchResult]:
        """
        ANN search with optional group_id filter and self-exclusion.
        Uses configured ef_search for recall/speed trade-off.
        """
        must_conditions = []
        must_not_conditions = []

        if group_id:
            must_conditions.append(
                qmodels.FieldCondition(
                    key="group_id",
                    match=qmodels.MatchValue(value=group_id),
                )
            )
        if exclude_content_hash:
            must_not_conditions.append(
                qmodels.FieldCondition(
                    key="content_hash",
                    match=qmodels.MatchValue(value=exclude_content_hash),
                )
            )

        search_filter = None
        if must_conditions or must_not_conditions:
            search_filter = qmodels.Filter(
                must=must_conditions or None,
                must_not=must_not_conditions or None,
            )

        with get_metrics().timed(
            get_metrics().qdrant_operation_duration, operation="search"
        ):
            hits = self._client.search(
                collection_name=self._collection,
                query_vector=query_vector,
                query_filter=search_filter,
                limit=limit,
                score_threshold=score_threshold,
                search_params=qmodels.SearchParams(
                    hnsw_ef=self._hnsw_ef_search,
                    exact=False,
                ),
                with_payload=True,
            )

        m = get_metrics()
        results = []
        for hit in hits:
            m.similarity_score.observe(hit.score)
            payload = hit.payload or {}
            results.append(
                SearchResult(
                    point_id=str(hit.id),
                    score=hit.score,
                    filename=payload.get("filename", "unknown"),
                    group_id=payload.get("group_id", ""),
                    page=payload.get("page"),
                    is_ai=payload.get("is_ai"),
                    content_hash=payload.get("content_hash"),
                    storage_key=payload.get("storage_key"),
                    storage_backend=payload.get("storage_backend", "local"),
                    payload=payload,
                )
            )

        logger.info(
            "search_completed",
            results=len(results),
            threshold=score_threshold,
            group_id=group_id,
        )
        return results

    def get_by_id(self, point_id: str) -> Optional[dict]:
        """Retrieve a single point's payload by ID."""
        with get_metrics().timed(
            get_metrics().qdrant_operation_duration, operation="retrieve"
        ):
            points = self._client.retrieve(
                collection_name=self._collection,
                ids=[point_id],
                with_payload=True,
                with_vectors=False,
            )
        return points[0].payload if points else None

    def scroll_all(
        self,
        limit: int = 100,
        offset: Optional[str] = None,
        group_id: Optional[str] = None,
    ) -> tuple[list[dict], Optional[str]]:
        """
        Paginated scroll. Returns (items, next_offset).
        next_offset is None when there are no more results.
        """
        search_filter = None
        if group_id:
            search_filter = qmodels.Filter(
                must=[
                    qmodels.FieldCondition(
                        key="group_id",
                        match=qmodels.MatchValue(value=group_id),
                    )
                ]
            )

        with get_metrics().timed(
            get_metrics().qdrant_operation_duration, operation="scroll"
        ):
            records, next_page_offset = self._client.scroll(
                collection_name=self._collection,
                scroll_filter=search_filter,
                limit=min(limit, 1000),  # hard cap per page
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )

        items = []
        for rec in records:
            item = {"id": str(rec.id)}
            if rec.payload:
                item.update(rec.payload)
            items.append(item)

        return items, str(next_page_offset) if next_page_offset else None

    # ------------------------------------------------------------------
    # Delete
    # ------------------------------------------------------------------

    def delete_by_id(self, point_id: str) -> None:
        with get_metrics().timed(
            get_metrics().qdrant_operation_duration, operation="delete"
        ):
            self._client.delete(
                collection_name=self._collection,
                points_selector=qmodels.PointIdsList(points=[point_id]),
            )
        logger.info("point_deleted", point_id=point_id)

    def delete_by_group(self, group_id: str) -> None:
        with get_metrics().timed(
            get_metrics().qdrant_operation_duration, operation="delete_by_group"
        ):
            self._client.delete(
                collection_name=self._collection,
                points_selector=qmodels.FilterSelector(
                    filter=qmodels.Filter(
                        must=[
                            qmodels.FieldCondition(
                                key="group_id",
                                match=qmodels.MatchValue(value=group_id),
                            )
                        ]
                    )
                ),
            )
        logger.info("group_deleted", group_id=group_id)

    def delete_by_content_hash(self, content_hash: str) -> None:
        """Used for de-duplication or targeted removal."""
        self._client.delete(
            collection_name=self._collection,
            points_selector=qmodels.FilterSelector(
                filter=qmodels.Filter(
                    must=[
                        qmodels.FieldCondition(
                            key="content_hash",
                            match=qmodels.MatchValue(value=content_hash),
                        )
                    ]
                )
            ),
        )

    # ------------------------------------------------------------------
    # Admin / health
    # ------------------------------------------------------------------

    def stats(self) -> CollectionStats:
        with get_metrics().timed(
            get_metrics().qdrant_operation_duration, operation="get_collection"
        ):
            info = self._client.get_collection(collection_name=self._collection)

        total = info.points_count or 0
        get_metrics().qdrant_collection_size.set(total)

        return CollectionStats(
            total_vectors=total,
            indexed_vectors_count=info.indexed_vectors_count or 0,
            collection_name=self._collection,
            status=str(info.status),
        )

    def health_check(self) -> dict:
        """Perform a lightweight check: can we issue a search?"""
        try:
            dummy = [0.0] * self._embedding_dim
            dummy[0] = 1.0
            self._client.search(
                collection_name=self._collection,
                query_vector=dummy,
                limit=1,
                search_params=qmodels.SearchParams(hnsw_ef=1),
            )
            return {"status": "ok"}
        except Exception as exc:
            return {"status": "error", "error": str(exc)}

    def reset_collection(
        self,
        hnsw_m: int = 16,
        hnsw_ef_construct: int = 200,
        full_scan_threshold: int = 10_000,
    ) -> None:
        """
        Drop and recreate collection. DESTRUCTIVE.
        Requires an explicit confirmation token at the API layer.
        """
        self._client.delete_collection(collection_name=self._collection)
        self._ensure_collection(hnsw_m, hnsw_ef_construct, full_scan_threshold)
        self._ensure_payload_indexes()
        logger.warning("collection_reset", collection=self._collection)
