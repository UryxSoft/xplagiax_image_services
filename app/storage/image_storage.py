"""
Object storage abstraction.

Supports:
  - local:             filesystem storage (dev / single-node)
  - seaweedfs_filer:   SeaweedFS via Filer HTTP API  <- PRODUCTION DEFAULT
  - seaweedfs_native:  SeaweedFS via Master + Volume native API

Why SeaweedFS over S3/MinIO:
  - Purpose-built for billions of small files (images, blobs)
  - Single binary — no external metadata DB like MinIO requires
  - Linear cost scaling: data volume not object count drives cost
  - Built-in replication and erasure coding without extra config
  - O(1) reads regardless of cluster size (direct volume lookup)
  - Filer adds a POSIX-like namespace on top of the volume layer

Architecture:
  Master  — coordinates topology, assigns file IDs (fid)
  Volume  — stores actual bytes, addressed by fid
  Filer   — translates paths to fids, provides REST API, enables dirs

storage_key format:
  Both backends use the same key format as LocalImageStorage:
    {group_id}/{hash[:2]}/{sha256_hash}.{ext}
  e.g.:  project_x/ab/abcdef1234...jpg

  Switching between backends is transparent — same keys, different bytes store.

Filer API (recommended — IMAGE_BACKEND=seaweedfs_filer):
  PUT    http://filer:8888/{path}   — upload
  GET    http://filer:8888/{path}   — download
  HEAD   http://filer:8888/{path}   — existence check
  DELETE http://filer:8888/{path}   — delete

Native API (IMAGE_BACKEND=seaweedfs_native):
  POST   http://master:9333/dir/assign   — get fid + volume URL
  PUT    http://volume_url/{fid}         — upload bytes
  GET    http://volume_url/{fid}         — download bytes
  DELETE http://volume_url/{fid}         — delete bytes
  In native mode, storage_key encodes both fid and volume url:
    {volume_public_url}|{fid}
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from app.observability.telemetry import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Shared HTTP session factory
# ---------------------------------------------------------------------------

def _make_session(
    timeout: float = 30.0,
    max_retries: int = 3,
    backoff_factor: float = 0.5,
) -> requests.Session:
    session = requests.Session()
    session._storage_timeout = timeout
    retry = Retry(
        total=max_retries,
        backoff_factor=backoff_factor,
        status_forcelist=[500, 502, 503, 504],
        allowed_methods=["GET", "PUT", "DELETE", "HEAD"],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(
        max_retries=retry,
        pool_connections=10,
        pool_maxsize=20,
    )
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


# ---------------------------------------------------------------------------
# Abstract interface
# ---------------------------------------------------------------------------

class ImageStorage(ABC):

    @abstractmethod
    def save(
        self,
        image_bytes: bytes,
        content_hash: str,
        group_id: str,
        filename: str,
        mime_type: str,
    ) -> str:
        """Persist image bytes. Returns a storage_key for later retrieval."""

    @abstractmethod
    def load(self, storage_key: str) -> bytes:
        """Load raw image bytes from storage."""

    @abstractmethod
    def get_url(self, storage_key: str, expiry_seconds: int = 3600) -> str:
        """Return a URL to directly access the image."""

    @abstractmethod
    def delete(self, storage_key: str) -> None:
        """Remove image from storage."""

    @abstractmethod
    def exists(self, storage_key: str) -> bool:
        """Check if image exists without loading it."""

    @abstractmethod
    def health_check(self) -> dict:
        """Verify the storage backend is reachable."""

    def backend_name(self) -> str:
        return self.__class__.__name__

    @staticmethod
    def _build_path(content_hash: str, group_id: str, mime_type: str) -> str:
        """
        Canonical storage path — identical across all backends.
        Shard by first 2 hash chars to avoid huge flat directories.
        """
        ext = "jpg" if mime_type == "jpeg" else mime_type
        return f"{group_id}/{content_hash[:2]}/{content_hash}.{ext}"


# ---------------------------------------------------------------------------
# Local filesystem backend (dev / single-node)
# ---------------------------------------------------------------------------

class LocalImageStorage(ImageStorage):

    def __init__(self, base_path: str) -> None:
        self._base = Path(base_path)
        self._base.mkdir(parents=True, exist_ok=True)
        logger.info("local_storage_ready", base_path=str(self._base))

    def save(self, image_bytes, content_hash, group_id, filename, mime_type):
        storage_key = self._build_path(content_hash, group_id, mime_type)
        full_path = self._base / storage_key
        full_path.parent.mkdir(parents=True, exist_ok=True)
        if not full_path.exists():
            full_path.write_bytes(image_bytes)
            logger.debug("image_saved_local", storage_key=storage_key)
        return storage_key

    def load(self, storage_key):
        path = self._base / storage_key
        if not path.exists():
            raise FileNotFoundError(f"Image not found locally: {storage_key}")
        return path.read_bytes()

    def get_url(self, storage_key, expiry_seconds=3600):
        return f"/api/v1/images/{storage_key}"

    def delete(self, storage_key):
        path = self._base / storage_key
        if path.exists():
            path.unlink()

    def exists(self, storage_key):
        return (self._base / storage_key).exists()

    def health_check(self):
        try:
            self._base.mkdir(parents=True, exist_ok=True)
            return {"status": "ok", "backend": "local", "path": str(self._base)}
        except Exception as exc:
            return {"status": "error", "error": str(exc)}


# ---------------------------------------------------------------------------
# SeaweedFS Filer backend  (IMAGE_BACKEND=seaweedfs_filer)
# ---------------------------------------------------------------------------

class SeaweedFSFilerStorage(ImageStorage):
    """
    SeaweedFS via Filer HTTP API — recommended production backend.

    The Filer provides a path-addressable REST interface on top of the
    SeaweedFS volume layer. Paths are human-readable and match the same
    key format used by LocalImageStorage, making backend migration trivial.

    Idempotency: storage keys embed the SHA-256 content hash, so the same bytes
    always map to the same immutable key (a re-upload simply overwrites identical
    content). Replication is handled transparently by the volume layer.
    TTL: passed per-request so transient uploads (temp_search) auto-expire even
    if a process crashes before explicit cleanup.

    Port: 8888 (default SeaweedFS filer port)
    """

    _FILER_BASE = "xplagiax/images"

    def __init__(
        self,
        filer_url: str,
        public_url: str,
        replication: str = "000",
        collection: str = "",
        ttl: str = "",
        request_timeout: float = 30.0,
        max_retries: int = 3,
    ) -> None:
        self._filer_url = filer_url.rstrip("/")
        self._public_url = public_url.rstrip("/")
        self._replication = replication
        self._collection = collection
        self._ttl = ttl
        self._session = _make_session(timeout=request_timeout, max_retries=max_retries)
        self._timeout = request_timeout
        logger.info(
            "seaweedfs_filer_storage_ready",
            filer_url=self._filer_url,
            public_url=self._public_url,
            replication=replication,
            collection=collection or "default",
            ttl=ttl or "permanent",
        )

    def _url(self, storage_key: str) -> str:
        return f"{self._filer_url}/{self._FILER_BASE}/{storage_key}"

    def save(self, image_bytes, content_hash, group_id, filename, mime_type):
        storage_key = self._build_path(content_hash, group_id, mime_type)
        url = self._url(storage_key)

        params = {}
        if self._replication:
            params["replication"] = self._replication
        if self._collection:
            params["collection"] = self._collection
        if self._ttl:
            params["ttl"] = self._ttl   # auto-expiry backstop for temp uploads

        start = time.perf_counter()
        resp = self._session.put(
            url,
            data=image_bytes,
            headers={
                "Content-Type":        f"image/{mime_type}",
                "X-Original-Filename": filename,
                "X-Content-Hash":      content_hash,
                "X-Group-ID":          group_id,
            },
            params=params,
            timeout=self._timeout,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000

        if resp.status_code not in (200, 201):
            raise IOError(
                f"SeaweedFS filer upload failed: HTTP {resp.status_code} "
                f"key='{storage_key}' body={resp.text[:200]}"
            )

        logger.info(
            "seaweedfs_filer_uploaded",
            storage_key=storage_key,
            size_bytes=len(image_bytes),
            elapsed_ms=round(elapsed_ms, 1),
        )
        return storage_key

    def load(self, storage_key):
        resp = self._session.get(self._url(storage_key), timeout=self._timeout)
        if resp.status_code == 404:
            raise FileNotFoundError(f"Image not found in SeaweedFS filer: {storage_key}")
        if resp.status_code != 200:
            raise IOError(
                f"SeaweedFS filer GET failed: HTTP {resp.status_code} key='{storage_key}'"
            )
        return resp.content

    def get_url(self, storage_key, expiry_seconds=3600):
        """
        Direct filer URL for the image.
        SeaweedFS filer has no pre-signed URLs (it is not S3).
        For access control, proxy through /api/v1/images/<point_id>.
        Set SEAWEEDFS_PUBLIC_URL to the API service host in that case.
        """
        return f"{self._public_url}/{self._FILER_BASE}/{storage_key}"

    def delete(self, storage_key):
        resp = self._session.delete(self._url(storage_key), timeout=self._timeout)
        if resp.status_code not in (200, 204, 404):
            raise IOError(
                f"SeaweedFS filer DELETE failed: HTTP {resp.status_code} key='{storage_key}'"
            )
        logger.info("seaweedfs_filer_deleted", storage_key=storage_key)

    def exists(self, storage_key):
        try:
            resp = self._session.head(self._url(storage_key), timeout=self._timeout)
            return resp.status_code == 200
        except requests.RequestException:
            return False

    def health_check(self):
        try:
            resp = self._session.get(
                f"{self._filer_url}/",
                timeout=min(self._timeout, 5.0),
            )
            if resp.status_code in (200, 301):
                return {"status": "ok", "backend": "seaweedfs_filer", "filer_url": self._filer_url}
            return {"status": "error", "backend": "seaweedfs_filer", "http_status": resp.status_code}
        except requests.RequestException as exc:
            return {"status": "error", "backend": "seaweedfs_filer", "error": str(exc)}


# ---------------------------------------------------------------------------
# SeaweedFS Native backend  (IMAGE_BACKEND=seaweedfs_native)
# ---------------------------------------------------------------------------

class SeaweedFSNativeStorage(ImageStorage):
    """
    SeaweedFS via Master + Volume native HTTP API (no filer).

    Flow:
      1. POST /dir/assign to master -> get fid + volume URL
      2. PUT to volume URL with fid -> store bytes
      3. storage_key = "{volume_public_url}|{fid}" stored in Qdrant payload

    Trade-offs vs filer:
      + Lower latency (one fewer hop)
      + Maximum throughput
      - No path-based addressing (keys are opaque fid strings)
      - Harder to browse / manage without filer UI

    Master port: 9333
    Volume port: 8080

    storage_key format: "{volume_public_url}|{fid}"
    Example:            "http://seaweedfs-volume:8080|3,01a2b3c4d5"
    """

    def __init__(
        self,
        master_url: str,
        public_volume_url: Optional[str] = None,
        replication: str = "000",
        collection: str = "",
        ttl: str = "",
        request_timeout: float = 30.0,
        max_retries: int = 3,
    ) -> None:
        """
        Args:
            master_url:          Internal master URL (http://seaweedfs-master:9333)
            public_volume_url:   Override volume URL returned by master assign.
                                 Use when volume is behind a reverse proxy.
                                 None = use the URL the master assigns directly.
            replication:         "000" no replication | "001" same rack |
                                 "010" different rack | "100" different DC
            collection:          Logical grouping in SeaweedFS.
            ttl:                 File TTL: "3m","4h","5d","6w","7M","8y" or "".
            request_timeout:     HTTP timeout in seconds.
            max_retries:         Retry on 5xx / transient network errors.
        """
        self._master_url = master_url.rstrip("/")
        self._public_volume_url = public_volume_url
        self._replication = replication
        self._collection = collection
        self._ttl = ttl
        self._session = _make_session(timeout=request_timeout, max_retries=max_retries)
        self._timeout = request_timeout
        logger.info(
            "seaweedfs_native_storage_ready",
            master_url=self._master_url,
            replication=replication,
            ttl=ttl or "permanent",
        )

    def _assign(self) -> tuple[str, str]:
        """Request a new file ID from the master. Returns (fid, volume_url)."""
        params: dict = {}
        if self._replication:
            params["replication"] = self._replication
        if self._collection:
            params["collection"] = self._collection
        if self._ttl:
            params["ttl"] = self._ttl

        resp = self._session.post(
            f"{self._master_url}/dir/assign",
            params=params,
            timeout=self._timeout,
        )
        if resp.status_code != 200:
            raise IOError(
                f"SeaweedFS master /dir/assign failed: HTTP {resp.status_code}. "
                f"Body: {resp.text[:200]}"
            )
        data = resp.json()
        fid = data["fid"]
        volume_url = self._public_volume_url or f"http://{data['url']}"
        return fid, volume_url

    @staticmethod
    def _encode_key(volume_url: str, fid: str) -> str:
        return f"{volume_url}|{fid}"

    @staticmethod
    def _decode_key(storage_key: str) -> tuple[str, str]:
        parts = storage_key.split("|", 1)
        if len(parts) != 2:
            raise ValueError(
                f"Invalid SeaweedFS native storage_key: '{storage_key}'. "
                "Expected: 'http://volume:8080|3,01a2b3c4'"
            )
        return parts[0], parts[1]

    def save(self, image_bytes, content_hash, group_id, filename, mime_type):
        fid, volume_url = self._assign()
        start = time.perf_counter()
        resp = self._session.put(
            f"{volume_url}/{fid}",
            data=image_bytes,
            headers={
                "Content-Type":        f"image/{mime_type}",
                "X-Original-Filename": filename,
                "X-Content-Hash":      content_hash,
                "X-Group-ID":          group_id,
            },
            timeout=self._timeout,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000

        if resp.status_code not in (200, 201):
            raise IOError(
                f"SeaweedFS volume upload failed: HTTP {resp.status_code} "
                f"fid={fid} body={resp.text[:200]}"
            )
        storage_key = self._encode_key(volume_url, fid)
        logger.info(
            "seaweedfs_native_uploaded",
            fid=fid,
            size_bytes=len(image_bytes),
            elapsed_ms=round(elapsed_ms, 1),
        )
        return storage_key

    def load(self, storage_key):
        volume_url, fid = self._decode_key(storage_key)
        resp = self._session.get(f"{volume_url}/{fid}", timeout=self._timeout)
        if resp.status_code == 404:
            raise FileNotFoundError(f"Image not found in SeaweedFS volume: fid={fid}")
        if resp.status_code != 200:
            raise IOError(f"SeaweedFS volume GET failed: HTTP {resp.status_code} fid={fid}")
        return resp.content

    def get_url(self, storage_key, expiry_seconds=3600):
        volume_url, fid = self._decode_key(storage_key)
        return f"{volume_url}/{fid}"

    def delete(self, storage_key):
        volume_url, fid = self._decode_key(storage_key)
        resp = self._session.delete(f"{volume_url}/{fid}", timeout=self._timeout)
        if resp.status_code not in (200, 204, 404):
            raise IOError(
                f"SeaweedFS volume DELETE failed: HTTP {resp.status_code} fid={fid}"
            )
        logger.info("seaweedfs_native_deleted", fid=fid)

    def exists(self, storage_key):
        try:
            volume_url, fid = self._decode_key(storage_key)
            resp = self._session.head(f"{volume_url}/{fid}", timeout=self._timeout)
            return resp.status_code == 200
        except (requests.RequestException, ValueError):
            return False

    def health_check(self):
        """Check master cluster status via /cluster/status."""
        try:
            resp = self._session.get(
                f"{self._master_url}/cluster/status",
                timeout=min(self._timeout, 5.0),
            )
            if resp.status_code == 200:
                data = resp.json()
                return {
                    "status":    "ok",
                    "backend":   "seaweedfs_native",
                    "leader":    data.get("Leader", "unknown"),
                    "is_leader": data.get("IsLeader", False),
                    "peers":     data.get("Peers", []),
                }
            return {"status": "error", "backend": "seaweedfs_native", "http_status": resp.status_code}
        except requests.RequestException as exc:
            return {"status": "error", "backend": "seaweedfs_native", "error": str(exc)}


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_storage(
    backend: str,
    local_base_path: str = "/data/images",
    seaweedfs_filer_url: Optional[str] = None,
    seaweedfs_public_url: Optional[str] = None,
    seaweedfs_replication: str = "000",
    seaweedfs_collection: str = "",
    seaweedfs_ttl: str = "",
    seaweedfs_request_timeout: float = 30.0,
    seaweedfs_max_retries: int = 3,
    seaweedfs_master_url: Optional[str] = None,
    seaweedfs_public_volume_url: Optional[str] = None,
) -> ImageStorage:
    """
    Instantiate the configured storage backend.

    backend options:
      "local"              - LocalImageStorage (dev only)
      "seaweedfs_filer"    - SeaweedFSFilerStorage  <- recommended
      "seaweedfs_native"   - SeaweedFSNativeStorage (no filer layer)
    """
    if backend == "local":
        return LocalImageStorage(local_base_path)

    elif backend == "seaweedfs_filer":
        if not seaweedfs_filer_url:
            raise ValueError(
                "SEAWEEDFS_FILER_URL must be set when IMAGE_BACKEND=seaweedfs_filer\n"
                "Example: SEAWEEDFS_FILER_URL=http://seaweedfs-filer:8888"
            )
        return SeaweedFSFilerStorage(
            filer_url=seaweedfs_filer_url,
            public_url=seaweedfs_public_url or seaweedfs_filer_url,
            replication=seaweedfs_replication,
            collection=seaweedfs_collection,
            ttl=seaweedfs_ttl,
            request_timeout=seaweedfs_request_timeout,
            max_retries=seaweedfs_max_retries,
        )

    elif backend == "seaweedfs_native":
        if not seaweedfs_master_url:
            raise ValueError(
                "SEAWEEDFS_MASTER_URL must be set when IMAGE_BACKEND=seaweedfs_native\n"
                "Example: SEAWEEDFS_MASTER_URL=http://seaweedfs-master:9333"
            )
        return SeaweedFSNativeStorage(
            master_url=seaweedfs_master_url,
            public_volume_url=seaweedfs_public_volume_url,
            replication=seaweedfs_replication,
            collection=seaweedfs_collection,
            ttl=seaweedfs_ttl,
            request_timeout=seaweedfs_request_timeout,
            max_retries=seaweedfs_max_retries,
        )

    else:
        raise ValueError(
            f"Unknown IMAGE_BACKEND='{backend}'. "
            "Valid options: local | seaweedfs_filer | seaweedfs_native"
        )
