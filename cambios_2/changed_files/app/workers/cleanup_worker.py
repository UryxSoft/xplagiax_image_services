"""
Cleanup worker — backstop sweep of orphaned temp_search uploads in SeaweedFS.

Primary protection against leaks is the per-request TTL set at save time
(SEAWEEDFS_TTL) plus the route's finally-block delete. This worker is a
defence-in-depth sweep that deletes temp_search files older than `max_age`
in case a process crashed before cleanup and TTL was not configured.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Optional

import requests
from redis import Redis
from rq import Queue

from app.observability.telemetry import get_logger

logger = get_logger(__name__)

_FILER_BASE = "xplagiax/images"
_TEMP_PREFIX = "temp_search"


def _parse_mtime(value) -> Optional[float]:
    """SeaweedFS may return Mtime as epoch int or RFC3339 string."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def _list_dir(session: requests.Session, filer_url: str, path: str, timeout: float):
    resp = session.get(
        f"{filer_url}/{path}/",
        headers={"Accept": "application/json"},
        params={"limit": 1000},
        timeout=timeout,
    )
    if resp.status_code != 200:
        return []
    return (resp.json() or {}).get("Entries") or []


def run_cleanup_job(
    seaweedfs_filer_url: str,
    max_age_seconds: int = 3600,
    max_depth: int = 3,
    timeout: float = 15.0,
) -> dict:
    """Delete temp_search files older than max_age_seconds. Returns a summary."""
    filer_url = seaweedfs_filer_url.rstrip("/")
    cutoff = time.time() - max_age_seconds
    session = requests.Session()
    deleted, scanned, errors = 0, 0, 0

    def walk(path: str, depth: int):
        nonlocal deleted, scanned, errors
        if depth > max_depth:
            return
        for entry in _list_dir(session, filer_url, path, timeout):
            full = (entry.get("FullPath") or "").lstrip("/")
            if not full:
                continue
            is_dir = bool(entry.get("Mode", 0) & 0o40000) or "chunks" not in entry and entry.get("FileSize") is None
            if entry.get("FileSize") is not None or not is_dir:
                scanned += 1
                mtime = _parse_mtime(entry.get("Mtime") or entry.get("Crtime"))
                if mtime is not None and mtime < cutoff:
                    try:
                        r = session.delete(f"{filer_url}/{full}", timeout=timeout)
                        if r.status_code in (200, 204, 404):
                            deleted += 1
                        else:
                            errors += 1
                    except requests.RequestException:
                        errors += 1
            else:
                # Recurse into shard subdirectories.
                rel = full[len(_FILER_BASE) + 1:] if full.startswith(_FILER_BASE) else full
                walk(f"{_FILER_BASE}/{rel}" if not full.startswith(_FILER_BASE) else full, depth + 1)

    logger.info("cleanup_job_started", prefix=_TEMP_PREFIX, max_age_seconds=max_age_seconds)
    try:
        walk(f"{_FILER_BASE}/{_TEMP_PREFIX}", 0)
    except Exception as exc:  # never let the backstop crash the worker
        logger.error("cleanup_job_failed", error=str(exc))
        errors += 1

    summary = {"deleted": deleted, "scanned": scanned, "errors": errors}
    logger.info("cleanup_job_completed", **summary)
    return summary


def schedule_cleanup(redis_url: str, seaweedfs_url: str, interval_seconds: int = 3600):
    """Enqueue a single cleanup pass (call from a cron/rq-scheduler)."""
    redis_conn = Redis.from_url(redis_url)
    q = Queue("cleanup", connection=redis_conn)
    q.enqueue(run_cleanup_job, seaweedfs_url, interval_seconds, job_timeout=600)
    logger.info("cleanup_job_scheduled", interval_seconds=interval_seconds)
