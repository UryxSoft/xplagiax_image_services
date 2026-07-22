"""
Flask routes for the reverse-image-search module.

  POST /api/v1/reverse-image-search         — single image. Returns exactly
                                               the six-field contract: found,
                                               website, url, similarity,
                                               provider, elapsed_ms. Nothing else.
  POST /api/v1/reverse-image-search/batch   — multiple images in one request.
                                               Each is independent (its own
                                               SHA-256, cache entry, and
                                               Early Stop chain) so they're
                                               fanned out concurrently with
                                               gevent — only the provider
                                               chain WITHIN one image stays
                                               sequential, since that's what
                                               Early Stop's cost saving
                                               depends on.
  GET  /api/v1/_tmp-image/<token>           — internal, single-use, short-TTL
                                               image serving for URL-based
                                               providers (see temp_hosting.py).
                                               Deliberately NOT behind
                                               require_auth: the external
                                               provider's crawler fetches this
                                               URL directly and cannot send
                                               our API key. Safe because
                                               tokens are random, single-use,
                                               and expire in seconds.
"""

from __future__ import annotations

from typing import Optional

from gevent.pool import Pool as GeventPool
from flask import Blueprint, Response, current_app, jsonify, request
from werkzeug.datastructures import FileStorage

from app.observability.telemetry import get_logger
from app.security.middleware import rate_limit, require_auth
from app.utils.image_fetcher import fetch_image_bytes
from app.utils.image_validation import ImageValidationError, validate_image_header

logger = get_logger(__name__)

reverse_search_bp = Blueprint("reverse_search", __name__, url_prefix="/api/v1")

_DEFAULT_MAX_BATCH_SIZE = 20
_DEFAULT_BATCH_CONCURRENCY = 10


def _svc(name: str):
    return current_app.extensions.get(f"xplagiax_{name}")


def _search_one(orchestrator, cfg, *, file_upload: Optional[FileStorage] = None,
                 image_url: Optional[str] = None) -> tuple[dict, int]:
    """
    Validate + search a single image. Shared by the single and batch
    endpoints so there's exactly one code path that reads bytes once,
    validates the header, and calls the orchestrator.

    Returns (payload, http_status). payload is either the 6-field result
    contract or an {"error", "code"} pair.
    """
    try:
        image_bytes, _filename = fetch_image_bytes(
            file_upload=file_upload,
            image_url=image_url,
            image_path=None,  # local-path reads are an LFI risk; not exposed here
            max_bytes=cfg.max_image_bytes,
            allow_local_path=False,
        )
        detected_mime = validate_image_header(
            image_bytes, max_bytes=cfg.max_image_bytes,
            allowed_mimes=cfg.allowed_mime_types, max_pixels=cfg.max_image_pixels,
        )
    except ImageValidationError as exc:
        return {"error": str(exc), "code": "INVALID_IMAGE"}, 400

    result = orchestrator.search(image_bytes, content_type=f"image/{detected_mime}")
    return result.to_response(), 200


@reverse_search_bp.route("/reverse-image-search", methods=["POST"])
@require_auth
@rate_limit
def reverse_image_search():
    cfg = _svc("security_config")
    orchestrator = _svc("reverse_search_orchestrator")
    if orchestrator is None:
        return jsonify({
            "error": "Reverse image search unavailable — no provider configured",
            "code": "SERVICE_UNAVAILABLE",
        }), 503

    params = (request.get_json(silent=True) or {}) if request.is_json else request.form.to_dict()
    file_upload = request.files.get("image") or request.files.get("file")
    image_url = params.get("image_url")

    if not file_upload and not image_url:
        return jsonify({
            "error": "No image source provided (image file or image_url).",
            "code": "MISSING_FILE",
        }), 400

    payload, status = _search_one(orchestrator, cfg, file_upload=file_upload, image_url=image_url)
    return jsonify(payload), status


@reverse_search_bp.route("/reverse-image-search/batch", methods=["POST"])
@require_auth
@rate_limit
def reverse_image_search_batch():
    cfg = _svc("security_config")
    orchestrator = _svc("reverse_search_orchestrator")
    if orchestrator is None:
        return jsonify({
            "error": "Reverse image search unavailable — no provider configured",
            "code": "SERVICE_UNAVAILABLE",
        }), 503

    rs_cfg = _svc("reverse_search_config")
    max_batch_size = rs_cfg.max_batch_size if rs_cfg else _DEFAULT_MAX_BATCH_SIZE
    batch_concurrency = rs_cfg.batch_concurrency if rs_cfg else _DEFAULT_BATCH_CONCURRENCY

    params = (request.get_json(silent=True) or {}) if request.is_json else request.form.to_dict()
    files = request.files.getlist("files")
    image_urls = params.get("image_urls") or []
    if isinstance(image_urls, str):
        image_urls = [image_urls]

    total = len(files) + len(image_urls)
    if total == 0:
        return jsonify({
            "error": "No images provided (files or image_urls).",
            "code": "MISSING_FILES",
        }), 400
    if total > max_batch_size:
        return jsonify({
            "error": f"Maximum {max_batch_size} images per batch",
            "code": "BATCH_TOO_LARGE",
        }), 400

    # Each job is independent — no shared state between images beyond the
    # already-thread/greenlet-safe pooled requests.Session inside each
    # provider adapter (the same session already serves concurrent client
    # requests under the gevent worker, so this is nothing new).
    jobs = []
    for f in files:
        if not f.filename:
            jobs.append(("unknown", None, None, "Empty filename", "EMPTY_FILENAME"))
            continue
        jobs.append((f.filename, f, None, None, None))
    for url in image_urls:
        jobs.append((url, None, url, None, None))

    def run_job(job):
        source, file_upload, image_url, preset_error, preset_code = job
        if preset_error:
            return {"source": source, "error": preset_error, "code": preset_code}
        payload, _status = _search_one(orchestrator, cfg, file_upload=file_upload, image_url=image_url)
        payload["source"] = source
        return payload

    # Bounded concurrency: images are independent so they run at the same
    # time, but a full batch firing every image at once would both risk
    # exhausting the shared connection pool and look like a burst to the
    # external provider. GeventPool caps how many run simultaneously,
    # queuing the rest, while still returning results in job order.
    pool = GeventPool(size=max(1, batch_concurrency))
    results = pool.map(run_job, jobs)

    return jsonify({"count": len(results), "results": list(results)}), 207


@reverse_search_bp.route("/_tmp-image/<token>", methods=["GET"])
def serve_temp_image(token: str):
    # Read from the shared xplagiax_temp_host extension (not the
    # orchestrator's own) so other features — e.g. patents_bp's file-upload
    # path in the standalone app — can reuse the exact same ephemeral
    # hosting without needing an orchestrator instance to exist at all.
    temp_host = _svc("temp_host")
    if temp_host is None:
        return jsonify({"error": "Not found", "code": "NOT_FOUND"}), 404

    served = temp_host.fetch_and_consume(token)
    if served is None:
        return jsonify({"error": "Not found or already consumed", "code": "NOT_FOUND"}), 404

    data, content_type = served
    return Response(data, mimetype=content_type)
