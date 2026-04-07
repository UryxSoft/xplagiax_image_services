"""
Flask route blueprints — all endpoints.

Each blueprint has a single responsibility:
  images_bp    — upload, retrieve, delete images
  search_bp    — similarity search and plagiarism analysis
  patents_bp   — external patent + reverse image search
  admin_bp     — collection management (authenticated + destructive ops)
  health_bp    — health, readiness, liveness probes
  jobs_bp      — async job status polling

All routes:
  - Use require_auth and rate_limit decorators
  - Validate inputs before touching ML or storage
  - Return structured JSON with consistent error codes
  - Never expose stack traces or internal paths
"""

from __future__ import annotations

import dataclasses
import hashlib
from typing import Optional

from flask import Blueprint, current_app, g, jsonify, request, send_file

from app.observability.telemetry import get_logger
from app.security.middleware import rate_limit, require_auth
from app.utils.image_validation import (
    ImageValidationError,
    sanitise_filename,
    sanitise_group_id,
    validate_and_load,
)

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Blueprint definitions
# ---------------------------------------------------------------------------

images_bp  = Blueprint("images",  __name__, url_prefix="/api/v1/images")
search_bp  = Blueprint("search",  __name__, url_prefix="/api/v1/search")
patents_bp = Blueprint("patents", __name__, url_prefix="/api/v1/patents")
admin_bp   = Blueprint("admin",   __name__, url_prefix="/api/v1/admin")
health_bp  = Blueprint("health",  __name__, url_prefix="")
jobs_bp    = Blueprint("jobs",    __name__, url_prefix="/api/v1/jobs")


# ---------------------------------------------------------------------------
# Helper: extract services from app extensions
# ---------------------------------------------------------------------------

def _svc(name: str):
    return current_app.extensions[f"xplagiax_{name}"]


def _parse_image_upload(required: bool = True):
    """Parse and validate uploaded image. Returns (image_bytes, pil_image, filename)."""
    cfg = _svc("security_config")

    if "file" not in request.files:
        if required:
            return None, None, None, {"error": "No file uploaded", "code": "MISSING_FILE"}, 400
        return None, None, None, None, None

    file = request.files["file"]
    if not file.filename:
        return None, None, None, {"error": "Empty filename", "code": "EMPTY_FILENAME"}, 400

    try:
        image_bytes = file.read(cfg.max_image_bytes + 1)
        pil_image, mime_type = validate_and_load(
            image_bytes,
            max_bytes=cfg.max_image_bytes,
            allowed_mimes=cfg.allowed_mime_types,
        )
        filename = sanitise_filename(file.filename)
        return image_bytes, pil_image, filename, None, None
    except ImageValidationError as exc:
        return None, None, None, {"error": str(exc), "code": "INVALID_IMAGE"}, 400


# ===========================================================================
# images_bp — Upload, retrieve, delete
# ===========================================================================

@images_bp.route("", methods=["POST"])
@require_auth
@rate_limit
def upload_and_index():
    """
    Upload an image for indexing.

    Form fields:
      file              (required) image file
      group_id          (optional) logical document group, default "default"
      page              (optional) page number within a document
      run_ai_detection  (optional) bool, default true
      extra_*           any extra_* fields are stored as metadata

    Returns 202 (async) or 200 (sync, degraded mode) with job info.
    """
    image_bytes, pil_image, filename, err, code = _parse_image_upload()
    if err:
        return jsonify(err), code

    group_id = sanitise_group_id(request.form.get("group_id", "default"))
    page_raw = request.form.get("page")
    page = int(page_raw) if page_raw and page_raw.isdigit() else None
    run_ai = request.form.get("run_ai_detection", "true").lower() != "false"

    extra = {
        k.removeprefix("extra_"): v
        for k, v in request.form.items()
        if k.startswith("extra_")
    }

    try:
        indexing = _svc("indexing")
        result = indexing.submit(
            image_bytes=image_bytes,
            pil_image=pil_image,
            filename=filename,
            group_id=group_id,
            page=page,
            run_ai_detection=run_ai,
            extra_metadata=extra,
        )
    except Exception as exc:
        logger.error("upload_failed", error=str(exc), exc_info=True)
        return jsonify({
            "error": "Indexing failed",
            "code": "INDEXING_ERROR",
            "request_id": g.request_id,
        }), 500

    status_code = 202 if result.get("status") == "queued" else 200
    return jsonify(result), status_code


@images_bp.route("/batch", methods=["POST"])
@require_auth
@rate_limit
def upload_batch():
    """
    Upload multiple images in a single request.
    Returns list of per-image results (or errors).
    Max 20 files per batch.
    """
    files = request.files.getlist("files")
    if not files:
        return jsonify({"error": "No files uploaded", "code": "MISSING_FILES"}), 400
    if len(files) > 20:
        return jsonify({
            "error": "Maximum 20 files per batch",
            "code": "BATCH_TOO_LARGE",
        }), 400

    cfg = _svc("security_config")
    group_id = sanitise_group_id(request.form.get("group_id", "default"))
    run_ai = request.form.get("run_ai_detection", "true").lower() != "false"
    indexing = _svc("indexing")

    results = []
    for file in files:
        if not file.filename:
            results.append({"error": "Empty filename", "code": "EMPTY_FILENAME"})
            continue
        try:
            raw = file.read(cfg.max_image_bytes + 1)
            pil_image, _ = validate_and_load(raw, max_bytes=cfg.max_image_bytes)
            filename = sanitise_filename(file.filename)
            result = indexing.submit(
                image_bytes=raw,
                pil_image=pil_image,
                filename=filename,
                group_id=group_id,
                run_ai_detection=run_ai,
            )
            results.append(result)
        except ImageValidationError as exc:
            results.append({
                "filename": sanitise_filename(file.filename or "unknown"),
                "error": str(exc),
                "code": "INVALID_IMAGE",
            })
        except Exception as exc:
            logger.error("batch_item_failed", file=file.filename, error=str(exc))
            results.append({
                "filename": sanitise_filename(file.filename or "unknown"),
                "error": "Indexing failed",
                "code": "INDEXING_ERROR",
                "request_id": g.request_id,
            })

    return jsonify({
        "total": len(results),
        "group_id": group_id,
        "results": results,
    }), 207  # Multi-Status


@images_bp.route("/<point_id>", methods=["GET"])
@require_auth
def get_image(point_id: str):
    """Retrieve image binary by Qdrant point ID."""
    try:
        repo = _svc("repo")
        storage = _svc("storage")

        payload = repo.get_by_id(point_id)
        if not payload:
            return jsonify({"error": "Not found", "code": "NOT_FOUND"}), 404

        storage_key = payload.get("storage_key")
        if not storage_key:
            return jsonify({
                "error": "No storage reference in metadata",
                "code": "MISSING_STORAGE_KEY",
            }), 404

        image_bytes = storage.load(storage_key)
        mime = payload.get("mime_type", "jpeg")
        return send_file(
            __import__("io").BytesIO(image_bytes),
            mimetype=f"image/{mime}",
            as_attachment=False,
        )
    except FileNotFoundError:
        return jsonify({"error": "Image file not found", "code": "FILE_NOT_FOUND"}), 404
    except Exception as exc:
        logger.error("get_image_failed", point_id=point_id, error=str(exc))
        return jsonify({
            "error": "Failed to retrieve image",
            "code": "RETRIEVAL_ERROR",
            "request_id": g.request_id,
        }), 500


@images_bp.route("/<point_id>/url", methods=["GET"])
@require_auth
def get_image_url(point_id: str):
    """Get a pre-signed or direct URL for an image."""
    repo = _svc("repo")
    storage = _svc("storage")
    cfg_storage = _svc("storage_config")

    payload = repo.get_by_id(point_id)
    if not payload:
        return jsonify({"error": "Not found", "code": "NOT_FOUND"}), 404

    storage_key = payload.get("storage_key")
    if not storage_key:
        return jsonify({"error": "No storage reference", "code": "MISSING_STORAGE_KEY"}), 404

    url = storage.get_url(storage_key, expiry_seconds=3600)
    return jsonify({
        "point_id": point_id,
        "url": url,
        "backend": storage.backend_name(),
    })


@images_bp.route("/<point_id>", methods=["DELETE"])
@require_auth
def delete_image(point_id: str):
    """Delete an image from Qdrant and optionally from storage."""
    repo = _svc("repo")
    storage = _svc("storage")

    payload = repo.get_by_id(point_id)
    if not payload:
        return jsonify({"error": "Not found", "code": "NOT_FOUND"}), 404

    storage_key = payload.get("storage_key")
    try:
        repo.delete_by_id(point_id)
        if storage_key:
            try:
                storage.delete(storage_key)
            except Exception as exc:
                logger.warning(
                    "storage_delete_failed_vector_deleted",
                    storage_key=storage_key,
                    error=str(exc),
                )
        return jsonify({"deleted": True, "point_id": point_id}), 200
    except Exception as exc:
        logger.error("delete_failed", point_id=point_id, error=str(exc))
        return jsonify({
            "error": "Delete failed",
            "code": "DELETE_ERROR",
            "request_id": g.request_id,
        }), 500


# ===========================================================================
# search_bp — Similarity + plagiarism
# ===========================================================================

@search_bp.route("/similar", methods=["POST"])
@require_auth
@rate_limit
def search_similar():
    """
    Search for visually similar images.

    Form fields:
      file       (required) query image
      limit      (optional) 1-50, default 10
      threshold  (optional) 0.0-1.0, default 0.0
      group_id   (optional) restrict search to a group
    """
    image_bytes, pil_image, _, err, code = _parse_image_upload()
    if err:
        return jsonify(err), code

    limit = min(int(request.form.get("limit", 10)), 50)
    threshold = float(request.form.get("threshold", 0.0))
    group_id_raw = request.form.get("group_id")
    group_id = sanitise_group_id(group_id_raw) if group_id_raw else None

    try:
        svc = _svc("similarity")
        matches = svc.search_similar(
            image_bytes=image_bytes,
            pil_image=pil_image,
            limit=limit,
            threshold=threshold,
            group_id=group_id,
        )
        return jsonify({
            "count": len(matches),
            "threshold": threshold,
            "group_id": group_id,
            "results": [dataclasses.asdict(m) for m in matches],
        }), 200
    except RuntimeError as exc:
        return jsonify({
            "error": str(exc),
            "code": "MODEL_NOT_READY",
        }), 503
    except Exception as exc:
        logger.error("search_failed", error=str(exc), exc_info=True)
        return jsonify({
            "error": "Search failed",
            "code": "SEARCH_ERROR",
            "request_id": g.request_id,
        }), 500


@search_bp.route("/plagiarism", methods=["POST"])
@require_auth
@rate_limit
def analyze_plagiarism():
    """
    Plagiarism analysis — finds copies and modified versions.

    Form fields:
      file                  (required) image to analyze
      similarity_threshold  (optional) default 0.90
      limit                 (optional) 1-20, default 5
      group_id              (optional) restrict to group
    """
    image_bytes, pil_image, _, err, code = _parse_image_upload()
    if err:
        return jsonify(err), code

    threshold = float(request.form.get("similarity_threshold", 0.90))
    threshold = max(0.5, min(1.0, threshold))  # clamp to sensible range
    limit = min(int(request.form.get("limit", 5)), 20)
    group_id_raw = request.form.get("group_id")
    group_id = sanitise_group_id(group_id_raw) if group_id_raw else None

    try:
        svc = _svc("similarity")
        report = svc.analyze_plagiarism(
            image_bytes=image_bytes,
            pil_image=pil_image,
            threshold=threshold,
            limit=limit,
            group_id=group_id,
            exclude_self=True,
        )
        return jsonify(dataclasses.asdict(report)), 200
    except RuntimeError as exc:
        return jsonify({"error": str(exc), "code": "MODEL_NOT_READY"}), 503
    except Exception as exc:
        logger.error("plagiarism_analysis_failed", error=str(exc), exc_info=True)
        return jsonify({
            "error": "Analysis failed",
            "code": "ANALYSIS_ERROR",
            "request_id": g.request_id,
        }), 500


@search_bp.route("/ai-detection", methods=["POST"])
@require_auth
@rate_limit
def analyze_ai_detection():
    """
    Classify whether an image is AI-generated or human-created.
    Does NOT index the image.
    """
    image_bytes, pil_image, _, err, code = _parse_image_upload()
    if err:
        return jsonify(err), code

    try:
        models = _svc("models")
        if not models.siglip_ready:
            return jsonify({
                "error": "AI detection model not available",
                "code": "MODEL_NOT_READY",
                "detail": models.get_status()["siglip"]["error"],
            }), 503

        result = models.classify_single(pil_image)
        return jsonify({
            "is_ai":        result.is_ai,
            "is_human":     result.is_human,
            "label":        result.label,
            "confidence":   round(result.confidence, 6),
            "ai_score":     result.ai_score,
            "human_score":  result.human_score,
            "all_scores":   result.all_scores,
            "model_id":     result.model_id,
            "duration_ms":  round(result.duration_ms, 1),
        }), 200
    except Exception as exc:
        logger.error("ai_detection_failed", error=str(exc), exc_info=True)
        return jsonify({
            "error": "AI detection failed",
            "code": "DETECTION_ERROR",
            "request_id": g.request_id,
        }), 500


# ===========================================================================
# patents_bp
# ===========================================================================

@patents_bp.route("/search/image", methods=["POST"])
@require_auth
@rate_limit
def patent_search_by_image():
    """
    Find patents related to an image (2-step: reverse image → patent search).
    Requires SERPAPI_KEY.
    """
    rotator = _svc("api_rotator")
    if not rotator:
        return jsonify({
            "error": "Patent search unavailable — SERPAPI_KEY not configured",
            "code": "SERVICE_UNAVAILABLE",
        }), 503

    image_url = request.form.get("image_url")
    num_results = min(int(request.form.get("num_results", 10)), 50)

    if not image_url and "file" in request.files:
        import base64, io as _io
        from PIL import Image as PILImage
        raw = request.files["file"].read()
        try:
            img = PILImage.open(_io.BytesIO(raw))
            fmt = (img.format or "jpeg").lower()
        except Exception:
            fmt = "jpeg"
        b64 = base64.b64encode(raw).decode()
        image_url = f"data:image/{fmt};base64,{b64}"

    if not image_url:
        return jsonify({"error": "Provide 'file' or 'image_url'", "code": "MISSING_INPUT"}), 400

    try:
        results = rotator.patent_image_search(image_url, num_results)
        return jsonify({
            "status": "success",
            "results": results,
            "usage": rotator.get_usage_status(),
        }), 200
    except Exception as exc:
        logger.error("patent_image_search_failed", error=str(exc))
        return jsonify({"error": str(exc), "code": "PATENT_SEARCH_ERROR"}), 500


@patents_bp.route("/search/text", methods=["POST"])
@require_auth
@rate_limit
def patent_search_by_text():
    rotator = _svc("api_rotator")
    if not rotator:
        return jsonify({"error": "Patent search unavailable", "code": "SERVICE_UNAVAILABLE"}), 503

    data = request.get_json(silent=True) or {}
    query = data.get("query", "").strip()
    if not query:
        return jsonify({"error": "Field 'query' is required", "code": "MISSING_QUERY"}), 400

    num_results = min(int(data.get("num_results", 10)), 50)
    try:
        results = rotator.patent_text_search(query, num_results)
        return jsonify({
            "status": "success",
            "results": results,
            "usage": rotator.get_usage_status(),
        }), 200
    except Exception as exc:
        return jsonify({"error": str(exc), "code": "PATENT_SEARCH_ERROR"}), 500


@patents_bp.route("/<patent_id>", methods=["GET"])
@require_auth
def get_patent_details(patent_id: str):
    rotator = _svc("api_rotator")
    if not rotator:
        return jsonify({"error": "Patent search unavailable", "code": "SERVICE_UNAVAILABLE"}), 503
    try:
        details = rotator.get_patent_details(patent_id)
        return jsonify({"status": "success", "patent": details}), 200
    except Exception as exc:
        return jsonify({"error": str(exc), "code": "PATENT_DETAILS_ERROR"}), 500


@patents_bp.route("/reverse-image", methods=["POST"])
@require_auth
@rate_limit
def reverse_image_search():
    rotator = _svc("api_rotator")
    if not rotator:
        return jsonify({"error": "Reverse search unavailable", "code": "SERVICE_UNAVAILABLE"}), 503

    import base64, io as _io
    from PIL import Image as PILImage

    image_url = request.form.get("image_url")
    num_results = min(int(request.form.get("num_results", 10)), 50)

    if not image_url and "file" in request.files:
        raw = request.files["file"].read()
        try:
            img = PILImage.open(_io.BytesIO(raw))
            fmt = (img.format or "jpeg").lower()
        except Exception:
            fmt = "jpeg"
        image_url = f"data:image/{fmt};base64,{base64.b64encode(raw).decode()}"

    if not image_url:
        return jsonify({"error": "Provide 'file' or 'image_url'", "code": "MISSING_INPUT"}), 400

    try:
        results = rotator.reverse_image_search(image_url, num_results)
        return jsonify({
            "status": "success",
            "results": results,
            "usage": rotator.get_usage_status(),
        }), 200
    except Exception as exc:
        return jsonify({"error": str(exc), "code": "REVERSE_SEARCH_ERROR"}), 500


@patents_bp.route("/usage", methods=["GET"])
@require_auth
def api_usage():
    rotator = _svc("api_rotator")
    if not rotator:
        return jsonify({"error": "API rotator unavailable", "code": "SERVICE_UNAVAILABLE"}), 503
    return jsonify(rotator.get_usage_status()), 200


# ===========================================================================
# admin_bp — destructive operations
# ===========================================================================

RESET_CONFIRMATION_TOKEN = "I_UNDERSTAND_THIS_WILL_DELETE_ALL_DATA"


@admin_bp.route("/collection/reset", methods=["DELETE"])
@require_auth
def reset_collection():
    """
    Drop and recreate the entire Qdrant collection.
    DESTRUCTIVE. Requires explicit confirmation token.
    """
    data = request.get_json(silent=True) or {}
    if data.get("confirm") != RESET_CONFIRMATION_TOKEN:
        return jsonify({
            "error": "Confirmation required",
            "code": "CONFIRMATION_REQUIRED",
            "hint": f"Send JSON body: {{\"confirm\": \"{RESET_CONFIRMATION_TOKEN}\"}}",
        }), 400

    try:
        repo = _svc("repo")
        repo.reset_collection()
        return jsonify({"status": "collection_reset", "warning": "All data deleted"}), 200
    except Exception as exc:
        logger.error("reset_collection_failed", error=str(exc))
        return jsonify({"error": "Reset failed", "code": "RESET_ERROR"}), 500


@admin_bp.route("/collection/groups/<group_id>", methods=["DELETE"])
@require_auth
def delete_group(group_id: str):
    safe_group_id = sanitise_group_id(group_id)
    try:
        repo = _svc("repo")
        repo.delete_by_group(safe_group_id)
        return jsonify({"deleted": True, "group_id": safe_group_id}), 200
    except Exception as exc:
        logger.error("delete_group_failed", group_id=safe_group_id, error=str(exc))
        return jsonify({"error": "Delete failed", "code": "DELETE_ERROR"}), 500


@admin_bp.route("/collection/items", methods=["GET"])
@require_auth
def list_items():
    """Paginated list. Default 100 per page, max 1000."""
    limit = min(int(request.args.get("limit", 100)), 1000)
    offset = request.args.get("offset") or None
    group_id_raw = request.args.get("group_id")
    group_id = sanitise_group_id(group_id_raw) if group_id_raw else None

    try:
        repo = _svc("repo")
        items, next_offset = repo.scroll_all(limit=limit, offset=offset, group_id=group_id)
        return jsonify({
            "count":       len(items),
            "items":       items,
            "next_offset": next_offset,
        }), 200
    except Exception as exc:
        logger.error("list_items_failed", error=str(exc))
        return jsonify({"error": "List failed", "code": "LIST_ERROR"}), 500


@admin_bp.route("/models", methods=["GET"])
@require_auth
def model_info():
    models = _svc("models")
    repo = _svc("repo")
    return jsonify({
        "models": models.get_status(),
        "collection": dataclasses.asdict(repo.stats()),
    }), 200


# ===========================================================================
# jobs_bp — async job status
# ===========================================================================

@jobs_bp.route("/<job_id>", methods=["GET"])
@require_auth
def job_status(job_id: str):
    cache = _svc("cache")
    status = cache.get_job(job_id)
    if status is None:
        return jsonify({"error": "Job not found", "code": "JOB_NOT_FOUND"}), 404
    return jsonify(status), 200


# ===========================================================================
# health_bp — probes
# ===========================================================================

@health_bp.route("/healthz", methods=["GET"])
def liveness():
    """Liveness probe — is the process alive? Always 200 unless crashed."""
    return jsonify({"status": "alive"}), 200


@health_bp.route("/readyz", methods=["GET"])
def readiness():
    """
    Readiness probe — is the service ready to handle traffic?
    Returns 200 only when Qdrant + CLIP are operational.
    """
    checks = {}
    overall_ok = True

    # Qdrant
    repo = _svc("repo")
    qdrant_health = repo.health_check()
    checks["qdrant"] = qdrant_health
    if qdrant_health["status"] != "ok":
        overall_ok = False

    # CLIP (required)
    models = _svc("models")
    model_status = models.get_status()
    checks["clip"] = {
        "loaded": model_status["clip"]["loaded"],
        "error":  model_status["clip"]["error"],
    }
    if not model_status["clip"]["loaded"]:
        overall_ok = False

    # SigLIP (optional — degraded if down, not unhealthy)
    checks["siglip"] = {
        "loaded":  model_status["siglip"]["loaded"],
        "error":   model_status["siglip"]["error"],
        "degraded": not model_status["siglip"]["loaded"],
    }

    # Redis (optional)
    cache = _svc("cache")
    checks["redis"] = cache.health_check()

    return jsonify({
        "status": "ready" if overall_ok else "not_ready",
        "checks": checks,
    }), 200 if overall_ok else 503


@health_bp.route("/health", methods=["GET"])
def health_detailed():
    """Detailed health — includes collection stats, model metadata, storage status."""
    repo = _svc("repo")
    models = _svc("models")
    cache = _svc("cache")
    rotator = _svc("api_rotator")
    storage = _svc("storage")

    return jsonify({
        "status":         "healthy",
        "models":         models.get_status(),
        "qdrant":         dataclasses.asdict(repo.stats()),
        "redis":          cache.health_check(),
        "storage":        storage.health_check(),
        "api_rotator":    {
            "available": rotator is not None,
            "usage":     rotator.get_usage_status() if rotator else None,
        },
    }), 200
