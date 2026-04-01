"""
Image validation — validates before any ML inference touches the bytes.

Rules enforced:
  1. Hard size limit (reject before loading into memory as PIL)
  2. Magic byte validation (not just file extension)
  3. PIL integrity check (catches truncated / corrupt files)
  4. Minimum dimensions (too small = unreliable inference)
  5. Path sanitisation (no directory traversal in filenames)
"""

from __future__ import annotations

import io
import re
import struct
from pathlib import Path
from typing import Optional, Tuple

from PIL import Image, UnidentifiedImageError

from app.observability.telemetry import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Magic bytes → (mime_type, extension)
_MAGIC: list[Tuple[bytes, str, str]] = [
    (b"\xff\xd8\xff",             "jpeg", "jpg"),
    (b"\x89PNG\r\n\x1a\n",       "png",  "png"),
    (b"GIF87a",                   "gif",  "gif"),
    (b"GIF89a",                   "gif",  "gif"),
    (b"RIFF",                     "webp", "webp"),   # checked further below
    (b"BM",                       "bmp",  "bmp"),
    (b"II\x2a\x00",               "tiff", "tiff"),
    (b"MM\x00\x2a",               "tiff", "tiff"),
]

_MIN_DIMENSION = 32     # pixels — below this CLIP/SigLIP results are unreliable
_MAX_DIMENSION = 16384  # pixels — above this we resize before inference (not reject)


class ImageValidationError(ValueError):
    """Raised when an image fails validation. Always a 4xx, not 5xx."""
    pass


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def validate_and_load(
    file_bytes: bytes,
    max_bytes: int = 20 * 1024 * 1024,
    allowed_mimes: Optional[frozenset] = None,
) -> Tuple[Image.Image, str]:
    """
    Validate image bytes and return a loaded PIL Image + detected mime type.

    Steps (fail-fast order):
      1. Size limit
      2. Magic byte detection
      3. Mime allowlist
      4. PIL open + verify (catches truncated files)
      5. Minimum dimension check

    Returns:
        (PIL.Image in RGB mode, detected_mime_type)

    Raises:
        ImageValidationError: on any validation failure (use 400/413 response)
    """
    if allowed_mimes is None:
        allowed_mimes = frozenset({"jpeg", "png", "webp", "bmp", "tiff", "gif"})

    # --- 1. Size ---
    size = len(file_bytes)
    if size == 0:
        raise ImageValidationError("Empty file received")
    if size > max_bytes:
        raise ImageValidationError(
            f"Image too large: {size / 1024 / 1024:.1f} MB "
            f"(max {max_bytes / 1024 / 1024:.0f} MB)"
        )

    # --- 2. Magic bytes ---
    detected_mime = _detect_mime(file_bytes)
    if detected_mime is None:
        raise ImageValidationError(
            "File does not appear to be a recognised image format. "
            "Accepted: JPEG, PNG, WebP, BMP, TIFF, GIF."
        )

    # --- 3. Allowlist ---
    if detected_mime not in allowed_mimes:
        raise ImageValidationError(
            f"Image format '{detected_mime}' is not allowed. "
            f"Accepted: {', '.join(sorted(allowed_mimes))}"
        )

    # --- 4. PIL integrity ---
    try:
        img = Image.open(io.BytesIO(file_bytes))
        img.verify()  # checks for truncation / corruption
    except (UnidentifiedImageError, Exception) as exc:
        raise ImageValidationError(f"Corrupt or unreadable image: {exc}") from exc

    # Re-open after verify() (PIL closes the image after verify)
    try:
        img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    except Exception as exc:
        raise ImageValidationError(f"Cannot decode image: {exc}") from exc

    # --- 5. Dimensions ---
    w, h = img.size
    if w < _MIN_DIMENSION or h < _MIN_DIMENSION:
        raise ImageValidationError(
            f"Image too small: {w}×{h}px (minimum {_MIN_DIMENSION}×{_MIN_DIMENSION}px)"
        )

    # Large images: resize before returning — PIL is lazy, this is cheap
    if w > _MAX_DIMENSION or h > _MAX_DIMENSION:
        img.thumbnail((_MAX_DIMENSION, _MAX_DIMENSION), Image.LANCZOS)
        logger.info(
            "image_resized_before_inference",
            original_w=w,
            original_h=h,
            new_w=img.width,
            new_h=img.height,
        )

    return img, detected_mime


def sanitise_filename(filename: str) -> str:
    """
    Strip directory components and normalise filename.
    Rejects names that are empty or consist only of dots.

    Safe to use in filesystem paths and as metadata values.
    """
    if not filename:
        return "unknown"

    # Strip directory traversal components
    name = Path(filename).name

    # Remove null bytes and control characters
    name = re.sub(r"[\x00-\x1f\x7f]", "", name)

    # Collapse multiple dots/spaces
    name = re.sub(r"\.{2,}", ".", name)
    name = name.strip(". ")

    if not name:
        return "unknown"

    # Length limit
    if len(name) > 255:
        stem = Path(name).stem[:200]
        suffix = Path(name).suffix[:10]
        name = stem + suffix

    return name


def sanitise_group_id(group_id: str) -> str:
    """
    Allowlist sanitisation for group_id used in filesystem paths and Qdrant filters.
    Only alphanumeric, hyphens, and underscores allowed.
    """
    if not group_id:
        return "default"
    sanitised = re.sub(r"[^a-zA-Z0-9_\-]", "", group_id)
    return sanitised[:128] if sanitised else "default"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _detect_mime(data: bytes) -> Optional[str]:
    """Identify image type from magic bytes (not file extension)."""
    header = data[:16]

    for magic, mime, _ in _MAGIC:
        if header.startswith(magic):
            # Extra WebP check: bytes 8-12 must be b'WEBP'
            if mime == "webp":
                if len(data) >= 12 and data[8:12] == b"WEBP":
                    return "webp"
                # It's a RIFF but not WebP
                return None
            return mime

    return None
