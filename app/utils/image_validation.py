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
from pathlib import Path
from typing import Optional, Tuple

from PIL import Image, UnidentifiedImageError

# Hard ceiling enforced by Pillow itself (raises DecompressionBombError above it).
# The per-request limit (max_pixels) is usually stricter; this is a backstop.
Image.MAX_IMAGE_PIXELS = 64_000_000  # ~64 MP backstop against decompression bombs

from app.observability.telemetry import get_logger

logger = get_logger(__name__)

_DEFAULT_MAX_PIXELS = 40_000_000   # 40 MP — validated BEFORE decoding pixels
_MAX_FRAMES = 1                    # reject animated/multi-page (GIF/TIFF) bombs

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
    max_pixels: int = _DEFAULT_MAX_PIXELS,
) -> Tuple[Image.Image, str]:
    """
    Validate image bytes and return a loaded PIL Image + detected mime type.

    Steps (fail-fast order):
      1. Size limit
      2. Magic byte detection
      3. Mime allowlist
      4. Header parse (lazy) → dimension + pixel-count + frame checks
         BEFORE decoding pixels (prevents decompression-bomb RAM blow-up)
      5. Integrity verify() then decode to RGB

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

    # --- 4. Header inspection BEFORE decoding (anti decompression bomb) ---
    try:
        with Image.open(io.BytesIO(file_bytes)) as header:
            w, h = header.size
            n_frames = getattr(header, "n_frames", 1)
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ImageValidationError(f"Corrupt or unreadable image: {exc}") from exc

    if w < _MIN_DIMENSION or h < _MIN_DIMENSION:
        raise ImageValidationError(
            f"Image too small: {w}×{h}px (minimum {_MIN_DIMENSION}×{_MIN_DIMENSION}px)"
        )
    if w * h > max_pixels:
        raise ImageValidationError(
            f"Image has too many pixels: {w}×{h} ({w * h / 1_000_000:.1f} MP, "
            f"max {max_pixels / 1_000_000:.0f} MP)"
        )
    if n_frames > _MAX_FRAMES:
        raise ImageValidationError(
            f"Animated / multi-page images are not allowed ({n_frames} frames)."
        )

    # --- 5. Integrity verify() + decode ---
    try:
        with Image.open(io.BytesIO(file_bytes)) as check:
            check.verify()  # truncation / corruption
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError) as exc:
        raise ImageValidationError(f"Corrupt or unreadable image: {exc}") from exc

    try:
        img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError) as exc:
        raise ImageValidationError(f"Cannot decode image: {exc}") from exc

    # Large (but allowed) images: downscale before inference. PIL is lazy → cheap.
    if img.width > _MAX_DIMENSION or img.height > _MAX_DIMENSION:
        img.thumbnail((_MAX_DIMENSION, _MAX_DIMENSION), Image.LANCZOS)
        logger.info(
            "image_resized_before_inference",
            original_w=w, original_h=h,
            new_w=img.width, new_h=img.height,
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
