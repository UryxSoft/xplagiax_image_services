"""
Secure and robust image fetching from multiple sources:
- HTTP/HTTPS URLs (streamed to avoid memory exhaustion)
- Local file paths (with LFI/Path Traversal awareness)
- Direct multipart file uploads
"""

import os
import requests
import urllib.parse
from typing import Tuple, Optional
from werkzeug.datastructures import FileStorage

from app.security.http_client import safe_download, SSRFViolationError, DownloadLimitExceededError
from app.utils.image_validation import ImageValidationError

def fetch_image_bytes(
    file_upload: Optional[FileStorage] = None,
    image_url: Optional[str] = None,
    image_path: Optional[str] = None,
    max_bytes: int = 20 * 1024 * 1024
) -> Tuple[bytes, str]:
    """
    Fetches image bytes from the first available source:
    1. file_upload (multipart form data)
    2. image_url (http/https)
    3. image_path (local filesystem)

    Returns:
        (image_bytes, filename)
    Raises:
        ImageValidationError: If no source is provided, or fetching fails.
    """
    if file_upload and file_upload.filename:
        image_bytes = file_upload.read(max_bytes + 1)
        if len(image_bytes) > max_bytes:
            raise ImageValidationError(f"File upload too large (max {max_bytes} bytes).")
        return image_bytes, file_upload.filename

    if image_url:
        try:
            # safe_download handles SSRF, Redirects, and DOS streaming automatically
            raw_bytes, content_type = safe_download(
                image_url, 
                max_bytes=max_bytes, 
                timeout=5.0
            )
            parsed = urllib.parse.urlparse(image_url)
            filename = os.path.basename(parsed.path) or "remote_image.jpg"
            return raw_bytes, filename
        except (SSRFViolationError, DownloadLimitExceededError) as e:
            raise ImageValidationError(str(e))
        except Exception as e:
            raise ImageValidationError(f"Failed to download image: {str(e)}")

    if image_path:
        if not os.path.isabs(image_path):
            raise ImageValidationError("Local image path must be an absolute path.")
        if not os.path.exists(image_path):
            raise ImageValidationError("Local image file does not exist.")
        if not os.path.isfile(image_path):
            raise ImageValidationError("Local image path is not a file.")

        file_size = os.path.getsize(image_path)
        if file_size > max_bytes:
            raise ImageValidationError(f"Local image too large ({file_size} bytes).")

        try:
            with open(image_path, "rb") as f:
                image_bytes = f.read(max_bytes + 1)
            filename = os.path.basename(image_path)
            return image_bytes, filename
        except Exception as e:
            raise ImageValidationError(f"Failed to read local image: {str(e)}")

    raise ImageValidationError("No image source provided (file, image_url, or image_path).")
