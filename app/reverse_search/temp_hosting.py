"""
Ephemeral public exposure for uploaded image bytes.

Some reverse-image-search providers (e.g. Serper's Lens endpoint) fetch the
image themselves from a URL and don't accept inline bytes. Rather than
standing up object storage for a few KB that need to live for two minutes,
we serve them directly from this process via a short-TTL, single-use,
unguessable token stored in Redis.

This path is only exercised when a URL-based provider is actually reached.
If an inline-bytes provider (Google Vision) already returns a confident
match, hosting is never created — zero extra cost in the common case.
"""

from __future__ import annotations

import secrets
from typing import Optional, Tuple
from urllib.parse import urlparse

_KEY_PREFIX = "reverse_search_tmp"
_DEFAULT_CONTENT_TYPE = "application/octet-stream"


class TempHostingUnavailableError(Exception):
    """A URL-based provider is enabled but there's no way to actually serve
    the bytes (missing REVERSE_SEARCH_PUBLIC_BASE_URL and/or Redis is down)."""


class TempImageHost:
    def __init__(self, cache_client, public_base_url: Optional[str], ttl_s: int) -> None:
        self._cache = cache_client  # CacheClient (Redis-backed, graceful degradation)
        self._public_base_url = (public_base_url or "").rstrip("/")
        self._ttl_s = ttl_s

    @property
    def available(self) -> bool:
        return bool(self._cache and self._cache.available and self._public_base_url)

    def host(self, image_bytes: bytes, content_type: str = _DEFAULT_CONTENT_TYPE) -> str:
        """Store the bytes (+ their real content type, so the provider's
        crawler sees a proper image/* response) under a random one-time
        token; return the public URL."""
        if not self.available:
            raise TempHostingUnavailableError(
                "No REVERSE_SEARCH_PUBLIC_BASE_URL / Redis configured — cannot "
                "expose the image to a URL-based provider."
            )
        token = secrets.token_urlsafe(24)
        self._cache.set_bytes(f"{_KEY_PREFIX}:{token}", image_bytes, self._ttl_s)
        self._cache.set(f"{_KEY_PREFIX}:{token}:ct", content_type, self._ttl_s)
        return f"{self._public_base_url}/api/v1/_tmp-image/{token}"

    def fetch_and_consume(self, token: str) -> Optional[Tuple[bytes, str]]:
        """Serve-once: return (bytes, content_type) and delete both keys
        immediately, so the window an image is publicly fetchable is as
        short as possible."""
        key = f"{_KEY_PREFIX}:{token}"
        data = self._cache.get_bytes(key)
        if data is None:
            return None
        content_type = self._cache.get(f"{key}:ct") or _DEFAULT_CONTENT_TYPE
        self._cache.delete(key)
        self._cache.delete(f"{key}:ct")
        return data, content_type

    def warn_if_unreachable(self, logger) -> None:
        """Heads-up if the configured base URL looks internal/loopback — mirrors
        the existing check for the legacy reverse-search feature: providers can
        never fetch a localhost/private URL, so this would silently return
        nothing without an explicit signal in the logs."""
        if not self._public_base_url:
            return
        host = (urlparse(self._public_base_url).hostname or "").lower()
        internal = (
            host in ("localhost", "127.0.0.1", "::1", "")
            or host.endswith(".local") or host.endswith(".internal")
            or host.startswith(("10.", "192.168.", "172."))
        )
        if internal:
            logger.warning(
                "reverse_search_public_url_not_public",
                host=host,
                hint="URL-based providers cannot fetch internal URLs. Set "
                     "REVERSE_SEARCH_PUBLIC_BASE_URL to a real public host.",
            )
