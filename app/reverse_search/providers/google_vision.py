"""
Google Vision API — WEB_DETECTION feature.

True reverse-image-search from Google: given raw image bytes (base64 inline,
NO public URL needed — the single biggest reason this is priority #1 by
default), Vision returns pages containing matching images, grouped into
confidence bands (full match / partial match / visually similar).

Vision does not return a numeric similarity score for web matches, so we map
its confidence bands to a fixed similarity percentage ourselves. This is a
deliberate, documented heuristic, not a claim that Google computed these
exact numbers — see _best_match() below.
"""

from __future__ import annotations

import base64
from typing import Optional
from urllib.parse import urlparse

import requests

from app.reverse_search.models import ProviderMatch
from app.reverse_search.providers.base import (
    ProviderAuthError,
    ProviderRateLimitError,
    ProviderResponseError,
    ProviderTimeoutError,
    ProviderTransientError,
    ReverseSearchProvider,
    retry_call,
)

_ENDPOINT = "https://vision.googleapis.com/v1/images:annotate"

# Vision groups web matches by confidence band, not a numeric score. Mapping
# each band to a fixed similarity lets it be compared against configurable
# stop_threshold values the same way as every other provider.
_SIMILARITY_FULL_MATCH = 99.0        # identical / near-identical image found verbatim
_SIMILARITY_PARTIAL_MATCH = 90.0     # cropped / edited / recoloured / watermarked
_SIMILARITY_PAGE_ONLY = 85.0         # page indexed with this image, no graded match list
_SIMILARITY_VISUALLY_SIMILAR = 65.0  # loosely related images only


class GoogleVisionProvider(ReverseSearchProvider):
    name = "google_vision"

    def __init__(
        self, api_key: str, session: Optional[requests.Session] = None, max_retries: int = 2
    ) -> None:
        self._api_key = api_key
        self._session = session or requests.Session()
        self._max_retries = max_retries

    def search(
        self, *, image_bytes: bytes, image_url: Optional[str], timeout_s: float
    ) -> Optional[ProviderMatch]:
        return retry_call(
            lambda: self._search_once(image_bytes, timeout_s),
            max_retries=self._max_retries,
        )

    def _search_once(self, image_bytes: bytes, timeout_s: float) -> Optional[ProviderMatch]:
        body = {
            "requests": [{
                "image": {"content": base64.b64encode(image_bytes).decode("ascii")},
                "features": [{"type": "WEB_DETECTION", "maxResults": 5}],
            }]
        }
        try:
            resp = self._session.post(
                _ENDPOINT,
                params={"key": self._api_key},
                json=body,
                timeout=(timeout_s, timeout_s),
            )
        except requests.exceptions.Timeout as exc:
            raise ProviderTimeoutError(self.name) from exc
        except requests.exceptions.RequestException as exc:
            raise ProviderTransientError(self.name, 0) from exc

        if resp.status_code in (401, 403):
            raise ProviderAuthError(self.name, resp.status_code)
        if resp.status_code == 429:
            retry_after = resp.headers.get("Retry-After")
            raise ProviderRateLimitError(self.name, retry_after=float(retry_after) if retry_after else None)
        if resp.status_code in (500, 502, 503, 504):
            raise ProviderTransientError(self.name, resp.status_code)
        if resp.status_code == 404:
            return None
        if resp.status_code != 200:
            raise ProviderResponseError(self.name, f"unexpected HTTP {resp.status_code}: {resp.text[:200]}")

        try:
            payload = resp.json()
        except ValueError as exc:
            raise ProviderResponseError(self.name, "invalid JSON response") from exc

        responses = payload.get("responses") or [{}]
        web = (responses[0] or {}).get("webDetection") or {}
        return self._best_match(web)

    def _best_match(self, web: dict) -> Optional[ProviderMatch]:
        full = web.get("fullMatchingImages") or []
        partial = web.get("partialMatchingImages") or []
        pages = web.get("pagesWithMatchingImages") or []
        similar = web.get("visuallySimilarImages") or []

        if pages:
            page = pages[0]
            url = page.get("url", "")
            if url:
                website = page.get("pageTitle") or _hostname(url)
                if full:
                    similarity = _SIMILARITY_FULL_MATCH
                elif partial:
                    similarity = _SIMILARITY_PARTIAL_MATCH
                else:
                    similarity = _SIMILARITY_PAGE_ONLY
                return ProviderMatch(website=website, url=url, similarity=similarity)

        if similar:
            url = similar[0].get("url", "")
            if url:
                return ProviderMatch(website=_hostname(url), url=url, similarity=_SIMILARITY_VISUALLY_SIMILAR)

        return None


def _hostname(url: str) -> str:
    try:
        return urlparse(url).hostname or url
    except ValueError:
        return url
