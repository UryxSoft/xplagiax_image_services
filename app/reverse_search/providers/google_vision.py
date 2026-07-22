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

import requests

from app.observability.telemetry import get_logger
from app.reverse_search.models import ProviderMatch
from app.reverse_search.providers.base import (
    ProviderAuthError,
    ProviderRateLimitError,
    ProviderResponseError,
    ProviderTimeoutError,
    ProviderTransientError,
    ReverseSearchProvider,
    hostname_of,
    retry_call,
)

logger = get_logger(__name__)

_ENDPOINT = "https://vision.googleapis.com/v1/images:annotate"

# Vision groups web matches by confidence band, not a numeric score, so we
# map each band to a similarity range ourselves. Within a band, more
# corroborating matches (more independent copies/pages Vision found) push
# the score toward the top of that band instead of a single fixed constant
# — still a heuristic, but one that differentiates "found once" from
# "found on a dozen pages" instead of reporting the exact same number for both.
_SIMILARITY_FULL_MATCH_BASE = 97.0
_SIMILARITY_FULL_MATCH_MAX = 99.9
_SIMILARITY_PARTIAL_MATCH_BASE = 88.0
_SIMILARITY_PARTIAL_MATCH_MAX = 94.0
_SIMILARITY_PAGE_ONLY = 85.0         # page indexed with this image, no graded match list
_SIMILARITY_VISUALLY_SIMILAR = 65.0  # loosely related images only
_CORROBORATION_STEP = 0.5
_CORROBORATION_STEP_CAP = 5          # extra matches beyond this stop adding to the score


class GoogleVisionProvider(ReverseSearchProvider):
    name = "google_vision"

    def __init__(
        self, api_key: str, session: Optional[requests.Session] = None, max_retries: int = 2
    ) -> None:
        self._api_key = api_key
        self._session = session or requests.Session()
        self._max_retries = max_retries

    def search(
        self, *, image_bytes: bytes, image_url: Optional[str], timeout_s: float,
        deadline: Optional[float] = None,
    ) -> Optional[ProviderMatch]:
        return retry_call(
            lambda: self._search_once(image_bytes, timeout_s),
            max_retries=self._max_retries,
            deadline=deadline,
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
                website = page.get("pageTitle") or hostname_of(url)
                if full:
                    similarity = min(
                        _SIMILARITY_FULL_MATCH_MAX,
                        _SIMILARITY_FULL_MATCH_BASE + _CORROBORATION_STEP * min(len(full) - 1, _CORROBORATION_STEP_CAP),
                    )
                elif partial:
                    similarity = min(
                        _SIMILARITY_PARTIAL_MATCH_MAX,
                        _SIMILARITY_PARTIAL_MATCH_BASE + _CORROBORATION_STEP * min(len(partial) - 1, _CORROBORATION_STEP_CAP),
                    )
                else:
                    similarity = _SIMILARITY_PAGE_ONLY
                return ProviderMatch(website=website, url=url, similarity=similarity)

        if similar:
            url = similar[0].get("url", "")
            if url:
                return ProviderMatch(website=hostname_of(url), url=url, similarity=_SIMILARITY_VISUALLY_SIMILAR)

        # Nothing matched our expected shape. This is the normal case for a
        # genuinely-unmatched image, but it's ALSO what an unnoticed contract
        # mismatch (wrong field names) would look like — this response was
        # never validated against a real Vision API key. Logged at DEBUG
        # (silent by default) so setting LOG_LEVEL=DEBUG during a real test
        # immediately shows whether the expected keys were present.
        logger.debug("google_vision_no_match_in_response", web_keys=sorted(web.keys()))
        return None
