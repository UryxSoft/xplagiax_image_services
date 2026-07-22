"""
Serper.dev — Google Lens endpoint.

IMPORTANT: this deliberately targets https://google.serper.dev/lens, NOT
Serper's plain /search endpoint. /search is a text web-search API and cannot
do reverse image lookup at all — it was easy to mix the two up because both
live under the same account/API key. /lens mirrors SerpApi's google_lens
engine: give it a *publicly reachable* image URL and it returns the visual
matches Google Lens found.

Because this endpoint needs a URL (not inline bytes), the orchestrator only
calls this provider after hosting the image via TempImageHost — see
orchestrator.py. Serper does not expose a numeric similarity for Lens
matches, so — like Vision — we assign a fixed confidence to the top match.
"""

from __future__ import annotations

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

_ENDPOINT = "https://google.serper.dev/lens"
# Serper Lens ranks by relevance but gives no numeric score. More independent
# organic matches for the same image is itself corroborating evidence, so we
# scale the top match's similarity by how many results came back instead of
# reporting the same fixed number regardless.
_SIMILARITY_TOP_MATCH_BASE = 88.0
_SIMILARITY_TOP_MATCH_MAX = 96.0
_CORROBORATION_STEP = 1.0
_CORROBORATION_STEP_CAP = 8


class SerperLensProvider(ReverseSearchProvider):
    name = "serper"

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
        if not image_url:
            # Should never happen: the orchestrator hosts the image before
            # calling any provider with requires_public_url=True.
            raise ProviderResponseError(self.name, "no public image URL available")
        return retry_call(
            lambda: self._search_once(image_url, timeout_s),
            max_retries=self._max_retries,
            deadline=deadline,
        )

    def _search_once(self, image_url: str, timeout_s: float) -> Optional[ProviderMatch]:
        try:
            resp = self._session.post(
                _ENDPOINT,
                headers={"X-API-KEY": self._api_key, "Content-Type": "application/json"},
                json={"url": image_url},
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

        matches = payload.get("organic") or []
        if not matches:
            logger.debug("serper_lens_no_match_in_response", response_keys=sorted(payload.keys()))
            return None
        top = matches[0]
        url = top.get("link") or ""
        if not url:
            return None
        website = top.get("title") or top.get("source") or hostname_of(url)
        similarity = min(
            _SIMILARITY_TOP_MATCH_MAX,
            _SIMILARITY_TOP_MATCH_BASE + _CORROBORATION_STEP * min(len(matches) - 1, _CORROBORATION_STEP_CAP),
        )
        return ProviderMatch(website=website, url=url, similarity=similarity)
