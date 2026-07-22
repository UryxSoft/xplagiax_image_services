"""
Mungfali — EXPERIMENTAL / UNVERIFIED adapter, disabled by default.

Mungfali's public product documentation (mungfali.net) describes a keyword /
stock-photo image search API ("search FOR images matching a text query"),
not a confirmed "upload this image, tell me where it appears online"
reverse-search endpoint. We could not verify a request/response contract for
actual image-to-source lookup, and shipping a guessed API shape here would
risk silently calling the wrong product (or a nonexistent endpoint) in
production.

This class exists as a structurally-complete extension point: it satisfies
the ReverseSearchProvider contract so the orchestrator can treat it like any
other provider in the chain, but search() always raises
ProviderUnavailableError — the orchestrator logs that and moves on to the
next provider, exactly as it would for a provider that's temporarily down.
MUNGFALI_ENABLED defaults to false so this never runs unless explicitly
turned on after the contract below is filled in.

To finish this integration: confirm with Mungfali's dashboard/docs whether
they expose an image-upload (or image-URL) reverse-search endpoint, what
auth scheme it uses, and the response schema — then implement _search_once()
the same way google_vision.py / serper_lens.py do.
"""

from __future__ import annotations

from typing import Optional

from app.reverse_search.models import ProviderMatch
from app.reverse_search.providers.base import ProviderUnavailableError, ReverseSearchProvider


class MungfaliProvider(ReverseSearchProvider):
    name = "mungfali"

    def __init__(self, api_key: str) -> None:
        self._api_key = api_key

    def search(
        self, *, image_bytes: bytes, image_url: Optional[str], timeout_s: float,
        deadline: Optional[float] = None,
    ) -> Optional[ProviderMatch]:
        raise ProviderUnavailableError(
            self.name,
            "Mungfali adapter is a template — its documented public API is a "
            "keyword image search product, not a verified reverse-image-search "
            "endpoint. Confirm the real contract with the vendor before enabling.",
        )
