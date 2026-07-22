"""
Dependency wiring for the reverse-image-search module.

Pure DI: builds provider adapters + the orchestrator from config. No Flask
here on purpose — this is reused by both the standalone microservice
entrypoint (app/reverse_search/app.py) and the existing monolith
(app/factory.py), and neither of those should have to duplicate this wiring.
"""

from __future__ import annotations

from typing import Optional

import requests
from requests.adapters import HTTPAdapter

from app.config import ReverseSearchConfig
from app.observability.telemetry import get_logger
from app.reverse_search.cache import ReverseSearchCache
from app.reverse_search.orchestrator import ProviderSlot, ReverseSearchOrchestrator
from app.reverse_search.providers.google_vision import GoogleVisionProvider
from app.reverse_search.providers.mungfali import MungfaliProvider
from app.reverse_search.providers.serper_lens import SerperLensProvider
from app.reverse_search.temp_hosting import TempImageHost

logger = get_logger(__name__)


def build_reverse_search_orchestrator(
    config: ReverseSearchConfig, cache_client
) -> Optional[ReverseSearchOrchestrator]:
    """
    Returns None when the feature is disabled or no provider is usable.
    Callers (routes) must handle that as a 503 — never assume a non-None result.
    """
    if not config.enabled:
        return None

    # One pooled, keep-alive session shared by every provider adapter in this
    # process — avoids a fresh TCP/TLS handshake on every external API call.
    session = requests.Session()
    adapter = HTTPAdapter(pool_connections=10, pool_maxsize=20, max_retries=0)
    session.mount("https://", adapter)
    session.mount("http://", adapter)

    slots: list[ProviderSlot] = []

    if config.google_vision.enabled and config.google_vision_api_key:
        slots.append(ProviderSlot(
            name="google_vision",
            provider=GoogleVisionProvider(
                config.google_vision_api_key, session=session, max_retries=config.max_retries
            ),
            priority=config.google_vision.priority,
            stop_threshold=config.google_vision.stop_threshold,
            timeout_s=config.google_vision.timeout_s,
            requires_public_url=config.google_vision.requires_public_url,
        ))

    if config.serper.enabled and config.serper_api_key:
        slots.append(ProviderSlot(
            name="serper",
            provider=SerperLensProvider(
                config.serper_api_key, session=session, max_retries=config.max_retries
            ),
            priority=config.serper.priority,
            stop_threshold=config.serper.stop_threshold,
            timeout_s=config.serper.timeout_s,
            requires_public_url=config.serper.requires_public_url,
        ))

    if config.mungfali.enabled and config.mungfali_api_key:
        slots.append(ProviderSlot(
            name="mungfali",
            provider=MungfaliProvider(config.mungfali_api_key),
            priority=config.mungfali.priority,
            stop_threshold=config.mungfali.stop_threshold,
            timeout_s=config.mungfali.timeout_s,
            requires_public_url=config.mungfali.requires_public_url,
        ))

    if not slots:
        logger.warning(
            "reverse_search_orchestrator_disabled",
            reason="No reverse-image-search provider is both enabled and configured with an API key.",
        )
        return None

    reverse_cache = ReverseSearchCache(
        cache_client, ttl_found=config.cache_ttl_found, ttl_not_found=config.cache_ttl_not_found
    )
    temp_host = TempImageHost(
        cache_client, public_base_url=config.public_base_url, ttl_s=config.temp_hosting_ttl
    ) if any(s.requires_public_url for s in slots) else None

    return ReverseSearchOrchestrator(
        slots=slots, cache=reverse_cache, temp_host=temp_host, max_providers=config.max_providers
    )
