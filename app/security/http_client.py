"""
Safe HTTP Client for downloading user-provided URLs.

Protects against:
  - SSRF (private/loopback/link-local/metadata IPs, protocol smuggling)
  - SSRF via DNS rebinding (TOCTOU): we resolve DNS ONCE, validate EVERY
    resolved IP, and pin the connection to a validated IP so the address used
    for validation is the exact address used for the request. The hostname is
    still used for the Host header and TLS SNI/cert validation.
  - DOS (infinite streaming / oversized bodies): hard byte cap while streaming.
  - Redirect-based SSRF: redirects are followed manually and each hop is
    re-validated and re-pinned.
"""

import ipaddress
import socket
import urllib.parse
from typing import List, Tuple

import requests
from requests.adapters import HTTPAdapter
from requests.exceptions import RequestException

from app.observability.telemetry import get_logger

logger = get_logger(__name__)


class SSRFViolationError(Exception):
    pass


class DownloadLimitExceededError(Exception):
    pass


def is_safe_ip(ip_str: str) -> bool:
    try:
        ip = ipaddress.ip_address(ip_str)
    except ValueError:
        return False
    # Block everything that is not a normal, routable public address.
    return not (
        ip.is_private
        or ip.is_loopback
        or ip.is_multicast
        or ip.is_link_local       # covers 169.254.169.254 cloud metadata
        or ip.is_unspecified
        or ip.is_reserved
    )


def _resolve_and_validate(hostname: str, port: int) -> List[str]:
    """
    Resolve the hostname and return the list of safe IPs.

    Security: if ANY resolved address is unsafe we reject the whole hostname.
    This defeats round-robin / fast-flux rebinding where the attacker mixes a
    public and a private address in the same DNS response.
    """
    try:
        addrinfo = socket.getaddrinfo(hostname, port, 0, socket.SOCK_STREAM)
    except socket.gaierror as exc:
        raise SSRFViolationError(f"DNS resolution failed for {hostname}: {exc}")

    ips = []
    for *_unused, sockaddr in addrinfo:
        ip = sockaddr[0]
        if not is_safe_ip(ip):
            raise SSRFViolationError(
                f"Host {hostname} resolved to forbidden IP {ip}"
            )
        ips.append(ip)

    if not ips:
        raise SSRFViolationError(f"Host {hostname} did not resolve to any IP")
    return ips


class _PinnedAdapter(HTTPAdapter):
    """
    Transport adapter that forces all connections to a single, pre-validated IP
    while preserving the original hostname for TLS SNI and certificate checks.
    """

    def __init__(self, hostname: str, pinned_ip: str, **kwargs):
        self._hostname = hostname
        self._pinned_ip = pinned_ip
        super().__init__(**kwargs)

    def init_poolmanager(self, *args, **kwargs):
        # Validate the cert against the real hostname even though we dial the IP.
        kwargs["server_hostname"] = self._hostname
        kwargs["assert_hostname"] = self._hostname
        super().init_poolmanager(*args, **kwargs)


def _pinned_get(url: str, timeout: float):
    """
    Issue a single GET to `url`, but force the TCP connection to a validated IP.
    Returns the streaming Response (caller must close it).
    """
    parsed = urllib.parse.urlsplit(url)
    hostname = parsed.hostname
    if not hostname:
        raise SSRFViolationError("URL missing hostname.")
    default_port = 443 if parsed.scheme == "https" else 80
    port = parsed.port or default_port

    safe_ips = _resolve_and_validate(hostname, port)
    pinned_ip = safe_ips[0]

    # Rewrite the netloc to the pinned IP; keep Host header = original hostname.
    ip_netloc = f"[{pinned_ip}]" if ":" in pinned_ip else pinned_ip
    if parsed.port:
        ip_netloc = f"{ip_netloc}:{parsed.port}"
    pinned_url = urllib.parse.urlunsplit(
        (parsed.scheme, ip_netloc, parsed.path or "/", parsed.query, "")
    )

    session = requests.Session()
    adapter = _PinnedAdapter(hostname=hostname, pinned_ip=pinned_ip)
    session.mount(f"{parsed.scheme}://{ip_netloc}", adapter)

    headers = {"Host": parsed.netloc, "Accept": "image/*"}
    try:
        return session.get(
            pinned_url,
            headers=headers,
            stream=True,
            allow_redirects=False,
            timeout=timeout,
        )
    except RequestException as exc:
        session.close()
        raise SSRFViolationError(f"Request failed: {exc}")


def validate_url_safe(url: str) -> None:
    """Validate scheme + resolve/validate the host (no rebinding)."""
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme not in ("http", "https"):
        raise SSRFViolationError(
            f"Protocol {parsed.scheme!r} is not allowed. Only HTTP/HTTPS."
        )
    if not parsed.hostname:
        raise SSRFViolationError("URL missing hostname.")
    default_port = 443 if parsed.scheme == "https" else 80
    _resolve_and_validate(parsed.hostname, parsed.port or default_port)


def safe_download(
    url: str,
    max_redirects: int = 5,
    max_bytes: int = 20 * 1024 * 1024,
    timeout: float = 10.0,
) -> Tuple[bytes, str]:
    """
    Securely download an image.
      - Each hop is validated AND pinned to a validated IP (anti-rebinding).
      - Redirects are followed manually so every hop is re-checked.
      - Total bytes are capped while streaming (anti-DOS); Content-Length ignored.
    Returns (content_bytes, content_type).
    """
    current_url = url

    for attempt in range(max_redirects + 1):
        resp = _pinned_get(current_url, timeout=timeout)
        try:
            if 300 <= resp.status_code < 400:
                if attempt == max_redirects:
                    raise SSRFViolationError("Too many redirects.")
                location = resp.headers.get("Location")
                if not location:
                    raise SSRFViolationError("Redirect missing Location header.")
                current_url = urllib.parse.urljoin(current_url, location)
                continue

            resp.raise_for_status()

            content_type = resp.headers.get("Content-Type", "")
            downloaded = bytearray()
            for chunk in resp.iter_content(chunk_size=16384):
                if not chunk:
                    continue
                downloaded.extend(chunk)
                if len(downloaded) > max_bytes:
                    raise DownloadLimitExceededError(
                        f"File exceeds maximum size of {max_bytes} bytes."
                    )
            return bytes(downloaded), content_type
        finally:
            resp.close()

    raise SSRFViolationError("Download failed after redirects.")
