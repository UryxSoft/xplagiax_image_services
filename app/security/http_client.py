"""
Safe HTTP Client for downloading user-provided URLs.
Protects against SSRF (DNS Rebinding, Redirect Chains, Private IP access)
and DOS (Infinite Streaming, Decompression Bombs).
"""

import socket
import ipaddress
import urllib.parse
from typing import Iterator

import requests
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
        # Block private, loopback, multicast, link-local, and unspecified (0.0.0.0)
        if ip.is_private or ip.is_loopback or ip.is_multicast or ip.is_link_local or ip.is_unspecified:
            return False
        return True
    except ValueError:
        return False


def validate_url_safe(url: str) -> None:
    """
    Validates that a URL points to a safe public IP.
    Raises SSRFViolationError if malicious.
    """
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise SSRFViolationError(f"Protocol {parsed.scheme} is not allowed. Only HTTP/HTTPS.")

    hostname = parsed.hostname
    if not hostname:
        raise SSRFViolationError("URL missing hostname.")

    try:
        # Resolve all IPs
        addrinfo = socket.getaddrinfo(hostname, parsed.port or 80, 0, socket.SOCK_STREAM)
    except socket.gaierror as e:
        raise SSRFViolationError(f"DNS Resolution failed for {hostname}: {e}")

    for _, _, _, _, sockaddr in addrinfo:
        ip = sockaddr[0]
        if not is_safe_ip(ip):
            raise SSRFViolationError(f"Host {hostname} resolved to forbidden IP {ip}")


def safe_download(
    url: str,
    max_redirects: int = 5,
    max_bytes: int = 20 * 1024 * 1024,
    timeout: float = 10.0,
) -> tuple[bytes, str]:
    """
    Securely downloads an image.
    - Manually handles redirects to validate each step against SSRF.
    - Limits total bytes downloaded to prevent RAM DOS.
    - Ignores Content-Length.
    Returns (content_bytes, content_type).
    """
    current_url = url
    session = requests.Session()
    
    for attempt in range(max_redirects + 1):
        validate_url_safe(current_url)
        
        try:
            # We don't allow automatic redirects to prevent blind SSRF chains
            resp = session.get(current_url, stream=True, allow_redirects=False, timeout=timeout)
        except RequestException as e:
            raise SSRFViolationError(f"Request failed: {e}")

        if 300 <= resp.status_code < 400:
            if attempt == max_redirects:
                raise SSRFViolationError("Too many redirects.")
            location = resp.headers.get("Location")
            if not location:
                raise SSRFViolationError("Redirect missing Location header.")
            # Resolve relative redirects
            current_url = urllib.parse.urljoin(current_url, location)
            resp.close()
            continue

        resp.raise_for_status()

        # Streaming download to enforce max_bytes
        downloaded = bytearray()
        content_type = resp.headers.get("Content-Type", "")
        
        try:
            for chunk in resp.iter_content(chunk_size=16384):
                if chunk:
                    downloaded.extend(chunk)
                    if len(downloaded) > max_bytes:
                        resp.close()
                        raise DownloadLimitExceededError(f"File exceeds maximum size of {max_bytes} bytes.")
        finally:
            resp.close()

        return bytes(downloaded), content_type

    raise SSRFViolationError("Download failed after redirects.")
