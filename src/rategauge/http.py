"""Shared HTTP client construction.

All outbound requests verify TLS against the OS trust store: corporate
networks commonly intercept TLS with a locally trusted root CA, where
certifi-only verification fails while browsers work. ``truststore`` keeps
verification on and delegates trust to the operating system.
"""

import ssl

import httpx
import truststore

# Browser-like because some official sites (Akamai-fronted federalreserve.gov)
# reject bare client user agents; the rategauge suffix keeps us identifiable.
BROWSER_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0 Safari/537.36 rategauge/0.0.1"
)


def default_client(timeout: float = 60.0, *, browser_headers: bool = False) -> httpx.Client:
    """HTTP client with OS-trust-store TLS verification (gzip on by default)."""
    context = truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    headers = {"User-Agent": BROWSER_USER_AGENT} if browser_headers else None
    return httpx.Client(
        timeout=timeout, follow_redirects=True, verify=context, headers=headers
    )
