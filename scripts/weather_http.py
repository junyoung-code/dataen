"""Bounded HTTPS reads with per-thread persistent connections for public weather files."""
import http.client
import threading
import urllib.error
from urllib.parse import urlsplit

_LOCAL = threading.local()


def get_bytes(url, *, headers=None, status=200, expected_headers=None, max_bytes=50 * 1024**2):
    parts = urlsplit(url)
    if parts.scheme != "https" or parts.username or parts.password or parts.fragment:
        raise ValueError("Only credential-free HTTPS object URLs are supported")
    connections = getattr(_LOCAL, "connections", None)
    if connections is None:
        connections = _LOCAL.connections = {}
    host = (parts.hostname, parts.port or 443)
    path = parts.path + ("?" + parts.query if parts.query else "")
    for attempt in range(3):
        connection = connections.get(host)
        if connection is None:
            connection = connections[host] = http.client.HTTPSConnection(*host, timeout=90)
        try:
            connection.request("GET", path, headers=headers or {})
            response = connection.getresponse()
            if response.status >= 400:
                raise urllib.error.HTTPError(url, response.status, response.reason, response.headers, None)
            if response.status != status:
                raise ValueError("Unexpected HTTP status; refusing response body")
            for key, value in (expected_headers or {}).items():
                if response.getheader(key) != value:
                    raise ValueError(f"Unexpected {key}; refusing response body")
            size = int(response.getheader("Content-Length"))
            if size > max_bytes or size < 0:
                raise ValueError("Unexpected response size; refusing response body")
            content = response.read(size + 1)
            if len(content) != size:
                raise OSError("Incomplete HTTP object")
            return content, {k.lower(): v for k, v in response.getheaders()}
        except Exception as error:
            connection.close()
            connections.pop(host, None)
            transient = isinstance(error, (OSError, http.client.HTTPException))
            if isinstance(error, urllib.error.HTTPError):
                transient = error.code in (429, 500, 502, 503, 504)
            if not transient or attempt == 2:
                raise
