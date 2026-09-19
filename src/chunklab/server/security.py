"""Localhost-only guards for the single-user web UI.

``chunklab ui`` binds to 127.0.0.1, but a page in the user's browser can still
reach it: a hostile site can point a DNS name at 127.0.0.1 (DNS rebinding) or
simply submit a cross-origin form. Both are blocked here, before any route
runs.

This is a pure ASGI middleware rather than ``BaseHTTPMiddleware`` so that
responses stream straight through instead of being buffered.
"""

from __future__ import annotations

from collections.abc import Iterable

from starlette.types import ASGIApp, Message, Receive, Scope, Send

DEFAULT_ALLOWED_HOSTS = frozenset({"127.0.0.1", "localhost"})

#: Methods that cannot change state, so a cross-origin one is harmless: the
#: browser's same-origin policy already keeps the attacker from reading it.
SAFE_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})

#: ``none`` is a user-initiated navigation (address bar, bookmark).
SAME_SITE_VALUES = frozenset({"same-origin", "none"})


def _hostname(value: str) -> str:
    """Host of ``value`` (a Host header or an Origin URL), without the port."""
    host = value.split("://", 1)[-1].split("/", 1)[0].strip()
    if host.startswith("["):  # IPv6 literal: [::1]:7860
        return host[1 : host.index("]")] if "]" in host else host
    return host.rsplit(":", 1)[0] if ":" in host else host


class LocalOnlyMiddleware:
    def __init__(self, app: ASGIApp, allowed_hosts: Iterable[str] | None = None) -> None:
        self.app = app
        self.allowed_hosts = frozenset(allowed_hosts or DEFAULT_ALLOWED_HOSTS)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        headers = {k.decode("latin-1").lower(): v.decode("latin-1") for k, v in scope["headers"]}
        denial = self._denial(scope, headers)
        if denial is not None:
            await _plain_text(send, *denial)
            return
        await self.app(scope, receive, send)

    def _denial(self, scope: Scope, headers: dict[str, str]) -> tuple[int, str] | None:
        if _hostname(headers.get("host", "")) not in self.allowed_hosts:
            return 421, "host not allowed"
        if scope["method"] in SAFE_METHODS:
            return None
        fetch_site = headers.get("sec-fetch-site")
        if fetch_site is not None:
            if fetch_site not in SAME_SITE_VALUES:
                return 403, "cross-origin request not allowed"
            return None
        origin = headers.get("origin")
        if origin is not None and _hostname(origin) not in self.allowed_hosts:
            return 403, "cross-origin request not allowed"
        return None


async def _plain_text(send: Send, status: int, body: str) -> None:
    payload = body.encode()
    start: Message = {
        "type": "http.response.start",
        "status": status,
        "headers": [
            (b"content-type", b"text/plain; charset=utf-8"),
            (b"content-length", str(len(payload)).encode()),
        ],
    }
    await send(start)
    await send({"type": "http.response.body", "body": payload})
