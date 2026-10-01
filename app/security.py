"""API authentication for the control plane.

The dashboard used to expose kill / shutdown / reset-account / alert-config
with NO auth on 0.0.0.0 — anyone who found the port could flatten positions
or steal the Telegram token. This module adds a shared-token guard on every
state-changing endpoint (all POST /api/control/*, settings, tunables, alerts).

Token resolution order:
  1. env var CRYPTOMIND_API_TOKEN
  2. a token auto-generated on first run and stored (0600) in api_token.txt

Reads (GET) stay open so the dashboard renders; mutations require the token,
sent as `Authorization: Bearer <token>` or `?token=<token>`. Requests from
loopback (127.0.0.1/::1) are allowed without a token so local ops keep
working behind an authenticated reverse proxy.
"""
from __future__ import annotations
import os
import secrets
import hmac
from fastapi import Request
from fastapi.responses import JSONResponse

TOKEN_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "api_token.txt")
_token: str | None = None


def _load_or_create_token() -> str:
    env = os.environ.get("CRYPTOMIND_API_TOKEN")
    if env:
        return env.strip()
    try:
        if os.path.exists(TOKEN_PATH):
            with open(TOKEN_PATH) as f:
                t = f.read().strip()
            if t:
                return t
    except OSError:
        pass
    t = secrets.token_urlsafe(32)
    try:
        with open(TOKEN_PATH, "w") as f:
            f.write(t)
        os.chmod(TOKEN_PATH, 0o600)
    except OSError:
        pass
    return t


def token() -> str:
    global _token
    if _token is None:
        _token = _load_or_create_token()
    return _token


def _is_loopback(request: Request) -> bool:
    host = request.client.host if request.client else ""
    return host in ("127.0.0.1", "::1", "localhost")


def _presented(request: Request) -> str | None:
    auth = request.headers.get("authorization", "")
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    tok = request.headers.get("x-api-token")
    if tok:
        return tok.strip()
    return request.query_params.get("token")


def check(request: Request) -> bool:
    if os.environ.get("CRYPTOMIND_ALLOW_LOOPBACK", "1") == "1" and _is_loopback(request):
        return True
    presented = _presented(request)
    if not presented:
        return False
    return hmac.compare_digest(presented, token())


# paths that mutate state / touch secrets — everything else is read-only
_PROTECTED_PREFIXES = ("/api/control/", "/api/settings", "/api/tunables",
                       "/api/alerts/config", "/api/alerts/test",
                       "/api/portfolio/migration")


async def auth_middleware(request: Request, call_next):
    path = request.url.path
    method = request.method.upper()
    needs = method in ("POST", "PUT", "DELETE", "PATCH") and \
        any(path.startswith(p) for p in _PROTECTED_PREFIXES)
    if needs and not check(request):
        return JSONResponse(
            {"error": "unauthorized",
             "detail": "This endpoint changes state and requires a token. "
                       "Send Authorization: Bearer <token> (see api_token.txt "
                       "or set CRYPTOMIND_API_TOKEN), or call from localhost."},
            status_code=401)
    return await call_next(request)
