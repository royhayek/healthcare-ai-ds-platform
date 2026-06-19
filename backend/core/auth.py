"""Authentication dependency.

Dev mode (DEV_MODE=true): reads the `X-User-Id` header. No JWT verification.

Production (DEV_MODE=false): verifies the Supabase access token. Supabase issues
asymmetric (ES256, sometimes RS256) access tokens signed with per-project keys,
so verification uses the project's public JWKS at
`{SUPABASE_URL}/auth/v1/.well-known/jwks.json` — NOT the legacy HS256
`SUPABASE_JWT_SECRET`. The key set is cached in-process and refreshed when a
token references an unknown `kid` (key rotation) or the soft TTL elapses.

The dependency signature is unchanged so no router code needs to change.
"""

from __future__ import annotations

import asyncio
import time

import httpx
from fastapi import HTTPException, Request, status

from backend.core.config import settings

# Algorithms we accept from the JWKS. Fixed allowlist (never taken from the
# token header) to avoid algorithm-confusion attacks. Supabase signs access
# tokens with ES256; RS256 is included for projects configured that way.
_ALLOWED_ALGS = ["ES256", "RS256"]

# In-process JWKS cache. Signing keys rotate rarely; refetch only on a `kid`
# miss or once the soft TTL expires.
_JWKS_TTL_SECONDS = 600.0
_jwks_keys_by_kid: dict[str, dict] = {}
_jwks_fetched_at: float = 0.0
_jwks_lock = asyncio.Lock()


def _jwks_url() -> str:
    base = settings.SUPABASE_URL.rstrip("/")
    if not base:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="SUPABASE_URL is not configured on the server.",
        )
    return f"{base}/auth/v1/.well-known/jwks.json"


async def _fetch_jwks() -> dict[str, dict]:
    """Fetch the project JWKS and index it by key id."""
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.get(_jwks_url())
        resp.raise_for_status()
        data = resp.json()
    return {k["kid"]: k for k in data.get("keys", []) if "kid" in k}


async def _get_signing_key(kid: str) -> dict:
    """Return the JWK for ``kid``, refreshing the cache on a miss or stale TTL."""
    global _jwks_keys_by_kid, _jwks_fetched_at

    now = time.monotonic()
    if kid in _jwks_keys_by_kid and (now - _jwks_fetched_at) < _JWKS_TTL_SECONDS:
        return _jwks_keys_by_kid[kid]

    async with _jwks_lock:
        # Re-check: another coroutine may have refreshed while we waited.
        now = time.monotonic()
        if kid in _jwks_keys_by_kid and (now - _jwks_fetched_at) < _JWKS_TTL_SECONDS:
            return _jwks_keys_by_kid[kid]
        try:
            keys = await _fetch_jwks()
        except (httpx.HTTPError, ValueError) as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Could not fetch Supabase JWKS: {exc}",
            ) from exc
        _jwks_keys_by_kid = keys
        _jwks_fetched_at = now

    if kid not in _jwks_keys_by_kid:
        raise HTTPException(status_code=401, detail="Unknown token key id (kid).")
    return _jwks_keys_by_kid[kid]


async def get_current_user(request: Request) -> str:
    """Return the authenticated user's ID.

    In dev mode this is whatever string is in the X-User-Id header. In production
    it is the UUID `sub` claim from a verified Supabase JWT.
    """
    if settings.DEV_MODE:
        user_id = request.headers.get("X-User-Id")
        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="X-User-Id header is required in dev mode (DEV_MODE=true)",
            )
        return user_id

    # ── Production path: Supabase JWKS verification ───────────────────────────
    from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
    from jose import JWTError, jwt

    bearer = HTTPBearer()
    credentials: HTTPAuthorizationCredentials = await bearer(request)
    token = credentials.credentials

    try:
        header = jwt.get_unverified_header(token)
    except JWTError as exc:
        raise HTTPException(status_code=401, detail=f"Malformed token: {exc}") from exc

    kid = header.get("kid")
    if not kid:
        raise HTTPException(status_code=401, detail="Token missing key id (kid).")

    signing_key = await _get_signing_key(kid)
    issuer = f"{settings.SUPABASE_URL.rstrip('/')}/auth/v1"
    try:
        payload = jwt.decode(
            token,
            signing_key,
            algorithms=_ALLOWED_ALGS,
            audience="authenticated",
            issuer=issuer,
        )
    except JWTError as exc:
        raise HTTPException(status_code=401, detail=f"Invalid token: {exc}") from exc

    user_id = payload.get("sub")
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid token: missing sub claim")
    return str(user_id)
