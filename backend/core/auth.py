"""Authentication dependency.

Dev mode (DEV_MODE=true): reads the `X-User-Id` header. No JWT verification.
Production: swap the stub block for Supabase JWT verification using
SUPABASE_JWT_SECRET and python-jose. The dependency signature is unchanged
so no router code needs to change.
"""

from fastapi import HTTPException, Request, status

from backend.core.config import settings


async def get_current_user(request: Request) -> str:
    """Return the authenticated user's ID.

    In dev mode this is whatever string is in the X-User-Id header.
    In production this will be the UUID `sub` claim from a Supabase JWT.
    """
    if settings.DEV_MODE:
        user_id = request.headers.get("X-User-Id")
        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="X-User-Id header is required in dev mode (DEV_MODE=true)",
            )
        return user_id

    # ── Production path: Supabase JWT verification ────────────────────────────
    if not settings.SUPABASE_JWT_SECRET:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="SUPABASE_JWT_SECRET is not configured on the server.",
        )

    from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
    from jose import JWTError, jwt

    bearer = HTTPBearer()
    credentials: HTTPAuthorizationCredentials = await bearer(request)
    try:
        payload = jwt.decode(
            credentials.credentials,
            settings.SUPABASE_JWT_SECRET,
            algorithms=["HS256"],
            audience="authenticated",
        )
        user_id = payload.get("sub")
        if not user_id:
            raise HTTPException(status_code=401, detail="Invalid token: missing sub claim")
        return str(user_id)
    except JWTError as exc:
        raise HTTPException(status_code=401, detail=f"Invalid token: {exc}") from exc
