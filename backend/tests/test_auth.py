"""Tests for the auth dependency (backend/core/auth.py).

Covers the dev-mode header path and the production JWKS/ES256 verification path.
The production path is the one that actually broke in practice: Supabase signs
access tokens with asymmetric ES256 keys, so verification must use the project
JWKS public key, not the legacy HS256 shared secret.
"""

from __future__ import annotations

import time

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec
from fastapi import HTTPException
from jose import jwt
from jose.utils import long_to_base64

import backend.core.auth as auth

ISSUER_BASE = "https://test-project.supabase.co"
ISSUER = f"{ISSUER_BASE}/auth/v1"
KID = "test-kid"


class FakeRequest:
    """Minimal stand-in for starlette Request: only headers are read here."""

    def __init__(self, headers: dict[str, str]):
        self.headers = headers


def _make_keypair_and_jwk() -> tuple[bytes, dict]:
    """Return (private PEM, public JWK) for a fresh EC P-256 key."""
    priv = ec.generate_private_key(ec.SECP256R1())
    nums = priv.public_key().public_numbers()
    jwk = {
        "kty": "EC", "crv": "P-256", "use": "sig", "alg": "ES256", "kid": KID,
        "x": long_to_base64(nums.x, size=32).decode(),
        "y": long_to_base64(nums.y, size=32).decode(),
    }
    pem = priv.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    return pem, jwk


def _sign(pem: bytes, **claims) -> str:
    payload = {
        "sub": "user-123",
        "aud": "authenticated",
        "iss": ISSUER,
        "exp": int(time.time()) + 3600,
    }
    payload.update(claims)
    return jwt.encode(payload, pem, algorithm="ES256", headers={"kid": KID})


@pytest.fixture
def prod_auth(monkeypatch):
    """Put auth in production mode with a patched JWKS fetch; yield the signer."""
    pem, jwk = _make_keypair_and_jwk()
    monkeypatch.setattr(auth.settings, "DEV_MODE", False)
    monkeypatch.setattr(auth.settings, "SUPABASE_URL", ISSUER_BASE)

    async def fake_fetch() -> dict[str, dict]:
        return {KID: jwk}

    monkeypatch.setattr(auth, "_fetch_jwks", fake_fetch)
    monkeypatch.setattr(auth, "_jwks_keys_by_kid", {})
    monkeypatch.setattr(auth, "_jwks_fetched_at", 0.0)
    return pem


@pytest.mark.asyncio
async def test_dev_mode_uses_x_user_id_header(monkeypatch):
    monkeypatch.setattr(auth.settings, "DEV_MODE", True)
    uid = await auth.get_current_user(FakeRequest({"X-User-Id": "dev-user-1"}))
    assert uid == "dev-user-1"


@pytest.mark.asyncio
async def test_dev_mode_missing_header_401(monkeypatch):
    monkeypatch.setattr(auth.settings, "DEV_MODE", True)
    with pytest.raises(HTTPException) as exc:
        await auth.get_current_user(FakeRequest({}))
    assert exc.value.status_code == 401


@pytest.mark.asyncio
async def test_prod_valid_es256_token_returns_sub(prod_auth):
    token = _sign(prod_auth)
    uid = await auth.get_current_user(
        FakeRequest({"Authorization": f"Bearer {token}"})
    )
    assert uid == "user-123"


@pytest.mark.asyncio
async def test_prod_tampered_token_rejected(prod_auth):
    token = _sign(prod_auth)
    bad = FakeRequest({"Authorization": f"Bearer {token[:-3]}abc"})
    with pytest.raises(HTTPException) as exc:
        await auth.get_current_user(bad)
    assert exc.value.status_code == 401


@pytest.mark.asyncio
async def test_prod_wrong_audience_rejected(prod_auth):
    token = _sign(prod_auth, aud="anon")
    with pytest.raises(HTTPException) as exc:
        await auth.get_current_user(
            FakeRequest({"Authorization": f"Bearer {token}"})
        )
    assert exc.value.status_code == 401


@pytest.mark.asyncio
async def test_prod_unknown_kid_rejected(prod_auth, monkeypatch):
    # JWKS has no key for this kid -> verification must fail with 401.
    async def empty_fetch() -> dict[str, dict]:
        return {}

    monkeypatch.setattr(auth, "_fetch_jwks", empty_fetch)
    monkeypatch.setattr(auth, "_jwks_keys_by_kid", {})
    monkeypatch.setattr(auth, "_jwks_fetched_at", 0.0)
    token = _sign(prod_auth)
    with pytest.raises(HTTPException) as exc:
        await auth.get_current_user(
            FakeRequest({"Authorization": f"Bearer {token}"})
        )
    assert exc.value.status_code == 401
