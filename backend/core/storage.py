"""Storage adapters (§5, §27).

Two backends share the StorageBackend protocol:
  LocalFileSystemStorage  - dev mode (STORAGE_ROOT on disk)
  SupabaseStorageBackend  - production (Supabase S3-compatible bucket)

The module-level `storage` singleton is chosen by config at import time.
All application code imports `storage` from here - never the adapter class.
"""

import hashlib
from pathlib import Path
from typing import Protocol, runtime_checkable

import aiofiles
import aiofiles.os

from backend.core.config import settings


@runtime_checkable
class StorageBackend(Protocol):
    async def upload(
        self,
        path: str,
        content: bytes,
        content_type: str = "application/octet-stream",
    ) -> str:
        """Upload bytes to path. Returns the storage path."""
        ...

    async def download(self, path: str) -> bytes:
        """Download and return bytes at path."""
        ...

    async def get_url(self, path: str, expires_in: int = 3600) -> str:
        """Return a URL for the resource (signed URL in prod; local file:// in dev)."""
        ...

    async def delete(self, path: str) -> None:
        """Delete the resource at path."""
        ...

    async def exists(self, path: str) -> bool:
        """Return True if path exists in storage."""
        ...


# The backend package root (parent of core/). A relative STORAGE_ROOT is
# anchored here so uploads and downloads resolve to the same directory
# regardless of the cwd each process (uvicorn, Celery) is launched from.
_BACKEND_ROOT = Path(__file__).parent.parent


class LocalFileSystemStorage:
    """Dev-mode storage that mirrors the Supabase Storage interface."""

    def __init__(self, root: str = settings.STORAGE_ROOT) -> None:
        root_path = Path(root)
        if not root_path.is_absolute():
            root_path = _BACKEND_ROOT / root_path
        self._root = root_path.resolve()
        self._root.mkdir(parents=True, exist_ok=True)

    def _abs(self, path: str) -> Path:
        return self._root / path

    async def upload(
        self,
        path: str,
        content: bytes,
        content_type: str = "application/octet-stream",
    ) -> str:
        target = self._abs(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        async with aiofiles.open(target, "wb") as f:
            await f.write(content)
        return path

    async def download(self, path: str) -> bytes:
        async with aiofiles.open(self._abs(path), "rb") as f:
            return await f.read()

    async def get_url(self, path: str, expires_in: int = 3600) -> str:
        # In dev mode, return the absolute local path as a file:// URL.
        # In production, Supabase returns a time-limited signed URL.
        return self._abs(path).as_uri()

    async def delete(self, path: str) -> None:
        await aiofiles.os.remove(self._abs(path))

    async def exists(self, path: str) -> bool:
        return self._abs(path).exists()

    @staticmethod
    def sha256(content: bytes) -> str:
        return hashlib.sha256(content).hexdigest()


class SupabaseStorageBackend:
    """Production storage using Supabase Storage (S3-compatible, RLS-enforced).

    Requires SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY in settings.
    The service-role key bypasses RLS at the storage layer; RLS on the
    database tables is enforced separately via Postgres policies.
    """

    def __init__(self) -> None:
        from supabase import create_client  # type: ignore[import-untyped]

        self._client = create_client(settings.SUPABASE_URL, settings.SUPABASE_SERVICE_ROLE_KEY)
        self._bucket = settings.SUPABASE_STORAGE_BUCKET

    async def upload(
        self,
        path: str,
        content: bytes,
        content_type: str = "application/octet-stream",
    ) -> str:
        import asyncio

        def _sync_upload() -> None:
            # upsert=True avoids 409 on re-upload
            self._client.storage.from_(self._bucket).upload(
                path=path,
                file=content,
                file_options={"content-type": content_type, "upsert": "true"},
            )

        await asyncio.get_running_loop().run_in_executor(None, _sync_upload)
        return path

    async def download(self, path: str) -> bytes:
        import asyncio

        def _sync_download() -> bytes:
            return self._client.storage.from_(self._bucket).download(path)

        return await asyncio.get_running_loop().run_in_executor(None, _sync_download)

    async def get_url(self, path: str, expires_in: int = 3600) -> str:
        import asyncio

        def _signed() -> str:
            result = self._client.storage.from_(self._bucket).create_signed_url(
                path, expires_in
            )
            return result["signedURL"]

        return await asyncio.get_running_loop().run_in_executor(None, _signed)

    async def delete(self, path: str) -> None:
        import asyncio

        await asyncio.get_running_loop().run_in_executor(
            None, lambda: self._client.storage.from_(self._bucket).remove([path])
        )

    async def exists(self, path: str) -> bool:
        import asyncio

        def _check() -> bool:
            try:
                result = self._client.storage.from_(self._bucket).list(
                    path.rsplit("/", 1)[0] if "/" in path else ""
                )
                filename = path.rsplit("/", 1)[-1]
                return any(f.get("name") == filename for f in (result or []))
            except Exception:
                return False

        return await asyncio.get_running_loop().run_in_executor(None, _check)

    @staticmethod
    def sha256(content: bytes) -> str:
        return hashlib.sha256(content).hexdigest()


def _build_storage() -> StorageBackend:
    if settings.SUPABASE_URL and settings.SUPABASE_SERVICE_ROLE_KEY:
        return SupabaseStorageBackend()
    return LocalFileSystemStorage()


storage: StorageBackend = _build_storage()
