"""Test configuration.

Sets DATABASE_URL before any backend module is imported so pydantic-settings
picks up aids_test as the target database.

NullPool strategy:
  The module-level SQLAlchemy engine is replaced with a NullPool variant after
  the test database is created. NullPool never caches connections, so there are
  no event-loop boundary issues when pytest-asyncio gives each test its own loop.
  This is the standard pattern for testing async SQLAlchemy apps.
"""

import asyncio
import os
from urllib.parse import urlsplit

# Host/port for the test Postgres. Defaults to the repo's docker-compose mapping
# (localhost:5433), and can be overridden with TEST_DB_HOST / TEST_DB_PORT (or by
# pointing TEST_DB_URL_BASE at any aids-owned Postgres). The previous hard-coded
# 5432 did not match docker-compose.yml, so the suite could not find the DB.
_BASE = os.environ.get(
    "TEST_DB_URL_BASE",
    "postgresql+asyncpg://aids:aids@localhost:5433",
)
_parts = urlsplit(_BASE)
_HOST = os.environ.get("TEST_DB_HOST", _parts.hostname or "localhost")
_PORT = os.environ.get("TEST_DB_PORT", str(_parts.port or 5433))
_USER = _parts.username or "aids"
_PASS = _parts.password or "aids"
_TEST_DB = "aids_test"
_TEST_DB_URL = f"postgresql+asyncpg://{_USER}:{_PASS}@{_HOST}:{_PORT}/{_TEST_DB}"
_ADMIN_DSN = f"postgresql://{_USER}:{_PASS}@{_HOST}:{_PORT}/postgres"

# Must be first - before any backend import - so pydantic-settings picks it up.
os.environ.setdefault("DATABASE_URL", _TEST_DB_URL)
os.environ.setdefault("DEV_MODE", "true")

import asyncpg  # noqa: E402
import pytest  # noqa: E402
from httpx import ASGITransport, AsyncClient  # noqa: E402
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine  # noqa: E402
from sqlalchemy.pool import NullPool  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def create_test_database() -> None:
    """Create aids_test, patch the engine to NullPool, init schema, tear down."""

    async def _create() -> None:
        conn = await asyncpg.connect(_ADMIN_DSN)
        try:
            await conn.execute(f"DROP DATABASE IF EXISTS {_TEST_DB} WITH (FORCE)")
            await conn.execute(f"CREATE DATABASE {_TEST_DB}")
        finally:
            await conn.close()

    asyncio.run(_create())

    # Patch the module-level engine BEFORE any router or endpoint code runs.
    # NullPool: every DB call creates and closes a connection immediately -
    # safe across multiple function-scoped event loops.
    from backend.core import database as db_module

    test_engine = create_async_engine(_TEST_DB_URL, poolclass=NullPool)
    db_module.engine = test_engine
    db_module.async_session_factory = async_sessionmaker(
        test_engine, class_=AsyncSession, expire_on_commit=False
    )

    async def _init() -> None:
        from backend.core.database import init_db

        await init_db()

    asyncio.run(_init())

    yield

    async def _teardown() -> None:
        await test_engine.dispose()
        conn = await asyncpg.connect(_ADMIN_DSN)
        try:
            await conn.execute(f"DROP DATABASE IF EXISTS {_TEST_DB} WITH (FORCE)")
        finally:
            await conn.close()

    asyncio.run(_teardown())


@pytest.fixture
async def client(create_test_database: None) -> AsyncClient:  # type: ignore[misc]
    from backend.main import app

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as c:
        yield c
