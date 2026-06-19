"""Dataset endpoint tests (§8).

Covers the DELETE endpoint: ownership scoping, 404 semantics, and that a
successful delete removes the row. Datasets are seeded directly via the DB
session so the test exercises the delete path without the upload route's
profiling / plot background tasks.
"""

import pytest
from httpx import AsyncClient


async def _create_project(client: AsyncClient, user: str) -> str:
    res = await client.post(
        "/projects",
        data={"name": "Dataset Delete Test"},
        headers={"X-User-Id": user},
    )
    assert res.status_code == 201
    return res.json()["id"]


async def _seed_dataset(project_id: str, role: str = "inference") -> str:
    from backend.core.database import Dataset, async_session_factory

    async with async_session_factory() as session:
        ds = Dataset(
            project_id=project_id,
            role=role,
            filename="seed.csv",
            storage_path=f"projects/{project_id}/datasets/none/seed.csv",
            file_size_bytes=10,
            sha256="0" * 64,
            schema_hash="schema",
            row_count=1,
            col_count=1,
        )
        session.add(ds)
        await session.commit()
        await session.refresh(ds)
        return ds.id


async def test_delete_dataset(client: AsyncClient) -> None:
    user = "ds-delete-user"
    project_id = await _create_project(client, user)
    dataset_id = await _seed_dataset(project_id)

    delete = await client.delete(
        f"/projects/{project_id}/datasets/{dataset_id}",
        headers={"X-User-Id": user},
    )
    assert delete.status_code == 204

    # Dataset must no longer appear in the project's list.
    listing = await client.get(
        f"/projects/{project_id}/datasets",
        headers={"X-User-Id": user},
    )
    assert listing.status_code == 200
    assert all(d["id"] != dataset_id for d in listing.json())


async def test_delete_dataset_wrong_user(client: AsyncClient) -> None:
    project_id = await _create_project(client, "owner-user")
    dataset_id = await _seed_dataset(project_id)

    delete = await client.delete(
        f"/projects/{project_id}/datasets/{dataset_id}",
        headers={"X-User-Id": "intruder-user"},
    )
    # 404 not 403 - do not reveal existence to non-owners.
    assert delete.status_code == 404


async def test_delete_dataset_not_found(client: AsyncClient) -> None:
    project_id = await _create_project(client, "ds-404-user")

    delete = await client.delete(
        f"/projects/{project_id}/datasets/00000000-0000-0000-0000-000000000000",
        headers={"X-User-Id": "ds-404-user"},
    )
    assert delete.status_code == 404
