import pytest
from httpx import AsyncClient


async def test_health_ok(client: AsyncClient) -> None:
    response = await client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["database"] == "connected"


async def test_create_project(client: AsyncClient) -> None:
    response = await client.post(
        "/projects",
        data={"name": "Telco Churn Analysis", "description": "Step 1 smoke test"},
        headers={"X-User-Id": "dev-user-1"},
    )
    assert response.status_code == 201
    body = response.json()
    assert body["name"] == "Telco Churn Analysis"
    assert body["user_id"] == "dev-user-1"
    assert "id" in body
    assert body["description"] == "Step 1 smoke test"


async def test_list_projects(client: AsyncClient) -> None:
    # Seed a project for this test's user
    await client.post(
        "/projects",
        data={"name": "List Test Project"},
        headers={"X-User-Id": "list-test-user"},
    )

    response = await client.get("/projects", headers={"X-User-Id": "list-test-user"})
    assert response.status_code == 200
    body = response.json()
    assert isinstance(body, list)
    assert any(p["name"] == "List Test Project" for p in body)


async def test_list_projects_scoped_to_user(client: AsyncClient) -> None:
    # Projects from user-A must not appear in user-B's list
    await client.post(
        "/projects",
        data={"name": "User A Project"},
        headers={"X-User-Id": "user-a"},
    )

    response = await client.get("/projects", headers={"X-User-Id": "user-b"})
    assert response.status_code == 200
    body = response.json()
    assert not any(p["name"] == "User A Project" for p in body)


async def test_get_project_not_found(client: AsyncClient) -> None:
    response = await client.get(
        "/projects/00000000-0000-0000-0000-000000000000",
        headers={"X-User-Id": "dev-user-1"},
    )
    assert response.status_code == 404


async def test_auth_required(client: AsyncClient) -> None:
    response = await client.get("/projects")
    assert response.status_code == 401


async def test_delete_project(client: AsyncClient) -> None:
    create = await client.post(
        "/projects",
        data={"name": "To Be Deleted"},
        headers={"X-User-Id": "delete-test-user"},
    )
    assert create.status_code == 201
    project_id = create.json()["id"]

    delete = await client.delete(
        f"/projects/{project_id}",
        headers={"X-User-Id": "delete-test-user"},
    )
    assert delete.status_code == 204

    # Project must no longer be accessible
    get = await client.get(
        f"/projects/{project_id}",
        headers={"X-User-Id": "delete-test-user"},
    )
    assert get.status_code == 404


async def test_delete_project_wrong_user(client: AsyncClient) -> None:
    create = await client.post(
        "/projects",
        data={"name": "Protected Project"},
        headers={"X-User-Id": "owner-user"},
    )
    project_id = create.json()["id"]

    delete = await client.delete(
        f"/projects/{project_id}",
        headers={"X-User-Id": "different-user"},
    )
    assert delete.status_code == 404  # 404 not 403 - don't reveal existence


async def test_delete_project_not_found(client: AsyncClient) -> None:
    response = await client.delete(
        "/projects/00000000-0000-0000-0000-000000000000",
        headers={"X-User-Id": "dev-user-1"},
    )
    assert response.status_code == 404
