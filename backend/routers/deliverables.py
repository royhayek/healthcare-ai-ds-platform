"""Deliverables endpoints (§4, §23).

GET  /runs/{run_id}/deliverables              - list all deliverables for a run
GET  /runs/{run_id}/deliverables/{name}/download - download one deliverable
POST /runs/{run_id}/deliverables/regenerate   - trigger on-demand regen
"""

from typing import Annotated, Any

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status
from fastapi.responses import Response
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.core.auth import get_current_user
from backend.core.database import Deliverable, Project, Run, get_db
from backend.core.storage import storage

router = APIRouter(prefix="/runs/{run_id}/deliverables", tags=["deliverables"])

_MIME_MAP = {
    "pdf": "application/pdf",
    "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "md": "text/markdown; charset=utf-8",
    "csv": "text/csv; charset=utf-8",
    "json": "application/json; charset=utf-8",
    "yaml": "application/yaml; charset=utf-8",
    "ipynb": "application/x-ipynb+json; charset=utf-8",
    "zip": "application/zip",
}

_SAFE_EXT_MAP = {
    "pdf": "pdf",
    "xlsx": "xlsx",
    "md": "md",
    "csv": "csv",
    "json": "json",
    "yaml": "yaml",
    "ipynb": "ipynb",
    "zip": "zip",
}


async def _get_run_or_404(run_id: str, session: AsyncSession) -> Run:
    run = await session.get(Run, run_id)
    if not run:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Run not found")
    return run


async def _assert_project_access(
    run: Run, user_id: str, session: AsyncSession
) -> None:
    project = await session.get(Project, run.project_id)
    if not project or project.user_id != user_id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied")


@router.get("", summary="List deliverables for a run")
async def list_deliverables(
    run_id: str,
    session: Annotated[AsyncSession, Depends(get_db)],
    user_id: Annotated[str, Depends(get_current_user)],
) -> list[dict[str, Any]]:
    run = await _get_run_or_404(run_id, session)
    await _assert_project_access(run, user_id, session)

    result = await session.execute(
        select(Deliverable)
        .where(Deliverable.run_id == run_id)
        .order_by(Deliverable.generated_at)
    )
    rows = result.scalars().all()

    return [
        {
            "id": d.id,
            "name": d.name,
            "format": d.format,
            "storage_path": d.storage_path,
            "checksum_sha256": d.checksum_sha256,
            "generator_version": d.generator_version,
            "inputs_used": d.inputs_used,
            "audience": d.audience,
            "generated_at": d.generated_at.isoformat() if d.generated_at else None,
        }
        for d in rows
    ]


@router.get("/{name}/download", summary="Download a single deliverable by name")
async def download_deliverable(
    run_id: str,
    name: str,
    session: Annotated[AsyncSession, Depends(get_db)],
    user_id: Annotated[str, Depends(get_current_user)],
) -> Response:
    run = await _get_run_or_404(run_id, session)
    await _assert_project_access(run, user_id, session)

    result = await session.execute(
        select(Deliverable)
        .where(Deliverable.run_id == run_id, Deliverable.name == name)
        .order_by(Deliverable.generated_at.desc())
        .limit(1)
    )
    deliverable = result.scalar_one_or_none()
    if not deliverable:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Deliverable '{name}' not found for run {run_id}",
        )

    try:
        content = await storage.download(deliverable.storage_path)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Storage retrieval failed: {exc}",
        ) from exc

    mime = _MIME_MAP.get(deliverable.format, "application/octet-stream")
    ext = _SAFE_EXT_MAP.get(deliverable.format, deliverable.format)
    filename = f"{name}.{ext}"

    return Response(
        content=content,
        media_type=mime,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.post("/notebook", summary="Generate on-demand Jupyter notebook export")
async def generate_notebook(
    run_id: str,
    session: Annotated[AsyncSession, Depends(get_db)],
    user_id: Annotated[str, Depends(get_current_user)],
) -> dict[str, str]:
    run = await _get_run_or_404(run_id, session)
    await _assert_project_access(run, user_id, session)

    if run.status not in ("completed", "failed", "running", "awaiting_checkpoint"):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Run is in status '{run.status}' - pipeline must have started before exporting a notebook",
        )

    from backend.tasks.deliverable_task import generate_notebook_export_task

    task = generate_notebook_export_task.delay(run_id)
    return {"message": "Notebook generation queued", "task_id": task.id}


@router.post("/regenerate", summary="Trigger on-demand deliverable regeneration")
async def regenerate_deliverables(
    run_id: str,
    session: Annotated[AsyncSession, Depends(get_db)],
    user_id: Annotated[str, Depends(get_current_user)],
) -> dict[str, str]:
    run = await _get_run_or_404(run_id, session)
    await _assert_project_access(run, user_id, session)

    if run.status not in ("completed", "failed"):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Run is in status '{run.status}' - cannot regenerate until complete",
        )

    from backend.tasks.deliverable_task import generate_deliverables_task

    task = generate_deliverables_task.delay(run_id)

    return {"message": "Deliverable regeneration queued", "task_id": task.id}
