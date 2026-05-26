"""Analysis run endpoints (§22, §26).

POST /projects/{project_id}/runs  - create a run and enqueue the pipeline task
GET  /projects/{project_id}/runs  - list runs for a project
GET  /runs/{run_id}               - run status + all pipeline outputs
POST /runs/{run_id}/resume        - resume from checkpoint (with optional overrides)
GET  /runs/{run_id}/audit/verify  - verify audit chain integrity
"""

from fastapi import APIRouter, Body, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.core.auth import get_current_user
from backend.core.database import Dataset, Project, Run, get_db
from backend.models.run import CheckpointResumeRequest, RunCreate, RunResponse

router = APIRouter(tags=["analysis"])

_RESUMABLE_STATUSES = {"awaiting_checkpoint", "failed"}


async def _get_project_or_404(
    project_id: str, user_id: str, db: AsyncSession
) -> Project:
    result = await db.execute(
        select(Project).where(Project.id == project_id, Project.user_id == user_id)
    )
    project = result.scalar_one_or_none()
    if project is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Project not found")
    return project


async def _get_run_or_404(run_id: str, user_id: str, db: AsyncSession) -> Run:
    result = await db.execute(select(Run).where(Run.id == run_id))
    run = result.scalar_one_or_none()
    if run is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Run not found")

    proj_result = await db.execute(
        select(Project).where(Project.id == run.project_id, Project.user_id == user_id)
    )
    if proj_result.scalar_one_or_none() is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Run not found")

    return run


@router.post(
    "/projects/{project_id}/runs",
    status_code=status.HTTP_201_CREATED,
    response_model=RunResponse,
)
async def create_run(
    project_id: str,
    payload: RunCreate,
    user_id: str = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> RunResponse:
    await _get_project_or_404(project_id, user_id, db)

    ds_result = await db.execute(
        select(Dataset).where(
            Dataset.id == payload.training_dataset_id,
            Dataset.project_id == project_id,
        )
    )
    if ds_result.scalar_one_or_none() is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="training_dataset_id not found in this project",
        )

    run = Run(
        project_id=project_id,
        training_dataset_id=payload.training_dataset_id,
        holdout_dataset_id=payload.holdout_dataset_id,
        threshold_config=payload.threshold_config,
        status="queued",
        created_by=user_id,
    )
    db.add(run)
    await db.commit()
    await db.refresh(run)

    from backend.tasks.analysis_task import run_analysis_task
    job = run_analysis_task.delay(run.id)

    run.job_id = job.id
    await db.commit()
    await db.refresh(run)

    return RunResponse.model_validate(run)


@router.get("/projects/{project_id}/runs", response_model=list[RunResponse])
async def list_runs(
    project_id: str,
    user_id: str = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> list[RunResponse]:
    await _get_project_or_404(project_id, user_id, db)
    result = await db.execute(
        select(Run).where(Run.project_id == project_id).order_by(Run.created_at.desc())
    )
    return [RunResponse.model_validate(r) for r in result.scalars()]


@router.get("/runs/{run_id}", response_model=RunResponse)
async def get_run(
    run_id: str,
    user_id: str = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> RunResponse:
    run = await _get_run_or_404(run_id, user_id, db)
    return RunResponse.model_validate(run)


@router.post("/runs/{run_id}/resume", response_model=RunResponse)
async def resume_run(
    run_id: str,
    payload: CheckpointResumeRequest | None = Body(default=None),
    user_id: str = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> RunResponse:
    """Resume a paused run from its current checkpoint.

    Applies any strategy_override fields before re-enqueueing. The task
    reads run.current_step to pick up exactly where it paused - crash-safe.
    """
    run = await _get_run_or_404(run_id, user_id, db)

    if run.status not in _RESUMABLE_STATUSES:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Run is {run.status!r} - only awaiting_checkpoint or failed runs can be resumed",
        )

    # Apply strategy overrides before re-enqueueing
    if payload and payload.strategy_override:
        _apply_strategy_override(run, payload.strategy_override)

    if payload and payload.threshold_config:
        run.threshold_config = payload.threshold_config

    run.status = "running"
    db.add(run)
    await db.commit()
    await db.refresh(run)

    from backend.tasks.analysis_task import run_analysis_task
    job = run_analysis_task.delay(run.id)

    run.job_id = job.id
    await db.commit()
    await db.refresh(run)

    return RunResponse.model_validate(run)


@router.post("/runs/{run_id}/rerun", response_model=RunResponse)
async def rerun_step(
    run_id: str,
    user_id: str = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> RunResponse:
    """Re-run the step that produced the current checkpoint.

    Unlike resume (which advances to the next step), this regenerates the
    current checkpoint - useful when the step's model output fell back to
    defaults (JSON parse failure or refusal), so the checkpoint looks
    "successful" but never reflected a real model decision. Safe because the
    pipeline pauses at each checkpoint: no downstream state exists yet.
    """
    run = await _get_run_or_404(run_id, user_id, db)

    if run.status not in _RESUMABLE_STATUSES:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Run is {run.status!r} - only awaiting_checkpoint or failed runs can be re-run",
        )

    from backend.tasks.analysis_task import rerun_step_for

    target_step = rerun_step_for(run.current_step)
    if target_step is None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Run step {run.current_step!r} cannot be re-run",
        )

    run.current_step = target_step
    run.status = "running"
    db.add(run)
    await db.commit()
    await db.refresh(run)

    from backend.tasks.analysis_task import run_analysis_task
    job = run_analysis_task.delay(run.id)

    run.job_id = job.id
    await db.commit()
    await db.refresh(run)

    return RunResponse.model_validate(run)


@router.get("/runs/{run_id}/audit/verify")
async def verify_audit_chain(
    run_id: str,
    user_id: str = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> dict[str, object]:
    """Verify the audit chain integrity for a run."""
    await _get_run_or_404(run_id, user_id, db)

    from backend.core.audit import verify_chain
    chain_ok = await verify_chain(db, run_id)

    return {"run_id": run_id, "chain_valid": chain_ok}


def _apply_strategy_override(run: Run, override: dict) -> None:
    """Apply partial strategy overrides to run fields in memory."""
    from copy import deepcopy

    if "model_selection" in override and run.model_selection:
        merged = deepcopy(run.model_selection)
        merged.update(override["model_selection"])
        run.model_selection = merged

    if "preprocessing_strategy" in override and run.preprocessing_strategy:
        merged = deepcopy(run.preprocessing_strategy)
        _deep_merge(merged, override["preprocessing_strategy"])
        run.preprocessing_strategy = merged


def _deep_merge(base: dict, updates: dict) -> None:
    """Recursively merge updates into base dict in-place."""
    for key, value in updates.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
