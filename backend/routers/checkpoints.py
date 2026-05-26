"""Checkpoint management endpoints (§2, §26).

Provides the spec §26 contract shape for checkpoint-based pipeline control:

GET  /runs/{run_id}/checkpoints           - list all checkpoints reached
GET  /runs/{run_id}/checkpoints/{n}       - details for checkpoint N
POST /runs/{run_id}/checkpoints/{n}/resume - resume from checkpoint N with optional overrides

The resume logic is identical to POST /runs/{run_id}/resume in analysis.py -
this router adds the checkpoint-numbered URL shape and per-checkpoint detail
responses on top of the existing infrastructure.
"""

from fastapi import APIRouter, Body, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Any

from backend.core.auth import get_current_user
from backend.core.database import Project, Run, get_db
from backend.models.run import CheckpointResumeRequest, RunResponse

router = APIRouter(tags=["checkpoints"])

# Map spec checkpoint numbers → current_step values that match "awaiting" state
_CHECKPOINT_STEPS: dict[int, str] = {
    1: "checkpoint_1_eda",
    2: "checkpoint_2_preprocessing",
    3: "checkpoint_3_model_selection",
    4: "checkpoint_4_training",
    5: "checkpoint_5_final",
}

# Human-readable labels and pipeline outputs available at each checkpoint
_CHECKPOINT_META: dict[int, dict[str, Any]] = {
    1: {
        "label": "EDA Review",
        "description": "Review EDA findings and quality issues before preprocessing.",
        "available_fields": ["eda_report"],
    },
    2: {
        "label": "Preprocessing Review",
        "description": "Review and override column-level preprocessing decisions.",
        "available_fields": ["preprocessing_strategy"],
    },
    3: {
        "label": "Model Selection Review",
        "description": "Review and override candidate model choices.",
        "available_fields": ["model_selection"],
    },
    4: {
        "label": "Training Review",
        "description": "Review stability leaderboard and stat-test results.",
        "available_fields": ["model_comparison", "stat_tests", "best_model_name", "best_model_score"],
    },
    5: {
        "label": "Final Review",
        "description": "Review all results before generating deliverables.",
        "available_fields": [
            "final_metrics", "eval_plots", "shap_summary",
            "calibration_report", "threshold_result",
            "drift_report", "fairness_report", "insight_report",
        ],
    },
}


class CheckpointSummary(BaseModel):
    checkpoint: int
    step: str
    label: str
    description: str
    reached: bool
    available_fields: list[str]


class CheckpointDetail(BaseModel):
    checkpoint: int
    step: str
    label: str
    description: str
    data: dict[str, Any]


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


def _run_has_reached(run: Run, checkpoint_n: int) -> bool:
    """Return True if the run has reached or passed this checkpoint."""
    step = _CHECKPOINT_STEPS[checkpoint_n]
    step_order = list(_CHECKPOINT_STEPS.values())
    current = run.current_step or ""
    if current in step_order:
        return step_order.index(current) >= step_order.index(step)
    # If current_step is a mid-step name, infer by checking later fields
    return False


@router.get("/runs/{run_id}/checkpoints", response_model=list[CheckpointSummary])
async def list_checkpoints(
    run_id: str,
    user_id: str = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> list[CheckpointSummary]:
    """Return all 5 checkpoints with reached/not-reached status."""
    run = await _get_run_or_404(run_id, user_id, db)
    summaries = []
    for n in range(1, 6):
        meta = _CHECKPOINT_META[n]
        summaries.append(
            CheckpointSummary(
                checkpoint=n,
                step=_CHECKPOINT_STEPS[n],
                label=meta["label"],
                description=meta["description"],
                reached=_run_has_reached(run, n),
                available_fields=meta["available_fields"],
            )
        )
    return summaries


@router.get("/runs/{run_id}/checkpoints/{n}", response_model=CheckpointDetail)
async def get_checkpoint(
    run_id: str,
    n: int,
    user_id: str = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> CheckpointDetail:
    """Return pipeline data available at checkpoint N."""
    if n not in _CHECKPOINT_STEPS:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Checkpoint {n} does not exist")
    run = await _get_run_or_404(run_id, user_id, db)
    if not _run_has_reached(run, n):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Checkpoint {n} has not been reached yet",
        )
    meta = _CHECKPOINT_META[n]
    data: dict[str, Any] = {}
    for field in meta["available_fields"]:
        val = getattr(run, field, None)
        if val is not None:
            data[field] = val
    return CheckpointDetail(
        checkpoint=n,
        step=_CHECKPOINT_STEPS[n],
        label=meta["label"],
        description=meta["description"],
        data=data,
    )


@router.post("/runs/{run_id}/checkpoints/{n}/resume", response_model=RunResponse)
async def resume_from_checkpoint(
    run_id: str,
    n: int,
    payload: CheckpointResumeRequest | None = Body(default=None),
    user_id: str = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> RunResponse:
    """Resume a run from checkpoint N with optional strategy overrides.

    Delegates to the same logic as POST /runs/{run_id}/resume. The run must
    currently be paused at this checkpoint (status=awaiting_checkpoint).
    """
    if n not in _CHECKPOINT_STEPS:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Checkpoint {n} does not exist")

    run = await _get_run_or_404(run_id, user_id, db)
    expected_step = _CHECKPOINT_STEPS[n]

    if run.current_step != expected_step:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                f"Run is at step {run.current_step!r}, not {expected_step!r}. "
                "Use POST /runs/{run_id}/resume to resume from the current step."
            ),
        )

    if run.status not in {"awaiting_checkpoint", "failed"}:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Run is {run.status!r} - only awaiting_checkpoint or failed runs can be resumed",
        )

    from copy import deepcopy

    if payload and payload.strategy_override:
        if "model_selection" in payload.strategy_override and run.model_selection:
            merged = deepcopy(run.model_selection)
            merged.update(payload.strategy_override["model_selection"])
            run.model_selection = merged
        if "preprocessing_strategy" in payload.strategy_override and run.preprocessing_strategy:
            merged = deepcopy(run.preprocessing_strategy)
            _deep_merge(merged, payload.strategy_override["preprocessing_strategy"])
            run.preprocessing_strategy = merged

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


def _deep_merge(base: dict, updates: dict) -> None:
    for key, value in updates.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
