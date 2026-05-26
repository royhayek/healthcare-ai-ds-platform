"""Results endpoints - structured access to pipeline outputs (§26).

The full run object is already returned by GET /runs/{run_id}. This router
adds granular endpoints that return specific result sections, enabling the
frontend to fetch only what it needs for a given results widget.

GET /runs/{run_id}/results/metrics        - final metrics + calibration + threshold
GET /runs/{run_id}/results/shap           - SHAP summary
GET /runs/{run_id}/results/eval-plots     - ROC/PR/confusion/calibration curve data
GET /runs/{run_id}/results/drift          - drift report
GET /runs/{run_id}/results/fairness       - fairness report
GET /runs/{run_id}/results/insight        - insight report text
GET /runs/{run_id}/results/model-card     - model card fields (name, metrics, SHAP, threshold)
"""

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.core.auth import get_current_user
from backend.core.database import Project, Run, get_db

router = APIRouter(tags=["results"])


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


def _require_field(run: Run, field: str) -> Any:
    val = getattr(run, field, None)
    if val is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Run {run.id} does not have {field} yet - pipeline not complete",
        )
    return val


@router.get("/runs/{run_id}/results/metrics")
async def get_metrics(
    run_id: str,
    user_id: str = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    run = await _get_run_or_404(run_id, user_id, db)
    final = _require_field(run, "final_metrics")
    return {
        "final_metrics": final,
        "best_model_name": run.best_model_name,
        "best_model_score": run.best_model_score,
        "calibration_report": run.calibration_report,
        "threshold_result": run.threshold_result,
        "threshold_config": run.threshold_config,
        "stat_tests": run.stat_tests,
        "stability_results": run.model_comparison,
    }


@router.get("/runs/{run_id}/results/shap")
async def get_shap(
    run_id: str,
    user_id: str = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    run = await _get_run_or_404(run_id, user_id, db)
    return {"shap_summary": _require_field(run, "shap_summary")}


@router.get("/runs/{run_id}/results/eval-plots")
async def get_eval_plots(
    run_id: str,
    user_id: str = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    run = await _get_run_or_404(run_id, user_id, db)
    return {"eval_plots": _require_field(run, "eval_plots")}


@router.get("/runs/{run_id}/results/drift")
async def get_drift(
    run_id: str,
    user_id: str = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    run = await _get_run_or_404(run_id, user_id, db)
    return {"drift_report": _require_field(run, "drift_report")}


@router.get("/runs/{run_id}/results/fairness")
async def get_fairness(
    run_id: str,
    user_id: str = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    run = await _get_run_or_404(run_id, user_id, db)
    return {"fairness_report": _require_field(run, "fairness_report")}


@router.get("/runs/{run_id}/results/insight")
async def get_insight(
    run_id: str,
    user_id: str = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    run = await _get_run_or_404(run_id, user_id, db)
    return {"insight_report": _require_field(run, "insight_report")}


@router.get("/runs/{run_id}/results/model-card")
async def get_model_card(
    run_id: str,
    user_id: str = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """Return the fields needed to render or preview a model card."""
    run = await _get_run_or_404(run_id, user_id, db)
    shap = run.shap_summary or {}
    return {
        "model_name": run.best_model_name,
        "task_type": (run.preprocessing_strategy or {}).get("task_type"),
        "final_metrics": run.final_metrics,
        "threshold_used": (run.threshold_config or {}).get("optimal_threshold", 0.5),
        "top_features": shap.get("top_k_features", [])[:10],
        "calibration_method": (run.calibration_report or {}).get("method"),
        "stability_mean": run.best_model_score,
        "tuning_params": (run.tuning_result or {}).get("best_params"),
        "drift_severity": (run.drift_report or {}).get("overall_severity"),
        "fairness_severity": (run.fairness_report or {}).get("overall_severity"),
        "completed_at": run.completed_at,
    }
