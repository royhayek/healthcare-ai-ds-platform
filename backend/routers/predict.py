"""Prediction endpoints (§26, §7).

POST /runs/{run_id}/predict              - single-row interactive prediction
POST /runs/{run_id}/predict/batch        - batch inference on an inference dataset
GET  /runs/{run_id}/predictions          - paginated list of stored predictions
GET  /runs/{run_id}/predictions/count    - count of stored predictions (batch progress)
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import Response
from pydantic import BaseModel
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.core import audit
from backend.core.auth import get_current_user
from backend.core.database import Dataset, Prediction, Run, get_db
from backend.core.json_utils import json_safe, safe_float
from backend.core.storage import storage
from backend.ml.predictor import load_model_artifacts, predict_single

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/runs", tags=["predict"])


class PredictRequest(BaseModel):
    input_data: dict[str, Any]


class PredictResponse(BaseModel):
    prediction: Any
    probability: float | None
    threshold_used: float
    confidence_band: str
    similarity_score: float | None
    shap_drivers: list[str]
    shap_dampeners: list[str]
    task_type: str
    prediction_id: str


class PredictionListItem(BaseModel):
    id: str
    prediction: dict[str, Any]
    probability: float | None
    similarity_score: float | None
    confidence_band: str | None
    threshold_used: float | None
    shap_values: dict[str, Any] | None
    risk_flag: bool
    created_at: str

    model_config = {"from_attributes": True}


@router.post("/{run_id}/predict", response_model=PredictResponse)
async def predict(
    run_id: str,
    body: PredictRequest,
    db: AsyncSession = Depends(get_db),
    user_id: str = Depends(get_current_user),
) -> PredictResponse:
    """Run inference for a single input row using the run's trained model."""
    run = await db.get(Run, run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")
    if run.status != "completed":
        raise HTTPException(
            status_code=400,
            detail=f"Run is not completed (status={run.status}). Train the model first.",
        )

    try:
        pipeline, sim_index = await load_model_artifacts(run, storage)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    run_meta = {
        "task_type": run.threshold_result and run.threshold_result.get("task_type")
            or (run.model_selection or {}).get("task_type", "binary_classification"),
        "threshold_result": run.threshold_result,
        "shap_summary": run.shap_summary,
    }

    result = predict_single(pipeline, sim_index, run_meta, body.input_data)

    # ── Persist prediction to DB ───────────────────────────────────────────────
    similarity = safe_float(result["similarity_score"])
    pred_record = Prediction(
        run_id=run_id,
        input_data=json_safe(body.input_data),
        prediction=json_safe({"value": result["prediction"]}),
        probability=safe_float(result["probability"]),
        similarity_score=similarity,
        confidence_band=result["confidence_band"],
        threshold_used=safe_float(result["threshold_used"]),
        shap_values=json_safe(
            {
                "drivers": result["shap_drivers"],
                "dampeners": result["shap_dampeners"],
            }
        ),
        risk_flag=(similarity is not None and similarity < 0.3)
        or result["confidence_band"] == "low",
    )
    db.add(pred_record)
    await db.flush()
    pred_id = pred_record.id

    await audit.append(
        db,
        run_id=run_id,
        actor="user",
        category="prediction",
        action="predict_single",
        payload={
            "prediction": result["prediction"],
            "probability": result["probability"],
            "confidence_band": result["confidence_band"],
            "threshold_used": result["threshold_used"],
            "similarity_score": result["similarity_score"],
        },
        reason="Interactive prediction via predict endpoint",
    )
    await db.commit()

    return PredictResponse(
        **result,
        prediction_id=pred_id,
    )


class BatchPredictRequest(BaseModel):
    inference_dataset_id: str


class BatchPredictResponse(BaseModel):
    job_id: str
    run_id: str
    inference_dataset_id: str
    n_rows: int
    status: str


@router.post("/{run_id}/predict/batch", response_model=BatchPredictResponse)
async def predict_batch(
    run_id: str,
    body: BatchPredictRequest,
    db: AsyncSession = Depends(get_db),
    user_id: str = Depends(get_current_user),
) -> BatchPredictResponse:
    """Trigger batch inference on an inference-role dataset (§7).

    The batch prediction runs as a Celery task. Poll
    GET /runs/{run_id}/predictions to watch rows accumulate, or subscribe
    to the SSE progress stream for live updates.
    """
    run = await db.get(Run, run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")
    if run.status != "completed":
        raise HTTPException(
            status_code=400,
            detail=f"Run is not completed (status={run.status}). Train the model first.",
        )

    ds = await db.get(Dataset, body.inference_dataset_id)
    if ds is None:
        raise HTTPException(status_code=404, detail="Inference dataset not found")
    if ds.role not in ("inference", "comparison"):
        raise HTTPException(
            status_code=400,
            detail=f"Dataset role is '{ds.role}' - must be 'inference' or 'comparison'.",
        )

    from backend.tasks.batch_prediction_task import batch_prediction_task

    celery_result = batch_prediction_task.delay(run_id, body.inference_dataset_id)

    await audit.append(
        db,
        run_id=run_id,
        actor="user",
        category="prediction",
        action="batch_predict_started",
        payload={
            "inference_dataset_id": body.inference_dataset_id,
            "filename": ds.filename,
            "job_id": celery_result.id,
        },
        reason=f"Batch inference on {ds.filename} queued by user",
    )
    await db.commit()

    return BatchPredictResponse(
        job_id=celery_result.id,
        run_id=run_id,
        inference_dataset_id=body.inference_dataset_id,
        n_rows=ds.row_count or 0,
        status="queued",
    )


_BATCH_ARTIFACT_FORMATS = {
    "xlsx": ("xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"),
    "csv": ("csv", "text/csv; charset=utf-8"),
    "parquet": ("parquet", "application/octet-stream"),
}


@router.get("/{run_id}/predict/batch/{inference_dataset_id}/download")
async def download_batch_artifact(
    run_id: str,
    inference_dataset_id: str,
    format: str = Query(default="xlsx"),
    db: AsyncSession = Depends(get_db),
    user_id: str = Depends(get_current_user),
) -> Response:
    """Stream a batch-inference predictions artifact (xlsx/csv/parquet).

    Mirrors the storage layout written by ``batch_prediction_task``:
    ``runs/{run_id}/inference_{inference_dataset_id}_predictions.{ext}``.
    """
    if format not in _BATCH_ARTIFACT_FORMATS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format '{format}'. Use one of: {', '.join(_BATCH_ARTIFACT_FORMATS)}.",
        )

    run = await db.get(Run, run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")

    ext, mime = _BATCH_ARTIFACT_FORMATS[format]
    path = f"runs/{run_id}/inference_{inference_dataset_id}_predictions.{ext}"

    if not await storage.exists(path):
        raise HTTPException(
            status_code=404,
            detail="Batch predictions artifact not found. Run batch inference first.",
        )

    try:
        content = await storage.download(path)
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail="Batch predictions artifact not found. Run batch inference first.",
        ) from exc

    filename = f"predictions_{inference_dataset_id}.{ext}"
    return Response(
        content=content,
        media_type=mime,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


class PredictionCountResponse(BaseModel):
    run_id: str
    count: int


@router.get("/{run_id}/predictions/count", response_model=PredictionCountResponse)
async def count_predictions(
    run_id: str,
    db: AsyncSession = Depends(get_db),
    user_id: str = Depends(get_current_user),
) -> PredictionCountResponse:
    """Return the number of stored predictions for a run.

    Lightweight progress signal for batch inference polling — avoids fetching
    full SHAP payloads just to count accumulated rows.
    """
    run = await db.get(Run, run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")

    stmt = select(func.count()).select_from(Prediction).where(Prediction.run_id == run_id)
    count = (await db.execute(stmt)).scalar_one()
    return PredictionCountResponse(run_id=run_id, count=count)


@router.get("/{run_id}/predictions", response_model=list[PredictionListItem])
async def list_predictions(
    run_id: str,
    limit: int = Query(default=50, le=200),
    offset: int = Query(default=0, ge=0),
    db: AsyncSession = Depends(get_db),
    user_id: str = Depends(get_current_user),
) -> list[PredictionListItem]:
    """Return stored predictions for a completed run, newest first."""
    run = await db.get(Run, run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")

    stmt = (
        select(Prediction)
        .where(Prediction.run_id == run_id)
        .order_by(Prediction.created_at.desc())
        .limit(limit)
        .offset(offset)
    )
    rows = (await db.execute(stmt)).scalars().all()

    return [
        PredictionListItem(
            id=r.id,
            prediction=r.prediction,
            probability=r.probability,
            similarity_score=r.similarity_score,
            confidence_band=r.confidence_band,
            threshold_used=r.threshold_used,
            shap_values=r.shap_values,
            risk_flag=r.risk_flag,
            created_at=r.created_at.isoformat(),
        )
        for r in rows
    ]
