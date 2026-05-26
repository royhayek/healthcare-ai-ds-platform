"""Batch inference Celery task (§7 - Train A → Predict B).

Triggered by POST /runs/{run_id}/predict/batch when an inference-role
dataset is present. Steps:

1. Load the trained + calibrated pipeline from storage.
2. Load the inference dataset; validate that its schema is compatible.
3. Run batch prediction row-by-row with per-row SHAP and similarity scoring.
4. Write a predictions artifact (Excel + CSV + Parquet) for the inference set.
5. Persist individual Prediction rows to the DB for query / audit.

Progress is emitted at every 10% interval so the frontend spinner never stalls.
"""

import asyncio
import io
import logging
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import select

from backend.core import audit
from backend.core.config import settings
from backend.core.database import Dataset, Prediction, Run, async_session_factory
from backend.core.events import ProgressEmitter
from backend.core.storage import storage
from backend.ml.predictor import load_model_artifacts, predict_single
from backend.tasks.celery_app import celery_app

logger = logging.getLogger(__name__)

_BATCH_SHAP_LIMIT = 2000  # max rows for per-row SHAP in batch mode


@celery_app.task(bind=True, name="prediction.batch", max_retries=0)
def batch_prediction_task(
    self,  # type: ignore[misc]
    run_id: str,
    inference_dataset_id: str,
) -> None:
    """Celery entry point. Bridges sync → async."""
    try:
        asyncio.run(_async_batch(run_id, inference_dataset_id))
    except Exception as exc:
        logger.exception(
            "Batch prediction failed for run %s, dataset %s", run_id, inference_dataset_id
        )
        raise


async def _async_batch(run_id: str, inference_dataset_id: str) -> None:
    from sqlalchemy.ext.asyncio import AsyncSession as _AS
    from sqlalchemy.ext.asyncio import async_sessionmaker as _asf
    from sqlalchemy.ext.asyncio import create_async_engine as _cae

    local_engine = _cae(settings.DATABASE_URL, pool_pre_ping=True)
    local_factory = _asf(local_engine, class_=_AS, expire_on_commit=False)
    emitter = ProgressEmitter(run_id)

    try:
        async with local_factory() as session:
            run = (
                await session.execute(select(Run).where(Run.id == run_id))
            ).scalar_one_or_none()
            if run is None:
                raise ValueError(f"Run {run_id} not found")

            ds = (
                await session.execute(select(Dataset).where(Dataset.id == inference_dataset_id))
            ).scalar_one_or_none()
            if ds is None:
                raise ValueError(f"Inference dataset {inference_dataset_id} not found")

            await emitter.emit_async("batch_predict", f"Loading model for batch inference…", 2)

            pipeline, sim_index = await load_model_artifacts(run, storage)

            await emitter.emit_async("batch_predict", f"Loading {ds.filename}…", 5)

            raw = await storage.download(ds.storage_path)
            df = _parse_df(raw, ds.filename)

            await emitter.emit_async(
                "batch_predict", f"Loaded {len(df)} rows - validating schema…", 8
            )

            # Schema validation: warn on missing columns (don't crash)
            from backend.models.strategy import PreprocessingStrategy

            prep_strategy = PreprocessingStrategy.model_validate(
                run.preprocessing_strategy or {}
            )
            expected = set(prep_strategy.feature_columns())
            actual = set(df.columns)
            missing_cols = expected - actual
            if missing_cols:
                logger.warning(
                    "Inference dataset %s is missing columns: %s - those will be NaN",
                    ds.filename, missing_cols
                )

            # Check for drift before predicting
            drift_warning: str | None = None
            try:
                from backend.ml.drift import compute_drift_report

                train_raw_bytes = await storage.download(
                    (
                        await session.execute(
                            select(Dataset).where(Dataset.id == run.training_dataset_id)
                        )
                    ).scalar_one().storage_path
                )
                df_train = _parse_df(train_raw_bytes, "train")
                numeric_cols = [c for c in prep_strategy.numeric_columns() if c in df.columns]
                cat_cols = [c for c in prep_strategy.categorical_columns() if c in df.columns]
                drift = compute_drift_report(df_train, df, numeric_cols, cat_cols)
                if drift.overall_severity in ("moderate", "severe"):
                    drift_warning = (
                        f"Drift detected ({drift.overall_severity}) - "
                        f"{drift.n_features_drifted} features drifted. "
                        "Predictions may be less reliable."
                    )
                    await emitter.emit_async("batch_predict", drift_warning, 10)
            except Exception as exc:
                logger.debug("Drift check during batch predict failed (non-fatal): %s", exc)

            # Build run_meta for predictor
            task_type: str = (
                (run.threshold_result or {}).get("task_type")
                or (run.model_selection or {}).get("task_type", "binary_classification")
            )
            run_meta = {
                "task_type": task_type,
                "threshold_result": run.threshold_result,
                "shap_summary": run.shap_summary,
            }
            optimal_threshold = float(
                (run.threshold_result or {}).get("optimal_threshold", 0.5)
            )

            # Per-row SHAP (limited to first BATCH_SHAP_LIMIT rows)
            shap_matrix: np.ndarray | None = None
            feature_names = list(df.columns)
            if len(df) <= _BATCH_SHAP_LIMIT:
                try:
                    from backend.ml.explainer import compute_shap

                    shap_result = compute_shap(
                        pipeline, df, feature_names, task_type, background_data=None
                    )
                    if (
                        hasattr(shap_result, "shap_values_matrix")
                        and shap_result.shap_values_matrix is not None
                    ):
                        shap_matrix = shap_result.shap_values_matrix
                except Exception as exc:
                    logger.debug("Batch SHAP failed (non-fatal): %s", exc)

            await emitter.emit_async("batch_predict", "Running predictions…", 15)

            n_rows = len(df)
            prediction_rows: list[Prediction] = []
            batch_results: list[dict[str, Any]] = []
            last_pct = 15

            for i, (_, row) in enumerate(df.iterrows()):
                result = predict_single(pipeline, sim_index, run_meta, row.to_dict())
                # Inject per-row SHAP if available
                if shap_matrix is not None and i < len(shap_matrix):
                    from backend.deliverables.predictions_artifact import _top_drivers

                    drivers, dampeners = _top_drivers(shap_matrix[i], feature_names)
                    result["shap_drivers"] = drivers.split(", ") if drivers else []
                    result["shap_dampeners"] = dampeners.split(", ") if dampeners else []

                batch_results.append(result)
                prediction_rows.append(
                    Prediction(
                        run_id=run_id,
                        inference_dataset_id=inference_dataset_id,
                        input_data=row.to_dict(),
                        prediction={"value": result["prediction"]},
                        probability=result["probability"],
                        similarity_score=result["similarity_score"],
                        confidence_band=result["confidence_band"],
                        threshold_used=result["threshold_used"],
                        shap_values={
                            "drivers": result["shap_drivers"],
                            "dampeners": result["shap_dampeners"],
                        },
                        risk_flag=(
                            (result["similarity_score"] is not None and result["similarity_score"] < 0.3)
                            or result["confidence_band"] == "low"
                        ),
                    )
                )

                # Emit progress every ~10%
                pct = 15 + int((i + 1) / n_rows * 75)
                if pct >= last_pct + 10:
                    await emitter.emit_async(
                        "batch_predict", f"Predicted {i + 1}/{n_rows} rows…", pct
                    )
                    last_pct = pct

            # Bulk insert predictions
            session.add_all(prediction_rows)
            await session.flush()

            await audit.append(
                session,
                run_id=run_id,
                actor="system",
                category="prediction",
                action="batch_predict_complete",
                payload={
                    "inference_dataset_id": inference_dataset_id,
                    "filename": ds.filename,
                    "n_rows": n_rows,
                    "task_type": task_type,
                    "drift_warning": drift_warning,
                },
                reason=f"Batch inference on {ds.filename} ({n_rows} rows)",
            )
            await session.commit()

            await emitter.emit_async("batch_predict", "Generating predictions artifact…", 90)

            # Build output DataFrame with all metadata columns
            df_out = df.copy().reset_index(drop=True)
            df_out["prediction"] = [r["prediction"] for r in batch_results]
            df_out["probability"] = [r["probability"] for r in batch_results]
            df_out["similarity_score"] = [r["similarity_score"] for r in batch_results]
            df_out["confidence_band"] = [r["confidence_band"] for r in batch_results]
            df_out["shap_drivers"] = [", ".join(r["shap_drivers"]) for r in batch_results]
            df_out["shap_dampeners"] = [", ".join(r["shap_dampeners"]) for r in batch_results]
            df_out["threshold_used"] = optimal_threshold
            df_out["risk_flag"] = [
                int(
                    (r["similarity_score"] is not None and r["similarity_score"] < 0.3)
                    or r["confidence_band"] == "low"
                )
                for r in batch_results
            ]

            # Write artifacts
            from backend.deliverables.predictions_artifact import _write_excel, _write_parquet

            excel_bytes = _write_excel(df_out, run_id)
            csv_bytes = df_out.to_csv(index=False).encode("utf-8")
            parquet_bytes = _write_parquet(df_out)

            prefix = f"runs/{run_id}/inference_{inference_dataset_id}"
            await storage.upload(f"{prefix}_predictions.xlsx", excel_bytes, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
            await storage.upload(f"{prefix}_predictions.csv", csv_bytes, "text/csv")
            await storage.upload(f"{prefix}_predictions.parquet", parquet_bytes, "application/octet-stream")

            await emitter.emit_async(
                "batch_predict",
                f"Batch inference complete - {n_rows} predictions stored",
                100,
                {
                    "n_rows": n_rows,
                    "drift_warning": drift_warning,
                    "artifact_prefix": prefix,
                },
            )

    finally:
        await local_engine.dispose()


def _parse_df(raw: bytes, filename: str) -> pd.DataFrame:
    if filename.endswith(".parquet"):
        return pd.read_parquet(io.BytesIO(raw))
    return pd.read_csv(io.BytesIO(raw))
