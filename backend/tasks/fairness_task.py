"""Celery task for a standalone fairness audit run (§19, §22).

Exposes `analysis.fairness` as a named Celery task. Within the main pipeline
the fairness audit is a sub-step of _step_tuning. This task allows fairness to
be re-run in isolation - e.g., when the user changes protected_columns after
the model is already trained - without re-running tuning/SHAP.

Usage:
    from backend.tasks.fairness_task import run_fairness_task
    run_fairness_task.delay(run_id, protected_columns=["gender", "age_band"])
"""

import asyncio
import logging
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import select

from backend.core import audit
from backend.core.database import Dataset, Run, async_session_factory
from backend.core.events import ProgressEmitter, emit_progress
from backend.core.storage import storage
from backend.tasks.celery_app import celery_app

logger = logging.getLogger(__name__)


@celery_app.task(bind=True, name="analysis.fairness", max_retries=0)
def run_fairness_task(
    self,  # type: ignore[misc]
    run_id: str,
    protected_columns: list[str] | None = None,
) -> None:
    """Standalone fairness audit; does not re-run the full pipeline."""
    emit_progress(run_id, "fairness", "Fairness audit starting…", 0)
    try:
        asyncio.run(_async_fairness(run_id, protected_columns or []))
    except Exception as exc:
        logger.exception("Fairness task failed for run %s", run_id)
        raise


async def _async_fairness(run_id: str, protected_columns: list[str]) -> None:
    from sqlalchemy.ext.asyncio import AsyncSession as _AS
    from sqlalchemy.ext.asyncio import async_sessionmaker as _asf
    from sqlalchemy.ext.asyncio import create_async_engine as _cae

    from backend.core.config import settings

    _local_engine = _cae(settings.DATABASE_URL, pool_pre_ping=True)
    _local_factory = _asf(_local_engine, class_=_AS, expire_on_commit=False)

    emitter = ProgressEmitter(run_id)
    try:
        async with _local_factory() as session:
            run_result = await session.execute(select(Run).where(Run.id == run_id))
            run: Run | None = run_result.scalar_one_or_none()
            if run is None:
                raise ValueError(f"Run {run_id} not found")

            if not run.model_storage_path:
                raise ValueError(f"Run {run_id} has no trained model - cannot run fairness audit")

            ds_result = await session.execute(
                select(Dataset).where(Dataset.id == run.training_dataset_id)
            )
            dataset: Dataset | None = ds_result.scalar_one_or_none()
            if dataset is None:
                raise ValueError(f"Dataset {run.training_dataset_id} not found")

            cols = protected_columns or (run.fairness_config or {}).get("protected_columns", [])
            if not cols:
                logger.info("No protected columns - fairness audit skipped for run %s", run_id)
                return

            await emitter.emit_async("fairness", f"Running fairness audit on {cols}…", 10)

            import joblib

            raw_bytes = await storage.download(run.model_storage_path)
            pipeline = joblib.load(__import__("io").BytesIO(raw_bytes))

            import pickle  # nosec
            splits_bytes = await storage.download(f"runs/{run_id}/splits.pkl")
            splits = pickle.loads(splits_bytes)  # nosec
            X_test: pd.DataFrame = splits["X_test"]
            y_test: pd.Series = splits["y_test"]

            from backend.ml.fairness import build_sensitive_features, fairness_audit
            from backend.ml.cleaner import _parse_csv_bytes  # noqa: F401 - exists in cleaner

            raw_ds_bytes = await storage.download(dataset.storage_path)
            df_raw = pd.read_csv(__import__("io").BytesIO(raw_ds_bytes))

            sensitive = build_sensitive_features(df_raw, cols, index=X_test.index)

            from backend.models.strategy import PreprocessingStrategy
            prep_strategy = PreprocessingStrategy.model_validate(run.preprocessing_strategy or {})
            task_type = prep_strategy.task_type

            optimal_threshold = (run.threshold_config or {}).get("optimal_threshold", 0.5)

            if task_type in ("binary_classification", "multiclass"):
                y_proba = pipeline.predict_proba(X_test)
                y_proba_1d = y_proba[:, 1] if task_type == "binary_classification" else y_proba
            else:
                y_proba_1d = None

            y_pred = pipeline.predict(X_test)
            if task_type == "binary_classification":
                y_pred = (pipeline.predict_proba(X_test)[:, 1] >= optimal_threshold).astype(int)

            fairness_report = await __import__("asyncio").get_running_loop().run_in_executor(
                None,
                lambda: fairness_audit(
                    np.asarray(y_test),
                    np.asarray(y_pred),
                    np.asarray(y_proba_1d) if y_proba_1d is not None else None,
                    sensitive,
                ),
            )

            await audit.append(
                session, run_id=run_id, actor="system", category="fairness",
                action="fairness_audit_complete",
                payload={
                    "protected_columns": cols,
                    "overall_severity": fairness_report.overall_severity,
                    "blocks_deliverables": fairness_report.blocks_deliverables,
                },
                reason=f"Standalone fairness re-run: severity={fairness_report.overall_severity}",
            )

            run.fairness_report = fairness_report.model_dump()
            run.fairness_config = {"protected_columns": cols}
            session.add(run)
            await session.commit()

            await emitter.emit_async(
                "fairness",
                f"Fairness audit complete - overall severity: {fairness_report.overall_severity}",
                100,
            )
    finally:
        await _local_engine.dispose()
