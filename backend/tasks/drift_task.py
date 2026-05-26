"""Celery task for a standalone drift detection run (§17, §22).

Exposes `analysis.drift` as a named Celery task. Within the main pipeline
drift is computed as a sub-step of _step_tuning. This task allows drift to be
re-computed in isolation - e.g., when the user uploads a new comparison dataset
after the model was already trained.

Usage:
    from backend.tasks.drift_task import run_drift_task
    run_drift_task.delay(run_id, comparison_dataset_id="<id>")
"""

import asyncio
import logging

import pandas as pd
from sqlalchemy import select

from backend.core import audit
from backend.core.database import Dataset, Run
from backend.core.events import ProgressEmitter, emit_progress
from backend.core.storage import storage
from backend.tasks.celery_app import celery_app

logger = logging.getLogger(__name__)


@celery_app.task(bind=True, name="analysis.drift", max_retries=0)
def run_drift_task(
    self,  # type: ignore[misc]
    run_id: str,
    comparison_dataset_id: str | None = None,
) -> None:
    """Standalone drift detection; does not re-run the full pipeline."""
    emit_progress(run_id, "drift", "Drift detection starting…", 0)
    try:
        asyncio.run(_async_drift(run_id, comparison_dataset_id))
    except Exception as exc:
        logger.exception("Drift task failed for run %s", run_id)
        raise


async def _async_drift(run_id: str, comparison_dataset_id: str | None) -> None:
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

            ds_result = await session.execute(
                select(Dataset).where(Dataset.id == run.training_dataset_id)
            )
            training_dataset: Dataset | None = ds_result.scalar_one_or_none()
            if training_dataset is None:
                raise ValueError(f"Training dataset not found for run {run_id}")

            # Resolve comparison dataset
            if comparison_dataset_id:
                comp_result = await session.execute(
                    select(Dataset).where(Dataset.id == comparison_dataset_id)
                )
                comparison_dataset: Dataset | None = comp_result.scalar_one_or_none()
            else:
                from sqlalchemy import desc
                comp_result = await session.execute(
                    select(Dataset)
                    .where(Dataset.project_id == run.project_id)
                    .where(Dataset.role.in_(["inference", "comparison"]))
                    .order_by(desc(Dataset.created_at))
                    .limit(1)
                )
                comparison_dataset = comp_result.scalar_one_or_none()

            if comparison_dataset is None:
                logger.info("No comparison dataset found - drift task skipped for run %s", run_id)
                return

            await emitter.emit_async(
                "drift", f"Computing drift vs. {comparison_dataset.filename}…", 10
            )

            from backend.ml.drift import compute_drift_report
            from backend.models.strategy import PreprocessingStrategy

            prep_strategy = PreprocessingStrategy.model_validate(run.preprocessing_strategy or {})
            numeric_cols = prep_strategy.numeric_columns()
            categorical_cols = prep_strategy.categorical_columns()

            import io
            train_bytes = await storage.download(training_dataset.storage_path)
            df_train = pd.read_csv(io.BytesIO(train_bytes))

            comp_bytes = await storage.download(comparison_dataset.storage_path)
            df_comp = pd.read_csv(io.BytesIO(comp_bytes))

            drift_report = compute_drift_report(
                df_train, df_comp,
                numeric_cols=[c for c in numeric_cols if c in df_comp.columns],
                categorical_cols=[c for c in categorical_cols if c in df_comp.columns],
            )

            await audit.append(
                session, run_id=run_id, actor="system", category="drift",
                action="drift_analysis_complete",
                payload={
                    "comparison_dataset": comparison_dataset.filename,
                    "overall_severity": drift_report.overall_severity,
                    "aggregate_psi": drift_report.aggregate_psi,
                    "n_features_drifted": drift_report.n_features_drifted,
                },
                reason=f"Standalone drift re-run vs. {comparison_dataset.filename}: {drift_report.overall_severity}",
            )

            run.drift_report = drift_report.model_dump()
            session.add(run)
            await session.commit()

            await emitter.emit_async(
                "drift",
                f"Drift analysis complete - severity: {drift_report.overall_severity}, "
                f"{drift_report.n_features_drifted} features drifted",
                100,
            )
    finally:
        await _local_engine.dispose()
