"""Celery task for the tuning/calibration/threshold/SHAP step (§22).

Exposes a named `analysis.tuning` Celery task. Routed by analysis_task when
current_step is "checkpoint_4_training" or any sub-step within the tuning
phase (tuning, calibration, threshold, shap, similarity, drift, fairness,
holdout, insight). Resumes from the last persisted sub-step on retry.
"""

import asyncio
import logging

from backend.core.events import emit_progress
from backend.tasks.celery_app import celery_app

logger = logging.getLogger(__name__)


@celery_app.task(bind=True, name="analysis.tuning", max_retries=0)
def run_tuning_task(self, run_id: str) -> None:  # type: ignore[misc]
    """Tuning step: Optuna → calibration → threshold → SHAP → insight → Checkpoint 5."""
    emit_progress(run_id, "tuning", "Tuning step starting…", 74)
    try:
        from backend.tasks.analysis_task import _async_pipeline
        asyncio.run(_async_pipeline(run_id))
    except Exception as exc:
        logger.exception("Tuning task failed for run %s", run_id)
        raise
