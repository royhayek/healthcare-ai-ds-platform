"""Celery task for the training step (§22).

Exposes a named `analysis.training` Celery task. Routed by analysis_task when
current_step is "checkpoint_3_model_selection" or "training". Runs stability
training (3 seeds × 5 folds per candidate), stat tests, and persists the model
leaderboard. Emits Checkpoint 4 on completion.
"""

import asyncio
import logging

from backend.core.events import emit_progress
from backend.tasks.celery_app import celery_app

logger = logging.getLogger(__name__)


@celery_app.task(bind=True, name="analysis.training", max_retries=0)
def run_training_task(self, run_id: str) -> None:  # type: ignore[misc]
    """Training step: stability runs → stat tests → Checkpoint 4."""
    emit_progress(run_id, "training", "Training step starting…", 54)
    try:
        from backend.tasks.analysis_task import _async_pipeline
        asyncio.run(_async_pipeline(run_id))
    except Exception as exc:
        logger.exception("Training task failed for run %s", run_id)
        raise
