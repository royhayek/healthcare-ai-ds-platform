"""Celery task for the preprocessing strategy step (§22).

Exposes a named `analysis.preprocessing` Celery task. Routed by analysis_task
when current_step is "checkpoint_1_eda" or "preprocessing". Calling this task
directly resumes from Checkpoint 1 and runs the preprocessing agent through
Checkpoint 2.
"""

import asyncio
import logging

from backend.core.events import emit_progress
from backend.tasks.celery_app import celery_app

logger = logging.getLogger(__name__)


@celery_app.task(bind=True, name="analysis.preprocessing", max_retries=2)
def run_preprocessing_task(self, run_id: str) -> None:  # type: ignore[misc]
    """Preprocessing step: build column strategy → Checkpoint 2."""
    emit_progress(run_id, "preprocessing", "Preprocessing step starting…", 42)
    try:
        from backend.tasks.analysis_task import _async_pipeline
        asyncio.run(_async_pipeline(run_id))
    except Exception as exc:
        logger.exception("Preprocessing task failed for run %s", run_id)
        raise
