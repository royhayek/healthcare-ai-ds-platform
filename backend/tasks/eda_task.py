"""Celery task for the EDA pipeline step (§22).

Structural wrapper that exposes a named `analysis.eda` Celery task. The
routing table in analysis_task.py dispatches to _step_eda() when current_step
is "init", "profiling", or "eda". Calling this task directly (e.g., for retry)
re-enters the pipeline from the last persisted state within the EDA phase.
"""

import asyncio
import logging

from backend.core.events import emit_progress
from backend.tasks.celery_app import celery_app

logger = logging.getLogger(__name__)


@celery_app.task(bind=True, name="analysis.eda", max_retries=2)
def run_eda_task(self, run_id: str) -> None:  # type: ignore[misc]
    """EDA step: load dataset → profile → run EDA agent → Checkpoint 1."""
    emit_progress(run_id, "eda", "EDA step starting…", 5)
    try:
        from backend.tasks.analysis_task import _async_pipeline
        asyncio.run(_async_pipeline(run_id))
    except Exception as exc:
        logger.exception("EDA task failed for run %s", run_id)
        raise
