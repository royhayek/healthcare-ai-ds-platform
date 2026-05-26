"""Celery task for pipeline-triggered inference dataset prediction (§7, §22).

Exposes `analysis.predict` as a named Celery task. This is the pipeline-level
prediction step that is triggered automatically when the run has an
inference-role dataset attached at pipeline start. For on-demand batch
prediction from the UI, see batch_prediction_task.py.

Usage:
    from backend.tasks.prediction_task import run_prediction_task
    run_prediction_task.delay(run_id, inference_dataset_id)
"""

import asyncio
import logging

from backend.core.events import emit_progress
from backend.tasks.celery_app import celery_app

logger = logging.getLogger(__name__)


@celery_app.task(bind=True, name="analysis.predict", max_retries=0)
def run_prediction_task(
    self,  # type: ignore[misc]
    run_id: str,
    inference_dataset_id: str,
) -> None:
    """Pipeline prediction step: run inference dataset through the trained model."""
    emit_progress(run_id, "prediction", "Inference prediction starting…", 0)
    try:
        from backend.tasks.batch_prediction_task import _async_batch
        asyncio.run(_async_batch(run_id, inference_dataset_id))
    except Exception as exc:
        logger.exception(
            "Prediction task failed for run %s, dataset %s", run_id, inference_dataset_id
        )
        raise
