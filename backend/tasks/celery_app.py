"""Celery application instance (§22).

Import this module to access the shared Celery app. Tasks are registered
by importing their modules (done in analysis_task.py).
"""

from celery import Celery

from backend.core.config import settings

celery_app = Celery(
    "ai_ds_platform",
    broker=settings.REDIS_URL,
    backend=settings.REDIS_URL,
    include=[
        "backend.tasks.analysis_task",
        "backend.tasks.deliverable_task",
        "backend.tasks.batch_prediction_task",
    ],
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_acks_late=True,
    worker_prefetch_multiplier=1,  # one task at a time per worker (pipeline tasks are heavy)
)
