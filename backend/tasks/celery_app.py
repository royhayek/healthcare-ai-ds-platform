"""Celery application instance (§22).

Import this module to access the shared Celery app. Tasks are registered
by importing their modules (done in analysis_task.py).
"""

# ── macOS fork safety ─────────────────────────────────────────────────────────
# Celery's default `prefork` pool uses fork() without exec(). On macOS the native
# BLAS/OpenMP threadpools behind numpy/scikit-learn are NOT fork-safe, which
# manifests as a worker dying with SIGSEGV (signal 11) the moment a forked child
# touches numpy. Pinning the native threadpools to a single thread BEFORE numpy
# is imported anywhere makes the prefork pool safe. These must be set before the
# first numpy import in this process; `setdefault` lets an operator override.
import os

for _var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_var, "1")
# Allow Objective-C runtimes (pulled in transitively on macOS) to be used after
# fork() instead of aborting the child process.
os.environ.setdefault("OBJC_DISABLE_INITIALIZE_FORK_SAFETY", "YES")

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
