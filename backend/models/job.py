"""Job state model - typed representation of a Celery task lifecycle (§22)."""

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

JobStatus = Literal[
    "queued",
    "running",
    "awaiting_checkpoint",
    "completed",
    "failed",
    "cancelled",
]


class JobProgress(BaseModel):
    """Snapshot of a running job's progress, emitted via SSE."""

    run_id: str
    step: str
    message: str
    pct: int  # 0-100


class JobState(BaseModel):
    """Full job state persisted to the Run ORM row and returned by GET /runs/{id}.

    This model is intentionally read-only - it is derived from the Run ORM row,
    not written to it directly. Writers go through the analysis_task pipeline.
    """

    run_id: str
    status: JobStatus
    current_step: str | None = None
    progress: int = 0
    celery_task_id: str | None = None

    started_at: datetime | None = None
    completed_at: datetime | None = None
    error_message: str | None = None

    # Pipeline step timestamps (populated as each step completes)
    step_timings: dict[str, float] = Field(default_factory=dict)

    @classmethod
    def from_run_orm(cls, run: Any) -> "JobState":
        """Construct a JobState from a SQLAlchemy Run ORM row."""
        return cls(
            run_id=str(run.id),
            status=run.status,
            current_step=run.current_step,
            progress=run.progress or 0,
            celery_task_id=run.job_id,
            completed_at=getattr(run, "completed_at", None),
            error_message=run.error_message,
        )
