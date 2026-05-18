"""Pydantic models for analysis run API (§22, §26)."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel


class RunCreate(BaseModel):
    training_dataset_id: str
    holdout_dataset_id: str | None = None
    threshold_config: dict[str, Any] | None = None


class CheckpointResumeRequest(BaseModel):
    """Request body for resuming a paused run from a checkpoint.

    strategy_override: optional partial strategy update applied before resuming.
    Example: {"model_selection": {"primary": "lightgbm"}} overrides the primary
    model before the training step runs.
    """
    strategy_override: dict[str, Any] | None = None
    threshold_config: dict[str, Any] | None = None


class RunResponse(BaseModel):
    id: str
    project_id: str
    training_dataset_id: str | None
    holdout_dataset_id: str | None
    job_id: str | None
    status: str
    current_step: str | None
    progress: int

    # Pipeline outputs
    eda_report: dict[str, Any] | None
    preprocessing_strategy: dict[str, Any] | None
    model_selection: dict[str, Any] | None
    model_comparison: list[Any] | None
    stat_tests: dict[str, Any] | None
    best_model_name: str | None
    best_model_score: float | None
    tuning_result: dict[str, Any] | None
    calibration_report: dict[str, Any] | None
    threshold_result: dict[str, Any] | None
    threshold_config: dict[str, Any] | None
    final_metrics: dict[str, Any] | None
    eval_plots: dict[str, Any] | None
    shap_summary: dict[str, Any] | None
    similarity_index_built: bool
    insight_report: str | None

    error_message: str | None
    created_at: datetime
    completed_at: datetime | None

    model_config = {"from_attributes": True}
