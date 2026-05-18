"""Pydantic models for dataset API (§8)."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel

VALID_ROLES = {"training", "inference", "holdout", "reference", "comparison"}


class DatasetResponse(BaseModel):
    id: str
    project_id: str
    role: str
    filename: str
    storage_path: str
    file_size_bytes: int | None
    sha256: str
    schema_hash: str
    row_count: int | None
    col_count: int | None
    target_column: str | None
    task_type: str | None
    profile: dict[str, Any] | None
    created_at: datetime

    model_config = {"from_attributes": True}
