"""Multi-dataset join endpoints (§7).

Covers the "Kaggle multi-file" pattern: upload train.csv + store.csv + features.csv,
define join relationships between them, materialize the result as a new Dataset,
and let the pipeline run on the joined dataset.

Flow:
  1. Upload each file as a Dataset (same project, role=training or supplementary).
  2. POST /projects/{id}/joins/suggest   → auto-detect join key candidates
  3. POST /projects/{id}/joins           → execute join, materialize, store result
  4. GET  /projects/{id}/joins           → list all join definitions in the project
  5. The materialized result Dataset shows in the project's dataset list with
     role=training and can be selected when starting a pipeline run.

Chaining joins (3+ files):
  Join A + B → dataset_ab, then join dataset_ab + C → dataset_abc.
  Each step is a separate POST /joins call.
"""

from __future__ import annotations

import hashlib
import io
import logging
from typing import Any

import pandas as pd
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.core.auth import get_current_user
from backend.core.database import Dataset, DatasetJoin, Project, get_db
from backend.core.storage import storage
from backend.ml.joiner import JoinType, auto_detect_join_keys, join_datasets
from backend.ml.profiler import compress_profile_for_claude, profile_dataset
from backend.models.dataset import DatasetResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/projects/{project_id}/joins", tags=["joins"])

_MAX_ROWS_SYNC = 500_000  # rows; above this we'd want async - fine for Kaggle files


# ── Request / response models ──────────────────────────────────────────────────


class JoinSuggestRequest(BaseModel):
    left_dataset_id: str
    right_dataset_id: str


class JoinKeyCandidate(BaseModel):
    column: str
    left_unique: int
    right_unique: int
    overlap_pct: float  # fraction of right values found in left
    recommended: bool  # True = high confidence


class JoinSuggestResponse(BaseModel):
    left_dataset_id: str
    right_dataset_id: str
    candidates: list[JoinKeyCandidate]
    recommended_join_type: str  # left | inner
    note: str


class JoinCreateRequest(BaseModel):
    left_dataset_id: str
    right_dataset_id: str
    join_type: JoinType = "left"
    join_keys: list[str]
    result_filename: str | None = None  # defaults to "{left}_{right}_joined.csv"


class JoinRecord(BaseModel):
    id: str
    left_dataset_id: str
    right_dataset_id: str
    result_dataset_id: str | None
    join_type: str
    join_keys: list[str]
    rows_before_left: int | None
    rows_before_right: int | None
    rows_after: int | None

    model_config = {"from_attributes": True}


# ── Endpoints ──────────────────────────────────────────────────────────────────


@router.post("/suggest", response_model=JoinSuggestResponse)
async def suggest_join_keys(
    project_id: str,
    body: JoinSuggestRequest,
    user_id: str = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> JoinSuggestResponse:
    """Auto-detect candidate join keys between two datasets.

    Loads both files, finds columns that share the same name and have
    significant value overlap - the same heuristic data scientists use
    when exploring an unfamiliar Kaggle dataset.
    """
    await _get_project_or_404(project_id, user_id, db)

    df_left, left_ds = await _load_dataset(body.left_dataset_id, project_id, db)
    df_right, right_ds = await _load_dataset(body.right_dataset_id, project_id, db)

    shared_cols = set(df_left.columns) & set(df_right.columns)
    candidates: list[JoinKeyCandidate] = []

    for col in sorted(shared_cols):
        left_vals = set(df_left[col].dropna().unique())
        right_vals = set(df_right[col].dropna().unique())
        if not left_vals or not right_vals:
            continue

        overlap = len(left_vals & right_vals) / len(right_vals)
        cardinality_ratio = len(left_vals) / max(len(df_left), 1)

        # Recommended: high overlap AND the column looks like a key (not a low-cardinality category)
        recommended = overlap >= 0.5 and (cardinality_ratio > 0.01 or len(left_vals) > 5)

        candidates.append(
            JoinKeyCandidate(
                column=col,
                left_unique=len(left_vals),
                right_unique=len(right_vals),
                overlap_pct=round(overlap, 4),
                recommended=recommended,
            )
        )

    # Sort: recommended first, then by overlap descending
    candidates.sort(key=lambda c: (-int(c.recommended), -c.overlap_pct))

    # Recommend join type: if right table has fewer unique key combos → dimension table → LEFT join
    recommended_type = "left"
    if candidates:
        best_key = candidates[0].column
        if df_right[best_key].nunique() < df_left[best_key].nunique() * 0.5:
            recommended_type = "left"  # right is smaller → dimension table → keep all left rows
        else:
            recommended_type = "inner"

    note = (
        f"{left_ds.filename} has {len(df_left):,} rows, "
        f"{right_ds.filename} has {len(df_right):,} rows. "
    )
    if candidates:
        best = candidates[0]
        note += (
            f"Best candidate key: '{best.column}' "
            f"({best.overlap_pct*100:.0f}% overlap). "
        )
    note += f"Recommended join type: {recommended_type}."

    return JoinSuggestResponse(
        left_dataset_id=body.left_dataset_id,
        right_dataset_id=body.right_dataset_id,
        candidates=candidates,
        recommended_join_type=recommended_type,
        note=note,
    )


@router.post("", status_code=status.HTTP_201_CREATED, response_model=DatasetResponse)
async def create_join(
    project_id: str,
    background_tasks: BackgroundTasks,
    body: JoinCreateRequest,
    user_id: str = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> DatasetResponse:
    """Execute a join between two datasets and materialise the result.

    The joined DataFrame is:
    - Profiled (same as any uploaded dataset)
    - Stored as a new Dataset in this project (role=training)
    - Immediately queued for EDA plot rendering

    The DatasetJoin record links the two source datasets to the result so the
    lineage is preserved in the audit trail.
    """
    await _get_project_or_404(project_id, user_id, db)

    df_left, left_ds = await _load_dataset(body.left_dataset_id, project_id, db)
    df_right, right_ds = await _load_dataset(body.right_dataset_id, project_id, db)

    total_rows = len(df_left) + len(df_right)
    if total_rows > _MAX_ROWS_SYNC:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=(
                f"Combined row count ({total_rows:,}) exceeds the synchronous join limit "
                f"({_MAX_ROWS_SYNC:,}). Split the operation or use a smaller sample."
            ),
        )

    # Validate join keys exist in both DataFrames
    for key in body.join_keys:
        if key not in df_left.columns:
            raise HTTPException(
                status_code=422,
                detail=f"Join key '{key}' not found in {left_ds.filename}",
            )
        if key not in df_right.columns:
            raise HTTPException(
                status_code=422,
                detail=f"Join key '{key}' not found in {right_ds.filename}",
            )

    # Execute join
    try:
        df_joined, join_result = join_datasets(df_left, df_right, body.join_type, body.join_keys)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Join failed: {exc}") from exc

    if df_joined.empty:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Join produced 0 rows. Check that the key values overlap - "
                f"got {join_result.rows_before_left:,} left rows and "
                f"{join_result.rows_before_right:,} right rows."
            ),
        )

    # Materialise to CSV
    result_filename = body.result_filename or (
        f"{left_ds.filename.rsplit('.', 1)[0]}_{right_ds.filename.rsplit('.', 1)[0]}_joined.csv"
    )
    csv_bytes = df_joined.to_csv(index=False).encode("utf-8")
    sha256 = hashlib.sha256(csv_bytes).hexdigest()
    storage_path = f"projects/{project_id}/datasets/{sha256}/{result_filename}"
    await storage.upload(storage_path, csv_bytes, "text/csv")

    # Profile the joined result
    target_col = left_ds.target_column  # inherit target from the left (primary) dataset
    prof = profile_dataset(df_joined, target_column=target_col)

    schema_hash = hashlib.sha256(
        str({col: str(dt) for col, dt in df_joined.dtypes.items()}).encode()
    ).hexdigest()[:16]

    # Persist result as a new Dataset
    result_ds = Dataset(
        project_id=project_id,
        role=left_ds.role,  # inherit role from left dataset
        filename=result_filename,
        storage_path=storage_path,
        file_size_bytes=len(csv_bytes),
        sha256=sha256,
        schema_hash=schema_hash,
        row_count=prof.n_rows,
        col_count=prof.n_cols,
        target_column=target_col,
        task_type=prof.task_type,
        profile=prof.model_dump(mode="json"),
    )
    db.add(result_ds)
    await db.flush()

    # Persist join record for lineage
    join_record = DatasetJoin(
        left_dataset_id=body.left_dataset_id,
        right_dataset_id=body.right_dataset_id,
        join_type=body.join_type,
        join_keys={"keys": body.join_keys},
        rows_before=join_result.rows_before_left,
        rows_after=join_result.rows_after,
    )
    db.add(join_record)
    await db.commit()
    await db.refresh(result_ds)

    # Trigger instant EDA plots as a background task
    from backend.tasks.dataset_plot_task import run_dataset_plots
    background_tasks.add_task(run_dataset_plots, result_ds.id, project_id)

    logger.info(
        "Join %s + %s → %s (%d rows, %d cols)",
        left_ds.filename, right_ds.filename, result_filename,
        prof.n_rows, prof.n_cols,
    )

    return DatasetResponse.model_validate(result_ds)


@router.get("", response_model=list[JoinRecord])
async def list_joins(
    project_id: str,
    user_id: str = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> list[JoinRecord]:
    """List all join definitions for a project."""
    await _get_project_or_404(project_id, user_id, db)

    # Get all dataset IDs in this project
    ds_ids_result = await db.execute(
        select(Dataset.id).where(Dataset.project_id == project_id)
    )
    ds_ids = {row[0] for row in ds_ids_result}

    joins_result = await db.execute(
        select(DatasetJoin).where(DatasetJoin.left_dataset_id.in_(ds_ids))
    )
    joins = joins_result.scalars().all()

    return [
        JoinRecord(
            id=j.id,
            left_dataset_id=j.left_dataset_id,
            right_dataset_id=j.right_dataset_id,
            result_dataset_id=None,  # could be stored in future iteration
            join_type=j.join_type,
            join_keys=j.join_keys.get("keys", []) if isinstance(j.join_keys, dict) else [],
            rows_before_left=j.rows_before,
            rows_before_right=None,
            rows_after=j.rows_after,
        )
        for j in joins
    ]


# ── Helpers ────────────────────────────────────────────────────────────────────


async def _get_project_or_404(
    project_id: str, user_id: str, db: AsyncSession
) -> Project:
    result = await db.execute(
        select(Project).where(Project.id == project_id, Project.user_id == user_id)
    )
    project = result.scalar_one_or_none()
    if project is None:
        raise HTTPException(status_code=404, detail="Project not found")
    return project


async def _load_dataset(
    dataset_id: str, project_id: str, db: AsyncSession
) -> tuple[pd.DataFrame, Dataset]:
    result = await db.execute(
        select(Dataset).where(Dataset.id == dataset_id, Dataset.project_id == project_id)
    )
    ds = result.scalar_one_or_none()
    if ds is None:
        raise HTTPException(status_code=404, detail=f"Dataset {dataset_id} not found")
    raw = await storage.download(ds.storage_path)
    df = (
        pd.read_parquet(io.BytesIO(raw))
        if ds.filename.endswith(".parquet")
        else pd.read_csv(io.BytesIO(raw))
    )
    return df, ds
