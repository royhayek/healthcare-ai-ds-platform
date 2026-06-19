"""Dataset management endpoints (§8).

POST /projects/{project_id}/datasets                         - upload CSV/Parquet, profile, persist
GET  /projects/{project_id}/datasets                         - list datasets for a project
GET  /projects/{project_id}/datasets/{dataset_id}
GET  /projects/{project_id}/datasets/{dataset_id}/preview    - first N rows
GET  /projects/{project_id}/datasets/{dataset_id}/plots      - list dataset-level plots
GET  /projects/{project_id}/datasets/{dataset_id}/plots/{plot_id}  - single plot (base64 PNG)
GET  /projects/{project_id}/datasets/{dataset_id}/plots/vs/{ref_id} - comparison plots
"""

import hashlib
import io
from typing import Annotated

import pandas as pd
from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, UploadFile, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.core.auth import get_current_user
from backend.core.database import Dataset, Project, get_db
from backend.core.storage import storage
from backend.ml.profiler import profile_dataset
from backend.models.dataset import VALID_ROLES, DatasetResponse

router = APIRouter(prefix="/projects/{project_id}/datasets", tags=["datasets"])

_PREVIEW_ROWS = 20
_MAX_FILE_BYTES = 500 * 1024 * 1024  # 500 MB


async def _get_project_or_404(
    project_id: str, user_id: str, db: AsyncSession
) -> Project:
    result = await db.execute(
        select(Project).where(Project.id == project_id, Project.user_id == user_id)
    )
    project = result.scalar_one_or_none()
    if project is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Project not found")
    return project


@router.post("", status_code=status.HTTP_201_CREATED, response_model=DatasetResponse)
async def upload_dataset(
    project_id: str,
    background_tasks: BackgroundTasks,
    file: Annotated[UploadFile, File(description="CSV or Parquet file")],
    role: Annotated[str, Form()] = "training",
    target_column: Annotated[str | None, Form()] = None,
    user_id: str = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> DatasetResponse:
    if role not in VALID_ROLES:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"role must be one of {sorted(VALID_ROLES)}",
        )

    await _get_project_or_404(project_id, user_id, db)

    raw = await file.read()
    if len(raw) > _MAX_FILE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds {_MAX_FILE_BYTES // 1024 // 1024} MB limit",
        )

    filename = file.filename or "upload.csv"
    sha256 = hashlib.sha256(raw).hexdigest()

    # Parse to validate and profile
    try:
        df = _parse_df(raw, filename)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Could not parse file: {exc}",
        ) from exc

    if target_column and target_column not in df.columns:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"target_column '{target_column}' not found in dataset",
        )

    prof = profile_dataset(df, target_column=target_column)
    schema_hash = _schema_hash(df)

    # Persist to storage
    storage_path = f"projects/{project_id}/datasets/{sha256}/{filename}"
    await storage.upload(storage_path, raw)

    dataset = Dataset(
        project_id=project_id,
        role=role,
        filename=filename,
        storage_path=storage_path,
        file_size_bytes=len(raw),
        sha256=sha256,
        schema_hash=schema_hash,
        row_count=prof.n_rows,
        col_count=prof.n_cols,
        target_column=target_column,
        task_type=prof.task_type,
        profile=prof.model_dump(mode="json"),
    )
    db.add(dataset)
    await db.commit()
    await db.refresh(dataset)

    # Trigger instant EDA plots as a background task (no Celery/Redis needed)
    from backend.tasks.dataset_plot_task import run_comparison_plots, run_dataset_plots

    background_tasks.add_task(run_dataset_plots, dataset.id, project_id)

    # If this is an inference/comparison dataset, also render comparison plots
    # against the project's most recent training dataset
    if role in ("inference", "comparison"):
        training_ds = await _find_training_dataset(project_id, db)
        if training_ds is not None:
            background_tasks.add_task(run_comparison_plots, training_ds.id, dataset.id)

    return DatasetResponse.model_validate(dataset)


@router.get("", response_model=list[DatasetResponse])
async def list_datasets(
    project_id: str,
    user_id: str = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> list[DatasetResponse]:
    await _get_project_or_404(project_id, user_id, db)
    result = await db.execute(
        select(Dataset)
        .where(Dataset.project_id == project_id)
        .order_by(Dataset.created_at.asc())
    )
    return [DatasetResponse.model_validate(d) for d in result.scalars()]


@router.get("/{dataset_id}", response_model=DatasetResponse)
async def get_dataset(
    project_id: str,
    dataset_id: str,
    user_id: str = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> DatasetResponse:
    await _get_project_or_404(project_id, user_id, db)
    result = await db.execute(
        select(Dataset).where(Dataset.id == dataset_id, Dataset.project_id == project_id)
    )
    dataset = result.scalar_one_or_none()
    if dataset is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Dataset not found")
    return DatasetResponse.model_validate(dataset)


@router.delete("/{dataset_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_dataset(
    project_id: str,
    dataset_id: str,
    user_id: str = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> None:
    """Delete a dataset and its stored file.

    Referential integrity is handled by the schema: dataset joins cascade,
    while runs (training/holdout) and predictions (inference) have their
    dataset references set to NULL - a completed run keeps its results even
    after its source dataset is removed. The stored file is deleted on a
    best-effort basis so a missing object never blocks the DB delete.
    """
    await _get_project_or_404(project_id, user_id, db)

    result = await db.execute(
        select(Dataset).where(Dataset.id == dataset_id, Dataset.project_id == project_id)
    )
    dataset = result.scalar_one_or_none()
    if dataset is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Dataset not found")

    # Best-effort storage cleanup - a missing object must not block deletion.
    try:
        if await storage.exists(dataset.storage_path):
            await storage.delete(dataset.storage_path)
    except Exception:  # noqa: BLE001 - storage cleanup is non-critical
        pass

    await db.delete(dataset)
    await db.commit()


@router.patch("/{dataset_id}/role", response_model=DatasetResponse)
async def update_dataset_role(
    project_id: str,
    dataset_id: str,
    body: dict,
    user_id: str = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> DatasetResponse:
    await _get_project_or_404(project_id, user_id, db)
    result = await db.execute(
        select(Dataset).where(Dataset.id == dataset_id, Dataset.project_id == project_id)
    )
    dataset = result.scalar_one_or_none()
    if dataset is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Dataset not found")

    role = body.get("role")
    if role not in VALID_ROLES:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"role must be one of {sorted(VALID_ROLES)}",
        )

    dataset.role = role
    db.add(dataset)
    await db.commit()
    await db.refresh(dataset)
    return DatasetResponse.model_validate(dataset)


@router.patch("/{dataset_id}/target-column", response_model=DatasetResponse)
async def update_target_column(
    project_id: str,
    dataset_id: str,
    body: dict,
    user_id: str = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> DatasetResponse:
    await _get_project_or_404(project_id, user_id, db)
    result = await db.execute(
        select(Dataset).where(Dataset.id == dataset_id, Dataset.project_id == project_id)
    )
    dataset = result.scalar_one_or_none()
    if dataset is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Dataset not found")

    target_column = body.get("target_column")
    if not target_column or not isinstance(target_column, str):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="target_column must be a non-empty string",
        )

    # Validate the column actually exists in the dataset
    # profile.columns is stored as a list of {name: ...} dicts
    col_dicts = (dataset.profile or {}).get("columns", [])
    columns = [c["name"] for c in col_dicts if isinstance(c, dict) and "name" in c]
    if columns and target_column not in columns:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"target_column '{target_column}' not found in dataset columns",
        )

    dataset.target_column = target_column
    db.add(dataset)
    await db.commit()
    await db.refresh(dataset)
    return DatasetResponse.model_validate(dataset)


@router.get("/{dataset_id}/preview")
async def preview_dataset(
    project_id: str,
    dataset_id: str,
    rows: int = _PREVIEW_ROWS,
    user_id: str = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> dict:
    await _get_project_or_404(project_id, user_id, db)
    result = await db.execute(
        select(Dataset).where(Dataset.id == dataset_id, Dataset.project_id == project_id)
    )
    dataset = result.scalar_one_or_none()
    if dataset is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Dataset not found")

    raw = await storage.download(dataset.storage_path)
    df = _parse_df(raw, dataset.filename)
    preview = df.head(min(rows, _PREVIEW_ROWS))
    return {
        "columns": list(df.columns),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "rows": preview.fillna("").to_dict(orient="records"),
        "total_rows": len(df),
    }


@router.get("/{dataset_id}/plots")
async def list_dataset_plots(
    project_id: str,
    dataset_id: str,
    user_id: str = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> dict:
    """Return the manifest of EDA plots for a dataset.

    The response is ``{plots, complete, error}``: the full planned plot set
    (each carrying a ``status`` of ready/failed/pending), a ``complete`` flag so
    the UI stops the loading indicator deterministically, and an ``error``
    string when the dataset could not be loaded for row-level plots.
    """
    await _get_project_or_404(project_id, user_id, db)
    import json

    manifest_path = f"datasets/{dataset_id}/plots/manifest.json"
    if not await storage.exists(manifest_path):
        return {"plots": [], "complete": False, "error": None}

    manifest = json.loads(await storage.download(manifest_path))

    status: dict = {}
    status_path = f"datasets/{dataset_id}/plots/render.json"
    if await storage.exists(status_path):
        try:
            status = json.loads(await storage.download(status_path))
        except Exception:  # noqa: BLE001 - corrupt sidecar means "still rendering"
            status = {}
    failed = set(status.get("failed", []))

    plots = []
    for entry in manifest:
        has_image = await storage.exists(
            f"datasets/{dataset_id}/plots/{entry['plot_id']}.png"
        )
        plot_status = (
            "ready" if has_image
            else "failed" if entry["plot_id"] in failed
            else "pending"
        )
        plots.append({**entry, "has_image": has_image, "status": plot_status})

    return {
        "plots": plots,
        "complete": bool(status.get("complete", False)),
        "error": status.get("df_error"),
    }


@router.get("/{dataset_id}/plots/{plot_id}")
async def get_dataset_plot(
    project_id: str,
    dataset_id: str,
    plot_id: str,
    user_id: str = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> dict:
    """Return a single dataset-level plot as a base64-encoded PNG."""
    await _get_project_or_404(project_id, user_id, db)
    png_path = f"datasets/{dataset_id}/plots/{plot_id}.png"
    if not await storage.exists(png_path):
        raise HTTPException(status_code=404, detail=f"Plot '{plot_id}' not yet rendered")
    import base64
    import json
    png_bytes = await storage.download(png_path)
    meta: dict = {}
    meta_path = f"datasets/{dataset_id}/plots/{plot_id}.meta.json"
    if await storage.exists(meta_path):
        meta = json.loads(await storage.download(meta_path))
    return {**meta, "image_b64": base64.b64encode(png_bytes).decode()}


@router.get("/{dataset_id}/plots/vs/{reference_dataset_id}")
async def list_comparison_plots(
    project_id: str,
    dataset_id: str,
    reference_dataset_id: str,
    user_id: str = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> list[dict]:
    """Return the comparison plot manifest for this dataset vs a reference (training) dataset.

    Triggered automatically when an inference/comparison dataset is uploaded.
    Returns [] while plots are still rendering.
    """
    await _get_project_or_404(project_id, user_id, db)
    manifest_path = f"datasets/{dataset_id}/plots/vs_{reference_dataset_id}/manifest.json"
    if not await storage.exists(manifest_path):
        return []
    import json
    raw = await storage.download(manifest_path)
    return json.loads(raw)


async def _find_training_dataset(
    project_id: str, db: AsyncSession
) -> "Dataset | None":
    """Return the most recently uploaded training-role dataset for a project."""
    from sqlalchemy import desc
    result = await db.execute(
        select(Dataset)
        .where(Dataset.project_id == project_id, Dataset.role == "training")
        .order_by(desc(Dataset.created_at))
        .limit(1)
    )
    return result.scalar_one_or_none()


def _parse_df(raw: bytes, filename: str) -> pd.DataFrame:
    if filename.endswith(".parquet"):
        return pd.read_parquet(io.BytesIO(raw))
    return pd.read_csv(io.BytesIO(raw))


def _schema_hash(df: pd.DataFrame) -> str:
    schema = {col: str(dtype) for col, dtype in df.dtypes.items()}
    import json
    return hashlib.sha256(
        json.dumps(schema, sort_keys=True).encode()
    ).hexdigest()[:16]
