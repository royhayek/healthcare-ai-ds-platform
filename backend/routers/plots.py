"""Plot API endpoints (§12, Category B1).

GET  /runs/{run_id}/plots              - list all plots for a run (by stage)
GET  /runs/{run_id}/plots/{plot_id}    - get a single plot as base64 PNG
POST /runs/{run_id}/plots              - trigger on-demand plot generation
"""

from __future__ import annotations

import json
import logging
from typing import Any

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from backend.core.auth import get_current_user
from backend.core.database import Dataset, Run, get_db
from backend.core.storage import storage

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/runs", tags=["plots"])

_PLOTS_INDEX_PATH = "runs/{run_id}/plots/manifest.json"
_PLOT_PNG_PATH = "runs/{run_id}/plots/{plot_id}.png"
_PLOT_META_PATH = "runs/{run_id}/plots/{plot_id}.meta.json"


class PlotSummary(BaseModel):
    plot_id: str
    plot_type: str
    title: str
    column: str | None
    priority: int
    stage: str
    has_image: bool


class PlotDetail(BaseModel):
    plot_id: str
    plot_type: str
    title: str
    column: str | None
    priority: int
    stage: str
    image_b64: str  # base64-encoded PNG


class GeneratePlotsRequest(BaseModel):
    stage: str = "eda"
    column: str | None = None  # when None: generate all plots for the stage


@router.get("/{run_id}/plots", response_model=list[PlotSummary])
async def list_plots(
    run_id: str,
    stage: str | None = Query(default=None),
    db: AsyncSession = Depends(get_db),
    user_id: str = Depends(get_current_user),
) -> list[PlotSummary]:
    """Return the plot manifest for a run, optionally filtered by stage."""
    run = await db.get(Run, run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")

    manifest_path = _PLOTS_INDEX_PATH.format(run_id=run_id)
    if not await storage.exists(manifest_path):
        return []

    raw = await storage.download(manifest_path)
    manifest: list[dict[str, Any]] = json.loads(raw)

    results: list[PlotSummary] = []
    for entry in manifest:
        if stage and entry.get("stage") != stage:
            continue
        png_path = _PLOT_PNG_PATH.format(run_id=run_id, plot_id=entry["plot_id"])
        has_image = await storage.exists(png_path)
        results.append(
            PlotSummary(
                plot_id=entry["plot_id"],
                plot_type=entry["plot_type"],
                title=entry["title"],
                column=entry.get("column"),
                priority=entry.get("priority", 1),
                stage=entry.get("stage", "eda"),
                has_image=has_image,
            )
        )
    return results


@router.get("/{run_id}/plots/{plot_id}", response_model=PlotDetail)
async def get_plot(
    run_id: str,
    plot_id: str,
    db: AsyncSession = Depends(get_db),
    user_id: str = Depends(get_current_user),
) -> PlotDetail:
    """Return a single plot as a base64-encoded PNG."""
    run = await db.get(Run, run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")

    png_path = _PLOT_PNG_PATH.format(run_id=run_id, plot_id=plot_id)
    if not await storage.exists(png_path):
        raise HTTPException(status_code=404, detail=f"Plot {plot_id} not found or not yet rendered")

    meta_path = _PLOT_META_PATH.format(run_id=run_id, plot_id=plot_id)
    meta: dict[str, Any] = {}
    if await storage.exists(meta_path):
        meta = json.loads(await storage.download(meta_path))

    import base64

    png_bytes = await storage.download(png_path)
    image_b64 = base64.b64encode(png_bytes).decode("utf-8")

    return PlotDetail(
        plot_id=plot_id,
        plot_type=meta.get("plot_type", "unknown"),
        title=meta.get("title", plot_id),
        column=meta.get("column"),
        priority=meta.get("priority", 1),
        stage=meta.get("stage", "eda"),
        image_b64=image_b64,
    )


@router.post("/{run_id}/plots", status_code=202)
async def generate_plots(
    run_id: str,
    background_tasks: BackgroundTasks,
    body: GeneratePlotsRequest,
    db: AsyncSession = Depends(get_db),
    user_id: str = Depends(get_current_user),
) -> dict[str, str]:
    """Trigger on-demand plot generation for a pipeline stage.

    Responds 202 Accepted immediately - poll GET /runs/{run_id}/plots to
    watch plots appear.
    """
    run = await db.get(Run, run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")

    from backend.tasks.plot_task import run_plots

    background_tasks.add_task(run_plots, run_id, body.stage, body.column)
    return {"status": "queued", "stage": body.stage}
