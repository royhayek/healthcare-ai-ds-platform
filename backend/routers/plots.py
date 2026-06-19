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


_RENDER_STATUS_PATH = "runs/{run_id}/plots/render_{stage}.json"
_KNOWN_STAGES = ("eda", "preprocessing", "preprocessing_after", "training", "drift")


class PlotSummary(BaseModel):
    plot_id: str
    plot_type: str
    title: str
    column: str | None
    priority: int
    stage: str
    has_image: bool
    status: str  # "ready" | "failed" | "pending"


class PlotManifestResponse(BaseModel):
    plots: list[PlotSummary]
    complete: bool
    error: str | None = None


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


async def _load_render_status(
    run_id: str, stage: str | None
) -> tuple[bool, str | None, set[str]]:
    """Return (complete, df_error, failed_plot_ids) from the render-status sidecar(s).

    When a stage is given, completion reflects exactly that stage. When no stage
    is given, completion requires every known stage that has started to be done.
    """
    stages = [stage] if stage else list(_KNOWN_STAGES)
    complete = True
    any_started = False
    error: str | None = None
    failed: set[str] = set()

    for s in stages:
        path = _RENDER_STATUS_PATH.format(run_id=run_id, stage=s)
        if not await storage.exists(path):
            if stage:  # the requested stage hasn't started rendering yet
                complete = False
            continue
        any_started = True
        try:
            status = json.loads(await storage.download(path))
        except Exception:  # noqa: BLE001 - a corrupt sidecar just means "still working"
            complete = False
            continue
        if not status.get("complete"):
            complete = False
        if status.get("df_error") and not error:
            error = status.get("df_error")
        failed.update(status.get("failed", []))

    if not any_started:
        complete = False
    return complete, error, failed


@router.get("/{run_id}/plots", response_model=PlotManifestResponse)
async def list_plots(
    run_id: str,
    stage: str | None = Query(default=None),
    db: AsyncSession = Depends(get_db),
    user_id: str = Depends(get_current_user),
) -> PlotManifestResponse:
    """Return the plot manifest for a run, optionally filtered by stage.

    The response includes a ``complete`` flag and per-plot ``status`` so the UI
    can show every planned plot up front and stop the loading indicator
    deterministically rather than guessing via a timeout.
    """
    run = await db.get(Run, run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")

    manifest_path = _PLOTS_INDEX_PATH.format(run_id=run_id)
    if not await storage.exists(manifest_path):
        return PlotManifestResponse(plots=[], complete=False, error=None)

    raw = await storage.download(manifest_path)
    manifest: list[dict[str, Any]] = json.loads(raw)
    complete, error, failed = await _load_render_status(run_id, stage)

    results: list[PlotSummary] = []
    for entry in manifest:
        if stage and entry.get("stage") != stage:
            continue
        png_path = _PLOT_PNG_PATH.format(run_id=run_id, plot_id=entry["plot_id"])
        has_image = await storage.exists(png_path)
        status = (
            "ready" if has_image
            else "failed" if entry["plot_id"] in failed
            else "pending"
        )
        results.append(
            PlotSummary(
                plot_id=entry["plot_id"],
                plot_type=entry["plot_type"],
                title=entry["title"],
                column=entry.get("column"),
                priority=entry.get("priority", 1),
                stage=entry.get("stage", "eda"),
                has_image=has_image,
                status=status,
            )
        )
    return PlotManifestResponse(plots=results, complete=complete, error=error)


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
