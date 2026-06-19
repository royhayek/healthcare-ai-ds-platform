"""Celery task: render visualisation plots immediately after dataset upload.

Fires from POST /projects/{project_id}/datasets as soon as the file is
committed to storage. This is the "instant EDA" layer - plots live at the
dataset level, independent of any pipeline run.

Two modes:
  render_dataset_plots   - single dataset (training / holdout / reference)
  render_comparison_plots - two datasets side-by-side (training vs inference)

Storage layout:
  datasets/{dataset_id}/plots/{plot_id}.png
  datasets/{dataset_id}/plots/manifest.json
  datasets/{dataset_id}/plots/vs_{reference_dataset_id}/{plot_id}.png
  datasets/{dataset_id}/plots/vs_{reference_dataset_id}/manifest.json
"""

import asyncio
import base64
import io
import json
import logging
from typing import Any

from backend.core.config import settings
from backend.tasks.celery_app import celery_app

logger = logging.getLogger(__name__)


# ── Public async entry points (used by FastAPI BackgroundTasks) ───────────────


async def run_dataset_plots(dataset_id: str, project_id: str) -> None:
    """Async entry point for FastAPI BackgroundTasks - no Celery/Redis needed."""
    await _async_single(dataset_id, project_id)


async def run_comparison_plots(training_dataset_id: str, inference_dataset_id: str) -> None:
    """Async entry point for FastAPI BackgroundTasks - no Celery/Redis needed."""
    await _async_compare(training_dataset_id, inference_dataset_id)


# ── Single-dataset plots ───────────────────────────────────────────────────────


@celery_app.task(bind=True, name="dataset.plots.render", max_retries=0)
def render_dataset_plots_task(
    self,  # type: ignore[misc]
    dataset_id: str,
    project_id: str,
) -> None:
    """Render exploratory plots for a single dataset immediately after upload."""
    try:
        asyncio.run(_async_single(dataset_id, project_id))
    except Exception as exc:
        logger.exception("Dataset plot render failed for %s", dataset_id)
        raise


async def _async_single(dataset_id: str, project_id: str) -> None:
    from sqlalchemy import select
    from sqlalchemy.ext.asyncio import AsyncSession as _AS
    from sqlalchemy.ext.asyncio import async_sessionmaker as _asf
    from sqlalchemy.ext.asyncio import create_async_engine as _cae

    import pandas as pd

    from backend.core.database import Dataset
    from backend.ml.profiler import DatasetProfile, compress_profile_for_claude
    from backend.agents.plot_selector_agent import select_plots_for_stage
    from backend.tasks.plot_task import load_dataframe_for_plots, render_specs_to_storage

    local_engine = _cae(settings.DATABASE_URL, pool_pre_ping=True)
    local_factory = _asf(local_engine, class_=_AS, expire_on_commit=False)

    try:
        async with local_factory() as session:
            ds = (
                await session.execute(select(Dataset).where(Dataset.id == dataset_id))
            ).scalar_one_or_none()
            if ds is None:
                logger.warning("Dataset %s not found - aborting plot render", dataset_id)
                return

            profile: DatasetProfile | None = None
            if ds.profile:
                try:
                    profile = DatasetProfile.model_validate(ds.profile)
                except Exception:
                    pass

            # Load a sampled DataFrame for row-level plots. A load failure is
            # reported through the manifest, never swallowed (that is what made
            # the page show only the 2-3 profile-based plots).
            df, df_error = await load_dataframe_for_plots(ds)

            compressed: dict[str, Any] = {}
            if profile:
                compressed = compress_profile_for_claude(profile)

            task_type = ds.task_type or compressed.get("task_type", "unknown")

            manifest = await select_plots_for_stage(
                compressed, stage="eda", task_type=task_type, run_id=dataset_id
            )

            await render_specs_to_storage(
                manifest.plots,
                base_dir=f"datasets/{dataset_id}/plots",
                profile=profile,
                df=df,
                df_error=df_error,
            )

    finally:
        await local_engine.dispose()


# ── Two-dataset comparison plots ───────────────────────────────────────────────


@celery_app.task(bind=True, name="dataset.plots.compare", max_retries=0)
def render_comparison_plots_task(
    self,  # type: ignore[misc]
    training_dataset_id: str,
    inference_dataset_id: str,
) -> None:
    """Render side-by-side comparison plots (training vs inference/comparison dataset).

    Automatically triggered when an inference or comparison-role dataset is
    uploaded to a project that already has a training dataset.
    """
    try:
        asyncio.run(_async_compare(training_dataset_id, inference_dataset_id))
    except Exception as exc:
        logger.exception(
            "Comparison plot render failed for %s vs %s",
            training_dataset_id, inference_dataset_id,
        )
        raise


async def _async_compare(train_id: str, inf_id: str) -> None:
    from sqlalchemy import select
    from sqlalchemy.ext.asyncio import AsyncSession as _AS
    from sqlalchemy.ext.asyncio import async_sessionmaker as _asf
    from sqlalchemy.ext.asyncio import create_async_engine as _cae

    import pandas as pd

    from backend.core.database import Dataset
    from backend.core.storage import storage
    from backend.ml.drift import compute_drift_report
    from backend.ml.plotter import PlotRenderer, PlotSpec, PlotType

    local_engine = _cae(settings.DATABASE_URL, pool_pre_ping=True)
    local_factory = _asf(local_engine, class_=_AS, expire_on_commit=False)

    try:
        async with local_factory() as session:
            train_ds = (
                await session.execute(select(Dataset).where(Dataset.id == train_id))
            ).scalar_one_or_none()
            inf_ds = (
                await session.execute(select(Dataset).where(Dataset.id == inf_id))
            ).scalar_one_or_none()

            if train_ds is None or inf_ds is None:
                logger.warning("Cannot compare: dataset %s or %s not found", train_id, inf_id)
                return

            def _load(ds: Dataset) -> pd.DataFrame:
                import asyncio as _asyncio

                raw = _asyncio.get_event_loop().run_until_complete(
                    storage.download(ds.storage_path)
                )
                return (
                    pd.read_parquet(io.BytesIO(raw))
                    if ds.filename.endswith(".parquet")
                    else pd.read_csv(io.BytesIO(raw))
                )

            # Load both datasets with sampling
            try:
                train_raw = await storage.download(train_ds.storage_path)
                df_train = (
                    pd.read_parquet(io.BytesIO(train_raw))
                    if train_ds.filename.endswith(".parquet")
                    else pd.read_csv(io.BytesIO(train_raw))
                )
                df_train = df_train.sample(min(5000, len(df_train)), random_state=42)
            except Exception as exc:
                logger.warning("Could not load training dataset for comparison: %s", exc)
                return

            try:
                inf_raw = await storage.download(inf_ds.storage_path)
                df_inf = (
                    pd.read_parquet(io.BytesIO(inf_raw))
                    if inf_ds.filename.endswith(".parquet")
                    else pd.read_csv(io.BytesIO(inf_raw))
                )
                df_inf = df_inf.sample(min(5000, len(df_inf)), random_state=42)
            except Exception as exc:
                logger.warning("Could not load inference dataset for comparison: %s", exc)
                return

            # Compute PSI for numeric columns present in both datasets
            common_numeric = [
                c for c in df_train.select_dtypes(include="number").columns
                if c in df_inf.columns
            ]
            common_cat = [
                c for c in df_train.select_dtypes(exclude="number").columns
                if c in df_inf.columns
            ]

            psi_values: dict[str, float] = {}
            try:
                drift = compute_drift_report(df_train, df_inf, common_numeric, common_cat)
                psi_values = {
                    f.feature: f.psi
                    for f in drift.features
                    if f.psi is not None
                }
            except Exception as exc:
                logger.debug("Drift compute during comparison plotting failed: %s", exc)

            renderer = PlotRenderer()
            rendered: list[dict[str, Any]] = []
            manifest_dir = f"datasets/{inf_id}/plots/vs_{train_id}"
            comparison_manifest_path = f"{manifest_dir}/manifest.json"

            async def _save_comparison_plot(spec: "PlotSpec", b64: str) -> None:  # type: ignore[name-defined]
                await storage.upload(
                    f"{manifest_dir}/{spec.plot_id}.png",
                    base64.b64decode(b64), "image/png",
                )
                await storage.upload(
                    f"{manifest_dir}/{spec.plot_id}.meta.json",
                    json.dumps(spec.to_dict()).encode(), "application/json",
                )
                rendered.append(spec.to_dict())
                await storage.upload(
                    comparison_manifest_path,
                    json.dumps(rendered).encode(), "application/json",
                )

            # PSI ranked bar chart
            if psi_values:
                psi_spec = PlotSpec(
                    plot_id="psi_bar_comparison",
                    plot_type=PlotType.PSI_BAR,
                    title=f"Feature drift: {inf_ds.filename} vs {train_ds.filename}",
                    priority=1,
                    stage="comparison",
                    extra={"psi_values": psi_values},
                )
                b64 = renderer.render(psi_spec, df=df_train, df_reference=df_inf)
                if b64:
                    await _save_comparison_plot(psi_spec, b64)

            # Top drifted features: KDE overlays
            top_drifted = sorted(psi_values, key=lambda c: psi_values[c], reverse=True)[:8]
            for col in top_drifted:
                kde_spec = PlotSpec(
                    plot_id=f"side_by_side_kde_{col}",
                    plot_type=PlotType.SIDE_BY_SIDE_KDE,
                    title=f"{col}: train vs inference",
                    column=col,
                    priority=1,
                    stage="comparison",
                )
                b64 = renderer.render(kde_spec, df=df_train, df_reference=df_inf)
                if b64:
                    await _save_comparison_plot(kde_spec, b64)

            # Categorical drift heatmaps for top drifted categoricals
            top_cat_drifted = sorted(
                [c for c in common_cat if c in psi_values],
                key=lambda c: psi_values.get(c, 0),
                reverse=True,
            )[:4]
            for col in top_cat_drifted:
                cat_spec = PlotSpec(
                    plot_id=f"cat_drift_heatmap_{col}",
                    plot_type=PlotType.CAT_DRIFT_HEATMAP,
                    title=f"{col}: category shift",
                    column=col,
                    priority=2,
                    stage="comparison",
                )
                b64 = renderer.render(cat_spec, df=df_train, df_reference=df_inf)
                if b64:
                    await _save_comparison_plot(cat_spec, b64)

            logger.info(
                "Comparison %s vs %s: rendered %d plots",
                inf_id, train_id, len(rendered),
            )

    finally:
        await local_engine.dispose()
