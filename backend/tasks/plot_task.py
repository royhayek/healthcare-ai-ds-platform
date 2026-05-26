"""Celery task: render plots for a pipeline stage (§12, Category B1).

Triggered by:
  - The analysis task at each checkpoint (automatic)
  - POST /runs/{run_id}/plots (on-demand from chat or UI)

For each PlotSpec in the manifest, PlotRenderer.render() is called and the
resulting PNG is stored under runs/{run_id}/plots/{plot_id}.png. A manifest
index at runs/{run_id}/plots/manifest.json tracks all rendered specs.
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


# ── Public async entry point (used by analysis_task + FastAPI BackgroundTasks) ─


async def run_plots(run_id: str, stage: str = "eda", column_filter: str | None = None) -> None:
    """Async entry point - no Celery/Redis needed."""
    await _async_render(run_id, stage, column_filter)


@celery_app.task(bind=True, name="plots.render", max_retries=0)
def render_plots_task(
    self,  # type: ignore[misc]
    run_id: str,
    stage: str = "eda",
    column_filter: str | None = None,
) -> None:
    """Celery entry point for plot rendering."""
    try:
        asyncio.run(_async_render(run_id, stage, column_filter))
    except Exception as exc:
        logger.exception("Plot render task failed for run %s stage %s", run_id, stage)
        raise


async def _async_render(run_id: str, stage: str, column_filter: str | None) -> None:
    from sqlalchemy.ext.asyncio import AsyncSession as _AS
    from sqlalchemy.ext.asyncio import async_sessionmaker as _asf
    from sqlalchemy.ext.asyncio import create_async_engine as _cae

    import pandas as pd
    from sqlalchemy import select

    from backend.core.database import Dataset, Run
    from backend.core.events import ProgressEmitter
    from backend.core.storage import storage
    from backend.ml.plotter import PlotRenderer, PlotSpec, PlotType
    from backend.ml.profiler import DatasetProfile
    from backend.agents.plot_selector_agent import select_plots_for_stage

    local_engine = _cae(settings.DATABASE_URL, pool_pre_ping=True)
    local_factory = _asf(local_engine, class_=_AS, expire_on_commit=False)
    emitter = ProgressEmitter(run_id)

    try:
        async with local_factory() as session:
            run = (
                await session.execute(select(Run).where(Run.id == run_id))
            ).scalar_one_or_none()
            if run is None:
                return

            ds = None
            df: pd.DataFrame | None = None
            profile: DatasetProfile | None = None

            if run.training_dataset_id:
                ds = (
                    await session.execute(select(Dataset).where(Dataset.id == run.training_dataset_id))
                ).scalar_one_or_none()

            if ds and ds.profile:
                try:
                    profile = DatasetProfile.model_validate(ds.profile)
                except Exception:
                    pass

            # Load a sampled DataFrame for row-level plots (max 5 000 rows)
            if ds:
                try:
                    raw = await storage.download(ds.storage_path)
                    import io as _io
                    df_full = (
                        pd.read_parquet(_io.BytesIO(raw))
                        if ds.filename.endswith(".parquet")
                        else pd.read_csv(_io.BytesIO(raw))
                    )
                    df = df_full.sample(min(5000, len(df_full)), random_state=42)
                except Exception as exc:
                    logger.warning("Could not load dataset for plot rendering: %s", exc)

            # Build compressed profile dict
            compressed: dict[str, Any] = {}
            if profile:
                from backend.ml.profiler import compress_profile_for_claude
                compressed = compress_profile_for_claude(profile)
            elif run.eda_report:
                compressed = run.eda_report

            task_type = (
                (ds.task_type if ds else None)
                or compressed.get("task_type", "binary_classification")
            )

            await emitter.emit_async("plots", f"Selecting plots for {stage} stage…", 5)

            # preprocessing_after: apply the strategy to produce a cleaned
            # DataFrame and generate specs from the strategy rather than profile.
            if stage == "preprocessing_after":
                if not run.preprocessing_strategy:
                    logger.warning("No preprocessing_strategy for run %s - skipping preprocessing_after plots", run_id)
                    return

                df_cleaned: pd.DataFrame | None = None
                if df is not None:
                    df_cleaned = _apply_preprocessing_for_viz(df, run.preprocessing_strategy)

                from backend.agents.plot_selector_agent import generate_preprocessing_after_specs
                manifest = generate_preprocessing_after_specs(
                    run.preprocessing_strategy,
                    compressed,
                    run_id,
                )

                specs_to_render = manifest.plots
                if column_filter:
                    specs_to_render = [s for s in specs_to_render if s.column == column_filter]

                renderer = PlotRenderer()
                rendered: list[dict[str, Any]] = []
                n = len(specs_to_render)

                manifest_path = f"runs/{run_id}/plots/manifest.json"
                existing_manifest: list[dict[str, Any]] = []
                if await storage.exists(manifest_path):
                    try:
                        existing_manifest = json.loads(await storage.download(manifest_path))
                    except Exception:
                        existing_manifest = []
                existing_ids = {e["plot_id"] for e in existing_manifest}

                for i, spec in enumerate(specs_to_render):
                    pct = 10 + int(i / max(n, 1) * 85)
                    await emitter.emit_async("plots", f"Rendering plot {i+1}/{n}: {spec.title}…", pct)
                    try:
                        b64 = renderer.render(spec, profile=profile, df=df_cleaned)
                    except Exception as exc:
                        logger.warning("Plot render failed for %s (%s) - skipping: %s", spec.plot_id, spec.title, exc)
                        continue
                    if not b64:
                        logger.debug("Empty render for %s - skipping", spec.plot_id)
                        continue
                    png_bytes = base64.b64decode(b64)
                    png_path = f"runs/{run_id}/plots/{spec.plot_id}.png"
                    meta_path = f"runs/{run_id}/plots/{spec.plot_id}.meta.json"
                    await storage.upload(png_path, png_bytes, "image/png")
                    await storage.upload(meta_path, json.dumps(spec.to_dict()).encode("utf-8"), "application/json")
                    entry = spec.to_dict()
                    if spec.plot_id not in existing_ids:
                        existing_manifest.append(entry)
                        existing_ids.add(spec.plot_id)
                    rendered.append(entry)
                    await storage.upload(manifest_path, json.dumps(existing_manifest).encode("utf-8"), "application/json")
                    await emitter.emit_async("plots", f"Plot ready: {spec.title}", pct, {"plot_id": spec.plot_id, "stage": stage})

                await emitter.emit_async("plots", f"Rendered {len(rendered)}/{n} plots for {stage} stage", 100, {"stage": stage, "n_rendered": len(rendered)})
                return

            manifest = await select_plots_for_stage(compressed, stage, task_type, run_id)

            # Filter by column if requested
            specs_to_render = manifest.plots
            if column_filter:
                specs_to_render = [s for s in specs_to_render if s.column == column_filter]

            renderer = PlotRenderer()
            rendered: list[dict[str, Any]] = []
            n = len(specs_to_render)

            # Load existing manifest so we can merge without overwriting other stages
            manifest_path = f"runs/{run_id}/plots/manifest.json"
            existing_manifest: list[dict[str, Any]] = []
            if await storage.exists(manifest_path):
                try:
                    existing_manifest = json.loads(await storage.download(manifest_path))
                except Exception:
                    existing_manifest = []

            existing_ids = {e["plot_id"] for e in existing_manifest}

            for i, spec in enumerate(specs_to_render):
                pct = 10 + int(i / max(n, 1) * 85)
                await emitter.emit_async("plots", f"Rendering plot {i+1}/{n}: {spec.title}…", pct)

                try:
                    b64 = renderer.render(spec, profile=profile, df=df)
                except Exception as exc:
                    logger.warning("Plot render failed for %s (%s) - skipping: %s", spec.plot_id, spec.title, exc)
                    continue
                if not b64:
                    logger.debug("Empty render for %s - skipping", spec.plot_id)
                    continue

                png_bytes = base64.b64decode(b64)
                png_path = f"runs/{run_id}/plots/{spec.plot_id}.png"
                meta_path = f"runs/{run_id}/plots/{spec.plot_id}.meta.json"

                await storage.upload(png_path, png_bytes, "image/png")
                await storage.upload(
                    meta_path,
                    json.dumps(spec.to_dict()).encode("utf-8"),
                    "application/json",
                )

                entry = spec.to_dict()
                if spec.plot_id not in existing_ids:
                    existing_manifest.append(entry)
                    existing_ids.add(spec.plot_id)
                rendered.append(entry)

                # Write manifest after every individual plot so the frontend
                # can display each one as soon as it is ready.
                await storage.upload(
                    manifest_path,
                    json.dumps(existing_manifest).encode("utf-8"),
                    "application/json",
                )
                await emitter.emit_async(
                    "plots",
                    f"Plot ready: {spec.title}",
                    pct,
                    {"plot_id": spec.plot_id, "stage": stage},
                )

            await emitter.emit_async(
                "plots",
                f"Rendered {len(rendered)}/{n} plots for {stage} stage",
                100,
                {"stage": stage, "n_rendered": len(rendered)},
            )

    finally:
        await local_engine.dispose()


# ── Helpers ────────────────────────────────────────────────────────────────────


def _apply_preprocessing_for_viz(
    df: "pd.DataFrame",
    prep_strategy_dict: dict,
) -> "pd.DataFrame | None":
    """Apply imputation + scaling to numeric columns and imputation to categorical
    columns (no encoding) so the cleaned DataFrame keeps original column names.

    Used exclusively for visualization - never used for model training.
    Returns None on any failure so a bad strategy never blocks plot rendering.
    """
    try:
        import pandas as _pd
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

        from backend.ml.cleaner import prepare_data
        from backend.models.strategy import PreprocessingStrategy

        prep_strategy = PreprocessingStrategy.model_validate(prep_strategy_dict)
        X, y = prepare_data(df, prep_strategy)
        df_cleaned = X.copy()

        scaler_map: dict[str, Any] = {
            "standard": StandardScaler(),
            "minmax": MinMaxScaler(),
            "robust": RobustScaler(),
        }

        for col, col_strat in prep_strategy.columns.items():
            if col not in df_cleaned.columns or col_strat.action == "drop":
                continue

            if col_strat.dtype_hint == "numeric":
                imp_strat = col_strat.impute_strategy or "median"
                if imp_strat not in ("none", "constant"):
                    imputer = SimpleImputer(strategy=imp_strat)
                    df_cleaned[[col]] = imputer.fit_transform(df_cleaned[[col]])
                scale_strat = col_strat.scale_strategy or "standard"
                scaler = scaler_map.get(scale_strat)
                if scaler is not None:
                    df_cleaned[[col]] = scaler.fit_transform(df_cleaned[[col]])

            elif col_strat.dtype_hint == "categorical":
                imp_strat = col_strat.impute_strategy or "most_frequent"
                if imp_strat not in ("none",) and df_cleaned[col].isnull().any():
                    imputer = SimpleImputer(strategy=imp_strat)
                    df_cleaned[[col]] = imputer.fit_transform(df_cleaned[[col]])

        # Add target back so class_dist plots work.
        target_col = prep_strategy.target_column
        if target_col and target_col not in df_cleaned.columns:
            df_cleaned[target_col] = y.values

        return df_cleaned
    except Exception as exc:
        logger.warning("preprocessing_after viz transform failed: %s", exc)
        return None
