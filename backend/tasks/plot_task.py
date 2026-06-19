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

_PLOT_SAMPLE_ROWS = 5000


# ── Shared helpers (used by plot_task + dataset_plot_task) ─────────────────────


async def load_dataframe_for_plots(ds: Any, sample_n: int = _PLOT_SAMPLE_ROWS):
    """Load a sampled DataFrame for row-level plots.

    Retries the download/parse once before giving up, and NEVER swallows the
    failure silently: returns ``(df, error_message)`` where ``df`` is ``None``
    and ``error_message`` describes the failure. Callers pass ``error_message``
    into :func:`render_specs_to_storage` so the UI can show why the row-level
    plots are missing instead of degrading to only the 2-3 profile-based plots.
    """
    import io as _io

    import pandas as pd

    from backend.core.storage import storage

    last_err: Exception | None = None
    for attempt in range(2):
        try:
            raw = await storage.download(ds.storage_path)
            df_full = (
                pd.read_parquet(_io.BytesIO(raw))
                if ds.filename.endswith(".parquet")
                else pd.read_csv(_io.BytesIO(raw))
            )
            if df_full.empty:
                return None, f"Dataset '{ds.filename}' loaded but contains no rows."
            return (
                df_full.sample(min(sample_n, len(df_full)), random_state=42),
                None,
            )
        except Exception as exc:  # noqa: BLE001 - reported back to the caller, not swallowed
            last_err = exc
            logger.warning(
                "DataFrame load attempt %d/2 failed for %s: %s",
                attempt + 1, ds.storage_path, exc,
            )
    return None, f"Could not load '{ds.filename}' for row-level plots: {last_err}"


async def _write_render_status(
    status_path: str,
    *,
    complete: bool,
    expected: int,
    rendered: int,
    failed: list[str],
    df_error: str | None,
) -> None:
    from backend.core.storage import storage

    payload = {
        "complete": complete,
        "expected": expected,
        "rendered": rendered,
        "failed": failed,
        "df_error": df_error,
    }
    await storage.upload(status_path, json.dumps(payload).encode("utf-8"), "application/json")


async def render_specs_to_storage(
    specs: list[Any],
    *,
    base_dir: str,
    profile: Any = None,
    df: Any = None,
    df_reference: Any = None,
    df_error: str | None = None,
    emitter: Any = None,
    stage: str | None = None,
    status_suffix: str = "",
) -> tuple[int, list[str]]:
    """Render every spec and persist PNG + meta + manifest + render-status.

    The full planned manifest is written UP FRONT so the frontend knows the
    total number of plots immediately, and a render-status sidecar records
    completion + any failures so the UI can stop the loading indicator
    deterministically (no more 90s "more incoming" guessing).

    Returns ``(n_rendered, failed_plot_ids)``.
    """
    from backend.ml.plotter import PlotRenderer
    from backend.core.storage import storage

    manifest_path = f"{base_dir}/manifest.json"
    status_path = f"{base_dir}/render{status_suffix}.json"
    expected = len(specs)

    # Merge the full planned spec list into any existing manifest (other stages
    # share manifest.json) so the UI can render placeholders immediately.
    existing: list[dict[str, Any]] = []
    if await storage.exists(manifest_path):
        try:
            existing = json.loads(await storage.download(manifest_path))
        except Exception:  # noqa: BLE001 - corrupt manifest is rebuilt from specs
            existing = []
    existing_ids = {e["plot_id"] for e in existing}
    for spec in specs:
        if spec.plot_id not in existing_ids:
            existing.append(spec.to_dict())
            existing_ids.add(spec.plot_id)
    await storage.upload(manifest_path, json.dumps(existing).encode("utf-8"), "application/json")

    failed: list[str] = []
    rendered = 0
    await _write_render_status(
        status_path, complete=False, expected=expected,
        rendered=0, failed=[], df_error=df_error,
    )

    renderer = PlotRenderer()
    for i, spec in enumerate(specs):
        pct = 10 + int(i / max(expected, 1) * 85)
        if emitter:
            await emitter.emit_async("plots", f"Rendering plot {i+1}/{expected}: {spec.title}…", pct)

        b64 = ""
        try:
            b64 = renderer.render(spec, profile=profile, df=df, df_reference=df_reference)
        except Exception as exc:  # noqa: BLE001 - one bad plot never blocks the rest
            logger.warning("Plot render failed for %s (%s): %s", spec.plot_id, spec.title, exc)

        if not b64:
            failed.append(spec.plot_id)
            await _write_render_status(
                status_path, complete=False, expected=expected,
                rendered=rendered, failed=failed, df_error=df_error,
            )
            continue

        png_bytes = base64.b64decode(b64)
        await storage.upload(f"{base_dir}/{spec.plot_id}.png", png_bytes, "image/png")
        await storage.upload(
            f"{base_dir}/{spec.plot_id}.meta.json",
            json.dumps(spec.to_dict()).encode("utf-8"), "application/json",
        )
        rendered += 1
        await _write_render_status(
            status_path, complete=False, expected=expected,
            rendered=rendered, failed=failed, df_error=df_error,
        )
        if emitter:
            await emitter.emit_async(
                "plots", f"Plot ready: {spec.title}", pct,
                {"plot_id": spec.plot_id, "stage": stage},
            )

    await _write_render_status(
        status_path, complete=True, expected=expected,
        rendered=rendered, failed=failed, df_error=df_error,
    )
    if emitter:
        await emitter.emit_async(
            "plots", f"Rendered {rendered}/{expected} plots",
            100, {"stage": stage, "n_rendered": rendered},
        )
    logger.info(
        "%s: rendered %d/%d plots (%d failed)%s",
        base_dir, rendered, expected, len(failed),
        f"; df_error={df_error}" if df_error else "",
    )
    return rendered, failed


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
    from backend.ml.profiler import DatasetProfile
    from backend.agents.plot_selector_agent import select_plots_for_stage

    local_engine = _cae(settings.DATABASE_URL, pool_pre_ping=True)
    local_factory = _asf(local_engine, class_=_AS, expire_on_commit=False)
    emitter = ProgressEmitter(run_id)
    base_dir = f"runs/{run_id}/plots"

    try:
        async with local_factory() as session:
            run = (
                await session.execute(select(Run).where(Run.id == run_id))
            ).scalar_one_or_none()
            if run is None:
                return

            ds = None
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

            # Load a sampled DataFrame for row-level plots. The error (if any) is
            # carried through to the manifest so the UI can explain why row-level
            # plots are missing instead of silently showing only profile plots.
            df: pd.DataFrame | None = None
            df_error: str | None = None
            if ds:
                df, df_error = await load_dataframe_for_plots(ds)
            else:
                df_error = "No training dataset is attached to this run."

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
                    if df_cleaned is None and not df_error:
                        df_error = "Preprocessing transform for visualization failed."

                from backend.agents.plot_selector_agent import generate_preprocessing_after_specs
                manifest = generate_preprocessing_after_specs(
                    run.preprocessing_strategy, compressed, run_id,
                )
                specs = [
                    s for s in manifest.plots
                    if not column_filter or s.column == column_filter
                ]
                await render_specs_to_storage(
                    specs, base_dir=base_dir, profile=profile, df=df_cleaned,
                    df_error=df_error, emitter=emitter, stage=stage,
                    status_suffix=f"_{stage}",
                )
                return

            manifest = await select_plots_for_stage(compressed, stage, task_type, run_id)
            specs = [
                s for s in manifest.plots
                if not column_filter or s.column == column_filter
            ]
            await render_specs_to_storage(
                specs, base_dir=base_dir, profile=profile, df=df,
                df_error=df_error, emitter=emitter, stage=stage,
                status_suffix=f"_{stage}",
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
