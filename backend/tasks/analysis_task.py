"""Full Celery analysis pipeline task (§22).

Pipeline with 5 checkpoint pauses:
  1. load → profile → EDA       → [CHECKPOINT 1: checkpoint_1_eda]
  2. preprocessing strategy     → [CHECKPOINT 2: checkpoint_2_preprocessing]
  3. model selection            → [CHECKPOINT 3: checkpoint_3_model_selection]
  4. train → stability → tests  → [CHECKPOINT 4: checkpoint_4_training]
  5. tune → calibrate → threshold → SHAP → similarity → insight
                                  → [CHECKPOINT 5: checkpoint_5_final]

Crash-safety protocol:
  At each checkpoint, the full strategy snapshot is persisted to the Run record
  BEFORE setting status=awaiting_checkpoint. On resume, the pipeline reads
  run.current_step and routes to the correct step function. A server restart
  mid-checkpoint always resumes rather than restarts from the beginning.

Resume flow:
  POST /runs/{run_id}/resume  → set status=running → re-enqueue this task →
  task reads current_step → dispatches to _step_N() function.

Hard rule from the spec §16:
  Calibration FIRST on X_cal (never X_test).
  Threshold optimization on calibrated probabilities from X_val (not X_test).
  X_test stays sealed until final_metrics.
"""

import asyncio
import io
import logging
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import select

from backend.core import audit
from backend.core.config import settings
from backend.core.database import ChatMessage as DBChatMessage, Dataset, Project, Run, async_session_factory
from backend.core.events import ProgressEmitter, emit_progress
from backend.core.storage import storage
from backend.core.strategy_mutator import flush_pending_intents
from backend.ml.profiler import compress_profile_for_claude, profile_dataset
from backend.tasks.celery_app import celery_app
from backend.tasks.plot_task import run_plots

logger = logging.getLogger(__name__)


# ── Checkpoint routing ─────────────────────────────────────────────────────────
# Maps current_step value → which async step function to call on resume.

_STEP_ROUTING: dict[str, str] = {
    "init": "_step_eda",
    "profiling": "_step_eda",
    "eda": "_step_eda",
    "checkpoint_1_eda": "_step_preprocessing",
    "preprocessing": "_step_preprocessing",
    "checkpoint_2_preprocessing": "_step_model_selection",
    "model_selection": "_step_model_selection",
    "checkpoint_3_model_selection": "_step_training",
    "training": "_step_training",
    "checkpoint_4_training": "_step_tuning",
    "tuning": "_step_tuning",
    "calibration": "_step_tuning",
    "threshold": "_step_tuning",
    "shap": "_step_tuning",
    "similarity": "_step_tuning",
    "drift": "_step_tuning",
    "fairness": "_step_tuning",
    "holdout": "_step_tuning",
    "insight": "_step_tuning",
    "checkpoint_5_final": "_step_deliverables_placeholder",
}


# Maps the checkpoint a run is paused at → the current_step value that re-runs
# the step which PRODUCED that checkpoint (rather than advancing). Used by the
# "Re-run step" action so a user can regenerate a checkpoint whose model output
# fell back to defaults (e.g. a JSON parse failure or a refusal). Re-running the
# current step is always safe: the pipeline pauses at each checkpoint, so no
# downstream state exists yet to invalidate.
_RERUN_STEP: dict[str, str] = {
    "checkpoint_1_eda": "eda",
    "checkpoint_2_preprocessing": "preprocessing",
    "checkpoint_3_model_selection": "model_selection",
    "checkpoint_4_training": "training",
    "checkpoint_5_final": "tuning",
}


def rerun_step_for(current_step: str | None) -> str | None:
    """Return the current_step value to re-run the producing step of a checkpoint.

    For a checkpoint key, returns the producing step's key. For a non-checkpoint
    value (e.g. a failed run paused mid-step), returns it unchanged - the
    existing dispatcher already re-runs that step. Returns None only for an
    unknown/empty step the caller should reject.
    """
    if not current_step:
        return None
    if current_step in _RERUN_STEP:
        return _RERUN_STEP[current_step]
    if current_step in _STEP_ROUTING:
        return current_step
    return None


# ── Override-driven step re-run ────────────────────────────────────────────────
# When a chat override mutates a pipeline decision, editing the field is not
# enough: the downstream step that CONSUMES the decision must re-run for the
# change to take effect. The target step consumes the decision (it does not
# regenerate it via an LLM agent), so re-running it realises the override
# without clobbering it - e.g. _step_training reads model_selection.primary and
# recomputes best_model_name, but never rewrites model_selection.

_OVERRIDE_RECOMPUTE_STEP: dict[str, str] = {
    # Re-run the PRODUCING agent so the human directive is sent to the AI and the
    # strategy is regenerated honouring it, then re-pause at checkpoint 2. The
    # full retrain Rule 8 requires still happens: resuming past checkpoint 2 flows
    # through model_selection → training on the regenerated strategy.
    "preprocessing": "preprocessing",
    # Target hygiene changes the task type, class distribution and every downstream
    # decision, so it re-runs from EDA and re-pauses at checkpoint 1.
    "target": "eda",
    "model_selection": "training",    # primary/candidate change → re-evaluate + re-pick
    "threshold": "tuning",            # threshold/cost matrix feeds calibration/eval in step 5
    "fairness": "tuning",             # protected-attribute change → re-run the fairness audit
}

# Ordinal position of each checkpoint pause and each recompute-target step in the
# pipeline. Used to decide whether an override invalidates ALREADY-computed state
# (recompute needed now) or only future state (the normal resume will pick it up).
_STAGE_ORDER: dict[str, int] = {
    "checkpoint_1_eda": 1,
    "checkpoint_2_preprocessing": 2,
    "checkpoint_3_model_selection": 3,
    "checkpoint_4_training": 4,
    "checkpoint_5_final": 5,
    "eda": 1,
    "preprocessing": 2,
    "model_selection": 3,
    "training": 4,
    "tuning": 5,
}


def resolve_training_primary(
    model_selection: dict[str, Any],
    result_names: list[str],
    fallback: str,
) -> str:
    """Resolve the winning model at the training checkpoint.

    A user override (model_selection["primary_source"] == "user_override") is
    AUTHORITATIVE: it is honoured exactly and never silently replaced by the
    leaderboard winner, even when scores tie. This is the whole point of the
    chat override - "use logistic_regression instead of lightgbm" forces that
    model, it does not request a re-comparison that might switch back.

    Without an override, the agent's recommended primary is used when it was
    actually trained, otherwise the top stability result (fallback) wins.
    """
    primary = model_selection.get("primary")
    user_forced = model_selection.get("primary_source") == "user_override"

    if user_forced and primary:
        if primary not in result_names:
            logger.error(
                "User-forced primary %r was not among trained candidates %s - "
                "honouring the override regardless; its leaderboard score will be unavailable.",
                primary, result_names,
            )
        else:
            logger.info("Honouring user override: primary model forced to %r", primary)
        return primary

    if primary and primary in result_names:
        return primary
    return fallback


def rerun_step_for_override(category: str, current_step: str | None) -> str | None:
    """Return the step to re-run so a chat override takes effect, or None.

    Returns the recompute-target step only when the override invalidates state
    that has ALREADY been computed at or before the current checkpoint. Returns
    None when the overridden decision's downstream step has not run yet (the
    normal resume will consume the change) or when the category drives no
    recompute (e.g. fairness/drift, applied in-place).
    """
    recompute = _OVERRIDE_RECOMPUTE_STEP.get(category)
    if recompute is None or not current_step:
        return None
    current_stage = _STAGE_ORDER.get(current_step)
    recompute_stage = _STAGE_ORDER.get(recompute)
    if current_stage is None or recompute_stage is None:
        return None
    return recompute if recompute_stage <= current_stage else None


# ── Celery entry point ─────────────────────────────────────────────────────────


@celery_app.task(bind=True, name="analysis.run", max_retries=0)
def run_analysis_task(self, run_id: str) -> None:  # type: ignore[misc]
    """Bridges sync Celery → async pipeline. Routes to correct step on resume."""
    emit_progress(run_id, "init", "Pipeline starting…", 2)
    try:
        asyncio.run(_async_pipeline(run_id))
    except Exception as exc:
        logger.exception("Analysis task failed for run %s", run_id)
        asyncio.run(_mark_failed(run_id, str(exc)))
        raise


# ── Async pipeline dispatcher ──────────────────────────────────────────────────


async def _async_pipeline(run_id: str) -> None:
    # Create a fresh engine inside this coroutine so it is always bound to the
    # current event loop. The module-level engine is created at import time and
    # its asyncpg connections reference whichever loop existed then - reusing it
    # across asyncio.run() calls (each of which creates and destroys a loop)
    # causes "Future attached to a different loop" errors.
    from sqlalchemy.ext.asyncio import AsyncSession as _AS
    from sqlalchemy.ext.asyncio import async_sessionmaker as _asf
    from sqlalchemy.ext.asyncio import create_async_engine as _cae
    _local_engine = _cae(settings.DATABASE_URL, pool_pre_ping=True)
    _local_factory = _asf(_local_engine, class_=_AS, expire_on_commit=False)

    emitter = ProgressEmitter(run_id)
    try:
        async with _local_factory() as session:
            result = await session.execute(select(Run).where(Run.id == run_id))
            run = result.scalar_one_or_none()
            if run is None:
                raise ValueError(f"Run {run_id} not found")

            # Route to correct step based on current_step (crash-safe resume)
            current = run.current_step or "init"
            step_name = _STEP_ROUTING.get(current, "_step_eda")
            logger.info("Dispatching run %s from step=%s → %s", run_id, current, step_name)

            run.status = "running"
            session.add(run)
            await session.commit()

            step_fn = globals()[step_name]
            await step_fn(session, run, emitter)
    finally:
        await _local_engine.dispose()


# ── Step 1: Load → Profile → EDA ──────────────────────────────────────────────


async def _step_eda(session: Any, run: Run, emitter: ProgressEmitter) -> None:
    run_id = run.id

    # Load dataset
    if not run.training_dataset_id:
        raise ValueError(f"Run {run_id} has no training_dataset_id")

    ds_result = await session.execute(
        select(Dataset).where(Dataset.id == run.training_dataset_id)
    )
    dataset = ds_result.scalar_one_or_none()
    if dataset is None:
        raise ValueError(f"Dataset {run.training_dataset_id} not found")

    run.current_step = "profiling"
    session.add(run)
    await session.commit()

    await emitter.emit_async("load", f"Loading {dataset.filename}…", 5)
    raw_bytes = await storage.download(dataset.storage_path)
    df = _parse_dataframe(raw_bytes, dataset.filename)
    await emitter.emit_async("load", "Dataset loaded", 8)

    # ── Target hygiene (§7, §10) ───────────────────────────────────────────────
    # Drop unlabelled rows + collapse the target to binary BEFORE profiling, so the
    # task type and class distribution reflect the real modelling target rather than
    # treating an "unknown" placeholder as a class. Derived from the case brief +
    # deterministic unlabelled detection, merged with any chat override recorded on
    # the run (the human's override wins). Persisted so training applies the same.
    if dataset.target_column:
        target_strategy = await _resolve_target_strategy(session, run, dataset, df)
        if not target_strategy.is_empty():
            from backend.ml.cleaner import apply_target_hygiene

            n_before = len(df)
            df = apply_target_hygiene(df, dataset.target_column, target_strategy)
            await emitter.emit_async(
                "target_hygiene",
                f"Applied target hygiene: {n_before - len(df)} unlabelled rows dropped"
                + ("; target collapsed to binary" if target_strategy.positive_labels else ""),
                9,
            )
            await audit.append(
                session, run_id=run_id, actor="system", category="target_hygiene",
                action="target_hygiene_applied",
                payload={
                    "drop_labels": target_strategy.drop_labels,
                    "positive_labels": target_strategy.positive_labels,
                    "rows_before": n_before,
                    "rows_after": len(df),
                    "source": target_strategy.note,
                },
                reason=(
                    f"Dropped {n_before - len(df)} unlabelled rows"
                    + (f"; collapsed target to binary (positive={target_strategy.positive_labels})"
                       if target_strategy.positive_labels else "")
                ),
            )
        run.target_strategy = target_strategy.model_dump(mode="json")
        session.add(run)
        await session.commit()

    # Profile
    await emitter.emit_async("profile", "Profiling dataset…", 10)
    full_profile = profile_dataset(df, target_column=dataset.target_column)
    compressed = compress_profile_for_claude(full_profile)
    await emitter.emit_async("profile", "Profile complete", 14)

    await audit.append(
        session, run_id=run_id, actor="system", category="profiler",
        action="profile_complete",
        payload={
            "n_rows": full_profile.n_rows,
            "n_cols": full_profile.n_cols,
            "duplicate_count": full_profile.duplicate_count,
            "task_type": full_profile.task_type,
        },
    )
    dataset.profile = full_profile.model_dump(mode="json")
    dataset.row_count = full_profile.n_rows
    dataset.col_count = full_profile.n_cols
    dataset.task_type = full_profile.task_type
    session.add(dataset)
    await session.commit()

    # EDA agent
    from backend.agents.eda_agent import run_eda_agent

    run.current_step = "eda"
    run.progress = 15
    session.add(run)
    await session.commit()

    eda_report = await run_eda_agent(session, run_id, compressed, emitter)

    # The target column and task type are deterministic facts from the dataset
    # and profiler - not LLM opinions. The agent's free-form target_analysis
    # often omits these structured fields, leaving the EDA checkpoint showing
    # "—", so always populate them here.
    eda_report.target_analysis["column"] = dataset.target_column or "(none specified)"
    eda_report.target_analysis["task_type"] = full_profile.task_type

    # Deterministic leakage + target-hygiene scan (domain-agnostic, §7/§9). Runs
    # regardless of what the LLM agent surfaced, so proxy leakage (a categorical
    # near-perfect predictor of the target) and "unlabeled" placeholder classes are
    # always caught - these are the silent causes of misleadingly perfect accuracy.
    if dataset.target_column and dataset.target_column in df.columns:
        from backend.ml.leakage_detector import (
            detect_leakage,
            detect_unlabeled_target_classes,
        )
        from backend.models.eda import QualityIssue

        await emitter.emit_async("leakage", "Scanning for label leakage…", 36)
        leak_report = detect_leakage(df, dataset.target_column)
        unlabeled = detect_unlabeled_target_classes(df[dataset.target_column])

        if leak_report.candidates:
            detected = [
                {"column": c.column, "reason": c.reason, "severity": c.severity}
                for c in leak_report.candidates
            ]
            existing = eda_report.correlations.get("leakage_risk") or []
            seen = {str(r.get("column")) for r in existing if isinstance(r, dict)}
            merged = list(existing) + [d for d in detected if d["column"] not in seen]
            eda_report.correlations["leakage_risk"] = merged
            await audit.append(
                session, run_id=run_id, actor="system", category="leakage",
                action="leakage_detected",
                payload={"n_flagged": leak_report.n_flagged, "columns": detected},
                reason=leak_report.recommendation,
            )

        if unlabeled.suspicious_classes:
            eda_report.quality_issues.insert(0, QualityIssue(
                column=dataset.target_column,
                issue=(
                    f"Target contains {unlabeled.affected_rows} rows "
                    f"({100*unlabeled.affected_rows/max(unlabeled.total_rows,1):.0f}%) "
                    f"labelled {unlabeled.suspicious_classes} - these look like "
                    "'unlabeled' placeholders, not a real outcome level."
                ),
                severity="high",
                recommendation=unlabeled.recommendation,
            ))
            await audit.append(
                session, run_id=run_id, actor="system", category="target_hygiene",
                action="unlabeled_target_classes_detected",
                payload={
                    "suspicious_classes": unlabeled.suspicious_classes,
                    "affected_rows": unlabeled.affected_rows,
                    "total_rows": unlabeled.total_rows,
                },
                reason=unlabeled.recommendation,
            )

    # Checkpoint 1: persist everything, await user review
    run.status = "awaiting_checkpoint"
    run.current_step = "checkpoint_1_eda"
    run.progress = 40
    run.eda_report = eda_report.model_dump(mode="json")
    session.add(run)
    await session.commit()

    await emitter.emit_async(
        "checkpoint",
        "EDA complete - review findings before preprocessing",
        40,
        {"checkpoint": 1, "step": "eda", "summary": eda_report.summary},
    )

    # Inject a proactive co-pilot message so the panel isn't empty at checkpoint
    high_issues = [q for q in eda_report.quality_issues if q.severity == "high"]
    leakage = (eda_report.correlations or {}).get("leakage_risk", [])
    issue_lines = "\n".join(
        f"- **{q.column or 'dataset'}**: {q.issue} - {q.recommendation}"
        for q in high_issues[:5]
    )
    leakage_lines = "\n".join(
        f"- **{r['column']}**: {r['reason']}" for r in leakage[:3]
    ) if leakage else ""
    proactive = (
        f"EDA complete. {eda_report.summary}\n\n"
        + (f"**High-severity issues ({len(high_issues)}):**\n{issue_lines}\n\n" if high_issues else "")
        + (f"**Leakage risks:**\n{leakage_lines}\n\n" if leakage_lines else "")
        + f"Initial model recommendation: `{eda_report.model_recommendation}`. "
        + "Use the preprocessing checkpoint to override any column strategy."
    )
    session.add(
        DBChatMessage(
            run_id=run_id,
            user_id=run.created_by or "system",
            role="assistant",
            content=proactive,
            model=settings.CLAUDE_SONNET_MODEL,
        )
    )
    await session.commit()

    # Trigger async plot rendering for EDA stage (non-blocking)
    await run_plots(run_id, "eda")


# ── Step 2: Preprocessing Strategy ────────────────────────────────────────────


async def _step_preprocessing(session: Any, run: Run, emitter: ProgressEmitter) -> None:
    run_id = run.id

    if not run.eda_report:
        raise ValueError(f"Run {run_id}: EDA report missing - cannot continue")

    # Re-load dataset and profile
    ds_result = await session.execute(
        select(Dataset).where(Dataset.id == run.training_dataset_id)
    )
    dataset = ds_result.scalar_one_or_none()
    if dataset is None:
        raise ValueError(f"Dataset {run.training_dataset_id} not found")

    if not dataset.target_column:
        raise ValueError(
            "No target column set on this dataset. "
            "Re-upload with a target_column specified, or set it via the dataset settings."
        )

    run.current_step = "preprocessing"
    run.progress = 42
    session.add(run)
    await session.commit()

    from backend.agents.preprocessing_agent import run_preprocessing_agent

    compressed = _compress_stored_profile(dataset.profile or {})
    # Replay any human overrides recorded in chat for this step so the agent
    # regenerates the strategy honouring them (§2, §21). On a re-run after an
    # override, these carry the clinician's verbatim instruction(s).
    directives = list((run.user_directives or {}).get("preprocessing", []))
    strategy = await run_preprocessing_agent(
        session, run_id,
        compressed_profile=compressed,
        eda_report=run.eda_report,
        target_column=dataset.target_column,
        task_type=dataset.task_type or run.eda_report.get("target_analysis", {}).get("task_type", "binary_classification"),
        emitter=emitter,
        user_directives=directives,
    )

    run.status = "awaiting_checkpoint"
    run.current_step = "checkpoint_2_preprocessing"
    run.progress = 48
    run.preprocessing_strategy = strategy.model_dump(mode="json")
    session.add(run)
    await session.commit()

    await emitter.emit_async(
        "checkpoint",
        "Preprocessing strategy ready - review column decisions",
        48,
        {
            "checkpoint": 2,
            "step": "preprocessing",
            "n_columns": len(strategy.columns),
            "dropped": [c for c, s in strategy.columns.items() if s.action == "drop"],
        },
    )

    # Plots are decorative and run after the strategy + checkpoint are committed.
    # A rendering failure must never fail the analysis run at this point.
    for _plot_stage in ("preprocessing", "preprocessing_after"):
        try:
            await run_plots(run_id, _plot_stage)
        except Exception as exc:
            logger.warning("Plot stage %s failed for run %s - continuing: %s", _plot_stage, run_id, exc)


# ── Step 3: Model Selection ────────────────────────────────────────────────────


async def _step_model_selection(session: Any, run: Run, emitter: ProgressEmitter) -> None:
    run_id = run.id

    if not run.preprocessing_strategy:
        raise ValueError(f"Run {run_id}: preprocessing strategy missing")

    ds_result = await session.execute(
        select(Dataset).where(Dataset.id == run.training_dataset_id)
    )
    dataset = ds_result.scalar_one_or_none()

    run.current_step = "model_selection"
    run.progress = 50
    session.add(run)
    await session.commit()

    from backend.agents.model_selector_agent import run_model_selector_agent

    compressed = _compress_stored_profile(dataset.profile or {})
    task_type = run.preprocessing_strategy.get("task_type", "binary_classification")
    n_rows = dataset.row_count or 1000

    selection = await run_model_selector_agent(
        session, run_id,
        compressed_profile=compressed,
        eda_report=run.eda_report or {},
        task_type=task_type,
        n_rows=n_rows,
        emitter=emitter,
    )

    run.status = "awaiting_checkpoint"
    run.current_step = "checkpoint_3_model_selection"
    run.progress = 52
    run.model_selection = selection.model_dump(mode="json")
    session.add(run)
    await session.commit()

    await emitter.emit_async(
        "checkpoint",
        f"Model candidates selected - primary: {selection.primary}",
        52,
        {
            "checkpoint": 3,
            "step": "model_selection",
            "primary": selection.primary,
            "candidates": selection.candidates,
        },
    )


# ── Step 4: Training + Stability + Stat Tests ──────────────────────────────────


async def _step_training(session: Any, run: Run, emitter: ProgressEmitter) -> None:
    run_id = run.id

    if not run.model_selection or not run.preprocessing_strategy:
        raise ValueError(f"Run {run_id}: model selection or preprocessing strategy missing")

    ds_result = await session.execute(
        select(Dataset).where(Dataset.id == run.training_dataset_id)
    )
    dataset = ds_result.scalar_one_or_none()
    if dataset is None:
        raise ValueError(f"Dataset {run.training_dataset_id} not found")

    run.current_step = "training"
    run.progress = 54
    session.add(run)
    await session.commit()

    await emitter.emit_async("training", "Loading and preparing data…", 55)

    # Load data
    raw_bytes = await storage.download(dataset.storage_path)
    df = _parse_dataframe(raw_bytes, dataset.filename)

    from backend.ml.cleaner import (
        build_preprocessor,
        prepare_data,
        resolve_task_type,
        split_train_test,
    )
    from backend.ml.trainer import (
        fit_final_pipeline,
        maybe_run_stat_test,
        train_all_candidates,
    )
    from backend.models.strategy import PreprocessingStrategy

    prep_strategy = PreprocessingStrategy.model_validate(run.preprocessing_strategy)
    if not prep_strategy.target_column:
        # Auto-heal: strategy was persisted before target_column was set on the dataset.
        # Pull the current value from the dataset and patch both the in-memory object
        # and the persisted run record so the fix survives future resumes.
        if not dataset or not dataset.target_column:
            raise ValueError(
                "Preprocessing strategy has no target column and neither does the dataset. "
                "Set a target column on the dataset, then retry."
            )
        prep_strategy = prep_strategy.model_copy(update={"target_column": dataset.target_column})
        run.preprocessing_strategy = prep_strategy.model_dump(mode="json")
        session.add(run)
        await session.commit()
        logger.info(
            "Run %s: patched missing target_column in preprocessing_strategy → %r",
            run.id, dataset.target_column,
        )
    # Auto-heal an unresolved task_type. The profiler returns "unknown" for a
    # string-dtype target it cannot classify; left as-is the trainer treats it
    # as regression and crashes on string labels. Resolve from the actual target
    # and persist so downstream steps (tuning, calibration, threshold) agree.
    if prep_strategy.task_type not in ("binary_classification", "multiclass", "regression"):
        if prep_strategy.target_column in df.columns:
            resolved_tt = resolve_task_type(prep_strategy.task_type, df[prep_strategy.target_column])
            logger.info(
                "Run %s: resolved task_type %r → %r",
                run.id, prep_strategy.task_type, resolved_tt,
            )
            prep_strategy = prep_strategy.model_copy(update={"task_type": resolved_tt})
            run.preprocessing_strategy = prep_strategy.model_dump(mode="json")
            session.add(run)
            await session.commit()

    model_selection = run.model_selection
    logger.info(
        "Run %s training: model_selection.primary=%r source=%r candidates=%s",
        run.id,
        model_selection.get("primary"),
        model_selection.get("primary_source"),
        model_selection.get("candidates"),
    )

    # Apply the same target hygiene used at profiling time so train/eval match the
    # EDA-time target (unlabelled rows dropped, binary collapse). §7, §10.
    X, y = prepare_data(df, prep_strategy, target_strategy=run.target_strategy)
    X_train, X_test, y_train, y_test = split_train_test(
        X, y, prep_strategy.task_type, test_size=0.2, random_state=42
    )

    preprocessor = build_preprocessor(prep_strategy, X_train)
    candidates = model_selection.get("candidates", [model_selection.get("primary", "xgboost")])

    await emitter.emit_async("training", f"Training {len(candidates)} candidates (3 seeds × 5 folds)…", 56)

    # Run stability training (3 seeds × 5 folds per candidate)
    # Run in threadpool to avoid blocking the event loop
    import asyncio
    stability_results = await asyncio.get_running_loop().run_in_executor(
        None,
        lambda: train_all_candidates(candidates, preprocessor, X_train, y_train, prep_strategy.task_type),
    )

    await emitter.emit_async("training", "Evaluating results…", 70)

    # Stat test when top-2 are close
    stat_test_result = maybe_run_stat_test(stability_results[:2], prep_strategy.task_type)
    stat_tests_dict = stat_test_result or {}

    # Determine winner. A chat override is authoritative (see
    # resolve_training_primary) - it is never silently replaced by the
    # leaderboard winner.
    result_names = [r.model_name for r in stability_results]
    fallback = stability_results[0].model_name if stability_results else "xgboost"
    primary = resolve_training_primary(model_selection, result_names, fallback)

    best_result = next((r for r in stability_results if r.model_name == primary), stability_results[0] if stability_results else None)
    best_score = best_result.mean if best_result else 0.0

    # Separate two distinct concepts that diverge under a user override:
    #   metric_winner  - the highest-scoring model (top of the sorted leaderboard)
    #   primary        - the model SELECTED to go forward (may be a human override)
    # Calling the selected model "best" when the user forced a lower-scoring model
    # is wrong, so the audit/checkpoint distinguish them explicitly.
    metric_winner_result = stability_results[0] if stability_results else None
    metric_winner = metric_winner_result.model_name if metric_winner_result else primary
    metric_winner_score = metric_winner_result.mean if metric_winner_result else 0.0
    metric = model_selection.get("primary_metric", "score")
    primary_is_override = (
        model_selection.get("primary_source") == "user_override" and primary != metric_winner
    )

    model_comparison = [r.model_dump() for r in stability_results]

    if primary_is_override:
        training_reason = (
            f"Selected primary: {primary} ({best_score:.4f} {metric}) - manual override. "
            f"Highest {metric}: {metric_winner} ({metric_winner_score:.4f})"
        )
    elif best_result:
        training_reason = f"Selected primary: {primary} ({best_score:.4f} mean ± {best_result.std:.4f} std)"
    else:
        training_reason = "training complete"

    await audit.append(
        session, run_id=run_id, actor="system", category="training",
        action="stability_training_complete",
        payload={
            "candidates": candidates,
            "n_seeds": 3,
            "n_folds": 5,
            "selected_primary": primary,
            "selected_primary_score": round(best_score, 6),
            "primary_is_override": primary_is_override,
            "metric_winner": metric_winner,
            "metric_winner_score": round(metric_winner_score, 6),
            "stat_test_run": stat_test_result is not None,
            "stat_test_p_value": stat_test_result.get("p_value") if stat_test_result else None,
        },
        reason=training_reason,
    )
    await session.commit()

    # Save raw data splits to storage for later steps (avoid re-loading)
    import pickle  # nosec - internal use only
    splits_bytes = pickle.dumps({  # nosec
        "X_train": X_train, "X_test": X_test,
        "y_train": y_train, "y_test": y_test,
        "prep_strategy": prep_strategy,
    })
    splits_path = f"runs/{run_id}/splits.pkl"
    await storage.upload(splits_path, splits_bytes)

    # Flush any modify intents that arrived during training (interrupt semantics §2)
    await flush_pending_intents(session, run)

    run.status = "awaiting_checkpoint"
    run.current_step = "checkpoint_4_training"
    run.progress = 72
    run.model_comparison = model_comparison
    run.stat_tests = stat_tests_dict
    run.best_model_name = primary
    run.best_model_score = round(best_score, 6)
    session.add(run)
    await session.commit()

    stat_msg = ""
    if stat_test_result:
        stat_msg = f" | p={stat_test_result.get('p_value', 1.0):.4f}"

    await run_plots(run_id, "training")

    if primary_is_override:
        checkpoint_msg = (
            f"Training complete - primary: {primary} ({best_score:.4f}, your override) | "
            f"highest {metric}: {metric_winner} ({metric_winner_score:.4f}){stat_msg}"
        )
    else:
        checkpoint_msg = (
            f"Training complete - primary: {primary} "
            f"({best_score:.4f}±{(best_result.std if best_result else 0):.4f}){stat_msg}"
        )

    await emitter.emit_async(
        "checkpoint",
        checkpoint_msg,
        72,
        {
            "checkpoint": 4,
            "step": "training",
            "selected_primary": primary,
            "selected_primary_score": round(best_score, 4),
            "primary_is_override": primary_is_override,
            "metric_winner": metric_winner,
            "metric_winner_score": round(metric_winner_score, 4),
            "stat_test": stat_test_result,
            "leaderboard": [
                {"name": r.model_name, "mean": round(r.mean, 4), "std": round(r.std, 4)}
                for r in stability_results
            ],
        },
    )


# ── Step 5: Tune → Calibrate → Threshold → SHAP → Similarity → Insight ────────


async def _step_tuning(session: Any, run: Run, emitter: ProgressEmitter) -> None:
    run_id = run.id

    ds_result = await session.execute(
        select(Dataset).where(Dataset.id == run.training_dataset_id)
    )
    dataset = ds_result.scalar_one_or_none()

    run.current_step = "tuning"
    run.progress = 74
    session.add(run)
    await session.commit()

    await emitter.emit_async("tuning", "Starting hyperparameter tuning…", 74)

    # Load splits from storage
    import pickle  # nosec
    splits_bytes = await storage.download(f"runs/{run_id}/splits.pkl")
    splits = pickle.loads(splits_bytes)  # nosec
    X_train: pd.DataFrame = splits["X_train"]
    X_test: pd.DataFrame = splits["X_test"]
    y_train: pd.Series = splits["y_train"]
    y_test: pd.Series = splits["y_test"]

    from backend.ml.cleaner import build_preprocessor, split_cal_val
    from backend.ml.trainer import fit_final_pipeline
    from backend.ml.tuner import tune_model
    from backend.models.strategy import PreprocessingStrategy

    prep_strategy = PreprocessingStrategy.model_validate(run.preprocessing_strategy)
    task_type = prep_strategy.task_type
    model_name = run.best_model_name or "xgboost"
    baseline_score = run.best_model_score or 0.0

    preprocessor = build_preprocessor(prep_strategy, X_train)

    import asyncio
    tuned_pipeline, tuning_result = await asyncio.get_running_loop().run_in_executor(
        None,
        lambda: tune_model(
            model_name, preprocessor, X_train, y_train, task_type,
            baseline_score=baseline_score,
        ),
    )

    run.current_step = "calibration"
    run.progress = 78
    run.tuning_result = tuning_result.model_dump(mode="json")
    session.add(run)
    await session.commit()
    await emitter.emit_async("tuning", f"Tuning complete (best {tuning_result.metric}={tuning_result.best_score:.4f})", 78)

    await audit.append(
        session, run_id=run_id, actor="system", category="tuning",
        action="tuning_complete",
        payload={
            "model_name": model_name,
            "best_params": tuning_result.best_params,
            "best_score": tuning_result.best_score,
            "n_trials": tuning_result.n_trials,
            "improvement": tuning_result.improvement_over_baseline,
        },
        reason=f"Optuna tuning: {tuning_result.n_trials} trials, best={tuning_result.best_score:.4f}",
    )
    await session.commit()

    # Split X_train into X_fit / X_cal / X_val for calibration + threshold
    X_fit, X_cal, X_val, y_fit, y_cal, y_val = split_cal_val(
        X_train, y_train, task_type, cal_size=0.20, val_size=0.20
    )

    # Fit final model on X_fit (with best tuning params)
    import copy
    final_prep = copy.deepcopy(build_preprocessor(prep_strategy, X_fit))
    final_pipeline = fit_final_pipeline(model_name, final_prep, X_fit, y_fit, task_type)

    # ── Calibration (classification only) ──────────────────────────────────────
    calibrated_pipeline = final_pipeline
    calibration_report_dict: dict[str, Any] | None = None

    if task_type == "binary_classification":
        from backend.ml.calibration import calibrate_classifier

        run.current_step = "calibration"
        session.add(run)
        await session.commit()
        await emitter.emit_async("calibration", "Calibrating probabilities…", 80)

        calibrated_pipeline, cal_report = await asyncio.get_running_loop().run_in_executor(
            None,
            lambda: calibrate_classifier(final_pipeline, X_cal, y_cal),
        )
        calibration_report_dict = cal_report.model_dump()

        await audit.append(
            session, run_id=run_id, actor="system", category="calibration",
            action="calibration_complete",
            payload={
                "method": cal_report.method,
                "brier_before": cal_report.brier_before,
                "brier_after": cal_report.brier_after,
                "ece_before": cal_report.ece_before,
                "ece_after": cal_report.ece_after,
                "improvement_pct": cal_report.improvement_pct,
            },
            reason=f"Calibrated with {cal_report.method}, improvement={cal_report.improvement_pct:.1f}%",
        )
        await session.commit()
    elif task_type == "multiclass":
        # The calibrator is a binary Platt/Isotonic implementation (operates on
        # the positive-class column). Multiclass calibration is a separate
        # method (per-class OvR or multinomial) not implemented here, so skip it
        # rather than crash - the uncalibrated multiclass probabilities are used.
        # Mirrors threshold optimization, which is likewise binary-only.
        await emitter.emit_async(
            "calibration",
            "Skipping calibration - not supported for multiclass; using uncalibrated probabilities",
            80,
        )
        await audit.append(
            session, run_id=run_id, actor="system", category="calibration",
            action="calibration_skipped",
            payload={"task_type": task_type, "reason": "multiclass not supported by binary calibrator"},
            reason="Calibration skipped: multiclass task, binary-only calibrator",
        )
        await session.commit()

    # ── Threshold optimization (binary classification only) ────────────────────
    threshold_result_dict: dict[str, Any] | None = None
    optimal_threshold = 0.5

    if task_type == "binary_classification":
        from backend.ml.threshold_optimizer import optimize_threshold
        from backend.models.strategy import CostMatrix, ThresholdConfig

        await emitter.emit_async("threshold", "Optimizing classification threshold…", 82)

        # Get calibrated probabilities on X_val (NEVER X_test)
        y_proba_val = calibrated_pipeline.predict_proba(X_val)[:, 1]

        # Use cost matrix from threshold_config or default
        threshold_cfg = run.threshold_config or {}
        cost_dict = threshold_cfg.get("cost_matrix", {})
        cost_matrix = CostMatrix(**cost_dict) if cost_dict else CostMatrix()

        thresh_result = await asyncio.get_running_loop().run_in_executor(
            None,
            lambda: optimize_threshold(
                np.asarray(y_val), y_proba_val, cost_matrix
            ),
        )
        threshold_result_dict = thresh_result.model_dump()
        optimal_threshold = thresh_result.optimal_threshold

        await audit.append(
            session, run_id=run_id, actor="system", category="threshold",
            action="threshold_optimization_complete",
            payload={
                "optimal_threshold": optimal_threshold,
                "cost_at_default": thresh_result.cost_at_default,
                "cost_at_optimal": thresh_result.cost_at_optimal,
                "improvement_pct": thresh_result.improvement_pct,
                "cost_matrix": cost_matrix.model_dump(),
            },
            reason=thresh_result.note or f"Optimal threshold={optimal_threshold:.3f}",
        )
        await session.commit()

    # ── Final evaluation on sealed test set ────────────────────────────────────
    await emitter.emit_async("eval", "Evaluating on test set…", 84)

    final_metrics = _compute_final_metrics(
        calibrated_pipeline, X_test, y_test, task_type, optimal_threshold
    )
    eval_plots = _compute_eval_plots(
        calibrated_pipeline, X_test, y_test, task_type, optimal_threshold
    )

    await audit.append(
        session, run_id=run_id, actor="system", category="eval",
        action="test_evaluation_complete",
        payload={"metrics": final_metrics, "threshold_used": optimal_threshold},
        reason="Final evaluation on sealed test set (opened once)",
    )
    await session.commit()

    # ── Persist model artifacts ────────────────────────────────────────────────
    import joblib

    model_bytes = io.BytesIO()
    joblib.dump(calibrated_pipeline, model_bytes)
    model_path = f"runs/{run_id}/model.joblib"
    await storage.upload(model_path, model_bytes.getvalue())

    # ── SHAP ───────────────────────────────────────────────────────────────────
    run.current_step = "shap"
    run.progress = 85
    session.add(run)
    await session.commit()
    await emitter.emit_async("shap", "Computing SHAP explanations…", 85)

    from backend.ml.explainer import compute_shap

    feature_names = list(X_test.columns)
    shap_summary = await asyncio.get_running_loop().run_in_executor(
        None,
        lambda: compute_shap(
            calibrated_pipeline, X_test, feature_names, task_type,
            background_data=X_train,
        ),
    )
    shap_dict = shap_summary.model_dump()

    await audit.append(
        session, run_id=run_id, actor="system", category="shap",
        action="shap_complete",
        payload={
            "top_features": shap_summary.top_k_features[:5],
            "explainer_type": shap_summary.explainer_type,
            "n_samples": shap_summary.n_samples,
        },
        reason=f"SHAP computed via {shap_summary.explainer_type}Explainer on {shap_summary.n_samples} samples",
    )
    await session.commit()

    # ── Similarity index ───────────────────────────────────────────────────────
    run.current_step = "similarity"
    run.progress = 87
    session.add(run)
    await session.commit()
    await emitter.emit_async("similarity", "Building similarity index…", 87)

    faiss_index_path: str | None = None
    try:
        from backend.ml.similarity import SimilarityIndex
        from backend.ml.cleaner import apply_preprocessor

        # Build a fresh preprocessor and transform X_train for the index.
        # We cannot reuse the tuning-step preprocessor here - it may have been
        # fitted on a sub-split (X_fit). Refit on the full X_train.
        sim_prep = build_preprocessor(prep_strategy, X_train)
        X_train_t, _ = apply_preprocessor(sim_prep, X_train, X_test)
        sim_index = SimilarityIndex()
        sim_index.fit(X_train_t)
        sim_bytes = sim_index.serialize()
        faiss_index_path = f"runs/{run_id}/similarity.index"
        await storage.upload(faiss_index_path, sim_bytes)

        await audit.append(
            session, run_id=run_id, actor="system", category="similarity",
            action="similarity_index_built",
            payload={"n_train_samples": len(X_train_t)},
            reason="BallTree L2 index built on preprocessed training features",
        )
        await session.commit()
    except ImportError:
        logger.warning("faiss-cpu not installed - skipping similarity index")
    except Exception as exc:
        logger.warning("Similarity index failed (non-fatal): %s", exc)

    # ── Drift detection (when comparison/inference dataset exists) ─────────────
    drift_report_dict: dict[str, Any] | None = None

    comparison_dataset = await _find_comparison_dataset(session, run.project_id)
    if comparison_dataset is not None:
        run.current_step = "drift"
        run.progress = 89
        session.add(run)
        await session.commit()
        await emitter.emit_async("drift", f"Computing drift vs. {comparison_dataset.filename}…", 89)

        try:
            comp_bytes = await storage.download(comparison_dataset.storage_path)
            df_comp = _parse_dataframe(comp_bytes, comparison_dataset.filename)

            from backend.ml.drift import compute_drift_report

            # Use raw columns from training frame (pre-transform) for distribution drift
            raw_bytes = await storage.download(dataset.storage_path)
            df_train_raw = _parse_dataframe(raw_bytes, dataset.filename)

            numeric_cols = prep_strategy.numeric_columns()
            categorical_cols = prep_strategy.categorical_columns()

            drift_report = compute_drift_report(
                df_train_raw, df_comp,
                numeric_cols=[c for c in numeric_cols if c in df_comp.columns],
                categorical_cols=[c for c in categorical_cols if c in df_comp.columns],
            )
            drift_report_dict = drift_report.model_dump()

            await audit.append(
                session, run_id=run_id, actor="system", category="drift",
                action="drift_analysis_complete",
                payload={
                    "comparison_dataset": comparison_dataset.filename,
                    "overall_severity": drift_report.overall_severity,
                    "aggregate_psi": drift_report.aggregate_psi,
                    "n_features_drifted": drift_report.n_features_drifted,
                    "significant_features": drift_report.significant_features[:10],
                },
                reason=f"Drift vs. {comparison_dataset.filename}: {drift_report.overall_severity}",
            )
            await session.commit()

            if drift_report.warning:
                await emitter.emit_async("drift", drift_report.warning, 89)
        except Exception as exc:
            logger.warning("Drift detection failed (non-fatal): %s", exc)

    # ── Fairness audit (when protected attributes are configured) ──────────────
    fairness_report_dict: dict[str, Any] | None = None
    fairness_blocks = False

    fairness_cfg = run.fairness_config or {}
    protected_cols = fairness_cfg.get("protected_columns", [])

    if protected_cols:
        run.current_step = "fairness"
        run.progress = 90
        session.add(run)
        await session.commit()
        await emitter.emit_async(
            "fairness", f"Running fairness audit on {protected_cols}…", 90
        )

        try:
            from backend.ml.fairness import build_sensitive_features, fairness_audit

            # Reload raw training data to extract protected attribute arrays
            raw_bytes_fair = await storage.download(dataset.storage_path)
            df_raw_fair = _parse_dataframe(raw_bytes_fair, dataset.filename)

            test_index = X_test.index
            sensitive = build_sensitive_features(df_raw_fair, protected_cols, index=test_index)

            task_type_fair = prep_strategy.task_type
            if task_type_fair in ("binary_classification", "multiclass"):
                y_proba_fair = calibrated_pipeline.predict_proba(X_test)
                if task_type_fair == "binary_classification":
                    y_proba_fair = y_proba_fair[:, 1]
            else:
                y_proba_fair = None

            y_pred_fair = calibrated_pipeline.predict(X_test)
            if task_type_fair == "binary_classification":
                y_pred_fair = (
                    calibrated_pipeline.predict_proba(X_test)[:, 1] >= optimal_threshold
                ).astype(int)

            fairness_report = await asyncio.get_running_loop().run_in_executor(
                None,
                lambda: fairness_audit(
                    np.asarray(y_test),
                    np.asarray(y_pred_fair),
                    np.asarray(y_proba_fair) if y_proba_fair is not None else None,
                    sensitive,
                ),
            )
            fairness_report_dict = fairness_report.model_dump()
            fairness_blocks = fairness_report.blocks_deliverables

            await audit.append(
                session, run_id=run_id, actor="system", category="fairness",
                action="fairness_audit_complete",
                payload={
                    "protected_columns": protected_cols,
                    "overall_severity": fairness_report.overall_severity,
                    "blocks_deliverables": fairness_blocks,
                    "attributes": [
                        {
                            "attribute": r.attribute,
                            "dp_diff": r.demographic_parity_diff,
                            "severity": r.severity,
                        }
                        for r in fairness_report.attributes
                    ],
                },
                reason=f"Fairness audit: overall_severity={fairness_report.overall_severity}",
            )
            await session.commit()

            if fairness_blocks:
                await emitter.emit_async(
                    "fairness",
                    f"Severe fairness disparity detected ({fairness_report.overall_severity}). "
                    "Deliverable generation is blocked until you acknowledge this in the chat.",
                    90,
                )
        except Exception as exc:
            logger.warning("Fairness audit failed (non-fatal): %s", exc)

    # ── Holdout evaluation (if holdout dataset is sealed on this run) ──────────
    holdout_metrics: dict[str, Any] | None = None
    if run.holdout_dataset_id:
        try:
            from sqlalchemy import select as sa_select

            hd_result = await session.execute(
                sa_select(Dataset).where(Dataset.id == run.holdout_dataset_id)
            )
            holdout_ds = hd_result.scalar_one_or_none()

            if holdout_ds is not None:
                await emitter.emit_async(
                    "holdout", f"Opening sealed holdout set: {holdout_ds.filename}…", 91
                )
                await audit.append(
                    session, run_id=run_id, actor="system", category="holdout",
                    action="holdout_open",
                    payload={
                        "filename": holdout_ds.filename,
                        "sha256": holdout_ds.sha256,
                        "row_count": holdout_ds.row_count,
                    },
                    reason="Holdout set opened - final evaluation only (opened once)",
                )
                await session.commit()

                hd_bytes = await storage.download(holdout_ds.storage_path)
                df_holdout = _parse_dataframe(hd_bytes, holdout_ds.filename)

                target_col = dataset.target_column or prep_strategy.target_column
                if target_col in df_holdout.columns:
                    from backend.ml.cleaner import prepare_data

                    X_ho, y_ho = prepare_data(
                        df_holdout, prep_strategy, target_strategy=run.target_strategy
                    )
                    holdout_metrics = _compute_final_metrics(
                        calibrated_pipeline, X_ho, y_ho,
                        prep_strategy.task_type, optimal_threshold
                    )

                    await audit.append(
                        session, run_id=run_id, actor="system", category="holdout",
                        action="holdout_metric",
                        payload={
                            "holdout_metrics": holdout_metrics,
                            "cv_estimate": run.best_model_score,
                        },
                        reason=f"Holdout evaluation complete: {holdout_metrics}",
                    )
                    await session.commit()

                    await emitter.emit_async(
                        "holdout",
                        f"Holdout evaluation complete: {holdout_metrics}",
                        91,
                    )
        except Exception as exc:
            logger.warning("Holdout evaluation failed (non-fatal): %s", exc)

    # ── Insight report ─────────────────────────────────────────────────────────
    run.current_step = "insight"
    run.progress = 92
    session.add(run)
    await session.commit()

    from backend.agents.insight_agent import run_insight_agent

    stability_list = run.model_comparison or []
    insight_text = await run_insight_agent(
        session, run_id,
        task_type=prep_strategy.task_type,
        model_name=model_name,
        stability_results=stability_list,
        calibration_report=calibration_report_dict,
        threshold_result=threshold_result_dict,
        shap_summary=shap_dict,
        eda_report=run.eda_report or {},
        stat_tests=run.stat_tests or None,
        emitter=emitter,
    )

    # Flush any modify intents that arrived during tuning/calibration/shap/similarity
    await flush_pending_intents(session, run)

    # ── Checkpoint 5: persist all results ─────────────────────────────────────
    run.status = "awaiting_checkpoint"
    run.current_step = "checkpoint_5_final"
    run.progress = 95
    run.calibration_report = calibration_report_dict
    run.threshold_result = threshold_result_dict
    run.threshold_config = run.threshold_config or {
        "optimal_threshold": optimal_threshold
    }
    run.final_metrics = final_metrics
    run.eval_plots = eval_plots
    run.shap_summary = shap_dict
    run.similarity_index_built = faiss_index_path is not None
    run.faiss_index_path = faiss_index_path
    run.model_storage_path = model_path
    run.drift_report = drift_report_dict
    run.fairness_report = fairness_report_dict
    run.insight_report = insight_text
    run.seeds = {"stability": [42, 0, 1], "final": 42}
    run.claude_models_used = {
        "eda": settings.CLAUDE_SONNET_MODEL,
        "preprocessing": settings.CLAUDE_SONNET_MODEL,
        "model_selection": settings.CLAUDE_SONNET_MODEL,
        "insight": settings.CLAUDE_OPUS_MODEL,
        "chat": settings.CLAUDE_SONNET_MODEL,
    }
    session.add(run)
    await session.commit()

    checkpoint_extra: dict[str, Any] = {
        "checkpoint": 5,
        "step": "final",
        "final_metrics": final_metrics,
        "optimal_threshold": optimal_threshold,
        "top_features": shap_summary.top_k_features[:3],
    }
    if drift_report_dict:
        checkpoint_extra["drift_severity"] = drift_report_dict.get("overall_severity")
    if fairness_report_dict:
        checkpoint_extra["fairness_severity"] = fairness_report_dict.get("overall_severity")
        checkpoint_extra["fairness_blocks"] = fairness_blocks
    if holdout_metrics:
        checkpoint_extra["holdout_metrics"] = holdout_metrics

    await emitter.emit_async(
        "checkpoint",
        "Pipeline complete - review results before generating deliverables",
        95,
        checkpoint_extra,
    )


# ── Step 7: Deliverables ───────────────────────────────────────────────────────


async def _step_deliverables_placeholder(
    session: Any, run: Run, emitter: ProgressEmitter
) -> None:
    """Dispatch the deliverable generation Celery task.

    The analysis task commits all pipeline results and then hands off to
    the dedicated deliverable task so the two phases are independently
    retriable and their Celery workers can be sized separately.
    """
    from backend.tasks.deliverable_task import generate_deliverables_task

    run_id = run.id

    # Commit all pipeline state before handing off
    await session.commit()

    await emitter.emit_async("deliverables_queued", "Queuing deliverable generation…", 96)

    generate_deliverables_task.delay(run_id)


# ── Helpers ────────────────────────────────────────────────────────────────────


def _compute_final_metrics(
    pipeline: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    task_type: str,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Compute final test-set metrics. TEST SET IS OPENED HERE - once only."""
    import numpy as np
    from sklearn.metrics import (
        accuracy_score, f1_score, mean_absolute_error,
        r2_score, roc_auc_score,
    )
    from sklearn.metrics import root_mean_squared_error

    y_true = np.asarray(y_test)

    metrics: dict[str, Any] = {}

    if task_type == "binary_classification":
        y_proba = pipeline.predict_proba(X_test)[:, 1]
        y_pred = (y_proba >= threshold).astype(int)
        metrics["auc"] = round(float(roc_auc_score(y_true, y_proba)), 6)
        metrics["f1"] = round(float(f1_score(y_true, y_pred, zero_division=0)), 6)
        metrics["accuracy"] = round(float(accuracy_score(y_true, y_pred)), 6)
        metrics["threshold_used"] = threshold
        from sklearn.metrics import precision_score, recall_score
        metrics["precision"] = round(float(precision_score(y_true, y_pred, zero_division=0)), 6)
        metrics["recall"] = round(float(recall_score(y_true, y_pred, zero_division=0)), 6)

    elif task_type == "multiclass":
        y_proba = pipeline.predict_proba(X_test)
        y_pred = pipeline.predict(X_test)
        metrics["macro_auc"] = round(float(roc_auc_score(y_true, y_proba, multi_class="ovr", average="weighted")), 6)
        metrics["macro_f1"] = round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 6)
        metrics["accuracy"] = round(float(accuracy_score(y_true, y_pred)), 6)

    else:  # regression
        y_pred = pipeline.predict(X_test)
        metrics["rmse"] = round(float(root_mean_squared_error(y_true, y_pred)), 6)
        metrics["mae"] = round(float(mean_absolute_error(y_true, y_pred)), 6)
        metrics["r2"] = round(float(r2_score(y_true, y_pred)), 6)

    return metrics


def _compute_eval_plots(
    pipeline: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    task_type: str,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Compute curve data for all evaluation plots. Downsamples to ≤200 points per curve."""
    import numpy as np

    plots: dict[str, Any] = {}
    y_true = np.asarray(y_test)

    def _downsample(arr: np.ndarray, n: int = 200) -> list[float]:
        if len(arr) <= n:
            return [round(float(v), 6) for v in arr]
        idx = np.linspace(0, len(arr) - 1, n, dtype=int)
        return [round(float(arr[i]), 6) for i in idx]

    if task_type == "binary_classification":
        from sklearn.metrics import (
            roc_curve, precision_recall_curve, average_precision_score,
            confusion_matrix,
        )
        from sklearn.calibration import calibration_curve
        y_proba = pipeline.predict_proba(X_test)[:, 1]
        y_pred = (y_proba >= threshold).astype(int)

        # ROC curve
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        plots["roc_curve"] = {
            "fpr": _downsample(fpr),
            "tpr": _downsample(tpr),
        }

        # Precision-Recall curve
        prec, rec, _ = precision_recall_curve(y_true, y_proba)
        plots["pr_curve"] = {
            "precision": _downsample(prec),
            "recall": _downsample(rec),
            "ap": round(float(average_precision_score(y_true, y_proba)), 6),
        }

        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        plots["confusion_matrix"] = {
            "tn": int(tn), "fp": int(fp),
            "fn": int(fn), "tp": int(tp),
        }

        # Score distribution - histogram of predicted probabilities by class
        bins = np.linspace(0, 1, 31)  # 30 bins
        neg_scores = y_proba[y_true == 0]
        pos_scores = y_proba[y_true == 1]
        neg_hist, _ = np.histogram(neg_scores, bins=bins)
        pos_hist, _ = np.histogram(pos_scores, bins=bins)
        bin_centers = [(bins[i] + bins[i + 1]) / 2 for i in range(len(bins) - 1)]
        plots["score_distribution"] = [
            {"score": round(c, 3), "negative": int(n), "positive": int(p)}
            for c, n, p in zip(bin_centers, neg_hist, pos_hist)
        ]

        # Calibration curve (reliability diagram)
        try:
            prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=10)
            plots["calibration_curve"] = {
                "prob_true": [round(float(v), 6) for v in prob_true],
                "prob_pred": [round(float(v), 6) for v in prob_pred],
            }
        except Exception:
            pass

    elif task_type == "multiclass":
        from sklearn.metrics import (
            confusion_matrix, roc_curve, roc_auc_score,
            precision_recall_curve, average_precision_score,
        )
        from sklearn.calibration import calibration_curve

        y_pred = pipeline.predict(X_test)
        # Use the fitted estimator's class order so it lines up with the columns
        # of predict_proba; fall back to sorted unique labels.
        classes_attr = getattr(pipeline, "classes_", None)
        classes = list(classes_attr) if classes_attr is not None else sorted(set(y_true.tolist()))

        cm = confusion_matrix(y_true, y_pred, labels=classes)
        plots["confusion_matrix_multi"] = {
            "matrix": cm.tolist(),
            "classes": [str(c) for c in classes],
        }

        # One-vs-rest curves per class. Capped to keep the payload small and the
        # results page readable; classes whose support is 0 or 100% are skipped
        # (their one-vs-rest curve is undefined).
        try:
            proba = pipeline.predict_proba(X_test)
        except Exception:  # noqa: BLE001 - some estimators lack predict_proba
            proba = None

        if proba is not None and proba.shape[1] == len(classes):
            max_classes = 10
            roc_multi: list[dict[str, Any]] = []
            pr_multi: list[dict[str, Any]] = []
            cal_multi: list[dict[str, Any]] = []
            for ci, cls in enumerate(classes[:max_classes]):
                y_bin = (y_true == cls).astype(int)
                pos = int(y_bin.sum())
                if pos == 0 or pos == len(y_bin):
                    continue
                p = proba[:, ci]

                fpr, tpr, _ = roc_curve(y_bin, p)
                roc_multi.append({
                    "label": str(cls),
                    "fpr": _downsample(fpr),
                    "tpr": _downsample(tpr),
                    "auc": round(float(roc_auc_score(y_bin, p)), 6),
                })

                prec, rec, _ = precision_recall_curve(y_bin, p)
                pr_multi.append({
                    "label": str(cls),
                    "precision": _downsample(prec),
                    "recall": _downsample(rec),
                    "ap": round(float(average_precision_score(y_bin, p)), 6),
                })

                try:
                    prob_true, prob_pred = calibration_curve(y_bin, p, n_bins=10)
                    cal_multi.append({
                        "label": str(cls),
                        "prob_true": [round(float(v), 6) for v in prob_true],
                        "prob_pred": [round(float(v), 6) for v in prob_pred],
                    })
                except Exception:  # noqa: BLE001 - too few points in some bins
                    pass

            if roc_multi:
                plots["roc_curve_multi"] = roc_multi
            if pr_multi:
                plots["pr_curve_multi"] = pr_multi
            if cal_multi:
                plots["calibration_curve_multi"] = cal_multi

    else:  # regression
        y_pred = pipeline.predict(X_test)
        residuals = y_true - y_pred
        # Downsample scatter to 500 points
        n = min(500, len(y_true))
        idx = np.random.default_rng(42).choice(len(y_true), n, replace=False)
        idx = np.sort(idx)
        plots["predicted_vs_actual"] = [
            {"actual": round(float(y_true[i]), 4), "predicted": round(float(y_pred[i]), 4)}
            for i in idx
        ]
        plots["residuals"] = [
            {"predicted": round(float(y_pred[i]), 4), "residual": round(float(residuals[i]), 4)}
            for i in idx
        ]

    return plots


def _compress_stored_profile(profile_dict: dict[str, Any]) -> dict[str, Any]:
    """Return a lightweight version of a stored profile dict for LLM context."""
    if not profile_dict:
        return {}
    # Re-use profiler's compress logic if available
    try:
        from backend.ml.profiler import DatasetProfile, compress_profile_for_claude
        dp = DatasetProfile.model_validate(profile_dict)
        return compress_profile_for_claude(dp)
    except Exception:
        # Fallback: return without heavy columns
        return {k: v for k, v in profile_dict.items() if k not in ("correlation_matrix",)}


async def _resolve_target_strategy(
    session: Any, run: Run, dataset: Dataset, df: pd.DataFrame
) -> "Any":
    """Merge target hygiene from three sources, human override winning (§7, §10).

    Precedence / union:
      - drop_labels: union of any chat override already on the run, the case brief's
        drop_labels, and deterministic unlabelled detection on the target. Union is
        safe - dropping a placeholder row is never harmful and the deterministic
        scan guarantees "unknown" is caught even if the brief parse refused.
      - positive_labels (binary collapse): the human override wins; otherwise the
        brief's. Never inferred deterministically (which class is "positive" is a
        domain decision), so absent a brief/override the target stays as-is.
    """
    from backend.ml.leakage_detector import detect_unlabeled_target_classes
    from backend.models.strategy import TargetStrategy

    base = (
        TargetStrategy.model_validate(run.target_strategy)
        if run.target_strategy else TargetStrategy()
    )

    project = (
        await session.execute(select(Project).where(Project.id == run.project_id))
    ).scalar_one_or_none() if hasattr(run, "project_id") else None
    brief = (project.case_brief if project else None) or {}
    brief_ts = brief.get("target_strategy") if brief.get("parsed") else None
    brief_drop = list((brief_ts or {}).get("drop_labels") or [])
    brief_pos = list((brief_ts or {}).get("positive_labels") or [])

    detected: list[str] = []
    if dataset.target_column in df.columns:
        detected = detect_unlabeled_target_classes(df[dataset.target_column]).suspicious_classes

    drop_labels: list[str] = []
    seen: set[str] = set()
    for label in [*base.drop_labels, *brief_drop, *detected]:
        key = str(label).strip().lower()
        if key and key not in seen:
            seen.add(key)
            drop_labels.append(str(label))

    positive_labels = base.positive_labels or brief_pos

    sources = []
    if base.drop_labels or base.positive_labels:
        sources.append("chat override")
    if brief_ts:
        sources.append("case brief")
    if detected:
        sources.append("unlabelled detection")

    return TargetStrategy(
        drop_labels=drop_labels,
        positive_labels=[str(p) for p in positive_labels],
        note="; ".join(sources),
    )


async def _find_comparison_dataset(session: Any, project_id: str) -> Dataset | None:
    """Return the most recently uploaded inference or comparison dataset for this project."""
    from sqlalchemy import desc

    result = await session.execute(
        select(Dataset)
        .where(Dataset.project_id == project_id)
        .where(Dataset.role.in_(["inference", "comparison"]))
        .order_by(desc(Dataset.created_at))
        .limit(1)
    )
    return result.scalar_one_or_none()


async def _mark_failed(run_id: str, error: str) -> None:
    from sqlalchemy.ext.asyncio import AsyncSession as _AS
    from sqlalchemy.ext.asyncio import async_sessionmaker as _asf
    from sqlalchemy.ext.asyncio import create_async_engine as _cae
    _local_engine = _cae(settings.DATABASE_URL, pool_pre_ping=True)
    _local_factory = _asf(_local_engine, class_=_AS, expire_on_commit=False)
    try:
        async with _local_factory() as session:
            result = await session.execute(select(Run).where(Run.id == run_id))
            run = result.scalar_one_or_none()
            if run:
                run.status = "failed"
                run.error_message = error[:2000]
                session.add(run)
                await session.commit()
    finally:
        await _local_engine.dispose()


def _parse_dataframe(raw_bytes: bytes, filename: str) -> pd.DataFrame:
    if filename.endswith(".parquet"):
        return pd.read_parquet(io.BytesIO(raw_bytes))
    return pd.read_csv(io.BytesIO(raw_bytes))


