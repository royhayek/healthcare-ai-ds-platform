"""Model selection agent - candidate ranking with hard exclusion rules (§13).

Hard exclusion rules are applied in Python BEFORE the LLM call, so the LLM
only chooses from the eligible set. This avoids the LLM recommending an SVM
on a 50k-row dataset.

Hard exclusion rules:
  - SVM: excluded if n_rows > 10_000 (quadratic training cost)
  - KNN: excluded if n_rows > 50_000 (inference cost)
  - Gradient Boosting (sklearn): excluded if n_rows > 100_000 (slow vs XGBoost/LGBM)

Expected JSON schema from the model:
{
  "candidates": ["xgboost", "lightgbm", "random_forest"],
  "primary": "xgboost",
  "primary_metric": "auc",
  "excluded": [{"name": "svm", "reason": "..."}],
  "reasoning": "2-3 sentence justification",
  "notes": null
}
"""

import logging
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from backend.agents.base import call_claude, extract_json
from backend.core import audit
from backend.core.config import settings
from backend.core.events import ProgressEmitter
from backend.models.strategy import ModelSelectionStrategy

logger = logging.getLogger(__name__)

_ALL_MODELS = [
    "logistic_regression",
    "random_forest",
    "gradient_boosting",
    "xgboost",
    "lightgbm",
    "linear_regression",
    "ridge_regression",
]

_CLASSIFICATION_MODELS = [
    "logistic_regression", "random_forest", "gradient_boosting", "xgboost", "lightgbm"
]
_REGRESSION_MODELS = [
    "linear_regression", "ridge_regression", "random_forest",
    "gradient_boosting", "xgboost", "lightgbm"
]

_SYSTEM = """You are a senior ML engineer selecting model candidates for a
standard supervised-learning task on a de-identified, aggregate statistical
profile of a tabular dataset. This is a routine model-selection step: choose
estimators based purely on dataset shape, task type, and class balance. You do
not see any raw records and the choice does not depend on the subject matter of
the columns. Return ONLY a valid JSON object. No markdown fences.

Select 2-4 candidates from the eligible list. The primary candidate should be
the single best choice. primary_metric must be one of: auc, f1, macro_auc, rmse, r2."""

_PROMPT_TEMPLATE = """Select model candidates for this supervised-learning task
based on the structural dataset characteristics below. Choose estimators by
dataset shape, task type, and class balance only.

Task type: {task_type}
Dataset size: {n_rows} rows × {n_cols} columns
Class balance: {class_balance}

Eligible models (after exclusion rules): {eligible_models}
Already excluded: {already_excluded}

Profile highlights (counts only, no row-level data):
{profile_highlights}

Return JSON with: candidates (list), primary (string), primary_metric (string),
excluded (list of {{name, reason}}), reasoning (string), notes (null or string)."""


def _apply_hard_exclusion_rules(
    n_rows: int,
    task_type: str,
) -> tuple[list[str], list[dict[str, str]]]:
    """Apply exclusion rules. Returns (eligible, excluded)."""
    base = _CLASSIFICATION_MODELS if task_type != "regression" else _REGRESSION_MODELS

    excluded: list[dict[str, str]] = []
    eligible: list[str] = []

    for model in base:
        reason = None
        if model == "gradient_boosting" and n_rows > 100_000:
            reason = f"sklearn GradientBoosting is slow on {n_rows:,} rows - prefer XGBoost or LightGBM"
        if reason:
            excluded.append({"name": model, "reason": reason})
        else:
            eligible.append(model)

    return eligible, excluded


async def run_model_selector_agent(
    session: AsyncSession,
    run_id: str,
    compressed_profile: dict[str, Any],
    eda_report: dict[str, Any],
    task_type: str,
    n_rows: int,
    emitter: ProgressEmitter,
) -> ModelSelectionStrategy:
    """Run the model selector agent and return a ModelSelectionStrategy.

    Hard exclusion rules are applied before the LLM call. Falls back to a
    rule-based selection if JSON parse fails.
    """
    import json as _json

    await emitter.emit_async("model_selection", "Selecting model candidates…", 48)

    eligible, hard_excluded = _apply_hard_exclusion_rules(n_rows, task_type)

    # Build context for the LLM
    profile_highlights = {
        "n_rows": n_rows,
        "n_cols": compressed_profile.get("n_cols", 0),
        "null_count": compressed_profile.get("null_count", 0),
        "task_type": task_type,
    }
    target_analysis = eda_report.get("target_analysis", {})
    class_balance_str = str(target_analysis.get("class_balance", "unknown"))

    prompt = _PROMPT_TEMPLATE.format(
        task_type=task_type,
        n_rows=n_rows,
        n_cols=profile_highlights["n_cols"],
        class_balance=class_balance_str,
        eligible_models=", ".join(eligible),
        already_excluded=", ".join(f"{e['name']} ({e['reason']})" for e in hard_excluded)
        if hard_excluded else "none",
        profile_highlights=_json.dumps(profile_highlights, indent=2),
    )

    # Guard the call so a transport error / retries-exhausted can't fail the run;
    # degrade to the rule-based fallback instead.
    try:
        raw_text = await call_claude(
            messages=[{"role": "user", "content": prompt}],
            model=settings.CLAUDE_SONNET_MODEL,
            system=_SYSTEM,
            max_tokens=2048,
        )
    except Exception as exc:
        logger.error("Model selector: model call failed for run %s: %s", run_id, exc)
        raw_text = ""

    try:
        import os
        debug_dir = os.path.join(os.path.dirname(__file__), "..", "..", "data", "debug")
        os.makedirs(debug_dir, exist_ok=True)
        with open(os.path.join(debug_dir, f"model_selection_raw_{run_id}.txt"), "w", encoding="utf-8") as _f:
            _f.write(raw_text)
    except Exception as _exc:
        logger.warning("Could not write model selection debug file: %s", _exc)

    parsed = extract_json(raw_text)
    parse_failed = not parsed

    if parse_failed:
        logger.error("Model selector: JSON parse failure for run %s (raw len=%d)", run_id, len(raw_text))
        strategy = _safe_fallback(eligible, task_type, hard_excluded)
    else:
        try:
            strategy = _build_strategy(parsed, eligible, hard_excluded)
        except Exception as exc:
            logger.error("Model selector: strategy build failed for run %s: %s", run_id, exc)
            strategy = _safe_fallback(eligible, task_type, hard_excluded)
            parse_failed = True

    action = "model_selection_parse_failure" if parse_failed else "model_selection_complete"
    await audit.append(
        session,
        run_id=run_id,
        actor="ai",
        category="model_selection",
        action=action,
        payload={
            "model": settings.CLAUDE_SONNET_MODEL,
            "candidates": strategy.candidates,
            "primary": strategy.primary,
            "primary_metric": strategy.primary_metric,
            "hard_excluded": [e["name"] for e in hard_excluded],
            "parse_failed": parse_failed,
        },
        reason=strategy.reasoning or "model selection complete",
    )
    await session.commit()

    await emitter.emit_async("model_selection", f"Primary candidate: {strategy.primary}", 50)
    return strategy


def _build_strategy(
    parsed: dict[str, Any],
    eligible: list[str],
    hard_excluded: list[dict[str, str]],
) -> ModelSelectionStrategy:
    candidates = [c for c in parsed.get("candidates", []) if c in eligible]
    if not candidates:
        candidates = eligible[:2]

    primary = parsed.get("primary", candidates[0] if candidates else eligible[0])
    if primary not in candidates:
        primary = candidates[0]

    all_excluded = hard_excluded + [
        e for e in parsed.get("excluded", [])
        if isinstance(e, dict) and "name" in e
    ]

    return ModelSelectionStrategy(
        candidates=candidates,
        primary=primary,
        primary_metric=parsed.get("primary_metric", "auc"),
        excluded=all_excluded,
        reasoning=parsed.get("reasoning", ""),
        notes=parsed.get("notes"),
    )


def _safe_fallback(
    eligible: list[str],
    task_type: str,
    hard_excluded: list[dict[str, str]],
) -> ModelSelectionStrategy:
    if task_type == "regression":
        primary = "ridge_regression" if "ridge_regression" in eligible else eligible[0]
        metric = "rmse"
    elif "xgboost" in eligible:
        primary = "xgboost"
        metric = "auc"
    elif "lightgbm" in eligible:
        primary = "lightgbm"
        metric = "auc"
    else:
        primary = eligible[0] if eligible else "logistic_regression"
        metric = "auc" if task_type != "regression" else "rmse"

    candidates = [primary] + [m for m in eligible[:3] if m != primary]

    return ModelSelectionStrategy(
        candidates=candidates[:3],
        primary=primary,
        primary_metric=metric,
        excluded=hard_excluded,
        reasoning="Safe fallback - model JSON parse failed; using rule-based selection.",
    )
