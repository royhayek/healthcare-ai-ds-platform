"""Training pipeline with stability runs (§14).

Stability protocol (project rule 12):
  - 3 seeds × k-fold CV per candidate
  - Leaderboard reports mean ± std across BOTH axes
  - When the top two are within 0.005, fire stat_tests and surface p-value
  - Single CV result reporting is explicitly disallowed

Parallel execution:
  - Candidates are evaluated in parallel via joblib.Parallel
  - sklearn's n_jobs=-1 handles fold parallelism inside each candidate

Seven supported model types:
  logistic_regression, random_forest, gradient_boosting, xgboost,
  lightgbm, linear_regression, ridge_regression
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.compose import ColumnTransformer
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.linear_model import LinearRegression as SKLinearRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    make_scorer,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    GroupKFold,
    KFold,
    StratifiedGroupKFold,
    StratifiedKFold,
    cross_validate as sk_cross_validate,
)
from sklearn.pipeline import Pipeline

from backend.ml.stat_tests import run_comparison_test
from backend.models.strategy import CVResult, StabilityResult

logger = logging.getLogger(__name__)

STABILITY_SEEDS = [42, 0, 1]
CLOSE_THRESHOLD = 0.005  # trigger stat test when top-2 scores are within this


def make_estimator(model_name: str, random_state: int = 42) -> Any:
    """Return an unfitted sklearn-compatible estimator."""
    mapping: dict[str, Any] = {
        "logistic_regression": LogisticRegression(
            max_iter=1000, random_state=random_state, class_weight="balanced"
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200, random_state=random_state, n_jobs=1, class_weight="balanced"
        ),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=200, random_state=random_state
        ),
        "xgboost": None,  # built on-demand - optional dep
        "lightgbm": None,  # built on-demand - optional dep
        "linear_regression": SKLinearRegression(),
        "ridge_regression": Ridge(random_state=random_state),
        # Regression variants of tree models
        "random_forest_regressor": RandomForestRegressor(
            n_estimators=200, random_state=random_state, n_jobs=1
        ),
        "gradient_boosting_regressor": GradientBoostingRegressor(
            n_estimators=200, random_state=random_state
        ),
    }

    if model_name == "xgboost":
        from xgboost import XGBClassifier
        return XGBClassifier(
            n_estimators=200,
            random_state=random_state,
            eval_metric="logloss",
            use_label_encoder=False,
            verbosity=0,
        )
    if model_name == "xgboost_regressor":
        from xgboost import XGBRegressor
        return XGBRegressor(n_estimators=200, random_state=random_state, verbosity=0)

    if model_name == "lightgbm":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(
            n_estimators=200, random_state=random_state, verbose=-1, is_unbalance=True
        )
    if model_name == "lightgbm_regressor":
        from lightgbm import LGBMRegressor
        return LGBMRegressor(n_estimators=200, random_state=random_state, verbose=-1)

    est = mapping.get(model_name)
    if est is None:
        raise ValueError(f"Unknown model: {model_name!r}")
    return est


def _resolve_model_name(name: str, task_type: str) -> str:
    """Map generic name to task-specific variant."""
    if task_type == "regression":
        remap = {
            "random_forest": "random_forest_regressor",
            "gradient_boosting": "gradient_boosting_regressor",
            "xgboost": "xgboost_regressor",
            "lightgbm": "lightgbm_regressor",
            "logistic_regression": "ridge_regression",
        }
        return remap.get(name, name)
    return name


def make_pipeline(
    model_name: str,
    preprocessor: ColumnTransformer,
    task_type: str,
    random_state: int = 42,
) -> Pipeline:
    """Return an unfitted Pipeline(preprocessor + model)."""
    resolved = _resolve_model_name(model_name, task_type)
    estimator = make_estimator(resolved, random_state=random_state)
    return Pipeline([("preprocessor", preprocessor), ("model", estimator)])


def _primary_scorer(task_type: str) -> tuple[str, str]:
    """Return (sklearn_metric_key, display_name) for the primary metric."""
    if task_type == "binary_classification":
        return "roc_auc", "auc"
    if task_type == "multiclass":
        return "roc_auc_ovr_weighted", "macro_auc"
    return "neg_root_mean_squared_error", "rmse"


def _scoring_dict(task_type: str) -> dict[str, Any]:
    from sklearn.metrics import make_scorer
    if task_type == "binary_classification":
        return {
            "roc_auc": "roc_auc",
            "f1": make_scorer(f1_score, zero_division=0),
            "accuracy": "accuracy",
        }
    if task_type == "multiclass":
        return {
            "roc_auc_ovr_weighted": "roc_auc_ovr_weighted",
            "f1_macro": make_scorer(f1_score, average="macro", zero_division=0),
            "accuracy": "accuracy",
        }
    return {
        "neg_root_mean_squared_error": "neg_root_mean_squared_error",
        "neg_mean_absolute_error": "neg_mean_absolute_error",
        "r2": "r2",
    }


def _make_cv_splitter(
    task_type: str, n_splits: int, random_state: int, grouped: bool = False
) -> Any:
    is_clf = task_type in ("binary_classification", "multiclass")
    if grouped:
        # Group-aware CV: all rows sharing a group (e.g. patient_id, isolate_id)
        # stay on the same side of every split, so repeated measurements of the
        # same entity cannot leak across train/test. This is the single most
        # common silent leak in medical datasets with multiple records per patient.
        if is_clf:
            return StratifiedGroupKFold(
                n_splits=n_splits, shuffle=True, random_state=random_state
            )
        return GroupKFold(n_splits=n_splits)
    if is_clf:
        return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    return KFold(n_splits=n_splits, shuffle=True, random_state=random_state)


def cross_validate_candidate(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    task_type: str,
    n_splits: int = 5,
    random_state: int = 42,
    groups: Any = None,
) -> CVResult:
    """Run k-fold CV on one candidate pipeline with one seed.

    If `groups` is provided, uses group-aware CV so rows sharing a group never
    split across train/test (prevents per-entity leakage).
    """
    primary_key, metric_name = _primary_scorer(task_type)
    scoring = _scoring_dict(task_type)
    cv = _make_cv_splitter(task_type, n_splits, random_state, grouped=groups is not None)

    results = sk_cross_validate(
        pipeline, X, y,
        cv=cv,
        scoring=scoring,
        groups=groups,
        return_train_score=True,
        error_score="raise",
    )

    fold_scores = results[f"test_{primary_key}"].tolist()
    fold_train_scores = results[f"train_{primary_key}"].tolist()

    # RMSE is negative in sklearn - flip sign for reporting
    if "rmse" in metric_name:
        fold_scores = [-s for s in fold_scores]
        fold_train_scores = [-s for s in fold_train_scores]

    mean_score = float(np.mean(fold_scores))
    std_score = float(np.std(fold_scores))

    model_name = pipeline.named_steps["model"].__class__.__name__

    return CVResult(
        model_name=model_name,
        seed=random_state,
        fold_scores=fold_scores,
        fold_train_scores=fold_train_scores,
        mean_score=mean_score,
        std_score=std_score,
        metric=metric_name,
    )


def _run_single_seed(
    model_name: str,
    preprocessor_factory: Any,
    X: pd.DataFrame,
    y: pd.Series,
    task_type: str,
    n_splits: int,
    seed: int,
    groups: Any = None,
) -> CVResult:
    """Run k-fold CV for one seed. Called via joblib.Parallel."""
    import copy
    prep_copy = copy.deepcopy(preprocessor_factory)
    pipeline = make_pipeline(model_name, prep_copy, task_type, random_state=seed)
    return cross_validate_candidate(pipeline, X, y, task_type, n_splits, seed, groups=groups)


def train_with_stability(
    model_name: str,
    preprocessor: ColumnTransformer,
    X: pd.DataFrame,
    y: pd.Series,
    task_type: str,
    n_seeds: int = 3,
    n_splits: int = 5,
    groups: Any = None,
) -> StabilityResult:
    """Run n_seeds × n_splits CV and return mean±std across ALL runs.

    If `groups` is provided, CV is group-aware (rows sharing a group never split
    across train/test) - prevents per-entity (e.g. per-patient) leakage.
    """
    seeds = STABILITY_SEEDS[:n_seeds]

    # n_jobs=1: Celery uses prefork workers; Loky cannot spawn child processes
    # inside a forked process and silently falls back to 1 anyway. Setting it
    # explicitly avoids the noisy UserWarning on every fold.
    cv_results: list[CVResult] = Parallel(n_jobs=1)(
        delayed(_run_single_seed)(model_name, preprocessor, X, y, task_type, n_splits, seed, groups)
        for seed in seeds
    )  # type: ignore[assignment]

    all_scores = [s for r in cv_results for s in r.fold_scores]
    all_train_scores = [s for r in cv_results for s in r.fold_train_scores]

    mean = float(np.mean(all_scores))
    std = float(np.std(all_scores))
    train_mean = float(np.mean(all_train_scores)) if all_train_scores else 0.0
    overfit_gap = max(0.0, train_mean - mean)

    return StabilityResult(
        model_name=model_name,
        scores=all_scores,
        mean=mean,
        std=std,
        train_scores=all_train_scores,
        train_mean=train_mean,
        overfit_gap=overfit_gap,
    )


def train_all_candidates(
    candidates: list[str],
    preprocessor: ColumnTransformer,
    X: pd.DataFrame,
    y: pd.Series,
    task_type: str,
    n_seeds: int = 3,
    n_splits: int = 5,
    groups: Any = None,
) -> list[StabilityResult]:
    """Evaluate all candidates and return sorted by mean score (best first).

    `groups`, if provided, enables group-aware CV across all candidates.
    """
    results: list[StabilityResult] = []
    for name in candidates:
        logger.info("Evaluating %s (%d seeds × %d folds)", name, n_seeds, n_splits)
        result = train_with_stability(
            name, preprocessor, X, y, task_type, n_seeds, n_splits, groups=groups
        )
        results.append(result)

    results.sort(key=lambda r: r.mean, reverse=True)
    return results


def maybe_run_stat_test(
    top_two: list[StabilityResult],
    task_type: str,
) -> dict[str, Any] | None:
    """Run McNemar or paired-t when top-2 scores are within CLOSE_THRESHOLD.

    Returns a stat test result dict or None if models are clearly separated.
    """
    if len(top_two) < 2:
        return None

    a, b = top_two[0], top_two[1]
    gap = abs(a.mean - b.mean)

    if gap > CLOSE_THRESHOLD:
        return None

    logger.info(
        "Top-2 within %.4f (%s=%.4f vs %s=%.4f) - running stat test",
        gap, a.model_name, a.mean, b.model_name, b.mean,
    )

    result = run_comparison_test(
        task_type=task_type,
        scores_a=a.scores,
        scores_b=b.scores,
    )
    result["model_a"] = a.model_name
    result["model_b"] = b.model_name
    result["score_gap"] = round(gap, 6)
    return result


def fit_final_pipeline(
    model_name: str,
    preprocessor: ColumnTransformer,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    task_type: str,
    random_state: int = 42,
) -> Pipeline:
    """Fit the final pipeline (preprocessor + model) on the full training set."""
    import copy
    prep = copy.deepcopy(preprocessor)
    pipeline = make_pipeline(model_name, prep, task_type, random_state=random_state)
    pipeline.fit(X_train, y_train)
    return pipeline
