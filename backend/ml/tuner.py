"""Optuna Bayesian hyperparameter tuning (§15).

Per-model search spaces cover all 7 supported model types.
Uses TPESampler. The fitted pipeline is returned - caller is responsible
for calibration and threshold optimization steps that follow (§16).
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score
from sklearn.pipeline import Pipeline

from backend.ml.trainer import fit_final_pipeline, make_pipeline, _primary_scorer
from backend.models.strategy import TuningResult

logger = logging.getLogger(__name__)

_DEFAULT_N_TRIALS = 30
_DEFAULT_TIMEOUT = 300  # seconds


def _suggest_params(trial: Any, model_name: str) -> dict[str, Any]:
    """Suggest hyperparameters for `model_name` using the Optuna trial."""
    if model_name in ("logistic_regression",):
        return {
            "model__C": trial.suggest_float("C", 1e-4, 100.0, log=True),
            "model__solver": trial.suggest_categorical("solver", ["lbfgs", "saga"]),
        }

    if model_name in ("random_forest", "random_forest_regressor"):
        return {
            "model__n_estimators": trial.suggest_int("n_estimators", 50, 500),
            "model__max_depth": trial.suggest_int("max_depth", 3, 20),
            "model__min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "model__min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "model__max_features": trial.suggest_categorical("max_features", ["sqrt", "log2"]),
        }

    if model_name in ("gradient_boosting", "gradient_boosting_regressor"):
        return {
            "model__n_estimators": trial.suggest_int("n_estimators", 50, 500),
            "model__learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.5, log=True),
            "model__max_depth": trial.suggest_int("max_depth", 2, 8),
            "model__subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "model__min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        }

    if model_name in ("xgboost", "xgboost_regressor"):
        return {
            "model__n_estimators": trial.suggest_int("n_estimators", 50, 500),
            "model__learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "model__max_depth": trial.suggest_int("max_depth", 2, 10),
            "model__subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "model__colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "model__reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "model__reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        }

    if model_name in ("lightgbm", "lightgbm_regressor"):
        return {
            "model__n_estimators": trial.suggest_int("n_estimators", 50, 500),
            "model__learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "model__max_depth": trial.suggest_int("max_depth", 2, 10),
            "model__num_leaves": trial.suggest_int("num_leaves", 16, 256),
            "model__subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "model__colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "model__reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "model__reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        }

    if model_name == "ridge_regression":
        return {
            "model__alpha": trial.suggest_float("alpha", 1e-4, 1000.0, log=True),
        }

    if model_name == "linear_regression":
        return {}  # No hyperparameters to tune

    logger.warning("No search space for %s - using baseline params", model_name)
    return {}


def tune_model(
    model_name: str,
    preprocessor: ColumnTransformer,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    task_type: str,
    baseline_score: float,
    n_trials: int = _DEFAULT_N_TRIALS,
    timeout: int = _DEFAULT_TIMEOUT,
    random_state: int = 42,
) -> tuple[Pipeline, TuningResult]:
    """Run Optuna tuning. Returns (fitted_pipeline_with_best_params, TuningResult).

    The returned pipeline is already fitted on the full X_train with best params.
    Caller should then split off X_cal and X_val for calibration + threshold.
    """
    import copy

    import optuna
    from sklearn.model_selection import StratifiedKFold, KFold

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    from backend.ml.trainer import _resolve_model_name, _primary_scorer
    resolved_name = _resolve_model_name(model_name, task_type)
    primary_key, metric_name = _primary_scorer(task_type)

    n_splits = 3  # faster CV during tuning

    if task_type in ("binary_classification", "multiclass"):
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    else:
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    def objective(trial: Any) -> float:
        params = _suggest_params(trial, resolved_name)
        prep_copy = copy.deepcopy(preprocessor)
        pipeline = make_pipeline(model_name, prep_copy, task_type, random_state=random_state)
        pipeline.set_params(**params)

        scores = cross_val_score(
            pipeline, X_train, y_train,
            cv=cv,
            scoring=primary_key,
            error_score=float("-inf"),
        )
        return float(np.mean(scores))

    sampler = optuna.samplers.TPESampler(seed=random_state)
    direction = "minimize" if "rmse" in metric_name else "maximize"
    study = optuna.create_study(direction=direction, sampler=sampler)

    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=False)

    # study.best_params keys match the trial.suggest_*(name) strings, which are
    # bare param names ("n_estimators"). Pipeline.set_params() requires the
    # "stepname__param" form ("model__n_estimators"). Re-prefix all keys.
    best_params = {f"model__{k}": v for k, v in study.best_params.items()}
    best_score = study.best_value
    if "rmse" in metric_name:
        best_score = -best_score  # convert back to positive RMSE for reporting

    improvement = abs(best_score - baseline_score)

    logger.info(
        "Tuning complete: %s best_%s=%.4f (baseline=%.4f, delta=%.4f) after %d trials",
        model_name, metric_name, best_score, baseline_score, improvement, n_trials,
    )

    # Fit final pipeline with best params on full X_train
    tuning_result = TuningResult(
        model_name=model_name,
        best_params=study.best_params,  # bare names for human readability in the audit log
        best_score=round(best_score, 6),
        n_trials=len(study.trials),
        metric=metric_name,
        improvement_over_baseline=round(improvement, 6),
    )

    final_pipeline = fit_final_pipeline(model_name, preprocessor, X_train, y_train, task_type)
    # Apply best params before fitting
    import copy as _copy
    final_prep = _copy.deepcopy(preprocessor)
    final_pipeline = make_pipeline(model_name, final_prep, task_type, random_state=random_state)
    final_pipeline.set_params(**best_params)
    final_pipeline.fit(X_train, y_train)

    return final_pipeline, tuning_result
