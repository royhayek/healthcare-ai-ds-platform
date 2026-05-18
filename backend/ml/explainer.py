"""SHAP-based feature importance (§18).

Explainer selection:
  - TreeExplainer  → XGBoost, LightGBM, RandomForest, GradientBoosting
  - LinearExplainer → LogisticRegression, Ridge, LinearRegression
  - KernelExplainer → fallback (slow - capped at 100 background samples)

All SHAP is computed on the TEST SET only. The global importance summary
stores mean |SHAP| per feature - the full SHAP matrix is never persisted
(Rule 7: no raw row-level data storage).

MAX_SHAP_SAMPLES = 500 - sampling applied to large test sets to keep
computation time under 60 seconds.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline

from backend.ml.calibration import _CalibratedPipeline

from backend.models.strategy import SHAPSummary

logger = logging.getLogger(__name__)

MAX_SHAP_SAMPLES = 500
MAX_KERNEL_BACKGROUND = 100

_TREE_MODELS = {
    "XGBClassifier", "XGBRegressor",
    "LGBMClassifier", "LGBMRegressor",
    "RandomForestClassifier", "RandomForestRegressor",
    "GradientBoostingClassifier", "GradientBoostingRegressor",
    "ExtraTreesClassifier", "ExtraTreesRegressor",
}

_LINEAR_MODELS = {
    "LogisticRegression", "Ridge", "LinearRegression",
    "Lasso", "ElasticNet", "SGDClassifier", "SGDRegressor",
}


def compute_shap(
    fitted_pipeline: Pipeline | CalibratedClassifierCV,
    X_test: pd.DataFrame | np.ndarray,
    feature_names: list[str],
    task_type: str,
    background_data: pd.DataFrame | np.ndarray | None = None,
) -> SHAPSummary:
    """Compute SHAP values on X_test and return a global importance summary.

    Parameters
    ----------
    fitted_pipeline  : fitted sklearn Pipeline or CalibratedClassifierCV
    X_test           : raw (unpreprocessed) test features - the pipeline handles preprocessing
    feature_names    : original feature column names (before preprocessing)
    task_type        : binary_classification | multiclass | regression
    background_data  : optional background for KernelExplainer fallback

    Returns SHAPSummary with mean |SHAP| per feature (never raw SHAP matrices).
    """
    import shap

    X_test_arr = _to_numpy(X_test)
    n = len(X_test_arr)

    # Sample if large
    if n > MAX_SHAP_SAMPLES:
        rng = np.random.default_rng(42)
        idx = rng.choice(n, size=MAX_SHAP_SAMPLES, replace=False)
        X_test_arr = X_test_arr[idx]
        logger.info("SHAP: sampled %d/%d test rows", MAX_SHAP_SAMPLES, n)
        n = MAX_SHAP_SAMPLES

    # Extract the inner model (unwrap Pipeline + CalibratedClassifierCV)
    inner_model, preprocessor = _unwrap(fitted_pipeline)
    model_class = inner_model.__class__.__name__

    # Transform features
    if preprocessor is not None:
        if isinstance(X_test, pd.DataFrame):
            X_transformed = preprocessor.transform(X_test.iloc[: n] if isinstance(X_test, pd.DataFrame) else X_test[:n])
        else:
            X_transformed = preprocessor.transform(X_test_arr)
    else:
        X_transformed = X_test_arr

    X_transformed = np.asarray(X_transformed, dtype=np.float32)

    # Pick explainer
    if model_class in _TREE_MODELS:
        explainer_type = "tree"
        explainer = shap.TreeExplainer(inner_model)
        shap_values = explainer.shap_values(X_transformed)
    elif model_class in _LINEAR_MODELS:
        explainer_type = "linear"
        if preprocessor is not None and background_data is not None:
            bg = _to_numpy(background_data)[:MAX_KERNEL_BACKGROUND]
            bg_transformed = np.asarray(preprocessor.transform(bg), dtype=np.float32)
        else:
            bg_transformed = X_transformed[:MAX_KERNEL_BACKGROUND]
        explainer = shap.LinearExplainer(inner_model, bg_transformed)
        shap_values = explainer.shap_values(X_transformed)
    else:
        logger.warning("Using KernelExplainer for %s (slow)", model_class)
        explainer_type = "kernel"
        bg = X_transformed[:MAX_KERNEL_BACKGROUND]
        explainer = shap.KernelExplainer(
            inner_model.predict_proba if hasattr(inner_model, "predict_proba") else inner_model.predict,
            bg,
        )
        shap_values = explainer.shap_values(X_transformed, nsamples=100)

    # Compute mean |SHAP| per (preprocessed) feature. SHAP returns different
    # shapes across versions/tasks and this must collapse all of them to a 1D
    # (n_features,) vector - otherwise a per-class axis survives and downstream
    # rounding hits a list ("type list doesn't define __round__").
    mean_abs_shap_transformed = _mean_abs_per_feature(shap_values)

    # Map back to original feature names (best effort - transformed dim may differ)
    transformed_feature_names = _get_transformed_feature_names(preprocessor, feature_names)
    if len(transformed_feature_names) != len(mean_abs_shap_transformed):
        # Fallback: use indices
        logger.warning(
            "SHAP feature name mismatch: %d names vs %d SHAP values",
            len(transformed_feature_names), len(mean_abs_shap_transformed),
        )
        transformed_feature_names = [f"feature_{i}" for i in range(len(mean_abs_shap_transformed))]

    sorted_pairs = sorted(
        zip(transformed_feature_names, mean_abs_shap_transformed.tolist()),
        key=lambda x: x[1],
        reverse=True,
    )
    top_k = [name for name, _ in sorted_pairs[:10]]

    return SHAPSummary(
        feature_names=[p[0] for p in sorted_pairs],
        mean_abs_shap=[round(p[1], 6) for p in sorted_pairs],
        top_k_features=top_k,
        explainer_type=explainer_type,
        n_samples=n,
    )


def _mean_abs_per_feature(shap_values: Any) -> np.ndarray:
    """Reduce any SHAP output to a 1D mean|SHAP| vector of length n_features.

    Handles every shape SHAP emits:
      - list of (n_samples, n_features) arrays - one per class/output (older API)
      - 3D array (n_samples, n_features, n_classes) - multiclass (newer API)
      - 2D array (n_samples, n_features) - binary / regression
    Multiclass importance is the mean of |SHAP| averaged over samples and classes.
    """
    if isinstance(shap_values, list):
        # Stack per-class arrays → (n_classes, n_samples, n_features), average |·|.
        stacked = np.stack([np.abs(np.asarray(a)) for a in shap_values], axis=0)
        return stacked.mean(axis=(0, 1))

    arr = np.abs(np.asarray(shap_values, dtype=np.float64))
    if arr.ndim == 3:
        # (n_samples, n_features, n_classes) → average over samples and classes.
        return arr.mean(axis=(0, 2))
    if arr.ndim == 2:
        return arr.mean(axis=0)
    return np.atleast_1d(arr)


def _to_numpy(X: pd.DataFrame | np.ndarray) -> np.ndarray:
    if isinstance(X, pd.DataFrame):
        return X.values
    return np.asarray(X)


def _unwrap(model: Any) -> tuple[Any, Any]:
    """Return (inner_estimator, preprocessor_or_None)."""
    if isinstance(model, _CalibratedPipeline):
        return _unwrap(model.base)

    if isinstance(model, CalibratedClassifierCV):
        inner = model.estimator
        return _unwrap(inner)

    if isinstance(model, Pipeline):
        preprocessor = model.named_steps.get("preprocessor")
        estimator = model.named_steps.get("model")
        if estimator is None:
            # Last step is the model
            *_, last_step = model.steps
            estimator = last_step[1]
        return estimator, preprocessor

    return model, None


def _get_transformed_feature_names(preprocessor: Any, original_names: list[str]) -> list[str]:
    """Extract feature names from a fitted ColumnTransformer, falling back to originals."""
    if preprocessor is None:
        return original_names

    try:
        names = preprocessor.get_feature_names_out()
        return list(names)
    except Exception:
        return original_names
