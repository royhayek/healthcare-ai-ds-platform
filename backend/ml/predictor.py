"""Inference engine for trained models (§26 prediction endpoint).

Loads the calibrated sklearn Pipeline (model.joblib) and optional
SimilarityIndex, applies the run's optimal threshold, and computes
per-row SHAP contributions.

SHAP strategy for interactive prediction:
  1. Try TreeExplainer / LinearExplainer on the underlying estimator after
     transforming the input through the preprocessor steps.
  2. Fall back to KernelExplainer with a zero-vector background (1 sample) if
     the estimator type is not recognised.
  3. If shap is not installed or computation fails, return top-k features from
     the stored global shap_summary ordered by the run's mean_abs_shap -
     still useful for the user even though it is not row-specific.

Per-row SHAP is capped at a single row so latency stays low.
"""

from __future__ import annotations

import io
import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_CONFIDENCE_THRESHOLDS = {"high": 0.30, "low": 0.10}

# Models that have native SHAP TreeExplainer support
_TREE_MODELS = {
    "XGBClassifier", "XGBRegressor",
    "LGBMClassifier", "LGBMRegressor",
    "RandomForestClassifier", "RandomForestRegressor",
    "GradientBoostingClassifier", "GradientBoostingRegressor",
}
_LINEAR_MODELS = {
    "LogisticRegression", "Ridge", "LinearRegression",
    "Lasso", "ElasticNet",
}


def _get_expected_columns(pipeline: Any) -> list[str]:
    """Return the column names the ColumnTransformer was fitted on, or []."""
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline

    obj = pipeline
    # Unwrap _CalibratedPipeline
    try:
        obj = object.__getattribute__(obj, "base")
    except AttributeError:
        pass
    if not isinstance(obj, Pipeline):
        return []
    for _, step in obj.steps:
        if isinstance(step, ColumnTransformer):
            if hasattr(step, "feature_names_in_"):
                return list(step.feature_names_in_)
            # Fall back to transformers_ for older sklearn
            cols: list[str] = []
            for _, _, c in getattr(step, "transformers_", []):
                if isinstance(c, list):
                    cols.extend(c)
            return cols
    return []


def _fill_missing_columns(df: pd.DataFrame, pipeline: Any) -> pd.DataFrame:
    """Add any columns the pipeline expects but the input omits, filled with NaN.

    The pipeline's own imputer will handle NaN - this just satisfies the
    ColumnTransformer's column-presence check so partial-feature payloads work.
    """
    expected = _get_expected_columns(pipeline)
    if not expected:
        return df
    missing = [c for c in expected if c not in df.columns]
    if missing:
        df = df.copy()
        for col in missing:
            df[col] = np.nan
    return df


def _confidence_band(prob: float, threshold: float) -> str:
    margin = abs(prob - threshold)
    if margin >= _CONFIDENCE_THRESHOLDS["high"]:
        return "high"
    if margin >= _CONFIDENCE_THRESHOLDS["low"]:
        return "medium"
    return "low"


def _extract_base_estimator(pipeline: Any) -> Any:
    """Walk a fitted sklearn Pipeline / CalibratedClassifierCV to reach the base estimator."""
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.pipeline import Pipeline

    obj = pipeline
    # Unwrap Pipeline (last step is the estimator)
    if isinstance(obj, Pipeline):
        obj = obj.steps[-1][1]
    # Unwrap CalibratedClassifierCV
    if isinstance(obj, CalibratedClassifierCV):
        if obj.calibrated_classifiers_:
            inner = obj.calibrated_classifiers_[0].estimator
            if isinstance(inner, Pipeline):
                inner = inner.steps[-1][1]
            return inner
    return obj


def _get_preprocessor(pipeline: Any) -> Any | None:
    """Return the preprocessing step(s) as a transformer, or None."""
    from sklearn.pipeline import Pipeline
    if not isinstance(pipeline, Pipeline):
        return None
    if len(pipeline.steps) < 2:
        return None
    # Slice everything except the last step
    from sklearn.pipeline import Pipeline as SKPipeline
    preproc_steps = pipeline.steps[:-1]
    if not preproc_steps:
        return None
    return SKPipeline(preproc_steps)


def _compute_row_shap(
    pipeline: Any,
    df_row: pd.DataFrame,
    feature_names: list[str],
    task_type: str,
) -> dict[str, float] | None:
    """Return {feature_name: shap_value} for a single row, or None on failure."""
    try:
        import shap
    except ImportError:
        return None

    try:
        preproc = _get_preprocessor(pipeline)
        estimator = _extract_base_estimator(pipeline)

        if preproc is not None:
            X_transformed = preproc.transform(df_row)
        else:
            X_transformed = df_row.to_numpy()

        X_arr = np.asarray(X_transformed, dtype=np.float32)
        est_type = type(estimator).__name__

        if est_type in _TREE_MODELS:
            explainer = shap.TreeExplainer(estimator)
            sv = explainer.shap_values(X_arr)
        elif est_type in _LINEAR_MODELS:
            explainer = shap.LinearExplainer(estimator, X_arr)
            sv = explainer.shap_values(X_arr)
        else:
            background = np.zeros((1, X_arr.shape[1]), dtype=np.float32)
            explainer = shap.KernelExplainer(estimator.predict_proba, background)
            sv = explainer.shap_values(X_arr, nsamples=50)

        # sv may be list (binary: [class0, class1]) or ndarray
        if isinstance(sv, list):
            sv = sv[1] if len(sv) > 1 else sv[0]

        row_sv = np.asarray(sv).flatten()

        # Feature names from the transformed space; fall back to indices
        try:
            if preproc is not None and hasattr(preproc, "get_feature_names_out"):
                transformed_names = list(preproc.get_feature_names_out())
            else:
                transformed_names = [f"f{i}" for i in range(len(row_sv))]
        except Exception:
            transformed_names = [f"f{i}" for i in range(len(row_sv))]

        n = min(len(transformed_names), len(row_sv))
        return {transformed_names[i]: float(row_sv[i]) for i in range(n)}

    except Exception as exc:
        logger.debug("Per-row SHAP failed (non-fatal): %s", exc)
        return None


def _shap_drivers_from_values(
    shap_map: dict[str, float], k: int = 3
) -> tuple[list[str], list[str]]:
    """Split SHAP map into top-k positive drivers and top-k negative dampeners."""
    sorted_items = sorted(shap_map.items(), key=lambda x: x[1], reverse=True)
    drivers = [name for name, v in sorted_items if v > 0][:k]
    dampeners = [name for name, v in sorted_items if v < 0][:k]
    return drivers, dampeners


def _shap_drivers_from_global(
    shap_summary: dict[str, Any], input_row: dict[str, Any], k: int = 3
) -> tuple[list[str], list[str]]:
    """Fall back: rank by global importance, split by whether row value is above median."""
    feature_names = shap_summary.get("feature_names", [])
    mean_abs = shap_summary.get("mean_abs_shap", [])
    top_k = shap_summary.get("top_k_features", feature_names[:k])[:k]
    # Return top-k as drivers; can't determine dampeners without per-row values
    return list(top_k), []


def _confidence_from_similarity(similarity: float | None) -> str:
    """Confidence band for regression tasks - uses similarity as a proxy."""
    if similarity is None:
        return "low"
    if similarity >= 0.70:
        return "high"
    if similarity >= 0.40:
        return "medium"
    return "low"


def predict_single(
    pipeline: Any,
    sim_index: Any | None,
    run_meta: dict[str, Any],
    input_data: dict[str, Any],
) -> dict[str, Any]:
    """Run prediction for a single input row.

    Parameters
    ----------
    pipeline   : fitted sklearn Pipeline loaded from model.joblib
    sim_index  : optional SimilarityIndex (None if not built)
    run_meta   : dict with threshold_result, shap_summary, task_type keys
    input_data : dict of {column_name: value} from the user

    Returns dict matching PredictionResult schema.
    """
    import warnings

    df = pd.DataFrame([input_data])
    df = _fill_missing_columns(df, pipeline)

    task_type: str = run_meta.get("task_type", "binary_classification")
    threshold_result = run_meta.get("threshold_result") or {}
    threshold = float(threshold_result.get("optimal_threshold", 0.5))

    # ── Probability / prediction ───────────────────────────────────────────────
    # Suppress sklearn feature-name warnings: models trained before the
    # set_output("default") fix may have been fitted with DataFrames but
    # now receive numpy from the ColumnTransformer in a different process.
    probability: float | None = None
    prediction: int | float | str | None = None

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=".*does not have valid feature names.*",
            category=UserWarning,
        )
        if task_type in ("binary_classification", "multiclass"):
            try:
                proba = pipeline.predict_proba(df)
                if task_type == "binary_classification":
                    probability = float(proba[0, 1])
                    prediction = int(probability >= threshold)
                else:
                    prediction = int(np.argmax(proba[0]))
                    probability = float(np.max(proba[0]))
            except AttributeError:
                # Model doesn't support predict_proba - fall back to predict.
                raw = pipeline.predict(df)
                prediction = int(raw[0]) if hasattr(raw[0], "__int__") else str(raw[0])
        else:
            raw = pipeline.predict(df)
            prediction = float(raw[0])

    # ── Similarity score ───────────────────────────────────────────────────────
    # Computed before confidence so regression tasks can use it as a proxy.
    similarity_score: float | None = None
    if sim_index is not None:
        try:
            preproc = _get_preprocessor(pipeline)
            if preproc is not None:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=UserWarning)
                    X_t = np.asarray(preproc.transform(df), dtype=np.float32)
            else:
                X_t = df.to_numpy(dtype=np.float32)
            scores = sim_index.score(X_t)
            similarity_score = float(scores[0])
        except Exception as exc:
            logger.warning("Similarity score failed (non-fatal): %s", exc)

    # ── Confidence band ────────────────────────────────────────────────────────
    # Classification: margin from threshold → high/medium/low.
    # Regression: no probability, so use similarity as proxy; low if unavailable.
    if probability is not None:
        confidence = _confidence_band(probability, threshold)
    elif task_type == "regression":
        confidence = _confidence_from_similarity(similarity_score)
    else:
        confidence = "low"

    # ── SHAP ───────────────────────────────────────────────────────────────────
    feature_names = list(run_meta.get("shap_summary", {}).get("feature_names", []))
    shap_drivers: list[str] = []
    shap_dampeners: list[str] = []

    shap_map = _compute_row_shap(pipeline, df, feature_names, task_type)
    if shap_map is not None:
        shap_drivers, shap_dampeners = _shap_drivers_from_values(shap_map)
    elif run_meta.get("shap_summary"):
        shap_drivers, shap_dampeners = _shap_drivers_from_global(
            run_meta["shap_summary"], input_data
        )

    return {
        "prediction": prediction,
        "probability": probability,
        "threshold_used": threshold,
        "confidence_band": confidence,
        "similarity_score": similarity_score,
        "shap_drivers": shap_drivers,
        "shap_dampeners": shap_dampeners,
        "task_type": task_type,
    }


async def load_model_artifacts(run: Any, storage: Any) -> tuple[Any, Any | None]:
    """Load fitted pipeline and optional SimilarityIndex from storage.

    Returns (pipeline, sim_index_or_None).
    Raises FileNotFoundError if model artifact is missing.
    """
    import joblib

    from backend.ml.similarity import SimilarityIndex

    if not run.model_storage_path:
        raise FileNotFoundError(f"Run {run.id} has no model artifact (not yet trained).")

    model_bytes = await storage.download(run.model_storage_path)
    pipeline = joblib.load(io.BytesIO(model_bytes))

    sim_index: SimilarityIndex | None = None
    if run.faiss_index_path:
        try:
            sim_bytes = await storage.download(run.faiss_index_path)
            sim_index = SimilarityIndex.deserialize(sim_bytes)
        except Exception as exc:
            logger.warning("Could not load similarity index for run %s: %s", run.id, exc)

    return pipeline, sim_index
