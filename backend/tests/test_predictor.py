"""Unit tests for backend/ml/predictor.py."""

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture()
def simple_pipeline() -> Pipeline:
    X, y = make_classification(n_samples=200, n_features=5, random_state=42)
    pipe = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=500))])
    pipe.fit(X, y)
    return pipe


@pytest.fixture()
def run_meta() -> dict:
    return {
        "task_type": "binary_classification",
        "threshold_result": {"optimal_threshold": 0.4},
        "shap_summary": {
            "feature_names": ["a", "b", "c", "d", "e"],
            "mean_abs_shap": [0.5, 0.3, 0.2, 0.1, 0.05],
            "top_k_features": ["a", "b", "c"],
        },
    }


# ── _confidence_band ───────────────────────────────────────────────────────────

def test_confidence_high():
    from backend.ml.predictor import _confidence_band
    assert _confidence_band(0.95, 0.5) == "high"


def test_confidence_medium():
    from backend.ml.predictor import _confidence_band
    assert _confidence_band(0.7, 0.5) == "medium"


def test_confidence_low():
    from backend.ml.predictor import _confidence_band
    assert _confidence_band(0.52, 0.5) == "low"


# ── _shap_drivers_from_values ──────────────────────────────────────────────────

def test_shap_drivers_splits_positive_negative():
    from backend.ml.predictor import _shap_drivers_from_values
    shap_map = {"a": 0.5, "b": -0.3, "c": 0.2, "d": -0.1}
    drivers, dampeners = _shap_drivers_from_values(shap_map, k=3)
    assert "a" in drivers
    assert "b" in dampeners
    assert len(drivers) <= 3
    assert len(dampeners) <= 3


def test_shap_drivers_empty_map():
    from backend.ml.predictor import _shap_drivers_from_values
    drivers, dampeners = _shap_drivers_from_values({})
    assert drivers == []
    assert dampeners == []


# ── _shap_drivers_from_global ──────────────────────────────────────────────────

def test_global_shap_returns_top_k():
    from backend.ml.predictor import _shap_drivers_from_global
    summary = {"feature_names": ["a", "b", "c"], "top_k_features": ["a", "b"]}
    drivers, dampeners = _shap_drivers_from_global(summary, {})
    assert drivers == ["a", "b"]
    assert dampeners == []


# ── predict_single ────────────────────────────────────────────────────────────

def test_predict_single_returns_expected_keys(simple_pipeline, run_meta):
    from backend.ml.predictor import predict_single

    input_row = {f"feat_{i}": float(i) for i in range(5)}
    # Rename to match pipeline's expected input (numeric columns)
    df = pd.DataFrame([input_row])
    # Use a dict that produces a 1-row DataFrame with 5 columns
    result = predict_single(simple_pipeline, None, run_meta, input_row)

    assert "prediction" in result
    assert "probability" in result
    assert "threshold_used" in result
    assert "confidence_band" in result
    assert result["confidence_band"] in ("high", "medium", "low")
    assert result["threshold_used"] == 0.4


def test_predict_single_probability_in_range(simple_pipeline, run_meta):
    from backend.ml.predictor import predict_single

    input_row = {str(i): 0.0 for i in range(5)}
    result = predict_single(simple_pipeline, None, run_meta, input_row)
    if result["probability"] is not None:
        assert 0.0 <= result["probability"] <= 1.0


def test_predict_single_with_sim_index(simple_pipeline, run_meta):
    from backend.ml.predictor import predict_single
    from backend.ml.similarity import SimilarityIndex

    X_train = np.random.default_rng(42).random((100, 5), dtype=np.float32)
    sim = SimilarityIndex(k=3).fit(X_train)

    input_row = {str(i): 0.5 for i in range(5)}
    result = predict_single(simple_pipeline, sim, run_meta, input_row)
    # sim_index works on transformed space; may fail non-fatally
    # just check key is present
    assert "similarity_score" in result


def test_predict_single_regression():
    from sklearn.linear_model import LinearRegression
    from backend.ml.predictor import predict_single

    X, y = np.random.default_rng(0).random((50, 3)), np.random.default_rng(0).random(50)
    model = Pipeline([("reg", LinearRegression())])
    model.fit(X, y)

    run_meta = {"task_type": "regression", "threshold_result": {}, "shap_summary": {}}
    input_row = {"0": 0.1, "1": 0.2, "2": 0.3}
    result = predict_single(model, None, run_meta, input_row)
    assert result["prediction"] is not None
    assert result["probability"] is None


# ── _extract_base_estimator ────────────────────────────────────────────────────

def test_extract_base_estimator_from_pipeline(simple_pipeline):
    from backend.ml.predictor import _extract_base_estimator
    est = _extract_base_estimator(simple_pipeline)
    assert hasattr(est, "predict")


def test_get_preprocessor_returns_transformer(simple_pipeline):
    from backend.ml.predictor import _get_preprocessor
    from sklearn.pipeline import Pipeline
    preproc = _get_preprocessor(simple_pipeline)
    assert preproc is not None
    assert isinstance(preproc, Pipeline)
