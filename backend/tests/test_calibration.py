"""Tests for ml/calibration.py - binary calibration + multiclass guard."""

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from backend.ml.calibration import calibrate_classifier


def _fit_binary_pipeline(n: int = 400, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 4))
    y = (X[:, 0] + rng.normal(scale=0.5, size=n) > 0).astype(int)
    pipe = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression())])
    pipe.fit(X, y)
    return pipe, X, y


def test_calibrate_binary_returns_report_and_proba():
    pipe, X, y = _fit_binary_pipeline()
    calibrated, report = calibrate_classifier(pipe, X, y)
    proba = calibrated.predict_proba(X)
    assert proba.shape == (len(y), 2)
    # Probabilities are valid and rows sum to 1.
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)
    assert report.brier_after >= 0.0
    assert report.method in ("isotonic", "sigmoid")


def test_calibrate_rejects_multiclass():
    # A 3-class target must raise a clear error, not sklearn's opaque
    # "multiclass but should be binary" from brier_score_loss.
    rng = np.random.default_rng(1)
    X = rng.normal(size=(150, 4))
    y = rng.integers(0, 3, size=150)  # 3 classes
    pipe = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression())])
    pipe.fit(X, y)
    with pytest.raises(ValueError, match="binary classification only"):
        calibrate_classifier(pipe, X, y)
