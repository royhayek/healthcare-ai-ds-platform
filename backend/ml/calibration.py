"""Probability calibration for classification models (§16).

Calibrates a pre-fitted sklearn Pipeline on a held-out calibration fold.
sklearn removed cv="prefit" in 1.6 - we replicate that behavior directly:
  1. Get raw probabilities from the fitted pipeline on X_cal
  2. Fit an IsotonicRegression (n_cal >= 1000) or Platt scaler (sigmoid) on them
  3. Wrap both in a thin _CalibratedPipeline that chains them at predict time

The caller must fit the base model BEFORE passing it here. Calibration is
performed on a held-out calibration fold (X_cal, y_cal) that was never seen
during model fitting - this prevents leakage (§16).
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from sklearn.pipeline import Pipeline

from backend.models.strategy import CalibrationReport

logger = logging.getLogger(__name__)

_ISO_MIN_SAMPLES = 1000


class _CalibratedPipeline:
    """Thin wrapper: base Pipeline → scalar calibrator → calibrated probabilities."""

    def __init__(self, base: Pipeline, calibrator: Any, method: str) -> None:
        self.base = base
        self.calibrator = calibrator
        self.method = method

    def predict_proba(self, X: Any) -> np.ndarray:
        raw = self.base.predict_proba(X)[:, 1]
        if self.method == "isotonic":
            cal = np.clip(self.calibrator.predict(raw), 0.0, 1.0)
        else:
            cal = self.calibrator.predict_proba(raw.reshape(-1, 1))[:, 1]
        return np.column_stack([1.0 - cal, cal])

    def predict(self, X: Any, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)

    # Passthrough for attribute access (e.g. pipeline.steps from predictor).
    # Must guard against infinite recursion during joblib unpickling: when the
    # object is being reconstructed, self.base is not yet in __dict__, so a
    # naive `self.base` access would trigger __getattr__ again → RecursionError.
    def __getattr__(self, name: str) -> Any:
        try:
            base = object.__getattribute__(self, "base")
        except AttributeError:
            raise AttributeError(name)
        return getattr(base, name)


def calibrate_classifier(
    fitted_pipeline: Pipeline,
    X_cal: Any,
    y_cal: Any,
) -> tuple[_CalibratedPipeline, CalibrationReport]:
    """Calibrate a fitted sklearn Pipeline on the calibration fold.

    Parameters
    ----------
    fitted_pipeline : a Pipeline already fit on X_fit. Must expose predict_proba.
    X_cal           : raw (unpreprocessed) calibration features
    y_cal           : calibration labels

    Returns (calibrated_pipeline, CalibrationReport).
    """
    y_cal = np.asarray(y_cal)
    n_cal = len(y_cal)

    # This calibrator is binary-only: it operates on the positive-class column
    # (predict_proba[:, 1]) and uses binary Brier/Platt/Isotonic. Guard against a
    # multiclass target so the failure is explicit rather than sklearn's opaque
    # "target ... is multiclass but should be binary" from brier_score_loss.
    n_classes = int(np.unique(y_cal).size)
    if n_classes > 2:
        raise ValueError(
            f"calibrate_classifier supports binary classification only "
            f"(got {n_classes} classes). Gate calibration to binary tasks."
        )

    method = "isotonic" if n_cal >= _ISO_MIN_SAMPLES else "sigmoid"
    logger.info("Calibrating with %s on %d samples", method, n_cal)

    y_proba_before = fitted_pipeline.predict_proba(X_cal)[:, 1]
    brier_before = float(brier_score_loss(y_cal, y_proba_before))
    ece_before = _compute_ece(y_cal, y_proba_before)

    if method == "isotonic":
        calibrator: Any = IsotonicRegression(out_of_bounds="clip")
        calibrator.fit(y_proba_before, y_cal)
    else:
        calibrator = LogisticRegression(C=1.0, solver="lbfgs")
        calibrator.fit(y_proba_before.reshape(-1, 1), y_cal)

    calibrated = _CalibratedPipeline(fitted_pipeline, calibrator, method)

    y_proba_after = calibrated.predict_proba(X_cal)[:, 1]
    brier_after = float(brier_score_loss(y_cal, y_proba_after))
    ece_after = _compute_ece(y_cal, y_proba_after)

    improvement_pct = (
        100.0 * (brier_before - brier_after) / brier_before if brier_before > 0 else 0.0
    )

    report = CalibrationReport(
        method=method,
        brier_before=brier_before,
        brier_after=brier_after,
        ece_before=ece_before,
        ece_after=ece_after,
        improvement_pct=improvement_pct,
    )

    logger.info(
        "Calibration: Brier %.4f → %.4f (%.1f%% improvement), ECE %.4f → %.4f",
        brier_before, brier_after, improvement_pct, ece_before, ece_after,
    )

    return calibrated, report


def _compute_ece(y_true: np.ndarray, y_proba: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error - weighted mean absolute calibration deviation."""
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    n = len(y_true)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (y_proba >= lo) & (y_proba < hi)
        if not mask.any():
            continue
        ece += (mask.sum() / n) * abs(y_proba[mask].mean() - y_true[mask].mean())
    return float(ece)
