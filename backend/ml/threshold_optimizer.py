"""Business cost-matrix threshold optimization for binary classification (§16).

KEY CONSTRAINT (per spec and project rule 11):
  1. Calibration runs FIRST on X_cal (held-out from model fitting).
  2. Threshold optimization runs on calibrated probabilities from X_val
     (a second held-out slice, never used for fitting or calibration).
  3. The test set (X_test / y_test) STAYS SEALED - it is never passed here.

Default threshold is NOT 0.5. This module finds the threshold that minimizes
total business cost given the cost matrix:
    cost(threshold) = FP * cost_fp + FN * cost_fn

The cost curve is stored for plotting and audit.
"""

from __future__ import annotations

import logging

import numpy as np

from backend.models.strategy import CostMatrix, ThresholdResult

logger = logging.getLogger(__name__)

_DEFAULT_STEPS = 999


def optimize_threshold(
    y_val: np.ndarray,
    y_proba_val: np.ndarray,
    cost_matrix: CostMatrix,
    n_steps: int = _DEFAULT_STEPS,
) -> ThresholdResult:
    """Find the threshold minimizing total cost on the VALIDATION fold.

    Parameters
    ----------
    y_val        : ground truth labels from the validation fold (NOT test set)
    y_proba_val  : calibrated probabilities from the validation fold
    cost_matrix  : business cost matrix (cost_fp, cost_fn, cost_tp, cost_tn)
    n_steps      : number of threshold candidates (default 999 → step ≈ 0.001)

    Returns ThresholdResult with optimal_threshold, cost curve, and improvement
    over the default threshold of 0.5.
    """
    y_val = np.asarray(y_val)
    y_proba_val = np.asarray(y_proba_val)

    thresholds = np.linspace(0.001, 0.999, n_steps)
    cost_curve: list[dict[str, float]] = []
    best_threshold = 0.5
    best_cost = float("inf")

    cost_at_default = _compute_cost(y_val, y_proba_val, 0.5, cost_matrix)

    for t in thresholds:
        cost = _compute_cost(y_val, y_proba_val, float(t), cost_matrix)
        metrics = _compute_metrics(y_val, y_proba_val >= t)
        cost_curve.append({
            "threshold": round(float(t), 4),
            "cost": round(cost, 4),
            "tp": float(metrics["tp"]),
            "fp": float(metrics["fp"]),
            "tn": float(metrics["tn"]),
            "fn": float(metrics["fn"]),
            "precision": round(metrics["precision"], 4),
            "recall": round(metrics["recall"], 4),
            "f1": round(metrics["f1"], 4),
        })
        if cost < best_cost:
            best_cost = cost
            best_threshold = float(t)

    improvement_pct = (
        100.0 * (cost_at_default - best_cost) / cost_at_default
        if cost_at_default > 0
        else 0.0
    )

    metrics_at_optimal = _compute_metrics(y_val, y_proba_val >= best_threshold)

    note = ""
    if abs(best_threshold - 0.5) < 0.01:
        note = "Optimal threshold is near default (0.5); cost matrix may not distinguish strongly."
    elif improvement_pct > 20:
        note = f"Threshold optimization delivers {improvement_pct:.1f}% cost reduction vs. default 0.5."

    logger.info(
        "Threshold optimization: optimal=%.3f cost=%.2f (default 0.5: %.2f, improvement %.1f%%)",
        best_threshold, best_cost, cost_at_default, improvement_pct,
    )

    return ThresholdResult(
        optimal_threshold=round(best_threshold, 4),
        cost_at_default=round(cost_at_default, 4),
        cost_at_optimal=round(best_cost, 4),
        improvement_pct=round(improvement_pct, 2),
        metric_at_optimal={k: round(v, 4) for k, v in metrics_at_optimal.items()},
        cost_curve=cost_curve,
        note=note,
    )


def _compute_cost(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float,
    cm: CostMatrix,
) -> float:
    y_pred = (y_proba >= threshold).astype(int)
    metrics = _compute_metrics(y_true, y_pred)
    return (
        metrics["tp"] * cm.cost_tp
        + metrics["fp"] * cm.cost_fp
        + metrics["tn"] * cm.cost_tn
        + metrics["fn"] * cm.cost_fn
    )


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    tp = float(np.sum((y_pred == 1) & (y_true == 1)))
    fp = float(np.sum((y_pred == 1) & (y_true == 0)))
    tn = float(np.sum((y_pred == 0) & (y_true == 0)))
    fn = float(np.sum((y_pred == 0) & (y_true == 1)))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "precision": precision, "recall": recall, "f1": f1,
    }
