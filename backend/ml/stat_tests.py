"""Statistical significance tests for model comparison (§14).

McNemar's test: binary classification - compares two models on the same test set.
Paired t-test: regression - compares CV score distributions of two models.

Both tests are used in trainer.py when the top-two candidates are within 0.005
on the primary metric. The p-value is surfaced in the chat context block so the
user can make an informed decision about which model to select.
"""

from __future__ import annotations

import logging

import numpy as np
from scipy.stats import ttest_rel

logger = logging.getLogger(__name__)


def mcnemar_test(
    y_true: np.ndarray,
    y_pred_a: np.ndarray,
    y_pred_b: np.ndarray,
) -> dict[str, float]:
    """McNemar's test for the difference between two classifiers.

    Uses the continuity-corrected version (Fleiss, Levin, Paik 2003):
        χ² = (|b - c| - 1)² / (b + c)  where b+c > 0, else p = 1.0

    Parameters
    ----------
    y_true:   ground truth labels (0/1)
    y_pred_a: predictions from model A (0/1)
    y_pred_b: predictions from model B (0/1)

    Returns {statistic, p_value, b, c} where:
      b = cases correct by A, wrong by B
      c = cases wrong by A, correct by B
    """
    from scipy.stats import chi2

    y_true = np.asarray(y_true)
    y_pred_a = np.asarray(y_pred_a)
    y_pred_b = np.asarray(y_pred_b)

    correct_a = y_pred_a == y_true
    correct_b = y_pred_b == y_true

    b = int(np.sum(correct_a & ~correct_b))   # A correct, B wrong
    c = int(np.sum(~correct_a & correct_b))   # A wrong, B correct

    if b + c == 0:
        logger.warning("McNemar: b+c=0, models are identical on all discordant pairs")
        return {"statistic": 0.0, "p_value": 1.0, "b": b, "c": c}

    # Continuity correction
    statistic = (abs(b - c) - 1) ** 2 / (b + c)
    p_value = float(1.0 - chi2.cdf(statistic, df=1))

    return {"statistic": float(statistic), "p_value": p_value, "b": b, "c": c}


def paired_t_test(
    scores_a: list[float],
    scores_b: list[float],
) -> dict[str, float]:
    """Paired t-test for the difference between two sets of CV scores.

    scores_a and scores_b must have the same length (matched folds × seeds).

    Returns {statistic, p_value, mean_diff, std_diff}.
    """
    a = np.asarray(scores_a, dtype=float)
    b = np.asarray(scores_b, dtype=float)

    if len(a) != len(b):
        raise ValueError(f"paired_t_test: length mismatch {len(a)} vs {len(b)}")

    if len(a) < 2:
        return {"statistic": 0.0, "p_value": 1.0, "mean_diff": 0.0, "std_diff": 0.0}

    diff = a - b
    if np.std(diff) == 0.0:
        # Degenerate case: all differences are identical - models agree everywhere
        return {"statistic": 0.0, "p_value": 1.0, "mean_diff": float(diff.mean()), "std_diff": 0.0}

    statistic, p_value = ttest_rel(a, b)
    diff = a - b

    return {
        "statistic": float(statistic),
        "p_value": float(p_value),
        "mean_diff": float(diff.mean()),
        "std_diff": float(diff.std()),
    }


def run_comparison_test(
    task_type: str,
    scores_a: list[float],
    scores_b: list[float],
    *,
    y_true: np.ndarray | None = None,
    y_pred_a: np.ndarray | None = None,
    y_pred_b: np.ndarray | None = None,
) -> dict[str, object]:
    """Run the appropriate test given the task type.

    For binary_classification: prefer McNemar (needs predictions) with
    paired-t fallback when predictions are unavailable.
    For everything else: paired-t.

    Returns a dict with 'test_name', 'p_value', 'statistic', and
    a human-readable 'interpretation'.
    """
    result: dict[str, object] = {}

    if (
        task_type == "binary_classification"
        and y_true is not None
        and y_pred_a is not None
        and y_pred_b is not None
    ):
        r = mcnemar_test(y_true, y_pred_a, y_pred_b)
        result = {"test_name": "mcnemar", **r}
    else:
        r = paired_t_test(scores_a, scores_b)
        result = {"test_name": "paired_t", **r}

    p = float(result["p_value"])  # type: ignore[arg-type]
    if p < 0.05:
        result["interpretation"] = f"Significant difference (p={p:.4f}); prefer the higher-scoring model."
    else:
        result["interpretation"] = (
            f"No significant difference (p={p:.4f}); models are statistically equivalent - "
            "prefer simpler or faster one."
        )

    return result
