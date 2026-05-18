"""Bias and fairness analysis (§19).

Uses fairlearn for group metrics. Falls back to manual metric computation if
fairlearn is not installed (so the module remains importable in minimal envs).

Severity thresholds follow the EEOC 80% rule and the spec's banding:
  < 5%   → none      (pass silently)
  5-10%  → mild      (note in report)
  10-20% → moderate  (surface in chat, recommend mitigation)
  > 20%  → severe    (block deliverables until acknowledged)
"""

from __future__ import annotations

import itertools
import logging
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class GroupMetrics(BaseModel):
    group: str
    n_samples: int
    selection_rate: float
    true_positive_rate: float | None = None
    false_positive_rate: float | None = None
    precision: float | None = None


class AttributeFairnessResult(BaseModel):
    attribute: str
    demographic_parity_diff: float
    equalized_odds_diff: float | None = None
    equal_opportunity_diff: float | None = None
    by_group: list[GroupMetrics] = Field(default_factory=list)
    severity: str = "none"  # none | mild | moderate | severe
    note: str = ""


class FairnessReport(BaseModel):
    attributes: list[AttributeFairnessResult] = Field(default_factory=list)
    intersectional: list[AttributeFairnessResult] = Field(default_factory=list)
    overall_severity: str = "none"  # none | mild | moderate | severe
    blocks_deliverables: bool = False
    requires_acknowledgment: bool = False
    acknowledged: bool = False


def _classify_severity(disparity: float) -> str:
    abs_disp = abs(disparity)
    if abs_disp < 0.05:
        return "none"
    if abs_disp < 0.10:
        return "mild"
    if abs_disp < 0.20:
        return "moderate"
    return "severe"


def _group_metrics_for_attr(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None,
    groups: np.ndarray,
    is_classification: bool,
) -> list[GroupMetrics]:
    results = []
    for group_val in np.unique(groups):
        mask = groups == group_val
        n = int(mask.sum())
        if n == 0:
            continue

        yt = y_true[mask]
        yp = y_pred[mask]

        selection_rate = float(yp.mean()) if len(yp) > 0 else 0.0

        tpr = None
        fpr = None
        precision = None

        if is_classification and len(np.unique(yt)) >= 2:
            tp = int(((yt == 1) & (yp == 1)).sum())
            fn = int(((yt == 1) & (yp == 0)).sum())
            fp = int(((yt == 0) & (yp == 1)).sum())
            tn = int(((yt == 0) & (yp == 0)).sum())

            tpr = tp / (tp + fn) if (tp + fn) > 0 else None
            fpr = fp / (fp + tn) if (fp + tn) > 0 else None
            precision = tp / (tp + fp) if (tp + fp) > 0 else None

        results.append(GroupMetrics(
            group=str(group_val),
            n_samples=n,
            selection_rate=round(selection_rate, 6),
            true_positive_rate=round(tpr, 6) if tpr is not None else None,
            false_positive_rate=round(fpr, 6) if fpr is not None else None,
            precision=round(precision, 6) if precision is not None else None,
        ))

    return results


def _audit_single_attribute(
    attr_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None,
    groups: np.ndarray,
) -> AttributeFairnessResult:
    is_classification = len(np.unique(y_true)) <= 20

    by_group = _group_metrics_for_attr(y_true, y_pred, y_proba, groups, is_classification)

    # Demographic parity difference: max selection_rate - min selection_rate
    selection_rates = [gm.selection_rate for gm in by_group]
    dp_diff = max(selection_rates) - min(selection_rates) if len(selection_rates) >= 2 else 0.0

    # Equal opportunity difference: max TPR - min TPR
    tpr_vals = [gm.true_positive_rate for gm in by_group if gm.true_positive_rate is not None]
    eo_diff = max(tpr_vals) - min(tpr_vals) if len(tpr_vals) >= 2 else None

    # Equalized odds difference: max of (TPR diff, FPR diff)
    fpr_vals = [gm.false_positive_rate for gm in by_group if gm.false_positive_rate is not None]
    fpr_diff = max(fpr_vals) - min(fpr_vals) if len(fpr_vals) >= 2 else None
    eqodds_diff: float | None = None
    if eo_diff is not None and fpr_diff is not None:
        eqodds_diff = max(abs(eo_diff), abs(fpr_diff))

    # Try to use fairlearn for more precise metrics
    try:
        from fairlearn.metrics import (
            demographic_parity_difference,
            equalized_odds_difference,
        )
        dp_diff = float(demographic_parity_difference(y_true, y_pred, sensitive_features=groups))
        if is_classification:
            eqodds_diff = float(equalized_odds_difference(y_true, y_pred, sensitive_features=groups))
    except ImportError:
        logger.debug("fairlearn not installed - using manual metric computation")
    except Exception as exc:
        logger.warning("fairlearn metric computation failed: %s", exc)

    severity = _classify_severity(dp_diff)
    if eo_diff is not None:
        severity = max(
            [severity, _classify_severity(eo_diff)],
            key=lambda s: ["none", "mild", "moderate", "severe"].index(s),
        )

    notes = []
    if severity in ("moderate", "severe"):
        min_tpr_group = min(by_group, key=lambda g: g.true_positive_rate or 1.0)
        max_tpr_group = max(by_group, key=lambda g: g.true_positive_rate or 0.0)
        if min_tpr_group.true_positive_rate is not None and max_tpr_group.true_positive_rate is not None:
            notes.append(
                f"TPR gap: {max_tpr_group.group}={max_tpr_group.true_positive_rate:.2f} vs "
                f"{min_tpr_group.group}={min_tpr_group.true_positive_rate:.2f}"
            )

    return AttributeFairnessResult(
        attribute=attr_name,
        demographic_parity_diff=round(dp_diff, 6),
        equalized_odds_diff=round(eqodds_diff, 6) if eqodds_diff is not None else None,
        equal_opportunity_diff=round(eo_diff, 6) if eo_diff is not None else None,
        by_group=by_group,
        severity=severity,
        note="; ".join(notes),
    )


def fairness_audit(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None,
    sensitive_features: dict[str, np.ndarray],
) -> FairnessReport:
    """Perform a full fairness audit across one or more protected attributes.

    Args:
        y_true: Ground-truth labels.
        y_pred: Hard predictions (thresholded).
        y_proba: Predicted probabilities (binary: shape (n,) or (n,2); None for regression).
        sensitive_features: Mapping of attribute name → array of group labels.

    Returns:
        FairnessReport with per-attribute results and intersectional analysis
        when >1 attribute is provided.
    """
    if len(sensitive_features) == 0:
        return FairnessReport()

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    attribute_results: list[AttributeFairnessResult] = []
    for attr_name, groups in sensitive_features.items():
        groups_arr = np.asarray(groups)
        result = _audit_single_attribute(attr_name, y_true, y_pred, y_proba, groups_arr)
        attribute_results.append(result)

    # Intersectional analysis when >1 protected attribute
    intersectional_results: list[AttributeFairnessResult] = []
    if len(sensitive_features) > 1:
        attr_names = list(sensitive_features.keys())
        attr_arrays = [np.asarray(v) for v in sensitive_features.values()]

        # Create composite group labels (e.g. "senior=1 × gender=Female")
        composite_groups = np.array([
            " × ".join(f"{n}={v}" for n, v in zip(attr_names, combo))
            for combo in zip(*attr_arrays)
        ])
        composite_name = " × ".join(attr_names)
        intersect_result = _audit_single_attribute(
            composite_name, y_true, y_pred, y_proba, composite_groups
        )
        intersectional_results.append(intersect_result)

    all_results = attribute_results + intersectional_results
    severity_rank = ["none", "mild", "moderate", "severe"]
    overall_severity = max(
        (r.severity for r in all_results),
        key=lambda s: severity_rank.index(s),
        default="none",
    )

    blocks = overall_severity == "severe"

    return FairnessReport(
        attributes=attribute_results,
        intersectional=intersectional_results,
        overall_severity=overall_severity,
        blocks_deliverables=blocks,
        requires_acknowledgment=overall_severity in ("moderate", "severe"),
    )


def build_sensitive_features(
    df: pd.DataFrame,
    protected_columns: list[str],
    index: pd.Index | None = None,
) -> dict[str, np.ndarray]:
    """Extract protected attribute arrays aligned to a given index.

    If `index` is provided (e.g. the test-set index after train/test split),
    only rows with that index are used.
    """
    result: dict[str, np.ndarray] = {}
    for col in protected_columns:
        if col not in df.columns:
            logger.warning("Protected column %r not found in DataFrame - skipping", col)
            continue
        series = df[col] if index is None else df.loc[index, col]
        result[col] = series.to_numpy()
    return result
