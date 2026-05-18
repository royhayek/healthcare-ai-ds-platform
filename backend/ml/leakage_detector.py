"""Leakage detection (§7, §9).

Three categories of label leakage are flagged:
  1. High-correlation features: numeric correlation with target ≥ threshold.
     Flags "features that may encode the answer" (e.g., a credit_score column
     in a loan default dataset where the credit score was computed post-default).
  2. Exact-copy columns: columns that are monotone functions of the target
     (e.g., a binary indicator that is 1 iff the event occurred).
  3. Categorical PROXY leakage (binary OR multiclass): a categorical feature
     whose values map near-deterministically onto a single target class, so a
     model just memorises a lookup table instead of learning. Domain-agnostic:
     it catches a `diagnosis_code → disease` proxy, a `discharge_unit → mortality`
     proxy, or any near-perfect categorical predictor - without knowing the domain.

Separately, `detect_unlabeled_target_classes` flags target values that are really
"unlabeled" placeholders (e.g., "unknown", "pending", "n/a") rather than genuine
outcome levels - these are missing labels and inflate accuracy if trained on.

This module does NOT drop columns or rows - it only reports. The EDA/preprocessing
agents (and the user) decide what to do with the findings.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class LeakageCandidate(BaseModel):
    column: str
    reason: str
    correlation: float | None = None
    severity: str  # low | medium | high


class LeakageReport(BaseModel):
    candidates: list[LeakageCandidate] = Field(default_factory=list)
    n_flagged: int = 0
    recommendation: str = ""


class UnlabeledTargetReport(BaseModel):
    """Target classes that look like 'missing label' placeholders rather than
    genuine outcome levels."""
    suspicious_classes: list[str] = Field(default_factory=list)
    affected_rows: int = 0
    total_rows: int = 0
    recommendation: str = ""


# Lexical placeholders that mean "we don't know the label", not a real outcome.
# Kept domain-agnostic - matched case-insensitively against target class values.
_UNLABELED_TOKENS = {
    "unknown", "unlabeled", "unlabelled", "unspecified", "undetermined",
    "not determined", "not_determined", "not determined yet", "indeterminate",
    "n/a", "na", "none", "null", "nan", "missing", "pending", "tbd", "to be determined",
    "unclassified", "not available", "notavailable", "not applicable", "?", "",
}


def detect_leakage(
    df: pd.DataFrame,
    target_column: str,
    high_corr_threshold: float = 0.95,
) -> LeakageReport:
    """Flag columns that may encode the target label.

    Args:
        df: The full DataFrame including the target column.
        target_column: Name of the target/label column.
        high_corr_threshold: Absolute Pearson/Spearman correlation above which
            a numeric column is flagged. Default 0.95 is conservative - only
            near-perfect predictors are flagged automatically.

    Returns:
        LeakageReport with ranked candidates and a plain-English recommendation.
    """
    if target_column not in df.columns:
        logger.warning("Target column %r not found - leakage detection skipped", target_column)
        return LeakageReport()

    y = df[target_column]
    candidates: list[LeakageCandidate] = []

    feature_cols = [c for c in df.columns if c != target_column]

    for col in feature_cols:
        series = df[col]

        # Skip all-null columns
        if series.isna().all():
            continue

        col_corr = _safe_correlation(series, y)

        if col_corr is not None and abs(col_corr) >= high_corr_threshold:
            severity = "high" if abs(col_corr) >= 0.99 else "medium"
            candidates.append(LeakageCandidate(
                column=col,
                reason=(
                    f"Very high correlation with target ({col_corr:.3f}). "
                    "This feature may have been computed after the outcome was known "
                    "(target encoding, derived field, or data entry error)."
                ),
                correlation=round(col_corr, 6),
                severity=severity,
            ))
            continue

        # Categorical PROXY leakage - works for binary AND multiclass targets.
        # A near-perfect categorical predictor lets the model memorise a lookup
        # table (e.g. clade→pathogenicity, diagnosis_code→disease). Guarded against
        # identifier columns, which would trivially score purity 1.0.
        purity, n_cat = _categorical_target_purity(series, y)
        if purity is not None and purity >= 0.95 and not _is_identifier_like(series, n_cat):
            severity = "high" if purity >= 0.98 else "medium"
            candidates.append(LeakageCandidate(
                column=col,
                reason=(
                    f"Categorical column maps {purity:.1%} of rows onto a single target "
                    f"class ({n_cat} categories) - a near-deterministic proxy for the "
                    "label. A model given this memorises a lookup table rather than "
                    "learning. Confirm it is not derived from / synonymous with the target."
                ),
                correlation=round(purity, 6),
                severity=severity,
            ))

    # Sort by severity
    severity_order = {"high": 0, "medium": 1, "low": 2}
    candidates.sort(key=lambda c: (severity_order.get(c.severity, 2), c.column))

    recommendation = ""
    if candidates:
        high_names = [c.column for c in candidates if c.severity == "high"]
        if high_names:
            recommendation = (
                f"Drop or quarantine high-severity columns before training: "
                f"{', '.join(high_names)}. "
                "Confirm with the data owner that these columns were NOT derived from the target."
            )
        else:
            recommendation = (
                "Review medium-severity columns with the data owner before including them "
                "in the feature set."
            )

    return LeakageReport(
        candidates=candidates,
        n_flagged=len(candidates),
        recommendation=recommendation,
    )


def _safe_correlation(series: pd.Series, y: pd.Series) -> float | None:
    """Return Pearson correlation for numeric, Spearman for ordered categorical."""
    col_clean = series.dropna()
    y_clean = y.loc[col_clean.index].dropna()
    common_idx = col_clean.index.intersection(y_clean.index)
    if len(common_idx) < 10:
        return None

    x_vals = col_clean.loc[common_idx]
    y_vals = y_clean.loc[common_idx]

    try:
        # Convert target to numeric for correlation
        if y_vals.dtype == object:
            y_vals = pd.Categorical(y_vals).codes  # type: ignore[assignment]

        if x_vals.dtype in (np.float64, np.float32, np.int64, np.int32):
            corr = float(np.corrcoef(x_vals.to_numpy(float), np.asarray(y_vals, float))[0, 1])
            return corr if np.isfinite(corr) else None
    except Exception:
        pass
    return None


def _categorical_target_purity(series: pd.Series, y: pd.Series) -> tuple[float | None, int]:
    """Fraction of rows whose feature value maps to that value's most common target
    class, averaged over the dataset. 1.0 ⇒ the feature perfectly determines the
    target (pure lookup table). Works for any number of target classes.

    Only meaningful for categorical / low-cardinality columns; returns (None, 0)
    for high-cardinality continuous numerics so they are scored by correlation only.
    """
    # Skip continuous numerics: only treat object dtype or genuinely low-cardinality
    # numeric columns (encoded categories) as categorical.
    is_object = series.dtype == object or str(series.dtype) in ("category", "string", "boolean")
    if not is_object and series.nunique(dropna=True) > 25:
        return None, 0

    combined = pd.DataFrame({"feature": series, "target": y}).dropna()
    if len(combined) < 10:
        return None, 0

    n_cat = int(combined["feature"].nunique())
    if n_cat < 2:
        return None, n_cat

    # Sum of the dominant-target-class count within each feature value.
    dominant = combined.groupby("feature", observed=True)["target"].agg(
        lambda s: int(s.value_counts().iloc[0])
    ).sum()
    purity = float(dominant) / float(len(combined))
    return purity, n_cat


def _is_identifier_like(series: pd.Series, n_cat: int) -> bool:
    """Heuristic to exclude ID-style columns (which trivially have purity 1.0).
    A column is identifier-like if its values are nearly unique per row."""
    n_valid = int(series.notna().sum())
    if n_valid == 0:
        return True
    # > 50% distinct values, or fewer than ~3 rows per category on average → ID-like.
    if n_cat > 0.5 * n_valid:
        return True
    if n_valid / max(n_cat, 1) < 3:
        return True
    return False


def detect_unlabeled_target_classes(y: pd.Series) -> UnlabeledTargetReport:
    """Flag target classes that are 'missing label' placeholders (e.g. "unknown",
    "pending", "n/a") rather than genuine outcome levels. Domain-agnostic: matches
    class values lexically against a placeholder vocabulary.

    These rows are missing labels, not an outcome - training on them inflates
    accuracy via a trivial placeholder lookup and is scientifically meaningless.
    """
    vals = y.dropna()
    if len(vals) == 0:
        return UnlabeledTargetReport()

    counts = vals.value_counts()
    suspicious = [str(v) for v in counts.index if str(v).strip().lower() in _UNLABELED_TOKENS]
    affected = int(counts[[v for v in counts.index if str(v) in suspicious]].sum()) if suspicious else 0

    recommendation = ""
    if suspicious:
        pct = 100 * affected / max(len(vals), 1)
        recommendation = (
            f"Target class(es) {suspicious} look like 'unlabeled' placeholders "
            f"({affected} rows, {pct:.1f}% of the data), not a genuine outcome level. "
            "Exclude these rows from supervised training (or model them separately as a "
            "novelty/abstention class) - keeping them inflates accuracy with a trivial "
            "placeholder lookup and produces meaningless predictions."
        )

    return UnlabeledTargetReport(
        suspicious_classes=suspicious,
        affected_rows=affected,
        total_rows=int(len(vals)),
        recommendation=recommendation,
    )
