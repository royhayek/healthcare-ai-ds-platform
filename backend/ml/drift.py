"""Drift detection between training and new datasets (§17).

Three metrics per numeric feature: PSI, KS statistic, Wasserstein distance.
Two metrics per categorical feature: chi-squared + JS-divergence.

All functions operate on raw pandas Series/DataFrames (pre-transform), so drift
is measured at the feature level before any preprocessing is applied.
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import jensenshannon
from pydantic import BaseModel, Field


class FeatureDriftResult(BaseModel):
    feature: str
    type: str  # numeric | categorical
    psi: float | None = None
    ks_statistic: float | None = None
    ks_p_value: float | None = None
    wasserstein: float | None = None
    wasserstein_relative: float | None = None  # normalized by training std
    chi2: float | None = None
    chi2_p_value: float | None = None
    js_divergence: float | None = None
    severity: str = "stable"  # stable | mild | significant


class DriftReport(BaseModel):
    overall_severity: str  # stable | mild | significant
    aggregate_psi: float
    features: list[FeatureDriftResult] = Field(default_factory=list)
    n_features_drifted: int = 0
    significant_features: list[str] = Field(default_factory=list)
    n_train_rows: int = 0
    n_new_rows: int = 0
    warning: str | None = None


def population_stability_index(
    expected: np.ndarray,
    actual: np.ndarray,
    bins: int = 10,
) -> float:
    """Standard PSI calculation.

    Bin edges are computed on expected; actual is binned identically.
    < 0.10 = stable, 0.10-0.25 = mild, > 0.25 = significant.
    """
    if len(expected) == 0 or len(actual) == 0:
        return 0.0

    breakpoints = np.linspace(0, 1, bins + 1)
    quantile_edges = np.quantile(expected, breakpoints)

    # Handle degenerate case: all values identical
    if np.all(quantile_edges == quantile_edges[0]):
        return 0.0

    quantile_edges[0] = -np.inf
    quantile_edges[-1] = np.inf

    expected_pct = np.histogram(expected, bins=quantile_edges)[0] / len(expected)
    actual_pct = np.histogram(actual, bins=quantile_edges)[0] / len(actual)

    expected_pct = np.clip(expected_pct, 1e-6, 1.0)
    actual_pct = np.clip(actual_pct, 1e-6, 1.0)

    return float(np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct)))


def _classify_drift_severity(psi: float | None, ks_p: float | None = None) -> str:
    if psi is not None:
        if psi >= 0.25:
            return "significant"
        if psi >= 0.10:
            return "mild"
    if ks_p is not None and ks_p < 0.01:
        return "significant"
    return "stable"


def _compute_numeric_drift(
    train_series: pd.Series,
    new_series: pd.Series,
) -> FeatureDriftResult:
    feature = str(train_series.name)
    train_arr = train_series.dropna().to_numpy(dtype=float)
    new_arr = new_series.dropna().to_numpy(dtype=float)

    if len(train_arr) == 0 or len(new_arr) == 0:
        return FeatureDriftResult(feature=feature, type="numeric", severity="stable")

    psi = population_stability_index(train_arr, new_arr)
    ks_stat, ks_p = stats.ks_2samp(train_arr, new_arr)

    # Wasserstein distance, normalized by training std to be scale-independent
    wass = float(stats.wasserstein_distance(train_arr, new_arr))
    train_std = float(np.std(train_arr))
    wass_relative = wass / train_std if train_std > 0 else 0.0

    severity = _classify_drift_severity(psi, float(ks_p))

    return FeatureDriftResult(
        feature=feature,
        type="numeric",
        psi=round(psi, 6),
        ks_statistic=round(float(ks_stat), 6),
        ks_p_value=round(float(ks_p), 6),
        wasserstein=round(wass, 6),
        wasserstein_relative=round(wass_relative, 6),
        severity=severity,
    )


def _compute_categorical_drift(
    train_series: pd.Series,
    new_series: pd.Series,
) -> FeatureDriftResult:
    feature = str(train_series.name)
    train_vals = train_series.dropna().astype(str)
    new_vals = new_series.dropna().astype(str)

    if len(train_vals) == 0 or len(new_vals) == 0:
        return FeatureDriftResult(feature=feature, type="categorical", severity="stable")

    # Build contingency table over union of categories
    all_cats = sorted(set(train_vals.unique()) | set(new_vals.unique()))
    train_counts = train_vals.value_counts().reindex(all_cats, fill_value=0)
    new_counts = new_vals.value_counts().reindex(all_cats, fill_value=0)

    # Chi-squared requires at least one non-zero in each column
    contingency = np.vstack([train_counts.values, new_counts.values])
    if contingency.sum() == 0:
        return FeatureDriftResult(feature=feature, type="categorical", severity="stable")

    try:
        chi2, p, _, _ = stats.chi2_contingency(contingency)
    except ValueError:
        chi2, p = 0.0, 1.0

    # JS-divergence
    train_prob = np.clip(train_counts.values / train_counts.sum(), 1e-10, 1.0)
    new_prob = np.clip(new_counts.values / new_counts.sum(), 1e-10, 1.0)
    js_div = float(jensenshannon(train_prob, new_prob))

    severity = "significant" if float(p) < 0.01 else ("mild" if float(p) < 0.05 else "stable")

    return FeatureDriftResult(
        feature=feature,
        type="categorical",
        chi2=round(float(chi2), 6),
        chi2_p_value=round(float(p), 6),
        js_divergence=round(js_div, 6),
        severity=severity,
    )


def compute_drift_report(
    X_train: pd.DataFrame,
    X_new: pd.DataFrame,
    numeric_cols: list[str],
    categorical_cols: list[str],
) -> DriftReport:
    """Compute per-feature drift metrics between training and new data.

    Only columns present in both DataFrames are analyzed. Missing columns
    produce a schema-incompatibility error upstream - this function receives
    already-validated frames.
    """
    feature_results: list[FeatureDriftResult] = []

    shared_cols = set(X_train.columns) & set(X_new.columns)

    for col in numeric_cols:
        if col not in shared_cols:
            continue
        result = _compute_numeric_drift(X_train[col], X_new[col])
        feature_results.append(result)

    for col in categorical_cols:
        if col not in shared_cols:
            continue
        result = _compute_categorical_drift(X_train[col], X_new[col])
        feature_results.append(result)

    numeric_psi_values = [
        r.psi for r in feature_results if r.type == "numeric" and r.psi is not None
    ]
    aggregate_psi = float(np.mean(numeric_psi_values)) if numeric_psi_values else 0.0

    significant = [r.feature for r in feature_results if r.severity == "significant"]
    mild = [r.feature for r in feature_results if r.severity == "mild"]

    if significant:
        overall_severity = "significant"
    elif mild:
        overall_severity = "mild"
    else:
        overall_severity = "stable"

    warning = None
    if overall_severity == "significant":
        warning = (
            f"Significant drift detected on {len(significant)} feature(s): "
            f"{', '.join(significant[:5])}. "
            "Predictions on this dataset may be unreliable. "
            "Consider retraining or restricting predictions to similar samples."
        )

    # Sort by severity for display: significant → mild → stable
    severity_order = {"significant": 0, "mild": 1, "stable": 2}
    feature_results.sort(key=lambda r: (severity_order.get(r.severity, 2), r.feature))

    return DriftReport(
        overall_severity=overall_severity,
        aggregate_psi=round(aggregate_psi, 6),
        features=feature_results,
        n_features_drifted=len(significant) + len(mild),
        significant_features=significant,
        n_train_rows=len(X_train),
        n_new_rows=len(X_new),
        warning=warning,
    )
