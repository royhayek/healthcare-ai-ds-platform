"""Dataset profiler (§8, §9).

Computes statistical summaries used by the EDA agent. Never sends raw rows
to the model - only the compressed profile (compress_profile_for_claude).

profile_dataset: full statistical profile of a DataFrame
compress_profile_for_claude: trims the profile to what the model needs
"""

import hashlib
import json
import math
import re
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel
from scipy import stats as scipy_stats

TOP_VALUES_COUNT = 10
HIGH_CORRELATION_THRESHOLD = 0.85
OUTLIER_IQR_MULTIPLIER = 1.5
COMPRESS_TOP_VALUES_COUNT = 5
_FLOAT_PRECISION = 4


class ColumnProfile(BaseModel):
    name: str
    dtype: str
    null_count: int
    null_pct: float
    n_unique: int
    # Numeric fields - None for categorical columns
    mean: float | None = None
    median: float | None = None
    std: float | None = None
    min_val: float | None = None
    max_val: float | None = None
    q25: float | None = None
    q75: float | None = None
    iqr: float | None = None
    skewness: float | None = None
    kurtosis: float | None = None
    outlier_count: int | None = None
    # Categorical fields - None for numeric columns
    entropy: float | None = None
    top_values: dict[str, int] | None = None


class DatasetProfile(BaseModel):
    n_rows: int
    n_cols: int
    duplicate_count: int
    numeric_columns: list[str]
    categorical_columns: list[str]
    columns: list[ColumnProfile]
    correlation_matrix: dict[str, dict[str, float]]
    high_correlation_pairs: list[dict[str, Any]]
    target_column: str | None
    task_type: str | None  # binary_classification | multiclass | regression | unknown
    # Enhanced data quality fields (Category D)
    isolation_score_summary: dict[str, Any] | None = None  # percentile distribution of anomaly scores
    missingness_correlation: dict[str, dict[str, float]] | None = None  # §D3
    vif: dict[str, float] | None = None  # §D4
    # Healthcare-specific fields
    phi_columns: list[dict[str, Any]] | None = None   # suspected PHI columns with confidence
    clinical_range_flags: dict[str, dict[str, Any]] | None = None  # out-of-range lab/vital flags
    icd_columns: list[str] | None = None  # columns detected as containing ICD-10 codes


def _safe_float(val: Any) -> float | None:
    try:
        f = float(val)
        return None if math.isnan(f) or math.isinf(f) else f
    except (TypeError, ValueError):
        return None


def _profile_numeric(series: pd.Series) -> dict[str, Any]:
    non_null = series.dropna()
    if non_null.empty:
        return {}

    q25 = _safe_float(non_null.quantile(0.25))
    q75 = _safe_float(non_null.quantile(0.75))
    iqr = (_safe_float(q75 - q25) if q25 is not None and q75 is not None else None)

    outlier_count: int | None = None
    if iqr is not None and q25 is not None and q75 is not None:
        lower = q25 - OUTLIER_IQR_MULTIPLIER * iqr
        upper = q75 + OUTLIER_IQR_MULTIPLIER * iqr
        outlier_count = int(((non_null < lower) | (non_null > upper)).sum())

    skew = _safe_float(scipy_stats.skew(non_null, nan_policy="omit"))
    kurt = _safe_float(scipy_stats.kurtosis(non_null, nan_policy="omit"))

    return dict(
        mean=_safe_float(non_null.mean()),
        median=_safe_float(non_null.median()),
        std=_safe_float(non_null.std()),
        min_val=_safe_float(non_null.min()),
        max_val=_safe_float(non_null.max()),
        q25=q25,
        q75=q75,
        iqr=iqr,
        skewness=skew,
        kurtosis=kurt,
        outlier_count=outlier_count,
    )


def _profile_categorical(series: pd.Series) -> dict[str, Any]:
    non_null = series.dropna().astype(str)
    if non_null.empty:
        return {}

    counts = non_null.value_counts()
    total = counts.sum()
    probs = counts / total
    entropy = _safe_float(scipy_stats.entropy(probs))
    top_values = {str(k): int(v) for k, v in counts.head(TOP_VALUES_COUNT).items()}

    return dict(entropy=entropy, top_values=top_values)


def _detect_task_type(series: pd.Series) -> str:
    """Infer classification vs regression from the target column."""
    non_null = series.dropna()
    n_unique = non_null.nunique()

    if n_unique == 2:
        return "binary_classification"

    if pd.api.types.is_float_dtype(series):
        if n_unique > 20:
            return "regression"
        return "multiclass"

    if pd.api.types.is_integer_dtype(series):
        if n_unique <= 20:
            return "multiclass"
        return "regression"

    # Any non-numeric target (object, pandas StringDtype/"str", category, bool)
    # is categorical. NOTE: is_object_dtype() is False for pandas 2.2+ StringDtype,
    # so checking "not numeric" rather than "is object" is essential - otherwise a
    # plain string target falls through to "unknown" and the trainer mis-handles
    # it as regression, crashing on string labels.
    if not pd.api.types.is_numeric_dtype(series):
        if n_unique <= 20:
            return "multiclass"
        return "regression"

    return "unknown"


def _compute_correlations(
    df: pd.DataFrame, numeric_cols: list[str]
) -> tuple[dict[str, dict[str, float]], list[dict[str, Any]]]:
    if len(numeric_cols) < 2:
        return {}, []

    corr_df = df[numeric_cols].corr(method="pearson")
    matrix: dict[str, dict[str, float]] = {}
    for col in corr_df.columns:
        matrix[col] = {}
        for other in corr_df.columns:
            val = _safe_float(corr_df.loc[col, other])
            if val is not None:
                matrix[col][other] = round(val, _FLOAT_PRECISION)

    pairs: list[dict[str, Any]] = []
    seen: set[frozenset[str]] = set()
    for col_a in numeric_cols:
        for col_b in numeric_cols:
            if col_a == col_b:
                continue
            pair_key = frozenset({col_a, col_b})
            if pair_key in seen:
                continue
            seen.add(pair_key)
            val = corr_df.loc[col_a, col_b]
            if not math.isnan(float(val)) and abs(float(val)) >= HIGH_CORRELATION_THRESHOLD:
                pairs.append(
                    dict(col_a=col_a, col_b=col_b, correlation=round(float(val), _FLOAT_PRECISION))
                )

    return matrix, sorted(pairs, key=lambda p: abs(p["correlation"]), reverse=True)


def compute_isolation_scores(df: pd.DataFrame, n_estimators: int = 100) -> dict[str, Any]:
    """Anomaly score distribution via IsolationForest (§D1).

    Returns a dict of percentile statistics so the profile stays aggregate-only
    (no raw row indices are stored). The EDA agent can reference outlier_pct_rough
    as an approximate "fraction of anomalous rows".
    """
    try:
        from sklearn.ensemble import IsolationForest

        numeric = df.select_dtypes(include="number").dropna(axis=1, how="all")
        if numeric.shape[1] == 0 or len(numeric) < 10:
            return {}

        filled = numeric.fillna(numeric.median())
        iso = IsolationForest(n_estimators=n_estimators, random_state=42, n_jobs=-1)
        iso.fit(filled)
        scores = iso.decision_function(filled)

        pcts = np.percentile(scores, [5, 25, 50, 75, 95])
        outlier_pct = float((scores < 0).mean())

        return {
            "p5": round(float(pcts[0]), _FLOAT_PRECISION),
            "p25": round(float(pcts[1]), _FLOAT_PRECISION),
            "p50": round(float(pcts[2]), _FLOAT_PRECISION),
            "p75": round(float(pcts[3]), _FLOAT_PRECISION),
            "p95": round(float(pcts[4]), _FLOAT_PRECISION),
            "outlier_pct_rough": round(outlier_pct, _FLOAT_PRECISION),
            "n_flagged": int((scores < 0).sum()),
            "n_total": len(scores),
        }
    except Exception:
        return {}


def compute_missingness_correlation(df: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Pearson correlation of boolean missingness indicators per column (§D3).

    Reveals whether columns tend to be missing together (MAR / MNAR patterns).
    Returns {} when fewer than 2 columns have any missing values.
    """
    cols_with_missing = [c for c in df.columns if df[c].isnull().any()]
    if len(cols_with_missing) < 2:
        return {}

    missing_indicators = df[cols_with_missing].isnull().astype(int)
    corr = missing_indicators.corr()

    result: dict[str, dict[str, float]] = {}
    for col in corr.columns:
        row: dict[str, float] = {}
        for other in corr.columns:
            if col != other:
                v = _safe_float(corr.loc[col, other])
                if v is not None:
                    row[other] = round(v, _FLOAT_PRECISION)
        if row:
            result[col] = row
    return result


def compute_vif(df: pd.DataFrame, numeric_cols: list[str]) -> dict[str, float]:
    """Variance Inflation Factor per numeric column (§D4).

    VIF > 10 indicates severe multicollinearity. Requires statsmodels.
    Returns {} when fewer than 2 columns are provided or on any error.
    """
    if len(numeric_cols) < 2:
        return {}
    try:
        from statsmodels.stats.outliers_influence import variance_inflation_factor  # type: ignore[import-untyped]

        sub = df[numeric_cols].dropna()
        if len(sub) < 10 or sub.shape[1] < 2:
            return {}

        vif_data: dict[str, float] = {}
        for i, col in enumerate(sub.columns.tolist()):
            try:
                v = variance_inflation_factor(sub.values, i)
                vif_data[col] = round(float(v), 2) if math.isfinite(float(v)) else float("nan")
            except Exception:
                vif_data[col] = float("nan")
        return vif_data
    except ImportError:
        return {}
    except Exception:
        return {}


# ── PHI column names that trigger a flag regardless of content ────────────────
_PHI_NAME_PATTERNS = re.compile(
    r"(^|_)(name|patient_name|first_name|last_name|full_name|"
    r"ssn|social_security|sin|"
    r"mrn|medical_record|patient_id(?!_hash)|"
    r"dob|date_of_birth|birthdate|birth_date|"
    r"address|street|zip|postal|"
    r"phone|telephone|mobile|"
    r"email|e_mail|"
    r"npi|national_provider)($|_)",
    re.IGNORECASE,
)
_SSN_RE = re.compile(r"^\d{3}-\d{2}-\d{4}$")
_MRN_RE = re.compile(r"^MRN\d{5,}$", re.IGNORECASE)
_DOB_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def detect_phi_columns(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Heuristically detect columns that likely contain Protected Health Information.

    Checks both column names and a sample of values. Returns a list of dicts
    with `column`, `reason`, and `confidence` (low / medium / high). Only
    columns that match at least one signal are returned.

    No raw values are included in the output - only pattern-match summaries.
    """
    flags: list[dict[str, Any]] = []

    for col in df.columns:
        reasons: list[str] = []
        confidence = "low"

        # --- Name-based signal ---
        if _PHI_NAME_PATTERNS.search(col):
            reasons.append(f"column name '{col}' matches PHI keyword pattern")
            confidence = "high"

        # --- Value-based signal (sample up to 20 non-null values) ---
        sample = df[col].dropna().astype(str).head(20).tolist()
        if sample:
            ssn_hits = sum(1 for v in sample if _SSN_RE.match(v.strip()))
            mrn_hits = sum(1 for v in sample if _MRN_RE.match(v.strip()))
            dob_hits = sum(1 for v in sample if _DOB_RE.match(v.strip()))
            email_hits = sum(1 for v in sample if _EMAIL_RE.match(v.strip()))

            if ssn_hits / len(sample) > 0.3:
                reasons.append(f"values match SSN pattern ({ssn_hits}/{len(sample)} sampled)")
                confidence = "high"
            if mrn_hits / len(sample) > 0.3:
                reasons.append(f"values match MRN pattern ({mrn_hits}/{len(sample)} sampled)")
                confidence = "high"
            if dob_hits / len(sample) > 0.5:
                reasons.append(f"values appear to be dates of birth ({dob_hits}/{len(sample)} sampled)")
                confidence = "medium" if confidence == "low" else confidence
            if email_hits / len(sample) > 0.3:
                reasons.append(f"values match email pattern ({email_hits}/{len(sample)} sampled)")
                confidence = "medium" if confidence == "low" else confidence

        if reasons:
            flags.append({
                "column": col,
                "reasons": reasons,
                "confidence": confidence,
            })

    return flags


# ── Clinical reference ranges for common lab / vital sign column names ────────
_CLINICAL_RANGES: dict[str, dict[str, Any]] = {
    # Lab values
    "hemoglobin":           {"min": 7.0,  "max": 18.0, "unit": "g/dL",   "critical_low": 7.0,  "critical_high": 20.0},
    "hgb":                  {"min": 7.0,  "max": 18.0, "unit": "g/dL",   "critical_low": 7.0,  "critical_high": 20.0},
    "hba1c":                {"min": 4.0,  "max": 15.0, "unit": "%",       "critical_low": 2.0,  "critical_high": 20.0},
    "hba1c_pct":            {"min": 4.0,  "max": 15.0, "unit": "%",       "critical_low": 2.0,  "critical_high": 20.0},
    "creatinine":           {"min": 0.4,  "max": 12.0, "unit": "mg/dL",  "critical_low": 0.2,  "critical_high": 15.0},
    "creatinine_mgdl":      {"min": 0.4,  "max": 12.0, "unit": "mg/dL",  "critical_low": 0.2,  "critical_high": 15.0},
    "glucose":              {"min": 50.0, "max": 400.0,"unit": "mg/dL",  "critical_low": 40.0, "critical_high": 500.0},
    "fasting_glucose":      {"min": 50.0, "max": 400.0,"unit": "mg/dL",  "critical_low": 40.0, "critical_high": 500.0},
    "fasting_glucose_mgdl": {"min": 50.0, "max": 400.0,"unit": "mg/dL",  "critical_low": 40.0, "critical_high": 500.0},
    "sodium":               {"min": 120.0,"max": 155.0,"unit": "mEq/L",  "critical_low": 115.0,"critical_high": 160.0},
    "potassium":            {"min": 2.5,  "max": 6.5,  "unit": "mEq/L",  "critical_low": 2.0,  "critical_high": 7.0},
    "bilirubin":            {"min": 0.1,  "max": 20.0, "unit": "mg/dL",  "critical_low": 0.0,  "critical_high": 30.0},
    "bilirubin_mgdl":       {"min": 0.1,  "max": 20.0, "unit": "mg/dL",  "critical_low": 0.0,  "critical_high": 30.0},
    "wbc":                  {"min": 1.0,  "max": 30.0, "unit": "×10³/µL","critical_low": 0.5,  "critical_high": 50.0},
    "platelet_count":       {"min": 20.0, "max": 600.0,"unit": "×10³/µL","critical_low": 10.0, "critical_high": 1000.0},
    "triglycerides":        {"min": 30.0, "max": 800.0,"unit": "mg/dL",  "critical_low": 20.0, "critical_high": 2000.0},
    "triglycerides_mgdl":   {"min": 30.0, "max": 800.0,"unit": "mg/dL",  "critical_low": 20.0, "critical_high": 2000.0},
    "hdl_cholesterol":      {"min": 15.0, "max": 120.0,"unit": "mg/dL",  "critical_low": 5.0,  "critical_high": 200.0},
    "hdl_cholesterol_mgdl": {"min": 15.0, "max": 120.0,"unit": "mg/dL",  "critical_low": 5.0,  "critical_high": 200.0},
    # Vital signs
    "systolic_bp":          {"min": 70.0, "max": 200.0,"unit": "mmHg",   "critical_low": 60.0, "critical_high": 220.0},
    "diastolic_bp":         {"min": 40.0, "max": 120.0,"unit": "mmHg",   "critical_low": 30.0, "critical_high": 130.0},
    "mean_arterial_pressure":{"min":40.0, "max": 140.0,"unit": "mmHg",   "critical_low": 30.0, "critical_high": 150.0},
    "heart_rate_bpm":       {"min": 35.0, "max": 200.0,"unit": "bpm",    "critical_low": 30.0, "critical_high": 250.0},
    "respiratory_rate":     {"min": 6.0,  "max": 50.0, "unit": "br/min", "critical_low": 4.0,  "critical_high": 60.0},
    "temperature_celsius":  {"min": 33.0, "max": 42.0, "unit": "°C",     "critical_low": 30.0, "critical_high": 43.0},
    "spo2_pct":             {"min": 70.0, "max": 100.0,"unit": "%",       "critical_low": 60.0, "critical_high": 100.0},
    "spo2":                 {"min": 70.0, "max": 100.0,"unit": "%",       "critical_low": 60.0, "critical_high": 100.0},
    "pao2_fio2_ratio":      {"min": 60.0, "max": 500.0,"unit": "mmHg",   "critical_low": 40.0, "critical_high": 600.0},
    "gcs_score":            {"min": 3.0,  "max": 15.0, "unit": "points",  "critical_low": 3.0,  "critical_high": 15.0},
    "bmi":                  {"min": 12.0, "max": 70.0, "unit": "kg/m²",  "critical_low": 10.0, "critical_high": 80.0},
}


def check_clinical_ranges(df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """Flag numeric columns whose values fall outside clinical reference ranges.

    Matches columns by exact name (case-insensitive). For each matched column
    computes: pct_below_min, pct_above_max, pct_critical (outside critical bounds),
    min_observed, max_observed.

    Returns only columns with at least one match. Never returns raw values.
    """
    result: dict[str, dict[str, Any]] = {}
    col_lower_map = {c.lower(): c for c in df.columns}

    for ref_name, ref in _CLINICAL_RANGES.items():
        col = col_lower_map.get(ref_name.lower())
        if col is None:
            continue

        series = pd.to_numeric(df[col], errors="coerce").dropna()
        if series.empty:
            continue

        n = len(series)
        pct_below = float((series < ref["min"]).sum()) / n
        pct_above = float((series > ref["max"]).sum()) / n
        pct_critical = float(
            ((series < ref["critical_low"]) | (series > ref["critical_high"])).sum()
        ) / n

        result[col] = {
            "reference_min": ref["min"],
            "reference_max": ref["max"],
            "unit": ref["unit"],
            "pct_below_reference_min": round(pct_below, _FLOAT_PRECISION),
            "pct_above_reference_max": round(pct_above, _FLOAT_PRECISION),
            "pct_critical_range": round(pct_critical, _FLOAT_PRECISION),
            "observed_min": round(float(series.min()), _FLOAT_PRECISION),
            "observed_max": round(float(series.max()), _FLOAT_PRECISION),
            "clinical_concern": pct_critical > 0.01 or pct_below + pct_above > 0.05,
        }

    return result


_ICD10_RE = re.compile(r"^[A-TV-Z]\d{2}(\.[\dA-Z]{1,4})?$")


def detect_icd_columns(df: pd.DataFrame) -> list[str]:
    """Detect columns whose values predominantly look like ICD-10 codes.

    Samples up to 30 non-null values per column. A column is flagged when
    ≥ 40% of sampled values match the ICD-10 pattern (letter + 2 digits,
    optionally dot + up to 4 alphanumeric chars).
    """
    icd_cols: list[str] = []
    for col in df.select_dtypes(include=["object", "string"]).columns:
        sample = df[col].dropna().astype(str).head(30).tolist()
        if not sample:
            continue
        hit_rate = sum(1 for v in sample if _ICD10_RE.match(v.strip())) / len(sample)
        if hit_rate >= 0.4:
            icd_cols.append(col)
    return icd_cols


def profile_dataset(df: pd.DataFrame, target_column: str | None = None) -> DatasetProfile:
    """Compute a full statistical profile of df.

    target_column: if provided, used for task_type detection and included
    in the profile. Must be a column in df.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

    column_profiles: list[ColumnProfile] = []
    for col in df.columns:
        series = df[col]
        null_count = int(series.isna().sum())
        n_unique = int(series.nunique(dropna=True))
        extra: dict[str, Any] = {}

        if col in numeric_cols:
            extra = _profile_numeric(series)
        else:
            extra = _profile_categorical(series)

        column_profiles.append(
            ColumnProfile(
                name=col,
                dtype=str(series.dtype),
                null_count=null_count,
                null_pct=round(null_count / len(df), _FLOAT_PRECISION) if len(df) else 0.0,
                n_unique=n_unique,
                **extra,
            )
        )

    corr_matrix, high_corr_pairs = _compute_correlations(df, numeric_cols)

    task_type: str | None = None
    if target_column and target_column in df.columns:
        task_type = _detect_task_type(df[target_column])

    # Enhanced quality metrics - computed after the base profile so they
    # don't block the pipeline on ImportError or unexpected data shapes.
    feature_numeric = [c for c in numeric_cols if c != target_column]
    isolation_summary = compute_isolation_scores(df) if len(df) >= 50 else {}
    missingness_corr = compute_missingness_correlation(df)
    vif = compute_vif(df, feature_numeric) if len(feature_numeric) >= 2 else {}

    # Healthcare-specific detection (safe - never raises, returns empty on failure)
    phi_cols = detect_phi_columns(df)
    clinical_flags = check_clinical_ranges(df)
    icd_cols = detect_icd_columns(df)

    return DatasetProfile(
        n_rows=len(df),
        n_cols=len(df.columns),
        duplicate_count=int(df.duplicated().sum()),
        numeric_columns=numeric_cols,
        categorical_columns=categorical_cols,
        columns=column_profiles,
        correlation_matrix=corr_matrix,
        high_correlation_pairs=high_corr_pairs,
        target_column=target_column,
        task_type=task_type,
        isolation_score_summary=isolation_summary or None,
        missingness_correlation=missingness_corr or None,
        vif=vif or None,
        phi_columns=phi_cols or None,
        clinical_range_flags=clinical_flags or None,
        icd_columns=icd_cols or None,
    )


def _round_floats(obj: Any, precision: int = _FLOAT_PRECISION) -> Any:
    if isinstance(obj, float):
        return round(obj, precision) if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _round_floats(v, precision) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_round_floats(v, precision) for v in obj]
    return obj


def compress_profile_for_claude(profile: DatasetProfile) -> dict[str, Any]:
    """Return a trimmed profile dict safe to pass to the model.

    Removes the full correlation_matrix (the model only needs the flagged pairs),
    caps top_values at COMPRESS_TOP_VALUES_COUNT, and rounds floats.
    Includes the enhanced quality fields (isolation scores, missingness
    correlation, VIF) so agents can reason about data quality issues.
    Never contains raw row-level data - only aggregates.

    PHI exclusion: columns flagged as high/medium PHI confidence have their
    full statistical summaries stripped - only name and PHI warning remain.
    This prevents the agent from reasoning about distributions of patient
    identifiers (name patterns, DOB ranges, etc.).
    """
    d = profile.model_dump()

    # Drop verbose full matrix - only keep flagged high-correlation pairs
    d.pop("correlation_matrix", None)

    # Build set of PHI column names (high/medium confidence) that should be redacted
    phi_raw = d.get("phi_columns") or []
    phi_redacted: set[str] = {
        p["column"]
        for p in phi_raw
        if p.get("confidence") in ("high", "medium")
    }

    # Process per-column stats
    for col_profile in d.get("columns", []):
        col_name = col_profile.get("name", "")
        if col_name in phi_redacted:
            # Strip all statistical fields - leave only name, dtype, and PHI flag
            for field in (
                "mean", "median", "std", "min_val", "max_val",
                "q25", "q75", "iqr", "skewness", "kurtosis",
                "outlier_count", "entropy", "top_values",
                "null_count", "null_pct", "n_unique",
            ):
                col_profile.pop(field, None)
            col_profile["phi_excluded"] = True
        else:
            tv = col_profile.get("top_values")
            if tv:
                col_profile["top_values"] = dict(list(tv.items())[:COMPRESS_TOP_VALUES_COUNT])

    # Trim VIF to only entries with VIF > 5 (only interesting ones for the model)
    vif = d.get("vif")
    if vif:
        d["vif"] = {k: v for k, v in vif.items() if isinstance(v, (int, float)) and v > 5}

    # Include PHI warning summary (column names and confidence only - no raw values ever)
    phi = d.get("phi_columns")
    if phi:
        d["phi_columns"] = [{"column": p["column"], "confidence": p["confidence"]} for p in phi]

    # Include clinical range flags - only columns with concerns
    clinical = d.get("clinical_range_flags")
    if clinical:
        d["clinical_range_flags"] = {
            col: flags for col, flags in clinical.items() if flags.get("clinical_concern")
        }

    # ICD columns: just the list of column names (already compact)

    return _round_floats(d)
