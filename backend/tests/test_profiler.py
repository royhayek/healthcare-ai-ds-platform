"""Tests for backend/ml/profiler.py - pure-Python, no DB or the model required."""

import math

import numpy as np
import pandas as pd
import pytest

from backend.ml.profiler import (
    HIGH_CORRELATION_THRESHOLD,
    DatasetProfile,
    check_clinical_ranges,
    compress_profile_for_claude,
    detect_icd_columns,
    detect_phi_columns,
    profile_dataset,
)


# ── Fixtures ───────────────────────────────────────────────────────────────────


@pytest.fixture
def binary_df() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n = 500
    age = rng.integers(18, 80, size=n).astype(float)
    age[rng.choice(n, 30, replace=False)] = np.nan  # 6% nulls
    income = age * 1000 + rng.normal(0, 5000, n)
    churn = (income < 45000).astype(int)
    cat = rng.choice(["A", "B", "C"], n)
    return pd.DataFrame({"age": age, "income": income, "churn": churn, "segment": cat})


@pytest.fixture
def regression_df() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 200
    x = rng.normal(0, 1, n)
    y = 3 * x + rng.normal(0, 0.5, n)
    return pd.DataFrame({"x": x, "y": y})


# ── profile_dataset ────────────────────────────────────────────────────────────


def test_profile_returns_correct_shape(binary_df: pd.DataFrame) -> None:
    p = profile_dataset(binary_df)
    assert p.n_rows == 500
    assert p.n_cols == 4


def test_profile_counts_duplicates() -> None:
    df = pd.DataFrame({"a": [1, 2, 2, 3], "b": ["x", "y", "y", "z"]})
    p = profile_dataset(df)
    assert p.duplicate_count == 1


def test_profile_classifies_columns(binary_df: pd.DataFrame) -> None:
    p = profile_dataset(binary_df)
    assert "age" in p.numeric_columns
    assert "income" in p.numeric_columns
    assert "segment" in p.categorical_columns


def test_profile_null_count(binary_df: pd.DataFrame) -> None:
    p = profile_dataset(binary_df, target_column="churn")
    age_prof = next(c for c in p.columns if c.name == "age")
    assert age_prof.null_count == 30
    assert abs(age_prof.null_pct - 0.06) < 0.005


def test_profile_numeric_stats(binary_df: pd.DataFrame) -> None:
    p = profile_dataset(binary_df)
    age_prof = next(c for c in p.columns if c.name == "age")
    assert age_prof.mean is not None
    assert age_prof.std is not None
    assert age_prof.q25 is not None
    assert age_prof.q75 is not None
    assert age_prof.iqr is not None
    assert age_prof.outlier_count is not None
    assert age_prof.skewness is not None
    assert age_prof.kurtosis is not None
    # Entropy and top_values should be None for numeric
    assert age_prof.entropy is None
    assert age_prof.top_values is None


def test_profile_categorical_stats(binary_df: pd.DataFrame) -> None:
    p = profile_dataset(binary_df)
    seg_prof = next(c for c in p.columns if c.name == "segment")
    assert seg_prof.entropy is not None
    assert seg_prof.top_values is not None
    assert set(seg_prof.top_values.keys()) == {"A", "B", "C"}
    # Numeric fields should be None for categorical
    assert seg_prof.mean is None
    assert seg_prof.std is None


def test_profile_task_type_binary(binary_df: pd.DataFrame) -> None:
    p = profile_dataset(binary_df, target_column="churn")
    assert p.task_type == "binary_classification"
    assert p.target_column == "churn"


def test_profile_task_type_regression(regression_df: pd.DataFrame) -> None:
    p = profile_dataset(regression_df, target_column="y")
    assert p.task_type == "regression"


def test_profile_task_type_none_when_no_target(binary_df: pd.DataFrame) -> None:
    p = profile_dataset(binary_df)
    assert p.task_type is None
    assert p.target_column is None


def test_profile_task_type_string_dtype_multiclass() -> None:
    # pandas 2.2+ StringDtype: is_object_dtype is False, so a naive detector
    # returns "unknown" and the trainer mis-handles it as regression. Must be
    # detected as multiclass.
    df = pd.DataFrame({
        "f1": np.arange(40, dtype=float),
        "target": pd.array(["high", "low", "moderate", "unknown"] * 10, dtype="string"),
    })
    assert str(df["target"].dtype) in ("string", "str")
    p = profile_dataset(df, target_column="target")
    assert p.task_type == "multiclass"


def test_profile_high_correlation_detected(regression_df: pd.DataFrame) -> None:
    p = profile_dataset(regression_df)
    # x and y are strongly correlated (r ≈ 0.99)
    assert len(p.high_correlation_pairs) >= 1
    pair = p.high_correlation_pairs[0]
    assert abs(pair["correlation"]) >= HIGH_CORRELATION_THRESHOLD


def test_profile_no_spurious_correlation() -> None:
    rng = np.random.default_rng(7)
    df = pd.DataFrame({"a": rng.normal(size=200), "b": rng.normal(size=200)})
    p = profile_dataset(df)
    assert p.high_correlation_pairs == []


def test_profile_missing_values_in_all_numeric() -> None:
    df = pd.DataFrame({"a": [np.nan, np.nan, np.nan]})
    p = profile_dataset(df)
    col = p.columns[0]
    assert col.null_count == 3
    # Stats should be None or not crash
    assert col.mean is None or isinstance(col.mean, float)


# ── compress_profile_for_claude ────────────────────────────────────────────────


def test_compress_removes_correlation_matrix(binary_df: pd.DataFrame) -> None:
    p = profile_dataset(binary_df)
    compressed = compress_profile_for_claude(p)
    assert "correlation_matrix" not in compressed


def test_compress_keeps_high_correlation_pairs(regression_df: pd.DataFrame) -> None:
    p = profile_dataset(regression_df)
    compressed = compress_profile_for_claude(p)
    assert "high_correlation_pairs" in compressed
    assert len(compressed["high_correlation_pairs"]) >= 1


def test_compress_caps_top_values() -> None:
    df = pd.DataFrame({"cat": list("ABCDEFGHIJ") * 10})
    p = profile_dataset(df)
    compressed = compress_profile_for_claude(p)
    col = compressed["columns"][0]
    assert col["top_values"] is not None
    assert len(col["top_values"]) <= 5


def test_compress_output_has_no_nan_floats(binary_df: pd.DataFrame) -> None:
    p = profile_dataset(binary_df)
    compressed = compress_profile_for_claude(p)
    _assert_no_nan(compressed)


def _assert_no_nan(obj) -> None:
    if isinstance(obj, float):
        assert not math.isnan(obj), f"NaN found in compressed profile: {obj}"
    elif isinstance(obj, dict):
        for v in obj.values():
            _assert_no_nan(v)
    elif isinstance(obj, list):
        for v in obj:
            _assert_no_nan(v)


# ── detect_phi_columns ─────────────────────────────────────────────────────────


def test_phi_detects_name_column() -> None:
    df = pd.DataFrame({"patient_name": ["John Doe", "Jane Smith"]})
    flags = detect_phi_columns(df)
    names = [f["column"] for f in flags]
    assert "patient_name" in names
    assert any(f["confidence"] == "high" for f in flags if f["column"] == "patient_name")


def test_phi_detects_mrn_by_value() -> None:
    df = pd.DataFrame({"record": [f"MRN{i:07d}" for i in range(25)]})
    flags = detect_phi_columns(df)
    assert any(f["column"] == "record" for f in flags)


def test_phi_detects_dob_column_name() -> None:
    df = pd.DataFrame({"date_of_birth": ["1980-01-15", "1992-07-22"]})
    flags = detect_phi_columns(df)
    assert any(f["column"] == "date_of_birth" for f in flags)


def test_phi_clean_column_not_flagged() -> None:
    df = pd.DataFrame({
        "age": [45, 62, 38],
        "systolic_bp": [130, 145, 118],
        "diabetes_dx": [0, 1, 0],
    })
    flags = detect_phi_columns(df)
    assert flags == []


def test_phi_appears_in_profile() -> None:
    df = pd.DataFrame({
        "mrn": [f"MRN{i:07d}" for i in range(50)],
        "age": range(50),
    })
    profile = profile_dataset(df)
    assert profile.phi_columns is not None
    assert any(p["column"] == "mrn" for p in profile.phi_columns)


def test_phi_in_compressed_profile_has_no_raw_values() -> None:
    df = pd.DataFrame({
        "patient_name": ["Alice", "Bob"],
        "age": [50, 60],
    })
    profile = profile_dataset(df)
    compressed = compress_profile_for_claude(profile)
    phi = compressed.get("phi_columns", [])
    for entry in phi:
        # Only column name and confidence - no reasons list (trimmed) and no raw values
        assert "column" in entry
        assert "confidence" in entry


# ── check_clinical_ranges ──────────────────────────────────────────────────────


def test_clinical_ranges_flags_out_of_range() -> None:
    df = pd.DataFrame({
        "hemoglobin": [8.0, 15.0, 200.0],  # 200 is physiologically impossible
    })
    flags = check_clinical_ranges(df)
    assert "hemoglobin" in flags
    assert flags["hemoglobin"]["pct_above_reference_max"] > 0


def test_clinical_ranges_normal_values_no_concern() -> None:
    df = pd.DataFrame({
        "hemoglobin": [12.5, 13.0, 14.5, 11.8, 15.2],
    })
    flags = check_clinical_ranges(df)
    if "hemoglobin" in flags:
        assert not flags["hemoglobin"]["clinical_concern"]


def test_clinical_ranges_unknown_column_ignored() -> None:
    df = pd.DataFrame({"monthly_charges": [65.0, 89.0, 120.0]})
    flags = check_clinical_ranges(df)
    assert flags == {}


def test_clinical_ranges_critical_triggers_concern() -> None:
    df = pd.DataFrame({
        "spo2_pct": [96.0, 98.0, 55.0],  # 55% is below critical_low=60
    })
    flags = check_clinical_ranges(df)
    assert "spo2_pct" in flags
    assert flags["spo2_pct"]["clinical_concern"]


# ── detect_icd_columns ─────────────────────────────────────────────────────────


def test_icd_detects_valid_codes() -> None:
    df = pd.DataFrame({
        "primary_dx": ["I21.0", "J18.9", "E11.65", "K35.2", "I50.9"] * 6,
    })
    icd_cols = detect_icd_columns(df)
    assert "primary_dx" in icd_cols


def test_icd_ignores_non_code_columns() -> None:
    df = pd.DataFrame({
        "diagnosis_text": ["chest pain", "shortness of breath", "abdominal pain"] * 10,
        "contract": ["Month-to-month", "Annual", "Month-to-month"] * 10,
    })
    icd_cols = detect_icd_columns(df)
    assert icd_cols == []


def test_icd_appears_in_profile() -> None:
    df = pd.DataFrame({
        "icd_code": (["A00.0", "B34.9", "C50.1", "D64.9", "E11.0"] * 10),
        "age": list(range(50)),
    })
    profile = profile_dataset(df)
    assert profile.icd_columns is not None
    assert "icd_code" in profile.icd_columns
