"""Unit tests for backend.ml.drift (§17)."""

import numpy as np
import pandas as pd
import pytest

from backend.ml.drift import (
    DriftReport,
    FeatureDriftResult,
    compute_drift_report,
    population_stability_index,
)


class TestPSI:
    def test_identical_distribution_returns_zero(self) -> None:
        rng = np.random.default_rng(42)
        data = rng.normal(0, 1, 1000)
        psi = population_stability_index(data, data.copy())
        assert psi < 0.01

    def test_shifted_distribution_returns_positive(self) -> None:
        rng = np.random.default_rng(42)
        expected = rng.normal(0, 1, 1000)
        actual = rng.normal(3, 1, 1000)  # large shift
        psi = population_stability_index(expected, actual)
        assert psi > 0.25

    def test_mild_shift_between_thresholds(self) -> None:
        rng = np.random.default_rng(42)
        expected = rng.normal(0, 1, 1000)
        actual = rng.normal(0.5, 1, 1000)
        psi = population_stability_index(expected, actual)
        assert 0.10 <= psi <= 0.25

    def test_degenerate_constant_column_returns_zero(self) -> None:
        constant = np.ones(100)
        psi = population_stability_index(constant, constant)
        assert psi == 0.0

    def test_empty_arrays_return_zero(self) -> None:
        psi = population_stability_index(np.array([]), np.array([]))
        assert psi == 0.0


class TestComputeDriftReport:
    def _make_frames(self, n: int = 500, shift: float = 0.0) -> tuple[pd.DataFrame, pd.DataFrame]:
        rng = np.random.default_rng(7)
        df_train = pd.DataFrame({
            "age": rng.normal(40, 10, n),
            "income": rng.normal(50000, 10000, n),
            "region": rng.choice(["north", "south", "east"], n),
        })
        df_new = pd.DataFrame({
            "age": rng.normal(40 + shift, 10, n),
            "income": rng.normal(50000 + shift * 2000, 10000, n),
            "region": rng.choice(["north", "south", "east"], n),
        })
        return df_train, df_new

    def test_no_drift_produces_stable(self) -> None:
        df_train, df_new = self._make_frames(shift=0.0)
        report = compute_drift_report(
            df_train, df_new,
            numeric_cols=["age", "income"],
            categorical_cols=["region"],
        )
        assert report.overall_severity in ("stable", "mild")

    def test_large_shift_produces_significant(self) -> None:
        df_train, df_new = self._make_frames(shift=5.0)
        report = compute_drift_report(
            df_train, df_new,
            numeric_cols=["age", "income"],
            categorical_cols=["region"],
        )
        assert report.overall_severity == "significant"
        assert len(report.significant_features) > 0
        assert report.warning is not None

    def test_report_contains_all_features(self) -> None:
        df_train, df_new = self._make_frames()
        report = compute_drift_report(
            df_train, df_new,
            numeric_cols=["age", "income"],
            categorical_cols=["region"],
        )
        feature_names = [f.feature for f in report.features]
        assert "age" in feature_names
        assert "income" in feature_names
        assert "region" in feature_names

    def test_numeric_features_have_psi_and_ks(self) -> None:
        df_train, df_new = self._make_frames()
        report = compute_drift_report(
            df_train, df_new,
            numeric_cols=["age"],
            categorical_cols=[],
        )
        age_result = next(f for f in report.features if f.feature == "age")
        assert age_result.psi is not None
        assert age_result.ks_statistic is not None
        assert age_result.wasserstein is not None

    def test_categorical_features_have_chi2_and_js(self) -> None:
        df_train, df_new = self._make_frames()
        report = compute_drift_report(
            df_train, df_new,
            numeric_cols=[],
            categorical_cols=["region"],
        )
        region_result = next(f for f in report.features if f.feature == "region")
        assert region_result.chi2 is not None
        assert region_result.js_divergence is not None

    def test_missing_columns_in_new_data_are_skipped(self) -> None:
        df_train, df_new = self._make_frames()
        df_new_partial = df_new.drop(columns=["income"])
        report = compute_drift_report(
            df_train, df_new_partial,
            numeric_cols=["age", "income"],
            categorical_cols=[],
        )
        feature_names = [f.feature for f in report.features]
        assert "age" in feature_names
        assert "income" not in feature_names

    def test_row_counts_recorded(self) -> None:
        df_train, df_new = self._make_frames(n=300)
        df_new_small = df_new.iloc[:100]
        report = compute_drift_report(df_train, df_new_small, numeric_cols=["age"], categorical_cols=[])
        assert report.n_train_rows == 300
        assert report.n_new_rows == 100

    def test_pydantic_serialization(self) -> None:
        df_train, df_new = self._make_frames()
        report = compute_drift_report(
            df_train, df_new,
            numeric_cols=["age"],
            categorical_cols=["region"],
        )
        d = report.model_dump()
        assert "overall_severity" in d
        assert "features" in d
        # Round-trip
        DriftReport.model_validate(d)
