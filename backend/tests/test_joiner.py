"""Unit tests for backend.ml.joiner and backend.ml.leakage_detector (§7)."""

import numpy as np
import pandas as pd
import pytest

from backend.ml.joiner import (
    JoinResult,
    SchemaCompatibilityResult,
    auto_detect_join_keys,
    check_schema_compatibility,
    join_datasets,
)
from backend.ml.leakage_detector import LeakageReport, detect_leakage


class TestSchemaCompatibility:
    def _base_df(self) -> pd.DataFrame:
        return pd.DataFrame({"id": [1, 2], "age": [30, 40], "region": ["A", "B"]})

    def test_identical_schemas_compatible(self) -> None:
        df = self._base_df()
        result = check_schema_compatibility(df, df.copy())
        assert result.compatible

    def test_new_column_is_info_not_error(self) -> None:
        df_ref = self._base_df()
        df_new = df_ref.copy()
        df_new["extra"] = 0
        result = check_schema_compatibility(df_ref, df_new)
        assert result.compatible  # new columns are OK
        assert "extra" in result.new_columns
        issues_by_type = {i.issue_type for i in result.issues}
        assert "new_col" in issues_by_type

    def test_missing_column_is_error(self) -> None:
        df_ref = self._base_df()
        df_new = df_ref.drop(columns=["age"])
        result = check_schema_compatibility(df_ref, df_new)
        assert not result.compatible
        assert "age" in result.missing_columns

    def test_target_column_excluded_from_required(self) -> None:
        df_ref = self._base_df()
        df_new = df_ref.drop(columns=["region"])
        result = check_schema_compatibility(df_ref, df_new, target_column="region")
        assert result.compatible  # target absence is fine for inference

    def test_int_float_mismatch_is_warning_not_error(self) -> None:
        df_ref = pd.DataFrame({"age": [30, 40]})
        df_new = pd.DataFrame({"age": [30.0, 40.0]})
        result = check_schema_compatibility(df_ref, df_new)
        assert result.compatible
        assert any(i.severity == "warning" and i.issue_type == "dtype_mismatch" for i in result.issues)

    def test_category_mismatch_is_warning(self) -> None:
        df_ref = pd.DataFrame({"region": ["A", "B", "C"]})
        df_new = pd.DataFrame({"region": ["A", "D", "E"]})  # D and E are new
        result = check_schema_compatibility(df_ref, df_new)
        assert any(i.issue_type == "category_mismatch" for i in result.issues)


class TestAutoDetectJoinKeys:
    def test_detects_high_cardinality_shared_column(self) -> None:
        ids = list(range(100))
        df_left = pd.DataFrame({"customer_id": ids, "age": range(100)})
        df_right = pd.DataFrame({"customer_id": ids, "plan": ["basic"] * 100})
        keys = auto_detect_join_keys(df_left, df_right)
        assert "customer_id" in keys

    def test_low_overlap_not_detected(self) -> None:
        df_left = pd.DataFrame({"customer_id": list(range(100))})
        df_right = pd.DataFrame({"customer_id": list(range(200, 300))})  # no overlap
        keys = auto_detect_join_keys(df_left, df_right)
        assert "customer_id" not in keys

    def test_low_cardinality_column_excluded(self) -> None:
        df_left = pd.DataFrame({"flag": [0, 1, 0, 1] * 25, "value": range(100)})
        df_right = pd.DataFrame({"flag": [0, 1, 0, 1] * 25, "price": range(100)})
        keys = auto_detect_join_keys(df_left, df_right)
        assert "flag" not in keys  # cardinality ratio too low


class TestJoinDatasets:
    def _left(self) -> pd.DataFrame:
        return pd.DataFrame({"id": [1, 2, 3], "age": [30, 40, 50]})

    def _right(self) -> pd.DataFrame:
        return pd.DataFrame({"id": [1, 2, 4], "plan": ["a", "b", "c"]})

    def test_inner_join_drops_unmatched(self) -> None:
        merged, result = join_datasets(self._left(), self._right(), "inner", ["id"])
        assert len(merged) == 2
        assert result.dropped_rows == 1
        assert result.join_type == "inner"

    def test_left_join_keeps_all_left(self) -> None:
        merged, result = join_datasets(self._left(), self._right(), "left", ["id"])
        assert len(merged) == 3
        assert result.dropped_rows == 0

    def test_outer_join_keeps_all_rows(self) -> None:
        merged, result = join_datasets(self._left(), self._right(), "outer", ["id"])
        assert len(merged) == 4

    def test_new_columns_tracked(self) -> None:
        _, result = join_datasets(self._left(), self._right(), "inner", ["id"])
        assert "plan" in result.new_columns_added

    def test_missing_join_key_raises(self) -> None:
        with pytest.raises(ValueError, match="Join key 'missing'"):
            join_datasets(self._left(), self._right(), "inner", ["missing"])

    def test_empty_join_keys_raises(self) -> None:
        with pytest.raises(ValueError, match="join_keys must not be empty"):
            join_datasets(self._left(), self._right(), "inner", [])


class TestLeakageDetector:
    def test_high_corr_feature_flagged(self) -> None:
        rng = np.random.default_rng(42)
        y = rng.integers(0, 2, 500).astype(float)
        df = pd.DataFrame({"target": y, "proxy": y + rng.normal(0, 0.001, 500)})
        report = detect_leakage(df, "target", high_corr_threshold=0.95)
        flagged_cols = [c.column for c in report.candidates]
        assert "proxy" in flagged_cols

    def test_uncorrelated_feature_not_flagged(self) -> None:
        rng = np.random.default_rng(1)
        y = rng.integers(0, 2, 500).astype(float)
        noise = rng.normal(0, 1, 500)
        df = pd.DataFrame({"target": y, "noise": noise})
        report = detect_leakage(df, "target", high_corr_threshold=0.95)
        flagged_cols = [c.column for c in report.candidates]
        assert "noise" not in flagged_cols

    def test_missing_target_returns_empty(self) -> None:
        df = pd.DataFrame({"a": [1, 2, 3]})
        report = detect_leakage(df, "missing_target")
        assert report.n_flagged == 0

    def test_pydantic_serialization(self) -> None:
        rng = np.random.default_rng(9)
        y = rng.integers(0, 2, 100).astype(float)
        df = pd.DataFrame({"target": y, "f1": y, "f2": rng.normal(0, 1, 100)})
        report = detect_leakage(df, "target")
        d = report.model_dump()
        LeakageReport.model_validate(d)
