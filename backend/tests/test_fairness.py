"""Unit tests for backend.ml.fairness (§19)."""

import numpy as np
import pytest

from backend.ml.fairness import (
    FairnessReport,
    _classify_severity,
    build_sensitive_features,
    fairness_audit,
)
import pandas as pd


class TestClassifySeverity:
    def test_below_5pct_is_none(self) -> None:
        assert _classify_severity(0.03) == "none"
        assert _classify_severity(-0.04) == "none"

    def test_5_to_10_is_mild(self) -> None:
        assert _classify_severity(0.07) == "mild"

    def test_10_to_20_is_moderate(self) -> None:
        assert _classify_severity(0.15) == "moderate"

    def test_above_20_is_severe(self) -> None:
        assert _classify_severity(0.25) == "severe"
        assert _classify_severity(-0.21) == "severe"


class TestFairnessAudit:
    def _perfect_predictions(self, n: int = 200) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        rng = np.random.default_rng(1)
        y_true = rng.integers(0, 2, n)
        return y_true, y_true.copy(), y_true.astype(float)

    def _biased_predictions(
        self, n: int = 200
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Group 0 gets perfect predictions; group 1 gets random predictions."""
        rng = np.random.default_rng(42)
        y_true = rng.integers(0, 2, n)
        groups = (np.arange(n) % 2).astype(int)
        y_pred = y_true.copy()
        # Degrade group 1 predictions significantly
        group1_idx = np.where(groups == 1)[0]
        y_pred[group1_idx] = rng.integers(0, 2, len(group1_idx))
        y_proba = y_pred.astype(float)
        return y_true, y_pred, y_proba, groups

    def test_empty_sensitive_features_returns_empty_report(self) -> None:
        y_true, y_pred, y_proba = self._perfect_predictions()
        report = fairness_audit(y_true, y_pred, y_proba, {})
        assert len(report.attributes) == 0
        assert report.overall_severity == "none"

    def test_perfectly_fair_model_has_low_disparity(self) -> None:
        rng = np.random.default_rng(99)
        n = 400
        y_true = rng.integers(0, 2, n)
        groups = (np.arange(n) % 2).astype(int)
        # Perfect model: no disparity
        report = fairness_audit(y_true, y_true.copy(), y_true.astype(float), {"gender": groups})
        assert report.overall_severity in ("none", "mild")
        assert not report.blocks_deliverables

    def test_biased_model_detects_disparity(self) -> None:
        y_true, y_pred, y_proba, groups = self._biased_predictions(n=400)
        report = fairness_audit(y_true, y_pred, y_proba, {"gender": groups})
        assert len(report.attributes) == 1
        attr = report.attributes[0]
        assert attr.attribute == "gender"
        assert attr.demographic_parity_diff != 0.0

    def test_severe_disparity_blocks_deliverables(self) -> None:
        rng = np.random.default_rng(77)
        n = 400
        y_true = rng.integers(0, 2, n)
        groups = (np.arange(n) % 2).astype(int)
        # Group 1 always predicted 0
        y_pred = y_true.copy()
        y_pred[groups == 1] = 0
        report = fairness_audit(y_true, y_pred, None, {"age_group": groups})
        if report.overall_severity == "severe":
            assert report.blocks_deliverables

    def test_intersectional_analysis_with_two_attributes(self) -> None:
        rng = np.random.default_rng(55)
        n = 400
        y_true = rng.integers(0, 2, n)
        gender = (np.arange(n) % 2).astype(int)
        age_group = (np.arange(n) % 3).astype(int)
        report = fairness_audit(
            y_true, y_true.copy(), y_true.astype(float),
            {"gender": gender, "age_group": age_group},
        )
        assert len(report.attributes) == 2
        assert len(report.intersectional) == 1
        assert "gender × age_group" in report.intersectional[0].attribute

    def test_no_intersectional_for_single_attribute(self) -> None:
        rng = np.random.default_rng(3)
        n = 200
        y_true = rng.integers(0, 2, n)
        groups = (np.arange(n) % 2).astype(int)
        report = fairness_audit(y_true, y_true.copy(), None, {"gender": groups})
        assert len(report.intersectional) == 0

    def test_by_group_metrics_present(self) -> None:
        rng = np.random.default_rng(4)
        n = 200
        y_true = rng.integers(0, 2, n)
        groups = (np.arange(n) % 2).astype(int)
        report = fairness_audit(y_true, y_true.copy(), y_true.astype(float), {"gender": groups})
        attr = report.attributes[0]
        assert len(attr.by_group) == 2
        group_names = {g.group for g in attr.by_group}
        assert "0" in group_names
        assert "1" in group_names

    def test_pydantic_serialization(self) -> None:
        rng = np.random.default_rng(5)
        n = 200
        y_true = rng.integers(0, 2, n)
        groups = (np.arange(n) % 2).astype(int)
        report = fairness_audit(y_true, y_true.copy(), None, {"g": groups})
        d = report.model_dump()
        assert "attributes" in d
        FairnessReport.model_validate(d)


class TestBuildSensitiveFeatures:
    def test_extracts_columns(self) -> None:
        df = pd.DataFrame({"gender": ["M", "F", "M", "F"], "age": [25, 30, 35, 40]})
        result = build_sensitive_features(df, ["gender"])
        assert "gender" in result
        assert list(result["gender"]) == ["M", "F", "M", "F"]

    def test_missing_column_is_skipped(self) -> None:
        df = pd.DataFrame({"gender": ["M", "F"]})
        result = build_sensitive_features(df, ["gender", "nonexistent"])
        assert "gender" in result
        assert "nonexistent" not in result

    def test_respects_index_filter(self) -> None:
        df = pd.DataFrame(
            {"gender": ["M", "F", "M", "F"]},
            index=[10, 11, 12, 13],
        )
        result = build_sensitive_features(df, ["gender"], index=pd.Index([10, 12]))
        assert list(result["gender"]) == ["M", "M"]
