"""Unit tests for ml/leakage_detector.py.

Covers the domain-agnostic guardrails that prevent misleadingly-perfect accuracy:
proxy (lookup-table) leakage for binary AND multiclass targets, the identifier
guard, numeric high-correlation leakage, and unlabeled-target detection.
"""

import numpy as np
import pandas as pd

from backend.ml.leakage_detector import (
    detect_leakage,
    detect_unlabeled_target_classes,
)


def _rng(seed: int = 0) -> np.random.RandomState:
    return np.random.RandomState(seed)


class TestProxyLeakage:
    def test_multiclass_categorical_proxy_is_flagged(self):
        # A categorical feature that maps near 1:1 onto a multiclass target -
        # the clade→pathogenicity / diagnosis_code→disease pattern.
        rng = _rng()
        family = rng.choice(["A", "B", "C", "D"], 400)
        proxy_map = {"A": "high", "B": "high", "C": "low", "D": "moderate"}
        target = pd.Series([proxy_map[f] for f in family])
        df = pd.DataFrame({
            "family": family,
            "noise": rng.normal(0, 1, 400),
            "label": target,
        })
        report = detect_leakage(df, "label")
        flagged = {c.column for c in report.candidates}
        assert "family" in flagged
        fam = next(c for c in report.candidates if c.column == "family")
        assert fam.severity == "high"
        assert "noise" not in flagged

    def test_binary_categorical_proxy_is_flagged(self):
        rng = _rng(1)
        unit = rng.choice(["icu", "ward", "stepdown"], 300)
        proxy_map = {"icu": 1, "ward": 0, "stepdown": 0}
        df = pd.DataFrame({
            "discharge_unit": unit,
            "age": rng.randint(20, 90, 300),
            "y": [proxy_map[u] for u in unit],
        })
        report = detect_leakage(df, "y")
        assert "discharge_unit" in {c.column for c in report.candidates}

    def test_identifier_column_not_flagged_as_proxy(self):
        # A near-unique ID column trivially has purity 1.0 but is NOT leakage.
        rng = _rng(2)
        n = 300
        df = pd.DataFrame({
            "patient_id": [f"P{i:05d}" for i in range(n)],
            "x": rng.normal(0, 1, n),
            "y": rng.choice([0, 1], n),
        })
        report = detect_leakage(df, "y")
        assert "patient_id" not in {c.column for c in report.candidates}

    def test_clean_dataset_flags_nothing(self):
        rng = _rng(3)
        n = 400
        df = pd.DataFrame({
            "age": rng.randint(20, 80, n),
            "bp": rng.normal(120, 15, n),
            "sex": rng.choice(["M", "F"], n),
            "y": rng.choice([0, 1], n),
        })
        report = detect_leakage(df, "y")
        assert report.n_flagged == 0

    def test_numeric_high_correlation_still_flagged(self):
        rng = _rng(4)
        n = 300
        y = rng.choice([0, 1], n)
        df = pd.DataFrame({
            "leaky_score": y + rng.normal(0, 0.001, n),  # ~perfect numeric copy
            "x": rng.normal(0, 1, n),
            "y": y,
        })
        report = detect_leakage(df, "y")
        assert "leaky_score" in {c.column for c in report.candidates}

    def test_missing_target_returns_empty(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        report = detect_leakage(df, "nonexistent")
        assert report.n_flagged == 0


class TestUnlabeledTarget:
    def test_unknown_placeholder_detected(self):
        y = pd.Series(["high"] * 50 + ["low"] * 30 + ["unknown"] * 20)
        report = detect_unlabeled_target_classes(y)
        assert "unknown" in report.suspicious_classes
        assert report.affected_rows == 20
        assert report.total_rows == 100
        assert report.recommendation

    def test_multiple_placeholder_tokens_detected(self):
        y = pd.Series(["yes"] * 40 + ["no"] * 40 + ["pending"] * 10 + ["N/A"] * 10)
        report = detect_unlabeled_target_classes(y)
        assert set(report.suspicious_classes) == {"pending", "N/A"}
        assert report.affected_rows == 20

    def test_genuine_labels_not_flagged(self):
        y = pd.Series(["high"] * 40 + ["moderate"] * 30 + ["low"] * 30)
        report = detect_unlabeled_target_classes(y)
        assert report.suspicious_classes == []
        assert report.affected_rows == 0

    def test_numeric_target_not_flagged(self):
        y = pd.Series([0, 1, 0, 1, 1, 0] * 20)
        report = detect_unlabeled_target_classes(y)
        assert report.suspicious_classes == []
