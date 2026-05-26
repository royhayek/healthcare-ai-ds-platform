"""Unit tests for ml/stat_tests.py."""

import numpy as np
import pytest

from backend.ml.stat_tests import mcnemar_test, paired_t_test, run_comparison_test


class TestMcNemar:
    def test_identical_predictions_p1(self):
        y_true = np.array([0, 1, 0, 1, 1])
        y_pred = np.array([0, 1, 0, 1, 1])
        result = mcnemar_test(y_true, y_pred, y_pred)
        assert result["p_value"] == 1.0
        assert result["b"] == 0
        assert result["c"] == 0

    def test_completely_different_predictions(self):
        n = 1000
        rng = np.random.default_rng(42)
        y_true = rng.integers(0, 2, n)
        y_pred_a = (rng.random(n) > 0.5).astype(int)
        y_pred_b = 1 - y_pred_a  # perfectly inverted
        result = mcnemar_test(y_true, y_pred_a, y_pred_b)
        assert 0.0 <= result["p_value"] <= 1.0
        assert result["b"] + result["c"] > 0

    def test_returns_required_keys(self):
        y = np.array([0, 1, 0, 1])
        result = mcnemar_test(y, y, 1 - y)
        assert "statistic" in result
        assert "p_value" in result
        assert "b" in result
        assert "c" in result

    def test_significant_difference(self):
        # Model A always correct, Model B always wrong
        n = 200
        y_true = np.ones(n, dtype=int)
        y_pred_a = np.ones(n, dtype=int)
        y_pred_b = np.zeros(n, dtype=int)
        result = mcnemar_test(y_true, y_pred_a, y_pred_b)
        assert result["p_value"] < 0.001


class TestPairedT:
    def test_same_scores_p1(self):
        scores = [0.8, 0.81, 0.79, 0.80, 0.82]
        result = paired_t_test(scores, scores)
        assert result["p_value"] == pytest.approx(1.0)
        assert result["mean_diff"] == pytest.approx(0.0)

    def test_significant_difference(self):
        # Differences are non-constant so ttest_rel runs properly
        a = [0.90, 0.88, 0.92, 0.87, 0.91, 0.89, 0.93, 0.86, 0.91, 0.90]
        b = [0.60, 0.65, 0.58, 0.70, 0.62, 0.55, 0.68, 0.63, 0.61, 0.66]
        result = paired_t_test(a, b)
        assert result["p_value"] < 0.001
        assert result["mean_diff"] > 0

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="length mismatch"):
            paired_t_test([0.8, 0.9], [0.8])

    def test_single_element_returns_p1(self):
        result = paired_t_test([0.8], [0.9])
        assert result["p_value"] == 1.0


class TestRunComparisonTest:
    def test_binary_uses_paired_t_without_predictions(self):
        result = run_comparison_test(
            task_type="binary_classification",
            scores_a=[0.85, 0.86, 0.84],
            scores_b=[0.85, 0.86, 0.84],
        )
        assert result["test_name"] == "paired_t"

    def test_regression_uses_paired_t(self):
        result = run_comparison_test(
            task_type="regression",
            scores_a=[0.8, 0.82, 0.79],
            scores_b=[0.7, 0.72, 0.68],
        )
        assert result["test_name"] == "paired_t"

    def test_binary_with_predictions_uses_mcnemar(self):
        y_true = np.array([0, 1, 0, 1, 0, 1, 1, 0])
        y_a = np.array([0, 1, 0, 1, 0, 1, 1, 0])
        y_b = np.array([1, 0, 1, 0, 1, 0, 0, 1])
        result = run_comparison_test(
            task_type="binary_classification",
            scores_a=[0.85], scores_b=[0.84],
            y_true=y_true, y_pred_a=y_a, y_pred_b=y_b,
        )
        assert result["test_name"] == "mcnemar"

    def test_interpretation_present(self):
        result = run_comparison_test(
            task_type="regression",
            scores_a=[0.8, 0.82, 0.79],
            scores_b=[0.8, 0.82, 0.79],
        )
        assert "interpretation" in result
        assert isinstance(result["interpretation"], str)
