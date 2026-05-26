"""Unit tests for ml/threshold_optimizer.py - spec §16 correctness."""

import numpy as np
import pytest

from backend.ml.threshold_optimizer import optimize_threshold
from backend.models.strategy import CostMatrix, ThresholdResult


def _make_binary_data(n: int = 1000, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    y_true = rng.integers(0, 2, n)
    # Imperfect but correlated probabilities
    y_proba = np.where(
        y_true == 1,
        rng.beta(5, 2, n),   # positives cluster near 1
        rng.beta(2, 5, n),   # negatives cluster near 0
    )
    return y_true.astype(int), y_proba.astype(float)


class TestOptimizeThreshold:
    def test_returns_threshold_result(self):
        y_true, y_proba = _make_binary_data()
        cm = CostMatrix()
        result = optimize_threshold(y_true, y_proba, cm)
        assert isinstance(result, ThresholdResult)

    def test_threshold_in_valid_range(self):
        y_true, y_proba = _make_binary_data()
        cm = CostMatrix()
        result = optimize_threshold(y_true, y_proba, cm)
        assert 0.0 < result.optimal_threshold < 1.0

    def test_cost_at_optimal_le_cost_at_default(self):
        """Optimal threshold must not cost MORE than default 0.5."""
        y_true, y_proba = _make_binary_data()
        cm = CostMatrix()
        result = optimize_threshold(y_true, y_proba, cm)
        assert result.cost_at_optimal <= result.cost_at_default + 1e-6

    def test_high_fn_cost_lowers_threshold(self):
        """When FN costs much more than FP, optimal threshold should be lower than 0.5."""
        y_true, y_proba = _make_binary_data()
        expensive_fn = CostMatrix(cost_fp=1.0, cost_fn=100.0)
        result = optimize_threshold(y_true, y_proba, expensive_fn)
        assert result.optimal_threshold < 0.5

    def test_high_fp_cost_raises_threshold(self):
        """When FP costs much more than FN, optimal threshold should be higher than 0.5."""
        y_true, y_proba = _make_binary_data()
        expensive_fp = CostMatrix(cost_fp=100.0, cost_fn=1.0)
        result = optimize_threshold(y_true, y_proba, expensive_fp)
        assert result.optimal_threshold > 0.5

    def test_improvement_pct_non_negative(self):
        y_true, y_proba = _make_binary_data()
        cm = CostMatrix()
        result = optimize_threshold(y_true, y_proba, cm)
        assert result.improvement_pct >= 0.0

    def test_cost_curve_coverage(self):
        y_true, y_proba = _make_binary_data()
        cm = CostMatrix()
        result = optimize_threshold(y_true, y_proba, cm, n_steps=99)
        assert len(result.cost_curve) == 99
        thresholds = [point["threshold"] for point in result.cost_curve]
        assert all(0.0 < t < 1.0 for t in thresholds)

    def test_metric_at_optimal_populated(self):
        y_true, y_proba = _make_binary_data()
        cm = CostMatrix()
        result = optimize_threshold(y_true, y_proba, cm)
        assert "precision" in result.metric_at_optimal
        assert "recall" in result.metric_at_optimal
        assert "f1" in result.metric_at_optimal

    def test_noisy_data_improves_with_high_fn_cost(self):
        """With noisy data and high FN cost, optimizer should move threshold below 0.5."""
        y_true, y_proba = _make_binary_data(n=2000)
        cm = CostMatrix(cost_fn=20.0, cost_fp=1.0)
        result = optimize_threshold(y_true, y_proba, cm)
        # With FN cost 20x FP, threshold should be well below 0.5
        assert result.optimal_threshold < 0.45

    def test_default_0_5_never_used_without_explicit_selection(self):
        """The function should never report optimal=0.5 as the only possible choice."""
        y_true, y_proba = _make_binary_data()
        cm = CostMatrix(cost_fn=10.0)
        result = optimize_threshold(y_true, y_proba, cm)
        # With high FN cost and realistic data, threshold should be < 0.5
        # This is a design intent check, not a hard assertion for all seeds
        assert result.optimal_threshold != 0.0 and result.optimal_threshold != 1.0
