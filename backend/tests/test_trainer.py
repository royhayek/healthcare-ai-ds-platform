"""Unit tests for ml/trainer.py - stability runs and stat test trigger."""

import numpy as np
import pandas as pd
import pytest

from backend.ml.cleaner import build_preprocessor, prepare_data, split_train_test
from backend.ml.trainer import (
    CLOSE_THRESHOLD,
    maybe_run_stat_test,
    train_with_stability,
    fit_final_pipeline,
)
from backend.models.strategy import (
    ColumnPreprocessingStrategy,
    PreprocessingStrategy,
    StabilityResult,
)


def _binary_df(n: int = 300) -> tuple[pd.DataFrame, pd.Series, PreprocessingStrategy]:
    rng = np.random.default_rng(42)
    df = pd.DataFrame({
        "x1": rng.normal(0, 1, n),
        "x2": rng.normal(2, 1, n),
        "x3": rng.choice(["A", "B", "C"], n),
        "y": (rng.normal(0, 1, n) > 0).astype(int),
    })
    strategy = PreprocessingStrategy(
        columns={
            "x1": ColumnPreprocessingStrategy(action="keep", dtype_hint="numeric",
                                               impute_strategy="median", scale_strategy="standard"),
            "x2": ColumnPreprocessingStrategy(action="keep", dtype_hint="numeric",
                                               impute_strategy="median", scale_strategy="standard"),
            "x3": ColumnPreprocessingStrategy(action="keep", dtype_hint="categorical",
                                               impute_strategy="most_frequent", encode_strategy="onehot"),
        },
        target_column="y",
        task_type="binary_classification",
    )
    X, y = prepare_data(df, strategy)
    return X, y, strategy


class TestStabilityResult:
    def test_mean_std_populated(self):
        X, y, strategy = _binary_df(n=200)
        X_tr, X_te, y_tr, y_te = split_train_test(X, y, "binary_classification")
        preprocessor = build_preprocessor(strategy, X_tr)

        result = train_with_stability(
            "logistic_regression", preprocessor, X_tr, y_tr,
            task_type="binary_classification",
            n_seeds=2, n_splits=3,
        )

        assert result.model_name == "logistic_regression"
        assert len(result.scores) == 2 * 3  # 2 seeds × 3 folds
        assert 0.0 <= result.mean <= 1.0
        assert result.std >= 0.0
        assert result.mean == pytest.approx(np.mean(result.scores), abs=1e-6)

    def test_overfit_gap_non_negative(self):
        X, y, strategy = _binary_df(n=200)
        X_tr, _, y_tr, _ = split_train_test(X, y, "binary_classification")
        preprocessor = build_preprocessor(strategy, X_tr)

        result = train_with_stability(
            "logistic_regression", preprocessor, X_tr, y_tr,
            task_type="binary_classification",
            n_seeds=2, n_splits=3,
        )
        assert result.overfit_gap >= 0.0


class TestGroupAwareCV:
    def test_grouped_cv_runs_and_returns_valid_result(self):
        # Many records per entity; grouped CV must keep an entity's rows together.
        X, y, strategy = _binary_df(n=300)
        X_tr, _, y_tr, _ = split_train_test(X, y, "binary_classification")
        preprocessor = build_preprocessor(strategy, X_tr)

        # 60 groups of ~4 records each, aligned to the training rows.
        groups = np.arange(len(X_tr)) % 60

        result = train_with_stability(
            "logistic_regression", preprocessor, X_tr, y_tr,
            task_type="binary_classification",
            n_seeds=2, n_splits=3,
            groups=groups,
        )
        assert len(result.scores) == 2 * 3
        assert 0.0 <= result.mean <= 1.0

    def test_grouped_splitter_keeps_groups_disjoint(self):
        from backend.ml.trainer import _make_cv_splitter

        rng = np.random.default_rng(0)
        n = 200
        X = pd.DataFrame({"x": rng.normal(0, 1, n)})
        y = pd.Series(rng.integers(0, 2, n))
        groups = np.arange(n) % 40  # 40 groups

        cv = _make_cv_splitter("binary_classification", n_splits=4, random_state=0, grouped=True)
        for train_idx, test_idx in cv.split(X, y, groups):
            train_groups = set(groups[train_idx])
            test_groups = set(groups[test_idx])
            assert train_groups.isdisjoint(test_groups)


class TestStatTestTrigger:
    def _make_stability(self, name: str, mean: float) -> StabilityResult:
        """Create a StabilityResult with a guaranteed exact mean for trigger testing."""
        return StabilityResult(
            model_name=name,
            scores=[mean] * 15,
            mean=mean,
            std=0.0,
        )

    def test_no_test_when_clearly_separated(self):
        a = self._make_stability("xgboost", 0.85)
        b = self._make_stability("lightgbm", 0.80)  # 0.05 gap > CLOSE_THRESHOLD
        result = maybe_run_stat_test([a, b], "binary_classification")
        assert result is None

    def test_test_runs_when_close(self):
        a = self._make_stability("xgboost", 0.8500)
        b = self._make_stability("lightgbm", 0.8504)  # within 0.005
        result = maybe_run_stat_test([a, b], "binary_classification")
        assert result is not None
        assert "p_value" in result
        assert "interpretation" in result
        assert 0.0 <= result["p_value"] <= 1.0

    def test_returns_none_with_single_candidate(self):
        a = self._make_stability("xgboost", 0.85)
        result = maybe_run_stat_test([a], "binary_classification")
        assert result is None

    def test_result_includes_model_names(self):
        a = self._make_stability("xgboost", 0.8500)
        b = self._make_stability("lightgbm", 0.8504)
        result = maybe_run_stat_test([a, b], "binary_classification")
        assert result is not None
        assert result["model_a"] == "xgboost"
        assert result["model_b"] == "lightgbm"


class TestFitFinalPipeline:
    def test_pipeline_predicts(self):
        X, y, strategy = _binary_df(n=200)
        X_tr, X_te, y_tr, y_te = split_train_test(X, y, "binary_classification")
        preprocessor = build_preprocessor(strategy, X_tr)

        pipeline = fit_final_pipeline(
            "logistic_regression", preprocessor, X_tr, y_tr, "binary_classification"
        )

        assert hasattr(pipeline, "predict_proba")
        proba = pipeline.predict_proba(X_te)
        assert proba.shape == (len(X_te), 2)
        assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)
