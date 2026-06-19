"""Unit tests for ml/cleaner.py."""

import numpy as np
import pandas as pd
import pytest

from backend.ml.cleaner import (
    apply_preprocessor,
    apply_target_hygiene,
    build_preprocessor,
    prepare_data,
    resolve_task_type,
    split_cal_val,
    split_train_test,
)
from backend.models.strategy import (
    ColumnPreprocessingStrategy,
    PreprocessingStrategy,
    TargetStrategy,
)


class TestTargetHygiene:
    """Target-level hygiene: drop unlabelled rows + collapse to binary (§7, §10)."""

    def _df(self):
        return pd.DataFrame({
            "f1": np.arange(8, dtype=float),
            "pathogenicity_class": [
                "high", "low", "moderate", "unknown",
                "high", "unknown", "low", "high",
            ],
        })

    def test_drop_unlabelled_rows(self):
        out = apply_target_hygiene(
            self._df(), "pathogenicity_class", TargetStrategy(drop_labels=["unknown"])
        )
        assert len(out) == 6
        assert "unknown" not in out["pathogenicity_class"].tolist()

    def test_collapse_to_binary_high_vs_rest(self):
        # Drop unknown, then collapse: high → 1, low/moderate → 0.
        out = apply_target_hygiene(
            self._df(), "pathogenicity_class",
            TargetStrategy(drop_labels=["unknown"], positive_labels=["high"]),
        )
        assert set(out["pathogenicity_class"].unique()) == {0, 1}
        assert out["pathogenicity_class"].sum() == 3  # three "high" rows

    def test_case_insensitive_matching(self):
        df = self._df()
        df.loc[0, "pathogenicity_class"] = "HIGH"
        out = apply_target_hygiene(
            df, "pathogenicity_class",
            TargetStrategy(drop_labels=["UNKNOWN"], positive_labels=["High"]),
        )
        assert "unknown" not in [str(v).lower() for v in out["pathogenicity_class"]]
        assert out["pathogenicity_class"].sum() == 3

    def test_empty_strategy_is_noop(self):
        df = self._df()
        out = apply_target_hygiene(df, "pathogenicity_class", TargetStrategy())
        assert out.equals(df)
        assert apply_target_hygiene(df, "pathogenicity_class", None).equals(df)

    def test_accepts_dict(self):
        out = apply_target_hygiene(
            self._df(), "pathogenicity_class", {"drop_labels": ["unknown"]}
        )
        assert len(out) == 6

    def test_positive_label_no_match_skips_collapse(self):
        # A positive label that matches nothing must NOT produce an all-zero target.
        df = self._df()
        out = apply_target_hygiene(
            df, "pathogenicity_class", TargetStrategy(positive_labels=["nonexistent"])
        )
        assert out["pathogenicity_class"].tolist() == df["pathogenicity_class"].tolist()

    def test_prepare_data_applies_hygiene(self):
        df = self._df()
        strategy = PreprocessingStrategy(
            columns={"f1": ColumnPreprocessingStrategy(action="keep", dtype_hint="numeric")},
            target_column="pathogenicity_class",
            task_type="binary_classification",
        )
        X, y = prepare_data(
            df, strategy,
            target_strategy=TargetStrategy(drop_labels=["unknown"], positive_labels=["high"]),
        )
        assert len(y) == 6
        assert set(y.unique()) == {0, 1}


class TestTaskTypeResolution:
    """Regression: a string-dtype target the profiler left as task_type='unknown'
    must not reach the model as string labels ('could not convert string to
    float: \"unknown\"')."""

    def _df(self):
        return pd.DataFrame({
            "f1": np.arange(40, dtype=float),
            "cat": (["a", "b"] * 20),
            "target": (["high", "low", "moderate", "unknown"] * 10),
        })

    def _strategy(self, task_type="unknown"):
        return PreprocessingStrategy(
            columns={
                "f1": ColumnPreprocessingStrategy(action="keep", dtype_hint="numeric"),
                "cat": ColumnPreprocessingStrategy(action="keep", dtype_hint="categorical"),
            },
            target_column="target",
            task_type=task_type,
        )

    def test_resolve_unknown_to_multiclass(self):
        df = self._df()
        assert resolve_task_type("unknown", df["target"]) == "multiclass"

    def test_resolve_binary(self):
        s = pd.Series(["yes", "no"] * 10)
        assert resolve_task_type(None, s) == "binary_classification"

    def test_resolve_passthrough_valid(self):
        assert resolve_task_type("regression", pd.Series([1.0, 2.0])) == "regression"

    def test_prepare_data_encodes_target_when_task_type_unknown(self):
        df = self._df()
        X, y = prepare_data(df, self._strategy("unknown"))
        # Target must be integer-encoded despite the 'unknown' task_type.
        assert pd.api.types.is_integer_dtype(y)
        assert set(y) == {0, 1, 2, 3}

    def test_string_target_does_not_crash_training_path(self):
        df = self._df()
        strat = self._strategy("unknown").model_copy(
            update={"task_type": resolve_task_type("unknown", df["target"])}
        )
        X, y = prepare_data(df, strat)
        pre = build_preprocessor(strat, X)
        X_tr, X_te, _, _ = split_train_test(X, y, strat.task_type)
        Xt, _ = apply_preprocessor(pre, X_tr, X_te)
        assert Xt.dtype == np.float32


def _make_strategy(target: str = "y", task_type: str = "binary_classification") -> PreprocessingStrategy:
    return PreprocessingStrategy(
        columns={
            "age": ColumnPreprocessingStrategy(
                action="keep", dtype_hint="numeric",
                impute_strategy="median", scale_strategy="standard"
            ),
            "income": ColumnPreprocessingStrategy(
                action="keep", dtype_hint="numeric",
                impute_strategy="mean", scale_strategy="standard"
            ),
            "category": ColumnPreprocessingStrategy(
                action="keep", dtype_hint="categorical",
                impute_strategy="most_frequent", encode_strategy="onehot"
            ),
            "drop_me": ColumnPreprocessingStrategy(action="drop"),
        },
        target_column=target,
        task_type=task_type,
    )


def _make_df(n: int = 200, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "age": rng.integers(18, 80, n).astype(float),
        "income": rng.normal(50000, 20000, n),
        "category": rng.choice(["A", "B", "C"], n),
        "drop_me": rng.random(n),
        "y": rng.integers(0, 2, n),
    })


class TestNumericPlaceholderCoercion:
    """Regression: a numeric-hinted column carrying 'unknown' placeholders must
    not crash training with "could not convert string to float: 'unknown'"."""

    def _df_with_unknown(self, n: int = 200, seed: int = 1) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        # collection_year is semantically numeric but stored as strings with
        # 'unknown' placeholders for missing entries.
        years = rng.integers(1990, 2024, n).astype(str).astype(object)
        years[rng.random(n) < 0.2] = "unknown"
        return pd.DataFrame({
            "collection_year": years,
            "income": rng.normal(50000, 20000, n),
            "category": rng.choice(["A", "B", "C"], n),
            "y": rng.integers(0, 2, n),
        })

    def _strategy(self) -> PreprocessingStrategy:
        return PreprocessingStrategy(
            columns={
                "collection_year": ColumnPreprocessingStrategy(
                    action="keep", dtype_hint="numeric",
                    impute_strategy="median", scale_strategy="standard",
                ),
                "income": ColumnPreprocessingStrategy(
                    action="keep", dtype_hint="numeric",
                    impute_strategy="mean", scale_strategy="standard",
                ),
                "category": ColumnPreprocessingStrategy(
                    action="keep", dtype_hint="categorical",
                    impute_strategy="most_frequent", encode_strategy="onehot",
                ),
            },
            target_column="y",
            task_type="binary_classification",
        )

    def test_numeric_column_with_unknown_does_not_crash(self):
        df = self._df_with_unknown()
        strategy = self._strategy()
        X, y = prepare_data(df, strategy)
        pre = build_preprocessor(strategy, X)
        X_tr, X_te, y_tr, y_te = split_train_test(X, y, "binary_classification")
        # Must not raise "could not convert string to float: 'unknown'".
        Xt_tr, Xt_te = apply_preprocessor(pre, X_tr, X_te)
        assert Xt_tr.dtype == np.float32
        assert not np.isnan(Xt_tr).any()  # placeholders coerced→NaN→imputed

    def test_genuinely_categorical_column_hinted_numeric_is_demoted(self):
        # A truly categorical column wrongly hinted numeric should be encoded,
        # not coerced to all-NaN.
        rng = np.random.default_rng(0)
        df = pd.DataFrame({
            "host": rng.choice(["mouse", "rat", "unknown"], 150),
            "income": rng.normal(50000, 20000, 150),
            "y": rng.integers(0, 2, 150),
        })
        strategy = PreprocessingStrategy(
            columns={
                "host": ColumnPreprocessingStrategy(
                    action="keep", dtype_hint="numeric",  # wrong hint
                    impute_strategy="median", scale_strategy="standard",
                ),
                "income": ColumnPreprocessingStrategy(
                    action="keep", dtype_hint="numeric",
                    impute_strategy="mean", scale_strategy="standard",
                ),
            },
            target_column="y",
            task_type="binary_classification",
        )
        X, y = prepare_data(df, strategy)
        pre = build_preprocessor(strategy, X)
        X_tr, X_te, _, _ = split_train_test(X, y, "binary_classification")
        Xt_tr, _ = apply_preprocessor(pre, X_tr, X_te)
        # host one-hot expands columns; result is finite floats, no crash.
        assert Xt_tr.dtype == np.float32
        assert Xt_tr.shape[1] >= 3


class TestPrepareData:
    def test_removes_target(self):
        df = _make_df()
        strategy = _make_strategy()
        X, y = prepare_data(df, strategy)
        assert "y" not in X.columns
        assert "y" == y.name

    def test_drops_tagged_columns(self):
        df = _make_df()
        strategy = _make_strategy()
        X, _ = prepare_data(df, strategy)
        assert "drop_me" not in X.columns

    def test_keeps_feature_columns(self):
        df = _make_df()
        strategy = _make_strategy()
        X, _ = prepare_data(df, strategy)
        assert "age" in X.columns
        assert "income" in X.columns
        assert "category" in X.columns

    def test_missing_target_raises(self):
        df = _make_df().drop(columns=["y"])
        strategy = _make_strategy()
        with pytest.raises(ValueError, match="Target column"):
            prepare_data(df, strategy)

    def test_drops_all_null_rows(self):
        df = _make_df(n=100)
        # Make first row all-null in features
        df.loc[0, ["age", "income", "category", "drop_me"]] = np.nan
        strategy = _make_strategy()
        X, y = prepare_data(df, strategy)
        assert len(X) == 99


class TestSplitTrainTest:
    def test_correct_sizes_classification(self):
        df = _make_df(n=200)
        strategy = _make_strategy()
        X, y = prepare_data(df, strategy)
        X_tr, X_te, y_tr, y_te = split_train_test(X, y, "binary_classification", test_size=0.2)
        assert len(X_tr) == 160
        assert len(X_te) == 40

    def test_correct_sizes_regression(self):
        df = _make_df(n=100)
        strategy = _make_strategy(task_type="regression")
        X, y = prepare_data(df, strategy)
        X_tr, X_te, y_tr, y_te = split_train_test(X, y, "regression", test_size=0.2)
        assert len(X_tr) == 80
        assert len(X_te) == 20


class TestSplitCalVal:
    def test_three_way_split(self):
        df = _make_df(n=500)
        strategy = _make_strategy()
        X, y = prepare_data(df, strategy)
        X_tr, X_te, y_tr, y_te = split_train_test(X, y, "binary_classification")
        X_fit, X_cal, X_val, y_fit, y_cal, y_val = split_cal_val(X_tr, y_tr, "binary_classification")

        # X_fit ≈ 60%, X_cal ≈ 20%, X_val ≈ 20% of X_train
        total = len(X_tr)
        assert abs(len(X_fit) - int(0.6 * total)) <= 5
        assert abs(len(X_cal) - int(0.2 * total)) <= 5
        assert abs(len(X_val) - int(0.2 * total)) <= 5
        assert len(X_fit) + len(X_cal) + len(X_val) == total

    def test_no_overlap(self):
        df = _make_df(n=500)
        strategy = _make_strategy()
        X, y = prepare_data(df, strategy)
        X_tr, X_te, y_tr, y_te = split_train_test(X, y, "binary_classification")
        X_fit, X_cal, X_val, y_fit, y_cal, y_val = split_cal_val(X_tr, y_tr, "binary_classification")

        fit_idx = set(X_fit.index)
        cal_idx = set(X_cal.index)
        val_idx = set(X_val.index)
        assert len(fit_idx & cal_idx) == 0
        assert len(fit_idx & val_idx) == 0
        assert len(cal_idx & val_idx) == 0


class TestBuildPreprocessor:
    def test_returns_column_transformer(self):
        from sklearn.compose import ColumnTransformer
        df = _make_df()
        strategy = _make_strategy()
        X, _ = prepare_data(df, strategy)
        prep = build_preprocessor(strategy, X)
        assert isinstance(prep, ColumnTransformer)

    def test_transforms_without_error(self):
        df = _make_df(n=200)
        strategy = _make_strategy()
        X, y = prepare_data(df, strategy)
        X_tr, X_te, y_tr, y_te = split_train_test(X, y, "binary_classification")
        prep = build_preprocessor(strategy, X_tr)
        X_tr_t, X_te_t = apply_preprocessor(prep, X_tr, X_te)
        assert X_tr_t.shape[0] == len(X_tr)
        assert X_te_t.shape[0] == len(X_te)
        assert not np.any(np.isnan(X_tr_t))
        assert not np.any(np.isnan(X_te_t))

    def test_no_test_leakage(self):
        df = _make_df(n=200)
        strategy = _make_strategy()
        X, y = prepare_data(df, strategy)
        X_tr, X_te, y_tr, y_te = split_train_test(X, y, "binary_classification")

        # Introduce a strong outlier in test only
        X_te_mod = X_te.copy()
        X_te_mod.iloc[0, 0] = 1e9  # extreme outlier

        prep = build_preprocessor(strategy, X_tr)
        X_tr_t, X_te_t = apply_preprocessor(prep, X_tr, X_te_mod)

        # Training set should not be affected by test outlier
        assert X_tr_t.max() < 100  # standard-scaled values should be reasonable
