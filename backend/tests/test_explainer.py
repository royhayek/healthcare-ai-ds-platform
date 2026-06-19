"""Tests for ml/explainer.py - focus on multiclass SHAP aggregation.

Regression: multiclass SHAP returns a 3D array (n_samples, n_features,
n_classes) on modern SHAP; a per-class axis must not survive into the summary
or downstream rounding crashes ("type list doesn't define __round__").
"""

import numpy as np

from backend.ml.explainer import _mean_abs_per_feature


def test_mean_abs_2d_binary():
    sv = np.array([[1.0, -2.0], [3.0, -4.0]])  # (2 samples, 2 features)
    out = _mean_abs_per_feature(sv)
    assert out.shape == (2,)
    assert np.allclose(out, [2.0, 3.0])


def test_mean_abs_3d_multiclass():
    # (2 samples, 2 features, 3 classes) - newer SHAP API.
    sv = np.abs(np.arange(12, dtype=float)).reshape(2, 2, 3)
    out = _mean_abs_per_feature(sv)
    assert out.shape == (2,)  # one scalar per feature, classes collapsed
    assert out.ndim == 1


def test_mean_abs_list_of_arrays_multiclass():
    # Older SHAP API: list of (n_samples, n_features), one per class.
    sv = [
        np.array([[1.0, 0.0], [1.0, 0.0]]),
        np.array([[0.0, 2.0], [0.0, 2.0]]),
        np.array([[0.0, 0.0], [0.0, 0.0]]),
    ]
    out = _mean_abs_per_feature(sv)
    assert out.shape == (2,)
    # feature 0: mean|·| over classes&samples = (1+0+0)/3 ; feature 1 = (0+2+0)/3
    assert np.allclose(out, [1 / 3, 2 / 3])


def test_mean_abs_values_are_scalars_not_lists():
    sv = np.abs(np.random.default_rng(0).normal(size=(5, 4, 3)))
    out = _mean_abs_per_feature(sv)
    # Each entry must be a plain float so SHAPSummary rounding works.
    for v in out.tolist():
        assert isinstance(v, float)


def test_compute_shap_linear_model_with_column_transformer():
    """Regression: a linear model (e.g. forced logistic_regression) goes through
    the LinearExplainer path, which transforms the background set. The preprocessor
    is a ColumnTransformer with string column selectors, so the background must
    stay a DataFrame - converting to numpy first raised "Specifying the columns
    using strings is only supported for dataframes" and crashed the pipeline.
    """
    import pandas as pd
    from sklearn.compose import ColumnTransformer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    from backend.ml.explainer import compute_shap

    rng = np.random.default_rng(0)
    n = 60
    df = pd.DataFrame({
        "age": rng.normal(50, 10, n),
        "score": rng.normal(0, 1, n),
        "region": rng.choice(["north", "south", "east"], n),
    })
    y = (df["age"] + df["score"] * 5 > 50).astype(int)

    pre = ColumnTransformer([
        ("num", StandardScaler(), ["age", "score"]),
        ("cat", OneHotEncoder(handle_unknown="ignore"), ["region"]),
    ])
    pipe = Pipeline([("preprocessor", pre), ("model", LogisticRegression(max_iter=500))])
    pipe.fit(df, y)

    summary = compute_shap(
        pipe, df, ["age", "score", "region"], "binary_classification", background_data=df
    )

    assert summary.explainer_type == "linear"
    assert summary.n_samples > 0
    assert len(summary.mean_abs_shap) == len(summary.feature_names)
