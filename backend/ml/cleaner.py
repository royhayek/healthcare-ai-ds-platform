"""Data preparation and preprocessing pipeline builder (§10).

Three public functions:
  prepare_data()      - drop/filter, returns (X, y) DataFrames
  build_preprocessor() - unfitted ColumnTransformer from strategy
  apply_preprocessor() - fits on train, transforms train + test

The preprocessor is always built unfitted so that sklearn Pipelines can
wrap it for cross-validation without leaking fit statistics across folds.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    MinMaxScaler,
    OneHotEncoder,
    OrdinalEncoder,
    RobustScaler,
    StandardScaler,
)

from sklearn.preprocessing import LabelEncoder

from backend.models.strategy import PreprocessingStrategy

logger = logging.getLogger(__name__)


def prepare_data(
    df: pd.DataFrame,
    strategy: PreprocessingStrategy,
) -> tuple[pd.DataFrame, pd.Series]:
    """Drop tagged columns and extract target. No fitting.

    Returns (X, y) where X contains only feature columns (not target,
    not drop-tagged columns) and y is the target series.

    y is always integer-encoded for classification targets so sklearn scorers
    (roc_auc, f1, etc.) receive 0/1 labels rather than raw strings.
    """
    target = strategy.target_column
    if target not in df.columns:
        raise ValueError(f"Target column {target!r} not found in DataFrame")

    # Columns explicitly tagged drop
    drop_cols = [
        col for col, strat in strategy.columns.items()
        if strat.action == "drop" and col in df.columns
    ]
    # High-correlation drops from EDA
    drop_cols += [c for c in strategy.drop_high_correlation if c in df.columns]
    drop_cols = list(set(drop_cols))

    feature_cols = [
        c for c in df.columns
        if c != target and c not in drop_cols
    ]

    X = df[feature_cols].copy()
    y = df[target].copy()

    # Drop rows where ALL features are null
    all_null_mask = X.isnull().all(axis=1)
    if all_null_mask.any():
        n = all_null_mask.sum()
        logger.warning("Dropping %d all-null rows", n)
        X = X[~all_null_mask]
        y = y[~all_null_mask]

    # Encode non-numeric targets to integers so sklearn estimators and scorers
    # receive numeric labels. Driven by dtype, not by task_type: a stale or
    # unresolved task_type (e.g. "unknown" from a string-dtype target the
    # profiler could not classify) must NOT leave string labels like 'unknown'
    # to reach the model, which raises "could not convert string to float".
    # Covers object, pandas StringDtype, bool, and category dtypes.
    if strategy.task_type != "regression" and not pd.api.types.is_numeric_dtype(y):
        le = LabelEncoder()
        y = pd.Series(le.fit_transform(y.astype(str)), index=y.index, name=y.name, dtype=int)
        logger.info("Label-encoded target %r: %s", target, dict(zip(le.classes_, le.transform(le.classes_))))

    return X, y


def resolve_task_type(task_type: str | None, y: pd.Series) -> str:
    """Return a valid task_type, inferring from the target when unset/unknown.

    The profiler can leave task_type as "unknown" for a string-dtype target
    (see profiler._detect_task_type). Left unresolved it reaches the trainer,
    which treats anything non-classification as regression and crashes on
    string/categorical labels. This normalises to a known value before training.
    """
    if task_type in ("binary_classification", "multiclass", "regression"):
        return task_type
    non_null = y.dropna()
    n_unique = non_null.nunique()
    if pd.api.types.is_numeric_dtype(y) and not pd.api.types.is_bool_dtype(y):
        if n_unique > 20:
            return "regression"
        return "binary_classification" if n_unique == 2 else "multiclass"
    # Non-numeric target → classification.
    return "binary_classification" if n_unique == 2 else "multiclass"


def split_train_test(
    X: pd.DataFrame,
    y: pd.Series,
    task_type: str,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Stratified train/test split for classification; random for regression."""
    if task_type in ("binary_classification", "multiclass"):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
    return X_train, X_test, y_train, y_test


def split_cal_val(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    task_type: str,
    cal_size: float = 0.20,
    val_size: float = 0.20,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Split X_train into X_fit / X_cal / X_val for calibration + threshold work.

    Returns (X_fit, X_cal, X_val, y_fit, y_cal, y_val).

    X_fit  - used for fitting the final model (≈60% of original train)
    X_cal  - held out for probability calibration (≈20%)
    X_val  - held out for threshold optimization on calibrated probs (≈20%)
    X_test - sealed test set, never touched here
    """
    # First split off X_cal
    if task_type in ("binary_classification", "multiclass"):
        X_fit_tmp, X_cal, y_fit_tmp, y_cal = train_test_split(
            X_train, y_train,
            test_size=cal_size,
            random_state=random_state,
            stratify=y_train,
        )
        X_fit, X_val, y_fit, y_val = train_test_split(
            X_fit_tmp, y_fit_tmp,
            test_size=val_size / (1.0 - cal_size),
            random_state=random_state,
            stratify=y_fit_tmp,
        )
    else:
        X_fit_tmp, X_cal, y_fit_tmp, y_cal = train_test_split(
            X_train, y_train, test_size=cal_size, random_state=random_state
        )
        X_fit, X_val, y_fit, y_val = train_test_split(
            X_fit_tmp, y_fit_tmp,
            test_size=val_size / (1.0 - cal_size),
            random_state=random_state,
        )

    return X_fit, X_cal, X_val, y_fit, y_cal, y_val


def build_preprocessor(
    strategy: PreprocessingStrategy,
    X: pd.DataFrame,
) -> ColumnTransformer:
    """Build an unfitted ColumnTransformer from the preprocessing strategy.

    Column groupings are determined by dtype_hint in the strategy. For columns
    not in the strategy (e.g., newly discovered columns), dtype is inferred from
    the DataFrame dtypes and sane defaults are applied.
    """
    numeric_cols: list[str] = []
    categorical_cols: list[str] = []

    for col in X.columns:
        if col not in strategy.columns:
            # Infer from pandas dtype
            if _is_numeric_like(X[col]):
                numeric_cols.append(col)
            else:
                categorical_cols.append(col)
            continue

        col_strat = strategy.columns[col]
        if col_strat.action == "drop":
            continue

        hint = col_strat.dtype_hint
        # A "numeric" hint is only honoured if the column is actually
        # numeric-coercible. Columns stored as strings with placeholder values
        # like 'unknown' (common when a numeric field has missing entries) would
        # otherwise reach StandardScaler and raise "could not convert string to
        # float: 'unknown'". Genuinely non-numeric columns are demoted to
        # categorical regardless of the (possibly fallback-generated) hint.
        if hint == "categorical":
            categorical_cols.append(col)
        elif hint == "numeric":
            if _is_numeric_like(X[col]):
                numeric_cols.append(col)
            else:
                logger.warning(
                    "Column %r hinted numeric but is not numeric-coercible - treating as categorical",
                    col,
                )
                categorical_cols.append(col)
        else:
            if _is_numeric_like(X[col]):
                numeric_cols.append(col)
            else:
                categorical_cols.append(col)

    transformers: list[tuple[str, Any, list[str]]] = []

    if numeric_cols:
        num_imputer = _pick_imputer_for_group(strategy, numeric_cols, "median")
        num_scaler = _pick_scaler_for_group(strategy, numeric_cols)
        # Coerce first: object-dtype numeric columns carrying placeholder strings
        # (e.g. 'unknown') become NaN here, then the imputer fills them. A no-op
        # for columns already stored as int/float.
        num_steps: list[tuple[str, Any]] = [
            # feature_names_out="one-to-one": coercion preserves columns 1:1, so
            # this keeps ColumnTransformer.get_feature_names_out() working - which
            # SHAP relies on to label features (otherwise they degrade to
            # "feature_0", "feature_1", …).
            ("coerce", FunctionTransformer(_coerce_numeric_array, feature_names_out="one-to-one")),
            ("imputer", num_imputer),
        ]
        if num_scaler is not None:
            num_steps.append(("scaler", num_scaler))
        transformers.append(("num", Pipeline(num_steps), numeric_cols))

    if categorical_cols:
        cat_imputer = _pick_imputer_for_group(strategy, categorical_cols, "most_frequent")
        cat_encoder = _pick_encoder_for_group(strategy, categorical_cols, X)
        cat_steps: list[tuple[str, Any]] = [("imputer", cat_imputer), ("encoder", cat_encoder)]
        transformers.append(("cat", Pipeline(cat_steps), categorical_cols))

    if not transformers:
        ct = ColumnTransformer([], remainder="passthrough")
    else:
        ct = ColumnTransformer(transformers, remainder="drop")

    # Force numpy output so downstream estimators (e.g. LightGBM) are always
    # fitted and called with arrays, not DataFrames. Without this, sklearn 1.4+
    # global set_output settings can cause LightGBM to be fitted with named
    # features and then warned on numpy predict inputs.
    ct.set_output(transform="default")
    return ct


def apply_preprocessor(
    preprocessor: ColumnTransformer,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit preprocessor on X_train, transform both splits. No test leakage."""
    X_train_t = preprocessor.fit_transform(X_train)
    X_test_t = preprocessor.transform(X_test)
    return np.asarray(X_train_t, dtype=np.float32), np.asarray(X_test_t, dtype=np.float32)


# ── Helpers ────────────────────────────────────────────────────────────────────


def _is_numeric_like(s: pd.Series, threshold: float = 0.5) -> bool:
    """True if the column is numeric or majority numeric-coercible.

    A column counts as numeric when it is already an int/float dtype, or when at
    least ``threshold`` of its non-null values parse as numbers. The remaining
    values (placeholders like 'unknown', 'n/a') are treated as missing and
    handled by the imputer. Columns below the threshold are genuinely
    categorical and routed to the encoder instead.
    """
    if pd.api.types.is_numeric_dtype(s):
        return True
    non_null = s.dropna()
    if non_null.empty:
        return False
    coerced = pd.to_numeric(non_null, errors="coerce")
    return bool(coerced.notna().mean() >= threshold)


def _coerce_numeric_array(arr: Any) -> np.ndarray:
    """Coerce a numeric-column slice to float, mapping unparseable strings to NaN.

    Module-level (not a lambda) so the fitted ColumnTransformer stays picklable
    for joblib persistence and the reproducibility manifest hash.
    """
    df = pd.DataFrame(arr)
    return df.apply(lambda c: pd.to_numeric(c, errors="coerce")).to_numpy()


def _pick_imputer_for_group(
    strategy: PreprocessingStrategy,
    cols: list[str],
    default_strategy: str,
) -> SimpleImputer:
    """Pick the most common imputation strategy across the given columns.

    The strategy is constrained to those valid for the group. The default given
    by the caller signals the group: "median"/"mean" → numeric group (all four
    strategies allowed), otherwise → categorical group, where only
    most_frequent/constant are valid. This prevents a column demoted from
    numeric to categorical (whose strategy may still say "median") from forcing
    a median imputer onto string data - which raises "could not convert string
    to float".
    """
    numeric_group = default_strategy in ("median", "mean")
    allowed = (
        {"median", "mean", "most_frequent", "constant"} if numeric_group
        else {"most_frequent", "constant"}
    )
    votes: dict[str, int] = {}
    for col in cols:
        if col in strategy.columns:
            imp = strategy.columns[col].impute_strategy or default_strategy
            if imp not in allowed:
                continue
            votes[imp] = votes.get(imp, 0) + 1
    chosen = max(votes, key=lambda k: votes[k]) if votes else default_strategy
    return SimpleImputer(strategy=chosen)


def _pick_scaler_for_group(
    strategy: PreprocessingStrategy,
    cols: list[str],
) -> Any | None:
    votes: dict[str, int] = {}
    for col in cols:
        if col in strategy.columns:
            sc = strategy.columns[col].scale_strategy or "standard"
            votes[sc] = votes.get(sc, 0) + 1
    chosen = max(votes, key=lambda k: votes[k]) if votes else "standard"
    return {
        "standard": StandardScaler(),
        "minmax": MinMaxScaler(),
        "robust": RobustScaler(),
        "none": None,
    }.get(chosen, StandardScaler())


def _pick_encoder_for_group(
    strategy: PreprocessingStrategy,
    cols: list[str],
    X: pd.DataFrame,
) -> Any:
    votes: dict[str, int] = {}
    for col in cols:
        if col in strategy.columns:
            enc = strategy.columns[col].encode_strategy or "onehot"
            votes[enc] = votes.get(enc, 0) + 1
    chosen = max(votes, key=lambda k: votes[k]) if votes else "onehot"
    if chosen == "ordinal":
        return OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    # Default: OneHotEncoder
    return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
