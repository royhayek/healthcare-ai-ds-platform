"""Multi-dataset join logic (§7).

Handles schema compatibility checking, join-key auto-detection, and the
actual join operation with before/after row-count audit.
"""

from __future__ import annotations

import logging
from typing import Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

JoinType = Literal["inner", "left", "right", "outer"]


class SchemaIssue(BaseModel):
    column: str
    issue_type: str  # new_col | missing_col | dtype_mismatch | category_mismatch
    severity: str  # info | warning | error
    detail: str


class SchemaCompatibilityResult(BaseModel):
    compatible: bool
    issues: list[SchemaIssue] = Field(default_factory=list)
    new_columns: list[str] = Field(default_factory=list)
    missing_columns: list[str] = Field(default_factory=list)
    dtype_mismatches: list[str] = Field(default_factory=list)


class JoinResult(BaseModel):
    rows_before_left: int
    rows_before_right: int
    rows_after: int
    dropped_rows: int
    join_type: str
    join_keys: list[str]
    new_columns_added: list[str]


def check_schema_compatibility(
    df_reference: pd.DataFrame,
    df_new: pd.DataFrame,
    target_column: str | None = None,
) -> SchemaCompatibilityResult:
    """Compare df_new against df_reference (the training schema).

    | Case                        | Severity |
    | --------------------------- | -------- |
    | New columns in df_new       | info     |
    | Missing columns in df_new   | error    |
    | Dtype mismatch (safe cast)  | warning  |
    | Dtype mismatch (lossy)      | error    |
    | Category mismatch           | warning  |

    A result is `compatible=True` iff there are no error-severity issues.
    """
    ref_cols = set(df_reference.columns)
    new_cols = set(df_new.columns)

    # Exclude target column from required-column check (absent in inference)
    required = ref_cols - ({target_column} if target_column else set())

    issues: list[SchemaIssue] = []
    new_columns = sorted(new_cols - ref_cols)
    missing_columns = sorted(required - new_cols)
    dtype_mismatches: list[str] = []

    for col in new_columns:
        issues.append(SchemaIssue(
            column=col,
            issue_type="new_col",
            severity="info",
            detail=f"Column '{col}' is new (not in training schema) - ignored for prediction",
        ))

    for col in missing_columns:
        issues.append(SchemaIssue(
            column=col,
            issue_type="missing_col",
            severity="error",
            detail=f"Column '{col}' is required (present in training) but missing - cannot predict",
        ))

    # Dtype checks for shared columns
    shared = ref_cols & new_cols - ({target_column} if target_column else set())
    for col in sorted(shared):
        ref_dtype = df_reference[col].dtype
        new_dtype = df_new[col].dtype

        if ref_dtype == new_dtype:
            continue

        ref_kind = ref_dtype.kind  # 'f' float, 'i' int, 'O' object, 'b' bool
        new_kind = new_dtype.kind

        # int ↔ float is a safe cast
        if {ref_kind, new_kind} <= {"f", "i"}:
            issues.append(SchemaIssue(
                column=col,
                issue_type="dtype_mismatch",
                severity="warning",
                detail=f"'{col}': training={ref_dtype} vs new={new_dtype} (will auto-coerce)",
            ))
        else:
            issues.append(SchemaIssue(
                column=col,
                issue_type="dtype_mismatch",
                severity="error",
                detail=(
                    f"'{col}': training={ref_dtype} vs new={new_dtype} - "
                    "lossy type change; manual review required"
                ),
            ))
            dtype_mismatches.append(col)

    # Categorical mismatch: new values in a string/object column not seen during training
    def _is_string_col(s: pd.Series) -> bool:
        import pandas.api.types as ptypes
        return ptypes.is_object_dtype(s) or ptypes.is_string_dtype(s)

    for col in sorted(shared):
        if _is_string_col(df_reference[col]) and _is_string_col(df_new[col]):
            ref_cats = set(df_reference[col].dropna().unique())
            new_cats = set(df_new[col].dropna().unique())
            unseen = new_cats - ref_cats
            if unseen:
                issues.append(SchemaIssue(
                    column=col,
                    issue_type="category_mismatch",
                    severity="warning",
                    detail=(
                        f"'{col}': {len(unseen)} unseen category value(s) in new data "
                        f"(e.g. {sorted(str(v) for v in list(unseen)[:3])}). "
                        "Encoder will handle via 'unknown' token."
                    ),
                ))

    has_errors = any(i.severity == "error" for i in issues)

    return SchemaCompatibilityResult(
        compatible=not has_errors,
        issues=issues,
        new_columns=new_columns,
        missing_columns=missing_columns,
        dtype_mismatches=dtype_mismatches,
    )


def auto_detect_join_keys(
    df_left: pd.DataFrame,
    df_right: pd.DataFrame,
    min_overlap: float = 0.5,
) -> list[str]:
    """Heuristically find columns that could serve as join keys.

    A column is a candidate if:
    - It exists in both DataFrames with the same name.
    - At least `min_overlap` fraction of its values appear in both DataFrames.
    - It looks like an identifier (high cardinality relative to row count).
    """
    shared = set(df_left.columns) & set(df_right.columns)
    candidates: list[str] = []

    for col in sorted(shared):
        left_vals = set(df_left[col].dropna().unique())
        right_vals = set(df_right[col].dropna().unique())

        if not left_vals or not right_vals:
            continue

        overlap = len(left_vals & right_vals) / min(len(left_vals), len(right_vals))
        if overlap < min_overlap:
            continue

        # Prefer high-cardinality columns (likely IDs, not categories)
        cardinality_ratio = len(left_vals) / len(df_left)
        if cardinality_ratio > 0.05:  # > 5% unique values
            candidates.append(col)

    return candidates


def join_datasets(
    df_left: pd.DataFrame,
    df_right: pd.DataFrame,
    join_type: JoinType,
    join_keys: list[str],
) -> tuple[pd.DataFrame, JoinResult]:
    """Join two DataFrames and return the result with audit metadata.

    Columns that appear in both DataFrames (and are not join keys) are
    suffixed _left and _right in the output. The caller is responsible for
    resolving duplicates.
    """
    if not join_keys:
        raise ValueError("join_keys must not be empty")

    for key in join_keys:
        if key not in df_left.columns:
            raise ValueError(f"Join key '{key}' not found in left dataset")
        if key not in df_right.columns:
            raise ValueError(f"Join key '{key}' not found in right dataset")

    rows_before_left = len(df_left)
    rows_before_right = len(df_right)

    how_map: dict[str, str] = {
        "inner": "inner",
        "left": "left",
        "right": "right",
        "outer": "outer",
    }
    how = how_map[join_type]

    # Determine new columns added by the right frame (excluding join keys)
    right_new_cols = [c for c in df_right.columns if c not in df_left.columns and c not in join_keys]

    merged = df_left.merge(df_right, on=join_keys, how=how, suffixes=("_left", "_right"))

    return merged, JoinResult(
        rows_before_left=rows_before_left,
        rows_before_right=rows_before_right,
        rows_after=len(merged),
        dropped_rows=rows_before_left - len(merged) if join_type == "inner" else 0,
        join_type=join_type,
        join_keys=join_keys,
        new_columns_added=right_new_cols,
    )
