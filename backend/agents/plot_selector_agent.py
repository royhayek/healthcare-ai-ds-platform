"""Local plot selector for checkpoint decision cards (§12, Category B1).

Purely deterministic - no model API call. Selects plots from the compressed
dataset profile using rule-based logic and the mandatory injection pass.

Priority semantics:
  1 = always rendered and shown first
  2 = rendered but shown in collapsible "More plots" section
"""

import logging
import uuid
from typing import Any

from backend.ml.plotter import PlotManifest, PlotSpec, PlotType

logger = logging.getLogger(__name__)


async def select_plots_for_stage(
    compressed_profile: dict[str, Any],
    stage: str,
    task_type: str,
    run_id: str,
) -> PlotManifest:
    """Return a PlotManifest for the given pipeline stage.

    Purely local - no API calls. The mandatory EDA injection pass guarantees
    class_dist, box plots, and missingness_matrix are always present for EDA.
    """
    specs = _default_specs(compressed_profile, stage, task_type, run_id)

    if stage == "eda":
        specs = _inject_mandatory_eda_plots(specs, compressed_profile, task_type)

    return PlotManifest(run_id=run_id, stage=stage, plots=specs)


def _inject_mandatory_eda_plots(
    specs: list[PlotSpec],
    profile: dict[str, Any],
    task_type: str,
) -> list[PlotSpec]:
    """Ensure critical EDA plots are always present.

    Adds at priority=1 if missing:
      - class_dist for target column (classification imbalance)
      - box for every numeric column except target (outlier detection)
      - missingness_matrix when nulls exist
    """
    existing_types_cols = {(s.plot_type.value, s.column) for s in specs}

    target = profile.get("target_column")
    numeric_cols: list[str] = profile.get("numeric_columns", [])
    any_missing = any(c.get("null_pct", 0) > 0 for c in profile.get("columns", []))

    injected: list[PlotSpec] = []

    if task_type in ("binary_classification", "multiclass") and target:
        if ("class_dist", target) not in existing_types_cols:
            injected.append(PlotSpec(
                plot_id="class_dist_target",
                plot_type=PlotType.CLASS_DIST,
                title=f"Class distribution - {target}",
                column=target,
                priority=1,
                stage="eda",
            ))

    for col in numeric_cols:
        if col == target:
            continue
        if ("box", col) not in existing_types_cols:
            injected.append(PlotSpec(
                plot_id=f"box_{col}",
                plot_type=PlotType.BOX,
                title=f"Outliers - {col}",
                column=col,
                priority=1,
                stage="eda",
            ))

    if any_missing and ("missingness_matrix", None) not in existing_types_cols:
        injected.append(PlotSpec(
            plot_id="missingness_matrix_dataset",
            plot_type=PlotType.MISSINGNESS_MATRIX,
            title="Missing value pattern",
            column=None,
            priority=1,
            stage="eda",
        ))

    seen: set[str] = set()
    result: list[PlotSpec] = []
    for s in injected + specs:
        if s.plot_id not in seen:
            seen.add(s.plot_id)
            result.append(s)

    return result[:30]


def generate_preprocessing_after_specs(
    prep_strategy_dict: dict[str, Any],
    original_profile: dict[str, Any],
    run_id: str,
) -> PlotManifest:
    """Build a fixed PlotManifest for the preprocessing_after stage.

    Derived directly from the preprocessing strategy (not the raw profile) so
    the plots reflect the decisions the AI actually made.  The cleaned DataFrame
    (imputed + scaled numeric, imputed categorical, no encoding) is passed as
    ``df`` at render time so all existing plot types work unchanged.
    """
    specs: list[PlotSpec] = []

    columns_strategy: dict[str, Any] = prep_strategy_dict.get("columns", {})
    task_type: str = prep_strategy_dict.get("task_type", "binary_classification")
    target_col: str = prep_strategy_dict.get("target_column", "")

    profile_cols = {
        c.get("name"): c
        for c in original_profile.get("columns", [])
        if c.get("name")
    }

    numeric_kept = [
        col for col, s in columns_strategy.items()
        if s.get("action") == "keep" and s.get("dtype_hint") == "numeric"
    ]
    categorical_kept = [
        col for col, s in columns_strategy.items()
        if s.get("action") == "keep" and s.get("dtype_hint") == "categorical"
    ]

    # Correlation heatmap on kept features only - verifies that dropped
    # high-corr features are gone.
    if len(numeric_kept) >= 2:
        specs.append(PlotSpec(
            plot_id="post_corr_heatmap",
            plot_type=PlotType.CORR_HEATMAP,
            title="Correlation after dropping redundant features",
            priority=1,
            stage="preprocessing_after",
        ))

    # Class distribution - confirms target balance is preserved after row drops.
    if task_type in ("binary_classification", "multiclass") and target_col:
        specs.append(PlotSpec(
            plot_id="post_class_dist",
            plot_type=PlotType.CLASS_DIST,
            title=f"Class distribution - {target_col} (post-preprocessing)",
            column=target_col,
            priority=1,
            stage="preprocessing_after",
        ))

    # Missing-values-resolved chart - shows before-imputation % for every column
    # that had nulls; confirms imputation set them to 0.
    missing_before = {
        col: profile_cols[col].get("null_pct", 0.0)
        for col in (numeric_kept + categorical_kept)
        if col in profile_cols and (profile_cols[col].get("null_pct") or 0.0) > 0
    }
    if missing_before:
        specs.append(PlotSpec(
            plot_id="post_missing_resolved",
            plot_type=PlotType.MISSING_RESOLVED,
            title="Missing values resolved by imputation",
            priority=1,
            stage="preprocessing_after",
            extra={"missing_before": missing_before},
        ))

    # Distributions of numeric columns after imputation + scaling - verifies
    # scaling normalised the range and imputation filled gaps correctly.
    for i, col in enumerate(numeric_kept[:10]):
        specs.append(PlotSpec(
            plot_id=f"post_hist_{col}",
            plot_type=PlotType.HISTOGRAM_KDE,
            title=f"Distribution after preprocessing - {col}",
            column=col,
            priority=1 if i < 5 else 2,
            stage="preprocessing_after",
        ))

    # Box plots on scaled data - shows whether outlier situation improved.
    for col in numeric_kept[:6]:
        specs.append(PlotSpec(
            plot_id=f"post_box_{col}",
            plot_type=PlotType.BOX,
            title=f"Outliers after scaling - {col}",
            column=col,
            priority=2,
            stage="preprocessing_after",
        ))

    return PlotManifest(run_id=run_id, stage="preprocessing_after", plots=specs[:25])


def _default_specs(
    profile: dict[str, Any],
    stage: str,
    task_type: str,
    run_id: str,
) -> list[PlotSpec]:
    """Deterministic plot selection from the dataset profile."""
    specs: list[PlotSpec] = []

    columns: list[dict] = profile.get("columns", [])
    numeric_cols = profile.get("numeric_columns", [])
    categorical_cols = profile.get("categorical_columns", [])
    target = profile.get("target_column")
    high_corr = profile.get("high_correlation_pairs", [])
    any_missing = any(c.get("null_pct", 0) > 0 for c in columns)

    if stage in ("eda", "profiling"):
        if len(numeric_cols) >= 2:
            specs.append(PlotSpec(
                plot_id="corr_heatmap_dataset",
                plot_type=PlotType.CORR_HEATMAP,
                title="Feature correlation heatmap",
                priority=1, stage=stage,
            ))

        if task_type in ("binary_classification", "multiclass") and target:
            specs.append(PlotSpec(
                plot_id="class_dist_target",
                plot_type=PlotType.CLASS_DIST,
                title=f"Class distribution - {target}",
                column=target, priority=1, stage=stage,
            ))

        for name in numeric_cols:
            if name != target:
                specs.append(PlotSpec(
                    plot_id=f"box_{name}",
                    plot_type=PlotType.BOX,
                    title=f"Outliers - {name}",
                    column=name, priority=1, stage=stage,
                ))

        for col_info in columns:
            if col_info.get("name") == target:
                continue
            name = col_info.get("name", "")
            skew = col_info.get("skewness")
            if name in numeric_cols and skew is not None and abs(skew) > 1.0:
                specs.append(PlotSpec(
                    plot_id=f"histogram_kde_{name}",
                    plot_type=PlotType.HISTOGRAM_KDE,
                    title=f"Distribution of {name}",
                    column=name, priority=1 if abs(skew) > 2 else 2, stage=stage,
                ))

        for name in categorical_cols[:6]:
            if name != target:
                specs.append(PlotSpec(
                    plot_id=f"cat_bar_{name}",
                    plot_type=PlotType.CAT_BAR,
                    title=f"Value distribution - {name}",
                    column=name, priority=2, stage=stage,
                ))

        if any_missing:
            specs.append(PlotSpec(
                plot_id="missingness_matrix_dataset",
                plot_type=PlotType.MISSINGNESS_MATRIX,
                title="Missing value pattern",
                priority=1, stage=stage,
            ))

        if high_corr:
            specs.append(PlotSpec(
                plot_id="vif_bar_dataset",
                plot_type=PlotType.VIF_BAR,
                title="Variance Inflation Factor",
                priority=2, stage=stage,
            ))

        if profile.get("isolation_score_summary"):
            specs.append(PlotSpec(
                plot_id="isolation_scores_dataset",
                plot_type=PlotType.ISOLATION_SCORES,
                title="Anomaly score distribution",
                priority=2, stage=stage,
            ))

    elif stage == "preprocessing":
        for name in numeric_cols[:8]:
            if name != target:
                specs.append(PlotSpec(
                    plot_id=f"before_after_cap_{name}",
                    plot_type=PlotType.BEFORE_AFTER_CAP,
                    title=f"Capping effect - {name}",
                    column=name, priority=2, stage=stage,
                ))

    elif stage == "training":
        if task_type in ("binary_classification", "multiclass") and target:
            # Use a stage-prefixed ID to avoid colliding with the EDA class_dist
            # entry in the manifest. Without the prefix the manifest deduplicates
            # by plot_id and the training-stage card never has a class_dist plot.
            specs.append(PlotSpec(
                plot_id="training_class_dist_target",
                plot_type=PlotType.CLASS_DIST,
                title=f"Class distribution ({target})",
                column=target, priority=1, stage=stage,
            ))
        for name in numeric_cols[:3]:
            if name != target:
                specs.append(PlotSpec(
                    plot_id=f"feature_vs_target_{name}",
                    plot_type=PlotType.FEATURE_VS_TARGET,
                    title=f"{name} vs {target}",
                    column=name, priority=2, stage=stage,
                ))

    return specs[:20]
