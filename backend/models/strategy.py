"""Pipeline strategy Pydantic models (§10, §13, §16).

These models are serialized to JSONB columns on the Run table and are read
by every downstream ML module. The dict-keyed `columns` field in
PreprocessingStrategy matches the format expected by strategy_mutator.py.
"""

from typing import Any

from pydantic import BaseModel, Field


class ColumnPreprocessingStrategy(BaseModel):
    """Per-column decisions produced by the preprocessing agent."""

    action: str = "keep"  # keep | drop
    impute_strategy: str | None = None  # mean | median | most_frequent | constant | none
    impute_fill_value: str | float | None = None
    encode_strategy: str | None = None  # onehot | ordinal | binary | none
    scale_strategy: str | None = None  # standard | minmax | robust | none
    dtype_hint: str | None = None  # numeric | categorical
    reason: str = ""


class PreprocessingStrategy(BaseModel):
    """Full preprocessing plan for one run.

    `columns` is keyed by column name - this matches the format that
    strategy_mutator._preprocessing_mutator reads and writes.
    """

    columns: dict[str, ColumnPreprocessingStrategy] = Field(default_factory=dict)
    target_column: str
    task_type: str  # binary_classification | multiclass | regression
    drop_high_correlation: list[str] = Field(default_factory=list)
    notes: str | None = None

    def feature_columns(self) -> list[str]:
        """Return columns that are kept (action != drop) and are not the target."""
        return [
            col for col, strat in self.columns.items()
            if strat.action != "drop" and col != self.target_column
        ]

    def numeric_columns(self) -> list[str]:
        return [
            col for col in self.feature_columns()
            if self.columns[col].dtype_hint == "numeric"
        ]

    def categorical_columns(self) -> list[str]:
        return [
            col for col in self.feature_columns()
            if self.columns[col].dtype_hint == "categorical"
        ]


class ModelSelectionStrategy(BaseModel):
    """Model candidates and ranking produced by the model selector agent."""

    candidates: list[str] = Field(default_factory=list)
    primary: str
    primary_metric: str  # auc | f1 | macro_auc | rmse | r2
    excluded: list[dict[str, str]] = Field(default_factory=list)
    reasoning: str = ""
    notes: str | None = None


class CostMatrix(BaseModel):
    """Business cost matrix for threshold optimization (§16).

    Defaults encode: a false negative (missed positive) costs 5× a false positive.
    The threshold optimizer minimizes total_cost = FP*cost_fp + FN*cost_fn.
    """

    cost_fp: float = 1.0
    cost_fn: float = 5.0
    cost_tp: float = 0.0
    cost_tn: float = 0.0
    override_threshold: float | None = None  # user override bypasses optimization


class ThresholdConfig(BaseModel):
    """Persisted threshold configuration (stored in run.threshold_config)."""

    cost_matrix: CostMatrix = Field(default_factory=CostMatrix)
    override_threshold: float | None = None

    def effective_threshold(self, optimized: float) -> float:
        return self.override_threshold if self.override_threshold is not None else optimized


class StabilityResult(BaseModel):
    """Per-candidate stability across seeds × folds."""

    model_name: str
    scores: list[float]  # one score per (seed × fold) combination
    mean: float
    std: float
    train_scores: list[float] = Field(default_factory=list)
    train_mean: float = 0.0
    overfit_gap: float = 0.0  # train_mean - mean; > 0.15 = high overfit risk


class CVResult(BaseModel):
    """Single candidate × single seed cross-validation result."""

    model_name: str
    seed: int
    fold_scores: list[float]
    fold_train_scores: list[float] = Field(default_factory=list)
    mean_score: float
    std_score: float
    metric: str


class TuningResult(BaseModel):
    """Optuna tuning result for the best model."""

    model_name: str
    best_params: dict[str, Any]
    best_score: float
    n_trials: int
    metric: str
    improvement_over_baseline: float = 0.0


class CalibrationReport(BaseModel):
    """Calibration diagnostics (§16)."""

    method: str  # isotonic | sigmoid
    brier_before: float
    brier_after: float
    ece_before: float
    ece_after: float
    improvement_pct: float


class ThresholdResult(BaseModel):
    """Threshold optimization output (§16)."""

    optimal_threshold: float
    cost_at_default: float
    cost_at_optimal: float
    improvement_pct: float
    metric_at_optimal: dict[str, float] = Field(default_factory=dict)
    cost_curve: list[dict[str, float]] = Field(default_factory=list)
    note: str = ""


class SHAPSummary(BaseModel):
    """SHAP global importance summary (§18).

    Stores aggregates only - never individual SHAP matrices.
    """

    feature_names: list[str]
    mean_abs_shap: list[float]  # aligned with feature_names
    top_k_features: list[str]  # top-10 by |SHAP|
    explainer_type: str  # tree | linear | kernel
    n_samples: int
    note: str = ""
