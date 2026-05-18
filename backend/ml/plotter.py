"""Server-side plot renderer (§12, Category B1 + D1-D7).

Uses matplotlib with the Agg non-interactive backend so it is safe inside
Celery workers and FastAPI processes that have no display.

PlotRenderer.render(spec, profile, df) → base64-encoded PNG string.

All methods receive either the DatasetProfile (for aggregate-level plots) or
a raw DataFrame (for row-level plots like missingness matrix or pairplot).
Raw DataFrames must be kept <= 5 000 rows before calling - the caller must
sample first to keep plot generation under 10 s.
"""

from __future__ import annotations

import base64
import io
import logging
import math
from enum import Enum
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")  # Must be set before pyplot import
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import cm

from backend.ml.profiler import DatasetProfile

logger = logging.getLogger(__name__)

_DPI = 120
_FIG_W = 8
_FIG_H = 5


# ── Plot type catalogue ────────────────────────────────────────────────────────


class PlotType(str, Enum):
    # D1 - Outlier detection
    BOX = "box"
    ZSCORE_DIST = "zscore_dist"
    OUTLIER_HEATMAP = "outlier_heatmap"
    ISOLATION_SCORES = "isolation_scores"
    BEFORE_AFTER_CAP = "before_after_cap"
    # D2 - Distribution
    HISTOGRAM_KDE = "histogram_kde"
    QQ_PLOT = "qq_plot"
    LOG_PREVIEW = "log_preview"
    CAT_BAR = "cat_bar"
    CARDINALITY_WARNING = "cardinality_warning"
    BINARY_BAR = "binary_bar"
    # D3 - Missingness
    MISSINGNESS_MATRIX = "missingness_matrix"
    MISSINGNESS_CORRELATION = "missingness_correlation"
    MISSING_VS_TARGET = "missing_vs_target"
    # D4 - Correlation & redundancy
    CORR_HEATMAP = "corr_heatmap"
    PAIRPLOT = "pairplot"
    VIF_BAR = "vif_bar"
    # D5 - Class imbalance
    CLASS_DIST = "class_dist"
    TARGET_DIST = "target_dist"
    FEATURE_VS_TARGET = "feature_vs_target"
    # D6 - Integrity
    DUPLICATE_SUMMARY = "duplicate_summary"
    CONSTANT_INDICATOR = "constant_indicator"
    # D7 - Multi-dataset comparison
    SIDE_BY_SIDE_KDE = "side_by_side_kde"
    PSI_BAR = "psi_bar"
    CAT_DRIFT_HEATMAP = "cat_drift_heatmap"
    # D8 - Preprocessing verification
    MISSING_RESOLVED = "missing_resolved"


# ── Plot spec / manifest ───────────────────────────────────────────────────────


class PlotSpec:
    """Descriptor for a single plot to render."""

    __slots__ = ("plot_id", "plot_type", "title", "column", "priority", "stage", "extra")

    def __init__(
        self,
        plot_id: str,
        plot_type: PlotType,
        title: str,
        column: str | None = None,
        priority: int = 1,
        stage: str = "eda",
        extra: dict[str, Any] | None = None,
    ) -> None:
        self.plot_id = plot_id
        self.plot_type = plot_type
        self.title = title
        self.column = column
        self.priority = priority
        self.stage = stage
        self.extra = extra or {}

    def to_dict(self) -> dict[str, Any]:
        return {
            "plot_id": self.plot_id,
            "plot_type": self.plot_type.value,
            "title": self.title,
            "column": self.column,
            "priority": self.priority,
            "stage": self.stage,
            "extra": self.extra,
        }


class PlotManifest:
    """Ordered list of PlotSpec objects for a pipeline stage."""

    def __init__(self, run_id: str, stage: str, plots: list[PlotSpec]) -> None:
        self.run_id = run_id
        self.stage = stage
        self.plots = plots

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "stage": self.stage,
            "plots": [p.to_dict() for p in self.plots],
        }


# ── Renderer ──────────────────────────────────────────────────────────────────


class PlotRenderer:
    """Renders plot specs to base64-encoded PNG strings.

    Usage:
        renderer = PlotRenderer()
        png_b64 = renderer.render(spec, profile=profile, df=df)
    """

    def render(
        self,
        spec: PlotSpec,
        profile: DatasetProfile | None = None,
        df: pd.DataFrame | None = None,
        df_reference: pd.DataFrame | None = None,
    ) -> str:
        """Dispatch to the appropriate render method and return base64 PNG.

        Returns an empty string on any rendering failure - never raises so
        one bad plot doesn't block the whole manifest.
        """
        try:
            method_name = f"_render_{spec.plot_type.value}"
            method = getattr(self, method_name, None)
            if method is None:
                logger.warning("No renderer for plot type %s", spec.plot_type)
                return ""
            fig = method(spec, profile=profile, df=df, df_reference=df_reference)
            if fig is None:
                return ""
            return _fig_to_b64(fig)
        except Exception as exc:
            logger.warning("Plot render failed for %s: %s", spec.plot_id, exc)
            return ""

    # ── D1: Outlier detection ──────────────────────────────────────────────────

    def _render_box(
        self,
        spec: PlotSpec,
        profile: DatasetProfile | None = None,
        df: pd.DataFrame | None = None,
        **_: Any,
    ) -> plt.Figure | None:
        if df is None or spec.column not in df.columns:
            return None
        series = df[spec.column].dropna()
        if series.empty:
            return None
        fig, ax = plt.subplots(figsize=(_FIG_W, _FIG_H), dpi=_DPI)
        q25, q75 = series.quantile(0.25), series.quantile(0.75)
        iqr = q75 - q25
        lower, upper = q25 - 1.5 * iqr, q75 + 1.5 * iqr
        outliers = series[(series < lower) | (series > upper)]
        inliers = series[(series >= lower) & (series <= upper)]
        ax.boxplot(
            inliers.values,
            vert=True,
            patch_artist=True,
            boxprops=dict(facecolor="#DBEAFE"),
            medianprops=dict(color="#1E40AF", linewidth=2),
            flierprops=dict(marker="o", markerfacecolor="#EF4444", markersize=4),
        )
        if len(outliers):
            ax.scatter(
                [1] * len(outliers), outliers.values,
                color="#EF4444", s=20, zorder=5, label=f"{len(outliers)} outliers"
            )
            ax.legend(fontsize=8)
        ax.set_title(spec.title, fontsize=11)
        ax.set_xticklabels([spec.column])
        _style_ax(ax)
        fig.tight_layout()
        return fig

    def _render_histogram_kde(
        self,
        spec: PlotSpec,
        df: pd.DataFrame | None = None,
        **_: Any,
    ) -> plt.Figure | None:
        if df is None or spec.column not in df.columns:
            return None
        series = df[spec.column].dropna()
        if series.empty or not pd.api.types.is_numeric_dtype(series):
            return None
        fig, ax = plt.subplots(figsize=(_FIG_W, _FIG_H), dpi=_DPI)
        ax.hist(series, bins=min(50, max(10, len(series) // 20)), color="#93C5FD", edgecolor="white", density=True, alpha=0.8)
        try:
            from scipy.stats import gaussian_kde
            kde = gaussian_kde(series, bw_method="scott")
            xs = np.linspace(series.min(), series.max(), 300)
            ax.plot(xs, kde(xs), color="#1D4ED8", linewidth=2)
        except Exception:
            pass
        ax.set_title(spec.title, fontsize=11)
        ax.set_xlabel(spec.column)
        ax.set_ylabel("Density")
        _style_ax(ax)
        fig.tight_layout()
        return fig

    def _render_qq_plot(
        self,
        spec: PlotSpec,
        df: pd.DataFrame | None = None,
        **_: Any,
    ) -> plt.Figure | None:
        if df is None or spec.column not in df.columns:
            return None
        series = df[spec.column].dropna()
        if len(series) < 10:
            return None
        from scipy import stats

        fig, ax = plt.subplots(figsize=(_FIG_W, _FIG_H), dpi=_DPI)
        (osm, osr), (slope, intercept, _) = stats.probplot(series, dist="norm")
        ax.scatter(osm, osr, color="#3B82F6", s=8, alpha=0.6)
        xmin, xmax = min(osm), max(osm)
        ax.plot([xmin, xmax], [slope * xmin + intercept, slope * xmax + intercept],
                color="#EF4444", linewidth=1.5, linestyle="--", label="Normal reference")
        ax.set_title(spec.title, fontsize=11)
        ax.set_xlabel("Theoretical quantiles")
        ax.set_ylabel("Sample quantiles")
        ax.legend(fontsize=8)
        _style_ax(ax)
        fig.tight_layout()
        return fig

    def _render_log_preview(
        self,
        spec: PlotSpec,
        df: pd.DataFrame | None = None,
        **_: Any,
    ) -> plt.Figure | None:
        if df is None or spec.column not in df.columns:
            return None
        series = df[spec.column].dropna()
        if series.empty or series.min() <= 0:
            return None
        fig, axes = plt.subplots(1, 2, figsize=(_FIG_W, _FIG_H), dpi=_DPI)
        for ax, (label, data) in zip(axes, [("Raw", series), ("log1p", np.log1p(series))]):
            ax.hist(data, bins=40, color="#93C5FD", edgecolor="white", alpha=0.8)
            ax.set_title(label, fontsize=10)
            _style_ax(ax)
        fig.suptitle(spec.title, fontsize=11)
        fig.tight_layout()
        return fig

    def _render_cat_bar(
        self,
        spec: PlotSpec,
        df: pd.DataFrame | None = None,
        **_: Any,
    ) -> plt.Figure | None:
        if df is None or spec.column not in df.columns:
            return None
        series = df[spec.column].dropna().astype(str)
        counts = series.value_counts()
        top_n = 20
        if len(counts) > top_n:
            other = counts.iloc[top_n:].sum()
            counts = counts.iloc[:top_n]
            counts["(other)"] = other
        fig, ax = plt.subplots(figsize=(_FIG_W, max(4, len(counts) * 0.35)), dpi=_DPI)
        bars = ax.barh(counts.index[::-1], counts.values[::-1], color="#93C5FD", edgecolor="white")
        for bar, val in zip(bars, counts.values[::-1]):
            ax.text(bar.get_width() + max(counts) * 0.01, bar.get_y() + bar.get_height() / 2,
                    f"{val:,}", va="center", fontsize=7)
        ax.set_title(spec.title, fontsize=11)
        _style_ax(ax)
        fig.tight_layout()
        return fig

    def _render_binary_bar(
        self,
        spec: PlotSpec,
        df: pd.DataFrame | None = None,
        **_: Any,
    ) -> plt.Figure | None:
        return self._render_cat_bar(spec, df=df)

    def _render_cardinality_warning(
        self,
        spec: PlotSpec,
        df: pd.DataFrame | None = None,
        **_: Any,
    ) -> plt.Figure | None:
        return self._render_cat_bar(spec, df=df)

    # ── D3: Missingness ────────────────────────────────────────────────────────

    def _render_missingness_matrix(
        self,
        spec: PlotSpec,
        df: pd.DataFrame | None = None,
        **_: Any,
    ) -> plt.Figure | None:
        if df is None:
            return None
        cols_with_missing = [c for c in df.columns if df[c].isnull().any()]
        if not cols_with_missing:
            return None
        sample = df[cols_with_missing].head(200)
        matrix = sample.isnull().astype(int).T
        fig, ax = plt.subplots(figsize=(_FIG_W, max(3, len(cols_with_missing) * 0.4)), dpi=_DPI)
        ax.imshow(matrix.values, aspect="auto", cmap="gray_r", interpolation="none")
        ax.set_yticks(range(len(cols_with_missing)))
        ax.set_yticklabels(cols_with_missing, fontsize=7)
        ax.set_xlabel("Row index (first 200)", fontsize=8)
        ax.set_title(spec.title, fontsize=11)
        _style_ax(ax)
        fig.tight_layout()
        return fig

    def _render_missingness_correlation(
        self,
        spec: PlotSpec,
        profile: DatasetProfile | None = None,
        df: pd.DataFrame | None = None,
        **_: Any,
    ) -> plt.Figure | None:
        corr_data = (profile.missingness_correlation if profile else None) or {}
        if not corr_data:
            if df is None:
                return None
            from backend.ml.profiler import compute_missingness_correlation
            corr_data = compute_missingness_correlation(df)
        if not corr_data:
            return None
        cols = list(corr_data.keys())
        mat = np.zeros((len(cols), len(cols)))
        for i, c1 in enumerate(cols):
            for j, c2 in enumerate(cols):
                mat[i, j] = corr_data.get(c1, {}).get(c2, 0.0)
        fig, ax = plt.subplots(figsize=(_FIG_W, _FIG_H), dpi=_DPI)
        im = ax.imshow(mat, cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_xticks(range(len(cols)))
        ax.set_yticks(range(len(cols)))
        ax.set_xticklabels(cols, rotation=45, ha="right", fontsize=7)
        ax.set_yticklabels(cols, fontsize=7)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(spec.title, fontsize=11)
        fig.tight_layout()
        return fig

    def _render_missing_vs_target(
        self,
        spec: PlotSpec,
        df: pd.DataFrame | None = None,
        profile: DatasetProfile | None = None,
        **_: Any,
    ) -> plt.Figure | None:
        if df is None or not (profile and profile.target_column):
            return None
        target = profile.target_column
        if target not in df.columns or spec.column not in df.columns:
            return None
        missing_mask = df[spec.column].isnull()
        if missing_mask.sum() < 5:
            return None
        fig, ax = plt.subplots(figsize=(_FIG_W, _FIG_H), dpi=_DPI)
        target_series = df[target]
        if pd.api.types.is_numeric_dtype(target_series) and target_series.nunique() > 10:
            ax.hist(target_series[~missing_mask], bins=30, alpha=0.6, label="Not missing", color="#3B82F6")
            ax.hist(target_series[missing_mask], bins=30, alpha=0.6, label="Missing", color="#EF4444")
        else:
            for grp, label, color in [(~missing_mask, "Not missing", "#3B82F6"), (missing_mask, "Missing", "#EF4444")]:
                counts = target_series[grp].value_counts(normalize=True)
                ax.bar([str(v) for v in counts.index], counts.values, alpha=0.7, label=label, color=color)
        ax.legend(fontsize=8)
        ax.set_title(spec.title, fontsize=11)
        _style_ax(ax)
        fig.tight_layout()
        return fig

    # ── D4: Correlation & redundancy ──────────────────────────────────────────

    def _render_corr_heatmap(
        self,
        spec: PlotSpec,
        profile: DatasetProfile | None = None,
        df: pd.DataFrame | None = None,
        **_: Any,
    ) -> plt.Figure | None:
        corr_matrix: dict[str, dict[str, float]] | None = None
        if profile and profile.correlation_matrix:
            corr_matrix = profile.correlation_matrix
        elif df is not None:
            numeric = df.select_dtypes(include="number")
            if numeric.shape[1] < 2:
                return None
            corr_matrix = numeric.corr().to_dict()
        if not corr_matrix:
            return None

        cols = list(corr_matrix.keys())
        if len(cols) > 25:
            cols = cols[:25]
        mat = np.array([[corr_matrix.get(c1, {}).get(c2, 0.0) for c2 in cols] for c1 in cols])

        fig_size = max(_FIG_H, len(cols) * 0.4)
        fig, ax = plt.subplots(figsize=(fig_size, fig_size), dpi=_DPI)
        im = ax.imshow(mat, cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_xticks(range(len(cols)))
        ax.set_yticks(range(len(cols)))
        ax.set_xticklabels(cols, rotation=45, ha="right", fontsize=7)
        ax.set_yticklabels(cols, fontsize=7)
        for i in range(len(cols)):
            for j in range(len(cols)):
                val = mat[i, j]
                if abs(val) >= 0.7:
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=6,
                            color="white" if abs(val) > 0.85 else "black")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(spec.title, fontsize=11)
        fig.tight_layout()
        return fig

    def _render_vif_bar(
        self,
        spec: PlotSpec,
        profile: DatasetProfile | None = None,
        **_: Any,
    ) -> plt.Figure | None:
        vif_data = profile.vif if profile else None
        if not vif_data:
            return None
        cols = list(vif_data.keys())
        vals = [vif_data[c] for c in cols]
        colors = ["#EF4444" if v > 10 else "#F59E0B" if v > 5 else "#22C55E" for v in vals]
        order = sorted(range(len(vals)), key=lambda i: vals[i], reverse=True)
        cols_s = [cols[i] for i in order]
        vals_s = [vals[i] for i in order]
        colors_s = [colors[i] for i in order]
        fig, ax = plt.subplots(figsize=(_FIG_W, max(4, len(cols) * 0.35)), dpi=_DPI)
        ax.barh(cols_s[::-1], vals_s[::-1], color=colors_s[::-1])
        ax.axvline(10, color="#EF4444", linestyle="--", linewidth=1, label="VIF = 10 (severe)")
        ax.axvline(5, color="#F59E0B", linestyle="--", linewidth=1, label="VIF = 5 (moderate)")
        ax.legend(fontsize=8)
        ax.set_title(spec.title, fontsize=11)
        ax.set_xlabel("VIF")
        _style_ax(ax)
        fig.tight_layout()
        return fig

    def _render_pairplot(
        self,
        spec: PlotSpec,
        df: pd.DataFrame | None = None,
        profile: DatasetProfile | None = None,
        **_: Any,
    ) -> plt.Figure | None:
        if df is None:
            return None
        target = profile.target_column if profile else spec.extra.get("target_column")
        cols_to_use = spec.extra.get("columns")
        if not cols_to_use:
            numeric = df.select_dtypes(include="number").columns.tolist()
            if target in numeric:
                numeric.remove(target)
            cols_to_use = numeric[:5]
        if not cols_to_use or len(cols_to_use) < 2:
            return None
        sample = df[cols_to_use + ([target] if target and target in df.columns else [])].dropna().head(500)
        n = len(cols_to_use)
        fig, axes = plt.subplots(n, n, figsize=(n * 2.5, n * 2.5), dpi=_DPI)
        if n == 1:
            axes = [[axes]]
        hue_col = target if target and target in sample.columns else None
        if hue_col and sample[hue_col].nunique() <= 10:
            classes = sample[hue_col].unique()
            palette = cm.tab10(np.linspace(0, 1, len(classes)))
            for i, c1 in enumerate(cols_to_use):
                for j, c2 in enumerate(cols_to_use):
                    ax = axes[i][j]
                    if i == j:
                        for cls, color in zip(classes, palette):
                            subset = sample[sample[hue_col] == cls][c1]
                            ax.hist(subset, bins=20, alpha=0.5, color=color, density=True)
                    else:
                        for cls, color in zip(classes, palette):
                            subset = sample[sample[hue_col] == cls]
                            ax.scatter(subset[c2], subset[c1], s=4, alpha=0.4, color=color)
                    ax.set_xlabel(c2 if i == n - 1 else "", fontsize=7)
                    ax.set_ylabel(c1 if j == 0 else "", fontsize=7)
                    ax.tick_params(labelsize=6)
        else:
            for i, c1 in enumerate(cols_to_use):
                for j, c2 in enumerate(cols_to_use):
                    ax = axes[i][j]
                    if i == j:
                        ax.hist(sample[c1].dropna(), bins=20, color="#93C5FD", alpha=0.8)
                    else:
                        ax.scatter(sample[c2], sample[c1], s=4, alpha=0.4, color="#3B82F6")
                    ax.set_xlabel(c2 if i == n - 1 else "", fontsize=7)
                    ax.set_ylabel(c1 if j == 0 else "", fontsize=7)
                    ax.tick_params(labelsize=6)
        fig.suptitle(spec.title, fontsize=10, y=1.01)
        fig.tight_layout()
        return fig

    # ── D5: Class imbalance ────────────────────────────────────────────────────

    def _render_class_dist(
        self,
        spec: PlotSpec,
        df: pd.DataFrame | None = None,
        profile: DatasetProfile | None = None,
        **_: Any,
    ) -> plt.Figure | None:
        if df is None or not (profile and profile.target_column):
            return None
        target = profile.target_column
        if target not in df.columns:
            return None
        counts = df[target].value_counts()
        total = len(df)
        labels = [str(v) for v in counts.index]
        values = counts.values
        colors = ["#3B82F6", "#EF4444", "#22C55E", "#F59E0B", "#8B5CF6"]
        fig, ax = plt.subplots(figsize=(_FIG_W, _FIG_H), dpi=_DPI)
        bars = ax.bar(labels, values, color=colors[:len(labels)], edgecolor="white")
        for bar, val in zip(bars, values):
            pct = 100 * val / total
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + total * 0.005,
                    f"{val:,}\n({pct:.1f}%)", ha="center", va="bottom", fontsize=8)
        ax.set_title(spec.title, fontsize=11)
        ax.set_ylabel("Count")
        _style_ax(ax)
        fig.tight_layout()
        return fig

    def _render_target_dist(
        self,
        spec: PlotSpec,
        df: pd.DataFrame | None = None,
        profile: DatasetProfile | None = None,
        **_: Any,
    ) -> plt.Figure | None:
        if df is None or not (profile and profile.target_column):
            return None
        target = profile.target_column
        if target not in df.columns:
            return None
        series = df[target].dropna()
        if not pd.api.types.is_numeric_dtype(series):
            return self._render_class_dist(spec, df=df, profile=profile)
        fig, ax = plt.subplots(figsize=(_FIG_W, _FIG_H), dpi=_DPI)
        ax.hist(series, bins=50, color="#3B82F6", edgecolor="white", alpha=0.85)
        ax.axvline(series.mean(), color="#EF4444", linestyle="--", label=f"Mean {series.mean():.3f}")
        ax.axvline(series.median(), color="#22C55E", linestyle="--", label=f"Median {series.median():.3f}")
        ax.legend(fontsize=8)
        ax.set_title(spec.title, fontsize=11)
        ax.set_xlabel(target)
        ax.set_ylabel("Count")
        _style_ax(ax)
        fig.tight_layout()
        return fig

    def _render_feature_vs_target(
        self,
        spec: PlotSpec,
        df: pd.DataFrame | None = None,
        profile: DatasetProfile | None = None,
        **_: Any,
    ) -> plt.Figure | None:
        if df is None or spec.column is None or not (profile and profile.target_column):
            return None
        target = profile.target_column
        if target not in df.columns or spec.column not in df.columns:
            return None
        series = df[spec.column].dropna()
        target_series = df.loc[series.index, target]
        if not pd.api.types.is_numeric_dtype(series):
            return None
        classes = target_series.dropna().unique()
        if len(classes) > 10:
            return None
        fig, ax = plt.subplots(figsize=(_FIG_W, _FIG_H), dpi=_DPI)
        palette = cm.tab10(np.linspace(0, 1, len(classes)))
        for cls, color in zip(sorted(classes), palette):
            mask = target_series == cls
            subset = series[mask].dropna()
            if len(subset) > 0:
                try:
                    from scipy.stats import gaussian_kde
                    kde = gaussian_kde(subset, bw_method="scott")
                    xs = np.linspace(series.min(), series.max(), 300)
                    ax.fill_between(xs, kde(xs), alpha=0.35, color=color, label=str(cls))
                    ax.plot(xs, kde(xs), color=color, linewidth=1.5)
                except Exception:
                    ax.hist(subset, bins=30, alpha=0.4, color=color, label=str(cls), density=True)
        ax.legend(fontsize=8)
        ax.set_title(spec.title, fontsize=11)
        ax.set_xlabel(spec.column)
        ax.set_ylabel("Density")
        _style_ax(ax)
        fig.tight_layout()
        return fig

    # ── D6: Integrity ─────────────────────────────────────────────────────────

    def _render_duplicate_summary(
        self,
        spec: PlotSpec,
        profile: DatasetProfile | None = None,
        df: pd.DataFrame | None = None,
        **_: Any,
    ) -> plt.Figure | None:
        dup_count = profile.duplicate_count if profile else (df.duplicated().sum() if df is not None else 0)
        total = profile.n_rows if profile else (len(df) if df is not None else 0)
        if total == 0:
            return None
        unique_count = total - dup_count
        fig, ax = plt.subplots(figsize=(5, 4), dpi=_DPI)
        ax.pie(
            [unique_count, dup_count],
            labels=[f"Unique ({unique_count:,})", f"Duplicates ({dup_count:,})"],
            colors=["#3B82F6", "#EF4444"],
            autopct="%1.1f%%",
            startangle=90,
        )
        ax.set_title(spec.title, fontsize=11)
        fig.tight_layout()
        return fig

    def _render_constant_indicator(
        self,
        spec: PlotSpec,
        profile: DatasetProfile | None = None,
        **_: Any,
    ) -> plt.Figure | None:
        if profile is None:
            return None
        near_const = [
            col.name for col in profile.columns
            if col.n_unique is not None and col.n_unique <= 2
               and profile.n_rows > 0
        ]
        if not near_const:
            return None
        fig, ax = plt.subplots(figsize=(_FIG_W, max(3, len(near_const) * 0.4)), dpi=_DPI)
        unique_counts = [
            next((c.n_unique for c in profile.columns if c.name == col), 1)
            for col in near_const
        ]
        ax.barh(near_const[::-1], unique_counts[::-1], color="#F59E0B", edgecolor="white")
        ax.set_title(spec.title, fontsize=11)
        ax.set_xlabel("Unique value count")
        _style_ax(ax)
        fig.tight_layout()
        return fig

    # ── D7: Multi-dataset comparison ──────────────────────────────────────────

    def _render_side_by_side_kde(
        self,
        spec: PlotSpec,
        df: pd.DataFrame | None = None,
        df_reference: pd.DataFrame | None = None,
        **_: Any,
    ) -> plt.Figure | None:
        if df is None or df_reference is None or spec.column is None:
            return None
        if spec.column not in df.columns or spec.column not in df_reference.columns:
            return None
        s_train = df[spec.column].dropna()
        s_inf = df_reference[spec.column].dropna()
        if not (pd.api.types.is_numeric_dtype(s_train) and len(s_train) > 5 and len(s_inf) > 5):
            return None
        fig, ax = plt.subplots(figsize=(_FIG_W, _FIG_H), dpi=_DPI)
        ax.hist(s_train, bins=40, color="#3B82F6", alpha=0.5, density=True, label="Train")
        ax.hist(s_inf, bins=40, color="#F97316", alpha=0.5, density=True, label="Inference")
        try:
            from scipy.stats import gaussian_kde
            for data, color in [(s_train, "#1D4ED8"), (s_inf, "#EA580C")]:
                kde = gaussian_kde(data, bw_method="scott")
                xs = np.linspace(min(data.min(), s_train.min()), max(data.max(), s_inf.max()), 300)
                ax.plot(xs, kde(xs), color=color, linewidth=1.5)
        except Exception:
            pass
        ax.legend(fontsize=8)
        ax.set_title(spec.title, fontsize=11)
        ax.set_xlabel(spec.column)
        _style_ax(ax)
        fig.tight_layout()
        return fig

    def _render_psi_bar(
        self,
        spec: PlotSpec,
        **kwargs: Any,
    ) -> plt.Figure | None:
        psi_values: dict[str, float] = spec.extra.get("psi_values", {})
        if not psi_values:
            return None
        cols = sorted(psi_values.keys(), key=lambda c: psi_values[c], reverse=True)[:25]
        vals = [psi_values[c] for c in cols]
        colors = ["#EF4444" if v > 0.25 else "#F59E0B" if v > 0.1 else "#22C55E" for v in vals]
        fig, ax = plt.subplots(figsize=(_FIG_W, max(4, len(cols) * 0.35)), dpi=_DPI)
        ax.barh(cols[::-1], vals[::-1], color=colors[::-1])
        ax.axvline(0.25, color="#EF4444", linestyle="--", linewidth=1, label="PSI > 0.25 (severe)")
        ax.axvline(0.1, color="#F59E0B", linestyle="--", linewidth=1, label="PSI > 0.10 (moderate)")
        ax.legend(fontsize=8)
        ax.set_title(spec.title, fontsize=11)
        ax.set_xlabel("PSI")
        _style_ax(ax)
        fig.tight_layout()
        return fig

    def _render_cat_drift_heatmap(
        self,
        spec: PlotSpec,
        df: pd.DataFrame | None = None,
        df_reference: pd.DataFrame | None = None,
        **_: Any,
    ) -> plt.Figure | None:
        if df is None or df_reference is None or spec.column is None:
            return None
        if spec.column not in df.columns or spec.column not in df_reference.columns:
            return None
        cats_train = df[spec.column].value_counts(normalize=True)
        cats_inf = df_reference[spec.column].value_counts(normalize=True)
        all_cats = sorted(set(cats_train.index) | set(cats_inf.index))[:20]
        mat = np.array([
            [cats_train.get(c, 0), cats_inf.get(c, 0)]
            for c in all_cats
        ])
        fig, ax = plt.subplots(figsize=(_FIG_W, max(4, len(all_cats) * 0.35)), dpi=_DPI)
        im = ax.imshow(mat.T, aspect="auto", cmap="Blues")
        ax.set_xticks(range(len(all_cats)))
        ax.set_xticklabels([str(c) for c in all_cats], rotation=45, ha="right", fontsize=7)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["Train", "Inference"])
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(spec.title, fontsize=11)
        fig.tight_layout()
        return fig

    # ── D8: Preprocessing verification ───────────────────────────────────────

    def _render_missing_resolved(
        self,
        spec: PlotSpec,
        **_: Any,
    ) -> plt.Figure | None:
        """Bar chart showing missing % before imputation with ✓ resolved annotations."""
        missing_before: dict[str, float] = spec.extra.get("missing_before", {})
        cols_sorted = sorted(
            [(c, pct) for c, pct in missing_before.items() if pct > 0],
            key=lambda x: x[1],
            reverse=True,
        )
        if not cols_sorted:
            return None
        names = [c for c, _ in cols_sorted]
        pcts = [p * 100 for _, p in cols_sorted]
        fig, ax = plt.subplots(figsize=(_FIG_W, max(3, len(names) * 0.45)), dpi=_DPI)
        bars = ax.barh(range(len(names)), pcts, color="#EF4444", alpha=0.7)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=8)
        ax.set_xlabel("Missing % (before imputation)")
        ax.set_title(spec.title, fontsize=11)
        max_pct = max(pcts) if pcts else 1
        for i, (bar, pct) in enumerate(zip(bars, pcts)):
            ax.text(
                bar.get_width() + max_pct * 0.02, i,
                f"{pct:.1f}% → 0% ✓",
                va="center", fontsize=7, color="#22C55E", fontweight="bold",
            )
        _style_ax(ax)
        fig.tight_layout()
        return fig

    # ── Placeholder for unimplemented types ───────────────────────────────────

    def _render_zscore_dist(self, spec: PlotSpec, df: pd.DataFrame | None = None, **_: Any) -> plt.Figure | None:
        if df is None or spec.column not in df.columns:
            return None
        series = df[spec.column].dropna()
        if not pd.api.types.is_numeric_dtype(series) or series.std() == 0:
            return None
        z = (series - series.mean()) / series.std()
        fig, ax = plt.subplots(figsize=(_FIG_W, _FIG_H), dpi=_DPI)
        ax.hist(z, bins=50, color="#93C5FD", edgecolor="white", density=True, alpha=0.85)
        ax.axvline(3, color="#EF4444", linestyle="--", linewidth=1.5, label="+3σ")
        ax.axvline(-3, color="#EF4444", linestyle="--", linewidth=1.5, label="-3σ")
        ax.legend(fontsize=8)
        ax.set_title(spec.title, fontsize=11)
        ax.set_xlabel("Z-score")
        _style_ax(ax)
        fig.tight_layout()
        return fig

    def _render_outlier_heatmap(self, spec: PlotSpec, df: pd.DataFrame | None = None, **_: Any) -> plt.Figure | None:
        if df is None:
            return None
        numeric = df.select_dtypes(include="number").head(200)
        if numeric.empty:
            return None
        q25 = numeric.quantile(0.25)
        q75 = numeric.quantile(0.75)
        iqr = q75 - q25
        lower = q25 - 1.5 * iqr
        upper = q75 + 1.5 * iqr
        outlier_matrix = ((numeric < lower) | (numeric > upper)).astype(int)
        fig, ax = plt.subplots(figsize=(_FIG_W, max(3, numeric.shape[1] * 0.3)), dpi=_DPI)
        ax.imshow(outlier_matrix.T.values, aspect="auto", cmap="Reds", interpolation="none")
        ax.set_yticks(range(len(outlier_matrix.columns)))
        ax.set_yticklabels(outlier_matrix.columns, fontsize=7)
        ax.set_xlabel("Row index (first 200)", fontsize=8)
        ax.set_title(spec.title, fontsize=11)
        fig.tight_layout()
        return fig

    def _render_isolation_scores(self, spec: PlotSpec, profile: DatasetProfile | None = None, **_: Any) -> plt.Figure | None:
        if not (profile and profile.isolation_score_summary):
            return None
        d = profile.isolation_score_summary
        pcts = [d.get(f"p{k}") for k in [5, 25, 50, 75, 95]]
        if any(v is None for v in pcts):
            return None
        fig, ax = plt.subplots(figsize=(6, 4), dpi=_DPI)
        labels = ["p5", "p25", "p50", "p75", "p95"]
        colors = ["#EF4444" if v < 0 else "#22C55E" for v in pcts]
        ax.bar(labels, pcts, color=colors, edgecolor="white")
        ax.axhline(0, color="#111827", linewidth=1, linestyle="--")
        outlier_pct = d.get("outlier_pct_rough", 0)
        ax.set_title(f"{spec.title}\n{outlier_pct*100:.1f}% rows flagged as anomalous", fontsize=10)
        ax.set_ylabel("Isolation score (negative = outlier)")
        _style_ax(ax)
        fig.tight_layout()
        return fig

    def _render_before_after_cap(self, spec: PlotSpec, df: pd.DataFrame | None = None, **_: Any) -> plt.Figure | None:
        if df is None or spec.column not in df.columns:
            return None
        series = df[spec.column].dropna()
        cap_low = spec.extra.get("cap_low", series.quantile(0.01))
        cap_high = spec.extra.get("cap_high", series.quantile(0.99))
        capped = series.clip(lower=cap_low, upper=cap_high)
        fig, axes = plt.subplots(1, 2, figsize=(_FIG_W, _FIG_H), dpi=_DPI, sharey=False)
        for ax, (label, data) in zip(axes, [("Before cap", series), ("After cap", capped)]):
            ax.hist(data, bins=40, color="#93C5FD", edgecolor="white", alpha=0.85)
            ax.set_title(label, fontsize=10)
            _style_ax(ax)
        fig.suptitle(spec.title, fontsize=11)
        fig.tight_layout()
        return fig


# ── Helpers ────────────────────────────────────────────────────────────────────


def _fig_to_b64(fig: plt.Figure) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=_DPI)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def _style_ax(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=8)
