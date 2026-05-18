"""Shared types for the eight-deliverable suite (§4, §23).

Every generator returns a GeneratedDeliverable. The deliverable_task.py
orchestrator collects them all, persists to storage, inserts Deliverable ORM
rows, and bundles into a ZIP.
"""

from __future__ import annotations

import hashlib
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

GENERATOR_VERSION = "1.0.0"

CLINICAL_DISCLAIMER = (
    "⚕ CLINICAL ADVISORY: This analysis is intended to assist clinical decision-making and "
    "does not constitute a medical diagnosis or treatment recommendation. All AI-generated "
    "risk scores, predictions, and recommendations must be reviewed and verified by a licensed "
    "clinician before any clinical action is taken. Model performance may vary across patient "
    "populations not represented in the training cohort."
)

CLINICAL_DISCLAIMER_SHORT = (
    "This analysis assists clinical decision-making and must be reviewed by a licensed "
    "clinician before clinical action is taken."
)

# On macOS with Homebrew, weasyprint's Pango/GObject dependencies live under
# /opt/homebrew/lib and are not on the default dyld search path. Patch the env
# before any weasyprint import so dlopen() finds them.
if sys.platform == "darwin":
    _homebrew_lib = "/opt/homebrew/lib"
    _existing = os.environ.get("DYLD_LIBRARY_PATH", "")
    if _homebrew_lib not in _existing:
        os.environ["DYLD_LIBRARY_PATH"] = (
            f"{_homebrew_lib}:{_existing}" if _existing else _homebrew_lib
        )


def render_pdf(html: str, css_path: str) -> bytes:
    """Render *html* to PDF bytes via weasyprint.

    Raises on failure - callers must not swallow the exception and return
    raw HTML bytes (that produces a corrupt file with a .pdf extension).
    """
    from weasyprint import CSS, HTML
    from weasyprint.text.fonts import FontConfiguration

    font_config = FontConfiguration()
    css = CSS(filename=css_path)
    return HTML(string=html).write_pdf(stylesheets=[css], font_config=font_config)


@dataclass
class GeneratedDeliverable:
    """Output of one generator: the file bytes + metadata."""

    name: str                   # executive_summary | model_card | ...
    fmt: str                    # pdf | xlsx | md | csv | json | yaml | zip
    content: bytes              # the actual file bytes
    storage_path: str           # path within storage root
    checksum_sha256: str        # sha256(content)
    generated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    generator_version: str = GENERATOR_VERSION
    inputs_used: list[str] = field(default_factory=list)
    audience: str = ""

    @classmethod
    def build(
        cls,
        name: str,
        fmt: str,
        content: bytes,
        run_id: str,
        inputs_used: list[str] | None = None,
        audience: str = "",
        extra_path: str = "",
    ) -> "GeneratedDeliverable":
        sha = hashlib.sha256(content).hexdigest()
        ext = _ext(fmt)
        suffix = f"_{extra_path}" if extra_path else ""
        path = f"runs/{run_id}/deliverables/{name}{suffix}.{ext}"
        return cls(
            name=name,
            fmt=fmt,
            content=content,
            storage_path=path,
            checksum_sha256=sha,
            inputs_used=inputs_used or [],
            audience=audience,
        )


def _ext(fmt: str) -> str:
    return {
        "pdf": "pdf",
        "xlsx": "xlsx",
        "md": "md",
        "csv": "csv",
        "json": "json",
        "yaml": "yaml",
        "zip": "zip",
    }.get(fmt, fmt)


def run_summary_context(run: Any, dataset: Any, case_brief: dict[str, Any] | None = None) -> dict[str, Any]:
    """Build a compact, serializable context dict from ORM objects.

    Passed to the model and to Jinja templates so they never touch raw ORM rows.
    All sensitive row-level data is excluded - only aggregates and metadata.
    """
    eda = run.eda_report or {}
    prep = dict(run.preprocessing_strategy or {})
    # Normalize columns to dict - the template calls .items() on it.
    # the model occasionally returns a list; old runs may be stored that way too.
    if isinstance(prep.get("columns"), list):
        prep["columns"] = {
            c.get("name", str(i)): {k: v for k, v in c.items() if k != "name"}
            for i, c in enumerate(prep["columns"])
            if isinstance(c, dict)
        }
    sel = run.model_selection or {}
    fm = run.final_metrics or {}
    shap = run.shap_summary or {}
    cal = run.calibration_report or {}
    thr = run.threshold_result or {}
    thr_cfg = run.threshold_config or {}
    drift = run.drift_report or {}
    fairness = run.fairness_report or {}

    return {
        "run_id": run.id,
        "project_id": run.project_id,
        "created_at": run.created_at.isoformat() if run.created_at else "",
        "completed_at": run.completed_at.isoformat() if run.completed_at else "",
        "created_by": run.created_by or "unknown",
        "dataset": {
            "filename": dataset.filename if dataset else "unknown",
            "sha256": dataset.sha256[:16] + "…" if dataset and dataset.sha256 else "",
            "row_count": dataset.row_count if dataset else None,
            "col_count": dataset.col_count if dataset else None,
            "target_column": dataset.target_column if dataset else None,
        },
        "task_type": prep.get("task_type", eda.get("target_analysis", {}).get("task_type", "unknown")),
        "model_name": run.best_model_name or "unknown",
        "final_metrics": fm,
        "calibration": cal,
        "threshold": {
            "optimal": thr.get("optimal_threshold"),
            "cost_at_default": thr.get("cost_at_default"),
            "cost_at_optimal": thr.get("cost_at_optimal"),
            "improvement_pct": thr.get("improvement_pct"),
            "note": thr.get("note", ""),
        },
        "threshold_config": thr_cfg,
        "shap": {
            "top_features": shap.get("top_k_features", [])[:10],
            "mean_abs": shap.get("mean_abs_shap", [])[:10],
            "feature_names": shap.get("feature_names", []),
            "explainer_type": shap.get("explainer_type", ""),
        },
        "eda_summary": eda.get("summary", ""),
        "quality_issues": eda.get("quality_issues", []),
        "model_comparison": run.model_comparison or [],
        "stat_tests": run.stat_tests or {},
        "tuning_result": run.tuning_result or {},
        "preprocessing_strategy": prep,
        "drift": drift,
        "fairness": fairness,
        "seeds": run.seeds or {},
        "claude_models_used": run.claude_models_used or {},
        "library_versions": run.library_versions or {},
        "case_brief": {
            "objectives": (case_brief or {}).get("objectives", []),
            "cost_matrix": (case_brief or {}).get("cost_matrix"),
            "known_data_issues": (case_brief or {}).get("known_data_issues", []),
            "deliverable_requirements": (case_brief or {}).get("deliverable_requirements", []),
            "evaluation_criteria": (case_brief or {}).get("evaluation_criteria", []),
            "stakeholder_name": (case_brief or {}).get("stakeholder_name"),
            "stakeholder_role": (case_brief or {}).get("stakeholder_role"),
        } if case_brief else None,
    }
