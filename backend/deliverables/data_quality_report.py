"""Data quality report - PDF (§4.4).

Rendered from the profiler output and preprocessing strategy via Jinja + weasyprint.
No model call - the data speaks for itself.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from backend.deliverables.base import CLINICAL_DISCLAIMER_SHORT, GeneratedDeliverable, render_pdf

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession
    from backend.core.database import Dataset, Run

logger = logging.getLogger(__name__)

_TEMPLATES_DIR = os.path.join(os.path.dirname(__file__), "templates")
_CSS_PATH = os.path.join(_TEMPLATES_DIR, "styles", "base.css")


def _build_column_profiles(profile: dict[str, Any]) -> list[dict[str, Any]]:
    raw = profile.get("columns", [])
    # profile["columns"] is a list[dict] from the profiler, not a dict
    if isinstance(raw, dict):
        items = raw.items()
    else:
        items = ((c["name"], c) for c in raw if isinstance(c, dict) and "name" in c)

    columns: list[dict[str, Any]] = []
    for col_name, col_data in items:
        missing_pct = col_data.get("missing_pct", 0.0) or 0.0
        notes_parts = []
        if col_data.get("has_outliers"):
            notes_parts.append("outliers present")
        if col_data.get("skewness") and abs(col_data.get("skewness", 0)) > 2:
            notes_parts.append(f"high skew ({col_data['skewness']:.2f})")
        if col_data.get("cardinality_ratio") and col_data["cardinality_ratio"] > 0.9:
            notes_parts.append("high cardinality")
        columns.append({
            "name": col_name,
            "dtype": col_data.get("dtype", ""),
            "missing_pct": missing_pct,
            "n_unique": col_data.get("n_unique"),
            "notes": "; ".join(notes_parts),
        })
    return columns


def _extract_clinical_findings(profile: dict[str, Any]) -> dict[str, Any]:
    """Pull PHI flags, clinical range violations, and ICD column detections from profile."""
    phi_columns = profile.get("phi_columns") or []
    clinical_range_flags = profile.get("clinical_range_flags") or {}
    icd_columns = profile.get("icd_columns") or []

    # Only keep range flags with clinical concerns
    range_concerns = {
        col: flags for col, flags in clinical_range_flags.items()
        if isinstance(flags, dict) and flags.get("clinical_concern")
    }

    return {
        "phi_columns": phi_columns,
        "clinical_range_concerns": range_concerns,
        "icd_columns": icd_columns,
        "has_phi": len(phi_columns) > 0,
        "has_range_concerns": len(range_concerns) > 0,
        "has_icd": len(icd_columns) > 0,
    }


async def generate_data_quality_report(
    run: "Run",
    dataset: "Dataset | None",
    session: "AsyncSession",
    ctx: dict[str, Any],
) -> GeneratedDeliverable:
    from jinja2 import Environment, FileSystemLoader

    profile = (dataset.profile or {}) if dataset else {}
    column_profiles = _build_column_profiles(profile)
    clinical_findings = _extract_clinical_findings(profile)

    env = Environment(loader=FileSystemLoader(_TEMPLATES_DIR), autoescape=False)
    env.filters["truncate"] = lambda s, n: (str(s)[:n] + "…") if len(str(s)) > n else str(s)
    tmpl = env.get_template("data_quality_report.html")

    html = tmpl.render(
        ctx=ctx,
        column_profiles=column_profiles,
        clinical_findings=clinical_findings,
        clinical_disclaimer=CLINICAL_DISCLAIMER_SHORT,
        generated_at=datetime.now(timezone.utc).isoformat(),
        generator_version="1.0.0",
    )

    pdf_bytes = render_pdf(html, _CSS_PATH)

    return GeneratedDeliverable.build(
        name="data_quality_report",
        fmt="pdf",
        content=pdf_bytes,
        run_id=run.id,
        inputs_used=["dataset_profile", "preprocessing_strategy"],
        audience="clinical data steward, data engineer, data owner",
    )


