"""Technical report - PDF (§4.2).

Full methodology document for the data science / ML team. Rendered via Jinja + weasyprint.
Includes an inline base64 SHAP bar chart generated with matplotlib.
"""

from __future__ import annotations

import base64
import io
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


def _shap_bar_chart_b64(top_features: list[str], mean_abs: list[float]) -> str | None:
    if not top_features or not mean_abs:
        return None
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        n = min(len(top_features), len(mean_abs), 15)
        feats = top_features[:n][::-1]
        vals = mean_abs[:n][::-1]

        fig, ax = plt.subplots(figsize=(6, max(3, n * 0.35)))
        bars = ax.barh(feats, vals, color="#2563eb", edgecolor="none")
        ax.set_xlabel("Mean |SHAP value|", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode()
    except Exception as exc:
        logger.debug("SHAP chart generation failed: %s", exc)
        return None


async def generate_technical_report(
    run: "Run",
    dataset: "Dataset | None",
    session: "AsyncSession",
    ctx: dict[str, Any],
) -> GeneratedDeliverable:
    from jinja2 import Environment, FileSystemLoader

    shap = ctx.get("shap", {})
    top_features = shap.get("top_features", [])
    mean_abs = shap.get("mean_abs", [])
    shap_chart_b64 = _shap_bar_chart_b64(top_features, mean_abs)

    env = Environment(loader=FileSystemLoader(_TEMPLATES_DIR), autoescape=False)
    env.filters["truncate"] = lambda s, n: (str(s)[:n] + "…") if len(str(s)) > n else str(s)

    tmpl = env.get_template("technical_report.html")
    html = tmpl.render(
        ctx=ctx,
        shap_chart_b64=shap_chart_b64,
        clinical_disclaimer=CLINICAL_DISCLAIMER_SHORT,
        generated_at=datetime.now(timezone.utc).isoformat(),
        generator_version="1.0.0",
    )

    pdf_bytes = render_pdf(html, _CSS_PATH)

    return GeneratedDeliverable.build(
        name="technical_report",
        fmt="pdf",
        content=pdf_bytes,
        run_id=run.id,
        inputs_used=[
            "eda_report", "preprocessing_strategy", "model_comparison",
            "final_metrics", "calibration_report", "threshold_result",
            "shap_summary", "drift_report", "fairness_report",
        ],
        audience="clinical informatics team, data scientist, ML engineer",
    )


