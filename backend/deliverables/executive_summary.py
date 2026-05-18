"""Executive summary - PDF (§4.1, §23).

the model writes structured JSON content; Jinja renders it to HTML;
weasyprint converts to PDF. One page maximum for the exec audience.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from backend.deliverables.base import CLINICAL_DISCLAIMER_SHORT, GeneratedDeliverable, render_pdf
from backend.core.config import settings

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession
    from backend.core.database import Dataset, Run

logger = logging.getLogger(__name__)

_TEMPLATES_DIR = os.path.join(os.path.dirname(__file__), "templates")
_CSS_PATH = os.path.join(_TEMPLATES_DIR, "styles", "base.css")

_SYSTEM = """You are a senior clinical data scientist writing an executive summary for
healthcare leadership (CMO, CMIO, clinical leads, risk owners).
Be concise, decisive, and specific. Use actual numbers from the run data.
Use clinical language: detection rate, missed patients, unnecessary interventions.
Avoid raw ML jargon. Do not hedge everything - lead with the clinical result.
Return valid JSON only. No preamble, no markdown fences."""

_PROMPT = """Write an executive summary for this ML model run.

Run context:
{context_json}

Return exactly this JSON structure (no extra keys):
{{
  "one_liner": "<one sentence: what the model does and its headline performance>",
  "performance_paragraph": "<2-3 sentences: key metrics, threshold choice, calibration quality>",
  "business_paragraph": "<2-3 sentences: what this means for the business, cost/benefit if threshold data is available>",
  "insights": ["<insight 1>", "<insight 2>", "<insight 3>"],
  "risks": ["<risk 1>", "<risk 2>", "<risk 3>"],
  "next_steps_paragraph": "<2-3 sentences: concrete recommended next steps before or after production deployment>"
}}

Rules:
- one_liner: include the primary metric value (e.g. AUC, F1, RMSE)
- performance_paragraph: use actual numbers from final_metrics and threshold data
- business_paragraph: translate model quality to business terms; if cost matrix data is present, use it
- insights: must be actionable, specific, and derived from this run's data
- risks: must reference actual findings (drift severity, fairness issues, low sample counts, etc.)
- next_steps_paragraph: concrete actions, not generic advice
"""

_FALLBACK_CONTENT = {
    "one_liner": "Model training completed - see metrics below.",
    "performance_paragraph": "Final metrics are available in the technical report.",
    "business_paragraph": "Review the full technical report for business impact analysis.",
    "insights": [
        "Review SHAP feature importance for actionable signals.",
        "Validate optimal threshold against business cost matrix.",
        "Monitor model performance after deployment.",
    ],
    "risks": [
        "Ensure drift monitoring is configured for production.",
        "Review fairness findings before deployment.",
        "Validate on holdout dataset before launch.",
    ],
    "next_steps_paragraph": "Review the full technical report and risk register before production deployment.",
}


def _summarize_for_exec(ctx: dict[str, Any]) -> dict[str, Any]:
    drift = ctx.get("drift", {})
    fairness = ctx.get("fairness", {})
    comparison = ctx.get("model_comparison", [])

    return {
        "model_name": ctx.get("model_name", "unknown"),
        "task_type": ctx.get("task_type", "unknown"),
        "dataset_rows": ctx.get("dataset", {}).get("row_count"),
        "target_column": ctx.get("dataset", {}).get("target_column"),
        "final_metrics": ctx.get("final_metrics", {}),
        "optimal_threshold": ctx.get("threshold", {}).get("optimal"),
        "threshold_improvement_pct": ctx.get("threshold", {}).get("improvement_pct"),
        "top_features": ctx.get("shap", {}).get("top_features", [])[:5],
        "models_evaluated": len(comparison),
        "best_model": comparison[0] if comparison else {},
        "drift_severity": drift.get("overall_severity", "not_run"),
        "drift_significant_features": drift.get("significant_features", [])[:3],
        "fairness_severity": fairness.get("overall_severity", "not_run"),
        "fairness_blocks": fairness.get("blocks_deliverables", False),
        "calibration_method": ctx.get("calibration", {}).get("method"),
        "calibration_improvement": ctx.get("calibration", {}).get("improvement_pct"),
    }


async def generate_executive_summary(
    run: Any,
    dataset: "Dataset | None",
    session: "AsyncSession",
    ctx: dict[str, Any],
) -> GeneratedDeliverable:
    from backend.agents.base import call_claude, extract_json
    from jinja2 import Environment, FileSystemLoader

    exec_ctx = _summarize_for_exec(ctx)
    context_json = json.dumps(exec_ctx, indent=2, default=str)
    prompt = _PROMPT.format(context_json=context_json)

    content_dict = _FALLBACK_CONTENT
    try:
        raw = await call_claude(
            messages=[{"role": "user", "content": prompt}],
            model=settings.CLAUDE_OPUS_MODEL,
            system=_SYSTEM,
            max_tokens=1500,
        )
        parsed = extract_json(raw)
        if isinstance(parsed, dict) and "one_liner" in parsed:
            content_dict = parsed
        else:
            logger.warning("Executive summary the model returned unexpected structure - using fallback")
    except Exception as exc:
        logger.warning("Executive summary model call failed (%s) - using fallback", exc)

    env = Environment(loader=FileSystemLoader(_TEMPLATES_DIR), autoescape=False)
    tmpl = env.get_template("executive_summary.html")
    html = tmpl.render(
        ctx=ctx,
        content=content_dict,
        clinical_disclaimer=CLINICAL_DISCLAIMER_SHORT,
        generated_at=datetime.now(timezone.utc).isoformat(),
        generator_version="1.0.0",
    )

    pdf_bytes = render_pdf(html, _CSS_PATH)

    return GeneratedDeliverable.build(
        name="executive_summary",
        fmt="pdf",
        content=pdf_bytes,
        run_id=run.id,
        inputs_used=[
            "final_metrics", "threshold_result", "shap_summary",
            "drift_report", "fairness_report", "model_comparison",
        ],
        audience="C-suite, product lead, risk owner",
    )


