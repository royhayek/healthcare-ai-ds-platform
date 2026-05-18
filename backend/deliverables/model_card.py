"""Model card - Markdown + PDF (§4.3, §23).

the model fills the narrative sections; Jinja renders to HTML + weasyprint
for PDF, and to Markdown for machine-readable consumption.
Both files are returned as a list so the orchestrator can persist both.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from backend.deliverables.base import CLINICAL_DISCLAIMER, GeneratedDeliverable, render_pdf
from backend.core.config import settings

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession
    from backend.core.database import Dataset, Run

logger = logging.getLogger(__name__)

_TEMPLATES_DIR = os.path.join(os.path.dirname(__file__), "templates")
_CSS_PATH = os.path.join(_TEMPLATES_DIR, "styles", "base.css")

_SYSTEM = """You are a senior clinical ML engineer writing a model card following the
Mitchell et al. (2019) framework, adapted for clinical AI governance.
The audience is clinical governance, ethics review, and technical leads deploying
AI in a healthcare setting.
Be specific - use actual numbers from the run data. Flag genuine risks clearly.
Use clinical terminology: "false negatives" are missed patients, "false positives"
are unnecessary interventions, "sensitivity" is the detection rate.
Return valid JSON only. No preamble, no markdown fences."""

_PROMPT = """Write model card narrative sections for this clinical AI model.

Run context:
{context_json}

Return exactly this JSON structure (no extra keys):
{{
  "intended_use": "<paragraph: what clinical problem this model addresses and in what care setting>",
  "primary_intended_uses": "<paragraph: specific validated clinical deployment scenarios>",
  "out_of_scope_uses": "<paragraph: explicitly what this model should NOT be used for, including patient populations not represented in training>",
  "intended_clinical_population": "<paragraph: age range, condition, acuity level, and any exclusion criteria for safe use>",
  "contraindications": "<paragraph: patient types or situations where this model must NOT be used>",
  "relevant_factors": "<paragraph: demographic, clinical, and data factors that affect model performance - include any equity findings>",
  "evaluation_factors": "<paragraph: how the model was evaluated - CV strategy, seeds, cost matrix, threshold optimisation>",
  "unitary_results": "<paragraph: overall clinical performance - detection rate, miss rate, false alarm rate, with actual metric values>",
  "intersectional_results": "<paragraph: equity analysis across demographic groups, or explicitly state it was not run and flag as a gap>",
  "ethical_considerations": "<paragraph: clinical safety risks (missed patients), demographic bias findings, and required mitigations>",
  "caveats": "<paragraph: limitations, known failure modes for edge-case populations, and mandatory deployment safeguards>"
}}

Rules:
- Use actual metric values from final_metrics throughout
- Reference sensitivity/specificity with clinical framing (missed patients, unnecessary referrals)
- If fairness was not run, say so explicitly in intersectional_results and flag it as a clinical governance gap
- out_of_scope_uses and contraindications: be specific about populations NOT validated
- ethical_considerations: reference actual fairness severity and blocks_deliverables flag if present
"""

_FALLBACK_NARRATIVE = {
    "intended_use": "This model was trained to perform the specified clinical prediction task on the provided patient cohort.",
    "primary_intended_uses": "Use in the clinical domain and patient population represented by the training cohort.",
    "out_of_scope_uses": "Do not use outside the training data distribution without validation. Do not use for high-stakes clinical decisions without clinician review.",
    "intended_clinical_population": "Patients whose profile is similar to the training cohort. See dataset documentation for inclusion/exclusion criteria.",
    "contraindications": "Do not apply this model to patient populations not represented in the training cohort without prospective validation. Do not use as the sole basis for clinical action.",
    "relevant_factors": "Clinical performance may vary across demographic subgroups. Equity analysis results are reported separately.",
    "evaluation_factors": "Model evaluated using cross-validation with multiple seeds for stability. Threshold optimised using clinical cost matrix. See technical report for full methodology.",
    "unitary_results": "See clinical performance metrics (detection rate, miss rate, false alarm rate) in the performance section of this card.",
    "intersectional_results": "Fairness analysis results are available in the equity section of the technical report. If not run, this represents a clinical governance gap that should be addressed before deployment.",
    "ethical_considerations": "Review fairness findings and risk register before clinical deployment. Missed patients (false negatives) represent the primary clinical safety risk.",
    "caveats": "This model must be validated on prospective data before clinical deployment. Clinician oversight is mandatory. Monitor for population drift and demographic performance gaps.",
}


def _summarize_for_card(ctx: dict[str, Any]) -> dict[str, Any]:
    drift = ctx.get("drift", {})
    fairness = ctx.get("fairness", {})
    comparison = ctx.get("model_comparison", [])
    stat_tests = ctx.get("stat_tests", {})

    return {
        "model_name": ctx.get("model_name", "unknown"),
        "task_type": ctx.get("task_type", "unknown"),
        "dataset_rows": ctx.get("dataset", {}).get("row_count"),
        "dataset_cols": ctx.get("dataset", {}).get("col_count"),
        "target_column": ctx.get("dataset", {}).get("target_column"),
        "final_metrics": ctx.get("final_metrics", {}),
        "optimal_threshold": ctx.get("threshold", {}).get("optimal"),
        "threshold_improvement_pct": ctx.get("threshold", {}).get("improvement_pct"),
        "top_features": ctx.get("shap", {}).get("top_features", [])[:10],
        "models_evaluated": len(comparison),
        "runner_up": comparison[1] if len(comparison) > 1 else None,
        "stat_test_significant": stat_tests.get("significant", False),
        "calibration_method": ctx.get("calibration", {}).get("method"),
        "drift_severity": drift.get("overall_severity", "not_run"),
        "drift_significant_features": drift.get("significant_features", []),
        "fairness_severity": fairness.get("overall_severity", "not_run"),
        "fairness_blocks": fairness.get("blocks_deliverables", False),
        "fairness_protected_columns": [
            b.get("attribute") for b in fairness.get("attributes", [])
        ],
        "seeds_used": ctx.get("seeds", {}),
    }


def _render_markdown(ctx: dict[str, Any], narrative: dict[str, Any]) -> str:
    fm = ctx.get("final_metrics", {})
    shap = ctx.get("shap", {})
    drift = ctx.get("drift", {})
    fairness = ctx.get("fairness", {})
    dataset = ctx.get("dataset", {})
    threshold = ctx.get("threshold", {})

    lines = [
        f"# Model Card - {ctx.get('model_name', 'unknown')}",
        "",
        f"**Task type:** {ctx.get('task_type', 'unknown').replace('_', ' ').title()}  ",
        f"**Run ID:** {ctx.get('run_id', '')}  ",
        f"**Training dataset:** {dataset.get('filename', 'unknown')} ({dataset.get('row_count')} rows)  ",
        f"**Target column:** {dataset.get('target_column', 'unknown')}  ",
        f"**Generated:** {datetime.now(timezone.utc).date().isoformat()}  ",
        "",
        "## Performance Metrics",
        "",
    ]
    for name, val in fm.items():
        lines.append(f"- **{name.upper()}:** {val:.4f}")
    if threshold.get("optimal"):
        lines.append(f"- **Optimal threshold:** {threshold['optimal']:.4f}")
    if threshold.get("improvement_pct"):
        lines.append(f"- **Threshold improvement:** {threshold['improvement_pct']:.1f}% over default 0.5")

    lines += [
        "",
        "## Intended Use",
        "",
        narrative["intended_use"],
        "",
        "### Primary Intended Uses",
        "",
        narrative["primary_intended_uses"],
        "",
        "### Out-of-Scope Uses",
        "",
        narrative["out_of_scope_uses"],
        "",
        "## Relevant Factors",
        "",
        narrative["relevant_factors"],
        "",
        "## Evaluation",
        "",
        narrative["evaluation_factors"],
        "",
        "### Results",
        "",
        narrative["unitary_results"],
        "",
    ]

    if shap.get("top_features"):
        lines += ["### Top Predictive Features", ""]
        for i, feat in enumerate(shap["top_features"]):
            mean_abs_list = shap.get("mean_abs", [])
            val_str = f" (mean |SHAP| = {mean_abs_list[i]:.4f})" if i < len(mean_abs_list) else ""
            lines.append(f"{i+1}. {feat}{val_str}")
        lines.append("")

    lines += [
        "## Fairness Analysis",
        "",
        narrative["intersectional_results"],
        "",
    ]

    if drift.get("overall_severity") and drift["overall_severity"] != "not_run":
        lines += [
            "## Drift Assessment",
            "",
            f"**Overall severity:** {drift['overall_severity'].upper()}",
            "",
        ]
        if drift.get("significant_features"):
            lines.append(f"Significant drift in: {', '.join(drift['significant_features'])}")
            lines.append("")

    lines += [
        "## Intended Clinical Population",
        "",
        narrative.get("intended_clinical_population", "See training dataset documentation."),
        "",
        "### Contraindications",
        "",
        narrative.get("contraindications", "Do not apply outside the validated patient population."),
        "",
        "## Ethical Considerations",
        "",
        narrative["ethical_considerations"],
        "",
        "## Caveats and Recommendations",
        "",
        narrative["caveats"],
        "",
        "---",
        "",
        f"*{CLINICAL_DISCLAIMER}*",
    ]

    return "\n".join(lines)


async def generate_model_card(
    run: Any,
    dataset: "Dataset | None",
    session: "AsyncSession",
    ctx: dict[str, Any],
) -> list[GeneratedDeliverable]:
    """Returns two deliverables: Markdown and PDF."""
    from backend.agents.base import call_claude, extract_json
    from jinja2 import Environment, FileSystemLoader

    card_ctx = _summarize_for_card(ctx)
    context_json = json.dumps(card_ctx, indent=2, default=str)
    prompt = _PROMPT.format(context_json=context_json)

    narrative = _FALLBACK_NARRATIVE.copy()
    try:
        raw = await call_claude(
            messages=[{"role": "user", "content": prompt}],
            model=settings.CLAUDE_OPUS_MODEL,
            system=_SYSTEM,
            max_tokens=2500,
        )
        parsed = extract_json(raw)
        if isinstance(parsed, dict) and "intended_use" in parsed:
            # Merge with fallback so any missing clinical keys still have content
            narrative = {**_FALLBACK_NARRATIVE, **parsed}
        else:
            logger.warning("Model card the model returned unexpected structure - using fallback")
    except Exception as exc:
        logger.warning("Model card model call failed (%s) - using fallback", exc)

    now = datetime.now(timezone.utc).isoformat()

    # Markdown deliverable
    md_content = _render_markdown(ctx, narrative)
    md_deliverable = GeneratedDeliverable.build(
        name="model_card",
        fmt="md",
        content=md_content.encode("utf-8"),
        run_id=run.id,
        inputs_used=[
            "final_metrics", "shap_summary", "drift_report",
            "fairness_report", "model_comparison", "threshold_result",
        ],
        audience="model governance, ethics review, technical lead",
    )

    # PDF deliverable
    env = Environment(loader=FileSystemLoader(_TEMPLATES_DIR), autoescape=False)
    tmpl = env.get_template("model_card.html")
    html = tmpl.render(
        ctx=ctx,
        narrative=narrative,
        generated_at=now,
        generator_version="1.0.0",
    )
    pdf_bytes = render_pdf(html, _CSS_PATH)
    pdf_deliverable = GeneratedDeliverable.build(
        name="model_card_pdf",
        fmt="pdf",
        content=pdf_bytes,
        run_id=run.id,
        inputs_used=[
            "final_metrics", "shap_summary", "drift_report",
            "fairness_report", "model_comparison", "threshold_result",
        ],
        audience="model governance, ethics review, technical lead",
    )

    return [md_deliverable, pdf_deliverable]


