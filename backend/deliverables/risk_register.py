"""Risk register - Markdown (§4.8).

the model writes the risk register using all pipeline results as context.
The output is a machine-readable Markdown document for compliance and operations.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from backend.deliverables.base import CLINICAL_DISCLAIMER, GeneratedDeliverable
from backend.core.config import settings

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession
    from backend.core.database import Run

logger = logging.getLogger(__name__)

_SYSTEM = """You are a senior clinical AI risk officer writing a model risk register
for a healthcare AI system.
The audience is clinical governance, compliance, operations, and patient safety officers.
Be specific. Use actual numbers from the run data. Flag concrete patient safety risks.
Rank risks by patient impact severity. Use clinical framing: missed patients (FN),
unnecessary interventions (FP), demographic disparities (equity).
Output valid Markdown only. No preamble, no conclusion outside the structure."""

_PROMPT = """Write a clinical model risk register for this healthcare AI model.

Run context:
{context_json}

Structure the register as:

# Clinical Model Risk Register - {model_name}

## 1. Patient Safety Risks
List specific patient safety concerns with evidence from the run (missed detection rate,
high-risk demographic subgroups, low similarity cohort patients).

## 2. Known Limitations
List specific technical and data limitations (low sample count for a segment,
high missingness, low performance on a subgroup).

## 3. Population Equity Concerns
If fairness was analyzed: describe the TPR disparity findings across demographic groups
and any acknowledged equity gaps.
If not analyzed: flag the absence as a clinical governance gap requiring action before deployment.

## 4. Drift & Population Shift Triggers
Specify the conditions that should trigger retraining or human clinical review, using
actual PSI values and performance thresholds from this run.

## 5. Edge Cases Not Validated
List specific patient types and clinical scenarios the model has NOT been validated against.

## 6. Monitoring Cadence
Recommend monitoring intervals - include both model performance metrics AND equity metrics
(TPR by demographic group should be re-evaluated quarterly).

## 7. Retraining Triggers
Specify concrete, measurable conditions that should trigger a retrain.

## 8. Clinical Governance Sign-off
List what approvals are needed before deploying this model in clinical practice.
Include: clinical lead, ethics review, IT security, and patient safety officer.

---

*{disclaimer}*
"""


def _summarize_for_risk(ctx: dict[str, Any]) -> dict[str, Any]:
    """Build a compact risk-focused context for the model prompt."""
    drift = ctx.get("drift", {})
    fairness = ctx.get("fairness", {})
    quality_issues = ctx.get("quality_issues", [])

    return {
        "model_name": ctx.get("model_name", "unknown"),
        "task_type": ctx.get("task_type", "unknown"),
        "final_metrics": ctx.get("final_metrics", {}),
        "dataset_rows": ctx.get("dataset", {}).get("row_count"),
        "dataset_cols": ctx.get("dataset", {}).get("col_count"),
        "optimal_threshold": ctx.get("threshold", {}).get("optimal"),
        "threshold_improvement_pct": ctx.get("threshold", {}).get("improvement_pct"),
        "top_features": ctx.get("shap", {}).get("top_features", [])[:5],
        "high_quality_issues": [
            qi for qi in quality_issues if qi.get("severity") in ("high", "medium")
        ][:5],
        "drift_overall": drift.get("overall_severity", "not_run"),
        "drift_aggregate_psi": drift.get("aggregate_psi"),
        "drift_significant_features": drift.get("significant_features", [])[:5],
        "fairness_overall": fairness.get("overall_severity", "not_run"),
        "fairness_blocks": fairness.get("blocks_deliverables", False),
        "fairness_acknowledged": fairness.get("acknowledged", False),
        "model_comparison_count": len(ctx.get("model_comparison", [])),
        "calibration_improvement_pct": ctx.get("calibration", {}).get("improvement_pct"),
    }


async def generate_risk_register(
    run: Any,
    session: "AsyncSession",
    ctx: dict[str, Any],
) -> GeneratedDeliverable:
    from backend.agents.base import call_claude

    risk_ctx = _summarize_for_risk(ctx)

    import json
    context_json = json.dumps(risk_ctx, indent=2, default=str)
    prompt = _PROMPT.format(
        model_name=ctx.get("model_name", "unknown"),
        context_json=context_json,
        disclaimer=CLINICAL_DISCLAIMER,
    )

    try:
        md_content = await call_claude(
            messages=[{"role": "user", "content": prompt}],
            model=settings.CLAUDE_OPUS_MODEL,
            system=_SYSTEM,
            max_tokens=3000,
        )
    except Exception as exc:
        logger.warning("Risk register model call failed (%s) - using stub", exc)
        md_content = _stub_risk_register(ctx)

    content_bytes = md_content.encode("utf-8")

    return GeneratedDeliverable.build(
        name="risk_register",
        fmt="md",
        content=content_bytes,
        run_id=run.id,
        inputs_used=[
            "final_metrics", "drift_report", "fairness_report",
            "shap_summary", "threshold_result",
        ],
        audience="compliance, operations, model governance",
    )


def _stub_risk_register(ctx: dict[str, Any]) -> str:
    model = ctx.get("model_name", "unknown")
    metrics = ctx.get("final_metrics", {})
    drift = ctx.get("drift", {})
    fairness = ctx.get("fairness", {})

    lines = [
        f"# Clinical Model Risk Register - {model}",
        "",
        "## 1. Patient Safety Risks",
        f"- Model: {model}",
        f"- Final metrics: {metrics}",
        "- Review miss rate (false negative rate) before clinical deployment.",
        "",
        "## 2. Known Limitations",
        f"- See Technical Report for full data quality findings.",
        "",
        "## 3. Population Equity Concerns",
        f"- Fairness severity: {fairness.get('overall_severity', 'not analyzed')}",
        "- Equity analysis must be completed before clinical deployment.",
        "",
        "## 4. Drift & Population Shift Triggers",
        f"- Drift overall severity: {drift.get('overall_severity', 'not run')}",
        f"- Aggregate PSI: {drift.get('aggregate_psi', 'N/A')}",
        "- Trigger clinical review when aggregate PSI > 0.25 on any top feature.",
        "",
        "## 5. Edge Cases Not Validated",
        "- See Technical Report for population coverage details.",
        "",
        "## 6. Monitoring Cadence",
        "- Weekly PSI checks on top features.",
        "- Monthly clinical performance review.",
        "- Quarterly equity (TPR by demographic group) review.",
        "",
        "## 7. Retraining Triggers",
        "- PSI > 0.25 on any feature in the top-5 by |SHAP|.",
        "- Detection rate drop > 5% from baseline.",
        "",
        "## 8. Clinical Governance Sign-off",
        "- Clinical lead, ethics review, IT security, and patient safety officer approval required.",
        "",
        "---",
        "",
        f"*{CLINICAL_DISCLAIMER}*",
    ]
    return "\n".join(lines)
