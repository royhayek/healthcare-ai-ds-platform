"""Insight report agent - final analysis using the model (§18).

Uses Opus for the insight report because this is the highest-stakes output
in the pipeline: it synthesises all pipeline results into a business-facing
narrative. Quality matters more than speed here.

Consumes SHAP summary + training profile - NEVER raw rows.

Streams the response to the progress feed. The full text is stored as
run.insight_report (plain markdown, not JSON).
"""

import logging
import time
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from backend.agents.base import call_claude_stream
from backend.core import audit
from backend.core.config import settings
from backend.core.events import ProgressEmitter

logger = logging.getLogger(__name__)

_SYSTEM = """You are a senior clinical AI specialist writing a final analysis report \
for a healthcare predictive model.
Write for a clinical informatics audience: clinical data analysts, ML engineers, \
and clinical leadership.
Be precise. Use actual numbers. Frame findings in clinical terms (patient risk, \
population health, clinical equity).

Structure the report with these sections:
1. Clinical Summary (2-3 sentences for clinical leadership - patient impact framing)
2. Model Performance (metrics, stability across seeds and folds, calibration quality)
3. Clinical Risk Factors (top SHAP features interpreted clinically, not just statistically)
4. Clinical Decision Threshold (optimal threshold, cost of missed cases vs false alarms)
5. Population Equity Analysis (fairness across demographic groups - mandatory section)
6. Data Quality and Limitations (PHI handling, missing values, known failure modes)
7. Clinical Recommendations (actionable, specific to the patient population)

Important clinical framing:
- False negatives = missed patients at risk. Always quantify: "At threshold X, \
  Y% of positive cases are missed."
- Calibration matters for clinical use: Brier score and ECE tell clinicians how \
  much to trust the probability.
- Equity gaps > 5% between demographic groups should be flagged as requiring \
  clinical review before deployment.
- End with: "This model is intended to assist, not replace, clinician judgment."
"""

_PROMPT_TEMPLATE = """Write the final clinical AI insight report for this healthcare model.

Task type: {task_type}
Best model: {model_name}

Performance (mean ± std across {n_seeds} seeds × {n_folds} folds):
{performance_summary}

Calibration:
{calibration_summary}

Threshold optimization (clinical cost matrix):
{threshold_summary}

Top SHAP predictors (mean |SHAP| - interpret clinically):
{shap_summary}

EDA summary:
{eda_summary}

Data quality and clinical range concerns:
{quality_notes}

Statistical test results (if run):
{stat_test_summary}

Write a complete clinical report. Use markdown headers. Quote actual numbers.
Frame all findings in terms of patient risk and clinical decision-making.
This report goes to clinical reviewers - be clinically precise."""


async def run_insight_agent(
    session: AsyncSession,
    run_id: str,
    task_type: str,
    model_name: str,
    stability_results: list[dict[str, Any]],
    calibration_report: dict[str, Any] | None,
    threshold_result: dict[str, Any] | None,
    shap_summary: dict[str, Any],
    eda_report: dict[str, Any],
    stat_tests: dict[str, Any] | None,
    emitter: ProgressEmitter,
) -> str:
    """Run the insight agent. Returns the full report as a markdown string.

    Uses Opus for higher quality. Streams to the progress feed.
    Commits one audit event on completion.
    """
    import json as _json

    await emitter.emit_async("insight", "Generating insight report (Opus)…", 88)

    # Build structured performance summary
    best_result = stability_results[0] if stability_results else {}
    perf_lines = []
    for r in stability_results[:3]:
        perf_lines.append(
            f"  {r.get('model_name', '?')}: {r.get('mean', 0):.4f} ± {r.get('std', 0):.4f} "
            f"(overfit gap: {r.get('overfit_gap', 0):.4f})"
        )
    performance_summary = "\n".join(perf_lines) if perf_lines else "No stability results available."

    cal_summary = "Not run (regression task)."
    if calibration_report:
        cal_summary = (
            f"Method: {calibration_report.get('method', '?')}, "
            f"Brier {calibration_report.get('brier_before', 0):.4f} → "
            f"{calibration_report.get('brier_after', 0):.4f} "
            f"({calibration_report.get('improvement_pct', 0):.1f}% improvement), "
            f"ECE {calibration_report.get('ece_before', 0):.4f} → "
            f"{calibration_report.get('ece_after', 0):.4f}"
        )

    thr_summary = "Not run (regression or not applicable)."
    if threshold_result:
        thr_summary = (
            f"Optimal threshold: {threshold_result.get('optimal_threshold', 0.5):.3f} "
            f"(default 0.5 cost: {threshold_result.get('cost_at_default', 0):.2f}, "
            f"optimal cost: {threshold_result.get('cost_at_optimal', 0):.2f}, "
            f"improvement: {threshold_result.get('improvement_pct', 0):.1f}%)"
        )

    top_features = shap_summary.get("top_k_features", [])
    mean_abs = shap_summary.get("mean_abs_shap", [])
    feature_names = shap_summary.get("feature_names", [])
    shap_lines = []
    for i, feat in enumerate(top_features[:10]):
        try:
            idx = feature_names.index(feat)
            val = mean_abs[idx]
            shap_lines.append(f"  {feat}: {val:.4f}")
        except (ValueError, IndexError):
            shap_lines.append(f"  {feat}")
    shap_text = "\n".join(shap_lines) if shap_lines else "SHAP not computed."

    quality_notes = "; ".join(
        issue.get("recommendation", "") for issue in eda_report.get("quality_issues", [])
        if issue.get("severity") in ("medium", "high")
    ) or "No significant quality issues."

    stat_summary = "Not run (models were clearly separated)."
    if stat_tests:
        stat_summary = (
            f"{stat_tests.get('test_name', '?').upper()}: "
            f"p={stat_tests.get('p_value', 1.0):.4f} - "
            f"{stat_tests.get('interpretation', '')}"
        )

    n_seeds = len(set(s.get("seed", 42) for s in stability_results)) if stability_results else 3
    n_folds = (
        len(stability_results[0].get("scores", [])) // n_seeds
        if stability_results
        else 5
    )

    prompt = _PROMPT_TEMPLATE.format(
        task_type=task_type,
        model_name=model_name,
        n_seeds=n_seeds,
        n_folds=n_folds,
        performance_summary=performance_summary,
        calibration_summary=cal_summary,
        threshold_summary=thr_summary,
        shap_summary=shap_text,
        eda_summary=eda_report.get("summary", ""),
        quality_notes=quality_notes,
        stat_test_summary=stat_summary,
    )

    # Streaming with rate-limited progress
    last_emit_ts = [time.monotonic()]
    chars_received = [0]

    async def on_chunk(text: str) -> None:
        chars_received[0] += len(text)
        now = time.monotonic()
        if now - last_emit_ts[0] >= 2.0:
            await emitter.emit_async(
                "insight", "Writing report…", 90,
                {"chars_received": chars_received[0]}
            )
            last_emit_ts[0] = now

    report_text = await call_claude_stream(
        messages=[{"role": "user", "content": prompt}],
        model=settings.CLAUDE_OPUS_MODEL,
        system=_SYSTEM,
        max_tokens=8192,
        on_chunk=on_chunk,
    )

    await audit.append(
        session,
        run_id=run_id,
        actor="ai",
        category="insight",
        action="insight_complete",
        payload={
            "model": settings.CLAUDE_OPUS_MODEL,
            "report_length_chars": len(report_text),
            "top_features": top_features[:5],
        },
        reason=f"Insight report generated using {settings.CLAUDE_OPUS_MODEL}",
    )
    await session.commit()

    await emitter.emit_async("insight", "Insight report complete", 92)
    return report_text
