"""EDA agent - interprets a dataset profile and produces an EdaReport (§9).

Model: Sonnet (settings.CLAUDE_SONNET_MODEL) - reasoning quality matters
here; this report drives preprocessing, model selection, and the insight report.

Audit: appends one event (category="eda", action="eda_complete") after a
successful run, and one (action="eda_parse_failure") if JSON extraction fails.

Expected JSON schema from the model:
{
  "overview": "string - 2-3 sentence plain-language description of the dataset",
  "target_analysis": {
    "column": "string",
    "task_type": "binary_classification | multiclass | regression",
    "class_balance": {...} or null,
    "notes": "string"
  },
  "quality_issues": [
    {"column": "string or null", "issue": "string", "severity": "low|medium|high",
     "recommendation": "string"}
  ],
  "correlations": {
    "high_pairs": [...],
    "leakage_risk": [{"column": "string", "reason": "string"}]
  },
  "preprocessing_recommendations": [
    {"column": "string or null", "strategy": "string", "reason": "string"}
  ],
  "model_recommendation": "string - one of: logistic_regression, random_forest,
    gradient_boosting, xgboost, lightgbm, linear_regression, ridge_regression",
  "summary": "string - 1-2 sentence business-facing summary"
}
"""

import logging
import os
import time
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from backend.agents.base import call_claude_stream, extract_json
from backend.core import audit
from backend.core.config import settings
from backend.core.events import ProgressEmitter
from backend.models.eda import EdaReport

logger = logging.getLogger(__name__)

_SYSTEM = """You are a senior ML engineer and clinical AI specialist performing exploratory \
data analysis on healthcare datasets.
You receive a statistical profile of a patient dataset (no raw rows - only aggregates and \
population-level statistics).
Your analysis must reflect clinical domain knowledge: reference ranges for lab values, \
physiological plausibility, known risk factors, and the regulatory context of clinical AI.

Return ONLY a valid JSON object matching the schema documented in the caller.
Do not include any text outside the JSON object. Do not use markdown fences.

Clinical analysis guidelines:
- PHI columns flagged in the profile (phi_columns) must be noted in quality_issues as \
  privacy concerns that should be excluded before model training.
- Clinical range violations (clinical_range_flags) indicate physiologically implausible \
  values or known disease states that require investigation.
- ICD-10 code columns (icd_columns) should be recommended for chapter-level grouping \
  rather than raw numeric encoding.
- For imbalanced clinical outcomes (e.g., rare disease, mortality): always recommend \
  class-weight balancing or SMOTE, and note that the production threshold must be \
  calibrated against a clinical cost matrix - not left at 0.5.
- Use clinical terminology: 'patient cohort', 'clinical outcome', 'risk stratification', \
  'lab values', 'vital signs'. Do not use generic DS jargon where clinical terms apply."""

_PROMPT_TEMPLATE = """Analyze this patient dataset profile and return a JSON EDA report.

Dataset profile:
{profile_json}

Requirements:
- overview: 2-3 sentences describing the patient cohort, clinical outcome, and data quality
- target_analysis: analyze the target column; detect class imbalance
  (flag if minority class < 20% for classification; note clinical implications of imbalance)
- quality_issues: list all columns with null_pct > 5%, high skewness (|skew| > 2),
  high cardinality (n_unique > 50 for categoricals), outlier_count > 5% of rows,
  PHI columns from phi_columns, and any clinical_range_flags with clinical_concern=true.
  Each item MUST include all four fields: column, issue, severity, recommendation.
  Example: {{"column": "hba1c", "issue": "5% of values above 14% - physiologically \
  implausible; may indicate data entry errors",
            "severity": "high", "recommendation": "cap at 14.0% and flag for clinical review"}}
- correlations: summarize high_correlation_pairs; flag any that look like leakage
  (e.g., a derived column perfectly correlated with target)
- preprocessing_recommendations: concrete, column-level recommendations including
  clinically-appropriate imputation (LOCF for longitudinal vitals, median for labs,
  indicator flag for systematically missing values that may be MNAR).
  Each item MUST include all three fields: column, strategy, reason.
- model_recommendation: one model type from the allowed list - justify with clinical
  context (e.g., interpretability requirements, handling of missing lab values)
- summary: 1-2 sentences a clinical director would understand - frame around patient
  risk, not just model performance

You are writing for a clinical ML engineer. Use precise clinical terminology."""


async def run_eda_agent(
    session: AsyncSession,
    run_id: str,
    compressed_profile: dict[str, Any],
    emitter: ProgressEmitter,
) -> EdaReport:
    """Run the EDA agent and return a structured EdaReport.

    compressed_profile: output of ml.profiler.compress_profile_for_claude()
    Commits one audit event regardless of parse success/failure.
    """
    import json as _json

    await emitter.emit_async("eda", "Starting EDA analysis…", 15)

    prompt = _PROMPT_TEMPLATE.format(profile_json=_json.dumps(compressed_profile, indent=2))

    # Rate-limit chunk-level progress emissions to avoid Redis spam
    last_emit_ts = [time.monotonic()]
    accumulated = [""]

    async def on_chunk(text: str) -> None:
        accumulated[0] += text
        now = time.monotonic()
        if now - last_emit_ts[0] >= 1.0:
            await emitter.emit_async(
                "eda",
                "Analyzing dataset…",
                20,
                {"chars_received": len(accumulated[0])},
            )
            last_emit_ts[0] = now

    raw_text = await call_claude_stream(
        messages=[{"role": "user", "content": prompt}],
        model=settings.CLAUDE_SONNET_MODEL,
        system=_SYSTEM,
        max_tokens=8192,
        on_chunk=on_chunk,
    )

    await emitter.emit_async("eda", "Parsing EDA report…", 35)

    # Write raw model output to disk for debugging parse failures
    try:
        debug_dir = os.path.join(os.path.dirname(__file__), "..", "..", "data", "debug")
        os.makedirs(debug_dir, exist_ok=True)
        debug_path = os.path.join(debug_dir, f"eda_raw_{run_id}.txt")
        with open(debug_path, "w", encoding="utf-8") as _f:
            _f.write(raw_text)
        logger.info("EDA raw output saved to %s (%d chars)", debug_path, len(raw_text))
    except Exception as _exc:
        logger.warning("Could not write EDA debug file: %s", _exc)

    parsed = extract_json(raw_text)
    parse_failed = not parsed

    if parse_failed:
        logger.error("EDA agent: JSON parse failure for run %s (raw len=%d)", run_id, len(raw_text))
        report = EdaReport.safe_fallback("JSON parse failed")
    else:
        try:
            report = EdaReport.model_validate(parsed)
        except Exception as exc:
            logger.error("EDA agent: EdaReport validation failed for run %s: %s", run_id, exc)
            report = EdaReport.safe_fallback(str(exc))
            parse_failed = True

    action = "eda_parse_failure" if parse_failed else "eda_complete"
    await audit.append(
        session,
        run_id=run_id,
        actor="ai",
        category="eda",
        action=action,
        payload={
            "model": settings.CLAUDE_SONNET_MODEL,
            "raw_output_len": len(raw_text),
            "parse_failed": parse_failed,
            "quality_issues_count": len(report.quality_issues),
            "model_recommendation": report.model_recommendation,
        },
        reason=report.summary if not parse_failed else "parse failure - using safe fallback",
    )
    await session.commit()

    await emitter.emit_async("eda", "EDA complete", 40, {"model_recommendation": report.model_recommendation})
    return report
