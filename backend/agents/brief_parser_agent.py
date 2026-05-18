"""Case brief parser agent (§2, §20).

Parses raw case brief text (extracted from PDF/DOCX/TXT/MD) into the structured
CaseBrief fields consumed by downstream agents:
  - objectives       → what questions the stakeholder wants answered
  - cost_matrix      → FP/FN cost and TP/TN value for threshold optimization
  - known_data_issues → leaky features or data problems mentioned in the brief
  - deliverable_requirements → what the stakeholder explicitly asked for
  - evaluation_criteria → what they will push back on
  - stakeholder_name / stakeholder_role → who commissioned the work

Expected JSON response schema:
{
  "objectives": ["string", ...],
  "cost_matrix": {"fp_cost": float, "fn_cost": float, "tp_value": float, "tn_value": float} | null,
  "known_data_issues": ["string", ...],
  "deliverable_requirements": ["string", ...],
  "evaluation_criteria": ["string", ...],
  "stakeholder_name": "string | null",
  "stakeholder_role": "string | null"
}
"""

import logging
from typing import Any

from backend.agents.base import call_claude, extract_json
from backend.core.config import settings

logger = logging.getLogger(__name__)

_SYSTEM = """You are parsing a business case brief written by a non-technical stakeholder
for a data science engagement. Extract the following fields as JSON:

- objectives: list of specific questions or goals the stakeholder wants answered.
  Quote or closely paraphrase the actual questions, do not generalize.
- cost_matrix: if the brief mentions specific costs or values per action/outcome,
  extract them as {"fp_cost": <float>, "fn_cost": <float>, "tp_value": <float>, "tn_value": <float>}.
  fp_cost = cost of a false positive (e.g. cost of a wasted call, bad loan approved).
  fn_cost = cost of a false negative (e.g. missed churn, missed fraud).
  tp_value = value/revenue of a true positive (e.g. margin per conversion, fraud caught).
  tn_value = value of a true negative (usually 0).
  Set to null if no specific numbers are given.
- known_data_issues: list of data quality concerns, leaky features, or caveats the
  stakeholder explicitly mentions (e.g. "call duration is known to leak the outcome").
- deliverable_requirements: list of specific outputs the stakeholder asked for
  (e.g. "a lift/gains curve", "a reproducible notebook", "a 5-page report").
  Only include what was explicitly requested - do not infer.
- evaluation_criteria: list of things the stakeholder said they will push back on
  (e.g. "model evaluated only on accuracy", "black-box recommendations").
- stakeholder_name: the name of the person who wrote the brief, or null.
- stakeholder_role: their role/title, or null.

Return only valid JSON matching the schema above. No prose, no markdown fences."""


async def parse_case_brief(raw_text: str) -> dict[str, Any]:
    """Parse raw brief text into structured fields.

    Returns a dict suitable for merging into a CaseBrief model.
    Never raises - returns empty structure on parse failure.
    """
    if not raw_text.strip():
        return _empty()

    prompt = f"""<case_brief>
{raw_text}
</case_brief>

Extract the structured fields from this brief."""

    try:
        raw = await call_claude(
            messages=[{"role": "user", "content": prompt}],
            model=settings.CLAUDE_SONNET_MODEL,
            system=_SYSTEM,
            max_tokens=1024,
        )
    except Exception as exc:
        logger.warning("brief_parser_agent: model call failed: %s", exc)
        return _empty()

    parsed = extract_json(raw, fallback=None)
    if not parsed:
        logger.warning("brief_parser_agent: could not parse JSON from model output")
        return _empty()

    return {
        "objectives": _coerce_list(parsed.get("objectives")),
        "cost_matrix": _coerce_cost_matrix(parsed.get("cost_matrix")),
        "known_data_issues": _coerce_list(parsed.get("known_data_issues")),
        "deliverable_requirements": _coerce_list(parsed.get("deliverable_requirements")),
        "evaluation_criteria": _coerce_list(parsed.get("evaluation_criteria")),
        "stakeholder_name": parsed.get("stakeholder_name") or None,
        "stakeholder_role": parsed.get("stakeholder_role") or None,
        "parsed": True,
    }


def _empty() -> dict[str, Any]:
    return {
        "objectives": [],
        "cost_matrix": None,
        "known_data_issues": [],
        "deliverable_requirements": [],
        "evaluation_criteria": [],
        "stakeholder_name": None,
        "stakeholder_role": None,
        "parsed": False,
    }


def _coerce_list(val: Any) -> list[str]:
    if isinstance(val, list):
        return [str(v) for v in val if v]
    return []


def _coerce_cost_matrix(val: Any) -> dict[str, float] | None:
    if not isinstance(val, dict):
        return None
    try:
        return {
            "fp_cost": float(val.get("fp_cost", 0)),
            "fn_cost": float(val.get("fn_cost", 0)),
            "tp_value": float(val.get("tp_value", 0)),
            "tn_value": float(val.get("tn_value", 0)),
        }
    except (TypeError, ValueError):
        return None
