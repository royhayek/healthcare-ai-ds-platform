"""Doc generator agent - the model wrapper for narrative deliverable sections (§4, §20).

The individual deliverable generators (executive_summary.py, technical_report.py,
model_card.py) call the model directly. This agent provides a shared, reusable entry
point for any deliverable section that needs the model to write structured
narrative prose from run data.

Expected JSON schema returned by the model:
{
  "title": "<document title>",
  "sections": [
    {"heading": "<section heading>", "body": "<markdown prose>"},
    ...
  ],
  "key_findings": ["<finding 1>", ...],
  "caveats": ["<caveat 1>", ...],
  "generated_by_model": "claude-opus-4-7"
}
"""

import logging
from typing import Any

from backend.agents.base import call_claude, extract_json
from backend.core.config import settings

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """You are a senior ML engineer writing professional deliverable
documentation for a production ML project. Your audience is technical stakeholders
(data scientists, ML engineers, product leads). Write in clear, precise language.
Do not hedge or add unnecessary caveats beyond what is technically warranted.

Return ONLY valid JSON matching this schema:
{
  "title": "<document title>",
  "sections": [
    {"heading": "<section heading>", "body": "<markdown prose, 2-4 paragraphs>"},
    ...
  ],
  "key_findings": ["<concise finding>", ...],
  "caveats": ["<technical caveat>", ...],
  "generated_by_model": "claude-opus-4-7"
}
"""

_SAFE_FALLBACK: dict[str, Any] = {
    "title": "Deliverable Section",
    "sections": [{"heading": "Summary", "body": "Document generation incomplete - retry available."}],
    "key_findings": [],
    "caveats": ["model response could not be parsed - deliverable may be incomplete"],
    "generated_by_model": settings.CLAUDE_OPUS_MODEL,
}


async def generate_document_section(
    doc_type: str,
    run_summary: dict[str, Any],
    additional_context: str | None = None,
) -> dict[str, Any]:
    """Generate structured narrative text for a deliverable section.

    Args:
        doc_type: One of "executive_summary", "technical_report", "model_card",
                  "data_quality_report", "risk_register".
        run_summary: Compressed run data (metrics, model name, key findings).
                     Must never contain raw row-level data.
        additional_context: Optional free-text context appended to the prompt.

    Returns:
        Parsed JSON dict matching the schema above. Falls back to _SAFE_FALLBACK
        on parse failure - never raises.
    """
    prompt_parts = [
        f"Generate the {doc_type.replace('_', ' ')} document for this ML run.",
        "",
        "Run summary (no raw data):",
        str(run_summary),
    ]
    if additional_context:
        prompt_parts += ["", "Additional context:", additional_context]

    prompt = "\n".join(prompt_parts)

    try:
        response_text = await call_claude(
            model=settings.CLAUDE_OPUS_MODEL,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=8192,
        )
        result = extract_json(response_text)
        if not isinstance(result, dict) or "sections" not in result:
            logger.warning("doc_generator_agent: unexpected JSON shape, using fallback")
            return {**_SAFE_FALLBACK, "title": doc_type.replace("_", " ").title()}
        return result
    except Exception as exc:
        logger.warning("doc_generator_agent failed for %s: %s", doc_type, exc)
        return {**_SAFE_FALLBACK, "title": doc_type.replace("_", " ").title()}
