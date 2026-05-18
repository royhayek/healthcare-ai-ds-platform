"""Haiku-based intent classifier for the chat co-pilot (§21).

Runs in parallel with the Sonnet streaming response via asyncio.gather in
chat_agent.py. Returns a ChatIntent even on parse failure - never None -
so the caller never needs to guard against None.

Expected JSON schema from the model:
{
  "intent": "question | modify | abort | request_artifact | navigate",
  "confidence": 0.0-1.0,
  "category": "eda | preprocessing | model_selection | threshold | fairness | drift | deliverables | request_plot | general",
  "structured_payload": {
    // For "modify" intent - what field to change and to what value:
    "column": "age",                    // optional: which column
    "field": "imputation_strategy",     // optional: which strategy field
    "value": "median"                   // optional: new value
    // For model_selection:
    "model": "random_forest"            // optional
    // For abort: {} (empty)
    // For question: {} (empty)
  },
  "needs_confirmation": true,
  "reasoning": "one-sentence explanation"
}
"""

import logging

from backend.agents.base import call_claude, extract_json
from backend.core.config import settings
from backend.models.chat import ChatIntent

logger = logging.getLogger(__name__)

_SYSTEM = (
    "You are a precise intent classifier for a clinical AI pipeline chat interface. "
    "The user is a clinical ML engineer or healthcare professional. "
    "Return only valid JSON matching the schema - no prose, no markdown fences."
)

_PROMPT = """Classify the intent of this user message in the context of a healthcare AI pipeline.

User message:
{message}

Pipeline context summary:
{context_summary}

Return JSON with this schema:
{{
  "intent": "question | modify | abort | request_artifact | navigate",
  "confidence": 0.0-1.0,
  "category": "eda | preprocessing | model_selection | threshold | fairness | drift | \
deliverables | request_plot | clinical_query | equity_query | threshold_query | general",
  "structured_payload": {{}},
  "needs_confirmation": false,
  "reasoning": "brief explanation"
}}

Rules:
- "modify" intent: structured_payload must include at minimum one of: column, field, value, model
- "abort" intent: needs_confirmation must be true
- "question" intent: structured_payload is {{}}
- confidence < 0.5 → set intent to "question" when uncertain
- category must match the pipeline area the message is about

Clinical category rules:
- use category "clinical_query" for questions about clinical meaning of model outputs
  (e.g. "what are the top risk factors?", "which patient characteristics drive the score?",
  "what does a high risk score mean clinically?")
- use category "equity_query" for questions about demographic fairness or disparity
  (e.g. "are predictions fair across groups?", "is there bias toward elderly patients?",
  "show me equity across insurance types", "do outcomes differ by gender?")
- use category "threshold_query" for questions about decision threshold and clinical costs
  (e.g. "how does the threshold affect missed diagnoses?", "what's the cost of a false negative?",
  "adjust the threshold for higher sensitivity", "set FN cost to 10x FP")

General rules:
- use "request_plot" when the user asks to see or generate a specific plot
  (e.g. "show me the outlier distribution", "plot the correlation heatmap");
  set structured_payload {{"stage": "<stage>"}} where stage ∈ {{eda, preprocessing, training, drift}}
- use intent "request_artifact" with category "deliverables" and
  structured_payload {{"artifact_type": "notebook"}} when the user asks to
  export a Jupyter notebook (e.g. "export as notebook", "give me a .ipynb")"""


async def classify_intent(user_message: str, context_summary: str) -> ChatIntent:
    """Classify the user message using Haiku. Never raises - returns fallback on failure."""
    prompt = _PROMPT.format(message=user_message, context_summary=context_summary)
    try:
        raw = await call_claude(
            messages=[{"role": "user", "content": prompt}],
            model=settings.CLAUDE_HAIKU_MODEL,
            system=_SYSTEM,
            max_tokens=512,
        )
        parsed = extract_json(raw)
        if not parsed:
            logger.warning("intent_extractor: empty JSON from Haiku")
            return ChatIntent.question_fallback()
        return ChatIntent.model_validate(parsed)
    except Exception as exc:
        logger.warning("intent_extractor: classification failed: %s", exc)
        return ChatIntent.question_fallback()
