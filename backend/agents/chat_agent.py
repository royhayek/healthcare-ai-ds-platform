"""Persistent chat co-pilot (§2, §21).

Two-pass parallel pattern - visible at call site, not buried:

    response_text, intent = await asyncio.gather(
        call_claude_stream(..., on_chunk=on_chunk),   # Sonnet, streaming
        classify_intent(user_message, context_summary),  # Haiku, concurrent
    )

Both calls fire simultaneously. Sonnet streams chunks to the SSE queue while
Haiku classifies intent. Total latency ≈ max(sonnet, haiku), not their sum.

Context window management:
  - Last 20 turns included verbatim in messages list (history slice in router).
  - Turns 21-80 summarized by Haiku [deferred: Step 5+, when history grows].
  - Context block includes run state, datasets, EDA report, recent audit events.
"""

import asyncio
import json
import logging
from collections.abc import Awaitable, Callable
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.agents.base import call_claude_stream
from backend.agents.intent_extractor import classify_intent
from backend.core.config import settings
from backend.core.database import AuditEvent, ChatMessage as DBChatMessage, Dataset, Project, Run
from backend.core.strategy_mutator import apply_intent_to_strategy, queue_intent_if_busy
from backend.models.chat import ChatIntent, StrategyDiff

logger = logging.getLogger(__name__)

# Expensive pipeline steps where a modify intent should be queued, not applied immediately.
_EXPENSIVE_STEPS = {"training", "tuning", "calibration", "shap", "similarity"}

# Static instructions - cached across all turns on the same model (§28).
# The dynamic context JSON is injected as the first user-turn content block so
# the static instructions always map to the same cache key.
_CHAT_SYSTEM_STATIC = """You are the clinical AI co-pilot embedded in a healthcare ML pipeline.
The user is a clinical ML engineer, clinical data analyst, or healthcare professional
working on predictive models for patient outcomes. Use precise clinical and ML terminology.
Never condescend or explain basic clinical concepts unless explicitly asked.

Domain context:
- Patient outcomes (readmission, mortality, disease risk) are the primary prediction targets.
- False negatives (missing a patient at risk) are almost always more costly than false positives.
- Calibrated probabilities matter - clinicians use them to stratify intervention intensity.
- Model fairness across demographic groups is a clinical equity requirement, not optional.
- All AI-assisted decisions must be reviewable and overrideable by the clinician.

You have full context of the current analysis run injected at the start of each conversation turn.
Answer questions from that context. When asked about a specific metric, quote it exactly.

When the user wants to modify a pipeline decision, confirm what will change and why.
Clinical override examples:
  - "Use XGBoost instead" → change model_selection.primary
  - "Set FN cost to 10x FP" → update threshold_config.cost_matrix
  - "Add age_group to protected attributes" → update fairness_config.protected_columns
  - "Exclude patient_name" → update preprocessing to drop that column (PHI)
If the intent is ambiguous, ask for clarification once - don't guess.

Pipeline interrupt policy:
If run.status == "running" and run.current_step is one of [training, tuning, calibration,
shap, similarity], a modify request cannot be applied immediately. Tell the user:
"I've queued that change - it will be applied as soon as [current_step] finishes
(roughly 1-3 minutes). Type 'abort' if you want to stop the run and apply it now."

Clinical equity queries:
When asked about fairness or equity ("Are predictions fair across groups?",
"Is there bias toward a demographic?"), summarise the fairness_report from context:
group-level sensitivity, specificity, and any flagged gaps > 5%.

Notebook export:
When the user asks to export as a notebook (e.g. "export as notebook", "give me a .ipynb",
"export to Jupyter"), respond: "Generating your reproducibility notebook - it will appear
in the deliverables section in a few seconds." The backend triggers the generation
automatically; do not describe how to build it manually.

Important: this platform assists clinical decision-making. Always include a brief note
that AI-generated risk scores must be reviewed by a licensed clinician before acting."""


async def _build_context_block(session: AsyncSession, run_id: str) -> dict[str, Any]:
    """Assemble a fresh structured context block per chat turn."""
    run = (
        await session.execute(select(Run).where(Run.id == run_id))
    ).scalar_one_or_none()
    if run is None:
        return {}

    project = (
        await session.execute(select(Project).where(Project.id == run.project_id))
    ).scalar_one_or_none()

    datasets = list(
        (
            await session.execute(
                select(Dataset).where(Dataset.project_id == run.project_id)
            )
        ).scalars()
    )

    recent_audit = list(
        reversed(
            list(
                (
                    await session.execute(
                        select(AuditEvent)
                        .where(AuditEvent.run_id == run_id)
                        .order_by(AuditEvent.seq.desc())
                        .limit(10)
                    )
                ).scalars()
            )
        )
    )

    brief = project.case_brief if project else None
    brief_context: dict[str, Any] | None = None
    if brief and brief.get("parsed"):
        brief_context = {
            "objectives": brief.get("objectives", []),
            "cost_matrix": brief.get("cost_matrix"),
            "known_data_issues": brief.get("known_data_issues", []),
            "deliverable_requirements": brief.get("deliverable_requirements", []),
            "evaluation_criteria": brief.get("evaluation_criteria", []),
            "stakeholder": {
                "name": brief.get("stakeholder_name"),
                "role": brief.get("stakeholder_role"),
            },
        }

    return {
        "run": {
            "id": run.id,
            "status": run.status,
            "current_step": run.current_step,
            "progress": run.progress,
        },
        "case_brief": brief_context,
        "datasets": [
            {
                "filename": d.filename,
                "role": d.role,
                "rows": d.row_count,
                "cols": d.col_count,
                "task_type": d.task_type,
            }
            for d in datasets
        ],
        "eda_report": run.eda_report,
        "preprocessing_strategy": run.preprocessing_strategy,
        "model_selection": run.model_selection,
        "best_model": run.best_model_name,
        "final_metrics": run.final_metrics,
        "threshold_result": run.threshold_result,
        "fairness_report": run.fairness_report,
        "recent_decisions": [
            {"category": e.category, "action": e.action, "reason": e.reason}
            for e in recent_audit
        ],
    }


def _summarize_context(context: dict[str, Any]) -> str:
    """Produce a short context summary for Haiku's intent classifier."""
    run = context.get("run", {})
    parts = [f"Run status: {run.get('status')}, step: {run.get('current_step')}"]
    if context.get("eda_report"):
        eda = context["eda_report"]
        parts.append(f"EDA complete. Model recommendation: {eda.get('model_recommendation')}")
    if context.get("preprocessing_strategy"):
        parts.append("Preprocessing strategy: set")
    if context.get("model_selection"):
        parts.append(f"Model selection: {context['model_selection'].get('primary')}")
    if context.get("best_model"):
        parts.append(f"Best model: {context['best_model']}")
    return ". ".join(parts)


def _format_messages(
    history: list[dict[str, Any]],
    user_message: str,
    context: dict[str, Any],
) -> list[dict[str, Any]]:
    """Build the messages list with context injected as a cached first-turn block.

    Injecting context here (not in the system prompt) means the static system
    instructions hit the prompt cache on every turn while only the context block
    (which changes per turn) is re-tokenised.
    """
    context_block = {
        "type": "text",
        "text": f"<pipeline_context>\n{json.dumps(context, indent=2)}\n</pipeline_context>",
        "cache_control": {"type": "ephemeral"},
    }

    messages: list[dict[str, Any]] = []

    if history:
        # Prepend context to the first history user turn so the model always sees it
        first = history[0]
        if first["role"] == "user":
            messages.append({
                "role": "user",
                "content": [
                    context_block,
                    {"type": "text", "text": first["content"]},
                ],
            })
            messages.extend(
                {"role": m["role"], "content": m["content"]} for m in history[1:]
            )
        else:
            # Edge-case: history starts with an assistant turn - inject context block first
            messages.append({"role": "user", "content": [context_block, {"type": "text", "text": ""}]})
            messages.extend({"role": m["role"], "content": m["content"]} for m in history)
    else:
        # No history - context block + the current message in one user turn
        messages.append({
            "role": "user",
            "content": [
                context_block,
                {"type": "text", "text": user_message},
            ],
        })
        return messages

    messages.append({"role": "user", "content": user_message})
    return messages


async def stream_chat_response(
    session: AsyncSession,
    run_id: str,
    user_id: str,
    user_message: str,
    history: list[dict[str, Any]],
    on_chunk: Callable[[str], Awaitable[None]],
) -> tuple[str, ChatIntent | None, list[StrategyDiff]]:
    """Drive one chat turn and return (response_text, intent, strategy_diffs).

    Two-pass parallel: Sonnet streams the response while Haiku classifies intent
    concurrently. This is the canonical latency-optimized pattern (§21).

    Interrupt semantics (§2, B6): if the pipeline is in an expensive step
    (training/tuning/calibration/shap/similarity), modify intents are queued
    rather than applied immediately.
    """
    context = await _build_context_block(session, run_id)
    context_summary = _summarize_context(context)
    messages = _format_messages(history, user_message, context)

    # ── Two passes in parallel ─────────────────────────────────────────────────
    response_text, intent = await asyncio.gather(
        call_claude_stream(
            messages=messages,
            model=settings.CLAUDE_SONNET_MODEL,
            system=_CHAT_SYSTEM_STATIC,
            max_tokens=2048,
            on_chunk=on_chunk,
        ),
        classify_intent(user_message, context_summary),
    )
    # ── End two-pass ──────────────────────────────────────────────────────────

    # Apply or queue strategy mutation
    diffs: list[StrategyDiff] = []
    if intent and intent.intent == "modify" and not intent.needs_confirmation:
        run = (
            await session.execute(select(Run).where(Run.id == run_id))
        ).scalar_one_or_none()
        if run:
            queued = await queue_intent_if_busy(session, run, intent)
            if not queued:
                diffs = await apply_intent_to_strategy(session, run, intent)

    # Persist assistant message
    session.add(
        DBChatMessage(
            run_id=run_id,
            user_id=user_id,
            role="assistant",
            content=response_text,
            intent=intent.model_dump() if intent else None,
            strategy_diff=[d.model_dump() for d in diffs] if diffs else None,
            model=settings.CLAUDE_SONNET_MODEL,
        )
    )
    await session.commit()

    return response_text, intent, diffs
