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
import re
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

# Short replies that confirm / reject a previously-gated modify intent. A modify
# with needs_confirmation=True is NOT applied on the turn it arrives - the model
# asks the user to confirm. The bare confirmation ("yes") carries no structured
# payload of its own, so we resolve it against the pending intent stored on the
# last assistant turn. Without this, the model narrates "override applied" while
# the backend applies nothing.
_AFFIRMATIONS = frozenset({
    "yes", "y", "yeah", "yep", "yup", "ya", "sure", "ok", "okay", "confirm",
    "confirmed", "apply", "proceed", "go", "do it", "go ahead", "yes please",
    "please do", "approved", "accept", "agreed", "affirmative", "yes confirm",
})
_NEGATIONS = frozenset({
    "no", "n", "nope", "nah", "cancel", "stop", "abort", "never mind",
    "nevermind", "dont", "do not", "keep it", "leave it", "no thanks",
})

# Shown when the model returns an empty completion (zero text blocks). Without
# this the SSE stream emits no text_chunk and no error, and the panel finalizes
# a silent empty assistant bubble (the observed failure). The fallback is also
# streamed via on_chunk so the user always sees a visible turn to retry against.
_EMPTY_RESPONSE_FALLBACK = (
    "I wasn't able to generate a response to that just now - please re-send your "
    "message. If you were asking me to change a pipeline decision, it has NOT been "
    "applied; re-state it and I'll confirm before making the change. "
    "(AI-generated outputs must be reviewed by a licensed clinician before acting.)"
)


_CLINICIAN_NOTE = (
    "AI-generated risk scores must be reviewed by a licensed clinician before acting."
)


def _deterministic_modify_message(intent: ChatIntent) -> str:
    """Describe a modify override without the chat model.

    Used when the chat model returns an empty completion (e.g. a refusal on this
    clinical domain) but the intent classifier produced an actionable override.
    The clinician's ability to steer the pipeline must not depend on the chat
    model agreeing to converse, so this message is built entirely from the
    classified intent. When confirmation is required it tells the user exactly
    how to confirm, so the gated-modify flow still completes on the next turn.
    """
    what = (intent.reasoning or "").strip() or f"a change to the {intent.category} step"
    if intent.needs_confirmation:
        return (
            f"I've read that as an override of the **{intent.category}** decision: {what}. "
            "Reply **confirm** to apply it (or **cancel** to discard). "
            f"{_CLINICIAN_NOTE}"
        )
    return (
        f"Applying your override to the **{intent.category}** decision: {what}. "
        f"{_CLINICIAN_NOTE}"
    )


def _classify_confirmation(message: str) -> str | None:
    """Return 'confirm', 'reject', or None for an ambiguous / substantive message.

    Only short replies (<= 4 words) are treated as pure confirmations - anything
    longer (e.g. "yes but use the threshold instead") is a substantive turn and
    falls through to normal intent classification so it isn't misread as consent.
    """
    norm = re.sub(r"[^a-z\s]", "", message.lower())
    norm = re.sub(r"\s+", " ", norm).strip()
    if not norm or len(norm.split()) > 4:
        return None
    if norm in _AFFIRMATIONS:
        return "confirm"
    if norm in _NEGATIONS:
        return "reject"
    first = norm.split()[0]
    if first in _AFFIRMATIONS:
        return "confirm"
    if first in _NEGATIONS:
        return "reject"
    return None


async def _pending_confirmation_intent(
    session: AsyncSession, run_id: str
) -> ChatIntent | None:
    """Return the modify intent the last assistant turn asked the user to confirm.

    The pending intent is recovered from the most recent assistant ChatMessage's
    stored `intent` (persisted at the end of every turn). Returns None when the
    last assistant turn was not a confirmation-gated modify.
    """
    row = (
        await session.execute(
            select(DBChatMessage)
            .where(DBChatMessage.run_id == run_id, DBChatMessage.role == "assistant")
            .order_by(DBChatMessage.created_at.desc())
            .limit(1)
        )
    ).scalar_one_or_none()
    if row is None or not row.intent:
        return None
    try:
        intent = ChatIntent.model_validate(row.intent)
    except Exception:
        return None
    if intent.intent == "modify" and intent.needs_confirmation:
        return intent
    return None

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
When a user CONFIRMS an override at a checkpoint, the affected step is re-run
automatically so the change actually takes effect, then the pipeline re-pauses at
the SAME checkpoint for review - it does NOT auto-advance to the next checkpoint.
A model-selection override is AUTHORITATIVE: "use logistic_regression instead of
lightgbm" FORCES logistic_regression as the primary model. It is not a request to
re-compare and possibly switch back - the chosen model becomes primary regardless
of which candidate scores highest. The other candidates may still appear in the
leaderboard for reference, but the user's pick is the primary. Do NOT tell the
user you will "confirm logistic regression or switch back based on the results."
Tell the user what is being recomputed (e.g. "Re-running training with
logistic_regression forced as the primary model; the others remain as reference
candidates - review, then resume"). Do not claim a change is "committed" or that
"the run will proceed" as if downstream steps already ran; they have not until the
re-run completes and the user resumes.
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
        "target_strategy": run.target_strategy,
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

    # Drop any history turn with empty/whitespace-only content: the Anthropic API
    # rejects empty text blocks ("text content blocks must be non-empty"), and a
    # blank turn (e.g. a previously failed assistant response) carries no signal.
    clean_history = [
        m for m in history if isinstance(m.get("content"), str) and m["content"].strip()
    ]

    messages: list[dict[str, Any]] = []

    if clean_history and clean_history[0]["role"] == "user":
        # Prepend context to the first history user turn so the model always sees it.
        first = clean_history[0]
        messages.append({
            "role": "user",
            "content": [
                context_block,
                {"type": "text", "text": first["content"]},
            ],
        })
        messages.extend(
            {"role": m["role"], "content": m["content"]} for m in clean_history[1:]
        )
    elif clean_history:
        # History starts with an assistant turn - the context block alone is the
        # first user turn. Do NOT append an empty text block here; a single
        # non-empty block is a valid user message.
        messages.append({"role": "user", "content": [context_block]})
        messages.extend(
            {"role": m["role"], "content": m["content"]} for m in clean_history
        )
    else:
        # No usable history - context block + the current message in one user turn.
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

    # If this turn confirms / rejects a previously-gated modify, recover the
    # pending intent now (before the parallel passes) so a bare "yes" actually
    # applies the stored change rather than being re-classified from scratch.
    confirmation = _classify_confirmation(user_message)
    pending = (
        await _pending_confirmation_intent(session, run_id)
        if confirmation is not None
        else None
    )

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

    # Resolve a pending confirmation: the stored modify intent is authoritative
    # over Haiku's read of the bare reply ("yes"/"no" carry no payload).
    if pending is not None:
        if confirmation == "confirm":
            intent = pending.model_copy(update={"needs_confirmation": False})
        elif confirmation == "reject":
            intent = None  # user declined - apply nothing

    # Empty model completion handling. On this clinical domain the chat model
    # sometimes returns no text at all (stop_reason="refusal") even for a
    # legitimate pipeline override - e.g. "a missed high-pathogenicity strain is
    # worse than a false alarm, set FN cost 10x". A chat-model refusal must NOT
    # strip the clinician's ability to steer the pipeline. So when Haiku still
    # classified an actionable override, we keep it and synthesise a deterministic
    # message describing the change (built from the intent, never from the chat
    # model). Only a genuinely non-actionable empty turn falls back to the generic
    # retry message and suppresses mutation.
    if not response_text.strip():
        if intent and intent.intent == "modify":
            logger.warning(
                "chat: empty model completion for run %s - synthesising deterministic "
                "message for modify intent (override preserved despite refusal)", run_id,
            )
            response_text = _deterministic_modify_message(intent)
        else:
            logger.warning(
                "chat: empty model completion for run %s - emitting fallback, suppressing mutation",
                run_id,
            )
            response_text = _EMPTY_RESPONSE_FALLBACK
            intent = None
        await on_chunk(response_text)

    # Carry the user's verbatim message on the intent so producer-regenerated
    # overrides (e.g. preprocessing) can replay the exact instruction to the agent
    # when the step re-runs. Set once, when the modify is first seen, so a later
    # bare "yes" confirmation still resolves the original instruction.
    if intent and intent.intent == "modify":
        payload = dict(intent.structured_payload or {})
        if not payload.get("instruction"):
            payload["instruction"] = user_message
            intent = intent.model_copy(update={"structured_payload": payload})

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
