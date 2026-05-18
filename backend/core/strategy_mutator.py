"""Strategy mutation dispatch for the chat co-pilot (§21).

This is the seam every pipeline stage plugs into. When a user override arrives
via chat, apply_intent_to_strategy:
  1. Dispatches to the registered category mutator.
  2. The mutator modifies the Run object IN MEMORY only - no DB writes.
  3. apply_intent_to_strategy commits strategy changes + audit entry atomically.

Adding a new category (e.g., "threshold", "fairness") means registering one
function with @_register("category") - nothing else changes.

Mutator signature:
    async def _(session, run, payload) -> list[StrategyDiff]

The mutator may return [] to signal "no applicable change" (e.g., strategy not
initialized yet, or no-op because the value is already set). This is not an
error - the caller just produces no diff card.
"""

import logging
from collections.abc import Callable, Coroutine
from copy import deepcopy
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from backend.core import audit
from backend.core.database import Run
from backend.models.chat import ChatIntent, PendingIntent, StrategyDiff

logger = logging.getLogger(__name__)

MutatorFn = Callable[
    [AsyncSession, Run, dict[str, Any]],
    Coroutine[Any, Any, list[StrategyDiff]],
]

_MUTATORS: dict[str, MutatorFn] = {}


def _register(category: str) -> Callable[[MutatorFn], MutatorFn]:
    def decorator(fn: MutatorFn) -> MutatorFn:
        _MUTATORS[category] = fn
        return fn
    return decorator


# ── Category mutators ──────────────────────────────────────────────────────────
# Each mutator: modifies run fields in memory, returns list[StrategyDiff].
# Do NOT call session.begin() here - apply_intent_to_strategy commits everything.


@_register("preprocessing")
async def _preprocessing_mutator(
    session: AsyncSession, run: Run, payload: dict[str, Any]
) -> list[StrategyDiff]:
    if not run.preprocessing_strategy:
        return []

    col = payload.get("column")
    field = payload.get("field")
    value = payload.get("value")
    if not (col and field and value is not None):
        return []

    strategy = deepcopy(run.preprocessing_strategy)
    cols = strategy.get("columns", {})
    if col not in cols:
        return []

    before = cols[col].get(field)
    if before == value:
        return []  # no-op

    cols[col][field] = value
    strategy["columns"] = cols
    run.preprocessing_strategy = strategy

    return [
        StrategyDiff(
            field_path=f"preprocessing.columns.{col}.{field}",
            before=before,
            after=value,
            summary=f"Changed {col} {field}: {before!r} → {value!r}",
            run_id=run.id,
        )
    ]


@_register("model_selection")
async def _model_selection_mutator(
    session: AsyncSession, run: Run, payload: dict[str, Any]
) -> list[StrategyDiff]:
    if not run.model_selection:
        return []

    model = payload.get("model")
    if not model:
        return []

    selection = deepcopy(run.model_selection)
    before = selection.get("primary")
    if before == model:
        return []

    selection["primary"] = model
    run.model_selection = selection

    return [
        StrategyDiff(
            field_path="model_selection.primary",
            before=before,
            after=model,
            summary=f"Changed primary model: {before!r} → {model!r}",
            run_id=run.id,
        )
    ]


@_register("threshold")
async def _threshold_mutator(
    session: AsyncSession, run: Run, payload: dict[str, Any]
) -> list[StrategyDiff]:
    if not run.threshold_config:
        return []

    threshold = payload.get("threshold")
    if threshold is None:
        return []

    config = deepcopy(run.threshold_config)
    before = config.get("override_threshold")
    config["override_threshold"] = float(threshold)
    run.threshold_config = config

    return [
        StrategyDiff(
            field_path="threshold_config.override_threshold",
            before=before,
            after=float(threshold),
            summary=f"Overrode classification threshold: {before} → {threshold}",
            run_id=run.id,
        )
    ]


# ── Interrupt semantics (§2, B6) ──────────────────────────────────────────────

_EXPENSIVE_STEPS = frozenset({"training", "tuning", "calibration", "shap", "similarity"})


async def queue_intent_if_busy(
    session: AsyncSession,
    run: Run,
    intent: ChatIntent,
) -> bool:
    """Queue a modify intent if the pipeline is in an expensive step.

    Returns True when the intent was queued (pipeline is busy), False when the
    caller should apply it immediately via apply_intent_to_strategy.
    """
    if run.status != "running" or run.current_step not in _EXPENSIVE_STEPS:
        return False

    pending = PendingIntent(
        intent=intent,
        step_at_queue_time=run.current_step or "unknown",
    )
    current_queue: list = list(run.pending_intents or [])
    current_queue.append(pending.to_dict())
    run.pending_intents = current_queue
    session.add(run)

    await audit.append(
        session,
        run_id=run.id,
        actor="user",
        category=intent.category,
        action="intent_queued",
        payload={
            "intent_category": intent.category,
            "step_at_queue_time": run.current_step,
            "queue_depth": len(current_queue),
        },
        reason=f"Intent queued while {run.current_step} was running",
    )
    await session.commit()
    return True


async def flush_pending_intents(
    session: AsyncSession,
    run: Run,
) -> list[StrategyDiff]:
    """Apply all queued intents in order. Call after an expensive step completes.

    Called by the analysis task at each checkpoint so changes requested during
    heavy computation are visible before the user reviews the checkpoint card.
    """
    raw_queue: list = list(run.pending_intents or [])
    if not raw_queue:
        return []

    all_diffs: list[StrategyDiff] = []
    for raw in raw_queue:
        try:
            pending = PendingIntent.from_dict(raw)
            diffs = await apply_intent_to_strategy(session, run, pending.intent)
            all_diffs.extend(diffs)
        except Exception as exc:
            logger.warning("Failed to apply queued intent: %s - %s", raw, exc)

    run.pending_intents = []
    session.add(run)
    await session.commit()

    if all_diffs:
        logger.info(
            "Flushed %d pending intent(s) for run %s → %d diffs",
            len(raw_queue), run.id, len(all_diffs),
        )
    return all_diffs


# ── Dispatch ───────────────────────────────────────────────────────────────────


async def apply_intent_to_strategy(
    session: AsyncSession,
    run: Run,
    intent: ChatIntent,
) -> list[StrategyDiff]:
    """Dispatch intent to the appropriate mutator and commit atomically.

    Returns the list of StrategyDiff objects (may be empty if no change applies).
    """
    mutator = _MUTATORS.get(intent.category)
    if mutator is None:
        return []

    diffs = await mutator(session, run, intent.structured_payload)

    if diffs:
        session.add(run)
        await audit.append(
            session,
            run_id=run.id,
            actor="user",
            category=intent.category,
            action="strategy_override",
            payload={
                "diffs": [d.model_dump() for d in diffs],
                "intent_category": intent.category,
                "confidence": intent.confidence,
            },
            reason=intent.reasoning,
        )
        await session.commit()

    return diffs
