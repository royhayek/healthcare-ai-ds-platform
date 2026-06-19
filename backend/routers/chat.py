"""Chat co-pilot SSE endpoint (§21).

POST /runs/{run_id}/chat → text/event-stream

SSE event protocol (one JSON object per data: line):
  {"type": "text_chunk", "content": "..."}     - incremental response text
  {"type": "strategy_diff", "diffs": [...]}    - before/after strategy change
  {"type": "intent", "intent": {...}}          - classified intent
  {"type": "done"}                             - stream complete
  {"type": "error", "error": "..."}            - pipeline error

Implementation uses asyncio.Queue to decouple the SSE generator from the
background processing coroutine. The generator drains the queue and yields
SSE frames. On client disconnect, the generator cancels the background task.
"""

import asyncio
import json
import logging
from collections.abc import AsyncGenerator

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.agents.chat_agent import stream_chat_response
from backend.core.auth import get_current_user
from backend.core.database import ChatMessage as DBChatMessage
from backend.core.database import Project, Run, get_db
from backend.models.chat import ChatMessageCreate

router = APIRouter(tags=["chat"])
logger = logging.getLogger(__name__)

_HISTORY_LIMIT = 20


async def _verify_run_access(run_id: str, user_id: str, db: AsyncSession) -> Run:
    run = (await db.execute(select(Run).where(Run.id == run_id))).scalar_one_or_none()
    if run is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Run not found")
    project = (
        await db.execute(
            select(Project).where(Project.id == run.project_id, Project.user_id == user_id)
        )
    ).scalar_one_or_none()
    if project is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Run not found")
    return run


async def _load_history(db: AsyncSession, run_id: str) -> list[dict]:
    result = await db.execute(
        select(DBChatMessage)
        .where(DBChatMessage.run_id == run_id, DBChatMessage.role.in_(["user", "assistant"]))
        .order_by(DBChatMessage.created_at.desc())
        .limit(_HISTORY_LIMIT)
    )
    msgs = list(reversed(result.scalars().all()))
    return [{"role": m.role, "content": m.content} for m in msgs]


def _sse(event: dict) -> str:
    return f"data: {json.dumps(event)}\n\n"


async def _maybe_trigger_override_rerun(
    db: AsyncSession, run_id: str, category: str
) -> str | None:
    """Re-run the step a just-applied override invalidated, so it takes effect.

    A chat override edits a decision field but does not recompute anything; the
    human's intent is only realised once the consuming step re-runs. Fires only
    when the run is paused at a checkpoint and the override invalidates state
    already computed at/before it. Flips the run to `running`, re-enqueues the
    analysis task at the recompute-target step (which re-pauses at the same
    checkpoint for review), and returns the step name - or None if no re-run.
    """
    from backend.core.strategy_mutator import canonical_category
    from backend.tasks.analysis_task import rerun_step_for_override, run_analysis_task

    run = (await db.execute(select(Run).where(Run.id == run_id))).scalar_one_or_none()
    if run is None or run.status != "awaiting_checkpoint":
        return None

    category = canonical_category(category)
    target_step = rerun_step_for_override(category, run.current_step)
    if target_step is None:
        return None

    paused_at = run.current_step
    run.current_step = target_step
    run.status = "running"
    db.add(run)

    from backend.core import audit
    await audit.append(
        db,
        run_id=run_id,
        actor="system",
        category=category,
        action="override_rerun_triggered",
        payload={"override_category": category, "rerun_step": target_step, "paused_at": paused_at},
        reason=f"Chat override to {category} re-runs {target_step} to take effect",
    )
    await db.commit()

    job = run_analysis_task.delay(run_id)
    run.job_id = job.id
    await db.commit()
    return target_step


@router.post("/runs/{run_id}/chat")
async def chat(
    run_id: str,
    payload: ChatMessageCreate,
    user_id: str = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> StreamingResponse:
    await _verify_run_access(run_id, user_id, db)

    # Load history before persisting the current message so it isn't
    # included twice - once in history and once appended by _format_messages.
    history = await _load_history(db, run_id)

    db.add(
        DBChatMessage(
            run_id=run_id,
            user_id=user_id,
            role="user",
            content=payload.content,
        )
    )
    await db.commit()

    async def event_stream() -> AsyncGenerator[str, None]:
        queue: asyncio.Queue[str | None] = asyncio.Queue()

        async def on_chunk(text: str) -> None:
            await queue.put(_sse({"type": "text_chunk", "content": text}))

        async def process() -> None:
            try:
                _, intent, diffs = await stream_chat_response(
                    session=db,
                    run_id=run_id,
                    user_id=user_id,
                    user_message=payload.content,
                    history=history,
                    on_chunk=on_chunk,
                )
                if diffs:
                    await queue.put(
                        _sse({"type": "strategy_diff", "diffs": [d.model_dump() for d in diffs]})
                    )
                if intent:
                    await queue.put(_sse({"type": "intent", "intent": intent.model_dump()}))

                # A successful override must actually take effect: re-run the step
                # that consumes the changed decision so the human's intent is
                # realised before the pipeline advances to the next checkpoint.
                if diffs and intent:
                    rerun_step = await _maybe_trigger_override_rerun(db, run_id, intent.category)
                    if rerun_step:
                        await queue.put(
                            _sse({"type": "rerun_triggered", "step": rerun_step})
                        )

                # Notebook export: enqueue Celery task and emit artifact_task event
                if (
                    intent
                    and intent.intent == "request_artifact"
                    and (intent.structured_payload or {}).get("artifact_type") == "notebook"
                ):
                    try:
                        from backend.tasks.deliverable_task import generate_notebook_export_task
                        task = generate_notebook_export_task.delay(run_id)
                        await queue.put(
                            _sse({"type": "artifact_task", "task_id": task.id, "artifact_type": "notebook"})
                        )
                    except Exception as nb_exc:
                        logger.warning("notebook export enqueue failed: %s", nb_exc)

                # Plot request: trigger render task for the requested stage
                if intent and intent.intent == "request_plot":
                    stage = (intent.structured_payload or {}).get("stage") or "training"
                    try:
                        from backend.tasks.plot_task import render_plots_task
                        plot_task = render_plots_task.delay(run_id, stage)
                        await queue.put(
                            _sse({"type": "artifact_task", "task_id": plot_task.id, "artifact_type": f"plots:{stage}"})
                        )
                    except Exception as plot_exc:
                        logger.warning("plot render enqueue failed for stage %s: %s", stage, plot_exc)

                await queue.put(_sse({"type": "done"}))
            except Exception as exc:
                logger.exception("chat stream error for run %s", run_id)
                await queue.put(_sse({"type": "error", "error": str(exc)}))
            finally:
                await queue.put(None)  # sentinel - always unblocks the generator

        task = asyncio.create_task(process())
        try:
            while True:
                item = await queue.get()
                if item is None:
                    break
                yield item
        finally:
            # Client disconnect: cancel background task cleanly
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # disable nginx buffering
        },
    )


@router.get("/runs/{run_id}/chat/history")
async def get_chat_history(
    run_id: str,
    user_id: str = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> list[dict]:
    await _verify_run_access(run_id, user_id, db)
    result = await db.execute(
        select(DBChatMessage)
        .where(DBChatMessage.run_id == run_id)
        .order_by(DBChatMessage.created_at.asc())
    )
    return [
        {
            "role": m.role,
            "content": m.content,
            "intent": m.intent,
            "strategy_diff": m.strategy_diff,
            "created_at": m.created_at.isoformat(),
        }
        for m in result.scalars()
    ]
