"""Real-time progress emitter (§22).

Celery tasks and async pipeline code publish ProgressEvent objects to a
per-run Redis pub/sub channel. The SSE endpoint subscribes to that channel
and forwards chunks to the browser.

Hard rule (the spec §9): any operation taking > 1 second must emit at least
one progress event. Never leave the user with an unexplained spinner.

Usage - sync (Celery tasks):
    emitter = ProgressEmitter(run_id)
    emitter.emit("eda", "Profiling columns…", 15)

Usage - async (FastAPI, async pipeline steps):
    emitter = ProgressEmitter(run_id)
    await emitter.emit_async("model_selection", "Ranking candidates…", 55)

Module-level shortcuts when you only emit once from a call site:
    emit_progress(run_id, "threshold", "Sweeping thresholds…", 72)
    await emit_progress_async(run_id, "shap", "Computing SHAP values…", 85)
"""

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field

from backend.core.config import settings

_CHANNEL_PREFIX = "progress"


class ProgressEvent(BaseModel):
    run_id: str
    step: str
    message: str
    pct: int  # 0-100
    detail: dict[str, Any] | None = None
    ts: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ProgressEmitter:
    def __init__(self, run_id: str) -> None:
        self.run_id = run_id
        self.channel = f"{_CHANNEL_PREFIX}:{run_id}"

    def emit(self, step: str, message: str, pct: int, detail: dict[str, Any] | None = None) -> None:
        """Sync publish - for use in Celery tasks."""
        import redis as redis_sync

        event = ProgressEvent(run_id=self.run_id, step=step, message=message, pct=pct, detail=detail)
        r = redis_sync.from_url(settings.REDIS_URL)
        try:
            r.publish(self.channel, event.model_dump_json())
        finally:
            r.close()

    async def emit_async(
        self, step: str, message: str, pct: int, detail: dict[str, Any] | None = None
    ) -> None:
        """Async publish - for use in FastAPI endpoints and async code."""
        import redis.asyncio as aioredis

        event = ProgressEvent(run_id=self.run_id, step=step, message=message, pct=pct, detail=detail)
        r = aioredis.from_url(settings.REDIS_URL)
        try:
            await r.publish(self.channel, event.model_dump_json())
        finally:
            await r.aclose()


def emit_progress(
    run_id: str, step: str, message: str, pct: int, detail: dict[str, Any] | None = None
) -> None:
    ProgressEmitter(run_id).emit(step, message, pct, detail)


async def emit_progress_async(
    run_id: str, step: str, message: str, pct: int, detail: dict[str, Any] | None = None
) -> None:
    await ProgressEmitter(run_id).emit_async(step, message, pct, detail)
