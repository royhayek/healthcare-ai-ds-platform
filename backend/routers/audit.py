"""Audit log viewer endpoints (§24, §26).

GET /runs/{run_id}/audit          - paginated audit event list
GET /runs/{run_id}/audit/verify   - verify chain integrity
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.core import audit
from backend.core.auth import get_current_user
from backend.core.database import AuditEvent, Run, get_db

router = APIRouter(prefix="/runs", tags=["audit"])


class AuditEventResponse(BaseModel):
    id: str
    seq: int
    timestamp: str
    actor: str
    category: str
    action: str
    payload: dict
    reason: str | None
    prev_hash: str
    self_hash: str

    model_config = {"from_attributes": True}


class AuditVerifyResponse(BaseModel):
    run_id: str
    chain_valid: bool
    total_events: int
    error: str | None


@router.get("/{run_id}/audit", response_model=list[AuditEventResponse])
async def get_audit_log(
    run_id: str,
    limit: int = Query(default=100, le=500),
    offset: int = Query(default=0, ge=0),
    db: AsyncSession = Depends(get_db),
    user_id: str = Depends(get_current_user),
) -> list[AuditEventResponse]:
    """Return hash-chained audit events for a run, ordered by sequence number."""
    run = await db.get(Run, run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")

    stmt = (
        select(AuditEvent)
        .where(AuditEvent.run_id == run_id)
        .order_by(AuditEvent.seq)
        .limit(limit)
        .offset(offset)
    )
    events = (await db.execute(stmt)).scalars().all()

    return [
        AuditEventResponse(
            id=e.id,
            seq=e.seq,
            timestamp=e.timestamp.isoformat(),
            actor=e.actor,
            category=e.category,
            action=e.action,
            payload=e.payload,
            reason=e.reason,
            prev_hash=e.prev_hash,
            self_hash=e.self_hash,
        )
        for e in events
    ]


@router.get("/{run_id}/audit/verify", response_model=AuditVerifyResponse)
async def verify_audit_chain(
    run_id: str,
    db: AsyncSession = Depends(get_db),
    user_id: str = Depends(get_current_user),
) -> AuditVerifyResponse:
    """Verify the cryptographic integrity of the run's audit chain."""
    run = await db.get(Run, run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")

    count_result = await db.execute(
        select(func.count()).where(AuditEvent.run_id == run_id)
    )
    total = count_result.scalar_one()

    chain_valid = False
    error: str | None = None
    try:
        chain_valid = await audit.verify_chain(db, run_id)
    except Exception as exc:
        error = str(exc)

    return AuditVerifyResponse(
        run_id=run_id,
        chain_valid=chain_valid,
        total_events=total,
        error=error,
    )
