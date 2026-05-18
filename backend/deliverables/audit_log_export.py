"""Audit log export - CSV + JSON (§4.6).

Exports the full hash-chained audit log for the run.
Chain verification is the caller's responsibility (deliverable_task.py does it
before invoking any generator).
"""

from __future__ import annotations

import csv
import io
import json
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from backend.deliverables.base import GENERATOR_VERSION, GeneratedDeliverable

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession
    from backend.core.database import Run


async def generate_audit_log_export(
    run: "Run",
    session: "AsyncSession",
    ctx: dict[str, Any],
) -> list[GeneratedDeliverable]:
    """Return two deliverables: CSV and JSON of the full audit chain."""
    from sqlalchemy import select, asc
    from backend.core.database import AuditEvent

    result = await session.execute(
        select(AuditEvent)
        .where(AuditEvent.run_id == run.id)
        .order_by(asc(AuditEvent.seq))
    )
    events = result.scalars().all()

    rows = [
        {
            "seq": e.seq,
            "timestamp": e.timestamp.isoformat(),
            "actor": e.actor,
            "category": e.category,
            "action": e.action,
            "reason": e.reason or "",
            "payload": json.dumps(e.payload),
            "prev_hash": e.prev_hash[:16] + "…" if len(e.prev_hash) > 16 else e.prev_hash,
            "self_hash": e.self_hash[:16] + "…" if len(e.self_hash) > 16 else e.self_hash,
        }
        for e in events
    ]

    # CSV
    csv_buf = io.StringIO()
    if rows:
        writer = csv.DictWriter(csv_buf, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    csv_bytes = csv_buf.getvalue().encode()

    # Count clinical-specific event categories for the summary
    phi_events = sum(1 for e in events if e.category == "phi" or "phi" in (e.action or ""))
    override_events = sum(1 for e in events if e.action in ("clinician_override", "equity_acknowledged"))
    threshold_events = sum(1 for e in events if e.category == "threshold")

    # JSON (full hashes, machine-readable)
    json_payload = {
        "run_id": run.id,
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "clinical_domain": "healthcare",
        "chain_length": len(events),
        "clinical_event_summary": {
            "phi_exclusion_events": phi_events,
            "clinician_override_events": override_events,
            "threshold_decision_events": threshold_events,
        },
        "events": [
            {
                "seq": e.seq,
                "timestamp": e.timestamp.isoformat(),
                "actor": e.actor,
                "category": e.category,
                "action": e.action,
                "reason": e.reason,
                "payload": e.payload,
                "prev_hash": e.prev_hash,
                "self_hash": e.self_hash,
            }
            for e in events
        ],
    }
    json_bytes = json.dumps(json_payload, indent=2, ensure_ascii=False).encode()

    now = datetime.now(timezone.utc).isoformat()
    return [
        GeneratedDeliverable.build(
            name="audit_log",
            fmt="csv",
            content=csv_bytes,
            run_id=run.id,
            inputs_used=["audit_events"],
            audience="clinical auditor, compliance officer, patient safety officer",
            extra_path="chain",
        ),
        GeneratedDeliverable.build(
            name="audit_log",
            fmt="json",
            content=json_bytes,
            run_id=run.id,
            inputs_used=["audit_events"],
            audience="clinical auditor, compliance officer, patient safety officer",
            extra_path="chain_full",
        ),
    ]
