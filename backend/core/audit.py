"""Hash-chained, append-only audit log (§21, §24).

Chain convention:
- The first event in any run has prev_hash = GENESIS_SENTINEL, the literal
  string "GENESIS" - not a hash of that string. append() and verify_chain()
  both check against this literal, so there is no ambiguity: if you see
  "GENESIS" in prev_hash, the event is first in its chain.
- Every subsequent event's prev_hash equals the previous event's self_hash.
- self_hash = SHA-256 of the canonical string:
      {prev_hash}:{run_id}:{seq}:{actor}:{category}:{action}:{payload_json}:{reason}
  where payload_json is compact JSON with sorted keys, and reason is the
  empty string when None.

Delivery guarantee:
- append() calls session.flush() but does NOT commit. The caller controls the
  transaction boundary. This means callers can append several events atomically
  or roll back the entire transaction on error.
- verify_chain() is a read-only operation. Call it before every deliverable
  export (§23) - a broken chain must refuse bundling.

PHI redaction:
- redact_phi_from_payload() scans all string values in the payload and replaces
  values matching SSN, MRN, email, or DOB patterns with "[REDACTED]".
- append() calls this automatically before computing the hash, so the chain
  hash is always computed over the redacted payload - never over raw PHI.
- The redaction is irreversible by design: once written to the chain, the
  redacted form is canonical.
"""

import hashlib
import json
import re
from datetime import datetime, timezone
from typing import Any

# PHI patterns - same regexes as profiler.py but applied to values, not column names
_SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
_MRN_RE = re.compile(r"\bMRN\d{6,10}\b", re.IGNORECASE)
_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b")
_DOB_RE = re.compile(r"\b(19|20)\d{2}-(0[1-9]|1[0-2])-(0[1-9]|[12]\d|3[01])\b")

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.core.database import AuditEvent

GENESIS_SENTINEL = "GENESIS"

_PHI_PATTERNS = [_SSN_RE, _MRN_RE, _EMAIL_RE, _DOB_RE]


def redact_phi_from_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Return a deep copy of payload with PHI-matching string values replaced.

    Walks the full nested structure (dicts and lists). Numeric values, booleans,
    and None are passed through unchanged. String values matching any PHI regex
    have the matched fragment replaced with "[REDACTED]".

    This is called by append() before hashing so the chain is always computed
    over the redacted form.
    """
    def _redact_value(val: Any) -> Any:
        if isinstance(val, str):
            for pattern in _PHI_PATTERNS:
                val = pattern.sub("[REDACTED]", val)
            return val
        if isinstance(val, dict):
            return {k: _redact_value(v) for k, v in val.items()}
        if isinstance(val, list):
            return [_redact_value(item) for item in val]
        return val

    return {k: _redact_value(v) for k, v in payload.items()}


def compute_self_hash(
    prev_hash: str,
    run_id: str,
    seq: int,
    actor: str,
    category: str,
    action: str,
    payload: dict[str, Any],
    reason: str | None,
) -> str:
    payload_json = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    canonical = f"{prev_hash}:{run_id}:{seq}:{actor}:{category}:{action}:{payload_json}:{reason or ''}"
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


async def append(
    session: AsyncSession,
    run_id: str,
    actor: str,
    category: str,
    action: str,
    payload: dict[str, Any],
    reason: str | None = None,
) -> AuditEvent:
    """Append one event to the hash chain for run_id.

    actor: "ai" | "user" | "system"
    category: pipeline subsystem - "eda", "threshold", "chat", "fairness", …
    action: specific event within category - "eda_complete", "override", …
    payload: statistical aggregates or metadata - never raw row-level data
    reason: required for AI decisions; optional for system events

    Returns the persisted AuditEvent (flushed, not yet committed).

    PHI safety: payload is automatically scanned and any PHI-matching strings
    are replaced with "[REDACTED]" before the hash is computed and the event
    is persisted. This is irreversible by design.
    """
    payload = redact_phi_from_payload(payload)

    result = await session.execute(
        select(AuditEvent)
        .where(AuditEvent.run_id == run_id)
        .order_by(AuditEvent.seq.desc())
        .limit(1)
    )
    last = result.scalar_one_or_none()

    if last is None:
        seq = 0
        prev_hash = GENESIS_SENTINEL
    else:
        seq = last.seq + 1
        prev_hash = last.self_hash

    self_hash = compute_self_hash(prev_hash, run_id, seq, actor, category, action, payload, reason)

    event = AuditEvent(
        run_id=run_id,
        seq=seq,
        timestamp=datetime.now(timezone.utc),
        actor=actor,
        category=category,
        action=action,
        payload=payload,
        reason=reason,
        prev_hash=prev_hash,
        self_hash=self_hash,
    )
    session.add(event)
    await session.flush()
    return event


def _verify_event_chain(events: list[AuditEvent]) -> bool:
    """Pure function: verify the hash chain of a pre-fetched, ordered event list.

    Separated from the DB layer so unit tests can exercise it with synthetic
    events without requiring a database session.

    Returns True for an empty list - a run that hasn't started has a trivially
    valid (empty) chain.
    """
    if not events:
        return True

    if events[0].prev_hash != GENESIS_SENTINEL:
        return False

    for i, event in enumerate(events):
        expected = compute_self_hash(
            event.prev_hash,
            event.run_id,
            event.seq,
            event.actor,
            event.category,
            event.action,
            event.payload,
            event.reason,
        )
        if event.self_hash != expected:
            return False

        if i > 0 and event.prev_hash != events[i - 1].self_hash:
            return False

    return True


async def verify_chain(session: AsyncSession, run_id: str) -> bool:
    """Verify the complete audit chain for run_id.

    Must return True before any deliverable export. A False result means
    the chain was tampered with or persisted incorrectly - refuse bundling.
    """
    result = await session.execute(
        select(AuditEvent)
        .where(AuditEvent.run_id == run_id)
        .order_by(AuditEvent.seq.asc())
    )
    events = list(result.scalars().all())
    return _verify_event_chain(events)
