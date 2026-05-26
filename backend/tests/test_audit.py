"""Tests for backend/core/audit.py - hash-chained audit log.

Unit tests (no Postgres required):
  Run with: pytest backend/tests/test_audit.py -k "not integration"

Integration tests (require Postgres via docker-compose):
  Run with: pytest backend/tests/test_audit.py -m integration
"""

import uuid
from unittest.mock import MagicMock

import pytest

from backend.core.audit import (
    GENESIS_SENTINEL,
    _verify_event_chain,
    compute_self_hash,
    redact_phi_from_payload,
)


# ── Helpers ────────────────────────────────────────────────────────────────────


def _make_event(prev_hash: str, run_id: str, seq: int) -> MagicMock:
    """Build a synthetic AuditEvent with a correct self_hash for chain tests."""
    event = MagicMock()
    event.run_id = run_id
    event.seq = seq
    event.actor = "ai"
    event.category = "test"
    event.action = "step"
    event.payload = {"seq": seq}
    event.reason = None
    event.prev_hash = prev_hash
    event.self_hash = compute_self_hash(prev_hash, run_id, seq, "ai", "test", "step", {"seq": seq}, None)
    return event


def _make_chain(run_id: str, n: int) -> list[MagicMock]:
    events: list[MagicMock] = []
    prev = GENESIS_SENTINEL
    for i in range(n):
        e = _make_event(prev, run_id, i)
        events.append(e)
        prev = e.self_hash
    return events


# ── Unit tests - pure, no Postgres ────────────────────────────────────────────


# ── PHI redaction tests ────────────────────────────────────────────────────────


def test_redact_phi_ssn():
    payload = {"note": "Patient SSN is 123-45-6789"}
    result = redact_phi_from_payload(payload)
    assert "123-45-6789" not in result["note"]
    assert "[REDACTED]" in result["note"]


def test_redact_phi_mrn():
    payload = {"id": "MRN1234567"}
    result = redact_phi_from_payload(payload)
    assert "MRN1234567" not in result["id"]
    assert "[REDACTED]" in result["id"]


def test_redact_phi_email():
    payload = {"contact": "Send results to john.doe@hospital.org"}
    result = redact_phi_from_payload(payload)
    assert "john.doe@hospital.org" not in result["contact"]
    assert "[REDACTED]" in result["contact"]


def test_redact_phi_dob():
    payload = {"history": "DOB: 1985-03-15"}
    result = redact_phi_from_payload(payload)
    assert "1985-03-15" not in result["history"]
    assert "[REDACTED]" in result["history"]


def test_redact_phi_nested_dict():
    payload = {"outer": {"inner": "ssn=111-22-3333"}}
    result = redact_phi_from_payload(payload)
    assert "111-22-3333" not in result["outer"]["inner"]


def test_redact_phi_in_list():
    payload = {"values": ["normal text", "ssn: 444-55-6666", 42]}
    result = redact_phi_from_payload(payload)
    assert "444-55-6666" not in result["values"][1]
    assert result["values"][2] == 42  # numeric unchanged


def test_redact_phi_clean_payload_unchanged():
    payload = {"rows": 500, "auc": 0.85, "model": "XGBClassifier"}
    result = redact_phi_from_payload(payload)
    assert result == payload


def test_redact_phi_none_and_bool_unchanged():
    payload = {"flag": None, "active": True, "count": 0}
    result = redact_phi_from_payload(payload)
    assert result["flag"] is None
    assert result["active"] is True


# ── Hash-chain unit tests ──────────────────────────────────────────────────────


def test_genesis_sentinel_is_literal_string():
    assert GENESIS_SENTINEL == "GENESIS"


def test_compute_self_hash_is_deterministic():
    run_id = str(uuid.uuid4())
    h1 = compute_self_hash("GENESIS", run_id, 0, "ai", "eda", "eda_complete", {"rows": 100}, "done")
    h2 = compute_self_hash("GENESIS", run_id, 0, "ai", "eda", "eda_complete", {"rows": 100}, "done")
    assert h1 == h2


def test_compute_self_hash_sensitive_to_every_field():
    run_id = str(uuid.uuid4())
    base = compute_self_hash("GENESIS", run_id, 0, "ai", "eda", "eda_complete", {"rows": 100}, None)
    assert base != compute_self_hash("GENESIS", run_id, 0, "ai", "eda", "eda_complete", {"rows": 101}, None)
    assert base != compute_self_hash("GENESIS", run_id, 1, "ai", "eda", "eda_complete", {"rows": 100}, None)
    assert base != compute_self_hash("GENESIS", run_id, 0, "user", "eda", "eda_complete", {"rows": 100}, None)
    assert base != compute_self_hash("GENESIS", run_id, 0, "ai", "eda", "override", {"rows": 100}, None)
    assert base != compute_self_hash("GENESIS", run_id, 0, "ai", "eda", "eda_complete", {"rows": 100}, "reason")


def test_compute_self_hash_none_reason_equals_empty_string():
    run_id = str(uuid.uuid4())
    h_none = compute_self_hash("GENESIS", run_id, 0, "ai", "cat", "act", {}, None)
    h_empty = compute_self_hash("GENESIS", run_id, 0, "ai", "cat", "act", {}, "")
    assert h_none == h_empty


def test_verify_event_chain_empty():
    assert _verify_event_chain([]) is True


def test_verify_event_chain_single_event():
    chain = _make_chain(str(uuid.uuid4()), 1)
    assert _verify_event_chain(chain) is True


def test_verify_event_chain_50_events():
    chain = _make_chain(str(uuid.uuid4()), 50)
    assert _verify_event_chain(chain) is True


def test_verify_event_chain_rejects_non_genesis_first_prev_hash():
    chain = _make_chain(str(uuid.uuid4()), 5)
    chain[0].prev_hash = "not-GENESIS"
    assert _verify_event_chain(chain) is False


def test_verify_event_chain_rejects_corrupted_self_hash():
    chain = _make_chain(str(uuid.uuid4()), 5)
    original = chain[2].self_hash
    chain[2].self_hash = "0" * len(original)
    assert _verify_event_chain(chain) is False


def test_verify_event_chain_rejects_broken_link():
    chain = _make_chain(str(uuid.uuid4()), 5)
    chain[3].prev_hash = "wrong-hash"
    assert _verify_event_chain(chain) is False


def test_verify_event_chain_rejects_mutated_payload():
    """Mutating payload after the event is persisted invalidates self_hash."""
    chain = _make_chain(str(uuid.uuid4()), 5)
    chain[2].payload = {"tampered": True}  # self_hash was computed over {"seq": 2}
    assert _verify_event_chain(chain) is False


def test_verify_event_chain_rejects_mutated_actor():
    chain = _make_chain(str(uuid.uuid4()), 5)
    chain[1].actor = "user"  # was "ai" when self_hash was computed
    assert _verify_event_chain(chain) is False


# ── Integration tests (require Postgres) ───────────────────────────────────────


@pytest.mark.integration
async def test_append_and_verify_50_events(create_test_database: None) -> None:
    """50-event chain appended to Postgres must verify successfully."""
    from backend.core import database as db_module
    from backend.core.audit import append, verify_chain
    from backend.core.database import Project, Run

    project_id = str(uuid.uuid4())
    run_id = str(uuid.uuid4())

    async with db_module.async_session_factory() as session:
        async with session.begin():
            session.add(Project(id=project_id, user_id="test-user", name="Audit test project"))
            session.add(Run(id=run_id, project_id=project_id))

        for i in range(50):
            async with session.begin():
                await append(
                    session,
                    run_id=run_id,
                    actor="ai",
                    category="test",
                    action=f"step_{i}",
                    payload={"index": i},
                    reason=f"Step {i}",
                )

        async with session.begin():
            result = await verify_chain(session, run_id)

    assert result is True


@pytest.mark.integration
async def test_verify_chain_empty_run(create_test_database: None) -> None:
    """A run with no audit events has a trivially valid (empty) chain."""
    from backend.core import database as db_module
    from backend.core.audit import verify_chain
    from backend.core.database import Project, Run

    project_id = str(uuid.uuid4())
    run_id = str(uuid.uuid4())

    async with db_module.async_session_factory() as session:
        async with session.begin():
            session.add(Project(id=project_id, user_id="test-user", name="Empty chain project"))
            session.add(Run(id=run_id, project_id=project_id))
        async with session.begin():
            result = await verify_chain(session, run_id)

    assert result is True


@pytest.mark.integration
async def test_append_only_trigger_rejects_update(create_test_database: None) -> None:
    """The Postgres trigger must reject UPDATE on audit_events."""
    import sqlalchemy.exc

    from backend.core import database as db_module
    from backend.core.audit import append
    from backend.core.database import AuditEvent, Project, Run

    project_id = str(uuid.uuid4())
    run_id = str(uuid.uuid4())

    async with db_module.async_session_factory() as session:
        async with session.begin():
            session.add(Project(id=project_id, user_id="test-user", name="Trigger test"))
            session.add(Run(id=run_id, project_id=project_id))

        async with session.begin():
            event = await append(
                session, run_id=run_id, actor="ai", category="test", action="created",
                payload={}, reason=None,
            )
            event_id = event.id

        # Attempt an UPDATE - must be rejected by trigger
        with pytest.raises(sqlalchemy.exc.DBAPIError):
            async with session.begin():
                from sqlalchemy import update
                await session.execute(
                    update(AuditEvent)
                    .where(AuditEvent.id == event_id)
                    .values(reason="tampered")
                )


@pytest.mark.integration
async def test_append_only_trigger_rejects_delete(create_test_database: None) -> None:
    """A plain DELETE on audit_events must still be rejected by the trigger.

    The project-purge path sets app.allow_audit_purge; an ordinary DELETE that
    does not set the flag must remain forbidden so the trail stays tamper-proof.
    """
    import sqlalchemy.exc

    from backend.core import database as db_module
    from backend.core.audit import append
    from backend.core.database import AuditEvent, Project, Run

    project_id = str(uuid.uuid4())
    run_id = str(uuid.uuid4())

    async with db_module.async_session_factory() as session:
        async with session.begin():
            session.add(Project(id=project_id, user_id="test-user", name="Delete-trigger test"))
            session.add(Run(id=run_id, project_id=project_id))

        async with session.begin():
            event = await append(
                session, run_id=run_id, actor="ai", category="test", action="created",
                payload={}, reason=None,
            )
            event_id = event.id

        with pytest.raises(sqlalchemy.exc.DBAPIError):
            async with session.begin():
                from sqlalchemy import delete
                await session.execute(delete(AuditEvent).where(AuditEvent.id == event_id))


@pytest.mark.integration
async def test_delete_project_cascades_through_audit(create_test_database: None, client) -> None:  # type: ignore[no-untyped-def]
    """Deleting a project that has audit events must succeed.

    This is the regression test for the append-only trigger aborting the
    cascade: Project → Run → AuditEvent. Without the authorized-purge flag the
    DELETE transaction rolls back and the project is never removed.
    """
    from sqlalchemy import select

    from backend.core import database as db_module
    from backend.core.audit import append
    from backend.core.database import AuditEvent, Project, Run

    user = "purge-test-user"
    project_id = str(uuid.uuid4())
    run_id = str(uuid.uuid4())

    # Seed a project with a run and a real hash-chained audit event.
    async with db_module.async_session_factory() as session:
        async with session.begin():
            session.add(Project(id=project_id, user_id=user, name="Has audit trail"))
            session.add(Run(id=run_id, project_id=project_id))
        async with session.begin():
            await append(
                session, run_id=run_id, actor="system", category="test",
                action="seed", payload={"k": "v"}, reason=None,
            )

    # Delete through the real endpoint.
    resp = await client.delete(f"/projects/{project_id}", headers={"X-User-Id": user})
    assert resp.status_code == 204

    # Project and its audit events must both be gone.
    async with db_module.async_session_factory() as session:
        assert await session.get(Project, project_id) is None
        remaining = await session.execute(
            select(AuditEvent).where(AuditEvent.run_id == run_id)
        )
        assert remaining.scalar_one_or_none() is None
