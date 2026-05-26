"""SSE integration test - end-to-end wire-up (§21).

Verifies that:
1. POST /runs/{run_id}/chat returns text/event-stream
2. text_chunk events arrive in the response body
3. intent event arrives
4. done event is the final event
5. strategy_diff arrives and is ordered before done when intent is "modify"

the model is mocked at the module level - no real API calls.
Postgres is required (run with docker-compose up, then pytest -m integration).
"""

import json
import uuid
from unittest.mock import AsyncMock, patch

import pytest

from backend.models.chat import ChatIntent, StrategyDiff

_MOCK_CHUNKS = ["Based on the EDA, ", "gradient boosting ", "is the best choice."]
_MOCK_RESPONSE = "".join(_MOCK_CHUNKS)

_QUESTION_INTENT = ChatIntent(
    intent="question",
    confidence=0.95,
    category="eda",
    structured_payload={},
    needs_confirmation=False,
    reasoning="User asked about model recommendation",
)

_MODIFY_INTENT = ChatIntent(
    intent="modify",
    confidence=0.88,
    category="model_selection",
    structured_payload={"model": "random_forest"},
    needs_confirmation=False,
    reasoning="User wants random forest",
)


async def _stream_fake(*args, on_chunk=None, **kwargs):
    """Fake call_claude_stream that emits chunks through on_chunk."""
    for chunk in _MOCK_CHUNKS:
        if on_chunk:
            result = on_chunk(chunk)
            if hasattr(result, "__await__"):
                await result
    return _MOCK_RESPONSE


def _parse_sse_events(body: bytes) -> list[dict]:
    events = []
    for line in body.decode().split("\n"):
        line = line.strip()
        if line.startswith("data: "):
            events.append(json.loads(line[6:]))
    return events


@pytest.mark.integration
async def test_sse_question_intent(create_test_database: None, client) -> None:
    """Full SSE round-trip: question intent produces text_chunk → intent → done."""
    from backend.core import database as db_module
    from backend.core.database import Dataset, Project, Run

    project_id = str(uuid.uuid4())
    run_id = str(uuid.uuid4())
    dataset_id = str(uuid.uuid4())

    async with db_module.async_session_factory() as session:
        async with session.begin():
            session.add(Project(id=project_id, user_id="dev-user-1", name="SSE Test Project"))
    async with db_module.async_session_factory() as session:
        async with session.begin():
            session.add(Dataset(
                id=dataset_id,
                project_id=project_id,
                role="training",
                filename="telco_churn.csv",
                storage_path="dummy/path.csv",
                sha256="abc123",
                schema_hash="deadbeef",
            ))
    async with db_module.async_session_factory() as session:
        async with session.begin():
            session.add(Run(
                id=run_id,
                project_id=project_id,
                training_dataset_id=dataset_id,
                status="awaiting_checkpoint",
                current_step="checkpoint_1_eda",
                progress=40,
                eda_report={
                    "model_recommendation": "gradient_boosting",
                    "summary": "Churn dataset, 14% minority class.",
                },
            ))

    with (
        patch("backend.agents.chat_agent.call_claude_stream", new=_stream_fake),
        patch("backend.agents.intent_extractor.call_claude", new=AsyncMock(
            return_value=json.dumps(_QUESTION_INTENT.model_dump())
        )),
    ):
        response = await client.post(
            f"/runs/{run_id}/chat",
            json={"content": "What model should I use?"},
            headers={"X-User-Id": "dev-user-1"},
        )

    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]

    events = _parse_sse_events(response.content)
    types = [e["type"] for e in events]

    assert "text_chunk" in types, f"Expected text_chunk events, got: {types}"
    assert types[-1] == "done", f"Expected 'done' as last event, got: {types}"

    # Verify chunks reconstruct the full response
    full_text = "".join(e["content"] for e in events if e["type"] == "text_chunk")
    assert full_text == _MOCK_RESPONSE

    # Intent event must arrive before done
    intent_idx = next((i for i, e in enumerate(events) if e["type"] == "intent"), None)
    done_idx = types.index("done")
    assert intent_idx is not None, "No intent event found"
    assert intent_idx < done_idx


@pytest.mark.integration
async def test_sse_modify_intent_produces_strategy_diff(create_test_database: None, client) -> None:
    """Modify intent causes strategy_diff event before done."""
    from backend.core import database as db_module
    from backend.core.database import Dataset, Project, Run

    project_id = str(uuid.uuid4())
    run_id = str(uuid.uuid4())
    dataset_id = str(uuid.uuid4())

    async with db_module.async_session_factory() as session:
        async with session.begin():
            session.add(Project(id=project_id, user_id="dev-user-1", name="Diff Test Project"))
    async with db_module.async_session_factory() as session:
        async with session.begin():
            session.add(Dataset(
                id=dataset_id,
                project_id=project_id,
                role="training",
                filename="telco_churn.csv",
                storage_path="dummy/path.csv",
                sha256="def456",
                schema_hash="deadbeef",
            ))
    async with db_module.async_session_factory() as session:
        async with session.begin():
            session.add(Run(
                id=run_id,
                project_id=project_id,
                training_dataset_id=dataset_id,
                status="awaiting_checkpoint",
                model_selection={"primary": "gradient_boosting", "candidates": []},
            ))

    mock_diff = StrategyDiff(
        field_path="model_selection.primary",
        before="gradient_boosting",
        after="random_forest",
        summary="Changed primary model",
        run_id=run_id,
    )

    with (
        patch("backend.agents.chat_agent.call_claude_stream", new=_stream_fake),
        patch("backend.agents.intent_extractor.call_claude", new=AsyncMock(
            return_value=json.dumps(_MODIFY_INTENT.model_dump())
        )),
        patch(
            "backend.agents.chat_agent.apply_intent_to_strategy",
            new=AsyncMock(return_value=[mock_diff]),
        ),
    ):
        response = await client.post(
            f"/runs/{run_id}/chat",
            json={"content": "Switch to random forest"},
            headers={"X-User-Id": "dev-user-1"},
        )

    assert response.status_code == 200
    events = _parse_sse_events(response.content)
    types = [e["type"] for e in events]

    assert "strategy_diff" in types, f"Expected strategy_diff, got: {types}"
    diff_idx = types.index("strategy_diff")
    done_idx = types.index("done")
    assert diff_idx < done_idx, "strategy_diff must arrive before done"

    diff_event = next(e for e in events if e["type"] == "strategy_diff")
    assert diff_event["diffs"][0]["field_path"] == "model_selection.primary"
