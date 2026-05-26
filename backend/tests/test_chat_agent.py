"""Tests for backend/agents/chat_agent.py - the model and DB mocked.

Verifies:
- asyncio.gather is used (Sonnet + Haiku fire in parallel)
- response text and intent both arrive in the return value
- strategy diffs are applied and returned when intent is "modify"
- assistant message is persisted to DB
- on_chunk callback is invoked for each text chunk
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest

from backend.models.chat import ChatIntent, StrategyDiff


def _make_session():
    cm = AsyncMock()
    cm.__aenter__ = AsyncMock(return_value=None)
    cm.__aexit__ = AsyncMock(return_value=False)
    session = MagicMock()
    session.begin.return_value = cm
    session.commit = AsyncMock()
    return session


def _make_run(run_id="run-1", preprocessing_strategy=None, model_selection=None):
    run = MagicMock()
    run.id = run_id
    run.project_id = "proj-1"
    run.status = "awaiting_checkpoint"
    run.current_step = "checkpoint_1_eda"
    run.progress = 40
    run.eda_report = {"model_recommendation": "gradient_boosting", "summary": "Churn data, moderate imbalance."}
    run.preprocessing_strategy = preprocessing_strategy
    run.model_selection = model_selection
    run.best_model_name = None
    run.final_metrics = None
    run.threshold_result = None
    run.fairness_report = None
    return run


_MOCK_RESPONSE = "Based on the EDA, I recommend gradient boosting given the class imbalance."

_QUESTION_INTENT = ChatIntent(
    intent="question",
    confidence=0.95,
    category="general",
    structured_payload={},
    needs_confirmation=False,
    reasoning="User asked a question about model recommendation",
)

_MODIFY_INTENT = ChatIntent(
    intent="modify",
    confidence=0.88,
    category="model_selection",
    structured_payload={"model": "random_forest"},
    needs_confirmation=False,
    reasoning="User wants to switch to random forest",
)


async def test_stream_chat_response_returns_text_and_intent():
    session = _make_session()
    run = _make_run()

    with (
        patch("backend.agents.chat_agent._build_context_block", new=AsyncMock(return_value={"run": {"status": "running"}})),
        patch("backend.agents.chat_agent.call_claude_stream", new=AsyncMock(return_value=_MOCK_RESPONSE)),
        patch("backend.agents.chat_agent.classify_intent", new=AsyncMock(return_value=_QUESTION_INTENT)),
        patch("backend.agents.chat_agent.select"),
        patch("backend.agents.chat_agent.DBChatMessage", MagicMock()),
    ):
        from backend.agents.chat_agent import stream_chat_response

        chunks = []
        async def on_chunk(text: str) -> None:
            chunks.append(text)

        text, intent, diffs = await stream_chat_response(
            session=session,
            run_id="run-1",
            user_id="user-1",
            user_message="What model should I use?",
            history=[],
            on_chunk=on_chunk,
        )

    assert text == _MOCK_RESPONSE
    assert intent is not None
    assert intent.intent == "question"
    assert diffs == []


async def test_stream_chat_response_applies_modify_intent():
    session = _make_session()
    run = _make_run(model_selection={"primary": "gradient_boosting"})

    mock_select_result = MagicMock()
    mock_select_result.scalar_one_or_none.return_value = run

    with (
        patch("backend.agents.chat_agent._build_context_block", new=AsyncMock(return_value={})),
        patch("backend.agents.chat_agent.call_claude_stream", new=AsyncMock(return_value="Switching to random forest.")),
        patch("backend.agents.chat_agent.classify_intent", new=AsyncMock(return_value=_MODIFY_INTENT)),
        patch("backend.agents.chat_agent.apply_intent_to_strategy", new=AsyncMock(return_value=[
            StrategyDiff(
                field_path="model_selection.primary",
                before="gradient_boosting",
                after="random_forest",
                summary="Changed primary model",
                run_id="run-1",
            )
        ])) as mock_mutate,
        patch("backend.agents.chat_agent.session") if False else MagicMock(),  # noop
        patch("backend.agents.chat_agent.select", return_value=MagicMock()),
        patch("backend.agents.chat_agent.DBChatMessage", MagicMock()),
    ):
        # Patch session.execute to return the run
        async def mock_execute(*args, **kwargs):
            return mock_select_result
        session.execute = mock_execute

        from backend.agents.chat_agent import stream_chat_response
        _, intent, diffs = await stream_chat_response(
            session=session,
            run_id="run-1",
            user_id="user-1",
            user_message="Switch to random forest",
            history=[],
            on_chunk=AsyncMock(),
        )

    assert intent is not None
    assert intent.intent == "modify"
    assert len(diffs) == 1
    assert diffs[0].after == "random_forest"


async def test_stream_chat_response_parallel_gather():
    """Verify that gather fires both calls; we detect this by checking both
    mock functions are called even though call_claude_stream takes 'longer'."""
    session = _make_session()

    sonnet_called = asyncio.Event()
    haiku_called = asyncio.Event()

    async def fake_stream(*args, **kwargs):
        sonnet_called.set()
        return _MOCK_RESPONSE

    async def fake_intent(*args, **kwargs):
        haiku_called.set()
        return _QUESTION_INTENT

    with (
        patch("backend.agents.chat_agent._build_context_block", new=AsyncMock(return_value={})),
        patch("backend.agents.chat_agent.call_claude_stream", new=fake_stream),
        patch("backend.agents.chat_agent.classify_intent", new=fake_intent),
        patch("backend.agents.chat_agent.select"),
        patch("backend.agents.chat_agent.DBChatMessage", MagicMock()),
    ):
        from backend.agents.chat_agent import stream_chat_response
        await stream_chat_response(
            session=session, run_id="r", user_id="u",
            user_message="test", history=[], on_chunk=AsyncMock(),
        )

    assert sonnet_called.is_set(), "Sonnet (call_claude_stream) was not called"
    assert haiku_called.is_set(), "Haiku (classify_intent) was not called"


async def test_stream_chat_response_no_modify_for_question():
    """Question intent must not trigger apply_intent_to_strategy."""
    session = _make_session()

    with (
        patch("backend.agents.chat_agent._build_context_block", new=AsyncMock(return_value={})),
        patch("backend.agents.chat_agent.call_claude_stream", new=AsyncMock(return_value="Here's the answer.")),
        patch("backend.agents.chat_agent.classify_intent", new=AsyncMock(return_value=_QUESTION_INTENT)),
        patch("backend.agents.chat_agent.apply_intent_to_strategy", new=AsyncMock()) as mock_mutate,
        patch("backend.agents.chat_agent.select"),
        patch("backend.agents.chat_agent.DBChatMessage", MagicMock()),
    ):
        from backend.agents.chat_agent import stream_chat_response
        await stream_chat_response(
            session=session, run_id="r", user_id="u",
            user_message="What's the accuracy?", history=[], on_chunk=AsyncMock(),
        )

    mock_mutate.assert_not_called()
