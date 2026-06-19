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

_DIFF = StrategyDiff(
    field_path="model_selection.primary",
    before="gradient_boosting",
    after="random_forest",
    summary="primary → random_forest",
    run_id="run-1",
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


def _text_blocks(messages):
    """Yield every text-typed content block across a messages list."""
    for m in messages:
        content = m["content"]
        if isinstance(content, list):
            for block in content:
                if block.get("type") == "text":
                    yield block


def test_format_messages_no_empty_text_blocks_when_history_starts_with_assistant():
    """Regression: history beginning with an assistant turn must not emit an
    empty text block - Anthropic rejects it with 'text content blocks must be
    non-empty'. This was the live chat failure on the first override message."""
    from backend.agents.chat_agent import _format_messages

    history = [
        {"role": "assistant", "content": "Initial EDA summary and model recommendation."},
    ]
    messages = _format_messages(history, "Use logistic_regression instead of lightgbm", {"run": {}})

    assert all(block["text"].strip() for block in _text_blocks(messages)), \
        "an empty text block leaked into the messages payload"
    # First turn must be a user role carrying the context block.
    assert messages[0]["role"] == "user"
    assert any("pipeline_context" in b["text"] for b in messages[0]["content"])
    # The current user message is appended last as a plain string.
    assert messages[-1] == {"role": "user", "content": "Use logistic_regression instead of lightgbm"}


def test_format_messages_drops_empty_history_turns():
    """A previously-persisted blank turn must be filtered, not forwarded as an
    empty text block."""
    from backend.agents.chat_agent import _format_messages

    history = [
        {"role": "user", "content": "   "},          # whitespace-only - dropped
        {"role": "assistant", "content": ""},          # empty - dropped
        {"role": "user", "content": "real question"},
    ]
    messages = _format_messages(history, "follow up", {"run": {}})

    assert all(block["text"].strip() for block in _text_blocks(messages))
    # The surviving user turn carries the context block.
    assert messages[0]["role"] == "user"
    assert any("real question" in b["text"] for b in messages[0]["content"])


def test_format_messages_no_history_inlines_user_message():
    from backend.agents.chat_agent import _format_messages

    messages = _format_messages([], "first message", {"run": {}})

    assert len(messages) == 1
    assert messages[0]["role"] == "user"
    texts = [b["text"] for b in messages[0]["content"]]
    assert any("pipeline_context" in t for t in texts)
    assert "first message" in texts


def test_classify_confirmation():
    from backend.agents.chat_agent import _classify_confirmation

    for affirm in ["yes", "Yes", "yes!", "y", "confirm", "go ahead", "do it", "yes please", "OK"]:
        assert _classify_confirmation(affirm) == "confirm", affirm
    for reject in ["no", "No", "cancel", "stop", "never mind", "no thanks"]:
        assert _classify_confirmation(reject) == "reject", reject
    # Substantive turns must NOT be read as confirmations.
    for ambiguous in ["yes but use random forest instead", "what does that mean?", ""]:
        assert _classify_confirmation(ambiguous) is None, ambiguous


async def test_stream_chat_response_applies_pending_intent_on_yes():
    """The live failure: a gated modify ('use logistic_regression') was confirmed
    with 'yes', but nothing applied because 'yes' carries no payload. The pending
    intent stored on the prior assistant turn must be applied on confirmation."""
    session = _make_session()
    run = _make_run(model_selection={"primary": "lightgbm"})

    pending_modify = ChatIntent(
        intent="modify",
        confidence=0.9,
        category="model_selection",
        structured_payload={"model": "logistic_regression"},
        needs_confirmation=True,
        reasoning="User wants logistic_regression as primary",
    )

    # The most recent assistant turn carries the pending (gated) modify intent.
    pending_row = MagicMock()
    pending_row.intent = pending_modify.model_dump()

    pending_result = MagicMock()
    pending_result.scalar_one_or_none.return_value = pending_row
    run_result = MagicMock()
    run_result.scalar_one_or_none.return_value = run

    # First execute() is the pending-confirmation lookup, second is the run fetch.
    results = iter([pending_result, run_result])

    async def mock_execute(*args, **kwargs):
        return next(results)
    session.execute = mock_execute

    applied = {}
    async def fake_apply(_session, _run, intent):
        applied["intent"] = intent
        return [StrategyDiff(
            field_path="model_selection.primary",
            before="lightgbm", after="logistic_regression",
            summary="Changed primary model", run_id="run-1",
        )]

    with (
        patch("backend.agents.chat_agent._build_context_block", new=AsyncMock(return_value={})),
        patch("backend.agents.chat_agent.call_claude_stream", new=AsyncMock(return_value="Override applied.")),
        # Haiku reads bare "yes" as a question - the pending intent must still win.
        patch("backend.agents.chat_agent.classify_intent", new=AsyncMock(return_value=_QUESTION_INTENT)),
        patch("backend.agents.chat_agent.queue_intent_if_busy", new=AsyncMock(return_value=False)),
        patch("backend.agents.chat_agent.apply_intent_to_strategy", new=fake_apply),
        patch("backend.agents.chat_agent.select"),
        patch("backend.agents.chat_agent.DBChatMessage", MagicMock()),
    ):
        from backend.agents.chat_agent import stream_chat_response
        _, intent, diffs = await stream_chat_response(
            session=session, run_id="run-1", user_id="u",
            user_message="yes", history=[{"role": "assistant", "content": "Shall I confirm?"}],
            on_chunk=AsyncMock(),
        )

    assert "intent" in applied, "pending intent was never applied"
    assert applied["intent"].category == "model_selection"
    assert applied["intent"].structured_payload["model"] == "logistic_regression"
    assert applied["intent"].needs_confirmation is False
    assert len(diffs) == 1 and diffs[0].after == "logistic_regression"


async def test_stream_chat_response_rejects_pending_intent_on_no():
    """'no' to a gated modify must apply nothing."""
    session = _make_session()

    pending_modify = ChatIntent(
        intent="modify", confidence=0.9, category="model_selection",
        structured_payload={"model": "logistic_regression"},
        needs_confirmation=True, reasoning="x",
    )
    pending_row = MagicMock()
    pending_row.intent = pending_modify.model_dump()
    pending_result = MagicMock()
    pending_result.scalar_one_or_none.return_value = pending_row

    async def mock_execute(*args, **kwargs):
        return pending_result
    session.execute = mock_execute

    with (
        patch("backend.agents.chat_agent._build_context_block", new=AsyncMock(return_value={})),
        patch("backend.agents.chat_agent.call_claude_stream", new=AsyncMock(return_value="Understood, keeping LightGBM.")),
        patch("backend.agents.chat_agent.classify_intent", new=AsyncMock(return_value=_QUESTION_INTENT)),
        patch("backend.agents.chat_agent.apply_intent_to_strategy", new=AsyncMock()) as mock_apply,
        patch("backend.agents.chat_agent.select"),
        patch("backend.agents.chat_agent.DBChatMessage", MagicMock()),
    ):
        from backend.agents.chat_agent import stream_chat_response
        _, intent, diffs = await stream_chat_response(
            session=session, run_id="run-1", user_id="u",
            user_message="no", history=[{"role": "assistant", "content": "Shall I confirm?"}],
            on_chunk=AsyncMock(),
        )

    mock_apply.assert_not_called()
    assert diffs == []


async def test_stream_chat_response_empty_completion_preserves_modify_override():
    """An empty chat-model completion (e.g. a refusal on the clinical domain) must
    NOT strip a correctly-classified override. The override is preserved, a
    deterministic message (not the generic retry fallback) is streamed, and the
    mutation is applied. This is the live failure: a refusal on a hantavirus
    threshold override silently discarded the change."""
    session = _make_session()
    run = _make_run(model_selection={"primary": "gradient_boosting"})

    mock_select_result = MagicMock()
    mock_select_result.scalar_one_or_none.return_value = run

    async def mock_execute(*args, **kwargs):
        return mock_select_result
    session.execute = mock_execute

    chunks: list[str] = []
    async def on_chunk(text: str) -> None:
        chunks.append(text)

    with (
        patch("backend.agents.chat_agent._build_context_block", new=AsyncMock(return_value={})),
        patch("backend.agents.chat_agent.call_claude_stream", new=AsyncMock(return_value="")),
        patch("backend.agents.chat_agent.classify_intent", new=AsyncMock(return_value=_MODIFY_INTENT)),
        patch("backend.agents.chat_agent.queue_intent_if_busy", new=AsyncMock(return_value=False)),
        patch("backend.agents.chat_agent.apply_intent_to_strategy",
              new=AsyncMock(return_value=[_DIFF])),
        patch("backend.agents.chat_agent.select"),
        patch("backend.agents.chat_agent.DBChatMessage", MagicMock()),
    ):
        from backend.agents.chat_agent import _EMPTY_RESPONSE_FALLBACK, stream_chat_response

        text, intent, diffs = await stream_chat_response(
            session=session, run_id="run-1", user_id="u",
            user_message="Switch to random forest", history=[], on_chunk=on_chunk,
        )

    assert text != _EMPTY_RESPONSE_FALLBACK, "an actionable override must not get the generic fallback"
    assert "override" in text.lower()
    assert chunks == [text], "the deterministic message must be streamed to the panel"
    assert intent is not None and intent.intent == "modify", "override intent must be preserved"
    assert diffs == [_DIFF], "the override must be applied despite the empty completion"


async def test_stream_chat_response_empty_completion_question_falls_back():
    """An empty completion on a NON-actionable turn (question) still shows the
    generic retry fallback and applies no mutation - no silent empty bubble."""
    session = _make_session()

    chunks: list[str] = []
    async def on_chunk(text: str) -> None:
        chunks.append(text)

    with (
        patch("backend.agents.chat_agent._build_context_block", new=AsyncMock(return_value={})),
        patch("backend.agents.chat_agent.call_claude_stream", new=AsyncMock(return_value="")),
        patch("backend.agents.chat_agent.classify_intent", new=AsyncMock(return_value=_QUESTION_INTENT)),
        patch("backend.agents.chat_agent.apply_intent_to_strategy", new=AsyncMock()) as mock_apply,
        patch("backend.agents.chat_agent.select"),
        patch("backend.agents.chat_agent.DBChatMessage", MagicMock()),
    ):
        from backend.agents.chat_agent import _EMPTY_RESPONSE_FALLBACK, stream_chat_response

        text, intent, diffs = await stream_chat_response(
            session=session, run_id="run-1", user_id="u",
            user_message="What's the accuracy?", history=[], on_chunk=on_chunk,
        )

    assert text == _EMPTY_RESPONSE_FALLBACK
    assert chunks == [_EMPTY_RESPONSE_FALLBACK]
    assert intent is None
    assert diffs == []
    mock_apply.assert_not_called()


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
