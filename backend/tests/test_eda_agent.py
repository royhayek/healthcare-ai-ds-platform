"""Tests for backend/agents/eda_agent.py - the model is mocked.

These tests verify that:
1. A valid model JSON response produces a correct EdaReport.
2. A parse-failure response falls back to EdaReport.safe_fallback().
3. audit.append is called exactly once with the right action in each case.
4. ProgressEmitter.emit_async is called (i.e., progress events are fired).
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.models.eda import EdaReport


_VALID_CLAUDE_RESPONSE = json.dumps({
    "overview": "Telco churn dataset with 7032 rows and moderate class imbalance.",
    "target_analysis": {
        "column": "churn",
        "task_type": "binary_classification",
        "class_balance": {"0": 0.855, "1": 0.145},
        "notes": "Imbalanced - minority class is 14.5%.",
    },
    "quality_issues": [
        {
            "column": "total_charges",
            "issue": "11 null values (0.16%)",
            "severity": "low",
            "recommendation": "Impute with median",
        }
    ],
    "correlations": {
        "high_pairs": [{"col_a": "monthly_charges", "col_b": "total_charges", "correlation": 0.83}],
        "leakage_risk": [],
    },
    "preprocessing_recommendations": [
        {"column": "total_charges", "strategy": "median_imputation", "reason": "Low null count, right-skewed"},
        {"column": "tenure", "strategy": "standard_scaling", "reason": "Continuous, approximately normal"},
    ],
    "model_recommendation": "gradient_boosting",
    "summary": "Churn prediction dataset suitable for binary classification with SMOTE or class_weight balancing.",
})

_INVALID_CLAUDE_RESPONSE = "I'm sorry, I cannot parse this dataset. The data seems unusual."


@pytest.fixture
def mock_session():
    # session.begin() must be a regular call returning an async context manager,
    # not a coroutine. AsyncMock wraps everything as a coroutine, so use MagicMock
    # for session and set up begin() to return a proper async CM.
    cm = AsyncMock()
    cm.__aenter__ = AsyncMock(return_value=None)
    cm.__aexit__ = AsyncMock(return_value=False)

    session = MagicMock()
    session.begin.return_value = cm
    session.commit = AsyncMock()
    return session


@pytest.fixture
def mock_emitter():
    emitter = MagicMock()
    emitter.emit_async = AsyncMock()
    return emitter


@pytest.fixture
def compressed_profile():
    return {
        "n_rows": 7032,
        "n_cols": 21,
        "target_column": "churn",
        "task_type": "binary_classification",
        "columns": [],
        "high_correlation_pairs": [],
    }


async def test_eda_agent_valid_response(mock_session, mock_emitter, compressed_profile):
    with (
        patch("backend.agents.eda_agent.call_claude_stream", new=AsyncMock(return_value=_VALID_CLAUDE_RESPONSE)),
        patch("backend.agents.eda_agent.audit.append", new=AsyncMock()) as mock_audit,
    ):
        from backend.agents.eda_agent import run_eda_agent
        report = await run_eda_agent(mock_session, "run-123", compressed_profile, mock_emitter)

    assert isinstance(report, EdaReport)
    assert report.model_recommendation == "gradient_boosting"
    assert len(report.quality_issues) == 1
    assert len(report.preprocessing_recommendations) == 2

    # Audit was called once with eda_complete
    mock_audit.assert_called_once()
    call_kwargs = mock_audit.call_args[1]
    assert call_kwargs["action"] == "eda_complete"
    assert call_kwargs["payload"]["parse_failed"] is False


async def test_eda_agent_parse_failure_uses_fallback(mock_session, mock_emitter, compressed_profile):
    with (
        patch("backend.agents.eda_agent.call_claude_stream", new=AsyncMock(return_value=_INVALID_CLAUDE_RESPONSE)),
        patch("backend.agents.eda_agent.audit.append", new=AsyncMock()) as mock_audit,
    ):
        from backend.agents.eda_agent import run_eda_agent
        report = await run_eda_agent(mock_session, "run-456", compressed_profile, mock_emitter)

    assert isinstance(report, EdaReport)
    # Safe fallback uses gradient_boosting as default
    assert report.model_recommendation == "gradient_boosting"

    mock_audit.assert_called_once()
    call_kwargs = mock_audit.call_args[1]
    assert call_kwargs["action"] == "eda_parse_failure"
    assert call_kwargs["payload"]["parse_failed"] is True


async def test_eda_agent_emits_progress_events(mock_session, mock_emitter, compressed_profile):
    with (
        patch("backend.agents.eda_agent.call_claude_stream", new=AsyncMock(return_value=_VALID_CLAUDE_RESPONSE)),
        patch("backend.agents.eda_agent.audit.append", new=AsyncMock()),
    ):
        from backend.agents.eda_agent import run_eda_agent
        await run_eda_agent(mock_session, "run-789", compressed_profile, mock_emitter)

    assert mock_emitter.emit_async.call_count >= 3  # start, parse, complete


async def test_eda_agent_passes_correct_model_to_claude(mock_session, mock_emitter, compressed_profile):
    with (
        patch("backend.agents.eda_agent.call_claude_stream", new=AsyncMock(return_value=_VALID_CLAUDE_RESPONSE)) as mock_stream,
        patch("backend.agents.eda_agent.audit.append", new=AsyncMock()),
    ):
        from backend.agents.eda_agent import run_eda_agent
        from backend.core.config import settings
        await run_eda_agent(mock_session, "run-abc", compressed_profile, mock_emitter)

    call_kwargs = mock_stream.call_args[1]
    assert call_kwargs["model"] == settings.CLAUDE_SONNET_MODEL
    assert "claude-sonnet-4-6" == settings.CLAUDE_SONNET_MODEL


async def test_eda_agent_audit_includes_model_string(mock_session, mock_emitter, compressed_profile):
    with (
        patch("backend.agents.eda_agent.call_claude_stream", new=AsyncMock(return_value=_VALID_CLAUDE_RESPONSE)),
        patch("backend.agents.eda_agent.audit.append", new=AsyncMock()) as mock_audit,
    ):
        from backend.agents.eda_agent import run_eda_agent
        await run_eda_agent(mock_session, "run-def", compressed_profile, mock_emitter)

    payload = mock_audit.call_args[1]["payload"]
    assert payload["model"] == "claude-sonnet-4-6"
