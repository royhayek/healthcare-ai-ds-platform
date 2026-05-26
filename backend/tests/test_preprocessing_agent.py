"""Unit tests for agents/preprocessing_agent.py."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.agents.preprocessing_agent import _safe_fallback, run_preprocessing_agent
from backend.models.strategy import PreprocessingStrategy


@pytest.fixture()
def mock_session():
    session = MagicMock()
    cm = AsyncMock()
    cm.__aenter__ = AsyncMock(return_value=cm)
    cm.__aexit__ = AsyncMock(return_value=False)
    session.begin = MagicMock(return_value=cm)
    session.commit = AsyncMock()
    return session


@pytest.fixture()
def mock_emitter():
    emitter = AsyncMock()
    emitter.emit_async = AsyncMock()
    return emitter


VALID_RESPONSE = """{
  "columns": {
    "tenure": {
      "action": "keep",
      "dtype_hint": "numeric",
      "impute_strategy": "median",
      "scale_strategy": "standard",
      "reason": "Numeric, low missing rate"
    },
    "MonthlyCharges": {
      "action": "keep",
      "dtype_hint": "numeric",
      "impute_strategy": "median",
      "scale_strategy": "standard",
      "reason": "Numeric, slight right skew"
    },
    "Contract": {
      "action": "keep",
      "dtype_hint": "categorical",
      "impute_strategy": "most_frequent",
      "encode_strategy": "onehot",
      "reason": "Low cardinality categorical"
    }
  },
  "drop_high_correlation": [],
  "notes": "Standard preprocessing for telecom churn dataset."
}"""


class TestRunPreprocessingAgent:
    @pytest.mark.asyncio
    async def test_success_returns_preprocessing_strategy(self, mock_session, mock_emitter):
        with (
            patch("backend.agents.preprocessing_agent.call_claude", return_value=VALID_RESPONSE),
            patch("backend.agents.preprocessing_agent.audit") as mock_audit,
        ):
            mock_audit.append = AsyncMock()

            strategy = await run_preprocessing_agent(
                session=mock_session,
                run_id="test-run-123",
                compressed_profile={"columns": [], "n_rows": 7032},
                eda_report={"quality_issues": [], "summary": "Test"},
                target_column="Churn",
                task_type="binary_classification",
                emitter=mock_emitter,
            )

        assert isinstance(strategy, PreprocessingStrategy)
        assert "tenure" in strategy.columns
        assert strategy.columns["tenure"].action == "keep"
        assert strategy.columns["tenure"].dtype_hint == "numeric"
        assert strategy.target_column == "Churn"
        assert strategy.task_type == "binary_classification"

    @pytest.mark.asyncio
    async def test_parse_failure_returns_fallback(self, mock_session, mock_emitter):
        with (
            patch("backend.agents.preprocessing_agent.call_claude", return_value="NOT JSON"),
            patch("backend.agents.preprocessing_agent.audit") as mock_audit,
        ):
            mock_audit.append = AsyncMock()

            profile = {
                "columns": [
                    {"name": "tenure", "dtype": "int64", "null_pct": 0.0},
                    {"name": "MonthlyCharges", "dtype": "float64", "null_pct": 0.02},
                    {"name": "Contract", "dtype": "object", "null_pct": 0.0},
                ]
            }

            strategy = await run_preprocessing_agent(
                session=mock_session,
                run_id="test-run-456",
                compressed_profile=profile,
                eda_report={"quality_issues": []},
                target_column="Churn",
                task_type="binary_classification",
                emitter=mock_emitter,
            )

        assert isinstance(strategy, PreprocessingStrategy)
        assert "Safe fallback" in (strategy.notes or "")

    @pytest.mark.asyncio
    async def test_audit_event_emitted_on_success(self, mock_session, mock_emitter):
        with (
            patch("backend.agents.preprocessing_agent.call_claude", return_value=VALID_RESPONSE),
            patch("backend.agents.preprocessing_agent.audit") as mock_audit,
        ):
            mock_audit.append = AsyncMock()

            await run_preprocessing_agent(
                session=mock_session,
                run_id="audit-check-run",
                compressed_profile={},
                eda_report={},
                target_column="target",
                task_type="binary_classification",
                emitter=mock_emitter,
            )

        mock_audit.append.assert_called_once()
        call_kwargs = mock_audit.append.call_args.kwargs
        assert call_kwargs["action"] == "preprocessing_strategy_complete"
        assert call_kwargs["category"] == "preprocessing"


class TestSafeFallback:
    def test_numeric_column_gets_standard_scaling(self):
        profile = {
            "columns": [{"name": "age", "dtype": "int64", "null_pct": 0.0}]
        }
        strategy = _safe_fallback(profile, "y", "binary_classification")
        assert "age" in strategy.columns
        assert strategy.columns["age"].dtype_hint == "numeric"
        assert strategy.columns["age"].scale_strategy == "standard"

    def test_high_null_column_dropped(self):
        profile = {
            "columns": [{"name": "sparse_col", "dtype": "float64", "null_pct": 0.75}]
        }
        strategy = _safe_fallback(profile, "y", "binary_classification")
        assert strategy.columns["sparse_col"].action == "drop"

    def test_categorical_column_gets_onehot(self):
        profile = {
            "columns": [{"name": "city", "dtype": "object", "null_pct": 0.0}]
        }
        strategy = _safe_fallback(profile, "y", "binary_classification")
        assert strategy.columns["city"].dtype_hint == "categorical"
        assert strategy.columns["city"].encode_strategy == "onehot"
