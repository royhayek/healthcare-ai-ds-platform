"""Unit tests for agents/model_selector_agent.py."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.agents.model_selector_agent import (
    _apply_hard_exclusion_rules,
    _safe_fallback,
    run_model_selector_agent,
)
from backend.models.strategy import ModelSelectionStrategy


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
    e = AsyncMock()
    e.emit_async = AsyncMock()
    return e


VALID_RESPONSE = """{
  "candidates": ["xgboost", "lightgbm", "random_forest"],
  "primary": "xgboost",
  "primary_metric": "auc",
  "excluded": [],
  "reasoning": "XGBoost handles imbalanced data well and has strong performance on tabular data.",
  "notes": null
}"""


class TestHardExclusionRules:
    def test_gradient_boosting_excluded_large_dataset(self):
        eligible, excluded = _apply_hard_exclusion_rules(150_000, "binary_classification")
        names = [e["name"] for e in excluded]
        assert "gradient_boosting" in names
        assert "gradient_boosting" not in eligible

    def test_no_exclusions_small_dataset(self):
        eligible, excluded = _apply_hard_exclusion_rules(5_000, "binary_classification")
        assert "gradient_boosting" in eligible
        assert len(excluded) == 0

    def test_regression_excludes_classification_models(self):
        eligible, _ = _apply_hard_exclusion_rules(1_000, "regression")
        assert "logistic_regression" not in eligible
        assert "linear_regression" in eligible
        assert "ridge_regression" in eligible

    def test_classification_excludes_regression_models(self):
        eligible, _ = _apply_hard_exclusion_rules(1_000, "binary_classification")
        assert "linear_regression" not in eligible
        assert "logistic_regression" in eligible


class TestRunModelSelectorAgent:
    @pytest.mark.asyncio
    async def test_success_returns_strategy(self, mock_session, mock_emitter):
        with (
            patch("backend.agents.model_selector_agent.call_claude", return_value=VALID_RESPONSE),
            patch("backend.agents.model_selector_agent.audit") as mock_audit,
        ):
            mock_audit.append = AsyncMock()

            strategy = await run_model_selector_agent(
                session=mock_session,
                run_id="run-sel-001",
                compressed_profile={"n_rows": 7032, "n_cols": 10},
                eda_report={"summary": "Telco churn data", "quality_issues": []},
                task_type="binary_classification",
                n_rows=7032,
                emitter=mock_emitter,
            )

        assert isinstance(strategy, ModelSelectionStrategy)
        assert strategy.primary == "xgboost"
        assert "xgboost" in strategy.candidates
        assert strategy.primary_metric == "auc"

    @pytest.mark.asyncio
    async def test_parse_failure_returns_fallback(self, mock_session, mock_emitter):
        with (
            patch("backend.agents.model_selector_agent.call_claude", return_value="GARBAGE"),
            patch("backend.agents.model_selector_agent.audit") as mock_audit,
        ):
            mock_audit.append = AsyncMock()

            strategy = await run_model_selector_agent(
                session=mock_session,
                run_id="run-sel-002",
                compressed_profile={},
                eda_report={},
                task_type="binary_classification",
                n_rows=1000,
                emitter=mock_emitter,
            )

        assert isinstance(strategy, ModelSelectionStrategy)
        assert strategy.primary in strategy.candidates

    @pytest.mark.asyncio
    async def test_audit_appended(self, mock_session, mock_emitter):
        with (
            patch("backend.agents.model_selector_agent.call_claude", return_value=VALID_RESPONSE),
            patch("backend.agents.model_selector_agent.audit") as mock_audit,
        ):
            mock_audit.append = AsyncMock()

            await run_model_selector_agent(
                session=mock_session,
                run_id="run-sel-003",
                compressed_profile={},
                eda_report={},
                task_type="binary_classification",
                n_rows=5000,
                emitter=mock_emitter,
            )

        mock_audit.append.assert_called_once()
        kwargs = mock_audit.append.call_args.kwargs
        assert kwargs["category"] == "model_selection"
        assert kwargs["action"] == "model_selection_complete"

    @pytest.mark.asyncio
    async def test_hard_exclusion_applied_to_large_dataset(self, mock_session, mock_emitter):
        """gradient_boosting must be excluded from candidates on 150k-row dataset."""
        response_with_gb = """{
          "candidates": ["xgboost", "lightgbm", "gradient_boosting"],
          "primary": "gradient_boosting",
          "primary_metric": "auc",
          "excluded": [],
          "reasoning": "Test",
          "notes": null
        }"""

        with (
            patch("backend.agents.model_selector_agent.call_claude", return_value=response_with_gb),
            patch("backend.agents.model_selector_agent.audit") as mock_audit,
        ):
            mock_audit.append = AsyncMock()

            strategy = await run_model_selector_agent(
                session=mock_session,
                run_id="run-sel-004",
                compressed_profile={},
                eda_report={},
                task_type="binary_classification",
                n_rows=150_000,
                emitter=mock_emitter,
            )

        # gradient_boosting should have been filtered from candidates
        assert "gradient_boosting" not in strategy.candidates


class TestSafeFallback:
    def test_xgboost_preferred_for_classification(self):
        strategy = _safe_fallback(
            ["xgboost", "lightgbm", "random_forest"],
            "binary_classification",
            [],
        )
        assert strategy.primary == "xgboost"
        assert strategy.primary_metric == "auc"

    def test_ridge_preferred_for_regression(self):
        strategy = _safe_fallback(
            ["ridge_regression", "linear_regression", "random_forest_regressor"],
            "regression",
            [],
        )
        assert strategy.primary == "ridge_regression"
        assert strategy.primary_metric == "rmse"
