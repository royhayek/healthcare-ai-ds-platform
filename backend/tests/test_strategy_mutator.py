"""Tests for backend/core/strategy_mutator.py - pure logic, no DB required.

Tests verify:
- mutators produce correct StrategyDiff values
- no-op inputs return empty lists
- missing strategy returns empty list gracefully
- apply_intent_to_strategy dispatches correctly and commits atomically
- unrecognized category returns empty list
"""

from copy import deepcopy
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.core.strategy_mutator import (
    _model_selection_mutator,
    _preprocessing_mutator,
    _threshold_mutator,
    apply_intent_to_strategy,
)
from backend.models.chat import ChatIntent, StrategyDiff


def _make_run(preprocessing_strategy=None, model_selection=None, threshold_config=None):
    run = MagicMock()
    run.id = "run-test-1"
    run.preprocessing_strategy = preprocessing_strategy
    run.model_selection = model_selection
    run.threshold_config = threshold_config
    return run


def _make_session():
    cm = AsyncMock()
    cm.__aenter__ = AsyncMock(return_value=None)
    cm.__aexit__ = AsyncMock(return_value=False)
    session = MagicMock()
    session.begin.return_value = cm
    session.commit = AsyncMock()
    return session


def _make_modify_intent(category: str, payload: dict) -> ChatIntent:
    return ChatIntent(
        intent="modify",
        confidence=0.9,
        category=category,
        structured_payload=payload,
        needs_confirmation=False,
        reasoning="test",
    )


# ── preprocessing mutator ──────────────────────────────────────────────────────


async def test_preprocessing_mutator_changes_field():
    run = _make_run(preprocessing_strategy={
        "columns": {"age": {"imputation_strategy": "mean", "scaling": "standard"}}
    })
    session = _make_session()
    payload = {"column": "age", "field": "imputation_strategy", "value": "median"}
    diffs = await _preprocessing_mutator(session, run, payload)

    assert len(diffs) == 1
    assert diffs[0].field_path == "preprocessing.columns.age.imputation_strategy"
    assert diffs[0].before == "mean"
    assert diffs[0].after == "median"
    assert run.preprocessing_strategy["columns"]["age"]["imputation_strategy"] == "median"


async def test_preprocessing_mutator_noop_same_value():
    run = _make_run(preprocessing_strategy={
        "columns": {"age": {"imputation_strategy": "median"}}
    })
    diffs = await _preprocessing_mutator(_make_session(), run,
                                          {"column": "age", "field": "imputation_strategy", "value": "median"})
    assert diffs == []


async def test_preprocessing_mutator_missing_strategy():
    run = _make_run(preprocessing_strategy=None)
    diffs = await _preprocessing_mutator(_make_session(), run, {"column": "age", "field": "x", "value": "y"})
    assert diffs == []


async def test_preprocessing_mutator_unknown_column():
    run = _make_run(preprocessing_strategy={"columns": {"income": {}}})
    diffs = await _preprocessing_mutator(_make_session(), run,
                                          {"column": "age", "field": "x", "value": "y"})
    assert diffs == []


async def test_preprocessing_mutator_incomplete_payload():
    run = _make_run(preprocessing_strategy={"columns": {"age": {}}})
    diffs = await _preprocessing_mutator(_make_session(), run, {"column": "age"})
    assert diffs == []


# ── model_selection mutator ────────────────────────────────────────────────────


async def test_model_selection_mutator_changes_primary():
    run = _make_run(model_selection={"primary": "gradient_boosting", "candidates": []})
    diffs = await _model_selection_mutator(_make_session(), run, {"model": "random_forest"})
    assert len(diffs) == 1
    assert diffs[0].before == "gradient_boosting"
    assert diffs[0].after == "random_forest"
    assert run.model_selection["primary"] == "random_forest"


async def test_model_selection_mutator_noop():
    run = _make_run(model_selection={"primary": "random_forest"})
    diffs = await _model_selection_mutator(_make_session(), run, {"model": "random_forest"})
    assert diffs == []


async def test_model_selection_mutator_missing_strategy():
    run = _make_run(model_selection=None)
    diffs = await _model_selection_mutator(_make_session(), run, {"model": "random_forest"})
    assert diffs == []


# ── threshold mutator ──────────────────────────────────────────────────────────


async def test_threshold_mutator_sets_override():
    run = _make_run(threshold_config={"optimal": 0.42, "override_threshold": None})
    diffs = await _threshold_mutator(_make_session(), run, {"threshold": 0.35})
    assert len(diffs) == 1
    assert diffs[0].after == 0.35
    assert run.threshold_config["override_threshold"] == 0.35


async def test_threshold_mutator_missing_config():
    run = _make_run(threshold_config=None)
    diffs = await _threshold_mutator(_make_session(), run, {"threshold": 0.35})
    assert diffs == []


# ── apply_intent_to_strategy ───────────────────────────────────────────────────


async def test_apply_intent_dispatches_preprocessing():
    run = _make_run(preprocessing_strategy={"columns": {"income": {"scaling": "minmax"}}})
    session = _make_session()
    intent = _make_modify_intent("preprocessing",
                                  {"column": "income", "field": "scaling", "value": "standard"})

    with patch("backend.core.strategy_mutator.audit.append", new=AsyncMock()):
        diffs = await apply_intent_to_strategy(session, run, intent)

    assert len(diffs) == 1
    assert diffs[0].field_path == "preprocessing.columns.income.scaling"


async def test_apply_intent_unknown_category_returns_empty():
    run = _make_run()
    intent = _make_modify_intent("unknown_category", {})
    diffs = await apply_intent_to_strategy(_make_session(), run, intent)
    assert diffs == []


async def test_apply_intent_writes_audit_on_change():
    run = _make_run(preprocessing_strategy={"columns": {"age": {"imputation_strategy": "mean"}}})
    session = _make_session()
    intent = _make_modify_intent("preprocessing",
                                  {"column": "age", "field": "imputation_strategy", "value": "median"})

    with patch("backend.core.strategy_mutator.audit.append", new=AsyncMock()) as mock_audit:
        await apply_intent_to_strategy(session, run, intent)

    mock_audit.assert_called_once()
    assert mock_audit.call_args[1]["actor"] == "user"
    assert mock_audit.call_args[1]["action"] == "strategy_override"


async def test_apply_intent_no_audit_on_noop():
    run = _make_run(preprocessing_strategy={"columns": {"age": {"imputation_strategy": "median"}}})
    session = _make_session()
    intent = _make_modify_intent("preprocessing",
                                  {"column": "age", "field": "imputation_strategy", "value": "median"})

    with patch("backend.core.strategy_mutator.audit.append", new=AsyncMock()) as mock_audit:
        diffs = await apply_intent_to_strategy(session, run, intent)

    assert diffs == []
    mock_audit.assert_not_called()
