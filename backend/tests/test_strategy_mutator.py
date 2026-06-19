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
    _fairness_mutator,
    _model_selection_mutator,
    _target_mutator,
    _threshold_mutator,
    apply_intent_to_strategy,
    canonical_category,
    extract_drop_columns,
    record_directive,
)
from backend.models.chat import ChatIntent, StrategyDiff


def _make_run(
    preprocessing_strategy=None,
    model_selection=None,
    threshold_config=None,
    fairness_config=None,
    user_directives=None,
    target_strategy=None,
):
    run = MagicMock()
    run.id = "run-test-1"
    run.preprocessing_strategy = preprocessing_strategy
    run.model_selection = model_selection
    run.threshold_config = threshold_config
    run.fairness_config = fairness_config
    run.user_directives = user_directives
    run.target_strategy = target_strategy
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


# ── threshold mutator: direct threshold + cost matrix ──────────────────────────


async def test_threshold_mutator_cost_matrix_creates_config():
    # Config absent: a cost-matrix override must still apply (created on the fly).
    run = _make_run(threshold_config=None)
    diffs = await _threshold_mutator(_make_session(), run, {"cost_matrix": {"cost_fn": 10, "cost_fp": 1}})
    assert len(diffs) == 1
    assert run.threshold_config["cost_matrix"] == {"cost_fn": 10.0, "cost_fp": 1.0}


async def test_threshold_mutator_flat_cost_keys():
    run = _make_run(threshold_config={})
    diffs = await _threshold_mutator(_make_session(), run, {"cost_fn": 5})
    assert len(diffs) == 1
    assert run.threshold_config["cost_matrix"]["cost_fn"] == 5.0


async def test_threshold_mutator_direct_threshold():
    run = _make_run(threshold_config={})
    diffs = await _threshold_mutator(_make_session(), run, {"threshold": 0.3})
    assert run.threshold_config["override_threshold"] == 0.3
    assert diffs[0].after == 0.3


# ── fairness mutator: protected attributes ─────────────────────────────────────


async def test_fairness_mutator_adds_column():
    run = _make_run(fairness_config={"protected_columns": ["sex"]})
    diffs = await _fairness_mutator(_make_session(), run, {"column": "age_group"})
    assert len(diffs) == 1
    assert run.fairness_config["protected_columns"] == ["sex", "age_group"]


async def test_fairness_mutator_creates_config_from_list():
    run = _make_run(fairness_config=None)
    diffs = await _fairness_mutator(_make_session(), run, {"protected_columns": ["sex", "race"]})
    assert len(diffs) == 1
    assert run.fairness_config["protected_columns"] == ["sex", "race"]


async def test_fairness_mutator_noop_already_protected():
    run = _make_run(fairness_config={"protected_columns": ["sex"]})
    diffs = await _fairness_mutator(_make_session(), run, {"value": "sex"})
    assert diffs == []


# ── target mutator: drop labels + binary collapse ──────────────────────────────


async def test_target_mutator_adds_drop_labels():
    run = _make_run(target_strategy=None)
    diffs = await _target_mutator(_make_session(), run, {"drop_labels": ["unknown"]})
    assert len(diffs) == 1
    assert diffs[0].field_path == "target_strategy.drop_labels"
    assert run.target_strategy["drop_labels"] == ["unknown"]


async def test_target_mutator_sets_positive_labels():
    run = _make_run(target_strategy={"drop_labels": ["unknown"], "positive_labels": []})
    diffs = await _target_mutator(_make_session(), run, {"positive_labels": ["high"]})
    assert len(diffs) == 1
    assert diffs[0].field_path == "target_strategy.positive_labels"
    assert run.target_strategy["positive_labels"] == ["high"]
    assert run.target_strategy["drop_labels"] == ["unknown"]  # preserved


async def test_target_mutator_unions_drop_labels():
    run = _make_run(target_strategy={"drop_labels": ["unknown"]})
    await _target_mutator(_make_session(), run, {"drop_labels": ["pending", "unknown"]})
    assert run.target_strategy["drop_labels"] == ["unknown", "pending"]


async def test_target_mutator_noop_when_unchanged():
    run = _make_run(target_strategy={"drop_labels": ["unknown"], "positive_labels": []})
    diffs = await _target_mutator(_make_session(), run, {"drop_labels": ["unknown"]})
    assert diffs == []


async def test_apply_intent_routes_target_query_to_target():
    run = _make_run(target_strategy=None)
    intent = _make_modify_intent("target_query", {"positive_labels": ["high"]})
    with patch("backend.core.strategy_mutator.audit.append", new=AsyncMock()):
        diffs = await apply_intent_to_strategy(_make_session(), run, intent)
    assert len(diffs) == 1
    assert run.target_strategy["positive_labels"] == ["high"]


# ── canonical category routing ─────────────────────────────────────────────────


def test_canonical_category_maps_query_forms():
    assert canonical_category("equity_query") == "fairness"
    assert canonical_category("threshold_query") == "threshold"
    assert canonical_category("model_selection") == "model_selection"
    assert canonical_category("preprocessing") == "preprocessing"


async def test_apply_intent_routes_equity_query_to_fairness():
    # A modify mislabeled "equity_query" must still reach the fairness mutator.
    run = _make_run(fairness_config={"protected_columns": []})
    intent = _make_modify_intent("equity_query", {"column": "age_group"})
    with patch("backend.core.strategy_mutator.audit.append", new=AsyncMock()):
        diffs = await apply_intent_to_strategy(_make_session(), run, intent)
    assert len(diffs) == 1
    assert run.fairness_config["protected_columns"] == ["age_group"]


# ── preprocessing overrides → verbatim directives (not field edits) ──────────────


def test_extract_drop_columns_explicit_list():
    cols = extract_drop_columns({"columns_to_drop": ["clade", "clinical_syndrome"]})
    assert cols == ["clade", "clinical_syndrome"]


def test_extract_drop_columns_from_columns_action_drop():
    cols = extract_drop_columns(
        {"columns": ["clade", "clinical_syndrome"], "field": "action", "value": "drop"}
    )
    assert cols == ["clade", "clinical_syndrome"]


def test_extract_drop_columns_single_legacy():
    assert extract_drop_columns(
        {"column": "patient_id", "field": "action", "value": "drop"}
    ) == ["patient_id"]


def test_extract_drop_columns_dedupes_and_ignores_non_drop():
    # A non-drop edit (impute) yields no drop columns from column/columns keys.
    assert extract_drop_columns(
        {"column": "age", "field": "impute_strategy", "value": "median"}
    ) == []
    # Explicit hint + overlapping columns list de-duplicates, order preserved.
    cols = extract_drop_columns(
        {"columns_to_drop": ["clade"], "columns": ["clade", "clinical_syndrome"],
         "field": "action", "value": "drop"}
    )
    assert cols == ["clade", "clinical_syndrome"]


async def test_record_directive_appends_and_audits():
    # Run carries the agent's strategy listing both columns as kept features.
    run = _make_run(
        preprocessing_strategy={"columns": {
            "clinical_syndrome": {"action": "keep"}, "clade": {"action": "keep"}
        }},
        user_directives=None,
    )
    session = _make_session()
    intent = _make_modify_intent(
        "preprocessing",
        {"instruction": "drop both clinical_syndrome and clade - they're relabeled targets",
         "columns_to_drop": ["clinical_syndrome", "clade"]},
    )
    with patch("backend.core.strategy_mutator.audit.append", new=AsyncMock()) as mock_audit:
        diffs = await record_directive(session, run, intent)

    # One per-column "keep → drop" card per column, matching the demo.
    assert len(diffs) == 2
    assert {d.field_path for d in diffs} == {
        "preprocessing.columns.clinical_syndrome.action",
        "preprocessing.columns.clade.action",
    }
    assert all(d.before == "keep" and d.after == "drop" for d in diffs)
    recorded = run.user_directives["preprocessing"]
    assert len(recorded) == 1
    assert recorded[0]["columns_to_drop"] == ["clinical_syndrome", "clade"]
    assert "relabeled targets" in recorded[0]["instruction"]
    assert mock_audit.call_args.kwargs["action"] == "directive_recorded"


async def test_record_directive_accumulates_across_overrides():
    run = _make_run(user_directives={"preprocessing": [
        {"category": "preprocessing", "instruction": "drop clade",
         "columns_to_drop": ["clade"], "created_at": "2026-06-19T00:00:00"}
    ]})
    intent = _make_modify_intent("preprocessing", {"instruction": "also impute age with median"})
    with patch("backend.core.strategy_mutator.audit.append", new=AsyncMock()):
        await record_directive(_make_session(), run, intent)
    assert len(run.user_directives["preprocessing"]) == 2


# ── model_selection mutator ────────────────────────────────────────────────────


async def test_model_selection_mutator_changes_primary():
    run = _make_run(model_selection={"primary": "gradient_boosting", "candidates": []})
    diffs = await _model_selection_mutator(_make_session(), run, {"model": "random_forest"})
    assert len(diffs) == 1
    assert diffs[0].before == "gradient_boosting"
    assert diffs[0].after == "random_forest"
    assert run.model_selection["primary"] == "random_forest"
    # Override is marked authoritative and the chosen model is added as a candidate
    # so training actually evaluates it.
    assert run.model_selection["primary_source"] == "user_override"
    assert "random_forest" in run.model_selection["candidates"]


async def test_model_selection_mutator_keeps_existing_candidates():
    run = _make_run(model_selection={
        "primary": "lightgbm",
        "candidates": ["lightgbm", "xgboost", "logistic_regression"],
    })
    await _model_selection_mutator(_make_session(), run, {"model": "logistic_regression"})
    # logistic_regression already a candidate - not duplicated
    assert run.model_selection["candidates"].count("logistic_regression") == 1
    assert run.model_selection["primary"] == "logistic_regression"
    assert run.model_selection["primary_source"] == "user_override"


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


async def test_threshold_mutator_creates_config_when_missing():
    # A user may set the threshold before the threshold step has run; the config
    # is created on the fly rather than silently ignored.
    run = _make_run(threshold_config=None)
    diffs = await _threshold_mutator(_make_session(), run, {"threshold": 0.35})
    assert len(diffs) == 1
    assert run.threshold_config["override_threshold"] == 0.35


async def test_threshold_mutator_noop_empty_payload():
    run = _make_run(threshold_config={"override_threshold": 0.4})
    diffs = await _threshold_mutator(_make_session(), run, {})
    assert diffs == []


# ── apply_intent_to_strategy ───────────────────────────────────────────────────


async def test_apply_intent_routes_preprocessing_to_directive():
    # A preprocessing override is recorded as a verbatim directive (replayed to the
    # agent on re-run), NOT applied as a deterministic field edit.
    run = _make_run(
        preprocessing_strategy={"columns": {
            "clade": {"action": "keep"}, "clinical_syndrome": {"action": "keep"}
        }},
        user_directives=None,
    )
    session = _make_session()
    intent = _make_modify_intent(
        "preprocessing",
        {"instruction": "drop both clade and clinical_syndrome",
         "columns_to_drop": ["clade", "clinical_syndrome"]},
    )

    with patch("backend.core.strategy_mutator.audit.append", new=AsyncMock()):
        diffs = await apply_intent_to_strategy(session, run, intent)

    # Override is recorded as a directive AND previewed as per-column drop cards.
    assert {d.field_path for d in diffs} == {
        "preprocessing.columns.clade.action",
        "preprocessing.columns.clinical_syndrome.action",
    }
    assert run.user_directives["preprocessing"][0]["columns_to_drop"] == [
        "clade", "clinical_syndrome"
    ]


async def test_apply_intent_unknown_category_returns_empty():
    run = _make_run()
    intent = _make_modify_intent("unknown_category", {})
    diffs = await apply_intent_to_strategy(_make_session(), run, intent)
    assert diffs == []


async def test_apply_intent_writes_audit_on_change():
    # A model_selection override is a deterministic field edit and audits as a
    # strategy_override.
    run = _make_run(model_selection={"primary": "lightgbm", "candidates": ["lightgbm"]})
    session = _make_session()
    intent = _make_modify_intent("model_selection", {"model": "logistic_regression"})

    with patch("backend.core.strategy_mutator.audit.append", new=AsyncMock()) as mock_audit:
        await apply_intent_to_strategy(session, run, intent)

    mock_audit.assert_called_once()
    assert mock_audit.call_args[1]["actor"] == "user"
    assert mock_audit.call_args[1]["action"] == "strategy_override"


async def test_apply_intent_no_audit_on_noop():
    run = _make_run(model_selection={"primary": "random_forest"})
    session = _make_session()
    intent = _make_modify_intent("model_selection", {"model": "random_forest"})

    with patch("backend.core.strategy_mutator.audit.append", new=AsyncMock()) as mock_audit:
        diffs = await apply_intent_to_strategy(session, run, intent)

    assert diffs == []
    mock_audit.assert_not_called()
