"""Unit tests for agents/brief_parser_agent.py - target_strategy extraction."""

from unittest.mock import patch

import pytest

from backend.agents.brief_parser_agent import _coerce_target_strategy, parse_case_brief


def test_coerce_target_strategy_extracts_both():
    out = _coerce_target_strategy(
        {"drop_labels": ["unknown"], "positive_labels": ["high"]}
    )
    assert out == {"drop_labels": ["unknown"], "positive_labels": ["high"]}


def test_coerce_target_strategy_none_when_empty():
    assert _coerce_target_strategy({"drop_labels": [], "positive_labels": []}) is None
    assert _coerce_target_strategy(None) is None
    assert _coerce_target_strategy("nonsense") is None


def test_coerce_target_strategy_drop_only():
    assert _coerce_target_strategy({"drop_labels": ["unknown", "pending"]}) == {
        "drop_labels": ["unknown", "pending"],
        "positive_labels": [],
    }


_BRIEF_JSON = """{
  "objectives": ["flag high-pathogenicity strains"],
  "cost_matrix": null,
  "known_data_issues": ["clade and clinical_syndrome leak the label"],
  "deliverable_requirements": [],
  "evaluation_criteria": [],
  "target_strategy": {"drop_labels": ["unknown"], "positive_labels": ["high"]},
  "stakeholder_name": null,
  "stakeholder_role": null
}"""


@pytest.mark.asyncio
async def test_parse_case_brief_includes_target_strategy():
    with patch("backend.agents.brief_parser_agent.call_claude", return_value=_BRIEF_JSON):
        parsed = await parse_case_brief("some brief text")
    assert parsed["parsed"] is True
    assert parsed["target_strategy"] == {
        "drop_labels": ["unknown"],
        "positive_labels": ["high"],
    }


@pytest.mark.asyncio
async def test_parse_case_brief_empty_has_target_strategy_key():
    # Refusal / empty text path must still return the key (None), not KeyError.
    parsed = await parse_case_brief("")
    assert parsed["target_strategy"] is None
    assert parsed["parsed"] is False
