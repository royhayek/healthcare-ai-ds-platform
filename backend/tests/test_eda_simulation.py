"""Simulate the full EDA agent step without calling the model API.

Run with:
    cd /path/to/AI_DS
    python -m pytest backend/tests/test_eda_simulation.py -v
    # or directly:
    python backend/tests/test_eda_simulation.py

Tests three realistic model response shapes observed in production:
  A) Well-formed JSON - exactly the schema we asked for
  B) Alt field names - the model uses "action"/"description" instead of "strategy"/"reason"
  C) model_recommendation as dict - the model returns {"model": "GradientBoosting", ...}
"""

import asyncio
import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch

# Make sure project root is on path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# ── Realistic mock responses ───────────────────────────────────────────────────

# Shape A: exactly the schema we documented (ideal case)
_RESPONSE_A = json.dumps({
    "overview": "Telco churn dataset - 7043 rows, 21 features. "
                "Binary classification with 26.5% positive rate (churners). "
                "TotalCharges stored as object due to whitespace values.",
    "target_analysis": {
        "column": "Churn",
        "task_type": "binary_classification",
        "class_balance": {"No": 0.735, "Yes": 0.265},
        "notes": "Class imbalance ratio 2.77× - recommend class_weight or SMOTE.",
    },
    "quality_issues": [
        {
            "column": "TotalCharges",
            "issue": "dtype=object - 11 rows contain whitespace, should be float64",
            "severity": "high",
            "recommendation": "coerce to numeric with pd.to_numeric(errors='coerce'), then median-impute NaN",
        },
        {
            "column": "customerID",
            "issue": "high cardinality (n_unique == n_rows) - pure identifier",
            "severity": "medium",
            "recommendation": "drop before modeling - zero predictive value",
        },
        {
            "column": "SeniorCitizen",
            "issue": "encoded as int 0/1 rather than boolean categorical",
            "severity": "low",
            "recommendation": "leave as-is or convert to bool; already binary",
        },
    ],
    "correlations": {
        "high_pairs": [
            {"col_a": "TotalCharges", "col_b": "tenure", "r": 0.826},
            {"col_a": "TotalCharges", "col_b": "MonthlyCharges", "r": 0.651},
        ],
        "leakage_risk": [],
    },
    "preprocessing_recommendations": [
        {"column": "customerID",  "strategy": "drop",           "reason": "identifier - n_unique == n_rows"},
        {"column": "TotalCharges","strategy": "median_impute",  "reason": "1.6% nulls after dtype fix; right-skewed"},
        {"column": "SeniorCitizen","strategy":"binary_encode",  "reason": "already 0/1; keep as numeric"},
        {"column": "tenure",      "strategy": "standard_scale","reason": "numeric, right-skewed, no outliers"},
        {"column": "Churn",       "strategy": "label_encode",  "reason": "target - encode Yes=1 No=0"},
    ],
    "model_recommendation": "gradient_boosting",
    "summary": "Dataset is suitable for binary churn prediction. "
               "Fix TotalCharges dtype first; drop customerID. "
               "26.5% positive rate requires imbalance handling.",
})

# Shape B: alt field names - the model uses "action"/"description" instead of "strategy"/"reason"
# and omits "severity"/"recommendation" from quality_issues
_RESPONSE_B = json.dumps({
    "overview": "Telco churn dataset - 7043 rows, 21 features.",
    "target_analysis": {"column": "Churn", "task_type": "binary_classification", "class_balance": {}},
    "quality_issues": [
        # no "severity" or "recommendation" keys - just column + issue
        {"column": "TotalCharges", "issue": "whitespace treated as NaN."},
        {"column": "customerID",   "issue": "high cardinality - should be dropped before modeling."},
    ],
    "correlations": {},
    "preprocessing_recommendations": [
        # uses "action" instead of "strategy", "rationale" instead of "reason"
        {"column": "customerID",  "action": "drop",          "rationale": "pure ID (n_unique = n_rows)."},
        {"column": "TotalCharges","action": "median_impute", "rationale": "1.6% nulls after coerce."},
        {"column": "tenure",      "recommendation": "standard_scale", "description": "numeric, no outliers."},
    ],
    "model_recommendation": "gradient_boosting",
    "summary": "Fix dtype issues; drop customerID.",
})

# Shape C: model_recommendation returned as dict
_RESPONSE_C = json.dumps({
    "overview": "Telco churn dataset.",
    "target_analysis": {"column": "Churn", "task_type": "binary_classification"},
    "quality_issues": [
        {"column": "TotalCharges", "issue": "dtype object.", "severity": "high",
         "recommendation": "coerce to float"},
    ],
    "correlations": {},
    "preprocessing_recommendations": [
        {"column": "customerID", "strategy": "drop", "reason": "identifier"},
    ],
    "model_recommendation": {
        "model": "GradientBoosting",
        "justification": "handles mixed types well; interpretable for customer-facing explanations.",
    },
    "summary": "Binary churn prediction dataset.",
})


# ── Helpers ────────────────────────────────────────────────────────────────────

class _FakeEmitter:
    async def emit_async(self, *args, **kwargs): pass


class _FakeSession:
    """Minimal async session stub - audit.append needs execute + add + commit."""
    def add(self, obj): pass
    async def commit(self): pass
    async def execute(self, stmt):
        class _R:
            def scalars(self): return self
            def all(self): return []
            def scalar_one_or_none(self): return None
        return _R()


async def _run(mock_response: str, label: str) -> None:
    from backend.agents.eda_agent import run_eda_agent
    from backend.models.eda import EdaReport

    session = _FakeSession()
    emitter = _FakeEmitter()

    with patch("backend.agents.eda_agent.call_claude_stream", new=AsyncMock(return_value=mock_response)):
        with patch("backend.core.audit.append", new=AsyncMock()):
            report = await run_eda_agent(
                session=session,
                run_id="test-run-id",
                compressed_profile={"target_column": "Churn", "task_type": "binary_classification"},
                emitter=emitter,
            )

    assert isinstance(report, EdaReport), f"{label}: expected EdaReport, got {type(report)}"

    # Must NOT be the fallback (which sets quality_issues=[])
    is_fallback = report.overview.startswith("EDA completed with parse error") or report.overview.startswith("EDA parse failed")
    if is_fallback:
        print(f"  FAIL [{label}]: got fallback - overview: {report.overview[:120]}")
        sys.exit(1)

    print(f"  PASS [{label}]")
    print(f"    overview          : {report.overview[:80]}…")
    print(f"    quality_issues    : {len(report.quality_issues)} items")
    if report.quality_issues:
        q = report.quality_issues[0]
        print(f"      [0] severity={q.severity!r}  recommendation={q.recommendation[:40]!r}")
    print(f"    preprocessing_recs: {len(report.preprocessing_recommendations)} items")
    if report.preprocessing_recommendations:
        p = report.preprocessing_recommendations[0]
        print(f"      [0] strategy={p.strategy!r}  reason={p.reason[:40]!r}")
    print(f"    model_recommendation: {report.model_recommendation!r}")
    print()


def main() -> None:
    print("=== EDA agent simulation (no model API) ===\n")
    asyncio.run(_run(_RESPONSE_A, "Shape A - well-formed JSON"))
    asyncio.run(_run(_RESPONSE_B, "Shape B - alt field names (action/rationale/recommendation)"))
    asyncio.run(_run(_RESPONSE_C, "Shape C - model_recommendation as dict"))
    print("All simulations passed.")


if __name__ == "__main__":
    main()
