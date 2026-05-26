"""Tests for the eight deliverable generators (§4, §23).

These tests work without a running DB, Celery, or model API:
- Generators are called with fake run/dataset stubs
- the model calls are patched to return valid structured responses
- Storage upload/download is patched to use in-memory dicts
- PDF rendering falls back to HTML bytes (weasyprint not required)
"""

from __future__ import annotations

import json
import pickle
from io import BytesIO
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

# ── Fixtures ──────────────────────────────────────────────────────────────────


def _make_run(run_id: str = "run-test-001") -> MagicMock:
    run = MagicMock()
    run.id = run_id
    run.project_id = "proj-001"
    run.created_at = datetime(2024, 1, 15, 12, 0, tzinfo=timezone.utc)
    run.completed_at = datetime(2024, 1, 15, 13, 0, tzinfo=timezone.utc)
    run.created_by = "test-user"
    run.best_model_name = "XGBClassifier"
    run.best_model_score = 0.87
    run.status = "completed"
    run.model_storage_path = f"runs/{run_id}/model.joblib"
    run.faiss_index_path = f"runs/{run_id}/similarity.index"
    run.threshold_config = {"optimal_threshold": 0.42}
    run.preprocessing_strategy = {"task_type": "binary_classification"}
    run.model_selection = {"candidates": ["XGBClassifier", "LGBMClassifier"]}
    run.model_comparison = [
        {"model_name": "XGBClassifier", "mean": 0.87, "std": 0.02},
        {"model_name": "LGBMClassifier", "mean": 0.85, "std": 0.03},
    ]
    run.stat_tests = {"significant": True}
    run.tuning_result = {"n_trials": 50, "best_value": 0.87}
    run.final_metrics = {"auc": 0.87, "f1": 0.76, "accuracy": 0.83}
    run.calibration_report = {
        "method": "isotonic",
        "brier_before": 0.14,
        "brier_after": 0.11,
        "improvement_pct": 21.4,
    }
    run.threshold_result = {
        "optimal_threshold": 0.42,
        "cost_at_default": 1200,
        "cost_at_optimal": 950,
        "improvement_pct": 20.8,
    }
    run.shap_summary = {
        "top_k_features": ["age", "balance", "duration", "campaign", "pdays"],
        "mean_abs_shap": [0.32, 0.21, 0.18, 0.12, 0.09],
        "feature_names": ["age", "balance", "duration", "campaign", "pdays"],
        "explainer_type": "TreeExplainer",
    }
    run.eda_report = {"summary": "Dataset has 7032 rows and 20 columns.", "quality_issues": []}
    run.drift_report = {
        "overall_severity": "mild",
        "aggregate_psi": 0.14,
        "significant_features": ["balance"],
        "n_features_drifted": 1,
    }
    run.fairness_report = {
        "overall_severity": "mild",
        "blocks_deliverables": False,
        "attributes": [{"attribute": "gender", "severity": "mild"}],
    }
    run.seeds = {"global": 42}
    run.claude_models_used = {"eda": "claude-sonnet-4-6"}
    run.library_versions = {"python": "3.11.0", "scikit-learn": "1.3.0"}
    run.fairness_config = {"protected_columns": ["gender"]}
    run.similarity_index_built = True
    return run


def _make_dataset() -> MagicMock:
    ds = MagicMock()
    ds.id = "ds-001"
    ds.filename = "telco_churn.csv"
    ds.sha256 = "abc123def456abc123def456abc123def456"
    ds.schema_hash = "schema_hash_001"
    ds.row_count = 7032
    ds.col_count = 20
    ds.target_column = "Churn"
    ds.storage_path = "datasets/ds-001/telco_churn.csv"
    ds.profile = {
        "columns": {
            "age": {"dtype": "int64", "missing_pct": 0.0, "has_outliers": False},
            "balance": {"dtype": "float64", "missing_pct": 0.01, "has_outliers": True},
        }
    }
    return ds


def _make_ctx(run: MagicMock, dataset: MagicMock) -> dict:
    from backend.deliverables.base import run_summary_context
    return run_summary_context(run, dataset)


# ── base.py ───────────────────────────────────────────────────────────────────


class TestRunSummaryContext:
    def test_produces_dict_with_required_keys(self):
        run = _make_run()
        dataset = _make_dataset()
        from backend.deliverables.base import run_summary_context
        ctx = run_summary_context(run, dataset)
        assert "run_id" in ctx
        assert "model_name" in ctx
        assert "final_metrics" in ctx
        assert "dataset" in ctx
        assert ctx["dataset"]["row_count"] == 7032
        assert ctx["model_name"] == "XGBClassifier"

    def test_no_raw_row_data(self):
        run = _make_run()
        dataset = _make_dataset()
        from backend.deliverables.base import run_summary_context
        ctx = run_summary_context(run, dataset)
        ctx_str = json.dumps(ctx)
        # Must not contain raw row-level data - only aggregates
        assert "profile" not in ctx_str or "columns" not in ctx_str

    def test_handles_none_dataset(self):
        run = _make_run()
        from backend.deliverables.base import run_summary_context
        ctx = run_summary_context(run, None)
        assert ctx["dataset"]["filename"] == "unknown"
        assert ctx["dataset"]["row_count"] is None

    def test_generated_deliverable_build(self):
        from backend.deliverables.base import GeneratedDeliverable
        content = b"hello world"
        d = GeneratedDeliverable.build(
            name="test_doc",
            fmt="pdf",
            content=content,
            run_id="run-abc",
        )
        assert d.storage_path == "runs/run-abc/deliverables/test_doc.pdf"
        assert len(d.checksum_sha256) == 64
        assert d.fmt == "pdf"


# ── repro_manifest.py ─────────────────────────────────────────────────────────


class TestReproManifest:
    @pytest.mark.asyncio
    async def test_generates_yaml_deliverable(self):
        run = _make_run()
        dataset = _make_dataset()
        ctx = _make_ctx(run, dataset)
        session = AsyncMock()

        from backend.deliverables.repro_manifest import generate_repro_manifest
        d = await generate_repro_manifest(run, dataset, session, ctx)

        assert d.name == "repro_manifest"
        assert d.fmt == "yaml"
        assert len(d.content) > 0
        content_str = d.content.decode()
        assert "run_id" in content_str or "run-test-001" in content_str

    @pytest.mark.asyncio
    async def test_includes_library_versions(self):
        run = _make_run()
        dataset = _make_dataset()
        ctx = _make_ctx(run, dataset)
        session = AsyncMock()

        from backend.deliverables.repro_manifest import generate_repro_manifest
        d = await generate_repro_manifest(run, dataset, session, ctx)
        content_str = d.content.decode()
        assert "python" in content_str

    @pytest.mark.asyncio
    async def test_strategy_hash_present(self):
        run = _make_run()
        dataset = _make_dataset()
        ctx = _make_ctx(run, dataset)
        session = AsyncMock()

        from backend.deliverables.repro_manifest import generate_repro_manifest
        d = await generate_repro_manifest(run, dataset, session, ctx)
        content_str = d.content.decode()
        assert "sha256" in content_str or "preprocessing_strategy" in content_str


# ── risk_register.py ──────────────────────────────────────────────────────────


class TestRiskRegister:
    @pytest.mark.asyncio
    async def test_generates_markdown_deliverable(self):
        run = _make_run()
        ctx = _make_ctx(run, _make_dataset())
        session = AsyncMock()

        with patch("backend.agents.base.call_claude", new_callable=AsyncMock) as mock_claude:
            mock_claude.return_value = """# Model Risk Register - XGBClassifier

## 1. Known Limitations
- Sample size is sufficient (7032 rows).

## 2. Drift Triggers
- Retrain when PSI > 0.25.

## 3. Fairness Concerns
- Mild disparity on gender attribute.

## 4. Edge Cases Not Validated
- Extreme outliers in balance column.

## 5. Monitoring Cadence
- Weekly PSI checks.

## 6. Retraining Triggers
- PSI > 0.25 on top features.

## 7. Sign-off Requirements
- DS lead approval required."""

            from backend.deliverables.risk_register import generate_risk_register
            d = await generate_risk_register(run, session, ctx)

        assert d.name == "risk_register"
        assert d.fmt == "md"
        content = d.content.decode()
        assert "Risk Register" in content

    @pytest.mark.asyncio
    async def test_falls_back_to_stub_on_claude_failure(self):
        run = _make_run()
        ctx = _make_ctx(run, _make_dataset())
        session = AsyncMock()

        with patch("backend.agents.base.call_claude", new_callable=AsyncMock) as mock_claude:
            mock_claude.side_effect = RuntimeError("API error")

            from backend.deliverables.risk_register import generate_risk_register
            d = await generate_risk_register(run, session, ctx)

        assert d.fmt == "md"
        content = d.content.decode()
        assert "Risk Register" in content
        assert "XGBClassifier" in content


# ── executive_summary.py ──────────────────────────────────────────────────────


class TestExecutiveSummary:
    @pytest.mark.asyncio
    async def test_generates_pdf_deliverable(self):
        run = _make_run()
        dataset = _make_dataset()
        ctx = _make_ctx(run, dataset)
        session = AsyncMock()

        exec_json = json.dumps({
            "one_liner": "XGBClassifier achieves AUC 0.87 on churn prediction.",
            "performance_paragraph": "AUC 0.87, F1 0.76. Optimal threshold 0.42 improves cost by 20.8%.",
            "business_paragraph": "20.8% cost reduction vs. default threshold. Identify top 30% churners.",
            "insights": ["Balance is top driver.", "Short duration raises churn risk.", "Recent contact reduces churn."],
            "risks": ["Mild drift on balance.", "Fairness gap on gender.", "Limited validation on edge cases."],
            "next_steps_paragraph": "Deploy with monitoring. Review fairness report before launch.",
        })

        with patch("backend.agents.base.call_claude", new_callable=AsyncMock) as mock_claude, \
             patch("backend.agents.base.extract_json") as mock_extract:
            mock_claude.return_value = exec_json
            mock_extract.return_value = json.loads(exec_json)

            from backend.deliverables.executive_summary import generate_executive_summary
            d = await generate_executive_summary(run, dataset, session, ctx)

        assert d.name == "executive_summary"
        assert d.fmt == "pdf"
        assert len(d.content) > 0

    @pytest.mark.asyncio
    async def test_falls_back_on_claude_failure(self):
        run = _make_run()
        dataset = _make_dataset()
        ctx = _make_ctx(run, dataset)
        session = AsyncMock()

        with patch("backend.agents.base.call_claude", new_callable=AsyncMock) as mock_claude:
            mock_claude.side_effect = RuntimeError("API timeout")

            from backend.deliverables.executive_summary import generate_executive_summary
            d = await generate_executive_summary(run, dataset, session, ctx)

        assert d.fmt == "pdf"
        assert len(d.content) > 0


# ── model_card.py ─────────────────────────────────────────────────────────────


class TestModelCard:
    @pytest.mark.asyncio
    async def test_returns_two_deliverables(self):
        run = _make_run()
        dataset = _make_dataset()
        ctx = _make_ctx(run, dataset)
        session = AsyncMock()

        card_json = json.dumps({
            "intended_use": "Predict customer churn for telecom.",
            "primary_intended_uses": "Monthly churn scoring for retention campaigns.",
            "out_of_scope_uses": "Not for credit scoring or insurance.",
            "relevant_factors": "Gender and age are protected attributes.",
            "evaluation_factors": "5-fold CV with 3 seeds. AUC primary metric.",
            "unitary_results": "AUC 0.87, F1 0.76, accuracy 0.83.",
            "intersectional_results": "Mild fairness disparity on gender (5-10% demographic parity diff).",
            "ethical_considerations": "Review gender disparity before production use.",
            "caveats": "Validated on 2024 data only. Retrain annually.",
        })

        with patch("backend.agents.base.call_claude", new_callable=AsyncMock) as mock_claude, \
             patch("backend.agents.base.extract_json") as mock_extract:
            mock_claude.return_value = card_json
            mock_extract.return_value = json.loads(card_json)

            from backend.deliverables.model_card import generate_model_card
            deliverables = await generate_model_card(run, dataset, session, ctx)

        assert len(deliverables) == 2
        formats = {d.fmt for d in deliverables}
        assert "md" in formats
        assert "pdf" in formats

    @pytest.mark.asyncio
    async def test_markdown_contains_required_sections(self):
        run = _make_run()
        dataset = _make_dataset()
        ctx = _make_ctx(run, dataset)
        session = AsyncMock()

        card_json = json.dumps({
            "intended_use": "Predict customer churn.",
            "primary_intended_uses": "Retention campaigns.",
            "out_of_scope_uses": "Not for credit scoring.",
            "relevant_factors": "Age and balance matter.",
            "evaluation_factors": "5-fold CV.",
            "unitary_results": "AUC 0.87.",
            "intersectional_results": "Fairness not analyzed.",
            "ethical_considerations": "Review before deployment.",
            "caveats": "Retrain annually.",
        })

        with patch("backend.agents.base.call_claude", new_callable=AsyncMock) as mock_claude, \
             patch("backend.agents.base.extract_json") as mock_extract:
            mock_claude.return_value = card_json
            mock_extract.return_value = json.loads(card_json)

            from backend.deliverables.model_card import generate_model_card
            deliverables = await generate_model_card(run, dataset, session, ctx)

        md_d = next(d for d in deliverables if d.fmt == "md")
        content = md_d.content.decode()
        assert "# Model Card" in content
        assert "Intended Use" in content
        assert "Fairness" in content
        assert "Performance Metrics" in content


# ── data_quality_report.py ────────────────────────────────────────────────────


class TestDataQualityReport:
    @pytest.mark.asyncio
    async def test_generates_pdf_deliverable(self):
        run = _make_run()
        dataset = _make_dataset()
        ctx = _make_ctx(run, dataset)
        session = AsyncMock()

        from backend.deliverables.data_quality_report import generate_data_quality_report
        d = await generate_data_quality_report(run, dataset, session, ctx)

        assert d.name == "data_quality_report"
        assert d.fmt == "pdf"
        assert len(d.content) > 0

    @pytest.mark.asyncio
    async def test_handles_none_dataset(self):
        run = _make_run()
        ctx = _make_ctx(run, None)
        session = AsyncMock()

        from backend.deliverables.data_quality_report import generate_data_quality_report
        d = await generate_data_quality_report(run, None, session, ctx)

        assert d.fmt == "pdf"


# ── technical_report.py ───────────────────────────────────────────────────────


class TestTechnicalReport:
    @pytest.mark.asyncio
    async def test_generates_pdf_deliverable(self):
        run = _make_run()
        dataset = _make_dataset()
        ctx = _make_ctx(run, dataset)
        session = AsyncMock()

        from backend.deliverables.technical_report import generate_technical_report
        d = await generate_technical_report(run, dataset, session, ctx)

        assert d.name == "technical_report"
        assert d.fmt == "pdf"
        assert len(d.content) > 0


# ── predictions_artifact.py ───────────────────────────────────────────────────


class TestPredictionsArtifact:
    @pytest.mark.asyncio
    async def test_returns_empty_artifact_when_model_unavailable(self):
        run = _make_run()
        dataset = _make_dataset()
        ctx = _make_ctx(run, dataset)
        session = AsyncMock()

        with patch("backend.core.storage.storage") as mock_storage:
            mock_storage.download = AsyncMock(side_effect=FileNotFoundError("not found"))

            from backend.deliverables.predictions_artifact import generate_predictions_artifact
            d = await generate_predictions_artifact(run, dataset, session, ctx)

        assert d.name == "predictions"
        # Empty artifact falls back to CSV bytes
        assert b"run_id" in d.content or b"error" in d.content

    @pytest.mark.asyncio
    async def test_generates_excel_with_real_data(self):
        import joblib
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline

        run = _make_run()
        dataset = _make_dataset()
        ctx = _make_ctx(run, dataset)
        session = AsyncMock()

        # Build a tiny real model
        X = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0], "b": [0.1, 0.2, 0.3, 0.4, 0.5]})
        y = pd.Series([0, 0, 1, 1, 1])
        clf = LogisticRegression().fit(X, y)

        model_buf = BytesIO()
        joblib.dump(clf, model_buf)
        model_bytes = model_buf.getvalue()

        splits = {"X_test": X, "y_test": y}
        splits_bytes = pickle.dumps(splits)

        run.faiss_index_path = None  # skip similarity

        async def mock_download(path: str) -> bytes:
            if "model" in path:
                return model_bytes
            if "splits" in path:
                return splits_bytes
            raise FileNotFoundError(path)

        with patch("backend.core.storage.storage") as mock_storage:
            mock_storage.download = AsyncMock(side_effect=mock_download)
            mock_storage.upload = AsyncMock()

            from backend.deliverables.predictions_artifact import generate_predictions_artifact
            d = await generate_predictions_artifact(run, dataset, session, ctx)

        assert d.name == "predictions"
        assert d.fmt == "xlsx"
        assert len(d.content) > 0


# ── audit_log_export.py ───────────────────────────────────────────────────────


class TestAuditLogExport:
    @pytest.mark.asyncio
    async def test_returns_two_deliverables(self):
        run = _make_run()
        ctx = _make_ctx(run, _make_dataset())
        session = AsyncMock()

        # Mock audit event query
        fake_events = [
            MagicMock(
                seq=1,
                timestamp=datetime(2024, 1, 15, 12, 0, tzinfo=timezone.utc),
                actor="system",
                category="pipeline",
                action="start",
                payload={"step": "eda"},
                reason="Pipeline started",
                prev_hash="GENESIS",
                self_hash="abc123",
            ),
            MagicMock(
                seq=2,
                timestamp=datetime(2024, 1, 15, 12, 5, tzinfo=timezone.utc),
                actor="ai",
                category="eda",
                action="eda_complete",
                payload={"rows": 7032},
                reason="EDA complete",
                prev_hash="abc123",
                self_hash="def456",
            ),
        ]

        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = fake_events
        session.execute = AsyncMock(return_value=mock_result)

        from backend.deliverables.audit_log_export import generate_audit_log_export
        deliverables = await generate_audit_log_export(run, session, ctx)

        assert len(deliverables) == 2
        formats = {d.fmt for d in deliverables}
        assert "csv" in formats
        assert "json" in formats

    @pytest.mark.asyncio
    async def test_csv_contains_audit_columns(self):
        run = _make_run()
        ctx = _make_ctx(run, _make_dataset())
        session = AsyncMock()

        fake_events = [
            MagicMock(
                seq=1,
                timestamp=datetime(2024, 1, 15, 12, 0, tzinfo=timezone.utc),
                actor="system",
                category="pipeline",
                action="start",
                payload={},
                reason=None,
                prev_hash="GENESIS",
                self_hash="abc123",
            ),
        ]
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = fake_events
        session.execute = AsyncMock(return_value=mock_result)

        from backend.deliverables.audit_log_export import generate_audit_log_export
        deliverables = await generate_audit_log_export(run, session, ctx)

        csv_d = next(d for d in deliverables if d.fmt == "csv")
        csv_str = csv_d.content.decode()
        assert "seq" in csv_str
        assert "actor" in csv_str
        assert "action" in csv_str
        assert "self_hash" in csv_str
