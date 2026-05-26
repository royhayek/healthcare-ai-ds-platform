"""Unit tests for notebook_export.py (§4.9)."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

from backend.deliverables.notebook_export import generate_notebook


def _mock_run(overrides: dict[str, Any] | None = None) -> Any:
    run = SimpleNamespace(
        id="test-run-uuid-1234",
        best_model_name="random_forest",
        preprocessing_strategy={
            "task_type": "binary_classification",
            "target_column": "churn",
            "columns": {
                "tenure": {"action": "keep", "impute_strategy": "median", "encode_strategy": None},
                "monthly_charges": {"action": "keep", "impute_strategy": "mean", "encode_strategy": None},
                "contract": {"action": "keep", "impute_strategy": None, "encode_strategy": "onehot"},
                "customer_id": {"action": "drop", "reason": "identifier"},
            },
            "drop_high_correlation": ["total_charges"],
            "notes": None,
        },
        model_selection={"primary": "random_forest", "candidates": ["random_forest", "xgboost"]},
        tuning_result={"best_params": {"n_estimators": 200, "max_depth": 8}, "n_trials": 50},
        shap_summary={"top_k_features": ["tenure", "monthly_charges", "contract_two_year"], "mean_abs_shap": [0.12, 0.09, 0.07]},
        final_metrics={"auc": 0.84, "f1": 0.71},
        seeds={"random_state": 42},
        library_versions={"python": "3.11.0"},
    )
    if overrides:
        for k, v in overrides.items():
            setattr(run, k, v)
    return run


def _mock_dataset() -> Any:
    return SimpleNamespace(
        filename="telco_churn.csv",
        storage_path="datasets/telco_churn.csv",
        sha256="abc123def456",
        row_count=7032,
        col_count=21,
        target_column="churn",
        task_type="binary_classification",
    )


@pytest.mark.asyncio
async def test_notebook_generates_valid_ipynb() -> None:
    run = _mock_run()
    dataset = _mock_dataset()
    ctx = {"task_type": "binary_classification"}

    deliverable = await generate_notebook(run, dataset, MagicMock(), ctx)

    assert deliverable.name == "notebook"
    assert deliverable.fmt == "ipynb"
    assert deliverable.storage_path == f"runs/{run.id}/deliverables/notebook.ipynb"
    assert len(deliverable.content) > 100

    nb = json.loads(deliverable.content.decode())
    assert nb["nbformat"] == 4
    assert isinstance(nb["cells"], list)
    assert len(nb["cells"]) >= 6


@pytest.mark.asyncio
async def test_notebook_contains_expected_code() -> None:
    run = _mock_run()
    dataset = _mock_dataset()
    ctx = {"task_type": "binary_classification"}

    deliverable = await generate_notebook(run, dataset, MagicMock(), ctx)
    nb = json.loads(deliverable.content.decode())

    source_text = "\n".join(
        "".join(cell["source"]) for cell in nb["cells"] if cell["cell_type"] == "code"
    )

    assert "TARGET = " in source_text
    assert '"churn"' in source_text
    assert "train_test_split" in source_text
    assert "RandomForestClassifier" in source_text
    assert "model.fit" in source_text
    assert "n_estimators" in source_text  # tuned params


@pytest.mark.asyncio
async def test_notebook_drop_and_impute_cells() -> None:
    run = _mock_run()
    dataset = _mock_dataset()
    ctx = {"task_type": "binary_classification"}

    deliverable = await generate_notebook(run, dataset, MagicMock(), ctx)
    nb = json.loads(deliverable.content.decode())

    source_text = "\n".join(
        "".join(cell["source"]) for cell in nb["cells"] if cell["cell_type"] == "code"
    )

    assert "drop_cols" in source_text
    assert "customer_id" in source_text
    assert "total_charges" in source_text
    assert "impute_map" in source_text
    assert "encode_map" in source_text


@pytest.mark.asyncio
async def test_notebook_no_dataset() -> None:
    run = _mock_run()
    ctx = {"task_type": "binary_classification"}

    deliverable = await generate_notebook(run, None, MagicMock(), ctx)
    nb = json.loads(deliverable.content.decode())

    assert nb["nbformat"] == 4
    source_text = "\n".join(
        "".join(cell["source"]) for cell in nb["cells"] if cell["cell_type"] == "code"
    )
    assert "path/to/dataset.csv" in source_text


@pytest.mark.asyncio
async def test_notebook_regression_task() -> None:
    run = _mock_run({
        "best_model_name": "ridge_regression",
        "preprocessing_strategy": {
            "task_type": "regression",
            "target_column": "sale_price",
            "columns": {},
            "drop_high_correlation": [],
        },
    })
    ctx = {"task_type": "regression"}

    deliverable = await generate_notebook(run, None, MagicMock(), ctx)
    nb = json.loads(deliverable.content.decode())

    source_text = "\n".join(
        "".join(cell["source"]) for cell in nb["cells"] if cell["cell_type"] == "code"
    )

    assert "mean_squared_error" in source_text
    assert "r2_score" in source_text


@pytest.mark.asyncio
async def test_notebook_shap_cell_present() -> None:
    run = _mock_run()
    dataset = _mock_dataset()
    ctx = {"task_type": "binary_classification"}

    deliverable = await generate_notebook(run, dataset, MagicMock(), ctx)
    nb = json.loads(deliverable.content.decode())

    source_text = "\n".join(
        "".join(cell["source"]) for cell in nb["cells"] if cell["cell_type"] == "code"
    )
    assert "shap" in source_text
    assert "summary_plot" in source_text
