"""On-demand Jupyter notebook export (§4.9).

Generates a reproducible .ipynb for the run: data loading, preprocessing,
model training with the tuned hyperparameters, evaluation, and SHAP.
Triggered by POST /runs/{run_id}/deliverables/notebook or via chat
("export this as a notebook").

No raw row-level data is embedded - the notebook loads the dataset from
its storage path at runtime, preserving the §7 no-PII-to-model rule.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from backend.deliverables.base import GeneratedDeliverable

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession
    from backend.core.database import Dataset, Run

_COUNTER: list[int] = [0]  # mutable box so inner functions can mutate it


def _cell_id() -> str:
    _COUNTER[0] += 1
    return f"cell-{_COUNTER[0]:04d}"


def _md(lines: str | list[str]) -> dict[str, Any]:
    src = lines if isinstance(lines, list) else [lines]
    return {"cell_type": "markdown", "id": _cell_id(), "metadata": {}, "source": src}


def _code(lines: str | list[str]) -> dict[str, Any]:
    src = lines if isinstance(lines, list) else [lines]
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": _cell_id(),
        "metadata": {},
        "outputs": [],
        "source": src,
    }


def _model_import(model_name: str, best_params: dict[str, Any]) -> list[str]:
    """Return a code-cell source list that imports and instantiates the model."""
    params_repr = json.dumps(best_params, indent=2) if best_params else "{}"
    nm = model_name.lower()

    if "random_forest" in nm or "randomforest" in nm:
        cls = "RandomForestClassifier"
        imp = "from sklearn.ensemble import RandomForestClassifier"
    elif "gradient_boosting" in nm or "gradientboosting" in nm:
        cls = "GradientBoostingClassifier"
        imp = "from sklearn.ensemble import GradientBoostingClassifier"
    elif "extra_trees" in nm or "extratrees" in nm:
        cls = "ExtraTreesClassifier"
        imp = "from sklearn.ensemble import ExtraTreesClassifier"
    elif "xgb" in nm or "xgboost" in nm:
        cls = "XGBClassifier"
        imp = "from xgboost import XGBClassifier"
    elif "lgbm" in nm or "lightgbm" in nm:
        cls = "LGBMClassifier"
        imp = "from lightgbm import LGBMClassifier"
    elif "catboost" in nm:
        cls = "CatBoostClassifier"
        imp = "from catboost import CatBoostClassifier"
    elif "logistic" in nm:
        cls = "LogisticRegression"
        imp = "from sklearn.linear_model import LogisticRegression"
    elif "ridge" in nm and "regression" not in nm:
        cls = "Ridge"
        imp = "from sklearn.linear_model import Ridge"
    elif "ridge_regression" in nm or ("ridge" in nm and "regression" in nm):
        cls = "Ridge"
        imp = "from sklearn.linear_model import Ridge"
    elif "linear_regression" in nm or ("linear" in nm and "regression" in nm):
        cls = "LinearRegression"
        imp = "from sklearn.linear_model import LinearRegression"
    elif "svm" in nm or "svc" in nm:
        cls = "SVC"
        imp = "from sklearn.svm import SVC"
    elif "knn" in nm or "neighbor" in nm:
        cls = "KNeighborsClassifier"
        imp = "from sklearn.neighbors import KNeighborsClassifier"
    else:
        return [
            f"# Model: {model_name} - add the correct import below\n",
            f"best_params = {params_repr}\n",
            "# model = YourModel(**best_params)\n",
        ]

    return [
        f"{imp}\n",
        f"\n",
        f"best_params = {params_repr}\n",
        f"model = {cls}(**{{k: v for k, v in best_params.items() if v is not None}})\n",
    ]


def _eval_cells(task_type: str) -> list[dict[str, Any]]:
    cells: list[dict[str, Any]] = []
    if "regression" in task_type:
        cells.append(_code([
            "from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score\n",
            "import numpy as np\n",
            "\n",
            "y_pred = model.predict(X_test)\n",
            "print(f'RMSE : {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}')\n",
            "print(f'MAE  : {mean_absolute_error(y_test, y_pred):.4f}')\n",
            "print(f'R²   : {r2_score(y_test, y_pred):.4f}')\n",
        ]))
    else:
        cells.append(_code([
            "from sklearn.metrics import (\n",
            "    roc_auc_score, f1_score, accuracy_score, classification_report\n",
            ")\n",
            "\n",
            "y_pred = model.predict(X_test)\n",
            "y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None\n",
            "\n",
            "print(classification_report(y_test, y_pred))\n",
            "if y_prob is not None:\n",
            "    print(f'AUC: {roc_auc_score(y_test, y_prob):.4f}')\n",
        ]))
    return cells


async def generate_notebook(
    run: "Run",
    dataset: "Dataset | None",
    session: "AsyncSession",
    ctx: dict[str, Any],
) -> GeneratedDeliverable:
    """Build and return a .ipynb deliverable for the run."""
    _COUNTER[0] = 0  # reset cell IDs per call

    prep: dict[str, Any] = run.preprocessing_strategy or {}
    tuning: dict[str, Any] = run.tuning_result or {}
    shap: dict[str, Any] = run.shap_summary or {}

    task_type: str = prep.get("task_type") or ctx.get("task_type", "unknown")
    target: str = prep.get("target_column") or (dataset.target_column if dataset else "target") or "target"
    model_name: str = run.best_model_name or "unknown"
    best_params: dict[str, Any] = tuning.get("best_params") or {}
    seeds: dict[str, Any] = run.seeds or {}
    random_state: int = int(seeds.get("random_state", 42))

    col_configs: dict[str, Any] = {}
    if isinstance(prep.get("columns"), dict):
        col_configs = prep["columns"]

    drop_cols: list[str] = [
        c for c, cfg in col_configs.items()
        if isinstance(cfg, dict) and cfg.get("action") == "drop"
    ]
    high_corr: list[str] = prep.get("drop_high_correlation") or []
    all_drops = drop_cols + [c for c in high_corr if c not in drop_cols]

    impute_map: dict[str, str] = {
        c: cfg["impute_strategy"]
        for c, cfg in col_configs.items()
        if isinstance(cfg, dict) and cfg.get("action") == "keep" and cfg.get("impute_strategy")
    }
    encode_map: dict[str, str] = {
        c: cfg["encode_strategy"]
        for c, cfg in col_configs.items()
        if isinstance(cfg, dict) and cfg.get("action") == "keep" and cfg.get("encode_strategy")
    }

    dataset_path: str = dataset.storage_path if dataset else "path/to/dataset.csv"
    dataset_name: str = dataset.filename if dataset else "dataset.csv"
    row_count: str = str(dataset.row_count) if dataset and dataset.row_count else "?"
    py_version: str = (run.library_versions or {}).get("python", "3.11.0")

    cells: list[dict[str, Any]] = []

    # ── Title ────────────────────────────────────────────────────────────────
    cells.append(_md([
        f"# Reproducibility Notebook - Run `{run.id[:8]}…`\n",
        "\n",
        f"| | |\n",
        f"|---|---|\n",
        f"| **Run ID** | `{run.id}` |\n",
        f"| **Model** | {model_name} |\n",
        f"| **Task** | {task_type} |\n",
        f"| **Dataset** | {dataset_name} ({row_count} rows) |\n",
        f"| **Generated** | {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} |\n",
        "\n",
        "Running this notebook end-to-end reproduces the training pipeline exactly as "
        "the AI co-pilot ran it, using the same hyperparameters, random seeds, and "
        "preprocessing decisions.\n",
    ]))

    # ── Imports ──────────────────────────────────────────────────────────────
    cells.append(_md("## 1. Setup"))
    cells.append(_code([
        "import warnings; warnings.filterwarnings('ignore')\n",
        "import pandas as pd\n",
        "import numpy as np\n",
        "import joblib\n",
        "import matplotlib.pyplot as plt\n",
        "from sklearn.model_selection import train_test_split\n",
        "from sklearn.preprocessing import LabelEncoder\n",
    ]))

    # ── Load data ────────────────────────────────────────────────────────────
    cells.append(_md("## 2. Load data"))
    cells.append(_code([
        f"# Replace with your local path if running outside the platform\n",
        f'df = pd.read_csv("{dataset_path}")\n',
        f"print(f'Loaded {{df.shape[0]}} rows × {{df.shape[1]}} columns')\n",
        "df.head(3)\n",
    ]))

    # ── Preprocessing ────────────────────────────────────────────────────────
    cells.append(_md("## 3. Preprocessing"))

    if all_drops:
        cells.append(_code([
            "# Columns dropped by the AI pipeline\n",
            f"drop_cols = {json.dumps(all_drops)}\n",
            "df = df.drop(columns=[c for c in drop_cols if c in df.columns])\n",
        ]))

    if impute_map:
        cells.append(_code([
            "# Imputation strategies\n",
            f"impute_map = {json.dumps(impute_map, indent=2)}\n",
            "for col, strategy in impute_map.items():\n",
            "    if col not in df.columns:\n",
            "        continue\n",
            "    if strategy == 'mean':\n",
            "        df[col] = df[col].fillna(df[col].mean())\n",
            "    elif strategy == 'median':\n",
            "        df[col] = df[col].fillna(df[col].median())\n",
            "    elif strategy == 'mode':\n",
            "        df[col] = df[col].fillna(df[col].mode().iloc[0])\n",
            "    else:\n",
            "        df[col] = df[col].fillna(strategy)\n",
        ]))

    if encode_map:
        cells.append(_code([
            "# Categorical encoding\n",
            f"encode_map = {json.dumps(encode_map, indent=2)}\n",
            "for col, strategy in encode_map.items():\n",
            "    if col not in df.columns:\n",
            "        continue\n",
            "    if strategy == 'label':\n",
            "        df[col] = LabelEncoder().fit_transform(df[col].astype(str))\n",
            "    elif strategy == 'onehot':\n",
            "        dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)\n",
            "        df = pd.concat([df.drop(columns=[col]), dummies], axis=1)\n",
        ]))

    cells.append(_code([
        f"TARGET = {json.dumps(target)}\n",
        f"RANDOM_STATE = {random_state}\n",
        "\n",
        "X = df.drop(columns=[TARGET])\n",
        "y = df[TARGET]\n",
        "\n",
        "stratify = y if y.nunique() < 20 else None\n",
        "X_train, X_test, y_train, y_test = train_test_split(\n",
        "    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=stratify\n",
        ")\n",
        "print(f'Train: {X_train.shape}  Test: {X_test.shape}')\n",
    ]))

    # ── Model ────────────────────────────────────────────────────────────────
    cells.append(_md(f"## 4. Train: {model_name}"))
    cells.append(_code(_model_import(model_name, best_params)))
    cells.append(_code([
        "model.fit(X_train, y_train)\n",
        "print('Training complete')\n",
    ]))

    # ── Evaluation ───────────────────────────────────────────────────────────
    cells.append(_md("## 5. Evaluation"))
    cells.extend(_eval_cells(task_type))

    # ── SHAP ─────────────────────────────────────────────────────────────────
    if shap.get("top_k_features"):
        top5 = shap.get("top_k_features", [])[:5]
        cells.append(_md("## 6. SHAP feature importance"))
        cells.append(_code([
            "import shap\n",
            "\n",
            f"# Top-5 features from the run: {top5}\n",
            "sample = X_test.iloc[:min(200, len(X_test))]\n",
            "explainer = shap.Explainer(model, X_train)\n",
            "shap_values = explainer(sample)\n",
            "shap.summary_plot(shap_values, sample)\n",
        ]))

    # ── Save / reload ─────────────────────────────────────────────────────────
    short_id = run.id.replace("-", "")[:8]
    cells.append(_md("## 7. Save model"))
    cells.append(_code([
        f'MODEL_PATH = "model_{short_id}.joblib"\n',
        "joblib.dump(model, MODEL_PATH)\n",
        "print(f'Saved to {MODEL_PATH}')\n",
        "\n",
        "# Reload:\n",
        "# model = joblib.load(MODEL_PATH)\n",
    ]))

    notebook: dict[str, Any] = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "pygments_lexer": "ipython3",
                "version": py_version,
            },
            "ai_ds_platform": {
                "run_id": run.id,
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "model": model_name,
                "task_type": task_type,
            },
        },
        "cells": cells,
    }

    content = json.dumps(notebook, indent=1, ensure_ascii=False).encode()

    return GeneratedDeliverable.build(
        name="notebook",
        fmt="ipynb",
        content=content,
        run_id=run.id,
        inputs_used=[
            "preprocessing_strategy",
            "model_selection",
            "tuning_result",
            "shap_summary",
        ],
        audience="ML engineer, data scientist",
    )
