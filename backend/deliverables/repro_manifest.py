"""Reproducibility manifest - YAML (§4.7).

Records everything required to reproduce the exact run:
seeds, library versions, dataset hashes, strategy hashes, model IDs.
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from backend.deliverables.base import GeneratedDeliverable

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession
    from backend.core.database import Dataset, Run


def _strategy_hash(obj: Any) -> str:
    if not obj:
        return "none"
    serialized = json.dumps(obj, sort_keys=True, default=str).encode()
    return "sha256:" + hashlib.sha256(serialized).hexdigest()[:16]


def _collect_library_versions() -> dict[str, str]:
    """Collect versions of all key ML libraries at runtime."""
    versions: dict[str, str] = {
        "python": sys.version.split()[0],
    }
    for pkg in ["pandas", "numpy", "scikit_learn", "xgboost", "lightgbm",
                "optuna", "shap", "faiss", "anthropic"]:
        try:
            import importlib.metadata
            versions[pkg.replace("_", "-")] = importlib.metadata.version(pkg.replace("_", "-"))
        except Exception:
            pass
    return versions


async def generate_repro_manifest(
    run: "Run",
    dataset: "Dataset | None",
    session: "AsyncSession",
    ctx: dict[str, Any],
) -> GeneratedDeliverable:
    """Generate the YAML reproducibility manifest."""
    try:
        import yaml  # pyyaml is a transitive dep of many packages
    except ImportError:
        yaml = None  # type: ignore[assignment]

    lib_versions = run.library_versions or _collect_library_versions()

    manifest: dict[str, Any] = {
        "run_id": run.id,
        "created_at": run.created_at.isoformat() if run.created_at else "",
        "completed_at": run.completed_at.isoformat() if run.completed_at else "",
        "created_by": run.created_by or "unknown",
        "platform": "ai-ds-platform",
        "clinical_domain": "healthcare",
        "generator_version": "1.0.0",
        "datasets": [],
        "random_seeds": run.seeds or {},
        "split": {
            "test_size": 0.2,
            "stratify": ctx.get("task_type") in ("binary_classification", "multiclass"),
            "random_state": 42,
        },
        "hashes": {
            "preprocessing_strategy": _strategy_hash(run.preprocessing_strategy),
            "model_selection": _strategy_hash(run.model_selection),
            "tuning_result": _strategy_hash(run.tuning_result),
        },
        "model": {
            "name": run.best_model_name or "unknown",
            "score": run.best_model_score,
            "final_metrics": run.final_metrics or {},
            "storage_path": run.model_storage_path or "",
            "faiss_index_path": run.faiss_index_path or "",
        },
        "claude_models_used": run.claude_models_used or {},
        "library_versions": lib_versions,
        "artifacts": {
            "model": run.model_storage_path or f"runs/{run.id}/model.joblib",
            "faiss_index": run.faiss_index_path or f"runs/{run.id}/similarity.index",
            "splits": f"runs/{run.id}/splits.pkl",
        },
    }

    if dataset:
        manifest["datasets"].append({
            "role": "training",
            "filename": dataset.filename,
            "sha256": dataset.sha256,
            "schema_hash": dataset.schema_hash,
            "rows": dataset.row_count,
            "cols": dataset.col_count,
            "target_column": dataset.target_column,
            "storage_path": dataset.storage_path,
        })

    if yaml:
        content = yaml.dump(manifest, default_flow_style=False, allow_unicode=True).encode()
    else:
        # Fallback: pretty JSON if pyyaml not available
        content = json.dumps(manifest, indent=2, default=str).encode()
        # Still label as yaml - content is valid
        content = b"# YAML (rendered as JSON - install pyyaml for proper YAML)\n" + content

    return GeneratedDeliverable.build(
        name="repro_manifest",
        fmt="yaml",
        content=content,
        run_id=run.id,
        inputs_used=["run_metadata", "dataset_metadata", "model_storage_path"],
        audience="MLOps team, technical reviewer",
    )
