"""
End-to-end pipeline simulation - drives the backend API exactly as the frontend would.

Usage:
    python simulate_run.py

Requires:
    - FastAPI running on localhost:8000
    - Celery worker running
    - Redis running
    - ANTHROPIC_API_KEY set in .env or environment
"""

import json
import sys
import time
from pathlib import Path

import requests

BASE = "http://localhost:8000"
HEADERS = {"X-User-Id": "dev-user-1"}
FIXTURE = Path(__file__).parent / "backend/tests/fixtures/telco_churn.csv"

CHECKPOINT_LABELS = {
    "checkpoint_1_eda": "EDA",
    "checkpoint_2_preprocessing": "Preprocessing",
    "checkpoint_3_model_selection": "Model Selection",
    "checkpoint_4_training": "Training",
    "checkpoint_5_final": "Final",
}


def req(method: str, path: str, **kwargs) -> dict:
    r = getattr(requests, method)(f"{BASE}{path}", headers=HEADERS, **kwargs)
    if not r.ok:
        print(f"\n[ERROR] {method.upper()} {path} → {r.status_code}")
        print(r.text[:1000])
        sys.exit(1)
    return r.json()


def poll_run(run_id: str, timeout: int = 600) -> dict:
    deadline = time.time() + timeout
    last_step = None
    last_progress = -1

    while time.time() < deadline:
        run = req("get", f"/runs/{run_id}")
        step = run.get("current_step", "")
        status = run.get("status", "")
        progress = run.get("progress", 0)

        if step != last_step or progress != last_progress:
            print(f"  [{progress:3d}%] {status} / {step}")
            last_step = step
            last_progress = progress

        if status == "failed":
            print(f"\n[FAILED] {run.get('error_message', 'unknown error')}")
            sys.exit(1)

        if status == "completed":
            return run

        if status == "awaiting_checkpoint":
            label = CHECKPOINT_LABELS.get(step, step)
            print(f"\n  → Checkpoint: {label} - auto-approving…")
            req("post", f"/runs/{run_id}/resume")
            time.sleep(1)
            continue

        time.sleep(3)

    print("\n[TIMEOUT] Pipeline did not complete within limit")
    sys.exit(1)


def main() -> None:
    print("=== AI-DS Platform - End-to-End Simulation ===\n")

    # 1. Health check
    print("1. Health check…")
    r = requests.get(f"{BASE}/health")
    if not r.ok:
        print(f"   [ERROR] FastAPI not reachable: {r.status_code}")
        sys.exit(1)
    print(f"   OK - {r.json()}")

    # 2. Create project
    print("\n2. Creating project…")
    project = req("post", "/projects", json={
        "name": "Telco Churn Simulation",
        "description": "Automated simulation run using telco_churn fixture",
        "business_context": "Predict which customers will churn. FN is costly - missed churner costs $500, FP costs $50 (discount voucher).",
    })
    project_id = project["id"]
    print(f"   Project: {project_id}")

    # 3. Upload dataset
    print("\n3. Uploading telco_churn.csv…")
    if not FIXTURE.exists():
        print(f"   [ERROR] Fixture not found: {FIXTURE}")
        sys.exit(1)

    with open(FIXTURE, "rb") as f:
        dataset = req(
            "post",
            f"/projects/{project_id}/datasets",
            files={"file": ("telco_churn.csv", f, "text/csv")},
            data={"role": "training", "target_column": "Churn"},
        )
    dataset_id = dataset["id"]
    print(f"   Dataset: {dataset_id}  ({dataset.get('row_count', '?')} rows)")

    # 4. Start analysis run
    print("\n4. Starting analysis run…")
    run = req("post", f"/projects/{project_id}/runs", json={
        "training_dataset_id": dataset_id,
        "threshold_config": {
            "cost_matrix": {
                "fn_cost": 500,
                "fp_cost": 50,
                "tp_cost": 0,
                "tn_cost": 0,
            }
        },
    })
    run_id = run["id"]
    print(f"   Run: {run_id}")

    # 5. Poll until done, auto-approving checkpoints
    print("\n5. Pipeline running - polling for progress…\n")
    final_run = poll_run(run_id, timeout=900)

    # 6. Report results
    print("\n=== Pipeline Complete ===\n")
    metrics = final_run.get("final_metrics") or {}
    threshold = (final_run.get("threshold_result") or {}).get("optimal_threshold", 0.5)
    top_features = ((final_run.get("shap_summary") or {}).get("top_k_features") or [])[:5]

    print(f"Best model    : {final_run.get('best_model_name', '?')}")
    print(f"CV score      : {final_run.get('best_model_score', '?'):.4f}")
    print(f"Threshold     : {threshold:.3f}  (optimized vs default 0.5)")
    print(f"Test AUC      : {metrics.get('roc_auc', '?')}")
    print(f"Test F1       : {metrics.get('f1', '?')}")
    print(f"Top features  : {', '.join(top_features)}")
    print(f"\nInsight report preview:\n{(final_run.get('insight_report') or '')[:400]}…")

    print("\n[OK] Simulation complete.")


if __name__ == "__main__":
    main()
