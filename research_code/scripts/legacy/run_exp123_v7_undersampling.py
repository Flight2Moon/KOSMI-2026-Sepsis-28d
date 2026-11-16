#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from los_baseline.config import load_config, resolve_path
from los_baseline.exp1_runner import load_exp1_discovery_task_data
from los_baseline.stage1_runner import load_stage1_discovery_task_data
from los_baseline.stage2_runner import load_stage2_discovery_task_data
from los_baseline.imbalance import run_undersampling_search
from los_baseline.progress import configure_progress, emit_progress
from los_baseline.utils import atomic_write_json

TASKS = {
    "exp1": (load_exp1_discovery_task_data, "exp1_flat_3class"),
    "stage1": (load_stage1_discovery_task_data, "stage1_mortality"),
    "stage2": (load_stage2_discovery_task_data, "stage2_long_stay"),
}


def parse_args():
    p = argparse.ArgumentParser(description="Run Track 3 fold-wise majority undersampling.")
    p.add_argument("--config", required=True)
    p.add_argument("--run-id", required=True, help="Track 3 output run ID")
    p.add_argument("--optimized-run-id", required=True, help="Track 2 run providing hyperparameters and fixed CV")
    p.add_argument("--models", default="decision_tree,logistic_regression,random_forest,svm,xgboost")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    config_path = Path(args.config).expanduser().resolve()
    config = load_config(config_path)
    runs_dir = resolve_path(config, config["paths"]["runs_dir"])
    run_dir = runs_dir / args.run_id
    optimized_dir = runs_dir / args.optimized_run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    configure_progress(output_dir=run_dir / "common" / "progress", run_id=args.run_id, config=config, worker_label="track3_undersampling")
    models = [v.strip() for v in args.models.split(",") if v.strip()]
    results = {}
    for task_key, (loader, task_name) in TASKS.items():
        task_data = loader(config=config, run_dir=optimized_dir)
        fixed_cv_dir = optimized_dir / "common" / "fixed_cv" / task_name
        results[task_key] = {}
        for model in models:
            selection = optimized_dir / "optimization" / task_name / model / "selected_hyperparameters.json"
            if not selection.is_file():
                results[task_key][model] = {"status": "skipped", "reason": "optimized hyperparameters missing"}
                continue
            model_config = json.loads(selection.read_text(encoding="utf-8"))["model_config"]
            emit_progress("track3_model_started", operation=f"{task_key}/{model}", status="running")
            payload = run_undersampling_search(
                task_data=task_data, model_name=model, model_config=model_config,
                project_config=config, fixed_cv_dir=fixed_cv_dir,
                output_dir=run_dir / "TRACK3_undersampled" / task_name / model,
            )
            results[task_key][model] = {"status": "completed", "manifest": payload}
    summary_rows = []
    for task_key, model_rows in results.items():
        for model_name, row in model_rows.items():
            manifest_row = row.get("manifest", {}) if isinstance(row, dict) else {}
            summary_rows.append({
                "task": task_key,
                "model": model_name,
                "status": row.get("status") if isinstance(row, dict) else "unknown",
                "selected_ratio": manifest_row.get("selected_ratio"),
                "member_count": len(manifest_row.get("sampling_seeds", [])),
                "fixed_cv_hash": manifest_row.get("fixed_cv_hash"),
            })
    summary_path = run_dir / "TRACK3_undersampled" / "track3_summary.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    manifest = {
        "schema_version": "exp123_v7_track3_run_v1",
        "run_id": args.run_id,
        "optimized_run_id": args.optimized_run_id,
        "results": results,
        "confirmatory_splits_accessed": False,
    }
    atomic_write_json(run_dir / "TRACK3_undersampled" / "track3_manifest.json", manifest)
    print(f"[OK] Track 3 fold-wise undersampling completed: {args.run_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
