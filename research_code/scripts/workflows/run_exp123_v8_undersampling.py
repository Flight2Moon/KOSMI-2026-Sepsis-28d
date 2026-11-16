#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from los_baseline.config import load_config, resolve_path
from los_baseline.exp1_runner import load_exp1_discovery_task_data
from los_baseline.stage1_runner import load_stage1_discovery_task_data
from los_baseline.stage2_runner import load_stage2_discovery_task_data
from los_baseline.imbalance import run_undersampling_search
from los_baseline.progress import configure_progress, emit_progress
from los_baseline.utils import atomic_write_json
from los_baseline.runner_common import validate_selected_hyperparameters
from los_baseline.exp2_combined_v8 import build_exp2_combined_from_prediction_files
from los_baseline.pipeline_predictions_v8 import materialize_member_ensemble_pipeline_predictions
from los_baseline.exp3_reporting_v8 import build_exp3_report_from_stage2_predictions
from los_baseline.v8_reporting import write_exp1_sampled_clinical_reports, write_stage_binary_clinical_reports
from los_baseline.experiment_scheduler import ResourceAwareScheduler, ScheduledJob
from los_baseline.resource_manager import request_for_job

TASKS = {
    "exp1": (load_exp1_discovery_task_data, "exp1_flat_3class"),
    "stage1": (load_stage1_discovery_task_data, "stage1_mortality"),
    "stage2": (load_stage2_discovery_task_data, "stage2_long_stay"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run optional V8 Track4 fold-wise majority undersampling."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--optimized-run-id", required=True)
    parser.add_argument("--models", default="decision_tree,logistic_regression,random_forest,svm,xgboost")
    parser.add_argument(
        "--confirm-optional-track",
        action="store_true",
        help="Required guard confirming that T4 was prespecified before confirmatory access.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.confirm_optional_track:
        raise SystemExit(
            "[BLOCKED] T4 Undersampling is optional. Re-run with "
            "--confirm-optional-track after prespecifying include_undersampling=true."
        )
    config_path = Path(args.config).expanduser().resolve()
    config = load_config(config_path)
    runs_dir = resolve_path(config, config["paths"]["runs_dir"])
    run_dir = runs_dir / args.run_id
    optimized_dir = runs_dir / args.optimized_run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    configure_progress(
        output_dir=run_dir / "common" / "progress",
        run_id=args.run_id,
        config=config,
        worker_label="track4_undersampling",
    )
    models = [v.strip() for v in args.models.split(",") if v.strip()]
    # Phase 6: each task/model undersampling search is independent. Run these
    # searches through the shared resource-aware scheduler instead of serially.
    results: dict[str, dict[str, object]] = {key: {} for key in TASKS}
    jobs: list[ScheduledJob] = []
    for task_key, (loader, task_name) in TASKS.items():
        task_data = loader(config=config, run_dir=optimized_dir)
        for model in models:
            selection = optimized_dir / "optimization" / task_name / model / "selected_hyperparameters.json"
            if not selection.is_file():
                results[task_key][model] = {
                    "status": "skipped",
                    "reason": "optimized hyperparameters missing",
                }
                continue
            validate_selected_hyperparameters(
                path=selection,
                model_name=model,
                task=task_data.task,
                shared_split_hash=task_data.shared_split_hash,
            )
            completion = Path("TRACK4_undersampled") / task_name / model / "worker_completed.json"
            command = (
                sys.executable,
                str(PROJECT_ROOT / "scripts" / "internal" / "run_exp123_v8_sampling_worker.py"),
                "--mode", "undersampling-model",
                "--config", str(config_path),
                "--run-id", args.run_id,
                "--optimized-run-id", args.optimized_run_id,
                "--task", task_key,
                "--model", model,
            )
            jobs.append(ScheduledJob(
                name=f"t4_{task_key}_{model}",
                command=command,
                request=request_for_job(config, model_name=model, job_kind="training"),
                completion_path=completion.as_posix(),
                model_name=model,
                task_key=task_key,
                kind="track4_undersampling_model",
            ))

    scheduler_result = None
    if jobs:
        scheduler = ResourceAwareScheduler(
            jobs=jobs, config=config, project_root=PROJECT_ROOT, run_dir=run_dir,
            run_id=args.run_id, mode="v8_track4_undersampling_parallel",
        )
        scheduler_result = scheduler.run()
        failed_states = {"failed", "blocked", "resource_exceeded", "cancelled"}
        failed = {name: status for name, status in scheduler.status.items() if status in failed_states}
        if failed:
            raise RuntimeError(f"Track4 parallel task/model jobs failed: {failed}")

    for task_key, (_, task_name) in TASKS.items():
        for model in models:
            if model in results[task_key] and results[task_key][model].get("status") == "skipped":
                continue
            manifest_path = run_dir / "TRACK4_undersampled" / task_name / model / "undersampling_manifest.json"
            if not manifest_path.is_file():
                results[task_key][model] = {"status": "missing", "reason": str(manifest_path)}
                continue
            results[task_key][model] = {
                "status": "completed",
                "manifest": json.loads(manifest_path.read_text(encoding="utf-8")),
            }

    rows = []
    for task_key, model_rows in results.items():
        for model_name, row in model_rows.items():
            manifest = row.get("manifest", {}) if isinstance(row, dict) else {}
            rows.append(
                {
                    "task": task_key,
                    "model": model_name,
                    "status": row.get("status") if isinstance(row, dict) else "unknown",
                    "selected_ratio": manifest.get("selected_ratio") if isinstance(manifest, dict) else None,
                    "member_count": len(manifest.get("sampling_seeds", [])) if isinstance(manifest, dict) else 0,
                    "fixed_cv_hash": manifest.get("fixed_cv_hash") if isinstance(manifest, dict) else None,
                }
            )
    root = run_dir / "TRACK4_undersampled"
    root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(root / "track4_summary.csv", index=False)

    completed_models = {
        task_key: [model_name for model_name, row in model_rows.items() if isinstance(row, dict) and row.get("status") == "completed"]
        for task_key, model_rows in results.items()
    }
    full_validation = load_stage1_discovery_task_data(config=config, run_dir=optimized_dir).splits["validation"]
    pipeline_root = root / "EXP2_two_stage" / "pipeline_inputs"
    stage1_pipeline_paths = {}
    stage2_pipeline_paths = {}
    undersampling_seeds = [int(v) for v in config.get("undersampling", {}).get("seeds", [42, 142, 242, 342, 442])]
    for model in completed_models.get("stage1", []):
        member_dirs = [root / "stage1_mortality" / model / "members" / f"seed_{seed}" for seed in undersampling_seeds]
        stage1_pipeline_paths[model] = materialize_member_ensemble_pipeline_predictions(
            member_dirs=member_dirs,
            full_validation=full_validation,
            task_kind="stage1",
            survivor_validation_prediction_path=root / "stage1_mortality" / model / "ensemble" / "validation" / "predictions.csv.gz",
            output_path=pipeline_root / "stage1" / model / "validation_full.csv.gz",
            default_threshold=float(config.get("evaluation", {}).get("default_decision_threshold", 0.5)),
        )
    for model in completed_models.get("stage2", []):
        member_dirs = [root / "stage2_long_stay" / model / "members" / f"seed_{seed}" for seed in undersampling_seeds]
        stage2_pipeline_paths[model] = materialize_member_ensemble_pipeline_predictions(
            member_dirs=member_dirs,
            full_validation=full_validation,
            task_kind="stage2",
            survivor_validation_prediction_path=root / "stage2_long_stay" / model / "ensemble" / "validation" / "predictions.csv.gz",
            output_path=pipeline_root / "stage2" / model / "validation_full.csv.gz",
            default_threshold=float(config.get("evaluation", {}).get("default_decision_threshold", 0.5)),
        )

    exp1_prediction_paths = {
        model: root / "exp1_flat_3class" / model / "ensemble" / "validation" / "predictions.csv.gz"
        for model in completed_models.get("exp1", [])
    }
    stage1_prediction_paths = {
        model: root / "stage1_mortality" / model / "ensemble" / "validation" / "predictions.csv.gz"
        for model in completed_models.get("stage1", [])
    }
    stage2_prediction_paths = {
        model: root / "stage2_long_stay" / model / "ensemble" / "validation" / "predictions.csv.gz"
        for model in completed_models.get("stage2", [])
    }
    if exp1_prediction_paths:
        write_exp1_sampled_clinical_reports(
            model_prediction_paths=exp1_prediction_paths,
            output_root=root / "EXP1_flat_3class" / "clinical",
            clinical_config=config.get("clinical_evaluation", {}),
        )
    if stage1_prediction_paths:
        write_stage_binary_clinical_reports(
            model_prediction_paths=stage1_prediction_paths,
            output_root=root / "EXP2_two_stage" / "stage1_mortality" / "clinical",
            clinical_config=config.get("clinical_evaluation", {}),
        )
    exp2_combined = build_exp2_combined_from_prediction_files(
        run_id=args.run_id,
        strategy_id="T4_undersampling",
        stage1_prediction_paths=stage1_pipeline_paths,
        stage2_prediction_paths=stage2_pipeline_paths,
        output_dir=root / "EXP2_two_stage" / "combined",
        primary_metric=str(config.get("analysis_plan", {}).get("task_selection_metrics", {}).get("exp2_combined", "macro_auprc")),
        default_threshold=float(config.get("evaluation", {}).get("default_decision_threshold", 0.5)),
        clinical_config=config.get("clinical_evaluation", {}),
    )
    exp3_report = build_exp3_report_from_stage2_predictions(
        run_id=args.run_id,
        strategy_id="T4_undersampling",
        stage2_prediction_paths=stage2_prediction_paths,
        output_dir=root / "EXP3_survivor_binary",
        clinical_config=config.get("clinical_evaluation", {}),
    )

    atomic_write_json(
        root / "track4_manifest.json",
        {
            "schema_version": "exp123_v8_track4_run_v1",
            "run_id": args.run_id,
            "optimized_run_id": args.optimized_run_id,
            "strategy_id": "T4_undersampling",
            "optional_track": True,
            "prespecified_before_confirmatory_access": True,
            "results": results,
            "exp2_combined": exp2_combined,
            "exp3_report": exp3_report,
            "exp2_combined_evaluation_status": "completed",
            "exp3_reference_status": "completed",
            "parallel_execution": {
                "enabled": True,
                "scheduler_mode": "resource_aware",
                "task_model_job_count": len(jobs),
                "scheduler_result": scheduler_result,
            },
            "confirmatory_splits_accessed": False,
        },
    )
    print(f"[OK] Optional V8 Track4 undersampling completed: {args.run_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
