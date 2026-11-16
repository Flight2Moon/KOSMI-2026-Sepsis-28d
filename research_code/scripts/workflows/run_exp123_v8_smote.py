#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from los_baseline.config import load_config, resolve_path
from los_baseline.exp1_runner import load_exp1_discovery_task_data
from los_baseline.stage2_runner import load_stage2_discovery_task_data
from los_baseline.stage1_runner import load_stage1_discovery_task_data
from los_baseline.imbalance import (
    ClassCSmoteSampler,
    build_class_c_synthetic_pool,
    save_synthetic_pool,
)
from los_baseline.imbalance.ensemble import ensemble_member_prediction_files
from los_baseline.progress import configure_progress, emit_progress
from los_baseline.training_engine import train_and_evaluate_single_model
from los_baseline.runner_common import validate_selected_hyperparameters
from los_baseline.utils import atomic_write_json, sha256_file
from los_baseline.exp2_combined_v8 import build_exp2_combined_from_prediction_files
from los_baseline.pipeline_predictions_v8 import (
    materialize_member_ensemble_pipeline_predictions,
    materialize_standard_artifact_pipeline_predictions,
)
from los_baseline.exp3_reporting_v8 import build_exp3_report_from_stage2_predictions
from los_baseline.v8_reporting import write_exp1_sampled_clinical_reports, write_stage_binary_clinical_reports
from los_baseline.experiment_scheduler import ResourceAwareScheduler, ScheduledJob
from los_baseline.resource_manager import request_for_job


MODELS_DEFAULT = "decision_tree,logistic_regression,random_forest,svm,xgboost"
TASKS = {
    "exp1": {
        "task_name": "exp1_flat_3class",
        "minority_class": 2,
        "loader": load_exp1_discovery_task_data,
    },
    "stage2": {
        "task_name": "stage2_long_stay",
        "minority_class": 1,
        "loader": load_stage2_discovery_task_data,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run V8 Track 3 Class-C SMOTE x2 with five-seed ensembles."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--optimized-run-id", required=True)
    parser.add_argument("--models", default=MODELS_DEFAULT)
    return parser.parse_args()


def _selected_model_config(optimized_dir: Path, task_data: Any, model_name: str) -> dict[str, Any]:
    task_name = task_data.task.task_name
    path = optimized_dir / "optimization" / task_name / model_name / "selected_hyperparameters.json"
    if not path.is_file():
        raise FileNotFoundError(f"Selected hyperparameters were not found: {path}")
    model_config, _ = validate_selected_hyperparameters(
        path=path,
        model_name=model_name,
        task=task_data.task,
        shared_split_hash=task_data.shared_split_hash,
    )
    model_config.setdefault("training_policy", {})["class_weight_mode"] = "none"
    return model_config


def _write_stage1_reference(run_dir: Path, optimized_dir: Path, models: list[str]) -> dict[str, Any]:
    refs: dict[str, Any] = {}
    for model in models:
        source = optimized_dir / "EXP2_two_stage" / "stage1_mortality" / model
        if not source.is_dir():
            raise FileNotFoundError(
                f"T3 Stage1 must reference the completed T2 natural Stage1 model: {source}"
            )
        hashes = {
            str(path.relative_to(source)): sha256_file(path)
            for path in sorted(source.rglob("*"))
            if path.is_file()
        }
        refs[model] = {
            "status": "referenced",
            "source": str(source),
            "artifact_sha256": hashes,
        }
    payload = {
        "schema_version": "exp123_v8_track3_stage1_reference_v1",
        "strategy_id": "T3_smote",
        "policy": "reuse_track2_natural_stage1",
        "optimized_run_dir": str(optimized_dir),
        "models": refs,
        "training_performed": False,
        "reason": "Class-C SMOTE is intentionally excluded from Stage1 to avoid altering mortality prevalence.",
    }
    out = run_dir / "TRACK3_smote" / "EXP2_two_stage"
    out.mkdir(parents=True, exist_ok=True)
    atomic_write_json(out / "stage1_reference.json", payload)
    return payload


def _write_exp3_reference(run_dir: Path, task_results: dict[str, Any]) -> dict[str, Any]:
    model_refs: dict[str, Any] = {}
    for model_name, row in dict(task_results.get("stage2", {})).items():
        model_refs[model_name] = {
            "source_model_dir": row["source_model_dir"],
            "model_manifest_path": row["model_manifest_path"],
            "model_manifest_sha256": row["model_manifest_sha256"],
            "ensemble_dir": row["ensemble_dir"],
        }
    payload = {
        "schema_version": "exp123_v8_track3_exp3_reference_v1",
        "strategy_id": "T3_smote",
        "experiment": "EXP3_survivor_binary",
        "source_task": "stage2_long_stay",
        "reuse_stage2_artifacts": True,
        "duplicate_training": False,
        "model_references": model_refs,
        "confirmatory_splits_accessed": False,
    }
    out = run_dir / "TRACK3_smote" / "EXP3_survivor_binary"
    out.mkdir(parents=True, exist_ok=True)
    atomic_write_json(out / "artifact_reference.json", payload)
    return payload


def main() -> int:
    args = parse_args()
    config_path = Path(args.config).expanduser().resolve()
    config = load_config(config_path)
    runs_dir = resolve_path(config, config["paths"]["runs_dir"])
    run_dir = runs_dir / args.run_id
    optimized_dir = runs_dir / args.optimized_run_id
    if not optimized_dir.is_dir():
        raise FileNotFoundError(f"Track2 optimized run was not found: {optimized_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    configure_progress(
        output_dir=run_dir / "common" / "progress",
        run_id=args.run_id,
        config=config,
        worker_label="track3_smote",
    )

    smote_cfg = dict(config.get("smote", {}))
    if not bool(smote_cfg.get("enabled", True)):
        raise ValueError("config.smote.enabled must be true for Track3 SMOTE.")
    multiplier = float(smote_cfg.get("minority_multiplier", 2.0))
    k_neighbors = int(smote_cfg.get("k_neighbors", 5))
    seeds = [int(v) for v in smote_cfg.get("seeds", [42, 142, 242, 342, 442])]
    patient_id_column = str(config.get("data", {}).get("patient_id_column", "subject_id"))
    models = [value.strip() for value in args.models.split(",") if value.strip()]

    exp1_data = load_exp1_discovery_task_data(config=config, run_dir=optimized_dir)
    stage2_data = load_stage2_discovery_task_data(config=config, run_dir=optimized_dir)
    if tuple(exp1_data.feature_names) != tuple(stage2_data.feature_names):
        raise ValueError("EXP1 and Stage2 must use identical ordered feature sets for shared SMOTE pools.")
    if exp1_data.shared_split_hash != stage2_data.shared_split_hash:
        raise ValueError("EXP1 and Stage2 must share the same patient split hash.")

    pools: dict[int, tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]] = {}
    for seed in seeds:
        pool_raw, pool_provenance, manifest = build_class_c_synthetic_pool(
            train_split=exp1_data.splits["train"],
            feature_names=exp1_data.feature_names,
            minority_class=2,
            multiplier=multiplier,
            seed=seed,
            k_neighbors=k_neighbors,
            patient_id_column=patient_id_column,
            binary_columns=tuple(smote_cfg.get("binary_columns", [])),
            integer_columns=tuple(smote_cfg.get("integer_columns", [])),
            value_ranges=dict(smote_cfg.get("value_ranges", {})),
        )
        pool_dir = run_dir / "common" / "synthetic" / f"seed_{seed}"
        save_synthetic_pool(pool_dir, pool_raw, pool_provenance, manifest)
        pools[seed] = (pool_raw, pool_provenance, manifest.to_dict())

    # Phase 6: member fits are independent and therefore scheduled as separate
    # resource-aware subprocesses. This allows CPU/GPU/RAM reservations to be
    # honored while multiple models/seeds run concurrently.
    jobs: list[ScheduledJob] = []
    for task_key, spec in TASKS.items():
        task_data = exp1_data if task_key == "exp1" else stage2_data
        task_name = str(spec["task_name"])
        # Validate selected hyperparameters before launching any worker so a
        # missing T2 artifact fails fast rather than halfway through the pool.
        for model_name in models:
            _selected_model_config(optimized_dir, task_data, model_name)
            for seed in seeds:
                completion = (
                    Path("TRACK3_smote")
                    / task_name
                    / model_name
                    / "members"
                    / f"seed_{seed}"
                    / "worker_completed.json"
                )
                command = (
                    sys.executable,
                    str(PROJECT_ROOT / "scripts" / "internal" / "run_exp123_v8_sampling_worker.py"),
                    "--mode", "smote-member",
                    "--config", str(config_path),
                    "--run-id", args.run_id,
                    "--optimized-run-id", args.optimized_run_id,
                    "--task", task_key,
                    "--model", model_name,
                    "--seed", str(seed),
                )
                jobs.append(
                    ScheduledJob(
                        name=f"t3_{task_key}_{model_name}_seed_{seed}",
                        command=command,
                        request=request_for_job(config, model_name=model_name, job_kind="training"),
                        completion_path=completion.as_posix(),
                        model_name=model_name,
                        task_key=task_key,
                        kind="track3_smote_member",
                    )
                )
    scheduler = ResourceAwareScheduler(
        jobs=jobs,
        config=config,
        project_root=PROJECT_ROOT,
        run_dir=run_dir,
        run_id=args.run_id,
        mode="v8_track3_smote_parallel",
    )
    scheduler_result = scheduler.run()
    failed_states = {"failed", "blocked", "resource_exceeded", "cancelled"}
    failed = {name: status for name, status in scheduler.status.items() if status in failed_states}
    if failed:
        raise RuntimeError(f"Track3 parallel member jobs failed: {failed}")

    task_results: dict[str, Any] = {}
    for task_key, spec in TASKS.items():
        task_data = exp1_data if task_key == "exp1" else stage2_data
        task_name = str(spec["task_name"])
        minority_class = int(spec["minority_class"])
        task_results[task_key] = {}
        for model_name in models:
            member_dirs: list[Path] = []
            members: list[dict[str, Any]] = []
            for seed in seeds:
                member_dir = run_dir / "TRACK3_smote" / task_name / model_name / "members" / f"seed_{seed}"
                completed_path = member_dir / "worker_completed.json"
                completed = json.loads(completed_path.read_text(encoding="utf-8"))
                member_dirs.append(member_dir)
                members.append({
                    "seed": seed,
                    "pool_sha256": completed["pool_sha256"],
                    "result": completed["result"],
                })
            model_root = run_dir / "TRACK3_smote" / task_name / model_name
            ensemble = ensemble_member_prediction_files(
                task=task_data.task,
                member_dirs=member_dirs,
                split_names=tuple(task_data.splits.keys()),
                output_dir=model_root / "ensemble",
            )
            manifest = {
                "schema_version": "exp123_v8_track3_smote_model_v2_parallel",
                "strategy_id": "T3_smote",
                "task_name": task_name,
                "model_name": model_name,
                "minority_class": minority_class,
                "multiplier": multiplier,
                "seeds": seeds,
                "shared_final_pool_across_exp1_stage2": True,
                "preprocessor_fit_distribution": "natural_train_before_sampling",
                "calibration_target_distribution": "natural_oof_only",
                "validation_distribution_modified": False,
                "internal_test_distribution_modified": False,
                "external_distribution_modified": False,
                "parallel_execution": {
                    "enabled": True,
                    "scheduler_mode": "resource_aware",
                    "member_job_count": len(member_dirs),
                },
                "members": members,
                "ensemble": ensemble,
            }
            model_manifest_path = model_root / "smote_model_manifest.json"
            atomic_write_json(model_manifest_path, manifest)
            task_results[task_key][model_name] = {
                "source_model_dir": str(model_root),
                "model_manifest_path": str(model_manifest_path),
                "model_manifest_sha256": sha256_file(model_manifest_path),
                "ensemble_dir": str(model_root / "ensemble"),
                "member_count": len(member_dirs),
                "pool_sha256_by_seed": {str(item["seed"]): item["pool_sha256"] for item in members},
            }

    stage1_reference = _write_stage1_reference(run_dir, optimized_dir, models)
    exp3_reference = _write_exp3_reference(run_dir, task_results)

    full_validation = load_stage1_discovery_task_data(config=config, run_dir=optimized_dir).splits["validation"]
    pipeline_root = run_dir / "TRACK3_smote" / "EXP2_two_stage" / "pipeline_inputs"
    stage1_pipeline_paths = {}
    stage2_pipeline_paths = {}
    for model in models:
        stage1_pipeline_paths[model] = materialize_standard_artifact_pipeline_predictions(
            artifact_dir=optimized_dir / "EXP2_two_stage" / "stage1_mortality" / model / "artifact",
            full_validation=full_validation,
            task_kind="stage1",
            output_path=pipeline_root / "stage1" / model / "validation_full.csv.gz",
            default_threshold=float(config.get("evaluation", {}).get("default_decision_threshold", 0.5)),
        )
        member_dirs = [
            run_dir / "TRACK3_smote" / "stage2_long_stay" / model / "members" / f"seed_{seed}"
            for seed in seeds
        ]
        stage2_pipeline_paths[model] = materialize_member_ensemble_pipeline_predictions(
            member_dirs=member_dirs,
            full_validation=full_validation,
            task_kind="stage2",
            survivor_validation_prediction_path=run_dir / "TRACK3_smote" / "stage2_long_stay" / model / "ensemble" / "validation" / "predictions.csv.gz",
            output_path=pipeline_root / "stage2" / model / "validation_full.csv.gz",
            default_threshold=float(config.get("evaluation", {}).get("default_decision_threshold", 0.5)),
        )

    exp1_prediction_paths = {
        model: run_dir / "TRACK3_smote" / "exp1_flat_3class" / model / "ensemble" / "validation" / "predictions.csv.gz"
        for model in models
    }
    stage1_prediction_paths = {
        model: optimized_dir / "EXP2_two_stage" / "stage1_mortality" / model / "evaluation" / "validation" / "predictions.csv.gz"
        for model in models
    }
    stage2_prediction_paths = {
        model: run_dir / "TRACK3_smote" / "stage2_long_stay" / model / "ensemble" / "validation" / "predictions.csv.gz"
        for model in models
    }
    write_exp1_sampled_clinical_reports(
        model_prediction_paths=exp1_prediction_paths,
        output_root=run_dir / "TRACK3_smote" / "EXP1_flat_3class" / "clinical",
        clinical_config=config.get("clinical_evaluation", {}),
    )
    write_stage_binary_clinical_reports(
        model_prediction_paths=stage1_prediction_paths,
        output_root=run_dir / "TRACK3_smote" / "EXP2_two_stage" / "stage1_mortality" / "clinical_reference_T2",
        clinical_config=config.get("clinical_evaluation", {}),
    )
    exp2_combined = build_exp2_combined_from_prediction_files(
        run_id=args.run_id,
        strategy_id="T3_smote",
        stage1_prediction_paths=stage1_pipeline_paths,
        stage2_prediction_paths=stage2_pipeline_paths,
        output_dir=run_dir / "TRACK3_smote" / "EXP2_two_stage" / "combined",
        primary_metric=str(config.get("analysis_plan", {}).get("task_selection_metrics", {}).get("exp2_combined", "macro_auprc")),
        default_threshold=float(config.get("evaluation", {}).get("default_decision_threshold", 0.5)),
        clinical_config=config.get("clinical_evaluation", {}),
    )
    exp3_report = build_exp3_report_from_stage2_predictions(
        run_id=args.run_id,
        strategy_id="T3_smote",
        stage2_prediction_paths=stage2_prediction_paths,
        output_dir=run_dir / "TRACK3_smote" / "EXP3_survivor_binary",
        clinical_config=config.get("clinical_evaluation", {}),
    )
    run_manifest = {
        "schema_version": "exp123_v8_track3_smote_run_v1",
        "run_id": args.run_id,
        "optimized_run_id": args.optimized_run_id,
        "strategy_id": "T3_smote",
        "outcome_horizon_days": float(config["outcomes"]["outcome_horizon_days"]),
        "smote": {
            "multiplier": multiplier,
            "k_neighbors": k_neighbors,
            "seeds": seeds,
            "stage1_policy": "reuse_track2_natural_stage1",
        },
        "task_results": task_results,
        "stage1_reference": stage1_reference,
        "exp3_reference": exp3_reference,
        "exp2_combined": exp2_combined,
        "exp3_report": exp3_report,
        "exp2_combined_evaluation_status": "completed",
        "exp3_reporting_status": "completed",
        "parallel_execution": {
            "enabled": True,
            "scheduler_mode": "resource_aware",
            "member_job_count": len(jobs),
            "scheduler_result": scheduler_result,
        },
        "confirmatory_splits_accessed": False,
    }
    atomic_write_json(run_dir / "TRACK3_smote" / "track3_manifest.json", run_manifest)
    print(f"[OK] V8 Track3 SMOTE completed: {args.run_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
