#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from los_baseline.config import load_config, resolve_path
from los_baseline.exp2_combined_v8 import build_exp2_combined_from_prediction_files
from los_baseline.stage1_runner import load_stage1_discovery_task_data
from los_baseline.pipeline_predictions_v8 import materialize_standard_artifact_pipeline_predictions
from los_baseline.exp3_reporting_v8 import build_exp3_report_from_stage2_predictions
from los_baseline.v8_reporting import write_exp1_sampled_clinical_reports, write_stage_binary_clinical_reports
from los_baseline.utils import atomic_write_json


def _models(config: dict, task: str) -> list[str]:
    if task == "exp1":
        return list(config["experiments"]["exp1_flat_3class"]["models"])
    key = "stage1_mortality" if task == "stage1" else "stage2_long_stay"
    return list(config["experiments"]["exp2_two_stage"][key]["models"])


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate V8 reporting overlays for natural T1/T2 runs.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--strategy", required=True, choices=["T1_baseline", "T2_optuna"])
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    config = load_config(Path(args.config).expanduser().resolve())
    run_dir = resolve_path(config, config["paths"]["runs_dir"]) / args.run_id
    if not run_dir.is_dir():
        raise FileNotFoundError(run_dir)
    root = run_dir / "V8_reporting"
    completed_manifest = root / "reporting_manifest.json"
    if completed_manifest.is_file() and not args.overwrite:
        print(f"[OK] V8 reporting overlay already completed: {completed_manifest}")
        return 0
    if root.exists() and any(root.iterdir()) and args.overwrite:
        import shutil
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)

    exp1_paths = {
        model: run_dir / "EXP1_flat_3class" / model / "evaluation" / "validation" / "predictions.csv.gz"
        for model in _models(config, "exp1")
    }
    stage1_paths = {
        model: run_dir / "EXP2_two_stage" / "stage1_mortality" / model / "evaluation" / "validation" / "predictions.csv.gz"
        for model in _models(config, "stage1")
    }
    stage2_paths = {
        model: run_dir / "EXP2_two_stage" / "stage2_long_stay" / model / "evaluation" / "validation" / "predictions.csv.gz"
        for model in _models(config, "stage2")
    }
    for mapping_name, mapping in (("EXP1", exp1_paths), ("Stage1", stage1_paths), ("Stage2", stage2_paths)):
        missing = [str(path) for path in mapping.values() if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"{mapping_name} validation predictions are incomplete: {missing}")

    full_validation = load_stage1_discovery_task_data(config=config, run_dir=run_dir).splits["validation"]
    pipeline_root = root / "EXP2_two_stage" / "pipeline_inputs"
    stage1_pipeline_paths = {}
    stage2_pipeline_paths = {}
    for model in _models(config, "stage1"):
        stage1_pipeline_paths[model] = materialize_standard_artifact_pipeline_predictions(
            artifact_dir=run_dir / "EXP2_two_stage" / "stage1_mortality" / model / "artifact",
            full_validation=full_validation,
            task_kind="stage1",
            output_path=pipeline_root / "stage1" / model / "validation_full.csv.gz",
            default_threshold=float(config.get("evaluation", {}).get("default_decision_threshold", 0.5)),
        )
    for model in _models(config, "stage2"):
        stage2_pipeline_paths[model] = materialize_standard_artifact_pipeline_predictions(
            artifact_dir=run_dir / "EXP2_two_stage" / "stage2_long_stay" / model / "artifact",
            full_validation=full_validation,
            task_kind="stage2",
            output_path=pipeline_root / "stage2" / model / "validation_full.csv.gz",
            default_threshold=float(config.get("evaluation", {}).get("default_decision_threshold", 0.5)),
        )

    exp1_clinical = write_exp1_sampled_clinical_reports(
        model_prediction_paths=exp1_paths,
        output_root=root / "EXP1_flat_3class" / "clinical",
        clinical_config=config.get("clinical_evaluation", {}),
    )
    stage1_clinical = write_stage_binary_clinical_reports(
        model_prediction_paths=stage1_paths,
        output_root=root / "EXP2_two_stage" / "stage1_mortality" / "clinical",
        clinical_config=config.get("clinical_evaluation", {}),
    )
    stage2_clinical = write_stage_binary_clinical_reports(
        model_prediction_paths=stage2_paths,
        output_root=root / "EXP2_two_stage" / "stage2_long_stay" / "clinical",
        clinical_config=config.get("clinical_evaluation", {}),
    )
    combined = build_exp2_combined_from_prediction_files(
        run_id=args.run_id,
        strategy_id=args.strategy,
        stage1_prediction_paths=stage1_pipeline_paths,
        stage2_prediction_paths=stage2_pipeline_paths,
        output_dir=root / "EXP2_two_stage" / "combined",
        primary_metric=str(config.get("analysis_plan", {}).get("task_selection_metrics", {}).get("exp2_combined", "macro_auprc")),
        default_threshold=float(config.get("evaluation", {}).get("default_decision_threshold", 0.5)),
        overwrite=args.overwrite,
        clinical_config=config.get("clinical_evaluation", {}),
    )
    exp3 = build_exp3_report_from_stage2_predictions(
        run_id=args.run_id,
        strategy_id=args.strategy,
        stage2_prediction_paths=stage2_paths,
        output_dir=root / "EXP3_survivor_binary",
        overwrite=args.overwrite,
        clinical_config=config.get("clinical_evaluation", {}),
    )
    manifest = {
        "schema_version": "exp123_v8_natural_reporting_overlay_v1",
        "run_id": args.run_id,
        "strategy_id": args.strategy,
        "outcome_horizon_days": float(config["outcomes"]["outcome_horizon_days"]),
        "feature_set": config.get("feature_set"),
        "ablation": config.get("ablation"),
        "exp1_clinical": exp1_clinical,
        "stage1_clinical": stage1_clinical,
        "stage2_clinical": stage2_clinical,
        "exp2_combined": combined,
        "exp3_report": exp3,
        "confirmatory_splits_accessed": False,
    }
    atomic_write_json(root / "reporting_manifest.json", manifest)
    print(f"[OK] V8 reporting overlay completed: {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
