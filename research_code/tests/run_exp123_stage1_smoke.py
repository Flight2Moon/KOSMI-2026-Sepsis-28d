#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from los_baseline.exp123_data import TaskDataBundle, TaskSplitData
from los_baseline.stage1_runner import run_stage1_mortality
from los_baseline.task_spec import STAGE1_MORTALITY_TASK
from los_baseline.utils import atomic_write_json, sha256_file


def _split(name: str, labels: np.ndarray, offset: int) -> TaskSplitData:
    labels = np.asarray(labels, dtype=np.int64)
    rng = np.random.default_rng(100 + offset)
    features = pd.DataFrame(
        {
            "feature_1": labels * 1.2 + rng.normal(0.0, 0.8, len(labels)),
            "feature_2": labels * 0.7 + rng.normal(0.0, 1.0, len(labels)),
            "feature_3": rng.normal(0.0, 1.0, len(labels)),
            "feature_4": rng.normal(0.0, 1.0, len(labels)),
        }
    )
    features.loc[::19, "feature_4"] = np.nan
    metadata = pd.DataFrame(
        {
            "source_row_index": np.arange(offset, offset + len(labels)),
            "split": name,
            "subject_id": [f"{name}_{index}" for index in range(len(labels))],
        }
    )
    return TaskSplitData(name, features, labels, metadata)


def _bundle() -> TaskDataBundle:
    train = np.array([0] * 120 + [1] * 60, dtype=np.int64)
    validation = np.array([0] * 48 + [1] * 24, dtype=np.int64)
    return TaskDataBundle(
        task=STAGE1_MORTALITY_TASK,
        feature_names=("feature_1", "feature_2", "feature_3", "feature_4"),
        splits={
            "train": _split("train", train, 0),
            "validation": _split("validation", validation, 10000),
        },
        shared_split_hash="e" * 64,
    )


def _model_config(model_name: str) -> dict:
    if model_name == "logistic_regression":
        return {
            "epochs": 5,
            "batch_size": 32,
            "learning_rate": 0.01,
            "metric_interval": 1,
            "device": "cpu",
            "fixed_epoch_training": True,
        }
    if model_name == "decision_tree":
        return {"parameters": {"max_depth": 4, "min_samples_split": 2}}
    if model_name == "random_forest":
        return {
            "parameters": {
                "n_estimators": 15,
                "max_depth": 5,
                "n_jobs": 1,
            }
        }
    if model_name == "svm":
        return {
            "parameters": {
                "kernel": "rbf",
                "C": 1.0,
                "gamma": "scale",
                "probability": True,
                "cache_size": 256,
            }
        }
    if model_name == "xgboost":
        return {
            "parameters": {
                "n_estimators": 15,
                "max_depth": 3,
                "learning_rate": 0.1,
                "tree_method": "hist",
                "device": "cpu",
                "n_jobs": 1,
                "verbosity": 0,
            }
        }
    raise KeyError(model_name)


def _prepare_optimization_files(
    run_dir: Path,
    model_names: list[str],
    split_hash: str,
) -> None:
    for model_name in model_names:
        output_dir = (
            run_dir
            / "optimization"
            / STAGE1_MORTALITY_TASK.task_name
            / model_name
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "study_id": f"{STAGE1_MORTALITY_TASK.task_name}:{model_name}",
            "task_name": STAGE1_MORTALITY_TASK.task_name,
            "model_name": model_name,
            "selected_on_split": "validation",
            "confirmatory_splits_accessed": False,
            "forbidden_splits": ["internal_test", "external_eicu"],
            "selected_candidate_id": "smoke_candidate",
            "source_stage": "smoke",
            "source_trial_number": 0,
            "model_config": _model_config(model_name),
            "metrics": {},
            "objective": {
                "primary": "auprc",
                "direction": "maximize",
                "secondary": "brier_score",
                "secondary_direction": "minimize",
            },
            "selection_guards": {},
            "shared_split_hash": split_hash,
        }
        path = output_dir / "selected_hyperparameters.json"
        atomic_write_json(path, payload)
        (output_dir / "selected_hyperparameters.json.sha256").write_text(
            sha256_file(path) + "\n", encoding="utf-8"
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    run_dir = Path(args.output).expanduser().resolve()
    if run_dir.exists() and args.overwrite:
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=False)

    config = json.loads((PROJECT_ROOT / "configs" / "experiment.json").read_text())
    config["_project_root"] = str(PROJECT_ROOT)
    config["evaluation"]["save_plots"] = True
    config["evaluation"]["calibration_bins"] = 5
    config["evaluation"]["binary_bootstrap_repeats"] = 10
    models = list(
        config["experiments"]["exp2_two_stage"]["stage1_mortality"]["models"]
    )
    bundle = _bundle()
    _prepare_optimization_files(run_dir, models, str(bundle.shared_split_hash))

    result = run_stage1_mortality(
        task_data=bundle,
        config=config,
        run_id=run_dir.name,
        run_dir=run_dir,
        model_names=models,
        bootstrap_repeats=10,
    )
    if result.confirmatory_splits_accessed:
        raise RuntimeError("Stage1 smoke unexpectedly accessed confirmatory splits.")
    stage1_dir = run_dir / "EXP2_two_stage" / "stage1_mortality"
    required = [
        stage1_dir / "validation_reference_model.json",
        stage1_dir / "frozen_stage1_design.json",
        stage1_dir / "model_summary_validation.csv",
        stage1_dir / "stage1_manifest.json",
    ]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    for model_name in models:
        validation_dir = stage1_dir / model_name / "evaluation" / "validation"
        for filename in [
            "predictions.csv.gz",
            "binary_metrics.json",
            "threshold_metrics.csv",
            "bootstrap_confidence_intervals.csv",
            "confusion_matrix_raw.csv",
            "roc_curve.csv",
        ]:
            if not (validation_dir / filename).is_file():
                raise FileNotFoundError(validation_dir / filename)
    print(f"[OK] Completed Stage1 smoke run for {len(models)} models.")
    print(f"[OK] Validation reference model: {result.validation_reference_model}")
    print("[OK] Confirmatory splits were not accessed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
