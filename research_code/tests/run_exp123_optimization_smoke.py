#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from los_baseline.exp123_data import TaskDataBundle, TaskSplitData
from los_baseline.optimization import run_optimization_study
from los_baseline.task_spec import EXP1_FLAT_3CLASS_TASK, STAGE1_MORTALITY_TASK


def _split(name: str, labels: np.ndarray, offset: int) -> TaskSplitData:
    labels = np.asarray(labels, dtype=np.int64)
    rng = np.random.default_rng(100 + offset)
    features = pd.DataFrame(
        {
            "feature_1": labels.astype(float) + rng.normal(0.0, 0.3, len(labels)),
            "feature_2": (labels == 0).astype(float) + rng.normal(0.0, 0.4, len(labels)),
            "feature_3": rng.normal(0.0, 1.0, len(labels)),
        }
    )
    features.loc[::11, "feature_3"] = np.nan
    metadata = pd.DataFrame(
        {
            "source_row_index": np.arange(offset, offset + len(labels)),
            "split": name,
            "subject_id": [f"{name}_subject_{index // 2}" for index in range(len(labels))],
        }
    )
    return TaskSplitData(name, features, labels, metadata)


def _bundle(task_type: str) -> TaskDataBundle:
    if task_type == "binary":
        task = STAGE1_MORTALITY_TASK
        train_labels = np.tile([0, 1, 0, 1], 20)
        validation_labels = np.tile([0, 1, 0, 1], 8)
        test_labels = np.tile([0, 1, 0, 1], 8)
    else:
        task = EXP1_FLAT_3CLASS_TASK
        train_labels = np.tile([0, 1, 2, 0, 1, 2], 15)
        validation_labels = np.tile([0, 1, 2, 0, 1, 2], 6)
        test_labels = np.tile([0, 1, 2, 0, 1, 2], 6)
    return TaskDataBundle(
        task=task,
        feature_names=("feature_1", "feature_2", "feature_3"),
        splits={
            "train": _split("train", train_labels, 0),
            "validation": _split("validation", validation_labels, 1000),
            "internal_test": _split("internal_test", test_labels, 2000),
        },
        shared_split_hash="optimization_smoke_split_hash",
    )


def _base_model_config(model_name: str) -> dict:
    if model_name == "logistic_regression":
        return {
            "epochs": 3,
            "batch_size": 32,
            "learning_rate": 0.01,
            "metric_interval": 1,
            "device": "cpu",
        }
    if model_name == "decision_tree":
        return {"parameters": {"max_depth": 4, "min_samples_split": 2}}
    if model_name == "random_forest":
        return {"parameters": {"n_estimators": 10, "max_depth": 5, "n_jobs": 1}}
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
                "n_estimators": 10,
                "max_depth": 3,
                "learning_rate": 0.1,
                "tree_method": "hist",
                "device": "cpu",
                "n_jobs": 1,
                "verbosity": 0,
            }
        }
    raise KeyError(model_name)


def _config() -> dict:
    return {
        "data": {"patient_id_column": "subject_id"},
        "optimization": {
            "seed": 42,
            "cv_folds": 2,
            "top_k_for_validation": 1,
            "search_stages": {
                "smoke": {"trials": 1, "cv_folds": 2},
                "coarse": {"trials": 1, "cv_folds": 2},
                "fine": {"trials": 1, "cv_folds": 2},
            },
            "selection": {
                "stability_penalty": 0.5,
                "practical_tie_margin": 0.005,
                "auroc_margin_guard": 0.02,
                "prefer_simpler_model_on_tie": True,
            },
            "runtime_limits": {
                "trial_timeout_seconds": 600,
                "study_timeout_seconds": 3600,
            },
            "resume": True,
        },
        "optimization_objectives": {
            "exp1_flat_3class": {
                "primary": "macro_auroc",
                "direction": "maximize",
                "secondary": "multiclass_brier",
                "secondary_direction": "minimize",
            },
            "stage1_mortality": {
                "primary": "auprc",
                "direction": "maximize",
                "secondary": "brier_score",
                "secondary_direction": "minimize",
            },
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    output_root = Path(args.output).expanduser().resolve()
    if output_root.exists() and args.overwrite:
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=False)

    models = [
        "decision_tree",
        "logistic_regression",
        "random_forest",
        "svm",
        "xgboost",
    ]
    completed = 0
    for task_type in ("binary", "multiclass"):
        bundle = _bundle(task_type)
        for model_name in models:
            print(f"[INFO] Optimization smoke: {task_type}/{model_name}")
            _, selection = run_optimization_study(
                task_data=bundle,
                model_name=model_name,
                base_model_config=copy.deepcopy(_base_model_config(model_name)),
                config=_config(),
                output_dir=output_root / task_type / model_name,
                search_stages=("smoke",),
            )
            if not Path(selection.output_path).is_file():
                raise RuntimeError(
                    f"Missing selection artifact: {task_type}/{model_name}"
                )
            completed += 1
    print(f"[OK] Completed {completed} optimization smoke studies.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
