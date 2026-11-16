#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from los_baseline.exp123_data import TaskDataBundle, TaskSplitData
from los_baseline.task_spec import EXP1_FLAT_3CLASS_TASK, STAGE1_MORTALITY_TASK
from los_baseline.training_engine import train_and_evaluate_single_model


def _split(name: str, labels: np.ndarray, offset: int) -> TaskSplitData:
    labels = np.asarray(labels, dtype=np.int64)
    rng = np.random.default_rng(42 + offset)
    features = pd.DataFrame(
        {
            "feature_1": labels + rng.normal(0.0, 0.35, len(labels)),
            "feature_2": (labels == 0).astype(float)
            + rng.normal(0.0, 0.4, len(labels)),
            "feature_3": rng.normal(0.0, 1.0, len(labels)),
        }
    )
    features.loc[::13, "feature_3"] = np.nan
    metadata = pd.DataFrame(
        {
            "source_row_index": np.arange(offset, offset + len(labels)),
            "split": name,
            "subject_id": [f"{name}_{index}" for index in range(len(labels))],
        }
    )
    return TaskSplitData(name, features, labels, metadata)


def _bundle(task_name: str) -> TaskDataBundle:
    if task_name == "binary":
        task = STAGE1_MORTALITY_TASK
        train = np.tile([0, 1], 30)
        validation = np.tile([0, 1], 12)
        test = np.tile([0, 1], 12)
    else:
        task = EXP1_FLAT_3CLASS_TASK
        train = np.tile([0, 1, 2], 20)
        validation = np.tile([0, 1, 2], 8)
        test = np.tile([0, 1, 2], 8)
    return TaskDataBundle(
        task=task,
        feature_names=("feature_1", "feature_2", "feature_3"),
        splits={
            "train": _split("train", train, 0),
            "validation": _split("validation", validation, 1000),
            "internal_test": _split("internal_test", test, 2000),
        },
        shared_split_hash="training_engine_smoke_hash",
    )


def _config(model_name: str) -> dict:
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
        return {
            "parameters": {
                "n_estimators": 10,
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
            print(f"[INFO] Smoke training: {task_type}/{model_name}")
            result = train_and_evaluate_single_model(
                task_data=bundle,
                model_name=model_name,
                model_config=_config(model_name),
                output_dir=output_root / task_type / model_name,
                seed=42,
                prediction_tolerance=1e-6,
            )
            if not all(
                item.matches for item in result.prediction_integrity.values()
            ):
                raise RuntimeError(
                    f"Prediction integrity failed: {task_type}/{model_name}"
                )
            completed += 1
    print(f"[OK] Completed {completed} binary/multiclass engine smoke runs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
