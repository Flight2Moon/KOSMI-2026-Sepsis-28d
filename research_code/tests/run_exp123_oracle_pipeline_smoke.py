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
from los_baseline.oracle_pipeline import ConfirmatoryCohort, run_oracle_pipeline_for_cohort
from los_baseline.stacking import SuperLearnerArtifact, save_super_learner_artifact
from los_baseline.task_spec import STAGE1_MORTALITY_TASK, STAGE2_LONG_STAY_TASK
from los_baseline.training_engine import train_and_evaluate_single_model
from los_baseline.utils import sha256_file

MODELS = ("decision_tree", "random_forest")
FEATURES = ("feature_1", "feature_2", "feature_3", "feature_4")
SPLIT_HASH = "d" * 64


def _model_config(model_name: str) -> dict:
    if model_name == "decision_tree":
        return {"parameters": {"max_depth": 4, "min_samples_split": 4}}
    if model_name == "random_forest":
        return {
            "parameters": {
                "n_estimators": 20,
                "max_depth": 5,
                "n_jobs": 1,
            }
        }
    raise KeyError(model_name)


def _features_from_three_class(y3: np.ndarray, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    y3 = np.asarray(y3, dtype=np.int64)
    death = (y3 == 1).astype(float)
    long_stay = (y3 == 2).astype(float)
    frame = pd.DataFrame(
        {
            "feature_1": 1.4 * death + 0.5 * long_stay + rng.normal(0, 0.7, len(y3)),
            "feature_2": 0.4 * death + 1.2 * long_stay + rng.normal(0, 0.8, len(y3)),
            "feature_3": rng.normal(0, 1.0, len(y3)),
            "feature_4": rng.normal(0, 1.0, len(y3)),
        }
    )
    frame.loc[::19, "feature_4"] = np.nan
    return frame


def _task_split(name: str, y3: np.ndarray, offset: int, task_name: str) -> TaskSplitData:
    y3 = np.asarray(y3, dtype=np.int64)
    features = _features_from_three_class(y3, seed=100 + offset)
    metadata = pd.DataFrame(
        {
            "source_row_index": np.arange(offset, offset + len(y3), dtype=np.int64),
            "subject_id": [f"{name}_p_{index}" for index in range(len(y3))],
            "split": name,
            "three_class_outcome": y3,
            "stage1_mortality_label": (y3 == 1).astype(np.int64),
            "stage2_eligible": y3 != 1,
            "stage2_long_stay_label": np.where(y3 == 1, np.nan, (y3 == 2).astype(float)),
        }
    )
    if task_name == STAGE1_MORTALITY_TASK.task_name:
        labels = (y3 == 1).astype(np.int64)
        return TaskSplitData(name, features, labels, metadata)
    mask = y3 != 1
    stage2_features = features.loc[mask].reset_index(drop=True)
    stage2_metadata = metadata.loc[mask].reset_index(drop=True)
    labels = (y3[mask] == 2).astype(np.int64)
    return TaskSplitData(name, stage2_features, labels, stage2_metadata)


def _bundle(task_name: str) -> TaskDataBundle:
    train_y3 = np.array(([0] * 80) + ([1] * 45) + ([2] * 55), dtype=np.int64)
    validation_y3 = np.array(([0] * 36) + ([1] * 18) + ([2] * 26), dtype=np.int64)
    task = STAGE1_MORTALITY_TASK if task_name == STAGE1_MORTALITY_TASK.task_name else STAGE2_LONG_STAY_TASK
    return TaskDataBundle(
        task=task,
        feature_names=FEATURES,
        splits={
            "train": _task_split("train", train_y3, 0, task_name),
            "validation": _task_split("validation", validation_y3, 10000, task_name),
        },
        shared_split_hash=SPLIT_HASH,
    )


def _train_stage(run_dir: Path, task_name: str) -> None:
    bundle = _bundle(task_name)
    stage_dir = "stage1_mortality" if task_name == STAGE1_MORTALITY_TASK.task_name else "stage2_long_stay"
    for model_name in MODELS:
        train_and_evaluate_single_model(
            task_data=bundle,
            model_name=model_name,
            model_config=_model_config(model_name),
            output_dir=run_dir / "EXP2_two_stage" / stage_dir / model_name,
            seed=42,
        )

    prediction_hashes = {}
    for model_name in MODELS:
        path = run_dir / "EXP2_two_stage" / stage_dir / model_name / "evaluation" / "validation" / "predictions.csv.gz"
        prediction_hashes[model_name] = sha256_file(path)
    artifact = SuperLearnerArtifact(
        artifact_schema_version="exp123_super_learner_v1",
        task_name=task_name,
        meta_learner_method="nnls",
        library_members=MODELS,
        weights=(0.5, 0.5),
        validation_decision_threshold=0.5,
        fit_split="validation_meta",
        threshold_selection_split="validation_threshold",
        shared_split_hash=SPLIT_HASH,
        validation_subsplit_hash="e" * 64,
        base_prediction_hashes=prediction_hashes,
        heterogeneity_frozen_design_sha256="f" * 64,
        library_candidates_sha256="a" * 64,
        selection_metric="auprc" if task_name == STAGE1_MORTALITY_TASK.task_name else "auroc",
        selection_metric_value=0.7,
        auroc_margin_guard=0.02,
        practical_tie_margin=0.005,
        seed=42,
    )
    output = run_dir / "EXP2_two_stage" / "super_learner" / task_name / "artifact.json"
    save_super_learner_artifact(output, artifact)


def _cohort(split_name: str, offset: int, seed: int) -> ConfirmatoryCohort:
    y3 = np.array(([0] * 45) + ([1] * 25) + ([2] * 35), dtype=np.int64)
    rng = np.random.default_rng(seed)
    rng.shuffle(y3)
    features = _features_from_three_class(y3, seed=seed + 10)
    metadata = pd.DataFrame(
        {
            "source_row_index": np.arange(offset, offset + len(y3), dtype=np.int64),
            "subject_id": [f"{split_name}_{index}" for index in range(len(y3))],
            "split": split_name,
            "three_class_outcome": y3,
        }
    )
    return ConfirmatoryCohort(
        split_name=split_name,
        raw_features=features,
        y_true_3class=y3,
        metadata=metadata,
        feature_names=FEATURES,
    )


def _verify(run_dir: Path, split_name: str) -> None:
    root = run_dir / "EXP2_two_stage" / "pipeline_evaluation" / split_name
    required = [
        root / "pipeline_3class" / "predictions.csv.gz",
        root / "pipeline_3class" / "evaluation" / "multiclass_metrics.json",
        root / "oracle_stage2" / "predictions.csv.gz",
        root / "oracle_stage2" / "evaluation" / "binary_metrics.json",
        root / "error_propagation" / "error_propagation_summary.csv",
        root / "probability_integrity.json",
        root / "confirmatory_manifest.json",
        root / "COMPLETED.lock.json",
        root / "oracle_pipeline_result.json",
    ]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    predictions = pd.read_csv(root / "pipeline_3class" / "predictions.csv.gz")
    probability_columns = ["prob_class_0", "prob_class_1", "prob_class_2"]
    if not np.allclose(predictions[probability_columns].sum(axis=1), 1.0):
        raise RuntimeError("Pipeline class probabilities do not sum to one.")
    if set(predictions["pipeline_predicted_class"].unique()) - {0, 1, 2}:
        raise RuntimeError("Pipeline produced an invalid class.")
    oracle = pd.read_csv(root / "oracle_stage2" / "predictions.csv.gz")
    if set(oracle["three_class_outcome"].unique()) != {0, 2}:
        raise RuntimeError("Oracle Stage2 evaluation includes mortality class B.")
    exp3 = run_dir / "EXP3_survivor_binary" / f"oracle_evaluation_reference_{split_name}.json"
    if not exp3.is_file():
        raise FileNotFoundError(exp3)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    run_dir = Path(args.output).expanduser().resolve()
    if run_dir.exists() and args.overwrite:
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=False)
    split_dir = run_dir / "common" / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    (split_dir / "split_hash.txt").write_text(SPLIT_HASH + "\n", encoding="utf-8")

    _train_stage(run_dir, STAGE1_MORTALITY_TASK.task_name)
    _train_stage(run_dir, STAGE2_LONG_STAY_TASK.task_name)

    config = json.loads((PROJECT_ROOT / "configs" / "experiment.json").read_text())
    config["evaluation"]["save_plots"] = True
    config["evaluation"]["calibration_bins"] = 5
    config["oracle_pipeline"] = {"bootstrap_repeats": 10, "seed": 42}

    for split_name, offset, seed in (
        ("internal_test", 20000, 71),
        ("external_eicu", 30000, 81),
    ):
        result = run_oracle_pipeline_for_cohort(
            run_dir=run_dir,
            cohort=_cohort(split_name, offset, seed),
            config=config,
        )
        if result.sample_count <= result.survivor_count:
            raise RuntimeError("Synthetic confirmatory cohort did not include deaths.")
        _verify(run_dir, split_name)
        print(
            f"[OK] {split_name}: macro-AUROC={result.pipeline_macro_auroc:.4f}, "
            f"oracle AUROC={result.oracle_stage2_auroc:.4f}"
        )

    print("[OK] Completed Internal Test and eICU Oracle/Pipeline smoke analyses.")
    print("[OK] Stage1 and Stage2 artifacts were loaded without refitting.")
    print("[OK] Three-class probability restoration and error propagation were verified.")
    print("[OK] Confirmatory overwrite locks and EXP3 Oracle references were created.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
