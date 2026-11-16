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
from los_baseline.final_comparison import run_final_statistical_comparison_for_cohort
from los_baseline.oracle_pipeline import ConfirmatoryCohort, run_oracle_pipeline_for_cohort
from los_baseline.stacking import SuperLearnerArtifact, save_super_learner_artifact
from los_baseline.task_spec import (
    EXP1_FLAT_3CLASS_TASK,
    STAGE1_MORTALITY_TASK,
    STAGE2_LONG_STAY_TASK,
)
from los_baseline.training_engine import train_and_evaluate_single_model
from los_baseline.utils import atomic_write_json, sha256_file

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
    labels = np.asarray(y3, dtype=np.int64)
    death = (labels == 1).astype(float)
    long_stay = (labels == 2).astype(float)
    frame = pd.DataFrame(
        {
            "feature_1": 1.4 * death + 0.5 * long_stay + rng.normal(0, 0.7, len(labels)),
            "feature_2": 0.4 * death + 1.2 * long_stay + rng.normal(0, 0.8, len(labels)),
            "feature_3": rng.normal(0, 1.0, len(labels)),
            "feature_4": rng.normal(0, 1.0, len(labels)),
        }
    )
    frame.loc[::19, "feature_4"] = np.nan
    return frame


def _metadata(name: str, y3: np.ndarray, offset: int) -> pd.DataFrame:
    labels = np.asarray(y3, dtype=np.int64)
    return pd.DataFrame(
        {
            "source_row_index": np.arange(offset, offset + len(labels), dtype=np.int64),
            "subject_id": [f"{name}_p_{index}" for index in range(len(labels))],
            "split": name,
            "three_class_outcome": labels,
            "stage1_mortality_label": (labels == 1).astype(np.int64),
            "stage2_eligible": labels != 1,
            "stage2_long_stay_label": np.where(
                labels == 1, np.nan, (labels == 2).astype(float)
            ),
        }
    )


def _task_split(name: str, y3: np.ndarray, offset: int, task_name: str) -> TaskSplitData:
    labels = np.asarray(y3, dtype=np.int64)
    features = _features_from_three_class(labels, seed=100 + offset)
    metadata = _metadata(name, labels, offset)
    if task_name == EXP1_FLAT_3CLASS_TASK.task_name:
        return TaskSplitData(name, features, labels, metadata)
    if task_name == STAGE1_MORTALITY_TASK.task_name:
        return TaskSplitData(name, features, (labels == 1).astype(np.int64), metadata)
    mask = labels != 1
    return TaskSplitData(
        name,
        features.loc[mask].reset_index(drop=True),
        (labels[mask] == 2).astype(np.int64),
        metadata.loc[mask].reset_index(drop=True),
    )


def _bundle(task_name: str) -> TaskDataBundle:
    train_y3 = np.array(([0] * 80) + ([1] * 45) + ([2] * 55), dtype=np.int64)
    validation_y3 = np.array(([0] * 36) + ([1] * 18) + ([2] * 26), dtype=np.int64)
    task = {
        EXP1_FLAT_3CLASS_TASK.task_name: EXP1_FLAT_3CLASS_TASK,
        STAGE1_MORTALITY_TASK.task_name: STAGE1_MORTALITY_TASK,
        STAGE2_LONG_STAY_TASK.task_name: STAGE2_LONG_STAY_TASK,
    }[task_name]
    return TaskDataBundle(
        task=task,
        feature_names=FEATURES,
        splits={
            "train": _task_split("train", train_y3, 0, task_name),
            "validation": _task_split("validation", validation_y3, 10000, task_name),
        },
        shared_split_hash=SPLIT_HASH,
    )


def _train_exp1(run_dir: Path) -> None:
    model_name = "decision_tree"
    train_and_evaluate_single_model(
        task_data=_bundle(EXP1_FLAT_3CLASS_TASK.task_name),
        model_name=model_name,
        model_config=_model_config(model_name),
        output_dir=run_dir / "EXP1_flat_3class" / model_name,
        seed=42,
    )
    payload = {
        "task_name": EXP1_FLAT_3CLASS_TASK.task_name,
        "selected_on_split": "validation",
        "representative_metric": "macro_auroc",
        "practical_tie_margin": 0.005,
        "selected_model": model_name,
        "selected_model_validation_metrics": {},
        "candidate_models": [model_name],
        "confirmatory_splits_accessed": False,
        "shared_split_hash": SPLIT_HASH,
    }
    path = run_dir / "EXP1_flat_3class" / "representative_model.json"
    atomic_write_json(path, payload)
    path.with_name(path.name + ".sha256").write_text(
        sha256_file(path) + "\n", encoding="utf-8"
    )


def _train_stage(run_dir: Path, task_name: str) -> None:
    bundle = _bundle(task_name)
    stage_dir = (
        "stage1_mortality"
        if task_name == STAGE1_MORTALITY_TASK.task_name
        else "stage2_long_stay"
    )
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
        path = (
            run_dir
            / "EXP2_two_stage"
            / stage_dir
            / model_name
            / "evaluation"
            / "validation"
            / "predictions.csv.gz"
        )
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
        selection_metric=(
            "auprc" if task_name == STAGE1_MORTALITY_TASK.task_name else "auroc"
        ),
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
    metadata = _metadata(split_name, y3, offset)
    return ConfirmatoryCohort(
        split_name=split_name,
        raw_features=features,
        y_true_3class=y3,
        metadata=metadata,
        feature_names=FEATURES,
    )


def _verify(run_dir: Path, split_name: str) -> None:
    root = run_dir / "final_comparison" / split_name
    required = [
        root / "flat_representative" / "predictions.csv.gz",
        root / "pipeline" / "predictions.csv.gz",
        root / "flat_vs_pipeline_metrics.csv",
        root / "paired_bootstrap_differences.csv",
        root / "delong_class_auroc_tests.csv",
        root / "mcnemar_correctness.json",
        root / "statistical_tests.csv",
        root / "manuscript_tables" / "main_comparison_table.csv",
        root / "comparison_manifest.json",
        root / "COMPLETED.lock.json",
    ]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    tests = pd.read_csv(root / "statistical_tests.csv")
    if len(tests) != 10:
        raise RuntimeError(f"Expected 10 inferential tests, found {len(tests)}")
    if tests["p_fdr"].isna().any():
        raise RuntimeError("FDR-adjusted p-values are missing.")
    table = pd.read_csv(root / "manuscript_tables" / "main_comparison_table.csv")
    if table["analysis"].tolist() != [
        "EXP1_flat_3class",
        "EXP2_pipeline_3class",
        "EXP2_oracle_equals_EXP3",
    ]:
        raise RuntimeError("Manuscript table rows are not in the expected order.")


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

    _train_exp1(run_dir)
    _train_stage(run_dir, STAGE1_MORTALITY_TASK.task_name)
    _train_stage(run_dir, STAGE2_LONG_STAY_TASK.task_name)

    config = json.loads((PROJECT_ROOT / "configs" / "experiment.json").read_text())
    config["evaluation"]["save_plots"] = True
    config["evaluation"]["calibration_bins"] = 5
    config["oracle_pipeline"] = {"bootstrap_repeats": 10, "seed": 42}
    config["final_comparison"] = {
        "bootstrap_repeats": 20,
        "confidence_level": 0.95,
        "seed": 42,
        "minimum_successful_bootstrap_repeats": 10,
        "comparison_metrics": [
            "macro_auroc",
            "macro_auprc",
            "macro_f1",
            "balanced_accuracy",
            "multiclass_brier",
            "multiclass_log_loss",
        ],
        "fdr_method": "fdr_bh",
        "fdr_alpha": 0.05,
    }

    for split_name, offset, seed in (
        ("internal_test", 20000, 71),
        ("external_eicu", 30000, 81),
    ):
        cohort = _cohort(split_name, offset, seed)
        run_oracle_pipeline_for_cohort(
            run_dir=run_dir,
            cohort=cohort,
            config=config,
        )
        result = run_final_statistical_comparison_for_cohort(
            run_dir=run_dir,
            cohort=cohort,
            config=config,
        )
        _verify(run_dir, split_name)
        print(
            f"[OK] {split_name}: flat macro-AUROC={result.flat_macro_auroc:.4f}, "
            f"pipeline macro-AUROC={result.pipeline_macro_auroc:.4f}"
        )

    print("[OK] Completed Internal Test and eICU final comparison smoke analyses.")
    print("[OK] Paired cluster bootstrap, class-specific DeLong, and McNemar passed.")
    print("[OK] BH-FDR and manuscript table generation were verified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
