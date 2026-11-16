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

from los_baseline.stacking import run_stage_super_learner
from los_baseline.stage_heterogeneity import stage_analysis_root, stage_prediction_path
from los_baseline.task_spec import STAGE1_MORTALITY_TASK, STAGE2_LONG_STAY_TASK
from los_baseline.utils import atomic_write_json, sha256_file


MODELS = (
    "decision_tree",
    "logistic_regression",
    "random_forest",
    "svm",
    "xgboost",
)


def _sigmoid(value: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(value, -20.0, 20.0)))


def _write_stage_predictions(
    run_dir: Path,
    task_name: str,
    *,
    seed: int,
    n: int = 240,
) -> None:
    rng = np.random.default_rng(seed)
    latent = rng.normal(0.0, 1.0, n)
    baseline = -1.2 if task_name == STAGE1_MORTALITY_TASK.task_name else -0.4
    probability = _sigmoid(baseline + 1.0 * latent)
    y = rng.binomial(1, probability).astype(np.int64)
    if np.unique(y).size != 2:
        raise RuntimeError("Synthetic labels must contain both classes.")
    source_rows = np.arange(seed * 10000, seed * 10000 + n, dtype=np.int64)
    subject_ids = np.asarray([f"{task_name}_p_{index}" for index in range(n)])

    definitions = {
        "decision_tree": (0.65, 0.45),
        "logistic_regression": (0.95, 0.23),
        "random_forest": (0.80, 0.30),
        "svm": (0.55, 0.40),
        "xgboost": (1.05, 0.20),
    }
    for model_name, (signal, noise) in definitions.items():
        model_probability = _sigmoid(
            baseline + signal * latent + rng.normal(0.0, noise, n)
        )
        path = stage_prediction_path(
            run_dir,
            task_name,
            model_name,
            "validation",
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            {
                "source_row_index": source_rows,
                "subject_id": subject_ids,
                "split": "validation",
                "task_name": task_name,
                "model_name": model_name,
                "y_true": y,
                "positive_probability": model_probability,
                "predicted_at_validation_threshold": (model_probability >= 0.5).astype(int),
                "predicted_at_0_5": (model_probability >= 0.5).astype(int),
            }
        ).to_csv(path, index=False, compression="gzip")


def _write_step8_evidence(run_dir: Path, task_name: str) -> None:
    discovery = stage_analysis_root(run_dir, task_name) / "discovery_validation"
    discovery.mkdir(parents=True, exist_ok=True)
    candidate_payload = {
        "status": "validation_evidence_frozen",
        "purpose": "Stage9 Super Learner subset-search candidate pool",
        "selection_is_final_super_learner": False,
        "candidate_models": list(MODELS),
        "excluded_models": [],
        "rule": "smoke_all_models",
        "model_evidence": [],
    }
    atomic_write_json(discovery / "library_candidates.json", candidate_payload)
    atomic_write_json(
        discovery / "frozen_design.json",
        {
            "design_status": "frozen",
            "task_name": task_name,
            "model_names": list(MODELS),
            "confirmatory_splits_viewed": False,
        },
    )


def _verify_output(run_dir: Path, task_name: str) -> None:
    output = run_dir / "EXP2_two_stage" / "super_learner" / task_name
    required = [
        output / "artifact.json",
        output / "artifact.json.sha256",
        output / "artifact_integrity.json",
        output / "subset_search_results.csv",
        output / "selected_library.json",
        output / "validation_subsplit_assignments.csv.gz",
        output / "validation_subsplit_manifest.json",
        output / "predictions" / "validation_meta.csv.gz",
        output / "predictions" / "validation_threshold.csv.gz",
        output / "predictions" / "validation_full.csv.gz",
        output / "evaluation" / "validation_threshold" / "binary_metrics.json",
        output / "super_learner_manifest.json",
        output / "super_learner_result.json",
    ]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    artifact = json.loads((output / "artifact.json").read_text(encoding="utf-8"))
    weights = np.asarray(artifact["weights"], dtype=float)
    if np.any(weights < -1e-12) or not np.isclose(weights.sum(), 1.0, atol=1e-10):
        raise RuntimeError("Invalid Super Learner weights.")
    if sha256_file(output / "artifact.json") != (
        output / "artifact.json.sha256"
    ).read_text(encoding="utf-8").strip():
        raise RuntimeError("Super Learner artifact SHA-256 mismatch.")
    subsplit = pd.read_csv(output / "validation_subsplit_assignments.csv.gz")
    meta = set(subsplit.loc[subsplit["validation_subsplit"] == "validation_meta", "subject_id"])
    threshold = set(
        subsplit.loc[
            subsplit["validation_subsplit"] == "validation_threshold", "subject_id"
        ]
    )
    if meta & threshold:
        raise RuntimeError("Patient overlap detected in validation subsplits.")
    integrity = json.loads((output / "artifact_integrity.json").read_text())
    if not integrity["all_validation_predictions_match"]:
        raise RuntimeError("Artifact reload prediction integrity failed.")
    manifest = json.loads((output / "super_learner_manifest.json").read_text())
    if manifest["data_access"]["internal_test"] or manifest["data_access"]["external_eicu"]:
        raise RuntimeError("Confirmatory split access was recorded in Step 9.")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    run_dir = Path(args.output).expanduser().resolve()
    if run_dir.exists() and args.overwrite:
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "common" / "splits").mkdir(parents=True, exist_ok=True)
    (run_dir / "common" / "splits" / "split_hash.txt").write_text(
        "a" * 64 + "\n", encoding="utf-8"
    )

    config = json.loads((PROJECT_ROOT / "configs" / "experiment.json").read_text())
    config["_project_root"] = str(PROJECT_ROOT)
    config["evaluation"]["save_plots"] = True
    config["evaluation"]["calibration_bins"] = 5
    config["super_learner"]["bootstrap_repeats"] = 10
    config["super_learner"]["seed"] = 42

    for task_name, seed in (
        (STAGE1_MORTALITY_TASK.task_name, 31),
        (STAGE2_LONG_STAY_TASK.task_name, 41),
    ):
        _write_stage_predictions(run_dir, task_name, seed=seed)
        _write_step8_evidence(run_dir, task_name)
        result = run_stage_super_learner(
            run_dir=run_dir,
            task_name=task_name,
            config=config,
        )
        if result.confirmatory_splits_accessed:
            raise RuntimeError("Super Learner smoke accessed confirmatory splits.")
        _verify_output(run_dir, task_name)
        print(
            f"[OK] {task_name}: selected {list(result.selected_members)} "
            f"with threshold {result.validation_decision_threshold:.6f}"
        )

    print("[OK] Completed Stage1 and Stage2 NNLS Super Learner smoke runs.")
    print("[OK] Validation meta/threshold patient separation was verified.")
    print("[OK] Artifact reload and base prediction hash contracts were verified.")
    print("[OK] Internal Test and eICU were not accessed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
