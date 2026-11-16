#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from los_baseline.stage_heterogeneity import (
    StageHeterogeneityConfig,
    run_stage_heterogeneity_confirmatory,
    run_stage_heterogeneity_discovery,
    stage_analysis_root,
    stage_prediction_path,
)
from los_baseline.task_spec import STAGE1_MORTALITY_TASK, STAGE2_LONG_STAY_TASK


MODELS = (
    "decision_tree",
    "logistic_regression",
    "random_forest",
    "svm",
    "xgboost",
)
SUBGROUPS = (
    "age",
    "sofa_score",
    "aps3",
    "charlson_comorbidity_index",
)


def _sigmoid(value: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(value, -20.0, 20.0)))


def _clinical_frame(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    age = rng.integers(25, 91, size=n)
    sofa = rng.integers(0, 13, size=n)
    aps3 = np.clip(15 + sofa * 5 + rng.normal(0, 12, size=n), 5, 130)
    cci = np.clip(rng.poisson(3.0, size=n) + (age >= 70), 0, 12)
    return pd.DataFrame(
        {
            "LOS": np.clip(rng.gamma(2.0, 5.0, size=n), 0.25, 50.0),
            "subject_id": np.arange(seed * 10000, seed * 10000 + n),
            "hadm_id": np.arange(seed * 20000, seed * 20000 + n),
            "stay_id": np.arange(seed * 30000, seed * 30000 + n),
            "age": age.astype(float),
            "sofa_score": sofa.astype(float),
            "aps3": aps3.astype(float),
            "charlson_comorbidity_index": cci.astype(float),
        }
    )


def _labels(frame: pd.DataFrame, task_name: str, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if task_name == STAGE1_MORTALITY_TASK.task_name:
        logit = (
            -3.0
            + 0.025 * (frame["age"].to_numpy() - 60.0)
            + 0.18 * frame["sofa_score"].to_numpy()
            + 0.012 * (frame["aps3"].to_numpy() - 40.0)
            + 0.08 * frame["charlson_comorbidity_index"].to_numpy()
        )
    else:
        logit = (
            -1.8
            + 0.012 * (frame["age"].to_numpy() - 60.0)
            + 0.11 * frame["sofa_score"].to_numpy()
            + 0.010 * (frame["aps3"].to_numpy() - 40.0)
            + 0.05 * frame["charlson_comorbidity_index"].to_numpy()
        )
    probability = _sigmoid(logit)
    labels = rng.binomial(1, probability).astype(np.int32)
    if labels.sum() < 30:
        order = np.argsort(probability)[-30:]
        labels[order] = 1
    if (labels == 0).sum() < 30:
        order = np.argsort(probability)[:30]
        labels[order] = 0
    return labels


def _model_probabilities(
    frame: pd.DataFrame,
    y: np.ndarray,
    model_name: str,
    task_name: str,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    age = frame["age"].to_numpy(dtype=float)
    sofa = frame["sofa_score"].to_numpy(dtype=float)
    aps3 = frame["aps3"].to_numpy(dtype=float)
    cci = frame["charlson_comorbidity_index"].to_numpy(dtype=float)
    signal = (
        -1.1
        + 1.9 * y
        + 0.008 * (age - 60.0)
        + 0.025 * sofa
        + 0.003 * (aps3 - 40.0)
        + 0.015 * cci
    )
    if task_name == STAGE2_LONG_STAY_TASK.task_name:
        signal += 0.12 * (aps3 >= 55)
    if model_name == "xgboost":
        logit = signal + 0.20 * (sofa >= 4) + rng.normal(0, 0.55, len(y))
    elif model_name == "logistic_regression":
        logit = signal + 0.10 * (age < 60) + rng.normal(0, 0.65, len(y))
    elif model_name == "random_forest":
        logit = signal - 0.18 * (aps3 >= 55) + rng.normal(0, 0.85, len(y))
    elif model_name == "decision_tree":
        logit = signal + rng.normal(0, 1.20, len(y))
    elif model_name == "svm":
        prevalence = float(np.mean(y))
        return np.clip(
            prevalence + rng.normal(0, 0.008, len(y)), 0.01, 0.99
        )
    else:
        raise KeyError(model_name)
    return np.clip(_sigmoid(logit), 1e-5, 1 - 1e-5)


def _write_predictions(
    *,
    run_dir: Path,
    task_name: str,
    split: str,
    source_indices: np.ndarray,
    dataset_frame: pd.DataFrame,
    seed: int,
) -> None:
    selected = dataset_frame.iloc[source_indices].reset_index(drop=True)
    y = _labels(selected, task_name, seed)
    for model_index, model_name in enumerate(MODELS):
        probability = _model_probabilities(
            selected,
            y,
            model_name,
            task_name,
            seed + 100 + model_index,
        )
        output = pd.DataFrame(
            {
                "source_row_index": source_indices,
                "subject_id": selected["subject_id"].to_numpy(),
                "split": split,
                "task_name": task_name,
                "model_name": model_name,
                "y_true": y,
                "positive_probability": probability,
                "predicted_at_0_5": (probability >= 0.5).astype(np.int32),
            }
        )
        path = stage_prediction_path(
            run_dir, task_name, model_name, split
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        output.to_csv(path, index=False, compression="gzip")


def _assert_output(root: Path, task_name: str) -> None:
    discovery = stage_analysis_root(root, task_name) / "discovery_validation"
    required = [
        "subgroup_metrics.csv",
        "model_eligibility_per_stratum.csv",
        "best_model_per_stratum.csv",
        "interaction_tests.json",
        "strata_definitions.json",
        "library_candidate_evidence.csv",
        "library_candidates.json",
        "frozen_design.json",
        "frozen_design.json.sha256",
        "run_config.json",
    ]
    for filename in required:
        if not (discovery / filename).is_file():
            raise FileNotFoundError(discovery / filename)
    metrics = pd.read_csv(discovery / "subgroup_metrics.csv")
    if set(metrics["task_name"].unique()) != {task_name}:
        raise RuntimeError("Task name was not preserved in subgroup metrics.")
    eligibility = pd.read_csv(discovery / "model_eligibility_per_stratum.csv")
    if not (~eligibility["eligible_by_auroc_guard"].astype(bool)).any():
        raise RuntimeError("Smoke data did not exercise the AUROC margin guard.")
    rankings = pd.read_csv(discovery / "best_model_per_stratum.csv")
    allowed_rules = {"auroc_guard_then_lowest_brier", "no_eligible_model"}
    if not set(rankings["selection_rule"].astype(str)).issubset(allowed_rules):
        raise RuntimeError("Unexpected Stage heterogeneity ranking rule.")
    if not (rankings["selection_rule"] == "auroc_guard_then_lowest_brier").any():
        raise RuntimeError("No analyzable stratum exercised guarded Brier ranking.")
    design = json.loads((discovery / "frozen_design.json").read_text())
    if design["task_name"] != task_name:
        raise RuntimeError("Frozen design task mismatch.")
    if not math.isclose(float(design["auroc_margin_guard"]), 0.02):
        raise RuntimeError("Frozen design AUROC guard mismatch.")
    candidates = json.loads((discovery / "library_candidates.json").read_text())
    if candidates["selection_is_final_super_learner"] is not False:
        raise RuntimeError("Step 8 must not fit the final Super Learner.")
    for split in ("internal_test", "external_eicu"):
        confirmatory = stage_analysis_root(root, task_name) / "confirmatory" / split
        config = json.loads((confirmatory / "run_config.json").read_text())
        if config["no_refitting_of_strata"] is not True:
            raise RuntimeError("Confirmatory analysis refit the strata.")
        if config["stage9_library_may_be_changed_from_this_result"] is not False:
            raise RuntimeError("Confirmatory results may not change Stage9 library design.")


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
        "f" * 64 + "\n", encoding="utf-8"
    )

    mimic = _clinical_frame(500, 11)
    eicu = _clinical_frame(240, 23)
    mimic_path = run_dir / "mimic_smoke.csv"
    eicu_path = run_dir / "eicu_smoke.csv"
    mimic.to_csv(mimic_path, index=False)
    eicu.to_csv(eicu_path, index=False)

    task_indices = {
        STAGE1_MORTALITY_TASK.task_name: {
            "validation": np.arange(0, 300, dtype=np.int64),
            "internal_test": np.arange(300, 500, dtype=np.int64),
            "external_eicu": np.arange(0, 240, dtype=np.int64),
        },
        STAGE2_LONG_STAY_TASK.task_name: {
            "validation": np.arange(0, 240, dtype=np.int64),
            "internal_test": np.arange(300, 460, dtype=np.int64),
            "external_eicu": np.arange(0, 200, dtype=np.int64),
        },
    }
    for task_index, (task_name, splits) in enumerate(task_indices.items()):
        for split_index, (split, indices) in enumerate(splits.items()):
            dataset = eicu if split == "external_eicu" else mimic
            _write_predictions(
                run_dir=run_dir,
                task_name=task_name,
                split=split,
                source_indices=indices,
                dataset_frame=dataset,
                seed=1000 + task_index * 100 + split_index * 10,
            )
        config = StageHeterogeneityConfig(
            run_dir=run_dir,
            task_name=task_name,
            model_names=MODELS,
            subgroup_variables=SUBGROUPS,
            n_bins=3,
            minimum_bin_size=30,
            minimum_class_count=10,
            auroc_margin_guard=0.02,
            brier_tie_margin=0.005,
            trend_bins=20,
        )
        discovery = run_stage_heterogeneity_discovery(
            config,
            dataset_path=mimic_path,
        )
        frozen = discovery / "frozen_design.json"
        run_stage_heterogeneity_confirmatory(
            run_dir=run_dir,
            task_name=task_name,
            split="internal_test",
            dataset_path=mimic_path,
            frozen_design_path=frozen,
        )
        run_stage_heterogeneity_confirmatory(
            run_dir=run_dir,
            task_name=task_name,
            split="external_eicu",
            dataset_path=eicu_path,
            frozen_design_path=frozen,
        )
        _assert_output(run_dir, task_name)

    print("[OK] Completed Stage1 and Stage2 heterogeneity discovery analyses.")
    print("[OK] Completed frozen Internal Test and eICU confirmatory smoke analyses.")
    print("[OK] AUROC margin guard and Stage9 candidate evidence were verified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
