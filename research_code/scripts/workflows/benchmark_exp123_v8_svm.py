#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from los_baseline.config import load_config, resolve_path
from los_baseline.exp1_runner import load_exp1_discovery_task_data
from los_baseline.hpo.fixed_cv import load_fixed_folds
from los_baseline.optimization import _fit_predict_partition
from los_baseline.stage1_runner import load_stage1_discovery_task_data
from los_baseline.stage2_runner import load_stage2_discovery_task_data
from los_baseline.svm_runtime import svm_runtime_settings
from los_baseline.utils import atomic_write_json

TASKS = {
    "exp1": (load_exp1_discovery_task_data, "exp1_flat_3class"),
    "stage1": (load_stage1_discovery_task_data, "stage1_mortality"),
    "stage2": (load_stage2_discovery_task_data, "stage2_long_stay"),
}


def _stratified_subset(indices: np.ndarray, labels: np.ndarray, fraction: float, seed: int) -> np.ndarray:
    if not 0.0 < fraction <= 1.0:
        raise ValueError("train fraction must be in (0, 1].")
    if fraction >= 1.0:
        return np.asarray(indices, dtype=np.int64)
    rng = np.random.default_rng(seed)
    selected: list[int] = []
    local_labels = labels[indices]
    for class_code in sorted(np.unique(local_labels).astype(int).tolist()):
        members = np.asarray(indices[local_labels == class_code], dtype=np.int64)
        target = max(2, int(round(len(members) * fraction)))
        target = min(len(members), target)
        selected.extend(rng.choice(members, size=target, replace=False).astype(int).tolist())
    return np.asarray(sorted(selected), dtype=np.int64)


def _canonical_configs(config: dict[str, Any], train_count: int) -> dict[str, dict[str, Any]]:
    runtime = svm_runtime_settings(config)
    exact = runtime["exact_rbf"]
    approx = runtime["nystroem_linear"]
    exact_gamma = math.sqrt(float(exact["gamma_min"]) * float(exact["gamma_max"]))
    approx_gamma = math.sqrt(float(approx["gamma_min"]) * float(approx["gamma_max"]))
    components = min(
        max(int(approx["n_components_min"]), 512),
        int(approx["n_components_max"]),
        max(2, train_count),
    )
    common = {"training_policy": {"class_weight_mode": "none"}}
    return {
        "exact_rbf": {
            **common,
            "parameters": {
                "svm_backend": "exact_rbf",
                "kernel": "rbf",
                "C": 1.0,
                "gamma": exact_gamma,
                "probability": False,
                "decision_function_shape": "ovr",
                "cache_size": 2048,
                "max_iter": int(exact["max_iter"]),
                "tol": float(exact["tol"]),
            },
        },
        "nystroem_linear": {
            **common,
            "parameters": {
                "svm_backend": "nystroem_linear",
                "C": 1.0,
                "nystroem_gamma": approx_gamma,
                "nystroem_components": components,
                "linear_svc_max_iter": int(approx["max_iter"]),
                "linear_svc_tol": float(approx["tol"]),
                "probability": False,
                "decision_function_shape": "ovr",
            },
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train-only fixed-CV runtime benchmark for V8 Phase 4-B SVM backends."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--source-run-id", required=True)
    parser.add_argument("--task", choices=sorted(TASKS), default="exp1")
    parser.add_argument("--fractions", default=None, help="Comma-separated train fractions. Defaults to config.")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    config = load_config(Path(args.config).expanduser().resolve())
    runtime = svm_runtime_settings(config)
    if not runtime["enabled"]:
        raise RuntimeError("svm_runtime_optimization.enabled must be true for the Phase 4-B benchmark.")
    runs_dir = resolve_path(config, config["paths"]["runs_dir"])
    run_dir = runs_dir / args.source_run_id
    if not run_dir.is_dir():
        raise FileNotFoundError(run_dir)

    loader, task_name = TASKS[args.task]
    task_data = loader(config=config, run_dir=run_dir)
    train = task_data.splits["train"]
    labels = np.asarray(train.labels, dtype=np.int64)
    group_column = str(config.get("data", {}).get("patient_id_column", "subject_id"))
    groups = train.metadata[group_column].to_numpy()

    fixed_dir = run_dir / "common" / "fixed_cv" / task_name
    folds, fixed_manifest = load_fixed_folds(fixed_dir)
    primary = sorted((fold for fold in folds if fold.repeat == 1), key=lambda value: value.fold)
    if not primary:
        raise RuntimeError(f"No primary fixed-CV folds found: {fixed_dir}")
    benchmark_fold = primary[0]
    if set(groups[benchmark_fold.train_indices]) & set(groups[benchmark_fold.validation_indices]):
        raise RuntimeError("Benchmark fixed fold contains patient-group leakage.")

    raw_fractions = args.fractions
    if raw_fractions:
        fractions = [float(value.strip()) for value in raw_fractions.split(",") if value.strip()]
    else:
        fractions = [float(v) for v in runtime.get("benchmark", {}).get("train_fractions", [0.25, 0.5, 1.0])]
    fractions = sorted(set(fractions))
    for fraction in fractions:
        if not 0.0 < fraction <= 1.0:
            raise ValueError(f"Invalid fraction: {fraction}")

    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else run_dir / "V8_reporting" / "svm_runtime_benchmark" / task_name
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    plan_rows: list[dict[str, Any]] = []
    for fraction in fractions:
        fit_indices = _stratified_subset(
            np.asarray(benchmark_fold.train_indices, dtype=np.int64),
            labels,
            fraction,
            seed=int(runtime.get("benchmark", {}).get("seed", 42)) + int(round(fraction * 1000)),
        )
        configs = _canonical_configs(config, len(fit_indices))
        for backend, model_config in configs.items():
            plan_rows.append({
                "task": task_name,
                "backend": backend,
                "train_fraction": fraction,
                "train_rows": int(len(fit_indices)),
                "evaluation_rows": int(len(benchmark_fold.validation_indices)),
                "model_config_json": json.dumps(model_config, sort_keys=True),
            })
    pd.DataFrame(plan_rows).to_csv(output_dir / "benchmark_plan.csv", index=False)

    manifest: dict[str, Any] = {
        "schema_version": "exp123_v8_phase4b_svm_benchmark_v1",
        "source_run_id": args.source_run_id,
        "task": task_name,
        "fixed_cv_hash": fixed_manifest.get("split_hash"),
        "shared_split_hash": task_data.shared_split_hash,
        "benchmark_source": "train_fixed_cv_only",
        "project_validation_accessed": False,
        "internal_test_accessed": False,
        "external_eicu_accessed": False,
        "fractions": fractions,
        "runtime_policy": runtime,
        "executed": bool(args.execute),
    }
    if not args.execute:
        atomic_write_json(output_dir / "benchmark_manifest.json", manifest)
        print(f"[PLAN] SVM runtime benchmark written: {output_dir}")
        return 0

    result_rows: list[dict[str, Any]] = []
    for fraction in fractions:
        fit_indices = _stratified_subset(
            np.asarray(benchmark_fold.train_indices, dtype=np.int64),
            labels,
            fraction,
            seed=int(runtime.get("benchmark", {}).get("seed", 42)) + int(round(fraction * 1000)),
        )
        eval_indices = np.asarray(benchmark_fold.validation_indices, dtype=np.int64)
        configs = _canonical_configs(config, len(fit_indices))
        for backend, model_config in configs.items():
            _, metrics, diagnostics, effective = _fit_predict_partition(
                model_name="svm",
                model_config=model_config,
                task=task_data.task,
                feature_names=task_data.feature_names,
                train_features=train.raw_features.iloc[fit_indices].reset_index(drop=True),
                train_labels=labels[fit_indices],
                evaluation_features=train.raw_features.iloc[eval_indices].reset_index(drop=True),
                evaluation_labels=labels[eval_indices],
                seed=42,
                train_groups=groups[fit_indices],
            )
            row = {
                "task": task_name,
                "backend": backend,
                "train_fraction": fraction,
                "train_rows": int(len(fit_indices)),
                "evaluation_rows": int(len(eval_indices)),
                "fit_seconds": float(diagnostics.get("fit_seconds", float("nan"))),
                "prediction_kind": diagnostics.get("prediction_kind"),
                "svm_convergence_warning": bool(diagnostics.get("svm_convergence_warning", False)),
                "effective_config_json": json.dumps(effective, sort_keys=True, default=str),
                **{key: float(value) for key, value in metrics.items()},
            }
            result_rows.append(row)
            pd.DataFrame(result_rows).to_csv(output_dir / "benchmark_results.csv", index=False)
            print(
                f"[BENCH] {task_name} {backend} fraction={fraction:.2f} "
                f"n={len(fit_indices)} fit={row['fit_seconds']:.3f}s"
            )
    manifest["result_count"] = len(result_rows)
    manifest["executed"] = True
    atomic_write_json(output_dir / "benchmark_manifest.json", manifest)
    print(f"[OK] SVM runtime benchmark completed: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
