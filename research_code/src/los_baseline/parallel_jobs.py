from __future__ import annotations

import json
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .evaluation_binary import (
    BinaryBootstrapSettings,
    probabilities_from_binary_prediction_frame,
    write_binary_evaluation_artifacts,
)
from .evaluation_multiclass import (
    BootstrapSettings,
    probabilities_from_prediction_frame,
    write_multiclass_evaluation_artifacts,
)
from .exp1_runner import (
    Exp1ModelResult,
    Exp1RunResult,
    _validate_selected_hyperparameters as validate_exp1_hyperparameters,
    load_exp1_discovery_task_data,
    select_exp1_representative_model,
)
from .exp3_runner import create_exp3_artifact_reference
from .hardware import apply_acceleration_to_model_config
from .optimization import run_optimization_study
from .hpo import run_optuna_optimization_study
from .stage1_runner import (
    Stage1ModelResult,
    Stage1RunResult,
    _stage1_configuration,
    _validate_selected_hyperparameters as validate_stage1_hyperparameters,
    load_stage1_discovery_task_data,
    select_stage1_reference_model,
)
from .stage2_runner import (
    Stage2ModelResult,
    Stage2RunResult,
    _stage2_configuration,
    _validate_selected_hyperparameters as validate_stage2_hyperparameters,
    load_stage2_discovery_task_data,
    select_stage2_reference_model,
)
from .task_spec import (
    EXP1_FLAT_3CLASS_TASK,
    STAGE1_MORTALITY_TASK,
    STAGE2_LONG_STAY_TASK,
)
from .training_engine import train_and_evaluate_single_model
from .utils import atomic_write_json, environment_info, sha256_file


TASKS: dict[str, dict[str, Any]] = {
    "exp1": {
        "task": EXP1_FLAT_3CLASS_TASK,
        "output_relative": Path("EXP1_flat_3class"),
        "loader": load_exp1_discovery_task_data,
    },
    "stage1": {
        "task": STAGE1_MORTALITY_TASK,
        "output_relative": Path("EXP2_two_stage") / "stage1_mortality",
        "loader": load_stage1_discovery_task_data,
    },
    "stage2": {
        "task": STAGE2_LONG_STAY_TASK,
        "output_relative": Path("EXP2_two_stage") / "stage2_long_stay",
        "loader": load_stage2_discovery_task_data,
    },
}


def _models_for_task(config: Mapping[str, Any], task_key: str) -> list[str]:
    if task_key == "exp1":
        return list(config["experiments"]["exp1_flat_3class"]["models"])
    if task_key == "stage1":
        return list(_stage1_configuration(config)["models"])
    if task_key == "stage2":
        return list(_stage2_configuration(config)["models"])
    raise KeyError(task_key)


def _task_data(config: Mapping[str, Any], run_dir: Path, task_key: str):
    return TASKS[task_key]["loader"](config=config, run_dir=run_dir)


def _completion_lock(path: Path, payload: Mapping[str, Any]) -> None:
    atomic_write_json(path, dict(payload))
    path.with_name(path.name + ".sha256").write_text(
        sha256_file(path) + "\n", encoding="utf-8"
    )


def _safe_remove_partial(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)


def run_model_optimization_job(
    *,
    task_key: str,
    model_name: str,
    config: Mapping[str, Any],
    run_id: str,
    run_dir: Path,
    search_stages: Sequence[str],
    resume: bool = True,
) -> dict[str, Any]:
    task_data = _task_data(config, run_dir, task_key)
    configured = _models_for_task(config, task_key)
    if model_name not in configured:
        raise ValueError(f"{model_name} is not configured for {task_key}.")
    task = TASKS[task_key]["task"]
    output_dir = run_dir / "optimization" / task.task_name / model_name
    lock_path = output_dir / "MODEL_OPTIMIZATION_COMPLETED.lock.json"
    if lock_path.is_file() and resume:
        return json.loads(lock_path.read_text(encoding="utf-8"))
    base_config = apply_acceleration_to_model_config(
        project_config=config,
        model_name=model_name,
        model_config=dict(config["models"][model_name]),
    )
    optimizer_engine = str(
        config.get("optimization_v2", {}).get("engine", "legacy_random")
    ).lower()
    if optimizer_engine == "optuna_tpe":
        stages, selection = run_optuna_optimization_study(
            task_data=task_data,
            model_name=model_name,
            base_model_config=base_config,
            config=config,
            output_dir=output_dir,
            search_stages=search_stages,
        )
    else:
        stages, selection = run_optimization_study(
            task_data=task_data,
            model_name=model_name,
            base_model_config=base_config,
            config=config,
            output_dir=output_dir,
            search_stages=search_stages,
        )
    payload = {
        "job_type": "model_optimization",
        "run_id": run_id,
        "task_key": task_key,
        "task_name": task.task_name,
        "model_name": model_name,
        "shared_split_hash": task_data.shared_split_hash,
        "stage_results": [item.to_dict() for item in stages],
        "selection": selection.to_dict(),
        "selected_hyperparameters_sha256": sha256_file(
            output_dir / "selected_hyperparameters.json"
        ),
        "confirmatory_splits_accessed": False,
        "environment": environment_info(),
    }
    _completion_lock(lock_path, payload)
    return payload


def finalize_optimization_task(
    *,
    task_key: str,
    config: Mapping[str, Any],
    run_id: str,
    run_dir: Path,
) -> dict[str, Any]:
    task_data = _task_data(config, run_dir, task_key)
    task = TASKS[task_key]["task"]
    model_names = _models_for_task(config, task_key)
    results: dict[str, Any] = {}
    missing: list[str] = []
    for model_name in model_names:
        model_dir = run_dir / "optimization" / task.task_name / model_name
        lock = model_dir / "MODEL_OPTIMIZATION_COMPLETED.lock.json"
        selected = model_dir / "selected_hyperparameters.json"
        if not lock.is_file() or not selected.is_file():
            missing.append(model_name)
            continue
        results[model_name] = json.loads(lock.read_text(encoding="utf-8"))
    if missing:
        raise RuntimeError(
            f"Optimization cannot be finalized for {task_key}; incomplete models: {missing}"
        )
    manifest = {
        "step": f"{task.task_name}_optimization",
        "task": task.to_dict(),
        "models": model_names,
        "shared_split_hash": task_data.shared_split_hash,
        "execution_mode": "resource_aware_model_jobs",
        "data_access": {
            "train": True,
            "validation": True,
            "internal_test": False,
            "external_eicu": False,
        },
        "results": results,
        "environment": environment_info(),
    }
    path = run_dir / "optimization" / task.task_name / "optimization_manifest.json"
    atomic_write_json(path, manifest)
    _completion_lock(
        path.with_name("OPTIMIZATION_TASK_COMPLETED.lock.json"),
        {
            "run_id": run_id,
            "task_key": task_key,
            "manifest_sha256": sha256_file(path),
            "models": model_names,
        },
    )
    return manifest


def _train_exp1_model(
    *,
    model_name: str,
    config: Mapping[str, Any],
    run_id: str,
    run_dir: Path,
    output_dir: Path,
    bootstrap_repeats: int | None,
) -> Exp1ModelResult:
    task_data = load_exp1_discovery_task_data(config=config, run_dir=run_dir)
    selection_path = (
        run_dir / "optimization" / EXP1_FLAT_3CLASS_TASK.task_name /
        model_name / "selected_hyperparameters.json"
    )
    model_config, selection_hash = validate_exp1_hyperparameters(
        path=selection_path,
        model_name=model_name,
        shared_split_hash=task_data.shared_split_hash,
    )
    engine = train_and_evaluate_single_model(
        task_data=task_data,
        model_name=model_name,
        model_config=model_config,
        output_dir=output_dir,
        seed=int(config["split"]["seed"]),
        integrity_config=config.get("artifact_integrity"),
        project_config=config,
    )
    evaluation = config["evaluation"]
    repeats = int(bootstrap_repeats if bootstrap_repeats is not None else evaluation.get("multiclass_bootstrap_repeats", 500))
    settings = BootstrapSettings(
        repeats=repeats,
        confidence_level=float(evaluation.get("bootstrap_confidence_level", 0.95)),
        seed=int(config["split"]["seed"]),
        minimum_successful_repeats=min(100, repeats),
    )
    for split_name in ("train", "validation"):
        evaluation_dir = output_dir / "evaluation" / split_name
        frame = pd.read_csv(evaluation_dir / "predictions.csv.gz", compression="infer", low_memory=False)
        write_multiclass_evaluation_artifacts(
            y_true=frame["y_true"].to_numpy(dtype=np.int64),
            probabilities=probabilities_from_prediction_frame(frame, EXP1_FLAT_3CLASS_TASK.num_classes),
            class_names=EXP1_FLAT_3CLASS_TASK.class_names,
            output_dir=evaluation_dir,
            split_name=split_name,
            model_name=model_name,
            calibration_bins=int(evaluation["calibration_bins"]),
            bootstrap_settings=settings if split_name == "validation" else None,
            save_plots=bool(evaluation.get("save_plots", True)),
        )
    validation = engine.split_metrics["validation"]
    result = Exp1ModelResult(
        model_name=model_name,
        output_dir=str(output_dir),
        selected_hyperparameters_path=str(selection_path),
        selected_hyperparameters_sha256=selection_hash,
        validation_macro_auroc=float(validation["macro_auroc"]),
        validation_macro_auprc=float(validation["macro_auprc"]),
        validation_multiclass_brier=float(validation["multiclass_brier"]),
        validation_macro_f1=float(validation["macro_f1"]),
        training_seconds=float(engine.training_seconds),
    )
    atomic_write_json(output_dir / "model_result.json", result.to_dict())
    return result


def _train_binary_model(
    *,
    task_key: str,
    model_name: str,
    config: Mapping[str, Any],
    run_id: str,
    run_dir: Path,
    output_dir: Path,
    bootstrap_repeats: int | None,
) -> Stage1ModelResult | Stage2ModelResult:
    if task_key == "stage1":
        task = STAGE1_MORTALITY_TASK
        task_data = load_stage1_discovery_task_data(config=config, run_dir=run_dir)
        validate = validate_stage1_hyperparameters
        result_cls = Stage1ModelResult
    elif task_key == "stage2":
        task = STAGE2_LONG_STAY_TASK
        task_data = load_stage2_discovery_task_data(config=config, run_dir=run_dir)
        validate = validate_stage2_hyperparameters
        result_cls = Stage2ModelResult
    else:
        raise KeyError(task_key)
    selection_path = run_dir / "optimization" / task.task_name / model_name / "selected_hyperparameters.json"
    model_config, selection_hash = validate(
        path=selection_path,
        model_name=model_name,
        shared_split_hash=task_data.shared_split_hash,
    )
    engine = train_and_evaluate_single_model(
        task_data=task_data,
        model_name=model_name,
        model_config=model_config,
        output_dir=output_dir,
        seed=int(config["split"]["seed"]),
        integrity_config=config.get("artifact_integrity"),
        project_config=config,
    )
    threshold = engine.validation_decision_threshold
    if threshold is None:
        raise RuntimeError(f"Validation threshold is missing for {task_key}/{model_name}.")
    evaluation = config["evaluation"]
    repeats = int(bootstrap_repeats if bootstrap_repeats is not None else evaluation.get("binary_bootstrap_repeats", 500))
    settings = BinaryBootstrapSettings(
        repeats=repeats,
        seed=int(config["split"]["seed"]),
        confidence_level=float(evaluation.get("bootstrap_confidence_level", 0.95)),
    )
    detailed: dict[str, Any] = {}
    for split_name in ("train", "validation"):
        evaluation_dir = output_dir / "evaluation" / split_name
        frame = pd.read_csv(evaluation_dir / "predictions.csv.gz", compression="infer", low_memory=False)
        detailed[split_name] = write_binary_evaluation_artifacts(
            y_true=frame["y_true"].to_numpy(dtype="int64"),
            probabilities=probabilities_from_binary_prediction_frame(frame),
            selected_threshold=float(threshold),
            default_threshold=float(evaluation["default_decision_threshold"]),
            output_dir=evaluation_dir,
            split_name=split_name,
            model_name=model_name,
            positive_class_name=task.class_names[1],
            calibration_bins=int(evaluation["calibration_bins"]),
            bootstrap_settings=settings if split_name == "validation" else None,
            save_plots=bool(evaluation.get("save_plots", True)),
        )
    metrics = detailed["validation"]["threshold_views"]["validation_selected"]
    result = result_cls(
        model_name=model_name,
        output_dir=str(output_dir),
        selected_hyperparameters_path=str(selection_path),
        selected_hyperparameters_sha256=selection_hash,
        validation_auroc=float(metrics["auroc"]),
        validation_auprc=float(metrics["auprc"]),
        validation_brier_score=float(metrics["brier_score"]),
        validation_log_loss=float(metrics["log_loss"]),
        validation_balanced_accuracy=float(metrics["balanced_accuracy"]),
        validation_f1=float(metrics["f1_score"]),
        validation_sensitivity=float(metrics["sensitivity_recall"]),
        validation_specificity=float(metrics["specificity"]),
        validation_decision_threshold=float(threshold),
        training_seconds=float(engine.training_seconds),
    )
    atomic_write_json(output_dir / "model_result.json", result.to_dict())
    return result


def run_model_training_job(
    *,
    task_key: str,
    model_name: str,
    config: Mapping[str, Any],
    run_id: str,
    run_dir: Path,
    bootstrap_repeats: int | None = None,
    resume: bool = True,
) -> dict[str, Any]:
    configured = _models_for_task(config, task_key)
    if model_name not in configured:
        raise ValueError(f"{model_name} is not configured for {task_key}.")
    output_dir = run_dir / TASKS[task_key]["output_relative"] / model_name
    lock_path = output_dir / "MODEL_TRAINING_COMPLETED.lock.json"
    if lock_path.is_file() and resume:
        return json.loads(lock_path.read_text(encoding="utf-8"))
    if output_dir.exists() and any(output_dir.iterdir()):
        _safe_remove_partial(output_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    if task_key == "exp1":
        result = _train_exp1_model(
            model_name=model_name,
            config=config,
            run_id=run_id,
            run_dir=run_dir,
            output_dir=output_dir,
            bootstrap_repeats=bootstrap_repeats,
        )
    else:
        result = _train_binary_model(
            task_key=task_key,
            model_name=model_name,
            config=config,
            run_id=run_id,
            run_dir=run_dir,
            output_dir=output_dir,
            bootstrap_repeats=bootstrap_repeats,
        )
    payload = {
        "job_type": "model_training",
        "run_id": run_id,
        "task_key": task_key,
        "model_name": model_name,
        "model_result": result.to_dict(),
        "engine_result_sha256": sha256_file(output_dir / "engine_result.json"),
        "artifact_integrity_sha256": sha256_file(output_dir / "artifact_integrity.json"),
        "confirmatory_splits_accessed": False,
    }
    _completion_lock(lock_path, payload)
    return payload


def _load_model_results(run_dir: Path, task_key: str, models: Sequence[str]):
    output_root = run_dir / TASKS[task_key]["output_relative"]
    rows = []
    missing = []
    for model in models:
        path = output_root / model / "model_result.json"
        lock = output_root / model / "MODEL_TRAINING_COMPLETED.lock.json"
        if not path.is_file() or not lock.is_file():
            missing.append(model)
            continue
        rows.append(json.loads(path.read_text(encoding="utf-8")))
    if missing:
        raise RuntimeError(f"Cannot finalize {task_key}; incomplete models: {missing}")
    return rows


def finalize_training_task(
    *,
    task_key: str,
    config: Mapping[str, Any],
    run_id: str,
    run_dir: Path,
) -> dict[str, Any]:
    models = _models_for_task(config, task_key)
    task_data = _task_data(config, run_dir, task_key)
    output_dir = run_dir / TASKS[task_key]["output_relative"]
    rows = _load_model_results(run_dir, task_key, models)
    evaluation = config["evaluation"]
    practical = float(config["optimization"]["selection"]["practical_tie_margin"])
    selected_hashes = {row["model_name"]: row["selected_hyperparameters_sha256"] for row in rows}
    if task_key == "exp1":
        results = [Exp1ModelResult(**row) for row in rows]
        representative_metric = str(config["experiments"]["exp1_flat_3class"]["representative_metric"])
        selected, selection_table = select_exp1_representative_model(
            results,
            representative_metric=representative_metric,
            practical_tie_margin=practical,
            configured_model_order=models,
        )
        pd.DataFrame(selection_table).to_csv(output_dir / "model_selection_validation.csv", index=False)
        pd.DataFrame(rows).to_csv(output_dir / "model_summary_validation.csv", index=False)
        representative_payload = {
            "task_name": EXP1_FLAT_3CLASS_TASK.task_name,
            "selected_on_split": "validation",
            "representative_metric": representative_metric,
            "practical_tie_margin": practical,
            "selected_model": selected.model_name,
            "selected_model_validation_metrics": selected.to_dict(),
            "candidate_models": models,
            "confirmatory_splits_accessed": False,
            "shared_split_hash": task_data.shared_split_hash,
        }
        representative_path = output_dir / "representative_model.json"
        atomic_write_json(representative_path, representative_payload)
        representative_path.with_name(representative_path.name + ".sha256").write_text(sha256_file(representative_path) + "\n", encoding="utf-8")
        repeats = int(evaluation.get("multiclass_bootstrap_repeats", 500))
        bootstrap = BootstrapSettings(repeats=repeats, confidence_level=float(evaluation.get("bootstrap_confidence_level", .95)), seed=int(config["split"]["seed"]), minimum_successful_repeats=min(100, repeats))
        frozen = {
            "design_type": "exp1_flat_3class_discovery",
            "task": EXP1_FLAT_3CLASS_TASK.to_dict(),
            "shared_split_hash": task_data.shared_split_hash,
            "feature_names": list(task_data.feature_names),
            "model_names": models,
            "selected_hyperparameter_sha256": selected_hashes,
            "representative_model": selected.model_name,
            "representative_model_sha256": sha256_file(representative_path),
            "selection_split": "validation",
            "confirmatory_splits_accessed": False,
            "forbidden_splits": ["internal_test", "external_eicu"],
            "bootstrap_settings": bootstrap.to_dict(),
            "execution_mode": "resource_aware_model_jobs",
        }
        frozen_path = output_dir / "frozen_exp1_design.json"
        atomic_write_json(frozen_path, frozen)
        frozen_path.with_name(frozen_path.name + ".sha256").write_text(sha256_file(frozen_path) + "\n", encoding="utf-8")
        manifest = {
            "step": "exp1_flat_3class",
            "run_id": run_id,
            "status": "discovery_completed",
            "task": EXP1_FLAT_3CLASS_TASK.to_dict(),
            "data_access": {"train": True, "validation": True, "internal_test": False, "external_eicu": False},
            "model_results": rows,
            "representative_model": representative_payload,
            "frozen_design_path": str(frozen_path),
            "execution_mode": "resource_aware_model_jobs",
            "environment": environment_info(),
            "next_required_step": "stage1_mortality",
        }
        atomic_write_json(output_dir / "exp1_manifest.json", manifest)
        result = Exp1RunResult(run_id=run_id, output_dir=str(output_dir), shared_split_hash=str(task_data.shared_split_hash), representative_model=selected.model_name, model_results=tuple(results), confirmatory_splits_accessed=False)
        atomic_write_json(output_dir / "exp1_result.json", result.to_dict())
        return result.to_dict()

    if task_key == "stage1":
        results = [Stage1ModelResult(**row) for row in rows]
        stage_config = _stage1_configuration(config)
        reference_metric = str(stage_config.get("reference_metric", "auprc"))
        selected, selection_table = select_stage1_reference_model(results, reference_metric=reference_metric, practical_tie_margin=practical, configured_model_order=models)
        task = STAGE1_MORTALITY_TASK
        prefix = "stage1"
    else:
        results = [Stage2ModelResult(**row) for row in rows]
        stage_config = _stage2_configuration(config)
        reference_metric = str(stage_config.get("reference_metric", "auroc"))
        selected, selection_table = select_stage2_reference_model(results, reference_metric=reference_metric, practical_tie_margin=practical, configured_model_order=models)
        task = STAGE2_LONG_STAY_TASK
        prefix = "stage2"
    pd.DataFrame(selection_table).to_csv(output_dir / "model_selection_validation.csv", index=False)
    pd.DataFrame(rows).to_csv(output_dir / "model_summary_validation.csv", index=False)
    reference_payload = {
        "task_name": task.task_name,
        "selected_on_split": "validation",
        "reference_metric": reference_metric,
        "practical_tie_margin": practical,
        "selected_model": selected.model_name,
        "selected_model_validation_metrics": selected.to_dict(),
        "candidate_models": models,
        "role": "validation_reference_only",
        "final_pipeline_model_selected": False,
        "final_pipeline_selection_step": f"{prefix}_super_learner",
        "confirmatory_splits_accessed": False,
        "shared_split_hash": task_data.shared_split_hash,
    }
    reference_path = output_dir / "validation_reference_model.json"
    atomic_write_json(reference_path, reference_payload)
    reference_path.with_name(reference_path.name + ".sha256").write_text(sha256_file(reference_path) + "\n", encoding="utf-8")
    repeats = int(evaluation.get("binary_bootstrap_repeats", 500))
    bootstrap = BinaryBootstrapSettings(repeats=repeats, seed=int(config["split"]["seed"]), confidence_level=float(evaluation.get("bootstrap_confidence_level", .95)))
    thresholds = {row["model_name"]: row["validation_decision_threshold"] for row in rows}
    frozen = {
        "design_type": f"{task.task_name}_discovery",
        "task": task.to_dict(),
        "shared_split_hash": task_data.shared_split_hash,
        "feature_names": list(task_data.feature_names),
        "model_names": models,
        "selected_hyperparameter_sha256": selected_hashes,
        "validation_thresholds": thresholds,
        "validation_reference_model": selected.model_name,
        "reference_model_is_not_final_pipeline_model": True,
        "selection_split": "validation",
        "confirmatory_splits_accessed": False,
        "forbidden_splits": ["internal_test", "external_eicu"],
        "bootstrap_settings": bootstrap.to_dict(),
        "execution_mode": "resource_aware_model_jobs",
    }
    if task_key == "stage2":
        frozen.update({
            "population_definition": "ground_truth_survivors_A_or_C",
            "stage1_predictions_used_for_training_filter": False,
            "survivor_population_per_split": {
                split: {
                    "sample_count": int(len(task_data.splits[split].labels)),
                    "class_counts": {str(code): int((task_data.splits[split].labels == code).sum()) for code in task.expected_classes},
                } for split in ("train", "validation")
            },
        })
    frozen_path = output_dir / f"frozen_{prefix}_design.json"
    atomic_write_json(frozen_path, frozen)
    frozen_path.with_name(frozen_path.name + ".sha256").write_text(sha256_file(frozen_path) + "\n", encoding="utf-8")
    manifest = {
        "step": task.task_name,
        "run_id": run_id,
        "status": "discovery_completed",
        "task": task.to_dict(),
        "data_access": {"train": True, "validation": True, "internal_test": False, "external_eicu": False},
        "model_results": rows,
        "validation_reference_model": reference_payload,
        "frozen_design_path": str(frozen_path),
        "execution_mode": "resource_aware_model_jobs",
        "environment": environment_info(),
        "next_required_step": "stage_heterogeneity" if task_key == "stage2" else "stage2_long_stay_and_exp3",
    }
    atomic_write_json(output_dir / f"{prefix}_manifest.json", manifest)
    if task_key == "stage1":
        result = Stage1RunResult(run_id=run_id, output_dir=str(output_dir), shared_split_hash=str(task_data.shared_split_hash), validation_reference_model=selected.model_name, model_results=tuple(results), confirmatory_splits_accessed=False)
        atomic_write_json(output_dir / "stage1_result.json", result.to_dict())
        return result.to_dict()
    exp3 = create_exp3_artifact_reference(run_id=run_id, run_dir=run_dir, model_names=models, shared_split_hash=str(task_data.shared_split_hash))
    result = Stage2RunResult(run_id=run_id, output_dir=str(output_dir), shared_split_hash=str(task_data.shared_split_hash), validation_reference_model=selected.model_name, model_results=tuple(results), confirmatory_splits_accessed=False, exp3_artifact_reference_path=str(Path(exp3.output_dir) / "artifact_reference.json"))
    atomic_write_json(output_dir / "stage2_result.json", result.to_dict())
    return result.to_dict()
