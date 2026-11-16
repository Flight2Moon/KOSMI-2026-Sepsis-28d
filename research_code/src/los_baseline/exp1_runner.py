from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .config import resolve_path
from .evaluation_multiclass import (
    BootstrapSettings,
    probabilities_from_prediction_frame,
    write_multiclass_evaluation_artifacts,
)
from .exp123_data import TaskDataBundle, build_task_data_bundle, load_feature_frame
from .optimization import run_optimization_study
from .hardware import apply_acceleration_to_model_config
from .progress import progress_iter, tracked_operation
from .runner_common import (
    load_json_object,
    load_shared_split_hash,
    select_validation_reference_model,
    validate_selected_hyperparameters,
)
from .task_spec import EXP1_FLAT_3CLASS_TASK
from .training_engine import TrainingEngineResult, train_and_evaluate_single_model
from .utils import atomic_write_json, environment_info


@dataclass(frozen=True)
class Exp1ModelResult:
    model_name: str
    output_dir: str
    selected_hyperparameters_path: str
    selected_hyperparameters_sha256: str
    validation_macro_auroc: float
    validation_macro_auprc: float
    validation_multiclass_brier: float
    validation_macro_f1: float
    training_seconds: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Exp1RunResult:
    run_id: str
    output_dir: str
    shared_split_hash: str
    representative_model: str
    model_results: tuple[Exp1ModelResult, ...]
    confirmatory_splits_accessed: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)



def _validate_selected_hyperparameters(
    *,
    path: Path,
    model_name: str,
    shared_split_hash: str | None,
) -> tuple[dict[str, Any], str]:
    """Backward-compatible wrapper around the shared runner validator."""
    return validate_selected_hyperparameters(
        path=path,
        model_name=model_name,
        task=EXP1_FLAT_3CLASS_TASK,
        shared_split_hash=shared_split_hash,
    )


def load_exp1_discovery_task_data(
    *,
    config: Mapping[str, Any],
    run_dir: Path,
) -> TaskDataBundle:
    """Load only Train and Validation data for EXP1 discovery/model selection."""
    feature_names = tuple(str(value) for value in config["data"]["feature_columns"])
    mimic_path = resolve_path(dict(config), config["paths"]["mimic_csv"])
    feature_frame = load_feature_frame(mimic_path, feature_names)
    split_path = run_dir / "common" / "splits" / "split_assignments.csv.gz"
    if not split_path.is_file():
        raise FileNotFoundError(f"Common split assignment was not found: {split_path}")
    split_assignment = pd.read_csv(split_path, compression="infer", low_memory=False)
    shared_split_hash = load_shared_split_hash(run_dir)
    return build_task_data_bundle(
        feature_frame=feature_frame,
        split_assignment=split_assignment,
        task=EXP1_FLAT_3CLASS_TASK,
        feature_names=feature_names,
        shared_split_hash=shared_split_hash,
        split_names=("train", "validation"),
    )



def run_exp1_optimization(
    *,
    task_data: TaskDataBundle,
    config: Mapping[str, Any],
    run_dir: Path,
    model_names: Sequence[str] | None = None,
    search_stages: Sequence[str] = ("smoke", "coarse", "fine"),
) -> dict[str, Any]:
    if task_data.task.task_name != EXP1_FLAT_3CLASS_TASK.task_name:
        raise ValueError("run_exp1_optimization requires EXP1 task data.")
    configured = list(config["experiments"]["exp1_flat_3class"]["models"])
    selected_models = list(model_names) if model_names is not None else configured
    unknown = sorted(set(selected_models) - set(configured))
    if unknown:
        raise ValueError(f"Models are not configured for EXP1: {unknown}")

    results: dict[str, Any] = {}
    for model_name in progress_iter(
        selected_models,
        total=len(selected_models),
        description="EXP1 models",
        unit="model",
        details={"task": "EXP1"},
    ):
        base_config = apply_acceleration_to_model_config(
            project_config=config,
            model_name=model_name,
            model_config=dict(config["models"][model_name]),
        )
        output_dir = (
            run_dir
            / "optimization"
            / EXP1_FLAT_3CLASS_TASK.task_name
            / model_name
        )
        stages, selection = run_optimization_study(
            task_data=task_data,
            model_name=model_name,
            base_model_config=base_config,
            config=config,
            output_dir=output_dir,
            search_stages=search_stages,
        )
        results[model_name] = {
            "stage_results": [stage.to_dict() for stage in stages],
            "selection": selection.to_dict(),
        }
    manifest = {
        "step": "exp1_flat_3class_optimization",
        "task": EXP1_FLAT_3CLASS_TASK.to_dict(),
        "models": selected_models,
        "shared_split_hash": task_data.shared_split_hash,
        "data_access": {
            "train": True,
            "validation": True,
            "internal_test": False,
            "external_eicu": False,
        },
        "results": results,
        "environment": environment_info(),
    }
    atomic_write_json(
        run_dir
        / "optimization"
        / EXP1_FLAT_3CLASS_TASK.task_name
        / "optimization_manifest.json",
        manifest,
    )
    return manifest


def select_exp1_representative_model(
    model_results: Sequence[Exp1ModelResult],
    *,
    representative_metric: str,
    practical_tie_margin: float,
    configured_model_order: Sequence[str],
) -> tuple[Exp1ModelResult, list[dict[str, Any]]]:
    return select_validation_reference_model(
        model_results,
        reference_metric=representative_metric,
        metric_attributes={
            "macro_auroc": "validation_macro_auroc",
            "macro_auprc": "validation_macro_auprc",
            "macro_f1": "validation_macro_f1",
        },
        secondary_attribute="validation_multiclass_brier",
        practical_tie_margin=practical_tie_margin,
        configured_model_order=configured_model_order,
        selected_flag="selected",
    )


def run_exp1_flat_multiclass(
    *,
    task_data: TaskDataBundle,
    config: Mapping[str, Any],
    run_id: str,
    run_dir: Path,
    model_names: Sequence[str] | None = None,
    bootstrap_repeats: int | None = None,
    overwrite: bool = False,
) -> Exp1RunResult:
    """Train and select EXP1 models using Train and Validation only."""
    if task_data.task.task_name != EXP1_FLAT_3CLASS_TASK.task_name:
        raise ValueError("run_exp1_flat_multiclass requires EXP1 task data.")
    forbidden = sorted(set(task_data.splits) - {"train", "validation"})
    if forbidden:
        raise ValueError(
            "EXP1 discovery training must not receive confirmatory splits: "
            f"{forbidden}"
        )

    configured_models = list(config["experiments"]["exp1_flat_3class"]["models"])
    selected_models = list(model_names) if model_names is not None else configured_models
    unknown = sorted(set(selected_models) - set(configured_models))
    if unknown:
        raise ValueError(f"Models are not configured for EXP1: {unknown}")

    output_dir = run_dir / "EXP1_flat_3class"
    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(f"EXP1 output already exists: {output_dir}")
        raise RuntimeError(
            "EXP1 model outputs are immutable. Use a new Run ID instead of overwrite."
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    evaluation_config = config["evaluation"]
    repeats = int(
        bootstrap_repeats
        if bootstrap_repeats is not None
        else evaluation_config.get("multiclass_bootstrap_repeats", 500)
    )
    bootstrap_settings = BootstrapSettings(
        repeats=repeats,
        confidence_level=float(
            evaluation_config.get("bootstrap_confidence_level", 0.95)
        ),
        seed=int(config["split"]["seed"]),
        minimum_successful_repeats=min(100, repeats),
    )

    model_results: list[Exp1ModelResult] = []
    selected_parameter_hashes: dict[str, str] = {}
    for model_name in progress_iter(
        selected_models,
        total=len(selected_models),
        description="EXP1 models",
        unit="model",
        details={"task": "EXP1"},
    ):
        selection_path = (
            run_dir
            / "optimization"
            / EXP1_FLAT_3CLASS_TASK.task_name
            / model_name
            / "selected_hyperparameters.json"
        )
        model_config, selection_hash = validate_selected_hyperparameters(
            path=selection_path,
            model_name=model_name,
            task=EXP1_FLAT_3CLASS_TASK,
            shared_split_hash=task_data.shared_split_hash,
        )
        selected_parameter_hashes[model_name] = selection_hash
        model_output = output_dir / model_name
        engine_result: TrainingEngineResult = train_and_evaluate_single_model(
            task_data=task_data,
            model_name=model_name,
            model_config=model_config,
            output_dir=model_output,
            seed=int(config["split"]["seed"]),
            project_config=config,
        )

        for split_name in ("train", "validation"):
            prediction_path = (
                model_output
                / "evaluation"
                / split_name
                / "predictions.csv.gz"
            )
            prediction_frame = pd.read_csv(
                prediction_path, compression="infer", low_memory=False
            )
            probabilities = probabilities_from_prediction_frame(
                prediction_frame, EXP1_FLAT_3CLASS_TASK.num_classes
            )
            write_multiclass_evaluation_artifacts(
                y_true=prediction_frame["y_true"].to_numpy(dtype=np.int64),
                probabilities=probabilities,
                class_names=EXP1_FLAT_3CLASS_TASK.class_names,
                output_dir=model_output / "evaluation" / split_name,
                split_name=split_name,
                model_name=model_name,
                calibration_bins=int(evaluation_config["calibration_bins"]),
                bootstrap_settings=(
                    bootstrap_settings if split_name == "validation" else None
                ),
                save_plots=bool(evaluation_config.get("save_plots", True)),
            )

        validation = engine_result.split_metrics["validation"]
        required_metrics = [
            "macro_auroc",
            "macro_auprc",
            "multiclass_brier",
            "macro_f1",
        ]
        missing = [name for name in required_metrics if validation.get(name) is None]
        if missing:
            raise RuntimeError(
                f"Validation metrics are missing for {model_name}: {missing}"
            )
        model_results.append(
            Exp1ModelResult(
                model_name=model_name,
                output_dir=str(model_output),
                selected_hyperparameters_path=str(selection_path),
                selected_hyperparameters_sha256=selection_hash,
                validation_macro_auroc=float(validation["macro_auroc"]),
                validation_macro_auprc=float(validation["macro_auprc"]),
                validation_multiclass_brier=float(
                    validation["multiclass_brier"]
                ),
                validation_macro_f1=float(validation["macro_f1"]),
                training_seconds=float(engine_result.training_seconds),
            )
        )

    representative_metric = str(
        config["experiments"]["exp1_flat_3class"]["representative_metric"]
    )
    practical_tie_margin = float(
        config["optimization"]["selection"]["practical_tie_margin"]
    )
    representative, selection_table = select_exp1_representative_model(
        model_results,
        representative_metric=representative_metric,
        practical_tie_margin=practical_tie_margin,
        configured_model_order=configured_models,
    )
    pd.DataFrame(selection_table).to_csv(
        output_dir / "model_selection_validation.csv", index=False
    )
    pd.DataFrame([result.to_dict() for result in model_results]).to_csv(
        output_dir / "model_summary_validation.csv", index=False
    )

    representative_payload = {
        "task_name": EXP1_FLAT_3CLASS_TASK.task_name,
        "selected_on_split": "validation",
        "representative_metric": representative_metric,
        "practical_tie_margin": practical_tie_margin,
        "selected_model": representative.model_name,
        "selected_model_validation_metrics": representative.to_dict(),
        "candidate_models": selected_models,
        "confirmatory_splits_accessed": False,
        "shared_split_hash": task_data.shared_split_hash,
    }
    representative_path = output_dir / "representative_model.json"
    atomic_write_json(representative_path, representative_payload)
    (output_dir / "representative_model.json.sha256").write_text(
        sha256_file(representative_path) + "\n", encoding="utf-8"
    )

    frozen_design = {
        "design_type": "exp1_flat_3class_discovery",
        "task": EXP1_FLAT_3CLASS_TASK.to_dict(),
        "shared_split_hash": task_data.shared_split_hash,
        "feature_names": list(task_data.feature_names),
        "model_names": selected_models,
        "selected_hyperparameter_sha256": selected_parameter_hashes,
        "representative_model": representative.model_name,
        "representative_model_sha256": sha256_file(representative_path),
        "selection_split": "validation",
        "confirmatory_splits_accessed": False,
        "forbidden_splits": ["internal_test", "external_eicu"],
        "bootstrap_settings": bootstrap_settings.to_dict(),
    }
    frozen_path = output_dir / "frozen_exp1_design.json"
    atomic_write_json(frozen_path, frozen_design)
    (output_dir / "frozen_exp1_design.json.sha256").write_text(
        sha256_file(frozen_path) + "\n", encoding="utf-8"
    )

    manifest = {
        "step": "exp1_flat_3class",
        "run_id": run_id,
        "status": "discovery_completed",
        "task": EXP1_FLAT_3CLASS_TASK.to_dict(),
        "data_access": {
            "train": True,
            "validation": True,
            "internal_test": False,
            "external_eicu": False,
        },
        "model_results": [result.to_dict() for result in model_results],
        "representative_model": representative_payload,
        "frozen_design_path": str(frozen_path),
        "environment": environment_info(),
        "next_required_step": "stage1_mortality",
    }
    atomic_write_json(output_dir / "exp1_manifest.json", manifest)

    result = Exp1RunResult(
        run_id=run_id,
        output_dir=str(output_dir),
        shared_split_hash=str(task_data.shared_split_hash),
        representative_model=representative.model_name,
        model_results=tuple(model_results),
        confirmatory_splits_accessed=False,
    )
    atomic_write_json(output_dir / "exp1_result.json", result.to_dict())
    return result
