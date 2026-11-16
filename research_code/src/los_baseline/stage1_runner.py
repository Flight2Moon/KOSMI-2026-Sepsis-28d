from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from .config import resolve_path
from .evaluation_binary import (
    BinaryBootstrapSettings,
    probabilities_from_binary_prediction_frame,
    write_binary_evaluation_artifacts,
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
from .task_spec import STAGE1_MORTALITY_TASK
from .training_engine import TrainingEngineResult, train_and_evaluate_single_model
from .utils import atomic_write_json, environment_info


@dataclass(frozen=True)
class Stage1ModelResult:
    model_name: str
    output_dir: str
    selected_hyperparameters_path: str
    selected_hyperparameters_sha256: str
    validation_auroc: float
    validation_auprc: float
    validation_brier_score: float
    validation_log_loss: float
    validation_balanced_accuracy: float
    validation_f1: float
    validation_sensitivity: float
    validation_specificity: float
    validation_decision_threshold: float
    training_seconds: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Stage1RunResult:
    run_id: str
    output_dir: str
    shared_split_hash: str
    validation_reference_model: str
    model_results: tuple[Stage1ModelResult, ...]
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
        task=STAGE1_MORTALITY_TASK,
        shared_split_hash=shared_split_hash,
    )


def load_stage1_discovery_task_data(
    *,
    config: Mapping[str, Any],
    run_dir: Path,
) -> TaskDataBundle:
    """Load only Train and Validation for Stage1 discovery and selection."""
    feature_names = tuple(str(value) for value in config["data"]["feature_columns"])
    mimic_path = resolve_path(dict(config), config["paths"]["mimic_csv"])
    feature_frame = load_feature_frame(mimic_path, feature_names)
    split_path = run_dir / "common" / "splits" / "split_assignments.csv.gz"
    if not split_path.is_file():
        raise FileNotFoundError(f"Common split assignment was not found: {split_path}")
    split_assignment = pd.read_csv(split_path, compression="infer", low_memory=False)
    return build_task_data_bundle(
        feature_frame=feature_frame,
        split_assignment=split_assignment,
        task=STAGE1_MORTALITY_TASK,
        feature_names=feature_names,
        shared_split_hash=load_shared_split_hash(run_dir),
        split_names=("train", "validation"),
    )


def _stage1_configuration(config: Mapping[str, Any]) -> Mapping[str, Any]:
    exp2 = config["experiments"]["exp2_two_stage"]
    stage1 = exp2.get("stage1_mortality")
    if not isinstance(stage1, Mapping):
        raise KeyError("experiments.exp2_two_stage.stage1_mortality is required.")
    return stage1



def run_stage1_optimization(
    *,
    task_data: TaskDataBundle,
    config: Mapping[str, Any],
    run_dir: Path,
    model_names: Sequence[str] | None = None,
    search_stages: Sequence[str] = ("smoke", "coarse", "fine"),
) -> dict[str, Any]:
    if task_data.task.task_name != STAGE1_MORTALITY_TASK.task_name:
        raise ValueError("run_stage1_optimization requires Stage1 task data.")
    stage1_config = _stage1_configuration(config)
    configured = list(stage1_config["models"])
    selected_models = list(model_names) if model_names is not None else configured
    unknown = sorted(set(selected_models) - set(configured))
    if unknown:
        raise ValueError(f"Models are not configured for Stage1: {unknown}")

    results: dict[str, Any] = {}
    for model_name in progress_iter(
        selected_models,
        total=len(selected_models),
        description="Stage1 models",
        unit="model",
        details={"task": "Stage1"},
    ):
        output_dir = (
            run_dir
            / "optimization"
            / STAGE1_MORTALITY_TASK.task_name
            / model_name
        )
        stages, selection = run_optimization_study(
            task_data=task_data,
            model_name=model_name,
            base_model_config=apply_acceleration_to_model_config(
                project_config=config,
                model_name=model_name,
                model_config=dict(config["models"][model_name]),
            ),
            config=config,
            output_dir=output_dir,
            search_stages=search_stages,
        )
        results[model_name] = {
            "stage_results": [stage.to_dict() for stage in stages],
            "selection": selection.to_dict(),
        }

    manifest = {
        "step": "stage1_mortality_optimization",
        "task": STAGE1_MORTALITY_TASK.to_dict(),
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
        / STAGE1_MORTALITY_TASK.task_name
        / "optimization_manifest.json",
        manifest,
    )
    return manifest


def select_stage1_reference_model(
    model_results: Sequence[Stage1ModelResult],
    *,
    reference_metric: str,
    practical_tie_margin: float,
    configured_model_order: Sequence[str],
) -> tuple[Stage1ModelResult, list[dict[str, Any]]]:
    return select_validation_reference_model(
        model_results,
        reference_metric=reference_metric,
        metric_attributes={
            "auprc": "validation_auprc",
            "auroc": "validation_auroc",
            "balanced_accuracy": "validation_balanced_accuracy",
        },
        secondary_attribute="validation_brier_score",
        practical_tie_margin=practical_tie_margin,
        configured_model_order=configured_model_order,
        selected_flag="selected_as_validation_reference",
    )


def run_stage1_mortality(
    *,
    task_data: TaskDataBundle,
    config: Mapping[str, Any],
    run_id: str,
    run_dir: Path,
    model_names: Sequence[str] | None = None,
    bootstrap_repeats: int | None = None,
    overwrite: bool = False,
) -> Stage1RunResult:
    """Train Stage1 mortality models using Train and Validation only."""
    if task_data.task.task_name != STAGE1_MORTALITY_TASK.task_name:
        raise ValueError("run_stage1_mortality requires Stage1 task data.")
    forbidden = sorted(set(task_data.splits) - {"train", "validation"})
    if forbidden:
        raise ValueError(
            "Stage1 discovery training must not receive confirmatory splits: "
            f"{forbidden}"
        )

    stage1_config = _stage1_configuration(config)
    configured_models = list(stage1_config["models"])
    selected_models = list(model_names) if model_names is not None else configured_models
    unknown = sorted(set(selected_models) - set(configured_models))
    if unknown:
        raise ValueError(f"Models are not configured for Stage1: {unknown}")

    output_dir = run_dir / "EXP2_two_stage" / "stage1_mortality"
    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(f"Stage1 output already exists: {output_dir}")
        raise RuntimeError(
            "Stage1 model outputs are immutable. Use a new Run ID instead of overwrite."
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    evaluation_config = config["evaluation"]
    repeats = int(
        bootstrap_repeats
        if bootstrap_repeats is not None
        else evaluation_config.get("binary_bootstrap_repeats", 500)
    )
    bootstrap_settings = BinaryBootstrapSettings(
        repeats=repeats,
        seed=int(config["split"]["seed"]),
        confidence_level=float(
            evaluation_config.get("bootstrap_confidence_level", 0.95)
        ),
    )
    default_threshold = float(evaluation_config["default_decision_threshold"])

    model_results: list[Stage1ModelResult] = []
    selected_parameter_hashes: dict[str, str] = {}
    selected_thresholds: dict[str, float] = {}
    for model_name in progress_iter(
        selected_models,
        total=len(selected_models),
        description="Stage1 models",
        unit="model",
        details={"task": "Stage1"},
    ):
        selection_path = (
            run_dir
            / "optimization"
            / STAGE1_MORTALITY_TASK.task_name
            / model_name
            / "selected_hyperparameters.json"
        )
        model_config, selection_hash = validate_selected_hyperparameters(
            path=selection_path,
            model_name=model_name,
            task=STAGE1_MORTALITY_TASK,
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
        threshold = engine_result.validation_decision_threshold
        if threshold is None:
            raise RuntimeError(f"Stage1 threshold is missing for {model_name}.")
        selected_thresholds[model_name] = float(threshold)

        detailed_payloads: dict[str, dict[str, Any]] = {}
        for split_name in ("train", "validation"):
            evaluation_dir = model_output / "evaluation" / split_name
            prediction_frame = pd.read_csv(
                evaluation_dir / "predictions.csv.gz",
                compression="infer",
                low_memory=False,
            )
            detailed_payloads[split_name] = write_binary_evaluation_artifacts(
                y_true=prediction_frame["y_true"].to_numpy(dtype="int64"),
                probabilities=probabilities_from_binary_prediction_frame(
                    prediction_frame
                ),
                selected_threshold=float(threshold),
                default_threshold=default_threshold,
                output_dir=evaluation_dir,
                split_name=split_name,
                model_name=model_name,
                positive_class_name=STAGE1_MORTALITY_TASK.class_names[1],
                calibration_bins=int(evaluation_config["calibration_bins"]),
                bootstrap_settings=(
                    bootstrap_settings if split_name == "validation" else None
                ),
                save_plots=bool(evaluation_config.get("save_plots", True)),
            )

        validation_metrics = detailed_payloads["validation"]["threshold_views"]
        threshold_free = validation_metrics["validation_selected"]
        model_results.append(
            Stage1ModelResult(
                model_name=model_name,
                output_dir=str(model_output),
                selected_hyperparameters_path=str(selection_path),
                selected_hyperparameters_sha256=selection_hash,
                validation_auroc=float(threshold_free["auroc"]),
                validation_auprc=float(threshold_free["auprc"]),
                validation_brier_score=float(threshold_free["brier_score"]),
                validation_log_loss=float(threshold_free["log_loss"]),
                validation_balanced_accuracy=float(
                    threshold_free["balanced_accuracy"]
                ),
                validation_f1=float(threshold_free["f1_score"]),
                validation_sensitivity=float(
                    threshold_free["sensitivity_recall"]
                ),
                validation_specificity=float(threshold_free["specificity"]),
                validation_decision_threshold=float(threshold),
                training_seconds=float(engine_result.training_seconds),
            )
        )

    reference_metric = str(stage1_config.get("reference_metric", "auprc"))
    practical_tie_margin = float(
        config["optimization"]["selection"]["practical_tie_margin"]
    )
    reference, selection_table = select_stage1_reference_model(
        model_results,
        reference_metric=reference_metric,
        practical_tie_margin=practical_tie_margin,
        configured_model_order=configured_models,
    )
    pd.DataFrame(selection_table).to_csv(
        output_dir / "model_selection_validation.csv", index=False
    )
    pd.DataFrame([result.to_dict() for result in model_results]).to_csv(
        output_dir / "model_summary_validation.csv", index=False
    )

    reference_payload = {
        "task_name": STAGE1_MORTALITY_TASK.task_name,
        "selected_on_split": "validation",
        "reference_metric": reference_metric,
        "practical_tie_margin": practical_tie_margin,
        "selected_model": reference.model_name,
        "selected_model_validation_metrics": reference.to_dict(),
        "candidate_models": selected_models,
        "role": "validation_reference_only",
        "final_pipeline_model_selected": False,
        "final_pipeline_selection_step": "stage1_super_learner",
        "confirmatory_splits_accessed": False,
        "shared_split_hash": task_data.shared_split_hash,
    }
    reference_path = output_dir / "validation_reference_model.json"
    atomic_write_json(reference_path, reference_payload)
    (output_dir / "validation_reference_model.json.sha256").write_text(
        sha256_file(reference_path) + "\n", encoding="utf-8"
    )

    frozen_design = {
        "design_type": "stage1_mortality_discovery",
        "task": STAGE1_MORTALITY_TASK.to_dict(),
        "shared_split_hash": task_data.shared_split_hash,
        "feature_names": list(task_data.feature_names),
        "model_names": selected_models,
        "selected_hyperparameter_sha256": selected_parameter_hashes,
        "validation_thresholds": selected_thresholds,
        "validation_reference_model": reference.model_name,
        "reference_model_is_not_final_pipeline_model": True,
        "selection_split": "validation",
        "confirmatory_splits_accessed": False,
        "forbidden_splits": ["internal_test", "external_eicu"],
        "bootstrap_settings": bootstrap_settings.to_dict(),
        "next_design_steps": [
            "stage2_long_stay",
            "stage1_heterogeneity",
            "stage1_super_learner",
        ],
    }
    frozen_path = output_dir / "frozen_stage1_design.json"
    atomic_write_json(frozen_path, frozen_design)
    (output_dir / "frozen_stage1_design.json.sha256").write_text(
        sha256_file(frozen_path) + "\n", encoding="utf-8"
    )

    manifest = {
        "step": "stage1_mortality",
        "run_id": run_id,
        "status": "discovery_completed",
        "task": STAGE1_MORTALITY_TASK.to_dict(),
        "data_access": {
            "train": True,
            "validation": True,
            "internal_test": False,
            "external_eicu": False,
        },
        "model_results": [result.to_dict() for result in model_results],
        "validation_reference_model": reference_payload,
        "frozen_design_path": str(frozen_path),
        "environment": environment_info(),
        "next_required_step": "stage2_long_stay_and_exp3",
    }
    atomic_write_json(output_dir / "stage1_manifest.json", manifest)

    result = Stage1RunResult(
        run_id=run_id,
        output_dir=str(output_dir),
        shared_split_hash=str(task_data.shared_split_hash),
        validation_reference_model=reference.model_name,
        model_results=tuple(model_results),
        confirmatory_splits_accessed=False,
    )
    atomic_write_json(output_dir / "stage1_result.json", result.to_dict())
    return result
