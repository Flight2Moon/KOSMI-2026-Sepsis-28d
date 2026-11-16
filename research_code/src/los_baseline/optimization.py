from __future__ import annotations

import gc
import hashlib
import json
import math
import time
import warnings
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

from .exp123_data import TaskDataBundle
from .svm_runtime import svm_backend_label
from .models import (
    build_task_sklearn_model,
    compute_balanced_class_weights,
    resolve_class_weights,
    training_sample_weights,
    uses_sklearn_logistic,
    normalize_task_model_config,
)
from .optimization_objectives import (
    ObjectiveDefinition,
    aggregate_fold_metrics,
    load_objective_definition,
    robust_selection_score,
    select_validation_candidate,
    threshold_free_metrics,
)
from .optimization_registry import OptimizationRegistry
from .optimization_report import write_optimization_reports, write_validation_candidates
from .preprocessing import build_preprocessor, transform_features
from .progress import progress_iter, emit_progress, tracked_operation
from .search_spaces import (
    extract_tunable_parameters,
    sample_model_config,
    search_space_manifest,
)
from .task_spec import TaskDefinition
from .torch_model import (
    train_weighted_task_logistic_regression,
    train_weighted_task_logistic_regression_fixed_epochs,
)
from .training_engine import (
    NullSummaryWriter,
    predict_task_optimization_output,
    validate_task_optimization_output_contract,
)
from .utils import atomic_write_json, environment_info, set_global_seed, sha256_file


@dataclass(frozen=True)
class OptimizationSettings:
    seed: int
    cv_folds: int
    top_k_for_validation: int
    stability_penalty: float
    practical_tie_margin: float
    auroc_margin_guard: float
    prefer_simpler_model_on_tie: bool
    trial_timeout_seconds: float
    study_timeout_seconds: float
    resume: bool
    patient_id_column: str = "subject_id"
    overfit_penalty: float = 0.5

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "OptimizationSettings":
        optimization = config["optimization"]
        selection = optimization.get("selection", {})
        runtime = optimization.get("runtime_limits", {})
        return cls(
            seed=int(optimization.get("seed", 42)),
            cv_folds=int(optimization.get("cv_folds", 5)),
            top_k_for_validation=int(
                optimization.get("top_k_for_validation", 3)
            ),
            stability_penalty=float(selection.get("stability_penalty", 0.5)),
            overfit_penalty=float(selection.get("overfit_penalty", 0.5)),
            practical_tie_margin=float(
                selection.get("practical_tie_margin", 0.005)
            ),
            auroc_margin_guard=float(
                selection.get("auroc_margin_guard", 0.02)
            ),
            prefer_simpler_model_on_tie=bool(
                selection.get("prefer_simpler_model_on_tie", True)
            ),
            trial_timeout_seconds=float(
                runtime.get("trial_timeout_seconds", 3600)
            ),
            study_timeout_seconds=float(
                runtime.get("study_timeout_seconds", 86400)
            ),
            resume=bool(optimization.get("resume", True)),
            patient_id_column=str(
                config.get("data", {}).get("patient_id_column", "subject_id")
            ),
        )


@dataclass(frozen=True)
class TrialEvaluation:
    fold_metrics: list[dict[str, float]]
    aggregate_metrics: dict[str, float]
    selection_score: float
    duration_seconds: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class OptimizationStageResult:
    study_id: str
    search_stage: str
    requested_trials: int
    completed_trials: int
    failed_trials: int
    skipped_existing_trials: int
    stopped_by_study_timeout: bool
    output_dir: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class OptimizationSelectionResult:
    study_id: str
    selected_candidate_id: str
    selected_model_config: dict[str, Any]
    selected_metrics: dict[str, float]
    output_path: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _groups_from_training_data(
    task_data: TaskDataBundle,
    patient_id_column: str,
) -> np.ndarray:
    metadata = task_data.splits["train"].metadata
    if patient_id_column in metadata.columns:
        values = metadata[patient_id_column].astype("string")
        fallback = metadata["source_row_index"].astype(str).map(lambda value: f"row_{value}")
        groups = values.fillna(fallback).to_numpy(dtype=str)
    else:
        groups = metadata["source_row_index"].astype(str).to_numpy()
    if len(groups) != len(task_data.splits["train"].labels):
        raise ValueError("Optimization group count does not match training labels.")
    return groups


def _validate_fold_classes(
    labels: np.ndarray,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    task: TaskDefinition,
    fold_index: int,
) -> None:
    expected = set(task.expected_classes)
    train_classes = set(np.unique(labels[train_indices]).astype(int).tolist())
    validation_classes = set(
        np.unique(labels[validation_indices]).astype(int).tolist()
    )
    if train_classes != expected or validation_classes != expected:
        raise ValueError(
            f"CV fold {fold_index} does not contain all classes. "
            f"train={sorted(train_classes)}, validation={sorted(validation_classes)}, "
            f"expected={sorted(expected)}"
        )


def _release_model(model: Any) -> None:
    del model
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()
    except Exception:
        pass


def _select_nested_logistic_epoch(
    *,
    model_name: str,
    model_config: dict[str, Any],
    task: TaskDefinition,
    feature_names: Sequence[str],
    train_features: pd.DataFrame,
    train_labels: np.ndarray,
    train_groups: np.ndarray,
    seed: int,
) -> int:
    if model_name != "logistic_regression":
        raise ValueError("Nested epoch selection is only defined for logistic regression.")
    unique_groups = np.unique(train_groups)
    inner_folds = min(3, len(unique_groups))
    if inner_folds < 2:
        return max(1, int(model_config.get("epochs", 100)))
    splitter = StratifiedGroupKFold(
        n_splits=inner_folds,
        shuffle=True,
        random_state=seed,
    )
    try:
        fit_indices, stop_indices = next(
            splitter.split(train_features, train_labels, train_groups)
        )
    except ValueError:
        return max(1, int(model_config.get("epochs", 100)))
    fit_indices = np.asarray(fit_indices, dtype=np.int64)
    stop_indices = np.asarray(stop_indices, dtype=np.int64)
    _validate_fold_classes(
        train_labels, fit_indices, stop_indices, task, fold_index=0
    )
    columns = list(feature_names)
    preprocessor = build_preprocessor(model_name)
    preprocessor.fit(train_features.iloc[fit_indices].loc[:, columns])
    inner_train = transform_features(
        preprocessor, train_features.iloc[fit_indices].loc[:, columns]
    )
    inner_stop = transform_features(
        preprocessor, train_features.iloc[stop_indices].loc[:, columns]
    )
    class_weights, _ = resolve_class_weights(
        train_labels[fit_indices], task.expected_classes, model_config
    )
    result = train_weighted_task_logistic_regression(
        inner_train,
        train_labels[fit_indices],
        inner_stop,
        train_labels[stop_indices],
        model_config,
        NullSummaryWriter(),
        seed,
        task.task_type,
        task.num_classes,
        class_weights,
    )
    selected_epoch = int(result.best_epoch)
    _release_model(result.model)
    if selected_epoch < 1:
        selected_epoch = max(1, int(model_config.get("epochs", 100)))
    return selected_epoch


def _inner_early_stopping_split(
    *,
    labels: np.ndarray,
    groups: np.ndarray | None,
    task: TaskDefinition,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, str]:
    """Build a deterministic inner split without touching the outer CV fold."""
    y = np.asarray(labels, dtype=np.int64)
    candidate_splits = range(5, 1, -1)
    if groups is not None:
        group_values = np.asarray(groups).reshape(-1)
        maximum = min(5, len(np.unique(group_values)))
        candidate_splits = range(maximum, 1, -1)
        for n_splits in candidate_splits:
            splitter = StratifiedGroupKFold(
                n_splits=n_splits,
                shuffle=True,
                random_state=seed,
            )
            try:
                candidates = splitter.split(np.zeros((len(y), 1)), y, group_values)
                for fit_idx, stop_idx in candidates:
                    fit_idx = np.asarray(fit_idx, dtype=np.int64)
                    stop_idx = np.asarray(stop_idx, dtype=np.int64)
                    try:
                        _validate_fold_classes(y, fit_idx, stop_idx, task, fold_index=0)
                    except ValueError:
                        continue
                    if set(group_values[fit_idx]) & set(group_values[stop_idx]):
                        continue
                    digest = hashlib.sha256()
                    digest.update(fit_idx.tobytes())
                    digest.update(b"|")
                    digest.update(stop_idx.tobytes())
                    return fit_idx, stop_idx, digest.hexdigest()
            except ValueError:
                continue
    else:
        for n_splits in candidate_splits:
            splitter = StratifiedKFold(
                n_splits=n_splits,
                shuffle=True,
                random_state=seed,
            )
            try:
                fit_idx, stop_idx = next(
                    splitter.split(np.zeros((len(y), 1)), y)
                )
                fit_idx = np.asarray(fit_idx, dtype=np.int64)
                stop_idx = np.asarray(stop_idx, dtype=np.int64)
                _validate_fold_classes(y, fit_idx, stop_idx, task, fold_index=0)
                digest = hashlib.sha256()
                digest.update(fit_idx.tobytes())
                digest.update(b"|")
                digest.update(stop_idx.tobytes())
                return fit_idx, stop_idx, digest.hexdigest()
            except ValueError:
                continue
    raise ValueError(
        "Unable to create an inner early-stopping split containing every class."
    )


def _fit_sklearn_hpo_model(
    *,
    model_name: str,
    model_config: dict[str, Any],
    task: TaskDefinition,
    class_weights: dict[int, float],
    class_weight_method: str,
    train_features: pd.DataFrame,
    train_labels: np.ndarray,
    transformed_train: np.ndarray,
    train_groups: np.ndarray | None,
    feature_names: Sequence[str],
    seed: int,
    diagnostics: dict[str, Any],
) -> tuple[Any, dict[str, Any]]:
    """Fit an HPO candidate with leakage-safe XGBoost early stopping."""
    parameters = dict(model_config.get("parameters", {}))
    use_nested_early_stopping = (
        model_name == "xgboost"
        and int(parameters.get("early_stopping_rounds", 0) or 0) > 0
    )

    if use_nested_early_stopping:
        fit_idx, stop_idx, split_hash = _inner_early_stopping_split(
            labels=train_labels,
            groups=train_groups,
            task=task,
            seed=seed + 701,
        )
        selector_preprocessor = build_preprocessor(model_name)
        columns = list(feature_names)
        selector_preprocessor.fit(train_features.iloc[fit_idx].loc[:, columns])
        selector_train = transform_features(
            selector_preprocessor,
            train_features.iloc[fit_idx].loc[:, columns],
        )
        selector_stop = transform_features(
            selector_preprocessor,
            train_features.iloc[stop_idx].loc[:, columns],
        )
        selector_model = build_task_sklearn_model(
            model_name=model_name,
            model_config=model_config,
            seed=seed,
            task_type=task.task_type,
            num_classes=task.num_classes,
            class_weights=class_weights,
        )
        selector_fit_kwargs: dict[str, Any] = {
            "eval_set": [(selector_stop, train_labels[stop_idx])],
            "verbose": False,
        }
        if class_weight_method != "none":
            selector_fit_kwargs["sample_weight"] = training_sample_weights(
                train_labels[fit_idx], class_weights
            )
        with tracked_operation(
            f"Select XGBoost iterations {task.task_name}/{model_name}",
            details={
                "task_name": task.task_name,
                "model_name": model_name,
                "inner_fit_samples": int(len(fit_idx)),
                "inner_stop_samples": int(len(stop_idx)),
                "inner_split_hash": split_hash,
            },
        ):
            selector_model.fit(
                selector_train,
                train_labels[fit_idx],
                **selector_fit_kwargs,
            )
        best_iteration = getattr(selector_model, "best_iteration", None)
        if best_iteration is None:
            best_iteration = int(parameters.get("n_estimators", 1)) - 1
        selected_estimators = max(1, int(best_iteration) + 1)
        diagnostics.update(
            {
                "best_iteration": int(best_iteration),
                "selected_n_estimators": selected_estimators,
                "early_stopping_protocol": (
                    "nested_group_holdout_then_refit_full_outer_train"
                    if train_groups is not None
                    else "nested_stratified_holdout_then_refit_full_outer_train"
                ),
                "early_stopping_split_hash": split_hash,
                "early_stopping_outer_validation_accessed": False,
            }
        )
        _release_model(selector_model)

        refit_config = json.loads(json.dumps(model_config, default=str))
        refit_parameters = dict(refit_config.get("parameters", {}))
        refit_parameters["n_estimators"] = selected_estimators
        refit_parameters.pop("early_stopping_rounds", None)
        refit_config["parameters"] = refit_parameters
        model = build_task_sklearn_model(
            model_name=model_name,
            model_config=refit_config,
            seed=seed,
            task_type=task.task_type,
            num_classes=task.num_classes,
            class_weights=class_weights,
        )
        fit_kwargs: dict[str, Any] = {}
        if class_weight_method != "none":
            fit_kwargs["sample_weight"] = training_sample_weights(
                train_labels, class_weights
            )
        with tracked_operation(
            f"Refit {task.task_name}/{model_name} on full outer training fold",
            details={
                "task_name": task.task_name,
                "model_name": model_name,
                "train_samples": int(len(train_labels)),
                "selected_n_estimators": selected_estimators,
                "device": refit_parameters.get("device", "cpu"),
            },
        ):
            model.fit(transformed_train, train_labels, **fit_kwargs)
        return model, refit_config

    model = build_task_sklearn_model(
        model_name=model_name,
        model_config=model_config,
        seed=seed,
        task_type=task.task_type,
        num_classes=task.num_classes,
        class_weights=class_weights,
    )
    fit_kwargs = {}
    if model_name == "xgboost" and class_weight_method != "none":
        fit_kwargs["sample_weight"] = training_sample_weights(
            train_labels, class_weights
        )
    fit_started = time.perf_counter()
    convergence_messages: list[str] = []
    with tracked_operation(
        f"Fit {task.task_name}/{model_name} partition",
        details={
            "task_name": task.task_name,
            "model_name": model_name,
            "train_samples": int(len(train_labels)),
            "device": parameters.get("device", "cpu"),
            "svm_backend": svm_backend_label(model_config) if model_name == "svm" else None,
        },
    ):
        if model_name == "svm":
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", ConvergenceWarning)
                model.fit(transformed_train, train_labels, **fit_kwargs)
            convergence_messages = [
                str(item.message) for item in caught
                if issubclass(item.category, ConvergenceWarning)
            ]
        else:
            model.fit(transformed_train, train_labels, **fit_kwargs)
    fit_seconds = float(time.perf_counter() - fit_started)
    diagnostics["fit_seconds"] = fit_seconds
    if model_name == "svm":
        diagnostics["svm_backend"] = svm_backend_label(model_config)
        diagnostics["svm_convergence_warning"] = bool(convergence_messages)
        diagnostics["svm_convergence_messages"] = convergence_messages
        n_iter = getattr(model, "n_iter_", None)
        if n_iter is None and hasattr(model, "named_steps"):
            final = model.named_steps.get("linear_svc")
            n_iter = getattr(final, "n_iter_", None) if final is not None else None
        if n_iter is not None:
            diagnostics["svm_n_iter"] = np.asarray(n_iter).astype(int).reshape(-1).tolist()
    return model, model_config


def _fit_predict_partition(
    *,
    model_name: str,
    model_config: dict[str, Any],
    task: TaskDefinition,
    feature_names: Sequence[str],
    train_features: pd.DataFrame,
    train_labels: np.ndarray,
    evaluation_features: pd.DataFrame,
    evaluation_labels: np.ndarray,
    seed: int,
    train_groups: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, float], dict[str, Any], dict[str, Any]]:
    set_global_seed(seed)
    effective_config = normalize_task_model_config(
        model_name,
        json.loads(json.dumps(model_config, default=str)),
        task_type=task.task_type,
        num_classes=task.num_classes,
        hpo_mode=True,
    )
    diagnostics: dict[str, Any] = {}
    prediction_config = effective_config
    if model_name == "logistic_regression" and not uses_sklearn_logistic(effective_config):
        if train_groups is None:
            train_groups = np.arange(len(train_labels)).astype(str)
        selected_epoch = _select_nested_logistic_epoch(
            model_name=model_name,
            model_config=model_config,
            task=task,
            feature_names=feature_names,
            train_features=train_features,
            train_labels=train_labels,
            train_groups=np.asarray(train_groups),
            seed=seed + 17,
        )
        effective_config["epochs"] = selected_epoch
        effective_config["fixed_epoch_training"] = True
        effective_config["epoch_selection"] = "nested_train_group_holdout"
        diagnostics["selected_epoch"] = selected_epoch

    preprocessor = build_preprocessor(model_name)
    columns = list(feature_names)
    preprocessor.fit(train_features.loc[:, columns])
    transformed_train = transform_features(preprocessor, train_features.loc[:, columns])
    transformed_evaluation = transform_features(
        preprocessor, evaluation_features.loc[:, columns]
    )
    class_weights, class_weight_method = resolve_class_weights(
        train_labels, task.expected_classes, effective_config
    )
    if model_name == "logistic_regression" and not uses_sklearn_logistic(effective_config):
        training_result = train_weighted_task_logistic_regression_fixed_epochs(
            transformed_train,
            train_labels,
            effective_config,
            NullSummaryWriter(),
            seed,
            task.task_type,
            task.num_classes,
            class_weights,
        )
        model = training_result.model
    else:
        model, prediction_config = _fit_sklearn_hpo_model(
            model_name=model_name,
            model_config=effective_config,
            task=task,
            class_weights=class_weights,
            class_weight_method=class_weight_method,
            train_features=train_features,
            train_labels=train_labels,
            transformed_train=transformed_train,
            train_groups=train_groups,
            feature_names=columns,
            seed=seed,
            diagnostics=diagnostics,
        )
    predictions, prediction_kind = predict_task_optimization_output(
        model_name=model_name,
        model=model,
        transformed=transformed_evaluation,
        task=task,
        model_config=prediction_config,
    )
    validate_task_optimization_output_contract(
        predictions, task, len(evaluation_labels), prediction_kind
    )
    metrics = threshold_free_metrics(
        evaluation_labels,
        predictions,
        task,
        prediction_kind=prediction_kind,
    )
    train_predictions, train_prediction_kind = predict_task_optimization_output(
        model_name=model_name,
        model=model,
        transformed=transformed_train,
        task=task,
        model_config=prediction_config,
    )
    if train_prediction_kind != prediction_kind:
        raise RuntimeError("Training and validation prediction kinds differ.")
    validate_task_optimization_output_contract(
        train_predictions, task, len(train_labels), prediction_kind
    )
    diagnostics["train_metrics"] = threshold_free_metrics(
        train_labels,
        train_predictions,
        task,
        prediction_kind=prediction_kind,
    )
    diagnostics["prediction_kind"] = prediction_kind
    _release_model(model)
    return predictions, metrics, diagnostics, effective_config

def evaluate_candidate_cv(
    *,
    task_data: TaskDataBundle,
    model_name: str,
    model_config: dict[str, Any],
    objective: ObjectiveDefinition,
    cv_folds: int,
    seed: int,
    stability_penalty: float,
    patient_id_column: str,
    overfit_penalty: float = 0.5,
) -> TrialEvaluation:
    train = task_data.splits["train"]
    labels = np.asarray(train.labels, dtype=np.int64)
    groups = _groups_from_training_data(task_data, patient_id_column)
    unique_groups = np.unique(groups)
    if len(unique_groups) < cv_folds:
        raise ValueError(
            f"Not enough unique patient groups for {cv_folds}-fold CV: "
            f"{len(unique_groups)}"
        )
    splitter = StratifiedGroupKFold(
        n_splits=cv_folds,
        shuffle=True,
        random_state=seed,
    )
    start = time.perf_counter()
    fold_metrics: list[dict[str, float]] = []
    fold_iterator = enumerate(
        splitter.split(train.raw_features, labels, groups), start=1
    )
    for fold_index, (fit_indices, validation_indices) in progress_iter(
        fold_iterator,
        total=cv_folds,
        description=f"{task_data.task.task_name}/{model_name} CV",
        unit="fold",
        leave=False,
        details={"task": task_data.task.task_name, "model": model_name},
    ):
        fit_indices = np.asarray(fit_indices, dtype=np.int64)
        validation_indices = np.asarray(validation_indices, dtype=np.int64)
        _validate_fold_classes(
            labels, fit_indices, validation_indices, task_data.task, fold_index
        )
        fit_groups = set(groups[fit_indices].tolist())
        validation_groups = set(groups[validation_indices].tolist())
        if fit_groups & validation_groups:
            raise RuntimeError(f"Patient group leakage detected in CV fold {fold_index}.")
        _, metrics, diagnostics, _ = _fit_predict_partition(
            model_name=model_name,
            model_config=model_config,
            task=task_data.task,
            feature_names=task_data.feature_names,
            train_features=train.raw_features.iloc[fit_indices].reset_index(drop=True),
            train_labels=labels[fit_indices],
            evaluation_features=train.raw_features.iloc[validation_indices].reset_index(
                drop=True
            ),
            evaluation_labels=labels[validation_indices],
            seed=seed + fold_index,
            train_groups=groups[fit_indices],
        )
        train_metrics = diagnostics.get("train_metrics", {})
        if isinstance(train_metrics, Mapping):
            for metric_name, metric_value in train_metrics.items():
                metrics[f"train_{metric_name}"] = float(metric_value)
        if "selected_epoch" in diagnostics:
            metrics["selected_epoch"] = float(diagnostics["selected_epoch"])
        metrics["fold_index"] = float(fold_index)
        metrics["train_sample_count"] = float(len(fit_indices))
        metrics["validation_sample_count"] = float(len(validation_indices))
        metrics["train_group_count"] = float(len(fit_groups))
        metrics["validation_group_count"] = float(len(validation_groups))
        fold_metrics.append(metrics)
    aggregate = aggregate_fold_metrics(fold_metrics)
    score = robust_selection_score(
        aggregate,
        objective,
        stability_penalty=stability_penalty,
        overfit_penalty=overfit_penalty,
    )
    return TrialEvaluation(
        fold_metrics=fold_metrics,
        aggregate_metrics=aggregate,
        selection_score=score,
        duration_seconds=float(time.perf_counter() - start),
    )


def _complexity_score(model_name: str, model_config: Mapping[str, Any]) -> float:
    if model_name == "logistic_regression":
        if uses_sklearn_logistic(dict(model_config)):
            parameters = dict(model_config.get("parameters", {}))
            l1_ratio = float(parameters.get("l1_ratio", 0.0))
            return l1_ratio + math.log10(1.0 + float(parameters.get("C", 1.0)))
        epochs = float(model_config.get("epochs", 100))
        weight_decay = float(model_config.get("weight_decay", 0.0))
        return epochs + math.log10(1.0 + weight_decay * 1e6)
    parameters = dict(model_config.get("parameters", {}))
    if model_name == "decision_tree":
        depth = parameters.get("max_depth")
        return float(1000 if depth is None else depth) + float(
            parameters.get("min_samples_leaf", 1)
        ) ** -1
    if model_name == "random_forest":
        depth = parameters.get("max_depth")
        depth_value = 100 if depth is None else float(depth)
        return float(parameters.get("n_estimators", 100)) * depth_value
    if model_name == "svm":
        return float(parameters.get("C", 1.0))
    if model_name == "xgboost":
        return float(parameters.get("n_estimators", 100)) * float(
            parameters.get("max_depth", 6)
        )
    return 0.0


def _best_anchor_parameters(
    registry: OptimizationRegistry,
    study_id: str,
) -> dict[str, Any]:
    trials = registry.completed_trials(study_id, search_stages=("coarse",))
    if not trials:
        raise RuntimeError("Fine search requires at least one completed coarse trial.")
    return dict(trials[0]["tunable_parameters"])


def run_optimization_stage(
    *,
    task_data: TaskDataBundle,
    model_name: str,
    base_model_config: Mapping[str, Any],
    objective: ObjectiveDefinition,
    settings: OptimizationSettings,
    output_dir: Path,
    search_stage: str,
    trial_count: int,
    cv_folds: int,
) -> OptimizationStageResult:
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    study_id = f"{task_data.task.task_name}:{model_name}"
    registry_path = output_dir / "study.sqlite"
    stage_start = time.perf_counter()
    completed = 0
    failed = 0
    skipped = 0
    stopped_by_timeout = False

    with OptimizationRegistry(registry_path) as registry:
        registry.register_study(
            study_id=study_id,
            task_name=task_data.task.task_name,
            model_name=model_name,
            seed=settings.seed,
            shared_split_hash=task_data.shared_split_hash,
            objective=objective.to_dict(),
        )
        anchor = (
            _best_anchor_parameters(registry, study_id)
            if search_stage == "fine"
            else None
        )
        atomic_write_json(
            output_dir / f"search_space_{search_stage}.json",
            search_space_manifest(model_name, search_stage, anchor),
        )
        for trial_number in progress_iter(
            range(int(trial_count)),
            total=int(trial_count),
            description=f"{study_id}/{search_stage}",
            unit="trial",
            details={
                "study_id": study_id,
                "search_stage": search_stage,
                "cv_folds": int(cv_folds),
            },
        ):
            if time.perf_counter() - stage_start >= settings.study_timeout_seconds:
                stopped_by_timeout = True
                break
            if settings.resume and registry.trial_exists(
                study_id, search_stage, trial_number, completed_only=True
            ):
                skipped += 1
                continue
            model_config = sample_model_config(
                model_name=model_name,
                base_model_config=base_model_config,
                task_name=task_data.task.task_name,
                search_stage=search_stage,
                trial_number=trial_number,
                seed=settings.seed,
                anchor_parameters=anchor,
            )
            tunable = extract_tunable_parameters(model_name, model_config)
            registry.upsert_trial(
                study_id=study_id,
                search_stage=search_stage,
                trial_number=trial_number,
                status="running",
                tunable_parameters=tunable,
                model_config=model_config,
            )
            started = time.perf_counter()
            try:
                result = evaluate_candidate_cv(
                    task_data=task_data,
                    model_name=model_name,
                    model_config=model_config,
                    objective=objective,
                    cv_folds=cv_folds,
                    seed=settings.seed + trial_number * 100,
                    stability_penalty=settings.stability_penalty,
                    patient_id_column=settings.patient_id_column,
                    overfit_penalty=settings.overfit_penalty,
                )
                duration = float(time.perf_counter() - started)
                if duration > settings.trial_timeout_seconds:
                    registry.upsert_trial(
                        study_id=study_id,
                        search_stage=search_stage,
                        trial_number=trial_number,
                        status="timeout",
                        tunable_parameters=tunable,
                        model_config=model_config,
                        fold_metrics=result.fold_metrics,
                        aggregate_metrics=result.aggregate_metrics,
                        duration_seconds=duration,
                        failure_type="SoftTrialTimeout",
                        failure_message=(
                            "Trial completed after the configured soft timeout. "
                            "It is excluded from candidate selection."
                        ),
                    )
                    failed += 1
                else:
                    registry.upsert_trial(
                        study_id=study_id,
                        search_stage=search_stage,
                        trial_number=trial_number,
                        status="completed",
                        tunable_parameters=tunable,
                        model_config=model_config,
                        fold_metrics=result.fold_metrics,
                        aggregate_metrics=result.aggregate_metrics,
                        selection_score=result.selection_score,
                        duration_seconds=duration,
                    )
                    completed += 1
            except Exception as exception:
                registry.upsert_trial(
                    study_id=study_id,
                    search_stage=search_stage,
                    trial_number=trial_number,
                    status="failed",
                    tunable_parameters=tunable,
                    model_config=model_config,
                    duration_seconds=float(time.perf_counter() - started),
                    failure_type=type(exception).__name__,
                    failure_message=str(exception),
                    traceback_text=traceback.format_exc(),
                )
                failed += 1
        write_optimization_reports(registry, study_id, output_dir)

    result = OptimizationStageResult(
        study_id=study_id,
        search_stage=search_stage,
        requested_trials=int(trial_count),
        completed_trials=completed,
        failed_trials=failed,
        skipped_existing_trials=skipped,
        stopped_by_study_timeout=stopped_by_timeout,
        output_dir=str(output_dir),
    )
    atomic_write_json(output_dir / f"{search_stage}_summary.json", result.to_dict())
    return result


def finalize_validation_selection(
    *,
    task_data: TaskDataBundle,
    model_name: str,
    objective: ObjectiveDefinition,
    settings: OptimizationSettings,
    output_dir: Path,
    preferred_search_stages: Sequence[str] = ("fine", "coarse"),
) -> OptimizationSelectionResult:
    output_dir = output_dir.expanduser().resolve()
    study_id = f"{task_data.task.task_name}:{model_name}"
    registry_path = output_dir / "study.sqlite"
    with OptimizationRegistry(registry_path) as registry:
        trials = registry.completed_trials(study_id, preferred_search_stages)
        if not trials:
            trials = registry.completed_trials(study_id, ("smoke",))
        if not trials:
            raise RuntimeError(f"No completed optimization trials for {study_id}.")

        unique: list[dict[str, Any]] = []
        seen: set[str] = set()
        for trial in trials:
            canonical = json.dumps(trial["model_config"], sort_keys=True, default=str)
            if canonical in seen:
                continue
            seen.add(canonical)
            unique.append(trial)
            if len(unique) >= settings.top_k_for_validation:
                break

        train = task_data.splits["train"]
        validation = task_data.splits["validation"]
        flat_candidates: list[dict[str, Any]] = []
        for index, trial in progress_iter(
            list(enumerate(unique, start=1)),
            total=len(unique),
            description=f"{study_id}/validation candidates",
            unit="candidate",
            leave=False,
            details={"study_id": study_id},
        ):
            _, metrics, diagnostics, effective_config = _fit_predict_partition(
                model_name=model_name,
                model_config=trial["model_config"],
                task=task_data.task,
                feature_names=task_data.feature_names,
                train_features=train.raw_features,
                train_labels=train.labels,
                evaluation_features=validation.raw_features,
                evaluation_labels=validation.labels,
                seed=settings.seed + 10000 + index,
                train_groups=_groups_from_training_data(
                    task_data, settings.patient_id_column
                ),
            )
            flat_candidates.append(
                {
                    "candidate_id": f"candidate_{index:02d}",
                    "source_stage": trial["search_stage"],
                    "source_trial_number": int(trial["trial_number"]),
                    "cv_selection_score": float(trial["selection_score"]),
                    "model_config": effective_config,
                    "selected_epoch": diagnostics.get("selected_epoch"),
                    "complexity_score": _complexity_score(
                        model_name, effective_config
                    ),
                    **metrics,
                }
            )

        selected, decorated = select_validation_candidate(
            candidates=flat_candidates,
            task=task_data.task,
            objective=objective,
            practical_tie_margin=settings.practical_tie_margin,
            auroc_margin_guard=settings.auroc_margin_guard,
        )
        registry_candidates: list[dict[str, Any]] = []
        for row in decorated:
            metrics = {
                key: value
                for key, value in row.items()
                if key
                not in {
                    "candidate_id",
                    "source_stage",
                    "source_trial_number",
                    "cv_selection_score",
                    "model_config",
                    "complexity_score",
                    "primary_tie_guard_passed",
                    "auroc_margin_guard_passed",
                    "eligible_for_selection",
                    "selected",
                }
            }
            registry_candidates.append(
                {
                    **row,
                    "metrics": metrics,
                }
            )
        registry.replace_validation_candidates(study_id, registry_candidates)

    write_validation_candidates(decorated, output_dir / "validation_candidates.csv")
    selected_metrics = {
        key: float(value)
        for key, value in selected.items()
        if key
        not in {
            "candidate_id",
            "source_stage",
            "source_trial_number",
            "cv_selection_score",
            "model_config",
            "complexity_score",
            "primary_tie_guard_passed",
            "auroc_margin_guard_passed",
            "eligible_for_selection",
            "selected",
        }
        and isinstance(value, (int, float, np.integer, np.floating))
    }
    payload = {
        "study_id": study_id,
        "task_name": task_data.task.task_name,
        "model_name": model_name,
        "selected_on_split": "validation",
        "confirmatory_splits_accessed": False,
        "forbidden_splits": ["internal_test", "external_eicu"],
        "selected_candidate_id": selected["candidate_id"],
        "source_stage": selected["source_stage"],
        "source_trial_number": int(selected["source_trial_number"]),
        "model_config": selected["model_config"],
        "metrics": selected_metrics,
        "objective": objective.to_dict(),
        "selection_guards": {
            "practical_tie_margin": settings.practical_tie_margin,
            "auroc_margin_guard": settings.auroc_margin_guard,
            "prefer_simpler_model_on_tie": settings.prefer_simpler_model_on_tie,
        },
        "shared_split_hash": task_data.shared_split_hash,
    }
    output_path = output_dir / "selected_hyperparameters.json"
    atomic_write_json(output_path, payload)
    (output_dir / "selected_hyperparameters.json.sha256").write_text(
        sha256_file(output_path) + "\n", encoding="utf-8"
    )
    result = OptimizationSelectionResult(
        study_id=study_id,
        selected_candidate_id=str(selected["candidate_id"]),
        selected_model_config=dict(selected["model_config"]),
        selected_metrics=selected_metrics,
        output_path=str(output_path),
    )
    return result


def resolve_search_stage_config(
    *,
    config: Mapping[str, Any],
    task_name: str,
    model_name: str,
    search_stage: str,
) -> dict[str, Any]:
    """Resolve global, model-specific, then task/model-specific search budgets."""
    optimization = config["optimization"]
    resolved = dict(optimization["search_stages"][search_stage])

    model_budgets = optimization.get("model_search_budgets", {})
    if isinstance(model_budgets, Mapping):
        model_config = model_budgets.get(model_name, {})
        if isinstance(model_config, Mapping):
            stage_config = model_config.get(search_stage, {})
            if isinstance(stage_config, Mapping):
                resolved.update(dict(stage_config))

    task_budgets = optimization.get("task_model_search_budgets", {})
    if isinstance(task_budgets, Mapping):
        task_config = task_budgets.get(task_name, {})
        if isinstance(task_config, Mapping):
            model_config = task_config.get(model_name, {})
            if isinstance(model_config, Mapping):
                stage_config = model_config.get(search_stage, {})
                if isinstance(stage_config, Mapping):
                    resolved.update(dict(stage_config))

    trials = int(resolved.get("trials", 0))
    cv_folds = int(resolved.get("cv_folds", 0))
    if trials < 1:
        raise ValueError(
            f"Resolved search budget has no trials for {task_name}/{model_name}/{search_stage}."
        )
    if cv_folds < 2:
        raise ValueError(
            f"Resolved search budget has fewer than 2 folds for {task_name}/{model_name}/{search_stage}."
        )
    return {"trials": trials, "cv_folds": cv_folds}


def run_optimization_study(
    *,
    task_data: TaskDataBundle,
    model_name: str,
    base_model_config: Mapping[str, Any],
    config: Mapping[str, Any],
    output_dir: Path,
    search_stages: Sequence[str] = ("smoke", "coarse", "fine"),
) -> tuple[list[OptimizationStageResult], OptimizationSelectionResult]:
    settings = OptimizationSettings.from_config(config)
    objective = load_objective_definition(
        task_data.task.task_name, config["optimization_objectives"]
    )
    stage_results: list[OptimizationStageResult] = []
    for stage in progress_iter(
        search_stages,
        total=len(search_stages),
        description=f"{task_data.task.task_name}/{model_name} search stages",
        unit="stage",
        details={"task": task_data.task.task_name, "model": model_name},
    ):
        stage_config = resolve_search_stage_config(
            config=config,
            task_name=task_data.task.task_name,
            model_name=model_name,
            search_stage=stage,
        )
        stage_results.append(
            run_optimization_stage(
                task_data=task_data,
                model_name=model_name,
                base_model_config=base_model_config,
                objective=objective,
                settings=settings,
                output_dir=output_dir,
                search_stage=stage,
                trial_count=int(stage_config["trials"]),
                cv_folds=int(stage_config["cv_folds"]),
            )
        )
    selection = finalize_validation_selection(
        task_data=task_data,
        model_name=model_name,
        objective=objective,
        settings=settings,
        output_dir=output_dir,
        preferred_search_stages=tuple(reversed(search_stages)),
    )
    manifest = {
        "step": "optimization_engine",
        "task_name": task_data.task.task_name,
        "model_name": model_name,
        "study_id": selection.study_id,
        "shared_split_hash": task_data.shared_split_hash,
        "data_access": {
            "cross_validation": "train_only",
            "candidate_selection": "validation_only",
            "internal_test_accessed": False,
            "external_eicu_accessed": False,
        },
        "stage_results": [result.to_dict() for result in stage_results],
        "selection": selection.to_dict(),
        "environment": environment_info(),
    }
    atomic_write_json(output_dir / "study_manifest.json", manifest)
    return stage_results, selection
