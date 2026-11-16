from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    roc_auc_score,
)

from .artifacts import load_task_artifact, save_task_artifact
from .calibration_v7 import build_group_calibrated_classifier, explainability_estimator
from .calibration_sampling_v8 import (
    FrozenOOFSigmoidClassifier,
    fit_sampling_aware_calibration_map,
)
from .imbalance.sampling import TrainingSampler, sampling_manifest_payload
from .exp123_data import TaskDataBundle, TaskSplitData
from .explainability import save_explainability
from .metrics import select_youden_threshold
from .models import (
    build_task_sklearn_model,
    compute_balanced_class_weights,
    resolve_class_weights,
    training_sample_weights,
    uses_sklearn_logistic,
)
from .progress import tracked_operation
from .experiment_tracking import get_tracker, log_metrics
from .preprocessing import (
    build_preprocessor,
    transform_features,
    validate_training_features,
)
from .task_spec import TaskDefinition
from .svm_runtime import svm_backend_label, svm_runtime_settings
from .torch_model import (
    predict_task_torch_probabilities,
    select_device,
    train_weighted_task_logistic_regression,
    train_weighted_task_logistic_regression_fixed_epochs,
)
from .utils import atomic_write_json, sha256_file
from .xgboost_device_adapter import predict_proba_device_aware


class NullSummaryWriter:
    def add_scalar(self, *args: Any, **kwargs: Any) -> None:
        return None

    def close(self) -> None:
        return None


@dataclass(frozen=True)
class ClassWeightSummary:
    class_counts: dict[int, int]
    class_weights: dict[int, float]
    sample_count: int
    method: str = "balanced_inverse_frequency"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PredictionIntegrityResult:
    split_name: str
    sample_count: int
    shape: tuple[int, ...]
    maximum_absolute_difference: float
    absolute_tolerance: float
    relative_tolerance: float
    probabilities_match: bool
    decisions_match: bool
    mismatched_decision_count: int
    comparison_method: str
    matches: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TrainingEngineResult:
    task_name: str
    model_name: str
    output_dir: str
    artifact_dir: str
    validation_decision_threshold: float | None
    class_weights: dict[int, float]
    split_metrics: dict[str, dict[str, float | int | None]]
    prediction_integrity: dict[str, PredictionIntegrityResult]
    training_seconds: float
    artifact_reload_seconds: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _class_weight_summary(
    labels: np.ndarray,
    task: TaskDefinition,
    model_config: Mapping[str, Any] | None = None,
) -> ClassWeightSummary:
    weights, method = resolve_class_weights(
        labels, task.expected_classes, dict(model_config or {})
    )
    counts = {
        class_code: int((labels == class_code).sum())
        for class_code in task.expected_classes
    }
    return ClassWeightSummary(
        class_counts=counts,
        class_weights=weights,
        sample_count=int(len(labels)),
        method=method,
    )


def _predict_in_memory(
    model_name: str,
    model: Any,
    transformed: np.ndarray,
    task: TaskDefinition,
    model_config: Mapping[str, Any],
) -> np.ndarray:
    if model_name == "logistic_regression" and not uses_sklearn_logistic(dict(model_config)):
        device = select_device(str(model_config.get("device", "auto")))
        return predict_task_torch_probabilities(
            model,
            transformed,
            device,
            task.task_type,
        )
    if (
        model_name == "xgboost"
        and not hasattr(model, "calibrated_classifiers_")
        and not bool(getattr(model, "_exp123_calibrated_wrapper", False))
    ):
        probabilities, _ = predict_proba_device_aware(
            model,
            transformed,
            model_config=model_config,
        )
    else:
        probabilities = np.asarray(model.predict_proba(transformed), dtype=np.float64)
    if task.task_type == "binary":
        if probabilities.ndim == 2 and probabilities.shape[1] >= 2:
            return probabilities[:, 1]
        return probabilities.reshape(-1)
    if probabilities.ndim != 2 or probabilities.shape[1] != task.num_classes:
        raise ValueError(
            f"Model {model_name} returned invalid multiclass probabilities: "
            f"{probabilities.shape}"
        )
    row_sums = probabilities.sum(axis=1, keepdims=True)
    if (row_sums <= 0).any():
        raise ValueError("Multiclass model returned a non-positive probability sum.")
    return probabilities / row_sums


def _validate_probability_contract(
    probabilities: np.ndarray,
    task: TaskDefinition,
    sample_count: int,
) -> None:
    values = np.asarray(probabilities, dtype=np.float64)
    if task.task_type == "binary":
        if values.shape != (sample_count,):
            raise ValueError(
                f"Binary probability shape must be {(sample_count,)}: {values.shape}"
            )
    else:
        expected = (sample_count, task.num_classes)
        if values.shape != expected:
            raise ValueError(
                f"Multiclass probability shape must be {expected}: {values.shape}"
            )
        row_sums = values.sum(axis=1)
        if not np.allclose(row_sums, 1.0, atol=1e-6, rtol=1e-6):
            maximum_error = float(np.max(np.abs(row_sums - 1.0)))
            raise ValueError(
                "Multiclass probabilities do not sum to one. "
                f"Maximum error: {maximum_error}"
            )
    if not np.isfinite(values).all():
        raise ValueError("Predicted probabilities contain non-finite values.")
    if ((values < -1e-8) | (values > 1.0 + 1e-8)).any():
        raise ValueError("Predicted probabilities are outside [0, 1].")


def compare_prediction_integrity(
    *,
    split_name: str,
    memory_probability: np.ndarray,
    reloaded_probability: np.ndarray,
    task: TaskDefinition,
    threshold: float | None,
    model_config: Mapping[str, Any],
    integrity_config: Mapping[str, Any] | None,
    default_cpu_tolerance: float = 1e-7,
) -> PredictionIntegrityResult:
    """Compare saved/reloaded predictions with backend-aware numeric tolerance.

    CUDA and MPS commonly use float32 kernels, so a one-machine-epsilon
    probability difference is acceptable only when the final binary threshold
    or multiclass argmax decision remains unchanged.
    """
    memory = np.asarray(memory_probability)
    reloaded = np.asarray(reloaded_probability)
    if memory.shape != reloaded.shape:
        raise ValueError(
            f"Prediction shape mismatch on {split_name}: {memory.shape} != {reloaded.shape}"
        )
    maximum_difference = float(np.max(np.abs(memory - reloaded))) if memory.size else 0.0
    parameters = dict(model_config.get("parameters", {}))
    backend = str(model_config.get("device", parameters.get("device", "cpu"))).lower()
    backend_name = "cuda" if backend.startswith("cuda") else ("mps" if backend.startswith("mps") else "cpu")
    configured = dict(integrity_config or {})
    tolerance_values = dict(dict(configured.get("probability_tolerance", {})).get(backend_name, {}))
    default_atol = 1e-6 if backend_name in {"cuda", "mps"} else float(default_cpu_tolerance)
    absolute_tolerance = float(tolerance_values.get("atol", default_atol))
    relative_tolerance = float(tolerance_values.get("rtol", 1e-6 if backend_name in {"cuda", "mps"} else 0.0))
    probabilities_match = bool(np.allclose(memory, reloaded, atol=absolute_tolerance, rtol=relative_tolerance, equal_nan=False))
    if task.task_type == "binary":
        if threshold is None:
            raise ValueError("Binary artifact integrity requires a frozen threshold.")
        memory_decisions = (memory >= threshold).astype(np.int64)
        reloaded_decisions = (reloaded >= threshold).astype(np.int64)
    else:
        memory_decisions = np.argmax(memory, axis=1)
        reloaded_decisions = np.argmax(reloaded, axis=1)
    mismatched_decision_count = int(np.count_nonzero(memory_decisions != reloaded_decisions))
    maximum_allowed = int(configured.get("maximum_decision_mismatches", 0))
    require_decision_equality = bool(configured.get("require_decision_equality", True))
    decisions_match = bool(mismatched_decision_count <= maximum_allowed if require_decision_equality else True)
    return PredictionIntegrityResult(
        split_name=split_name,
        sample_count=int(memory.shape[0]) if memory.ndim else int(memory.size),
        shape=tuple(int(value) for value in reloaded.shape),
        maximum_absolute_difference=maximum_difference,
        absolute_tolerance=absolute_tolerance,
        relative_tolerance=relative_tolerance,
        probabilities_match=probabilities_match,
        decisions_match=decisions_match,
        mismatched_decision_count=mismatched_decision_count,
        comparison_method="numpy_allclose_plus_decision_equality",
        matches=bool(probabilities_match and decisions_match),
    )



def predict_task_probabilities(
    model_name: str,
    model: Any,
    transformed: np.ndarray,
    task: TaskDefinition,
    model_config: Mapping[str, Any],
) -> np.ndarray:
    """Public probability adapter shared by training and optimization."""
    return _predict_in_memory(
        model_name=model_name,
        model=model,
        transformed=transformed,
        task=task,
        model_config=model_config,
    )


def predict_task_optimization_output(
    *,
    model_name: str,
    model: Any,
    transformed: np.ndarray,
    task: TaskDefinition,
    model_config: Mapping[str, Any],
) -> tuple[np.ndarray, str]:
    """Return probabilities or raw ranking scores for HPO evaluation.

    SVC candidates are intentionally fitted with ``probability=False`` during
    HPO.  Calling ``predict_proba`` in that mode is invalid and enabling it
    would add an expensive hidden calibration CV to every trial.  Raw decision
    scores are sufficient for AUROC/AUPRC candidate ranking; probability
    metrics are computed only after the selected model receives frozen OOF
    calibration in the normal training engine.
    """
    parameters = dict(model_config.get("parameters", {}))
    raw_svm = (
        model_name == "svm"
        and not bool(parameters.get("probability", True))
        and not hasattr(model, "calibrated_classifiers_")
    )
    if not raw_svm:
        return (
            _predict_in_memory(
                model_name=model_name,
                model=model,
                transformed=transformed,
                task=task,
                model_config=model_config,
            ),
            "probability",
        )

    fitted_classes = np.asarray(getattr(model, "classes_", ()), dtype=np.int64)
    expected_classes = np.asarray(task.expected_classes, dtype=np.int64)
    if fitted_classes.shape != expected_classes.shape or not np.array_equal(
        fitted_classes, expected_classes
    ):
        raise ValueError(
            "SVM optimization class order does not match the frozen task contract: "
            f"expected {expected_classes.tolist()}, received {fitted_classes.tolist()}."
        )
    scores = np.asarray(model.decision_function(transformed), dtype=np.float64)
    if task.task_type == "binary":
        scores = scores.reshape(-1)
    else:
        expected = (len(transformed), task.num_classes)
        if scores.shape != expected:
            raise ValueError(
                "Multiclass SVM decision_function must use one-vs-rest scores: "
                f"expected {expected}, received {scores.shape}."
            )
    if not np.isfinite(scores).all():
        raise ValueError("Optimization decision scores contain non-finite values.")
    return scores, "decision_score"


def validate_task_optimization_output_contract(
    values: np.ndarray,
    task: TaskDefinition,
    sample_count: int,
    prediction_kind: str,
) -> None:
    """Validate either probability or decision-score HPO output."""
    if prediction_kind == "probability":
        _validate_probability_contract(values, task, sample_count)
        return
    if prediction_kind != "decision_score":
        raise ValueError(f"Unsupported optimization prediction kind: {prediction_kind}")
    array = np.asarray(values, dtype=np.float64)
    expected = (
        (sample_count,)
        if task.task_type == "binary"
        else (sample_count, task.num_classes)
    )
    if array.shape != expected:
        raise ValueError(
            f"Optimization decision-score shape must be {expected}: {array.shape}"
        )
    if not np.isfinite(array).all():
        raise ValueError("Optimization decision scores contain non-finite values.")


def validate_task_probability_contract(
    probabilities: np.ndarray,
    task: TaskDefinition,
    sample_count: int,
) -> None:
    """Validate the binary or multiclass probability output contract."""
    _validate_probability_contract(probabilities, task, sample_count)

def _safe_binary_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, float | int | None]:
    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    predictions = (probabilities >= threshold).astype(np.int64)
    metrics: dict[str, float | int | None] = {
        "sample_count": int(len(labels)),
        "positive_count": int((labels == 1).sum()),
        "negative_count": int((labels == 0).sum()),
        "positive_rate": float(labels.mean()) if len(labels) else float("nan"),
        "decision_threshold": float(threshold),
        "brier_score": float(brier_score_loss(labels, probabilities)),
        "log_loss": float(
            -np.mean(
                labels * np.log(np.clip(probabilities, 1e-15, 1.0))
                + (1 - labels)
                * np.log(np.clip(1.0 - probabilities, 1e-15, 1.0))
            )
        ),
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
    }
    if np.unique(labels).size == 2:
        metrics["auroc"] = float(roc_auc_score(labels, probabilities))
        metrics["auprc"] = float(average_precision_score(labels, probabilities))
    else:
        metrics["auroc"] = None
        metrics["auprc"] = None
    return metrics


def _safe_multiclass_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    task: TaskDefinition,
) -> dict[str, float | int | None]:
    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    predictions = np.argmax(probabilities, axis=1).astype(np.int64)
    one_hot = np.eye(task.num_classes, dtype=np.float64)[labels]
    per_class_auroc: list[float] = []
    per_class_auprc: list[float] = []
    output: dict[str, float | int | None] = {
        "sample_count": int(len(labels)),
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "macro_f1": float(
            f1_score(labels, predictions, average="macro", zero_division=0)
        ),
        "micro_f1": float(
            f1_score(labels, predictions, average="micro", zero_division=0)
        ),
        "multiclass_brier": float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1))),
        "multiclass_log_loss": float(
            -np.mean(
                np.log(
                    np.clip(
                        probabilities[np.arange(len(labels)), labels],
                        1e-15,
                        1.0,
                    )
                )
            )
        ),
    }
    for class_code, class_name in enumerate(task.class_names):
        binary = (labels == class_code).astype(np.int64)
        key = class_name.lower()
        output[f"class_{class_code}_{key}_count"] = int(binary.sum())
        if np.unique(binary).size == 2:
            auroc = float(roc_auc_score(binary, probabilities[:, class_code]))
            auprc = float(
                average_precision_score(binary, probabilities[:, class_code])
            )
            output[f"class_{class_code}_auroc"] = auroc
            output[f"class_{class_code}_auprc"] = auprc
            per_class_auroc.append(auroc)
            per_class_auprc.append(auprc)
        else:
            output[f"class_{class_code}_auroc"] = None
            output[f"class_{class_code}_auprc"] = None
    output["macro_auroc"] = (
        float(np.mean(per_class_auroc)) if per_class_auroc else None
    )
    output["macro_auprc"] = (
        float(np.mean(per_class_auprc)) if per_class_auprc else None
    )
    return output


def _prediction_frame(
    split: TaskSplitData,
    task: TaskDefinition,
    probabilities: np.ndarray,
    threshold: float | None,
    model_name: str,
) -> pd.DataFrame:
    output = split.metadata.copy().reset_index(drop=True)
    output["task_name"] = task.task_name
    output["model_name"] = model_name
    output["y_true"] = split.labels.astype(np.int64)
    if task.task_type == "binary":
        assert threshold is not None
        output["positive_probability"] = probabilities
        output["predicted_at_validation_threshold"] = (
            probabilities >= threshold
        ).astype(np.int64)
        output["predicted_at_0_5"] = (probabilities >= 0.5).astype(np.int64)
    else:
        for class_code, class_name in enumerate(task.class_names):
            output[f"prob_class_{class_code}"] = probabilities[:, class_code]
            output[f"prob_{class_name}"] = probabilities[:, class_code]
        output["predicted_class"] = np.argmax(probabilities, axis=1).astype(
            np.int64
        )
    return output


def _save_training_history(history: list[dict[str, float]], output_dir: Path) -> None:
    if history:
        pd.DataFrame(history).to_csv(
            output_dir / "training_history.csv", index=False
        )


def _artifact_file_hashes(artifact_dir: Path) -> dict[str, str]:
    return {
        str(path.relative_to(artifact_dir)): sha256_file(path)
        for path in sorted(artifact_dir.rglob("*"))
        if path.is_file()
    }


def train_and_evaluate_single_model(
    task_data: TaskDataBundle,
    model_name: str,
    model_config: dict[str, Any],
    output_dir: Path,
    seed: int = 42,
    writer: Any | None = None,
    prediction_tolerance: float = 1e-7,
    integrity_config: Mapping[str, Any] | None = None,
    project_config: Mapping[str, Any] | None = None,
    training_sampler: TrainingSampler | None = None,
) -> TrainingEngineResult:
    """Run the shared EXP123 train/save/reload/evaluate contract for one model."""
    task = task_data.task
    output_dir = output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Training engine output already exists: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    writer = writer or NullSummaryWriter()

    train_split = task_data.splits["train"]
    validation_split = task_data.splits["validation"]
    validate_training_features(
        train_split.raw_features,
        list(task_data.feature_names),
    )

    # V8 invariant: preprocessing is always fitted on the complete natural
    # training partition before any sampling strategy changes the model-fit
    # distribution. Evaluation splits remain untouched natural patients.
    preprocessor = build_preprocessor(model_name)
    preprocessor.fit(train_split.raw_features.loc[:, list(task_data.feature_names)])
    transformed = {
        split_name: transform_features(
            preprocessor,
            split.raw_features.loc[:, list(task_data.feature_names)],
        )
        for split_name, split in task_data.splits.items()
    }
    sampled_training = (
        training_sampler.sample(
            train_split,
            feature_names=task_data.feature_names,
            seed=int(seed),
            purpose="final_train",
        )
        if training_sampler is not None
        else None
    )
    fit_raw_features = (
        sampled_training.raw_features if sampled_training is not None
        else train_split.raw_features.loc[:, list(task_data.feature_names)].copy().reset_index(drop=True)
    )
    fit_labels = (
        np.asarray(sampled_training.labels, dtype=np.int64)
        if sampled_training is not None
        else np.asarray(train_split.labels, dtype=np.int64)
    )
    transformed_fit = transform_features(
        preprocessor, fit_raw_features.loc[:, list(task_data.feature_names)]
    )
    class_summary = _class_weight_summary(fit_labels, task, model_config)
    atomic_write_json(output_dir / "class_weights.json", class_summary.to_dict())
    if sampled_training is not None:
        atomic_write_json(
            output_dir / "sampling_manifest.json",
            sampling_manifest_payload(sampled_training),
        )

    training_start = time.perf_counter()
    training_history: list[dict[str, float]] = []
    training_info: dict[str, Any]
    calibration_summary: dict[str, Any] = {"enabled": False, "method": "none"}
    model_serialization = "native"
    calibration_config = dict((project_config or {}).get("calibration", {}))
    calibration_enabled = bool(calibration_config.get("enabled", False))
    if (
        model_name == "svm"
        and not bool(dict(model_config.get("parameters", {})).get("probability", True))
        and not calibration_enabled
    ):
        raise ValueError(
            "SVM probability=False requires the frozen OOF calibration stage. "
            "Enable project_config.calibration or provide an explicitly "
            "probabilistic final SVM configuration."
        )
    sklearn_logistic = model_name == "logistic_regression" and uses_sklearn_logistic(model_config)
    if model_name == "logistic_regression" and not sklearn_logistic:
        if calibration_enabled:
            raise ValueError(
                "V7 OOF calibration requires the sklearn logistic-regression implementation."
            )
        if bool(model_config.get("fixed_epoch_training", False)):
            result = train_weighted_task_logistic_regression_fixed_epochs(
                transformed_fit,
                fit_labels,
                model_config,
                writer,
                seed,
                task.task_type,
                task.num_classes,
                class_summary.class_weights,
            )
        else:
            result = train_weighted_task_logistic_regression(
                transformed_fit,
                fit_labels,
                transformed["validation"],
                validation_split.labels,
                model_config,
                writer,
                seed,
                task.task_type,
                task.num_classes,
                class_summary.class_weights,
            )
        model = result.model
        explain_model = model
        training_history = result.history
        training_info = {
            "best_epoch": result.best_epoch,
            "parameters": dict(model_config),
        }
    else:
        base_model = build_task_sklearn_model(
            model_name=model_name,
            model_config=model_config,
            seed=seed,
            task_type=task.task_type,
            num_classes=task.num_classes,
            class_weights=class_summary.class_weights,
        )
        fit_kwargs: dict[str, Any] = {}
        if model_name == "xgboost" and class_summary.method != "none":
            fit_kwargs["sample_weight"] = training_sample_weights(
                fit_labels, class_summary.class_weights
            )
        with tracked_operation(
            f"Train {task.task_name}/{model_name}",
            details={
                "task_name": task.task_name,
                "model_name": model_name,
                "split": "train",
                "sample_count": int(len(fit_labels)),
                "natural_train_sample_count": int(len(train_split.labels)),
                "sampling_strategy": (
                    getattr(training_sampler, "strategy_name", "none")
                    if training_sampler is not None else "none"
                ),
                "device": dict(model_config.get("parameters", {})).get("device", "cpu"),
                "calibration": calibration_enabled,
            },
        ):
            if calibration_enabled:
                group_column = str(
                    calibration_config.get(
                        "group_column",
                        dict((project_config or {}).get("data", {})).get(
                            "patient_id_column", "subject_id"
                        ),
                    )
                )
                if group_column not in train_split.metadata.columns:
                    raise KeyError(
                        f"Calibration group column is missing from train metadata: {group_column}"
                    )
                if fit_kwargs:
                    raise ValueError(
                        "Calibrated primary tracks require class_weight_mode=none; "
                        "sample-weighted calibration is intentionally disabled."
                    )
                if training_sampler is not None:
                    calibrators, fitted_calibration, score_kind = (
                        fit_sampling_aware_calibration_map(
                            natural_train=train_split,
                            feature_names=task_data.feature_names,
                            task=task,
                            model_name=model_name,
                            model_config=model_config,
                            sampler=training_sampler,
                            config=calibration_config,
                            group_column=group_column,
                            sampling_member_seed=int(seed),
                        )
                    )
                    # The final base estimator is fitted on the sampled full train.
                    # Calibration never refits it on the natural distribution.
                    base_model.fit(transformed_fit, fit_labels)
                    model = FrozenOOFSigmoidClassifier(
                        estimator=base_model,
                        calibrators=calibrators,
                        task_type=task.task_type,
                        expected_classes=task.expected_classes,
                        score_kind=score_kind,
                    )
                    calibration_summary = fitted_calibration.to_dict()
                    model_serialization = "joblib"
                    explain_model = model if model_name == "svm" else base_model
                else:
                    model, fitted_calibration = build_group_calibrated_classifier(
                        estimator=base_model,
                        features=transformed["train"],
                        labels=train_split.labels,
                        groups=train_split.metadata[group_column].to_numpy(),
                        config=calibration_config,
                        group_column=group_column,
                    )
                    calibration_summary = fitted_calibration.to_dict()
                    model_serialization = "joblib"
                    explain_model = (
                        model if model_name == "svm" else explainability_estimator(model)
                    )
            else:
                base_model.fit(transformed_fit, fit_labels, **fit_kwargs)
                model = base_model
                explain_model = model
        effective_parameters = (
            explain_model.get_params()
            if hasattr(explain_model, "get_params")
            else dict(model_config)
        )
        training_info = {"parameters": effective_parameters}
    training_seconds = float(time.perf_counter() - training_start)
    training_info["training_seconds"] = training_seconds
    atomic_write_json(output_dir / "training_info.json", training_info)
    _save_training_history(training_history, output_dir)

    memory_probabilities = {
        split_name: _predict_in_memory(
            model_name,
            model,
            transformed[split_name],
            task,
            model_config,
        )
        for split_name in task_data.splits
    }
    for split_name, probabilities in memory_probabilities.items():
        _validate_probability_contract(
            probabilities,
            task,
            len(task_data.splits[split_name].labels),
        )

    threshold: float | None
    if task.threshold_required:
        threshold = float(
            select_youden_threshold(
                validation_split.labels,
                memory_probabilities["validation"],
            )
        )
    else:
        threshold = None

    artifact_metadata: dict[str, Any] = {
        "artifact_schema_version": "exp123_task_v1",
        "model_name": model_name,
        "task_name": task.task_name,
        "task_type": task.task_type,
        "population": task.population,
        "num_classes": task.num_classes,
        "class_names": list(task.class_names),
        "feature_names": list(task_data.feature_names),
        "validation_decision_threshold": threshold,
        "threshold_selection_split": (
            "validation" if task.threshold_required else None
        ),
        "decision_rule": (
            "validation_youden" if task.threshold_required else "argmax"
        ),
        "training_seed": int(seed),
        "shared_split_hash": task_data.shared_split_hash,
        "class_weight_summary": class_summary.to_dict(),
        "model_implementation": (
            "sklearn" if sklearn_logistic else ("torch" if model_name == "logistic_regression" else "native")
        ),
        "model_serialization": model_serialization,
        "calibration": calibration_summary,
        "model_config": training_info["parameters"],
        "model_config_requested": model_config,
        "model_config_effective": training_info["parameters"],
        "inference_device": training_info["parameters"].get(
            "device", model_config.get("device", "auto")
        ) if isinstance(training_info["parameters"], dict) else model_config.get("device", "auto"),
        "split_sample_counts": {
            name: int(len(split.labels))
            for name, split in task_data.splits.items()
        },
        "model_fit_sample_count": int(len(fit_labels)),
        "natural_train_sample_count": int(len(train_split.labels)),
        "sampling": (
            sampling_manifest_payload(sampled_training)
            if sampled_training is not None
            else {
                "schema_version": "exp123_sampling_v8_1",
                "strategy": "natural",
                "natural_count": int(len(train_split.labels)),
                "synthetic_count": 0,
                "fit_count": int(len(train_split.labels)),
            }
        ),
        "training_seconds": training_seconds,
        "svm_runtime": (
            {
                "backend": svm_backend_label(model_config),
                "policy": svm_runtime_settings(project_config),
                "probability_during_base_fit": bool(
                    dict(model_config.get("parameters", {})).get("probability", True)
                ),
            }
            if model_name == "svm" else None
        ),
    }
    feature_set_payload = None
    if project_config is not None:
        raw_feature_set = project_config.get("feature_set")
        if isinstance(raw_feature_set, dict):
            feature_set_payload = {
                "set_id": raw_feature_set.get("set_id"),
                "sha256": raw_feature_set.get("sha256"),
                "feature_count": raw_feature_set.get("feature_count", len(task_data.feature_names)),
                "registry_schema_version": raw_feature_set.get("registry_schema_version"),
            }
        elif project_config.get("data", {}).get("feature_set_id"):
            feature_set_payload = {
                "set_id": project_config["data"].get("feature_set_id"),
                "sha256": project_config["data"].get("feature_set_sha256"),
                "feature_count": len(task_data.feature_names),
            }
    if feature_set_payload is not None:
        artifact_metadata["feature_set"] = feature_set_payload
    explainability_config = dict((project_config or {}).get("explainability", {})) if project_config is not None else {}
    explainability_result = save_explainability(
        model_name=model_name,
        model=explain_model,
        X_background=transformed["train"],
        X_test=transformed["validation"],
        feature_names=list(task_data.feature_names),
        output_dir=output_dir / "explainability",
        config=explainability_config,
        task_type=task.task_type,
        class_names=list(task.class_names),
    ) if explainability_config else {"status": "not_configured"}
    artifact_metadata["explainability"] = explainability_result

    artifact_dir = output_dir / "artifact"
    save_task_artifact(
        artifact_dir=artifact_dir,
        model_name=model_name,
        model=model,
        preprocessor=preprocessor,
        metadata=artifact_metadata,
    )

    reload_start = time.perf_counter()
    loaded = load_task_artifact(artifact_dir)
    artifact_reload_seconds = float(time.perf_counter() - reload_start)

    prediction_integrity: dict[str, PredictionIntegrityResult] = {}
    split_metrics: dict[str, dict[str, float | int | None]] = {}
    for split_name, split in task_data.splits.items():
        reloaded_probability = loaded.predict_proba(split.raw_features)
        _validate_probability_contract(
            reloaded_probability,
            task,
            len(split.labels),
        )
        memory_probability = memory_probabilities[split_name]
        integrity = compare_prediction_integrity(
            split_name=split_name,
            memory_probability=memory_probability,
            reloaded_probability=reloaded_probability,
            task=task,
            threshold=threshold,
            model_config=model_config,
            integrity_config=integrity_config,
            default_cpu_tolerance=prediction_tolerance,
        )
        prediction_integrity[split_name] = integrity
        if not integrity.matches:
            raise RuntimeError(
                f"Reloaded artifact predictions differ on {split_name}: "
                f"max_abs_diff={integrity.maximum_absolute_difference}, "
                f"atol={integrity.absolute_tolerance}, "
                f"rtol={integrity.relative_tolerance}, "
                f"decision_mismatches={integrity.mismatched_decision_count}"
            )

        if task.task_type == "binary":
            assert threshold is not None
            metrics = _safe_binary_metrics(
                split.labels,
                reloaded_probability,
                threshold,
            )
        else:
            metrics = _safe_multiclass_metrics(
                split.labels,
                reloaded_probability,
                task,
            )
        split_metrics[split_name] = metrics
        log_metrics({
            f"metrics/{task.task_name}/{model_name}/{split_name}/{key}": value
            for key, value in metrics.items()
            if isinstance(value, (int, float)) and value is not None
        })
        split_dir = output_dir / "evaluation" / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        _prediction_frame(
            split,
            task,
            reloaded_probability,
            threshold,
            model_name,
        ).to_csv(
            split_dir / "predictions.csv.gz",
            index=False,
            compression="gzip",
        )
        atomic_write_json(split_dir / "metrics.json", metrics)

    integrity_payload = {
        "all_splits_match": all(
            value.matches for value in prediction_integrity.values()
        ),
        "prediction_tolerance": prediction_tolerance,
        "integrity_config": dict(integrity_config or {}),
        "splits": {
            key: value.to_dict()
            for key, value in prediction_integrity.items()
        },
        "artifact_file_sha256": _artifact_file_hashes(artifact_dir),
        "xgboost_prediction_adapter": loaded.metadata.get(
            "last_xgboost_prediction_adapter"
        ),
    }
    atomic_write_json(output_dir / "artifact_integrity.json", integrity_payload)
    atomic_write_json(output_dir / "task_definition.json", task.to_dict())

    result = TrainingEngineResult(
        task_name=task.task_name,
        model_name=model_name,
        output_dir=str(output_dir),
        artifact_dir=str(artifact_dir),
        validation_decision_threshold=threshold,
        class_weights=class_summary.class_weights,
        split_metrics=split_metrics,
        prediction_integrity=prediction_integrity,
        training_seconds=training_seconds,
        artifact_reload_seconds=artifact_reload_seconds,
    )
    atomic_write_json(output_dir / "engine_result.json", result.to_dict())
    tracker = get_tracker()
    if tracker is not None:
        tracker.log_artifact(
            artifact_dir,
            name=f"{task.task_name}-{model_name}-artifact",
            artifact_type="model",
            metadata={
                "task_name": task.task_name,
                "model_name": model_name,
                "shared_split_hash": task_data.shared_split_hash,
            },
        )
    writer.close()
    return result


def load_engine_result(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)
