from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    roc_auc_score,
)

from .task_spec import TaskDefinition


@dataclass(frozen=True)
class ObjectiveDefinition:
    primary: str
    direction: str
    secondary: str
    secondary_direction: str

    def __post_init__(self) -> None:
        if self.direction not in {"maximize", "minimize"}:
            raise ValueError(f"Invalid primary direction: {self.direction}")
        if self.secondary_direction not in {"maximize", "minimize"}:
            raise ValueError(
                f"Invalid secondary direction: {self.secondary_direction}"
            )

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


def load_objective_definition(
    task_name: str,
    objective_config: Mapping[str, Any],
) -> ObjectiveDefinition:
    try:
        payload = objective_config[task_name]
    except KeyError as exception:
        raise KeyError(f"Missing optimization objective for {task_name}") from exception
    return ObjectiveDefinition(
        primary=str(payload["primary"]),
        direction=str(payload["direction"]),
        secondary=str(payload["secondary"]),
        secondary_direction=str(payload["secondary_direction"]),
    )


def threshold_free_metrics(
    labels: np.ndarray,
    predictions: np.ndarray,
    task: TaskDefinition,
    *,
    prediction_kind: str = "probability",
) -> dict[str, float]:
    y = np.asarray(labels, dtype=np.int64)
    p = np.asarray(predictions, dtype=np.float64)
    if prediction_kind not in {"probability", "decision_score"}:
        raise ValueError(f"Unsupported prediction_kind: {prediction_kind}")
    if task.task_type == "binary":
        if p.shape != (len(y),):
            raise ValueError(f"Invalid binary prediction shape: {p.shape}")
        if np.unique(y).size < 2:
            raise ValueError("Binary optimization fold must contain both classes.")
        output = {
            "auroc": float(roc_auc_score(y, p)),
            "auprc": float(average_precision_score(y, p)),
        }
        if prediction_kind == "decision_score":
            pred = (p >= 0.0).astype(np.int64)
            output.update(
                {
                    "balanced_accuracy": float(
                        balanced_accuracy_score(y, pred)
                    ),
                    "f1": float(f1_score(y, pred, zero_division=0)),
                }
            )
            return output
        pred = (p >= 0.5).astype(np.int64)
        output.update(
            {
                "brier_score": float(brier_score_loss(y, p)),
                "log_loss": float(
                    log_loss(
                        y,
                        np.column_stack([1.0 - p, p]),
                        labels=[0, 1],
                    )
                ),
                "balanced_accuracy": float(
                    balanced_accuracy_score(y, pred)
                ),
                "f1": float(f1_score(y, pred, zero_division=0)),
            }
        )
        return output

    expected = (len(y), task.num_classes)
    if p.shape != expected:
        raise ValueError(
            f"Invalid multiclass prediction shape: {p.shape}, expected {expected}"
        )
    predictions = np.argmax(p, axis=1)
    class_aurocs: list[float] = []
    class_auprcs: list[float] = []
    output: dict[str, float] = {
        "macro_f1": float(f1_score(y, predictions, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(y, predictions, average="micro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y, predictions)),
    }
    if prediction_kind == "probability":
        row_sums = p.sum(axis=1, keepdims=True)
        if (row_sums <= 0).any():
            raise ValueError(
                "Multiclass probabilities contain a non-positive row sum."
            )
        p = p / row_sums
        one_hot = np.eye(task.num_classes, dtype=np.float64)[y]
        output.update(
            {
                "multiclass_brier": float(
                    np.mean(np.sum((p - one_hot) ** 2, axis=1))
                ),
                "multiclass_log_loss": float(
                    log_loss(y, p, labels=list(task.expected_classes))
                ),
            }
        )
    for class_code in task.expected_classes:
        binary = (y == class_code).astype(np.int64)
        if np.unique(binary).size < 2:
            raise ValueError(
                f"Multiclass optimization fold is missing class {class_code}."
            )
        auroc = float(roc_auc_score(binary, p[:, class_code]))
        auprc = float(average_precision_score(binary, p[:, class_code]))
        output[f"class_{class_code}_auroc"] = auroc
        output[f"class_{class_code}_auprc"] = auprc
        class_aurocs.append(auroc)
        class_auprcs.append(auprc)
    output["macro_auroc"] = float(np.mean(class_aurocs))
    output["macro_auprc"] = float(np.mean(class_auprcs))
    return output


def aggregate_fold_metrics(
    fold_metrics: Sequence[Mapping[str, float]],
) -> dict[str, float]:
    if not fold_metrics:
        raise ValueError("fold_metrics must not be empty.")
    common_keys = set(fold_metrics[0])
    for values in fold_metrics[1:]:
        common_keys &= set(values)
    output: dict[str, float] = {}
    for key in sorted(common_keys):
        samples = np.asarray([float(values[key]) for values in fold_metrics], dtype=float)
        output[f"mean_{key}"] = float(np.mean(samples))
        output[f"std_{key}"] = float(np.std(samples, ddof=0))
        output[f"min_{key}"] = float(np.min(samples))
        output[f"max_{key}"] = float(np.max(samples))
    output["fold_count"] = float(len(fold_metrics))
    return output


def robust_selection_score(
    aggregate_metrics: Mapping[str, float],
    objective: ObjectiveDefinition,
    stability_penalty: float,
    overfit_penalty: float = 0.5,
) -> float:
    mean_key = f"mean_{objective.primary}"
    std_key = f"std_{objective.primary}"
    if mean_key not in aggregate_metrics or std_key not in aggregate_metrics:
        raise KeyError(
            f"Aggregate metrics do not contain objective {objective.primary}."
        )
    mean_value = float(aggregate_metrics[mean_key])
    std_value = float(aggregate_metrics[std_key])
    if objective.direction == "maximize":
        score = mean_value - float(stability_penalty) * std_value
        train_key = f"mean_train_{objective.primary}"
        if train_key in aggregate_metrics:
            gap = max(0.0, float(aggregate_metrics[train_key]) - mean_value)
            score -= float(overfit_penalty) * gap
        return score
    score = -mean_value - float(stability_penalty) * std_value
    train_key = f"mean_train_{objective.primary}"
    if train_key in aggregate_metrics:
        # For minimized metrics, validation worse than training means a positive gap.
        gap = max(0.0, mean_value - float(aggregate_metrics[train_key]))
        score -= float(overfit_penalty) * gap
    return score


def metric_is_better(
    candidate: float,
    incumbent: float,
    direction: str,
) -> bool:
    if direction == "maximize":
        return candidate > incumbent
    if direction == "minimize":
        return candidate < incumbent
    raise ValueError(f"Unsupported direction: {direction}")


def select_validation_candidate(
    candidates: Sequence[Mapping[str, Any]],
    task: TaskDefinition,
    objective: ObjectiveDefinition,
    practical_tie_margin: float,
    auroc_margin_guard: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not candidates:
        raise ValueError("Validation candidates must not be empty.")
    rows = [dict(row) for row in candidates]
    primary_values = [float(row[objective.primary]) for row in rows]
    best_primary = (
        max(primary_values)
        if objective.direction == "maximize"
        else min(primary_values)
    )
    best_auroc = None
    if task.task_type == "binary":
        best_auroc = max(float(row["auroc"]) for row in rows)

    eligible: list[dict[str, Any]] = []
    for row in rows:
        primary = float(row[objective.primary])
        if objective.direction == "maximize":
            primary_ok = primary >= best_primary - practical_tie_margin
        else:
            primary_ok = primary <= best_primary + practical_tie_margin
        auroc_ok = True
        if best_auroc is not None:
            auroc_ok = float(row["auroc"]) >= best_auroc - auroc_margin_guard
        row["primary_tie_guard_passed"] = bool(primary_ok)
        row["auroc_margin_guard_passed"] = bool(auroc_ok)
        row["eligible_for_selection"] = bool(primary_ok and auroc_ok)
        if row["eligible_for_selection"]:
            eligible.append(row)

    if not eligible:
        raise RuntimeError("No validation candidate passed the selection guards.")

    def sort_key(row: Mapping[str, Any]) -> tuple[float, float, float]:
        secondary = float(row[objective.secondary])
        secondary_key = (
            -secondary
            if objective.secondary_direction == "maximize"
            else secondary
        )
        primary = float(row[objective.primary])
        primary_key = -primary if objective.direction == "maximize" else primary
        complexity = float(row.get("complexity_score", 0.0))
        return secondary_key, primary_key, complexity

    selected = min(eligible, key=sort_key)
    selected_id = str(selected["candidate_id"])
    for row in rows:
        row["selected"] = str(row["candidate_id"]) == selected_id
    return dict(selected), rows
