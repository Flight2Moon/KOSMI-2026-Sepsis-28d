from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence, TypeVar

from .task_spec import TaskDefinition
from .utils import sha256_file

T = TypeVar("T")


def load_json_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required JSON file was not found: {path}")
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a JSON object in {path}")
    return payload


def load_shared_split_hash(run_dir: Path) -> str:
    path = run_dir / "common" / "splits" / "split_hash.txt"
    if not path.is_file():
        raise FileNotFoundError(f"Shared split hash was not found: {path}")
    value = path.read_text(encoding="utf-8").strip()
    if len(value) != 64:
        raise ValueError(f"Invalid shared split SHA-256 in {path}: {value}")
    return value


def validate_selected_hyperparameters(
    *,
    path: Path,
    model_name: str,
    task: TaskDefinition,
    shared_split_hash: str | None,
) -> tuple[dict[str, Any], str]:
    payload = load_json_object(path)
    if payload.get("task_name") != task.task_name:
        raise ValueError(
            f"Optimization result has the wrong task: {payload.get('task_name')}"
        )
    if payload.get("model_name") != model_name:
        raise ValueError(
            f"Optimization result has the wrong model: {payload.get('model_name')}"
        )
    if payload.get("confirmatory_splits_accessed") is not False:
        raise ValueError(
            "Optimization selection must explicitly record that confirmatory "
            "splits were not accessed."
        )
    if shared_split_hash is not None and payload.get("shared_split_hash") != shared_split_hash:
        raise ValueError(
            "Optimization selection and training use different shared split hashes."
        )
    model_config = payload.get("model_config")
    if not isinstance(model_config, dict):
        raise TypeError(f"model_config must be a JSON object in {path}")
    digest = sha256_file(path)
    digest_path = path.with_name(path.name + ".sha256")
    if digest_path.is_file():
        recorded = digest_path.read_text(encoding="utf-8").strip()
        if recorded != digest:
            raise ValueError(f"Selected hyperparameter hash mismatch: {path}")
    return dict(model_config), digest


def select_validation_reference_model(
    model_results: Sequence[T],
    *,
    reference_metric: str,
    metric_attributes: Mapping[str, str],
    secondary_attribute: str,
    practical_tie_margin: float,
    configured_model_order: Sequence[str],
    selected_flag: str,
) -> tuple[T, list[dict[str, Any]]]:
    if not model_results:
        raise ValueError("At least one model result is required.")
    metric_attribute = metric_attributes.get(reference_metric)
    if metric_attribute is None:
        raise ValueError(f"Unsupported reference metric: {reference_metric}")
    rows = [result.to_dict() for result in model_results]  # type: ignore[attr-defined]
    best_primary = max(float(row[metric_attribute]) for row in rows)
    order = {name: index for index, name in enumerate(configured_model_order)}
    for row in rows:
        primary = float(row[metric_attribute])
        row[reference_metric] = primary
        row["primary_metric"] = reference_metric
        row["primary_metric_value"] = primary
        row["within_practical_tie"] = bool(
            primary >= best_primary - practical_tie_margin
        )
        row[selected_flag] = False
    selected_row = min(
        rows,
        key=lambda row: (
            0 if row["within_practical_tie"] else 1,
            float(row[secondary_attribute]),
            int(order[str(row["model_name"])]),
        ),
    )
    selected_row[selected_flag] = True
    selected = next(
        result for result in model_results
        if getattr(result, "model_name") == selected_row["model_name"]
    )
    return selected, rows
