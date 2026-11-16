from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class NestedCVProtocol:
    outer_folds: int
    inner_folds: int
    seed: int
    selection_metric: str
    selection_feature_set: str
    hyperparameter_grid: tuple[dict[str, int | float], ...]
    enforce_expected_locked_parameters: bool


@dataclass(frozen=True)
class TemporalSplitProtocol:
    column: str
    development_groups: tuple[str, ...]
    evaluation_groups: tuple[str, ...]
    expected_development_n: int | None
    expected_evaluation_n: int | None


@dataclass(frozen=True)
class PrimaryModelProtocol:
    name: str
    implementation: str
    parameters: dict[str, Any]
    expected_locked_search_parameters: dict[str, Any]


@dataclass(frozen=True)
class Study28Protocol:
    schema_version: str
    study_id: str
    primary_feature_set: str
    comparison_feature_sets: tuple[str, ...]
    nested_cv: NestedCVProtocol
    temporal_split: TemporalSplitProtocol
    primary_model: PrimaryModelProtocol
    primary_metric: str
    required_external_sites: tuple[str, ...]
    source_path: str
    sha256: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tuple_of_strings(values: Sequence[Any], *, name: str) -> tuple[str, ...]:
    result = tuple(str(value) for value in values)
    if not result:
        raise ValueError(f"{name} must not be empty.")
    if len(result) != len(set(result)):
        raise ValueError(f"{name} contains duplicates: {result}")
    return result


def load_study28_protocol(path: Path) -> Study28Protocol:
    path = Path(path).expanduser().resolve()
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping):
        raise TypeError("study28 protocol must be a JSON object.")

    development = payload.get("development", {})
    nested = development.get("nested_cv", {})
    temporal = development.get("temporal_split", {})
    features = payload.get("features", {})
    primary_model = payload.get("primary_model", {})
    evaluation = payload.get("evaluation", {})

    grid_payload = nested.get("hyperparameter_grid", [])
    grid = tuple(dict(item) for item in grid_payload)
    if not grid:
        raise ValueError("nested_cv.hyperparameter_grid must not be empty.")
    for candidate in grid:
        if "num_leaves" not in candidate or "n_estimators" not in candidate:
            raise ValueError(
                "Every Study28 LightGBM grid candidate must contain num_leaves and n_estimators."
            )

    nested_protocol = NestedCVProtocol(
        outer_folds=int(nested.get("outer_folds", 5)),
        inner_folds=int(nested.get("inner_folds", 3)),
        seed=int(nested.get("seed", 2026)),
        selection_metric=str(nested.get("selection_metric", "macro_auroc")),
        selection_feature_set=str(nested.get("selection_feature_set", "FS_FULL")),
        hyperparameter_grid=grid,
        enforce_expected_locked_parameters=bool(
            nested.get("enforce_expected_locked_parameters", True)
        ),
    )
    if nested_protocol.outer_folds < 2 or nested_protocol.inner_folds < 2:
        raise ValueError("Nested CV requires at least two outer and inner folds.")
    if nested_protocol.selection_metric != "macro_auroc":
        raise ValueError(
            "The frozen professor protocol selects LightGBM hyperparameters by macro_auroc."
        )

    development_groups = _tuple_of_strings(
        temporal.get("development_groups", []), name="temporal development groups"
    )
    evaluation_groups = _tuple_of_strings(
        temporal.get("evaluation_groups", []), name="temporal evaluation groups"
    )
    overlap = sorted(set(development_groups) & set(evaluation_groups))
    if overlap:
        raise ValueError(f"Temporal development/evaluation groups overlap: {overlap}")
    temporal_protocol = TemporalSplitProtocol(
        column=str(temporal.get("column", "anchor_year_group")),
        development_groups=development_groups,
        evaluation_groups=evaluation_groups,
        expected_development_n=(
            int(temporal["expected_development_n"])
            if temporal.get("expected_development_n") is not None
            else None
        ),
        expected_evaluation_n=(
            int(temporal["expected_evaluation_n"])
            if temporal.get("expected_evaluation_n") is not None
            else None
        ),
    )

    model_parameters = dict(primary_model.get("parameters", {}))
    expected_locked = dict(
        primary_model.get(
            "expected_locked_search_parameters",
            {
                key: model_parameters[key]
                for key in ("num_leaves", "n_estimators")
                if key in model_parameters
            },
        )
    )
    if not expected_locked:
        raise ValueError("primary_model.expected_locked_search_parameters is required.")

    model_protocol = PrimaryModelProtocol(
        name=str(primary_model.get("name", "lightgbm_multiclass_noapsiii")),
        implementation=str(primary_model.get("implementation", "lightgbm")),
        parameters=model_parameters,
        expected_locked_search_parameters=expected_locked,
    )
    if model_protocol.implementation != "lightgbm":
        raise ValueError("Study28 Alpha 2 primary implementation must be lightgbm.")

    return Study28Protocol(
        schema_version=str(payload.get("schema_version", "study28_protocol_v2")),
        study_id=str(payload.get("study_id", "sepsis3_28d_three_class_disposition")),
        primary_feature_set=str(features.get("primary_set", "FS_NOAPS")),
        comparison_feature_sets=tuple(str(v) for v in features.get("comparison_sets", [])),
        nested_cv=nested_protocol,
        temporal_split=temporal_protocol,
        primary_model=model_protocol,
        primary_metric=str(evaluation.get("primary_metric", "macro_auroc")),
        required_external_sites=tuple(
            str(v) for v in evaluation.get("required_external_sites", [])
        ),
        source_path=str(path),
        sha256=_sha256(path),
    )
