from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from .task_spec import TaskDefinition

SPLIT_ORDER = ("train", "validation", "internal_test", "external_eicu")


@dataclass(frozen=True)
class TaskSplitData:
    split_name: str
    raw_features: pd.DataFrame
    labels: np.ndarray
    metadata: pd.DataFrame

    def __post_init__(self) -> None:
        sample_count = len(self.raw_features)
        if len(self.labels) != sample_count or len(self.metadata) != sample_count:
            raise ValueError(
                f"Split {self.split_name} has inconsistent lengths: "
                f"features={sample_count}, labels={len(self.labels)}, "
                f"metadata={len(self.metadata)}"
            )
        if self.raw_features.index.has_duplicates:
            raise ValueError(f"Split {self.split_name} feature index is duplicated.")
        if "source_row_index" not in self.metadata.columns:
            raise KeyError(
                f"Split {self.split_name} metadata must contain source_row_index."
            )
        if self.metadata["source_row_index"].duplicated().any():
            raise ValueError(
                f"Split {self.split_name} contains duplicated source_row_index."
            )
        if "is_synthetic" in self.metadata.columns:
            synthetic = self.metadata["is_synthetic"].fillna(False).astype(bool)
            if bool(synthetic.any()):
                raise ValueError(
                    f"Split {self.split_name} contains {int(synthetic.sum())} synthetic rows. "
                    "V8 natural Train/Validation/Internal-Test/External splits must remain "
                    "synthetic-free; sampled rows belong in SampledTrainingData only."
                )


@dataclass(frozen=True)
class TaskDataBundle:
    task: TaskDefinition
    feature_names: tuple[str, ...]
    splits: Mapping[str, TaskSplitData]
    shared_split_hash: str | None = None

    def __post_init__(self) -> None:
        required = {"train", "validation"}
        missing = sorted(required - set(self.splits))
        if missing:
            raise ValueError(f"Task data bundle is missing required splits: {missing}")
        if not self.feature_names:
            raise ValueError("feature_names must not be empty.")
        for split_name, split in self.splits.items():
            missing_features = sorted(
                set(self.feature_names) - set(split.raw_features.columns)
            )
            if missing_features:
                raise KeyError(
                    f"Split {split_name} is missing features: {missing_features}"
                )
            observed = set(np.unique(split.labels).astype(int).tolist())
            invalid = sorted(observed - set(self.task.expected_classes))
            if invalid:
                raise ValueError(
                    f"Split {split_name} has invalid labels for {self.task.task_name}: "
                    f"{invalid}"
                )
            if split_name in {"train", "validation"} and observed != set(
                self.task.expected_classes
            ):
                raise ValueError(
                    f"Split {split_name} must contain all task classes. "
                    f"Observed: {sorted(observed)}"
                )


def _task_population_mask(
    assignment: pd.DataFrame,
    task: TaskDefinition,
) -> pd.Series:
    if task.population == "all":
        return pd.Series(True, index=assignment.index)
    return assignment["stage2_eligible"].astype(bool)


def _task_labels(
    assignment: pd.DataFrame,
    task: TaskDefinition,
) -> np.ndarray:
    if task.task_name == "exp1_flat_3class":
        return assignment["three_class_outcome"].to_numpy(dtype=np.int64)
    if task.task_name == "stage1_mortality":
        return assignment["stage1_mortality_label"].to_numpy(dtype=np.int64)
    if task.task_name == "stage2_long_stay":
        values = pd.to_numeric(
            assignment["stage2_long_stay_label"], errors="raise"
        )
        return values.to_numpy(dtype=np.int64)
    raise KeyError(f"Unsupported task assignment mapping: {task.task_name}")


def build_task_data_bundle(
    feature_frame: pd.DataFrame,
    split_assignment: pd.DataFrame,
    task: TaskDefinition,
    feature_names: Sequence[str],
    external_feature_frame: pd.DataFrame | None = None,
    external_outcome_assignment: pd.DataFrame | None = None,
    shared_split_hash: str | None = None,
    split_names: Sequence[str] = ("train", "validation", "internal_test"),
) -> TaskDataBundle:
    """Build task-specific split data from the single shared split assignment."""
    feature_names = tuple(str(value) for value in feature_names)
    split_names = tuple(str(value) for value in split_names)
    allowed_split_names = {"train", "validation", "internal_test"}
    unknown_split_names = sorted(set(split_names) - allowed_split_names)
    if unknown_split_names:
        raise ValueError(f"Unknown requested split names: {unknown_split_names}")
    if not {"train", "validation"}.issubset(split_names):
        raise ValueError("split_names must include train and validation.")
    required_assignment_columns = {
        "source_row_index",
        "split",
        "three_class_outcome",
        "stage1_mortality_label",
        "stage2_eligible",
        "stage2_long_stay_label",
    }
    missing_assignment = sorted(
        required_assignment_columns - set(split_assignment.columns)
    )
    if missing_assignment:
        raise KeyError(
            f"Common split assignment is missing columns: {missing_assignment}"
        )
    missing_features = sorted(set(feature_names) - set(feature_frame.columns))
    if missing_features:
        raise KeyError(f"Feature frame is missing columns: {missing_features}")
    source_column = (
        "_source_row_index"
        if "_source_row_index" in feature_frame.columns
        else "source_row_index"
    )
    if source_column not in feature_frame.columns:
        raise KeyError("Feature frame must contain a stable source row index.")
    if feature_frame[source_column].duplicated().any():
        raise ValueError("Feature frame source row indices must be unique.")

    feature_lookup = feature_frame.set_index(source_column, drop=False)
    split_data: dict[str, TaskSplitData] = {}
    for split_name in split_names:
        assigned = split_assignment.loc[
            split_assignment["split"] == split_name
        ].copy()
        assigned = assigned.loc[_task_population_mask(assigned, task)].copy()
        assigned = assigned.sort_values("source_row_index", kind="mergesort")
        missing_source = sorted(
            set(assigned["source_row_index"].astype(int))
            - set(feature_lookup.index.astype(int))
        )
        if missing_source:
            raise ValueError(
                f"Split {split_name} has {len(missing_source)} rows that are "
                "absent from the feature frame."
            )
        raw = feature_lookup.loc[
            assigned["source_row_index"].to_numpy(), list(feature_names)
        ].copy()
        raw.index = pd.RangeIndex(len(raw))
        labels = _task_labels(assigned, task)
        metadata_columns = [
            column
            for column in assigned.columns
            if column not in set(feature_names)
        ]
        metadata = assigned.loc[:, metadata_columns].reset_index(drop=True)
        split_data[split_name] = TaskSplitData(
            split_name=split_name,
            raw_features=raw,
            labels=labels,
            metadata=metadata,
        )

    if (external_feature_frame is None) != (external_outcome_assignment is None):
        raise ValueError(
            "external_feature_frame and external_outcome_assignment must be "
            "provided together."
        )
    if external_feature_frame is not None and external_outcome_assignment is not None:
        external = external_outcome_assignment.copy()
        if "source_row_index" not in external.columns:
            if "_source_row_index" in external.columns:
                external = external.rename(
                    columns={"_source_row_index": "source_row_index"}
                )
            else:
                raise KeyError(
                    "External outcome assignment must contain source_row_index."
                )
        external = external.loc[_task_population_mask(external, task)].copy()
        external = external.sort_values("source_row_index", kind="mergesort")
        external_source_column = (
            "_source_row_index"
            if "_source_row_index" in external_feature_frame.columns
            else "source_row_index"
        )
        external_lookup = external_feature_frame.set_index(
            external_source_column, drop=False
        )
        raw = external_lookup.loc[
            external["source_row_index"].to_numpy(), list(feature_names)
        ].copy()
        raw.index = pd.RangeIndex(len(raw))
        split_data["external_eicu"] = TaskSplitData(
            split_name="external_eicu",
            raw_features=raw,
            labels=_task_labels(external, task),
            metadata=external.reset_index(drop=True),
        )

    return TaskDataBundle(
        task=task,
        feature_names=feature_names,
        splits=split_data,
        shared_split_hash=shared_split_hash,
    )


def load_feature_frame(
    csv_path: Path,
    feature_names: Sequence[str],
) -> pd.DataFrame:
    csv_path = csv_path.expanduser().resolve()
    header = pd.read_csv(csv_path, nrows=0, compression="infer")
    available = set(header.columns.astype(str).str.strip())
    missing = sorted(set(feature_names) - available)
    if missing:
        raise KeyError(f"Dataset is missing EXP123 features: {missing}")
    frame = pd.read_csv(
        csv_path,
        usecols=list(feature_names),
        low_memory=False,
        compression="infer",
    )
    frame.columns = frame.columns.astype(str).str.strip()
    frame.insert(0, "_source_row_index", np.arange(len(frame), dtype=np.int64))
    if "gender" in feature_names:
        normalized = frame["gender"].astype("string").str.strip().str.upper()
        frame["gender"] = normalized.map(
            {"M": 0.0, "MALE": 0.0, "0": 0.0, "F": 1.0, "FEMALE": 1.0, "1": 1.0}
        )
    for column in feature_names:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame.loc[:, list(feature_names)] = frame.loc[:, list(feature_names)].replace(
        [np.inf, -np.inf], np.nan
    )
    return frame
