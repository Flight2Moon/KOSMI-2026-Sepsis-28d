"""Shared dataset primitives plus legacy LOS-threshold data loading.

The EXP123 V6 paper pipeline uses the split primitives in this module, while
outcome construction is handled by outcomes.py and task loading by exp123_data.py.
The ClinicalDataset/LOS-threshold helpers remain for backward compatibility
with run.sh and run_all_experiments.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, StratifiedShuffleSplit

from .los_derivation import (
    derive_hospital_los,
    normalise_los_derivation_config,
    required_los_derivation_columns,
)


@dataclass
class ClinicalDataset:
    frame: pd.DataFrame
    feature_columns: list[str]
    target_column: str
    identifier_columns: list[str]
    source_path: Path
    removed_repeated_headers: int
    removed_missing_targets: int
    los_derivation_summary: dict[str, Any] | None = None

    @property
    def X(self) -> pd.DataFrame:
        return self.frame.loc[:, self.feature_columns]

    @property
    def los(self) -> np.ndarray:
        return self.frame[self.target_column].to_numpy(dtype=np.float64)

    @property
    def source_row_index(self) -> np.ndarray:
        return self.frame["_source_row_index"].to_numpy(dtype=np.int64)

    def labels(self, threshold_days: float) -> np.ndarray:
        return (self.los >= float(threshold_days)).astype(np.int32)


@dataclass
class DatasetSplits:
    train_indices: np.ndarray
    validation_indices: np.ndarray
    test_indices: np.ndarray
    seed_used: int
    grouping_column: str | None
    stratification_method: str

    def as_dict(self) -> dict[str, np.ndarray]:
        return {
            "train": self.train_indices,
            "validation": self.validation_indices,
            "test": self.test_indices,
        }


def _read_header(csv_path: Path) -> list[str]:
    frame = pd.read_csv(csv_path, nrows=0, compression="infer")
    return frame.columns.astype(str).str.strip().tolist()


def load_clinical_dataset(
    csv_path: Path,
    feature_columns: list[str],
    target_column: str,
    candidate_identifier_columns: list[str],
    outcomes_config: dict[str, Any] | None = None,
    dataset_name: str = "mimic",
) -> ClinicalDataset:
    csv_path = csv_path.expanduser().resolve()
    if not csv_path.is_file():
        raise FileNotFoundError(f"Dataset was not found: {csv_path}")

    available_columns = _read_header(csv_path)
    outcomes_config = dict(outcomes_config or {})
    mortality_column = str(
        outcomes_config.get("mortality_column", "hospital_expire_flag")
    )
    derivation_config = outcomes_config.get("los_derivation")
    derivation_policy = normalise_los_derivation_config(
        derivation_config,
        dataset_name=dataset_name,
    )
    derivation_columns = required_los_derivation_columns(
        derivation_config,
        dataset_name=dataset_name,
        mortality_column=mortality_column,
    )
    required_columns = feature_columns + [target_column]
    if derivation_policy["enabled"]:
        required_columns.extend(derivation_columns)
    required_columns = list(dict.fromkeys(required_columns))
    missing_columns = [
        column for column in required_columns if column not in available_columns
    ]
    if missing_columns:
        raise KeyError(
            "Required dataset columns are missing.\n"
            f"Missing: {missing_columns}\n"
            f"Available: {available_columns}"
        )

    identifier_columns = [
        column
        for column in candidate_identifier_columns
        if column in available_columns and column not in required_columns
    ]
    use_columns = list(dict.fromkeys(required_columns + identifier_columns))

    frame = pd.read_csv(
        csv_path,
        usecols=use_columns,
        low_memory=False,
        compression="infer",
    )
    frame.columns = frame.columns.astype(str).str.strip()
    frame.insert(
        0,
        "_source_row_index",
        np.arange(len(frame), dtype=np.int64),
    )

    header_matches = pd.Series(0, index=frame.index, dtype=np.int64)
    for column in required_columns:
        values = frame[column].astype("string").str.strip().str.lower()
        header_matches += (
            values.eq(column.lower()).fillna(False).astype(np.int64)
        )

    repeated_header_mask = header_matches >= 2
    removed_repeated_headers = int(repeated_header_mask.sum())
    if removed_repeated_headers:
        frame = frame.loc[~repeated_header_mask].copy()

    if "gender" in feature_columns:
        normalized_gender = (
            frame["gender"].astype("string").str.strip().str.upper()
        )
        gender_mapping = {
            "M": 0.0,
            "MALE": 0.0,
            "0": 0.0,
            "F": 1.0,
            "FEMALE": 1.0,
            "1": 1.0,
        }
        frame["gender"] = normalized_gender.map(gender_mapping)

    for column in feature_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if target_column in frame.columns:
        frame[target_column] = pd.to_numeric(
            frame[target_column], errors="coerce"
        )
    if mortality_column in frame.columns:
        frame[mortality_column] = pd.to_numeric(
            frame[mortality_column], errors="coerce"
        )
    frame, los_summary = derive_hospital_los(
        frame,
        dataset_name=dataset_name,
        mortality_column=mortality_column,
        los_column=target_column,
        config=derivation_config,
    )

    frame.loc[:, feature_columns] = frame[feature_columns].replace(
        [np.inf, -np.inf], np.nan
    )
    frame[target_column] = frame[target_column].replace(
        [np.inf, -np.inf], np.nan
    )

    missing_target_mask = frame[target_column].isna()
    removed_missing_targets = int(missing_target_mask.sum())
    if removed_missing_targets:
        frame = frame.loc[~missing_target_mask].copy()

    frame = frame.reset_index(drop=True)
    if frame.empty:
        raise ValueError("No rows remain after target validation.")

    all_missing_features = [
        column for column in feature_columns if frame[column].isna().all()
    ]
    if all_missing_features:
        raise ValueError(
            "The following features contain only missing values: "
            f"{all_missing_features}"
        )

    return ClinicalDataset(
        frame=frame,
        feature_columns=feature_columns,
        target_column=target_column,
        identifier_columns=identifier_columns,
        source_path=csv_path,
        removed_repeated_headers=removed_repeated_headers,
        removed_missing_targets=removed_missing_targets,
        los_derivation_summary=los_summary.to_dict(),
    )


def los_strata(los: np.ndarray) -> np.ndarray:
    los = np.asarray(los, dtype=np.float64)
    return np.select(
        [los < 7.0, los < 14.0, los < 28.0],
        [0, 1, 2],
        default=3,
    ).astype(np.int32)


def _can_stratify(labels: np.ndarray, test_size: float) -> bool:
    labels = np.asarray(labels)
    _, counts = np.unique(labels, return_counts=True)
    if len(counts) < 2 or counts.min() < 2:
        return False
    expected_test = np.floor(counts * float(test_size))
    expected_train = counts - expected_test
    return bool((expected_test >= 1).all() and (expected_train >= 1).all())


def _split_units(
    units: np.ndarray,
    labels: np.ndarray,
    test_size: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, str]:
    if _can_stratify(labels, test_size):
        splitter = StratifiedShuffleSplit(
            n_splits=1,
            test_size=test_size,
            random_state=seed,
        )
        train_pos, test_pos = next(splitter.split(units, labels))
        return units[train_pos], units[test_pos], "stratified"

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=test_size,
        random_state=seed,
    )
    dummy = np.zeros(len(units), dtype=np.int8)
    train_pos, test_pos = next(splitter.split(dummy, groups=units))
    return units[train_pos], units[test_pos], "group_shuffle_fallback"


def _all_thresholds_have_both_classes(
    los: np.ndarray,
    split_indices: dict[str, np.ndarray],
    thresholds: list[float],
) -> bool:
    for indices in split_indices.values():
        for threshold in thresholds:
            labels = (los[indices] >= threshold).astype(np.int32)
            if np.unique(labels).size < 2:
                return False
    return True


def create_common_splits(
    dataset: ClinicalDataset,
    thresholds: list[float],
    train_ratio: float,
    validation_ratio: float,
    test_ratio: float,
    seed: int,
    max_attempts: int,
    patient_id_column: str | None,
) -> DatasetSplits:
    row_indices = np.arange(len(dataset.frame), dtype=np.int64)
    strata = los_strata(dataset.los)

    use_groups = (
        patient_id_column is not None
        and patient_id_column in dataset.frame.columns
        and dataset.frame[patient_id_column].notna().any()
    )

    if use_groups:
        raw_groups = dataset.frame[patient_id_column].astype("string")
        missing_groups = raw_groups.isna()
        if missing_groups.any():
            raw_groups = raw_groups.copy()
            raw_groups.loc[missing_groups] = [
                f"__missing_group_row_{index}"
                for index in row_indices[missing_groups.to_numpy()]
            ]
        groups = raw_groups.astype(str).to_numpy()

        group_frame = pd.DataFrame({"group": groups, "stratum": strata})
        group_table = (
            group_frame.groupby("group", as_index=False)["stratum"].max()
        )
        units = group_table["group"].to_numpy()
        unit_labels = group_table["stratum"].to_numpy(dtype=np.int32)
        grouping_column = patient_id_column
    else:
        groups = row_indices.astype(str)
        units = row_indices
        unit_labels = strata
        grouping_column = None

    validation_fraction_in_train_validation = validation_ratio / (
        train_ratio + validation_ratio
    )

    last_method = "unknown"
    for attempt in range(max_attempts):
        current_seed = seed + attempt
        train_validation_units, test_units, method_1 = _split_units(
            units=units,
            labels=unit_labels,
            test_size=test_ratio,
            seed=current_seed,
        )

        if use_groups:
            unit_label_map = dict(zip(units.tolist(), unit_labels.tolist()))
            train_validation_labels = np.asarray(
                [unit_label_map[value] for value in train_validation_units],
                dtype=np.int32,
            )
        else:
            train_validation_labels = strata[train_validation_units]

        train_units, validation_units, method_2 = _split_units(
            units=train_validation_units,
            labels=train_validation_labels,
            test_size=validation_fraction_in_train_validation,
            seed=current_seed + 10000,
        )

        if use_groups:
            train_mask = np.isin(groups, train_units)
            validation_mask = np.isin(groups, validation_units)
            test_mask = np.isin(groups, test_units)
            train_indices = row_indices[train_mask]
            validation_indices = row_indices[validation_mask]
            test_indices = row_indices[test_mask]
        else:
            train_indices = np.asarray(train_units, dtype=np.int64)
            validation_indices = np.asarray(validation_units, dtype=np.int64)
            test_indices = np.asarray(test_units, dtype=np.int64)

        split_indices = {
            "train": np.sort(train_indices),
            "validation": np.sort(validation_indices),
            "test": np.sort(test_indices),
        }

        all_rows = np.concatenate(list(split_indices.values()))
        no_overlap = len(np.unique(all_rows)) == len(all_rows)
        complete = len(all_rows) == len(row_indices)
        class_valid = _all_thresholds_have_both_classes(
            dataset.los,
            split_indices,
            thresholds,
        )

        if no_overlap and complete and class_valid:
            return DatasetSplits(
                train_indices=split_indices["train"],
                validation_indices=split_indices["validation"],
                test_indices=split_indices["test"],
                seed_used=current_seed,
                grouping_column=grouping_column,
                stratification_method=f"{method_1}+{method_2}",
            )
        last_method = f"{method_1}+{method_2}"

    raise RuntimeError(
        "Unable to create a common split with both classes present for "
        f"all LOS thresholds after {max_attempts} attempts. "
        f"Last method: {last_method}"
    )



def _all_required_classes_present(
    strata: np.ndarray,
    split_indices: dict[str, np.ndarray],
    required_classes: Sequence[int],
) -> bool:
    required = set(int(value) for value in required_classes)
    if not required:
        raise ValueError("required_classes must not be empty.")
    for indices in split_indices.values():
        observed = set(
            np.asarray(strata, dtype=np.int64)[indices].astype(int).tolist()
        )
        if not required.issubset(observed):
            return False
    return True


def create_stratified_group_splits(
    frame: pd.DataFrame,
    strata: np.ndarray,
    train_ratio: float,
    validation_ratio: float,
    test_ratio: float,
    seed: int,
    max_attempts: int,
    patient_id_column: str | None,
    required_classes: Sequence[int] | None = None,
    stratification_target: str = "provided_strata",
) -> DatasetSplits:
    """Create reusable patient-level splits for externally supplied strata.

    This function is used by the integrated EXP1/EXP2/EXP3 pipeline. It does
    not derive labels internally, so every downstream task can share the same
    row and patient assignment while using task-specific labels later.
    """
    if frame.empty:
        raise ValueError("Cannot split an empty frame.")

    ratios = [float(train_ratio), float(validation_ratio), float(test_ratio)]
    if any(value <= 0 for value in ratios):
        raise ValueError("All split ratios must be greater than zero.")
    if abs(sum(ratios) - 1.0) > 1e-8:
        raise ValueError(f"Split ratios must sum to 1.0: {ratios}")
    if int(max_attempts) < 1:
        raise ValueError("max_attempts must be at least 1.")

    strata = np.asarray(strata, dtype=np.int64).reshape(-1)
    if len(strata) != len(frame):
        raise ValueError(
            "The supplied strata length does not match the frame: "
            f"{len(strata)} != {len(frame)}"
        )
    if not np.isfinite(strata.astype(np.float64)).all():
        raise ValueError("The supplied strata contain non-finite values.")

    observed_classes = sorted(np.unique(strata).astype(int).tolist())
    classes = (
        sorted(set(int(value) for value in required_classes))
        if required_classes is not None
        else observed_classes
    )
    missing_required = sorted(set(classes) - set(observed_classes))
    if missing_required:
        raise ValueError(
            "Required stratification classes are absent from the full cohort: "
            f"{missing_required}"
        )

    row_indices = np.arange(len(frame), dtype=np.int64)
    use_groups = (
        patient_id_column is not None
        and patient_id_column in frame.columns
        and frame[patient_id_column].notna().any()
    )

    if use_groups:
        raw_groups = frame[patient_id_column].astype("string")
        missing_groups = raw_groups.isna()
        if missing_groups.any():
            raw_groups = raw_groups.copy()
            raw_groups.loc[missing_groups] = [
                f"__missing_group_row_{index}"
                for index in row_indices[missing_groups.to_numpy()]
            ]
        groups = raw_groups.astype(str).to_numpy()

        group_frame = pd.DataFrame({"group": groups, "stratum": strata})
        # Keep the legacy deterministic rule for patients with multiple rows.
        # Split integrity is checked again at row level after assignment.
        group_table = (
            group_frame.groupby("group", as_index=False)["stratum"].max()
        )
        units = group_table["group"].to_numpy()
        unit_labels = group_table["stratum"].to_numpy(dtype=np.int64)
        grouping_column = patient_id_column
    else:
        groups = row_indices.astype(str)
        units = row_indices
        unit_labels = strata
        grouping_column = None

    validation_fraction_in_train_validation = validation_ratio / (
        train_ratio + validation_ratio
    )

    last_method = "unknown"
    for attempt in range(int(max_attempts)):
        current_seed = int(seed) + attempt
        train_validation_units, test_units, method_1 = _split_units(
            units=units,
            labels=unit_labels,
            test_size=test_ratio,
            seed=current_seed,
        )

        if use_groups:
            unit_label_map = dict(zip(units.tolist(), unit_labels.tolist()))
            train_validation_labels = np.asarray(
                [unit_label_map[value] for value in train_validation_units],
                dtype=np.int64,
            )
        else:
            train_validation_labels = strata[train_validation_units]

        train_units, validation_units, method_2 = _split_units(
            units=train_validation_units,
            labels=train_validation_labels,
            test_size=validation_fraction_in_train_validation,
            seed=current_seed + 10000,
        )

        if use_groups:
            train_mask = np.isin(groups, train_units)
            validation_mask = np.isin(groups, validation_units)
            test_mask = np.isin(groups, test_units)
            train_indices = row_indices[train_mask]
            validation_indices = row_indices[validation_mask]
            test_indices = row_indices[test_mask]
        else:
            train_indices = np.asarray(train_units, dtype=np.int64)
            validation_indices = np.asarray(validation_units, dtype=np.int64)
            test_indices = np.asarray(test_units, dtype=np.int64)

        split_indices = {
            "train": np.sort(train_indices),
            "validation": np.sort(validation_indices),
            "test": np.sort(test_indices),
        }
        all_rows = np.concatenate(list(split_indices.values()))
        no_overlap = len(np.unique(all_rows)) == len(all_rows)
        complete = len(all_rows) == len(row_indices)
        classes_valid = _all_required_classes_present(
            strata=strata,
            split_indices=split_indices,
            required_classes=classes,
        )

        if no_overlap and complete and classes_valid:
            return DatasetSplits(
                train_indices=split_indices["train"],
                validation_indices=split_indices["validation"],
                test_indices=split_indices["test"],
                seed_used=current_seed,
                grouping_column=grouping_column,
                stratification_method=(
                    f"{stratification_target}:{method_1}+{method_2}"
                ),
            )
        last_method = f"{method_1}+{method_2}"

    raise RuntimeError(
        "Unable to create a common split with all required classes present "
        f"after {max_attempts} attempts. Required classes: {classes}. "
        f"Last method: {last_method}"
    )

def save_splits(
    dataset: ClinicalDataset,
    splits: DatasetSplits,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / "split_indices.npz",
        train_indices=splits.train_indices,
        validation_indices=splits.validation_indices,
        test_indices=splits.test_indices,
        train_source_row_indices=dataset.source_row_index[splits.train_indices],
        validation_source_row_indices=dataset.source_row_index[
            splits.validation_indices
        ],
        test_source_row_indices=dataset.source_row_index[splits.test_indices],
    )

    assignment = pd.DataFrame(
        {
            "dataset_row_index": np.arange(len(dataset.frame), dtype=np.int64),
            "source_row_index": dataset.source_row_index,
            "split": "",
        }
    )
    assignment.loc[splits.train_indices, "split"] = "train"
    assignment.loc[splits.validation_indices, "split"] = "validation"
    assignment.loc[splits.test_indices, "split"] = "test"

    for column in dataset.identifier_columns:
        assignment[column] = dataset.frame[column].to_numpy()
    assignment[dataset.target_column] = dataset.los
    assignment["los_stratum"] = los_strata(dataset.los)
    assignment.to_csv(output_dir / "split_assignments.csv.gz", index=False)
