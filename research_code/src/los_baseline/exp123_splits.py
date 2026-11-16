from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from .config import resolve_outcome_horizon_days
from .constants import OUTCOME_CLASS_LABELS
from .data import DatasetSplits, create_stratified_group_splits
from .utils import atomic_write_json, environment_info, sha256_file

SPLIT_NAMES: tuple[str, ...] = ("train", "validation", "internal_test")
REQUIRED_OUTCOME_CLASSES: tuple[int, ...] = (0, 1, 2)


@dataclass(frozen=True)
class SplitIntegrityResult:
    total_rows: int
    assigned_rows: int
    unique_source_rows: int
    complete_row_coverage: bool
    no_row_overlap: bool
    grouping_column: str | None
    total_groups: int | None
    patient_overlap_counts: dict[str, int]
    no_patient_overlap: bool
    all_classes_present_per_split: bool
    class_presence: dict[str, list[int]]
    stage1_both_classes_per_split: bool
    stage2_both_classes_per_split: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CommonSplitResult:
    run_id: str
    split_hash: str
    output_dir: str
    seed_requested: int
    seed_used: int
    grouping_column: str | None
    stratification_method: str
    source_outcome_assignments_sha256: str
    source_outcome_manifest_sha256: str
    split_counts: dict[str, int]
    split_rates: dict[str, float]
    integrity: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required JSON file was not found: {path}")
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a JSON object in {path}")
    return payload


def load_ready_mimic_outcomes(run_dir: Path) -> tuple[pd.DataFrame, Path, Path]:
    """Load the frozen Step 1 MIMIC outcome assignments for split creation."""
    outcomes_dir = run_dir / "common" / "outcomes"
    manifest_path = outcomes_dir / "outcome_step_manifest.json"
    assignments_path = outcomes_dir / "mimic_outcome_assignments.csv.gz"

    manifest = _load_json(manifest_path)
    mimic_audit = manifest.get("dataset_audits", {}).get("mimic")
    if not isinstance(mimic_audit, dict) or mimic_audit.get("status") != "ready":
        raise RuntimeError(
            "The MIMIC outcome source is not ready in the Step 1 manifest."
        )
    if not assignments_path.is_file():
        raise FileNotFoundError(
            f"MIMIC outcome assignments were not found: {assignments_path}"
        )

    frame = pd.read_csv(assignments_path, compression="infer", low_memory=False)
    required_columns = {
        "_source_row_index",
        "three_class_outcome",
        "three_class_label",
        "stage1_mortality_label",
        "stage2_eligible",
        "stage2_long_stay_label",
    }
    missing = sorted(required_columns - set(frame.columns))
    if missing:
        raise KeyError(
            "MIMIC outcome assignments are missing required columns: "
            f"{missing}"
        )
    if frame.empty:
        raise ValueError("MIMIC outcome assignments are empty.")
    if frame["_source_row_index"].isna().any():
        raise ValueError("_source_row_index contains missing values.")
    if frame["_source_row_index"].duplicated().any():
        duplicates = int(frame["_source_row_index"].duplicated().sum())
        raise ValueError(
            f"_source_row_index must be unique; found {duplicates} duplicates."
        )

    labels = pd.to_numeric(
        frame["three_class_outcome"], errors="coerce"
    )
    if labels.isna().any():
        raise ValueError("three_class_outcome contains missing/non-numeric values.")
    labels = labels.astype(np.int64)
    invalid = sorted(set(labels.unique().tolist()) - set(REQUIRED_OUTCOME_CLASSES))
    if invalid:
        raise ValueError(f"Invalid three-class labels were found: {invalid}")
    missing_classes = sorted(
        set(REQUIRED_OUTCOME_CLASSES) - set(labels.unique().tolist())
    )
    if missing_classes:
        raise ValueError(
            "The full MIMIC cohort does not contain all required outcome "
            f"classes: {missing_classes}"
        )
    frame["three_class_outcome"] = labels
    frame["_source_row_index"] = pd.to_numeric(
        frame["_source_row_index"], errors="raise"
    ).astype(np.int64)
    return frame, assignments_path, manifest_path


def _pairwise_patient_overlap(
    assignment: pd.DataFrame,
    patient_id_column: str | None,
) -> tuple[dict[str, int], bool, int | None]:
    pair_names = [
        ("train", "validation"),
        ("train", "internal_test"),
        ("validation", "internal_test"),
    ]
    if patient_id_column is None or patient_id_column not in assignment.columns:
        return ({f"{left}_vs_{right}": 0 for left, right in pair_names}, True, None)

    normalized = assignment[patient_id_column].astype("string")
    missing = normalized.isna()
    if missing.any():
        normalized = normalized.copy()
        normalized.loc[missing] = [
            f"__missing_patient_row_{value}"
            for value in assignment.loc[missing, "dataset_row_index"]
        ]
    temporary = assignment.assign(_patient_group=normalized.astype(str))
    groups = {
        split_name: set(
            temporary.loc[
                temporary["split"] == split_name,
                "_patient_group",
            ].tolist()
        )
        for split_name in SPLIT_NAMES
    }
    overlaps = {
        f"{left}_vs_{right}": len(groups[left] & groups[right])
        for left, right in pair_names
    }
    return overlaps, all(value == 0 for value in overlaps.values()), int(
        temporary["_patient_group"].nunique()
    )


def validate_common_split_assignment(
    assignment: pd.DataFrame,
    patient_id_column: str | None,
) -> SplitIntegrityResult:
    required_columns = {
        "dataset_row_index",
        "source_row_index",
        "split",
        "three_class_outcome",
        "stage1_mortality_label",
        "stage2_eligible",
        "stage2_long_stay_label",
    }
    missing = sorted(required_columns - set(assignment.columns))
    if missing:
        raise KeyError(f"Split assignment is missing columns: {missing}")

    split_counts = assignment["split"].value_counts().to_dict()
    unknown_splits = sorted(set(split_counts) - set(SPLIT_NAMES))
    if unknown_splits:
        raise ValueError(f"Unknown split labels were found: {unknown_splits}")
    if set(split_counts) != set(SPLIT_NAMES):
        raise ValueError(
            "All train/validation/internal_test splits must be present."
        )

    assigned_rows = int(len(assignment))
    unique_source_rows = int(assignment["source_row_index"].nunique())
    complete_row_coverage = bool(
        assigned_rows == unique_source_rows
        and assignment["dataset_row_index"].nunique() == assigned_rows
    )
    no_row_overlap = complete_row_coverage

    class_presence: dict[str, list[int]] = {}
    stage1_valid = True
    stage2_valid = True
    for split_name in SPLIT_NAMES:
        split_frame = assignment.loc[assignment["split"] == split_name]
        classes = sorted(
            split_frame["three_class_outcome"].astype(int).unique().tolist()
        )
        class_presence[split_name] = classes
        if not set(REQUIRED_OUTCOME_CLASSES).issubset(classes):
            stage1_valid = False
            stage2_valid = False
            continue

        stage1_classes = set(
            split_frame["stage1_mortality_label"].astype(int).unique().tolist()
        )
        stage1_valid = stage1_valid and stage1_classes == {0, 1}

        stage2_frame = split_frame.loc[split_frame["stage2_eligible"].astype(bool)]
        stage2_values = pd.to_numeric(
            stage2_frame["stage2_long_stay_label"], errors="coerce"
        ).dropna().astype(int)
        stage2_valid = stage2_valid and set(stage2_values.unique().tolist()) == {0, 1}

    overlaps, no_patient_overlap, total_groups = _pairwise_patient_overlap(
        assignment=assignment,
        patient_id_column=patient_id_column,
    )
    all_classes = all(
        set(REQUIRED_OUTCOME_CLASSES).issubset(values)
        for values in class_presence.values()
    )
    result = SplitIntegrityResult(
        total_rows=assigned_rows,
        assigned_rows=assigned_rows,
        unique_source_rows=unique_source_rows,
        complete_row_coverage=complete_row_coverage,
        no_row_overlap=no_row_overlap,
        grouping_column=(
            patient_id_column
            if patient_id_column and patient_id_column in assignment.columns
            else None
        ),
        total_groups=total_groups,
        patient_overlap_counts=overlaps,
        no_patient_overlap=no_patient_overlap,
        all_classes_present_per_split=all_classes,
        class_presence=class_presence,
        stage1_both_classes_per_split=stage1_valid,
        stage2_both_classes_per_split=stage2_valid,
    )
    failures = [
        name
        for name, value in {
            "complete_row_coverage": result.complete_row_coverage,
            "no_row_overlap": result.no_row_overlap,
            "no_patient_overlap": result.no_patient_overlap,
            "all_classes_present_per_split": result.all_classes_present_per_split,
            "stage1_both_classes_per_split": result.stage1_both_classes_per_split,
            "stage2_both_classes_per_split": result.stage2_both_classes_per_split,
        }.items()
        if not value
    ]
    if failures:
        raise ValueError(
            "Common split integrity validation failed: " + ", ".join(failures)
        )
    return result


def build_split_assignment(
    outcomes: pd.DataFrame,
    splits: DatasetSplits,
    identifier_columns: Sequence[str],
) -> pd.DataFrame:
    assignment = pd.DataFrame(
        {
            "dataset_row_index": np.arange(len(outcomes), dtype=np.int64),
            "source_row_index": outcomes["_source_row_index"].to_numpy(
                dtype=np.int64
            ),
            "split": "",
        }
    )
    assignment.loc[splits.train_indices, "split"] = "train"
    assignment.loc[splits.validation_indices, "split"] = "validation"
    assignment.loc[splits.test_indices, "split"] = "internal_test"

    for column in identifier_columns:
        if column in outcomes.columns:
            assignment[column] = outcomes[column].to_numpy()
    for column in [
        "three_class_outcome",
        "three_class_label",
        "stage1_mortality_label",
        "stage2_eligible",
        "stage2_long_stay_label",
    ]:
        assignment[column] = outcomes[column].to_numpy()
    return assignment


def compute_split_hash(assignment: pd.DataFrame) -> str:
    """Hash the semantic row-to-split assignment without gzip metadata."""
    ordered = assignment.loc[:, ["source_row_index", "split"]].copy()
    ordered = ordered.sort_values("source_row_index", kind="mergesort")
    digest = hashlib.sha256()
    for source_index, split_name in ordered.itertuples(index=False, name=None):
        digest.update(f"{int(source_index)}\t{split_name}\n".encode("utf-8"))
    return digest.hexdigest()


def create_class_distribution(assignment: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for split_name in ["overall", *SPLIT_NAMES]:
        current = (
            assignment
            if split_name == "overall"
            else assignment.loc[assignment["split"] == split_name]
        )
        denominator = max(len(current), 1)
        for class_code, class_label in sorted(OUTCOME_CLASS_LABELS.items()):
            count = int((current["three_class_outcome"] == class_code).sum())
            rows.append(
                {
                    "split": split_name,
                    "class_code": int(class_code),
                    "class_label": class_label,
                    "count": count,
                    "rate": float(count / denominator),
                    "split_sample_count": int(len(current)),
                }
            )
    return pd.DataFrame(rows)


def create_task_population_distribution(assignment: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for split_name in SPLIT_NAMES:
        current = assignment.loc[assignment["split"] == split_name]

        for class_code, class_label in sorted(OUTCOME_CLASS_LABELS.items()):
            count = int((current["three_class_outcome"] == class_code).sum())
            rows.append(
                {
                    "task": "exp1_flat_3class",
                    "split": split_name,
                    "label": int(class_code),
                    "label_name": class_label,
                    "count": count,
                    "task_sample_count": int(len(current)),
                    "rate": float(count / max(len(current), 1)),
                }
            )

        for label, label_name in [(0, "A_or_C_non_death"), (1, "B_died")]:
            count = int((current["stage1_mortality_label"] == label).sum())
            rows.append(
                {
                    "task": "stage1_mortality",
                    "split": split_name,
                    "label": label,
                    "label_name": label_name,
                    "count": count,
                    "task_sample_count": int(len(current)),
                    "rate": float(count / max(len(current), 1)),
                }
            )

        stage2 = current.loc[current["stage2_eligible"].astype(bool)].copy()
        stage2_labels = pd.to_numeric(
            stage2["stage2_long_stay_label"], errors="coerce"
        ).astype(int)
        for label, label_name in [
            (0, "A_discharged_alive"),
            (1, "C_still_admitted"),
        ]:
            count = int((stage2_labels == label).sum())
            rows.append(
                {
                    "task": "stage2_long_stay_shared_with_exp3",
                    "split": split_name,
                    "label": label,
                    "label_name": label_name,
                    "count": count,
                    "task_sample_count": int(len(stage2)),
                    "rate": float(count / max(len(stage2), 1)),
                }
            )
    return pd.DataFrame(rows)


def save_common_split_outputs(
    output_dir: Path,
    assignment: pd.DataFrame,
    splits: DatasetSplits,
    integrity: SplitIntegrityResult,
    result_payload: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / "split_indices.npz",
        train_indices=splits.train_indices,
        validation_indices=splits.validation_indices,
        internal_test_indices=splits.test_indices,
        train_source_row_indices=assignment.loc[
            assignment["split"] == "train", "source_row_index"
        ].to_numpy(dtype=np.int64),
        validation_source_row_indices=assignment.loc[
            assignment["split"] == "validation", "source_row_index"
        ].to_numpy(dtype=np.int64),
        internal_test_source_row_indices=assignment.loc[
            assignment["split"] == "internal_test", "source_row_index"
        ].to_numpy(dtype=np.int64),
    )
    assignment.to_csv(
        output_dir / "split_assignments.csv.gz",
        index=False,
        compression="gzip",
    )
    create_class_distribution(assignment).to_csv(
        output_dir / "class_distribution.csv",
        index=False,
    )
    create_task_population_distribution(assignment).to_csv(
        output_dir / "task_population_distribution.csv",
        index=False,
    )
    atomic_write_json(output_dir / "split_integrity.json", integrity.to_dict())
    atomic_write_json(output_dir / "split_manifest.json", result_payload)
    (output_dir / "split_hash.txt").write_text(
        str(result_payload["split_hash"]) + "\n",
        encoding="utf-8",
    )


def run_common_split_step(
    config: dict[str, Any],
    run_id: str,
    overwrite: bool = False,
) -> CommonSplitResult:
    runs_dir_value = Path(config["paths"]["runs_dir"]).expanduser()
    if not runs_dir_value.is_absolute():
        runs_dir_value = Path(config["_project_root"]) / runs_dir_value
    run_dir = runs_dir_value.resolve() / run_id
    output_dir = run_dir / "common" / "splits"

    downstream_paths = [
        run_dir / "EXP1_flat_3class",
        run_dir / "EXP2_two_stage",
        run_dir / "EXP3_survivor_binary",
        run_dir / "optimization",
        run_dir / "frozen_design",
    ]
    if overwrite and any(path.exists() for path in downstream_paths):
        existing = [str(path) for path in downstream_paths if path.exists()]
        raise RuntimeError(
            "Common splits cannot be overwritten after downstream outputs "
            f"exist: {existing}"
        )
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(
            f"Common split output already exists: {output_dir}. "
            "Use --overwrite only before downstream experiments begin."
        )
    if overwrite and output_dir.exists():
        for path in sorted(output_dir.rglob("*"), reverse=True):
            if path.is_file() or path.is_symlink():
                path.unlink()
            elif path.is_dir():
                path.rmdir()
    output_dir.mkdir(parents=True, exist_ok=True)

    outcomes, assignments_path, outcome_manifest_path = (
        load_ready_mimic_outcomes(run_dir)
    )
    split_config = config["split"]
    patient_id_column = config["data"].get("patient_id_column")
    patient_id_column = str(patient_id_column) if patient_id_column else None
    labels = outcomes["three_class_outcome"].to_numpy(dtype=np.int64)

    splits = create_stratified_group_splits(
        frame=outcomes,
        strata=labels,
        train_ratio=float(split_config["train_ratio"]),
        validation_ratio=float(split_config["validation_ratio"]),
        test_ratio=float(split_config["test_ratio"]),
        seed=int(split_config["seed"]),
        max_attempts=int(split_config.get("max_attempts", 100)),
        patient_id_column=patient_id_column,
        required_classes=REQUIRED_OUTCOME_CLASSES,
        stratification_target="three_class_outcome",
    )
    identifier_columns = list(config["data"].get("identifier_columns", []))
    assignment = build_split_assignment(
        outcomes=outcomes,
        splits=splits,
        identifier_columns=identifier_columns,
    )
    integrity = validate_common_split_assignment(
        assignment=assignment,
        patient_id_column=patient_id_column,
    )
    split_hash = compute_split_hash(assignment)
    split_counts = {
        split_name: int((assignment["split"] == split_name).sum())
        for split_name in SPLIT_NAMES
    }
    total = max(len(assignment), 1)
    split_rates = {
        split_name: float(count / total)
        for split_name, count in split_counts.items()
    }

    result = CommonSplitResult(
        run_id=run_id,
        split_hash=split_hash,
        output_dir=str(output_dir),
        seed_requested=int(split_config["seed"]),
        seed_used=int(splits.seed_used),
        grouping_column=splits.grouping_column,
        stratification_method=splits.stratification_method,
        source_outcome_assignments_sha256=sha256_file(assignments_path),
        source_outcome_manifest_sha256=sha256_file(outcome_manifest_path),
        split_counts=split_counts,
        split_rates=split_rates,
        integrity=integrity.to_dict(),
    )
    horizon_days = resolve_outcome_horizon_days(config, required=False)
    manifest = {
        **result.to_dict(),
        "step": "common_three_class_splits",
        "status": "ready",
        "prediction_landmark_hours": float(config.get("cohort_contract", {}).get("prediction_landmark_hours", 24.0)),
        "required_outcome_classes": list(REQUIRED_OUTCOME_CLASSES),
        "class_labels": OUTCOME_CLASS_LABELS,
        "requested_ratios": {
            "train": float(split_config["train_ratio"]),
            "validation": float(split_config["validation_ratio"]),
            "internal_test": float(split_config["test_ratio"]),
        },
        "task_split_references": {
            "exp1_flat_3class": split_hash,
            "stage1_mortality": split_hash,
            "stage2_long_stay": split_hash,
            "exp3_survivor_binary": split_hash,
        },
        "external_eicu_policy": {
            "split": False,
            "role": "external_validation_only",
            "outcome_assignments": str(
                run_dir
                / "common"
                / "outcomes"
                / "eicu_outcome_assignments.csv.gz"
            ),
            "outcome_ready": bool(
                (run_dir / "common" / "outcomes" / "eicu_outcome_assignments.csv.gz").is_file()
            ),
        },
        "environment": environment_info(),
        "next_required_step": "common_training_engine",
        "model_training_enabled": False,
    }
    if horizon_days is not None:
        manifest["outcome_horizon_days"] = float(horizon_days)
    save_common_split_outputs(
        output_dir=output_dir,
        assignment=assignment,
        splits=splits,
        integrity=integrity,
        result_payload=manifest,
    )
    return result
