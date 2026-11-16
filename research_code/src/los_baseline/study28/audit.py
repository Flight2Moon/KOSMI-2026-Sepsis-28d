from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

from .contract import LABEL_TO_Y, FeatureContract, load_feature_contract
from .datasets import DatasetRegistry, DatasetSpec, load_dataset_registry


@dataclass(frozen=True)
class DatasetAudit:
    dataset: str
    source_file: str
    role: str
    source_sha256: str
    expected_sha256: str | None
    hash_matches: bool | None
    row_count: int
    unique_id_count: int | None
    duplicate_id_rows: int | None
    class_early_dc: int
    class_death: int
    class_prolonged: int
    invalid_y_count: int
    invalid_label_count: int
    label_y_mismatch_count: int
    missing_required_columns: tuple[str, ...]
    feature_count: int
    feature_order_matches: bool
    all_nan_features: tuple[str, ...]
    consistency_warning_count: int
    consistency_warning_note: str | None
    passed: bool
    blocking_reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AuditResult:
    passed: bool
    dataset_root: str
    feature_list_path: str
    feature_list_sha256: str
    feature_count: int
    required_external_sites: tuple[str, ...]
    datasets: tuple[DatasetAudit, ...]
    blocking_reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "dataset_root": self.dataset_root,
            "feature_list_path": self.feature_list_path,
            "feature_list_sha256": self.feature_list_sha256,
            "feature_count": self.feature_count,
            "required_external_sites": list(self.required_external_sites),
            "datasets": [item.to_dict() for item in self.datasets],
            "blocking_reasons": list(self.blocking_reasons),
        }


def assert_required_external_sites(
    completed_sites: Iterable[str],
    required_sites: Iterable[str],
) -> None:
    completed = {str(value) for value in completed_sites}
    required = {str(value) for value in required_sites}
    missing = sorted(required - completed)
    if missing:
        raise RuntimeError(
            "Required external validation is incomplete. Missing sites: " + ", ".join(missing)
        )


def _compound_duplicate_rows(frame: pd.DataFrame, id_columns: tuple[str, ...]) -> tuple[int | None, int | None]:
    if not id_columns or any(column not in frame.columns for column in id_columns):
        return None, None
    ids = frame.loc[:, list(id_columns)]
    valid = ~ids.isna().all(axis=1)
    if not bool(valid.any()):
        return 0, 0
    duplicate_mask = ids.loc[valid].duplicated(keep=False)
    unique_count = int(ids.loc[valid].drop_duplicates().shape[0])
    return unique_count, int(duplicate_mask.sum())


def _label_y_checks(frame: pd.DataFrame) -> tuple[int, int, int, dict[str, int]]:
    numeric_y = pd.to_numeric(frame["y"], errors="coerce")
    invalid_y = int((numeric_y.isna() | ~numeric_y.isin([0, 1, 2])).sum())
    normalized_label = frame["label28"].astype("string").str.strip()
    invalid_label = int((~normalized_label.isin(LABEL_TO_Y)).sum())
    expected_y = normalized_label.map(LABEL_TO_Y)
    mismatch = int((expected_y.notna() & numeric_y.notna() & (expected_y != numeric_y)).sum())
    counts = {
        label: int((normalized_label == label).sum())
        for label in ("early_dc", "death", "prolonged")
    }
    return invalid_y, invalid_label, mismatch, counts


def _derived_outcome_consistency_warning(frame: pd.DataFrame) -> tuple[int, str | None]:
    """Run a non-blocking plausibility check only.

    The final CSV label is authoritative because the original label-generation
    scripts were not supplied. This check must never rewrite or drop rows.
    """
    required = {"hosp_los_d", "hospital_expire_flag", "label28"}
    if not required.issubset(frame.columns):
        return 0, None
    los = pd.to_numeric(frame["hosp_los_d"], errors="coerce")
    death = pd.to_numeric(frame["hospital_expire_flag"], errors="coerce")
    provided = frame["label28"].astype("string").str.strip()

    inferred = pd.Series(pd.NA, index=frame.index, dtype="string")
    valid = los.notna() & death.isin([0, 1])
    inferred.loc[valid & (los >= 28.0)] = "prolonged"
    inferred.loc[valid & (los < 28.0) & (death == 1)] = "death"
    inferred.loc[valid & (los < 28.0) & (death == 0)] = "early_dc"
    mismatch = inferred.notna() & provided.isin(LABEL_TO_Y) & (inferred != provided)
    count = int(mismatch.sum())
    if not count:
        return 0, None
    note = (
        "Non-blocking warning only: provided label28 differs from a simplified "
        "LOS/death-flag reconstruction. The professor-provided label remains authoritative "
        "because original dataset-generation scripts are unavailable."
    )
    return count, note


def _audit_one_dataset(
    *,
    spec: DatasetSpec,
    dataset_root: Path,
    feature_contract: FeatureContract,
) -> DatasetAudit:
    path = spec.resolve_path(dataset_root)
    source_sha256 = spec.source_sha256(dataset_root)
    hash_matches = (
        None if spec.expected_sha256 is None else source_sha256 == spec.expected_sha256
    )

    # Load the full source once, then apply the cohort filter. The source hash is
    # always computed before filtering so the frozen input file is what is verified.
    source = pd.read_csv(path, low_memory=False)
    frame = source
    if spec.filter is not None:
        if spec.filter.column not in frame.columns:
            raise KeyError(
                f"Dataset '{spec.name}' filter column is missing: {spec.filter.column}"
            )
        frame = frame.loc[frame[spec.filter.column] == spec.filter.equals].copy()
    frame = frame.reset_index(drop=True)

    required_columns = list(feature_contract.columns) + ["label28", "y"]
    missing = tuple(column for column in required_columns if column not in frame.columns)
    blocking: list[str] = []
    if missing:
        blocking.append(f"missing_required_columns:{','.join(missing)}")

    if hash_matches is False:
        blocking.append("source_sha256_mismatch")

    if missing:
        invalid_y = invalid_label = mismatch = 0
        counts = {"early_dc": 0, "death": 0, "prolonged": 0}
        feature_order_matches = False
        all_nan: tuple[str, ...] = ()
        consistency_count, consistency_note = 0, None
    else:
        invalid_y, invalid_label, mismatch, counts = _label_y_checks(frame)
        if invalid_y:
            blocking.append(f"invalid_y:{invalid_y}")
        if invalid_label:
            blocking.append(f"invalid_label28:{invalid_label}")
        if mismatch:
            blocking.append(f"label_y_mismatch:{mismatch}")

        # The dataset may have metadata before the 57 features, but once the first
        # model feature begins the feature subsequence itself must match exactly.
        observed_feature_order = tuple(column for column in frame.columns if column in feature_contract.columns)
        feature_order_matches = observed_feature_order == feature_contract.columns
        if not feature_order_matches:
            blocking.append("feature_order_mismatch")
        all_nan = tuple(column for column in feature_contract.columns if frame[column].isna().all())
        consistency_count, consistency_note = _derived_outcome_consistency_warning(frame)

    unique_id_count, duplicate_id_rows = _compound_duplicate_rows(frame, spec.id_columns)
    if duplicate_id_rows:
        blocking.append(f"duplicate_id_rows:{duplicate_id_rows}")

    return DatasetAudit(
        dataset=spec.name,
        source_file=str(path),
        role=spec.role,
        source_sha256=source_sha256,
        expected_sha256=spec.expected_sha256,
        hash_matches=hash_matches,
        row_count=int(len(frame)),
        unique_id_count=unique_id_count,
        duplicate_id_rows=duplicate_id_rows,
        class_early_dc=counts["early_dc"],
        class_death=counts["death"],
        class_prolonged=counts["prolonged"],
        invalid_y_count=invalid_y,
        invalid_label_count=invalid_label,
        label_y_mismatch_count=mismatch,
        missing_required_columns=missing,
        feature_count=len(feature_contract.columns),
        feature_order_matches=feature_order_matches,
        all_nan_features=all_nan,
        consistency_warning_count=consistency_count,
        consistency_warning_note=consistency_note,
        passed=not blocking,
        blocking_reasons=tuple(blocking),
    )


def audit_study_datasets(
    *,
    dataset_root: Path,
    registry_path: Path,
    feature_list_path: Path | None = None,
    output_dir: Path | None = None,
) -> AuditResult:
    dataset_root = Path(dataset_root).expanduser().resolve()
    registry: DatasetRegistry = load_dataset_registry(registry_path)
    feature_list_path = (
        Path(feature_list_path).expanduser().resolve()
        if feature_list_path is not None
        else (dataset_root / "feature_list.csv").resolve()
    )
    feature_contract = load_feature_contract(feature_list_path)

    audits = tuple(
        _audit_one_dataset(
            spec=registry.datasets[name],
            dataset_root=dataset_root,
            feature_contract=feature_contract,
        )
        for name in registry.datasets
    )

    blocking: list[str] = []
    failed = [item.dataset for item in audits if not item.passed]
    if failed:
        blocking.append("dataset_audit_failed:" + ",".join(failed))

    available_external = [
        item.dataset
        for item in audits
        if item.dataset in registry.required_external_sites and item.passed
    ]
    try:
        assert_required_external_sites(available_external, registry.required_external_sites)
    except RuntimeError as exc:
        blocking.append(str(exc))

    result = AuditResult(
        passed=not blocking,
        dataset_root=str(dataset_root),
        feature_list_path=str(feature_list_path),
        feature_list_sha256=feature_contract.sha256,
        feature_count=len(feature_contract.columns),
        required_external_sites=registry.required_external_sites,
        datasets=audits,
        blocking_reasons=tuple(blocking),
    )

    if output_dir is not None:
        output_dir = Path(output_dir).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        with (output_dir / "study28_dataset_audit.json").open("w", encoding="utf-8") as handle:
            json.dump(result.to_dict(), handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        summary = pd.DataFrame([item.to_dict() for item in audits])
        for column in ["missing_required_columns", "all_nan_features", "blocking_reasons"]:
            if column in summary.columns:
                summary[column] = summary[column].apply(lambda value: "|".join(value) if isinstance(value, tuple) else value)
        summary.to_csv(output_dir / "study28_dataset_audit.csv", index=False)

    return result
