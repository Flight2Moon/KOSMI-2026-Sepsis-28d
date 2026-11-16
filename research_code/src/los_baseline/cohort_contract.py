from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


class CohortContractError(ValueError):
    """Raised when a dataset violates the frozen V7 cohort contract."""


@dataclass(frozen=True)
class CohortContractAudit:
    dataset_name: str
    enabled: bool
    input_rows: int
    output_rows: int
    removed_duplicate_stay_rows: int
    removed_later_patient_stays: int
    removed_before_landmark_rows: int
    unique_subjects: int | None
    unique_admissions: int | None
    unique_stays: int | None
    duplicate_stays_before: int | None
    duplicate_stays_after: int | None
    duplicate_subjects_after: int | None
    landmark_hours: float
    landmark_column: str
    analysis_unit: str
    strict_unique_stay: bool
    strict_unique_subject: bool
    selection_order: tuple[str, ...]
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def normalise_cohort_contract(config: Mapping[str, Any] | None) -> dict[str, Any]:
    source = dict(config or {})
    selection_order = tuple(
        str(value)
        for value in source.get(
            "patient_selection_order", ["icu_intime", "hadm_id", "stay_id"]
        )
    )
    return {
        "enabled": bool(source.get("enabled", False)),
        "analysis_unit": str(
            source.get("analysis_unit", "first_icu_stay_per_patient")
        ),
        "subject_id_column": str(source.get("subject_id_column", "subject_id")),
        "admission_id_column": str(source.get("admission_id_column", "hadm_id")),
        "stay_id_column": str(source.get("stay_id_column", "stay_id")),
        "icu_intime_column": str(source.get("icu_intime_column", "icu_intime")),
        "discharge_time_column": str(
            source.get("discharge_time_column", "dischtime")
        ),
        "mortality_column": str(
            source.get("mortality_column", "hospital_expire_flag")
        ),
        "patient_selection_order": selection_order,
        "strict_unique_stay": bool(source.get("strict_unique_stay", True)),
        "strict_unique_subject": bool(source.get("strict_unique_subject", True)),
        "deduplicate_exact_stays": bool(
            source.get("deduplicate_exact_stays", True)
        ),
        "prediction_landmark_hours": float(
            source.get("prediction_landmark_hours", 24.0)
        ),
        "require_alive_and_admitted_at_landmark": bool(
            source.get("require_alive_and_admitted_at_landmark", True)
        ),
        "allow_missing_identifiers": bool(
            source.get("allow_missing_identifiers", False)
        ),
    }


def _required_columns(policy: Mapping[str, Any]) -> set[str]:
    required = {
        str(policy["subject_id_column"]),
        str(policy["stay_id_column"]),
        str(policy["icu_intime_column"]),
        str(policy["discharge_time_column"]),
    }
    if str(policy["analysis_unit"]) == "first_icu_stay_per_patient":
        required.add(str(policy["subject_id_column"]))
    return required


def _safe_nunique(frame: pd.DataFrame, column: str) -> int | None:
    if column not in frame.columns:
        return None
    return int(frame[column].nunique(dropna=True))


def _duplicate_count(frame: pd.DataFrame, column: str) -> int | None:
    if column not in frame.columns:
        return None
    return int(frame[column].duplicated(keep=False).sum())


def _normalise_sort_columns(
    frame: pd.DataFrame, selection_order: Sequence[str]
) -> list[str]:
    columns = [column for column in selection_order if column in frame.columns]
    if not columns:
        raise CohortContractError(
            "None of the configured patient selection columns exist in the dataset."
        )
    return columns


def apply_cohort_contract(
    frame: pd.DataFrame,
    *,
    dataset_name: str,
    config: Mapping[str, Any] | None,
) -> tuple[pd.DataFrame, CohortContractAudit, pd.DataFrame]:
    """Apply the frozen first-ICU-stay and 24-hour landmark contract.

    The source row index is preserved. Excluded rows are returned with a reason so
    every cohort decision remains auditable.
    """
    policy = normalise_cohort_contract(config)
    source = frame.copy()
    input_rows = int(len(source))
    if not policy["enabled"]:
        audit = CohortContractAudit(
            dataset_name=str(dataset_name),
            enabled=False,
            input_rows=input_rows,
            output_rows=input_rows,
            removed_duplicate_stay_rows=0,
            removed_later_patient_stays=0,
            removed_before_landmark_rows=0,
            unique_subjects=_safe_nunique(source, policy["subject_id_column"]),
            unique_admissions=_safe_nunique(source, policy["admission_id_column"]),
            unique_stays=_safe_nunique(source, policy["stay_id_column"]),
            duplicate_stays_before=_duplicate_count(
                source, policy["stay_id_column"]
            ),
            duplicate_stays_after=_duplicate_count(source, policy["stay_id_column"]),
            duplicate_subjects_after=_duplicate_count(
                source, policy["subject_id_column"]
            ),
            landmark_hours=float(policy["prediction_landmark_hours"]),
            landmark_column=str(policy["icu_intime_column"]),
            analysis_unit=str(policy["analysis_unit"]),
            strict_unique_stay=bool(policy["strict_unique_stay"]),
            strict_unique_subject=bool(policy["strict_unique_subject"]),
            selection_order=tuple(policy["patient_selection_order"]),
        )
        return source, audit, pd.DataFrame()

    missing = sorted(_required_columns(policy) - set(source.columns))
    if missing:
        raise CohortContractError(
            f"Cohort contract columns are missing from {dataset_name}: {missing}"
        )

    excluded_parts: list[pd.DataFrame] = []
    warnings: list[str] = []
    subject_column = str(policy["subject_id_column"])
    admission_column = str(policy["admission_id_column"])
    stay_column = str(policy["stay_id_column"])
    intime_column = str(policy["icu_intime_column"])
    discharge_column = str(policy["discharge_time_column"])

    for identifier in (subject_column, stay_column):
        missing_count = int(source[identifier].isna().sum())
        if missing_count and not policy["allow_missing_identifiers"]:
            raise CohortContractError(
                f"{identifier} contains {missing_count} missing values."
            )

    duplicate_stays_before = _duplicate_count(source, stay_column)
    removed_duplicate_stay_rows = 0
    if source[stay_column].duplicated(keep=False).any():
        duplicate_rows = source.loc[
            source[stay_column].duplicated(keep=False)
        ].copy()
        value_columns = [
            column
            for column in source.columns
            if column not in {"_source_row_index"}
        ]
        conflicts = (
            duplicate_rows.groupby(stay_column, dropna=False)[value_columns]
            .nunique(dropna=False)
            .max(axis=1)
            .gt(1)
        )
        conflict_ids = conflicts[conflicts].index.tolist()
        if conflict_ids:
            preview = conflict_ids[:10]
            raise CohortContractError(
                "Duplicate stay_id groups contain conflicting values; fix the "
                f"upstream join before modelling. Example stay IDs: {preview}"
            )
        if not policy["deduplicate_exact_stays"]:
            raise CohortContractError(
                "Exact duplicate stay rows were found and automatic de-duplication "
                "is disabled."
            )
        keep_mask = ~source[stay_column].duplicated(keep="first")
        removed = source.loc[~keep_mask].copy()
        removed["cohort_exclusion_reason"] = "exact_duplicate_stay"
        excluded_parts.append(removed)
        removed_duplicate_stay_rows = int(len(removed))
        source = source.loc[keep_mask].copy()

    removed_later_patient_stays = 0
    if str(policy["analysis_unit"]) == "first_icu_stay_per_patient":
        sort_columns = _normalise_sort_columns(
            source, policy["patient_selection_order"]
        )
        temporary_sort_columns: list[str] = []
        for column in sort_columns:
            if column.endswith("time") or "time" in column:
                temp = f"__sort_{column}"
                source[temp] = pd.to_datetime(source[column], errors="coerce", format="mixed")
                temporary_sort_columns.append(temp)
            else:
                temporary_sort_columns.append(column)
        source = source.sort_values(
            [subject_column, *temporary_sort_columns],
            kind="mergesort",
            na_position="last",
        )
        keep_mask = ~source[subject_column].duplicated(keep="first")
        removed = source.loc[~keep_mask].copy()
        if not removed.empty:
            removed["cohort_exclusion_reason"] = "later_icu_stay_for_patient"
            excluded_parts.append(removed)
        removed_later_patient_stays = int(len(removed))
        source = source.loc[keep_mask].copy()
        source = source.drop(
            columns=[c for c in temporary_sort_columns if c.startswith("__sort_")],
            errors="ignore",
        )

    removed_before_landmark_rows = 0
    landmark_hours = float(policy["prediction_landmark_hours"])
    if policy["require_alive_and_admitted_at_landmark"]:
        icu_intime = pd.to_datetime(source[intime_column], errors="coerce", format="mixed")
        discharge_time = pd.to_datetime(source[discharge_column], errors="coerce", format="mixed")
        invalid_time = icu_intime.isna() | discharge_time.isna()
        if invalid_time.any():
            raise CohortContractError(
                "ICU admission/discharge timestamps contain missing or invalid values "
                "required for the landmark contract."
            )
        landmark_time = icu_intime + pd.to_timedelta(landmark_hours, unit="h")
        eligible = discharge_time.gt(landmark_time)
        removed = source.loc[~eligible].copy()
        if not removed.empty:
            removed["cohort_exclusion_reason"] = "outcome_before_prediction_landmark"
            removed["prediction_landmark_time"] = landmark_time.loc[~eligible]
            excluded_parts.append(removed)
        removed_before_landmark_rows = int((~eligible).sum())
        source = source.loc[eligible].copy()
        source["prediction_landmark_time"] = landmark_time.loc[eligible]

    source = source.sort_index(kind="mergesort").reset_index(drop=True)
    duplicate_stays_after = _duplicate_count(source, stay_column)
    duplicate_subjects_after = _duplicate_count(source, subject_column)
    if policy["strict_unique_stay"] and duplicate_stays_after:
        raise CohortContractError(
            f"stay_id is not unique after cohort filtering: {duplicate_stays_after} rows."
        )
    if policy["strict_unique_subject"] and duplicate_subjects_after:
        raise CohortContractError(
            "subject_id is not unique after first-stay selection: "
            f"{duplicate_subjects_after} rows."
        )

    if len(source) == 0:
        raise CohortContractError("No rows remain after applying the cohort contract.")

    excluded = (
        pd.concat(excluded_parts, ignore_index=True, sort=False)
        if excluded_parts
        else pd.DataFrame()
    )
    audit = CohortContractAudit(
        dataset_name=str(dataset_name),
        enabled=True,
        input_rows=input_rows,
        output_rows=int(len(source)),
        removed_duplicate_stay_rows=removed_duplicate_stay_rows,
        removed_later_patient_stays=removed_later_patient_stays,
        removed_before_landmark_rows=removed_before_landmark_rows,
        unique_subjects=_safe_nunique(source, subject_column),
        unique_admissions=_safe_nunique(source, admission_column),
        unique_stays=_safe_nunique(source, stay_column),
        duplicate_stays_before=duplicate_stays_before,
        duplicate_stays_after=duplicate_stays_after,
        duplicate_subjects_after=duplicate_subjects_after,
        landmark_hours=landmark_hours,
        landmark_column=intime_column,
        analysis_unit=str(policy["analysis_unit"]),
        strict_unique_stay=bool(policy["strict_unique_stay"]),
        strict_unique_subject=bool(policy["strict_unique_subject"]),
        selection_order=tuple(policy["patient_selection_order"]),
        warnings=tuple(warnings),
    )
    return source, audit, excluded


def validate_mortality_endpoint(
    frame: pd.DataFrame,
    *,
    mortality_column: str,
    config: Mapping[str, Any] | None,
) -> list[str]:
    if config is None:
        return []
    policy = dict(config)
    if not bool(policy.get("enabled", False)):
        return []
    if mortality_column not in frame.columns:
        return [f"Mortality column not found: {mortality_column}"]
    raw = frame[mortality_column]
    values = pd.to_numeric(raw, errors="coerce")
    missing_count = int(values.isna().sum())
    invalid_mask = values.notna() & ~values.isin([0, 1])
    invalid_count = int(invalid_mask.sum())
    valid = values.loc[values.isin([0, 1])]
    death_count = int(valid.eq(1).sum())
    alive_count = int(valid.eq(0).sum())
    total = max(int(len(valid)), 1)
    death_rate = float(death_count / total)
    issues: list[str] = []
    if invalid_count:
        issues.append(f"Mortality endpoint contains {invalid_count} values outside 0/1.")
    if missing_count and bool(policy.get("fail_on_missing", True)):
        issues.append(f"Mortality endpoint contains {missing_count} missing/unparseable values.")
    if valid.nunique() < 2:
        issues.append("Mortality endpoint does not contain both 0 and 1 classes.")
    minimum_count = int(policy.get("minimum_death_count", 10))
    if death_count < minimum_count:
        issues.append(
            f"Death count is below the minimum: {death_count} < {minimum_count}."
        )
    minimum_rate = float(policy.get("minimum_death_rate", 0.001))
    maximum_rate = float(policy.get("maximum_death_rate", 0.5))
    if death_rate < minimum_rate:
        issues.append(
            f"Death rate is below the minimum: {death_rate:.8f} < {minimum_rate:.8f}."
        )
    if death_rate > maximum_rate:
        issues.append(
            f"Death rate is above the maximum: {death_rate:.8f} > {maximum_rate:.8f}."
        )
    if alive_count == 0:
        issues.append("No alive cases were found in the mortality endpoint.")
    return issues
