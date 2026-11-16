from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np
import pandas as pd

from .constants import OUTCOME_CLASS_LABELS
from .cohort_contract import apply_cohort_contract, validate_mortality_endpoint
from .los_derivation import (
    derive_hospital_los,
    normalise_los_derivation_config,
    required_los_derivation_columns,
)
from .utils import atomic_write_json, sha256_file


@dataclass(frozen=True)
class OutcomeDefinition:
    landmark_days: float
    mortality_column: str
    class_counts: dict[str, int]
    class_counts_per_split: dict[str, dict[str, int]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class OutcomeSourceAudit:
    dataset_name: str
    source_path: str
    source_sha256: str | None
    status: str
    available_columns: tuple[str, ...]
    mortality_column: str
    los_column: str
    landmark_days: float
    input_row_count: int
    retained_row_count: int
    removed_repeated_headers: int
    mortality_missing_count: int | None
    los_missing_count: int | None
    negative_los_count: int | None
    invalid_mortality_values: tuple[str, ...]
    class_counts: dict[str, int]
    class_rates: dict[str, float]
    boundary_counts: dict[str, int]
    blocking_issues: tuple[str, ...]
    loaded_row_count: int = 0
    excluded_row_count: int = 0
    exclusion_rate: float = 0.0
    exclusion_reason_counts: dict[str, int] | None = None
    excluded_mortality_counts: dict[str, int] | None = None
    exclusion_policy: dict[str, Any] | None = None
    quality_warnings: tuple[str, ...] = ()
    los_derivation_enabled: bool = False
    los_derivation_summary: dict[str, Any] | None = None
    cohort_contract_summary: dict[str, Any] | None = None
    cohort_excluded_row_count: int = 0
    cohort_eligible_row_count: int = 0
    cohort_exclusion_rate: float = 0.0
    cohort_exclusion_reason_counts: dict[str, int] | None = None
    cohort_excluded_mortality_counts: dict[str, int] | None = None
    cohort_exclusion_reason_mortality_counts: dict[str, dict[str, int]] | None = None
    quality_excluded_row_count: int = 0
    quality_exclusion_rate: float = 0.0
    quality_exclusion_reason_counts: dict[str, int] | None = None
    quality_excluded_mortality_counts: dict[str, int] | None = None
    quality_exclusion_reason_mortality_counts: dict[str, dict[str, int]] | None = None
    label_derivation_status: str = "not_attempted"

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["exclusion_reason_counts"] = self.exclusion_reason_counts or {}
        payload["excluded_mortality_counts"] = self.excluded_mortality_counts or {}
        payload["exclusion_policy"] = self.exclusion_policy or {}
        payload["los_derivation_summary"] = self.los_derivation_summary or {}
        payload["cohort_contract_summary"] = self.cohort_contract_summary or {}
        payload["cohort_exclusion_reason_counts"] = (
            self.cohort_exclusion_reason_counts or {}
        )
        payload["cohort_excluded_mortality_counts"] = (
            self.cohort_excluded_mortality_counts or {}
        )
        payload["cohort_exclusion_reason_mortality_counts"] = (
            self.cohort_exclusion_reason_mortality_counts or {}
        )
        payload["quality_exclusion_reason_counts"] = (
            self.quality_exclusion_reason_counts or {}
        )
        payload["quality_excluded_mortality_counts"] = (
            self.quality_excluded_mortality_counts or {}
        )
        payload["quality_exclusion_reason_mortality_counts"] = (
            self.quality_exclusion_reason_mortality_counts or {}
        )
        return payload


@dataclass(frozen=True)
class OutcomeSourceAuditResult:
    """Audit result that remains backward-compatible with two-value unpacking."""

    audit: OutcomeSourceAudit
    assignments: pd.DataFrame | None
    excluded_rows: pd.DataFrame

    def __iter__(self) -> Iterator[Any]:
        yield self.audit
        yield self.assignments


def _normalise_exclusion_policy(
    policy: dict[str, Any] | None,
    dataset_name: str,
) -> dict[str, Any]:
    source = dict(policy or {})
    enabled = bool(source.get("enabled", False))
    datasets = [str(value) for value in source.get("datasets", ["mimic", "eicu"])]
    applies = enabled and str(dataset_name) in set(datasets)
    rules = dict(source.get("rules", {}))
    normalised = {
        "enabled": enabled,
        "applies_to_dataset": applies,
        "datasets": datasets,
        "rules": {
            "negative_los": bool(rules.get("negative_los", False)),
            "missing_los": bool(rules.get("missing_los", False)),
            "missing_mortality": bool(rules.get("missing_mortality", False)),
            "invalid_mortality": bool(rules.get("invalid_mortality", False)),
        },
        "max_excluded_fraction": float(source.get("max_excluded_fraction", 0.01)),
        "max_excluded_rows": int(source.get("max_excluded_rows", 100)),
        "fail_if_limit_exceeded": bool(source.get("fail_if_limit_exceeded", True)),
        "preserve_source_row_index": True,
        "silent_exclusion_allowed": False,
    }
    return normalised


def _apply_outcome_exclusion_policy(
    frame: pd.DataFrame,
    mortality_column: str,
    los_column: str,
    policy: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    """Quarantine explicitly allowed invalid rows without changing source indices."""
    reasons: dict[int, list[str]] = {int(index): [] for index in frame.index}
    if policy["applies_to_dataset"]:
        mortality = pd.to_numeric(frame[mortality_column], errors="coerce")
        los = pd.to_numeric(frame[los_column], errors="coerce")
        rules = policy["rules"]
        masks = {
            "negative_los": los.lt(0).fillna(False),
            "missing_los": los.isna(),
            "missing_mortality": mortality.isna(),
            "invalid_mortality": mortality.notna() & ~mortality.isin([0.0, 1.0]),
        }
        for reason, mask in masks.items():
            if not rules.get(reason, False):
                continue
            for index in frame.index[mask]:
                reasons[int(index)].append(reason)

    exclusion_mask = pd.Series(
        [bool(reasons[int(index)]) for index in frame.index],
        index=frame.index,
        dtype=bool,
    )
    excluded = frame.loc[exclusion_mask].copy()
    if not excluded.empty:
        excluded["exclusion_reason"] = [
            "|".join(reasons[int(index)]) for index in excluded.index
        ]
        excluded["excluded_from_modelling"] = True
    filtered = frame.loc[~exclusion_mask].copy()
    reason_counts: dict[str, int] = {}
    for values in reasons.values():
        for reason in values:
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
    return filtered, excluded, reason_counts


def _normalise_excluded_rows(
    frame: pd.DataFrame,
    *,
    category: str,
    reason_column: str,
) -> pd.DataFrame:
    """Return an auditable exclusion frame with a common schema."""
    if frame is None or frame.empty:
        return pd.DataFrame()
    result = frame.copy()
    if reason_column in result.columns:
        result["exclusion_reason"] = result[reason_column].astype("string")
    elif "exclusion_reason" not in result.columns:
        result["exclusion_reason"] = "unspecified"
    result["exclusion_category"] = str(category)
    result["excluded_from_modelling"] = True
    return result


def _reason_counts(frame: pd.DataFrame) -> dict[str, int]:
    if frame is None or frame.empty or "exclusion_reason" not in frame.columns:
        return {}
    counts: dict[str, int] = {}
    for value in frame["exclusion_reason"].astype("string").fillna("unspecified"):
        for reason in str(value).split("|"):
            clean = reason.strip() or "unspecified"
            counts[clean] = counts.get(clean, 0) + 1
    return counts


def _normalise_mortality_series(series: pd.Series) -> pd.Series:
    """Normalize common numeric and textual in-hospital mortality values."""
    numeric = pd.to_numeric(series, errors="coerce")
    text = series.astype("string").str.strip().str.casefold()
    positive = text.isin(
        {"expired", "dead", "death", "deceased", "yes", "true"}
    )
    negative = text.isin(
        {"alive", "survived", "survivor", "no", "false"}
    )
    return numeric.mask(positive, 1.0).mask(negative, 0.0)


def _mortality_counts(
    frame: pd.DataFrame,
    mortality_column: str,
) -> dict[str, int]:
    if frame is None or frame.empty or mortality_column not in frame.columns:
        return {}
    values = _normalise_mortality_series(frame[mortality_column])
    return {
        str(key): int(value)
        for key, value in values.value_counts(dropna=False).to_dict().items()
    }


def _reason_mortality_counts(
    frame: pd.DataFrame,
    mortality_column: str,
) -> dict[str, dict[str, int]]:
    if (
        frame is None
        or frame.empty
        or "exclusion_reason" not in frame.columns
        or mortality_column not in frame.columns
    ):
        return {}
    result: dict[str, dict[str, int]] = {}
    mortality = _normalise_mortality_series(frame[mortality_column])
    for index, value in frame["exclusion_reason"].astype("string").items():
        raw_mortality = mortality.loc[index]
        mortality_key = "<NA>" if pd.isna(raw_mortality) else str(float(raw_mortality))
        for reason in str(value).split("|"):
            clean = reason.strip() or "unspecified"
            bucket = result.setdefault(clean, {})
            bucket[mortality_key] = bucket.get(mortality_key, 0) + 1
    return result


def validate_outcome_source_columns(
    frame: pd.DataFrame,
    mortality_column: str,
    los_column: str,
) -> None:
    """Validate source columns required for the complete three-class outcome."""
    missing_columns = [
        column
        for column in (mortality_column, los_column)
        if column not in frame.columns
    ]
    if missing_columns:
        raise KeyError(
            "Outcome source columns are missing: "
            f"{missing_columns}. Available columns: {list(frame.columns)}"
        )

    mortality = pd.to_numeric(frame[mortality_column], errors="coerce")
    los = pd.to_numeric(frame[los_column], errors="coerce")

    mortality_missing = int(mortality.isna().sum())
    los_missing = int(los.isna().sum())
    if mortality_missing:
        raise ValueError(
            f"{mortality_column} contains {mortality_missing} missing or non-numeric values."
        )
    if los_missing:
        raise ValueError(
            f"{los_column} contains {los_missing} missing or non-numeric values."
        )

    invalid_mortality = sorted(
        set(mortality.astype(float).unique().tolist()) - {0.0, 1.0}
    )
    if invalid_mortality:
        raise ValueError(
            f"{mortality_column} must contain only 0/1 values. "
            f"Invalid values: {invalid_mortality}"
        )

    negative_los_count = int((los < 0).sum())
    if negative_los_count:
        raise ValueError(
            f"{los_column} contains {negative_los_count} negative values."
        )


def derive_three_class_outcome(
    frame: pd.DataFrame,
    los_column: str,
    mortality_column: str,
    landmark_days: float = 28.0,
) -> np.ndarray:
    """Return A=0, B=1, C=2 using the frozen day-landmark definition."""
    landmark_days = float(landmark_days)
    if landmark_days <= 0:
        raise ValueError("landmark_days must be greater than zero.")

    validate_outcome_source_columns(
        frame=frame,
        mortality_column=mortality_column,
        los_column=los_column,
    )

    mortality = pd.to_numeric(
        frame[mortality_column], errors="raise"
    ).to_numpy(dtype=np.int64)
    los = pd.to_numeric(frame[los_column], errors="raise").to_numpy(
        dtype=np.float64
    )

    outcome = np.full(len(frame), 2, dtype=np.int64)

    discharged_alive_before_landmark = (
        (mortality == 0) & (los < landmark_days)
    )
    died_by_landmark = (mortality == 1) & (los <= landmark_days)

    outcome[discharged_alive_before_landmark] = 0
    outcome[died_by_landmark] = 1

    if not np.isin(outcome, [0, 1, 2]).all():
        raise AssertionError("Unexpected class code generated.")
    return outcome


def derive_stage_labels(
    three_class: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Derive Stage1 B vs A/C and Stage2 C vs A labels."""
    labels = np.asarray(three_class, dtype=np.int64).reshape(-1)
    if not np.isin(labels, [0, 1, 2]).all():
        invalid = sorted(set(labels.tolist()) - {0, 1, 2})
        raise ValueError(f"three_class contains invalid labels: {invalid}")

    y1 = (labels == 1).astype(np.int64)
    survivor_mask = labels != 1
    y2 = (labels[survivor_mask] == 2).astype(np.int64)
    return y1, y2


def outcome_class_counts(labels: np.ndarray) -> dict[str, int]:
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    return {
        OUTCOME_CLASS_LABELS[class_code]: int((labels == class_code).sum())
        for class_code in sorted(OUTCOME_CLASS_LABELS)
    }


def create_outcome_definition(
    labels: np.ndarray,
    landmark_days: float,
    mortality_column: str,
    split_labels: dict[str, np.ndarray] | None = None,
) -> OutcomeDefinition:
    split_counts = {
        split_name: outcome_class_counts(split_values)
        for split_name, split_values in (split_labels or {}).items()
    }
    return OutcomeDefinition(
        landmark_days=float(landmark_days),
        mortality_column=str(mortality_column),
        class_counts=outcome_class_counts(labels),
        class_counts_per_split=split_counts,
    )


def _read_header(csv_path: Path) -> list[str]:
    header = pd.read_csv(csv_path, nrows=0, compression="infer")
    return header.columns.astype(str).str.strip().tolist()


def load_outcome_source_frame(
    csv_path: Path,
    mortality_column: str,
    los_column: str,
    candidate_identifier_columns: Sequence[str] = (),
    los_derivation_config: dict[str, Any] | None = None,
    dataset_name: str = "mimic",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load only outcome and available identifier columns with stable row indices."""
    csv_path = csv_path.expanduser().resolve()
    if not csv_path.is_file():
        raise FileNotFoundError(f"Dataset was not found: {csv_path}")

    available_columns = _read_header(csv_path)
    derivation_policy = normalise_los_derivation_config(
        los_derivation_config,
        dataset_name=dataset_name,
    )
    required_columns = [mortality_column]
    if derivation_policy["enabled"]:
        required_columns.extend(
            required_los_derivation_columns(
                los_derivation_config,
                dataset_name=dataset_name,
                mortality_column=mortality_column,
            )
        )
    else:
        required_columns.append(los_column)
    required_columns = list(dict.fromkeys(required_columns))
    missing_columns = [
        column for column in required_columns if column not in available_columns
    ]
    if missing_columns:
        raise KeyError(
            f"Required outcome columns are missing: {missing_columns}. "
            f"Available columns: {available_columns}"
        )

    optional_columns = [los_column] if los_column in available_columns else []
    identifier_columns = [
        column
        for column in candidate_identifier_columns
        if column in available_columns and column not in required_columns
    ]
    use_columns = list(
        dict.fromkeys(required_columns + optional_columns + identifier_columns)
    )
    frame = pd.read_csv(
        csv_path,
        usecols=use_columns,
        low_memory=False,
        compression="infer",
    )
    frame.columns = frame.columns.astype(str).str.strip()
    input_row_count = int(len(frame))
    frame.insert(
        0,
        "_source_row_index",
        np.arange(input_row_count, dtype=np.int64),
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

    frame[mortality_column] = _normalise_mortality_series(
        frame[mortality_column]
    )
    if los_column in frame.columns:
        frame[los_column] = pd.to_numeric(frame[los_column], errors="coerce")
    frame, los_derivation_summary = derive_hospital_los(
        frame,
        dataset_name=dataset_name,
        mortality_column=mortality_column,
        los_column=los_column,
        config=los_derivation_config,
    )
    frame = frame.reset_index(drop=True)

    metadata = {
        "available_columns": available_columns,
        "identifier_columns": identifier_columns,
        "input_row_count": input_row_count,
        "retained_row_count": int(len(frame)),
        "removed_repeated_headers": removed_repeated_headers,
        "los_derivation_summary": los_derivation_summary.to_dict(),
    }
    return frame, metadata


def audit_outcome_source(
    dataset_name: str,
    csv_path: Path,
    mortality_column: str,
    los_column: str,
    landmark_days: float,
    candidate_identifier_columns: Sequence[str] = (),
    exclusion_policy: dict[str, Any] | None = None,
    los_derivation_config: dict[str, Any] | None = None,
    cohort_contract_config: dict[str, Any] | None = None,
    mortality_guard_config: dict[str, Any] | None = None,
) -> OutcomeSourceAuditResult:
    """Audit an outcome source and quarantine explicitly permitted invalid rows."""
    path = csv_path.expanduser().resolve()
    blocking_issues: list[str] = []
    quality_warnings: list[str] = []
    available_columns: list[str] = []
    source_sha256: str | None = None
    input_row_count = 0
    loaded_row_count = 0
    retained_row_count = 0
    removed_repeated_headers = 0
    mortality_missing_count: int | None = None
    los_missing_count: int | None = None
    negative_los_count: int | None = None
    invalid_values: list[str] = []
    counts = {label: 0 for label in OUTCOME_CLASS_LABELS.values()}
    rates = {label: float("nan") for label in OUTCOME_CLASS_LABELS.values()}
    boundary_counts = {
        "los_equal_landmark": 0,
        "death_equal_landmark": 0,
        "alive_equal_landmark": 0,
    }
    assignments: pd.DataFrame | None = None
    excluded_rows = pd.DataFrame()
    exclusion_reason_counts: dict[str, int] = {}
    excluded_mortality_counts: dict[str, int] = {}
    cohort_excluded_row_count = 0
    cohort_eligible_row_count = 0
    cohort_exclusion_rate = 0.0
    cohort_exclusion_reason_counts: dict[str, int] = {}
    cohort_excluded_mortality_counts: dict[str, int] = {}
    cohort_exclusion_reason_mortality_counts: dict[str, dict[str, int]] = {}
    quality_excluded_row_count = 0
    quality_exclusion_rate = 0.0
    quality_exclusion_reason_counts: dict[str, int] = {}
    quality_excluded_mortality_counts: dict[str, int] = {}
    quality_exclusion_reason_mortality_counts: dict[str, dict[str, int]] = {}
    label_derivation_status = "not_attempted"
    policy = _normalise_exclusion_policy(exclusion_policy, dataset_name)
    los_derivation_policy = normalise_los_derivation_config(
        los_derivation_config,
        dataset_name=dataset_name,
    )
    los_derivation_summary: dict[str, Any] = {}
    cohort_contract_summary: dict[str, Any] = {}

    if not path.is_file():
        blocking_issues.append(f"Dataset file not found: {path}")
    else:
        source_sha256 = sha256_file(path)
        try:
            available_columns = _read_header(path)
        except Exception as exception:
            blocking_issues.append(f"Failed to read CSV header: {exception}")

    if path.is_file() and available_columns:
        required_columns = [mortality_column]
        if los_derivation_policy["enabled"]:
            required_columns.extend(
                required_los_derivation_columns(
                    los_derivation_config,
                    dataset_name=dataset_name,
                    mortality_column=mortality_column,
                )
            )
        else:
            required_columns.append(los_column)
        missing_columns = [
            column
            for column in dict.fromkeys(required_columns)
            if column not in available_columns
        ]
        if missing_columns:
            blocking_issues.append(
                f"Missing required outcome columns: {missing_columns}"
            )
        else:
            try:
                frame, metadata = load_outcome_source_frame(
                    csv_path=path,
                    mortality_column=mortality_column,
                    los_column=los_column,
                    candidate_identifier_columns=candidate_identifier_columns,
                    los_derivation_config=los_derivation_config,
                    dataset_name=dataset_name,
                )
                los_derivation_summary = dict(
                    metadata.get("los_derivation_summary", {})
                )
                input_row_count = int(metadata["input_row_count"])
                loaded_row_count = int(metadata["retained_row_count"])
                removed_repeated_headers = int(
                    metadata["removed_repeated_headers"]
                )
                frame, cohort_audit, cohort_excluded = apply_cohort_contract(
                    frame,
                    dataset_name=dataset_name,
                    config=cohort_contract_config,
                )
                cohort_excluded_rows = cohort_excluded.copy()
                cohort_contract_summary = cohort_audit.to_dict()
                if los_derivation_summary.get("enabled"):
                    discrepancy_rows = int(
                        los_derivation_summary.get("discrepancy_rows", 0)
                    )
                    fallback_rows = int(
                        los_derivation_summary.get("fallback_rows", 0)
                    )
                    quality_warnings.append(
                        "Outcome LOS was derived from the configured ICU-origin and endpoint "
                        "timestamps; in-hospital deaths use the configured death "
                        "endpoint and other admissions use discharge time."
                    )
                    if discrepancy_rows:
                        quality_warnings.append(
                            f"Derived LOS differed from the source LOS beyond the "
                            f"configured tolerance for {discrepancy_rows} rows."
                        )
                    if fallback_rows:
                        quality_warnings.append(
                            f"Existing LOS fallback was used for {fallback_rows} rows."
                        )

                frame, policy_excluded_rows, policy_reason_counts = (
                    _apply_outcome_exclusion_policy(
                        frame=frame,
                        mortality_column=mortality_column,
                        los_column=los_column,
                        policy=policy,
                    )
                )

                cohort_excluded_rows = _normalise_excluded_rows(
                    cohort_excluded_rows,
                    category="cohort_eligibility",
                    reason_column="cohort_exclusion_reason",
                )
                policy_excluded_rows = _normalise_excluded_rows(
                    policy_excluded_rows,
                    category="data_quality",
                    reason_column="exclusion_reason",
                )
                excluded_parts = [
                    value
                    for value in (cohort_excluded_rows, policy_excluded_rows)
                    if value is not None and not value.empty
                ]
                excluded_rows = (
                    pd.concat(excluded_parts, ignore_index=True, sort=False)
                    if excluded_parts
                    else pd.DataFrame()
                )

                cohort_excluded_row_count = int(len(cohort_excluded_rows))
                cohort_eligible_row_count = max(
                    int(loaded_row_count) - cohort_excluded_row_count,
                    0,
                )
                quality_excluded_row_count = int(len(policy_excluded_rows))
                cohort_exclusion_rate = float(
                    cohort_excluded_row_count / max(loaded_row_count, 1)
                )
                quality_exclusion_rate = float(
                    quality_excluded_row_count
                    / max(cohort_eligible_row_count, 1)
                )
                cohort_exclusion_reason_counts = _reason_counts(
                    cohort_excluded_rows
                )
                quality_exclusion_reason_counts = dict(policy_reason_counts)
                cohort_excluded_mortality_counts = _mortality_counts(
                    cohort_excluded_rows, mortality_column
                )
                quality_excluded_mortality_counts = _mortality_counts(
                    policy_excluded_rows, mortality_column
                )
                cohort_exclusion_reason_mortality_counts = (
                    _reason_mortality_counts(
                        cohort_excluded_rows, mortality_column
                    )
                )
                quality_exclusion_reason_mortality_counts = (
                    _reason_mortality_counts(
                        policy_excluded_rows, mortality_column
                    )
                )
                exclusion_reason_counts = dict(
                    cohort_exclusion_reason_counts
                )
                for reason, value in quality_exclusion_reason_counts.items():
                    exclusion_reason_counts[reason] = (
                        exclusion_reason_counts.get(reason, 0) + int(value)
                    )
                excluded_mortality_counts = _mortality_counts(
                    excluded_rows, mortality_column
                )
                retained_row_count = int(len(frame))
                excluded_row_count = int(len(excluded_rows))
                exclusion_rate = float(
                    excluded_row_count / max(loaded_row_count, 1)
                )

                if cohort_excluded_row_count:
                    quality_warnings.append(
                        "Cohort eligibility rules excluded "
                        f"{cohort_excluded_row_count} rows "
                        f"({cohort_exclusion_rate:.6%}); these planned exclusions "
                        "are reported but are not evaluated against data-quality "
                        "exclusion limits."
                    )

                if quality_excluded_row_count:
                    quality_warnings.append(
                        "Data-quality rules excluded "
                        f"{quality_excluded_row_count} rows "
                        f"({quality_exclusion_rate:.6%})."
                    )
                    if policy["fail_if_limit_exceeded"]:
                        if quality_excluded_row_count > int(
                            policy["max_excluded_rows"]
                        ):
                            blocking_issues.append(
                                "Data-quality exclusions exceeded "
                                "max_excluded_rows: "
                                f"{quality_excluded_row_count} > "
                                f"{policy['max_excluded_rows']}."
                            )
                        if quality_exclusion_rate > float(
                            policy["max_excluded_fraction"]
                        ):
                            blocking_issues.append(
                                "Data-quality exclusions exceeded "
                                "max_excluded_fraction: "
                                f"{quality_exclusion_rate:.8f} > "
                                f"{policy['max_excluded_fraction']:.8f}."
                            )

                mortality = frame[mortality_column]
                los = frame[los_column]
                mortality_missing_count = int(mortality.isna().sum())
                los_missing_count = int(los.isna().sum())
                negative_los_count = int((los < 0).fillna(False).sum())
                valid_mortality_values = set(
                    mortality.dropna().astype(float).unique().tolist()
                )
                invalid_values = [
                    str(value)
                    for value in sorted(valid_mortality_values - {0.0, 1.0})
                ]

                if mortality_missing_count:
                    blocking_issues.append(
                        f"{mortality_column} has {mortality_missing_count} "
                        "missing/non-numeric values after exclusions."
                    )
                if los_missing_count:
                    blocking_issues.append(
                        f"{los_column} has {los_missing_count} "
                        "missing/non-numeric values after exclusions."
                    )
                if negative_los_count:
                    blocking_issues.append(
                        f"{los_column} has {negative_los_count} negative values "
                        "after exclusions."
                    )
                if invalid_values:
                    blocking_issues.append(
                        f"{mortality_column} has invalid values after exclusions: "
                        f"{invalid_values}"
                    )
                if frame.empty:
                    blocking_issues.append(
                        "No rows remained after outcome quality exclusions."
                    )

                can_derive_labels = bool(
                    not frame.empty
                    and mortality_missing_count == 0
                    and los_missing_count == 0
                    and negative_los_count == 0
                    and not invalid_values
                )
                if can_derive_labels:
                    labels = derive_three_class_outcome(
                        frame=frame,
                        los_column=los_column,
                        mortality_column=mortality_column,
                        landmark_days=landmark_days,
                    )
                    label_derivation_status = "ready"
                    counts = outcome_class_counts(labels)
                    total = max(len(labels), 1)
                    rates = {
                        label: float(count / total)
                        for label, count in counts.items()
                    }
                    boundary_mask = np.isclose(
                        los.to_numpy(dtype=np.float64),
                        float(landmark_days),
                        rtol=0.0,
                        atol=1e-12,
                    )
                    mortality_values = mortality.to_numpy(dtype=np.int64)
                    boundary_counts = {
                        "los_equal_landmark": int(boundary_mask.sum()),
                        "death_equal_landmark": int(
                            (boundary_mask & (mortality_values == 1)).sum()
                        ),
                        "alive_equal_landmark": int(
                            (boundary_mask & (mortality_values == 0)).sum()
                        ),
                    }
                    assignment_columns = [
                        "_source_row_index",
                        *metadata["identifier_columns"],
                        mortality_column,
                        los_column,
                    ]
                    los_audit_columns = [
                        column
                        for column in (
                            str(los_derivation_policy["admission_time_column"]),
                            str(los_derivation_policy["discharge_time_column"]),
                            str(
                                los_derivation_policy[
                                    "in_hospital_death_time_column"
                                ]
                            ),
                            f"{los_column}_source_value",
                            f"{los_column}_derived_value",
                            f"{los_column}_absolute_discrepancy_days",
                            "_los_endpoint_time",
                            "_los_endpoint_source",
                            "_los_used_existing_fallback",
                            "prediction_landmark_time",
                        )
                        if column in frame.columns
                    ]
                    assignments = frame.loc[
                        :,
                        list(dict.fromkeys(assignment_columns + los_audit_columns)),
                    ].copy()
                    assignments["three_class_outcome"] = labels
                    assignments["three_class_label"] = [
                        OUTCOME_CLASS_LABELS[int(value)] for value in labels
                    ]
                    y1, _ = derive_stage_labels(labels)
                    assignments["stage1_mortality_label"] = y1
                    assignments["stage2_eligible"] = labels != 1
                    assignments["stage2_long_stay_label"] = np.where(
                        labels == 1,
                        np.nan,
                        (labels == 2).astype(np.float64),
                    )
                else:
                    label_derivation_status = "blocked"

                endpoint_issues = validate_mortality_endpoint(
                    frame,
                    mortality_column=mortality_column,
                    config=mortality_guard_config,
                )
                if endpoint_issues:
                    blocking_issues.extend(endpoint_issues)
            except Exception as exception:
                blocking_issues.append(
                    "Outcome source processing failed: "
                    f"{type(exception).__name__}: {exception}"
                )

    excluded_row_count = int(len(excluded_rows))
    exclusion_rate = float(excluded_row_count / max(loaded_row_count, 1))
    status = "ready" if not blocking_issues else "blocked"
    audit = OutcomeSourceAudit(
        dataset_name=str(dataset_name),
        source_path=str(path),
        source_sha256=source_sha256,
        status=status,
        available_columns=tuple(available_columns),
        mortality_column=str(mortality_column),
        los_column=str(los_column),
        landmark_days=float(landmark_days),
        input_row_count=input_row_count,
        retained_row_count=retained_row_count,
        removed_repeated_headers=removed_repeated_headers,
        mortality_missing_count=mortality_missing_count,
        los_missing_count=los_missing_count,
        negative_los_count=negative_los_count,
        invalid_mortality_values=tuple(invalid_values),
        class_counts=counts,
        class_rates=rates,
        boundary_counts=boundary_counts,
        blocking_issues=tuple(blocking_issues),
        loaded_row_count=loaded_row_count,
        excluded_row_count=excluded_row_count,
        exclusion_rate=exclusion_rate,
        exclusion_reason_counts=exclusion_reason_counts,
        excluded_mortality_counts=excluded_mortality_counts,
        exclusion_policy=policy,
        quality_warnings=tuple(quality_warnings),
        los_derivation_enabled=bool(los_derivation_policy["enabled"]),
        los_derivation_summary=los_derivation_summary,
        cohort_contract_summary=cohort_contract_summary,
        cohort_excluded_row_count=cohort_excluded_row_count,
        cohort_eligible_row_count=cohort_eligible_row_count,
        cohort_exclusion_rate=cohort_exclusion_rate,
        cohort_exclusion_reason_counts=cohort_exclusion_reason_counts,
        cohort_excluded_mortality_counts=cohort_excluded_mortality_counts,
        cohort_exclusion_reason_mortality_counts=(
            cohort_exclusion_reason_mortality_counts
        ),
        quality_excluded_row_count=quality_excluded_row_count,
        quality_exclusion_rate=quality_exclusion_rate,
        quality_exclusion_reason_counts=quality_exclusion_reason_counts,
        quality_excluded_mortality_counts=quality_excluded_mortality_counts,
        quality_exclusion_reason_mortality_counts=(
            quality_exclusion_reason_mortality_counts
        ),
        label_derivation_status=label_derivation_status,
    )
    return OutcomeSourceAuditResult(
        audit=audit,
        assignments=assignments,
        excluded_rows=excluded_rows.reset_index(drop=True),
    )


def save_outcome_source_audit(
    output_dir: Path,
    audit: OutcomeSourceAudit,
    assignments: pd.DataFrame | None,
    excluded_rows: pd.DataFrame | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_name = audit.dataset_name
    atomic_write_json(
        output_dir / f"{dataset_name}_outcome_audit.json",
        audit.to_dict(),
    )
    if audit.los_derivation_enabled:
        atomic_write_json(
            output_dir / f"{dataset_name}_los_derivation_audit.json",
            audit.los_derivation_summary or {},
        )
    if audit.cohort_contract_summary:
        atomic_write_json(
            output_dir / f"{dataset_name}_cohort_contract_audit.json",
            audit.cohort_contract_summary or {},
        )

    distribution = pd.DataFrame(
        [
            {
                "dataset": dataset_name,
                "class_code": class_code,
                "class_label": class_label,
                "count": int(audit.class_counts[class_label]),
                "rate": float(audit.class_rates[class_label]),
            }
            for class_code, class_label in sorted(
                OUTCOME_CLASS_LABELS.items()
            )
        ]
    )
    distribution.to_csv(
        output_dir / f"{dataset_name}_three_class_distribution.csv",
        index=False,
    )

    after_cohort_count = int(audit.cohort_eligible_row_count)
    flow = pd.DataFrame(
        [
            {"stage": "input_rows", "count": int(audit.input_row_count)},
            {
                "stage": "repeated_headers_removed",
                "count": int(audit.removed_repeated_headers),
            },
            {"stage": "loaded_rows", "count": int(audit.loaded_row_count)},
            {
                "stage": "cohort_eligibility_excluded_rows",
                "count": int(audit.cohort_excluded_row_count),
            },
            {"stage": "rows_after_cohort_contract", "count": after_cohort_count},
            {
                "stage": "data_quality_excluded_rows",
                "count": int(audit.quality_excluded_row_count),
            },
            {"stage": "modelling_eligible_rows", "count": int(audit.retained_row_count)},
        ]
    )
    flow.to_csv(
        output_dir / f"{dataset_name}_outcome_cohort_flow.csv",
        index=False,
    )

    reason_rows: list[dict[str, Any]] = []
    for category, reasons in (
        ("cohort_eligibility", audit.cohort_exclusion_reason_counts or {}),
        ("data_quality", audit.quality_exclusion_reason_counts or {}),
    ):
        for reason, count in sorted(reasons.items()):
            reason_rows.append(
                {
                    "dataset": dataset_name,
                    "exclusion_category": category,
                    "exclusion_reason": reason,
                    "count": int(count),
                }
            )
    pd.DataFrame(
        reason_rows,
        columns=[
            "dataset",
            "exclusion_category",
            "exclusion_reason",
            "count",
        ],
    ).to_csv(
        output_dir / f"{dataset_name}_outcome_exclusion_reasons.csv",
        index=False,
    )

    reason_mortality_rows: list[dict[str, Any]] = []
    for category, values in (
        (
            "cohort_eligibility",
            audit.cohort_exclusion_reason_mortality_counts or {},
        ),
        (
            "data_quality",
            audit.quality_exclusion_reason_mortality_counts or {},
        ),
    ):
        for reason, mortality_counts in sorted(values.items()):
            for mortality_value, count in sorted(mortality_counts.items()):
                reason_mortality_rows.append(
                    {
                        "dataset": dataset_name,
                        "exclusion_category": category,
                        "exclusion_reason": reason,
                        "mortality_value": mortality_value,
                        "count": int(count),
                    }
                )
    pd.DataFrame(
        reason_mortality_rows,
        columns=[
            "dataset",
            "exclusion_category",
            "exclusion_reason",
            "mortality_value",
            "count",
        ],
    ).to_csv(
        output_dir
        / f"{dataset_name}_outcome_exclusion_reason_mortality.csv",
        index=False,
    )

    impact_rows: list[dict[str, Any]] = []
    if excluded_rows is not None and not excluded_rows.empty:
        excluded_rows.to_csv(
            output_dir / f"{dataset_name}_excluded_outcome_rows.csv.gz",
            index=False,
            compression="gzip",
        )
        for category in ("cohort_eligibility", "data_quality"):
            if "exclusion_category" not in excluded_rows.columns:
                category_rows = pd.DataFrame()
            else:
                category_rows = excluded_rows.loc[
                    excluded_rows["exclusion_category"].eq(category)
                ].copy()
            if category_rows.empty:
                continue
            category_rows.to_csv(
                output_dir
                / f"{dataset_name}_{category}_excluded_rows.csv.gz",
                index=False,
                compression="gzip",
            )
            excluded_mortality = _normalise_mortality_series(
                category_rows[audit.mortality_column]
            )
            for value, count in excluded_mortality.value_counts(
                dropna=False
            ).items():
                impact_rows.append(
                    {
                        "dataset": dataset_name,
                        "cohort": f"excluded_{category}",
                        "mortality_value": str(value),
                        "count": int(count),
                    }
                )

    if assignments is not None:
        retained_mortality = pd.to_numeric(
            assignments[audit.mortality_column], errors="coerce"
        )
        for value, count in retained_mortality.value_counts(
            dropna=False
        ).items():
            impact_rows.append(
                {
                    "dataset": dataset_name,
                    "cohort": "retained",
                    "mortality_value": str(value),
                    "count": int(count),
                }
            )

    pd.DataFrame(
        impact_rows,
        columns=["dataset", "cohort", "mortality_value", "count"],
    ).to_csv(
        output_dir / f"{dataset_name}_outcome_exclusion_impact.csv",
        index=False,
    )

    if assignments is not None:
        assignments.to_csv(
            output_dir / f"{dataset_name}_outcome_assignments.csv.gz",
            index=False,
            compression="gzip",
        )
        discrepancy_column = f"{audit.los_column}_absolute_discrepancy_days"
        if discrepancy_column in assignments.columns:
            tolerance = float(
                (audit.los_derivation_summary or {})
                .get("policy", {})
                .get("discrepancy_tolerance_days", 0.05)
            )
            discrepancy = pd.to_numeric(
                assignments[discrepancy_column], errors="coerce"
            )
            mismatch = assignments.loc[
                discrepancy.gt(tolerance).fillna(False)
            ].copy()
            if not mismatch.empty:
                mismatch.to_csv(
                    output_dir / f"{dataset_name}_los_discrepancies.csv.gz",
                    index=False,
                    compression="gzip",
                )
