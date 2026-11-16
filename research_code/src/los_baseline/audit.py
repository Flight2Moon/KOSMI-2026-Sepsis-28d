from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from .data import ClinicalDataset, DatasetSplits
from .utils import atomic_write_json, sha256_file


_AUDIT_SOURCE_COLUMN_CANDIDATES = {
    "gender": "gender_source",
    "age": "age_source",
    "sofa_score": "sofa_method",
    "charlson_comorbidity_index": "charlson_source",
    "aps3": "aps_source_version",
    "heart_rate_mean": "heart_rate_final_source",
    "sbp_mean": "sbp_final_source",
    "dbp_mean": "dbp_final_source",
    "resp_rate_mean": "resp_rate_final_source",
    "temperature_mean": "temperature_final_source",
    "hemoglobin_mean": "hemoglobin_final_source",
    "platelets_mean": "platelets_final_source",
    "wbc_mean": "wbc_final_source",
    "bun_mean": "bun_final_source",
    "creatinine_mean": "creatinine_final_source",
    "glucose_mean": "glucose_final_source",
    "chloride_mean": "chloride_final_source",
    "sodium_mean": "sodium_final_source",
    "potassium_mean": "potassium_final_source",
}


def _split_index_map(
    dataset: ClinicalDataset,
    splits: DatasetSplits | None,
) -> dict[str, np.ndarray]:
    output = {
        "full": np.arange(len(dataset.frame), dtype=np.int64),
    }

    if splits is not None:
        output.update(
            {
                "train": splits.train_indices,
                "validation": splits.validation_indices,
                "internal_test": splits.test_indices,
            }
        )

    return output


def _safe_numeric_summary(series: pd.Series) -> dict[str, Any]:
    numeric = pd.to_numeric(series, errors="coerce").replace(
        [np.inf, -np.inf],
        np.nan,
    )
    valid = numeric.dropna()

    if valid.empty:
        return {
            "count": 0,
            "mean": float("nan"),
            "std": float("nan"),
            "min": float("nan"),
            "q25": float("nan"),
            "median": float("nan"),
            "q75": float("nan"),
            "max": float("nan"),
        }

    return {
        "count": int(valid.size),
        "mean": float(valid.mean()),
        "std": float(valid.std(ddof=1)) if valid.size > 1 else 0.0,
        "min": float(valid.min()),
        "q25": float(valid.quantile(0.25)),
        "median": float(valid.median()),
        "q75": float(valid.quantile(0.75)),
        "max": float(valid.max()),
    }


def save_dataset_audit(
    dataset_name: str,
    dataset: ClinicalDataset,
    thresholds: list[float],
    output_dir: Path,
    splits: DatasetSplits | None = None,
    strict_non_missing_columns: list[str] | None = None,
    fail_on_strict_missing: bool = False,
) -> dict[str, Any]:
    """Save missingness, distributions, and schema information without modifying data."""
    output_dir.mkdir(parents=True, exist_ok=True)
    strict_non_missing_columns = list(strict_non_missing_columns or [])
    split_map = _split_index_map(dataset, splits)

    audited_columns = dataset.feature_columns + [dataset.target_column]
    missingness_rows: list[dict[str, Any]] = []
    numeric_rows: list[dict[str, Any]] = []

    for split_name, indices in split_map.items():
        split_frame = dataset.frame.iloc[indices]

        for column in audited_columns:
            series = split_frame[column]
            missing_count = int(series.isna().sum())
            row_count = int(len(series))

            missingness_rows.append(
                {
                    "dataset": dataset_name,
                    "split": split_name,
                    "column": column,
                    "dtype": str(series.dtype),
                    "row_count": row_count,
                    "non_missing_count": row_count - missing_count,
                    "missing_count": missing_count,
                    "missing_rate": (
                        float(missing_count / row_count)
                        if row_count
                        else float("nan")
                    ),
                    "unique_non_missing_count": int(series.nunique(dropna=True)),
                }
            )

            numeric_rows.append(
                {
                    "dataset": dataset_name,
                    "split": split_name,
                    "column": column,
                    **_safe_numeric_summary(series),
                }
            )

    missingness = pd.DataFrame(missingness_rows)
    numeric_summary = pd.DataFrame(numeric_rows)
    missingness.to_csv(
        output_dir / f"{dataset_name}_missingness.csv",
        index=False,
    )
    numeric_summary.to_csv(
        output_dir / f"{dataset_name}_numeric_summary.csv",
        index=False,
    )

    class_rows: list[dict[str, Any]] = []

    for threshold_days in thresholds:
        labels = dataset.labels(threshold_days)

        for split_name, indices in split_map.items():
            split_labels = labels[indices]
            positive_count = int((split_labels == 1).sum())
            negative_count = int((split_labels == 0).sum())
            sample_count = int(len(split_labels))

            class_rows.append(
                {
                    "dataset": dataset_name,
                    "split": split_name,
                    "threshold_days": float(threshold_days),
                    "sample_count": sample_count,
                    "positive_count": positive_count,
                    "negative_count": negative_count,
                    "positive_rate": (
                        float(positive_count / sample_count)
                        if sample_count
                        else float("nan")
                    ),
                    "random_auprc_baseline": (
                        float(positive_count / sample_count)
                        if sample_count
                        else float("nan")
                    ),
                }
            )

    class_distribution = pd.DataFrame(class_rows)
    class_distribution.to_csv(
        output_dir / f"{dataset_name}_class_distribution.csv",
        index=False,
    )

    duplicate_counts = {}

    for column in dataset.identifier_columns:
        duplicate_counts[column] = int(
            dataset.frame[column].duplicated(keep=False).sum()
        )

    strict_missing: dict[str, int] = {}

    for column in strict_non_missing_columns:
        if column not in dataset.frame.columns:
            strict_missing[column] = -1
        else:
            strict_missing[column] = int(dataset.frame[column].isna().sum())

    strict_violations = {
        column: count
        for column, count in strict_missing.items()
        if count != 0
    }

    overview = {
        "dataset": dataset_name,
        "source_path": str(dataset.source_path),
        "source_sha256": sha256_file(dataset.source_path),
        "rows_after_validation": int(len(dataset.frame)),
        "feature_count": int(len(dataset.feature_columns)),
        "feature_columns": list(dataset.feature_columns),
        "target_column": dataset.target_column,
        "identifier_columns": list(dataset.identifier_columns),
        "removed_repeated_headers": int(dataset.removed_repeated_headers),
        "removed_missing_targets": int(dataset.removed_missing_targets),
        "total_feature_missing_values": int(
            dataset.frame[dataset.feature_columns].isna().sum().sum()
        ),
        "duplicate_identifier_rows": duplicate_counts,
        "strict_non_missing_columns": strict_non_missing_columns,
        "strict_missing_counts": strict_missing,
        "strict_violations": strict_violations,
        "strict_check_passed": not bool(strict_violations),
        "split_names": list(split_map),
    }
    atomic_write_json(
        output_dir / f"{dataset_name}_overview.json",
        overview,
    )

    if fail_on_strict_missing and strict_violations:
        raise ValueError(
            f"Strict non-missing validation failed for {dataset_name}: "
            f"{strict_violations}"
        )

    return {
        "overview": overview,
        "missingness_csv": str(
            output_dir / f"{dataset_name}_missingness.csv"
        ),
        "numeric_summary_csv": str(
            output_dir / f"{dataset_name}_numeric_summary.csv"
        ),
        "class_distribution_csv": str(
            output_dir / f"{dataset_name}_class_distribution.csv"
        ),
    }


def save_preprocessing_audit(
    model_name: str,
    feature_names: list[str],
    preprocessor: Pipeline,
    raw_split_frames: dict[str, pd.DataFrame],
    output_dir: Path,
) -> dict[str, Any]:
    """Record train-fitted imputer/scaler values and split-level missingness."""
    output_dir.mkdir(parents=True, exist_ok=True)
    missing_rows: list[dict[str, Any]] = []

    for split_name, frame in raw_split_frames.items():
        ordered = frame.loc[:, feature_names]

        for column in feature_names:
            missing_count = int(ordered[column].isna().sum())
            row_count = int(len(ordered))
            missing_rows.append(
                {
                    "model_name": model_name,
                    "split": split_name,
                    "feature": column,
                    "row_count": row_count,
                    "missing_count": missing_count,
                    "missing_rate": (
                        float(missing_count / row_count)
                        if row_count
                        else float("nan")
                    ),
                    "values_imputed": missing_count,
                }
            )

    missingness = pd.DataFrame(missing_rows)
    missingness.to_csv(
        output_dir / "raw_missingness_by_split.csv",
        index=False,
    )

    imputer = preprocessor.named_steps.get("imputer")
    if imputer is None or not hasattr(imputer, "statistics_"):
        raise ValueError("The fitted preprocessor does not contain imputer statistics.")

    imputer_frame = pd.DataFrame(
        {
            "feature": feature_names,
            "train_fitted_median": np.asarray(
                imputer.statistics_,
                dtype=np.float64,
            ),
        }
    )
    imputer_frame.to_csv(
        output_dir / "imputer_statistics.csv",
        index=False,
    )

    scaler_path: str | None = None
    scaler = preprocessor.named_steps.get("scaler")

    if scaler is not None:
        scaler_frame = pd.DataFrame(
            {
                "feature": feature_names,
                "train_fitted_mean": np.asarray(
                    scaler.mean_,
                    dtype=np.float64,
                ),
                "train_fitted_scale": np.asarray(
                    scaler.scale_,
                    dtype=np.float64,
                ),
                "train_fitted_variance": np.asarray(
                    scaler.var_,
                    dtype=np.float64,
                ),
            }
        )
        scaler_output = output_dir / "scaler_statistics.csv"
        scaler_frame.to_csv(scaler_output, index=False)
        scaler_path = str(scaler_output)

    split_totals = (
        missingness.groupby("split", as_index=False)["missing_count"]
        .sum()
        .set_index("split")["missing_count"]
        .astype(int)
        .to_dict()
    )

    summary = {
        "model_name": model_name,
        "fit_scope": "mimic_train_only",
        "imputer_strategy": str(getattr(imputer, "strategy", "unknown")),
        "scaler_applied": scaler is not None,
        "feature_count": len(feature_names),
        "raw_missing_values_by_split": split_totals,
        "raw_missingness_csv": str(
            output_dir / "raw_missingness_by_split.csv"
        ),
        "imputer_statistics_csv": str(
            output_dir / "imputer_statistics.csv"
        ),
        "scaler_statistics_csv": scaler_path,
    }
    atomic_write_json(output_dir / "preprocessing_audit.json", summary)
    return summary


def _strip_compression_suffix(path: Path) -> str:
    name = path.name
    for suffix in [".gz", ".bz2", ".xz", ".zip"]:
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    for suffix in [".csv"]:
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    return name


def resolve_eicu_audit_path(
    configured_path: str | None,
    eicu_path: Path,
    project_root: Path,
    *,
    auto_discovery: bool = False,
) -> tuple[Path | None, list[str]]:
    """Resolve an optional CSV harmonizer-provenance table.

    EXP123 V7.2 validates eICU directly from ``paths.eicu_csv``. Harmonizer
    provenance is an opt-in supplementary analysis and is never required for
    model training or external validation. Automatic discovery is disabled by
    default so that a nearby Parquet file cannot become an accidental runtime
    dependency.
    """
    candidates: list[Path] = []

    if configured_path:
        configured = Path(configured_path).expanduser()
        if not configured.is_absolute():
            configured = project_root / configured
        candidates.append(configured.resolve())

    if auto_discovery:
        base_name = _strip_compression_suffix(eicu_path)
        parent = eicu_path.parent
        candidates.extend(
            [
                parent / f"{base_name}_audit.csv",
                parent / f"{base_name}_audit.csv.gz",
                parent / "eicu_mimic_schema_complete_audit.csv",
                parent / "eicu_mimic_schema_complete_audit.csv.gz",
            ]
        )

    unique_candidates: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        resolved = candidate.expanduser().resolve()
        key = str(resolved)
        if key in seen:
            continue
        seen.add(key)
        unique_candidates.append(resolved)

    for candidate in unique_candidates:
        if candidate.is_file():
            return candidate, [str(path) for path in unique_candidates]

    return None, [str(path) for path in unique_candidates]


def _read_audit_table(path: Path) -> pd.DataFrame:
    """Read optional harmonizer provenance without a Parquet dependency."""
    lower_name = path.name.lower()
    if lower_name.endswith((".csv", ".csv.gz", ".csv.bz2", ".csv.xz")):
        return pd.read_csv(path, low_memory=False, compression="infer")
    raise ValueError(
        "EXP123 V7.2 accepts optional eICU harmonizer provenance only as "
        "CSV/CSV.GZ. The eICU model dataset itself must be supplied through "
        "paths.eicu_csv; no Parquet audit file is required."
    )


def _normalized_key(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip().fillna("<NA>")


def _fallback_group(ratio: float) -> str:
    if not np.isfinite(ratio):
        return "unknown"
    if ratio == 0:
        return "0%"
    if ratio <= 0.2:
        return ">0%-20%"
    if ratio <= 0.5:
        return ">20%-50%"
    return ">50%"


def attach_eicu_harmonizer_audit(
    dataset: ClinicalDataset,
    configured_path: str | None,
    project_root: Path,
    output_dir: Path,
    imputed_feature_denominator: int = 18,
    required: bool = False,
    *,
    enabled: bool = False,
    auto_discovery: bool = False,
) -> dict[str, Any]:
    """Optionally attach harmonizer provenance without affecting CSV validation.

    The primary eICU audit, cohort construction, outcome generation and external
    validation always operate on ``paths.eicu_csv``. This helper is only for
    supplementary source/fallback subgroup analysis.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    if not enabled:
        payload = {
            "status": "disabled",
            "required": False,
            "message": (
                "Optional eICU harmonizer provenance is disabled. The eICU CSV "
                "is still loaded, audited and used for external validation."
            ),
        }
        atomic_write_json(output_dir / "eicu_harmonizer_audit.json", payload)
        return payload

    audit_path, attempted_paths = resolve_eicu_audit_path(
        configured_path,
        dataset.source_path,
        project_root,
        auto_discovery=auto_discovery,
    )

    if audit_path is None:
        payload = {
            "status": "not_available",
            "required": bool(required),
            "attempted_paths": attempted_paths,
            "message": (
                "The optional eICU harmonizer provenance CSV was not found. Model "
                "evaluation can continue, but source/fallback subgroup "
                "analysis is unavailable."
            ),
        }
        atomic_write_json(output_dir / "eicu_harmonizer_audit.json", payload)

        if required:
            raise FileNotFoundError(payload["message"] + " " + str(attempted_paths))

        return payload

    try:
        audit = _read_audit_table(audit_path)
    except Exception as exception:
        payload = {
            "status": "read_failed",
            "required": bool(required),
            "audit_path": str(audit_path),
            "error": repr(exception),
        }
        atomic_write_json(output_dir / "eicu_harmonizer_audit.json", payload)

        if required:
            raise RuntimeError(
                f"Unable to read required eICU audit table: {audit_path}"
            ) from exception

        return payload

    audit.columns = audit.columns.astype(str).str.strip()
    alignment_method = "unavailable"
    aligned: pd.DataFrame | None = None
    key_used: str | None = None

    for key in ["stay_id", "hadm_id", "subject_id"]:
        if key not in audit.columns or key not in dataset.frame.columns:
            continue

        left_key = _normalized_key(dataset.frame[key])
        right_key = _normalized_key(audit[key])

        if left_key.duplicated().any() or right_key.duplicated().any():
            continue

        right = audit.copy()
        right["__audit_key"] = right_key
        left = pd.DataFrame(
            {
                "__dataset_order": np.arange(len(dataset.frame), dtype=np.int64),
                "__audit_key": left_key,
            }
        )
        merged = left.merge(
            right.drop(columns=[key], errors="ignore"),
            on="__audit_key",
            how="left",
            sort=False,
            validate="one_to_one",
        ).sort_values("__dataset_order")

        matched = int(merged["__audit_key"].isin(set(right_key)).sum())

        if matched == len(dataset.frame):
            aligned = merged.reset_index(drop=True)
            alignment_method = f"identifier:{key}"
            key_used = key
            break

    if aligned is None and len(audit) == len(dataset.frame):
        aligned = audit.reset_index(drop=True).copy()
        aligned.insert(
            0,
            "__dataset_order",
            np.arange(len(aligned), dtype=np.int64),
        )
        alignment_method = "row_order_equal_length"

    if aligned is None:
        payload = {
            "status": "alignment_failed",
            "required": bool(required),
            "audit_path": str(audit_path),
            "dataset_rows": int(len(dataset.frame)),
            "audit_rows": int(len(audit)),
            "attempted_identifier_columns": ["stay_id", "hadm_id", "subject_id"],
        }
        atomic_write_json(output_dir / "eicu_harmonizer_audit.json", payload)

        if required:
            raise RuntimeError(
                "The required eICU audit table could not be aligned to the "
                "external dataset."
            )

        return payload

    denominator = max(int(imputed_feature_denominator), 1)

    if "imputed_feature_count" in aligned.columns:
        imputed_count = pd.to_numeric(
            aligned["imputed_feature_count"],
            errors="coerce",
        )
    else:
        imputed_count = pd.Series(np.nan, index=aligned.index, dtype=np.float64)

    imputed_ratio = imputed_count / float(denominator)
    dataset.frame["_audit_imputed_feature_count"] = imputed_count.to_numpy()
    dataset.frame["_audit_imputed_feature_ratio"] = imputed_ratio.to_numpy()
    dataset.frame["_audit_fallback_group"] = [
        _fallback_group(float(value))
        for value in imputed_ratio.to_numpy(dtype=np.float64)
    ]

    if "completion_method" in aligned.columns:
        dataset.frame["_audit_completion_method"] = aligned[
            "completion_method"
        ].astype("string").to_numpy()

    source_rows: list[dict[str, Any]] = []

    for feature in dataset.feature_columns:
        source_column = _AUDIT_SOURCE_COLUMN_CANDIDATES.get(feature)
        if source_column is None or source_column not in aligned.columns:
            continue

        values = aligned[source_column].astype("string").fillna("<missing>")
        counts = values.value_counts(dropna=False)

        for source_value, count in counts.items():
            source_rows.append(
                {
                    "feature": feature,
                    "source_column": source_column,
                    "source_value": str(source_value),
                    "count": int(count),
                    "rate": float(count / len(aligned)) if len(aligned) else float("nan"),
                }
            )

    source_summary = pd.DataFrame(source_rows)
    source_summary_path = output_dir / "eicu_harmonizer_value_sources.csv"
    source_summary.to_csv(source_summary_path, index=False)

    burden_columns = [
        column
        for column in ["subject_id", "hadm_id", "stay_id"]
        if column in dataset.frame.columns
    ]
    burden = dataset.frame.loc[:, burden_columns].copy()
    burden.insert(0, "dataset_row_index", np.arange(len(dataset.frame)))
    burden["source_row_index"] = dataset.source_row_index
    burden["imputed_feature_count"] = imputed_count.to_numpy()
    burden["imputed_feature_denominator"] = denominator
    burden["imputed_feature_ratio"] = imputed_ratio.to_numpy()
    burden["fallback_group"] = dataset.frame["_audit_fallback_group"].to_numpy()
    burden_path = output_dir / "eicu_harmonizer_patient_burden.csv.gz"
    burden.to_csv(burden_path, index=False, compression="gzip")

    group_summary = (
        burden.groupby("fallback_group", dropna=False)
        .agg(
            sample_count=("dataset_row_index", "size"),
            mean_imputed_feature_count=("imputed_feature_count", "mean"),
            median_imputed_feature_count=("imputed_feature_count", "median"),
            mean_imputed_feature_ratio=("imputed_feature_ratio", "mean"),
        )
        .reset_index()
    )
    group_summary_path = output_dir / "eicu_harmonizer_burden_groups.csv"
    group_summary.to_csv(group_summary_path, index=False)

    payload = {
        "status": "attached",
        "required": bool(required),
        "audit_path": str(audit_path),
        "audit_sha256": sha256_file(audit_path),
        "dataset_rows": int(len(dataset.frame)),
        "audit_rows": int(len(audit)),
        "alignment_method": alignment_method,
        "alignment_identifier": key_used,
        "imputed_feature_denominator": denominator,
        "imputed_feature_count_available": "imputed_feature_count" in aligned.columns,
        "mean_imputed_feature_count": (
            float(imputed_count.mean())
            if imputed_count.notna().any()
            else float("nan")
        ),
        "median_imputed_feature_count": (
            float(imputed_count.median())
            if imputed_count.notna().any()
            else float("nan")
        ),
        "value_sources_csv": str(source_summary_path),
        "patient_burden_csv": str(burden_path),
        "burden_groups_csv": str(group_summary_path),
        "attached_prediction_columns": [
            "_audit_imputed_feature_count",
            "_audit_imputed_feature_ratio",
            "_audit_fallback_group",
            "_audit_completion_method",
        ],
    }
    atomic_write_json(output_dir / "eicu_harmonizer_audit.json", payload)
    return payload
