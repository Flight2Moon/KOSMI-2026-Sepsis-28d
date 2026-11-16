from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class LOSDerivationSummary:
    enabled: bool
    dataset_name: str
    admission_time_column: str
    discharge_time_column: str
    mortality_column: str
    los_column: str
    input_rows: int
    derived_rows: int
    fallback_rows: int
    missing_admission_rows: int
    missing_endpoint_rows: int
    negative_derived_los_rows: int
    discrepancy_rows: int
    maximum_absolute_discrepancy_days: float | None
    endpoint_source_counts: dict[str, int]
    policy: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def normalise_los_derivation_config(
    config: Mapping[str, Any] | None,
    *,
    dataset_name: str,
) -> dict[str, Any]:
    source = dict(config or {})
    datasets = [str(value) for value in source.get("datasets", ["mimic", "eicu"])]
    enabled = bool(source.get("enabled", False)) and dataset_name in set(datasets)
    admission_column = str(source.get("admission_time_column", "admittime"))
    discharge_column = str(source.get("discharge_time_column", "dischtime"))
    death_column = str(
        source.get("in_hospital_death_time_column", discharge_column)
    )
    return {
        "enabled": enabled,
        "datasets": datasets,
        "admission_time_column": admission_column,
        "discharge_time_column": discharge_column,
        "in_hospital_death_time_column": death_column,
        "overwrite_existing_los": bool(source.get("overwrite_existing_los", True)),
        "allow_existing_los_fallback": bool(
            source.get("allow_existing_los_fallback", False)
        ),
        "discrepancy_tolerance_days": float(
            source.get("discrepancy_tolerance_days", 0.05)
        ),
        "fail_on_discrepancy": bool(source.get("fail_on_discrepancy", False)),
        "round_decimals": int(source.get("round_decimals", 6)),
        "preserve_original_los": True,
        "post_discharge_death_changes_hospital_los": False,
        "hospital_expire_flag_semantics": "in_hospital_death_only",
    }


def required_los_derivation_columns(
    config: Mapping[str, Any] | None,
    *,
    dataset_name: str,
    mortality_column: str,
) -> list[str]:
    policy = normalise_los_derivation_config(config, dataset_name=dataset_name)
    if not policy["enabled"]:
        return []
    columns = [
        mortality_column,
        str(policy["admission_time_column"]),
        str(policy["discharge_time_column"]),
        str(policy["in_hospital_death_time_column"]),
    ]
    return list(dict.fromkeys(columns))


def derive_hospital_los(
    frame: pd.DataFrame,
    *,
    dataset_name: str,
    mortality_column: str,
    los_column: str,
    config: Mapping[str, Any] | None,
) -> tuple[pd.DataFrame, LOSDerivationSummary]:
    """Derive LOS from the configured origin to discharge/death endpoint.

    The origin may be hospital admission or ICU admission.
    hospital_expire_flag=1 means the configured death timestamp is used as the
    endpoint; hospital_expire_flag=0 uses discharge. Later post-discharge death
    never extends the in-hospital LOS.
    """
    policy = normalise_los_derivation_config(config, dataset_name=dataset_name)
    output = frame.copy()
    row_count = int(len(output))

    if not policy["enabled"]:
        summary = LOSDerivationSummary(
            enabled=False,
            dataset_name=str(dataset_name),
            admission_time_column=str(policy["admission_time_column"]),
            discharge_time_column=str(policy["discharge_time_column"]),
            mortality_column=str(mortality_column),
            los_column=str(los_column),
            input_rows=row_count,
            derived_rows=0,
            fallback_rows=0,
            missing_admission_rows=0,
            missing_endpoint_rows=0,
            negative_derived_los_rows=0,
            discrepancy_rows=0,
            maximum_absolute_discrepancy_days=None,
            endpoint_source_counts={},
            policy=policy,
        )
        return output, summary

    required = required_los_derivation_columns(
        config,
        dataset_name=dataset_name,
        mortality_column=mortality_column,
    )
    missing = [column for column in required if column not in output.columns]
    if missing:
        raise KeyError(
            "LOS derivation columns are missing: "
            f"{missing}. Available columns: {list(output.columns)}"
        )

    mortality = pd.to_numeric(output[mortality_column], errors="coerce")
    admission = pd.to_datetime(
        output[str(policy["admission_time_column"])],
        errors="coerce",
    )
    discharge = pd.to_datetime(
        output[str(policy["discharge_time_column"])],
        errors="coerce",
    )
    death_time = pd.to_datetime(
        output[str(policy["in_hospital_death_time_column"])],
        errors="coerce",
    )

    in_hospital_death = mortality.eq(1.0)
    endpoint = discharge.copy()
    endpoint.loc[in_hospital_death] = death_time.loc[in_hospital_death]
    endpoint_source = pd.Series(
        np.where(
            in_hospital_death,
            "in_hospital_death_at_discharge",
            "hospital_discharge",
        ),
        index=output.index,
        dtype="string",
    )

    derived = (endpoint - admission).dt.total_seconds() / 86400.0
    derived = derived.round(int(policy["round_decimals"]))
    current_los = (
        pd.to_numeric(output[los_column], errors="coerce")
        if los_column in output.columns
        else pd.Series(np.nan, index=output.index, dtype=float)
    )
    source_value_column = f"{los_column}_source_value"
    original = (
        pd.to_numeric(output[source_value_column], errors="coerce")
        if source_value_column in output.columns
        else current_los.copy()
    )

    valid_derived = admission.notna() & endpoint.notna() & derived.notna()
    fallback_mask = ~valid_derived & current_los.notna()
    if fallback_mask.any() and not policy["allow_existing_los_fallback"]:
        missing_admission = int(admission.isna().sum())
        missing_endpoint = int(endpoint.isna().sum())
        raise ValueError(
            "LOS derivation could not be completed for all rows and fallback is "
            "disabled. "
            f"missing_admission={missing_admission}, "
            f"missing_endpoint={missing_endpoint}, "
            f"fallback_candidates={int(fallback_mask.sum())}."
        )

    effective = derived.copy()
    if policy["allow_existing_los_fallback"]:
        effective.loc[fallback_mask] = current_los.loc[fallback_mask]
        endpoint_source.loc[fallback_mask] = "existing_los_fallback"

    if not bool(policy["overwrite_existing_los"]):
        effective = current_los.where(current_los.notna(), effective)

    discrepancy = (original - derived).abs()
    comparable = original.notna() & derived.notna()
    discrepancy_mask = comparable & discrepancy.gt(
        float(policy["discrepancy_tolerance_days"])
    )
    discrepancy_count = int(discrepancy_mask.sum())
    maximum_discrepancy = (
        float(discrepancy.loc[comparable].max()) if comparable.any() else None
    )
    if discrepancy_count and policy["fail_on_discrepancy"]:
        raise ValueError(
            "Existing and derived LOS values disagree beyond the configured "
            "tolerance. "
            f"count={discrepancy_count}, "
            f"tolerance_days={policy['discrepancy_tolerance_days']}, "
            f"max_abs_difference_days={maximum_discrepancy}."
        )

    output[f"{los_column}_source_value"] = original
    output[f"{los_column}_derived_value"] = derived
    output[f"{los_column}_absolute_discrepancy_days"] = discrepancy
    output["_los_endpoint_time"] = endpoint
    output["_los_endpoint_source"] = endpoint_source
    output["_los_used_existing_fallback"] = fallback_mask.astype(bool)
    output[los_column] = effective

    endpoint_counts = {
        str(key): int(value)
        for key, value in endpoint_source.value_counts(dropna=False).to_dict().items()
    }
    summary = LOSDerivationSummary(
        enabled=True,
        dataset_name=str(dataset_name),
        admission_time_column=str(policy["admission_time_column"]),
        discharge_time_column=str(policy["discharge_time_column"]),
        mortality_column=str(mortality_column),
        los_column=str(los_column),
        input_rows=row_count,
        derived_rows=int(valid_derived.sum()),
        fallback_rows=int(fallback_mask.sum()),
        missing_admission_rows=int(admission.isna().sum()),
        missing_endpoint_rows=int(endpoint.isna().sum()),
        negative_derived_los_rows=int(derived.lt(0).fillna(False).sum()),
        discrepancy_rows=discrepancy_count,
        maximum_absolute_discrepancy_days=maximum_discrepancy,
        endpoint_source_counts=endpoint_counts,
        policy=policy,
    )
    return output, summary
