from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from los_baseline.los_derivation import derive_hospital_los
from los_baseline.outcomes import audit_outcome_source, save_outcome_source_audit


POLICY = {
    "enabled": True,
    "datasets": ["mimic", "eicu"],
    "admission_time_column": "admittime",
    "discharge_time_column": "dischtime",
    "in_hospital_death_time_column": "dischtime",
    "overwrite_existing_los": True,
    "allow_existing_los_fallback": False,
    "discrepancy_tolerance_days": 0.05,
    "fail_on_discrepancy": False,
    "round_decimals": 6,
}


def test_in_hospital_death_uses_discharge_as_death_endpoint() -> None:
    frame = pd.DataFrame(
        {
            "admittime": ["2026-01-01 00:00:00"],
            "dischtime": ["2026-01-04 12:00:00"],
            "LOS": [99.0],
            "hospital_expire_flag": [1],
        }
    )
    derived, summary = derive_hospital_los(
        frame,
        dataset_name="mimic",
        mortality_column="hospital_expire_flag",
        los_column="LOS",
        config=POLICY,
    )
    assert np.isclose(float(derived.loc[0, "LOS"]), 3.5)
    assert derived.loc[0, "_los_endpoint_source"] == "in_hospital_death_at_discharge"
    assert summary.derived_rows == 1
    assert summary.endpoint_source_counts == {"in_hospital_death_at_discharge": 1}


def test_non_in_hospital_death_flag_uses_hospital_discharge_endpoint() -> None:
    frame = pd.DataFrame(
        {
            "admittime": ["2026-01-01 00:00:00"],
            "dischtime": ["2026-01-11 00:00:00"],
            "dod": ["2026-02-01 00:00:00"],
            "LOS": [31.0],
            "hospital_expire_flag": [0],
        }
    )
    derived, _ = derive_hospital_los(
        frame,
        dataset_name="mimic",
        mortality_column="hospital_expire_flag",
        los_column="LOS",
        config=POLICY,
    )
    assert np.isclose(float(derived.loc[0, "LOS"]), 10.0)
    assert derived.loc[0, "_los_endpoint_source"] == "hospital_discharge"


def test_valid_death_rows_are_retained_and_assigned_to_mortality_stage() -> None:
    with tempfile.TemporaryDirectory() as temporary_dir:
        path = Path(temporary_dir) / "clinical.csv"
        pd.DataFrame(
            {
                "subject_id": [1, 2, 3],
                "admittime": [
                    "2026-01-01 00:00:00",
                    "2026-01-01 00:00:00",
                    "2026-01-01 00:00:00",
                ],
                "dischtime": [
                    "2026-01-06 00:00:00",
                    "2026-01-11 00:00:00",
                    "2026-02-10 00:00:00",
                ],
                "LOS": [100.0, 10.0, 40.0],
                "hospital_expire_flag": [1, 0, 1],
            }
        ).to_csv(path, index=False)
        result = audit_outcome_source(
            dataset_name="mimic",
            csv_path=path,
            mortality_column="hospital_expire_flag",
            los_column="LOS",
            landmark_days=28,
            candidate_identifier_columns=["subject_id"],
            los_derivation_config=POLICY,
        )
        audit, assignments = result
        assert audit.status == "ready"
        assert audit.retained_row_count == 3
        assert audit.excluded_row_count == 0
        assert audit.los_derivation_enabled is True
        assert assignments is not None
        assert assignments["LOS"].tolist() == [5.0, 10.0, 40.0]
        assert assignments["stage1_mortality_label"].tolist() == [1, 0, 0]
        assert assignments["three_class_outcome"].tolist() == [1, 0, 2]
        assert "LOS_source_value" in assignments.columns
        assert "LOS_derived_value" in assignments.columns
        output_dir = Path(temporary_dir) / "audit"
        save_outcome_source_audit(
            output_dir, audit, assignments, result.excluded_rows
        )
        assert (output_dir / "mimic_los_derivation_audit.json").is_file()
        assert (output_dir / "mimic_los_discrepancies.csv.gz").is_file()



def test_user_example_is_recomputed_from_admission_and_discharge() -> None:
    frame = pd.DataFrame(
        {
            "admittime": ["2150-11-02 18:02:00"],
            "dischtime": ["2150-11-12 13:45:00"],
            "LOS": [9.8],
            "hospital_expire_flag": [0],
        }
    )
    derived, _ = derive_hospital_los(
        frame,
        dataset_name="mimic",
        mortality_column="hospital_expire_flag",
        los_column="LOS",
        config=POLICY,
    )
    assert np.isclose(float(derived.loc[0, "LOS"]), 9.821528)
    assert np.isclose(float(derived.loc[0, "LOS_source_value"]), 9.8)
    assert derived.loc[0, "_los_endpoint_source"] == "hospital_discharge"

def test_missing_endpoint_is_blocking_when_fallback_is_disabled() -> None:
    frame = pd.DataFrame(
        {
            "admittime": ["2026-01-01 00:00:00"],
            "dischtime": [None],
            "LOS": [5.0],
            "hospital_expire_flag": [0],
        }
    )
    try:
        derive_hospital_los(
            frame,
            dataset_name="mimic",
            mortality_column="hospital_expire_flag",
            los_column="LOS",
            config=POLICY,
        )
    except ValueError as exception:
        assert "fallback is disabled" in str(exception)
    else:
        raise AssertionError("Missing discharge time must be blocking.")
