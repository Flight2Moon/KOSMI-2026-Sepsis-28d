from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from los_baseline.constants import OUTCOME_CLASS_LABELS
from los_baseline.outcomes import (
    audit_outcome_source,
    derive_stage_labels,
    derive_three_class_outcome,
    save_outcome_source_audit,
    validate_outcome_source_columns,
)


class OutcomeDefinitionTests(unittest.TestCase):
    def test_landmark_boundaries(self) -> None:
        frame = pd.DataFrame(
            {
                "LOS": [27.999, 28.0, 10.0, 28.0, 35.0, 35.0],
                "hospital_expire_flag": [0, 0, 1, 1, 0, 1],
            }
        )
        labels = derive_three_class_outcome(
            frame,
            los_column="LOS",
            mortality_column="hospital_expire_flag",
            landmark_days=28.0,
        )
        np.testing.assert_array_equal(labels, [0, 2, 1, 1, 2, 2])

    def test_stage_labels_use_ground_truth_survivors(self) -> None:
        labels = np.asarray([0, 1, 2, 0, 2, 1], dtype=np.int64)
        y1, y2 = derive_stage_labels(labels)
        np.testing.assert_array_equal(y1, [0, 1, 0, 0, 0, 1])
        np.testing.assert_array_equal(y2, [0, 1, 0, 1])

    def test_missing_mortality_is_blocking(self) -> None:
        frame = pd.DataFrame({"LOS": [1.0, 2.0]})
        with self.assertRaises(KeyError):
            validate_outcome_source_columns(
                frame,
                mortality_column="hospital_expire_flag",
                los_column="LOS",
            )

    def test_invalid_mortality_is_blocking(self) -> None:
        frame = pd.DataFrame(
            {
                "LOS": [1.0, 2.0],
                "hospital_expire_flag": [0, 2],
            }
        )
        with self.assertRaises(ValueError):
            validate_outcome_source_columns(
                frame,
                mortality_column="hospital_expire_flag",
                los_column="LOS",
            )

    def test_outcome_audit_reports_missing_source_column(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            path = Path(temporary_dir) / "clinical.csv"
            pd.DataFrame({"LOS": [1.0, 2.0]}).to_csv(path, index=False)
            audit, assignments = audit_outcome_source(
                dataset_name="mimic",
                csv_path=path,
                mortality_column="hospital_expire_flag",
                los_column="LOS",
                landmark_days=28.0,
            )
            self.assertEqual(audit.status, "blocked")
            self.assertIsNone(assignments)
            self.assertTrue(audit.blocking_issues)

    def test_outcome_audit_writes_complete_assignments(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            path = Path(temporary_dir) / "clinical.csv"
            frame = pd.DataFrame(
                {
                    "subject_id": [1, 2, 3, 4, 5, 6],
                    "LOS": [5.0, 10.0, 28.0, 30.0, 12.0, 35.0],
                    "hospital_expire_flag": [0, 1, 0, 0, 0, 1],
                }
            )
            frame.to_csv(path, index=False)
            audit, assignments = audit_outcome_source(
                dataset_name="mimic",
                csv_path=path,
                mortality_column="hospital_expire_flag",
                los_column="LOS",
                landmark_days=28.0,
                candidate_identifier_columns=["subject_id"],
            )
            self.assertEqual(audit.status, "ready")
            self.assertIsNotNone(assignments)
            assert assignments is not None
            self.assertEqual(len(assignments), len(frame))
            self.assertEqual(
                sum(audit.class_counts.values()),
                len(frame),
            )
            self.assertEqual(
                set(audit.class_counts),
                set(OUTCOME_CLASS_LABELS.values()),
            )
            self.assertEqual(audit.boundary_counts["los_equal_landmark"], 1)

    def test_negative_los_remains_blocking_without_exclusion_policy(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            path = Path(temporary_dir) / "clinical.csv"
            pd.DataFrame(
                {
                    "subject_id": [1, 2, 3],
                    "LOS": [5.0, -0.1, 30.0],
                    "hospital_expire_flag": [0, 1, 0],
                }
            ).to_csv(path, index=False)
            result = audit_outcome_source(
                dataset_name="mimic",
                csv_path=path,
                mortality_column="hospital_expire_flag",
                los_column="LOS",
                landmark_days=28.0,
                candidate_identifier_columns=["subject_id"],
            )
            audit, assignments = result
            self.assertEqual(audit.status, "blocked")
            self.assertIsNone(assignments)
            self.assertEqual(audit.excluded_row_count, 0)

    def test_negative_los_can_be_excluded_with_full_traceability(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            path = Path(temporary_dir) / "clinical.csv"
            output_dir = Path(temporary_dir) / "audit"
            pd.DataFrame(
                {
                    "subject_id": [1, 2, 3, 4],
                    "hadm_id": [11, 22, 33, 44],
                    "LOS": [5.0, -0.1, 30.0, 10.0],
                    "hospital_expire_flag": [0, 1, 0, 1],
                }
            ).to_csv(path, index=False)
            result = audit_outcome_source(
                dataset_name="mimic",
                csv_path=path,
                mortality_column="hospital_expire_flag",
                los_column="LOS",
                landmark_days=28.0,
                candidate_identifier_columns=["subject_id", "hadm_id"],
                exclusion_policy={
                    "enabled": True,
                    "datasets": ["mimic"],
                    "rules": {"negative_los": True},
                    "max_excluded_fraction": 0.5,
                    "max_excluded_rows": 10,
                    "fail_if_limit_exceeded": True,
                },
            )
            audit, assignments = result
            self.assertEqual(audit.status, "ready")
            self.assertEqual(audit.excluded_row_count, 1)
            self.assertEqual(audit.retained_row_count, 3)
            self.assertEqual(audit.exclusion_reason_counts, {"negative_los": 1})
            self.assertIsNotNone(assignments)
            assert assignments is not None
            self.assertEqual(assignments["_source_row_index"].tolist(), [0, 2, 3])
            self.assertEqual(
                result.excluded_rows["_source_row_index"].tolist(), [1]
            )
            self.assertEqual(
                result.excluded_rows["exclusion_reason"].tolist(),
                ["negative_los"],
            )
            save_outcome_source_audit(
                output_dir, audit, assignments, result.excluded_rows
            )
            self.assertTrue(
                (output_dir / "mimic_excluded_outcome_rows.csv.gz").is_file()
            )
            self.assertTrue(
                (output_dir / "mimic_outcome_cohort_flow.csv").is_file()
            )
            self.assertTrue(
                (output_dir / "mimic_outcome_exclusion_reasons.csv").is_file()
            )
            self.assertTrue(
                (output_dir / "mimic_outcome_exclusion_impact.csv").is_file()
            )

    def test_exclusion_limit_exceedance_is_blocking(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            path = Path(temporary_dir) / "clinical.csv"
            pd.DataFrame(
                {
                    "LOS": [-0.1, -0.2, 5.0],
                    "hospital_expire_flag": [1, 1, 0],
                }
            ).to_csv(path, index=False)
            result = audit_outcome_source(
                dataset_name="mimic",
                csv_path=path,
                mortality_column="hospital_expire_flag",
                los_column="LOS",
                landmark_days=28.0,
                exclusion_policy={
                    "enabled": True,
                    "datasets": ["mimic"],
                    "rules": {"negative_los": True},
                    "max_excluded_fraction": 0.1,
                    "max_excluded_rows": 1,
                    "fail_if_limit_exceeded": True,
                },
            )
            audit, assignments = result
            self.assertEqual(audit.status, "blocked")
            self.assertIsNotNone(assignments)
            self.assertEqual(audit.label_derivation_status, "ready")
            self.assertEqual(sum(audit.class_counts.values()), 1)
            self.assertEqual(audit.excluded_row_count, 2)
            self.assertTrue(
                any(
                    "max_excluded_rows" in issue
                    for issue in audit.blocking_issues
                )
            )

    def test_non_enabled_invalid_rule_remains_blocking(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            path = Path(temporary_dir) / "clinical.csv"
            pd.DataFrame(
                {
                    "LOS": [1.0, 2.0],
                    "hospital_expire_flag": [0, 2],
                }
            ).to_csv(path, index=False)
            result = audit_outcome_source(
                dataset_name="mimic",
                csv_path=path,
                mortality_column="hospital_expire_flag",
                los_column="LOS",
                landmark_days=28.0,
                exclusion_policy={
                    "enabled": True,
                    "datasets": ["mimic"],
                    "rules": {
                        "negative_los": True,
                        "invalid_mortality": False,
                    },
                    "max_excluded_fraction": 1.0,
                    "max_excluded_rows": 100,
                },
            )
            audit, assignments = result
            self.assertEqual(audit.status, "blocked")
            self.assertIsNone(assignments)
            self.assertEqual(audit.excluded_row_count, 0)


if __name__ == "__main__":
    unittest.main()
