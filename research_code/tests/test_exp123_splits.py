from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from los_baseline.data import create_stratified_group_splits
from los_baseline.exp123_splits import (
    SPLIT_NAMES,
    build_split_assignment,
    compute_split_hash,
    create_class_distribution,
    create_task_population_distribution,
    load_ready_mimic_outcomes,
    run_common_split_step,
    validate_common_split_assignment,
)
from los_baseline.utils import atomic_write_json


class Exp123CommonSplitTests(unittest.TestCase):
    @staticmethod
    def _outcome_frame(n_per_class: int = 40) -> pd.DataFrame:
        rows: list[dict[str, object]] = []
        source_index = 0
        for class_code in [0, 1, 2]:
            for patient_index in range(n_per_class):
                rows.append(
                    {
                        "_source_row_index": source_index,
                        "subject_id": f"class_{class_code}_patient_{patient_index}",
                        "three_class_outcome": class_code,
                        "three_class_label": [
                            "A_discharged_alive",
                            "B_died",
                            "C_still_admitted",
                        ][class_code],
                        "stage1_mortality_label": int(class_code == 1),
                        "stage2_eligible": bool(class_code != 1),
                        "stage2_long_stay_label": (
                            np.nan if class_code == 1 else float(class_code == 2)
                        ),
                    }
                )
                source_index += 1
        return pd.DataFrame(rows)

    def test_shared_split_has_no_overlap_and_all_classes(self) -> None:
        frame = self._outcome_frame()
        splits = create_stratified_group_splits(
            frame=frame,
            strata=frame["three_class_outcome"].to_numpy(),
            train_ratio=0.70,
            validation_ratio=0.15,
            test_ratio=0.15,
            seed=42,
            max_attempts=100,
            patient_id_column="subject_id",
            required_classes=[0, 1, 2],
            stratification_target="three_class_outcome",
        )
        assignment = build_split_assignment(
            outcomes=frame,
            splits=splits,
            identifier_columns=["subject_id"],
        )
        integrity = validate_common_split_assignment(
            assignment=assignment,
            patient_id_column="subject_id",
        )
        self.assertTrue(integrity.complete_row_coverage)
        self.assertTrue(integrity.no_row_overlap)
        self.assertTrue(integrity.no_patient_overlap)
        self.assertTrue(integrity.all_classes_present_per_split)
        self.assertTrue(integrity.stage1_both_classes_per_split)
        self.assertTrue(integrity.stage2_both_classes_per_split)
        self.assertEqual(set(assignment["split"]), set(SPLIT_NAMES))

    def test_same_seed_produces_identical_split_hash(self) -> None:
        frame = self._outcome_frame()
        hashes: list[str] = []
        for _ in range(2):
            splits = create_stratified_group_splits(
                frame=frame,
                strata=frame["three_class_outcome"].to_numpy(),
                train_ratio=0.70,
                validation_ratio=0.15,
                test_ratio=0.15,
                seed=42,
                max_attempts=100,
                patient_id_column="subject_id",
                required_classes=[0, 1, 2],
                stratification_target="three_class_outcome",
            )
            assignment = build_split_assignment(
                outcomes=frame,
                splits=splits,
                identifier_columns=["subject_id"],
            )
            hashes.append(compute_split_hash(assignment))
        self.assertEqual(hashes[0], hashes[1])

    def test_distributions_cover_all_tasks_and_splits(self) -> None:
        frame = self._outcome_frame()
        splits = create_stratified_group_splits(
            frame=frame,
            strata=frame["three_class_outcome"].to_numpy(),
            train_ratio=0.70,
            validation_ratio=0.15,
            test_ratio=0.15,
            seed=42,
            max_attempts=100,
            patient_id_column="subject_id",
            required_classes=[0, 1, 2],
        )
        assignment = build_split_assignment(
            outcomes=frame,
            splits=splits,
            identifier_columns=["subject_id"],
        )
        class_distribution = create_class_distribution(assignment)
        task_distribution = create_task_population_distribution(assignment)
        self.assertEqual(len(class_distribution), 12)
        self.assertEqual(
            set(task_distribution["task"]),
            {
                "exp1_flat_3class",
                "stage1_mortality",
                "stage2_long_stay_shared_with_exp3",
            },
        )
        for split_name in SPLIT_NAMES:
            self.assertEqual(
                int(
                    class_distribution.loc[
                        class_distribution["split"] == split_name,
                        "count",
                    ].sum()
                ),
                int((assignment["split"] == split_name).sum()),
            )

    def test_missing_required_class_is_rejected(self) -> None:
        frame = self._outcome_frame()
        frame = frame.loc[frame["three_class_outcome"] != 2].reset_index(drop=True)
        with self.assertRaises(ValueError):
            create_stratified_group_splits(
                frame=frame,
                strata=frame["three_class_outcome"].to_numpy(),
                train_ratio=0.70,
                validation_ratio=0.15,
                test_ratio=0.15,
                seed=42,
                max_attempts=10,
                patient_id_column="subject_id",
                required_classes=[0, 1, 2],
            )

    def test_duplicate_source_index_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            run_dir = Path(temporary_dir) / "runs" / "test_run"
            outcomes_dir = run_dir / "common" / "outcomes"
            outcomes_dir.mkdir(parents=True)
            frame = self._outcome_frame()
            frame.loc[1, "_source_row_index"] = frame.loc[0, "_source_row_index"]
            frame.to_csv(
                outcomes_dir / "mimic_outcome_assignments.csv.gz",
                index=False,
                compression="gzip",
            )
            atomic_write_json(
                outcomes_dir / "outcome_step_manifest.json",
                {
                    "status": "ready",
                    "dataset_audits": {"mimic": {"status": "ready"}},
                },
            )
            with self.assertRaises(ValueError):
                load_ready_mimic_outcomes(run_dir)

    def test_mimic_ready_allows_split_when_eicu_is_blocked(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            run_dir = Path(temporary_dir) / "runs" / "test_run"
            outcomes_dir = run_dir / "common" / "outcomes"
            outcomes_dir.mkdir(parents=True)
            frame = self._outcome_frame()
            frame.to_csv(
                outcomes_dir / "mimic_outcome_assignments.csv.gz",
                index=False,
                compression="gzip",
            )
            atomic_write_json(
                outcomes_dir / "outcome_step_manifest.json",
                {
                    "status": "blocked",
                    "dataset_audits": {
                        "mimic": {"status": "ready"},
                        "eicu": {"status": "blocked"},
                    },
                },
            )
            loaded, _, _ = load_ready_mimic_outcomes(run_dir)
            self.assertEqual(len(loaded), len(frame))

    def test_run_common_split_step_writes_shared_hash_references(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            run_id = "split_test"
            outcomes_dir = root / "runs" / run_id / "common" / "outcomes"
            outcomes_dir.mkdir(parents=True)
            frame = self._outcome_frame()
            frame.to_csv(
                outcomes_dir / "mimic_outcome_assignments.csv.gz",
                index=False,
                compression="gzip",
            )
            atomic_write_json(
                outcomes_dir / "outcome_step_manifest.json",
                {
                    "status": "ready",
                    "dataset_audits": {"mimic": {"status": "ready"}},
                },
            )
            config = {
                "_project_root": str(root),
                "paths": {"runs_dir": "runs"},
                "data": {
                    "patient_id_column": "subject_id",
                    "identifier_columns": ["subject_id"],
                },
                "split": {
                    "train_ratio": 0.70,
                    "validation_ratio": 0.15,
                    "test_ratio": 0.15,
                    "seed": 42,
                    "max_attempts": 100,
                },
            }
            result = run_common_split_step(config, run_id)
            output_dir = Path(result.output_dir)
            self.assertTrue((output_dir / "split_assignments.csv.gz").is_file())
            self.assertTrue((output_dir / "split_indices.npz").is_file())
            self.assertTrue((output_dir / "class_distribution.csv").is_file())
            self.assertTrue((output_dir / "task_population_distribution.csv").is_file())
            with (output_dir / "split_manifest.json").open("r", encoding="utf-8") as file:
                manifest = json.load(file)
            self.assertEqual(
                set(manifest["task_split_references"].values()),
                {result.split_hash},
            )
            self.assertFalse(manifest["model_training_enabled"])


if __name__ == "__main__":
    unittest.main()
