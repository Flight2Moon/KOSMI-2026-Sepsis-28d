from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from los_baseline.exp123_data import build_task_data_bundle
from los_baseline.task_spec import (
    EXP1_FLAT_3CLASS_TASK,
    STAGE1_MORTALITY_TASK,
    STAGE2_LONG_STAY_TASK,
)


class Exp123TaskDataTests(unittest.TestCase):
    @staticmethod
    def _frames() -> tuple[pd.DataFrame, pd.DataFrame]:
        rows = []
        assignment = []
        split_names = ["train"] * 6 + ["validation"] * 6 + ["internal_test"] * 6
        classes = [0, 1, 2, 0, 1, 2] * 3
        for index, (split_name, class_code) in enumerate(zip(split_names, classes)):
            rows.append(
                {
                    "_source_row_index": index,
                    "feature_1": float(index),
                    "feature_2": float(class_code),
                }
            )
            assignment.append(
                {
                    "source_row_index": index,
                    "split": split_name,
                    "subject_id": f"patient_{index}",
                    "three_class_outcome": class_code,
                    "stage1_mortality_label": int(class_code == 1),
                    "stage2_eligible": bool(class_code != 1),
                    "stage2_long_stay_label": (
                        np.nan if class_code == 1 else int(class_code == 2)
                    ),
                }
            )
        return pd.DataFrame(rows), pd.DataFrame(assignment)

    def test_all_tasks_share_source_assignments(self) -> None:
        features, assignment = self._frames()
        bundles = [
            build_task_data_bundle(
                feature_frame=features,
                split_assignment=assignment,
                task=task,
                feature_names=["feature_1", "feature_2"],
                shared_split_hash="same_hash",
            )
            for task in (
                EXP1_FLAT_3CLASS_TASK,
                STAGE1_MORTALITY_TASK,
                STAGE2_LONG_STAY_TASK,
            )
        ]
        self.assertTrue(all(bundle.shared_split_hash == "same_hash" for bundle in bundles))
        for split_name in ("train", "validation", "internal_test"):
            exp1_sources = set(
                bundles[0].splits[split_name].metadata["source_row_index"]
            )
            stage1_sources = set(
                bundles[1].splits[split_name].metadata["source_row_index"]
            )
            stage2_sources = set(
                bundles[2].splits[split_name].metadata["source_row_index"]
            )
            self.assertEqual(exp1_sources, stage1_sources)
            expected_survivors = set(
                assignment.loc[
                    (assignment["split"] == split_name)
                    & assignment["stage2_eligible"],
                    "source_row_index",
                ]
            )
            self.assertEqual(stage2_sources, expected_survivors)

    def test_stage2_labels_are_ground_truth_survivor_labels(self) -> None:
        features, assignment = self._frames()
        bundle = build_task_data_bundle(
            feature_frame=features,
            split_assignment=assignment,
            task=STAGE2_LONG_STAY_TASK,
            feature_names=["feature_1", "feature_2"],
        )
        for split in bundle.splits.values():
            self.assertEqual(set(split.labels.tolist()), {0, 1})
            self.assertNotIn(1, set(
                split.metadata["three_class_outcome"].astype(int).tolist()
            ))


if __name__ == "__main__":
    unittest.main()
