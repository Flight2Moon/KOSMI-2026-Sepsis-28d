from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from los_baseline.exp123_data import TaskDataBundle, TaskSplitData
from los_baseline.stage2_runner import (
    Stage2ModelResult,
    run_stage2_long_stay,
    select_stage2_reference_model,
)
from los_baseline.task_spec import STAGE2_LONG_STAY_TASK


class Stage2RunnerTests(unittest.TestCase):
    @staticmethod
    def _result(model_name: str, auroc: float, brier: float) -> Stage2ModelResult:
        return Stage2ModelResult(
            model_name=model_name,
            output_dir=f"/{model_name}",
            selected_hyperparameters_path=f"/{model_name}/selected.json",
            selected_hyperparameters_sha256="a" * 64,
            validation_auroc=auroc,
            validation_auprc=0.65,
            validation_brier_score=brier,
            validation_log_loss=0.5,
            validation_balanced_accuracy=0.65,
            validation_f1=0.6,
            validation_sensitivity=0.7,
            validation_specificity=0.6,
            validation_decision_threshold=0.4,
            training_seconds=1.0,
        )

    def test_brier_breaks_auroc_practical_tie(self) -> None:
        results = [
            self._result("xgboost", 0.700, 0.20),
            self._result("logistic_regression", 0.697, 0.17),
            self._result("random_forest", 0.650, 0.15),
        ]
        selected, table = select_stage2_reference_model(
            results,
            reference_metric="auroc",
            practical_tie_margin=0.005,
            configured_model_order=(
                "logistic_regression",
                "random_forest",
                "xgboost",
            ),
        )
        self.assertEqual(selected.model_name, "logistic_regression")
        self.assertEqual(
            sum(bool(row["selected_as_validation_reference"]) for row in table),
            1,
        )

    def test_model_outside_auroc_tie_cannot_win_on_brier(self) -> None:
        results = [
            self._result("xgboost", 0.700, 0.20),
            self._result("random_forest", 0.680, 0.10),
        ]
        selected, _ = select_stage2_reference_model(
            results,
            reference_metric="auroc",
            practical_tie_margin=0.005,
            configured_model_order=("random_forest", "xgboost"),
        )
        self.assertEqual(selected.model_name, "xgboost")

    def test_stage2_rejects_mortality_class_in_metadata(self) -> None:
        features = pd.DataFrame({"x": [0.0, 1.0, 2.0, 3.0]})
        labels = np.array([0, 1, 0, 1], dtype=np.int64)
        train_meta = pd.DataFrame(
            {
                "source_row_index": [0, 1, 2, 3],
                "stage2_eligible": [True, True, True, True],
                "three_class_outcome": [0, 2, 1, 2],
            }
        )
        validation_meta = pd.DataFrame(
            {
                "source_row_index": [10, 11, 12, 13],
                "stage2_eligible": [True, True, True, True],
                "three_class_outcome": [0, 2, 0, 2],
            }
        )
        bundle = TaskDataBundle(
            task=STAGE2_LONG_STAY_TASK,
            feature_names=("x",),
            splits={
                "train": TaskSplitData("train", features.copy(), labels, train_meta),
                "validation": TaskSplitData(
                    "validation", features.copy(), labels, validation_meta
                ),
            },
            shared_split_hash="f" * 64,
        )
        config = {
            "experiments": {
                "exp2_two_stage": {
                    "stage2_long_stay": {
                        "models": ["decision_tree"],
                        "reference_metric": "auroc",
                    }
                }
            }
        }
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "mortality class B"):
                run_stage2_long_stay(
                    task_data=bundle,
                    config=config,
                    run_id="test",
                    run_dir=Path(directory),
                    model_names=["decision_tree"],
                )


if __name__ == "__main__":
    unittest.main()
