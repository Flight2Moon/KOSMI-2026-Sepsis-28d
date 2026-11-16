from __future__ import annotations

import unittest

from los_baseline.stage1_runner import (
    Stage1ModelResult,
    select_stage1_reference_model,
)


class Stage1RunnerTests(unittest.TestCase):
    @staticmethod
    def _result(model_name: str, auprc: float, brier: float) -> Stage1ModelResult:
        return Stage1ModelResult(
            model_name=model_name,
            output_dir=f"/{model_name}",
            selected_hyperparameters_path=f"/{model_name}/selected.json",
            selected_hyperparameters_sha256="a" * 64,
            validation_auroc=0.7,
            validation_auprc=auprc,
            validation_brier_score=brier,
            validation_log_loss=0.5,
            validation_balanced_accuracy=0.65,
            validation_f1=0.6,
            validation_sensitivity=0.7,
            validation_specificity=0.6,
            validation_decision_threshold=0.4,
            training_seconds=1.0,
        )

    def test_brier_breaks_auprc_practical_tie(self) -> None:
        results = [
            self._result("xgboost", 0.700, 0.20),
            self._result("logistic_regression", 0.697, 0.17),
            self._result("random_forest", 0.650, 0.15),
        ]
        selected, table = select_stage1_reference_model(
            results,
            reference_metric="auprc",
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

    def test_model_outside_tie_cannot_win_on_brier(self) -> None:
        results = [
            self._result("xgboost", 0.700, 0.20),
            self._result("random_forest", 0.680, 0.10),
        ]
        selected, _ = select_stage1_reference_model(
            results,
            reference_metric="auprc",
            practical_tie_margin=0.005,
            configured_model_order=("random_forest", "xgboost"),
        )
        self.assertEqual(selected.model_name, "xgboost")


if __name__ == "__main__":
    unittest.main()
