from __future__ import annotations

import unittest

from los_baseline.exp1_runner import (
    Exp1ModelResult,
    select_exp1_representative_model,
)


class Exp1RunnerTests(unittest.TestCase):
    @staticmethod
    def _result(
        model_name: str,
        auroc: float,
        brier: float,
    ) -> Exp1ModelResult:
        return Exp1ModelResult(
            model_name=model_name,
            output_dir=f"/{model_name}",
            selected_hyperparameters_path=f"/{model_name}/selected.json",
            selected_hyperparameters_sha256="a" * 64,
            validation_macro_auroc=auroc,
            validation_macro_auprc=0.6,
            validation_multiclass_brier=brier,
            validation_macro_f1=0.5,
            training_seconds=1.0,
        )

    def test_representative_model_uses_brier_inside_primary_tie(self) -> None:
        results = [
            self._result("xgboost", 0.800, 0.42),
            self._result("logistic_regression", 0.797, 0.38),
            self._result("random_forest", 0.760, 0.35),
        ]
        selected, table = select_exp1_representative_model(
            results,
            representative_metric="macro_auroc",
            practical_tie_margin=0.005,
            configured_model_order=(
                "logistic_regression",
                "random_forest",
                "xgboost",
            ),
        )
        self.assertEqual(selected.model_name, "logistic_regression")
        selected_rows = [row for row in table if row["selected"]]
        self.assertEqual(len(selected_rows), 1)

    def test_model_outside_primary_tie_cannot_win_on_brier(self) -> None:
        results = [
            self._result("xgboost", 0.800, 0.42),
            self._result("random_forest", 0.780, 0.20),
        ]
        selected, _ = select_exp1_representative_model(
            results,
            representative_metric="macro_auroc",
            practical_tie_margin=0.005,
            configured_model_order=("random_forest", "xgboost"),
        )
        self.assertEqual(selected.model_name, "xgboost")


if __name__ == "__main__":
    unittest.main()
