from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from los_baseline.heterogeneity import (
    evaluate_model_eligibility_per_stratum,
    rank_models_per_stratum,
)
from los_baseline.stage_heterogeneity import (
    StageHeterogeneityConfig,
    load_frozen_stage_heterogeneity_design,
    load_stage_predictions_with_covariates,
    stage_prediction_path,
)
from los_baseline.task_spec import STAGE1_MORTALITY_TASK
from los_baseline.utils import atomic_write_json, sha256_file


class StageHeterogeneityTests(unittest.TestCase):
    def test_auroc_guard_excludes_low_brier_low_discrimination_model(self) -> None:
        metrics = pd.DataFrame(
            [
                {
                    "task_name": "stage1_mortality",
                    "split": "validation",
                    "stratify_by": "age",
                    "bin_label": "Low",
                    "model_name": "svm",
                    "n": 100,
                    "positive_count": 30,
                    "negative_count": 70,
                    "positive_rate": 0.3,
                    "auroc": 0.61,
                    "brier_mean": 0.12,
                    "log_loss_mean": 0.5,
                    "accuracy_at_0_5": 0.7,
                    "insufficient_data": False,
                },
                {
                    "task_name": "stage1_mortality",
                    "split": "validation",
                    "stratify_by": "age",
                    "bin_label": "Low",
                    "model_name": "xgboost",
                    "n": 100,
                    "positive_count": 30,
                    "negative_count": 70,
                    "positive_rate": 0.3,
                    "auroc": 0.75,
                    "brier_mean": 0.15,
                    "log_loss_mean": 0.48,
                    "accuracy_at_0_5": 0.72,
                    "insufficient_data": False,
                },
                {
                    "task_name": "stage1_mortality",
                    "split": "validation",
                    "stratify_by": "age",
                    "bin_label": "Low",
                    "model_name": "logistic_regression",
                    "n": 100,
                    "positive_count": 30,
                    "negative_count": 70,
                    "positive_rate": 0.3,
                    "auroc": 0.74,
                    "brier_mean": 0.14,
                    "log_loss_mean": 0.47,
                    "accuracy_at_0_5": 0.71,
                    "insufficient_data": False,
                },
            ]
        )
        eligibility = evaluate_model_eligibility_per_stratum(
            metrics,
            auroc_margin_guard=0.02,
        )
        svm = eligibility.loc[eligibility["model_name"] == "svm"].iloc[0]
        self.assertFalse(bool(svm["eligible_by_auroc_guard"]))
        self.assertEqual(svm["eligibility_reason"], "excluded_by_auroc_guard")
        ranking = rank_models_per_stratum(
            metrics,
            tie_margin=0.005,
            auroc_margin_guard=0.02,
        )
        self.assertEqual(ranking.iloc[0]["best_model"], "logistic_regression")
        self.assertIn("svm", ranking.iloc[0]["excluded_models"])


    def test_legacy_ranking_without_guard_keeps_lowest_brier_rule(self) -> None:
        metrics = pd.DataFrame(
            [
                {
                    "threshold_days": 28,
                    "split": "validation",
                    "stratify_by": "age",
                    "bin_label": "Low",
                    "model_name": "svm",
                    "auroc": 0.55,
                    "brier_mean": 0.10,
                    "insufficient_data": False,
                },
                {
                    "threshold_days": 28,
                    "split": "validation",
                    "stratify_by": "age",
                    "bin_label": "Low",
                    "model_name": "xgboost",
                    "auroc": 0.75,
                    "brier_mean": 0.14,
                    "insufficient_data": False,
                },
            ]
        )
        ranking = rank_models_per_stratum(metrics, tie_margin=0.005)
        self.assertEqual(ranking.iloc[0]["best_model"], "svm")
        self.assertEqual(ranking.iloc[0]["selection_rule"], "lowest_brier")

    def test_stage_prediction_loader_requires_common_source_rows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = pd.DataFrame(
                {
                    "LOS": [1.0, 2.0, 3.0, 4.0],
                    "age": [50, 60, 70, 80],
                    "sofa_score": [1, 2, 3, 4],
                    "aps3": [20, 30, 40, 50],
                    "charlson_comorbidity_index": [1, 2, 3, 4],
                }
            )
            dataset_path = root / "mimic.csv"
            dataset.to_csv(dataset_path, index=False)
            task_name = STAGE1_MORTALITY_TASK.task_name
            for model_name, indices in {
                "decision_tree": [0, 1, 2, 3],
                "xgboost": [0, 1, 2, 99],
            }.items():
                path = stage_prediction_path(
                    root, task_name, model_name, "validation"
                )
                path.parent.mkdir(parents=True, exist_ok=True)
                pd.DataFrame(
                    {
                        "source_row_index": indices,
                        "y_true": [0, 1, 0, 1],
                        "positive_probability": [0.1, 0.8, 0.2, 0.7],
                    }
                ).to_csv(path, index=False, compression="gzip")
            with self.assertRaisesRegex(ValueError, "same source_row_index"):
                load_stage_predictions_with_covariates(
                    run_dir=root,
                    task_name=task_name,
                    model_names=("decision_tree", "xgboost"),
                    split="validation",
                    dataset_path=dataset_path,
                )

    def test_stage_config_rejects_negative_auroc_guard(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-negative"):
            StageHeterogeneityConfig(
                run_dir=Path("."),
                task_name=STAGE1_MORTALITY_TASK.task_name,
                auroc_margin_guard=-0.01,
            )

    def test_frozen_design_sha256_is_verified(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "frozen_design.json"
            atomic_write_json(
                path,
                {
                    "design_status": "frozen",
                    "design_type": "stage_model_subgroup_heterogeneity",
                    "task_name": "stage1_mortality",
                },
            )
            path.with_name(path.name + ".sha256").write_text(
                sha256_file(path) + "\n", encoding="utf-8"
            )
            loaded = load_frozen_stage_heterogeneity_design(path)
            self.assertEqual(loaded["task_name"], "stage1_mortality")
            payload = json.loads(path.read_text())
            payload["task_name"] = "stage2_long_stay"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                load_frozen_stage_heterogeneity_design(path)


if __name__ == "__main__":
    unittest.main()
