from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from los_baseline.artifacts import load_task_artifact
from los_baseline.exp123_data import TaskDataBundle, TaskSplitData
from los_baseline.models import compute_balanced_class_weights
from los_baseline.task_spec import (
    EXP1_FLAT_3CLASS_TASK,
    STAGE1_MORTALITY_TASK,
)
from los_baseline.training_engine import train_and_evaluate_single_model


class TrainingEngineTests(unittest.TestCase):
    @staticmethod
    def _split(
        split_name: str,
        labels: np.ndarray,
        offset: int,
        validation_shift: float = 0.0,
    ) -> TaskSplitData:
        labels = np.asarray(labels, dtype=np.int64)
        x1 = labels.astype(float) + np.linspace(-0.2, 0.2, len(labels))
        x2 = np.arange(len(labels), dtype=float) % 7
        x1 = x1 + validation_shift
        frame = pd.DataFrame({"feature_1": x1, "feature_2": x2})
        frame.loc[::11, "feature_2"] = np.nan
        metadata = pd.DataFrame(
            {
                "source_row_index": np.arange(offset, offset + len(labels)),
                "split": split_name,
                "subject_id": [
                    f"{split_name}_patient_{index}" for index in range(len(labels))
                ],
            }
        )
        return TaskSplitData(
            split_name=split_name,
            raw_features=frame,
            labels=labels,
            metadata=metadata,
        )

    @classmethod
    def _multiclass_bundle(cls) -> TaskDataBundle:
        train_labels = np.tile(np.asarray([0, 1, 2]), 30)
        validation_labels = np.tile(np.asarray([0, 1, 2]), 12)
        test_labels = np.tile(np.asarray([0, 1, 2]), 12)
        return TaskDataBundle(
            task=EXP1_FLAT_3CLASS_TASK,
            feature_names=("feature_1", "feature_2"),
            splits={
                "train": cls._split("train", train_labels, 0),
                "validation": cls._split(
                    "validation", validation_labels, 1000, validation_shift=100.0
                ),
                "internal_test": cls._split(
                    "internal_test", test_labels, 2000
                ),
            },
            shared_split_hash="synthetic_shared_hash",
        )

    @classmethod
    def _binary_bundle(cls) -> TaskDataBundle:
        train_labels = np.tile(np.asarray([0, 1]), 45)
        validation_labels = np.tile(np.asarray([0, 1]), 18)
        test_labels = np.tile(np.asarray([0, 1]), 18)
        return TaskDataBundle(
            task=STAGE1_MORTALITY_TASK,
            feature_names=("feature_1", "feature_2"),
            splits={
                "train": cls._split("train", train_labels, 0),
                "validation": cls._split("validation", validation_labels, 1000),
                "internal_test": cls._split(
                    "internal_test", test_labels, 2000
                ),
            },
            shared_split_hash="synthetic_shared_hash",
        )

    def test_balanced_class_weights_use_train_counts(self) -> None:
        labels = np.asarray([0, 0, 0, 1])
        weights = compute_balanced_class_weights(labels, (0, 1))
        self.assertAlmostEqual(weights[0], 4 / 6)
        self.assertAlmostEqual(weights[1], 2.0)

    def test_binary_engine_selects_validation_threshold_and_reloads(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            output = Path(temporary_dir) / "binary"
            result = train_and_evaluate_single_model(
                task_data=self._binary_bundle(),
                model_name="decision_tree",
                model_config={
                    "parameters": {
                        "max_depth": 3,
                        "min_samples_split": 2,
                    }
                },
                output_dir=output,
                seed=42,
            )
            self.assertIsNotNone(result.validation_decision_threshold)
            self.assertTrue(
                all(value.matches for value in result.prediction_integrity.values())
            )
            predictions = pd.read_csv(
                output
                / "evaluation"
                / "validation"
                / "predictions.csv.gz"
            )
            self.assertIn("positive_probability", predictions.columns)
            self.assertIn(
                "predicted_at_validation_threshold", predictions.columns
            )
            artifact = load_task_artifact(output / "artifact")
            self.assertIn("model_config_requested", artifact.metadata)
            self.assertIn("model_config_effective", artifact.metadata)
            self.assertEqual(
                artifact.metadata["model_config"],
                artifact.metadata["model_config_effective"],
            )
            self.assertEqual(artifact.task_type, "binary")
            self.assertEqual(
                artifact.validation_decision_threshold,
                result.validation_decision_threshold,
            )

    def test_multiclass_engine_uses_argmax_and_train_only_preprocessor(self) -> None:
        bundle = self._multiclass_bundle()
        expected_train_median = float(
            bundle.splits["train"].raw_features["feature_2"].median()
        )
        with tempfile.TemporaryDirectory() as temporary_dir:
            output = Path(temporary_dir) / "multiclass"
            result = train_and_evaluate_single_model(
                task_data=bundle,
                model_name="decision_tree",
                model_config={
                    "parameters": {
                        "max_depth": 4,
                        "min_samples_split": 2,
                    }
                },
                output_dir=output,
                seed=42,
            )
            self.assertIsNone(result.validation_decision_threshold)
            predictions = pd.read_csv(
                output
                / "evaluation"
                / "validation"
                / "predictions.csv.gz"
            )
            probability_columns = [
                "prob_class_0",
                "prob_class_1",
                "prob_class_2",
            ]
            np.testing.assert_allclose(
                predictions[probability_columns].sum(axis=1).to_numpy(),
                1.0,
                atol=1e-7,
            )
            self.assertIn("predicted_class", predictions.columns)
            preprocessor = joblib.load(output / "artifact" / "preprocessor.joblib")
            statistics = preprocessor.named_steps["imputer"].statistics_
            self.assertAlmostEqual(float(statistics[1]), expected_train_median)
            with (output / "artifact_integrity.json").open(
                "r", encoding="utf-8"
            ) as file:
                integrity = json.load(file)
            self.assertTrue(integrity["all_splits_match"])
            self.assertEqual(
                result.split_metrics["validation"]["sample_count"],
                len(bundle.splits["validation"].labels),
            )

    def test_weighted_torch_binary_artifact_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            output = Path(temporary_dir) / "torch_binary"
            result = train_and_evaluate_single_model(
                task_data=self._binary_bundle(),
                model_name="logistic_regression",
                model_config={
                    "epochs": 4,
                    "batch_size": 32,
                    "learning_rate": 0.01,
                    "metric_interval": 1,
                    "device": "cpu",
                },
                output_dir=output,
                seed=42,
                prediction_tolerance=1e-6,
            )
            self.assertTrue(
                all(value.matches for value in result.prediction_integrity.values())
            )
            self.assertTrue((output / "training_history.csv").is_file())
            artifact = load_task_artifact(output / "artifact")
            probabilities = artifact.predict_proba(
                self._binary_bundle().splits["validation"].raw_features
            )
            self.assertEqual(probabilities.ndim, 1)


if __name__ == "__main__":
    unittest.main()


def test_cuda_machine_epsilon_difference_preserves_integrity() -> None:
    from los_baseline.task_spec import STAGE1_MORTALITY_TASK
    from los_baseline.training_engine import compare_prediction_integrity

    memory = np.asarray([0.2, 0.8], dtype=np.float32)
    reloaded = memory.astype(np.float64)
    reloaded[0] += float(np.finfo(np.float32).eps)
    result = compare_prediction_integrity(
        split_name="train",
        memory_probability=memory,
        reloaded_probability=reloaded,
        task=STAGE1_MORTALITY_TASK,
        threshold=0.5,
        model_config={"parameters": {"device": "cuda:1"}},
        integrity_config={
            "probability_tolerance": {"cuda": {"atol": 1e-6, "rtol": 1e-6}},
            "require_decision_equality": True,
            "maximum_decision_mismatches": 0,
        },
    )
    assert result.matches is True
    assert result.maximum_absolute_difference == pytest.approx(np.finfo(np.float32).eps)
    assert result.mismatched_decision_count == 0


def test_integrity_rejects_threshold_decision_change() -> None:
    from los_baseline.task_spec import STAGE1_MORTALITY_TASK
    from los_baseline.training_engine import compare_prediction_integrity

    result = compare_prediction_integrity(
        split_name="validation",
        memory_probability=np.asarray([0.4999999, 0.8]),
        reloaded_probability=np.asarray([0.5000001, 0.8]),
        task=STAGE1_MORTALITY_TASK,
        threshold=0.5,
        model_config={"parameters": {"device": "cuda:0"}},
        integrity_config={
            "probability_tolerance": {"cuda": {"atol": 1e-6, "rtol": 1e-6}},
            "require_decision_equality": True,
            "maximum_decision_mismatches": 0,
        },
    )
    assert result.probabilities_match is True
    assert result.decisions_match is False
    assert result.matches is False
