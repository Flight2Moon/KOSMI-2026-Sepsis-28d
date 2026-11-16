from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from los_baseline.evaluation_multiclass import (
    BootstrapSettings,
    bootstrap_multiclass_confidence_intervals,
    calculate_multiclass_metrics,
    create_confusion_matrix_tables,
    write_multiclass_evaluation_artifacts,
)


class MulticlassEvaluationTests(unittest.TestCase):
    @staticmethod
    def _data() -> tuple[np.ndarray, np.ndarray]:
        labels = np.tile(np.asarray([0, 1, 2], dtype=np.int64), 30)
        probabilities = np.full((len(labels), 3), 0.05, dtype=np.float64)
        probabilities[np.arange(len(labels)), labels] = 0.90
        return labels, probabilities

    def test_metrics_and_confusion_matrix(self) -> None:
        labels, probabilities = self._data()
        metrics = calculate_multiclass_metrics(
            labels,
            probabilities,
            ("A", "B", "C"),
        )
        self.assertAlmostEqual(float(metrics["accuracy"]), 1.0)
        self.assertAlmostEqual(float(metrics["macro_auroc"]), 1.0)
        self.assertAlmostEqual(float(metrics["macro_auprc"]), 1.0)
        raw, normalized = create_confusion_matrix_tables(
            labels,
            probabilities,
            ("A", "B", "C"),
        )
        self.assertEqual(int(np.trace(raw.to_numpy())), len(labels))
        np.testing.assert_allclose(
            normalized.sum(axis=1).to_numpy(),
            np.ones(3),
        )

    def test_bootstrap_intervals_are_created(self) -> None:
        labels, probabilities = self._data()
        intervals = bootstrap_multiclass_confidence_intervals(
            labels,
            probabilities,
            ("A", "B", "C"),
            BootstrapSettings(
                repeats=20,
                confidence_level=0.95,
                seed=42,
                minimum_successful_repeats=10,
            ),
        )
        by_name = {interval.metric: interval for interval in intervals}
        self.assertIn("macro_auroc", by_name)
        self.assertEqual(by_name["macro_auroc"].successful_repeats, 20)
        self.assertAlmostEqual(by_name["macro_auroc"].lower, 1.0)
        self.assertAlmostEqual(by_name["macro_auroc"].upper, 1.0)

    def test_artifact_writer_outputs_required_files(self) -> None:
        labels, probabilities = self._data()
        with tempfile.TemporaryDirectory() as temporary_dir:
            output = Path(temporary_dir)
            write_multiclass_evaluation_artifacts(
                y_true=labels,
                probabilities=probabilities,
                class_names=("A", "B", "C"),
                output_dir=output,
                split_name="validation",
                model_name="test_model",
                calibration_bins=5,
                bootstrap_settings=BootstrapSettings(
                    repeats=10,
                    minimum_successful_repeats=5,
                ),
                save_plots=False,
            )
            required = [
                "multiclass_metrics.json",
                "class_metrics.csv",
                "confusion_matrix_raw.csv",
                "confusion_matrix_normalized.csv",
                "calibration_curve.csv",
                "bootstrap_confidence_intervals.csv",
                "bootstrap_confidence_intervals.json",
                "multiclass_evaluation_manifest.json",
            ]
            for filename in required:
                self.assertTrue((output / filename).is_file(), filename)


if __name__ == "__main__":
    unittest.main()


def test_custom_predicted_classes_control_threshold_based_metrics() -> None:
    from los_baseline.evaluation_multiclass import calculate_multiclass_metrics

    y_true = np.array([0, 1, 2], dtype=np.int64)
    probabilities = np.array(
        [
            [0.60, 0.20, 0.20],
            [0.20, 0.60, 0.20],
            [0.45, 0.10, 0.45],
        ],
        dtype=float,
    )
    argmax_metrics = calculate_multiclass_metrics(
        y_true,
        probabilities,
        ("A", "B", "C"),
    )
    threshold_predictions = np.array([0, 1, 2], dtype=np.int64)
    threshold_metrics = calculate_multiclass_metrics(
        y_true,
        probabilities,
        ("A", "B", "C"),
        predicted_classes=threshold_predictions,
    )
    assert argmax_metrics["accuracy"] == 2 / 3
    assert threshold_metrics["accuracy"] == 1.0
    assert threshold_metrics["multiclass_brier"] == argmax_metrics["multiclass_brier"]
