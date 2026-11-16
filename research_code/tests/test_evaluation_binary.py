from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from los_baseline.evaluation_binary import (
    BinaryBootstrapSettings,
    probabilities_from_binary_prediction_frame,
    write_binary_evaluation_artifacts,
)


class BinaryEvaluationTests(unittest.TestCase):
    def test_probability_parser(self) -> None:
        frame = pd.DataFrame({"positive_probability": [0.1, 0.8]})
        values = probabilities_from_binary_prediction_frame(frame)
        np.testing.assert_allclose(values, np.array([0.1, 0.8]))

    def test_writes_threshold_free_and_threshold_views(self) -> None:
        labels = np.array([0, 0, 0, 1, 1, 1], dtype=np.int64)
        probabilities = np.array([0.1, 0.2, 0.4, 0.55, 0.8, 0.9])
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            payload = write_binary_evaluation_artifacts(
                y_true=labels,
                probabilities=probabilities,
                selected_threshold=0.5,
                default_threshold=0.5,
                output_dir=output_dir,
                split_name="validation",
                model_name="test_model",
                positive_class_name="B_died",
                calibration_bins=3,
                bootstrap_settings=BinaryBootstrapSettings(repeats=10, seed=42),
                save_plots=False,
            )
            self.assertIn("default_0_5", payload["threshold_views"])
            self.assertIn("validation_selected", payload["threshold_views"])
            for filename in [
                "binary_metrics.json",
                "threshold_metrics.csv",
                "confusion_matrix_raw.csv",
                "confusion_matrix_normalized.csv",
                "roc_curve.csv",
                "precision_recall_curve.csv",
                "calibration_curve.csv",
                "bootstrap_confidence_intervals.csv",
                "binary_evaluation_manifest.json",
            ]:
                self.assertTrue((output_dir / filename).is_file(), filename)


if __name__ == "__main__":
    unittest.main()
