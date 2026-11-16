from __future__ import annotations

import numpy as np

from los_baseline.metrics import select_youden_threshold


def test_youden_threshold_is_placed_between_observed_probabilities() -> None:
    labels = np.asarray([0, 0, 1, 1], dtype=np.int32)
    probability = np.asarray([0.10, 0.20, 0.70, 0.90], dtype=np.float64)

    threshold = select_youden_threshold(labels, probability)

    assert 0.20 < threshold < 0.70
    assert np.array_equal(
        (probability >= threshold).astype(np.int64),
        np.asarray([0, 0, 1, 1], dtype=np.int64),
    )


def test_threshold_decisions_survive_sub_ulp_reload_noise() -> None:
    labels = np.asarray([0, 0, 1, 1], dtype=np.int32)
    probability = np.asarray([0.10, 0.20, 0.70, 0.90], dtype=np.float64)
    threshold = select_youden_threshold(labels, probability)
    reloaded = probability.copy()
    reloaded[2] = np.nextafter(reloaded[2], 0.0)

    assert np.array_equal(
        (probability >= threshold).astype(np.int64),
        (reloaded >= threshold).astype(np.int64),
    )
