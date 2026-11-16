from __future__ import annotations

import numpy as np

from los_baseline.study28.adaptation import (
    apply_platt_map,
    fit_platt_map,
    repeated_local_recalibration,
)


def _example(n: int = 600):
    rng = np.random.default_rng(2026)
    y = np.tile(np.array([0, 1, 2], dtype=int), n // 3)
    raw = rng.uniform(0.05, 1.0, size=(len(y), 3))
    raw[np.arange(len(y)), y] += 0.8
    p = raw / raw.sum(axis=1, keepdims=True)
    return y, p


def test_platt_map_returns_normalized_probabilities():
    y, p = _example()
    mapping = fit_platt_map(p[:300], y[:300], ("early_dc", "death", "prolonged"))
    q = apply_platt_map(p[300:], mapping)
    assert q.shape == p[300:].shape
    assert np.isfinite(q).all()
    np.testing.assert_allclose(q.sum(axis=1), 1.0, atol=1e-10)


def test_repeated_local_recalibration_runs_requested_seeds():
    y, p = _example()
    raw, agg, artifacts = repeated_local_recalibration(
        site="demo", y_true=y, probabilities=p,
        class_names=("early_dc", "death", "prolonged"),
        calibration_fraction=0.2, seeds=4, base_seed=2026,
        clip_probability=1e-6, logistic_C=1e6, max_iter=1000,
    )
    assert raw.seed.nunique() == 4
    assert set(raw.model) == {"raw", "recalibrated"}
    assert len(artifacts) == 4
    assert not agg.empty
