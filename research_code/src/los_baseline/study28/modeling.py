from __future__ import annotations

import os
from typing import Any, Mapping

import numpy as np


def resolve_lightgbm_threads(default: int = -1) -> int:
    raw = os.environ.get("EXP123_ASSIGNED_CPU_THREADS", "").strip()
    if not raw:
        return int(default)
    try:
        return max(1, int(raw))
    except ValueError:
        return int(default)


def build_lightgbm_multiclass(
    parameters: Mapping[str, Any],
    *,
    n_jobs: int | None = None,
):
    try:
        from lightgbm import LGBMClassifier
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "LightGBM is required for Study28 Alpha 2. Install requirements.txt first."
        ) from exc

    params = dict(parameters)
    params["objective"] = "multiclass"
    params["num_class"] = 3
    params.setdefault("learning_rate", 0.05)
    params.setdefault("min_child_samples", 50)
    params.setdefault("random_state", 2026)
    params.setdefault("verbose", -1)
    params.setdefault("n_jobs", resolve_lightgbm_threads() if n_jobs is None else int(n_jobs))
    return LGBMClassifier(**params)


def predict_three_class_probabilities(model: Any, X: Any) -> np.ndarray:
    probabilities = np.asarray(model.predict_proba(X), dtype=np.float64)
    classes = np.asarray(getattr(model, "classes_", []), dtype=np.int64)
    if probabilities.ndim != 2:
        raise ValueError(f"LightGBM predict_proba returned invalid shape: {probabilities.shape}")
    if tuple(classes.tolist()) != (0, 1, 2):
        raise ValueError(f"Study28 model classes must be exactly [0,1,2], got {classes.tolist()}")
    if probabilities.shape[1] != 3:
        raise ValueError(f"Study28 requires three probability columns, got {probabilities.shape}")
    if not np.isfinite(probabilities).all():
        raise ValueError("LightGBM produced non-finite probabilities.")
    row_sums = probabilities.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-6, rtol=1e-6):
        raise ValueError("LightGBM probabilities do not sum to one.")
    return probabilities
