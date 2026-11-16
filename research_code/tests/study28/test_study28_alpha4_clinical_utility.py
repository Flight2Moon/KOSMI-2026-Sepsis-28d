from __future__ import annotations

import numpy as np
import pandas as pd

from los_baseline.study28.clinical_utility import (
    decision_curve_rows,
    dersimonian_laird_pool,
    hospital_level_heterogeneity,
    subgroup_performance,
)


def _probabilities(y: np.ndarray) -> np.ndarray:
    p = np.full((len(y), 3), 0.15, dtype=float)
    p[np.arange(len(y)), y] = 0.70
    return p / p.sum(axis=1, keepdims=True)


def test_decision_curve_treat_none_is_zero():
    y = np.array([0, 1, 2] * 20, dtype=int)
    p = _probabilities(y)
    dca = decision_curve_rows(
        setting="demo", y_true=y, model_probabilities={"model": p},
        classes=("death", "prolonged"), thresholds=np.array([0.1, 0.2]),
    )
    assert (dca.nb_treat_none == 0.0).all()
    assert len(dca) == 4


def test_subgroups_are_marked_reference_or_exploratory():
    n = 300
    y = np.tile(np.array([0, 1, 2], dtype=int), n // 3)
    frame = pd.DataFrame({
        "age": np.tile([50, 50, 50, 70, 70, 70, 85, 85, 85, 60, 75, 82], 25),
        "male": np.tile([0, 1, 0, 1, 0, 1], 50),
        "invasive_mechanical_ventilation": np.tile([0, 0, 1, 1, 0, 1], 50),
        "admission_elective": np.tile([1, 0, 0, 1, 0, 0], 50),
    })
    out = subgroup_performance(
        site="demo", frame=frame, y_true=y, probabilities=_probabilities(y),
        definitions=("age<65", "age65-79", "age>=80", "male", "female"),
        minimum_n=20, reference=False,
    )
    assert set(out.analysis_role) == {"exploratory_extension"}


def test_random_effects_pool_returns_finite_summary():
    out = dersimonian_laird_pool(
        np.array([0.70, 0.75, 0.80]), np.array([0.05, 0.04, 0.06])
    )
    assert out["k"] == 3
    assert np.isfinite(out["pooled_auc"])
    assert out["tau2"] >= 0


def test_hospital_heterogeneity_filters_small_hospitals():
    y = np.array(([0] * 40 + [1] * 10 + [2] * 10) * 3, dtype=int)
    frame = pd.DataFrame({"hospitalid": np.repeat([1, 2, 3], 60)})
    p = _probabilities(y)
    levels, summary = hospital_level_heterogeneity(
        frame=frame, y_true=y, probabilities=p,
        hospital_column="hospitalid", classes=("death", "prolonged"),
        minimum_hospital_n=50, minimum_events=5,
    )
    assert len(levels) == 6
    assert set(summary["class"]) == {"death", "prolonged"}
