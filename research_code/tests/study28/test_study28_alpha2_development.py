from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from los_baseline.study28.development import run_nested_cv
from los_baseline.study28.protocol import NestedCVProtocol, load_study28_protocol

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _synthetic_three_class(n_per_class: int = 30) -> pd.DataFrame:
    rng = np.random.default_rng(2026)
    y = np.repeat(np.arange(3), n_per_class)
    signal = y.astype(float) + rng.normal(0.0, 0.25, size=len(y))
    frame = pd.DataFrame(
        {
            "f1": signal,
            "f2": rng.normal(size=len(y)),
            "f3": rng.normal(size=len(y)),
            "y": y,
            "label28": np.array(["early_dc", "death", "prolonged"])[y],
            "subject_id": np.arange(len(y)),
        }
    )
    frame.loc[::11, "f2"] = np.nan
    return frame


def test_nested_cv_generates_complete_oof_and_deterministic_lock(tmp_path: Path):
    base = load_study28_protocol(PROJECT_ROOT / "configs" / "study28_final.json")
    # Tiny grid/fold contract for a fast unit test; the real 5x3 professor grid
    # is verified separately and exercised by the real-data integration run.
    nested = NestedCVProtocol(
        outer_folds=3,
        inner_folds=2,
        seed=2026,
        selection_metric="macro_auroc",
        selection_feature_set="FS_FULL",
        hyperparameter_grid=({"num_leaves": 7, "n_estimators": 20},),
        enforce_expected_locked_parameters=False,
    )
    protocol = replace(base, nested_cv=nested)
    frame = _synthetic_three_class()
    result = run_nested_cv(
        mimic=frame,
        feature_names=["f1", "f2", "f3"],
        protocol=protocol,
        output_dir=tmp_path,
    )
    assert result.locked_search_parameters == {"num_leaves": 7, "n_estimators": 20}
    oof = pd.read_csv(result.oof_predictions_path)
    assert len(oof) == len(frame)
    assert oof[["p_early_dc", "p_death", "p_prolonged"]].notna().all().all()
    assert np.allclose(
        oof[["p_early_dc", "p_death", "p_prolonged"]].sum(axis=1), 1.0
    )
