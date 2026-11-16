from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from los_baseline.hpo.fixed_cv import build_fixed_group_folds, save_fixed_folds


def test_fixed_cv_manifest_records_outcome_horizon(tmp_path: Path):
    labels = np.asarray([0, 1] * 20, dtype=np.int64)
    groups = np.asarray([f"p{i}" for i in range(40)])
    folds = build_fixed_group_folds(labels=labels, groups=groups, n_splits=2, seeds=[42])
    save_fixed_folds(
        tmp_path,
        folds=folds,
        seeds=[42],
        n_splits=2,
        sample_count=len(labels),
        outcome_horizon_days=14,
    )
    manifest = json.loads((tmp_path / "fixed_cv_manifest.json").read_text())
    assert manifest["outcome_horizon_days"] == 14.0
