from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from los_baseline.study28.contract import FeatureContract
from los_baseline.study28.development import split_mimic_temporal
from los_baseline.study28.feature_sets import resolve_feature_set
from los_baseline.study28.protocol import load_study28_protocol

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_alpha2_protocol_matches_professor_nested_cv_and_exact_temporal_labels():
    protocol = load_study28_protocol(PROJECT_ROOT / "configs" / "study28_final.json")
    assert protocol.nested_cv.outer_folds == 5
    assert protocol.nested_cv.inner_folds == 3
    assert protocol.nested_cv.seed == 2026
    assert protocol.nested_cv.selection_metric == "macro_auroc"
    assert protocol.nested_cv.selection_feature_set == "FS_FULL"
    assert protocol.nested_cv.hyperparameter_grid == (
        {"num_leaves": 15, "n_estimators": 200},
        {"num_leaves": 15, "n_estimators": 400},
        {"num_leaves": 31, "n_estimators": 200},
        {"num_leaves": 31, "n_estimators": 400},
    )
    assert protocol.primary_feature_set == "FS_NOAPS"
    assert protocol.primary_model.expected_locked_search_parameters == {
        "num_leaves": 15,
        "n_estimators": 200,
    }
    assert protocol.temporal_split.development_groups == (
        "2008 - 2010", "2011 - 2013", "2014 - 2016"
    )
    assert protocol.temporal_split.evaluation_groups == (
        "2017 - 2019", "2020 - 2022"
    )
    assert protocol.temporal_split.expected_development_n == 17009
    assert protocol.temporal_split.expected_evaluation_n == 5732


def test_temporal_split_rejects_unassigned_year_group():
    protocol = load_study28_protocol(PROJECT_ROOT / "configs" / "study28_final.json")
    frame = pd.DataFrame(
        {
            "anchor_year_group": ["2008 - 2010", "2017 - 2019", "2099 - 2100"],
            "y": [0, 1, 2],
        }
    )
    with pytest.raises(ValueError, match="not assigned"):
        split_mimic_temporal(frame, protocol)


def test_final_feature_sets_preserve_professor_order(tmp_path: Path):
    # The real feature list is not shipped with the code package, so reproduce
    # the contract from the frozen feature-set file for this unit test.
    import json
    import hashlib

    payload = json.loads((PROJECT_ROOT / "configs" / "feature_sets.json").read_text())
    full = tuple(payload["sets"]["FS_FULL"]["columns"])
    contract = FeatureContract(columns=full, sha256="x" * 64, source_path="test")
    full_resolved = resolve_feature_set(
        "FS_FULL", feature_sets_path=PROJECT_ROOT / "configs" / "feature_sets.json", contract=contract
    )
    noaps = resolve_feature_set(
        "FS_NOAPS", feature_sets_path=PROJECT_ROOT / "configs" / "feature_sets.json", contract=contract
    )
    assert full_resolved == full
    assert noaps == tuple(value for value in full if value != "apsiii")
    assert len(noaps) == 56
