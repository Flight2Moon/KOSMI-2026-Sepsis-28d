from pathlib import Path
import json

import pandas as pd
import pytest

from los_baseline.study28.contract import (
    PROHIBITED_MODEL_COLUMNS,
    LABEL_TO_Y,
    load_feature_contract,
)
from los_baseline.study28.audit import assert_required_external_sites
from los_baseline.study28.datasets import load_dataset_registry

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_feature_registry_matches_professor_order_contract():
    feature_sets = json.loads((PROJECT_ROOT / "configs" / "feature_sets.json").read_text())
    full = feature_sets["sets"]["FS_FULL"]["columns"]
    assert len(full) == 57
    assert full[10:13] == ["cci", "apsiii", "gcs_score"]
    assert full[26:32] == [
        "mbp_min", "mbp_max", "respiratory_rate_min", "respiratory_rate_max",
        "temperature_min", "temperature_max",
    ]
    assert full[44:54] == [
        "chloride_min", "chloride_max", "sodium_min", "sodium_max",
        "bicarbonate_min", "bicarbonate_max", "potassium_min", "potassium_max",
        "glucose_min", "glucose_max",
    ]
    assert not (set(full) & PROHIBITED_MODEL_COLUMNS)


def test_registry_requires_all_external_sites():
    registry = load_dataset_registry(PROJECT_ROOT / "configs" / "dataset_registry_28d.json")
    assert registry.required_external_sites == (
        "eicu_alt", "eicu_primary", "sicdb", "nwicu"
    )
    assert set(registry.required_external_sites).issubset(registry.datasets)


def test_missing_external_site_is_blocking():
    with pytest.raises(RuntimeError, match="nwicu"):
        assert_required_external_sites(
            ["eicu_alt", "eicu_primary", "sicdb"],
            ["eicu_alt", "eicu_primary", "sicdb", "nwicu"],
        )


def test_label_mapping_is_frozen():
    assert LABEL_TO_Y == {"early_dc": 0, "death": 1, "prolonged": 2}
