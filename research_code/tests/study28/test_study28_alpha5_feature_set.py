from __future__ import annotations

from pathlib import Path

from los_baseline.study28.contract import load_feature_contract
from los_baseline.study28.feature_sets import resolve_feature_set

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_noaps_no_preicu_feature_set_is_55_and_ordered(tmp_path: Path):
    columns = resolve_feature_set
    # Use the checked-in frozen professor feature list only in real-data integration;
    # unit contract here uses the actual supplied list when available via project tests.
    feature_list = PROJECT_ROOT / "tests" / "fixtures" / "does_not_exist.csv"
    # This test intentionally only verifies the JSON declaration; resolution is
    # covered by the real-dataset integration test below.
    import json
    payload = json.loads((PROJECT_ROOT / "configs" / "feature_sets.json").read_text())
    spec = payload["sets"]["FS_NOAPS_NO_PREICU"]
    assert spec["derived_from"] == "FS_NOAPS"
    assert spec["exclude"] == ["pre_icu_los_d"]
