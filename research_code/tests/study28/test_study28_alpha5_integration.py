from __future__ import annotations

import os
from pathlib import Path

import pytest

from los_baseline.study28.robustness import validate_alpha5_contract

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.integration
def test_alpha5_contract_against_real_outputs_if_available(tmp_path: Path):
    data = os.environ.get("STUDY28_DATA_ROOT")
    a2 = os.environ.get("STUDY28_ALPHA2_OUTPUT")
    a3 = os.environ.get("STUDY28_ALPHA3_OUTPUT")
    a4 = os.environ.get("STUDY28_ALPHA4_OUTPUT")
    if not all([data, a2, a3, a4]):
        pytest.skip("Study28 Alpha 5 real-data environment variables are not set")
    c = validate_alpha5_contract(
        dataset_root=Path(data), project_root=PROJECT_ROOT,
        alpha2_output_dir=Path(a2), alpha3_output_dir=Path(a3), alpha4_output_dir=Path(a4),
        audit_output_dir=tmp_path,
    )
    assert c.alpha4_manifest["schema_version"] == "study28_alpha4_sensitivity_v1"
    assert c.alpha4.frozen.model_sha256 == c.alpha4_manifest["alpha2_frozen_model_sha256"]
    from los_baseline.study28.feature_sets import resolve_feature_set
    features = resolve_feature_set("FS_NOAPS_NO_PREICU", feature_sets_path=PROJECT_ROOT / "configs" / "feature_sets.json", contract=c.alpha4.feature_contract)
    assert len(features) == 55
    assert "apsiii" not in features and "pre_icu_los_d" not in features
