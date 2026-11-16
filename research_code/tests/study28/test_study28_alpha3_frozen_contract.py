from __future__ import annotations

import os
from pathlib import Path

import pytest

from los_baseline.study28.external_validation import validate_external_contract

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.integration
def test_alpha2_frozen_model_contract_if_available(tmp_path: Path):
    data_root = os.environ.get("STUDY28_DATA_ROOT")
    alpha2_root = os.environ.get("STUDY28_ALPHA2_OUTPUT")
    if not data_root or not alpha2_root:
        pytest.skip("STUDY28_DATA_ROOT/STUDY28_ALPHA2_OUTPUT are not set")
    freeze_manifest = (
        Path(alpha2_root)
        / "03_external_model_freeze"
        / "study28_model_freeze.json"
    )
    protocol, external, registry, feature_contract, frozen = validate_external_contract(
        dataset_root=Path(data_root),
        project_root=PROJECT_ROOT,
        freeze_manifest_path=freeze_manifest,
        audit_output_dir=tmp_path,
    )
    assert frozen.protocol_sha256 == protocol.sha256 == external.frozen_protocol_sha256
    assert frozen.feature_names == feature_contract.no_apsiii
    assert set(external.required_external_sites) == set(registry.required_external_sites)
    assert frozen.locked_search_parameters == {"num_leaves": 15, "n_estimators": 200}
