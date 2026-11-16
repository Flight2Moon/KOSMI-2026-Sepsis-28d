from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from los_baseline.study28.manuscript_reporting import (
    _collect_from_release_snapshots,
    _privacy_check,
    _validate_finalization,
    run_alpha6,
)
from los_baseline.study28.reporting_protocol import load_reporting_protocol


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_alpha6_protocol_locks_evidence_taxonomy_and_four_external_sites():
    protocol = load_reporting_protocol(PROJECT_ROOT / "configs" / "study28_manuscript_reporting.json")
    assert set(protocol.required_external_sites) == {"eicu_alt", "eicu_primary", "sicdb", "nwicu"}
    assert protocol.source_phases == ("V9_ALPHA2", "V9_ALPHA3", "V9_ALPHA4", "V9_ALPHA5")
    assert "primary_raw_external" in protocol.reporting_taxonomy
    assert protocol.privacy_guard["forbid_patient_level_artifacts"] is True


def test_alpha6_snapshot_run_preserves_raw_external_values(tmp_path: Path):
    result = run_alpha6(project_root=PROJECT_ROOT, output_dir=tmp_path, use_release_verification_snapshots=True)
    assert result.finalization_passed is True
    main = pd.read_csv(tmp_path / "tables" / "main" / "table_2_raw_external_validation.csv")
    assert set(main["site"]) == {"eicu_alt", "eicu_primary", "sicdb", "nwicu"}
    assert set(main["analysis_role"]) == {"primary_raw_external"}
    observed = dict(zip(main["site"], main["macro_auroc"]))
    assert observed["eicu_alt"] == pytest.approx(0.7518519933977491, abs=1e-15)
    assert observed["eicu_primary"] == pytest.approx(0.7523371625203105, abs=1e-15)
    assert observed["sicdb"] == pytest.approx(0.7346593510576019, abs=1e-15)
    assert observed["nwicu"] == pytest.approx(0.7959274082114818, abs=1e-15)


def test_alpha6_registry_keeps_sensitivity_out_of_primary_external(tmp_path: Path):
    run_alpha6(project_root=PROJECT_ROOT, output_dir=tmp_path, use_release_verification_snapshots=True)
    registry = pd.read_csv(tmp_path / "analysis_registry.csv")
    raw = registry.loc[registry["analysis"].eq("Raw external validation: 4 cohorts")].iloc[0]
    recal = registry.loc[registry["analysis"].eq("Local recalibration")].iloc[0]
    assert raw["analysis_role"] == "primary_raw_external"
    assert recal["analysis_role"] == "sensitivity_recalibration"
    assert "Main Table 2" in raw["report_location"]
    assert "Supplementary" in recal["report_location"]


def test_alpha6_home_discharge_remains_not_executable_and_not_inferred(tmp_path: Path):
    run_alpha6(project_root=PROJECT_ROOT, output_dir=tmp_path, use_release_verification_snapshots=True)
    table = pd.read_csv(tmp_path / "tables" / "supplementary" / "table_s13_unavailable_analysis_registry.csv")
    assert table.loc[0, "analysis_role"] == "not_executable"
    assert bool(table.loc[0, "labels_inferred"]) is False


def test_alpha6_finalization_blocks_missing_mandatory_external_site():
    protocol = load_reporting_protocol(PROJECT_ROOT / "configs" / "study28_manuscript_reporting.json")
    bundle = _collect_from_release_snapshots(PROJECT_ROOT)
    bundle["alpha3_manifest"] = dict(bundle["alpha3_manifest"])
    bundle["alpha3_manifest"]["completed_external_sites"] = ["eicu_alt", "eicu_primary", "sicdb"]
    with pytest.raises(RuntimeError, match="all_required_external_complete"):
        _validate_finalization(bundle, protocol)


def test_alpha6_privacy_guard_blocks_identifier_columns(tmp_path: Path):
    protocol = load_reporting_protocol(PROJECT_ROOT / "configs" / "study28_manuscript_reporting.json")
    pd.DataFrame({"subject_id":[1], "macro_auroc":[0.8]}).to_csv(tmp_path / "bad_table.csv", index=False)
    with pytest.raises(RuntimeError, match="privacy guard failed"):
        _privacy_check(tmp_path, protocol)


def test_alpha6_manifest_records_no_patient_outputs(tmp_path: Path):
    run_alpha6(project_root=PROJECT_ROOT, output_dir=tmp_path, use_release_verification_snapshots=True)
    manifest = json.loads((tmp_path / "study28_alpha6_manuscript_evidence_manifest.json").read_text())
    assert manifest["raw_external_results_replaced"] is False
    assert manifest["patient_level_outputs_created"] is False
    assert manifest["privacy_guard"]["passed"] is True
    assert all(manifest["finalization"]["checks"].values())
