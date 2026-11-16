from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class Study28ReportingProtocol:
    schema_version: str
    study_id: str
    source_phases: tuple[str, ...]
    required_external_sites: tuple[str, ...]
    reporting_taxonomy: dict[str, str]
    rounding: dict[str, int]
    figures: dict[str, Any]
    privacy_guard: dict[str, Any]
    finalization_requirements: dict[str, Any]
    methodological_boundaries: dict[str, str]
    source_path: str
    sha256: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_reporting_protocol(path: Path) -> Study28ReportingProtocol:
    path = Path(path).expanduser().resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise TypeError("Study28 Alpha 6 reporting protocol must be a JSON object.")
    required = tuple(str(v) for v in payload.get("required_external_sites", []))
    if set(required) != {"eicu_alt", "eicu_primary", "sicdb", "nwicu"}:
        raise ValueError("Alpha 6 must require all four mandatory external cohorts.")
    source_phases = tuple(str(v) for v in payload.get("source_phases", []))
    if source_phases != ("V9_ALPHA2", "V9_ALPHA3", "V9_ALPHA4", "V9_ALPHA5"):
        raise ValueError("Alpha 6 source phases must be V9 Alpha 2 through Alpha 5 in order.")
    taxonomy = {str(k): str(v) for k, v in dict(payload.get("reporting_taxonomy", {})).items()}
    required_roles = {
        "primary_development", "supportive_temporal", "primary_raw_external",
        "secondary_calibration", "sensitivity_reverse", "sensitivity_recalibration",
        "clinical_utility", "subgroup_reference", "exploratory_subgroup", "robustness",
        "interpretability", "secondary_outcome", "professor_method_reproduction", "not_executable",
    }
    missing_roles = sorted(required_roles - set(taxonomy))
    if missing_roles:
        raise ValueError(f"Reporting taxonomy missing roles: {missing_roles}")
    privacy = dict(payload.get("privacy_guard", {}))
    if not bool(privacy.get("forbid_patient_level_artifacts", False)):
        raise ValueError("Alpha 6 must forbid patient-level reporting artifacts.")
    finalization = dict(payload.get("finalization_requirements", {}))
    if not all(bool(finalization.get(key, False)) for key in (
        "frozen_primary_model_must_match_alpha3_alpha5",
        "alpha3_all_required_external_complete",
        "alpha3_raw_external_only",
        "alpha4_must_not_modify_raw_external",
        "alpha4_must_not_modify_primary_model",
        "alpha5_must_not_modify_primary_model",
    )):
        raise ValueError("Alpha 6 finalization guards may not be disabled.")
    return Study28ReportingProtocol(
        schema_version=str(payload.get("schema_version", "study28_manuscript_reporting_v1")),
        study_id=str(payload.get("study_id", "sepsis3_28d_three_class_disposition")),
        source_phases=source_phases,
        required_external_sites=required,
        reporting_taxonomy=taxonomy,
        rounding={str(k): int(v) for k, v in dict(payload.get("rounding", {})).items()},
        figures=dict(payload.get("figures", {})),
        privacy_guard=privacy,
        finalization_requirements=finalization,
        methodological_boundaries={str(k): str(v) for k, v in dict(payload.get("methodological_boundaries", {})).items()},
        source_path=str(path),
        sha256=sha256_file(path),
    )
