from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class Study28RobustnessProtocol:
    schema_version: str
    study_id: str
    required_alpha3_external_complete: bool
    required_alpha4_complete: bool
    interpretability: dict[str, Any]
    pre_icu_los_sensitivity: dict[str, Any]
    feature_availability_sensitivity: dict[str, Any]
    missingness_robustness: dict[str, Any]
    hfd28_secondary: dict[str, Any]
    home_discharge_sensitivity: dict[str, Any]
    methodological_notes: dict[str, str]
    source_path: str
    sha256: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_robustness_protocol(path: Path) -> Study28RobustnessProtocol:
    path = Path(path).expanduser().resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise TypeError("Study28 Alpha 5 robustness protocol must be a JSON object.")
    interpretability = dict(payload.get("interpretability", {}))
    pre_icu = dict(payload.get("pre_icu_los_sensitivity", {}))
    availability = dict(payload.get("feature_availability_sensitivity", {}))
    missingness = dict(payload.get("missingness_robustness", {}))
    hfd = dict(payload.get("hfd28_secondary", {}))
    home = dict(payload.get("home_discharge_sensitivity", {}))

    if interpretability.get("feature_set") != "FS_NOAPS":
        raise ValueError("Alpha 5 SHAP must explain the frozen primary FS_NOAPS model.")
    if int(interpretability.get("sample_size", 0)) <= 0:
        raise ValueError("interpretability.sample_size must be positive.")
    if pre_icu.get("base_feature_set") != "FS_NOAPS":
        raise ValueError("pre_icu_los sensitivity must start from FS_NOAPS.")
    if tuple(pre_icu.get("excluded_features", [])) != ("pre_icu_los_d",):
        raise ValueError("Alpha 5 pre_icu_los sensitivity must remove only pre_icu_los_d.")
    if availability.get("base_feature_set") != "FS_NOAPS":
        raise ValueError("Feature-availability sensitivity must start from FS_NOAPS.")
    if set(availability.get("sites", [])) != {"eicu_alt", "eicu_primary", "sicdb", "nwicu"}:
        raise ValueError("Feature-availability sensitivity must cover all four external cohorts.")
    if set(missingness.get("sites", [])) != {"mimic_temporal", "eicu_alt", "eicu_primary", "sicdb", "nwicu"}:
        raise ValueError("Missingness robustness must include MIMIC temporal plus all four external cohorts.")
    bins = tuple(float(v) for v in missingness.get("bins", []))
    labels = tuple(str(v) for v in missingness.get("labels", []))
    if len(bins) != len(labels) + 1 or sorted(bins) != list(bins):
        raise ValueError("Missingness bins/labels are invalid.")
    if hfd.get("feature_set") != "FS_NOAPS":
        raise ValueError("HFD28 secondary model must use FS_NOAPS.")
    href = dict(hfd.get("professor_reference", {}))
    if href.get("enabled") and href.get("feature_set") != "FS_FULL":
        raise ValueError("Professor-reference HFD28 reproduction must use FS_FULL.")
    if home.get("required_column") != "discharge_location":
        raise ValueError("Home-discharge sensitivity requires discharge_location.")
    if not bool(home.get("do_not_infer_from_other_columns", False)):
        raise ValueError("Home-discharge labels must never be inferred from other final-dataset columns.")

    return Study28RobustnessProtocol(
        schema_version=str(payload.get("schema_version", "study28_robustness_interpretability_v1")),
        study_id=str(payload.get("study_id", "sepsis3_28d_three_class_disposition")),
        required_alpha3_external_complete=bool(payload.get("required_alpha3_external_complete", True)),
        required_alpha4_complete=bool(payload.get("required_alpha4_complete", True)),
        interpretability=interpretability,
        pre_icu_los_sensitivity=pre_icu,
        feature_availability_sensitivity=availability,
        missingness_robustness=missingness,
        hfd28_secondary=hfd,
        home_discharge_sensitivity=home,
        methodological_notes={str(k): str(v) for k, v in dict(payload.get("methodological_notes", {})).items()},
        source_path=str(path),
        sha256=_sha256(path),
    )
