from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from ..utils import sha256_file


@dataclass(frozen=True)
class ReverseValidationProtocol:
    pairs: tuple[tuple[str, str], ...]
    feature_set: str
    use_locked_primary_hyperparameters: bool
    bootstrap_repeats: int
    confidence_level: float
    minimum_successful_repeats: int
    seed: int


@dataclass(frozen=True)
class RecalibrationProtocol:
    sites: tuple[str, ...]
    calibration_fraction: float
    seeds: int
    base_seed: int
    clip_probability: float
    logistic_C: float
    max_iter: int
    minimum_successful_seeds: int
    representative_seed: int


@dataclass(frozen=True)
class DCAProtocol:
    classes: tuple[str, ...]
    threshold_min: float
    threshold_max: float
    threshold_step: float
    include_mimic_temporal_raw: bool
    include_external_raw: bool
    include_external_recalibrated_representative_split: bool
    include_reverse_validation: bool


@dataclass(frozen=True)
class SubgroupProtocol:
    reference_site: str
    exploratory_sites: tuple[str, ...]
    definitions: tuple[str, ...]
    minimum_n: int
    require_all_three_classes_for_macro_auroc: bool


@dataclass(frozen=True)
class HospitalHeterogeneityProtocol:
    site: str
    hospital_column: str
    classes: tuple[str, ...]
    minimum_hospital_n: int
    minimum_events: int
    method: str
    auc_standard_error: str


@dataclass(frozen=True)
class Study28SensitivityProtocol:
    schema_version: str
    study_id: str
    required_alpha3_external_complete: bool
    reverse_validation: ReverseValidationProtocol
    local_recalibration: RecalibrationProtocol
    dca: DCAProtocol
    subgroups: SubgroupProtocol
    hospital_heterogeneity: HospitalHeterogeneityProtocol
    methodological_notes: dict[str, str]
    sha256: str


def _tuple_pairs(values: list[Mapping[str, Any]]) -> tuple[tuple[str, str], ...]:
    return tuple((str(item["train"]), str(item["evaluate"])) for item in values)


def load_sensitivity_protocol(path: Path) -> Study28SensitivityProtocol:
    path = Path(path).expanduser().resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    rv = payload["reverse_validation"]
    rc = payload["local_recalibration"]
    dca = payload["decision_curve_analysis"]
    sg = payload["subgroups"]
    hh = payload["hospital_heterogeneity"]

    protocol = Study28SensitivityProtocol(
        schema_version=str(payload.get("schema_version", "study28_sensitivity_analysis_v1")),
        study_id=str(payload["study_id"]),
        required_alpha3_external_complete=bool(payload.get("required_alpha3_external_complete", True)),
        reverse_validation=ReverseValidationProtocol(
            pairs=_tuple_pairs(list(rv["pairs"])),
            feature_set=str(rv["feature_set"]),
            use_locked_primary_hyperparameters=bool(rv["use_locked_primary_hyperparameters"]),
            bootstrap_repeats=int(rv["bootstrap_repeats"]),
            confidence_level=float(rv["confidence_level"]),
            minimum_successful_repeats=int(rv["minimum_successful_repeats"]),
            seed=int(rv["seed"]),
        ),
        local_recalibration=RecalibrationProtocol(
            sites=tuple(str(x) for x in rc["sites"]),
            calibration_fraction=float(rc["calibration_fraction"]),
            seeds=int(rc["seeds"]),
            base_seed=int(rc["base_seed"]),
            clip_probability=float(rc["clip_probability"]),
            logistic_C=float(rc["logistic_C"]),
            max_iter=int(rc["max_iter"]),
            minimum_successful_seeds=int(rc["minimum_successful_seeds"]),
            representative_seed=int(rc["representative_seed"]),
        ),
        dca=DCAProtocol(
            classes=tuple(str(x) for x in dca["classes"]),
            threshold_min=float(dca["threshold_min"]),
            threshold_max=float(dca["threshold_max"]),
            threshold_step=float(dca["threshold_step"]),
            include_mimic_temporal_raw=bool(dca["include_mimic_temporal_raw"]),
            include_external_raw=bool(dca["include_external_raw"]),
            include_external_recalibrated_representative_split=bool(dca["include_external_recalibrated_representative_split"]),
            include_reverse_validation=bool(dca["include_reverse_validation"]),
        ),
        subgroups=SubgroupProtocol(
            reference_site=str(sg["reference_site"]),
            exploratory_sites=tuple(str(x) for x in sg["exploratory_sites"]),
            definitions=tuple(str(x) for x in sg["definitions"]),
            minimum_n=int(sg["minimum_n"]),
            require_all_three_classes_for_macro_auroc=bool(sg["require_all_three_classes_for_macro_auroc"]),
        ),
        hospital_heterogeneity=HospitalHeterogeneityProtocol(
            site=str(hh["site"]),
            hospital_column=str(hh["hospital_column"]),
            classes=tuple(str(x) for x in hh["classes"]),
            minimum_hospital_n=int(hh["minimum_hospital_n"]),
            minimum_events=int(hh["minimum_events"]),
            method=str(hh["method"]),
            auc_standard_error=str(hh["auc_standard_error"]),
        ),
        methodological_notes={str(k): str(v) for k, v in payload.get("methodological_notes", {}).items()},
        sha256=sha256_file(path),
    )
    if not (0.0 < protocol.local_recalibration.calibration_fraction < 1.0):
        raise ValueError("Local recalibration calibration_fraction must be in (0,1).")
    if protocol.local_recalibration.seeds < protocol.local_recalibration.minimum_successful_seeds:
        raise ValueError("minimum_successful_seeds cannot exceed requested recalibration seeds.")
    return protocol
