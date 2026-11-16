from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class ExternalBootstrapProtocol:
    repeats: int
    confidence_level: float
    seed: int
    minimum_successful_repeats: int


@dataclass(frozen=True)
class ExternalCalibrationProtocol:
    bins: int
    clip_probability: float
    logistic_C: float
    max_iter: int


@dataclass(frozen=True)
class ExternalOutputProtocol:
    save_patient_level_predictions: bool
    save_plots: bool
    prediction_files_are_sensitive: bool
    include_patient_level_predictions_in_release_zip: bool


@dataclass(frozen=True)
class Study28ExternalProtocol:
    schema_version: str
    study_id: str
    frozen_protocol_sha256: str
    required_external_sites: tuple[str, ...]
    bootstrap: ExternalBootstrapProtocol
    calibration: ExternalCalibrationProtocol
    outputs: ExternalOutputProtocol
    methodological_notes: dict[str, str]
    source_path: str
    sha256: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_external_protocol(path: Path) -> Study28ExternalProtocol:
    path = Path(path).expanduser().resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise TypeError("Study28 external protocol must be a JSON object.")

    required = tuple(str(value) for value in payload.get("required_external_sites", []))
    if not required:
        raise ValueError("required_external_sites must not be empty.")
    if len(required) != len(set(required)):
        raise ValueError("required_external_sites contains duplicates.")

    bootstrap_payload = payload.get("bootstrap", {})
    bootstrap = ExternalBootstrapProtocol(
        repeats=int(bootstrap_payload.get("repeats", 1000)),
        confidence_level=float(bootstrap_payload.get("confidence_level", 0.95)),
        seed=int(bootstrap_payload.get("seed", 2026)),
        minimum_successful_repeats=int(
            bootstrap_payload.get("minimum_successful_repeats", 900)
        ),
    )
    if bootstrap.repeats != 1000:
        raise ValueError("The frozen Alpha 3 external protocol requires exactly 1000 bootstrap repeats.")
    if not 0.0 < bootstrap.confidence_level < 1.0:
        raise ValueError("bootstrap.confidence_level must be inside (0,1).")
    if bootstrap.minimum_successful_repeats < 1:
        raise ValueError("bootstrap.minimum_successful_repeats must be positive.")

    calibration_payload = payload.get("calibration", {})
    calibration = ExternalCalibrationProtocol(
        bins=int(calibration_payload.get("bins", 10)),
        clip_probability=float(calibration_payload.get("clip_probability", 1e-6)),
        logistic_C=float(calibration_payload.get("logistic_C", 1e6)),
        max_iter=int(calibration_payload.get("max_iter", 1000)),
    )
    if calibration.bins != 10:
        raise ValueError("The professor protocol uses 10 quantile calibration bins.")
    if not 0.0 < calibration.clip_probability < 0.5:
        raise ValueError("calibration.clip_probability must be inside (0,0.5).")

    outputs_payload = payload.get("outputs", {})
    outputs = ExternalOutputProtocol(
        save_patient_level_predictions=bool(
            outputs_payload.get("save_patient_level_predictions", True)
        ),
        save_plots=bool(outputs_payload.get("save_plots", True)),
        prediction_files_are_sensitive=bool(
            outputs_payload.get("prediction_files_are_sensitive", True)
        ),
        include_patient_level_predictions_in_release_zip=bool(
            outputs_payload.get("include_patient_level_predictions_in_release_zip", False)
        ),
    )
    if outputs.include_patient_level_predictions_in_release_zip:
        raise ValueError("Patient-level external predictions must not be included in release archives.")

    notes = {
        str(key): str(value)
        for key, value in dict(payload.get("methodological_notes", {})).items()
    }
    frozen_protocol_sha256 = str(payload.get("frozen_protocol_sha256", "")).lower()
    if len(frozen_protocol_sha256) != 64:
        raise ValueError("frozen_protocol_sha256 must be a SHA-256 digest.")

    return Study28ExternalProtocol(
        schema_version=str(payload.get("schema_version", "study28_external_validation_v1")),
        study_id=str(payload.get("study_id", "sepsis3_28d_three_class_disposition")),
        frozen_protocol_sha256=frozen_protocol_sha256,
        required_external_sites=required,
        bootstrap=bootstrap,
        calibration=calibration,
        outputs=outputs,
        methodological_notes=notes,
        source_path=str(path),
        sha256=_sha256(path),
    )
