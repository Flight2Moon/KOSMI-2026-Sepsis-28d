from __future__ import annotations

import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd

from ..evaluation_multiclass import (
    BootstrapSettings,
    calculate_multiclass_metrics,
    write_multiclass_evaluation_artifacts,
)
from ..utils import atomic_write_json, sha256_file
from .audit import assert_required_external_sites, audit_study_datasets
from .calibration import study28_calibration_bins, study28_calibration_summary
from .contract import FeatureContract, load_feature_contract
from .datasets import DatasetRegistry, DatasetSpec, load_dataset_registry
from .external_protocol import Study28ExternalProtocol, load_external_protocol
from .modeling import predict_three_class_probabilities, resolve_lightgbm_threads
from .protocol import Study28Protocol, load_study28_protocol
from .resource_plan import child_thread_environment, read_int_env

CLASS_NAMES = ("early_dc", "death", "prolonged")


@dataclass(frozen=True)
class FrozenExternalModel:
    manifest_path: str
    manifest_sha256: str
    model_path: str
    model_sha256: str
    feature_names: tuple[str, ...]
    class_names: tuple[str, ...]
    locked_search_parameters: dict[str, Any]
    protocol_sha256: str
    feature_list_sha256: str
    mimic_sha256: str
    preprocessor: Any
    model: Any


@dataclass(frozen=True)
class ExternalSiteResult:
    site: str
    role: str
    n: int
    source_sha256: str
    model_sha256: str
    metrics: dict[str, Any]
    predictions_path: str | None
    site_manifest_path: str
    completed: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ExternalValidationResult:
    completed_sites: tuple[str, ...]
    required_sites: tuple[str, ...]
    all_required_complete: bool
    summary_csv: str
    bootstrap_ci_csv: str
    calibration_summary_csv: str
    calibration_bins_csv: str
    manifest_path: str
    frozen_model_sha256: str
    freeze_manifest_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"JSON object required: {path}")
    return payload


def _load_frozen_model(
    *,
    freeze_manifest_path: Path,
    dataset_root: Path,
    project_root: Path,
    feature_contract: FeatureContract,
    protocol: Study28Protocol,
    external_protocol: Study28ExternalProtocol,
    registry: DatasetRegistry,
) -> FrozenExternalModel:
    freeze_manifest_path = Path(freeze_manifest_path).expanduser().resolve()
    if not freeze_manifest_path.is_file():
        raise FileNotFoundError(f"Alpha 2 freeze manifest not found: {freeze_manifest_path}")
    manifest = _read_json(freeze_manifest_path)
    if manifest.get("freeze_schema") != "study28_model_freeze_v1":
        raise ValueError("Unsupported or missing Alpha 2 freeze schema.")
    if str(manifest.get("study_id")) != protocol.study_id:
        raise ValueError("Frozen model study_id does not match the Study28 protocol.")

    actual_protocol_path = Path(project_root) / "configs" / "study28_final.json"
    actual_protocol_sha = sha256_file(actual_protocol_path)
    expected_protocol_sha = external_protocol.frozen_protocol_sha256
    manifest_protocol_sha = str(manifest.get("protocol_sha256", ""))
    if not (
        actual_protocol_sha
        == expected_protocol_sha
        == manifest_protocol_sha
        == protocol.sha256
    ):
        raise RuntimeError(
            "Frozen Study28 protocol SHA-256 mismatch. External validation is blocked. "
            f"actual={actual_protocol_sha}, external_config={expected_protocol_sha}, "
            f"freeze_manifest={manifest_protocol_sha}, parsed={protocol.sha256}"
        )

    manifest_feature_sha = str(manifest.get("feature_list_sha256", ""))
    if manifest_feature_sha != feature_contract.sha256:
        raise RuntimeError("Frozen model feature_list SHA-256 does not match current feature_list.csv.")

    mimic_spec = registry["mimic"]
    mimic_sha = mimic_spec.source_sha256(dataset_root)
    manifest_mimic_sha = str(manifest.get("mimic_sha256", ""))
    if mimic_sha != manifest_mimic_sha:
        raise RuntimeError(
            "Frozen model MIMIC SHA-256 does not match the current professor-provided MIMIC file."
        )

    model_path = Path(str(manifest.get("external_model_path", ""))).expanduser().resolve()
    if not model_path.is_file():
        raise FileNotFoundError(f"Frozen external model artifact not found: {model_path}")
    model_sha = sha256_file(model_path)
    expected_model_sha = str(manifest.get("external_model_sha256", ""))
    if model_sha != expected_model_sha:
        raise RuntimeError(
            "Frozen external model artifact SHA-256 mismatch. External validation is blocked."
        )

    bundle = joblib.load(model_path)
    if not isinstance(bundle, Mapping):
        raise TypeError("Frozen external model artifact must contain a mapping bundle.")
    required_bundle_keys = {
        "preprocessor",
        "model",
        "feature_names",
        "class_names",
        "locked_search_parameters",
        "protocol_sha256",
        "feature_list_sha256",
        "mimic_sha256",
    }
    missing = sorted(required_bundle_keys - set(bundle))
    if missing:
        raise KeyError(f"Frozen model bundle is missing required keys: {missing}")

    feature_names = tuple(str(value) for value in bundle["feature_names"])
    class_names = tuple(str(value) for value in bundle["class_names"])
    if feature_names != feature_contract.no_apsiii:
        raise RuntimeError(
            "Frozen external model features are not the exact professor FS_NOAPS feature contract."
        )
    feature_contract.validate_model_features(feature_names)
    if class_names != CLASS_NAMES:
        raise RuntimeError(f"Frozen class order mismatch: {class_names}")

    if str(bundle["protocol_sha256"]) != manifest_protocol_sha:
        raise RuntimeError("Frozen model bundle protocol SHA does not match freeze manifest.")
    if str(bundle["feature_list_sha256"]) != manifest_feature_sha:
        raise RuntimeError("Frozen model bundle feature SHA does not match freeze manifest.")
    if str(bundle["mimic_sha256"]) != manifest_mimic_sha:
        raise RuntimeError("Frozen model bundle MIMIC SHA does not match freeze manifest.")

    # The serialized model remains byte-for-byte frozen.  Runtime inference
    # thread count may be reduced inside concurrent external workers to avoid
    # oversubscription; this does not retrain or modify the artifact on disk.
    runtime_model = bundle["model"]
    try:
        runtime_model.set_params(n_jobs=resolve_lightgbm_threads(default=1))
    except Exception:
        pass

    locked = dict(bundle["locked_search_parameters"])
    expected_locked = dict(protocol.primary_model.expected_locked_search_parameters)
    if locked != expected_locked:
        raise RuntimeError(
            f"Frozen locked search parameters differ from professor lock: {locked} != {expected_locked}"
        )

    manifest_features = tuple(str(value) for value in manifest.get("primary_features", []))
    if manifest_features != feature_names:
        raise RuntimeError("Freeze manifest feature order does not match frozen model bundle.")
    manifest_sites = tuple(str(value) for value in manifest.get("required_external_sites_next_phase", []))
    if set(manifest_sites) != set(external_protocol.required_external_sites):
        raise RuntimeError(
            "Frozen model manifest and Alpha 3 external protocol disagree on required external sites."
        )

    return FrozenExternalModel(
        manifest_path=str(freeze_manifest_path),
        manifest_sha256=sha256_file(freeze_manifest_path),
        model_path=str(model_path),
        model_sha256=model_sha,
        feature_names=feature_names,
        class_names=class_names,
        locked_search_parameters=locked,
        protocol_sha256=manifest_protocol_sha,
        feature_list_sha256=manifest_feature_sha,
        mimic_sha256=manifest_mimic_sha,
        preprocessor=bundle["preprocessor"],
        model=runtime_model,
    )


def validate_external_contract(
    *,
    dataset_root: Path,
    project_root: Path,
    freeze_manifest_path: Path,
    audit_output_dir: Path | None = None,
) -> tuple[
    Study28Protocol,
    Study28ExternalProtocol,
    DatasetRegistry,
    FeatureContract,
    FrozenExternalModel,
]:
    project_root = Path(project_root).expanduser().resolve()
    dataset_root = Path(dataset_root).expanduser().resolve()
    protocol = load_study28_protocol(project_root / "configs" / "study28_final.json")
    external_protocol = load_external_protocol(
        project_root / "configs" / "study28_external_validation.json"
    )
    registry = load_dataset_registry(project_root / "configs" / "dataset_registry_28d.json")
    feature_contract = load_feature_contract(dataset_root / "feature_list.csv")

    required_sets = {
        "main_protocol": set(protocol.required_external_sites),
        "external_protocol": set(external_protocol.required_external_sites),
        "registry": set(registry.required_external_sites),
    }
    if len({frozenset(value) for value in required_sets.values()}) != 1:
        raise RuntimeError(f"Required external site contracts disagree: {required_sets}")

    audit = audit_study_datasets(
        dataset_root=dataset_root,
        registry_path=project_root / "configs" / "dataset_registry_28d.json",
        feature_list_path=dataset_root / "feature_list.csv",
        output_dir=audit_output_dir,
    )
    if not audit.passed:
        raise RuntimeError(
            "Study28 dataset audit failed; external validation is blocked: "
            + "; ".join(audit.blocking_reasons)
        )

    frozen = _load_frozen_model(
        freeze_manifest_path=freeze_manifest_path,
        dataset_root=dataset_root,
        project_root=project_root,
        feature_contract=feature_contract,
        protocol=protocol,
        external_protocol=external_protocol,
        registry=registry,
    )
    return protocol, external_protocol, registry, feature_contract, frozen


def _prediction_frame(
    frame: pd.DataFrame,
    spec: DatasetSpec,
    probabilities: np.ndarray,
) -> pd.DataFrame:
    metadata_candidates = [
        *spec.id_columns,
        "hospitalid",
        "sepsis3_primary",
        "sepsis_admitdx",
        "admissionyear",
    ]
    metadata = []
    for column in metadata_candidates:
        if column in frame.columns and column not in metadata:
            metadata.append(column)
    prediction = frame.loc[:, metadata + ["y", "label28"]].copy()
    prediction["p_early_dc"] = probabilities[:, 0]
    prediction["p_death"] = probabilities[:, 1]
    prediction["p_prolonged"] = probabilities[:, 2]
    prediction["predicted_y"] = np.argmax(probabilities, axis=1).astype(np.int64)
    return prediction


def evaluate_external_site(
    *,
    site: str,
    dataset_root: Path,
    project_root: Path,
    freeze_manifest_path: Path,
    output_dir: Path,
) -> ExternalSiteResult:
    dataset_root = Path(dataset_root).expanduser().resolve()
    project_root = Path(project_root).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    protocol, external_protocol, registry, feature_contract, frozen = validate_external_contract(
        dataset_root=dataset_root,
        project_root=project_root,
        freeze_manifest_path=freeze_manifest_path,
        audit_output_dir=None,
    )
    if site not in external_protocol.required_external_sites:
        raise ValueError(f"Site is not a required Alpha 3 external cohort: {site}")
    spec = registry[site]
    if not spec.role.startswith("external"):
        raise ValueError(f"Dataset '{site}' is not registered as external: role={spec.role}")

    frame = spec.load(dataset_root)
    missing = sorted(set(frozen.feature_names + ("y", "label28")) - set(frame.columns))
    if missing:
        raise KeyError(f"External dataset '{site}' is missing required columns: {missing}")
    y = pd.to_numeric(frame["y"], errors="raise").to_numpy(dtype=np.int64)
    observed = tuple(sorted(np.unique(y).tolist()))
    if observed != (0, 1, 2):
        raise ValueError(f"External site '{site}' must contain all classes; got {observed}")

    X = frame.loc[:, list(frozen.feature_names)]
    transformed = frozen.preprocessor.transform(X)
    probabilities = predict_three_class_probabilities(frozen.model, transformed)
    point_metrics = calculate_multiclass_metrics(y, probabilities, frozen.class_names)

    bootstrap_settings = BootstrapSettings(
        repeats=external_protocol.bootstrap.repeats,
        confidence_level=external_protocol.bootstrap.confidence_level,
        seed=external_protocol.bootstrap.seed,
        minimum_successful_repeats=external_protocol.bootstrap.minimum_successful_repeats,
    )
    evaluation_manifest = write_multiclass_evaluation_artifacts(
        y_true=y,
        probabilities=probabilities,
        class_names=frozen.class_names,
        output_dir=output_dir,
        split_name=site,
        model_name=protocol.primary_model.name,
        calibration_bins=external_protocol.calibration.bins,
        bootstrap_settings=bootstrap_settings,
        save_plots=external_protocol.outputs.save_plots,
    )

    calibration_summary = study28_calibration_summary(
        y,
        probabilities,
        frozen.class_names,
        clip_probability=external_protocol.calibration.clip_probability,
        logistic_C=external_protocol.calibration.logistic_C,
        max_iter=external_protocol.calibration.max_iter,
    )
    calibration_summary.insert(0, "site", site)
    calibration_summary.to_csv(output_dir / "calibration_summary_professor_method.csv", index=False)

    calibration_bins = study28_calibration_bins(
        y,
        probabilities,
        frozen.class_names,
        bins=external_protocol.calibration.bins,
        clip_probability=external_protocol.calibration.clip_probability,
    )
    calibration_bins.insert(0, "site", site)
    calibration_bins.to_csv(output_dir / "calibration_bins_professor_method.csv", index=False)

    predictions_path: Path | None = None
    if external_protocol.outputs.save_patient_level_predictions:
        predictions_path = output_dir / "external_predictions.csv"
        _prediction_frame(frame, spec, probabilities).to_csv(predictions_path, index=False)

    site_manifest_path = output_dir / "external_site_manifest.json"
    site_manifest = {
        "schema_version": "study28_external_site_result_v1",
        "study_id": protocol.study_id,
        "site": site,
        "role": spec.role,
        "n": int(len(frame)),
        "class_counts": {
            class_name: int((y == class_code).sum())
            for class_code, class_name in enumerate(frozen.class_names)
        },
        "source_file": str(spec.resolve_path(dataset_root)),
        "source_sha256": spec.source_sha256(dataset_root),
        "freeze_manifest_path": frozen.manifest_path,
        "freeze_manifest_sha256": frozen.manifest_sha256,
        "frozen_model_path": frozen.model_path,
        "frozen_model_sha256": frozen.model_sha256,
        "frozen_protocol_sha256": frozen.protocol_sha256,
        "external_protocol_sha256": external_protocol.sha256,
        "feature_list_sha256": frozen.feature_list_sha256,
        "feature_set": protocol.primary_feature_set,
        "feature_count": len(frozen.feature_names),
        "locked_search_parameters": frozen.locked_search_parameters,
        "metrics": point_metrics,
        "bootstrap": {
            "repeats": external_protocol.bootstrap.repeats,
            "confidence_level": external_protocol.bootstrap.confidence_level,
            "seed": external_protocol.bootstrap.seed,
        },
        "calibration_method": "one_vs_rest_logistic_recalibration_on_logit_probability_professor_method",
        "calibration_bins": external_protocol.calibration.bins,
        "patient_level_predictions_saved": predictions_path is not None,
        "patient_level_predictions_sensitive": external_protocol.outputs.prediction_files_are_sensitive,
        "methodological_notes": external_protocol.methodological_notes,
        "generic_evaluation_manifest": evaluation_manifest,
        "completed": True,
    }
    atomic_write_json(site_manifest_path, site_manifest)
    return ExternalSiteResult(
        site=site,
        role=spec.role,
        n=int(len(frame)),
        source_sha256=spec.source_sha256(dataset_root),
        model_sha256=frozen.model_sha256,
        metrics=point_metrics,
        predictions_path=str(predictions_path) if predictions_path is not None else None,
        site_manifest_path=str(site_manifest_path),
        completed=True,
    )


def _load_completed_site_result(
    *,
    site: str,
    site_dir: Path,
    expected_model_sha256: str,
    expected_source_sha256: str,
    expected_external_protocol_sha256: str,
) -> ExternalSiteResult | None:
    manifest_path = site_dir / "external_site_manifest.json"
    if not manifest_path.is_file():
        return None
    manifest = _read_json(manifest_path)
    if manifest.get("completed") is not True:
        return None
    checks = {
        "site": str(manifest.get("site")) == site,
        "model": str(manifest.get("frozen_model_sha256")) == expected_model_sha256,
        "source": str(manifest.get("source_sha256")) == expected_source_sha256,
        "external_protocol": str(manifest.get("external_protocol_sha256"))
        == expected_external_protocol_sha256,
    }
    if not all(checks.values()):
        raise RuntimeError(
            f"Existing external checkpoint for '{site}' does not match the current frozen contracts: {checks}. "
            "Use a new output directory; do not overwrite confirmatory results in place."
        )
    predictions = site_dir / "external_predictions.csv"
    return ExternalSiteResult(
        site=site,
        role=str(manifest.get("role")),
        n=int(manifest.get("n", 0)),
        source_sha256=str(manifest.get("source_sha256")),
        model_sha256=str(manifest.get("frozen_model_sha256")),
        metrics=dict(manifest.get("metrics", {})),
        predictions_path=str(predictions) if predictions.is_file() else None,
        site_manifest_path=str(manifest_path),
        completed=True,
    )


def _aggregate_external_results(
    *,
    results: Sequence[ExternalSiteResult],
    required_sites: Sequence[str],
    output_dir: Path,
    frozen: FrozenExternalModel,
    external_protocol: Study28ExternalProtocol,
) -> ExternalValidationResult:
    output_dir = Path(output_dir).expanduser().resolve()
    completed_sites = tuple(result.site for result in results if result.completed)
    assert_required_external_sites(completed_sites, required_sites)

    summary_rows = []
    bootstrap_frames = []
    calibration_frames = []
    calibration_bin_frames = []
    for result in results:
        row = {
            "site": result.site,
            "role": result.role,
            "n": result.n,
            "source_sha256": result.source_sha256,
            "frozen_model_sha256": result.model_sha256,
        }
        row.update(result.metrics)
        summary_rows.append(row)

        site_dir = Path(result.site_manifest_path).parent
        bootstrap_path = site_dir / "bootstrap_confidence_intervals.csv"
        if not bootstrap_path.is_file():
            raise RuntimeError(f"Missing bootstrap result for completed site: {result.site}")
        boot = pd.read_csv(bootstrap_path)
        boot.insert(0, "site", result.site)
        bootstrap_frames.append(boot)

        cal = pd.read_csv(site_dir / "calibration_summary_professor_method.csv")
        calibration_frames.append(cal)
        bins = pd.read_csv(site_dir / "calibration_bins_professor_method.csv")
        calibration_bin_frames.append(bins)

    summary_path = output_dir / "external_validation_summary.csv"
    bootstrap_path = output_dir / "external_validation_bootstrap_ci.csv"
    calibration_path = output_dir / "external_calibration_summary.csv"
    calibration_bins_path = output_dir / "external_calibration_bins.csv"
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    pd.concat(bootstrap_frames, ignore_index=True).to_csv(bootstrap_path, index=False)
    pd.concat(calibration_frames, ignore_index=True).to_csv(calibration_path, index=False)
    pd.concat(calibration_bin_frames, ignore_index=True).to_csv(calibration_bins_path, index=False)

    manifest_path = output_dir / "study28_alpha3_external_validation_manifest.json"
    manifest = {
        "schema_version": "study28_alpha3_external_validation_v1",
        "required_external_sites": list(required_sites),
        "completed_external_sites": list(completed_sites),
        "all_required_external_complete": True,
        "freeze_manifest_path": frozen.manifest_path,
        "freeze_manifest_sha256": frozen.manifest_sha256,
        "frozen_model_path": frozen.model_path,
        "frozen_model_sha256": frozen.model_sha256,
        "frozen_protocol_sha256": frozen.protocol_sha256,
        "external_protocol_sha256": external_protocol.sha256,
        "feature_list_sha256": frozen.feature_list_sha256,
        "mimic_development_sha256": frozen.mimic_sha256,
        "bootstrap_repeats": external_protocol.bootstrap.repeats,
        "confidence_level": external_protocol.bootstrap.confidence_level,
        "raw_external_only": True,
        "local_recalibration_performed": False,
        "eicu_cohorts_independent": False,
        "patient_level_outputs_are_sensitive": external_protocol.outputs.prediction_files_are_sensitive,
        "methodological_notes": external_protocol.methodological_notes,
        "sites": [result.to_dict() for result in results],
    }
    atomic_write_json(manifest_path, manifest)
    return ExternalValidationResult(
        completed_sites=completed_sites,
        required_sites=tuple(required_sites),
        all_required_complete=True,
        summary_csv=str(summary_path),
        bootstrap_ci_csv=str(bootstrap_path),
        calibration_summary_csv=str(calibration_path),
        calibration_bins_csv=str(calibration_bins_path),
        manifest_path=str(manifest_path),
        frozen_model_sha256=frozen.model_sha256,
        freeze_manifest_sha256=frozen.manifest_sha256,
    )


def run_external_validation_isolated(
    *,
    dataset_root: Path,
    project_root: Path,
    freeze_manifest_path: Path,
    output_dir: Path,
) -> ExternalValidationResult:
    """Run mandatory raw external sites in isolated resource-planned workers.

    Site-level prediction/bootstrap jobs are statistically independent. Parallelizing
    them changes only wall-clock scheduling; frozen model/data/protocol hashes remain
    mandatory and each site keeps an independent checkpoint.
    """
    dataset_root = Path(dataset_root).expanduser().resolve()
    project_root = Path(project_root).expanduser().resolve()
    freeze_manifest_path = Path(freeze_manifest_path).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    protocol, external_protocol, registry, _feature_contract, frozen = validate_external_contract(
        dataset_root=dataset_root,
        project_root=project_root,
        freeze_manifest_path=freeze_manifest_path,
        audit_output_dir=output_dir / "00_dataset_audit",
    )
    required_sites = external_protocol.required_external_sites
    assert_required_external_sites(required_sites, protocol.required_external_sites)

    site_root = output_dir / "sites"
    site_root.mkdir(parents=True, exist_ok=True)
    worker_count = min(len(required_sites), read_int_env("STUDY28_EXTERNAL_SITE_WORKERS", 1))
    threads_per_worker = read_int_env("STUDY28_EXTERNAL_SITE_THREADS", 1)
    results_by_site: dict[str, ExternalSiteResult] = {}
    pending: list[str] = []

    for site in required_sites:
        spec = registry[site]
        site_dir = site_root / site
        site_dir.mkdir(parents=True, exist_ok=True)
        existing = _load_completed_site_result(
            site=site,
            site_dir=site_dir,
            expected_model_sha256=frozen.model_sha256,
            expected_source_sha256=spec.source_sha256(dataset_root),
            expected_external_protocol_sha256=external_protocol.sha256,
        )
        if existing is not None:
            print(f"[Study28 Alpha3] {site}: existing verified checkpoint; skip", flush=True)
            results_by_site[site] = existing
        else:
            pending.append(site)

    def execute_site(site: str) -> tuple[str, ExternalSiteResult]:
        spec = registry[site]
        site_dir = site_root / site
        print(f"[Study28 Alpha3] {site}: raw frozen-model external evaluation", flush=True)
        log_path = site_dir / "external_worker.log"
        command = [
            sys.executable, "-m", "los_baseline.study28.external_worker",
            "--site", site,
            "--dataset-root", str(dataset_root),
            "--project-root", str(project_root),
            "--freeze-manifest", str(freeze_manifest_path),
            "--output-dir", str(site_dir),
        ]
        env = child_thread_environment(os.environ.copy(), threads_per_worker)
        src_path = str(project_root / "src")
        current = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = src_path + (os.pathsep + current if current else "")
        with log_path.open("w", encoding="utf-8") as log_handle:
            completed = subprocess.run(
                command, cwd=project_root, env=env, stdout=log_handle,
                stderr=subprocess.STDOUT, text=True, check=False,
            )
        if completed.returncode != 0:
            tail = log_path.read_text(encoding="utf-8", errors="replace")[-6000:]
            raise RuntimeError(f"External site '{site}' failed. Log tail:\n{tail}")
        verified = _load_completed_site_result(
            site=site,
            site_dir=site_dir,
            expected_model_sha256=frozen.model_sha256,
            expected_source_sha256=spec.source_sha256(dataset_root),
            expected_external_protocol_sha256=external_protocol.sha256,
        )
        if verified is None:
            raise RuntimeError(f"External site '{site}' did not create a completed checkpoint.")
        return site, verified

    if pending:
        print(
            f"[Study28 Alpha3] resource plan: concurrent_sites={worker_count}, "
            f"threads_per_site={threads_per_worker}, pending={len(pending)}",
            flush=True,
        )
        if worker_count <= 1:
            for site in pending:
                key, result = execute_site(site)
                results_by_site[key] = result
        else:
            with ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="study28-ext") as pool:
                futures = {pool.submit(execute_site, site): site for site in pending}
                for future in as_completed(futures):
                    site, result = future.result()
                    results_by_site[site] = result
                    print(f"[Study28 Alpha3] external checkpoint completed: {site}", flush=True)

    results = [results_by_site[site] for site in required_sites]
    result = _aggregate_external_results(
        results=results,
        required_sites=required_sites,
        output_dir=output_dir,
        frozen=frozen,
        external_protocol=external_protocol,
    )
    print(
        "[Study28 Alpha3] all mandatory external cohorts completed: "
        + ", ".join(result.completed_sites),
        flush=True,
    )
    return result

