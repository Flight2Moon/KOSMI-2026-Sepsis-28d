from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import joblib
import numpy as np
import pandas as pd

from ..evaluation_multiclass import (
    BootstrapSettings,
    bootstrap_multiclass_confidence_intervals,
    calculate_multiclass_metrics,
)
from ..utils import atomic_write_json, sha256_file
from .adaptation import repeated_local_recalibration
from .audit import audit_study_datasets, assert_required_external_sites
from .clinical_utility import (
    decision_curve_rows,
    hospital_level_heterogeneity,
    subgroup_performance,
)
from .contract import FeatureContract, load_feature_contract
from .datasets import DatasetRegistry, DatasetSpec, load_dataset_registry
from .development import _model_parameters
from .external_protocol import Study28ExternalProtocol, load_external_protocol
from .external_validation import FrozenExternalModel, validate_external_contract
from .modeling import build_lightgbm_multiclass, predict_three_class_probabilities
from .preprocessing import build_study28_preprocessor
from .protocol import Study28Protocol, load_study28_protocol
from .sensitivity_protocol import Study28SensitivityProtocol, load_sensitivity_protocol
from .sensitivity_reporting import (
    dca_dominance_summary, plot_dca_by_setting, plot_hospital_forest,
    recalibration_change_summary,
)

CLASS_NAMES = ("early_dc", "death", "prolonged")
PROBABILITY_COLUMNS = ("p_early_dc", "p_death", "p_prolonged")


@dataclass(frozen=True)
class Alpha4Contract:
    protocol: Study28Protocol
    external_protocol: Study28ExternalProtocol
    sensitivity_protocol: Study28SensitivityProtocol
    registry: DatasetRegistry
    feature_contract: FeatureContract
    frozen: FrozenExternalModel
    alpha3_manifest: dict[str, Any]
    alpha3_manifest_path: Path


@dataclass(frozen=True)
class Alpha4Result:
    manifest_path: str
    reverse_validation_summary_csv: str
    reverse_validation_ci_csv: str
    recalibration_raw_csv: str
    recalibration_summary_csv: str
    dca_csv: str
    subgroup_csv: str
    hospital_level_csv: str
    hospital_summary_csv: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"JSON object required: {path}")
    return payload


def _alpha3_manifest_path(alpha3_output_dir: Path) -> Path:
    return Path(alpha3_output_dir) / "study28_alpha3_external_validation_manifest.json"


def validate_alpha4_contract(
    *,
    dataset_root: Path,
    project_root: Path,
    alpha2_output_dir: Path,
    alpha3_output_dir: Path,
    audit_output_dir: Path | None = None,
) -> Alpha4Contract:
    dataset_root = Path(dataset_root).expanduser().resolve()
    project_root = Path(project_root).expanduser().resolve()
    alpha2_output_dir = Path(alpha2_output_dir).expanduser().resolve()
    alpha3_output_dir = Path(alpha3_output_dir).expanduser().resolve()

    protocol = load_study28_protocol(project_root / "configs" / "study28_final.json")
    external_protocol = load_external_protocol(project_root / "configs" / "study28_external_validation.json")
    sensitivity_protocol = load_sensitivity_protocol(project_root / "configs" / "study28_sensitivity_analysis.json")
    registry = load_dataset_registry(project_root / "configs" / "dataset_registry_28d.json")
    feature_contract = load_feature_contract(dataset_root / "feature_list.csv")

    if sensitivity_protocol.study_id != protocol.study_id:
        raise RuntimeError("Alpha 4 sensitivity study_id differs from the frozen Study28 protocol.")
    if tuple(registry.reverse_sensitivity) != tuple(sensitivity_protocol.reverse_validation.pairs):
        raise RuntimeError(
            "Dataset registry reverse_sensitivity and Alpha 4 reverse-validation protocol disagree."
        )
    if sensitivity_protocol.reverse_validation.feature_set != "FS_NOAPS":
        raise RuntimeError("Alpha 4 reverse validation must use the final primary FS_NOAPS feature set.")
    if set(sensitivity_protocol.local_recalibration.sites) != set(external_protocol.required_external_sites):
        raise RuntimeError("Alpha 4 local recalibration must include all four Alpha 3 external cohorts.")

    audit = audit_study_datasets(
        dataset_root=dataset_root,
        registry_path=project_root / "configs" / "dataset_registry_28d.json",
        feature_list_path=dataset_root / "feature_list.csv",
        output_dir=audit_output_dir,
    )
    if not audit.passed:
        raise RuntimeError("Study28 dataset audit failed; Alpha 4 is blocked: " + "; ".join(audit.blocking_reasons))

    freeze_manifest_path = alpha2_output_dir / "03_external_model_freeze" / "study28_model_freeze.json"
    _, _, _, _, frozen = validate_external_contract(
        dataset_root=dataset_root,
        project_root=project_root,
        freeze_manifest_path=freeze_manifest_path,
        audit_output_dir=None,
    )

    alpha3_path = _alpha3_manifest_path(alpha3_output_dir)
    if not alpha3_path.is_file():
        raise FileNotFoundError(f"Alpha 3 manifest not found: {alpha3_path}")
    alpha3 = _read_json(alpha3_path)
    if alpha3.get("schema_version") != "study28_alpha3_external_validation_v1":
        raise RuntimeError("Unsupported Alpha 3 manifest schema.")
    if sensitivity_protocol.required_alpha3_external_complete and not bool(alpha3.get("all_required_external_complete")):
        raise RuntimeError("Alpha 4 is blocked until all mandatory Alpha 3 external cohorts are complete.")
    assert_required_external_sites(alpha3.get("completed_external_sites", []), external_protocol.required_external_sites)
    if str(alpha3.get("frozen_model_sha256")) != frozen.model_sha256:
        raise RuntimeError("Alpha 3 manifest model SHA does not match the Alpha 2 frozen model.")
    if str(alpha3.get("frozen_protocol_sha256")) != frozen.protocol_sha256:
        raise RuntimeError("Alpha 3 manifest frozen protocol SHA mismatch.")
    if str(alpha3.get("external_protocol_sha256")) != external_protocol.sha256:
        raise RuntimeError("Alpha 3 external protocol SHA mismatch.")
    if bool(alpha3.get("local_recalibration_performed")):
        raise RuntimeError("Alpha 3 raw external manifest unexpectedly reports local recalibration.")

    return Alpha4Contract(
        protocol=protocol,
        external_protocol=external_protocol,
        sensitivity_protocol=sensitivity_protocol,
        registry=registry,
        feature_contract=feature_contract,
        frozen=frozen,
        alpha3_manifest=alpha3,
        alpha3_manifest_path=alpha3_path,
    )


def _load_raw_external_predictions(
    *,
    site: str,
    alpha3_output_dir: Path,
    dataset_root: Path,
    spec: DatasetSpec,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    prediction_path = Path(alpha3_output_dir) / "sites" / site / "external_predictions.csv"
    if not prediction_path.is_file():
        raise FileNotFoundError(f"Alpha 3 external predictions missing for {site}: {prediction_path}")
    pred = pd.read_csv(prediction_path, low_memory=False)
    frame = spec.load(dataset_root)
    if len(pred) != len(frame):
        raise RuntimeError(f"Alpha 3 prediction row count differs from frozen {site} dataset.")
    for column in spec.id_columns:
        if column in pred.columns and column in frame.columns:
            left = pred[column].astype(str).fillna("<NA>").to_numpy()
            right = frame[column].astype(str).fillna("<NA>").to_numpy()
            if not np.array_equal(left, right):
                raise RuntimeError(f"Alpha 3 {site} prediction IDs are not row-aligned for {column}.")
    if not np.array_equal(pred["y"].to_numpy(dtype=int), frame["y"].to_numpy(dtype=int)):
        raise RuntimeError(f"Alpha 3 {site} prediction labels are not row-aligned.")
    P = pred[list(PROBABILITY_COLUMNS)].to_numpy(dtype=np.float64)
    y = pred["y"].to_numpy(dtype=np.int64)
    if not np.allclose(P.sum(axis=1), 1.0, atol=1e-6, rtol=1e-6):
        raise RuntimeError(f"Alpha 3 {site} probabilities do not sum to one.")
    return frame, y, P


def _fit_primary_on_frame(
    *,
    train: pd.DataFrame,
    feature_names: tuple[str, ...],
    protocol: Study28Protocol,
) -> tuple[Any, Any]:
    y = train["y"].to_numpy(dtype=np.int64)
    if tuple(sorted(np.unique(y).tolist())) != (0, 1, 2):
        raise ValueError("Reverse-validation training cohort must contain all three classes.")
    pre = build_study28_preprocessor("lightgbm", feature_names)
    X = pre.fit_transform(train[list(feature_names)])
    model = build_lightgbm_multiclass(
        _model_parameters(protocol, protocol.primary_model.expected_locked_search_parameters)
    )
    model.fit(X, y)
    return pre, model


def _interval_frame(intervals: list[Any], **extra: Any) -> pd.DataFrame:
    rows = []
    for interval in intervals:
        row = dict(extra)
        row.update(asdict(interval))
        rows.append(row)
    return pd.DataFrame(rows)


def _run_reverse_pair(
    *,
    contract: Alpha4Contract,
    dataset_root: Path,
    train_name: str,
    eval_name: str,
    output_dir: Path,
) -> tuple[dict[str, Any], pd.DataFrame, np.ndarray, np.ndarray]:
    output_dir.mkdir(parents=True, exist_ok=True)
    features = contract.feature_contract.no_apsiii
    proto = contract.sensitivity_protocol.reverse_validation
    train = contract.registry[train_name].load(dataset_root)
    evaluate = contract.registry[eval_name].load(dataset_root)
    pre, model = _fit_primary_on_frame(
        train=train, feature_names=features, protocol=contract.protocol
    )
    P = predict_three_class_probabilities(model, pre.transform(evaluate[list(features)]))
    y = evaluate["y"].to_numpy(dtype=np.int64)
    metrics = calculate_multiclass_metrics(y, P, CLASS_NAMES)
    setting = f"{train_name}->{eval_name}"
    row = {"setting": setting, "train_site": train_name, "evaluation_site": eval_name,
           "train_n": len(train), "evaluation_n": len(evaluate), **metrics}
    settings = BootstrapSettings(
        repeats=proto.bootstrap_repeats,
        confidence_level=proto.confidence_level,
        seed=proto.seed,
        minimum_successful_repeats=proto.minimum_successful_repeats,
    )
    intervals = bootstrap_multiclass_confidence_intervals(y, P, CLASS_NAMES, settings)
    ci = _interval_frame(intervals, setting=setting)
    bundle = {
        "preprocessor": pre,
        "model": model,
        "feature_names": features,
        "class_names": CLASS_NAMES,
        "training_site": train_name,
        "evaluation_site": eval_name,
        "sensitivity_only": True,
    }
    model_path = output_dir / "reverse_model.joblib"
    joblib.dump(bundle, model_path)
    pd.DataFrame([row]).to_csv(output_dir / "summary.csv", index=False)
    ci.to_csv(output_dir / "bootstrap_ci.csv", index=False)
    np.savez_compressed(output_dir / "reverse_probabilities_sensitive.npz", y=y, probabilities=P)
    atomic_write_json(
        output_dir / "pair_manifest.json",
        {
            "schema_version": "study28_alpha4_reverse_pair_v1",
            "setting": setting,
            "train_site": train_name,
            "evaluation_site": eval_name,
            "feature_set": "FS_NOAPS",
            "locked_parameters": contract.protocol.primary_model.expected_locked_search_parameters,
            "model_sha256": sha256_file(model_path),
            "bootstrap_repeats": proto.bootstrap_repeats,
            "sensitivity_only": True,
            "does_not_replace_frozen_primary_model": True,
        },
    )
    return row, ci, y, P


def run_reverse_validation(
    *,
    contract: Alpha4Contract,
    dataset_root: Path,
    project_root: Path,
    alpha2_output_dir: Path,
    alpha3_output_dir: Path,
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, tuple[np.ndarray, np.ndarray]]]:
    """Run each reverse direction in an isolated process to avoid OpenMP state accumulation."""
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries: list[pd.DataFrame] = []
    ci_frames: list[pd.DataFrame] = []
    predictions: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for train_name, eval_name in contract.sensitivity_protocol.reverse_validation.pairs:
        pair_dir = output_dir / f"{train_name}_to_{eval_name}"
        pair_dir.mkdir(parents=True, exist_ok=True)
        expected = [pair_dir / "summary.csv", pair_dir / "bootstrap_ci.csv",
                    pair_dir / "reverse_probabilities_sensitive.npz", pair_dir / "pair_manifest.json"]
        if not all(path.is_file() for path in expected):
            command = [
                sys.executable, "-m", "los_baseline.study28.reverse_worker",
                "--dataset-root", str(dataset_root),
                "--project-root", str(project_root),
                "--alpha2-output-dir", str(alpha2_output_dir),
                "--alpha3-output-dir", str(alpha3_output_dir),
                "--train-site", train_name,
                "--evaluation-site", eval_name,
                "--output-dir", str(pair_dir),
            ]
            env = os.environ.copy()
            env.setdefault("STUDY28_LIGHTGBM_THREADS", "1")
            subprocess.run(command, cwd=project_root, env=env, check=True)
        manifest = _read_json(pair_dir / "pair_manifest.json")
        if manifest.get("setting") != f"{train_name}->{eval_name}":
            raise RuntimeError(f"Reverse-validation checkpoint setting mismatch: {pair_dir}")
        if not bool(manifest.get("does_not_replace_frozen_primary_model")):
            raise RuntimeError("Reverse-validation artifact lacks sensitivity-only guard.")
        summaries.append(pd.read_csv(pair_dir / "summary.csv"))
        ci_frames.append(pd.read_csv(pair_dir / "bootstrap_ci.csv"))
        with np.load(pair_dir / "reverse_probabilities_sensitive.npz") as data:
            y = np.asarray(data["y"], dtype=np.int64)
            P = np.asarray(data["probabilities"], dtype=np.float64)
        predictions[f"{train_name}->{eval_name}"] = (y, P)
    summary = pd.concat(summaries, ignore_index=True)
    ci = pd.concat(ci_frames, ignore_index=True)
    summary.to_csv(output_dir / "reverse_validation_summary.csv", index=False)
    ci.to_csv(output_dir / "reverse_validation_bootstrap_ci.csv", index=False)
    return summary, ci, predictions


def run_local_recalibration(
    *,
    contract: Alpha4Contract,
    dataset_root: Path,
    alpha3_output_dir: Path,
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    p = contract.sensitivity_protocol.local_recalibration
    raw_frames: list[pd.DataFrame] = []
    agg_frames: list[pd.DataFrame] = []
    representative: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    for site in p.sites:
        _, y, P = _load_raw_external_predictions(
            site=site,
            alpha3_output_dir=alpha3_output_dir,
            dataset_root=dataset_root,
            spec=contract.registry[site],
        )
        raw, agg, seed_artifacts = repeated_local_recalibration(
            site=site,
            y_true=y,
            probabilities=P,
            class_names=CLASS_NAMES,
            calibration_fraction=p.calibration_fraction,
            seeds=p.seeds,
            base_seed=p.base_seed,
            clip_probability=p.clip_probability,
            logistic_C=p.logistic_C,
            max_iter=p.max_iter,
        )
        successful_seeds = raw["seed"].nunique() if not raw.empty else 0
        if successful_seeds < p.minimum_successful_seeds:
            raise RuntimeError(
                f"Too few successful local recalibration seeds for {site}: "
                f"{successful_seeds}/{p.seeds}"
            )
        if p.representative_seed not in seed_artifacts:
            raise RuntimeError(f"Representative recalibration seed failed for {site}.")
        app_idx, rawP, recalP = seed_artifacts[p.representative_seed]
        representative[site] = (y[app_idx], app_idx, rawP, recalP)
        raw_frames.append(raw)
        agg_frames.append(agg)
    raw_all = pd.concat(raw_frames, ignore_index=True)
    agg_all = pd.concat(agg_frames, ignore_index=True)
    raw_all.to_csv(output_dir / "local_recalibration_multiseed_raw.csv", index=False)
    agg_all.to_csv(output_dir / "local_recalibration_multiseed_summary.csv", index=False)
    return raw_all, agg_all, representative


def run_dca(
    *,
    contract: Alpha4Contract,
    dataset_root: Path,
    alpha2_output_dir: Path,
    alpha3_output_dir: Path,
    reverse_predictions: Mapping[str, tuple[np.ndarray, np.ndarray]],
    recalibration_representative: Mapping[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    output_dir: Path,
) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    p = contract.sensitivity_protocol.dca
    thresholds = np.arange(p.threshold_min, p.threshold_max + p.threshold_step / 2.0, p.threshold_step)
    frames: list[pd.DataFrame] = []
    if p.include_mimic_temporal_raw:
        tmp = pd.read_csv(Path(alpha2_output_dir) / "02_temporal_evaluation" / "mimic_temporal_predictions.csv")
        frames.append(decision_curve_rows(
            setting="mimic_temporal",
            y_true=tmp["y"].to_numpy(dtype=int),
            model_probabilities={"raw_primary": tmp[list(PROBABILITY_COLUMNS)].to_numpy(float)},
            classes=p.classes,
            thresholds=thresholds,
        ))
    if p.include_external_raw:
        for site in contract.sensitivity_protocol.local_recalibration.sites:
            _, y, P = _load_raw_external_predictions(
                site=site, alpha3_output_dir=alpha3_output_dir,
                dataset_root=dataset_root, spec=contract.registry[site]
            )
            frames.append(decision_curve_rows(
                setting=f"mimic->{site}_raw_full",
                y_true=y, model_probabilities={"raw_primary": P},
                classes=p.classes, thresholds=thresholds,
            ))
    if p.include_external_recalibrated_representative_split:
        for site, (y_app, _idx, rawP, recalP) in recalibration_representative.items():
            frames.append(decision_curve_rows(
                setting=f"mimic->{site}_local80_seed{contract.sensitivity_protocol.local_recalibration.representative_seed}",
                y_true=y_app,
                model_probabilities={"raw_primary": rawP, "site_recalibrated": recalP},
                classes=p.classes,
                thresholds=thresholds,
            ))
    if p.include_reverse_validation:
        for setting, (y, P) in reverse_predictions.items():
            frames.append(decision_curve_rows(
                setting=setting, y_true=y,
                model_probabilities={"reverse_primary": P},
                classes=p.classes, thresholds=thresholds,
            ))
    dca = pd.concat(frames, ignore_index=True)
    dca.to_csv(output_dir / "decision_curve_analysis.csv", index=False)
    return dca


def run_subgroups(
    *,
    contract: Alpha4Contract,
    dataset_root: Path,
    alpha3_output_dir: Path,
    output_dir: Path,
) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    p = contract.sensitivity_protocol.subgroups
    frames: list[pd.DataFrame] = []
    for site in (p.reference_site, *p.exploratory_sites):
        frame, y, P = _load_raw_external_predictions(
            site=site, alpha3_output_dir=alpha3_output_dir,
            dataset_root=dataset_root, spec=contract.registry[site]
        )
        frames.append(subgroup_performance(
            site=site, frame=frame, y_true=y, probabilities=P,
            definitions=p.definitions, minimum_n=p.minimum_n,
            reference=(site == p.reference_site),
        ))
    result = pd.concat(frames, ignore_index=True)
    result.to_csv(output_dir / "external_subgroup_performance.csv", index=False)
    return result


def run_hospital_heterogeneity(
    *,
    contract: Alpha4Contract,
    dataset_root: Path,
    alpha3_output_dir: Path,
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    output_dir.mkdir(parents=True, exist_ok=True)
    p = contract.sensitivity_protocol.hospital_heterogeneity
    frame, y, P = _load_raw_external_predictions(
        site=p.site, alpha3_output_dir=alpha3_output_dir,
        dataset_root=dataset_root, spec=contract.registry[p.site]
    )
    levels, summary = hospital_level_heterogeneity(
        frame=frame, y_true=y, probabilities=P,
        hospital_column=p.hospital_column, classes=p.classes,
        minimum_hospital_n=p.minimum_hospital_n,
        minimum_events=p.minimum_events,
    )
    levels.to_csv(output_dir / "hospital_level_auc.csv", index=False)
    summary.to_csv(output_dir / "hospital_random_effects_summary.csv", index=False)
    return levels, summary


def run_alpha4(
    *,
    dataset_root: Path,
    project_root: Path,
    alpha2_output_dir: Path,
    alpha3_output_dir: Path,
    output_dir: Path,
) -> Alpha4Result:
    dataset_root = Path(dataset_root).expanduser().resolve()
    project_root = Path(project_root).expanduser().resolve()
    alpha2_output_dir = Path(alpha2_output_dir).expanduser().resolve()
    alpha3_output_dir = Path(alpha3_output_dir).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    contract = validate_alpha4_contract(
        dataset_root=dataset_root,
        project_root=project_root,
        alpha2_output_dir=alpha2_output_dir,
        alpha3_output_dir=alpha3_output_dir,
        audit_output_dir=output_dir / "00_dataset_audit",
    )

    reverse_summary, reverse_ci, reverse_predictions = run_reverse_validation(
        contract=contract, dataset_root=dataset_root, project_root=project_root,
        alpha2_output_dir=alpha2_output_dir, alpha3_output_dir=alpha3_output_dir,
        output_dir=output_dir / "01_reverse_validation",
    )
    recal_raw, recal_summary, representative = run_local_recalibration(
        contract=contract, dataset_root=dataset_root,
        alpha3_output_dir=alpha3_output_dir,
        output_dir=output_dir / "02_local_recalibration",
    )
    dca = run_dca(
        contract=contract, dataset_root=dataset_root,
        alpha2_output_dir=alpha2_output_dir,
        alpha3_output_dir=alpha3_output_dir,
        reverse_predictions=reverse_predictions,
        recalibration_representative=representative,
        output_dir=output_dir / "03_dca",
    )
    subgroup = run_subgroups(
        contract=contract, dataset_root=dataset_root,
        alpha3_output_dir=alpha3_output_dir,
        output_dir=output_dir / "04_subgroups",
    )
    hospital, hospital_summary = run_hospital_heterogeneity(
        contract=contract, dataset_root=dataset_root,
        alpha3_output_dir=alpha3_output_dir,
        output_dir=output_dir / "05_hospital_heterogeneity",
    )

    reporting_dir = output_dir / "06_reporting"
    reporting_dir.mkdir(parents=True, exist_ok=True)
    recal_change = recalibration_change_summary(recal_summary)
    recal_change.to_csv(reporting_dir / "local_recalibration_change_summary.csv", index=False)
    dca_summary = dca_dominance_summary(dca)
    dca_summary.to_csv(reporting_dir / "dca_dominance_summary.csv", index=False)
    dca_plots = plot_dca_by_setting(dca, reporting_dir / "figures" / "dca")
    hospital_plots = plot_hospital_forest(
        hospital, hospital_summary, reporting_dir / "figures" / "hospital_heterogeneity"
    )

    manifest = {
        "schema_version": "study28_alpha4_sensitivity_v1",
        "study_id": contract.protocol.study_id,
        "alpha2_frozen_model_sha256": contract.frozen.model_sha256,
        "alpha3_manifest_path": str(contract.alpha3_manifest_path),
        "alpha3_manifest_sha256": sha256_file(contract.alpha3_manifest_path),
        "alpha3_raw_external_preserved": True,
        "sensitivity_protocol_sha256": contract.sensitivity_protocol.sha256,
        "reverse_validation_pairs": [list(x) for x in contract.sensitivity_protocol.reverse_validation.pairs],
        "reverse_models_change_primary_model": False,
        "local_recalibration_sites": list(contract.sensitivity_protocol.local_recalibration.sites),
        "local_recalibration_fraction": contract.sensitivity_protocol.local_recalibration.calibration_fraction,
        "local_recalibration_seeds": contract.sensitivity_protocol.local_recalibration.seeds,
        "dca_classes": list(contract.sensitivity_protocol.dca.classes),
        "subgroup_reference_site": contract.sensitivity_protocol.subgroups.reference_site,
        "subgroup_exploratory_sites": list(contract.sensitivity_protocol.subgroups.exploratory_sites),
        "hospital_heterogeneity_site": contract.sensitivity_protocol.hospital_heterogeneity.site,
        "aggregate_counts": {
            "reverse_settings": int(len(reverse_summary)),
            "recalibration_rows": int(len(recal_raw)),
            "dca_rows": int(len(dca)),
            "subgroup_rows": int(len(subgroup)),
            "eligible_hospital_rows": int(len(hospital)),
            "hospital_summary_rows": int(len(hospital_summary)),
            "recalibration_change_rows": int(len(recal_change)),
            "dca_dominance_rows": int(len(dca_summary)),
        },
        "reporting": {
            "dca_plot_count": len(dca_plots),
            "hospital_forest_plot_count": len(hospital_plots),
        },
        "methodological_notes": contract.sensitivity_protocol.methodological_notes,
    }
    manifest_path = output_dir / "study28_alpha4_sensitivity_manifest.json"
    atomic_write_json(manifest_path, manifest)

    return Alpha4Result(
        manifest_path=str(manifest_path),
        reverse_validation_summary_csv=str(output_dir / "01_reverse_validation" / "reverse_validation_summary.csv"),
        reverse_validation_ci_csv=str(output_dir / "01_reverse_validation" / "reverse_validation_bootstrap_ci.csv"),
        recalibration_raw_csv=str(output_dir / "02_local_recalibration" / "local_recalibration_multiseed_raw.csv"),
        recalibration_summary_csv=str(output_dir / "02_local_recalibration" / "local_recalibration_multiseed_summary.csv"),
        dca_csv=str(output_dir / "03_dca" / "decision_curve_analysis.csv"),
        subgroup_csv=str(output_dir / "04_subgroups" / "external_subgroup_performance.csv"),
        hospital_level_csv=str(output_dir / "05_hospital_heterogeneity" / "hospital_level_auc.csv"),
        hospital_summary_csv=str(output_dir / "05_hospital_heterogeneity" / "hospital_random_effects_summary.csv"),
    )
