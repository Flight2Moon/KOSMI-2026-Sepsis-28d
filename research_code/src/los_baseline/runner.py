"""Legacy 7/14/28-day LOS binary baseline runner.

This module is retained for reproducibility of the earlier threshold-based
pipeline invoked by run.sh. It is not the EXP123 V6 A/B/C confirmatory
pipeline. New research runs should use run_exp123_v6_cuda.sh or
run_exp123_v6_mps.sh.
"""

from __future__ import annotations

import json
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from .artifacts import load_artifact, save_artifact
from .audit import (
    attach_eicu_harmonizer_audit,
    save_dataset_audit,
    save_preprocessing_audit,
)
from .config import resolve_path
from .constants import MODEL_DISPLAY_NAMES, MODEL_ORDER
from .data import (
    ClinicalDataset,
    DatasetSplits,
    create_common_splits,
    load_clinical_dataset,
    save_splits,
)
from .evaluation import (
    evaluate_and_save_split,
    evaluate_external_audit_subgroups,
)
from .explainability import save_explainability
from .metrics import select_youden_threshold
from .models import build_sklearn_model
from .preprocessing import (
    build_preprocessor,
    transform_features,
    validate_training_features,
)
from .summary import save_run_summary
from .tensorboard_logging import (
    create_summary_writer,
    log_json_payload,
    log_metadata,
)
from .torch_model import (
    predict_torch_probabilities,
    select_device,
    train_logistic_regression,
)
from .utils import (
    atomic_write_json,
    environment_info,
    set_global_seed,
    sha256_file,
)


def _prediction_metadata(
    dataset: ClinicalDataset,
    indices: np.ndarray | None = None,
) -> pd.DataFrame:
    if indices is None:
        indices = np.arange(len(dataset.frame), dtype=np.int64)
    output = pd.DataFrame(
        {
            "dataset_row_index": indices,
            "source_row_index": dataset.source_row_index[indices],
            "LOS": dataset.los[indices],
        }
    )
    selected_frame = dataset.frame.iloc[indices]
    for column in dataset.identifier_columns:
        output[column] = selected_frame[column].to_numpy()
    for column in dataset.frame.columns:
        if column.startswith("_audit_"):
            output[column.removeprefix("_audit_")] = selected_frame[
                column
            ].to_numpy()
    return output


def _class_distribution(y: np.ndarray) -> dict[str, Any]:
    labels, counts = np.unique(y, return_counts=True)
    count_map = {
        str(int(label)): int(count)
        for label, count in zip(labels, counts)
    }
    positive_count = int(count_map.get("1", 0))
    negative_count = int(count_map.get("0", 0))
    sample_count = int(len(y))
    return {
        "counts": count_map,
        "sample_count": sample_count,
        "positive_count": positive_count,
        "negative_count": negative_count,
        "positive_rate": (
            float(positive_count / sample_count)
            if sample_count
            else float("nan")
        ),
    }


def _train_model(
    model_name: str,
    model_config: dict[str, Any],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_validation: np.ndarray,
    y_validation: np.ndarray,
    writer: Any,
    seed: int,
):
    if model_name == "logistic_regression":
        result = train_logistic_regression(
            X_train,
            y_train,
            X_validation,
            y_validation,
            model_config,
            writer,
            seed,
        )
        history = pd.DataFrame(result.history)
        return result.model, {
            "best_epoch": result.best_epoch,
            "history": history,
            "parameters": model_config,
        }

    model = build_sklearn_model(model_name, model_config, seed)
    model.fit(X_train, y_train)
    return model, {
        "parameters": model.get_params(),
    }


def _in_memory_probability(
    model_name: str,
    model: Any,
    X: np.ndarray,
    model_config: dict[str, Any],
) -> np.ndarray:
    if model_name == "logistic_regression":
        device = select_device(str(model_config.get("device", "auto")))
        return predict_torch_probabilities(model, X, device)
    output = np.asarray(model.predict_proba(X))
    return output[:, 1] if output.ndim == 2 else output.reshape(-1)


def _save_training_history(training_info: dict[str, Any], output_dir: Path) -> None:
    history = training_info.pop("history", None)
    if isinstance(history, pd.DataFrame):
        history.to_csv(output_dir / "training_history.csv", index=False)
    atomic_write_json(output_dir / "training_info.json", training_info)


def _enabled_models(config: dict[str, Any], selected: list[str] | None) -> list[str]:
    models = []
    for name in MODEL_ORDER:
        if selected is not None and name not in selected:
            continue
        if bool(config["models"].get(name, {}).get("enabled", False)):
            models.append(name)
    return models


def run_experiments(
    config: dict[str, Any],
    selected_models: list[str] | None = None,
    selected_thresholds: list[float] | None = None,
    run_id: str | None = None,
    dry_run: bool = False,
) -> Path:
    seed = int(config["split"].get("seed", 42))
    set_global_seed(seed)

    mimic_path = resolve_path(config, config["paths"]["mimic_csv"])
    eicu_path = resolve_path(config, config["paths"]["eicu_csv"])
    runs_root = resolve_path(config, config["paths"].get("runs_dir", "runs"))
    run_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = runs_root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    feature_names = list(config["data"]["feature_columns"])
    target_column = str(config["data"].get("target_column", "LOS"))
    identifier_columns = list(
        config["data"].get(
            "identifier_columns",
            ["subject_id", "hadm_id", "stay_id"],
        )
    )
    patient_id_column = config["data"].get(
        "patient_id_column", "subject_id"
    )
    thresholds = [
        float(value) for value in config["data"]["threshold_days"]
    ]
    if selected_thresholds is not None:
        allowed = {float(value) for value in selected_thresholds}
        thresholds = [value for value in thresholds if value in allowed]
    models = _enabled_models(config, selected_models)

    if not thresholds:
        raise ValueError("No LOS thresholds were selected.")
    if not models:
        raise ValueError("No models were selected.")

    print(f"[INFO] Loading MIMIC-IV dataset: {mimic_path}")
    mimic = load_clinical_dataset(
        mimic_path,
        feature_names,
        target_column,
        identifier_columns,
        outcomes_config=config.get("outcomes"),
        dataset_name="mimic",
    )
    print(f"[INFO] Loading eICU dataset: {eicu_path}")
    eicu = load_clinical_dataset(
        eicu_path,
        feature_names,
        target_column,
        identifier_columns,
        outcomes_config=config.get("outcomes"),
        dataset_name="eicu",
    )

    audit_config = dict(config.get("audit", {}))
    audit_dir = run_dir / "data_audit"
    provenance_config = dict(
        audit_config.get("eicu_harmonizer_provenance", {})
    )
    # Backward-compatible legacy keys are read only when provenance is
    # explicitly enabled. They never make a Parquet file a required input.
    provenance_enabled = bool(provenance_config.get("enabled", False))
    provenance_path = provenance_config.get("path")
    if provenance_path is None and provenance_enabled:
        provenance_path = config.get("paths", {}).get(
            "eicu_harmonizer_provenance_csv"
        )
    eicu_harmonizer_audit = attach_eicu_harmonizer_audit(
        dataset=eicu,
        configured_path=provenance_path,
        project_root=Path(config["_project_root"]),
        output_dir=audit_dir,
        imputed_feature_denominator=int(
            provenance_config.get(
                "imputed_feature_denominator",
                audit_config.get("eicu_imputed_feature_denominator", 18),
            )
        ),
        required=bool(provenance_config.get("required", False)),
        enabled=provenance_enabled,
        auto_discovery=bool(provenance_config.get("auto_discovery", False)),
    )

    splits = create_common_splits(
        dataset=mimic,
        thresholds=thresholds,
        train_ratio=float(config["split"]["train_ratio"]),
        validation_ratio=float(config["split"]["validation_ratio"]),
        test_ratio=float(config["split"]["test_ratio"]),
        seed=seed,
        max_attempts=int(config["split"].get("max_attempts", 100)),
        patient_id_column=str(patient_id_column)
        if patient_id_column
        else None,
    )
    save_splits(mimic, splits, run_dir / "splits")

    strict_columns = list(
        audit_config.get("strict_non_missing_columns", [])
    )
    fail_on_strict_missing = bool(
        audit_config.get("fail_on_strict_missing", False)
    )
    mimic_audit = save_dataset_audit(
        dataset_name="mimic",
        dataset=mimic,
        thresholds=thresholds,
        output_dir=audit_dir,
        splits=splits,
        strict_non_missing_columns=strict_columns,
        fail_on_strict_missing=fail_on_strict_missing,
    )
    eicu_audit = save_dataset_audit(
        dataset_name="eicu",
        dataset=eicu,
        thresholds=thresholds,
        output_dir=audit_dir,
        splits=None,
        strict_non_missing_columns=strict_columns,
        fail_on_strict_missing=fail_on_strict_missing,
    )

    manifest = {
        "run_id": run_id,
        "created_at": datetime.now().isoformat(),
        "config": config,
        "environment": environment_info(),
        "datasets": {
            "mimic": {
                "path": str(mimic_path),
                "sha256": sha256_file(mimic_path),
                "rows": len(mimic.frame),
                "removed_repeated_headers": mimic.removed_repeated_headers,
                "removed_missing_targets": mimic.removed_missing_targets,
            },
            "eicu": {
                "path": str(eicu_path),
                "sha256": sha256_file(eicu_path),
                "rows": len(eicu.frame),
                "removed_repeated_headers": eicu.removed_repeated_headers,
                "removed_missing_targets": eicu.removed_missing_targets,
            },
        },
        "split": {
            "seed_used": splits.seed_used,
            "grouping_column": splits.grouping_column,
            "stratification_method": splits.stratification_method,
            "train_rows": len(splits.train_indices),
            "validation_rows": len(splits.validation_indices),
            "test_rows": len(splits.test_indices),
        },
        "thresholds": thresholds,
        "models": models,
        "audit": {
            "configuration": audit_config,
            "mimic": mimic_audit,
            "eicu": eicu_audit,
            "eicu_harmonizer": eicu_harmonizer_audit,
        },
    }
    atomic_write_json(run_dir / "run_manifest.json", manifest)

    if dry_run:
        print("[INFO] Dry run completed. Paths, columns, and splits are valid.")
        return run_dir

    X_mimic = mimic.X
    X_eicu = eicu.X
    split_map = splits.as_dict()
    records: list[dict[str, Any]] = []
    default_threshold = float(
        config["evaluation"].get("default_decision_threshold", 0.5)
    )
    calibration_bins = int(
        config["evaluation"].get("calibration_bins", 10)
    )
    bootstrap_repeats = int(
        config["evaluation"].get("external_bootstrap_repeats", 500)
    )
    stop_on_error = bool(config["execution"].get("stop_on_error", False))

    for threshold_days in thresholds:
        threshold_label = f"LOS_{int(threshold_days):02d}"
        y_mimic = mimic.labels(threshold_days)
        y_eicu = eicu.labels(threshold_days)

        for model_name in models:
            experiment_dir = run_dir / threshold_label / model_name
            experiment_dir.mkdir(parents=True, exist_ok=True)
            status_path = experiment_dir / "status.json"
            writer = create_summary_writer(
                run_dir / "tensorboard" / threshold_label / model_name
            )
            record: dict[str, Any] = {
                "threshold_days": threshold_days,
                "model_name": model_name,
                "status": "running",
                "experiment_dir": str(experiment_dir),
                "evaluations": {},
            }

            try:
                experiment_start = time.perf_counter()
                print(
                    f"[INFO] Training {MODEL_DISPLAY_NAMES[model_name]} "
                    f"for LOS >= {threshold_days:g} days"
                )
                train_indices = split_map["train"]
                validation_indices = split_map["validation"]
                test_indices = split_map["test"]

                y_train = y_mimic[train_indices]
                y_validation = y_mimic[validation_indices]
                y_test = y_mimic[test_indices]

                class_distributions = {
                    "train": _class_distribution(y_train),
                    "validation": _class_distribution(y_validation),
                    "internal_test": _class_distribution(y_test),
                    "external_eicu": _class_distribution(y_eicu),
                }
                log_metadata(
                    writer,
                    {
                        "threshold_days": threshold_days,
                        "model_name": model_name,
                        "class_distributions": class_distributions,
                        "feature_names": feature_names,
                    },
                )

                X_train_raw = X_mimic.iloc[train_indices]
                X_validation_raw = X_mimic.iloc[validation_indices]
                X_test_raw = X_mimic.iloc[test_indices]
                X_eicu_raw = X_eicu
                raw_split_frames = {
                    "train": X_train_raw,
                    "validation": X_validation_raw,
                    "internal_test": X_test_raw,
                    "external_eicu": X_eicu_raw,
                }
                validate_training_features(X_train_raw, feature_names)

                preprocessor = build_preprocessor(model_name)
                preprocessor.fit(X_train_raw)
                preprocessing_audit = save_preprocessing_audit(
                    model_name=model_name,
                    feature_names=feature_names,
                    preprocessor=preprocessor,
                    raw_split_frames=raw_split_frames,
                    output_dir=experiment_dir / "data_audit",
                )
                record["preprocessing_audit"] = preprocessing_audit
                log_json_payload(
                    writer,
                    "data_audit/preprocessing",
                    preprocessing_audit,
                )
                log_json_payload(
                    writer,
                    "data_audit/eicu_harmonizer",
                    eicu_harmonizer_audit,
                )
                X_train = transform_features(preprocessor, X_train_raw)
                X_validation = transform_features(
                    preprocessor, X_validation_raw
                )

                model_config = dict(config["models"][model_name])
                training_start = time.perf_counter()
                model, training_info = _train_model(
                    model_name,
                    model_config,
                    X_train,
                    y_train,
                    X_validation,
                    y_validation,
                    writer,
                    seed,
                )
                training_seconds = time.perf_counter() - training_start
                record["training_seconds"] = training_seconds
                _save_training_history(training_info, experiment_dir)

                validation_memory_start = time.perf_counter()
                validation_probability_memory = _in_memory_probability(
                    model_name,
                    model,
                    X_validation,
                    model_config,
                )
                validation_memory_seconds = (
                    time.perf_counter() - validation_memory_start
                )
                threshold_start = time.perf_counter()
                selected_decision_threshold = select_youden_threshold(
                    y_validation,
                    validation_probability_memory,
                )
                threshold_selection_seconds = (
                    time.perf_counter() - threshold_start
                )
                record["validation_in_memory_inference_seconds"] = (
                    validation_memory_seconds
                )
                record["threshold_selection_seconds"] = (
                    threshold_selection_seconds
                )
                record["validation_decision_threshold"] = (
                    selected_decision_threshold
                )

                artifact_metadata = {
                    "model_name": model_name,
                    "model_display_name": MODEL_DISPLAY_NAMES[model_name],
                    "feature_names": feature_names,
                    "target_column": target_column,
                    "threshold_days": threshold_days,
                    "validation_decision_threshold": selected_decision_threshold,
                    "default_decision_threshold": default_threshold,
                    "training_seed": seed,
                    "training_rows": len(train_indices),
                    "validation_rows": len(validation_indices),
                    "internal_test_rows": len(test_indices),
                    "class_distributions": class_distributions,
                    "training_seconds": training_seconds,
                    "inference_device": model_config.get("device", "auto"),
                    "model_config": model_config,
                    "preprocessing_audit": preprocessing_audit,
                    "eicu_harmonizer_audit_status": (
                        eicu_harmonizer_audit.get("status")
                    ),
                    "mimic_dataset_sha256": manifest["datasets"]["mimic"][
                        "sha256"
                    ],
                }
                artifact_dir = experiment_dir / "artifact"
                save_artifact(
                    artifact_dir,
                    model_name,
                    model,
                    preprocessor,
                    artifact_metadata,
                )

                # Reload the saved artifact before final inference.
                reload_start = time.perf_counter()
                loaded = load_artifact(artifact_dir)
                artifact_reload_seconds = time.perf_counter() - reload_start
                record["artifact_reload_seconds"] = artifact_reload_seconds

                probabilities: dict[str, np.ndarray] = {}
                inference_seconds_by_split: dict[str, float] = {}

                for split_name, raw_frame in raw_split_frames.items():
                    split_inference_start = time.perf_counter()
                    probabilities[split_name] = loaded.predict_proba(raw_frame)
                    inference_seconds_by_split[split_name] = (
                        time.perf_counter() - split_inference_start
                    )

                validation_probability_loaded = probabilities["validation"]
                if not np.allclose(
                    validation_probability_memory,
                    validation_probability_loaded,
                    rtol=1e-5,
                    atol=1e-6,
                ):
                    maximum_difference = float(
                        np.max(
                            np.abs(
                                validation_probability_memory
                                - validation_probability_loaded
                            )
                        )
                    )
                    raise RuntimeError(
                        "Reloaded artifact predictions differ from the "
                        f"in-memory model. Max difference={maximum_difference}"
                    )

                record["inference_seconds_by_split"] = (
                    inference_seconds_by_split
                )
                record["inference_seconds"] = float(
                    sum(inference_seconds_by_split.values())
                )

                split_payloads = {
                    "train": (
                        y_train,
                        _prediction_metadata(mimic, train_indices),
                        0,
                    ),
                    "validation": (
                        y_validation,
                        _prediction_metadata(mimic, validation_indices),
                        0,
                    ),
                    "internal_test": (
                        y_test,
                        _prediction_metadata(mimic, test_indices),
                        0,
                    ),
                    "external_eicu": (
                        y_eicu,
                        _prediction_metadata(eicu),
                        bootstrap_repeats,
                    ),
                }

                evaluation_seconds_by_split: dict[str, float] = {}

                for split_name, (
                    labels,
                    metadata_frame,
                    split_bootstrap_repeats,
                ) in split_payloads.items():
                    evaluation_start = time.perf_counter()
                    evaluation = evaluate_and_save_split(
                        split_name=split_name,
                        y_true=labels,
                        probability=probabilities[split_name],
                        selected_threshold=selected_decision_threshold,
                        default_threshold=default_threshold,
                        calibration_bins=calibration_bins,
                        output_dir=experiment_dir / "evaluation" / split_name,
                        prediction_metadata=metadata_frame,
                        writer=writer,
                        bootstrap_repeats=split_bootstrap_repeats,
                        bootstrap_seed=seed,
                        save_plots=bool(
                            config["evaluation"].get("save_plots", True)
                        ),
                        inference_seconds=inference_seconds_by_split[split_name],
                    )
                    evaluation_seconds_by_split[split_name] = (
                        time.perf_counter() - evaluation_start
                    )
                    evaluation["timing"]["evaluation_seconds"] = (
                        evaluation_seconds_by_split[split_name]
                    )
                    atomic_write_json(
                        experiment_dir
                        / "evaluation"
                        / split_name
                        / "metrics.json",
                        evaluation,
                    )
                    record["evaluations"][split_name] = evaluation

                    if split_name == "external_eicu":
                        subgroup_result = evaluate_external_audit_subgroups(
                            prediction_metadata=metadata_frame,
                            y_true=labels,
                            probability=probabilities[split_name],
                            selected_threshold=selected_decision_threshold,
                            default_threshold=default_threshold,
                            calibration_bins=calibration_bins,
                            output_dir=(
                                experiment_dir
                                / "evaluation"
                                / split_name
                            ),
                        )
                        record["external_eicu_fallback_subgroups"] = (
                            subgroup_result
                        )

                record["evaluation_seconds_by_split"] = (
                    evaluation_seconds_by_split
                )

                # Explain the saved model using the internal test split.
                X_test_transformed = transform_features(
                    loaded.preprocessor,
                    X_mimic.iloc[test_indices],
                )
                X_background = transform_features(
                    loaded.preprocessor,
                    X_train_raw,
                )
                explainability_start = time.perf_counter()
                explainability = save_explainability(
                    model_name=model_name,
                    model=loaded.model,
                    X_background=X_background,
                    X_test=X_test_transformed,
                    feature_names=feature_names,
                    output_dir=experiment_dir / "explainability",
                    config=config["explainability"],
                )
                record["explainability_seconds"] = (
                    time.perf_counter() - explainability_start
                )
                record["explainability"] = explainability
                record["total_experiment_seconds"] = (
                    time.perf_counter() - experiment_start
                )

                record["status"] = "completed"
                atomic_write_json(status_path, record)
                records.append(record)
                writer.flush()
                writer.close()

                internal_metrics = record["evaluations"]["internal_test"][
                    "threshold_views"
                ]["validation_selected"]
                external_metrics = record["evaluations"]["external_eicu"][
                    "threshold_views"
                ]["validation_selected"]
                print(
                    f"[OK] {threshold_label}/{model_name}: "
                    f"internal AUROC={internal_metrics['auroc']:.4f}, "
                    f"external AUROC={external_metrics['auroc']:.4f}"
                )

            except Exception as exception:
                if "experiment_start" in locals():
                    record["total_experiment_seconds"] = (
                        time.perf_counter() - experiment_start
                    )
                record["status"] = "failed"
                record["error"] = repr(exception)
                record["traceback"] = traceback.format_exc()
                if "validation_decision_threshold" not in record:
                    record["validation_decision_threshold"] = float("nan")
                atomic_write_json(status_path, record)
                records.append(record)
                writer.close()
                print(
                    f"[ERROR] {threshold_label}/{model_name}: {exception}"
                )
                if stop_on_error:
                    save_run_summary(run_dir, records)
                    raise

            save_run_summary(run_dir, records)

    atomic_write_json(run_dir / "experiment_records.json", records)
    save_run_summary(run_dir, records)
    print(f"[INFO] All requested experiments finished: {run_dir}")
    return run_dir
