from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .artifacts import load_task_artifact
from .constants import OUTCOME_CLASS_LABELS
from .evaluation_multiclass import (
    BootstrapSettings,
    calculate_multiclass_metrics,
    probabilities_from_prediction_frame,
    write_multiclass_evaluation_artifacts,
)
from .oracle_pipeline import ConfirmatoryCohort, load_confirmatory_cohort
from .task_spec import EXP1_FLAT_3CLASS_TASK
from .utils import atomic_write_json, environment_info, sha256_file
from .progress import progress_iter
from .final_comparison_plots import (
    _plot_model_metric_comparison,
    _plot_paired_differences,
)
from .final_comparison_stats import (
    DelongTestResult,
    _apply_fdr,
    delong_roc_test,
    mcnemar_correctness_test,
    paired_cluster_bootstrap_differences,
)


CONFIRMATORY_SPLITS = {"internal_test", "external_eicu"}
DEFAULT_COMPARISON_METRICS = (
    "macro_auroc",
    "macro_auprc",
    "macro_f1",
    "balanced_accuracy",
    "multiclass_brier",
    "multiclass_log_loss",
)
LOWER_IS_BETTER = {"multiclass_brier", "multiclass_log_loss"}


@dataclass(frozen=True)
class FinalComparisonSettings:
    bootstrap_repeats: int = 500
    confidence_level: float = 0.95
    seed: int = 42
    fdr_method: str = "fdr_bh"
    fdr_alpha: float = 0.05
    minimum_successful_bootstrap_repeats: int = 100
    comparison_metrics: tuple[str, ...] = DEFAULT_COMPARISON_METRICS

    def __post_init__(self) -> None:
        if self.bootstrap_repeats < 1:
            raise ValueError("bootstrap_repeats must be positive.")
        if not 0.0 < self.confidence_level < 1.0:
            raise ValueError("confidence_level must be between zero and one.")
        if self.minimum_successful_bootstrap_repeats < 1:
            raise ValueError(
                "minimum_successful_bootstrap_repeats must be positive."
            )
        if not self.comparison_metrics:
            raise ValueError("At least one comparison metric is required.")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FinalComparisonRunResult:
    split_name: str
    output_dir: str
    sample_count: int
    representative_model: str
    flat_macro_auroc: float
    pipeline_macro_auroc: float
    flat_macro_f1: float
    pipeline_macro_f1: float
    paired_bootstrap_repeats: int
    inferential_test_count: int
    completion_lock_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a JSON object in {path}")
    return payload


def _validate_recorded_sha256(path: Path) -> str:
    digest = sha256_file(path)
    digest_path = path.with_name(path.name + ".sha256")
    if not digest_path.is_file():
        raise FileNotFoundError(digest_path)
    recorded = digest_path.read_text(encoding="utf-8").strip()
    if digest != recorded:
        raise ValueError(f"SHA-256 mismatch for {path}")
    return digest


def _load_exp1_representative_model(run_dir: Path) -> tuple[str, dict[str, Any], str]:
    representative_path = run_dir / "EXP1_flat_3class" / "representative_model.json"
    representative_sha256 = _validate_recorded_sha256(representative_path)
    payload = _read_json(representative_path)
    if payload.get("task_name") != EXP1_FLAT_3CLASS_TASK.task_name:
        raise ValueError("EXP1 representative model has the wrong task.")
    if payload.get("selected_on_split") != "validation":
        raise ValueError("EXP1 representative model was not selected on Validation.")
    if payload.get("confirmatory_splits_accessed") is not False:
        raise ValueError(
            "EXP1 representative selection must not access confirmatory splits."
        )
    model_name = str(payload.get("selected_model"))
    if not model_name:
        raise ValueError("EXP1 representative model name is missing.")
    return model_name, payload, representative_sha256


def _flat_confirmatory_prediction_frame(
    cohort: ConfirmatoryCohort,
    probabilities: np.ndarray,
    model_name: str,
) -> pd.DataFrame:
    values = np.asarray(probabilities, dtype=np.float64)
    if values.shape != (len(cohort.y_true_3class), 3):
        raise ValueError(
            "EXP1 confirmatory probabilities have the wrong shape: "
            f"{values.shape}"
        )
    output = cohort.metadata.copy().reset_index(drop=True)
    output["split"] = cohort.split_name
    output["task_name"] = EXP1_FLAT_3CLASS_TASK.task_name
    output["model_name"] = model_name
    output["y_true"] = cohort.y_true_3class
    for class_code, class_name in OUTCOME_CLASS_LABELS.items():
        output[f"prob_class_{class_code}"] = values[:, class_code]
        output[f"prob_{class_name}"] = values[:, class_code]
    output["predicted_class"] = np.argmax(values, axis=1).astype(np.int64)
    output["correct"] = (
        output["predicted_class"].to_numpy(dtype=np.int64)
        == cohort.y_true_3class
    ).astype(np.int64)
    return output


def _load_pipeline_predictions(run_dir: Path, split_name: str) -> pd.DataFrame:
    path = (
        run_dir
        / "EXP2_two_stage"
        / "pipeline_evaluation"
        / split_name
        / "pipeline_3class"
        / "predictions.csv.gz"
    )
    if not path.is_file():
        raise FileNotFoundError(
            f"Pipeline confirmatory predictions were not found: {path}"
        )
    frame = pd.read_csv(path, compression="infer", low_memory=False)
    required = {
        "source_row_index",
        "y_true_3class",
        "pipeline_predicted_class",
        "prob_class_0",
        "prob_class_1",
        "prob_class_2",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise KeyError(f"Pipeline prediction file is missing columns: {missing}")
    return frame


def align_flat_and_pipeline_predictions(
    flat: pd.DataFrame,
    pipeline: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    for name, frame in (("flat", flat), ("pipeline", pipeline)):
        if frame["source_row_index"].duplicated().any():
            raise ValueError(f"{name} predictions contain duplicate source_row_index.")
    flat_indices = set(flat["source_row_index"].astype(int).tolist())
    pipeline_indices = set(pipeline["source_row_index"].astype(int).tolist())
    if flat_indices != pipeline_indices:
        missing_from_pipeline = sorted(flat_indices - pipeline_indices)
        missing_from_flat = sorted(pipeline_indices - flat_indices)
        raise ValueError(
            "Flat and Pipeline predictions do not contain the same samples. "
            f"Missing from pipeline={len(missing_from_pipeline)}, "
            f"missing from flat={len(missing_from_flat)}"
        )
    flat_sorted = flat.sort_values("source_row_index", kind="mergesort").reset_index(
        drop=True
    )
    pipeline_sorted = pipeline.sort_values(
        "source_row_index", kind="mergesort"
    ).reset_index(drop=True)
    if not np.array_equal(
        flat_sorted["source_row_index"].to_numpy(dtype=np.int64),
        pipeline_sorted["source_row_index"].to_numpy(dtype=np.int64),
    ):
        raise RuntimeError("Prediction alignment failed after sorting.")
    flat_y = flat_sorted["y_true"].to_numpy(dtype=np.int64)
    pipeline_y = pipeline_sorted["y_true_3class"].to_numpy(dtype=np.int64)
    if not np.array_equal(flat_y, pipeline_y):
        raise ValueError("Flat and Pipeline y_true values differ.")
    return flat_sorted, pipeline_sorted


def _load_oracle_metrics(run_dir: Path, split_name: str) -> dict[str, Any]:
    path = (
        run_dir
        / "EXP2_two_stage"
        / "pipeline_evaluation"
        / split_name
        / "oracle_stage2"
        / "evaluation"
        / "binary_metrics.json"
    )
    payload = _read_json(path)
    metrics = payload.get("threshold_views", {}).get("validation_selected")
    if not isinstance(metrics, dict):
        metrics = payload.get("metrics")
    if not isinstance(metrics, dict):
        raise KeyError(f"Could not locate Oracle Stage2 metrics in {path}")
    return metrics


def _manuscript_main_table(
    *,
    split_name: str,
    representative_model: str,
    flat_metrics: Mapping[str, Any],
    pipeline_metrics: Mapping[str, Any],
    oracle_metrics: Mapping[str, Any],
) -> pd.DataFrame:
    rows = [
        {
            "split": split_name,
            "analysis": "EXP1_flat_3class",
            "model": representative_model,
            "population": "full_cohort",
            "task": "A_vs_B_vs_C",
            "sample_count": flat_metrics.get("sample_count"),
            "macro_auroc": flat_metrics.get("macro_auroc"),
            "macro_auprc": flat_metrics.get("macro_auprc"),
            "macro_f1": flat_metrics.get("macro_f1"),
            "balanced_accuracy": flat_metrics.get("balanced_accuracy"),
            "multiclass_brier": flat_metrics.get("multiclass_brier"),
            "multiclass_log_loss": flat_metrics.get("multiclass_log_loss"),
            "binary_auroc": np.nan,
            "binary_auprc": np.nan,
            "binary_brier": np.nan,
            "binary_f1": np.nan,
        },
        {
            "split": split_name,
            "analysis": "EXP2_pipeline_3class",
            "model": "stage1_and_stage2_super_learner",
            "population": "full_cohort",
            "task": "A_vs_B_vs_C",
            "sample_count": pipeline_metrics.get("sample_count"),
            "macro_auroc": pipeline_metrics.get("macro_auroc"),
            "macro_auprc": pipeline_metrics.get("macro_auprc"),
            "macro_f1": pipeline_metrics.get("macro_f1"),
            "balanced_accuracy": pipeline_metrics.get("balanced_accuracy"),
            "multiclass_brier": pipeline_metrics.get("multiclass_brier"),
            "multiclass_log_loss": pipeline_metrics.get("multiclass_log_loss"),
            "binary_auroc": np.nan,
            "binary_auprc": np.nan,
            "binary_brier": np.nan,
            "binary_f1": np.nan,
        },
        {
            "split": split_name,
            "analysis": "EXP2_oracle_equals_EXP3",
            "model": "stage2_super_learner",
            "population": "ground_truth_survivors_A_or_C",
            "task": "A_vs_C",
            "sample_count": oracle_metrics.get("sample_count"),
            "macro_auroc": np.nan,
            "macro_auprc": np.nan,
            "macro_f1": np.nan,
            "balanced_accuracy": oracle_metrics.get("balanced_accuracy"),
            "multiclass_brier": np.nan,
            "multiclass_log_loss": np.nan,
            "binary_auroc": oracle_metrics.get("auroc"),
            "binary_auprc": oracle_metrics.get("auprc"),
            "binary_brier": oracle_metrics.get("brier_score"),
            "binary_f1": oracle_metrics.get("f1_score"),
        },
    ]
    return pd.DataFrame(rows)


def _individual_bootstrap_long_table(
    *,
    split_name: str,
    analysis: str,
    evaluation_dir: Path,
) -> pd.DataFrame:
    metrics = _read_json(evaluation_dir / "multiclass_metrics.json")
    bootstrap_path = evaluation_dir / "bootstrap_confidence_intervals.csv"
    intervals = (
        pd.read_csv(bootstrap_path)
        if bootstrap_path.is_file()
        else pd.DataFrame()
    )
    interval_lookup = {
        str(row["metric"]): row
        for row in intervals.to_dict(orient="records")
    }
    rows: list[dict[str, Any]] = []
    for metric in DEFAULT_COMPARISON_METRICS:
        if metric not in metrics or metrics[metric] is None:
            continue
        interval = interval_lookup.get(metric, {})
        rows.append(
            {
                "split": split_name,
                "analysis": analysis,
                "metric": metric,
                "point_estimate": float(metrics[metric]),
                "ci_lower": interval.get("lower", np.nan),
                "ci_upper": interval.get("upper", np.nan),
                "confidence_level": interval.get("confidence_level", np.nan),
                "successful_repeats": interval.get("successful_repeats", np.nan),
            }
        )
    return pd.DataFrame(rows)


def _oracle_bootstrap_long_table(
    *,
    run_dir: Path,
    split_name: str,
) -> pd.DataFrame:
    evaluation_dir = (
        run_dir
        / "EXP2_two_stage"
        / "pipeline_evaluation"
        / split_name
        / "oracle_stage2"
        / "evaluation"
    )
    payload = _read_json(evaluation_dir / "binary_metrics.json")
    threshold_metrics = payload.get("threshold_views", {}).get("validation_selected", {})
    intervals = payload.get("bootstrap_95_confidence_intervals", {})
    metric_mapping = {
        "auroc": "auroc",
        "auprc": "auprc",
        "brier_score": "brier_score",
        "f1_score": "f1_score",
        "balanced_accuracy": "balanced_accuracy",
    }
    rows = []
    for output_metric, source_metric in metric_mapping.items():
        value = threshold_metrics.get(source_metric)
        if value is None:
            continue
        interval = intervals.get(source_metric, {})
        rows.append(
            {
                "split": split_name,
                "analysis": "EXP2_oracle_equals_EXP3",
                "metric": output_metric,
                "point_estimate": float(value),
                "ci_lower": interval.get("lower_95", np.nan),
                "ci_upper": interval.get("upper_95", np.nan),
                "confidence_level": 0.95,
                "successful_repeats": interval.get("valid_repeats", np.nan),
            }
        )
    return pd.DataFrame(rows)


def _refresh_aggregate_outputs(run_dir: Path) -> None:
    root = run_dir / "final_comparison"
    manuscript_root = root / "manuscript_tables"
    manuscript_root.mkdir(parents=True, exist_ok=True)
    main_tables = []
    metric_tables = []
    test_tables = []
    bootstrap_tables = []
    for split_name in sorted(CONFIRMATORY_SPLITS):
        split_dir = root / split_name
        main_path = split_dir / "manuscript_tables" / "main_comparison_table.csv"
        metric_path = split_dir / "flat_vs_pipeline_metrics.csv"
        test_path = split_dir / "statistical_tests.csv"
        ci_path = split_dir / "individual_model_bootstrap_intervals.csv"
        if main_path.is_file():
            table = pd.read_csv(main_path)
            table.to_csv(
                manuscript_root / f"{split_name}_main_table.csv", index=False
            )
            main_tables.append(table)
        if metric_path.is_file():
            metric_tables.append(pd.read_csv(metric_path))
        if test_path.is_file():
            test_tables.append(pd.read_csv(test_path))
        if ci_path.is_file():
            bootstrap_tables.append(pd.read_csv(ci_path))
    if main_tables:
        pd.concat(main_tables, ignore_index=True).to_csv(
            manuscript_root / "combined_main_comparison_table.csv", index=False
        )
    if metric_tables:
        pd.concat(metric_tables, ignore_index=True).to_csv(
            root / "comparison_exp1_vs_exp2_pipeline.csv", index=False
        )
    if test_tables:
        pd.concat(test_tables, ignore_index=True).to_csv(
            root / "statistical_tests_combined.csv", index=False
        )
    if bootstrap_tables:
        pd.concat(bootstrap_tables, ignore_index=True).to_csv(
            manuscript_root / "combined_bootstrap_intervals.csv", index=False
        )
    if main_tables:
        combined = pd.concat(main_tables, ignore_index=True)
        descriptive = combined.loc[
            combined["analysis"].isin(
                ["EXP2_pipeline_3class", "EXP2_oracle_equals_EXP3"]
            )
        ].copy()
        descriptive["comparison_note"] = (
            "Descriptive only: Oracle and Pipeline use different populations and tasks."
        )
        descriptive.to_csv(
            root / "comparison_oracle_vs_pipeline_descriptive.csv", index=False
        )


def final_comparison_output_dir(run_dir: Path, split_name: str) -> Path:
    return run_dir / "final_comparison" / split_name


def run_final_statistical_comparison_for_cohort(
    *,
    run_dir: Path,
    cohort: ConfirmatoryCohort,
    config: Mapping[str, Any],
) -> FinalComparisonRunResult:
    run_dir = run_dir.expanduser().resolve()
    output_dir = final_comparison_output_dir(run_dir, cohort.split_name)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Final comparison output already exists: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=False)

    representative_model, representative_payload, representative_sha256 = (
        _load_exp1_representative_model(run_dir)
    )
    flat_artifact_dir = (
        run_dir / "EXP1_flat_3class" / representative_model / "artifact"
    )
    flat_artifact = load_task_artifact(flat_artifact_dir)
    if flat_artifact.task_name != EXP1_FLAT_3CLASS_TASK.task_name:
        raise ValueError("EXP1 artifact has the wrong task.")
    if flat_artifact.task_type != "multiclass" or flat_artifact.num_classes != 3:
        raise ValueError("EXP1 artifact is not a three-class model.")
    if tuple(flat_artifact.feature_names) != tuple(cohort.feature_names):
        raise ValueError("EXP1 artifact feature order differs from the cohort.")
    expected_split_hash = representative_payload.get("shared_split_hash")
    if flat_artifact.metadata.get("shared_split_hash") != expected_split_hash:
        raise ValueError("EXP1 artifact and representative design use different splits.")

    flat_probabilities = flat_artifact.predict_proba(cohort.raw_features)
    flat_predictions = _flat_confirmatory_prediction_frame(
        cohort, flat_probabilities, representative_model
    )
    pipeline_predictions = _load_pipeline_predictions(run_dir, cohort.split_name)
    flat_predictions, pipeline_predictions = align_flat_and_pipeline_predictions(
        flat_predictions, pipeline_predictions
    )

    y_true = flat_predictions["y_true"].to_numpy(dtype=np.int64)
    flat_probabilities = probabilities_from_prediction_frame(flat_predictions, 3)
    pipeline_probabilities = probabilities_from_prediction_frame(
        pipeline_predictions, 3
    )
    flat_hard = flat_predictions["predicted_class"].to_numpy(dtype=np.int64)
    pipeline_hard = pipeline_predictions["pipeline_predicted_class"].to_numpy(
        dtype=np.int64
    )
    class_names = tuple(EXP1_FLAT_3CLASS_TASK.class_names)

    settings_payload = dict(config.get("final_comparison", {}))
    settings = FinalComparisonSettings(
        bootstrap_repeats=int(settings_payload.get("bootstrap_repeats", 500)),
        confidence_level=float(settings_payload.get("confidence_level", 0.95)),
        seed=int(settings_payload.get("seed", 42)),
        fdr_method=str(settings_payload.get("fdr_method", "fdr_bh")),
        fdr_alpha=float(settings_payload.get("fdr_alpha", 0.05)),
        minimum_successful_bootstrap_repeats=int(
            settings_payload.get("minimum_successful_bootstrap_repeats", 100)
        ),
        comparison_metrics=tuple(
            settings_payload.get(
                "comparison_metrics", list(DEFAULT_COMPARISON_METRICS)
            )
        ),
    )

    flat_dir = output_dir / "flat_representative"
    pipeline_dir = output_dir / "pipeline"
    flat_dir.mkdir(parents=True, exist_ok=True)
    pipeline_dir.mkdir(parents=True, exist_ok=True)
    flat_predictions.to_csv(flat_dir / "predictions.csv.gz", index=False, compression="gzip")
    pipeline_predictions.to_csv(
        pipeline_dir / "predictions.csv.gz", index=False, compression="gzip"
    )
    evaluation_config = config.get("evaluation", {})
    flat_eval = write_multiclass_evaluation_artifacts(
        y_true=y_true,
        probabilities=flat_probabilities,
        class_names=class_names,
        output_dir=flat_dir / "evaluation",
        split_name=cohort.split_name,
        model_name=f"EXP1_{representative_model}",
        calibration_bins=int(evaluation_config.get("calibration_bins", 10)),
        bootstrap_settings=BootstrapSettings(
            repeats=settings.bootstrap_repeats,
            confidence_level=settings.confidence_level,
            seed=settings.seed,
            minimum_successful_repeats=settings.minimum_successful_bootstrap_repeats,
        ),
        save_plots=bool(evaluation_config.get("save_plots", True)),
        predicted_classes=flat_hard,
        decision_rule="probability_argmax",
    )
    pipeline_eval = write_multiclass_evaluation_artifacts(
        y_true=y_true,
        probabilities=pipeline_probabilities,
        class_names=class_names,
        output_dir=pipeline_dir / "evaluation",
        split_name=cohort.split_name,
        model_name="EXP2_pipeline",
        calibration_bins=int(evaluation_config.get("calibration_bins", 10)),
        bootstrap_settings=BootstrapSettings(
            repeats=settings.bootstrap_repeats,
            confidence_level=settings.confidence_level,
            seed=settings.seed + 1,
            minimum_successful_repeats=settings.minimum_successful_bootstrap_repeats,
        ),
        save_plots=bool(evaluation_config.get("save_plots", True)),
        predicted_classes=pipeline_hard,
        decision_rule="stage1_threshold_then_stage2_threshold",
    )
    flat_metrics = dict(flat_eval["metrics"])
    pipeline_metrics = dict(pipeline_eval["metrics"])

    metric_rows = []
    all_metric_names = sorted(
        set(settings.comparison_metrics)
        | {"accuracy", "weighted_f1"}
    )
    for metric in all_metric_names:
        if metric not in flat_metrics or metric not in pipeline_metrics:
            continue
        metric_rows.append(
            {
                "split": cohort.split_name,
                "metric": metric,
                "flat": flat_metrics[metric],
                "pipeline": pipeline_metrics[metric],
                "difference_pipeline_minus_flat": (
                    float(pipeline_metrics[metric]) - float(flat_metrics[metric])
                ),
            }
        )
    model_comparison = pd.DataFrame(metric_rows)
    model_comparison.to_csv(output_dir / "flat_vs_pipeline_metrics.csv", index=False)

    bootstrap_table, bootstrap_manifest = paired_cluster_bootstrap_differences(
        y_true=y_true,
        flat_probabilities=flat_probabilities,
        flat_predicted_classes=flat_hard,
        pipeline_probabilities=pipeline_probabilities,
        pipeline_predicted_classes=pipeline_hard,
        metadata=flat_predictions,
        class_names=class_names,
        settings=settings,
    )
    bootstrap_table.to_csv(
        output_dir / "paired_bootstrap_differences.csv", index=False
    )
    atomic_write_json(
        output_dir / "paired_bootstrap_manifest.json", bootstrap_manifest
    )

    delong_rows: list[dict[str, Any]] = []
    for class_code, class_name in OUTCOME_CLASS_LABELS.items():
        binary = (y_true == class_code).astype(np.int64)
        result = delong_roc_test(
            binary,
            flat_probabilities[:, class_code],
            pipeline_probabilities[:, class_code],
            class_code=class_code,
            class_name=class_name,
        )
        row = result.to_dict()
        row["test_type"] = "delong_ovr_auroc"
        row["metric"] = f"class_{class_code}_auroc"
        delong_rows.append(row)
    delong_table = pd.DataFrame(delong_rows)
    delong_table.to_csv(output_dir / "delong_class_auroc_tests.csv", index=False)

    mcnemar_result = mcnemar_correctness_test(y_true, flat_hard, pipeline_hard)
    atomic_write_json(output_dir / "mcnemar_correctness.json", mcnemar_result)

    inferential_rows = []
    for row in bootstrap_table.to_dict(orient="records"):
        inferential_rows.append(
            {
                "test_type": row["test_type"],
                "metric": row["metric"],
                "effect": row["improvement_positive_favors_pipeline"],
                "p_raw": row["p_raw"],
            }
        )
    for row in delong_table.to_dict(orient="records"):
        inferential_rows.append(
            {
                "test_type": row["test_type"],
                "metric": row["metric"],
                "effect": row["difference_pipeline_minus_flat"],
                "p_raw": row["p_raw"],
            }
        )
    inferential_rows.append(
        {
            "test_type": mcnemar_result["test_type"],
            "metric": mcnemar_result["metric"],
            "effect": mcnemar_result[
                "difference_accuracy_pipeline_minus_flat"
            ],
            "p_raw": mcnemar_result["p_raw"],
        }
    )
    tests = _apply_fdr(
        pd.DataFrame(inferential_rows),
        method=settings.fdr_method,
        alpha=settings.fdr_alpha,
        family_id=f"{cohort.split_name}_flat_vs_pipeline",
    )
    tests.to_csv(output_dir / "statistical_tests.csv", index=False)
    atomic_write_json(
        output_dir / "statistical_tests.json",
        {
            "split": cohort.split_name,
            "fdr_method": settings.fdr_method,
            "fdr_alpha": settings.fdr_alpha,
            "fdr_family_size": int(tests["fdr_family_size"].iloc[0]),
            "tests": tests.to_dict(orient="records"),
        },
    )

    fdr_lookup = {
        (str(row["test_type"]), str(row["metric"])): row
        for row in tests.to_dict(orient="records")
    }
    for table in (bootstrap_table, delong_table):
        table["p_fdr"] = [
            fdr_lookup[(str(row["test_type"]), str(row["metric"]))]["p_fdr"]
            for row in table.to_dict(orient="records")
        ]
        table["reject_fdr"] = [
            fdr_lookup[(str(row["test_type"]), str(row["metric"]))]["reject_fdr"]
            for row in table.to_dict(orient="records")
        ]
        table["fdr_family_id"] = f"{cohort.split_name}_flat_vs_pipeline"
    bootstrap_table.to_csv(
        output_dir / "paired_bootstrap_differences.csv", index=False
    )
    delong_table.to_csv(
        output_dir / "delong_class_auroc_tests.csv", index=False
    )
    mcnemar_key = (
        str(mcnemar_result["test_type"]),
        str(mcnemar_result["metric"]),
    )
    mcnemar_result["p_fdr"] = fdr_lookup[mcnemar_key]["p_fdr"]
    mcnemar_result["reject_fdr"] = fdr_lookup[mcnemar_key]["reject_fdr"]
    mcnemar_result["fdr_family_id"] = f"{cohort.split_name}_flat_vs_pipeline"
    atomic_write_json(output_dir / "mcnemar_correctness.json", mcnemar_result)

    individual_intervals = pd.concat(
        [
            _individual_bootstrap_long_table(
                split_name=cohort.split_name,
                analysis="EXP1_flat_3class",
                evaluation_dir=flat_dir / "evaluation",
            ),
            _individual_bootstrap_long_table(
                split_name=cohort.split_name,
                analysis="EXP2_pipeline_3class",
                evaluation_dir=pipeline_dir / "evaluation",
            ),
            _oracle_bootstrap_long_table(
                run_dir=run_dir, split_name=cohort.split_name
            ),
        ],
        ignore_index=True,
    )
    individual_intervals.to_csv(
        output_dir / "individual_model_bootstrap_intervals.csv", index=False
    )

    oracle_metrics = _load_oracle_metrics(run_dir, cohort.split_name)
    manuscript_table = _manuscript_main_table(
        split_name=cohort.split_name,
        representative_model=representative_model,
        flat_metrics=flat_metrics,
        pipeline_metrics=pipeline_metrics,
        oracle_metrics=oracle_metrics,
    )
    manuscript_dir = output_dir / "manuscript_tables"
    manuscript_dir.mkdir(parents=True, exist_ok=True)
    manuscript_table.to_csv(manuscript_dir / "main_comparison_table.csv", index=False)
    model_comparison.to_csv(
        manuscript_dir / "flat_vs_pipeline_metric_table.csv", index=False
    )
    delong_table.to_csv(
        manuscript_dir / "class_specific_auroc_tests.csv", index=False
    )

    if bool(evaluation_config.get("save_plots", True)):
        figures_dir = output_dir / "figures"
        _plot_model_metric_comparison(
            model_comparison,
            figures_dir / "flat_vs_pipeline_metrics.png",
        )
        _plot_paired_differences(
            bootstrap_table,
            figures_dir / "paired_metric_differences.png",
        )

    flat_artifact_hashes = {
        str(path.relative_to(flat_artifact_dir)): sha256_file(path)
        for path in sorted(flat_artifact_dir.rglob("*"))
        if path.is_file()
    }
    pipeline_prediction_path = (
        run_dir
        / "EXP2_two_stage"
        / "pipeline_evaluation"
        / cohort.split_name
        / "pipeline_3class"
        / "predictions.csv.gz"
    )
    manifest = {
        "step": "final_statistical_comparison",
        "split": cohort.split_name,
        "sample_count": len(y_true),
        "representative_model": representative_model,
        "representative_model_sha256": representative_sha256,
        "flat_artifact_hashes": flat_artifact_hashes,
        "pipeline_prediction_sha256": sha256_file(pipeline_prediction_path),
        "source_dataset_path": cohort.source_dataset_path,
        "source_assignment_path": cohort.source_assignment_path,
        "settings": settings.to_dict(),
        "paired_bootstrap": bootstrap_manifest,
        "delong_assumption_note": (
            "Class-specific DeLong tests operate at the prediction-row level; "
            "paired bootstrap confidence intervals use patient clusters when "
            "subject_id is available."
        ),
        "confirmatory_split_reused_for_design": False,
        "environment": environment_info(),
    }
    atomic_write_json(output_dir / "comparison_manifest.json", manifest)

    completion_payload = {
        "split": cohort.split_name,
        "status": "completed",
        "comparison_manifest_sha256": sha256_file(
            output_dir / "comparison_manifest.json"
        ),
        "flat_predictions_sha256": sha256_file(
            flat_dir / "predictions.csv.gz"
        ),
        "pipeline_predictions_sha256": sha256_file(
            pipeline_dir / "predictions.csv.gz"
        ),
        "statistical_tests_sha256": sha256_file(
            output_dir / "statistical_tests.csv"
        ),
        "main_table_sha256": sha256_file(
            manuscript_dir / "main_comparison_table.csv"
        ),
    }
    completion_path = output_dir / "COMPLETED.lock.json"
    atomic_write_json(completion_path, completion_payload)
    completion_digest = sha256_file(completion_path)
    (completion_path.with_name(completion_path.name + ".sha256")).write_text(
        completion_digest + "\n", encoding="utf-8"
    )

    result = FinalComparisonRunResult(
        split_name=cohort.split_name,
        output_dir=str(output_dir),
        sample_count=len(y_true),
        representative_model=representative_model,
        flat_macro_auroc=float(flat_metrics["macro_auroc"]),
        pipeline_macro_auroc=float(pipeline_metrics["macro_auroc"]),
        flat_macro_f1=float(flat_metrics["macro_f1"]),
        pipeline_macro_f1=float(pipeline_metrics["macro_f1"]),
        paired_bootstrap_repeats=int(bootstrap_manifest["successful_repeats"]),
        inferential_test_count=len(tests),
        completion_lock_sha256=completion_digest,
    )
    atomic_write_json(output_dir / "final_comparison_result.json", result.to_dict())
    _refresh_aggregate_outputs(run_dir)
    return result


def run_final_statistical_comparison(
    *,
    run_dir: Path,
    config: Mapping[str, Any],
    split_name: str,
) -> FinalComparisonRunResult:
    if split_name not in CONFIRMATORY_SPLITS:
        raise ValueError(f"Unsupported final comparison split: {split_name}")
    cohort = load_confirmatory_cohort(
        config=config,
        run_dir=run_dir,
        split_name=split_name,
    )
    return run_final_statistical_comparison_for_cohort(
        run_dir=run_dir,
        cohort=cohort,
        config=config,
    )
