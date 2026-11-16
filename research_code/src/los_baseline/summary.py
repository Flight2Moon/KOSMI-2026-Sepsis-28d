from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


_SELECTED_METRIC_COLUMNS = {
    "auroc": "auroc",
    "auprc": "auprc",
    "accuracy": "accuracy",
    "balanced_accuracy": "balanced_accuracy",
    "f1_score": "f1_score",
    "sensitivity": "sensitivity_recall",
    "specificity": "specificity",
    "ppv": "ppv_precision",
    "npv": "npv",
    "mcc": "matthews_correlation_coefficient",
    "brier_score": "brier_score",
    "log_loss": "log_loss",
    "expected_calibration_error": "expected_calibration_error",
    "maximum_calibration_error": "maximum_calibration_error",
}

_DEFAULT_THRESHOLD_COLUMNS = {
    "accuracy_at_0_5": "accuracy",
    "balanced_accuracy_at_0_5": "balanced_accuracy",
    "f1_at_0_5": "f1_score",
    "sensitivity_at_0_5": "sensitivity_recall",
    "specificity_at_0_5": "specificity",
    "ppv_at_0_5": "ppv_precision",
    "npv_at_0_5": "npv",
    "mcc_at_0_5": "matthews_correlation_coefficient",
}


def _ci_value(
    evaluation: dict[str, Any],
    metric_name: str,
    bound: str,
) -> float | None:
    interval = evaluation.get(
        "bootstrap_95_confidence_intervals",
        {},
    ).get(metric_name, {})
    value = interval.get(bound)
    return float(value) if value is not None else None


def build_summary_rows(experiment_records: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for record in experiment_records:
        inference_times = record.get("inference_seconds_by_split", {})
        evaluation_times = record.get("evaluation_seconds_by_split", {})

        for split_name, evaluation in record.get("evaluations", {}).items():
            selected = evaluation["threshold_views"]["validation_selected"]
            default = evaluation["threshold_views"]["default_0_5"]

            row: dict[str, Any] = {
                "threshold_days": record["threshold_days"],
                "model_name": record["model_name"],
                "split": split_name,
                "status": record["status"],
                "sample_count": selected["sample_count"],
                "positive_count": selected["positive_count"],
                "negative_count": selected["negative_count"],
                "positive_rate": selected["positive_rate"],
                "random_auprc_baseline": selected["positive_rate"],
                "validation_decision_threshold": record[
                    "validation_decision_threshold"
                ],
            }

            for output_name, metric_name in _SELECTED_METRIC_COLUMNS.items():
                row[output_name] = selected[metric_name]

            for output_name, metric_name in _DEFAULT_THRESHOLD_COLUMNS.items():
                row[output_name] = default[metric_name]

            row.update(
                {
                    "auroc_ci_lower_95": _ci_value(
                        evaluation,
                        "auroc",
                        "lower_95",
                    ),
                    "auroc_ci_upper_95": _ci_value(
                        evaluation,
                        "auroc",
                        "upper_95",
                    ),
                    "auprc_ci_lower_95": _ci_value(
                        evaluation,
                        "auprc",
                        "lower_95",
                    ),
                    "auprc_ci_upper_95": _ci_value(
                        evaluation,
                        "auprc",
                        "upper_95",
                    ),
                    "model_training_seconds": record.get("training_seconds"),
                    "split_inference_seconds": inference_times.get(split_name),
                    "split_evaluation_seconds": evaluation_times.get(split_name),
                    "artifact_reload_seconds": record.get(
                        "artifact_reload_seconds"
                    ),
                    "validation_threshold_selection_seconds": record.get(
                        "threshold_selection_seconds"
                    ),
                    "explainability_seconds": record.get(
                        "explainability_seconds"
                    ),
                    "total_experiment_seconds": record.get(
                        "total_experiment_seconds"
                    ),
                    "experiment_dir": record.get("experiment_dir"),
                }
            )
            rows.append(row)

    return pd.DataFrame(rows)




def build_fallback_subgroup_rows(
    experiment_records: list[dict[str, Any]],
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []

    for record in experiment_records:
        payload = record.get("external_eicu_fallback_subgroups", {})
        output_csv = payload.get("output_csv")
        if not output_csv:
            continue
        path = Path(str(output_csv))
        if not path.is_file():
            continue
        frame = pd.read_csv(path)
        frame.insert(0, "model_name", record["model_name"])
        frame.insert(0, "threshold_days", record["threshold_days"])
        frames.append(frame)

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)


def _read_optional_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame()
    return pd.read_csv(path)


def _compact_performance_table(frame: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "threshold_days",
        "model_name",
        "sample_count",
        "positive_rate",
        "validation_decision_threshold",
        "auroc",
        "auprc",
        "balanced_accuracy",
        "f1_score",
        "sensitivity",
        "specificity",
        "ppv",
        "npv",
        "mcc",
        "brier_score",
    ]
    available = [column for column in columns if column in frame.columns]
    return frame.loc[:, available]


def save_run_summary(
    run_dir: Path,
    experiment_records: list[dict[str, Any]],
) -> None:
    frame = build_summary_rows(experiment_records)
    frame.to_csv(run_dir / "comparison_all_splits.csv", index=False)

    if frame.empty or "split" not in frame.columns:
        internal = pd.DataFrame()
        external = pd.DataFrame()
    else:
        internal = frame[frame["split"] == "internal_test"].copy()
        external = frame[frame["split"] == "external_eicu"].copy()

    internal.to_csv(run_dir / "comparison_internal_test.csv", index=False)
    external.to_csv(run_dir / "comparison_external_eicu.csv", index=False)
    fallback_subgroups = build_fallback_subgroup_rows(experiment_records)
    fallback_subgroups.to_csv(
        run_dir / "comparison_external_eicu_fallback_subgroups.csv",
        index=False,
    )

    audit_dir = run_dir / "data_audit"
    mimic_class = _read_optional_csv(
        audit_dir / "mimic_class_distribution.csv"
    )
    eicu_class = _read_optional_csv(
        audit_dir / "eicu_class_distribution.csv"
    )
    mimic_missing = _read_optional_csv(
        audit_dir / "mimic_missingness.csv"
    )
    eicu_missing = _read_optional_csv(
        audit_dir / "eicu_missingness.csv"
    )
    burden_groups = _read_optional_csv(
        audit_dir / "eicu_harmonizer_burden_groups.csv"
    )

    if not mimic_missing.empty:
        mimic_missing = mimic_missing[
            mimic_missing["missing_count"] > 0
        ].copy()
    if not eicu_missing.empty:
        eicu_missing = eicu_missing[
            eicu_missing["missing_count"] > 0
        ].copy()

    internal_display = _compact_performance_table(internal)
    external_display = _compact_performance_table(external)

    html_content = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>LOS Experiment Summary</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 32px; line-height: 1.5; }}
table {{ border-collapse: collapse; width: 100%; font-size: 0.82rem; margin-bottom: 24px; }}
th, td {{ border: 1px solid #ccc; padding: 6px; text-align: right; }}
th:first-child, td:first-child {{ text-align: left; }}
h1, h2, h3 {{ margin-top: 1.4em; }}
.note {{ background: #f5f5f5; border-left: 4px solid #777; padding: 10px 14px; }}
</style>
</head>
<body>
<h1>Integrated LOS Experiment Summary</h1>
<p class="note">
AUROC and AUPRC are threshold-independent. Metrics ending in <code>at_0_5</code>
use the fixed probability threshold 0.5; the primary classification metrics use
the threshold selected only from the MIMIC-IV validation split.
</p>
<h2>Data Audit</h2>
<h3>MIMIC-IV Class Distribution</h3>
{mimic_class.to_html(index=False, border=0) if not mimic_class.empty else '<p>Not available.</p>'}
<h3>eICU Class Distribution</h3>
{eicu_class.to_html(index=False, border=0) if not eicu_class.empty else '<p>Not available.</p>'}
<h3>MIMIC-IV Missing Values</h3>
{mimic_missing.to_html(index=False, border=0) if not mimic_missing.empty else '<p>No feature/target missing values were recorded after loading, or the audit is unavailable.</p>'}
<h3>eICU Missing Values</h3>
{eicu_missing.to_html(index=False, border=0) if not eicu_missing.empty else '<p>No feature/target missing values were recorded after loading, or the audit is unavailable.</p>'}
<h3>eICU Harmonizer Fallback Burden</h3>
{burden_groups.to_html(index=False, border=0) if not burden_groups.empty else '<p>Harmonizer audit table was not available.</p>'}
<h2>Internal Test</h2>
{internal_display.to_html(index=False, border=0) if not internal_display.empty else '<p>No completed internal-test results.</p>'}
<h2>External eICU</h2>
{external_display.to_html(index=False, border=0) if not external_display.empty else '<p>No completed external-validation results.</p>'}
<h2>External eICU Performance by Harmonizer Fallback Burden</h2>
{fallback_subgroups.to_html(index=False, border=0) if not fallback_subgroups.empty else '<p>Fallback subgroup performance is unavailable.</p>'}
<h2>All Structured Results</h2>
{frame.to_html(index=False, border=0) if not frame.empty else '<p>No completed results.</p>'}
</body>
</html>
"""
    (run_dir / "summary_report.html").write_text(
        html_content,
        encoding="utf-8",
    )
