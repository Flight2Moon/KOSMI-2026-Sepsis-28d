from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _binary_metrics(path: Path) -> dict[str, Any]:
    payload = _read_json(path)
    metrics = payload.get("threshold_views", {}).get("validation_selected")
    if not isinstance(metrics, dict):
        return {}
    return dict(metrics)


def collect_confirmatory_results(results_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    recal_rows: list[dict[str, Any]] = []
    for split in ("internal_test", "external_eicu"):
        split_root = results_root / split
        if not split_root.is_dir():
            continue
        for horizon_dir in sorted(split_root.glob("LOS*")):
            if not horizon_dir.is_dir():
                continue
            try:
                horizon = int(horizon_dir.name.replace("LOS", ""))
            except ValueError:
                continue
            for strategy_dir in sorted(path for path in horizon_dir.iterdir() if path.is_dir()):
                strategy = strategy_dir.name
                manifest_path = strategy_dir / "confirmatory_result_manifest.json"
                if not manifest_path.is_file():
                    continue
                manifest = _read_json(manifest_path)
                specs = [
                    ("EXP1", strategy_dir / "EXP1_flat_3class" / "evaluation" / "multiclass_metrics.json", "multiclass"),
                    ("EXP2_Stage1", strategy_dir / "EXP2_two_stage" / "stage1_mortality" / "evaluation" / "binary_metrics.json", "binary"),
                    ("EXP2_Stage2", strategy_dir / "EXP2_two_stage" / "stage2_long_stay" / "evaluation" / "binary_metrics.json", "binary"),
                    ("EXP2_Combined", strategy_dir / "EXP2_two_stage" / "combined" / "evaluation" / "multiclass_metrics.json", "multiclass"),
                    ("EXP3", strategy_dir / "EXP3_survivor_binary" / "evaluation" / "binary_metrics.json", "binary"),
                ]
                for experiment, path, kind in specs:
                    if not path.is_file():
                        continue
                    metrics = _read_json(path) if kind == "multiclass" else _binary_metrics(path)
                    rows.append({
                        "split": split,
                        "outcome_horizon_days": horizon,
                        "strategy": strategy,
                        "experiment": experiment,
                        "analysis_role": "raw_confirmatory_primary" if split == "external_eicu" else "internal_confirmatory",
                        "sample_count": metrics.get("sample_count"),
                        "macro_auroc": metrics.get("macro_auroc"),
                        "macro_auprc": metrics.get("macro_auprc"),
                        "auroc": metrics.get("auroc"),
                        "auprc": metrics.get("auprc"),
                        "macro_f1": metrics.get("macro_f1"),
                        "f1": metrics.get("f1_score"),
                        "balanced_accuracy": metrics.get("balanced_accuracy"),
                        "brier": metrics.get("multiclass_brier", metrics.get("brier_score")),
                        "log_loss": metrics.get("multiclass_log_loss", metrics.get("log_loss")),
                        "threshold_reselection_performed": bool(manifest.get("threshold_reselection_performed", False)),
                    })
                if split == "external_eicu":
                    for experiment, dirname in (("EXP1", "EXP1_flat_3class"), ("EXP2_Combined", "EXP2_two_stage_combined")):
                        path = strategy_dir / "site_recalibration" / dirname / "recalibration_multiseed_metrics_summary.csv"
                        if not path.is_file():
                            continue
                        frame = pd.read_csv(path)
                        for row in frame.to_dict(orient="records"):
                            recal_rows.append({
                                "split": split,
                                "outcome_horizon_days": horizon,
                                "strategy": strategy,
                                "experiment": experiment,
                                "analysis_role": "secondary_local_site_adaptation",
                                **row,
                            })
    return pd.DataFrame(rows), pd.DataFrame(recal_rows)


def write_confirmatory_comparison(results_root: Path, output_dir: Path) -> dict[str, str]:
    raw, recal = collect_confirmatory_results(results_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / "confirmatory_raw_results.csv"
    recal_path = output_dir / "confirmatory_site_recalibration_summary.csv"
    raw.to_csv(raw_path, index=False)
    recal.to_csv(recal_path, index=False)
    html = ["<html><head><meta charset='utf-8'><title>EXP123 V8 Confirmatory Results</title></head><body>"]
    html.append("<h1>EXP123 V8 Confirmatory Results</h1>")
    html.append("<p><strong>Primary external result:</strong> raw eICU transfer. Site recalibration is secondary local adaptation and is reported separately.</p>")
    html.append("<h2>Raw confirmatory results</h2>")
    html.append(raw.to_html(index=False, border=0) if len(raw) else "<p>No raw confirmatory results.</p>")
    html.append("<h2>Site recalibration summary</h2>")
    html.append(recal.to_html(index=False, border=0) if len(recal) else "<p>No site recalibration results.</p>")
    html.append("</body></html>")
    html_path = output_dir / "confirmatory_comparison.html"
    html_path.write_text("\n".join(html), encoding="utf-8")
    return {"raw_csv": str(raw_path), "recalibration_csv": str(recal_path), "html": str(html_path)}
