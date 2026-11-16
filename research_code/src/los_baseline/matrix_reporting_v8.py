from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _standard_task_rows(run_dir: Path, strategy: str, horizon: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    paths = {
        "EXP1": run_dir / "EXP1_flat_3class" / "model_summary_validation.csv",
        "EXP2_Stage1": run_dir / "EXP2_two_stage" / "stage1_mortality" / "model_summary_validation.csv",
        "EXP2_Stage2": run_dir / "EXP2_two_stage" / "stage2_long_stay" / "model_summary_validation.csv",
    }
    for experiment, path in paths.items():
        if not path.is_file():
            continue
        frame = pd.read_csv(path)
        for row in frame.to_dict(orient="records"):
            rows.append({
                "outcome_horizon_days": horizon,
                "strategy": strategy,
                "experiment": experiment,
                **row,
            })
    return rows


def _sampled_task_rows(run_dir: Path, strategy: str, horizon: int, *, runs_dir: Path | None = None) -> list[dict[str, Any]]:
    root_name = "TRACK3_smote" if strategy == "T3_smote" else "TRACK4_undersampled"
    root = run_dir / root_name
    rows: list[dict[str, Any]] = []
    task_map = {
        "exp1_flat_3class": "EXP1",
        "stage1_mortality": "EXP2_Stage1",
        "stage2_long_stay": "EXP2_Stage2",
    }
    if strategy == "T3_smote" and runs_dir is not None:
        track_manifest = root / "track3_manifest.json"
        if track_manifest.is_file():
            optimized_run_id = _read_json(track_manifest).get("optimized_run_id")
            optimized_dir = runs_dir / str(optimized_run_id) if optimized_run_id else None
            if optimized_dir is not None:
                summary = optimized_dir / "EXP2_two_stage" / "stage1_mortality" / "model_summary_validation.csv"
                if summary.is_file():
                    for row in pd.read_csv(summary).to_dict(orient="records"):
                        rows.append({
                            "outcome_horizon_days": horizon,
                            "strategy": strategy,
                            "experiment": "EXP2_Stage1",
                            "source_strategy": "T2_optuna_reference",
                            **row,
                        })
                else:
                    stage1_root = optimized_dir / "EXP2_two_stage" / "stage1_mortality"
                    if stage1_root.is_dir():
                        for model_dir in sorted(stage1_root.iterdir()):
                            metrics_path = model_dir / "evaluation" / "validation" / "metrics.json"
                            if metrics_path.is_file():
                                metrics = _read_json(metrics_path)
                                rows.append({
                                    "outcome_horizon_days": horizon,
                                    "strategy": strategy,
                                    "experiment": "EXP2_Stage1",
                                    "source_strategy": "T2_optuna_reference",
                                    "model_name": model_dir.name,
                                    **{f"validation_{key}": value for key, value in metrics.items()},
                                })

    for task_name, experiment in task_map.items():
        if strategy == "T3_smote" and task_name == "stage1_mortality":
            continue
        task_root = root / task_name
        if not task_root.is_dir():
            continue
        for model_dir in sorted(task_root.iterdir()):
            if not model_dir.is_dir():
                continue
            metrics_path = model_dir / "ensemble" / "validation" / "metrics.json"
            if not metrics_path.is_file():
                continue
            metrics = _read_json(metrics_path)
            rows.append({
                "outcome_horizon_days": horizon,
                "strategy": strategy,
                "experiment": experiment,
                "model_name": model_dir.name,
                **{f"validation_{key}": value for key, value in metrics.items()},
            })
    return rows


def _combined_row(run_dir: Path, strategy: str, horizon: int) -> dict[str, Any] | None:
    if strategy in {"T1_baseline", "T2_optuna"}:
        path = run_dir / "V8_reporting" / "EXP2_two_stage" / "combined" / "top_pair.json"
    elif strategy == "T3_smote":
        path = run_dir / "TRACK3_smote" / "EXP2_two_stage" / "combined" / "top_pair.json"
    else:
        path = run_dir / "TRACK4_undersampled" / "EXP2_two_stage" / "combined" / "top_pair.json"
    if not path.is_file():
        return None
    payload = _read_json(path)
    top = dict(payload.get("top_pair", {}))
    return {
        "outcome_horizon_days": horizon,
        "strategy": strategy,
        "experiment": "EXP2_Combined",
        "model_name": f"{top.get('stage1_model')}->{top.get('stage2_model')}",
        **{f"validation_{key}": value for key, value in top.items() if key not in {"stage1_model", "stage2_model", "strategy_id"}},
    }


def _exp3_rows(run_dir: Path, strategy: str, horizon: int) -> list[dict[str, Any]]:
    if strategy in {"T1_baseline", "T2_optuna"}:
        path = run_dir / "V8_reporting" / "EXP3_survivor_binary" / "model_summary_validation.csv"
    elif strategy == "T3_smote":
        path = run_dir / "TRACK3_smote" / "EXP3_survivor_binary" / "model_summary_validation.csv"
    else:
        path = run_dir / "TRACK4_undersampled" / "EXP3_survivor_binary" / "model_summary_validation.csv"
    if not path.is_file():
        return []
    frame = pd.read_csv(path)
    return [
        {
            **row,
            "outcome_horizon_days": horizon,
            "strategy": strategy,
            "experiment": "EXP3",
        }
        for row in frame.to_dict(orient="records")
    ]


def collect_matrix_rows(
    *,
    runs_dir: Path,
    runs: Sequence[Mapping[str, Any]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for item in runs:
        if not bool(item.get("enabled", True)) or str(item.get("status", "")) == "not_requested":
            continue
        run_id = item.get("run_id")
        if not run_id:
            continue
        strategy = str(item["strategy"])
        horizon = int(item["outcome_horizon_days"])
        run_dir = runs_dir / str(run_id)
        if not run_dir.is_dir():
            continue
        if strategy in {"T1_baseline", "T2_optuna"}:
            rows.extend(_standard_task_rows(run_dir, strategy, horizon))
        else:
            rows.extend(_sampled_task_rows(run_dir, strategy, horizon, runs_dir=runs_dir))
        combined = _combined_row(run_dir, strategy, horizon)
        if combined is not None:
            rows.append(combined)
        rows.extend(_exp3_rows(run_dir, strategy, horizon))
    return pd.DataFrame(rows)
