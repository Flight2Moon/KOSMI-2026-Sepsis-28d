from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .artifacts import load_task_artifact
from .pipeline_predictions_v8 import infer_threshold_from_prediction_frame
from .utils import atomic_write_json, sha256_file


@dataclass(frozen=True)
class PredictorSpec:
    task_name: str
    model_name: str
    kind: str
    artifact_dir: str | None = None
    member_artifact_dirs: tuple[str, ...] = ()
    validation_threshold: float | None = None
    validation_threshold_policies: tuple[dict[str, Any], ...] = ()
    source_strategy: str | None = None
    source_run_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["member_artifact_dirs"] = list(self.member_artifact_dirs)
        payload["validation_threshold_policies"] = list(self.validation_threshold_policies)
        return payload


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return payload


def _read_validation_threshold_policies(path: Path) -> tuple[dict[str, Any], ...]:
    manifest = path / "clinical_evaluation_manifest.json"
    if not manifest.is_file():
        return ()
    payload = _read_json(manifest)
    policies = payload.get("threshold_policies", [])
    if not isinstance(policies, list):
        return ()
    return tuple(dict(item) for item in policies if isinstance(item, dict))


def _rank_frame(
    frame: pd.DataFrame,
    *,
    primary_candidates: Sequence[str],
    secondary_candidates: Sequence[str],
    lower_is_better_candidates: Sequence[str] = (),
) -> pd.DataFrame:
    if frame.empty:
        raise ValueError("Cannot rank an empty validation summary.")
    primary = next((c for c in primary_candidates if c in frame.columns), None)
    if primary is None:
        raise KeyError(f"None of the primary metrics are present: {list(primary_candidates)}")
    secondary = next((c for c in secondary_candidates if c in frame.columns), None)
    lower = next((c for c in lower_is_better_candidates if c in frame.columns), None)
    ranked = frame.copy()
    ranked["_primary"] = pd.to_numeric(ranked[primary], errors="coerce")
    if ranked["_primary"].isna().all():
        raise ValueError(f"Primary metric contains no finite values: {primary}")
    ranked["_secondary"] = (
        pd.to_numeric(ranked[secondary], errors="coerce") if secondary else np.nan
    )
    ranked["_lower"] = pd.to_numeric(ranked[lower], errors="coerce") if lower else np.nan
    ranked["_model"] = ranked.get("model_name", pd.Series([""] * len(ranked))).astype(str)
    sort_cols = ["_primary"]
    ascending = [False]
    if secondary:
        sort_cols.append("_secondary"); ascending.append(False)
    if lower:
        sort_cols.append("_lower"); ascending.append(True)
    sort_cols.append("_model"); ascending.append(True)
    return ranked.sort_values(sort_cols, ascending=ascending, kind="mergesort").reset_index(drop=True)


def _standard_exp1_model(run_dir: Path, metric: str) -> str:
    path = run_dir / "EXP1_flat_3class" / "model_summary_validation.csv"
    frame = pd.read_csv(path)
    if metric == "macro_auprc":
        ranked = _rank_frame(
            frame,
            primary_candidates=("validation_macro_auprc", "macro_auprc"),
            secondary_candidates=("validation_macro_auroc", "macro_auroc"),
            lower_is_better_candidates=("validation_multiclass_brier", "multiclass_brier"),
        )
    else:
        ranked = _rank_frame(
            frame,
            primary_candidates=("validation_macro_auroc", "macro_auroc"),
            secondary_candidates=("validation_macro_auprc", "macro_auprc"),
            lower_is_better_candidates=("validation_multiclass_brier", "multiclass_brier"),
        )
    return str(ranked.iloc[0]["model_name"])


def _sampled_exp1_model(run_dir: Path, strategy: str, metric: str) -> str:
    root_name = "TRACK3_smote" if strategy == "T3_smote" else "TRACK4_undersampled"
    task_root = run_dir / root_name / "exp1_flat_3class"
    rows: list[dict[str, Any]] = []
    for model_dir in sorted(task_root.iterdir() if task_root.is_dir() else []):
        if not model_dir.is_dir():
            continue
        metrics_path = model_dir / "ensemble" / "validation" / "metrics.json"
        if not metrics_path.is_file():
            continue
        rows.append({"model_name": model_dir.name, **_read_json(metrics_path)})
    frame = pd.DataFrame(rows)
    if metric == "macro_auprc":
        ranked = _rank_frame(frame, primary_candidates=("macro_auprc",), secondary_candidates=("macro_auroc",), lower_is_better_candidates=("multiclass_brier",))
    else:
        ranked = _rank_frame(frame, primary_candidates=("macro_auroc",), secondary_candidates=("macro_auprc",), lower_is_better_candidates=("multiclass_brier",))
    return str(ranked.iloc[0]["model_name"])


def _member_artifact_dirs(model_root: Path) -> tuple[str, ...]:
    members = model_root / "members"
    dirs = sorted(path / "artifact" for path in members.iterdir() if path.is_dir() and (path / "artifact").is_dir())
    if not dirs:
        raise FileNotFoundError(f"No sampled member artifacts found: {members}")
    return tuple(str(path) for path in dirs)


def _standard_binary_spec(
    *,
    run_dir: Path,
    task_name: str,
    model_name: str,
    strategy: str,
    run_id: str,
    clinical_dir: Path | None = None,
) -> PredictorSpec:
    stage_dir = "stage1_mortality" if task_name == "stage1_mortality" else "stage2_long_stay"
    artifact_dir = run_dir / "EXP2_two_stage" / stage_dir / model_name / "artifact"
    artifact = load_task_artifact(artifact_dir)
    threshold = artifact.validation_decision_threshold
    return PredictorSpec(
        task_name=task_name,
        model_name=model_name,
        kind="standard_artifact",
        artifact_dir=str(artifact_dir),
        validation_threshold=(float(threshold) if threshold is not None else None),
        validation_threshold_policies=(
            _read_validation_threshold_policies(clinical_dir) if clinical_dir is not None else ()
        ),
        source_strategy=strategy,
        source_run_id=run_id,
    )


def _sampled_binary_spec(
    *,
    run_dir: Path,
    task_name: str,
    model_name: str,
    strategy: str,
    run_id: str,
    clinical_dir: Path | None = None,
) -> PredictorSpec:
    root_name = "TRACK3_smote" if strategy == "T3_smote" else "TRACK4_undersampled"
    task_root = run_dir / root_name / task_name / model_name
    validation_path = task_root / "ensemble" / "validation" / "predictions.csv.gz"
    threshold = infer_threshold_from_prediction_frame(pd.read_csv(validation_path))
    return PredictorSpec(
        task_name=task_name,
        model_name=model_name,
        kind="member_ensemble",
        member_artifact_dirs=_member_artifact_dirs(task_root),
        validation_threshold=float(threshold),
        validation_threshold_policies=(
            _read_validation_threshold_policies(clinical_dir) if clinical_dir is not None else ()
        ),
        source_strategy=strategy,
        source_run_id=run_id,
    )


def _exp3_reference_model(run_dir: Path, strategy: str) -> str:
    if strategy in {"T1_baseline", "T2_optuna"}:
        root = run_dir / "V8_reporting" / "EXP3_survivor_binary"
    elif strategy == "T3_smote":
        root = run_dir / "TRACK3_smote" / "EXP3_survivor_binary"
    else:
        root = run_dir / "TRACK4_undersampled" / "EXP3_survivor_binary"
    lock = _read_json(root / "EXP3_REPORT_COMPLETED.lock.json")
    model = lock.get("validation_reference_model")
    if not model:
        raise ValueError(f"EXP3 validation reference model is missing: {root}")
    return str(model)


def _top_pair(run_dir: Path, strategy: str) -> dict[str, Any]:
    if strategy in {"T1_baseline", "T2_optuna"}:
        path = run_dir / "V8_reporting" / "EXP2_two_stage" / "combined" / "top_pair.json"
    elif strategy == "T3_smote":
        path = run_dir / "TRACK3_smote" / "EXP2_two_stage" / "combined" / "top_pair.json"
    else:
        path = run_dir / "TRACK4_undersampled" / "EXP2_two_stage" / "combined" / "top_pair.json"
    payload = _read_json(path)
    top = payload.get("top_pair")
    if not isinstance(top, dict):
        raise KeyError(f"top_pair missing: {path}")
    return dict(top)


def build_strategy_selection(
    *,
    run_dir: Path,
    run_id: str,
    strategy: str,
    config: Mapping[str, Any],
    runs_dir: Path,
) -> dict[str, Any]:
    metrics = dict(config.get("analysis_plan", {}).get("task_selection_metrics", {}))
    exp1_metric = str(metrics.get("exp1_flat_3class", "macro_auprc"))
    exp1_model = (
        _standard_exp1_model(run_dir, exp1_metric)
        if strategy in {"T1_baseline", "T2_optuna"}
        else _sampled_exp1_model(run_dir, strategy, exp1_metric)
    )

    if strategy in {"T1_baseline", "T2_optuna"}:
        exp1_spec = PredictorSpec(
            task_name="exp1_flat_3class",
            model_name=exp1_model,
            kind="standard_artifact",
            artifact_dir=str(run_dir / "EXP1_flat_3class" / exp1_model / "artifact"),
            source_strategy=strategy,
            source_run_id=run_id,
        )
    else:
        root_name = "TRACK3_smote" if strategy == "T3_smote" else "TRACK4_undersampled"
        exp1_spec = PredictorSpec(
            task_name="exp1_flat_3class",
            model_name=exp1_model,
            kind="member_ensemble",
            member_artifact_dirs=_member_artifact_dirs(run_dir / root_name / "exp1_flat_3class" / exp1_model),
            source_strategy=strategy,
            source_run_id=run_id,
        )

    pair = _top_pair(run_dir, strategy)
    stage1_model = str(pair["stage1_model"])
    stage2_model = str(pair["stage2_model"])
    exp3_model = _exp3_reference_model(run_dir, strategy)

    if strategy == "T3_smote":
        manifest = _read_json(run_dir / "TRACK3_smote" / "track3_manifest.json")
        optimized_run_id = str(manifest["optimized_run_id"])
        optimized_dir = runs_dir / optimized_run_id
        stage1_spec = _standard_binary_spec(
            run_dir=optimized_dir,
            task_name="stage1_mortality",
            model_name=stage1_model,
            strategy="T2_optuna_reference",
            run_id=optimized_run_id,
            clinical_dir=run_dir / "TRACK3_smote" / "EXP2_two_stage" / "stage1_mortality" / "clinical_reference_T2" / stage1_model / "clinical_validation",
        )
    elif strategy in {"T1_baseline", "T2_optuna"}:
        stage1_spec = _standard_binary_spec(
            run_dir=run_dir,
            task_name="stage1_mortality",
            model_name=stage1_model,
            strategy=strategy,
            run_id=run_id,
            clinical_dir=run_dir / "V8_reporting" / "EXP2_two_stage" / "stage1_mortality" / "clinical" / stage1_model / "clinical_validation",
        )
    else:
        stage1_spec = _sampled_binary_spec(
            run_dir=run_dir,
            task_name="stage1_mortality",
            model_name=stage1_model,
            strategy=strategy,
            run_id=run_id,
            clinical_dir=run_dir / "TRACK4_undersampled" / "EXP2_two_stage" / "stage1_mortality" / "clinical" / stage1_model / "clinical_validation",
        )

    def stage2_spec_for(model_name: str) -> PredictorSpec:
        if strategy in {"T1_baseline", "T2_optuna"}:
            return _standard_binary_spec(
                run_dir=run_dir,
                task_name="stage2_long_stay",
                model_name=model_name,
                strategy=strategy,
                run_id=run_id,
                clinical_dir=run_dir / "V8_reporting" / "EXP2_two_stage" / "stage2_long_stay" / "clinical" / model_name / "clinical_validation",
            )
        root_name = "TRACK3_smote" if strategy == "T3_smote" else "TRACK4_undersampled"
        return _sampled_binary_spec(
            run_dir=run_dir,
            task_name="stage2_long_stay",
            model_name=model_name,
            strategy=strategy,
            run_id=run_id,
            clinical_dir=run_dir / root_name / "EXP3_survivor_binary" / "models" / model_name / "clinical_validation",
        )

    stage2_pair_spec = stage2_spec_for(stage2_model)
    exp3_spec = stage2_pair_spec if exp3_model == stage2_model else stage2_spec_for(exp3_model)

    if strategy == "T3_smote":
        cohort_source_run_id = str(_read_json(run_dir / "TRACK3_smote" / "track3_manifest.json")["optimized_run_id"])
    elif strategy == "T4_undersampling":
        cohort_source_run_id = str(_read_json(run_dir / "TRACK4_undersampled" / "track4_manifest.json")["optimized_run_id"])
    else:
        cohort_source_run_id = run_id

    source_paths: list[Path] = []
    for spec in (exp1_spec, stage1_spec, stage2_pair_spec, exp3_spec):
        if spec.artifact_dir:
            source_paths.extend(sorted(Path(spec.artifact_dir).rglob("*")))
        for member in spec.member_artifact_dirs:
            source_paths.extend(sorted(Path(member).rglob("*")))
    hashes = {
        str(path): sha256_file(path)
        for path in source_paths
        if path.is_file()
    }
    return {
        "schema_version": "exp123_v8_strategy_confirmatory_selection_v1",
        "run_id": run_id,
        "strategy": strategy,
        "cohort_source_run_id": cohort_source_run_id,
        "outcome_horizon_days": float(config["outcomes"]["outcome_horizon_days"]),
        "selection_split": "validation",
        "selection_metrics": metrics,
        "exp1": exp1_spec.to_dict(),
        "exp2": {
            "stage1": stage1_spec.to_dict(),
            "stage2": stage2_pair_spec.to_dict(),
            "top_pair_validation": pair,
        },
        "exp3": exp3_spec.to_dict(),
        "artifact_hashes": hashes,
        "confirmatory_splits_accessed": False,
    }


def write_strategy_selection(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(path, dict(payload))
    (path.with_name(path.name + ".sha256")).write_text(sha256_file(path) + "\n", encoding="utf-8")
    return path
