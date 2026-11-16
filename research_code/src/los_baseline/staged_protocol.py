from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .artifacts import load_task_artifact
from .evaluation_multiclass import calculate_multiclass_metrics
from .exp1_runner import load_exp1_discovery_task_data
from .hardware import apply_acceleration_to_model_config
from .oracle_pipeline import derive_pipeline_classes, restore_three_class_probabilities
from .stage1_runner import load_stage1_discovery_task_data
from .stage2_runner import load_stage2_discovery_task_data
from .task_spec import (
    EXP1_FLAT_3CLASS_TASK,
    STAGE1_MORTALITY_TASK,
    STAGE2_LONG_STAY_TASK,
    TaskDefinition,
)
from .utils import atomic_write_json, environment_info, sha256_file


TASK_META: dict[str, dict[str, Any]] = {
    "exp1": {
        "task": EXP1_FLAT_3CLASS_TASK,
        "loader": load_exp1_discovery_task_data,
        "output_relative": Path("EXP1_flat_3class"),
        "summary_file": "model_summary_validation.csv",
        "primary_metric": "validation_macro_auroc",
        "secondary_metric": "validation_multiclass_brier",
        "secondary_direction": "minimize",
    },
    "stage1": {
        "task": STAGE1_MORTALITY_TASK,
        "loader": load_stage1_discovery_task_data,
        "output_relative": Path("EXP2_two_stage") / "stage1_mortality",
        "summary_file": "model_summary_validation.csv",
        "primary_metric": "validation_auprc",
        "secondary_metric": "validation_brier_score",
        "secondary_direction": "minimize",
    },
    "stage2": {
        "task": STAGE2_LONG_STAY_TASK,
        "loader": load_stage2_discovery_task_data,
        "output_relative": Path("EXP2_two_stage") / "stage2_long_stay",
        "summary_file": "model_summary_validation.csv",
        "primary_metric": "validation_auroc",
        "secondary_metric": "validation_brier_score",
        "secondary_direction": "minimize",
    },
}


@dataclass(frozen=True)
class BaselineParameterResult:
    task_key: str
    task_name: str
    model_name: str
    output_path: str
    shared_split_hash: str
    model_config_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CandidatePlanResult:
    baseline_run_id: str
    output_path: str
    split_hash: str
    selected_models: dict[str, list[str]]
    pairwise_report_path: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _models_for_task(config: Mapping[str, Any], task_key: str) -> list[str]:
    if task_key == "exp1":
        return list(config["experiments"]["exp1_flat_3class"]["models"])
    stage_name = "stage1_mortality" if task_key == "stage1" else "stage2_long_stay"
    return list(config["experiments"]["exp2_two_stage"][stage_name]["models"])


def _task_data(config: Mapping[str, Any], run_dir: Path, task_key: str):
    return TASK_META[task_key]["loader"](config=config, run_dir=run_dir)


def _baseline_parameter_dir(run_dir: Path, task: TaskDefinition, model_name: str) -> Path:
    return run_dir / "optimization" / task.task_name / model_name


def prepare_baseline_model_parameters(
    *,
    task_key: str,
    model_name: str,
    config: Mapping[str, Any],
    run_id: str,
    run_dir: Path,
    overwrite: bool = False,
) -> BaselineParameterResult:
    if task_key not in TASK_META:
        raise KeyError(task_key)
    configured = _models_for_task(config, task_key)
    if model_name not in configured:
        raise ValueError(f"{model_name} is not configured for {task_key}.")

    task = TASK_META[task_key]["task"]
    task_data = _task_data(config, run_dir, task_key)
    output_dir = _baseline_parameter_dir(run_dir, task, model_name)
    output_path = output_dir / "selected_hyperparameters.json"
    lock_path = output_dir / "BASELINE_PARAMETERS_COMPLETED.lock.json"

    if output_path.is_file() and lock_path.is_file() and not overwrite:
        payload = json.loads(output_path.read_text(encoding="utf-8"))
        return BaselineParameterResult(
            task_key=task_key,
            task_name=task.task_name,
            model_name=model_name,
            output_path=str(output_path),
            shared_split_hash=str(payload["shared_split_hash"]),
            model_config_sha256=sha256_file(output_path),
        )
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(
            f"Baseline parameter output already exists but is incomplete: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    model_config = apply_acceleration_to_model_config(
        project_config=config,
        model_name=model_name,
        model_config=dict(config["models"][model_name]),
    )
    payload = {
        "study_id": f"{task.task_name}:{model_name}:baseline_default",
        "task_name": task.task_name,
        "model_name": model_name,
        "selection_source": "baseline_default_configuration",
        "selected_candidate_id": "baseline_default",
        "confirmatory_splits_accessed": False,
        "model_config": model_config,
        "metrics": {},
        "shared_split_hash": task_data.shared_split_hash,
        "baseline_run_id": run_id,
        "environment": environment_info(),
    }
    atomic_write_json(output_path, payload)
    (output_path.with_name(output_path.name + ".sha256")).write_text(
        sha256_file(output_path) + "\n", encoding="utf-8"
    )
    lock_payload = {
        "job_type": "baseline_parameter_materialization",
        "run_id": run_id,
        "task_key": task_key,
        "task_name": task.task_name,
        "model_name": model_name,
        "selected_hyperparameters_sha256": sha256_file(output_path),
        "shared_split_hash": task_data.shared_split_hash,
        "confirmatory_splits_accessed": False,
    }
    atomic_write_json(lock_path, lock_payload)
    (lock_path.with_name(lock_path.name + ".sha256")).write_text(
        sha256_file(lock_path) + "\n", encoding="utf-8"
    )
    return BaselineParameterResult(
        task_key=task_key,
        task_name=task.task_name,
        model_name=model_name,
        output_path=str(output_path),
        shared_split_hash=str(task_data.shared_split_hash),
        model_config_sha256=sha256_file(output_path),
    )


def _model_artifact_dir(run_dir: Path, task_key: str, model_name: str) -> Path:
    return run_dir / TASK_META[task_key]["output_relative"] / model_name / "artifact"


def _sorted_pairwise_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda row: (
            -float(row.get("macro_auroc") or -np.inf),
            -float(row.get("macro_auprc") or -np.inf),
            float(row.get("multiclass_brier") or np.inf),
            float(row.get("multiclass_log_loss") or np.inf),
            str(row["stage1_model"]),
            str(row["stage2_model"]),
        ),
    )


def build_exp2_pairwise_validation_report(
    *,
    config: Mapping[str, Any],
    run_id: str,
    run_dir: Path,
    phase: str,
    overwrite: bool = False,
) -> dict[str, Any]:
    if phase not in {"baseline", "optimized"}:
        raise ValueError("phase must be baseline or optimized.")
    stage1_models = _models_for_task(config, "stage1")
    stage2_models = _models_for_task(config, "stage2")
    task_data = load_stage1_discovery_task_data(config=config, run_dir=run_dir)
    validation = task_data.splits["validation"]
    raw_features = validation.raw_features.loc[:, list(task_data.feature_names)]
    y_true = validation.metadata["three_class_outcome"].to_numpy(dtype=np.int64)

    output_dir = run_dir / "common" / f"staged_{phase}" / "exp2_pairwise_validation"
    lock_path = output_dir / "PAIRWISE_VALIDATION_COMPLETED.lock.json"
    if lock_path.is_file() and not overwrite:
        return json.loads(lock_path.read_text(encoding="utf-8"))
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"Pairwise output already exists: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    stage1_probabilities: dict[str, np.ndarray] = {}
    stage1_thresholds: dict[str, float] = {}
    for model_name in stage1_models:
        artifact = load_task_artifact(_model_artifact_dir(run_dir, "stage1", model_name))
        if artifact.task_name != STAGE1_MORTALITY_TASK.task_name:
            raise ValueError(f"Wrong Stage1 artifact task for {model_name}.")
        stage1_probabilities[model_name] = np.asarray(
            artifact.predict_proba(raw_features), dtype=np.float64
        ).reshape(-1)
        stage1_thresholds[model_name] = float(
            artifact.validation_decision_threshold
            if artifact.validation_decision_threshold is not None
            else config["evaluation"]["default_decision_threshold"]
        )

    stage2_probabilities: dict[str, np.ndarray] = {}
    stage2_thresholds: dict[str, float] = {}
    for model_name in stage2_models:
        artifact = load_task_artifact(_model_artifact_dir(run_dir, "stage2", model_name))
        if artifact.task_name != STAGE2_LONG_STAY_TASK.task_name:
            raise ValueError(f"Wrong Stage2 artifact task for {model_name}.")
        stage2_probabilities[model_name] = np.asarray(
            artifact.predict_proba(raw_features), dtype=np.float64
        ).reshape(-1)
        stage2_thresholds[model_name] = float(
            artifact.validation_decision_threshold
            if artifact.validation_decision_threshold is not None
            else config["evaluation"]["default_decision_threshold"]
        )

    rows: list[dict[str, Any]] = []
    for stage1_model in stage1_models:
        for stage2_model in stage2_models:
            probabilities = restore_three_class_probabilities(
                stage1_probabilities[stage1_model],
                stage2_probabilities[stage2_model],
            )
            argmax_metrics = calculate_multiclass_metrics(
                y_true=y_true,
                probabilities=probabilities,
                class_names=EXP1_FLAT_3CLASS_TASK.class_names,
            )
            hard_classes = derive_pipeline_classes(
                stage1_probabilities[stage1_model],
                stage2_probabilities[stage2_model],
                stage1_threshold=stage1_thresholds[stage1_model],
                stage2_threshold=stage2_thresholds[stage2_model],
            )
            hard_metrics = calculate_multiclass_metrics(
                y_true=y_true,
                probabilities=probabilities,
                class_names=EXP1_FLAT_3CLASS_TASK.class_names,
                predicted_classes=hard_classes,
            )
            rows.append(
                {
                    "phase": phase,
                    "stage1_model": stage1_model,
                    "stage2_model": stage2_model,
                    "stage1_threshold": stage1_thresholds[stage1_model],
                    "stage2_threshold": stage2_thresholds[stage2_model],
                    **argmax_metrics,
                    "hard_gate_accuracy": hard_metrics["accuracy"],
                    "hard_gate_balanced_accuracy": hard_metrics["balanced_accuracy"],
                    "hard_gate_macro_f1": hard_metrics["macro_f1"],
                }
            )

    ranked = _sorted_pairwise_rows(rows)
    frame = pd.DataFrame(ranked)
    csv_path = output_dir / "pairwise_pipeline_validation.csv"
    frame.to_csv(csv_path, index=False)
    top_pair = ranked[0]
    top_path = output_dir / "top_pair.json"
    atomic_write_json(
        top_path,
        {
            "run_id": run_id,
            "phase": phase,
            "selection_split": "validation",
            "selection_rule": [
                "maximize_macro_auroc",
                "maximize_macro_auprc",
                "minimize_multiclass_brier",
                "minimize_multiclass_log_loss",
            ],
            "top_pair": top_pair,
            "pair_count": len(ranked),
            "shared_split_hash": task_data.shared_split_hash,
            "confirmatory_splits_accessed": False,
        },
    )
    top_stage1 = str(top_pair["stage1_model"])
    top_stage2 = str(top_pair["stage2_model"])
    top_probabilities = restore_three_class_probabilities(
        stage1_probabilities[top_stage1], stage2_probabilities[top_stage2]
    )
    prediction_frame = validation.metadata.copy().reset_index(drop=True)
    prediction_frame["y_true_3class"] = y_true
    prediction_frame["stage1_probability_B"] = stage1_probabilities[top_stage1]
    prediction_frame["stage2_probability_C_given_nonB"] = stage2_probabilities[top_stage2]
    prediction_frame["prob_class_0"] = top_probabilities[:, 0]
    prediction_frame["prob_class_1"] = top_probabilities[:, 1]
    prediction_frame["prob_class_2"] = top_probabilities[:, 2]
    prediction_frame["argmax_prediction"] = np.argmax(top_probabilities, axis=1)
    prediction_frame.to_csv(
        output_dir / "top_pair_predictions.csv.gz", index=False, compression="gzip"
    )
    manifest = {
        "job_type": "exp2_pairwise_validation",
        "run_id": run_id,
        "phase": phase,
        "pair_count": len(ranked),
        "stage1_models": stage1_models,
        "stage2_models": stage2_models,
        "shared_split_hash": task_data.shared_split_hash,
        "pairwise_csv": str(csv_path),
        "pairwise_csv_sha256": sha256_file(csv_path),
        "top_pair_path": str(top_path),
        "top_pair_sha256": sha256_file(top_path),
        "confirmatory_splits_accessed": False,
        "environment": environment_info(),
    }
    atomic_write_json(lock_path, manifest)
    (lock_path.with_name(lock_path.name + ".sha256")).write_text(
        sha256_file(lock_path) + "\n", encoding="utf-8"
    )
    return manifest


def _read_task_summary(run_dir: Path, task_key: str) -> pd.DataFrame:
    path = (
        run_dir
        / TASK_META[task_key]["output_relative"]
        / TASK_META[task_key]["summary_file"]
    )
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    required = {
        "model_name",
        TASK_META[task_key]["primary_metric"],
        TASK_META[task_key]["secondary_metric"],
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise KeyError(f"Summary {path} is missing columns: {missing}")
    return frame


def _rank_models(frame: pd.DataFrame, task_key: str) -> list[str]:
    primary = TASK_META[task_key]["primary_metric"]
    secondary = TASK_META[task_key]["secondary_metric"]
    ranked = frame.sort_values(
        [primary, secondary, "model_name"],
        ascending=[False, True, True],
        kind="mergesort",
    )
    return ranked["model_name"].astype(str).tolist()


def _select_with_cap(
    *,
    configured: Sequence[str],
    mandatory: Sequence[str],
    ranked: Sequence[str],
    pipeline_members: Sequence[str],
    top_k: int,
    max_models: int,
) -> list[str]:
    configured_set = set(configured)
    ordered: list[str] = []
    for source in (mandatory, list(ranked)[:top_k], pipeline_members, ranked):
        for model in source:
            model = str(model)
            if model in configured_set and model not in ordered:
                ordered.append(model)
            if len(ordered) >= max_models:
                return ordered
    return ordered


def select_targeted_candidates(
    *,
    config: Mapping[str, Any],
    run_id: str,
    run_dir: Path,
    overwrite: bool = False,
) -> CandidatePlanResult:
    staged = dict(config.get("staged_protocol", {}))
    selection = dict(staged.get("candidate_selection", {}))
    top_k = max(1, int(selection.get("top_models_per_task", 1)))
    top_pairs = max(1, int(selection.get("top_pipeline_pairs", 3)))
    max_models = max(1, int(selection.get("max_models_per_task", 3)))
    always_include = dict(selection.get("always_include", {}))

    output_dir = run_dir / "common" / "staged_baseline"
    output_path = output_dir / "candidate_plan.json"
    if output_path.is_file() and not overwrite:
        payload = json.loads(output_path.read_text(encoding="utf-8"))
        return CandidatePlanResult(
            baseline_run_id=run_id,
            output_path=str(output_path),
            split_hash=str(payload["shared_split_hash"]),
            selected_models={
                key: list(value) for key, value in payload["selected_models"].items()
            },
            pairwise_report_path=str(payload["pairwise_report_path"]),
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    summaries = {task: _read_task_summary(run_dir, task) for task in TASK_META}
    rankings = {task: _rank_models(frame, task) for task, frame in summaries.items()}
    pairwise_path = (
        output_dir
        / "exp2_pairwise_validation"
        / "pairwise_pipeline_validation.csv"
    )
    if not pairwise_path.is_file():
        raise FileNotFoundError(pairwise_path)
    pairwise = pd.read_csv(pairwise_path)
    if pairwise.empty:
        raise ValueError("EXP2 pairwise validation report is empty.")
    top_pair_rows = pairwise.head(top_pairs)
    stage1_pair_members = top_pair_rows["stage1_model"].astype(str).tolist()
    stage2_pair_members = top_pair_rows["stage2_model"].astype(str).tolist()

    selected_models = {
        "exp1": _select_with_cap(
            configured=_models_for_task(config, "exp1"),
            mandatory=always_include.get(
                "exp1", ["logistic_regression", "xgboost"]
            ),
            ranked=rankings["exp1"],
            pipeline_members=[],
            top_k=top_k,
            max_models=max_models,
        ),
        "stage1": _select_with_cap(
            configured=_models_for_task(config, "stage1"),
            mandatory=always_include.get(
                "stage1", ["logistic_regression", "xgboost"]
            ),
            ranked=rankings["stage1"],
            pipeline_members=stage1_pair_members,
            top_k=top_k,
            max_models=max_models,
        ),
        "stage2": _select_with_cap(
            configured=_models_for_task(config, "stage2"),
            mandatory=always_include.get(
                "stage2", ["logistic_regression", "xgboost"]
            ),
            ranked=rankings["stage2"],
            pipeline_members=stage2_pair_members,
            top_k=top_k,
            max_models=max_models,
        ),
    }
    split_hash_path = run_dir / "common" / "splits" / "split_hash.txt"
    split_hash = split_hash_path.read_text(encoding="utf-8").strip()
    payload = {
        "schema_version": "exp123_staged_candidate_plan_v1",
        "baseline_run_id": run_id,
        "shared_split_hash": split_hash,
        "selection_split": "validation",
        "selected_models": selected_models,
        "model_rankings": rankings,
        "selection_settings": {
            "top_models_per_task": top_k,
            "top_pipeline_pairs": top_pairs,
            "max_models_per_task": max_models,
            "always_include": always_include,
        },
        "top_pipeline_pairs": top_pair_rows.to_dict(orient="records"),
        "pairwise_report_path": str(pairwise_path),
        "pairwise_report_sha256": sha256_file(pairwise_path),
        "source_summary_sha256": {
            task: sha256_file(
                run_dir
                / TASK_META[task]["output_relative"]
                / TASK_META[task]["summary_file"]
            )
            for task in TASK_META
        },
        "confirmatory_splits_accessed": False,
        "environment": environment_info(),
    }
    atomic_write_json(output_path, payload)
    (output_path.with_name(output_path.name + ".sha256")).write_text(
        sha256_file(output_path) + "\n", encoding="utf-8"
    )
    lines = [
        "# EXP123 Targeted Optimization Candidate Plan",
        "",
        f"- Baseline Run ID: `{run_id}`",
        f"- Shared split hash: `{split_hash}`",
        "- Selection split: Validation only",
        "",
    ]
    for task_key in ("exp1", "stage1", "stage2"):
        lines.append(f"## {task_key}")
        for index, model in enumerate(selected_models[task_key], start=1):
            lines.append(f"{index}. `{model}`")
        lines.append("")
    lines.extend(["## EXP2 top baseline pipeline pairs", ""])
    for rank, row in enumerate(top_pair_rows.to_dict(orient="records"), start=1):
        lines.append(
            f"{rank}. Stage1=`{row['stage1_model']}`, "
            f"Stage2=`{row['stage2_model']}`, "
            f"Macro AUROC={float(row['macro_auroc']):.6f}, "
            f"Brier={float(row['multiclass_brier']):.6f}"
        )
    lines.append("")
    (output_dir / "candidate_plan.md").write_text("\n".join(lines), encoding="utf-8")
    return CandidatePlanResult(
        baseline_run_id=run_id,
        output_path=str(output_path),
        split_hash=split_hash,
        selected_models=selected_models,
        pairwise_report_path=str(pairwise_path),
    )


def write_targeted_optimization_config(
    *,
    base_config_path: Path,
    candidate_plan_path: Path,
    output_path: Path,
) -> Path:
    base_config_path = base_config_path.expanduser().resolve()
    candidate_plan_path = candidate_plan_path.expanduser().resolve()
    output_path = output_path.expanduser().resolve()
    config = json.loads(base_config_path.read_text(encoding="utf-8"))
    plan = json.loads(candidate_plan_path.read_text(encoding="utf-8"))
    selected = plan["selected_models"]
    config["experiments"]["exp1_flat_3class"]["models"] = list(selected["exp1"])
    config["experiments"]["exp2_two_stage"]["stage1_mortality"]["models"] = list(
        selected["stage1"]
    )
    config["experiments"]["exp2_two_stage"]["stage2_long_stay"]["models"] = list(
        selected["stage2"]
    )
    staged = dict(config.get("staged_protocol", {}))
    staged["phase"] = "targeted_optimization"
    staged["linked_baseline"] = {
        "run_id": plan["baseline_run_id"],
        "candidate_plan_path": str(candidate_plan_path),
        "candidate_plan_sha256": sha256_file(candidate_plan_path),
        "shared_split_hash": plan["shared_split_hash"],
    }
    config["staged_protocol"] = staged
    targeted = dict(staged.get("targeted_optimization", {}))
    if "bootstrap_repeats" in targeted:
        repeats = int(targeted["bootstrap_repeats"])
        config["evaluation"]["binary_bootstrap_repeats"] = repeats
        config["evaluation"]["multiclass_bootstrap_repeats"] = repeats
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return output_path


def verify_linked_baseline(
    *,
    config: Mapping[str, Any],
    run_id: str,
    run_dir: Path,
) -> dict[str, Any]:
    linked = dict(config.get("staged_protocol", {}).get("linked_baseline", {}))
    if not linked:
        raise ValueError("staged_protocol.linked_baseline is required.")
    plan_path = Path(str(linked["candidate_plan_path"])).expanduser().resolve()
    if not plan_path.is_file():
        raise FileNotFoundError(plan_path)
    actual_plan_hash = sha256_file(plan_path)
    if actual_plan_hash != str(linked["candidate_plan_sha256"]):
        raise ValueError("Baseline candidate plan SHA-256 mismatch.")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    current_split_hash = (
        run_dir / "common" / "splits" / "split_hash.txt"
    ).read_text(encoding="utf-8").strip()
    expected_split_hash = str(linked["shared_split_hash"])
    if current_split_hash != expected_split_hash:
        raise ValueError(
            "Targeted optimization split hash differs from baseline split hash: "
            f"{current_split_hash} != {expected_split_hash}"
        )
    configured = {
        "exp1": _models_for_task(config, "exp1"),
        "stage1": _models_for_task(config, "stage1"),
        "stage2": _models_for_task(config, "stage2"),
    }
    expected_models = {key: list(value) for key, value in plan["selected_models"].items()}
    if configured != expected_models:
        raise ValueError(
            f"Targeted config model lists differ from candidate plan: {configured} != {expected_models}"
        )
    output_dir = run_dir / "common" / "staged_optimized"
    output_dir.mkdir(parents=True, exist_ok=True)
    lock_path = output_dir / "BASELINE_LINK_VERIFIED.lock.json"
    payload = {
        "job_type": "baseline_link_verification",
        "run_id": run_id,
        "baseline_run_id": linked["run_id"],
        "candidate_plan_path": str(plan_path),
        "candidate_plan_sha256": actual_plan_hash,
        "shared_split_hash": current_split_hash,
        "selected_models": configured,
        "confirmatory_splits_accessed": False,
    }
    atomic_write_json(lock_path, payload)
    (lock_path.with_name(lock_path.name + ".sha256")).write_text(
        sha256_file(lock_path) + "\n", encoding="utf-8"
    )
    return payload


def finalize_staged_optimization(
    *,
    config: Mapping[str, Any],
    run_id: str,
    run_dir: Path,
) -> dict[str, Any]:
    linked = dict(config.get("staged_protocol", {}).get("linked_baseline", {}))
    baseline_run_id = str(linked.get("run_id", ""))
    baseline_runs_dir = Path(config["paths"]["runs_dir"])
    if not baseline_runs_dir.is_absolute():
        baseline_runs_dir = Path(str(config["_project_root"])) / baseline_runs_dir
    baseline_run_dir = baseline_runs_dir.resolve() / baseline_run_id
    if not baseline_run_dir.is_dir():
        raise FileNotFoundError(baseline_run_dir)

    output_dir = run_dir / "common" / "staged_optimized"
    output_dir.mkdir(parents=True, exist_ok=True)
    task_rows: list[dict[str, Any]] = []
    for task_key in TASK_META:
        baseline_frame = _read_task_summary(baseline_run_dir, task_key).copy()
        optimized_frame = _read_task_summary(run_dir, task_key).copy()
        baseline_frame["phase"] = "baseline"
        optimized_frame["phase"] = "optimized"
        task_rows.extend(baseline_frame.to_dict(orient="records"))
        task_rows.extend(optimized_frame.to_dict(orient="records"))
        combined = pd.concat([baseline_frame, optimized_frame], ignore_index=True)
        combined.to_csv(output_dir / f"{task_key}_baseline_vs_optimized.csv", index=False)

    baseline_pair = pd.read_csv(
        baseline_run_dir
        / "common"
        / "staged_baseline"
        / "exp2_pairwise_validation"
        / "pairwise_pipeline_validation.csv"
    )
    optimized_pair = pd.read_csv(
        run_dir
        / "common"
        / "staged_optimized"
        / "exp2_pairwise_validation"
        / "pairwise_pipeline_validation.csv"
    )
    pair_compare = pd.concat([baseline_pair, optimized_pair], ignore_index=True)
    pair_compare.to_csv(output_dir / "exp2_baseline_vs_optimized_pairs.csv", index=False)
    best_baseline = _sorted_pairwise_rows(baseline_pair.to_dict(orient="records"))[0]
    best_optimized = _sorted_pairwise_rows(optimized_pair.to_dict(orient="records"))[0]
    payload = {
        "schema_version": "exp123_staged_optimization_summary_v1",
        "run_id": run_id,
        "baseline_run_id": baseline_run_id,
        "shared_split_hash": linked.get("shared_split_hash"),
        "selected_models": {
            "exp1": _models_for_task(config, "exp1"),
            "stage1": _models_for_task(config, "stage1"),
            "stage2": _models_for_task(config, "stage2"),
        },
        "best_baseline_exp2_pair": best_baseline,
        "best_optimized_exp2_pair": best_optimized,
        "macro_auroc_change": float(best_optimized["macro_auroc"])
        - float(best_baseline["macro_auroc"]),
        "multiclass_brier_change": float(best_optimized["multiclass_brier"])
        - float(best_baseline["multiclass_brier"]),
        "confirmatory_splits_accessed": False,
        "environment": environment_info(),
    }
    summary_path = output_dir / "staged_optimization_summary.json"
    atomic_write_json(summary_path, payload)
    lines = [
        "# EXP123 Baseline-to-Optimization Summary",
        "",
        f"- Baseline Run ID: `{baseline_run_id}`",
        f"- Optimized Run ID: `{run_id}`",
        f"- Shared split hash: `{linked.get('shared_split_hash')}`",
        "",
        "## Best baseline EXP2 pair",
        "",
        f"- Stage1: `{best_baseline['stage1_model']}`",
        f"- Stage2: `{best_baseline['stage2_model']}`",
        f"- Macro AUROC: {float(best_baseline['macro_auroc']):.6f}",
        f"- Multiclass Brier: {float(best_baseline['multiclass_brier']):.6f}",
        "",
        "## Best optimized EXP2 pair",
        "",
        f"- Stage1: `{best_optimized['stage1_model']}`",
        f"- Stage2: `{best_optimized['stage2_model']}`",
        f"- Macro AUROC: {float(best_optimized['macro_auroc']):.6f}",
        f"- Multiclass Brier: {float(best_optimized['multiclass_brier']):.6f}",
        "",
        f"- Macro AUROC change: {payload['macro_auroc_change']:+.6f}",
        f"- Multiclass Brier change: {payload['multiclass_brier_change']:+.6f}",
        "",
    ]
    (output_dir / "staged_optimization_summary.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    lock_path = output_dir / "STAGED_OPTIMIZATION_COMPLETED.lock.json"
    atomic_write_json(
        lock_path,
        {
            "run_id": run_id,
            "baseline_run_id": baseline_run_id,
            "summary_sha256": sha256_file(summary_path),
            "confirmatory_splits_accessed": False,
        },
    )
    (lock_path.with_name(lock_path.name + ".sha256")).write_text(
        sha256_file(lock_path) + "\n", encoding="utf-8"
    )
    return payload
