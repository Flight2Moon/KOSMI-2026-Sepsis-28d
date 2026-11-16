from __future__ import annotations

import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from ..evaluation_multiclass import calculate_multiclass_metrics
from ..utils import atomic_write_json, sha256_file
from .contract import FeatureContract
from .modeling import build_lightgbm_multiclass, predict_three_class_probabilities
from .preprocessing import build_study28_preprocessor
from .protocol import Study28Protocol
from .resource_plan import child_thread_environment, read_int_env

CLASS_NAMES = ("early_dc", "death", "prolonged")


@dataclass(frozen=True)
class NestedCVResult:
    locked_search_parameters: dict[str, Any]
    locked_vote_count: int
    outer_folds: int
    inner_folds: int
    selection_feature_set: str
    selection_features: tuple[str, ...]
    fold_metrics_path: str
    inner_scores_path: str
    oof_predictions_path: str
    summary_path: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TemporalEvaluationResult:
    development_n: int
    evaluation_n: int
    development_groups: tuple[str, ...]
    evaluation_groups: tuple[str, ...]
    metrics: dict[str, float | int | None]
    predictions_path: str
    metrics_path: str
    model_path: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Study28FreezeResult:
    manifest_path: str
    manifest_sha256: str
    external_model_path: str
    external_model_sha256: str
    protocol_sha256: str
    feature_list_sha256: str
    mimic_sha256: str
    primary_feature_set: str
    primary_features: tuple[str, ...]
    locked_parameters: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _validate_three_class_y(y: np.ndarray, *, context: str) -> None:
    values = np.asarray(y, dtype=np.int64).reshape(-1)
    observed = tuple(sorted(np.unique(values).tolist()))
    if observed != (0, 1, 2):
        raise ValueError(f"{context} must contain all three Study28 classes; got {observed}")


def _model_parameters(
    protocol: Study28Protocol,
    search_parameters: Mapping[str, Any],
) -> dict[str, Any]:
    parameters = dict(protocol.primary_model.parameters)
    # Search parameters intentionally override their pre-declared locked values.
    parameters.update(dict(search_parameters))
    return parameters


def _fit_lightgbm(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    feature_names: Sequence[str],
    parameters: Mapping[str, Any],
    *,
    n_jobs: int | None = None,
):
    preprocessor = build_study28_preprocessor("lightgbm", feature_names)
    X_fit = preprocessor.fit_transform(X_train)
    model = build_lightgbm_multiclass(parameters, n_jobs=n_jobs)
    model.fit(X_fit, y_train)
    return preprocessor, model


def _score_macro_auroc(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    metrics = calculate_multiclass_metrics(y_true, probabilities, CLASS_NAMES)
    value = metrics.get("macro_auroc")
    if value is None:
        raise ValueError("macro_auroc is undefined for a Study28 validation fold.")
    return float(value)


def _canonical_candidate(candidate: Mapping[str, Any]) -> str:
    return json.dumps(dict(candidate), sort_keys=True, separators=(",", ":"))


def _evaluate_inner_candidate(
    *,
    candidate_index: int,
    candidate: Mapping[str, Any],
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    inner_splits: Sequence[tuple[np.ndarray, np.ndarray]],
    feature_names: Sequence[str],
    protocol: Study28Protocol,
    model_threads: int,
) -> tuple[int, dict[str, Any], list[float], list[dict[str, Any]]]:
    scores: list[float] = []
    rows: list[dict[str, Any]] = []
    for inner_index, (fit_index, valid_index) in enumerate(inner_splits):
        preprocessor, model = _fit_lightgbm(
            X_train.iloc[fit_index],
            y_train[fit_index],
            feature_names,
            _model_parameters(protocol, candidate),
            n_jobs=model_threads,
        )
        probabilities = predict_three_class_probabilities(
            model, preprocessor.transform(X_train.iloc[valid_index])
        )
        score = _score_macro_auroc(y_train[valid_index], probabilities)
        scores.append(score)
        rows.append(
            {
                "candidate_index": candidate_index,
                "inner_fold": inner_index,
                "num_leaves": int(candidate["num_leaves"]),
                "n_estimators": int(candidate["n_estimators"]),
                "macro_auroc": score,
            }
        )
    return candidate_index, dict(candidate), scores, rows


def _run_outer_fold(
    *,
    outer_index: int,
    train_index: np.ndarray,
    test_index: np.ndarray,
    X: pd.DataFrame,
    y: np.ndarray,
    features: tuple[str, ...],
    protocol: Study28Protocol,
    grid: tuple[dict[str, Any], ...],
    lightgbm_threads: int = -1,
) -> dict[str, Any]:
    X_train = X.iloc[train_index]
    y_train = y[train_index]
    inner = StratifiedKFold(
        n_splits=protocol.nested_cv.inner_folds,
        shuffle=True,
        random_state=protocol.nested_cv.seed + outer_index,
    )
    inner_splits = tuple(
        (fit_index, valid_index)
        for fit_index, valid_index in inner.split(X_train, y_train)
    )
    candidate_means: list[float] = []
    inner_rows: list[dict[str, Any]] = []
    for candidate_index, candidate in enumerate(grid):
        _, _, scores, rows = _evaluate_inner_candidate(
            candidate_index=candidate_index,
            candidate=candidate,
            X_train=X_train,
            y_train=y_train,
            inner_splits=inner_splits,
            feature_names=features,
            protocol=protocol,
            model_threads=lightgbm_threads,
        )
        candidate_means.append(float(np.mean(scores)))
        for row in rows:
            inner_rows.append({"outer_fold": outer_index, **row})

    best_index = int(np.argmax(np.asarray(candidate_means, dtype=np.float64)))
    selected = dict(grid[best_index])
    preprocessor, model = _fit_lightgbm(
        X_train,
        y_train,
        features,
        _model_parameters(protocol, selected),
        n_jobs=lightgbm_threads,
    )
    probabilities = predict_three_class_probabilities(
        model, preprocessor.transform(X.iloc[test_index])
    )
    metrics = calculate_multiclass_metrics(y[test_index], probabilities, CLASS_NAMES)
    row = {
        "outer_fold": outer_index,
        "train_n": int(len(train_index)),
        "evaluation_n": int(len(test_index)),
        "selected_candidate_index": best_index,
        "selected_num_leaves": int(selected["num_leaves"]),
        "selected_n_estimators": int(selected["n_estimators"]),
    }
    row.update(metrics)
    return {
        "outer_index": outer_index,
        "test_index": np.asarray(test_index, dtype=np.int64),
        "probabilities": probabilities,
        "selected": selected,
        "fold_row": row,
        "inner_rows": inner_rows,
    }


def _finalize_nested_cv_results(
    *,
    mimic: pd.DataFrame,
    features: tuple[str, ...],
    protocol: Study28Protocol,
    grid: tuple[dict[str, Any], ...],
    outer_results: Sequence[Mapping[str, Any]],
    output_dir: Path,
    execution_metadata: Mapping[str, Any] | None = None,
) -> NestedCVResult:
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    outer_results = sorted(outer_results, key=lambda item: int(item["outer_index"]))
    oof = np.full((len(mimic), 3), np.nan, dtype=np.float64)
    fold_rows: list[dict[str, Any]] = []
    inner_rows: list[dict[str, Any]] = []
    selected_candidates: list[dict[str, Any]] = []
    for result in outer_results:
        test_index = np.asarray(result["test_index"], dtype=np.int64)
        probabilities = np.asarray(result["probabilities"], dtype=np.float64)
        oof[test_index] = probabilities
        fold_rows.append(dict(result["fold_row"]))
        inner_rows.extend(dict(row) for row in result["inner_rows"])
        selected = dict(result["selected"])
        selected_candidates.append(selected)
        metrics = result["fold_row"]
        print(
            f"[Study28 nested CV] outer={int(result['outer_index']) + 1}/{protocol.nested_cv.outer_folds} "
            f"selected={selected} macro_auroc={float(metrics['macro_auroc']):.6f}",
            flush=True,
        )

    if len(outer_results) != protocol.nested_cv.outer_folds:
        raise RuntimeError(
            f"Expected {protocol.nested_cv.outer_folds} outer-fold results, got {len(outer_results)}."
        )
    if np.isnan(oof).any():
        raise RuntimeError("Nested CV did not produce exactly one out-of-fold prediction per MIMIC row.")

    counts = Counter(_canonical_candidate(value) for value in selected_candidates)
    locked = max(
        grid,
        key=lambda candidate: (
            counts[_canonical_candidate(candidate)],
            -grid.index(candidate),
        ),
    )
    locked = dict(locked)
    locked_count = int(counts[_canonical_candidate(locked)])
    expected = dict(protocol.primary_model.expected_locked_search_parameters)
    if protocol.nested_cv.enforce_expected_locked_parameters and locked != expected:
        raise RuntimeError(
            "Nested CV did not reproduce the professor-locked LightGBM search parameters. "
            f"Expected={expected}, observed={locked}. Do not continue to confirmatory evaluation."
        )

    fold_frame = pd.DataFrame(fold_rows)
    inner_frame = pd.DataFrame(inner_rows)
    fold_metrics_path = output_dir / "nested_cv_fold_metrics.csv"
    inner_scores_path = output_dir / "nested_cv_inner_scores.csv"
    oof_path = output_dir / "nested_cv_oof_predictions.csv"
    summary_path = output_dir / "nested_cv_summary.json"
    fold_frame.to_csv(fold_metrics_path, index=False)
    inner_frame.to_csv(inner_scores_path, index=False)

    id_columns = [column for column in ("subject_id", "hadm_id", "stay_id") if column in mimic.columns]
    oof_frame = mimic.loc[:, id_columns + ["y", "label28"]].copy()
    oof_frame["p_early_dc"] = oof[:, 0]
    oof_frame["p_death"] = oof[:, 1]
    oof_frame["p_prolonged"] = oof[:, 2]
    oof_frame.to_csv(oof_path, index=False)

    numeric_metric_columns = [
        column for column in fold_frame.columns
        if column not in {
            "outer_fold", "train_n", "evaluation_n", "selected_candidate_index",
            "selected_num_leaves", "selected_n_estimators",
            "class_0_name", "class_1_name", "class_2_name",
        }
        and pd.api.types.is_numeric_dtype(fold_frame[column])
    ]
    aggregate = {
        column: {
            "mean": float(fold_frame[column].mean()),
            "std": float(fold_frame[column].std(ddof=1)),
        }
        for column in numeric_metric_columns
    }
    summary = {
        "protocol": {
            "outer_folds": protocol.nested_cv.outer_folds,
            "inner_folds": protocol.nested_cv.inner_folds,
            "seed": protocol.nested_cv.seed,
            "selection_metric": protocol.nested_cv.selection_metric,
            "selection_feature_set": protocol.nested_cv.selection_feature_set,
            **dict(execution_metadata or {}),
        },
        "grid": list(grid),
        "outer_selected_candidates": selected_candidates,
        "vote_counts": {key: int(value) for key, value in counts.items()},
        "locked_search_parameters": locked,
        "locked_vote_count": locked_count,
        "aggregate_outer_metrics": aggregate,
    }
    atomic_write_json(summary_path, summary)
    return NestedCVResult(
        locked_search_parameters=locked,
        locked_vote_count=locked_count,
        outer_folds=protocol.nested_cv.outer_folds,
        inner_folds=protocol.nested_cv.inner_folds,
        selection_feature_set=protocol.nested_cv.selection_feature_set,
        selection_features=features,
        fold_metrics_path=str(fold_metrics_path),
        inner_scores_path=str(inner_scores_path),
        oof_predictions_path=str(oof_path),
        summary_path=str(summary_path),
    )


def run_nested_cv(
    *,
    mimic: pd.DataFrame,
    feature_names: Sequence[str],
    protocol: Study28Protocol,
    output_dir: Path,
) -> NestedCVResult:
    """In-process nested CV, intended for tests/small datasets.

    Production Alpha 2 uses ``run_nested_cv_isolated`` so every outer fold gets
    a fresh Python/OpenMP process and can be resumed from a persisted fold file.
    """
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    features = tuple(str(value) for value in feature_names)
    X = mimic.loc[:, list(features)]
    y = pd.to_numeric(mimic["y"], errors="raise").to_numpy(dtype=np.int64)
    _validate_three_class_y(y, context="MIMIC nested CV")
    outer = StratifiedKFold(
        n_splits=protocol.nested_cv.outer_folds,
        shuffle=True,
        random_state=protocol.nested_cv.seed,
    )
    grid = tuple(dict(item) for item in protocol.nested_cv.hyperparameter_grid)
    results = []
    for outer_index, (train_index, test_index) in enumerate(outer.split(X, y)):
        results.append(
            _run_outer_fold(
                outer_index=outer_index,
                train_index=train_index,
                test_index=test_index,
                X=X,
                y=y,
                features=features,
                protocol=protocol,
                grid=grid,
            )
        )
    return _finalize_nested_cv_results(
        mimic=mimic,
        features=features,
        protocol=protocol,
        grid=grid,
        outer_results=results,
        output_dir=output_dir,
        execution_metadata={"execution_mode": "in_process"},
    )


def run_nested_cv_isolated(
    *,
    mimic: pd.DataFrame,
    feature_names: Sequence[str],
    protocol: Study28Protocol,
    dataset_root: Path,
    project_root: Path,
    output_dir: Path,
) -> NestedCVResult:
    """Run outer folds in isolated processes with frozen resource-aware concurrency.

    Statistical folds, seeds, candidate grid, and scoring are unchanged. Only the
    execution schedule is parallelized. Each child receives an explicit BLAS/OpenMP
    thread budget, preventing nested oversubscription while allowing the machine-wide
    CPU budget to remain close to fully occupied.
    """
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_root = Path(dataset_root).expanduser().resolve()
    project_root = Path(project_root).expanduser().resolve()
    features = tuple(str(value) for value in feature_names)
    grid = tuple(dict(item) for item in protocol.nested_cv.hyperparameter_grid)
    fold_dir = output_dir / "outer_fold_checkpoints"
    fold_dir.mkdir(parents=True, exist_ok=True)

    worker_count = min(
        protocol.nested_cv.outer_folds,
        read_int_env("STUDY28_NESTED_CV_WORKERS", 1),
    )
    threads_per_worker = read_int_env("STUDY28_NESTED_CV_THREADS", 1)

    results_by_index: dict[int, dict[str, Any]] = {}
    missing: list[int] = []
    for outer_index in range(protocol.nested_cv.outer_folds):
        checkpoint = fold_dir / f"outer_fold_{outer_index}.joblib"
        if checkpoint.is_file():
            results_by_index[outer_index] = joblib.load(checkpoint)
        else:
            missing.append(outer_index)

    def execute_fold(outer_index: int) -> tuple[int, dict[str, Any]]:
        checkpoint = fold_dir / f"outer_fold_{outer_index}.joblib"
        log_path = fold_dir / f"outer_fold_{outer_index}.log"
        command = [
            sys.executable,
            "-m",
            "los_baseline.study28.cv_worker",
            "--dataset-root", str(dataset_root),
            "--project-root", str(project_root),
            "--outer-index", str(outer_index),
            "--output", str(checkpoint),
        ]
        env = child_thread_environment(os.environ.copy(), threads_per_worker)
        existing_pythonpath = env.get("PYTHONPATH", "")
        src_path = str(project_root / "src")
        env["PYTHONPATH"] = src_path + (os.pathsep + existing_pythonpath if existing_pythonpath else "")
        with log_path.open("w", encoding="utf-8") as log_handle:
            completed = subprocess.run(
                command, cwd=project_root, env=env, stdout=log_handle,
                stderr=subprocess.STDOUT, text=True, check=False,
            )
        if completed.returncode != 0 or not checkpoint.is_file():
            log_text = log_path.read_text(encoding="utf-8", errors="replace")[-4000:]
            raise RuntimeError(
                f"Study28 outer fold {outer_index} failed in isolated process. Log tail:\n{log_text}"
            )
        return outer_index, joblib.load(checkpoint)

    if missing:
        print(
            f"[Study28 nested CV] resource plan: concurrent_outer_folds={worker_count}, "
            f"threads_per_fold={threads_per_worker}, missing_folds={len(missing)}",
            flush=True,
        )
        if worker_count <= 1:
            for outer_index in missing:
                idx, result = execute_fold(outer_index)
                results_by_index[idx] = result
        else:
            with ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="study28-cv") as pool:
                futures = {pool.submit(execute_fold, idx): idx for idx in missing}
                for future in as_completed(futures):
                    idx, result = future.result()
                    results_by_index[idx] = result
                    print(f"[Study28 nested CV] outer fold checkpoint completed: {idx + 1}", flush=True)

    results = [results_by_index[index] for index in range(protocol.nested_cv.outer_folds)]
    return _finalize_nested_cv_results(
        mimic=mimic,
        features=features,
        protocol=protocol,
        grid=grid,
        outer_results=results,
        output_dir=output_dir,
        execution_metadata={
            "execution_mode": "isolated_outer_fold_processes_resource_planned",
            "outer_fold_checkpointing": True,
            "concurrent_outer_fold_workers": worker_count,
            "lightgbm_threads_per_outer_worker": threads_per_worker,
            "resource_plan_path": os.environ.get("STUDY28_RESOURCE_PLAN_PATH"),
        },
    )

def split_mimic_temporal(
    mimic: pd.DataFrame,
    protocol: Study28Protocol,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    spec = protocol.temporal_split
    if spec.column not in mimic.columns:
        raise KeyError(f"MIMIC temporal split column is missing: {spec.column}")
    observed = set(mimic[spec.column].dropna().astype(str).unique().tolist())
    configured = set(spec.development_groups) | set(spec.evaluation_groups)
    unassigned = sorted(observed - configured)
    missing_configured = sorted(configured - observed)
    if unassigned:
        raise ValueError(f"MIMIC contains temporal groups not assigned by protocol: {unassigned}")
    if missing_configured:
        raise ValueError(f"Protocol temporal groups are absent from MIMIC: {missing_configured}")

    development = mimic.loc[mimic[spec.column].isin(spec.development_groups)].copy()
    evaluation = mimic.loc[mimic[spec.column].isin(spec.evaluation_groups)].copy()
    if len(development) + len(evaluation) != len(mimic):
        raise ValueError("Temporal split does not partition all MIMIC rows exactly once.")
    if spec.expected_development_n is not None and len(development) != spec.expected_development_n:
        raise ValueError(
            f"Unexpected temporal development N: {len(development)} != {spec.expected_development_n}"
        )
    if spec.expected_evaluation_n is not None and len(evaluation) != spec.expected_evaluation_n:
        raise ValueError(
            f"Unexpected temporal evaluation N: {len(evaluation)} != {spec.expected_evaluation_n}"
        )
    _validate_three_class_y(development["y"].to_numpy(), context="Temporal development")
    _validate_three_class_y(evaluation["y"].to_numpy(), context="Temporal evaluation")
    return development.reset_index(drop=True), evaluation.reset_index(drop=True)


def run_temporal_evaluation(
    *,
    mimic: pd.DataFrame,
    feature_names: Sequence[str],
    protocol: Study28Protocol,
    locked_search_parameters: Mapping[str, Any],
    output_dir: Path,
) -> TemporalEvaluationResult:
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    development, evaluation = split_mimic_temporal(mimic, protocol)
    features = tuple(str(value) for value in feature_names)
    y_train = development["y"].to_numpy(dtype=np.int64)
    y_test = evaluation["y"].to_numpy(dtype=np.int64)
    preprocessor, model = _fit_lightgbm(
        development.loc[:, list(features)],
        y_train,
        features,
        _model_parameters(protocol, locked_search_parameters),
    )
    probabilities = predict_three_class_probabilities(
        model, preprocessor.transform(evaluation.loc[:, list(features)])
    )
    metrics = calculate_multiclass_metrics(y_test, probabilities, CLASS_NAMES)

    predictions_path = output_dir / "mimic_temporal_predictions.csv"
    metrics_path = output_dir / "mimic_temporal_metrics.json"
    model_path = output_dir / "mimic_temporal_model.joblib"
    id_columns = [column for column in ("subject_id", "hadm_id", "stay_id", "anchor_year_group") if column in evaluation.columns]
    pred = evaluation.loc[:, id_columns + ["y", "label28"]].copy()
    pred["p_early_dc"] = probabilities[:, 0]
    pred["p_death"] = probabilities[:, 1]
    pred["p_prolonged"] = probabilities[:, 2]
    pred.to_csv(predictions_path, index=False)
    atomic_write_json(
        metrics_path,
        {
            "setting": "mimic_temporal_2008_2016_to_2017_2022",
            "development_n": len(development),
            "evaluation_n": len(evaluation),
            "development_groups": list(protocol.temporal_split.development_groups),
            "evaluation_groups": list(protocol.temporal_split.evaluation_groups),
            "feature_set": protocol.primary_feature_set,
            "features": list(features),
            "locked_search_parameters": dict(locked_search_parameters),
            "model_parameters": _model_parameters(protocol, locked_search_parameters),
            "metrics": metrics,
            "methodological_note": (
                "Supportive temporal evaluation. The professor-provided locked model design was established "
                "before this V9 execution; do not describe this as a newly untouched hyperparameter-selection holdout."
            ),
        },
    )
    joblib.dump(
        {
            "preprocessor": preprocessor,
            "model": model,
            "feature_names": features,
            "class_names": CLASS_NAMES,
            "locked_search_parameters": dict(locked_search_parameters),
        },
        model_path,
    )
    return TemporalEvaluationResult(
        development_n=len(development),
        evaluation_n=len(evaluation),
        development_groups=protocol.temporal_split.development_groups,
        evaluation_groups=protocol.temporal_split.evaluation_groups,
        metrics=metrics,
        predictions_path=str(predictions_path),
        metrics_path=str(metrics_path),
        model_path=str(model_path),
    )


def freeze_external_model(
    *,
    mimic: pd.DataFrame,
    feature_names: Sequence[str],
    feature_contract: FeatureContract,
    protocol: Study28Protocol,
    locked_search_parameters: Mapping[str, Any],
    mimic_sha256: str,
    output_dir: Path,
) -> Study28FreezeResult:
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    features = tuple(str(value) for value in feature_names)
    y = mimic["y"].to_numpy(dtype=np.int64)
    _validate_three_class_y(y, context="Full MIMIC external-development fit")
    preprocessor, model = _fit_lightgbm(
        mimic.loc[:, list(features)],
        y,
        features,
        _model_parameters(protocol, locked_search_parameters),
    )
    model_path = output_dir / "external_primary_model_frozen.joblib"
    joblib.dump(
        {
            "preprocessor": preprocessor,
            "model": model,
            "feature_names": features,
            "class_names": CLASS_NAMES,
            "locked_search_parameters": dict(locked_search_parameters),
            "protocol_sha256": protocol.sha256,
            "feature_list_sha256": feature_contract.sha256,
            "mimic_sha256": str(mimic_sha256),
        },
        model_path,
    )
    model_sha = sha256_file(model_path)
    manifest_path = output_dir / "study28_model_freeze.json"
    manifest = {
        "freeze_schema": "study28_model_freeze_v1",
        "study_id": protocol.study_id,
        "purpose": "Frozen full-MIMIC primary model for future external validation; no external predictions are generated in Alpha 2.",
        "protocol_sha256": protocol.sha256,
        "feature_list_sha256": feature_contract.sha256,
        "mimic_sha256": str(mimic_sha256),
        "primary_feature_set": protocol.primary_feature_set,
        "primary_features": list(features),
        "class_names": list(CLASS_NAMES),
        "locked_search_parameters": dict(locked_search_parameters),
        "model_parameters": _model_parameters(protocol, locked_search_parameters),
        "external_model_path": str(model_path),
        "external_model_sha256": model_sha,
        "required_external_sites_next_phase": list(protocol.required_external_sites),
        "external_evaluation_performed": False,
    }
    atomic_write_json(manifest_path, manifest)
    manifest_sha = sha256_file(manifest_path)
    return Study28FreezeResult(
        manifest_path=str(manifest_path),
        manifest_sha256=manifest_sha,
        external_model_path=str(model_path),
        external_model_sha256=model_sha,
        protocol_sha256=protocol.sha256,
        feature_list_sha256=feature_contract.sha256,
        mimic_sha256=str(mimic_sha256),
        primary_feature_set=protocol.primary_feature_set,
        primary_features=features,
        locked_parameters=dict(locked_search_parameters),
    )
