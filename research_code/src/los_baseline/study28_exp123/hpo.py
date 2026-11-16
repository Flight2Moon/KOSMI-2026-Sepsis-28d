from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from ..study28.preprocessing import build_study28_preprocessor
from ..utils import atomic_write_json
from .core import (
    TASKS,
    TASK_EXP1,
    _build_model,
    binary_metrics,
    majority_downsample_indices,
    multiclass_metrics,
    task_frame,
)


def trial_model_config(
    trial: optuna.Trial,
    *,
    model_name: str,
    base_config: Mapping[str, Any],
) -> dict[str, Any]:
    cfg = json.loads(json.dumps(dict(base_config)))
    p = cfg.setdefault("parameters", {})
    if model_name == "logistic_regression":
        p["solver"] = "lbfgs"
        p["C"] = trial.suggest_float("C", 1e-4, 1e3, log=True)
        p["max_iter"] = 3000
        p["tol"] = trial.suggest_float("tol", 1e-7, 1e-4, log=True)
        p.pop("l1_ratio", None)
    elif model_name == "decision_tree":
        p["criterion"] = trial.suggest_categorical("criterion", ["gini", "entropy", "log_loss"])
        p["max_depth"] = trial.suggest_int("max_depth", 2, 20)
        p["min_samples_split"] = trial.suggest_int("min_samples_split", 2, 80, log=True)
        p["min_samples_leaf"] = trial.suggest_int("min_samples_leaf", 1, 50, log=True)
        p["max_features"] = trial.suggest_float("max_features", 0.3, 1.0)
    elif model_name == "random_forest":
        p["n_estimators"] = trial.suggest_int("n_estimators", 200, 1200, step=100)
        p["max_depth"] = trial.suggest_int("max_depth", 4, 30)
        p["min_samples_split"] = trial.suggest_int("min_samples_split", 2, 80, log=True)
        p["min_samples_leaf"] = trial.suggest_int("min_samples_leaf", 1, 40, log=True)
        p["max_features"] = trial.suggest_float("max_features", 0.2, 1.0)
        p["bootstrap"] = True
        p["max_samples"] = trial.suggest_float("max_samples", 0.6, 1.0)
    elif model_name == "xgboost":
        p["n_estimators"] = trial.suggest_int("n_estimators", 200, 1400, step=100)
        p["learning_rate"] = trial.suggest_float("learning_rate", 0.01, 0.2, log=True)
        p["max_depth"] = trial.suggest_int("max_depth", 2, 9)
        p["min_child_weight"] = trial.suggest_float("min_child_weight", 0.5, 30.0, log=True)
        p["subsample"] = trial.suggest_float("subsample", 0.6, 1.0)
        p["colsample_bytree"] = trial.suggest_float("colsample_bytree", 0.5, 1.0)
        p["gamma"] = trial.suggest_float("gamma", 1e-8, 5.0, log=True)
        p["reg_alpha"] = trial.suggest_float("reg_alpha", 1e-8, 20.0, log=True)
        p["reg_lambda"] = trial.suggest_float("reg_lambda", 1e-3, 100.0, log=True)
        p["tree_method"] = "hist"
        p["device"] = "cpu"
    elif model_name == "lightgbm":
        p["n_estimators"] = trial.suggest_int("n_estimators", 100, 1000, step=50)
        p["learning_rate"] = trial.suggest_float("learning_rate", 0.01, 0.15, log=True)
        p["num_leaves"] = trial.suggest_int("num_leaves", 7, 63)
        p["max_depth"] = trial.suggest_int("max_depth", -1, 16)
        p["min_child_samples"] = trial.suggest_int("min_child_samples", 20, 150)
        p["subsample"] = trial.suggest_float("subsample", 0.6, 1.0)
        p["colsample_bytree"] = trial.suggest_float("colsample_bytree", 0.5, 1.0)
        p["reg_alpha"] = trial.suggest_float("reg_alpha", 1e-8, 20.0, log=True)
        p["reg_lambda"] = trial.suggest_float("reg_lambda", 1e-4, 100.0, log=True)
    elif model_name == "svm":
        p["kernel"] = "rbf"
        p["C"] = trial.suggest_float("C", 1e-3, 1e3, log=True)
        p["gamma"] = trial.suggest_float("gamma", 1e-6, 1.0, log=True)
        p["probability"] = True
        p["cache_size"] = max(2048, int(p.get("cache_size", 2048)))
    else:
        raise KeyError(model_name)
    cfg.setdefault("training_policy", {})["class_weight_mode"] = "none"
    return cfg


def _objective_value(task_name: str, y: np.ndarray, prediction: np.ndarray) -> float:
    if task_name == TASK_EXP1:
        return float(multiclass_metrics(y, prediction)["macro_auroc"])
    return float(binary_metrics(y, prediction, 0.5)["auprc"])


def cross_validated_score(
    *,
    frame: pd.DataFrame,
    task_name: str,
    model_name: str,
    model_config: Mapping[str, Any],
    feature_names: Sequence[str],
    seed: int,
    folds: int,
    strategy: str = "natural",
    max_majority_ratio: float = 2.0,
) -> tuple[float, list[float]]:
    task_data, y = task_frame(frame, task_name)
    X = task_data.loc[:, list(feature_names)].reset_index(drop=True)
    y = np.asarray(y, dtype=np.int64)
    splitter = StratifiedKFold(n_splits=int(folds), shuffle=True, random_state=int(seed))
    scores: list[float] = []
    for fold_index, (fit_idx, val_idx) in enumerate(splitter.split(X, y)):
        fit_idx = np.asarray(fit_idx, dtype=np.int64)
        if strategy == "majority_downsampling":
            keep_local = majority_downsample_indices(
                y[fit_idx], max_majority_ratio=max_majority_ratio, seed=seed + fold_index
            )
            fit_idx = fit_idx[keep_local]
        pre = build_study28_preprocessor(model_name, feature_names)
        X_fit = pre.fit_transform(X.iloc[fit_idx])
        model = _build_model(
            model_name=model_name,
            model_config=model_config,
            task_type=TASKS[task_name].task_type,
            seed=seed + fold_index,
        )
        model.fit(X_fit, y[fit_idx])
        X_val = pre.transform(X.iloc[val_idx])
        if task_name == TASK_EXP1:
            prob = np.asarray(model.predict_proba(X_val), dtype=np.float64)
            classes = np.asarray(model.classes_, dtype=int)
            order = [int(np.where(classes == c)[0][0]) for c in (0, 1, 2)]
            pred = prob[:, order]
        else:
            if hasattr(model, "predict_proba"):
                prob = np.asarray(model.predict_proba(X_val), dtype=np.float64)
                classes = np.asarray(model.classes_, dtype=int)
                pred = prob[:, int(np.where(classes == 1)[0][0])]
            else:
                decision = np.asarray(model.decision_function(X_val), dtype=np.float64).reshape(-1)
                pred = 1.0 / (1.0 + np.exp(-np.clip(decision, -30.0, 30.0)))
        scores.append(_objective_value(task_name, y[val_idx], pred))
    return float(np.mean(scores)), scores


def optimize_model(
    *,
    frame: pd.DataFrame,
    task_name: str,
    model_name: str,
    base_model_config: Mapping[str, Any],
    feature_names: Sequence[str],
    output_dir: Path,
    seed: int,
    folds: int,
    n_trials: int,
    max_majority_ratio: float,
) -> dict[str, Any]:
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    storage = f"sqlite:///{(output_dir / 'optuna.sqlite3').resolve()}"
    study = optuna.create_study(
        study_name=f"study28_exp123__{task_name}__{model_name}",
        direction="maximize",
        storage=storage,
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=int(seed), multivariate=True),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=max(5, n_trials // 5), n_warmup_steps=2),
    )

    def objective(trial: optuna.Trial) -> float:
        cfg = trial_model_config(trial, model_name=model_name, base_config=base_model_config)
        score, fold_scores = cross_validated_score(
            frame=frame,
            task_name=task_name,
            model_name=model_name,
            model_config=cfg,
            feature_names=feature_names,
            seed=seed,
            folds=folds,
            strategy="natural",
            max_majority_ratio=max_majority_ratio,
        )
        trial.set_user_attr("fold_scores", fold_scores)
        trial.set_user_attr("model_config", cfg)
        return score

    completed = sum(1 for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE)
    remaining = max(0, int(n_trials) - completed)
    if remaining:
        study.optimize(objective, n_trials=remaining, gc_after_trial=True, show_progress_bar=False)

    best_trial = study.best_trial
    best_cfg = dict(best_trial.user_attrs["model_config"])
    natural_score = float(best_trial.value)
    downsample_score, downsample_folds = cross_validated_score(
        frame=frame,
        task_name=task_name,
        model_name=model_name,
        model_config=best_cfg,
        feature_names=feature_names,
        seed=seed,
        folds=folds,
        strategy="majority_downsampling",
        max_majority_ratio=max_majority_ratio,
    )
    selected_strategy = "majority_downsampling" if downsample_score > natural_score else "natural"
    payload = {
        "task": task_name,
        "model": model_name,
        "objective": TASKS[task_name].objective,
        "n_trials_requested": int(n_trials),
        "n_trials_complete": sum(1 for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE),
        "best_trial_number": int(best_trial.number),
        "best_natural_cv_score": natural_score,
        "downsampling_cv_score": float(downsample_score),
        "selected_strategy": selected_strategy,
        "max_majority_ratio": float(max_majority_ratio),
        "best_model_config": best_cfg,
        "natural_fold_scores": list(best_trial.user_attrs.get("fold_scores", [])),
        "downsampling_fold_scores": downsample_folds,
    }
    atomic_write_json(output_dir / "optimization_result.json", payload)
    return payload
