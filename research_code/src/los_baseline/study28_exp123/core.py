from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split

from ..models import build_task_sklearn_model
from ..study28.contract import load_feature_contract
from ..study28.datasets import DatasetRegistry, load_dataset_registry
from ..study28.feature_sets import resolve_feature_set
from ..study28.preprocessing import build_study28_preprocessor
from ..study28.protocol import Study28Protocol, load_study28_protocol
from ..utils import atomic_write_json


EXP1_NAME = "EXP1_flat_3class"
EXP2_NAME = "EXP2_two_stage"
EXP3_NAME = "EXP3_survivor_binary"

TASK_EXP1 = "exp1_flat_3class"
TASK_STAGE1 = "stage1_mortality"
TASK_STAGE2 = "stage2_long_stay"

CLASS_NAMES = ("early_dc", "death", "prolonged")
MODEL_FAMILIES = (
    "logistic_regression",
    "decision_tree",
    "random_forest",
    "xgboost",
    "lightgbm",
    "svm",
)


@dataclass(frozen=True)
class TaskSpec:
    name: str
    task_type: str
    population: str
    objective: str


TASKS: dict[str, TaskSpec] = {
    TASK_EXP1: TaskSpec(TASK_EXP1, "multiclass", "all", "macro_auroc"),
    TASK_STAGE1: TaskSpec(TASK_STAGE1, "binary", "all", "auprc"),
    TASK_STAGE2: TaskSpec(TASK_STAGE2, "binary", "survivors", "auprc"),
}


@dataclass(frozen=True)
class Study28Exp123Context:
    dataset_root: Path
    protocol: Study28Protocol
    registry: DatasetRegistry
    features: tuple[str, ...]
    mimic: pd.DataFrame
    development: pd.DataFrame
    temporal: pd.DataFrame
    external: dict[str, pd.DataFrame]


def load_json(path: Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return payload


def load_context(
    *,
    dataset_root: Path,
    project_root: Path,
    feature_set: str = "FS_NOAPS",
) -> Study28Exp123Context:
    dataset_root = Path(dataset_root).expanduser().resolve()
    project_root = Path(project_root).expanduser().resolve()
    protocol = load_study28_protocol(project_root / "configs" / "study28_final.json")
    registry = load_dataset_registry(project_root / "configs" / "dataset_registry_28d.json")
    feature_contract = load_feature_contract(dataset_root / "feature_list.csv")
    features = resolve_feature_set(
        feature_set,
        feature_sets_path=project_root / "configs" / "feature_sets.json",
        contract=feature_contract,
    )

    mimic = registry["mimic"].load(dataset_root)
    required = set(features) | {"y", "label28", protocol.temporal_split.column}
    missing = sorted(required - set(mimic.columns))
    if missing:
        raise KeyError(f"MIMIC is missing Study28 EXP123 columns: {missing}")
    y = pd.to_numeric(mimic["y"], errors="raise").astype(int)
    if set(y.unique()) != {0, 1, 2}:
        raise ValueError(f"MIMIC Study28 y must contain 0/1/2, got {sorted(y.unique())}")
    mimic = mimic.copy()
    mimic["y"] = y

    temporal_col = protocol.temporal_split.column
    development = mimic.loc[
        mimic[temporal_col].astype(str).isin(protocol.temporal_split.development_groups)
    ].copy().reset_index(drop=True)
    temporal = mimic.loc[
        mimic[temporal_col].astype(str).isin(protocol.temporal_split.evaluation_groups)
    ].copy().reset_index(drop=True)
    if protocol.temporal_split.expected_development_n is not None:
        if len(development) != protocol.temporal_split.expected_development_n:
            raise ValueError(
                "Unexpected MIMIC development N: "
                f"{len(development)} != {protocol.temporal_split.expected_development_n}"
            )
    if protocol.temporal_split.expected_evaluation_n is not None:
        if len(temporal) != protocol.temporal_split.expected_evaluation_n:
            raise ValueError(
                "Unexpected MIMIC temporal N: "
                f"{len(temporal)} != {protocol.temporal_split.expected_evaluation_n}"
            )

    external: dict[str, pd.DataFrame] = {}
    for site in registry.required_external_sites:
        frame = registry[site].load(dataset_root)
        missing = sorted((set(features) | {"y", "label28"}) - set(frame.columns))
        if missing:
            raise KeyError(f"External site {site} is missing columns: {missing}")
        frame = frame.copy()
        frame["y"] = pd.to_numeric(frame["y"], errors="raise").astype(int)
        external[site] = frame.reset_index(drop=True)

    return Study28Exp123Context(
        dataset_root=dataset_root,
        protocol=protocol,
        registry=registry,
        features=features,
        mimic=mimic,
        development=development,
        temporal=temporal,
        external=external,
    )


def task_frame(frame: pd.DataFrame, task_name: str) -> tuple[pd.DataFrame, np.ndarray]:
    if task_name == TASK_EXP1:
        return frame, frame["y"].to_numpy(dtype=np.int64)
    if task_name == TASK_STAGE1:
        return frame, (frame["y"].to_numpy(dtype=np.int64) == 1).astype(np.int64)
    if task_name == TASK_STAGE2:
        survivors = frame.loc[frame["y"] != 1].copy().reset_index(drop=True)
        target = (survivors["y"].to_numpy(dtype=np.int64) == 2).astype(np.int64)
        return survivors, target
    raise KeyError(f"Unknown task: {task_name}")


def task_labels_for_all(frame: pd.DataFrame, task_name: str) -> np.ndarray:
    if task_name == TASK_EXP1:
        return frame["y"].to_numpy(dtype=np.int64)
    if task_name == TASK_STAGE1:
        return (frame["y"].to_numpy(dtype=np.int64) == 1).astype(np.int64)
    if task_name == TASK_STAGE2:
        return (frame["y"].to_numpy(dtype=np.int64) == 2).astype(np.int64)
    raise KeyError(task_name)


def development_train_validation_indices(
    frame: pd.DataFrame,
    *,
    validation_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    indices = np.arange(len(frame), dtype=np.int64)
    train_idx, valid_idx = train_test_split(
        indices,
        test_size=float(validation_fraction),
        random_state=int(seed),
        stratify=frame["y"].to_numpy(dtype=np.int64),
    )
    return np.sort(train_idx), np.sort(valid_idx)


def model_threads(default: int | None = None) -> int:
    for key in ("EXP123_ASSIGNED_CPU_THREADS", "STUDY28_SERIAL_MODEL_THREADS"):
        raw = os.environ.get(key, "").strip()
        if raw:
            try:
                return max(1, int(raw))
            except ValueError:
                pass
    if default is not None:
        return max(1, int(default))
    return max(1, int(os.cpu_count() or 1))


def default_model_config(model_name: str, config: Mapping[str, Any]) -> dict[str, Any]:
    models = dict(config.get("models", {}))
    if model_name not in models:
        raise KeyError(f"Model {model_name} missing from Study28 EXP123 config")
    payload = json.loads(json.dumps(models[model_name]))
    payload.setdefault("parameters", {})
    payload.setdefault("training_policy", {})["class_weight_mode"] = "none"
    return payload


def _build_model(
    *,
    model_name: str,
    model_config: Mapping[str, Any],
    task_type: str,
    seed: int,
):
    params = dict(model_config.get("parameters", {}))
    threads = model_threads()
    if model_name == "lightgbm":
        from lightgbm import LGBMClassifier

        params.setdefault("learning_rate", 0.05)
        params.setdefault("n_estimators", 200)
        params.setdefault("num_leaves", 15)
        params.setdefault("min_child_samples", 50)
        params.setdefault("random_state", seed)
        params.setdefault("verbose", -1)
        params["n_jobs"] = threads
        if task_type == "multiclass":
            params["objective"] = "multiclass"
            params["num_class"] = 3
        else:
            params["objective"] = "binary"
        return LGBMClassifier(**params)

    if model_name == "logistic_regression":
        # Baseline uses an efficient deterministic L2 solver.  The old SAGA
        # implementation remains available by setting solver=saga in config.
        params.setdefault("solver", "lbfgs")
        params.setdefault("C", 1.0)
        params.setdefault("max_iter", 2000)
        params.setdefault("tol", 1e-6)
        params.setdefault("fit_intercept", True)
        params.setdefault("random_state", seed)
        if params.get("solver") == "lbfgs":
            params.pop("l1_ratio", None)
        cfg = {"parameters": params, "training_policy": {"class_weight_mode": "none"}}
        return build_task_sklearn_model(
            model_name,
            cfg,
            seed=seed,
            task_type=task_type,
            num_classes=3 if task_type == "multiclass" else 2,
            class_weights={0: 1.0, 1: 1.0, 2: 1.0} if task_type == "multiclass" else {0: 1.0, 1: 1.0},
        )

    if model_name in {"decision_tree", "random_forest", "xgboost", "svm"}:
        if model_name in {"random_forest", "xgboost"}:
            params["n_jobs"] = threads
        cfg = {"parameters": params, "training_policy": {"class_weight_mode": "none"}}
        return build_task_sklearn_model(
            model_name,
            cfg,
            seed=seed,
            task_type=task_type,
            num_classes=3 if task_type == "multiclass" else 2,
            class_weights={0: 1.0, 1: 1.0, 2: 1.0} if task_type == "multiclass" else {0: 1.0, 1: 1.0},
        )
    raise KeyError(model_name)


@dataclass
class FittedTaskModel:
    task_name: str
    model_name: str
    feature_names: tuple[str, ...]
    preprocessor: Any
    model: Any
    threshold: float | None
    parameters: dict[str, Any]
    strategy: str = "natural"

    def predict_probability(self, frame: pd.DataFrame) -> np.ndarray:
        X = self.preprocessor.transform(frame.loc[:, list(self.feature_names)])
        if self.task_name == TASK_EXP1:
            probabilities = np.asarray(self.model.predict_proba(X), dtype=np.float64)
            classes = np.asarray(self.model.classes_, dtype=int)
            if set(classes.tolist()) != {0, 1, 2}:
                raise ValueError(f"Unexpected multiclass classes: {classes.tolist()}")
            order = [int(np.where(classes == c)[0][0]) for c in (0, 1, 2)]
            probabilities = probabilities[:, order]
            return probabilities
        if hasattr(self.model, "predict_proba"):
            probabilities = np.asarray(self.model.predict_proba(X), dtype=np.float64)
            classes = np.asarray(self.model.classes_, dtype=int)
            positive_index = int(np.where(classes == 1)[0][0])
            return probabilities[:, positive_index]
        decision = np.asarray(self.model.decision_function(X), dtype=np.float64).reshape(-1)
        # Monotonic pseudo-probability for optional non-probabilistic SVM paths.
        return 1.0 / (1.0 + np.exp(-np.clip(decision, -30.0, 30.0)))

    def save(self, output_dir: Path) -> None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.preprocessor, output_dir / "preprocessor.joblib")
        joblib.dump(self.model, output_dir / "model.joblib")
        atomic_write_json(
            output_dir / "model_manifest.json",
            {
                "task_name": self.task_name,
                "model_name": self.model_name,
                "feature_names": list(self.feature_names),
                "threshold": self.threshold,
                "parameters": self.parameters,
                "strategy": self.strategy,
            },
        )



def load_fitted_task_model(output_dir: Path) -> FittedTaskModel:
    output_dir = Path(output_dir)
    manifest = load_json(output_dir / "model_manifest.json")
    return FittedTaskModel(
        task_name=str(manifest["task_name"]),
        model_name=str(manifest["model_name"]),
        feature_names=tuple(str(v) for v in manifest["feature_names"]),
        preprocessor=joblib.load(output_dir / "preprocessor.joblib"),
        model=joblib.load(output_dir / "model.joblib"),
        threshold=(float(manifest["threshold"]) if manifest.get("threshold") is not None else None),
        parameters=dict(manifest.get("parameters", {})),
        strategy=str(manifest.get("strategy", "natural")),
    )

def majority_downsample_indices(
    y: np.ndarray,
    *,
    max_majority_ratio: float,
    seed: int,
) -> np.ndarray:
    labels = np.asarray(y, dtype=np.int64).reshape(-1)
    classes, counts = np.unique(labels, return_counts=True)
    if len(classes) < 2:
        raise ValueError("Downsampling requires at least two classes")
    min_count = int(counts.min())
    cap = max(min_count, int(math.floor(min_count * float(max_majority_ratio))))
    rng = np.random.default_rng(int(seed))
    selected: list[np.ndarray] = []
    for class_value, count in zip(classes, counts):
        idx = np.flatnonzero(labels == class_value)
        keep = min(int(count), cap)
        if keep < len(idx):
            idx = np.sort(rng.choice(idx, size=keep, replace=False))
        selected.append(idx)
    return np.sort(np.concatenate(selected).astype(np.int64))


def select_youden_threshold(y_true: np.ndarray, probability: np.ndarray) -> float:
    labels = np.asarray(y_true, dtype=np.int64)
    prob = np.asarray(probability, dtype=np.float64)
    fpr, tpr, thresholds = roc_curve(labels, prob)
    finite = np.isfinite(thresholds)
    if not finite.any():
        return 0.5
    scores = tpr[finite] - fpr[finite]
    threshold_values = thresholds[finite]
    best = int(np.argmax(scores))
    return float(np.clip(threshold_values[best], 0.0, 1.0))


def fit_task_model(
    *,
    task_name: str,
    model_name: str,
    model_config: Mapping[str, Any],
    feature_names: Sequence[str],
    train_frame: pd.DataFrame,
    threshold_frame: pd.DataFrame | None,
    seed: int,
    strategy: str = "natural",
    max_majority_ratio: float = 2.0,
    fixed_threshold: float | None = None,
) -> FittedTaskModel:
    spec = TASKS[task_name]
    train_task_frame, y_train = task_frame(train_frame, task_name)
    X_train = train_task_frame.loc[:, list(feature_names)].reset_index(drop=True)
    if strategy == "majority_downsampling":
        keep = majority_downsample_indices(
            y_train,
            max_majority_ratio=max_majority_ratio,
            seed=seed,
        )
        X_train = X_train.iloc[keep].reset_index(drop=True)
        y_train = y_train[keep]
    elif strategy != "natural":
        raise ValueError(f"Unsupported training strategy: {strategy}")

    pre = build_study28_preprocessor(model_name, feature_names)
    X_fit = pre.fit_transform(X_train)
    model = _build_model(
        model_name=model_name,
        model_config=model_config,
        task_type=spec.task_type,
        seed=seed,
    )
    model.fit(X_fit, y_train)

    threshold: float | None = None
    if spec.task_type == "binary":
        if fixed_threshold is not None:
            threshold = float(fixed_threshold)
        else:
            if threshold_frame is None:
                raise ValueError("Binary task requires threshold_frame or fixed_threshold")
            threshold_task, y_threshold = task_frame(threshold_frame, task_name)
            p_threshold = _positive_probability(model, pre, threshold_task, feature_names)
            threshold = select_youden_threshold(y_threshold, p_threshold)
    return FittedTaskModel(
        task_name=task_name,
        model_name=model_name,
        feature_names=tuple(feature_names),
        preprocessor=pre,
        model=model,
        threshold=threshold,
        parameters=dict(model_config.get("parameters", {})),
        strategy=strategy,
    )


def _positive_probability(model: Any, pre: Any, frame: pd.DataFrame, features: Sequence[str]) -> np.ndarray:
    X = pre.transform(frame.loc[:, list(features)])
    if hasattr(model, "predict_proba"):
        probs = np.asarray(model.predict_proba(X), dtype=np.float64)
        classes = np.asarray(model.classes_, dtype=int)
        return probs[:, int(np.where(classes == 1)[0][0])]
    decision = np.asarray(model.decision_function(X), dtype=np.float64).reshape(-1)
    return 1.0 / (1.0 + np.exp(-np.clip(decision, -30.0, 30.0)))


def multiclass_metrics(y_true: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    y = np.asarray(y_true, dtype=np.int64)
    p = np.asarray(probabilities, dtype=np.float64)
    p = np.clip(p, 1e-15, 1.0)
    row_sums = p.sum(axis=1, keepdims=True)
    if np.any(row_sums <= 0.0):
        raise ValueError("Multiclass probabilities contain a non-positive row sum")
    p = p / row_sums
    predicted = np.argmax(p, axis=1)
    one_hot = np.eye(3, dtype=np.float64)[y]
    per_auroc = [roc_auc_score((y == k).astype(int), p[:, k]) for k in range(3)]
    per_auprc = [average_precision_score((y == k).astype(int), p[:, k]) for k in range(3)]
    return {
        "macro_auroc": float(np.mean(per_auroc)),
        "macro_auprc": float(np.mean(per_auprc)),
        "early_auroc": float(per_auroc[0]),
        "death_auroc": float(per_auroc[1]),
        "prolonged_auroc": float(per_auroc[2]),
        "early_auprc": float(per_auprc[0]),
        "death_auprc": float(per_auprc[1]),
        "prolonged_auprc": float(per_auprc[2]),
        "log_loss": float(log_loss(y, p, labels=[0, 1, 2])),
        "brier": float(np.mean(np.sum((p - one_hot) ** 2, axis=1))),
        "accuracy": float(accuracy_score(y, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(y, predicted)),
        "macro_f1": float(f1_score(y, predicted, average="macro", zero_division=0)),
    }


def binary_metrics(y_true: np.ndarray, probability: np.ndarray, threshold: float) -> dict[str, float]:
    y = np.asarray(y_true, dtype=np.int64)
    p = np.asarray(probability, dtype=np.float64)
    pred = (p >= float(threshold)).astype(np.int64)
    return {
        "auroc": float(roc_auc_score(y, p)),
        "auprc": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, np.column_stack([1.0 - p, p]), labels=[0, 1])),
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "threshold": float(threshold),
    }


def evaluate_task_model(model: FittedTaskModel, frame: pd.DataFrame) -> dict[str, float]:
    if model.task_name == TASK_EXP1:
        return multiclass_metrics(frame["y"].to_numpy(dtype=np.int64), model.predict_probability(frame))
    task_eval, y = task_frame(frame, model.task_name)
    p = model.predict_probability(task_eval)
    assert model.threshold is not None
    return binary_metrics(y, p, model.threshold)


def cascade_probabilities(
    *,
    stage1: FittedTaskModel,
    stage2: FittedTaskModel,
    frame: pd.DataFrame,
) -> np.ndarray:
    p_death = np.asarray(stage1.predict_probability(frame), dtype=np.float64)
    p_prolonged_cond = np.asarray(stage2.predict_probability(frame), dtype=np.float64)
    p_survive = 1.0 - p_death
    p_early = p_survive * (1.0 - p_prolonged_cond)
    p_prolonged = p_survive * p_prolonged_cond
    p = np.column_stack([p_early, p_death, p_prolonged])
    p = np.clip(p, 1e-12, 1.0)
    p /= p.sum(axis=1, keepdims=True)
    return p


def evaluate_cascade(
    *,
    stage1: FittedTaskModel,
    stage2: FittedTaskModel,
    frame: pd.DataFrame,
) -> dict[str, float]:
    return multiclass_metrics(frame["y"].to_numpy(dtype=np.int64), cascade_probabilities(stage1=stage1, stage2=stage2, frame=frame))


def model_output_row(
    *,
    phase: str,
    experiment: str,
    task: str,
    model_name: str,
    split_name: str,
    strategy: str,
    metrics: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "phase": phase,
        "experiment": experiment,
        "task": task,
        "model": model_name,
        "strategy": strategy,
        "split": split_name,
        **{str(k): v for k, v in metrics.items()},
    }
