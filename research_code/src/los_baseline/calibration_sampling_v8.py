from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold

from .exp123_data import TaskSplitData
from .imbalance.sampling import TrainingSampler, sampling_manifest_payload
from .models import build_task_sklearn_model, resolve_class_weights
from .preprocessing import build_preprocessor, transform_features, validate_training_features
from .task_spec import TaskDefinition


@dataclass(frozen=True)
class SamplingAwareCalibrationSummary:
    enabled: bool
    method: str
    cv_folds: int
    cv_seed: int
    group_column: str
    group_count: int
    split_hash: str
    natural_oof_rows: int
    synthetic_rows_in_calibration_targets: int
    sampling_strategy: str
    sampling_member_seed: int
    fold_sampling: tuple[dict[str, Any], ...]
    implementation: str = "exp123_v8_sampling_aware_oof_sigmoid"

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["fold_sampling"] = list(self.fold_sampling)
        return payload


def _split_hash(splits: Sequence[tuple[np.ndarray, np.ndarray]]) -> str:
    digest = hashlib.sha256()
    for index, (train_idx, valid_idx) in enumerate(splits, start=1):
        digest.update(f"fold:{index}|".encode())
        digest.update(np.asarray(train_idx, dtype=np.int64).tobytes())
        digest.update(b"|")
        digest.update(np.asarray(valid_idx, dtype=np.int64).tobytes())
        digest.update(b"\n")
    return digest.hexdigest()


def _ordered_class_indices(model: Any, expected: Sequence[int]) -> list[int]:
    classes = np.asarray(getattr(model, "classes_", expected), dtype=np.int64).reshape(-1)
    mapping = {int(value): index for index, value in enumerate(classes.tolist())}
    missing = [int(value) for value in expected if int(value) not in mapping]
    if missing:
        raise ValueError(f"Calibration estimator is missing classes: {missing}")
    return [mapping[int(value)] for value in expected]


def calibration_scores(
    model: Any,
    X: np.ndarray,
    task: TaskDefinition,
) -> tuple[np.ndarray, str]:
    """Return stable raw scores used by the frozen sigmoid mapping."""
    expected = list(task.expected_classes)
    if hasattr(model, "decision_function"):
        scores = np.asarray(model.decision_function(X), dtype=np.float64)
        if task.task_type == "binary":
            if scores.ndim == 2:
                order = _ordered_class_indices(model, expected)
                if scores.shape[1] == len(order):
                    return scores[:, order[1]], "decision_function"
            return scores.reshape(-1), "decision_function"
        if scores.ndim != 2:
            raise ValueError(
                "Multiclass decision_function must return a 2D matrix for calibration."
            )
        order = _ordered_class_indices(model, expected)
        return scores[:, order], "decision_function"

    if not hasattr(model, "predict_proba"):
        raise TypeError("Sampling-aware calibration requires decision_function or predict_proba.")
    probabilities = np.asarray(model.predict_proba(X), dtype=np.float64)
    if task.task_type == "binary":
        if probabilities.ndim != 2 or probabilities.shape[1] < 2:
            raise ValueError("Binary predict_proba returned an invalid shape.")
        order = _ordered_class_indices(model, expected)
        return probabilities[:, order[1]], "predict_proba"
    if probabilities.ndim != 2:
        raise ValueError("Multiclass predict_proba returned an invalid shape.")
    order = _ordered_class_indices(model, expected)
    return probabilities[:, order], "predict_proba"


class FrozenOOFSigmoidClassifier:
    """Final sampled estimator plus a calibration map fit on natural OOF rows."""

    _exp123_calibrated_wrapper = True

    def __init__(
        self,
        *,
        estimator: Any,
        calibrators: Sequence[LogisticRegression],
        task_type: str,
        expected_classes: Sequence[int],
        score_kind: str,
    ) -> None:
        self.estimator = estimator
        self.calibrators = list(calibrators)
        self.task_type = str(task_type)
        self.classes_ = np.asarray(list(expected_classes), dtype=np.int64)
        self.score_kind = str(score_kind)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        # Build a tiny task-like contract locally to avoid serialising a full
        # TaskDefinition inside every wrapper.
        class _Task:
            pass

        task = _Task()
        task.task_type = self.task_type
        task.expected_classes = tuple(int(v) for v in self.classes_.tolist())
        task.num_classes = len(task.expected_classes)
        scores, kind = calibration_scores(self.estimator, X, task)  # type: ignore[arg-type]
        if kind != self.score_kind:
            raise RuntimeError(
                f"Calibration score kind changed from {self.score_kind} to {kind}."
            )
        if self.task_type == "binary":
            positive = self.calibrators[0].predict_proba(
                np.asarray(scores, dtype=np.float64).reshape(-1, 1)
            )[:, 1]
            positive = np.clip(positive, 1e-12, 1.0 - 1e-12)
            return np.column_stack([1.0 - positive, positive])

        score_matrix = np.asarray(scores, dtype=np.float64)
        calibrated = np.column_stack(
            [
                calibrator.predict_proba(score_matrix[:, [index]])[:, 1]
                for index, calibrator in enumerate(self.calibrators)
            ]
        )
        calibrated = np.clip(calibrated, 1e-12, None)
        return calibrated / calibrated.sum(axis=1, keepdims=True)


def _fit_sigmoid_maps(
    labels: np.ndarray,
    scores: np.ndarray,
    task: TaskDefinition,
) -> list[LogisticRegression]:
    y = np.asarray(labels, dtype=np.int64).reshape(-1)
    if task.task_type == "binary":
        calibrator = LogisticRegression(
            solver="lbfgs",
            C=1e12,
            l1_ratio=0.0,
            max_iter=2000,
            random_state=0,
        )
        calibrator.fit(np.asarray(scores, dtype=np.float64).reshape(-1, 1), y)
        return [calibrator]

    score_matrix = np.asarray(scores, dtype=np.float64)
    calibrators: list[LogisticRegression] = []
    for score_index, class_code in enumerate(task.expected_classes):
        binary = (y == int(class_code)).astype(np.int64)
        if len(np.unique(binary)) != 2:
            raise ValueError(f"Calibration target for class {class_code} is degenerate.")
        calibrator = LogisticRegression(
            solver="lbfgs",
            C=1e12,
            l1_ratio=0.0,
            max_iter=2000,
            random_state=0,
        )
        calibrator.fit(score_matrix[:, [score_index]], binary)
        calibrators.append(calibrator)
    return calibrators


def fit_sampling_aware_calibration_map(
    *,
    natural_train: TaskSplitData,
    feature_names: Sequence[str],
    task: TaskDefinition,
    model_name: str,
    model_config: Mapping[str, Any],
    sampler: TrainingSampler,
    config: Mapping[str, Any],
    group_column: str,
    sampling_member_seed: int,
) -> tuple[list[LogisticRegression], SamplingAwareCalibrationSummary, str]:
    """Fit calibration maps from natural OOF predictions only.

    Every calibration fold fits preprocessing on the natural fold-train, then
    applies the sampling strategy only to that fold-train. Fold validation rows
    are untouched natural patients. The returned sigmoid map therefore learns
    the natural training prevalence while preserving the sampled base model.
    """
    cfg = dict(config)
    method = str(cfg.get("method", "sigmoid")).lower()
    if method != "sigmoid":
        raise ValueError(
            "Sampling-aware V8 calibration currently supports method='sigmoid' only."
        )
    folds = int(cfg.get("cv_folds", 5))
    seed = int(cfg.get("seed", 4242))
    if group_column not in natural_train.metadata.columns:
        raise KeyError(f"Calibration group column is missing: {group_column}")
    validate_training_features(natural_train.raw_features, list(feature_names))
    labels = np.asarray(natural_train.labels, dtype=np.int64)
    groups = natural_train.metadata[group_column].to_numpy()
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    splits = [
        (np.asarray(tr, dtype=np.int64), np.asarray(va, dtype=np.int64))
        for tr, va in splitter.split(natural_train.raw_features, labels, groups)
    ]

    score_shape = (len(labels),) if task.task_type == "binary" else (len(labels), task.num_classes)
    oof_scores = np.full(score_shape, np.nan, dtype=np.float64)
    score_kind: str | None = None
    fold_sampling: list[dict[str, Any]] = []

    for fold_index, (fit_idx, valid_idx) in enumerate(splits, start=1):
        if set(groups[fit_idx].tolist()) & set(groups[valid_idx].tolist()):
            raise RuntimeError(f"Calibration group leakage in fold {fold_index}.")
        if set(np.unique(labels[fit_idx]).tolist()) != set(task.expected_classes):
            raise ValueError(f"Calibration fold {fold_index} train is missing a class.")
        if set(np.unique(labels[valid_idx]).tolist()) != set(task.expected_classes):
            raise ValueError(f"Calibration fold {fold_index} validation is missing a class.")

        fold_train = TaskSplitData(
            split_name="train",
            raw_features=natural_train.raw_features.iloc[fit_idx].reset_index(drop=True),
            labels=labels[fit_idx],
            metadata=natural_train.metadata.iloc[fit_idx].reset_index(drop=True),
        )
        sampled = sampler.sample(
            fold_train,
            feature_names=feature_names,
            seed=int(sampling_member_seed) + fold_index * 100_003,
            purpose=f"calibration_oof_fold_{fold_index}",
        )
        preprocessor = build_preprocessor(model_name)
        preprocessor.fit(fold_train.raw_features.loc[:, list(feature_names)])
        X_fit = transform_features(preprocessor, sampled.raw_features.loc[:, list(feature_names)])
        X_valid = transform_features(
            preprocessor,
            natural_train.raw_features.iloc[valid_idx].loc[:, list(feature_names)],
        )
        class_weights, method_name = resolve_class_weights(
            sampled.labels,
            task.expected_classes,
            dict(model_config),
        )
        if method_name != "none":
            raise ValueError(
                "Sampled V8 tracks require class_weight_mode='none' so the isolated "
                "effect is the sampling strategy."
            )
        estimator = build_task_sklearn_model(
            model_name=model_name,
            model_config=dict(model_config),
            seed=seed + fold_index,
            task_type=task.task_type,
            num_classes=task.num_classes,
            class_weights=class_weights,
        )
        estimator.fit(X_fit, sampled.labels)
        fold_scores, current_kind = calibration_scores(estimator, X_valid, task)
        if score_kind is None:
            score_kind = current_kind
        elif score_kind != current_kind:
            raise RuntimeError("Calibration folds returned inconsistent score kinds.")
        oof_scores[valid_idx] = fold_scores
        fold_sampling.append(
            {
                "fold": fold_index,
                "natural_fit_count": int(len(fit_idx)),
                "natural_validation_count": int(len(valid_idx)),
                "sampled_fit_count": int(len(sampled.labels)),
                "synthetic_fit_count": int(sampled.synthetic_count),
                "sampling": sampling_manifest_payload(sampled),
            }
        )

    if not np.isfinite(oof_scores).all():
        raise RuntimeError("Sampling-aware calibration did not produce complete OOF scores.")
    calibrators = _fit_sigmoid_maps(labels, oof_scores, task)
    summary = SamplingAwareCalibrationSummary(
        enabled=True,
        method="sigmoid",
        cv_folds=folds,
        cv_seed=seed,
        group_column=str(group_column),
        group_count=int(len(np.unique(groups))),
        split_hash=_split_hash(splits),
        natural_oof_rows=int(len(labels)),
        synthetic_rows_in_calibration_targets=0,
        sampling_strategy=str(getattr(sampler, "strategy_name", type(sampler).__name__)),
        sampling_member_seed=int(sampling_member_seed),
        fold_sampling=tuple(fold_sampling),
    )
    assert score_kind is not None
    return calibrators, summary, score_kind
