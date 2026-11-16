from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Mapping

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedGroupKFold


@dataclass(frozen=True)
class CalibrationSummary:
    enabled: bool
    method: str
    cv_folds: int
    cv_seed: int
    ensemble: bool
    group_column: str
    group_count: int
    split_hash: str
    implementation: str = "sklearn_calibrated_classifier_cv_oof"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _split_hash(splits: list[tuple[np.ndarray, np.ndarray]]) -> str:
    import hashlib

    digest = hashlib.sha256()
    for fold_index, (train_idx, valid_idx) in enumerate(splits, start=1):
        digest.update(f"fold:{fold_index}|".encode())
        digest.update(np.asarray(train_idx, dtype=np.int64).tobytes())
        digest.update(b"|")
        digest.update(np.asarray(valid_idx, dtype=np.int64).tobytes())
        digest.update(b"\n")
    return digest.hexdigest()


def build_group_calibrated_classifier(
    *,
    estimator: Any,
    features: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    config: Mapping[str, Any] | None,
    group_column: str,
) -> tuple[Any, CalibrationSummary]:
    """Fit an OOF probability calibrator with frozen patient-group folds.

    ``ensemble=False`` uses cross-validated out-of-fold predictions to fit the
    calibrator, then refits one final base estimator on the complete training
    set. The resulting object has one deployable estimator plus the learned
    calibration mapping.
    """
    cfg = dict(config or {})
    method = str(cfg.get("method", "sigmoid")).lower()
    if method not in {"sigmoid", "isotonic"}:
        raise ValueError(f"Unsupported calibration method: {method}")
    folds = int(cfg.get("cv_folds", 5))
    seed = int(cfg.get("seed", 4242))
    if folds < 2:
        raise ValueError("calibration.cv_folds must be at least 2.")
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    groups = np.asarray(groups).reshape(-1)
    if len(labels) != len(groups) or len(labels) != len(features):
        raise ValueError("Calibration features, labels, and groups must align.")
    if len(np.unique(groups)) < folds:
        raise ValueError("Not enough patient groups for calibration CV.")
    splitter = StratifiedGroupKFold(
        n_splits=folds,
        shuffle=True,
        random_state=seed,
    )
    splits = [
        (np.asarray(train_idx, dtype=np.int64), np.asarray(valid_idx, dtype=np.int64))
        for train_idx, valid_idx in splitter.split(features, labels, groups)
    ]
    for fold_index, (train_idx, valid_idx) in enumerate(splits, start=1):
        if set(groups[train_idx].tolist()) & set(groups[valid_idx].tolist()):
            raise RuntimeError(f"Calibration group leakage in fold {fold_index}.")
        if set(np.unique(labels[train_idx]).tolist()) != set(np.unique(labels).tolist()):
            raise ValueError(f"Calibration training fold {fold_index} is missing a class.")
        if set(np.unique(labels[valid_idx]).tolist()) != set(np.unique(labels).tolist()):
            raise ValueError(f"Calibration validation fold {fold_index} is missing a class.")
    calibrated = CalibratedClassifierCV(
        estimator=estimator,
        method=method,
        cv=splits,
        ensemble=False,
        n_jobs=1,
    )
    calibrated.fit(features, labels)
    summary = CalibrationSummary(
        enabled=True,
        method=method,
        cv_folds=folds,
        cv_seed=seed,
        ensemble=False,
        group_column=str(group_column),
        group_count=int(len(np.unique(groups))),
        split_hash=_split_hash(splits),
    )
    return calibrated, summary


def explainability_estimator(model: Any) -> Any:
    """Return the final base estimator stored inside a calibrated wrapper."""
    candidate = getattr(model, "estimator", None)
    if bool(getattr(model, "_exp123_calibrated_wrapper", False)) and candidate is not None:
        return candidate
    calibrated = getattr(model, "calibrated_classifiers_", None)
    if calibrated:
        candidate = getattr(calibrated[0], "estimator", None)
        if candidate is not None:
            return candidate
    return model
