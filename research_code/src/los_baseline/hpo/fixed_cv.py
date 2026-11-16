from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from ..utils import atomic_write_json


@dataclass(frozen=True)
class FixedFold:
    repeat: int
    fold: int
    train_indices: np.ndarray
    validation_indices: np.ndarray


def _hash_folds(folds: Sequence[FixedFold]) -> str:
    digest = hashlib.sha256()
    for item in folds:
        digest.update(f"repeat={item.repeat};fold={item.fold}\n".encode())
        digest.update(np.asarray(item.train_indices, dtype=np.int64).tobytes())
        digest.update(b"|")
        digest.update(np.asarray(item.validation_indices, dtype=np.int64).tobytes())
        digest.update(b"\n")
    return digest.hexdigest()


def build_fixed_group_folds(
    *,
    labels: np.ndarray,
    groups: np.ndarray,
    n_splits: int,
    seeds: Sequence[int],
) -> list[FixedFold]:
    labels = np.asarray(labels, dtype=np.int64)
    groups = np.asarray(groups).astype(str)
    folds: list[FixedFold] = []
    expected = set(np.unique(labels).tolist())
    for repeat, seed in enumerate(seeds, start=1):
        splitter = StratifiedGroupKFold(
            n_splits=int(n_splits), shuffle=True, random_state=int(seed)
        )
        for fold, (train_idx, valid_idx) in enumerate(
            splitter.split(np.zeros(len(labels)), labels, groups), start=1
        ):
            train_idx = np.asarray(train_idx, dtype=np.int64)
            valid_idx = np.asarray(valid_idx, dtype=np.int64)
            if set(np.unique(labels[train_idx]).tolist()) != expected:
                raise ValueError(f"Fixed CV repeat {repeat} fold {fold} misses a train class.")
            if set(np.unique(labels[valid_idx]).tolist()) != expected:
                raise ValueError(f"Fixed CV repeat {repeat} fold {fold} misses a validation class.")
            if set(groups[train_idx].tolist()) & set(groups[valid_idx].tolist()):
                raise ValueError(f"Group leakage in repeat {repeat} fold {fold}.")
            folds.append(FixedFold(repeat, fold, train_idx, valid_idx))
    return folds


def save_fixed_folds(
    output_dir: Path,
    *,
    folds: Sequence[FixedFold],
    seeds: Sequence[int],
    n_splits: int,
    sample_count: int,
    outcome_horizon_days: float | None = None,
) -> tuple[Path, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {}
    rows: list[dict[str, Any]] = []
    for item in folds:
        prefix = f"r{item.repeat}_f{item.fold}"
        arrays[f"{prefix}_train"] = item.train_indices
        arrays[f"{prefix}_validation"] = item.validation_indices
        rows.append({
            "repeat": item.repeat,
            "fold": item.fold,
            "train_count": int(len(item.train_indices)),
            "validation_count": int(len(item.validation_indices)),
        })
    npz_path = output_dir / "fixed_folds.npz"
    np.savez_compressed(npz_path, **arrays)
    split_hash = _hash_folds(folds)
    manifest = {
        "schema_version": "exp123_fixed_group_cv_v1",
        "n_splits": int(n_splits),
        "seeds": [int(v) for v in seeds],
        "repeat_count": len(seeds),
        "fold_count": len(folds),
        "sample_count": int(sample_count),
        "split_hash": split_hash,
        "folds": rows,
        "npz_file": npz_path.name,
    }
    if outcome_horizon_days is not None:
        manifest["outcome_horizon_days"] = float(outcome_horizon_days)
    atomic_write_json(output_dir / "fixed_cv_manifest.json", manifest)
    (output_dir / "fixed_cv_hash.txt").write_text(split_hash + "\n", encoding="utf-8")
    return npz_path, split_hash


def load_fixed_folds(output_dir: Path) -> tuple[list[FixedFold], dict[str, Any]]:
    manifest = json.loads((output_dir / "fixed_cv_manifest.json").read_text(encoding="utf-8"))
    data = np.load(output_dir / manifest["npz_file"])
    folds: list[FixedFold] = []
    for row in manifest["folds"]:
        prefix = f"r{row['repeat']}_f{row['fold']}"
        folds.append(FixedFold(
            repeat=int(row["repeat"]),
            fold=int(row["fold"]),
            train_indices=np.asarray(data[f"{prefix}_train"], dtype=np.int64),
            validation_indices=np.asarray(data[f"{prefix}_validation"], dtype=np.int64),
        ))
    if _hash_folds(folds) != manifest["split_hash"]:
        raise ValueError("Fixed CV split hash mismatch.")
    return folds, manifest
