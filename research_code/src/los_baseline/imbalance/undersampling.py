from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from ..exp123_data import TaskDataBundle, TaskSplitData
from ..hpo.fixed_cv import load_fixed_folds
from ..optimization_objectives import load_objective_definition, robust_selection_score
from ..training_engine import train_and_evaluate_single_model
from ..utils import atomic_write_json
from .ensemble import ensemble_member_prediction_files
from .sampled_partition import fit_predict_sampled_partition
from .sampling import SampledTrainingData, TrainingSampler, _natural_provenance, assert_natural_split


@dataclass(frozen=True)
class UndersamplingPlan:
    ratio: float
    seed: int
    majority_class: int
    original_counts: dict[int, int]
    selected_counts: dict[int, int]
    selected_indices: tuple[int, ...]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["selected_indices"] = list(self.selected_indices)
        return payload


def majority_undersample_indices(
    labels: np.ndarray,
    *,
    ratio: float,
    seed: int,
) -> UndersamplingPlan:
    y = np.asarray(labels, dtype=np.int64).reshape(-1)
    if float(ratio) <= 0:
        raise ValueError("Undersampling ratio must be positive.")
    classes, counts = np.unique(y, return_counts=True)
    if len(classes) < 2:
        raise ValueError("Undersampling requires at least two classes.")
    count_map = {int(c): int(n) for c, n in zip(classes, counts)}
    majority = int(classes[np.argmax(counts)])
    minority_max = max(count for cls, count in count_map.items() if cls != majority)
    majority_limit = min(
        count_map[majority],
        max(1, int(math.ceil(minority_max * float(ratio)))),
    )
    rng = np.random.default_rng(int(seed))
    selected: list[int] = []
    for cls in classes:
        indices = np.flatnonzero(y == cls)
        if int(cls) == majority and len(indices) > majority_limit:
            indices = np.sort(
                rng.choice(indices, size=majority_limit, replace=False)
            )
        selected.extend(indices.astype(int).tolist())
    selected_array = np.asarray(sorted(selected), dtype=np.int64)
    selected_counts = {
        int(c): int((y[selected_array] == c).sum()) for c in classes
    }
    for cls in classes:
        if int(cls) != majority and selected_counts[int(cls)] != count_map[int(cls)]:
            raise AssertionError("A minority-class row was removed by undersampling.")
    return UndersamplingPlan(
        ratio=float(ratio),
        seed=int(seed),
        majority_class=majority,
        original_counts=count_map,
        selected_counts=selected_counts,
        selected_indices=tuple(selected_array.tolist()),
    )


@dataclass(frozen=True)
class MajorityUndersamplingSampler(TrainingSampler):
    ratio: float
    strategy_name: str = "majority_undersampling"

    def sample(
        self,
        train_split: TaskSplitData,
        *,
        feature_names: Sequence[str],
        seed: int,
        purpose: str,
    ) -> SampledTrainingData:
        assert_natural_split(train_split)
        plan = majority_undersample_indices(
            np.asarray(train_split.labels, dtype=np.int64),
            ratio=float(self.ratio),
            seed=int(seed),
        )
        selected = np.asarray(plan.selected_indices, dtype=np.int64)
        raw = (
            train_split.raw_features.iloc[selected]
            .loc[:, list(feature_names)]
            .copy()
            .reset_index(drop=True)
        )
        labels = np.asarray(train_split.labels, dtype=np.int64)[selected]
        provenance = _natural_provenance(train_split).iloc[selected].reset_index(drop=True)
        manifest = {
            "schema_version": "exp123_sampling_v8_1",
            "strategy": self.strategy_name,
            "purpose": str(purpose),
            "seed": int(seed),
            "ratio": float(self.ratio),
            "natural_count": int(len(train_split.labels)),
            "synthetic_count": 0,
            "fit_count": int(len(labels)),
            "majority_class": int(plan.majority_class),
            "original_counts": dict(plan.original_counts),
            "selected_counts": dict(plan.selected_counts),
            "preserve_all_minority_samples": True,
            "validation_distribution_modified": False,
            "internal_test_distribution_modified": False,
            "external_distribution_modified": False,
        }
        return SampledTrainingData(raw, labels, provenance, manifest)


def run_undersampling_search(
    *,
    task_data: TaskDataBundle,
    model_name: str,
    model_config: dict[str, Any],
    project_config: Mapping[str, Any],
    fixed_cv_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """V8 Track-4 ratio search and five-seed ensemble.

    Preprocessing is fitted on every natural fold-train before the selected
    majority rows are passed to the estimator. Fold validation and all final
    evaluation splits remain natural.
    """
    cfg = dict(project_config.get("undersampling", {}))
    ratios = [float(v) for v in cfg.get("ratios", [1, 2, 3, 4])]
    seeds = [int(v) for v in cfg.get("seeds", [42, 142, 242, 342, 442])]
    objective = load_objective_definition(
        task_data.task.task_name,
        project_config["optimization_objectives"],
    )
    folds, cv_manifest = load_fixed_folds(fixed_cv_dir)
    primary_folds = [f for f in folds if f.repeat == 1]
    natural_train = task_data.splits["train"]
    labels = np.asarray(natural_train.labels, dtype=np.int64)
    model_config = json.loads(json.dumps(model_config, default=str))
    model_config.setdefault("training_policy", {})["class_weight_mode"] = "none"

    rows: list[dict[str, Any]] = []
    for ratio in ratios:
        values: list[float] = []
        fold_rows: list[dict[str, Any]] = []
        sampler = MajorityUndersamplingSampler(ratio=ratio)
        for sampling_seed in seeds:
            for fixed in primary_folds:
                fold_train = TaskSplitData(
                    split_name="train",
                    raw_features=natural_train.raw_features.iloc[fixed.train_indices].reset_index(drop=True),
                    labels=labels[fixed.train_indices],
                    metadata=natural_train.metadata.iloc[fixed.train_indices].reset_index(drop=True),
                )
                _, metrics, diagnostics, _ = fit_predict_sampled_partition(
                    model_name=model_name,
                    model_config=model_config,
                    task=task_data.task,
                    feature_names=task_data.feature_names,
                    natural_train=fold_train,
                    evaluation_features=natural_train.raw_features.iloc[fixed.validation_indices].reset_index(drop=True),
                    evaluation_labels=labels[fixed.validation_indices],
                    sampler=sampler,
                    seed=sampling_seed + fixed.fold * 1000,
                )
                primary = float(metrics[objective.primary])
                values.append(primary)
                fold_rows.append(
                    {
                        "ratio": ratio,
                        "seed": sampling_seed,
                        "fold": fixed.fold,
                        **metrics,
                        "sampling": diagnostics["sampling"],
                        "preprocessor_fit_count": diagnostics["preprocessor_fit_count"],
                        "model_fit_count": diagnostics["model_fit_count"],
                    }
                )
        mean = float(np.mean(values))
        std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        score = robust_selection_score(
            aggregate_metrics={
                f"mean_{objective.primary}": mean,
                f"std_{objective.primary}": std,
            },
            objective=objective,
            stability_penalty=float(cfg.get("stability_penalty", 0.5)),
        )
        rows.append(
            {
                "ratio": ratio,
                "mean_primary": mean,
                "std_primary": std,
                "selection_score": score,
                "fold_results": fold_rows,
            }
        )

    selected = max(rows, key=lambda row: row["selection_score"])
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [{key: value for key, value in row.items() if key != "fold_results"} for row in rows]
    ).to_csv(output_dir / "ratio_search.csv", index=False)
    atomic_write_json(output_dir / "ratio_search_details.json", {"rows": rows})

    member_dirs: list[Path] = []
    seed_results: list[dict[str, Any]] = []
    for seed in seeds:
        sampler = MajorityUndersamplingSampler(ratio=float(selected["ratio"]))
        member_dir = output_dir / "members" / f"seed_{seed}"
        result = train_and_evaluate_single_model(
            task_data=task_data,
            model_name=model_name,
            model_config=model_config,
            output_dir=member_dir,
            seed=seed,
            project_config=project_config,
            integrity_config=project_config.get("artifact_integrity", {}),
            training_sampler=sampler,
        )
        member_dirs.append(member_dir)
        sampling_manifest = json.loads(
            (member_dir / "sampling_manifest.json").read_text(encoding="utf-8")
        )
        seed_results.append(
            {
                "seed": seed,
                "sampling": sampling_manifest,
                "result": result.to_dict(),
            }
        )

    ensemble = ensemble_member_prediction_files(
        task=task_data.task,
        member_dirs=member_dirs,
        split_names=tuple(task_data.splits.keys()),
        output_dir=output_dir / "ensemble",
    )
    payload = {
        "schema_version": "exp123_foldwise_undersampling_v8_1",
        "strategy_id": "T4_undersampling",
        "task_name": task_data.task.task_name,
        "model_name": model_name,
        "selected_ratio": float(selected["ratio"]),
        "sampling_seeds": seeds,
        "preserve_all_minority_samples": True,
        "class_weight_mode": "none",
        "preprocessor_fit_distribution": "natural_train_before_sampling",
        "validation_distribution_modified": False,
        "internal_test_distribution_modified": False,
        "external_distribution_modified": False,
        "fixed_cv_hash": cv_manifest["split_hash"],
        "shared_split_hash": task_data.shared_split_hash,
        "seed_results": seed_results,
        "ensemble": ensemble,
    }
    atomic_write_json(output_dir / "undersampling_manifest.json", payload)
    return payload
