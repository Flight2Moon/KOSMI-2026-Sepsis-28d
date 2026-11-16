from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import los_baseline.parallel_jobs as jobs
from los_baseline.exp123_data import TaskDataBundle, TaskSplitData
from los_baseline.task_spec import EXP1_FLAT_3CLASS_TASK


def _split(name: str, labels: np.ndarray, offset: int) -> TaskSplitData:
    rng = np.random.default_rng(42 + offset)
    labels = np.asarray(labels, dtype=np.int64)
    features = pd.DataFrame({
        "f1": labels + rng.normal(0, 0.2, len(labels)),
        "f2": (labels == 1).astype(float) + rng.normal(0, 0.2, len(labels)),
        "f3": rng.normal(size=len(labels)),
    })
    metadata = pd.DataFrame({
        "source_row_index": np.arange(offset, offset + len(labels)),
        "split": name,
        "subject_id": [f"{name}_{i}" for i in range(len(labels))],
    })
    return TaskSplitData(name, features, labels, metadata)


def _bundle() -> TaskDataBundle:
    return TaskDataBundle(
        task=EXP1_FLAT_3CLASS_TASK,
        feature_names=("f1", "f2", "f3"),
        splits={
            "train": _split("train", np.tile([0, 1, 2], 20), 0),
            "validation": _split("validation", np.tile([0, 1, 2], 8), 1000),
        },
        shared_split_hash="parallel_job_smoke_hash",
    )


def _config() -> dict:
    return {
        "data": {"patient_id_column": "subject_id", "feature_columns": ["f1", "f2", "f3"]},
        "split": {"seed": 42},
        "experiments": {"exp1_flat_3class": {"models": ["decision_tree"], "representative_metric": "macro_auroc"}},
        "models": {"decision_tree": {"parameters": {"max_depth": 3, "min_samples_split": 2}}},
        "hardware": {"accelerator": {"backend": "cpu"}, "cpu": {"threads_per_worker": 1}, "cuda": {"mode": "disabled"}},
        "optimization": {
            "seed": 42, "cv_folds": 2, "top_k_for_validation": 1,
            "search_stages": {"smoke": {"trials": 1, "cv_folds": 2}},
            "selection": {"stability_penalty": 0.5, "practical_tie_margin": 0.005, "auroc_margin_guard": 0.02, "prefer_simpler_model_on_tie": True},
            "runtime_limits": {"trial_timeout_seconds": 60, "study_timeout_seconds": 300},
            "resume": True,
        },
        "optimization_objectives": {"exp1_flat_3class": {"primary": "macro_auroc", "direction": "maximize", "secondary": "multiclass_brier", "secondary_direction": "minimize"}},
        "evaluation": {"multiclass_bootstrap_repeats": 5, "bootstrap_confidence_level": 0.95, "calibration_bins": 5, "save_plots": False, "default_decision_threshold": 0.5},
        "artifact_integrity": {"probability_tolerance": {"cpu": {"atol": 1e-7, "rtol": 0}}, "require_decision_equality": True, "maximum_decision_mismatches": 0},
    }


def test_model_level_optimization_training_and_finalize(tmp_path: Path, monkeypatch) -> None:
    bundle = _bundle()
    config = _config()
    run_dir = tmp_path / "run"
    monkeypatch.setattr(jobs, "_task_data", lambda config, run_dir, task_key: bundle)
    monkeypatch.setattr(jobs, "load_exp1_discovery_task_data", lambda config, run_dir: bundle)

    optimization = jobs.run_model_optimization_job(
        task_key="exp1", model_name="decision_tree", config=config,
        run_id="smoke", run_dir=run_dir, search_stages=("smoke",),
    )
    assert optimization["model_name"] == "decision_tree"
    jobs.finalize_optimization_task(task_key="exp1", config=config, run_id="smoke", run_dir=run_dir)

    training = jobs.run_model_training_job(
        task_key="exp1", model_name="decision_tree", config=config,
        run_id="smoke", run_dir=run_dir, bootstrap_repeats=5,
    )
    assert training["model_result"]["model_name"] == "decision_tree"
    finalized = jobs.finalize_training_task(task_key="exp1", config=config, run_id="smoke", run_dir=run_dir)
    assert finalized["representative_model"] == "decision_tree"
    assert (run_dir / "EXP1_flat_3class" / "exp1_result.json").is_file()
