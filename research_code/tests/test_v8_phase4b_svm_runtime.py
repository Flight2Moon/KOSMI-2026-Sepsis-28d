from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import optuna
import pandas as pd
import pytest

from los_baseline.config import validate_config
from los_baseline.experiment_scheduler import ResourceAwareScheduler, ScheduledJob
from los_baseline.hpo.optuna_engine import (
    HPO_PROTOCOL_VERSION,
    SVM_PHASE4B_HPO_PROTOCOL_VERSION,
    _evaluate_config,
    resolve_hpo_protocol_version,
)
from los_baseline.hpo.search_space import suggest_model_config
from los_baseline.models import build_task_sklearn_model
from los_baseline.optimization import _fit_predict_partition
from los_baseline.resource_manager import ResourceRequest, resource_config
from los_baseline.svm_runtime import svm_backend_label, svm_runtime_settings
from los_baseline.task_spec import EXP1_FLAT_3CLASS_TASK, STAGE1_MORTALITY_TASK
from los_baseline.training_engine import train_and_evaluate_single_model
from los_baseline.exp123_data import TaskDataBundle, TaskSplitData
from los_baseline.artifacts import load_task_artifact
from los_baseline.optimization_objectives import load_objective_definition


class FixedTrial:
    def __init__(self, backend: str):
        self.backend = backend

    def suggest_categorical(self, name, choices):
        if name == "svm_backend":
            assert self.backend in choices
            return self.backend
        return choices[0]

    def suggest_float(self, name, low, high, log=False):
        return float(np.sqrt(low * high) if log else (low + high) / 2.0)

    def suggest_int(self, name, low, high, step=1, log=False):
        if log:
            return int(max(low, min(high, round(np.sqrt(low * high)))))
        return int(low)


def _runtime_config(max_fold_fit_seconds: float = 7200.0) -> dict:
    return {
        "svm_runtime_optimization": {
            "enabled": True,
            "search_backends": ["exact_rbf", "nystroem_linear"],
            "max_fold_fit_seconds": max_fold_fit_seconds,
            "exact_rbf": {
                "c_min": 1e-2,
                "c_max": 1e2,
                "gamma_min": 1e-4,
                "gamma_max": 1.0,
                "max_iter": 2_000_000,
                "tol": 1e-3,
            },
            "nystroem_linear": {
                "c_min": 1e-2,
                "c_max": 1e2,
                "gamma_min": 1e-4,
                "gamma_max": 1.0,
                "n_components_min": 8,
                "n_components_max": 64,
                "max_iter": 20_000,
                "tol": 1e-4,
            },
        }
    }


def _split(name: str, labels: np.ndarray, seed: int, offset: int) -> TaskSplitData:
    rng = np.random.default_rng(seed)
    n = len(labels)
    features = pd.DataFrame({
        "x1": rng.normal(size=n),
        "x2": rng.normal(size=n),
        "x3": rng.normal(size=n),
    })
    metadata = pd.DataFrame({
        "source_row_index": np.arange(offset, offset + n, dtype=np.int64),
        "subject_id": [f"p_{offset+i}" for i in range(n)],
    })
    return TaskSplitData(name, features, np.asarray(labels, dtype=np.int64), metadata)


def _binary_bundle() -> TaskDataBundle:
    train_y = np.array([0, 1] * 30, dtype=np.int64)
    valid_y = np.array([0, 1] * 10, dtype=np.int64)
    return TaskDataBundle(
        task=STAGE1_MORTALITY_TASK,
        feature_names=("x1", "x2", "x3"),
        splits={
            "train": _split("train", train_y, 11, 0),
            "validation": _split("validation", valid_y, 12, 1000),
        },
        shared_split_hash="phase4b_binary",
    )


def _multiclass_bundle() -> TaskDataBundle:
    train_y = np.tile(np.array([0, 1, 2], dtype=np.int64), 24)
    valid_y = np.tile(np.array([0, 1, 2], dtype=np.int64), 8)
    return TaskDataBundle(
        task=EXP1_FLAT_3CLASS_TASK,
        feature_names=("x1", "x2", "x3"),
        splits={
            "train": _split("train", train_y, 21, 0),
            "validation": _split("validation", valid_y, 22, 1000),
        },
        shared_split_hash="phase4b_multiclass",
    )


def test_phase4b_exact_rbf_search_space_is_narrowed_and_bounded() -> None:
    base = {"parameters": {"probability": True, "cache_size": 2048}}
    config = _runtime_config()
    result = suggest_model_config(FixedTrial("exact_rbf"), "svm", base, project_config=config)
    p = result["parameters"]
    assert p["svm_backend"] == "exact_rbf"
    assert p["kernel"] == "rbf"
    assert 1e-2 <= p["C"] <= 1e2
    assert 1e-4 <= p["gamma"] <= 1.0
    assert p["max_iter"] == 2_000_000
    assert p["probability"] is False
    assert "nystroem_components" not in p


def test_phase4b_nystroem_search_space_has_no_svc_only_parameters() -> None:
    base = {"parameters": {"kernel": "rbf", "gamma": "scale", "cache_size": 2048}}
    result = suggest_model_config(
        FixedTrial("nystroem_linear"), "svm", base, project_config=_runtime_config()
    )
    p = result["parameters"]
    assert p["svm_backend"] == "nystroem_linear"
    assert 8 <= p["nystroem_components"] <= 64
    assert 1e-4 <= p["nystroem_gamma"] <= 1.0
    assert p["linear_svc_max_iter"] == 20_000
    assert p["probability"] is False
    for key in ("kernel", "gamma", "cache_size", "max_iter"):
        assert key not in p


def test_nystroem_binary_hpo_uses_decision_function() -> None:
    bundle = _binary_bundle()
    train = bundle.splits["train"]
    valid = bundle.splits["validation"]
    _, metrics, diagnostics, effective = _fit_predict_partition(
        model_name="svm",
        model_config={
            "parameters": {
                "svm_backend": "nystroem_linear",
                "C": 1.0,
                "nystroem_gamma": 0.1,
                "nystroem_components": 16,
                "linear_svc_max_iter": 5000,
                "probability": False,
            },
            "training_policy": {"class_weight_mode": "none"},
        },
        task=bundle.task,
        feature_names=bundle.feature_names,
        train_features=train.raw_features,
        train_labels=train.labels,
        evaluation_features=valid.raw_features,
        evaluation_labels=valid.labels,
        seed=42,
        train_groups=train.metadata["subject_id"].to_numpy(),
    )
    assert diagnostics["prediction_kind"] == "decision_score"
    assert diagnostics["svm_backend"] == "nystroem_linear"
    assert diagnostics["fit_seconds"] >= 0.0
    assert "auprc" in metrics and "auroc" in metrics
    assert "brier_score" not in metrics
    assert effective["parameters"]["svm_backend"] == "nystroem_linear"


def test_nystroem_multiclass_decision_shape_and_metrics() -> None:
    bundle = _multiclass_bundle()
    train = bundle.splits["train"]
    valid = bundle.splits["validation"]
    predictions, metrics, diagnostics, _ = _fit_predict_partition(
        model_name="svm",
        model_config={
            "parameters": {
                "svm_backend": "nystroem_linear",
                "C": 0.7,
                "nystroem_gamma": 0.1,
                "nystroem_components": 18,
                "linear_svc_max_iter": 5000,
                "probability": False,
            },
            "training_policy": {"class_weight_mode": "none"},
        },
        task=bundle.task,
        feature_names=bundle.feature_names,
        train_features=train.raw_features,
        train_labels=train.labels,
        evaluation_features=valid.raw_features,
        evaluation_labels=valid.labels,
        seed=43,
        train_groups=train.metadata["subject_id"].to_numpy(),
    )
    assert predictions.shape == (len(valid.labels), 3)
    assert diagnostics["prediction_kind"] == "decision_score"
    assert "macro_auprc" in metrics and "macro_auroc" in metrics


def test_nystroem_final_model_is_oof_calibrated_and_reloadable(tmp_path: Path) -> None:
    bundle = _binary_bundle()
    output = tmp_path / "nystroem_final"
    result = train_and_evaluate_single_model(
        task_data=bundle,
        model_name="svm",
        model_config={
            "parameters": {
                "svm_backend": "nystroem_linear",
                "C": 1.0,
                "nystroem_gamma": 0.1,
                "nystroem_components": 16,
                "linear_svc_max_iter": 5000,
                "probability": False,
            },
            "training_policy": {"class_weight_mode": "none"},
        },
        output_dir=output,
        seed=42,
        project_config={
            "data": {"patient_id_column": "subject_id"},
            "calibration": {
                "enabled": True,
                "method": "sigmoid",
                "cv_folds": 2,
                "seed": 4242,
                "group_column": "subject_id",
            },
            "svm_runtime_optimization": _runtime_config()["svm_runtime_optimization"],
            "explainability": {"enabled": False},
        },
        integrity_config={"require_decision_equality": True},
    )
    assert result.split_metrics["validation"]["auprc"] >= 0.0
    metadata = json.loads((output / "artifact" / "metadata.json").read_text())
    assert metadata["calibration"]["enabled"] is True
    assert metadata["model_serialization"] == "joblib"
    assert metadata["svm_runtime"]["backend"] == "nystroem_linear"
    loaded = load_task_artifact(output / "artifact")
    probabilities = loaded.predict_proba(bundle.splits["validation"].raw_features)
    assert probabilities.shape == (len(bundle.splits["validation"].labels),)
    assert np.all((probabilities >= 0.0) & (probabilities <= 1.0))


def test_nystroem_seed_is_deterministic() -> None:
    cfg = {
        "parameters": {
            "svm_backend": "nystroem_linear",
            "C": 1.0,
            "nystroem_gamma": 0.1,
            "nystroem_components": 12,
            "linear_svc_max_iter": 5000,
            "probability": False,
        }
    }
    rng = np.random.default_rng(44)
    X = rng.normal(size=(60, 4))
    y = np.array([0, 1] * 30)
    a = build_task_sklearn_model("svm", cfg, 42, "binary", 2, {0: 1.0, 1: 1.0})
    b = build_task_sklearn_model("svm", cfg, 42, "binary", 2, {0: 1.0, 1: 1.0})
    a.fit(X, y)
    b.fit(X, y)
    assert np.allclose(a.decision_function(X), b.decision_function(X))


def test_runtime_guard_prunes_after_an_over_budget_fold(monkeypatch) -> None:
    bundle = _binary_bundle()
    train = bundle.splits["train"]
    fixed = [SimpleNamespace(
        train_indices=np.arange(0, 40, dtype=np.int64),
        validation_indices=np.arange(40, 60, dtype=np.int64),
        repeat=1,
        fold=1,
    )]

    def fake_partition(**kwargs):
        return (
            np.linspace(-1, 1, len(kwargs["evaluation_labels"])),
            {"auprc": 0.5, "auroc": 0.6},
            {
                "train_metrics": {"auprc": 0.55, "auroc": 0.65},
                "prediction_kind": "decision_score",
                "fit_seconds": 3.0,
                "svm_backend": "exact_rbf",
            },
            kwargs["model_config"],
        )

    monkeypatch.setattr("los_baseline.hpo.optuna_engine._fit_predict_partition", fake_partition)
    objective = load_objective_definition(
        "stage1_mortality",
        {
            "stage1_mortality": {
                "primary": "auprc",
                "direction": "maximize",
                "secondary": "brier_score",
                "secondary_direction": "minimize",
            }
        },
    )

    class Trial:
        def __init__(self):
            self.attrs = {}
        def set_user_attr(self, key, value): self.attrs[key] = value
        def report(self, value, step): pass
        def should_prune(self): return False

    trial = Trial()
    with pytest.raises(optuna.TrialPruned):
        _evaluate_config(
            task_data=bundle,
            model_name="svm",
            model_config={
                "parameters": {"svm_backend": "exact_rbf", "probability": False},
                "training_policy": {"class_weight_mode": "none"},
            },
            folds=fixed,
            objective=objective,
            patient_id_column="subject_id",
            stability_penalty=0.5,
            overfit_penalty=0.5,
            seed=42,
            trial=trial,
            project_config=_runtime_config(max_fold_fit_seconds=1.0),
        )
    assert trial.attrs["runtime_guard_triggered"] is True
    assert trial.attrs["runtime_guard_fit_seconds"] == 3.0


def test_phase4b_hpo_protocol_version_is_svm_specific() -> None:
    enabled = _runtime_config()
    assert resolve_hpo_protocol_version("svm", enabled) == SVM_PHASE4B_HPO_PROTOCOL_VERSION
    assert resolve_hpo_protocol_version("xgboost", enabled) == HPO_PROTOCOL_VERSION
    assert resolve_hpo_protocol_version("svm", {}) == HPO_PROTOCOL_VERSION


def test_scheduler_limits_concurrent_svm_jobs(tmp_path: Path) -> None:
    config = {
        "resources": {
            "accelerator": {"backend": "cpu"},
            "scheduler": {
                "enabled": True,
                "max_concurrent_jobs": 8,
                "max_concurrent_by_model": {"svm": 2},
            },
            "cpu": {"total_thread_limit": 16, "reserve_threads": 0, "default_threads_per_job": 1},
            "memory": {"total_limit_gb": 64, "max_fraction": 1.0, "reserve_gb": 0, "minimum_free_gb_to_launch": 0},
            "cuda": {"gpu_ids": [0]},
            "mps": {"enabled": False},
            "resource_failure_policy": {"auto_retry": False},
        }
    }
    jobs = [ScheduledJob(
        name=f"svm_{i}",
        command=("true",),
        model_name="svm",
        request=ResourceRequest(cpu_threads=1, ram_gb=0.1, accelerator="cpu"),
    ) for i in range(3)]
    scheduler = ResourceAwareScheduler(
        jobs=jobs,
        config=config,
        project_root=tmp_path,
        run_dir=tmp_path / "run",
        run_id="phase4b",
    )
    fake_running = SimpleNamespace(job=jobs[0])
    scheduler.running = {"svm_0": fake_running, "svm_1": SimpleNamespace(job=jobs[1])}
    ok, _, reason = scheduler._can_launch(jobs[2], {"available_ram_gb": 64.0})
    assert ok is False
    assert reason == "model_concurrency:svm"
    assert resource_config(config)["scheduler"]["max_concurrent_by_model"]["svm"] == 2


def test_config_rejects_unknown_phase4b_backend() -> None:
    path = Path(__file__).resolve().parents[1] / "configs" / "experiment_cuda_v8_los28.json"
    config = json.loads(path.read_text())
    config["svm_runtime_optimization"]["search_backends"] = ["made_up_backend"]
    with pytest.raises(ValueError, match="Unsupported SVM runtime backends"):
        validate_config(config)


def test_phase4b_main_config_resolves_runtime_policy() -> None:
    path = Path(__file__).resolve().parents[1] / "configs" / "experiment_cuda_v8_los28.json"
    config = json.loads(path.read_text())
    # load_config/validate_config is exercised by the CLI; the raw contract is
    # asserted here to keep this unit test independent of local dataset paths.
    runtime = svm_runtime_settings(config)
    assert runtime["enabled"] is True
    assert runtime["search_backends"] == ["exact_rbf", "nystroem_linear"]
    assert runtime["exact_rbf"]["c_min"] == 1e-2
    assert runtime["exact_rbf"]["c_max"] == 1e2
    assert config["resources"]["scheduler"]["max_concurrent_by_model"]["svm"] == 2


def test_nystroem_sampled_track_calibration_is_reloadable(tmp_path: Path) -> None:
    from los_baseline.imbalance import MajorityUndersamplingSampler

    bundle = _binary_bundle()
    output = tmp_path / "nystroem_sampled"
    sampler = MajorityUndersamplingSampler(ratio=1.0)
    result = train_and_evaluate_single_model(
        task_data=bundle,
        model_name="svm",
        model_config={
            "parameters": {
                "svm_backend": "nystroem_linear",
                "C": 0.8,
                "nystroem_gamma": 0.1,
                "nystroem_components": 16,
                "linear_svc_max_iter": 5000,
                "probability": False,
            },
            "training_policy": {"class_weight_mode": "none"},
        },
        output_dir=output,
        seed=42,
        project_config={
            "data": {"patient_id_column": "subject_id"},
            "calibration": {
                "enabled": True,
                "method": "sigmoid",
                "cv_folds": 2,
                "seed": 4242,
                "group_column": "subject_id",
            },
            "svm_runtime_optimization": _runtime_config()["svm_runtime_optimization"],
            "explainability": {"enabled": False},
        },
        training_sampler=sampler,
        integrity_config={"require_decision_equality": True},
    )
    assert result.prediction_integrity["validation"].matches
    metadata = json.loads((output / "artifact" / "metadata.json").read_text())
    assert metadata["calibration"]["implementation"] == "exp123_v8_sampling_aware_oof_sigmoid"
    assert metadata["svm_runtime"]["backend"] == "nystroem_linear"
    loaded = load_task_artifact(output / "artifact")
    probabilities = loaded.predict_proba(bundle.splits["validation"].raw_features)
    assert probabilities.shape == (len(bundle.splits["validation"].labels),)


def test_v8_baseline_svm_disables_hidden_svc_probability_calibration() -> None:
    path = Path(__file__).resolve().parents[1] / "configs" / "experiment_cuda_v8_los28.json"
    config = json.loads(path.read_text())
    parameters = config["models"]["svm"]["parameters"]
    assert parameters["svm_backend"] == "exact_rbf"
    assert parameters["probability"] is False
    assert parameters["max_iter"] == 2_000_000
    assert config["calibration"]["enabled"] is True
