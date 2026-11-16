from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from los_baseline.exp123_data import TaskDataBundle, TaskSplitData
from los_baseline.imbalance import (
    ClassCSmoteSampler,
    MajorityUndersamplingSampler,
    build_class_c_synthetic_pool,
)
from los_baseline.imbalance.sampled_partition import fit_predict_sampled_partition
from los_baseline.task_spec import EXP1_FLAT_3CLASS_TASK, STAGE2_LONG_STAY_TASK
from los_baseline.training_engine import train_and_evaluate_single_model


def _split(labels: np.ndarray, *, start: int = 0, name: str = "train") -> TaskSplitData:
    labels = np.asarray(labels, dtype=np.int64)
    n = len(labels)
    x = np.arange(n, dtype=float)
    frame = pd.DataFrame(
        {
            "f1": x,
            "f2": np.where(labels % 2 == 0, x * 0.1, x * 0.2 + 1.0),
            "binary": (x.astype(int) % 2).astype(float),
        }
    )
    metadata = pd.DataFrame(
        {
            "source_row_index": np.arange(start, start + n, dtype=np.int64),
            "subject_id": [f"p{start + i}" for i in range(n)],
        }
    )
    return TaskSplitData(name, frame, labels, metadata)


def _bundle_binary() -> TaskDataBundle:
    # Enough positives/groups for 3-fold group calibration.
    train_y = np.asarray([0] * 30 + [1] * 10, dtype=np.int64)
    valid_y = np.asarray([0] * 12 + [1] * 6, dtype=np.int64)
    return TaskDataBundle(
        task=STAGE2_LONG_STAY_TASK,
        feature_names=("f1", "f2", "binary"),
        splits={
            "train": _split(train_y, start=0, name="train"),
            "validation": _split(valid_y, start=1000, name="validation"),
        },
        shared_split_hash="toyhash",
    )


def test_natural_task_split_rejects_synthetic_rows():
    metadata = pd.DataFrame(
        {
            "source_row_index": [1, 2],
            "subject_id": ["a", "b"],
            "is_synthetic": [False, True],
        }
    )
    with pytest.raises(ValueError, match="synthetic rows"):
        TaskSplitData(
            "train",
            pd.DataFrame({"f": [1.0, 2.0]}),
            np.asarray([0, 1]),
            metadata,
        )


def test_smote_multiplier_doubles_minority_and_is_deterministic():
    labels = np.asarray([0] * 18 + [1] * 6 + [2] * 6, dtype=np.int64)
    split = _split(labels)
    a_raw, a_meta, a_manifest = build_class_c_synthetic_pool(
        train_split=split,
        feature_names=("f1", "f2", "binary"),
        minority_class=2,
        multiplier=2.0,
        seed=42,
        k_neighbors=3,
        patient_id_column="subject_id",
    )
    b_raw, b_meta, b_manifest = build_class_c_synthetic_pool(
        train_split=split,
        feature_names=("f1", "f2", "binary"),
        minority_class=2,
        multiplier=2.0,
        seed=42,
        k_neighbors=3,
        patient_id_column="subject_id",
    )
    assert len(a_raw) == 6
    assert a_manifest.original_minority_count == 6
    assert a_manifest.target_minority_count == 12
    pd.testing.assert_frame_equal(a_raw, b_raw)
    pd.testing.assert_frame_equal(a_meta, b_meta)
    assert a_manifest.pool_sha256 == b_manifest.pool_sha256
    assert set(a_raw["binary"].unique()).issubset({0.0, 1.0})
    natural_sources = set(split.metadata["source_row_index"])
    assert set(a_meta["parent_a_source_row_index"]).issubset(natural_sources)
    assert set(a_meta["parent_b_source_row_index"]).issubset(natural_sources)


def test_same_fixed_pool_can_be_reused_by_exp1_and_stage2():
    exp1_labels = np.asarray([0] * 18 + [1] * 6 + [2] * 6, dtype=np.int64)
    exp1 = _split(exp1_labels)
    raw, provenance, manifest = build_class_c_synthetic_pool(
        train_split=exp1,
        feature_names=("f1", "f2", "binary"),
        minority_class=2,
        multiplier=2.0,
        seed=142,
        k_neighbors=3,
        patient_id_column="subject_id",
    )
    survivor_idx = np.flatnonzero(exp1_labels != 1)
    stage2 = TaskSplitData(
        "train",
        exp1.raw_features.iloc[survivor_idx].reset_index(drop=True),
        np.where(exp1_labels[survivor_idx] == 2, 1, 0).astype(np.int64),
        exp1.metadata.iloc[survivor_idx].reset_index(drop=True),
    )
    exp1_sampler = ClassCSmoteSampler(
        minority_class=2,
        multiplier=2.0,
        k_neighbors=3,
        fixed_pool_raw=raw,
        fixed_pool_provenance=provenance,
        fixed_pool_seed=142,
    )
    stage2_sampler = ClassCSmoteSampler(
        minority_class=1,
        multiplier=2.0,
        k_neighbors=3,
        fixed_pool_raw=raw,
        fixed_pool_provenance=provenance,
        fixed_pool_seed=142,
    )
    exp1_sampled = exp1_sampler.sample(
        exp1, feature_names=("f1", "f2", "binary"), seed=142, purpose="final_train"
    )
    stage2_sampled = stage2_sampler.sample(
        stage2, feature_names=("f1", "f2", "binary"), seed=142, purpose="final_train"
    )
    assert exp1_sampled.synthetic_count == stage2_sampled.synthetic_count == 6
    pd.testing.assert_frame_equal(
        exp1_sampled.raw_features.tail(6).reset_index(drop=True),
        stage2_sampled.raw_features.tail(6).reset_index(drop=True),
    )
    assert set(stage2_sampled.labels[-6:].tolist()) == {1}
    assert exp1_sampled.manifest["pool"]["source"] == "fixed_shared_pool"
    assert stage2_sampled.manifest["pool"]["source"] == "fixed_shared_pool"
    assert manifest.synthetic_count == 6


def test_sampled_partition_fits_preprocessor_on_natural_rows_before_undersampling():
    split = _split(np.asarray([0] * 20 + [1] * 5, dtype=np.int64))
    sampler = MajorityUndersamplingSampler(ratio=1.0)
    _, metrics, diagnostics, _ = fit_predict_sampled_partition(
        model_name="decision_tree",
        model_config={
            "parameters": {"max_depth": 2},
            "training_policy": {"class_weight_mode": "none"},
        },
        task=STAGE2_LONG_STAY_TASK,
        feature_names=("f1", "f2", "binary"),
        natural_train=split,
        evaluation_features=split.raw_features.copy(),
        evaluation_labels=split.labels.copy(),
        sampler=sampler,
        seed=42,
    )
    assert diagnostics["preprocessor_fit_count"] == 25
    assert diagnostics["model_fit_count"] == 10
    assert diagnostics["evaluation_count"] == 25
    assert diagnostics["evaluation_distribution_modified"] is False
    assert "auroc" in metrics


def test_sampled_final_training_uses_natural_preprocessor_and_natural_oof_calibration(tmp_path: Path):
    bundle = _bundle_binary()
    sampler = MajorityUndersamplingSampler(ratio=1.0)
    project_config = {
        "calibration": {
            "enabled": True,
            "method": "sigmoid",
            "cv_folds": 3,
            "seed": 4242,
            "group_column": "subject_id",
        },
        "data": {"patient_id_column": "subject_id"},
        "explainability": {},
    }
    result = train_and_evaluate_single_model(
        task_data=bundle,
        model_name="decision_tree",
        model_config={
            "parameters": {"max_depth": 3, "min_samples_split": 2},
            "training_policy": {"class_weight_mode": "none"},
        },
        output_dir=tmp_path / "model",
        seed=42,
        project_config=project_config,
        training_sampler=sampler,
    )
    metadata = json.loads((tmp_path / "model" / "artifact" / "metadata.json").read_text())
    calibration = metadata["calibration"]
    assert calibration["implementation"] == "exp123_v8_sampling_aware_oof_sigmoid"
    assert calibration["natural_oof_rows"] == 40
    assert calibration["synthetic_rows_in_calibration_targets"] == 0
    assert metadata["natural_train_sample_count"] == 40
    assert metadata["model_fit_sample_count"] == 20
    preprocessor = joblib.load(tmp_path / "model" / "artifact" / "preprocessor.joblib")
    expected_medians = np.nanmedian(bundle.splits["train"].raw_features.to_numpy(), axis=0)
    np.testing.assert_allclose(preprocessor.named_steps["imputer"].statistics_, expected_medians)
    assert "validation" in result.split_metrics


def test_smote_sampling_aware_calibration_has_no_synthetic_targets(tmp_path: Path):
    bundle = _bundle_binary()
    sampler = ClassCSmoteSampler(
        minority_class=1,
        multiplier=2.0,
        k_neighbors=3,
        patient_id_column="subject_id",
    )
    project_config = {
        "calibration": {
            "enabled": True,
            "method": "sigmoid",
            "cv_folds": 3,
            "seed": 4242,
            "group_column": "subject_id",
        },
        "data": {"patient_id_column": "subject_id"},
        "explainability": {},
    }
    train_and_evaluate_single_model(
        task_data=bundle,
        model_name="decision_tree",
        model_config={
            "parameters": {"max_depth": 3, "min_samples_split": 2},
            "training_policy": {"class_weight_mode": "none"},
        },
        output_dir=tmp_path / "smote_model",
        seed=42,
        project_config=project_config,
        training_sampler=sampler,
    )
    metadata = json.loads((tmp_path / "smote_model" / "artifact" / "metadata.json").read_text())
    calibration = metadata["calibration"]
    assert calibration["synthetic_rows_in_calibration_targets"] == 0
    assert all(row["synthetic_fit_count"] > 0 for row in calibration["fold_sampling"])
    assert metadata["sampling"]["synthetic_count"] == 10
    assert metadata["model_fit_sample_count"] == 50


def test_sampled_svm_probability_false_is_calibrated_and_reloadable(tmp_path: Path):
    bundle = _bundle_binary()
    sampler = MajorityUndersamplingSampler(ratio=1.0)
    config = {
        "calibration": {
            "enabled": True,
            "method": "sigmoid",
            "cv_folds": 3,
            "seed": 4242,
            "group_column": "subject_id",
        },
        "data": {"patient_id_column": "subject_id"},
        "explainability": {},
    }
    result = train_and_evaluate_single_model(
        task_data=bundle,
        model_name="svm",
        model_config={
            "parameters": {
                "kernel": "linear",
                "C": 0.1,
                "probability": False,
                "cache_size": 128,
            },
            "training_policy": {"class_weight_mode": "none"},
        },
        output_dir=tmp_path / "svm",
        seed=42,
        project_config=config,
        training_sampler=sampler,
    )
    meta = json.loads((tmp_path / "svm" / "artifact" / "metadata.json").read_text())
    assert meta["model_serialization"] == "joblib"
    assert meta["calibration"]["implementation"] == "exp123_v8_sampling_aware_oof_sigmoid"
    assert result.prediction_integrity["validation"].matches


def test_multiclass_smote_calibration_returns_three_class_probabilities(tmp_path: Path):
    train_y = np.asarray([0] * 30 + [1] * 12 + [2] * 9, dtype=np.int64)
    valid_y = np.asarray([0] * 12 + [1] * 6 + [2] * 6, dtype=np.int64)
    bundle = TaskDataBundle(
        task=EXP1_FLAT_3CLASS_TASK,
        feature_names=("f1", "f2", "binary"),
        splits={
            "train": _split(train_y, start=0, name="train"),
            "validation": _split(valid_y, start=2000, name="validation"),
        },
        shared_split_hash="multihash",
    )
    config = {
        "calibration": {
            "enabled": True,
            "method": "sigmoid",
            "cv_folds": 3,
            "seed": 4242,
            "group_column": "subject_id",
        },
        "data": {"patient_id_column": "subject_id"},
        "explainability": {},
    }
    train_and_evaluate_single_model(
        task_data=bundle,
        model_name="decision_tree",
        model_config={
            "parameters": {"max_depth": 3},
            "training_policy": {"class_weight_mode": "none"},
        },
        output_dir=tmp_path / "multi",
        seed=42,
        project_config=config,
        training_sampler=ClassCSmoteSampler(
            minority_class=2,
            multiplier=2.0,
            k_neighbors=3,
            patient_id_column="subject_id",
        ),
    )
    pred = pd.read_csv(tmp_path / "multi" / "evaluation" / "validation" / "predictions.csv.gz")
    probs = pred[["prob_class_0", "prob_class_1", "prob_class_2"]].to_numpy(float)
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-8)
    assert np.isfinite(probs).all()


def test_persisted_smote_pool_hashes_roundtrip(tmp_path: Path):
    from los_baseline.imbalance import load_synthetic_pool, save_synthetic_pool
    labels = np.asarray([0] * 18 + [1] * 6 + [2] * 6, dtype=np.int64)
    split = _split(labels)
    raw, provenance, manifest = build_class_c_synthetic_pool(
        train_split=split,
        feature_names=("f1", "f2", "binary"),
        minority_class=2,
        multiplier=2.0,
        seed=242,
        k_neighbors=3,
        patient_id_column="subject_id",
    )
    save_synthetic_pool(tmp_path, raw, provenance, manifest)
    loaded_raw, loaded_provenance, loaded_manifest = load_synthetic_pool(tmp_path)
    assert len(loaded_raw) == len(raw)
    assert len(loaded_provenance) == len(provenance)
    assert loaded_manifest["pool_sha256"] == manifest.pool_sha256
