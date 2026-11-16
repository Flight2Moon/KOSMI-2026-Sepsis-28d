from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from los_baseline.exp123_data import TaskDataBundle, TaskSplitData
from los_baseline.feature_sets import (
    dataset_header_readiness,
    inject_feature_set_into_config,
    resolve_feature_set,
)
from los_baseline.task_spec import STAGE2_LONG_STAY_TASK
from los_baseline.training_engine import train_and_evaluate_single_model

REGISTRY = PROJECT_ROOT / "configs" / "feature_sets.json"


def _split(name: str, n: int, offset: int) -> TaskSplitData:
    labels = np.asarray(([0, 1] * ((n + 1) // 2))[:n], dtype=np.int64)
    raw = pd.DataFrame({"f1": np.linspace(0.0, 1.0, n), "f2": np.linspace(1.0, 2.0, n)})
    metadata = pd.DataFrame({
        "source_row_index": np.arange(offset, offset + n),
        "subject_id": np.arange(10_000 + offset, 10_000 + offset + n),
    })
    return TaskSplitData(name, raw, labels, metadata)


def test_target_feature_sets_have_expected_counts_and_exact_aps_ablation():
    full = resolve_feature_set(REGISTRY, "FS_FULL")
    noaps = resolve_feature_set(REGISTRY, "FS_NOAPS")
    aps = resolve_feature_set(REGISTRY, "FS_APSONLY")
    assert len(full.columns) == 57
    assert len(noaps.columns) == 56
    assert aps.columns == ("apsiii",)
    assert "apsiii" in full.columns
    assert "apsiii" not in noaps.columns
    assert list(noaps.columns) == [v for v in full.columns if v != "apsiii"]
    assert full.sha256 != noaps.sha256


def test_legacy_feature_sets_preserve_phase3_columns():
    phase3 = json.loads((PROJECT_ROOT / "configs" / "experiment_cuda_v8_los28.json").read_text())
    legacy = resolve_feature_set(REGISTRY, "FS_LEGACY19")
    assert list(legacy.columns) == phase3["data"]["feature_columns"]
    assert len(legacy.columns) == 19


def test_feature_set_hash_is_stable():
    a = resolve_feature_set(REGISTRY, "FS_FULL")
    b = resolve_feature_set(REGISTRY, "FS_FULL")
    assert a.sha256 == b.sha256
    assert len(a.sha256) == 64


def test_inject_feature_set_replaces_only_feature_axis():
    base = json.loads((PROJECT_ROOT / "configs" / "experiment_cuda_v8_los28.json").read_text())
    resolved = resolve_feature_set(REGISTRY, "FS_NOAPS")
    derived = inject_feature_set_into_config(base, resolved)
    assert derived["data"]["feature_columns"] == list(resolved.columns)
    assert derived["data"]["feature_set_id"] == "FS_NOAPS"
    assert derived["data"]["feature_set_sha256"] == resolved.sha256
    assert derived["split"] == base["split"]
    assert derived["models"] == base["models"]
    assert derived["outcomes"] == base["outcomes"]


def test_builder_generates_los28_t1_ablation_config(tmp_path: Path):
    output = tmp_path / "ablation.json"
    subprocess.run([
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "workflows" / "build_exp123_ablation_config.py"),
        "--base-config", str(PROJECT_ROOT / "configs" / "experiment_cuda_v8_los28.json"),
        "--feature-set-registry", str(REGISTRY),
        "--feature-set-id", "FS_APSONLY",
        "--output", str(output),
    ], check=True, cwd=PROJECT_ROOT)
    payload = json.loads(output.read_text())
    assert payload["data"]["feature_columns"] == ["apsiii"]
    assert payload["ablation"]["fixed_outcome_horizon_days"] == 28
    assert payload["ablation"]["fixed_strategy"] == "T1_baseline"
    assert payload["ablation"]["varied_axis"] == "feature_set"


def test_builder_rejects_los14_ablation(tmp_path: Path):
    output = tmp_path / "bad.json"
    result = subprocess.run([
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "workflows" / "build_exp123_ablation_config.py"),
        "--base-config", str(PROJECT_ROOT / "legacy_reference" / "v8_multi_horizon" / "experiment_cuda_v8_los14.json"),
        "--feature-set-registry", str(REGISTRY),
        "--feature-set-id", "FS_FULL",
        "--output", str(output),
    ], cwd=PROJECT_ROOT, capture_output=True, text=True)
    assert result.returncode != 0
    assert "prespecified on LOS28" in (result.stderr + result.stdout)


def test_dataset_header_readiness_reports_missing_features(tmp_path: Path):
    path = tmp_path / "x.csv"
    pd.DataFrame(columns=["age", "apsiii"]).to_csv(path, index=False)
    audit = dataset_header_readiness(dataset_path=path, required_columns=["age", "apsiii", "cci"])
    assert audit["status"] == "missing_columns"
    assert audit["missing_columns"] == ["cci"]


def test_artifact_metadata_records_feature_set(tmp_path: Path):
    bundle = TaskDataBundle(
        task=STAGE2_LONG_STAY_TASK,
        feature_names=("f1", "f2"),
        splits={
            "train": _split("train", 12, 0),
            "validation": _split("validation", 8, 100),
        },
        shared_split_hash="same-split",
    )
    project_config = {
        "data": {"patient_id_column": "subject_id", "feature_set_id": "FS_TEST", "feature_set_sha256": "a" * 64},
        "feature_set": {
            "set_id": "FS_TEST",
            "sha256": "a" * 64,
            "feature_count": 2,
            "registry_schema_version": "exp123_feature_sets_v1",
        },
        "calibration": {"enabled": False},
        "explainability": {},
    }
    train_and_evaluate_single_model(
        task_data=bundle,
        model_name="decision_tree",
        model_config={
            "parameters": {"max_depth": 2, "min_samples_split": 2},
            "training_policy": {"class_weight_mode": "none"},
        },
        output_dir=tmp_path / "model",
        project_config=project_config,
    )
    metadata = json.loads((tmp_path / "model" / "artifact" / "metadata.json").read_text())
    assert metadata["feature_set"]["set_id"] == "FS_TEST"
    assert metadata["feature_set"]["sha256"] == "a" * 64
    assert metadata["feature_set"]["feature_count"] == 2


def test_run_exp123_preflight_skips_cuda_check_for_cpu_backend():
    source = (PROJECT_ROOT / "run_exp123.py").read_text(encoding="utf-8")
    assert 'if backend == "cuda":' in source
    assert '"skipped_cuda_validation": True' in source


def test_exp1_ablation_statistics_are_paired_and_validation_only(tmp_path: Path):
    from los_baseline.ablation_stats_v8 import AblationComparisonSettings, compare_exp1_feature_sets

    n = 60
    y = np.asarray([0, 1, 2] * 20, dtype=np.int64)
    full_prob = np.full((n, 3), 0.1, dtype=float)
    ab_prob = np.full((n, 3), 0.15, dtype=float)
    for i, cls in enumerate(y):
        full_prob[i, cls] = 0.8
        ab_prob[i, cls] = 0.7
        remaining = [c for c in range(3) if c != cls]
        full_prob[i, remaining] = 0.1
        ab_prob[i, remaining] = 0.15
    for run_id, prob in (("full", full_prob), ("noaps", ab_prob)):
        path = tmp_path / run_id / "EXP1_flat_3class" / "decision_tree" / "evaluation" / "validation"
        path.mkdir(parents=True)
        frame = pd.DataFrame({
            "source_row_index": np.arange(n),
            "subject_id": np.arange(1000, 1000 + n),
            "y_true": y,
            "predicted_class": prob.argmax(axis=1),
            "prob_class_0": prob[:, 0],
            "prob_class_1": prob[:, 1],
            "prob_class_2": prob[:, 2],
        })
        frame.to_csv(path / "predictions.csv.gz", index=False, compression="gzip")

    out = tmp_path / "stats"
    manifest = compare_exp1_feature_sets(
        runs_dir=tmp_path,
        full_run_id="full",
        ablation_run_id="noaps",
        full_feature_set_id="FS_FULL",
        ablation_feature_set_id="FS_NOAPS",
        model_names=["decision_tree"],
        settings=AblationComparisonSettings(
            bootstrap_repeats=20,
            minimum_successful_bootstrap_repeats=10,
            seed=42,
        ),
        output_dir=out,
    )
    bootstrap = pd.read_csv(out / "exp1_paired_bootstrap_vs_full.csv")
    delong = pd.read_csv(out / "exp1_class_delong_vs_full.csv")
    assert set(bootstrap["metric"]) == {"macro_auprc", "macro_auroc", "multiclass_brier", "macro_f1"}
    assert set(delong["class_code"]) == {0, 1, 2}
    assert bootstrap["p_fdr"].notna().all()
    assert manifest["comparison_split"] == "validation"
    assert manifest["confirmatory_splits_accessed"] is False
