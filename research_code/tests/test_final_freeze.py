from __future__ import annotations

import importlib.metadata
import json
from pathlib import Path

import pandas as pd
import pytest

from los_baseline.final_freeze import (
    _required_discovery_files,
    assert_confirmatory_step_allowed,
    finalize_exp123_run,
    freeze_confirmatory_design,
    freeze_exp123_release,
    parse_exact_requirements,
    run_exp123_preflight,
    validate_installed_versions,
    verify_confirmatory_design,
    verify_exp123_release,
    verify_finalized_run,
)


def _minimal_config(project_root: Path) -> dict:
    return {
        "paths": {
            "mimic_csv": str(project_root / "mimic.csv"),
            "eicu_csv": str(project_root / "eicu.csv"),
            "runs_dir": "runs",
        },
        "data": {
            "feature_columns": ["age", "sofa_score"],
            "target_column": "LOS",
            "identifier_columns": ["subject_id", "hadm_id", "stay_id"],
        },
        "outcomes": {
            "mortality_column": "hospital_expire_flag",
            "los_column": "LOS",
            "landmark_days": 28,
        },
        "development": {
            "completed_steps": [
                "outcomes",
                "common_three_class_splits",
                "shared_training_engine",
                "optimization_engine",
                "exp1_flat_multiclass",
                "stage1_mortality",
                "stage2_long_stay_and_exp3",
                "stage_heterogeneity",
                "super_learner",
                "oracle_pipeline",
                "final_statistical_comparison",
                "final_freeze",
            ],
            "model_training_enabled": True,
        },
        "final_freeze": {"minimum_free_disk_gb": 0.0},
    }


def _prepare_project(tmp_path: Path) -> tuple[Path, Path]:
    project_root = tmp_path / "project"
    (project_root / "src" / "los_baseline").mkdir(parents=True)
    (project_root / "configs").mkdir(parents=True)
    (project_root / "tests").mkdir(parents=True)
    (project_root / "src" / "los_baseline" / "dummy.py").write_text(
        "VALUE = 1\n", encoding="utf-8"
    )
    numpy_version = importlib.metadata.version("numpy")
    (project_root / "requirements.txt").write_text(
        f"numpy=={numpy_version}\n", encoding="utf-8"
    )
    config = _minimal_config(project_root)
    config_path = project_root / "configs" / "experiment.json"
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    columns = [
        "subject_id",
        "hadm_id",
        "stay_id",
        "age",
        "sofa_score",
        "LOS",
        "hospital_expire_flag",
    ]
    frame = pd.DataFrame(
        [
            [1, 11, 111, 60, 3, 5.0, 0],
            [2, 22, 222, 70, 7, 10.0, 1],
            [3, 33, 333, 80, 5, 30.0, 0],
        ],
        columns=columns,
    )
    frame.to_csv(project_root / "mimic.csv", index=False)
    frame.to_csv(project_root / "eicu.csv", index=False)
    return project_root, config_path


def _write_required_discovery_files(run_dir: Path) -> None:
    for path in _required_discovery_files(run_dir):
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.name == "split_hash.txt":
            path.write_text("a" * 64 + "\n", encoding="utf-8")
        elif path.suffix == ".gz":
            pd.DataFrame({"x": [1]}).to_csv(path, index=False, compression="gzip")
        else:
            path.write_text("{}\n", encoding="utf-8")
    task_roots = [
        run_dir / "EXP1_flat_3class",
        run_dir / "EXP2_two_stage/stage1_mortality",
        run_dir / "EXP2_two_stage/stage2_long_stay",
    ]
    for task_root in task_roots:
        for model_name in (
            "decision_tree",
            "logistic_regression",
            "random_forest",
            "svm",
            "xgboost",
        ):
            artifact = task_root / model_name / "artifact" / "model.bin"
            artifact.parent.mkdir(parents=True, exist_ok=True)
            artifact.write_bytes(b"artifact")


def _write_confirmatory_outputs(run_dir: Path) -> None:
    outputs = [
        run_dir / "EXP2_two_stage/heterogeneity_analysis/stage1_mortality/confirmatory/internal_test/run_config.json",
        run_dir / "EXP2_two_stage/heterogeneity_analysis/stage2_long_stay/confirmatory/internal_test/run_config.json",
        run_dir / "EXP2_two_stage/pipeline_evaluation/internal_test/COMPLETED.lock.json",
        run_dir / "final_comparison/internal_test/COMPLETED.lock.json",
        run_dir / "EXP2_two_stage/heterogeneity_analysis/stage1_mortality/confirmatory/external_eicu/run_config.json",
        run_dir / "EXP2_two_stage/heterogeneity_analysis/stage2_long_stay/confirmatory/external_eicu/run_config.json",
        run_dir / "EXP2_two_stage/pipeline_evaluation/external_eicu/COMPLETED.lock.json",
        run_dir / "final_comparison/external_eicu/COMPLETED.lock.json",
    ]
    for path in outputs:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n", encoding="utf-8")


def test_exact_requirements_reject_ranges(tmp_path: Path) -> None:
    path = tmp_path / "requirements.txt"
    path.write_text("numpy>=1.0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="exact == pin"):
        parse_exact_requirements(path)


def test_local_build_suffix_is_allowed() -> None:
    errors = validate_installed_versions(
        {"torch": "2.10.0"}, {"torch": "2.10.0+cpu"}
    )
    assert errors == []


def test_release_freeze_and_source_tamper_detection(tmp_path: Path) -> None:
    project_root, config_path = _prepare_project(tmp_path)
    result = freeze_exp123_release(
        project_root=project_root,
        config_path=config_path,
        strict_environment=True,
    )
    assert Path(result.manifest_path).is_file()
    assert verify_exp123_release(project_root=project_root)["status"] == "verified"
    source = project_root / "src" / "los_baseline" / "dummy.py"
    source.write_text("VALUE = 2\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="source"):
        verify_exp123_release(project_root=project_root)


def test_preflight_confirmatory_freeze_and_final_run_freeze(tmp_path: Path) -> None:
    project_root, config_path = _prepare_project(tmp_path)
    freeze_exp123_release(
        project_root=project_root,
        config_path=config_path,
        strict_environment=True,
    )
    preflight = run_exp123_preflight(
        project_root=project_root,
        config_path=config_path,
        run_id="exp123_test",
    )
    assert Path(preflight.lock_path).is_file()
    run_dir = project_root / "runs" / "exp123_test"
    _write_required_discovery_files(run_dir)
    frozen = freeze_confirmatory_design(
        project_root=project_root,
        config_path=config_path,
        run_dir=run_dir,
    )
    assert frozen.frozen_file_count >= 30
    assert verify_confirmatory_design(
        project_root=project_root,
        config_path=config_path,
        run_dir=run_dir,
    )["status"] == "verified"

    assert_confirmatory_step_allowed(
        project_root=project_root,
        config_path=config_path,
        run_dir=run_dir,
        step="stage1-heterogeneity-confirmatory",
        split="internal_test",
    )
    with pytest.raises(RuntimeError, match="prior step"):
        assert_confirmatory_step_allowed(
            project_root=project_root,
            config_path=config_path,
            run_dir=run_dir,
            step="stage2-heterogeneity-confirmatory",
            split="internal_test",
        )

    _write_confirmatory_outputs(run_dir)
    final = finalize_exp123_run(
        project_root=project_root,
        config_path=config_path,
        run_dir=run_dir,
    )
    assert final.frozen_file_count > frozen.frozen_file_count
    assert verify_finalized_run(run_dir=run_dir)["status"] == "verified"
