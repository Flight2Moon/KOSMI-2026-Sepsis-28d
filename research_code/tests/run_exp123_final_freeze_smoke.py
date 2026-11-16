from __future__ import annotations

import importlib.metadata
import json
import tempfile
from pathlib import Path

import pandas as pd

from los_baseline.final_freeze import (
    _required_discovery_files,
    finalize_exp123_run,
    freeze_confirmatory_design,
    freeze_exp123_release,
    run_exp123_preflight,
    verify_confirmatory_design,
    verify_exp123_release,
    verify_finalized_run,
)


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="exp123_final_freeze_") as temp:
        root = Path(temp) / "project"
        (root / "src" / "los_baseline").mkdir(parents=True)
        (root / "configs").mkdir(parents=True)
        (root / "src" / "los_baseline" / "smoke.py").write_text(
            "READY = True\n", encoding="utf-8"
        )
        numpy_version = importlib.metadata.version("numpy")
        (root / "requirements.txt").write_text(
            f"numpy=={numpy_version}\n", encoding="utf-8"
        )
        config = {
            "paths": {
                "mimic_csv": str(root / "mimic.csv"),
                "eicu_csv": str(root / "eicu.csv"),
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
        config_path = root / "configs" / "experiment.json"
        config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
        frame = pd.DataFrame(
            {
                "subject_id": [1, 2, 3],
                "hadm_id": [11, 22, 33],
                "stay_id": [111, 222, 333],
                "age": [50, 60, 70],
                "sofa_score": [2, 5, 8],
                "LOS": [5.0, 10.0, 30.0],
                "hospital_expire_flag": [0, 1, 0],
            }
        )
        frame.to_csv(root / "mimic.csv", index=False)
        frame.to_csv(root / "eicu.csv", index=False)

        release = freeze_exp123_release(
            project_root=root, config_path=config_path, strict_environment=True
        )
        verify_exp123_release(project_root=root)
        preflight = run_exp123_preflight(
            project_root=root, config_path=config_path, run_id="smoke"
        )
        run_dir = root / "runs" / "smoke"
        for path in _required_discovery_files(run_dir):
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.name == "split_hash.txt":
                path.write_text("b" * 64 + "\n", encoding="utf-8")
            elif path.suffix == ".gz":
                pd.DataFrame({"x": [1]}).to_csv(
                    path, index=False, compression="gzip"
                )
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
        confirmatory = freeze_confirmatory_design(
            project_root=root, config_path=config_path, run_dir=run_dir
        )
        verify_confirmatory_design(
            project_root=root, config_path=config_path, run_dir=run_dir
        )
        completion_paths = [
            run_dir / "EXP2_two_stage/heterogeneity_analysis/stage1_mortality/confirmatory/internal_test/run_config.json",
            run_dir / "EXP2_two_stage/heterogeneity_analysis/stage2_long_stay/confirmatory/internal_test/run_config.json",
            run_dir / "EXP2_two_stage/pipeline_evaluation/internal_test/COMPLETED.lock.json",
            run_dir / "final_comparison/internal_test/COMPLETED.lock.json",
            run_dir / "EXP2_two_stage/heterogeneity_analysis/stage1_mortality/confirmatory/external_eicu/run_config.json",
            run_dir / "EXP2_two_stage/heterogeneity_analysis/stage2_long_stay/confirmatory/external_eicu/run_config.json",
            run_dir / "EXP2_two_stage/pipeline_evaluation/external_eicu/COMPLETED.lock.json",
            run_dir / "final_comparison/external_eicu/COMPLETED.lock.json",
        ]
        for path in completion_paths:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("{}\n", encoding="utf-8")
        final = finalize_exp123_run(
            project_root=root, config_path=config_path, run_dir=run_dir
        )
        verify_finalized_run(run_dir=run_dir)
        print(f"[OK] Release manifest: {release.manifest_sha256}")
        print(f"[OK] Preflight lock: {preflight.lock_sha256}")
        print(f"[OK] Confirmatory design: {confirmatory.manifest_sha256}")
        print(f"[OK] Final run freeze: {final.manifest_sha256}")
        print("[OK] Step 12 final-freeze smoke completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
