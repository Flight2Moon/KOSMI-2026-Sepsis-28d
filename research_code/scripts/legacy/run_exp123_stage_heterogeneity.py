from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from los_baseline.config import load_config, resolve_path
from los_baseline.stage_heterogeneity import (
    run_stage_heterogeneity_confirmatory,
    run_stage_heterogeneity_discovery,
    stage_heterogeneity_config_from_project,
)
from los_baseline.task_spec import STAGE1_MORTALITY_TASK, STAGE2_LONG_STAY_TASK


TASK_ALIASES = {
    "stage1": STAGE1_MORTALITY_TASK.task_name,
    "stage1_mortality": STAGE1_MORTALITY_TASK.task_name,
    "stage2": STAGE2_LONG_STAY_TASK.task_name,
    "stage2_long_stay": STAGE2_LONG_STAY_TASK.task_name,
}


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Stage1 or Stage2 model-by-subgroup heterogeneity analysis."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "experiment.json",
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--task", choices=sorted(TASK_ALIASES), required=True)
    parser.add_argument("--mode", choices=["discovery", "confirmatory"], required=True)
    parser.add_argument(
        "--split",
        choices=["validation", "internal_test", "external_eicu"],
        default="validation",
    )
    parser.add_argument("--frozen-design", type=Path)
    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()
    config = load_config(arguments.config)
    if not bool(config.get("development", {}).get("model_training_enabled", False)):
        raise RuntimeError(
            "Clinical EXP123 execution remains locked until all implementation "
            "steps are complete. Use run_exp123_stage_heterogeneity_smoke.sh "
            "to validate Step 8."
        )
    runs_dir = resolve_path(config, config["paths"]["runs_dir"])
    run_dir = runs_dir / arguments.run_id
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run directory was not found: {run_dir}")
    task_name = TASK_ALIASES[arguments.task]
    target_column = str(config["data"]["target_column"])
    identifier_columns = tuple(config["data"].get("identifier_columns", []))

    if arguments.mode == "discovery":
        if arguments.split != "validation":
            raise ValueError("Discovery mode is restricted to validation.")
        analysis_config = stage_heterogeneity_config_from_project(
            run_dir=run_dir,
            task_name=task_name,
            project_config=config,
        )
        output = run_stage_heterogeneity_discovery(
            analysis_config,
            dataset_path=resolve_path(config, config["paths"]["mimic_csv"]),
            target_column=target_column,
            identifier_columns=identifier_columns,
        )
    else:
        if arguments.split == "validation":
            raise ValueError(
                "Confirmatory mode is restricted to internal_test or external_eicu."
            )
        if arguments.frozen_design is None:
            raise ValueError("--frozen-design is required in confirmatory mode.")
        dataset_key = (
            "eicu_csv" if arguments.split == "external_eicu" else "mimic_csv"
        )
        output = run_stage_heterogeneity_confirmatory(
            run_dir=run_dir,
            task_name=task_name,
            split=arguments.split,
            dataset_path=resolve_path(config, config["paths"][dataset_key]),
            frozen_design_path=arguments.frozen_design,
            target_column=target_column,
            identifier_columns=identifier_columns,
        )
    print(f"[OK] Stage heterogeneity analysis saved: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
