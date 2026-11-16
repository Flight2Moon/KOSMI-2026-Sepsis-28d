from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from los_baseline.config import load_config, resolve_path
from los_baseline.heterogeneity import (
    HeterogeneityConfig,
    run_heterogeneity_confirmatory,
    run_heterogeneity_discovery,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose model-by-subgroup performance heterogeneity."
    )
    parser.add_argument("--config", type=Path, default=Path("configs/experiment.json"))
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--threshold", type=int, choices=[7, 14, 28], required=True)
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
    runs_dir = resolve_path(config, config["paths"]["runs_dir"])
    run_dir = runs_dir / arguments.run_id
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run directory was not found: {run_dir}")

    if arguments.mode == "discovery":
        if arguments.split != "validation":
            raise ValueError("Discovery mode is restricted to the validation split.")
        analysis_config = HeterogeneityConfig(
            run_dir=run_dir,
            threshold_days=arguments.threshold,
        )
        output = run_heterogeneity_discovery(
            analysis_config,
            resolve_path(config, config["paths"]["mimic_csv"]),
        )
    else:
        if arguments.split == "validation":
            raise ValueError(
                "Confirmatory mode is for internal_test or external_eicu only."
            )
        if arguments.frozen_design is None:
            raise ValueError("--frozen-design is required in confirmatory mode.")
        dataset_key = "eicu_csv" if arguments.split == "external_eicu" else "mimic_csv"
        output = run_heterogeneity_confirmatory(
            run_dir,
            arguments.split,
            resolve_path(config, config["paths"][dataset_key]),
            arguments.frozen_design,
        )
    print(f"[OK] Heterogeneity analysis saved: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
