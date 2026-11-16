#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from los_baseline.config import load_config
from los_baseline.runner import run_experiments


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train MIMIC-IV LOS models for 7, 14, and 28 days, "
            "evaluate internal splits, and perform eICU external validation."
        )
    )
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs" / "experiment.json"),
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        choices=[
            "decision_tree",
            "logistic_regression",
            "random_forest",
            "svm",
            "xgboost",
        ],
    )
    parser.add_argument(
        "--thresholds",
        nargs="*",
        type=float,
        default=None,
    )
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(Path(args.config))
    run_experiments(
        config=config,
        selected_models=args.models,
        selected_thresholds=args.thresholds,
        run_id=args.run_id,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
