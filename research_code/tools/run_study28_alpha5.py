#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from los_baseline.study28.robustness_isolated import run_alpha5_isolated


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Study28 V9 Alpha 5: robustness, interpretability, and HFD28 secondary analysis.")
    p.add_argument("--dataset-root", required=True, type=Path)
    p.add_argument("--alpha2-output-dir", required=True, type=Path)
    p.add_argument("--alpha3-output-dir", required=True, type=Path)
    p.add_argument("--alpha4-output-dir", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    return p.parse_args()


def main() -> int:
    args = parse_args()
    result = run_alpha5_isolated(
        dataset_root=args.dataset_root, project_root=args.project_root,
        alpha2_output_dir=args.alpha2_output_dir, alpha3_output_dir=args.alpha3_output_dir,
        alpha4_output_dir=args.alpha4_output_dir, output_dir=args.output_dir,
    )
    print(json.dumps(result.to_dict(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
