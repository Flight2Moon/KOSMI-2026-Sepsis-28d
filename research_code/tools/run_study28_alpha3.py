#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from los_baseline.study28.external_validation import run_external_validation_isolated


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Study28 V9 Alpha 3 mandatory raw external validation"
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--alpha2-output-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    args = parser.parse_args()
    freeze_manifest = (
        args.alpha2_output_dir.expanduser().resolve()
        / "03_external_model_freeze"
        / "study28_model_freeze.json"
    )
    result = run_external_validation_isolated(
        dataset_root=args.dataset_root,
        project_root=args.project_root,
        freeze_manifest_path=freeze_manifest,
        output_dir=args.output_dir,
    )
    print(json.dumps(result.to_dict(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
