#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from los_baseline.study28.manuscript_reporting import run_alpha6


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Study28 V9 Alpha 6: manuscript evidence and final reporting layer.")
    p.add_argument("--alpha2-output-dir", type=Path)
    p.add_argument("--alpha3-output-dir", type=Path)
    p.add_argument("--alpha4-output-dir", type=Path)
    p.add_argument("--alpha5-output-dir", type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument("--use-release-verification-snapshots", action="store_true",
                   help="Build a non-patient-level verification report from packaged Alpha2–5 aggregate snapshots.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    result = run_alpha6(
        project_root=args.project_root, output_dir=args.output_dir,
        alpha2_output_dir=args.alpha2_output_dir, alpha3_output_dir=args.alpha3_output_dir,
        alpha4_output_dir=args.alpha4_output_dir, alpha5_output_dir=args.alpha5_output_dir,
        use_release_verification_snapshots=args.use_release_verification_snapshots,
    )
    print(json.dumps(result.to_dict(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
