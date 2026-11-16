#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from los_baseline.study28.audit import audit_study_datasets


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit the frozen professor-provided Sepsis-3 28-day datasets without modifying them."
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument(
        "--registry",
        type=Path,
        default=PROJECT_ROOT / "configs" / "dataset_registry_28d.json",
    )
    parser.add_argument("--feature-list", type=Path, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "runs" / "study28_audit",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result = audit_study_datasets(
        dataset_root=args.dataset_root,
        registry_path=args.registry,
        feature_list_path=args.feature_list,
        output_dir=args.output_dir,
    )
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    return 0 if result.passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
