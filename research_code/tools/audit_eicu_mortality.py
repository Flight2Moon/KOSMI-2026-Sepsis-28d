#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
PROJECT_ROOT=Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from los_baseline.config import load_config, resolve_path
from los_baseline.external_validation_guard import audit_external_mortality_flag

def main() -> int:
    parser=argparse.ArgumentParser(description="Audit the harmonized eICU mortality flag.")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "experiment_cuda.json"))
    parser.add_argument("--output", default=None)
    args=parser.parse_args()
    config=load_config(Path(args.config))
    settings=dict(config.get("external_validation", {}).get("mortality_guard", {}))
    result=audit_external_mortality_flag(
        dataset_path=resolve_path(config, config["paths"]["eicu_csv"]),
        mortality_column=str(config["outcomes"]["mortality_column"]),
        settings=settings,
    )
    payload=result.to_dict()
    if args.output:
        Path(args.output).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if result.passed else 2
if __name__ == "__main__":
    raise SystemExit(main())
