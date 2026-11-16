#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from los_baseline.study28.resource_plan import write_resource_plan


def main() -> int:
    p = argparse.ArgumentParser(description="Detect hardware and freeze the Study28 execution resource plan.")
    p.add_argument("--output-json", type=Path, required=True)
    p.add_argument("--output-env", type=Path, required=True)
    p.add_argument("--profile", choices=["maximum_throughput", "balanced"], default="maximum_throughput")
    a = p.parse_args()
    plan = write_resource_plan(json_path=a.output_json, env_path=a.output_env, profile=a.profile)
    print(json.dumps(plan.to_dict(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
