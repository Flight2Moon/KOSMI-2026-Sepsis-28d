#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from los_baseline.config import load_config, resolve_path
from los_baseline.confirmatory_selection_v8 import build_strategy_selection, write_strategy_selection
from los_baseline.utils import atomic_write_json, sha256_file


def _read_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Freeze V8 confirmatory candidates using Validation only.")
    parser.add_argument("--matrix-manifest", required=True)
    parser.add_argument("--output-dir")
    args = parser.parse_args()

    matrix_path = Path(args.matrix_manifest).expanduser().resolve()
    matrix = _read_json(matrix_path)
    if matrix.get("confirmatory_splits_accessed") is not False:
        raise ValueError("Matrix manifest must state confirmatory_splits_accessed=false before design freeze.")
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else matrix_path.parent / "confirmatory_v8"
    freeze_path = output_dir / "confirmatory_design_freeze.json"
    if freeze_path.exists():
        raise FileExistsError(f"Confirmatory design is already frozen: {freeze_path}")
    output_dir.mkdir(parents=True, exist_ok=True)

    selections = []
    config_hashes = {}
    runs_dir_expected = None
    for item in matrix.get("runs", []):
        if not bool(item.get("enabled", True)) or str(item.get("status")) == "not_requested":
            continue
        if str(item.get("status")) != "completed":
            raise RuntimeError(f"Cannot freeze confirmatory design before all requested discovery runs complete: {item}")
        run_id = str(item.get("run_id") or "")
        strategy = str(item.get("strategy"))
        horizon = int(item.get("outcome_horizon_days"))
        if not run_id:
            raise ValueError(f"Completed matrix entry is missing run_id: {item}")
        config_path = PROJECT_ROOT / "configs" / f"experiment_cuda_v8_los{horizon}.json"
        if not config_path.is_file():
            raise FileNotFoundError(config_path)
        config = load_config(config_path)
        runs_dir = resolve_path(config, config["paths"]["runs_dir"])
        if runs_dir_expected is None:
            runs_dir_expected = runs_dir
        elif runs_dir != runs_dir_expected:
            raise ValueError("All confirmatory candidates must use the same runs_dir.")
        run_dir = runs_dir / run_id
        if not run_dir.is_dir():
            raise FileNotFoundError(run_dir)
        for split in ("internal_test", "external_eicu"):
            existing = run_dir / "V8_confirmatory" / split
            if existing.exists() and any(existing.iterdir()):
                raise RuntimeError(f"Confirmatory output already exists before design freeze: {existing}")
        selection = build_strategy_selection(
            run_dir=run_dir,
            run_id=run_id,
            strategy=strategy,
            config=config,
            runs_dir=runs_dir,
        )
        selection_path = output_dir / "selections" / f"LOS{horizon}_{strategy}.json"
        write_strategy_selection(selection_path, selection)
        selections.append({
            "outcome_horizon_days": horizon,
            "strategy": strategy,
            "run_id": run_id,
            "config_path": str(config_path.resolve()),
            "config_sha256": sha256_file(config_path),
            "selection_path": str(selection_path),
            "selection_sha256": sha256_file(selection_path),
        })
        config_hashes[str(config_path.resolve())] = sha256_file(config_path)

    if not selections:
        raise ValueError("No completed prespecified V8 runs were found in the matrix manifest.")
    payload = {
        "schema_version": "exp123_v8_confirmatory_design_freeze_v1",
        "status": "frozen",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "matrix_manifest": str(matrix_path),
        "matrix_manifest_sha256": sha256_file(matrix_path),
        "runs_dir": str(runs_dir_expected),
        "candidate_count": len(selections),
        "selections": selections,
        "selection_basis": "validation_only",
        "internal_test_single_use": True,
        "external_eicu_single_use": True,
        "internal_test_opened": False,
        "external_eicu_opened": False,
        "confirmatory_overwrite_allowed": False,
        "threshold_reselection_on_confirmatory": False,
        "raw_external_validation_is_primary": True,
        "site_recalibration_is_secondary_local_adaptation": True,
        "config_hashes": config_hashes,
    }
    atomic_write_json(freeze_path, payload)
    (freeze_path.with_name(freeze_path.name + ".sha256")).write_text(sha256_file(freeze_path) + "\n", encoding="utf-8")
    (output_dir / "CONFIRMATORY_DESIGN_FROZEN.lock.txt").write_text(
        f"V8 confirmatory design frozen.\nSHA-256: {sha256_file(freeze_path)}\n", encoding="utf-8"
    )
    print(f"[OK] V8 confirmatory design frozen: {freeze_path}")
    print(f"[INFO] Frozen candidates: {len(selections)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
