#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import shutil
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from los_baseline.config import load_config
from los_baseline.confirmatory_v8 import run_strategy_confirmatory
from los_baseline.external_validation_guard import assert_external_mortality_ready
from los_baseline.site_recalibration_v8 import SiteRecalibrationSettings
from los_baseline.utils import atomic_write_json, sha256_file


def _read_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return payload


def _lock_payload(split: str, freeze: Path) -> dict:
    return {
        "schema_version": "exp123_v8_confirmatory_access_lock_v1",
        "split": split,
        "opened_at_utc": datetime.now(timezone.utc).isoformat(),
        "design_freeze": str(freeze),
        "design_freeze_sha256": sha256_file(freeze),
        "state": "opened_incomplete",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run single-use V8 confirmatory evaluation for all frozen candidates.")
    parser.add_argument("--design-freeze", required=True)
    parser.add_argument("--split", required=True, choices=["internal_test", "external_eicu"])
    parser.add_argument("--confirm-open-internal-test", action="store_true")
    parser.add_argument("--confirm-open-eicu", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    freeze_path = Path(args.design_freeze).expanduser().resolve()
    freeze = _read_json(freeze_path)
    if freeze.get("status") != "frozen":
        raise ValueError("Confirmatory design freeze is not frozen.")
    recorded = freeze_path.with_name(freeze_path.name + ".sha256")
    if not recorded.is_file() or recorded.read_text(encoding="utf-8").strip() != sha256_file(freeze_path):
        raise ValueError("Confirmatory design freeze SHA-256 verification failed.")
    matrix_path = Path(str(freeze.get("matrix_manifest", ""))).expanduser().resolve()
    if not matrix_path.is_file():
        raise FileNotFoundError(f"Frozen matrix manifest is missing: {matrix_path}")
    if sha256_file(matrix_path) != str(freeze.get("matrix_manifest_sha256", "")):
        raise ValueError("Frozen matrix manifest changed after confirmatory design freeze.")
    split = args.split
    if split == "internal_test" and not args.confirm_open_internal_test:
        raise RuntimeError("Opening Internal Test requires --confirm-open-internal-test.")
    if split == "external_eicu" and not args.confirm_open_eicu:
        raise RuntimeError("Opening eICU requires --confirm-open-eicu.")

    root = freeze_path.parent / "results"
    root.mkdir(parents=True, exist_ok=True)
    opened_lock = freeze_path.parent / ("INTERNAL_TEST_OPENED.lock.json" if split == "internal_test" else "EXTERNAL_EICU_OPENED.lock.json")
    completed_lock = freeze_path.parent / ("INTERNAL_TEST_COMPLETED.lock.json" if split == "internal_test" else "EXTERNAL_EICU_COMPLETED.lock.json")
    if completed_lock.exists():
        raise RuntimeError(f"{split} has already been completed and cannot be re-opened.")
    if opened_lock.exists() and not args.resume:
        raise RuntimeError(f"{split} was previously opened but not completed. Use --resume with the same frozen design.")
    if split == "external_eicu":
        internal_completed = freeze_path.parent / "INTERNAL_TEST_COMPLETED.lock.json"
        if not internal_completed.is_file():
            raise RuntimeError("eICU may only be opened after the frozen Internal Test batch is completed.")
        # Guard all unique cohort-source runs before marking the external split opened.
        checked = set()
        for entry in freeze["selections"]:
            selection = _read_json(Path(entry["selection_path"]))
            config = load_config(Path(entry["config_path"]))
            source_run = str(selection.get("cohort_source_run_id", selection["run_id"]))
            key = (source_run, entry["config_path"])
            if key in checked:
                continue
            assert_external_mortality_ready(config=config, run_dir=Path(freeze["runs_dir"]) / source_run)
            checked.add(key)
    if not opened_lock.exists():
        atomic_write_json(opened_lock, _lock_payload(split, freeze_path))

    results = []
    try:
        for entry in freeze["selections"]:
            selection_path = Path(entry["selection_path"])
            if sha256_file(selection_path) != str(entry["selection_sha256"]):
                raise ValueError(f"Frozen selection changed: {selection_path}")
            config_path = Path(entry["config_path"])
            if sha256_file(config_path) != str(entry["config_sha256"]):
                raise ValueError(f"Frozen config changed: {config_path}")
            selection = _read_json(selection_path)
            # Freeze the exact fitted artifacts, not only their selection metadata.
            for artifact_path, expected_sha in dict(selection.get("artifact_hashes", {})).items():
                source = Path(str(artifact_path))
                if not source.is_file():
                    raise FileNotFoundError(f"Frozen predictor artifact is missing: {source}")
                if sha256_file(source) != str(expected_sha):
                    raise ValueError(f"Frozen predictor artifact changed after design freeze: {source}")
            config = load_config(config_path)
            candidate_out = root / split
            candidate_dir = candidate_out / f"LOS{int(entry['outcome_horizon_days'])}" / str(entry["strategy"])
            candidate_manifest = candidate_dir / "confirmatory_result_manifest.json"
            if candidate_manifest.is_file() and args.resume:
                results.append(_read_json(candidate_manifest))
                continue
            if candidate_dir.exists() and not candidate_manifest.is_file():
                if not args.resume:
                    raise RuntimeError(
                        f"Incomplete confirmatory candidate output exists: {candidate_dir}. "
                        "Use --resume to rerun the same frozen candidate."
                    )
                # A partially written candidate is not a completed confirmatory result.
                # Remove only this candidate directory and rerun the exact frozen selection.
                shutil.rmtree(candidate_dir)
            site_settings = None
            if split == "external_eicu":
                cfg = dict(config.get("confirmatory_v8", {}).get("site_recalibration", {}))
                if bool(cfg.get("enabled", True)):
                    site_settings = SiteRecalibrationSettings(
                        calibration_fraction=float(cfg.get("calibration_fraction", 0.20)),
                        repeats=int(cfg.get("repeats", 20)),
                        seed=int(cfg.get("seed", 2026)),
                        minimum_successful_repeats=int(cfg.get("minimum_successful_repeats", 15)),
                    )
            result = run_strategy_confirmatory(
                selection=selection,
                config=config,
                runs_dir=Path(freeze["runs_dir"]),
                split_name=split,
                output_root=candidate_out,
                site_recalibration_settings=site_settings,
            )
            results.append(result)
    except Exception:
        state = _read_json(opened_lock)
        state["state"] = "opened_incomplete"
        state["last_failure_at_utc"] = datetime.now(timezone.utc).isoformat()
        atomic_write_json(opened_lock, state)
        raise

    summary = {
        "schema_version": "exp123_v8_confirmatory_batch_result_v1",
        "split": split,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "design_freeze_sha256": sha256_file(freeze_path),
        "candidate_count": len(results),
        "results": results,
        "threshold_reselection_performed": False,
        "raw_external_validation_is_primary": bool(split == "external_eicu"),
        "site_recalibration_role": "secondary_local_adaptation" if split == "external_eicu" else None,
    }
    summary_path = root / split / "confirmatory_batch_summary.json"
    atomic_write_json(summary_path, summary)
    atomic_write_json(completed_lock, {
        "schema_version": "exp123_v8_confirmatory_completion_lock_v1",
        "split": split,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "design_freeze_sha256": sha256_file(freeze_path),
        "summary_path": str(summary_path),
        "summary_sha256": sha256_file(summary_path),
        "single_use_complete": True,
    })
    state = _read_json(opened_lock); state["state"] = "completed"; state["completed_at_utc"] = datetime.now(timezone.utc).isoformat(); atomic_write_json(opened_lock, state)
    print(f"[OK] V8 confirmatory batch completed: {split}")
    print(f"[RESULT] {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
