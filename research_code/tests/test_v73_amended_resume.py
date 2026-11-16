from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from los_baseline.amendments import verify_amended_run_preflight
from los_baseline.utils import sha256_file


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _build_amended_fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    project = tmp_path / "project"
    release = project / "exp123_release_freeze"
    release.mkdir(parents=True)
    source_a = project / "src" / "a.py"
    source_b = project / "src" / "b.py"
    source_a.parent.mkdir(parents=True)
    source_a.write_text("VALUE = 1\n", encoding="utf-8")
    source_b.write_text("UNCHANGED = True\n", encoding="utf-8")
    inventory = [
        {
            "category": "project_source",
            "path": "src/a.py",
            "size_bytes": source_a.stat().st_size,
            "sha256": sha256_file(source_a),
        },
        {
            "category": "project_source",
            "path": "src/b.py",
            "size_bytes": source_b.stat().st_size,
            "sha256": sha256_file(source_b),
        },
    ]
    with (release / "source_inventory.csv").open(
        "w", encoding="utf-8", newline=""
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["category", "path", "size_bytes", "sha256"],
        )
        writer.writeheader()
        writer.writerows(inventory)
    _write_json(release / "release_manifest.json", {"status": "frozen"})

    config = project / "config.json"
    _write_json(config, {"paths": {}})
    dataset = project / "dataset.csv"
    dataset.write_text("x\n1\n", encoding="utf-8")
    run_dir = project / "runs" / "run_1"
    split_hash = "a" * 64
    split_path = run_dir / "common" / "splits" / "split_hash.txt"
    split_path.parent.mkdir(parents=True)
    split_path.write_text(split_hash + "\n", encoding="utf-8")
    preflight_path = (
        run_dir / "common" / "preflight" / "PREFLIGHT_PASSED.lock.json"
    )
    _write_json(
        preflight_path,
        {
            "status": "passed",
            "run_id": "run_1",
            "config_sha256": sha256_file(config),
            "release_manifest_sha256": sha256_file(
                release / "release_manifest.json"
            ),
            "datasets": {
                "mimic": {
                    "path": str(dataset),
                    "size_bytes": dataset.stat().st_size,
                    "sha256": sha256_file(dataset),
                }
            },
        },
    )
    preflight_path.with_suffix(preflight_path.suffix + ".sha256").write_text(
        sha256_file(preflight_path) + "\n", encoding="utf-8"
    )

    source_a.write_text("VALUE = 2\n", encoding="utf-8")
    amendment = run_dir / "common" / "amendments" / "v73.json"
    _write_json(
        amendment,
        {
            "run_id": "run_1",
            "base_release_manifest_sha256": sha256_file(
                release / "release_manifest.json"
            ),
            "preflight_lock_sha256": sha256_file(preflight_path),
            "shared_split_hash": split_hash,
            "confirmatory_splits_accessed": False,
            "patched_files": [
                {"path": "src/a.py", "sha256": sha256_file(source_a)}
            ],
        },
    )
    return project, config, run_dir, amendment


def test_amended_preflight_preserves_original_freeze_and_declared_change(
    tmp_path: Path,
) -> None:
    project, config, run_dir, amendment = _build_amended_fixture(tmp_path)
    result = verify_amended_run_preflight(
        project_root=project,
        config_path=config,
        run_dir=run_dir,
        amendment_path=amendment,
        verify_dataset_hashes=True,
    )
    assert result["status"] == "verified_with_amendment"
    assert result["declared_amended_file_count"] == 1


def test_amended_preflight_rejects_undeclared_source_change(tmp_path: Path) -> None:
    project, config, run_dir, amendment = _build_amended_fixture(tmp_path)
    (project / "src" / "b.py").write_text("UNCHANGED = False\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="without an amendment declaration"):
        verify_amended_run_preflight(
            project_root=project,
            config_path=config,
            run_dir=run_dir,
            amendment_path=amendment,
        )
