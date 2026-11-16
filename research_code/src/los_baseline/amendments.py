from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any, Mapping

from .utils import sha256_file

AMENDMENT_ENV = "EXP123_AMENDMENT_MANIFEST"


def _load_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required JSON file was not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a JSON object in {path}")
    return payload


def _sidecar_digest(path: Path) -> str:
    sidecar = path.with_suffix(path.suffix + ".sha256")
    if not sidecar.is_file():
        raise FileNotFoundError(f"Checksum sidecar was not found: {sidecar}")
    value = sidecar.read_text(encoding="utf-8").strip().split()[0]
    if len(value) != 64:
        raise ValueError(f"Invalid SHA-256 sidecar: {sidecar}")
    return value


def verify_amended_run_preflight(
    *,
    project_root: Path,
    config_path: Path,
    run_dir: Path,
    amendment_path: Path,
    verify_dataset_hashes: bool = False,
) -> dict[str, Any]:
    """Verify a recorded source amendment without rewriting the original preflight.

    The original release manifest and every unamended frozen source file remain
    immutable. Only files explicitly listed in the amendment may differ from the
    original release inventory.
    """
    project_root = project_root.resolve()
    config_path = config_path.resolve()
    run_dir = run_dir.resolve()
    amendment_path = amendment_path.resolve()

    preflight_path = run_dir / "common" / "preflight" / "PREFLIGHT_PASSED.lock.json"
    preflight = _load_object(preflight_path)
    amendment = _load_object(amendment_path)
    errors: list[str] = []

    actual_preflight_sha = sha256_file(preflight_path)
    if _sidecar_digest(preflight_path) != actual_preflight_sha:
        errors.append("Preflight lock checksum mismatch.")
    if preflight.get("status") != "passed":
        errors.append("Preflight status is not passed.")
    if str(preflight.get("run_id")) != run_dir.name:
        errors.append("Preflight Run ID does not match the target directory.")
    if sha256_file(config_path) != str(preflight.get("config_sha256")):
        errors.append("Configuration checksum changed after preflight.")

    release_manifest = project_root / "exp123_release_freeze" / "release_manifest.json"
    if not release_manifest.is_file():
        errors.append(f"Original release manifest is unavailable: {release_manifest}")
    else:
        release_sha = sha256_file(release_manifest)
        if release_sha != str(preflight.get("release_manifest_sha256")):
            errors.append("Original release manifest changed after preflight.")
        if release_sha != str(amendment.get("base_release_manifest_sha256")):
            errors.append("Amendment does not reference the original frozen release.")

    if str(amendment.get("run_id")) != run_dir.name:
        errors.append("Amendment Run ID does not match the target Run.")
    if str(amendment.get("preflight_lock_sha256")) != actual_preflight_sha:
        errors.append("Amendment preflight checksum does not match the Run lock.")
    if amendment.get("confirmatory_splits_accessed") is not False:
        errors.append("Amendment must state that confirmatory splits were not accessed.")

    declared: dict[str, str] = {}
    for entry in amendment.get("patched_files", []):
        if not isinstance(entry, Mapping):
            errors.append("Invalid patched_files entry in amendment.")
            continue
        relative = str(entry.get("path", ""))
        expected = str(entry.get("sha256", ""))
        if not relative or len(expected) != 64:
            errors.append(f"Invalid amended file declaration: {entry}")
            continue
        declared[relative] = expected
        file_path = project_root / relative
        if not file_path.is_file():
            errors.append(f"Amended file is unavailable: {relative}")
        elif sha256_file(file_path) != expected:
            errors.append(f"Amended file checksum mismatch: {relative}")

    inventory_path = project_root / "exp123_release_freeze" / "source_inventory.csv"
    if not inventory_path.is_file():
        errors.append(f"Original source inventory is unavailable: {inventory_path}")
    else:
        with inventory_path.open("r", encoding="utf-8", newline="") as file:
            inventory = list(csv.DictReader(file))
        frozen_paths = {str(row["path"]) for row in inventory}
        for row in inventory:
            relative = str(row["path"])
            if relative in declared:
                continue
            current = project_root / relative
            if not current.is_file():
                errors.append(f"Unamended frozen source file is missing: {relative}")
                continue
            if sha256_file(current) != str(row["sha256"]):
                errors.append(
                    "Frozen source changed without an amendment declaration: "
                    f"{relative}"
                )
        modified_existing = sorted(set(declared).intersection(frozen_paths))
        if not modified_existing:
            errors.append("Amendment does not declare any modified frozen source files.")

    for name, dataset in dict(preflight.get("datasets", {})).items():
        dataset_path = Path(str(dataset.get("path", "")))
        if not dataset_path.is_file():
            errors.append(f"{name} dataset is unavailable: {dataset_path}")
            continue
        if dataset_path.stat().st_size != int(dataset.get("size_bytes", -1)):
            errors.append(f"{name} dataset size changed after preflight.")
            continue
        if verify_dataset_hashes and sha256_file(dataset_path) != str(dataset.get("sha256")):
            errors.append(f"{name} dataset checksum changed after preflight.")

    split_hash_path = run_dir / "common" / "splits" / "split_hash.txt"
    if not split_hash_path.is_file():
        errors.append("Shared split hash is unavailable for amended resume.")
    else:
        split_hash = split_hash_path.read_text(encoding="utf-8").strip()
        if split_hash != str(amendment.get("shared_split_hash")):
            errors.append("Shared split hash does not match the amendment record.")

    if errors:
        raise RuntimeError(
            "EXP123 amended-run verification failed:\n"
            + "\n".join(f"- {error}" for error in errors)
        )
    return {
        "status": "verified_with_amendment",
        "run_id": run_dir.name,
        "amendment_path": str(amendment_path),
        "amendment_sha256": sha256_file(amendment_path),
        "preflight_lock_sha256": actual_preflight_sha,
        "declared_amended_file_count": len(declared),
        "datasets_verified_by_hash": verify_dataset_hashes,
    }


def verify_run_execution_contract(
    *,
    project_root: Path,
    config_path: Path,
    run_dir: Path,
    verify_dataset_hashes: bool = False,
) -> dict[str, Any]:
    amendment_value = os.environ.get(AMENDMENT_ENV, "").strip()
    if amendment_value:
        return verify_amended_run_preflight(
            project_root=project_root,
            config_path=config_path,
            run_dir=run_dir,
            amendment_path=Path(amendment_value),
            verify_dataset_hashes=verify_dataset_hashes,
        )
    from .final_freeze import verify_run_preflight

    return verify_run_preflight(
        project_root=project_root,
        config_path=config_path,
        run_dir=run_dir,
        verify_dataset_hashes=verify_dataset_hashes,
    )
