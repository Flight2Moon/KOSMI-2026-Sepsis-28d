from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

from .task_spec import STAGE2_LONG_STAY_TASK
from .utils import atomic_write_json, environment_info, sha256_file


@dataclass(frozen=True)
class Exp3ArtifactReferenceResult:
    run_id: str
    output_dir: str
    source_stage2_dir: str
    shared_split_hash: str
    model_names: tuple[str, ...]
    duplicate_training: bool
    confirmatory_splits_accessed: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required JSON file was not found: {path}")
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a JSON object in {path}")
    return payload


def _relative_file_hashes(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def create_exp3_artifact_reference(
    *,
    run_id: str,
    run_dir: Path,
    model_names: Sequence[str],
    shared_split_hash: str,
) -> Exp3ArtifactReferenceResult:
    """Create EXP3 references to Stage2 without duplicating model training."""
    run_dir = run_dir.expanduser().resolve()
    stage2_dir = run_dir / "EXP2_two_stage" / "stage2_long_stay"
    if not stage2_dir.is_dir():
        raise FileNotFoundError(f"Stage2 output directory was not found: {stage2_dir}")

    stage2_manifest_path = stage2_dir / "stage2_manifest.json"
    stage2_manifest = _load_json(stage2_manifest_path)
    if stage2_manifest.get("task", {}).get("task_name") != STAGE2_LONG_STAY_TASK.task_name:
        raise ValueError("EXP3 can only reference the Stage2 long-stay task.")
    data_access = stage2_manifest.get("data_access", {})
    if data_access.get("internal_test") or data_access.get("external_eicu"):
        raise ValueError(
            "EXP3 discovery reference cannot be created from Stage2 outputs that "
            "accessed confirmatory splits."
        )

    frozen_path = stage2_dir / "frozen_stage2_design.json"
    frozen_payload = _load_json(frozen_path)
    if frozen_payload.get("shared_split_hash") != shared_split_hash:
        raise ValueError("EXP3 and Stage2 shared split hashes do not match.")

    model_references: dict[str, dict[str, Any]] = {}
    for model_name in model_names:
        model_dir = stage2_dir / model_name
        artifact_dir = model_dir / "artifact"
        if not artifact_dir.is_dir():
            raise FileNotFoundError(f"Stage2 artifact directory was not found: {artifact_dir}")
        engine_result_path = model_dir / "engine_result.json"
        artifact_integrity_path = model_dir / "artifact_integrity.json"
        selection_path = (
            run_dir
            / "optimization"
            / STAGE2_LONG_STAY_TASK.task_name
            / model_name
            / "selected_hyperparameters.json"
        )
        required = [engine_result_path, artifact_integrity_path, selection_path]
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                "Stage2 reference inputs are incomplete: " + ", ".join(missing)
            )
        model_references[model_name] = {
            "source_model_dir": str(model_dir.relative_to(run_dir)),
            "artifact_dir": str(artifact_dir.relative_to(run_dir)),
            "artifact_file_sha256": _relative_file_hashes(artifact_dir),
            "engine_result_path": str(engine_result_path.relative_to(run_dir)),
            "engine_result_sha256": sha256_file(engine_result_path),
            "artifact_integrity_path": str(
                artifact_integrity_path.relative_to(run_dir)
            ),
            "artifact_integrity_sha256": sha256_file(artifact_integrity_path),
            "selected_hyperparameters_path": str(selection_path.relative_to(run_dir)),
            "selected_hyperparameters_sha256": sha256_file(selection_path),
        }

    output_dir = run_dir / "EXP3_survivor_binary"
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"EXP3 reference output already exists: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    reference_payload = {
        "experiment": "EXP3_survivor_binary",
        "task": STAGE2_LONG_STAY_TASK.to_dict(),
        "population": "ground_truth_survivors_A_or_C",
        "source_experiment": "EXP2_two_stage",
        "source_stage": "stage2_long_stay",
        "source_stage2_dir": str(stage2_dir.relative_to(run_dir)),
        "shared_split_hash": shared_split_hash,
        "model_names": list(model_names),
        "model_references": model_references,
        "reuse_stage2_artifacts": True,
        "duplicate_training": False,
        "duplicate_artifact_storage": False,
        "oracle_interpretation": (
            "EXP3 reports Stage2 performance as the survivor-only oracle track."
        ),
        "confirmatory_splits_accessed": False,
        "stage2_frozen_design_path": str(frozen_path.relative_to(run_dir)),
        "stage2_frozen_design_sha256": sha256_file(frozen_path),
        "stage2_manifest_path": str(stage2_manifest_path.relative_to(run_dir)),
        "stage2_manifest_sha256": sha256_file(stage2_manifest_path),
    }
    reference_path = output_dir / "artifact_reference.json"
    atomic_write_json(reference_path, reference_payload)
    (output_dir / "artifact_reference.json.sha256").write_text(
        sha256_file(reference_path) + "\n", encoding="utf-8"
    )

    manifest = {
        "step": "exp3_survivor_binary_reference",
        "run_id": run_id,
        "status": "reference_created",
        "training_performed": False,
        "source_stage2_dir": str(stage2_dir),
        "artifact_reference_path": str(reference_path),
        "data_access": {
            "train": False,
            "validation": False,
            "internal_test": False,
            "external_eicu": False,
        },
        "environment": environment_info(),
        "next_required_step": "stage_heterogeneity",
    }
    atomic_write_json(output_dir / "exp3_manifest.json", manifest)

    result = Exp3ArtifactReferenceResult(
        run_id=run_id,
        output_dir=str(output_dir),
        source_stage2_dir=str(stage2_dir),
        shared_split_hash=shared_split_hash,
        model_names=tuple(model_names),
        duplicate_training=False,
        confirmatory_splits_accessed=False,
    )
    atomic_write_json(output_dir / "exp3_result.json", result.to_dict())
    return result
