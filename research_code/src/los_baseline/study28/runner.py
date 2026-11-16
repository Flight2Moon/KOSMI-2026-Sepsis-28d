from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from ..utils import atomic_write_json
from .audit import audit_study_datasets
from .contract import load_feature_contract
from .datasets import load_dataset_registry
from .development import freeze_external_model, run_nested_cv_isolated, run_temporal_evaluation
from .feature_sets import resolve_feature_set
from .protocol import load_study28_protocol


def run_alpha2(
    *,
    dataset_root: Path,
    project_root: Path,
    output_dir: Path,
) -> dict[str, Any]:
    project_root = Path(project_root).expanduser().resolve()
    dataset_root = Path(dataset_root).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    protocol_path = project_root / "configs" / "study28_final.json"
    registry_path = project_root / "configs" / "dataset_registry_28d.json"
    feature_sets_path = project_root / "configs" / "feature_sets.json"
    feature_list_path = dataset_root / "feature_list.csv"

    print("[Study28 Alpha2] phase 0/3: frozen dataset audit", flush=True)
    audit = audit_study_datasets(
        dataset_root=dataset_root,
        registry_path=registry_path,
        feature_list_path=feature_list_path,
        output_dir=output_dir / "00_dataset_audit",
    )
    if not audit.passed:
        raise RuntimeError(
            "Study28 dataset audit failed; Alpha 2 training is blocked: "
            + "; ".join(audit.blocking_reasons)
        )

    protocol = load_study28_protocol(protocol_path)
    registry = load_dataset_registry(registry_path)
    feature_contract = load_feature_contract(feature_list_path)
    if feature_contract.sha256 != json.loads(protocol_path.read_text(encoding="utf-8"))["data_provenance"]["feature_list_sha256"]:
        raise RuntimeError("feature_list.csv SHA-256 does not match the frozen Study28 protocol.")

    mimic_spec = registry["mimic"]
    mimic = mimic_spec.load(dataset_root)
    selection_features = resolve_feature_set(
        protocol.nested_cv.selection_feature_set,
        feature_sets_path=feature_sets_path,
        contract=feature_contract,
    )
    primary_features = resolve_feature_set(
        protocol.primary_feature_set,
        feature_sets_path=feature_sets_path,
        contract=feature_contract,
    )

    print("[Study28 Alpha2] phase 1/3: professor 5x3 LightGBM nested CV", flush=True)
    nested = run_nested_cv_isolated(
        mimic=mimic,
        feature_names=selection_features,
        protocol=protocol,
        dataset_root=dataset_root,
        project_root=project_root,
        output_dir=output_dir / "01_nested_cv",
    )
    print("[Study28 Alpha2] phase 2/3: MIMIC temporal evaluation", flush=True)
    temporal = run_temporal_evaluation(
        mimic=mimic,
        feature_names=primary_features,
        protocol=protocol,
        locked_search_parameters=nested.locked_search_parameters,
        output_dir=output_dir / "02_temporal_evaluation",
    )
    print("[Study28 Alpha2] phase 3/3: freeze full-MIMIC primary model", flush=True)
    freeze = freeze_external_model(
        mimic=mimic,
        feature_names=primary_features,
        feature_contract=feature_contract,
        protocol=protocol,
        locked_search_parameters=nested.locked_search_parameters,
        mimic_sha256=mimic_spec.source_sha256(dataset_root),
        output_dir=output_dir / "03_external_model_freeze",
    )

    result = {
        "phase": "V9_ALPHA2",
        "dataset_audit_passed": audit.passed,
        "nested_cv": nested.to_dict(),
        "temporal_evaluation": temporal.to_dict(),
        "model_freeze": freeze.to_dict(),
        "external_validation_performed": False,
        "next_phase_required_external_sites": list(protocol.required_external_sites),
    }
    atomic_write_json(output_dir / "study28_alpha2_summary.json", result)
    return result
