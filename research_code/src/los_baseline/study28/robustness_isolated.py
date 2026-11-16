from __future__ import annotations

import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..evaluation_multiclass import calculate_multiclass_metrics
from ..utils import atomic_write_json, sha256_file
from .interpretability import run_primary_shap
from .resource_plan import child_thread_environment, read_int_env
from .robustness import (
    Alpha5Result,
    _all_nan_primary_features,
    _bootstrap_hfd,
    _fit_hfd_regressor,
    _hfd_metrics,
    _load_primary_predictions,
    _paired_bootstrap_macro_auc_difference,
    _predict_locked_sensitivity,
    _setting_frames,
    hfd28_from_final_dataset,
    run_home_discharge_guard,
    run_missingness_robustness,
    validate_alpha5_contract,
)

CLASS_NAMES = ("early_dc", "death", "prolonged")


def preicu_setting_row(*, contract, dataset_root: Path, alpha2_output_dir: Path, alpha3_output_dir: Path, setting: str) -> dict[str, Any]:
    cfg = contract.robustness_protocol.pre_icu_los_sensitivity
    base = tuple(contract.alpha4.feature_contract.no_apsiii)
    excluded = tuple(str(x) for x in cfg["excluded_features"])
    features = tuple(x for x in base if x not in excluded)
    train, evaluate = _setting_frames(contract, dataset_root, setting)
    y = evaluate["y"].to_numpy(dtype=np.int64)
    P_sens = _predict_locked_sensitivity(train=train, evaluate=evaluate, features=features, contract=contract)
    primary_y, P_primary = _load_primary_predictions(setting=setting, alpha2_output_dir=alpha2_output_dir, alpha3_output_dir=alpha3_output_dir)
    if not np.array_equal(y, primary_y):
        raise RuntimeError(f"Primary and no-pre-ICU labels are not aligned: {setting}")
    sens = calculate_multiclass_metrics(y, P_sens, CLASS_NAMES)
    pri = calculate_multiclass_metrics(y, P_primary, CLASS_NAMES)
    paired = _paired_bootstrap_macro_auc_difference(
        y, P_primary, P_sens, repeats=int(cfg["paired_bootstrap_repeats"]),
        confidence_level=float(cfg["confidence_level"]), seed=int(cfg["seed"]),
    )
    return {
        "setting": setting, "feature_set": "FS_NOAPS_NO_PREICU_LOS", "feature_count": len(features),
        "excluded_features": ";".join(excluded),
        "primary_macro_auroc": pri["macro_auroc"], "sensitivity_macro_auroc": sens["macro_auroc"],
        "delta_macro_auroc": float(sens["macro_auroc"] - pri["macro_auroc"]),
        "sensitivity_auc_early_dc": sens["class_0_auroc"], "sensitivity_auc_death": sens["class_1_auroc"],
        "sensitivity_auc_prolonged": sens["class_2_auroc"], **paired,
    }


def availability_site_row(*, contract, dataset_root: Path, alpha3_output_dir: Path, site: str) -> dict[str, Any]:
    cfg = contract.robustness_protocol.feature_availability_sensitivity
    mimic = contract.alpha4.registry["mimic"].load(dataset_root)
    frame = contract.alpha4.registry[site].load(dataset_root)
    primary_features = tuple(contract.alpha4.feature_contract.no_apsiii)
    dropped = _all_nan_primary_features(frame, primary_features)
    features = tuple(f for f in primary_features if f not in dropped)
    y = frame["y"].to_numpy(dtype=np.int64)
    P_sens = _predict_locked_sensitivity(train=mimic, evaluate=frame, features=features, contract=contract)
    primary_y, P_primary = _load_primary_predictions(setting=f"mimic->{site}", alpha2_output_dir=Path("."), alpha3_output_dir=alpha3_output_dir)
    if not np.array_equal(y, primary_y):
        raise RuntimeError(f"Primary and availability labels are not aligned: {site}")
    sens = calculate_multiclass_metrics(y, P_sens, CLASS_NAMES)
    pri = calculate_multiclass_metrics(y, P_primary, CLASS_NAMES)
    paired = _paired_bootstrap_macro_auc_difference(
        y, P_primary, P_sens, repeats=int(cfg["paired_bootstrap_repeats"]),
        confidence_level=float(cfg["confidence_level"]), seed=int(cfg["seed"]),
    )
    return {
        "site": site, "analysis_role": cfg.get("analysis_role", "exploratory"),
        "primary_feature_count": len(primary_features), "available_feature_count": len(features),
        "all_nan_primary_feature_count": len(dropped), "dropped_all_nan_features": ";".join(dropped),
        "primary_macro_auroc": pri["macro_auroc"], "availability_matched_macro_auroc": sens["macro_auroc"],
        "delta_macro_auroc": float(sens["macro_auroc"] - pri["macro_auroc"]),
        "availability_auc_early_dc": sens["class_0_auroc"], "availability_auc_death": sens["class_1_auroc"],
        "availability_auc_prolonged": sens["class_2_auroc"], **paired,
    }


def hfd_setting_row(*, contract, dataset_root: Path, setting: str) -> dict[str, Any]:
    cfg = contract.robustness_protocol.hfd28_secondary
    features = tuple(contract.alpha4.feature_contract.no_apsiii)
    train, evaluate = _setting_frames(contract, dataset_root, setting)
    pre, model = _fit_hfd_regressor(train, features, contract)
    pred = np.clip(model.predict(pre.transform(evaluate.loc[:, list(features)])), 0.0, 28.0)
    actual = hfd28_from_final_dataset(evaluate)
    point = _hfd_metrics(actual, pred)
    ci = _bootstrap_hfd(actual, pred, repeats=int(cfg["bootstrap_repeats"]), confidence_level=float(cfg["confidence_level"]), seed=int(cfg["seed"]))
    return {"setting": setting, "analysis_role": cfg.get("analysis_role", "secondary_outcome"), "feature_count": len(features), **point, **ci}



def hfd_professor_reference_row(*, contract, dataset_root: Path, setting: str) -> dict[str, Any]:
    cfg = dict(contract.robustness_protocol.hfd28_secondary["professor_reference"])
    features = tuple(contract.alpha4.feature_contract.columns)
    mimic = contract.alpha4.registry["mimic"].load(dataset_root)
    if setting == "mimic_temporal":
        train, evaluate = __import__("los_baseline.study28.development", fromlist=["split_mimic_temporal"]).split_mimic_temporal(mimic, contract.alpha4.protocol)
    elif setting.startswith("mimic->"):
        site = setting.split("->", 1)[1]
        train, evaluate = mimic, contract.alpha4.registry[site].load(dataset_root)
    elif setting == "eicu_alt->mimic":
        train, evaluate = contract.alpha4.registry["eicu_alt"].load(dataset_root), mimic
    else:
        raise ValueError(f"Unsupported professor HFD28 reference setting: {setting}")
    pre, model = _fit_hfd_regressor(train, features, contract)
    pred = np.clip(model.predict(pre.transform(evaluate.loc[:, list(features)])), 0.0, 28.0)
    actual = hfd28_from_final_dataset(evaluate)
    return {"setting": setting, "analysis_role": cfg.get("analysis_role", "professor_method_reproduction"), "feature_set":"FS_FULL", "feature_count":len(features), **_hfd_metrics(actual, pred)}

def _run_worker(*, mode: str, item: str, dataset_root: Path, project_root: Path, alpha2: Path, alpha3: Path, alpha4: Path, output: Path) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "result.json"
    if result_path.is_file():
        return json.loads(result_path.read_text(encoding="utf-8"))
    command = [
        sys.executable, "-m", "los_baseline.study28.robustness_worker",
        "--mode", mode, "--item", item,
        "--dataset-root", str(dataset_root), "--project-root", str(project_root),
        "--alpha2-output-dir", str(alpha2), "--alpha3-output-dir", str(alpha3),
        "--alpha4-output-dir", str(alpha4), "--output-dir", str(output),
    ]
    threads = read_int_env("STUDY28_ALPHA5_THREADS", 1)
    env = child_thread_environment(os.environ.copy(), threads)
    src = str(project_root / "src")
    current = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = src + (os.pathsep + current if current else "")
    with (output / "worker.log").open("w", encoding="utf-8") as log:
        completed = subprocess.run(command, cwd=project_root, env=env, stdout=log, stderr=subprocess.STDOUT, check=False, text=True)
    if completed.returncode != 0 or not result_path.is_file():
        tail = (output / "worker.log").read_text(encoding="utf-8", errors="replace")[-5000:]
        raise RuntimeError(f"Alpha 5 worker failed mode={mode} item={item}. Log tail:\n{tail}")
    return json.loads(result_path.read_text(encoding="utf-8"))


def run_alpha5_isolated(*, dataset_root: Path, project_root: Path, alpha2_output_dir: Path, alpha3_output_dir: Path, alpha4_output_dir: Path, output_dir: Path) -> Alpha5Result:
    dataset_root = Path(dataset_root).expanduser().resolve(); project_root = Path(project_root).expanduser().resolve()
    alpha2_output_dir = Path(alpha2_output_dir).expanduser().resolve(); alpha3_output_dir = Path(alpha3_output_dir).expanduser().resolve()
    alpha4_output_dir = Path(alpha4_output_dir).expanduser().resolve(); output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    contract = validate_alpha5_contract(
        dataset_root=dataset_root, project_root=project_root, alpha2_output_dir=alpha2_output_dir,
        alpha3_output_dir=alpha3_output_dir, alpha4_output_dir=alpha4_output_dir,
        audit_output_dir=output_dir / "00_dataset_audit",
    )
    max_workers = read_int_env("STUDY28_ALPHA5_WORKERS", 1)
    threads_per_worker = read_int_env("STUDY28_ALPHA5_THREADS", 1)
    print(
        f"[Study28 Alpha5] resource plan: concurrent_workers={max_workers}, "
        f"threads_per_worker={threads_per_worker}", flush=True,
    )

    def parallel_rows(mode: str, items: list[str], root: Path) -> list[dict[str, Any]]:
        if not items:
            return []
        def execute(item: str) -> tuple[str, dict[str, Any]]:
            print(f"[Study28 Alpha5] {mode} worker: {item}", flush=True)
            row = _run_worker(
                mode=mode, item=item, dataset_root=dataset_root, project_root=project_root,
                alpha2=alpha2_output_dir, alpha3=alpha3_output_dir, alpha4=alpha4_output_dir,
                output=root / item.replace("->", "_to_"),
            )
            return item, row
        resolved: dict[str, dict[str, Any]] = {}
        workers = min(max_workers, len(items))
        if workers <= 1:
            for item in items:
                key, row = execute(item); resolved[key] = row
        else:
            with ThreadPoolExecutor(max_workers=workers, thread_name_prefix=f"study28-a5-{mode}") as pool:
                futures = {pool.submit(execute, item): item for item in items}
                for future in as_completed(futures):
                    key, row = future.result(); resolved[key] = row
                    print(f"[Study28 Alpha5] checkpoint completed: {mode} {key}", flush=True)
        return [resolved[item] for item in items]

    preicu_items = [str(x) for x in contract.robustness_protocol.pre_icu_los_sensitivity["settings"]]
    preicu_rows = parallel_rows("preicu", preicu_items, output_dir / "02_pre_icu_los_sensitivity")
    preicu=pd.DataFrame(preicu_rows); preicu_path=output_dir/"02_pre_icu_los_sensitivity"/"pre_icu_los_removed_sensitivity.csv"; preicu.to_csv(preicu_path,index=False)

    availability_items = [str(x) for x in contract.robustness_protocol.feature_availability_sensitivity["sites"]]
    availability_rows = parallel_rows("availability", availability_items, output_dir / "03_feature_availability")
    availability=pd.DataFrame(availability_rows); availability_path=output_dir/"03_feature_availability"/"feature_availability_sensitivity.csv"; availability.to_csv(availability_path,index=False)

    hfd_items = [str(x) for x in contract.robustness_protocol.hfd28_secondary["settings"]]
    hfd_rows = parallel_rows("hfd", hfd_items, output_dir / "05_hfd28_secondary")
    hfd_models=pd.DataFrame(hfd_rows); hfd_model_path=output_dir/"05_hfd28_secondary"/"hfd28_model_performance.csv"; hfd_models.to_csv(hfd_model_path,index=False)

    hfd_ref_rows=[]
    href=dict(contract.robustness_protocol.hfd28_secondary.get("professor_reference", {}))
    hfd_ref_items = [str(x) for x in href.get("settings", [])] if href.get("enabled") else []
    if hfd_ref_items:
        hfd_ref_rows = parallel_rows("hfd_professor", hfd_ref_items, output_dir / "05_hfd28_secondary" / "professor_reference")
    hfd_ref=pd.DataFrame(hfd_ref_rows)
    hfd_ref_path=output_dir/"05_hfd28_secondary"/"hfd28_professor_reference.csv"
    if len(hfd_ref): hfd_ref.to_csv(hfd_ref_path,index=False)

    # Quick, non-LightGBM analyses run after isolated model workers.
    print("[Study28 Alpha5] primary FS_NOAPS SHAP", flush=True)
    mimic = contract.alpha4.registry["mimic"].load(dataset_root)
    shap_result = run_primary_shap(
        model=contract.alpha4.frozen.model, preprocessor=contract.alpha4.frozen.preprocessor,
        mimic=mimic, feature_names=contract.alpha4.frozen.feature_names,
        output_dir=output_dir/"01_interpretability", sample_size=int(contract.robustness_protocol.interpretability["sample_size"]),
        seed=int(contract.robustness_protocol.interpretability["seed"]), beeswarm_class=str(contract.robustness_protocol.interpretability["beeswarm_class"]),
        max_display=int(contract.robustness_protocol.interpretability["max_display"]),
    )
    print("[Study28 Alpha5] observed missingness robustness", flush=True)
    missingness = run_missingness_robustness(contract=contract, dataset_root=dataset_root, alpha2_output_dir=alpha2_output_dir, alpha3_output_dir=alpha3_output_dir, output_dir=output_dir/"04_missingness_robustness")

    summary_rows=[]
    for cohort in ["mimic","eicu_primary","eicu_alt","sicdb","nwicu"]:
        frame=contract.alpha4.registry[cohort].load(dataset_root); hfd=hfd28_from_final_dataset(frame); q=np.quantile(hfd,[.25,.5,.75])
        summary_rows.append({"cohort":cohort,"n":len(frame),"mean":float(hfd.mean()),"q25":float(q[0]),"median":float(q[1]),"q75":float(q[2]),"pct_zero":float((hfd==0).mean())})
    hfd_summary=pd.DataFrame(summary_rows); hfd_summary_path=output_dir/"05_hfd28_secondary"/"hfd28_summary.csv"; hfd_summary.to_csv(hfd_summary_path,index=False)
    home=run_home_discharge_guard(contract=contract,dataset_root=dataset_root,output_dir=output_dir/"06_home_discharge_guard")

    manifest={
      "schema_version":"study28_alpha5_robustness_interpretability_v1","study_id":contract.alpha4.protocol.study_id,
      "frozen_primary_model_sha256":contract.alpha4.frozen.model_sha256,
      "alpha3_manifest_sha256":sha256_file(contract.alpha4.alpha3_manifest_path),"alpha4_manifest_sha256":sha256_file(contract.alpha4_manifest_path),
      "robustness_protocol_sha256":contract.robustness_protocol.sha256,"primary_model_changed":False,"primary_external_results_overwritten":False,
      "new_models_are_sensitivity_only":True,"execution_mode":"isolated_setting_workers_with_resource_planned_parallelism","resource_plan_path":os.environ.get("STUDY28_RESOURCE_PLAN_PATH"),
      "concurrent_workers":max_workers,"threads_per_worker":threads_per_worker,"shap":shap_result,
      "pre_icu_los_sensitivity_rows":len(preicu),"feature_availability_rows":len(availability),"missingness_rows":len(missingness),
      "hfd28_cohort_rows":len(hfd_summary),"hfd28_model_rows":len(hfd_models),"hfd28_professor_reference_rows":len(hfd_ref),"hfd28_professor_reference_csv":str(hfd_ref_path) if len(hfd_ref) else None,"home_discharge_sensitivity":home,
      "methodological_notes":contract.robustness_protocol.methodological_notes,
    }
    manifest_path=output_dir/"study28_alpha5_robustness_interpretability_manifest.json"; atomic_write_json(manifest_path,manifest)
    return Alpha5Result(
      manifest_path=str(manifest_path), shap_mean_abs_csv=shap_result["mean_abs_csv"], pre_icu_los_summary_csv=str(preicu_path),
      availability_summary_csv=str(availability_path), missingness_summary_csv=str(output_dir/"04_missingness_robustness"/"missingness_burden_performance.csv"),
      hfd28_summary_csv=str(hfd_summary_path), hfd28_model_csv=str(hfd_model_path), home_discharge_status_json=str(output_dir/"06_home_discharge_guard"/"home_discharge_sensitivity_status.json")
    )

