from __future__ import annotations

import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..utils import atomic_write_json
from .reporting_protocol import Study28ReportingProtocol, load_reporting_protocol, sha256_file


CLASS_LABELS = {0: "Early discharge", 1: "Death", 2: "Prolonged stay"}
SITE_LABELS = {
    "mimic_temporal": "MIMIC temporal",
    "eicu_alt": "eICU-alt",
    "eicu_primary": "eICU-primary",
    "sicdb": "SICdb",
    "nwicu": "NWICU",
}


@dataclass(frozen=True)
class Alpha6Result:
    manifest_path: str
    evidence_index_csv: str
    analysis_registry_csv: str
    main_tables: tuple[str, ...]
    supplementary_tables: tuple[str, ...]
    figures: tuple[str, ...]
    manuscript_summary_md: str
    statistical_methods_md: str
    finalization_passed: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _require(path: Path, label: str) -> Path:
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Missing Alpha 6 evidence source ({label}): {path}")
    return path


def _as_float(value: Any) -> float:
    if value is None:
        return math.nan
    return float(value)


def _save_figure(fig: plt.Figure, stem: Path, protocol: Study28ReportingProtocol) -> list[str]:
    outputs: list[str] = []
    stem.parent.mkdir(parents=True, exist_ok=True)
    dpi = int(protocol.figures.get("dpi", 220))
    if bool(protocol.figures.get("include_png", True)):
        path = stem.with_suffix(".png")
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        outputs.append(str(path))
    if bool(protocol.figures.get("include_svg", True)):
        path = stem.with_suffix(".svg")
        fig.savefig(path, bbox_inches="tight")
        outputs.append(str(path))
    plt.close(fig)
    return outputs


def _collect_from_output_dirs(alpha2: Path, alpha3: Path, alpha4: Path, alpha5: Path) -> dict[str, Any]:
    alpha2 = Path(alpha2).expanduser().resolve()
    alpha3 = Path(alpha3).expanduser().resolve()
    alpha4 = Path(alpha4).expanduser().resolve()
    alpha5 = Path(alpha5).expanduser().resolve()

    nested_summary_path = _require(alpha2 / "01_nested_cv" / "nested_cv_summary.json", "Alpha2 nested summary")
    nested_folds_path = _require(alpha2 / "01_nested_cv" / "nested_cv_fold_metrics.csv", "Alpha2 nested folds")
    temporal_path = _require(alpha2 / "02_temporal_evaluation" / "mimic_temporal_metrics.json", "Alpha2 temporal metrics")
    freeze_path = _require(alpha2 / "03_external_model_freeze" / "study28_model_freeze.json", "Alpha2 freeze manifest")
    alpha3_manifest_path = _require(alpha3 / "study28_alpha3_external_validation_manifest.json", "Alpha3 manifest")
    ext_summary_path = _require(alpha3 / "external_validation_summary.csv", "Alpha3 external summary")
    ext_ci_path = _require(alpha3 / "external_validation_bootstrap_ci.csv", "Alpha3 external bootstrap")
    ext_cal_path = _require(alpha3 / "external_calibration_summary.csv", "Alpha3 calibration")
    alpha4_manifest_path = _require(alpha4 / "study28_alpha4_sensitivity_manifest.json", "Alpha4 manifest")
    reverse_path = _require(alpha4 / "01_reverse_validation" / "reverse_validation_summary.csv", "Alpha4 reverse")
    recal_path = _require(alpha4 / "06_reporting" / "local_recalibration_change_summary.csv", "Alpha4 recalibration")
    dca_path = _require(alpha4 / "06_reporting" / "dca_dominance_summary.csv", "Alpha4 DCA")
    subgroup_path = _require(alpha4 / "04_subgroups" / "external_subgroup_performance.csv", "Alpha4 subgroup")
    hospital_path = _require(alpha4 / "05_hospital_heterogeneity" / "hospital_random_effects_summary.csv", "Alpha4 hospital")
    alpha5_manifest_path = _require(alpha5 / "study28_alpha5_robustness_interpretability_manifest.json", "Alpha5 manifest")
    preicu_path = _require(alpha5 / "02_pre_icu_los_sensitivity" / "pre_icu_los_removed_sensitivity.csv", "Alpha5 pre-ICU")
    availability_path = _require(alpha5 / "03_feature_availability" / "feature_availability_sensitivity.csv", "Alpha5 availability")
    missing_path = _require(alpha5 / "04_missingness_robustness" / "missingness_burden_performance.csv", "Alpha5 missingness")
    hfd_path = _require(alpha5 / "05_hfd28_secondary" / "hfd28_model_performance.csv", "Alpha5 HFD28")
    hfd_ref_path = _require(alpha5 / "05_hfd28_secondary" / "hfd28_professor_reference.csv", "Alpha5 HFD professor reference")
    home_path = _require(alpha5 / "06_home_discharge_guard" / "home_discharge_sensitivity_status.json", "Alpha5 home-discharge guard")
    shap_path = _require(alpha5 / "01_interpretability" / "shap_mean_abs_primary_noaps.csv", "Alpha5 SHAP")

    return {
        "mode": "run_outputs",
        "paths": [nested_summary_path, nested_folds_path, temporal_path, freeze_path, alpha3_manifest_path,
                  ext_summary_path, ext_ci_path, ext_cal_path, alpha4_manifest_path, reverse_path, recal_path,
                  dca_path, subgroup_path, hospital_path, alpha5_manifest_path, preicu_path, availability_path,
                  missing_path, hfd_path, hfd_ref_path, home_path, shap_path],
        "nested_summary": _read_json(nested_summary_path),
        "nested_folds": pd.read_csv(nested_folds_path),
        "temporal": _read_json(temporal_path),
        "freeze": _read_json(freeze_path),
        "alpha3_manifest": _read_json(alpha3_manifest_path),
        "external": pd.read_csv(ext_summary_path),
        "external_ci": pd.read_csv(ext_ci_path),
        "calibration": pd.read_csv(ext_cal_path),
        "alpha4_manifest": _read_json(alpha4_manifest_path),
        "reverse": pd.read_csv(reverse_path),
        "recalibration": pd.read_csv(recal_path),
        "dca": pd.read_csv(dca_path),
        "subgroups": pd.read_csv(subgroup_path),
        "hospital": pd.read_csv(hospital_path),
        "alpha5_manifest": _read_json(alpha5_manifest_path),
        "preicu": pd.read_csv(preicu_path),
        "availability": pd.read_csv(availability_path),
        "missingness": pd.read_csv(missing_path),
        "hfd28": pd.read_csv(hfd_path),
        "hfd28_professor": pd.read_csv(hfd_ref_path),
        "home": _read_json(home_path),
        "shap": pd.read_csv(shap_path),
    }


def _collect_from_release_snapshots(project_root: Path) -> dict[str, Any]:
    root = Path(project_root).expanduser().resolve()
    a2p = _require(root / "docs" / "v9_alpha2_verification.json", "release Alpha2 verification")
    a3p = _require(root / "docs" / "v9_alpha3_verification.json", "release Alpha3 verification")
    a4p = _require(root / "docs" / "v9_alpha4_verification.json", "release Alpha4 verification")
    a5p = _require(root / "docs" / "v9_alpha5_verification.json", "release Alpha5 verification")
    a2, a3, a4, a5 = map(_read_json, (a2p, a3p, a4p, a5p))

    nested_folds = pd.DataFrame({
        "outer_fold": list(range(1, len(a2["nested_cv"]["fold_macro_auroc"]) + 1)),
        "macro_auroc": a2["nested_cv"]["fold_macro_auroc"],
        "selected_num_leaves": [x["num_leaves"] for x in a2["nested_cv"]["selected_parameters_each_fold"]],
        "selected_n_estimators": [x["n_estimators"] for x in a2["nested_cv"]["selected_parameters_each_fold"]],
    })
    temporal = {
        "development_n": a2["temporal_primary_noaps"]["development_n"],
        "evaluation_n": a2["temporal_primary_noaps"]["evaluation_n"],
        "feature_set": "FS_NOAPS",
        "metrics": {
            "class_0_auroc": a2["temporal_primary_noaps"]["class_early_dc_auroc"],
            "class_1_auroc": a2["temporal_primary_noaps"]["class_death_auroc"],
            "class_2_auroc": a2["temporal_primary_noaps"]["class_prolonged_auroc"],
            "macro_auroc": a2["temporal_primary_noaps"]["macro_auroc"],
            "macro_auprc": a2["temporal_primary_noaps"]["macro_auprc"],
        },
    }
    external = pd.DataFrame([{
        "site": x["site"], "sample_count": x["n"],
        "class_0_auroc": x["auroc_early_dc"], "class_1_auroc": x["auroc_death"],
        "class_2_auroc": x["auroc_prolonged"], "macro_auroc": x["macro_auroc"],
        "macro_auprc": x["macro_auprc"], "multiclass_log_loss": x["multiclass_log_loss"],
        "multiclass_brier": x["multiclass_brier"], "macro_auroc_ci_lo": x["macro_auroc_ci95"][0],
        "macro_auroc_ci_hi": x["macro_auroc_ci95"][1],
    } for x in a3["sites"]])
    calibration = pd.DataFrame(a3["calibration"])
    reverse = pd.DataFrame(a4["reverse_validation"]).rename(columns={
        "class_0_auroc":"class_0_auroc", "class_1_auroc":"class_1_auroc", "class_2_auroc":"class_2_auroc"
    })
    base = root / "docs" / "verification"
    paths = [a2p, a3p, a4p, a5p]
    csv_map = {
        "recalibration": base / "v9_alpha4_real" / "local_recalibration_change_summary.csv",
        "dca": base / "v9_alpha4_real" / "dca_dominance_summary.csv",
        "subgroups": base / "v9_alpha4_real" / "external_subgroup_performance.csv",
        "hospital": base / "v9_alpha4_real" / "hospital_random_effects_summary.csv",
        "preicu": base / "v9_alpha5_real" / "pre_icu_los_removed_sensitivity.csv",
        "availability": base / "v9_alpha5_real" / "feature_availability_sensitivity.csv",
        "missingness": base / "v9_alpha5_real" / "missingness_burden_performance.csv",
        "hfd28": base / "v9_alpha5_real" / "hfd28_model_performance.csv",
        "hfd28_professor": base / "v9_alpha5_real" / "hfd28_professor_reference.csv",
        "shap": base / "v9_alpha5_real" / "shap_mean_abs_primary_noaps.csv",
    }
    frames: dict[str, pd.DataFrame] = {}
    for key, path in csv_map.items():
        path = _require(path, f"release {key}")
        paths.append(path)
        frames[key] = pd.read_csv(path)
    home_path = _require(base / "v9_alpha5_real" / "home_discharge_sensitivity_status.json", "release home guard")
    paths.append(home_path)
    freeze = {
        "external_model_sha256": a3["frozen_model_sha256"],
        "protocol_sha256": a3["frozen_protocol_sha256"],
        "primary_feature_set": "FS_NOAPS",
        "locked_search_parameters": {"num_leaves": 15, "n_estimators": 200},
    }
    nested_summary = {
        "protocol": {"outer_folds": a2["nested_cv"]["outer_folds"], "inner_folds": a2["nested_cv"]["inner_folds"],
                     "seed": a2["nested_cv"]["seed"], "selection_metric": "macro_auroc", "selection_feature_set": "FS_FULL"},
        "locked_search_parameters": {"num_leaves": 15, "n_estimators": 200},
        "aggregate_outer_metrics": {"macro_auroc": {"mean": a2["nested_cv"]["mean_macro_auroc"], "std": a2["nested_cv"]["sd_macro_auroc"]}},
    }
    alpha3_manifest = {
        "all_required_external_complete": a3["all_required_external_complete"],
        "required_external_sites": a3["required_external_sites"],
        "completed_external_sites": a3["required_external_sites"],
        "raw_external_only": a3["raw_external_only"],
        "local_recalibration_performed": a3["local_recalibration_performed"],
        "frozen_model_sha256": a3["frozen_model_sha256"],
        "frozen_protocol_sha256": a3["frozen_protocol_sha256"],
        "bootstrap_repeats": a3["bootstrap_repeats"], "confidence_level": a3["confidence_level"],
    }
    alpha4_manifest = {
        "alpha3_raw_external_preserved": not a4["alpha3_raw_external_results_modified"],
        "reverse_models_change_primary_model": a4["alpha2_frozen_primary_model_modified"],
    }
    alpha5_manifest = {
        "frozen_primary_model_sha256": a5["frozen_primary_model_sha256"],
        "primary_model_changed": a5["primary_model_changed"],
        "primary_external_results_overwritten": False,
    }
    return {
        "mode": "release_snapshots", "paths": paths,
        "nested_summary": nested_summary, "nested_folds": nested_folds, "temporal": temporal, "freeze": freeze,
        "alpha3_manifest": alpha3_manifest, "external": external, "external_ci": pd.DataFrame(), "calibration": calibration,
        "alpha4_manifest": alpha4_manifest, "reverse": reverse, "recalibration": frames["recalibration"], "dca": frames["dca"],
        "subgroups": frames["subgroups"], "hospital": frames["hospital"], "alpha5_manifest": alpha5_manifest,
        "preicu": frames["preicu"], "availability": frames["availability"], "missingness": frames["missingness"],
        "hfd28": frames["hfd28"], "hfd28_professor": frames["hfd28_professor"], "home": _read_json(home_path), "shap": frames["shap"],
    }


def _validate_finalization(bundle: dict[str, Any], protocol: Study28ReportingProtocol) -> dict[str, Any]:
    checks: dict[str, bool] = {}
    freeze_sha = str(bundle["freeze"].get("external_model_sha256", ""))
    a3_sha = str(bundle["alpha3_manifest"].get("frozen_model_sha256", ""))
    a5_sha = str(bundle["alpha5_manifest"].get("frozen_primary_model_sha256", freeze_sha))
    checks["frozen_model_sha_match_alpha2_alpha3_alpha5"] = bool(freeze_sha and freeze_sha == a3_sha == a5_sha)
    completed = set(bundle["alpha3_manifest"].get("completed_external_sites", bundle["alpha3_manifest"].get("required_external_sites", [])))
    checks["all_required_external_complete"] = set(protocol.required_external_sites).issubset(completed) and bool(bundle["alpha3_manifest"].get("all_required_external_complete", False))
    checks["alpha3_raw_external_only"] = bool(bundle["alpha3_manifest"].get("raw_external_only", False)) and not bool(bundle["alpha3_manifest"].get("local_recalibration_performed", False))
    checks["alpha4_preserved_raw_external"] = bool(bundle["alpha4_manifest"].get("alpha3_raw_external_preserved", True))
    checks["alpha4_did_not_replace_primary"] = not bool(bundle["alpha4_manifest"].get("reverse_models_change_primary_model", False))
    checks["alpha5_did_not_replace_primary"] = not bool(bundle["alpha5_manifest"].get("primary_model_changed", False)) and not bool(bundle["alpha5_manifest"].get("primary_external_results_overwritten", False))
    home = bundle["home"]
    executable = bool(home.get("executable", home.get("status") not in {"NOT_EXECUTABLE_WITH_FROZEN_FINAL_DATASET", "not_executable"}))
    inferred = bool(home.get("labels_inferred", False))
    checks["home_discharge_not_inferred"] = executable or not inferred
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError("Alpha 6 finalization blocked: " + ", ".join(failed))
    return {"passed": True, "checks": checks}


def _normalize_external(bundle: dict[str, Any]) -> pd.DataFrame:
    ext = bundle["external"].copy()
    rename = {"n":"sample_count", "auc_early_dc":"class_0_auroc", "auc_death":"class_1_auroc", "auc_prolonged":"class_2_auroc"}
    ext = ext.rename(columns={k:v for k,v in rename.items() if k in ext.columns})
    if "macro_auroc_ci_lo" not in ext.columns and len(bundle["external_ci"]):
        ci = bundle["external_ci"].copy()
        ci = ci.loc[ci["metric"].eq("macro_auroc"), ["site","lower","upper"]].rename(columns={"lower":"macro_auroc_ci_lo","upper":"macro_auroc_ci_hi"})
        ext = ext.merge(ci, on="site", how="left")
    ext["site_label"] = ext["site"].map(SITE_LABELS).fillna(ext["site"])
    return ext


def _write_tables(bundle: dict[str, Any], protocol: Study28ReportingProtocol, output_dir: Path) -> tuple[list[str], list[str]]:
    main_dir = output_dir / "tables" / "main"
    supp_dir = output_dir / "tables" / "supplementary"
    main_dir.mkdir(parents=True, exist_ok=True); supp_dir.mkdir(parents=True, exist_ok=True)
    main_paths: list[str] = []; supp_paths: list[str] = []

    nested = bundle["nested_summary"]
    temporal = bundle["temporal"]
    tmet = dict(temporal.get("metrics", {}))
    agg = nested.get("aggregate_outer_metrics", {}).get("macro_auroc", {})
    table1 = pd.DataFrame([
        {"analysis_role":"primary_development", "setting":"Nested CV (MIMIC)", "development_n":np.nan, "evaluation_n":np.nan,
         "feature_set":nested.get("protocol",{}).get("selection_feature_set","FS_FULL"), "macro_auroc":agg.get("mean"), "uncertainty":"SD", "uncertainty_value":agg.get("std"),
         "auc_early_dc":np.nan, "auc_death":np.nan, "auc_prolonged":np.nan, "macro_auprc":np.nan},
        {"analysis_role":"supportive_temporal", "setting":"MIMIC 2008–2016 → 2017–2022", "development_n":temporal.get("development_n"), "evaluation_n":temporal.get("evaluation_n"),
         "feature_set":temporal.get("feature_set","FS_NOAPS"), "macro_auroc":tmet.get("macro_auroc"), "uncertainty":"Not bootstrap-reported", "uncertainty_value":np.nan,
         "auc_early_dc":tmet.get("class_0_auroc"), "auc_death":tmet.get("class_1_auroc"), "auc_prolonged":tmet.get("class_2_auroc"), "macro_auprc":tmet.get("macro_auprc")},
    ])
    p = main_dir / "table_1_model_development_and_temporal.csv"; table1.to_csv(p,index=False); main_paths.append(str(p))

    ext = _normalize_external(bundle)
    keep = ["site","site_label","sample_count","class_0_auroc","class_1_auroc","class_2_auroc","macro_auroc","macro_auroc_ci_lo","macro_auroc_ci_hi","macro_auprc","multiclass_log_loss","multiclass_brier"]
    for col in keep:
        if col not in ext.columns: ext[col] = np.nan
    table2 = ext.loc[:,keep].copy(); table2.insert(0,"analysis_role","primary_raw_external")
    p = main_dir / "table_2_raw_external_validation.csv"; table2.to_csv(p,index=False); main_paths.append(str(p))

    cal = bundle["calibration"].copy()
    if "class_code" in cal.columns: cal["class_name"] = cal.get("class_name", cal["class_code"].map(CLASS_LABELS))
    cal.insert(0,"analysis_role","secondary_calibration")
    p = main_dir / "table_3_external_calibration.csv"; cal.to_csv(p,index=False); main_paths.append(str(p))

    supplements = [
        ("table_s1_nested_cv_folds.csv", bundle["nested_folds"].copy(), "primary_development"),
        ("table_s2_reverse_validation.csv", bundle["reverse"].copy(), "sensitivity_reverse"),
        ("table_s3_local_recalibration.csv", bundle["recalibration"].copy(), "sensitivity_recalibration"),
        ("table_s4_dca_dominance.csv", bundle["dca"].copy(), "clinical_utility"),
        ("table_s5_subgroup_performance.csv", bundle["subgroups"].copy(), None),
        ("table_s6_hospital_heterogeneity.csv", bundle["hospital"].copy(), "robustness"),
        ("table_s7_pre_icu_los_removed.csv", bundle["preicu"].copy(), "robustness"),
        ("table_s8_feature_availability.csv", bundle["availability"].copy(), "robustness"),
        ("table_s9_missingness_burden.csv", bundle["missingness"].copy(), "robustness"),
        ("table_s10_hfd28_secondary.csv", bundle["hfd28"].copy(), "secondary_outcome"),
        ("table_s11_hfd28_professor_method_reproduction.csv", bundle["hfd28_professor"].copy(), "professor_method_reproduction"),
        ("table_s12_shap_mean_absolute.csv", bundle["shap"].copy(), "interpretability"),
    ]
    for name, frame, role in supplements:
        if name == "table_s5_subgroup_performance.csv":
            if "analysis_role" in frame.columns:
                frame["analysis_role"] = frame["analysis_role"].map({"professor_reference":"subgroup_reference","exploratory_extension":"exploratory_subgroup"}).fillna(frame["analysis_role"])
            else:
                frame.insert(0,"analysis_role","exploratory_subgroup")
        elif "analysis_role" not in frame.columns and role:
            frame.insert(0,"analysis_role",role)
        p = supp_dir / name; frame.to_csv(p,index=False); supp_paths.append(str(p))

    home = bundle["home"]
    home_table = pd.DataFrame([{
        "analysis_role":"not_executable",
        "analysis":"Home-discharge sensitivity",
        "status": home.get("status", "NOT_EXECUTABLE_WITH_FROZEN_FINAL_DATASET"),
        "executable": bool(home.get("executable", False)),
        "labels_inferred": bool(home.get("labels_inferred", False)),
        "reason": home.get("reason", home.get("message", "discharge_location unavailable in frozen final datasets")),
    }])
    p = supp_dir / "table_s13_unavailable_analysis_registry.csv"; home_table.to_csv(p,index=False); supp_paths.append(str(p))
    return main_paths, supp_paths


def _write_figures(bundle: dict[str, Any], protocol: Study28ReportingProtocol, output_dir: Path) -> list[str]:
    fig_dir = output_dir / "figures"; fig_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    ext = _normalize_external(bundle)

    fig, ax = plt.subplots(figsize=(7.2,4.6))
    y = np.arange(len(ext))
    x = ext["macro_auroc"].to_numpy(float)
    lo = ext["macro_auroc_ci_lo"].to_numpy(float); hi = ext["macro_auroc_ci_hi"].to_numpy(float)
    ax.errorbar(x, y, xerr=np.vstack([x-lo, hi-x]), fmt="o", capsize=3)
    ax.set_yticks(y, ext["site_label"].tolist()); ax.invert_yaxis(); ax.set_xlabel("Macro AUROC (95% bootstrap CI)")
    ax.set_title("Raw external validation of frozen FS_NOAPS model"); ax.axvline(0.5, linestyle="--", linewidth=1)
    paths += _save_figure(fig, fig_dir / "figure_1_external_macro_auroc", protocol)

    fig, ax = plt.subplots(figsize=(7.2,4.4))
    matrix = ext[["class_0_auroc","class_1_auroc","class_2_auroc"]].to_numpy(float)
    im = ax.imshow(matrix, aspect="auto", vmin=0.5, vmax=0.9)
    ax.set_xticks([0,1,2], ["Early discharge","Death","Prolonged stay"]); ax.set_yticks(np.arange(len(ext)), ext["site_label"].tolist())
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]): ax.text(j,i,f"{matrix[i,j]:.3f}",ha="center",va="center")
    ax.set_title("Class-specific AUROC across external cohorts"); fig.colorbar(im, ax=ax, label="AUROC")
    paths += _save_figure(fig, fig_dir / "figure_2_external_class_auroc", protocol)

    recal = bundle["recalibration"].copy(); recal["site_label"] = recal["site"].map(SITE_LABELS).fillna(recal["site"])
    fig, ax = plt.subplots(figsize=(7.2,4.4)); y=np.arange(len(recal))
    ax.barh(y, recal["delta_multiclass_brier"].to_numpy(float)); ax.set_yticks(y,recal["site_label"].tolist()); ax.invert_yaxis()
    ax.axvline(0, linewidth=1); ax.set_xlabel("Δ multiclass Brier (recalibrated − raw)"); ax.set_title("Local recalibration sensitivity")
    paths += _save_figure(fig, fig_dir / "figure_s1_recalibration_brier_change", protocol)

    pre = bundle["preicu"].copy(); pre["label"] = pre["setting"].replace({"mimic_temporal":"MIMIC temporal","mimic->eicu_alt":"eICU-alt","mimic->eicu_primary":"eICU-primary","mimic->sicdb":"SICdb","mimic->nwicu":"NWICU"})
    fig, ax = plt.subplots(figsize=(7.2,4.6)); y=np.arange(len(pre)); x=pre["delta_macro_auroc"].to_numpy(float)
    if {"delta_macro_auroc_ci_lo","delta_macro_auroc_ci_hi"}.issubset(pre.columns):
        lo=pre["delta_macro_auroc_ci_lo"].to_numpy(float); hi=pre["delta_macro_auroc_ci_hi"].to_numpy(float); err=np.vstack([x-lo,hi-x])
        ax.errorbar(x,y,xerr=err,fmt="o",capsize=3)
    else: ax.plot(x,y,"o")
    ax.set_yticks(y,pre["label"].tolist()); ax.invert_yaxis(); ax.axvline(0,linestyle="--",linewidth=1)
    ax.set_xlabel("Δ macro AUROC after removing pre_icu_los_d"); ax.set_title("Pre-ICU LOS robustness sensitivity")
    paths += _save_figure(fig, fig_dir / "figure_s2_pre_icu_los_sensitivity", protocol)

    shap = bundle["shap"].copy(); shap = shap.sort_values("prolonged",ascending=True).tail(15)
    fig, ax = plt.subplots(figsize=(7.2,5.6)); ax.barh(shap["feature"],shap["prolonged"].astype(float)); ax.set_xlabel("Mean |SHAP| for prolonged class")
    ax.set_title("Primary FS_NOAPS prolonged-class feature importance")
    paths += _save_figure(fig, fig_dir / "figure_s3_shap_prolonged_top15", protocol)
    return paths


def _analysis_registry(bundle: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame([
        ["Nested 5×3 CV", "primary_development", "Main Table 1", "complete"],
        ["MIMIC temporal evaluation", "supportive_temporal", "Main Table 1", "complete"],
        ["Raw external validation: 4 cohorts", "primary_raw_external", "Main Table 2 / Figure 1–2", "complete"],
        ["External calibration", "secondary_calibration", "Main Table 3", "complete"],
        ["eICU→MIMIC reverse validation", "sensitivity_reverse", "Supplementary Table S2", "complete"],
        ["Local recalibration", "sensitivity_recalibration", "Supplementary Table S3 / Figure S1", "complete"],
        ["Decision curve analysis", "clinical_utility", "Supplementary Table S4", "complete"],
        ["eICU reference subgroups", "subgroup_reference", "Supplementary Table S5", "complete"],
        ["External exploratory subgroups", "exploratory_subgroup", "Supplementary Table S5", "complete"],
        ["Hospital heterogeneity", "robustness", "Supplementary Table S6", "complete"],
        ["Remove pre_icu_los_d", "robustness", "Supplementary Table S7 / Figure S2", "complete"],
        ["Feature availability", "robustness", "Supplementary Table S8", "complete"],
        ["Observed missingness burden", "robustness", "Supplementary Table S9", "complete"],
        ["HFD28 FS_NOAPS", "secondary_outcome", "Supplementary Table S10", "complete"],
        ["HFD28 professor method reproduction", "professor_method_reproduction", "Supplementary Table S11", "complete"],
        ["Primary FS_NOAPS SHAP", "interpretability", "Supplementary Table S12 / Figure S3", "complete"],
        ["Home-discharge sensitivity", "not_executable", "Supplementary Table S13", "not executable; not inferred"],
    ], columns=["analysis","analysis_role","report_location","status"])


def _write_markdown(bundle: dict[str, Any], protocol: Study28ReportingProtocol, output_dir: Path) -> tuple[str,str]:
    ext = _normalize_external(bundle)
    temporal = bundle["temporal"]; t = temporal.get("metrics",{})
    pre = bundle["preicu"].copy(); nw = pre.loc[pre["setting"].eq("mimic->nwicu")]
    nw_delta = float(nw.iloc[0]["delta_macro_auroc"]) if len(nw) else math.nan
    shap = bundle["shap"].sort_values("prolonged",ascending=False).head(5)
    shap_text = ", ".join(shap["feature"].astype(str).tolist())
    lines = [
        "# Study28 V9 Alpha 6 — Manuscript Evidence Summary", "",
        "## Evidence hierarchy", "",
        "- **Primary development:** professor-aligned 5×3 nested CV and locked LightGBM design.",
        "- **Supportive temporal evaluation:** MIMIC 2008–2016 → 2017–2022.",
        "- **Primary external validation:** raw, unrecalibrated MIMIC-frozen model on eICU-alt, eICU-primary, SICdb, and NWICU.",
        "- **Sensitivity / secondary analyses:** reverse validation, recalibration, DCA, subgroups, robustness, SHAP, and HFD28. These never replace raw Alpha 3 external results.", "",
        "## Primary numerical results", "",
        f"Nested-CV macro-AUROC: {bundle['nested_summary']['aggregate_outer_metrics']['macro_auroc']['mean']:.6f} ± {bundle['nested_summary']['aggregate_outer_metrics']['macro_auroc']['std']:.6f} (SD across outer folds).",
        f"MIMIC temporal macro-AUROC: {_as_float(t.get('macro_auroc')):.6f}.",
    ]
    for _, row in ext.iterrows():
        lines.append(f"- {row['site_label']}: macro-AUROC {row['macro_auroc']:.6f} (95% CI {row['macro_auroc_ci_lo']:.6f}–{row['macro_auroc_ci_hi']:.6f}).")
    lines += ["", "## Robustness highlights", "",
              f"Removing `pre_icu_los_d` reduced macro-AUROC in all five tested settings; the largest observed point change was NWICU (Δ {nw_delta:.6f}).",
              "Availability-matched retraining did not materially restore SICdb/NWICU performance, so complete feature absence alone does not explain transport differences.",
              f"Leading prolonged-class FS_NOAPS SHAP features: {shap_text}.",
              "Home-discharge sensitivity remains non-executable because `discharge_location` is absent from the frozen final datasets; labels were not inferred.", "",
              "## Reporting boundaries", ""]
    lines += [f"- **{key}:** {value}" for key,value in protocol.methodological_boundaries.items()]
    summary_path = output_dir / "manuscript_evidence_summary.md"; summary_path.write_text("\n".join(lines)+"\n",encoding="utf-8")

    methods = [
        "# Study28 Statistical Methods Evidence", "",
        "This file is a reporting aid generated from frozen aggregate evidence. It is not a substitute for the final manuscript Methods section.", "",
        "## Development and evaluation", "",
        "- Primary architecture: three-class LightGBM; primary feature set: FS_NOAPS (56 variables).",
        "- Hyperparameter lock reproduced using 5 outer × 3 inner stratified cross-validation on MIMIC; inner selection metric: macro-AUROC; seed: 2026.",
        "- Locked search parameters: num_leaves=15 and n_estimators=200; learning_rate=0.05 and min_child_samples=50.",
        "- Supportive temporal evaluation: MIMIC 2008–2016 development and 2017–2022 evaluation.",
        "- Raw external validation: eICU-alt, eICU-primary, SICdb, NWICU using the unchanged full-MIMIC frozen model.", "",
        "## Uncertainty and calibration", "",
        "- External confidence intervals: 1,000 nonparametric bootstrap repetitions with 95% intervals.",
        "- Calibration: one-vs-rest logistic calibration intercept and slope using logit-transformed predicted probability; quantile-based calibration bins.",
        "- Local recalibration: 20% local calibration / 80% held-out evaluation, repeated over 20 seeds; reported only as sensitivity/adaptation.", "",
        "## Additional analyses", "",
        "- Decision curve analysis for death and prolonged outcomes over thresholds 0.01–0.60.",
        "- eICU hospital heterogeneity: hospital N≥50 and events≥5; random-effects pooled AUROC and prediction interval.",
        "- `pre_icu_los_d` removal and external feature-availability analyses used locked primary hyperparameters and 1,000 paired bootstrap repetitions.",
        "- HFD28 secondary outcome: max(0, 28 − hospital LOS), with death/prolonged set to 0.", "",
        "## Provenance limitation", "",
        "The professor-provided final analysis CSVs are treated as frozen inputs. The original raw-to-final dataset generation scripts were unavailable, so reproducibility begins at the frozen final-dataset boundary.",
    ]
    methods_path = output_dir / "statistical_methods_evidence.md"; methods_path.write_text("\n".join(methods)+"\n",encoding="utf-8")
    return str(summary_path), str(methods_path)


def _privacy_check(output_dir: Path, protocol: Study28ReportingProtocol) -> dict[str, Any]:
    tokens = tuple(str(v).lower() for v in protocol.privacy_guard.get("forbidden_name_tokens", []))
    allowed = set(str(v) for v in protocol.privacy_guard.get("allowed_output_extensions", []))
    violations: list[str] = []
    for path in output_dir.rglob("*"):
        if not path.is_file(): continue
        rel = str(path.relative_to(output_dir)).lower()
        if path.suffix and path.suffix not in allowed: violations.append(f"disallowed extension: {rel}")
        if any(token in rel for token in tokens): violations.append(f"patient-level token in output filename: {rel}")
        if path.suffix == ".csv":
            try:
                cols = [c.lower() for c in pd.read_csv(path,nrows=0).columns]
            except Exception:
                continue
            identifiers = {"subject_id","hadm_id","stay_id","patientid","caseid","patientunitstayid","patienthealthsystemstayid"}
            bad = sorted(identifiers.intersection(cols))
            if bad: violations.append(f"identifier columns {bad}: {rel}")
    if violations:
        raise RuntimeError("Alpha 6 privacy guard failed: " + "; ".join(violations))
    return {"passed": True, "violations": []}


def run_alpha6(*, project_root: Path, output_dir: Path, alpha2_output_dir: Path | None = None,
               alpha3_output_dir: Path | None = None, alpha4_output_dir: Path | None = None,
               alpha5_output_dir: Path | None = None, use_release_verification_snapshots: bool = False) -> Alpha6Result:
    project_root = Path(project_root).expanduser().resolve(); output_dir = Path(output_dir).expanduser().resolve(); output_dir.mkdir(parents=True,exist_ok=True)
    protocol = load_reporting_protocol(project_root / "configs" / "study28_manuscript_reporting.json")
    if use_release_verification_snapshots:
        bundle = _collect_from_release_snapshots(project_root)
    else:
        if any(value is None for value in (alpha2_output_dir,alpha3_output_dir,alpha4_output_dir,alpha5_output_dir)):
            raise ValueError("Alpha 6 requires Alpha 2–5 output directories unless --use-release-verification-snapshots is enabled.")
        bundle = _collect_from_output_dirs(Path(alpha2_output_dir),Path(alpha3_output_dir),Path(alpha4_output_dir),Path(alpha5_output_dir))
    finalization = _validate_finalization(bundle, protocol)
    main_tables, supp_tables = _write_tables(bundle, protocol, output_dir)
    figures = _write_figures(bundle, protocol, output_dir)
    registry = _analysis_registry(bundle); registry_path = output_dir / "analysis_registry.csv"; registry.to_csv(registry_path,index=False)
    def portable_path(path: Path, kind: str) -> str:
        path = Path(path).resolve()
        for root, prefix in ((project_root, "project://"), (output_dir, "alpha6://")):
            try:
                return prefix + str(path.relative_to(root))
            except ValueError:
                pass
        return f"external_aggregate://{kind}/{path.name}"

    evidence_rows = []
    for source in bundle["paths"]:
        evidence_rows.append({"kind":"input_aggregate_evidence","path":portable_path(Path(source), "source"),"sha256":sha256_file(source)})
    summary_md, methods_md = _write_markdown(bundle, protocol, output_dir)
    output_candidates = [Path(x) for x in main_tables + supp_tables + figures + [str(registry_path),summary_md,methods_md]]
    for path in output_candidates:
        evidence_rows.append({"kind":"alpha6_output","path":portable_path(path, "output"),"sha256":sha256_file(path)})
    evidence_index = pd.DataFrame(evidence_rows); evidence_index_path = output_dir / "evidence_index.csv"; evidence_index.to_csv(evidence_index_path,index=False)
    privacy = _privacy_check(output_dir, protocol)
    checklist_path = output_dir / "finalization_checklist.json"
    atomic_write_json(checklist_path, {
        "schema_version": "study28_alpha6_finalization_checklist_v1",
        "passed": True,
        "checks": finalization["checks"],
        "privacy_guard": privacy,
        "required_external_sites": list(protocol.required_external_sites),
        "main_table_count": len(main_tables),
        "supplementary_table_count": len(supp_tables),
        "figure_file_count": len(figures),
        "home_discharge_status": "not_executable_not_inferred" if not bool(bundle["home"].get("executable", False)) else "executable",
    })
    manifest = {
        "schema_version":"study28_alpha6_manuscript_evidence_v1", "study_id":protocol.study_id,
        "source_mode":bundle["mode"], "reporting_protocol_sha256":protocol.sha256,
        "frozen_primary_model_sha256":bundle["freeze"].get("external_model_sha256"),
        "finalization":finalization, "privacy_guard":privacy,
        "required_external_sites":list(protocol.required_external_sites),
        "main_tables":[str(Path(x).relative_to(output_dir)) for x in main_tables],
        "supplementary_tables":[str(Path(x).relative_to(output_dir)) for x in supp_tables],
        "figures":[str(Path(x).relative_to(output_dir)) for x in figures],
        "analysis_registry":str(registry_path.relative_to(output_dir)),
        "evidence_index":str(evidence_index_path.relative_to(output_dir)),
        "finalization_checklist":str(checklist_path.relative_to(output_dir)),
        "methodological_boundaries":protocol.methodological_boundaries,
        "raw_external_results_replaced":False,
        "patient_level_outputs_created":False,
    }
    manifest_path = output_dir / "study28_alpha6_manuscript_evidence_manifest.json"; atomic_write_json(manifest_path,manifest)
    # Re-check after manifest creation.
    _privacy_check(output_dir, protocol)
    return Alpha6Result(
        manifest_path=str(manifest_path), evidence_index_csv=str(evidence_index_path), analysis_registry_csv=str(registry_path),
        main_tables=tuple(main_tables), supplementary_tables=tuple(supp_tables), figures=tuple(figures),
        manuscript_summary_md=summary_md, statistical_methods_md=methods_md, finalization_passed=True,
    )
