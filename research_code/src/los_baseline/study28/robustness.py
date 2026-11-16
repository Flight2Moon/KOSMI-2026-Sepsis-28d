from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, roc_auc_score

from ..evaluation_multiclass import calculate_multiclass_metrics
from ..utils import atomic_write_json, sha256_file
from .development import _fit_lightgbm, _model_parameters, split_mimic_temporal
from .external_validation import FrozenExternalModel
from .feature_sets import resolve_feature_set
from .interpretability import run_primary_shap
from .modeling import predict_three_class_probabilities
from .preprocessing import build_study28_preprocessor
from .robustness_protocol import Study28RobustnessProtocol, load_robustness_protocol
from .sensitivity import Alpha4Contract, validate_alpha4_contract

CLASS_NAMES = ("early_dc", "death", "prolonged")
PROBABILITY_COLUMNS = ("p_early_dc", "p_death", "p_prolonged")


@dataclass(frozen=True)
class Alpha5Contract:
    alpha4: Alpha4Contract
    robustness_protocol: Study28RobustnessProtocol
    alpha4_manifest: dict[str, Any]
    alpha4_manifest_path: Path


@dataclass(frozen=True)
class Alpha5Result:
    manifest_path: str
    shap_mean_abs_csv: str
    pre_icu_los_summary_csv: str
    availability_summary_csv: str
    missingness_summary_csv: str
    hfd28_summary_csv: str
    hfd28_model_csv: str
    home_discharge_status_json: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"JSON object required: {path}")
    return payload


def validate_alpha5_contract(
    *,
    dataset_root: Path,
    project_root: Path,
    alpha2_output_dir: Path,
    alpha3_output_dir: Path,
    alpha4_output_dir: Path,
    audit_output_dir: Path | None = None,
) -> Alpha5Contract:
    alpha4 = validate_alpha4_contract(
        dataset_root=dataset_root,
        project_root=project_root,
        alpha2_output_dir=alpha2_output_dir,
        alpha3_output_dir=alpha3_output_dir,
        audit_output_dir=audit_output_dir,
    )
    rp = load_robustness_protocol(Path(project_root) / "configs" / "study28_robustness_interpretability.json")
    if rp.study_id != alpha4.protocol.study_id:
        raise RuntimeError("Alpha 5 study_id differs from frozen Study28 protocol.")
    alpha4_manifest_path = Path(alpha4_output_dir).expanduser().resolve() / "study28_alpha4_sensitivity_manifest.json"
    if not alpha4_manifest_path.is_file():
        raise FileNotFoundError(f"Alpha 4 manifest not found: {alpha4_manifest_path}")
    manifest = _read_json(alpha4_manifest_path)
    if manifest.get("schema_version") != "study28_alpha4_sensitivity_v1":
        raise RuntimeError("Unsupported Alpha 4 manifest schema.")
    if rp.required_alpha4_complete:
        expected_model = alpha4.frozen.model_sha256
        if str(manifest.get("alpha2_frozen_model_sha256")) != expected_model:
            raise RuntimeError("Alpha 4 manifest does not reference the Alpha 2 frozen primary model.")
        if str(manifest.get("alpha3_manifest_sha256")) != sha256_file(alpha4.alpha3_manifest_path):
            raise RuntimeError("Alpha 4 manifest Alpha 3 SHA mismatch.")
    return Alpha5Contract(
        alpha4=alpha4,
        robustness_protocol=rp,
        alpha4_manifest=manifest,
        alpha4_manifest_path=alpha4_manifest_path,
    )


def _load_primary_predictions(
    *, setting: str, alpha2_output_dir: Path, alpha3_output_dir: Path,
) -> tuple[np.ndarray, np.ndarray]:
    if setting == "mimic_temporal":
        path = Path(alpha2_output_dir) / "02_temporal_evaluation" / "mimic_temporal_predictions.csv"
    else:
        if not setting.startswith("mimic->"):
            raise ValueError(f"Unsupported primary prediction setting: {setting}")
        site = setting.split("->", 1)[1]
        path = Path(alpha3_output_dir) / "sites" / site / "external_predictions.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Primary predictions missing: {path}")
    frame = pd.read_csv(path, low_memory=False)
    y = frame["y"].to_numpy(dtype=np.int64)
    P = frame[list(PROBABILITY_COLUMNS)].to_numpy(dtype=np.float64)
    if not np.allclose(P.sum(axis=1), 1.0, atol=1e-6, rtol=1e-6):
        raise RuntimeError(f"Primary probabilities do not sum to one: {setting}")
    return y, P


def _macro_auc(y: np.ndarray, P: np.ndarray) -> float:
    y = np.asarray(y, dtype=int)
    if tuple(sorted(np.unique(y).tolist())) != (0, 1, 2):
        return float("nan")
    onehot = np.eye(3, dtype=int)[y]
    return float(roc_auc_score(onehot, P, average="macro", multi_class="ovr"))


def _paired_bootstrap_macro_auc_difference(
    y: np.ndarray,
    primary: np.ndarray,
    sensitivity: np.ndarray,
    *,
    repeats: int,
    confidence_level: float,
    seed: int,
) -> dict[str, float | int]:
    y = np.asarray(y, dtype=np.int64)
    primary = np.asarray(primary, dtype=float)
    sensitivity = np.asarray(sensitivity, dtype=float)
    if primary.shape != sensitivity.shape or primary.shape != (len(y), 3):
        raise ValueError("Paired probability arrays must align with y.")
    rng = np.random.default_rng(seed)
    n = len(y)
    try:
        workers = max(1, int(os.environ.get("EXP123_ASSIGNED_CPU_THREADS", "1")))
    except ValueError:
        workers = 1

    def one(idx: np.ndarray) -> float | None:
        yy = y[idx]
        if len(np.unique(yy)) < 3:
            return None
        return _macro_auc(yy, sensitivity[idx]) - _macro_auc(yy, primary[idx])

    diffs: list[float] = []
    if workers <= 1:
        for _ in range(int(repeats)):
            value = one(rng.integers(0, n, size=n))
            if value is not None:
                diffs.append(float(value))
    else:
        batch_size = workers * 2
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="paired-bootstrap") as pool:
            done = 0
            while done < int(repeats):
                count = min(batch_size, int(repeats) - done)
                batch = [rng.integers(0, n, size=n) for _ in range(count)]
                for value in pool.map(one, batch):
                    if value is not None:
                        diffs.append(float(value))
                done += count
    if not diffs:
        raise RuntimeError("No valid paired bootstrap replicates.")
    alpha = (1.0 - float(confidence_level)) / 2.0
    arr = np.asarray(diffs, dtype=float)
    return {
        "paired_bootstrap_successful": int(len(arr)),
        "delta_macro_auroc_mean": float(arr.mean()),
        "delta_macro_auroc_ci_lo": float(np.quantile(arr, alpha)),
        "delta_macro_auroc_ci_hi": float(np.quantile(arr, 1.0 - alpha)),
    }

def _predict_locked_sensitivity(
    *, train: pd.DataFrame, evaluate: pd.DataFrame, features: Sequence[str], contract: Alpha5Contract,
) -> np.ndarray:
    proto = contract.alpha4.protocol
    params = proto.primary_model.expected_locked_search_parameters
    pre, model = _fit_lightgbm(
        train.loc[:, list(features)], train["y"].to_numpy(dtype=np.int64),
        tuple(features), _model_parameters(proto, params),
    )
    return predict_three_class_probabilities(model, pre.transform(evaluate.loc[:, list(features)]))


def _setting_frames(contract: Alpha5Contract, dataset_root: Path, setting: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    mimic = contract.alpha4.registry["mimic"].load(dataset_root)
    if setting == "mimic_temporal":
        return split_mimic_temporal(mimic, contract.alpha4.protocol)
    if not setting.startswith("mimic->"):
        raise ValueError(f"Unsupported Alpha 5 forward setting: {setting}")
    site = setting.split("->", 1)[1]
    return mimic, contract.alpha4.registry[site].load(dataset_root)


def run_pre_icu_los_sensitivity(
    *, contract: Alpha5Contract, dataset_root: Path, alpha2_output_dir: Path,
    alpha3_output_dir: Path, output_dir: Path,
) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg = contract.robustness_protocol.pre_icu_los_sensitivity
    base = list(contract.alpha4.feature_contract.no_apsiii)
    excluded = tuple(str(x) for x in cfg["excluded_features"])
    features = tuple(x for x in base if x not in excluded)
    if len(features) != len(base) - len(excluded):
        raise RuntimeError("pre_icu_los sensitivity exclusion did not change the primary feature set exactly once.")
    rows = []
    for setting in cfg["settings"]:
        train, evaluate = _setting_frames(contract, dataset_root, str(setting))
        y = evaluate["y"].to_numpy(dtype=np.int64)
        P_sens = _predict_locked_sensitivity(train=train, evaluate=evaluate, features=features, contract=contract)
        primary_y, P_primary = _load_primary_predictions(
            setting=str(setting), alpha2_output_dir=alpha2_output_dir, alpha3_output_dir=alpha3_output_dir
        )
        if not np.array_equal(y, primary_y):
            raise RuntimeError(f"Primary and no-pre-ICU sensitivity labels are not aligned: {setting}")
        sens_metrics = calculate_multiclass_metrics(y, P_sens, CLASS_NAMES)
        primary_metrics = calculate_multiclass_metrics(y, P_primary, CLASS_NAMES)
        paired = _paired_bootstrap_macro_auc_difference(
            y, P_primary, P_sens,
            repeats=int(cfg["paired_bootstrap_repeats"]),
            confidence_level=float(cfg["confidence_level"]), seed=int(cfg["seed"]),
        )
        rows.append({
            "setting": setting, "feature_set": "FS_NOAPS_NO_PREICU_LOS", "feature_count": len(features),
            "excluded_features": ";".join(excluded),
            "primary_macro_auroc": primary_metrics["macro_auroc"],
            "sensitivity_macro_auroc": sens_metrics["macro_auroc"],
            "delta_macro_auroc": float(sens_metrics["macro_auroc"] - primary_metrics["macro_auroc"]),
            "sensitivity_auc_early_dc": sens_metrics["class_0_auroc"],
            "sensitivity_auc_death": sens_metrics["class_1_auroc"],
            "sensitivity_auc_prolonged": sens_metrics["class_2_auroc"],
            **paired,
        })
    out = pd.DataFrame(rows)
    out.to_csv(output_dir / "pre_icu_los_removed_sensitivity.csv", index=False)
    return out


def _all_nan_primary_features(frame: pd.DataFrame, features: Sequence[str]) -> tuple[str, ...]:
    return tuple(str(f) for f in features if frame[str(f)].isna().all())


def run_feature_availability_sensitivity(
    *, contract: Alpha5Contract, dataset_root: Path, alpha3_output_dir: Path, output_dir: Path,
) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg = contract.robustness_protocol.feature_availability_sensitivity
    mimic = contract.alpha4.registry["mimic"].load(dataset_root)
    primary_features = tuple(contract.alpha4.feature_contract.no_apsiii)
    rows = []
    for site in cfg["sites"]:
        frame = contract.alpha4.registry[str(site)].load(dataset_root)
        dropped = _all_nan_primary_features(frame, primary_features)
        features = tuple(f for f in primary_features if f not in dropped)
        y = frame["y"].to_numpy(dtype=np.int64)
        P_sens = _predict_locked_sensitivity(train=mimic, evaluate=frame, features=features, contract=contract)
        primary_y, P_primary = _load_primary_predictions(
            setting=f"mimic->{site}", alpha2_output_dir=Path("."), alpha3_output_dir=alpha3_output_dir
        )
        if not np.array_equal(y, primary_y):
            raise RuntimeError(f"Primary and feature-availability sensitivity labels are not aligned: {site}")
        sens_metrics = calculate_multiclass_metrics(y, P_sens, CLASS_NAMES)
        primary_metrics = calculate_multiclass_metrics(y, P_primary, CLASS_NAMES)
        paired = _paired_bootstrap_macro_auc_difference(
            y, P_primary, P_sens, repeats=int(cfg["paired_bootstrap_repeats"]),
            confidence_level=float(cfg["confidence_level"]), seed=int(cfg["seed"]),
        )
        rows.append({
            "site": site, "analysis_role": cfg.get("analysis_role", "exploratory"),
            "primary_feature_count": len(primary_features), "available_feature_count": len(features),
            "all_nan_primary_feature_count": len(dropped), "dropped_all_nan_features": ";".join(dropped),
            "primary_macro_auroc": primary_metrics["macro_auroc"],
            "availability_matched_macro_auroc": sens_metrics["macro_auroc"],
            "delta_macro_auroc": float(sens_metrics["macro_auroc"] - primary_metrics["macro_auroc"]),
            "availability_auc_early_dc": sens_metrics["class_0_auroc"],
            "availability_auc_death": sens_metrics["class_1_auroc"],
            "availability_auc_prolonged": sens_metrics["class_2_auroc"],
            **paired,
        })
    out = pd.DataFrame(rows)
    out.to_csv(output_dir / "feature_availability_sensitivity.csv", index=False)
    return out


def run_missingness_robustness(
    *, contract: Alpha5Contract, dataset_root: Path, alpha2_output_dir: Path,
    alpha3_output_dir: Path, output_dir: Path,
) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg = contract.robustness_protocol.missingness_robustness
    features = tuple(contract.alpha4.feature_contract.no_apsiii)
    bins = [float(x) for x in cfg["bins"]]
    labels = [str(x) for x in cfg["labels"]]
    rows = []
    mimic = contract.alpha4.registry["mimic"].load(dataset_root)
    _, mimic_eval = split_mimic_temporal(mimic, contract.alpha4.protocol)
    for site in cfg["sites"]:
        if site == "mimic_temporal":
            frame = mimic_eval
            setting = "mimic_temporal"
        else:
            frame = contract.alpha4.registry[str(site)].load(dataset_root)
            setting = f"mimic->{site}"
        y, P = _load_primary_predictions(
            setting=setting, alpha2_output_dir=alpha2_output_dir, alpha3_output_dir=alpha3_output_dir
        )
        if len(frame) != len(y) or not np.array_equal(frame["y"].to_numpy(dtype=int), y):
            raise RuntimeError(f"Missingness robustness row alignment failed: {site}")
        missing_count = frame.loc[:, list(features)].isna().sum(axis=1).to_numpy(dtype=int)
        missing_fraction = missing_count / float(len(features))
        groups = pd.cut(missing_fraction, bins=bins, labels=labels, include_lowest=True, right=True)
        for label in labels:
            mask = np.asarray(groups == label)
            n = int(mask.sum())
            if n < int(cfg["minimum_n"]):
                continue
            yy = y[mask]
            class_counts = [int((yy == i).sum()) for i in range(3)]
            if bool(cfg["require_all_three_classes"]) and any(c == 0 for c in class_counts):
                continue
            metrics = calculate_multiclass_metrics(yy, P[mask], CLASS_NAMES)
            rows.append({
                "site": site, "analysis_role": cfg.get("analysis_role", "robustness_subgroup"),
                "missingness_bin": label, "n": n, "n_early_dc": class_counts[0],
                "n_death": class_counts[1], "n_prolonged": class_counts[2],
                "mean_missing_features": float(missing_count[mask].mean()),
                "mean_missing_fraction": float(missing_fraction[mask].mean()),
                "macro_auroc": metrics["macro_auroc"], "auc_early_dc": metrics["class_0_auroc"],
                "auc_death": metrics["class_1_auroc"], "auc_prolonged": metrics["class_2_auroc"],
            })
    out = pd.DataFrame(rows)
    out.to_csv(output_dir / "missingness_burden_performance.csv", index=False)
    return out


def hfd28_from_final_dataset(frame: pd.DataFrame) -> np.ndarray:
    if "hosp_los_d" not in frame or "label28" not in frame:
        raise KeyError("HFD28 requires hosp_los_d and label28.")
    los = pd.to_numeric(frame["hosp_los_d"], errors="raise").to_numpy(dtype=float)
    hfd = np.maximum(0.0, 28.0 - los)
    labels = frame["label28"].astype(str).to_numpy()
    hfd[np.isin(labels, ["death", "prolonged"])] = 0.0
    return hfd


def _hfd_metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    rho = float(spearmanr(y, pred).statistic)
    mae = float(mean_absolute_error(y, pred))
    r = float(np.corrcoef(y, pred)[0, 1]) if np.std(y) > 0 and np.std(pred) > 0 else float("nan")
    return {"spearman": rho, "mae_days": mae, "r2_pearson": float(r * r)}


def _bootstrap_hfd(y: np.ndarray, pred: np.ndarray, *, repeats: int, confidence_level: float, seed: int) -> dict[str, float | int]:
    rng = np.random.default_rng(seed)
    n = len(y)
    try:
        workers = max(1, int(os.environ.get("EXP123_ASSIGNED_CPU_THREADS", "1")))
    except ValueError:
        workers = 1
    values = {"spearman": [], "mae_days": [], "r2_pearson": []}

    def one(idx: np.ndarray) -> dict[str, float]:
        return _hfd_metrics(y[idx], pred[idx])

    if workers <= 1:
        iterator = (one(rng.integers(0, n, size=n)) for _ in range(int(repeats)))
        for m in iterator:
            for key in values:
                if np.isfinite(m[key]): values[key].append(m[key])
    else:
        batch_size = workers * 2
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="hfd-bootstrap") as pool:
            done = 0
            while done < int(repeats):
                count = min(batch_size, int(repeats) - done)
                batch = [rng.integers(0, n, size=n) for _ in range(count)]
                for m in pool.map(one, batch):
                    for key in values:
                        if np.isfinite(m[key]): values[key].append(m[key])
                done += count
    alpha = (1.0 - float(confidence_level)) / 2.0
    out: dict[str, float | int] = {"bootstrap_successful": min(len(v) for v in values.values())}
    for key, arr in values.items():
        a = np.asarray(arr, dtype=float)
        out[f"{key}_ci_lo"] = float(np.quantile(a, alpha))
        out[f"{key}_ci_hi"] = float(np.quantile(a, 1.0 - alpha))
    return out

def _fit_hfd_regressor(train: pd.DataFrame, features: Sequence[str], contract: Alpha5Contract) -> tuple[Any, Any]:
    pre = build_study28_preprocessor("lightgbm", features)
    X = pre.fit_transform(train.loc[:, list(features)])
    params = _model_parameters(contract.alpha4.protocol, contract.alpha4.protocol.primary_model.expected_locked_search_parameters)
    params = {k: v for k, v in params.items() if k not in {"objective", "num_class"}}
    params.setdefault("verbose", -1)
    model = LGBMRegressor(**params)
    model.fit(X, hfd28_from_final_dataset(train))
    return pre, model


def run_hfd28_secondary(
    *, contract: Alpha5Contract, dataset_root: Path, output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg = contract.robustness_protocol.hfd28_secondary
    features = tuple(contract.alpha4.feature_contract.no_apsiii)
    summary_rows = []
    for cohort in ["mimic", "eicu_primary", "eicu_alt", "sicdb", "nwicu"]:
        frame = contract.alpha4.registry[cohort].load(dataset_root)
        hfd = hfd28_from_final_dataset(frame)
        q = np.quantile(hfd, [0.25, 0.5, 0.75])
        summary_rows.append({
            "cohort": cohort, "n": len(frame), "mean": float(hfd.mean()), "q25": float(q[0]),
            "median": float(q[1]), "q75": float(q[2]), "pct_zero": float((hfd == 0).mean()),
        })
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output_dir / "hfd28_summary.csv", index=False)

    model_rows = []
    for setting in cfg["settings"]:
        train, evaluate = _setting_frames(contract, dataset_root, str(setting))
        pre, model = _fit_hfd_regressor(train, features, contract)
        pred = np.clip(model.predict(pre.transform(evaluate.loc[:, list(features)])), 0.0, 28.0)
        actual = hfd28_from_final_dataset(evaluate)
        point = _hfd_metrics(actual, pred)
        ci = _bootstrap_hfd(
            actual, pred, repeats=int(cfg["bootstrap_repeats"]),
            confidence_level=float(cfg["confidence_level"]), seed=int(cfg["seed"]),
        )
        model_rows.append({"setting": setting, "analysis_role": cfg.get("analysis_role", "secondary_outcome"),
                           "feature_count": len(features), **point, **ci})
    models = pd.DataFrame(model_rows)
    models.to_csv(output_dir / "hfd28_model_performance.csv", index=False)
    return summary, models


def run_home_discharge_guard(
    *, contract: Alpha5Contract, dataset_root: Path, output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg = contract.robustness_protocol.home_discharge_sensitivity
    required = str(cfg["required_column"])
    missing_sites = []
    for site in ["mimic", "eicu_alt", "eicu_primary", "sicdb", "nwicu"]:
        frame = contract.alpha4.registry[site].load(dataset_root)
        if required not in frame.columns:
            missing_sites.append(site)
    executable = len(missing_sites) == 0
    result = {
        "analysis": "home_discharge_sensitivity",
        "executable": executable,
        "required_column": required,
        "missing_sites": missing_sites,
        "status": "READY" if executable else "NOT_EXECUTABLE_WITH_FROZEN_FINAL_DATASET",
        "labels_inferred": False,
        "reason": None if executable else (
            "The professor method requires discharge_location, which is absent from one or more frozen final datasets. "
            "Alpha 5 will not reconstruct or guess home-discharge labels from LOS or other columns."
        ),
    }
    atomic_write_json(output_dir / "home_discharge_sensitivity_status.json", result)
    return result


def run_alpha5(
    *, dataset_root: Path, project_root: Path, alpha2_output_dir: Path,
    alpha3_output_dir: Path, alpha4_output_dir: Path, output_dir: Path,
) -> Alpha5Result:
    dataset_root = Path(dataset_root).expanduser().resolve()
    project_root = Path(project_root).expanduser().resolve()
    alpha2_output_dir = Path(alpha2_output_dir).expanduser().resolve()
    alpha3_output_dir = Path(alpha3_output_dir).expanduser().resolve()
    alpha4_output_dir = Path(alpha4_output_dir).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    contract = validate_alpha5_contract(
        dataset_root=dataset_root, project_root=project_root,
        alpha2_output_dir=alpha2_output_dir, alpha3_output_dir=alpha3_output_dir,
        alpha4_output_dir=alpha4_output_dir, audit_output_dir=output_dir / "00_dataset_audit",
    )
    mimic = contract.alpha4.registry["mimic"].load(dataset_root)

    print("[Study28 Alpha5] 1/6: primary FS_NOAPS SHAP", flush=True)
    shap_result = run_primary_shap(
        model=contract.alpha4.frozen.model, preprocessor=contract.alpha4.frozen.preprocessor,
        mimic=mimic, feature_names=contract.alpha4.frozen.feature_names,
        output_dir=output_dir / "01_interpretability",
        sample_size=int(contract.robustness_protocol.interpretability["sample_size"]),
        seed=int(contract.robustness_protocol.interpretability["seed"]),
        beeswarm_class=str(contract.robustness_protocol.interpretability["beeswarm_class"]),
        max_display=int(contract.robustness_protocol.interpretability["max_display"]),
    )
    print("[Study28 Alpha5] 2/6: pre_icu_los_d removal sensitivity", flush=True)
    preicu = run_pre_icu_los_sensitivity(
        contract=contract, dataset_root=dataset_root, alpha2_output_dir=alpha2_output_dir,
        alpha3_output_dir=alpha3_output_dir, output_dir=output_dir / "02_pre_icu_los_sensitivity",
    )
    print("[Study28 Alpha5] 3/6: site feature-availability sensitivity", flush=True)
    availability = run_feature_availability_sensitivity(
        contract=contract, dataset_root=dataset_root, alpha3_output_dir=alpha3_output_dir,
        output_dir=output_dir / "03_feature_availability",
    )
    print("[Study28 Alpha5] 4/6: observed missingness-burden robustness", flush=True)
    missingness = run_missingness_robustness(
        contract=contract, dataset_root=dataset_root, alpha2_output_dir=alpha2_output_dir,
        alpha3_output_dir=alpha3_output_dir, output_dir=output_dir / "04_missingness_robustness",
    )
    print("[Study28 Alpha5] 5/6: HFD28 secondary outcome", flush=True)
    hfd_summary, hfd_models = run_hfd28_secondary(
        contract=contract, dataset_root=dataset_root, output_dir=output_dir / "05_hfd28_secondary",
    )
    print("[Study28 Alpha5] 6/6: home-discharge reproducibility guard", flush=True)
    home = run_home_discharge_guard(
        contract=contract, dataset_root=dataset_root, output_dir=output_dir / "06_home_discharge_guard",
    )

    manifest = {
        "schema_version": "study28_alpha5_robustness_interpretability_v1",
        "study_id": contract.alpha4.protocol.study_id,
        "frozen_primary_model_sha256": contract.alpha4.frozen.model_sha256,
        "alpha3_manifest_sha256": sha256_file(contract.alpha4.alpha3_manifest_path),
        "alpha4_manifest_sha256": sha256_file(contract.alpha4_manifest_path),
        "robustness_protocol_sha256": contract.robustness_protocol.sha256,
        "primary_model_changed": False,
        "primary_external_results_overwritten": False,
        "new_models_are_sensitivity_only": True,
        "shap": shap_result,
        "pre_icu_los_sensitivity_rows": int(len(preicu)),
        "feature_availability_rows": int(len(availability)),
        "missingness_rows": int(len(missingness)),
        "hfd28_cohort_rows": int(len(hfd_summary)),
        "hfd28_model_rows": int(len(hfd_models)),
        "home_discharge_sensitivity": home,
        "methodological_notes": contract.robustness_protocol.methodological_notes,
    }
    manifest_path = output_dir / "study28_alpha5_robustness_interpretability_manifest.json"
    atomic_write_json(manifest_path, manifest)
    return Alpha5Result(
        manifest_path=str(manifest_path),
        shap_mean_abs_csv=shap_result["mean_abs_csv"],
        pre_icu_los_summary_csv=str(output_dir / "02_pre_icu_los_sensitivity" / "pre_icu_los_removed_sensitivity.csv"),
        availability_summary_csv=str(output_dir / "03_feature_availability" / "feature_availability_sensitivity.csv"),
        missingness_summary_csv=str(output_dir / "04_missingness_robustness" / "missingness_burden_performance.csv"),
        hfd28_summary_csv=str(output_dir / "05_hfd28_secondary" / "hfd28_summary.csv"),
        hfd28_model_csv=str(output_dir / "05_hfd28_secondary" / "hfd28_model_performance.csv"),
        home_discharge_status_json=str(output_dir / "06_home_discharge_guard" / "home_discharge_sensitivity_status.json"),
    )
