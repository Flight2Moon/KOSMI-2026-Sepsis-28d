from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from ..evaluation_multiclass import calculate_multiclass_metrics

CLASS_TO_CODE = {"early_dc": 0, "death": 1, "prolonged": 2}


def decision_curve_rows(
    *,
    setting: str,
    y_true: np.ndarray,
    model_probabilities: Mapping[str, np.ndarray],
    classes: Sequence[str],
    thresholds: np.ndarray,
) -> pd.DataFrame:
    y = np.asarray(y_true, dtype=np.int64).reshape(-1)
    rows: list[dict[str, object]] = []
    for class_name in classes:
        code = CLASS_TO_CODE[class_name]
        yy = (y == code).astype(np.int64)
        prevalence = float(yy.mean())
        for model_name, P in model_probabilities.items():
            p = np.asarray(P, dtype=np.float64)[:, code]
            for threshold in thresholds:
                if not 0 < threshold < 1:
                    raise ValueError("DCA thresholds must be in (0,1).")
                positive = p >= threshold
                tp = int(((positive) & (yy == 1)).sum())
                fp = int(((positive) & (yy == 0)).sum())
                n = len(yy)
                odds = threshold / (1.0 - threshold)
                nb = tp / n - fp / n * odds
                nb_all = prevalence - (1.0 - prevalence) * odds
                rows.append(
                    {
                        "setting": setting,
                        "class": class_name,
                        "model": model_name,
                        "threshold": float(threshold),
                        "net_benefit": float(nb),
                        "nb_treat_all": float(nb_all),
                        "nb_treat_none": 0.0,
                        "prevalence": prevalence,
                        "n": int(n),
                    }
                )
    return pd.DataFrame(rows)


def _subgroup_masks(frame: pd.DataFrame) -> dict[str, pd.Series]:
    required = {"age", "male", "invasive_mechanical_ventilation", "admission_elective"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise KeyError(f"Subgroup variables are missing: {missing}")
    return {
        "age<65": frame.age < 65,
        "age65-79": (frame.age >= 65) & (frame.age < 80),
        "age>=80": frame.age >= 80,
        "male": frame.male == 1,
        "female": frame.male == 0,
        "vent": frame.invasive_mechanical_ventilation == 1,
        "no_vent": frame.invasive_mechanical_ventilation == 0,
        "elective": frame.admission_elective == 1,
        "non_elective": frame.admission_elective != 1,
    }


def subgroup_performance(
    *,
    site: str,
    frame: pd.DataFrame,
    y_true: np.ndarray,
    probabilities: np.ndarray,
    definitions: Sequence[str],
    minimum_n: int,
    reference: bool,
) -> pd.DataFrame:
    y = np.asarray(y_true, dtype=np.int64).reshape(-1)
    P = np.asarray(probabilities, dtype=np.float64)
    masks = _subgroup_masks(frame)
    rows: list[dict[str, object]] = []
    output_columns = [
        "site", "analysis_role", "subgroup", "n", "n_early_dc", "n_death",
        "n_prolonged", "minimum_class_count", "small_class_warning",
        "auc_early_dc", "auc_death", "auc_prolonged", "macro_auroc", "macro_auprc",
    ]
    for name in definitions:
        if name not in masks:
            raise KeyError(f"Unknown Study28 subgroup definition: {name}")
        mask = masks[name].fillna(False).to_numpy(dtype=bool)
        yy = y[mask]
        if len(yy) < minimum_n or np.unique(yy).size < 3:
            continue
        d = calculate_multiclass_metrics(yy, P[mask], ("early_dc", "death", "prolonged"))
        rows.append(
            {
                "site": site,
                "analysis_role": "professor_reference" if reference else "exploratory_extension",
                "subgroup": name,
                "n": int(mask.sum()),
                "n_early_dc": int((yy == 0).sum()),
                "n_death": int((yy == 1).sum()),
                "n_prolonged": int((yy == 2).sum()),
                "minimum_class_count": int(min((yy == 0).sum(), (yy == 1).sum(), (yy == 2).sum())),
                "small_class_warning": bool(min((yy == 0).sum(), (yy == 1).sum(), (yy == 2).sum()) < 10),
                "auc_early_dc": float(d["class_0_auroc"]),
                "auc_death": float(d["class_1_auroc"]),
                "auc_prolonged": float(d["class_2_auroc"]),
                "macro_auroc": float(d["macro_auroc"]),
                "macro_auprc": float(d["macro_auprc"]),
            }
        )
    return pd.DataFrame(rows, columns=output_columns)


def hanley_mcneil_auc_se(auc: float, n1: int, n0: int) -> float:
    if n1 <= 0 or n0 <= 0:
        raise ValueError("AUC standard error requires positive event and non-event counts.")
    q1 = auc / (2.0 - auc)
    q2 = 2.0 * auc * auc / (1.0 + auc)
    variance = (
        auc * (1.0 - auc)
        + (n1 - 1) * (q1 - auc * auc)
        + (n0 - 1) * (q2 - auc * auc)
    ) / (n1 * n0)
    return float(np.sqrt(max(variance, 1e-10)))


def dersimonian_laird_pool(aucs: np.ndarray, ses: np.ndarray) -> dict[str, float]:
    aucs = np.asarray(aucs, dtype=np.float64)
    ses = np.asarray(ses, dtype=np.float64)
    if len(aucs) < 2 or len(aucs) != len(ses):
        raise ValueError("Random-effects pooling requires at least two aligned hospitals.")
    w = 1.0 / (ses ** 2)
    fixed = float(np.sum(w * aucs) / np.sum(w))
    q = float(np.sum(w * (aucs - fixed) ** 2))
    dfree = len(aucs) - 1
    c = float(np.sum(w) - np.sum(w ** 2) / np.sum(w))
    tau2 = max(0.0, (q - dfree) / c) if c > 0 else 0.0
    w2 = 1.0 / (ses ** 2 + tau2)
    mu = float(np.sum(w2 * aucs) / np.sum(w2))
    se = float(np.sqrt(1.0 / np.sum(w2)))
    i2 = max(0.0, (q - dfree) / q) * 100.0 if q > 0 else 0.0
    pred_se = float(np.sqrt(tau2 + se ** 2))
    return {
        "pooled_auc": mu,
        "ci_lower": mu - 1.96 * se,
        "ci_upper": mu + 1.96 * se,
        "prediction_interval_lower": mu - 1.96 * pred_se,
        "prediction_interval_upper": mu + 1.96 * pred_se,
        "tau2": float(tau2),
        "i2_percent": float(i2),
        "q": q,
        "k": int(len(aucs)),
    }


def hospital_level_heterogeneity(
    *,
    frame: pd.DataFrame,
    y_true: np.ndarray,
    probabilities: np.ndarray,
    hospital_column: str,
    classes: Sequence[str],
    minimum_hospital_n: int,
    minimum_events: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if hospital_column not in frame.columns:
        raise KeyError(f"Hospital column not found: {hospital_column}")
    y = np.asarray(y_true, dtype=np.int64).reshape(-1)
    P = np.asarray(probabilities, dtype=np.float64)
    work = frame.reset_index(drop=True).copy()
    if len(work) != len(y) or len(y) != len(P):
        raise ValueError("Hospital heterogeneity inputs are not row-aligned.")
    work["__row__"] = np.arange(len(work))
    rows: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []
    for class_name in classes:
        code = CLASS_TO_CODE[class_name]
        aucs: list[float] = []
        ses: list[float] = []
        for hospital_id, group in work.groupby(hospital_column, dropna=False):
            idx = group["__row__"].to_numpy(dtype=int)
            yy = (y[idx] == code).astype(np.int64)
            events = int(yy.sum())
            n = int(len(idx))
            if n < minimum_hospital_n or events < minimum_events or events >= n:
                continue
            auc = float(roc_auc_score(yy, P[idx, code]))
            se = hanley_mcneil_auc_se(auc, events, n - events)
            aucs.append(auc)
            ses.append(se)
            rows.append(
                {
                    "class": class_name,
                    "hospitalid": hospital_id,
                    "n": n,
                    "events": events,
                    "non_events": n - events,
                    "auc": auc,
                    "auc_se": se,
                }
            )
        pooled = dersimonian_laird_pool(np.asarray(aucs), np.asarray(ses))
        pooled.update({"class": class_name, "eligible_hospitals": len(aucs)})
        summaries.append(pooled)
    return pd.DataFrame(rows), pd.DataFrame(summaries)
