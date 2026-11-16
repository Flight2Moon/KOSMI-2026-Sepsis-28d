"""Paired bootstrap comparison: multiclass model versus death-first two-stage model.

Purpose
-------
Quantify, with patient-level paired confidence intervals, whether the clinically
intuitive death-first two-stage architecture outperforms a single multiclass
model on the Study28 EXP1/EXP2 comparison.

What this script does
---------------------
1. Loads the frozen Stage-2 optimized models produced by the Study28 EXP123 run
   (`exp1_flat_3class`, `stage1_mortality`, `stage2_long_stay`).
2. Recomputes patient-level predicted probabilities on the temporal split and on
   the four external cohorts.
3. Combines the two stage models into two-stage 3-class probabilities using the
   same rule as the training pipeline.
4. Derives paired bootstrap confidence intervals for the macro-AUROC difference.

Nothing is refitted. The models, the preprocessors and the frozen feature order
are read from disk exactly as produced by the original run, so this script is a
pure re-scoring step and is therefore deterministic given the same artifacts.

Usage
-----
    python analysis/paired_bootstrap_flat_vs_cascade.py                 # all sites
    python analysis/paired_bootstrap_flat_vs_cascade.py sicdb nwicu     # selected
    python analysis/paired_bootstrap_flat_vs_cascade.py --merge         # merge parts

Output
------
    analysis/results/boot_<site>[_<site>...].csv  per invocation
    analysis/results/flat_vs_cascade_paired_bootstrap.csv  with --merge

Reproducibility notes
---------------------
* Seed 42, 1000 paired bootstrap replicates. Re-running on the same frozen
  artifacts reproduces every reported figure to four decimal places.
* Two environment workarounds are applied; both are no-ops on the pinned
  environment in `requirements.txt` and only matter on newer stacks:
  - Under pandas >= 3.0 the XGBoost / LightGBM sklearn wrappers no longer
    resolve DataFrame feature names against the stored booster, raising
    "data did not contain feature names". Native-missing models are therefore
    scored from a column-ordered NumPy array with name validation disabled. The
    column order comes from the frozen manifest, so the input matrix is
    identical to the one used at training time.
  - `sklearn.metrics.roc_auc_score` is far too slow inside a bootstrap loop.
    The rank-based equivalent below was verified against scikit-learn to 1e-9,
    tied scores included.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score

# --------------------------------------------------------------------------- #
# Paths: everything resolves from the project root, so the script runs from any
# working directory and on any machine without edits.
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_ROOT / "data"
MODEL_ROOT = PROJECT_ROOT / "runs/study28_exp123_main/stage2_optimization/models"
RESULTS_DIR = PROJECT_ROOT / "analysis/results"

# The pickled preprocessor is a project class, so the package must be importable.
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# --------------------------------------------------------------------------- #
# Experiment constants
# --------------------------------------------------------------------------- #
MODEL_NAMES = (
    "logistic_regression",
    "decision_tree",
    "random_forest",
    "xgboost",
    "lightgbm",
)
ALL_SITES = ("mimic_temporal", "eicu_alt", "eicu_primary", "sicdb", "nwicu")

# MIMIC-IV anchor year groups forming the development subset.
DEVELOPMENT_YEAR_GROUPS = ("2008 - 2010", "2011 - 2013", "2014 - 2016")

# Class encoding of the frozen label column `y`.
CLASS_EARLY_DISCHARGE = 0
CLASS_DEATH = 1
CLASS_PROLONGED = 2

N_BOOTSTRAP = 1000
SEED = 42

MERGED_NAME = "flat_vs_cascade_paired_bootstrap.csv"


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
def load_site(site: str) -> pd.DataFrame:
    """Return the evaluation frame for one site, applying its cohort filter."""
    if site == "mimic_temporal":
        frame = pd.read_csv(DATASET_ROOT / "mimic_final.csv", low_memory=False)
        selected = ~frame["anchor_year_group"].isin(DEVELOPMENT_YEAR_GROUPS)
        return frame.loc[selected].reset_index(drop=True)
    if site == "eicu_alt":
        frame = pd.read_csv(DATASET_ROOT / "eicu_final.csv", low_memory=False)
        return frame.loc[frame["sepsis_admitdx"] == 1].reset_index(drop=True)
    if site == "eicu_primary":
        frame = pd.read_csv(DATASET_ROOT / "eicu_final.csv", low_memory=False)
        return frame.loc[frame["sepsis3_primary"] == 1].reset_index(drop=True)
    if site == "sicdb":
        return pd.read_csv(DATASET_ROOT / "sicdb_final.csv", low_memory=False)
    if site == "nwicu":
        return pd.read_csv(DATASET_ROOT / "nwicu_final.csv", low_memory=False)
    raise KeyError(f"Unknown site: {site}")


def fit_scope_for(site: str) -> str:
    """Which frozen fit to use for a site.

    The temporal split is scored with the model fitted on the development years
    only. External cohorts are scored with the model refitted on the full
    MIMIC-IV cohort, matching the original protocol.
    """
    return "temporal_full_development" if site == "mimic_temporal" else "external_full_mimic"


def load_fitted(task: str, model_name: str, fit_scope: str):
    import json

    directory = MODEL_ROOT / task / model_name / fit_scope
    if not directory.is_dir():
        raise FileNotFoundError(
            f"[ERROR] Frozen model directory not found: {directory}. "
            "Confirm that runs/study28_exp123_main/stage2_optimization is present."
        )
    preprocessor = joblib.load(directory / "preprocessor.joblib")
    estimator = joblib.load(directory / "model.joblib")
    manifest = json.loads((directory / "model_manifest.json").read_text(encoding="utf-8"))
    return preprocessor, estimator, list(manifest["feature_names"])


def predict_probability(
    task: str, model_name: str, fit_scope: str, frame: pd.DataFrame
) -> np.ndarray:
    """Score one frozen model on one evaluation frame."""
    preprocessor, estimator, feature_names = load_fitted(task, model_name, fit_scope)
    transformed = preprocessor.transform(frame.loc[:, feature_names])

    if isinstance(transformed, pd.DataFrame):
        # Native-missing learners (xgboost, lightgbm): see module docstring.
        if list(transformed.columns) != list(feature_names):
            raise ValueError("Preprocessor returned an unexpected feature order.")
        get_booster = getattr(estimator, "get_booster", None)
        if get_booster is not None:
            get_booster().feature_names = None
        transformed = transformed.to_numpy(dtype=np.float64)

    return np.asarray(estimator.predict_proba(transformed), dtype=np.float64)


def cascade_probabilities(stage1: np.ndarray, stage2: np.ndarray) -> np.ndarray:
    """Combine the two stage models into 3-class probabilities.

        P(death)     = P1
        P(early)     = (1 - P1) * (1 - P2)
        P(prolonged) = (1 - P1) * P2

    Column order matches the `y` encoding: early, death, prolonged.
    """
    p_death = stage1[:, 1]
    p_prolonged_conditional = stage2[:, 1]
    p_survive = 1.0 - p_death
    probability = np.column_stack(
        [
            p_survive * (1.0 - p_prolonged_conditional),
            p_death,
            p_survive * p_prolonged_conditional,
        ]
    )
    probability = np.clip(probability, 1e-12, 1.0)
    return probability / probability.sum(axis=1, keepdims=True)


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def binary_auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Rank-based AUROC. Verified against sklearn to 1e-9, ties included."""
    n_positive = float(labels.sum())
    n_negative = float(len(labels) - n_positive)
    if n_positive == 0.0 or n_negative == 0.0:
        return float("nan")
    ranks = rankdata(scores, method="average")
    rank_sum = float(ranks[labels == 1].sum())
    return (rank_sum - n_positive * (n_positive + 1.0) / 2.0) / (n_positive * n_negative)


def macro_auroc(y_true: np.ndarray, probability: np.ndarray) -> float:
    """One-vs-rest AUROC averaged over the three classes."""
    return float(
        np.mean(
            [
                binary_auroc((y_true == index).astype(np.int64), probability[:, index])
                for index in range(probability.shape[1])
            ]
        )
    )


def macro_auprc(y_true: np.ndarray, probability: np.ndarray) -> float:
    return float(
        np.mean(
            [
                average_precision_score((y_true == index).astype(int), probability[:, index])
                for index in range(probability.shape[1])
            ]
        )
    )


# --------------------------------------------------------------------------- #
# Paired bootstrap
# --------------------------------------------------------------------------- #
def paired_bootstrap(
    y_true: np.ndarray,
    flat: np.ndarray,
    cascade: np.ndarray,
    *,
    n_bootstrap: int = N_BOOTSTRAP,
    seed: int = SEED,
) -> dict[str, float]:
    """Patient-level paired bootstrap of the macro-AUROC difference.

    Both architectures are scored on the same resampled patients, so the
    difference is paired and the between-model correlation is preserved.
    Resamples that lose a class are discarded, since macro-AUROC is undefined
    for them.
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)

    differences: list[float] = []
    flat_values: list[float] = []
    cascade_values: list[float] = []

    attempts = 0
    max_attempts = n_bootstrap * 5
    while len(differences) < n_bootstrap and attempts < max_attempts:
        attempts += 1
        index = rng.integers(0, n, size=n)
        labels = y_true[index]
        if len(np.unique(labels)) < 3:
            continue
        flat_score = macro_auroc(labels, flat[index])
        cascade_score = macro_auroc(labels, cascade[index])
        flat_values.append(flat_score)
        cascade_values.append(cascade_score)
        differences.append(flat_score - cascade_score)

    if len(differences) < n_bootstrap:
        print(
            f"[WARNING] Only {len(differences)}/{n_bootstrap} resamples retained "
            f"after {attempts} attempts."
        )

    differences_array = np.asarray(differences)
    flat_point = macro_auroc(y_true, flat)
    cascade_point = macro_auroc(y_true, cascade)

    return {
        "flat_macro_auroc": flat_point,
        "flat_ci_low": float(np.percentile(flat_values, 2.5)),
        "flat_ci_high": float(np.percentile(flat_values, 97.5)),
        "cascade_macro_auroc": cascade_point,
        "cascade_ci_low": float(np.percentile(cascade_values, 2.5)),
        "cascade_ci_high": float(np.percentile(cascade_values, 97.5)),
        "delta": flat_point - cascade_point,
        "delta_ci_low": float(np.percentile(differences_array, 2.5)),
        "delta_ci_high": float(np.percentile(differences_array, 97.5)),
        "n_resamples": int(len(differences_array)),
    }


COLUMNS = [
    "site", "model", "n", "prolonged_prevalence",
    "flat_macro_auroc", "flat_ci_low", "flat_ci_high",
    "cascade_macro_auroc", "cascade_ci_low", "cascade_ci_high",
    "delta", "delta_ci_low", "delta_ci_high",
    "flat_macro_auprc", "cascade_macro_auprc", "prolonged_auprc_flat",
    "n_resamples",
]


def merge_parts() -> int:
    """Concatenate per-site part files into the canonical results CSV."""
    parts = sorted(RESULTS_DIR.glob("boot_*.csv"))
    if not parts:
        raise SystemExit(f"[ERROR] No part files found in {RESULTS_DIR}")
    merged = pd.concat([pd.read_csv(path) for path in parts], ignore_index=True)
    order = {site: index for index, site in enumerate(ALL_SITES)}
    merged = merged.sort_values(
        ["site", "model"], key=lambda column: column.map(order).fillna(99)
        if column.name == "site" else column
    )
    destination = RESULTS_DIR / MERGED_NAME
    merged.loc[:, COLUMNS].to_csv(destination, index=False)
    print(f"[OK] Merged {len(parts)} part file(s) -> {destination}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sites", nargs="*", default=None)
    parser.add_argument(
        "--merge", action="store_true",
        help="Merge existing boot_*.csv part files into the canonical results CSV.",
    )
    arguments = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if arguments.merge:
        return merge_parts()

    sites = tuple(arguments.sites) or ALL_SITES
    unknown = [site for site in sites if site not in ALL_SITES]
    if unknown:
        raise SystemExit(f"[ERROR] Unknown site(s): {unknown}. Valid: {list(ALL_SITES)}")

    rows: list[dict[str, object]] = []

    for site in sites:
        frame = load_site(site)
        y_true = frame["y"].to_numpy(dtype=np.int64)
        fit_scope = fit_scope_for(site)
        print(f"[INFO] {site}: n={len(frame)}, fit_scope={fit_scope}")

        for model_name in MODEL_NAMES:
            flat = predict_probability("exp1_flat_3class", model_name, fit_scope, frame)
            stage1 = predict_probability("stage1_mortality", model_name, fit_scope, frame)
            stage2 = predict_probability("stage2_long_stay", model_name, fit_scope, frame)
            cascade = cascade_probabilities(stage1, stage2)

            result = paired_bootstrap(y_true, flat, cascade)
            result.update(
                {
                    "site": site,
                    "model": model_name,
                    "n": len(frame),
                    "prolonged_prevalence": float((y_true == CLASS_PROLONGED).mean()),
                    "flat_macro_auprc": macro_auprc(y_true, flat),
                    "cascade_macro_auprc": macro_auprc(y_true, cascade),
                    "prolonged_auprc_flat": float(
                        average_precision_score(
                            (y_true == CLASS_PROLONGED).astype(int),
                            flat[:, CLASS_PROLONGED],
                        )
                    ),
                }
            )
            rows.append(result)
            print(
                f"  {model_name:20s} "
                f"flat={result['flat_macro_auroc']:.4f} "
                f"cascade={result['cascade_macro_auroc']:.4f} "
                f"delta={result['delta']:+.4f} "
                f"[{result['delta_ci_low']:+.4f}, {result['delta_ci_high']:+.4f}]"
            )

    destination = RESULTS_DIR / f"boot_{'_'.join(sites)}.csv"
    pd.DataFrame(rows).loc[:, COLUMNS].to_csv(destination, index=False)
    print(f"[OK] Written: {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
