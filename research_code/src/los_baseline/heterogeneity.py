from __future__ import annotations

import json
import math
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import log_loss, roc_auc_score
from statsmodels.genmod.cov_struct import Exchangeable
from statsmodels.genmod.generalized_estimating_equations import GEE
from statsmodels.stats.contingency_tables import cochrans_q
from statsmodels.stats.multitest import multipletests

from .data import load_clinical_dataset
from .utils import atomic_write_json, sha256_file


DEFAULT_MODELS = (
    "decision_tree",
    "logistic_regression",
    "random_forest",
    "svm",
    "xgboost",
)
DEFAULT_SUBGROUP_VARIABLES = (
    "age",
    "sofa_score",
    "aps3",
    "charlson_comorbidity_index",
)


@dataclass(frozen=True)
class StrataDefinition:
    variable: str
    method: str
    labels: tuple[str, ...]
    edges: tuple[float, ...]
    requested_n_bins: int
    actual_n_bins: int
    fitted_on_split: str
    minimum_bin_size: int
    minimum_class_count: int
    duplicate_edges_dropped: bool


@dataclass(frozen=True)
class HeterogeneityConfig:
    run_dir: Path
    threshold_days: int
    discovery_split: str = "validation"
    model_names: tuple[str, ...] = DEFAULT_MODELS
    subgroup_variables: tuple[str, ...] = DEFAULT_SUBGROUP_VARIABLES
    n_bins: int = 3
    minimum_bin_size: int = 30
    minimum_class_count: int = 10
    default_probability_threshold: float = 0.5
    fdr_method: str = "fdr_bh"
    fdr_alpha: float = 0.05
    brier_tie_margin: float = 0.005
    random_seed: int = 42
    trend_bins: int = 50


def _los_directory_name(threshold_days: int | float) -> str:
    return f"LOS_{int(float(threshold_days)):02d}"


def _prediction_path(
    run_dir: Path,
    threshold_days: int,
    model_name: str,
    split: str,
) -> Path:
    return (
        run_dir
        / _los_directory_name(threshold_days)
        / model_name
        / "evaluation"
        / split
        / "predictions.csv.gz"
    )


def _load_covariates(
    dataset_path: Path,
    subgroup_variables: Iterable[str],
    target_column: str = "LOS",
    identifier_columns: Iterable[str] = ("subject_id", "hadm_id", "stay_id"),
) -> pd.DataFrame:
    variables = list(dict.fromkeys(subgroup_variables))
    dataset = load_clinical_dataset(
        csv_path=dataset_path,
        feature_columns=variables,
        target_column=target_column,
        candidate_identifier_columns=list(identifier_columns),
    )
    columns = ["_source_row_index", *variables]
    frame = dataset.frame.loc[:, columns].rename(
        columns={"_source_row_index": "source_row_index"}
    )
    if frame["source_row_index"].duplicated().any():
        raise ValueError("Covariate source_row_index values are not unique.")
    return frame


def load_predictions_with_covariates(
    run_dir: Path,
    threshold_days: int,
    model_names: Iterable[str],
    split: str,
    dataset_path: Path,
    subgroup_variables: Iterable[str] = DEFAULT_SUBGROUP_VARIABLES,
) -> pd.DataFrame:
    """Load stored predictions and join prespecified clinical covariates."""
    run_dir = run_dir.expanduser().resolve()
    covariates = _load_covariates(dataset_path, subgroup_variables)
    model_frames: list[pd.DataFrame] = []
    reference_indices: np.ndarray | None = None
    reference_y: np.ndarray | None = None

    for model_name in model_names:
        path = _prediction_path(run_dir, threshold_days, model_name, split)
        if not path.is_file():
            raise FileNotFoundError(f"Prediction file was not found: {path}")
        prediction = pd.read_csv(path, low_memory=False)
        required = {"source_row_index", "y_true", "positive_probability"}
        missing = sorted(required - set(prediction.columns))
        if missing:
            raise KeyError(f"Missing prediction columns in {path}: {missing}")

        prediction = prediction.loc[
            :, ["source_row_index", "y_true", "positive_probability"]
        ].copy()
        prediction["source_row_index"] = pd.to_numeric(
            prediction["source_row_index"], errors="raise"
        ).astype(np.int64)
        prediction["y_true"] = pd.to_numeric(
            prediction["y_true"], errors="raise"
        ).astype(np.int32)
        prediction["positive_probability"] = pd.to_numeric(
            prediction["positive_probability"], errors="raise"
        ).astype(np.float64)
        if prediction["source_row_index"].duplicated().any():
            raise ValueError(f"Duplicate source rows in {path}")
        prediction = prediction.sort_values("source_row_index").reset_index(drop=True)

        current_indices = prediction["source_row_index"].to_numpy()
        current_y = prediction["y_true"].to_numpy()
        if reference_indices is None:
            reference_indices = current_indices
            reference_y = current_y
        else:
            if not np.array_equal(reference_indices, current_indices):
                raise ValueError(
                    "The five models do not share the same source_row_index set. "
                    f"Mismatch detected for {model_name}/{split}."
                )
            if not np.array_equal(reference_y, current_y):
                raise ValueError(
                    "The five models do not share identical y_true values. "
                    f"Mismatch detected for {model_name}/{split}."
                )

        merged = prediction.merge(
            covariates,
            on="source_row_index",
            how="left",
            validate="one_to_one",
            indicator=True,
        )
        unmatched = int((merged["_merge"] != "both").sum())
        if unmatched:
            raise ValueError(
                f"Covariate join failed for {unmatched} rows in {model_name}/{split}."
            )
        merged = merged.drop(columns="_merge")
        missing_covariates = merged[list(subgroup_variables)].isna().sum()
        if int(missing_covariates.sum()) > 0:
            raise ValueError(
                "Subgroup covariates contain missing values after join: "
                f"{missing_covariates[missing_covariates > 0].to_dict()}"
            )
        merged = merged.rename(columns={"positive_probability": "y_prob"})
        merged["model_name"] = model_name
        merged["squared_error"] = (
            merged["y_true"].astype(float) - merged["y_prob"]
        ) ** 2
        clipped = np.clip(merged["y_prob"].to_numpy(), 1e-15, 1 - 1e-15)
        merged["log_loss_component"] = -(
            merged["y_true"].to_numpy() * np.log(clipped)
            + (1 - merged["y_true"].to_numpy()) * np.log(1 - clipped)
        )
        model_frames.append(merged)

    output = pd.concat(model_frames, ignore_index=True)
    output["threshold_days"] = int(threshold_days)
    output["split"] = split
    return output


def fit_strata_definition(
    frame: pd.DataFrame,
    variable: str,
    fitted_on_split: str = "validation",
    method: str = "quantile",
    n_bins: int = 3,
    bin_edges: Iterable[float] | None = None,
    minimum_bin_size: int = 30,
    minimum_class_count: int = 10,
) -> StrataDefinition:
    values = pd.to_numeric(frame[variable], errors="coerce")
    if values.isna().any():
        raise ValueError(f"Missing values found in subgroup variable: {variable}")

    duplicate_edges_dropped = False
    if method == "quantile":
        _, edges = pd.qcut(
            values,
            q=n_bins,
            retbins=True,
            duplicates="drop",
        )
        edges = np.asarray(edges, dtype=float)
        duplicate_edges_dropped = len(edges) - 1 < n_bins
    elif method == "fixed":
        if bin_edges is None:
            raise ValueError("bin_edges are required when method='fixed'.")
        edges = np.asarray(list(bin_edges), dtype=float)
    else:
        raise ValueError(f"Unsupported strata method: {method}")

    edges = np.unique(edges)
    if len(edges) < 3:
        raise ValueError(
            f"{variable} cannot form at least two strata with the requested method."
        )
    edges[0] = -np.inf
    edges[-1] = np.inf
    actual_n_bins = len(edges) - 1
    if actual_n_bins == 3:
        labels = ("Low", "Mid", "High")
    else:
        labels = tuple(f"Bin_{index + 1}" for index in range(actual_n_bins))

    assigned = pd.cut(
        values,
        bins=edges,
        labels=labels,
        include_lowest=True,
        ordered=True,
    )
    counts = assigned.value_counts(sort=False)
    too_small = counts[counts < minimum_bin_size]
    if not too_small.empty:
        warnings.warn(
            f"{variable} has bins smaller than {minimum_bin_size}: "
            f"{too_small.to_dict()}. The frozen edges are retained; cells will be flagged.",
            RuntimeWarning,
        )

    return StrataDefinition(
        variable=variable,
        method=method,
        labels=labels,
        edges=tuple(float(value) for value in edges),
        requested_n_bins=int(n_bins),
        actual_n_bins=int(actual_n_bins),
        fitted_on_split=fitted_on_split,
        minimum_bin_size=int(minimum_bin_size),
        minimum_class_count=int(minimum_class_count),
        duplicate_edges_dropped=duplicate_edges_dropped,
    )


def apply_strata_definition(
    frame: pd.DataFrame,
    definition: StrataDefinition,
) -> pd.DataFrame:
    output = frame.copy()
    column = f"{definition.variable}_bin"
    output[column] = pd.cut(
        pd.to_numeric(output[definition.variable], errors="coerce"),
        bins=np.asarray(definition.edges, dtype=float),
        labels=list(definition.labels),
        include_lowest=True,
        ordered=True,
    )
    if output[column].isna().any():
        raise ValueError(
            f"Some {definition.variable} values could not be assigned to frozen strata."
        )
    return output


def compute_subgroup_metrics(
    frame: pd.DataFrame,
    stratify_by: str,
    minimum_class_count: int = 10,
    minimum_bin_size: int = 30,
    default_threshold: float = 0.5,
) -> pd.DataFrame:
    """Aggregate threshold-free performance by model and frozen stratum.

    The function supports both the legacy LOS-threshold analysis and the
    EXP123 task-based analyses. Context columns are copied when present.
    """
    bin_column = f"{stratify_by}_bin"
    rows: list[dict[str, Any]] = []
    grouped = frame.groupby(["model_name", bin_column], observed=True, sort=False)
    for (model_name, bin_label), group in grouped:
        y = group["y_true"].to_numpy(dtype=np.int32)
        probability = group["y_prob"].to_numpy(dtype=np.float64)
        positive_count = int((y == 1).sum())
        negative_count = int((y == 0).sum())
        insufficient = (
            len(group) < minimum_bin_size
            or positive_count < minimum_class_count
            or negative_count < minimum_class_count
        )
        auroc = float("nan")
        if not insufficient:
            auroc = float(roc_auc_score(y, probability))
        prediction = (probability >= default_threshold).astype(np.int32)
        record: dict[str, Any] = {}
        if "threshold_days" in group.columns:
            record["threshold_days"] = int(group["threshold_days"].iloc[0])
        if "task_name" in group.columns:
            record["task_name"] = str(group["task_name"].iloc[0])
        record.update(
            {
                "split": str(group["split"].iloc[0]),
                "stratify_by": stratify_by,
                "bin_label": str(bin_label),
                "model_name": str(model_name),
                "n": int(len(group)),
                "positive_count": positive_count,
                "negative_count": negative_count,
                "positive_rate": float(y.mean()),
                "auroc": auroc,
                "brier_mean": float(group["squared_error"].mean()),
                "log_loss_mean": float(group["log_loss_component"].mean()),
                "accuracy_at_0_5": float((prediction == y).mean()),
                "insufficient_data": bool(insufficient),
            }
        )
        rows.append(record)
    return pd.DataFrame(rows)


def evaluate_model_eligibility_per_stratum(
    subgroup_metrics: pd.DataFrame,
    auroc_margin_guard: float | None = None,
) -> pd.DataFrame:
    """Mark models eligible for Brier ranking after an AUROC safety guard."""
    if auroc_margin_guard is not None and float(auroc_margin_guard) < 0:
        raise ValueError("auroc_margin_guard must be non-negative.")
    context_columns = [
        column
        for column in ("threshold_days", "task_name", "split", "stratify_by", "bin_label")
        if column in subgroup_metrics.columns
    ]
    if not {"split", "stratify_by", "bin_label"}.issubset(context_columns):
        raise KeyError("subgroup_metrics is missing required stratum context columns.")

    output_frames: list[pd.DataFrame] = []
    for _, group in subgroup_metrics.groupby(context_columns, sort=False, dropna=False):
        marked = group.copy()
        analyzable = marked.loc[
            (~marked["insufficient_data"].astype(bool))
            & np.isfinite(marked["auroc"].astype(float))
            & np.isfinite(marked["brier_mean"].astype(float))
        ]
        best_auroc = (
            float(analyzable["auroc"].max()) if not analyzable.empty else float("nan")
        )
        marked["best_auroc_in_stratum"] = best_auroc
        marked["auroc_gap_from_best"] = best_auroc - marked["auroc"].astype(float)
        marked["auroc_margin_guard"] = (
            float(auroc_margin_guard) if auroc_margin_guard is not None else float("nan")
        )
        marked["eligible_by_auroc_guard"] = False
        marked["eligibility_reason"] = "insufficient_data"
        valid_mask = (
            (~marked["insufficient_data"].astype(bool))
            & np.isfinite(marked["auroc"].astype(float))
            & np.isfinite(marked["brier_mean"].astype(float))
        )
        if auroc_margin_guard is None:
            eligible_mask = valid_mask
        else:
            eligible_mask = valid_mask & (
                marked["auroc_gap_from_best"] <= float(auroc_margin_guard) + 1e-12
            )
        marked.loc[valid_mask, "eligibility_reason"] = "excluded_by_auroc_guard"
        marked.loc[eligible_mask, "eligible_by_auroc_guard"] = True
        marked.loc[eligible_mask, "eligibility_reason"] = "eligible"
        output_frames.append(marked)
    if not output_frames:
        return subgroup_metrics.copy()
    return pd.concat(output_frames, ignore_index=True)


def rank_models_per_stratum(
    subgroup_metrics: pd.DataFrame,
    tie_margin: float = 0.005,
    auroc_margin_guard: float | None = None,
) -> pd.DataFrame:
    """Rank Brier performance among models that retain adequate AUROC."""
    eligibility = evaluate_model_eligibility_per_stratum(
        subgroup_metrics,
        auroc_margin_guard=auroc_margin_guard,
    )
    keys = [
        column
        for column in ("threshold_days", "task_name", "split", "stratify_by", "bin_label")
        if column in eligibility.columns
    ]
    rows: list[dict[str, Any]] = []
    for key, group in eligibility.groupby(keys, sort=False, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        eligible = group.loc[group["eligible_by_auroc_guard"]].sort_values(
            ["brier_mean", "model_name"], kind="stable"
        )
        record = dict(zip(keys, key))
        excluded_models = group.loc[
            ~group["eligible_by_auroc_guard"], "model_name"
        ].astype(str).tolist()
        if eligible.empty:
            record.update(
                {
                    "best_model": "insufficient",
                    "best_brier": float("nan"),
                    "best_model_auroc": float("nan"),
                    "second_model": "insufficient",
                    "second_brier": float("nan"),
                    "second_model_auroc": float("nan"),
                    "margin": float("nan"),
                    "practical_tie": False,
                    "best_auroc_in_stratum": float(group["best_auroc_in_stratum"].iloc[0]),
                    "auroc_margin_guard": (
                        float(auroc_margin_guard)
                        if auroc_margin_guard is not None
                        else float("nan")
                    ),
                    "eligible_model_count": 0,
                    "excluded_model_count": int(len(excluded_models)),
                    "excluded_models": "|".join(excluded_models),
                    "selection_rule": "no_eligible_model",
                }
            )
        else:
            first = eligible.iloc[0]
            second = eligible.iloc[1] if len(eligible) > 1 else None
            margin = (
                float(second["brier_mean"] - first["brier_mean"])
                if second is not None
                else float("nan")
            )
            record.update(
                {
                    "best_model": str(first["model_name"]),
                    "best_brier": float(first["brier_mean"]),
                    "best_model_auroc": float(first["auroc"]),
                    "second_model": (
                        str(second["model_name"]) if second is not None else "none"
                    ),
                    "second_brier": (
                        float(second["brier_mean"])
                        if second is not None
                        else float("nan")
                    ),
                    "second_model_auroc": (
                        float(second["auroc"])
                        if second is not None
                        else float("nan")
                    ),
                    "margin": margin,
                    "practical_tie": bool(
                        second is not None and margin <= float(tie_margin)
                    ),
                    "best_auroc_in_stratum": float(first["best_auroc_in_stratum"]),
                    "auroc_margin_guard": (
                        float(auroc_margin_guard)
                        if auroc_margin_guard is not None
                        else float("nan")
                    ),
                    "eligible_model_count": int(len(eligible)),
                    "excluded_model_count": int(len(excluded_models)),
                    "excluded_models": "|".join(excluded_models),
                    "selection_rule": (
                        "auroc_guard_then_lowest_brier"
                        if auroc_margin_guard is not None
                        else "lowest_brier"
                    ),
                }
            )
        rows.append(record)
    return pd.DataFrame(rows)

def _gee_interaction_test(frame: pd.DataFrame, stratify_by: str) -> dict[str, Any]:
    bin_column = f"{stratify_by}_bin"
    analysis = frame.loc[
        :, ["source_row_index", "model_name", bin_column, "squared_error"]
    ].copy()
    analysis["model_name"] = analysis["model_name"].astype("category")
    analysis[bin_column] = analysis[bin_column].astype("category")
    formula = f"squared_error ~ C(model_name) * C({bin_column})"
    try:
        model = GEE.from_formula(
            formula,
            groups="source_row_index",
            data=analysis,
            family=sm.families.Gaussian(),
            cov_struct=Exchangeable(),
        )
        result = model.fit(cov_type="robust", maxiter=100)
        table = result.wald_test_terms(skip_single=False, scalar=True).table
        interaction_rows = [index for index in table.index if ":" in str(index)]
        if not interaction_rows:
            raise RuntimeError("The GEE interaction term was not found.")
        interaction = table.loc[interaction_rows[0]]
        p_value_column = "pvalue" if "pvalue" in table.columns else "P>chi2"
        statistic_column = (
            "statistic" if "statistic" in table.columns else "chi2"
        )
        df_column = (
            "df_constraint"
            if "df_constraint" in table.columns
            else "df constraint"
        )
        p_value = float(interaction[p_value_column])
        statistic = float(interaction[statistic_column])
        df_constraint = int(interaction[df_column])
        return {
            "gee_status": "completed",
            "gee_formula": formula,
            "gee_family": "Gaussian",
            "gee_covariance": "robust_exchangeable",
            "gee_interaction_statistic": statistic,
            "gee_interaction_df": df_constraint,
            "gee_interaction_p_raw": p_value,
        }
    except Exception as exception:
        return {
            "gee_status": "failed",
            "gee_formula": formula,
            "gee_error": f"{type(exception).__name__}: {exception}",
            "gee_interaction_p_raw": float("nan"),
        }


def _cochran_tests(
    frame: pd.DataFrame,
    stratify_by: str,
    model_names: Iterable[str],
    default_threshold: float,
) -> list[dict[str, Any]]:
    bin_column = f"{stratify_by}_bin"
    rows: list[dict[str, Any]] = []
    for bin_label, group in frame.groupby(bin_column, observed=True, sort=False):
        correctness = group.assign(
            correct=(
                (group["y_prob"] >= default_threshold).astype(np.int32)
                == group["y_true"].astype(np.int32)
            ).astype(np.int32)
        ).pivot(index="source_row_index", columns="model_name", values="correct")
        expected_columns = list(model_names)
        missing = sorted(set(expected_columns) - set(correctness.columns))
        if missing or correctness[expected_columns].isna().any().any():
            rows.append(
                {
                    "bin_label": str(bin_label),
                    "status": "insufficient_or_unaligned",
                    "q_stat": float("nan"),
                    "p_raw": float("nan"),
                    "n_samples": int(len(correctness)),
                }
            )
            continue
        try:
            result = cochrans_q(correctness[expected_columns].to_numpy(dtype=np.int32))
            rows.append(
                {
                    "bin_label": str(bin_label),
                    "status": "completed",
                    "q_stat": float(result.statistic),
                    "p_raw": float(result.pvalue),
                    "n_samples": int(len(correctness)),
                }
            )
        except Exception as exception:
            rows.append(
                {
                    "bin_label": str(bin_label),
                    "status": "failed",
                    "error": f"{type(exception).__name__}: {exception}",
                    "q_stat": float("nan"),
                    "p_raw": float("nan"),
                    "n_samples": int(len(correctness)),
                }
            )
    return rows


def test_model_subgroup_interaction(
    frame: pd.DataFrame,
    stratify_by: str,
    model_names: Iterable[str] = DEFAULT_MODELS,
    default_threshold: float = 0.5,
) -> dict[str, Any]:
    result = {
        "stratify_by": stratify_by,
        "n_samples": int(frame["source_row_index"].nunique()),
        **_gee_interaction_test(frame, stratify_by),
        "cochrans_q": _cochran_tests(
            frame, stratify_by, model_names, default_threshold
        ),
    }
    return result


def apply_fdr_to_interaction_results(
    results: list[dict[str, Any]],
    method: str = "fdr_bh",
    alpha: float = 0.05,
) -> list[dict[str, Any]]:
    references: list[tuple[str, int, int | None]] = []
    p_values: list[float] = []
    for result_index, result in enumerate(results):
        p_value = result.get("gee_interaction_p_raw")
        if p_value is not None and np.isfinite(p_value):
            references.append(("gee", result_index, None))
            p_values.append(float(p_value))
        for q_index, q_result in enumerate(result.get("cochrans_q", [])):
            q_p = q_result.get("p_raw")
            if q_p is not None and np.isfinite(q_p):
                references.append(("cochran", result_index, q_index))
                p_values.append(float(q_p))

    if p_values:
        rejected, adjusted, _, _ = multipletests(
            p_values, alpha=alpha, method=method
        )
        for reference, adjusted_p, reject in zip(references, adjusted, rejected):
            kind, result_index, q_index = reference
            if kind == "gee":
                results[result_index]["gee_interaction_p_fdr"] = float(adjusted_p)
                results[result_index]["conclusion"] = (
                    "significant_after_fdr" if reject else "not_significant"
                )
            else:
                assert q_index is not None
                q_result = results[result_index]["cochrans_q"][q_index]
                q_result["p_fdr"] = float(adjusted_p)
                q_result["significant_after_fdr"] = bool(reject)

    for result in results:
        result.setdefault("gee_interaction_p_fdr", float("nan"))
        result.setdefault("conclusion", "not_testable")
        result["fdr_method"] = method
        result["fdr_alpha"] = float(alpha)
        result["fdr_family_size"] = int(len(p_values))
    return results


def plot_heterogeneity_heatmap(
    subgroup_metrics: pd.DataFrame,
    stratify_by: str,
    output_path: Path,
) -> None:
    data = subgroup_metrics.loc[
        subgroup_metrics["stratify_by"] == stratify_by
    ].copy()
    model_order = list(dict.fromkeys(data["model_name"].tolist()))
    bin_order = list(dict.fromkeys(data["bin_label"].tolist()))
    matrix = data.pivot(index="model_name", columns="bin_label", values="brier_mean")
    matrix = matrix.reindex(index=model_order, columns=bin_order)

    figure = plt.figure(figsize=(8, 5.5))
    axis = figure.add_subplot(111)
    image = axis.imshow(matrix.to_numpy(dtype=float), aspect="auto")
    axis.set_xticks(range(len(bin_order)))
    axis.set_xticklabels(bin_order)
    axis.set_yticks(range(len(model_order)))
    axis.set_yticklabels(model_order)
    axis.set_xlabel(f"{stratify_by} stratum")
    axis.set_ylabel("Model")
    axis.set_title(f"Model × {stratify_by} heterogeneity (Brier; lower is better)")

    lookup = data.set_index(["model_name", "bin_label"])
    for row, model_name in enumerate(model_order):
        for column, bin_label in enumerate(bin_order):
            if (model_name, bin_label) not in lookup.index:
                continue
            record = lookup.loc[(model_name, bin_label)]
            auroc = record["auroc"]
            auroc_text = "NA" if pd.isna(auroc) else f"{float(auroc):.3f}"
            axis.text(
                column,
                row,
                f"AUC {auroc_text}\nn={int(record['n'])}",
                ha="center",
                va="center",
                fontsize=8,
            )
    figure.colorbar(image, ax=axis, label="Mean squared error (Brier)")
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_error_vs_covariate(
    frame: pd.DataFrame,
    covariate: str,
    output_path: Path,
    trend_bins: int = 50,
) -> None:
    figure = plt.figure(figsize=(8, 5.5))
    axis = figure.add_subplot(111)
    for model_name, group in frame.groupby("model_name", sort=False):
        quantile_count = min(trend_bins, max(5, int(group[covariate].nunique())))
        try:
            bins = pd.qcut(group[covariate], q=quantile_count, duplicates="drop")
            trend = group.assign(_trend_bin=bins).groupby(
                "_trend_bin", observed=True
            ).agg(
                covariate_mean=(covariate, "mean"),
                squared_error_mean=("squared_error", "mean"),
            )
            axis.plot(
                trend["covariate_mean"],
                trend["squared_error_mean"],
                marker="o",
                markersize=3,
                label=str(model_name),
            )
        except ValueError:
            continue
    axis.set_xlabel(covariate)
    axis.set_ylabel("Mean squared prediction error")
    axis.set_title(f"Prediction error trend by {covariate}")
    axis.grid(True, alpha=0.25)
    axis.legend()
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def freeze_discovery_design(
    output_path: Path,
    config: HeterogeneityConfig,
    definitions: dict[str, StrataDefinition],
    interaction_results: list[dict[str, Any]],
) -> None:
    payload = {
        "design_status": "frozen",
        "threshold_days": int(config.threshold_days),
        "discovery_split": config.discovery_split,
        "model_names": list(config.model_names),
        "subgroup_variables": list(config.subgroup_variables),
        "minimum_bin_size": config.minimum_bin_size,
        "minimum_class_count": config.minimum_class_count,
        "default_probability_threshold": config.default_probability_threshold,
        "fdr_method": config.fdr_method,
        "fdr_alpha": config.fdr_alpha,
        "brier_tie_margin": config.brier_tie_margin,
        "strata_definitions": {
            variable: asdict(definition)
            for variable, definition in definitions.items()
        },
        "discovery_conclusions": {
            result["stratify_by"]: result.get("conclusion", "not_testable")
            for result in interaction_results
        },
    }
    atomic_write_json(output_path, payload)
    digest = sha256_file(output_path)
    output_path.with_suffix(output_path.suffix + ".sha256").write_text(
        f"{digest}  {output_path.name}\n", encoding="utf-8"
    )


def load_frozen_design(path: Path) -> dict[str, Any]:
    path = path.expanduser().resolve()
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if payload.get("design_status") != "frozen":
        raise ValueError("The supplied heterogeneity design is not frozen.")
    return payload


def _definition_from_payload(payload: dict[str, Any]) -> StrataDefinition:
    return StrataDefinition(
        variable=str(payload["variable"]),
        method=str(payload["method"]),
        labels=tuple(payload["labels"]),
        edges=tuple(float(value) for value in payload["edges"]),
        requested_n_bins=int(payload["requested_n_bins"]),
        actual_n_bins=int(payload["actual_n_bins"]),
        fitted_on_split=str(payload["fitted_on_split"]),
        minimum_bin_size=int(payload["minimum_bin_size"]),
        minimum_class_count=int(payload["minimum_class_count"]),
        duplicate_edges_dropped=bool(payload["duplicate_edges_dropped"]),
    )


def run_heterogeneity_discovery(
    config: HeterogeneityConfig,
    dataset_path: Path,
) -> Path:
    output_dir = (
        config.run_dir
        / _los_directory_name(config.threshold_days)
        / "heterogeneity_analysis"
        / "discovery_validation"
    )
    if (output_dir / "frozen_design.json").exists():
        raise FileExistsError(
            "A frozen discovery design already exists. Delete it only before "
            "any confirmatory analysis has been viewed, or use a new run copy. "
            f"Existing design: {output_dir / 'frozen_design.json'}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = load_predictions_with_covariates(
        config.run_dir,
        config.threshold_days,
        config.model_names,
        config.discovery_split,
        dataset_path,
        config.subgroup_variables,
    )
    all_metrics: list[pd.DataFrame] = []
    all_rankings: list[pd.DataFrame] = []
    definitions: dict[str, StrataDefinition] = {}
    interaction_results: list[dict[str, Any]] = []

    for variable in config.subgroup_variables:
        definition = fit_strata_definition(
            frame,
            variable,
            fitted_on_split=config.discovery_split,
            method="quantile",
            n_bins=config.n_bins,
            minimum_bin_size=config.minimum_bin_size,
            minimum_class_count=config.minimum_class_count,
        )
        definitions[variable] = definition
        stratified = apply_strata_definition(frame, definition)
        metrics = compute_subgroup_metrics(
            stratified,
            variable,
            config.minimum_class_count,
            config.minimum_bin_size,
            config.default_probability_threshold,
        )
        rankings = rank_models_per_stratum(metrics, config.brier_tie_margin)
        all_metrics.append(metrics)
        all_rankings.append(rankings)
        interaction_results.append(
            test_model_subgroup_interaction(
                stratified,
                variable,
                config.model_names,
                config.default_probability_threshold,
            )
        )
        plot_heterogeneity_heatmap(
            metrics,
            variable,
            output_dir / "figures" / f"heatmap_{variable}.png",
        )
        plot_error_vs_covariate(
            stratified,
            variable,
            output_dir / "figures" / f"error_vs_{variable}.png",
            config.trend_bins,
        )

    metrics_frame = pd.concat(all_metrics, ignore_index=True)
    ranking_frame = pd.concat(all_rankings, ignore_index=True)
    interaction_results = apply_fdr_to_interaction_results(
        interaction_results, config.fdr_method, config.fdr_alpha
    )
    metrics_frame.to_csv(output_dir / "subgroup_metrics.csv", index=False)
    ranking_frame.to_csv(output_dir / "best_model_per_stratum.csv", index=False)
    atomic_write_json(output_dir / "interaction_tests.json", interaction_results)
    atomic_write_json(
        output_dir / "strata_definitions.json",
        {key: asdict(value) for key, value in definitions.items()},
    )
    atomic_write_json(
        output_dir / "run_config.json",
        {
            **asdict(config),
            "run_dir": str(config.run_dir.resolve()),
            "dataset_path": str(dataset_path.expanduser().resolve()),
            "analysis_role": "discovery_only",
            "fdr_family": f"validation_LOS_{config.threshold_days:02d}",
        },
    )
    freeze_discovery_design(
        output_dir / "frozen_design.json",
        config,
        definitions,
        interaction_results,
    )
    return output_dir


def run_heterogeneity_confirmatory(
    run_dir: Path,
    split: str,
    dataset_path: Path,
    frozen_design_path: Path,
) -> Path:
    design = load_frozen_design(frozen_design_path)
    threshold_days = int(design["threshold_days"])
    model_names = tuple(design["model_names"])
    subgroup_variables = tuple(design["subgroup_variables"])
    output_dir = (
        run_dir.expanduser().resolve()
        / _los_directory_name(threshold_days)
        / "heterogeneity_analysis"
        / "confirmatory"
        / split
    )
    if (output_dir / "run_config.json").exists():
        raise FileExistsError(
            "This confirmatory split has already been analysed. Re-running it "
            "would violate the single-use reporting guard. "
            f"Existing output: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = load_predictions_with_covariates(
        run_dir,
        threshold_days,
        model_names,
        split,
        dataset_path,
        subgroup_variables,
    )
    all_metrics: list[pd.DataFrame] = []
    all_rankings: list[pd.DataFrame] = []
    interaction_results: list[dict[str, Any]] = []
    for variable in subgroup_variables:
        definition = _definition_from_payload(
            design["strata_definitions"][variable]
        )
        stratified = apply_strata_definition(frame, definition)
        metrics = compute_subgroup_metrics(
            stratified,
            variable,
            int(design["minimum_class_count"]),
            int(design["minimum_bin_size"]),
            float(design["default_probability_threshold"]),
        )
        all_metrics.append(metrics)
        all_rankings.append(
            rank_models_per_stratum(metrics, float(design["brier_tie_margin"]))
        )
        interaction_results.append(
            test_model_subgroup_interaction(
                stratified,
                variable,
                model_names,
                float(design["default_probability_threshold"]),
            )
        )
        plot_heterogeneity_heatmap(
            metrics,
            variable,
            output_dir / "figures" / f"heatmap_{variable}.png",
        )
        plot_error_vs_covariate(
            stratified,
            variable,
            output_dir / "figures" / f"error_vs_{variable}.png",
        )
    interaction_results = apply_fdr_to_interaction_results(
        interaction_results,
        str(design["fdr_method"]),
        float(design["fdr_alpha"]),
    )
    pd.concat(all_metrics, ignore_index=True).to_csv(
        output_dir / "subgroup_metrics.csv", index=False
    )
    pd.concat(all_rankings, ignore_index=True).to_csv(
        output_dir / "best_model_per_stratum.csv", index=False
    )
    atomic_write_json(output_dir / "interaction_tests.json", interaction_results)
    atomic_write_json(
        output_dir / "run_config.json",
        {
            "analysis_role": "confirmatory_single_use",
            "split": split,
            "dataset_path": str(dataset_path.expanduser().resolve()),
            "frozen_design_path": str(frozen_design_path.expanduser().resolve()),
            "frozen_design_sha256": sha256_file(
                frozen_design_path.expanduser().resolve()
            ),
            "no_refitting_of_strata": True,
        },
    )
    return output_dir
