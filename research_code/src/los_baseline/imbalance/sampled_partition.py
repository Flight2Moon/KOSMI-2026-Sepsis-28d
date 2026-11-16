from __future__ import annotations

import json
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from ..exp123_data import TaskSplitData
from ..models import build_task_sklearn_model, resolve_class_weights, uses_sklearn_logistic
from ..optimization_objectives import threshold_free_metrics
from ..preprocessing import build_preprocessor, transform_features, validate_training_features
from ..task_spec import TaskDefinition
from ..training_engine import (
    predict_task_optimization_output,
    validate_task_optimization_output_contract,
)
from .sampling import TrainingSampler, sampling_manifest_payload


def fit_predict_sampled_partition(
    *,
    model_name: str,
    model_config: Mapping[str, Any],
    task: TaskDefinition,
    feature_names: Sequence[str],
    natural_train: TaskSplitData,
    evaluation_features: pd.DataFrame,
    evaluation_labels: np.ndarray,
    sampler: TrainingSampler,
    seed: int,
) -> tuple[np.ndarray, dict[str, float], dict[str, Any], dict[str, Any]]:
    """Threshold-free fold evaluation with preprocessing fit before sampling."""
    if model_name == "logistic_regression" and not uses_sklearn_logistic(dict(model_config)):
        raise ValueError(
            "V8 sampled-track partition evaluation requires the sklearn logistic "
            "implementation. T2-selected parameters are expected to satisfy this."
        )
    validate_training_features(natural_train.raw_features, list(feature_names))
    sampled = sampler.sample(
        natural_train,
        feature_names=feature_names,
        seed=int(seed),
        purpose="sampling_search_fold",
    )
    preprocessor = build_preprocessor(model_name)
    preprocessor.fit(natural_train.raw_features.loc[:, list(feature_names)])
    X_fit = transform_features(preprocessor, sampled.raw_features.loc[:, list(feature_names)])
    X_eval = transform_features(preprocessor, evaluation_features.loc[:, list(feature_names)])

    effective = json.loads(json.dumps(dict(model_config), default=str))
    effective.setdefault("training_policy", {})["class_weight_mode"] = "none"
    parameters = dict(effective.get("parameters", {}))
    # T2-selected XGBoost configurations should already contain the selected
    # tree count. Defensive removal prevents a stale early-stopping parameter
    # from requesting an evaluation set during sampled sensitivity analysis.
    parameters.pop("early_stopping_rounds", None)
    effective["parameters"] = parameters
    class_weights, weight_method = resolve_class_weights(
        sampled.labels, task.expected_classes, effective
    )
    if weight_method != "none":
        raise ValueError("Sampled V8 tracks must keep class_weight_mode='none'.")
    estimator = build_task_sklearn_model(
        model_name=model_name,
        model_config=effective,
        seed=int(seed),
        task_type=task.task_type,
        num_classes=task.num_classes,
        class_weights=class_weights,
    )
    estimator.fit(X_fit, sampled.labels)
    predictions, prediction_kind = predict_task_optimization_output(
        model_name=model_name,
        model=estimator,
        transformed=X_eval,
        task=task,
        model_config=effective,
    )
    validate_task_optimization_output_contract(
        predictions, task, len(evaluation_labels), prediction_kind
    )
    metrics = threshold_free_metrics(
        np.asarray(evaluation_labels, dtype=np.int64),
        predictions,
        task,
        prediction_kind=prediction_kind,
    )
    diagnostics = {
        "prediction_kind": prediction_kind,
        "sampling": sampling_manifest_payload(sampled),
        "preprocessor_fit_count": int(len(natural_train.labels)),
        "model_fit_count": int(len(sampled.labels)),
        "evaluation_count": int(len(evaluation_labels)),
        "evaluation_distribution_modified": False,
    }
    return predictions, metrics, diagnostics, effective
