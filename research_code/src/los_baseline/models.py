from __future__ import annotations

import copy
import os
from typing import Any

from sklearn.ensemble import RandomForestClassifier
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier

from .svm_runtime import (
    SVM_BACKEND_EXACT_RBF,
    SVM_BACKEND_NYSTROEM_LINEAR,
    resolve_svm_backend,
)


# Keep native numerical libraries conservative on macOS and shared servers.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")


def _assigned_cpu_threads(default: int) -> int:
    """Resolve the scheduler-assigned per-process CPU budget."""
    raw = os.environ.get("EXP123_ASSIGNED_CPU_THREADS", "").strip()
    if not raw:
        return max(1, int(default))
    try:
        return max(1, int(raw))
    except ValueError:
        return max(1, int(default))


def build_sklearn_model(
    model_name: str,
    model_config: dict[str, Any],
    seed: int,
):
    parameters = dict(model_config.get("parameters", {}))

    if model_name == "decision_tree":
        parameters.setdefault("max_depth", 5)
        parameters.setdefault("min_samples_split", 10)
        parameters.setdefault("class_weight", "balanced")
        parameters.setdefault("random_state", seed)
        return DecisionTreeClassifier(**parameters)

    if model_name == "random_forest":
        parameters.setdefault("n_estimators", 100)
        parameters.setdefault("max_depth", 10)
        parameters.setdefault("n_jobs", -1)
        parameters.setdefault("class_weight", "balanced")
        parameters.setdefault("random_state", seed)
        return RandomForestClassifier(**parameters)

    if model_name == "svm":
        parameters.setdefault("kernel", "rbf")
        parameters.setdefault("C", 1.0)
        parameters.setdefault("gamma", "scale")
        parameters.setdefault("probability", True)
        parameters.setdefault("class_weight", "balanced")
        parameters.setdefault("cache_size", 1024)
        parameters.setdefault("random_state", seed)
        return SVC(**parameters)

    if model_name == "xgboost":
        from xgboost import XGBClassifier

        parameters.setdefault("n_estimators", 100)
        parameters.setdefault("max_depth", 6)
        parameters.setdefault("learning_rate", 0.1)
        parameters.setdefault("objective", "binary:logistic")
        parameters.setdefault("eval_metric", "logloss")
        parameters.setdefault("tree_method", "hist")
        parameters.setdefault("device", "cpu")
        parameters.setdefault("n_jobs", 1)
        parameters.setdefault("random_state", seed)
        parameters.setdefault("verbosity", 1)
        return XGBClassifier(**parameters)

    raise KeyError(f"Unsupported sklearn model: {model_name}")


def compute_balanced_class_weights(
    labels: Any,
    expected_classes: Any,
) -> dict[int, float]:
    """Return deterministic inverse-frequency weights from the training labels."""
    import numpy as np

    y = np.asarray(labels, dtype=np.int64).reshape(-1)
    classes = tuple(int(value) for value in expected_classes)
    if not classes:
        raise ValueError("expected_classes must not be empty.")
    if len(y) == 0:
        raise ValueError("Training labels must not be empty.")
    invalid = sorted(set(np.unique(y).astype(int).tolist()) - set(classes))
    if invalid:
        raise ValueError(f"Training labels contain unexpected classes: {invalid}")
    counts = {class_code: int((y == class_code).sum()) for class_code in classes}
    missing = [class_code for class_code, count in counts.items() if count == 0]
    if missing:
        raise ValueError(f"Training labels are missing required classes: {missing}")
    denominator_classes = float(len(classes))
    total = float(len(y))
    return {
        class_code: total / (denominator_classes * float(count))
        for class_code, count in counts.items()
    }



def resolve_class_weights(
    labels: Any,
    expected_classes: Any,
    model_config: dict[str, Any] | None = None,
) -> tuple[dict[int, float], str]:
    """Resolve the frozen track-level class-weight policy.

    Track 1/2/3 use uniform weights in the primary analysis. Legacy balanced
    weighting remains available only when explicitly requested.
    """
    config = dict(model_config or {})
    policy = dict(config.get("training_policy", {}))
    mode = str(policy.get("class_weight_mode", "balanced")).lower()
    classes = tuple(int(value) for value in expected_classes)
    if mode in {"none", "uniform", "off", "disabled"}:
        return ({class_code: 1.0 for class_code in classes}, "none")
    if mode in {"balanced", "inverse_frequency"}:
        return (compute_balanced_class_weights(labels, classes), "balanced_inverse_frequency")
    raise ValueError(f"Unsupported class_weight_mode: {mode}")


def uses_sklearn_logistic(model_config: dict[str, Any] | None) -> bool:
    config = dict(model_config or {})
    return str(config.get("implementation", "torch")).lower() in {
        "sklearn", "scikit-learn", "scikit_learn"
    }


def normalize_task_model_config(
    model_name: str,
    model_config: dict[str, Any],
    *,
    task_type: str,
    num_classes: int,
    hpo_mode: bool = False,
) -> dict[str, Any]:
    """Return a task-compatible model configuration.

    The packaged configuration historically stored binary XGBoost defaults and
    SVC probability settings shared by every task.  This adapter makes the
    effective task contract explicit before fitting or persisting a candidate.
    """
    config = copy.deepcopy(dict(model_config))
    parameters = dict(config.get("parameters", {}))

    if model_name == "svm":
        if hpo_mode:
            parameters["probability"] = False
            parameters["decision_function_shape"] = "ovr"
        else:
            parameters.setdefault("probability", True)
            parameters.setdefault("decision_function_shape", "ovr")

    if model_name == "xgboost":
        if task_type == "binary":
            parameters["objective"] = "binary:logistic"
            parameters["eval_metric"] = "logloss"
            parameters.pop("num_class", None)
        elif task_type == "multiclass":
            parameters["objective"] = "multi:softprob"
            parameters["num_class"] = int(num_classes)
            parameters["eval_metric"] = "mlogloss"
        else:
            raise ValueError(f"Unsupported task_type: {task_type}")

    config["parameters"] = parameters
    return config



def _build_task_svm_model(
    *,
    model_config: dict[str, Any],
    parameters: dict[str, Any],
    seed: int,
    class_weights: dict[int, float],
):
    """Build either the exact RBF SVC or the V8 Nystroem approximation.

    Legacy configurations without ``svm_backend`` retain the historical SVC
    behavior so previously frozen V7/V8 artifacts remain reproducible.
    """
    backend = resolve_svm_backend(model_config)
    if backend == SVM_BACKEND_NYSTROEM_LINEAR:
        c_value = float(parameters.get("C", 1.0))
        gamma = float(parameters.get("nystroem_gamma", parameters.get("gamma", 0.1)))
        n_components = max(2, int(parameters.get("nystroem_components", 1024)))
        max_iter = max(1, int(parameters.get("linear_svc_max_iter", parameters.get("max_iter", 20_000))))
        tol = float(parameters.get("linear_svc_tol", parameters.get("tol", 1e-4)))
        feature_map = Nystroem(
            kernel="rbf",
            gamma=gamma,
            n_components=n_components,
            random_state=seed,
        )
        classifier = LinearSVC(
            C=c_value,
            class_weight=dict(class_weights),
            max_iter=max_iter,
            tol=tol,
            dual="auto",
            random_state=seed,
        )
        return SklearnPipeline([
            ("nystroem", feature_map),
            ("linear_svc", classifier),
        ])

    # exact_rbf and legacy_svc both use SVC.  ``exact_rbf`` is explicit and
    # receives the narrowed V8 runtime/search parameters; legacy configs keep
    # their original kernel choice.
    svc_parameters = dict(parameters)
    svc_parameters.pop("svm_backend", None)
    svc_parameters.pop("nystroem_components", None)
    svc_parameters.pop("nystroem_gamma", None)
    svc_parameters.pop("linear_svc_max_iter", None)
    svc_parameters.pop("linear_svc_tol", None)
    if backend == SVM_BACKEND_EXACT_RBF:
        svc_parameters["kernel"] = "rbf"
    svc_parameters.setdefault("kernel", "rbf")
    svc_parameters.setdefault("C", 1.0)
    svc_parameters.setdefault("gamma", "scale")
    svc_parameters.setdefault("probability", True)
    svc_parameters.setdefault("decision_function_shape", "ovr")
    svc_parameters["class_weight"] = dict(class_weights)
    svc_parameters.setdefault("cache_size", 1024)
    svc_parameters.setdefault("random_state", seed)
    return SVC(**svc_parameters)


def build_task_sklearn_model(
    model_name: str,
    model_config: dict[str, Any],
    seed: int,
    task_type: str,
    num_classes: int,
    class_weights: dict[int, float],
):
    """Build an EXP123 sklearn/XGBoost model without changing legacy builders."""
    normalized_config = normalize_task_model_config(
        model_name,
        model_config,
        task_type=task_type,
        num_classes=num_classes,
        hpo_mode=False,
    )
    parameters = dict(normalized_config.get("parameters", {}))

    if model_name == "logistic_regression":
        parameters.setdefault("solver", "saga")
        parameters.setdefault("C", 1.0)
        parameters.setdefault("l1_ratio", 0.0)
        parameters.setdefault("max_iter", 5000)
        parameters.setdefault("tol", 1e-5)
        parameters.setdefault("fit_intercept", True)
        parameters.setdefault("random_state", seed)
        parameters.pop("penalty", None)
        parameters.pop("n_jobs", None)
        parameters["class_weight"] = dict(class_weights)
        return LogisticRegression(**parameters)

    if model_name == "decision_tree":
        parameters.setdefault("max_depth", 5)
        parameters.setdefault("min_samples_split", 10)
        parameters["class_weight"] = dict(class_weights)
        parameters.setdefault("random_state", seed)
        return DecisionTreeClassifier(**parameters)

    if model_name == "random_forest":
        parameters.setdefault("n_estimators", 100)
        parameters.setdefault("max_depth", 10)
        if os.environ.get("EXP123_ASSIGNED_CPU_THREADS", "").strip():
            parameters["n_jobs"] = _assigned_cpu_threads(1)
        else:
            parameters.setdefault("n_jobs", -1)
        parameters["class_weight"] = dict(class_weights)
        parameters.setdefault("random_state", seed)
        return RandomForestClassifier(**parameters)

    if model_name == "svm":
        return _build_task_svm_model(
            model_config=normalized_config,
            parameters=parameters,
            seed=seed,
            class_weights=class_weights,
        )

    if model_name == "xgboost":
        from xgboost import XGBClassifier

        parameters.setdefault("n_estimators", 100)
        parameters.setdefault("max_depth", 6)
        parameters.setdefault("learning_rate", 0.1)
        # objective/eval_metric were normalized above.  Assignment rather than
        # setdefault is intentional because the shared config stores binary
        # defaults that are invalid for EXP1 multiclass training.
        parameters.setdefault("tree_method", "hist")
        parameters.setdefault("device", "cpu")
        if os.environ.get("EXP123_ASSIGNED_CPU_THREADS", "").strip():
            parameters["n_jobs"] = _assigned_cpu_threads(1)
        else:
            parameters.setdefault("n_jobs", 1)
        parameters.setdefault("random_state", seed)
        parameters.setdefault("verbosity", 1)
        return XGBClassifier(**parameters)

    raise KeyError(f"Unsupported EXP123 sklearn model: {model_name}")


def training_sample_weights(
    labels: Any,
    class_weights: dict[int, float],
):
    import numpy as np

    y = np.asarray(labels, dtype=np.int64).reshape(-1)
    return np.asarray(
        [class_weights[int(value)] for value in y],
        dtype=np.float64,
    )
