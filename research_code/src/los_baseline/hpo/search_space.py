from __future__ import annotations

from typing import Any, Mapping

from ..svm_runtime import svm_runtime_settings


def suggest_model_config(
    trial: Any,
    model_name: str,
    base: Mapping[str, Any],
    project_config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    config = {**dict(base), "parameters": dict(dict(base).get("parameters", {}))}
    p = config["parameters"]
    if model_name == "logistic_regression":
        config["implementation"] = "sklearn"
        p["solver"] = "saga"
        p["C"] = trial.suggest_float("C", 1e-4, 1e3, log=True)
        p["l1_ratio"] = trial.suggest_float("l1_ratio", 0.0, 1.0)
        p["max_iter"] = 5000
        p["tol"] = trial.suggest_float("tol", 1e-6, 1e-3, log=True)
        p["fit_intercept"] = True
        p.pop("penalty", None)
        p.pop("n_jobs", None)
    elif model_name == "decision_tree":
        p["criterion"] = trial.suggest_categorical("criterion", ["gini", "entropy", "log_loss"])
        p["max_depth"] = trial.suggest_int("max_depth", 2, 20)
        p["min_samples_split"] = trial.suggest_int("min_samples_split", 2, 100, log=True)
        p["min_samples_leaf"] = trial.suggest_int("min_samples_leaf", 2, 100, log=True)
        p["max_features"] = trial.suggest_float("max_features", 0.3, 1.0)
        p["ccp_alpha"] = trial.suggest_float("ccp_alpha", 1e-8, 1e-1, log=True)
    elif model_name == "random_forest":
        p["n_estimators"] = trial.suggest_int("n_estimators", 300, 1500, step=100)
        p["criterion"] = trial.suggest_categorical("criterion", ["gini", "entropy", "log_loss"])
        p["max_depth"] = trial.suggest_int("max_depth", 4, 30)
        p["min_samples_split"] = trial.suggest_int("min_samples_split", 2, 100, log=True)
        p["min_samples_leaf"] = trial.suggest_int("min_samples_leaf", 2, 100, log=True)
        p["max_features"] = trial.suggest_float("max_features", 0.2, 1.0)
        p["bootstrap"] = True
        p["max_samples"] = trial.suggest_float("max_samples", 0.5, 1.0)
    elif model_name == "svm":
        runtime = svm_runtime_settings(project_config)
        if runtime["enabled"]:
            backends = [
                value for value in runtime["search_backends"]
                if value in {"exact_rbf", "nystroem_linear"}
            ]
            if not backends:
                raise ValueError("svm_runtime_optimization.search_backends is empty.")
            backend = trial.suggest_categorical("svm_backend", backends)
            p["svm_backend"] = backend
            p["probability"] = False
            p["decision_function_shape"] = "ovr"
            if backend == "exact_rbf":
                spec = runtime["exact_rbf"]
                p["kernel"] = "rbf"
                p["C"] = trial.suggest_float("C", spec["c_min"], spec["c_max"], log=True)
                p["gamma"] = trial.suggest_float("gamma", spec["gamma_min"], spec["gamma_max"], log=True)
                p["max_iter"] = int(spec["max_iter"])
                p["tol"] = float(spec["tol"])
                p["cache_size"] = max(1024, int(p.get("cache_size", 2048)))
                p.pop("nystroem_components", None)
                p.pop("nystroem_gamma", None)
                p.pop("linear_svc_max_iter", None)
                p.pop("linear_svc_tol", None)
            else:
                spec = runtime["nystroem_linear"]
                p["C"] = trial.suggest_float("C", spec["c_min"], spec["c_max"], log=True)
                p["nystroem_components"] = trial.suggest_int(
                    "nystroem_components", spec["n_components_min"], spec["n_components_max"], log=True
                )
                p["nystroem_gamma"] = trial.suggest_float(
                    "nystroem_gamma", spec["gamma_min"], spec["gamma_max"], log=True
                )
                p["linear_svc_max_iter"] = int(spec["max_iter"])
                p["linear_svc_tol"] = float(spec["tol"])
                # These are SVC-only and must never leak into LinearSVC.
                p.pop("kernel", None)
                p.pop("gamma", None)
                p.pop("cache_size", None)
                p.pop("max_iter", None)
                p.pop("tol", None)
        else:
            # Historical V7.3 search remains available for old configs/tests.
            p["kernel"] = trial.suggest_categorical("kernel", ["linear", "rbf"])
            p["C"] = trial.suggest_float("C", 1e-3, 1e3, log=True)
            if p["kernel"] == "rbf":
                p["gamma"] = trial.suggest_float("gamma", 1e-6, 1e1, log=True)
            else:
                p.pop("gamma", None)
            p["probability"] = False
            p["decision_function_shape"] = "ovr"
            p["cache_size"] = max(1024, int(p.get("cache_size", 2048)))
    elif model_name == "xgboost":
        p["n_estimators"] = 3000
        p["early_stopping_rounds"] = trial.suggest_int("early_stopping_rounds", 50, 125, step=25)
        p["learning_rate"] = trial.suggest_float("learning_rate", 3e-3, 0.2, log=True)
        p["max_depth"] = trial.suggest_int("max_depth", 2, 10)
        p["min_child_weight"] = trial.suggest_float("min_child_weight", 0.5, 50.0, log=True)
        p["subsample"] = trial.suggest_float("subsample", 0.5, 1.0)
        p["colsample_bytree"] = trial.suggest_float("colsample_bytree", 0.4, 1.0)
        p["gamma"] = trial.suggest_float("gamma", 1e-8, 10.0, log=True)
        p["reg_alpha"] = trial.suggest_float("reg_alpha", 1e-8, 100.0, log=True)
        p["reg_lambda"] = trial.suggest_float("reg_lambda", 1e-4, 1000.0, log=True)
        p["max_delta_step"] = trial.suggest_float("max_delta_step", 0.0, 10.0)
    else:
        raise KeyError(model_name)
    config.setdefault("training_policy", {})["class_weight_mode"] = "none"
    return config
