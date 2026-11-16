from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class ChoiceSpace:
    values: tuple[Any, ...]

    def __post_init__(self) -> None:
        if not self.values:
            raise ValueError("ChoiceSpace values must not be empty.")

    def sample(self, rng: random.Random) -> Any:
        return self.values[rng.randrange(len(self.values))]

    def neighborhood(self, anchor: Any) -> "ChoiceSpace":
        if anchor not in self.values:
            return self
        index = self.values.index(anchor)
        start = max(0, index - 1)
        stop = min(len(self.values), index + 2)
        return ChoiceSpace(self.values[start:stop])

    def to_dict(self) -> dict[str, Any]:
        return {"type": "choice", "values": list(self.values)}


MODEL_SEARCH_SPACES: dict[str, dict[str, ChoiceSpace]] = {
    "decision_tree": {
        "max_depth": ChoiceSpace((3, 5, 7, 10, 15, None)),
        "min_samples_split": ChoiceSpace((2, 5, 10, 20, 50, 100)),
        "min_samples_leaf": ChoiceSpace((1, 5, 10, 20, 50)),
        "max_features": ChoiceSpace((None, "sqrt", "log2", 0.5, 0.8)),
        "ccp_alpha": ChoiceSpace((0.0, 0.0001, 0.001, 0.01)),
    },
    "random_forest": {
        "n_estimators": ChoiceSpace((100, 200, 300, 500, 800)),
        "max_depth": ChoiceSpace((4, 6, 8, 10, 15)),
        "min_samples_split": ChoiceSpace((2, 5, 10, 20)),
        "min_samples_leaf": ChoiceSpace((2, 4, 8, 10, 20, 50)),
        "max_features": ChoiceSpace(("sqrt", "log2", 0.3, 0.5, 0.8)),
        "bootstrap": ChoiceSpace((True,)),
    },
    "svm": {
        "C": ChoiceSpace((0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0)),
        "gamma": ChoiceSpace(("scale", "auto", 0.0001, 0.001, 0.01, 0.1)),
        "kernel": ChoiceSpace(("rbf",)),
        "tol": ChoiceSpace((0.0001, 0.001, 0.01)),
        "cache_size": ChoiceSpace((2048, 4096)),
        "probability": ChoiceSpace((True,)),
    },
    "xgboost": {
        "n_estimators": ChoiceSpace((100, 200, 300, 500, 800)),
        "max_depth": ChoiceSpace((2, 3, 4, 5, 6, 8)),
        "learning_rate": ChoiceSpace((0.005, 0.01, 0.03, 0.05, 0.1, 0.2)),
        "min_child_weight": ChoiceSpace((1, 3, 5, 10, 20)),
        "subsample": ChoiceSpace((0.6, 0.7, 0.8, 0.9, 1.0)),
        "colsample_bytree": ChoiceSpace((0.5, 0.7, 0.8, 0.9, 1.0)),
        "gamma": ChoiceSpace((0.0, 0.1, 0.5, 1.0, 5.0)),
        "reg_alpha": ChoiceSpace((0.0, 0.01, 0.1, 1.0, 10.0)),
        "reg_lambda": ChoiceSpace((0.1, 1.0, 5.0, 10.0, 50.0)),
    },
    "logistic_regression": {
        "learning_rate": ChoiceSpace(
            (0.00001, 0.00003, 0.0001, 0.0003, 0.001, 0.003, 0.01)
        ),
        "weight_decay": ChoiceSpace(
            (0.0, 0.000001, 0.00001, 0.0001, 0.001, 0.01)
        ),
        "batch_size": ChoiceSpace((64, 128, 256, 512, 1024)),
        "epochs": ChoiceSpace((50, 100, 200, 400)),
        "optimizer": ChoiceSpace(("adam", "adamw", "sgd")),
        "metric_interval": ChoiceSpace((5, 10, 20)),
    },
}


SMOKE_OVERRIDES: dict[str, dict[str, Any]] = {
    "decision_tree": {
        "max_depth": 4,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "max_features": None,
        "ccp_alpha": 0.0,
    },
    "random_forest": {
        "n_estimators": 10,
        "max_depth": 5,
        "min_samples_split": 2,
        "min_samples_leaf": 2,
        "max_features": "sqrt",
        "bootstrap": True,
        "n_jobs": 1,
    },
    "svm": {
        "C": 1.0,
        "gamma": "scale",
        "kernel": "rbf",
        "tol": 0.001,
        "cache_size": 256,
        "probability": True,
    },
    "xgboost": {
        "n_estimators": 10,
        "max_depth": 3,
        "learning_rate": 0.1,
        "min_child_weight": 1,
        "subsample": 1.0,
        "colsample_bytree": 1.0,
        "gamma": 0.0,
        "reg_alpha": 0.0,
        "reg_lambda": 1.0,
        "tree_method": "hist",
        "n_jobs": 1,
        "verbosity": 0,
    },
    "logistic_regression": {
        "learning_rate": 0.01,
        "weight_decay": 0.0,
        "batch_size": 32,
        "epochs": 3,
        "optimizer": "adam",
        "metric_interval": 1,
    },
}


SMOKE_VARIANTS: dict[str, tuple[dict[str, Any], ...]] = {
    "decision_tree": (
        {"max_depth": 3, "min_samples_leaf": 1},
        {"max_depth": 4, "min_samples_leaf": 1},
        {"max_depth": 5, "min_samples_leaf": 2},
        {"max_depth": 7, "min_samples_leaf": 5},
        {"max_depth": 10, "min_samples_leaf": 10},
    ),
    "random_forest": (
        {"n_estimators": 10, "max_depth": 3},
        {"n_estimators": 15, "max_depth": 4},
        {"n_estimators": 20, "max_depth": 5},
        {"n_estimators": 25, "max_depth": 6},
        {"n_estimators": 30, "max_depth": 7},
    ),
    "svm": (
        {"C": 0.1, "gamma": "scale"},
        {"C": 0.3, "gamma": "scale"},
        {"C": 1.0, "gamma": "scale"},
        {"C": 3.0, "gamma": "scale"},
        {"C": 1.0, "gamma": 0.01},
    ),
    "xgboost": (
        {"n_estimators": 10, "max_depth": 2, "learning_rate": 0.1},
        {"n_estimators": 15, "max_depth": 3, "learning_rate": 0.1},
        {"n_estimators": 20, "max_depth": 3, "learning_rate": 0.05},
        {"n_estimators": 25, "max_depth": 4, "learning_rate": 0.05},
        {"n_estimators": 30, "max_depth": 4, "learning_rate": 0.03},
    ),
    "logistic_regression": (
        {"epochs": 3, "learning_rate": 0.01},
        {"epochs": 4, "learning_rate": 0.003},
        {"epochs": 5, "learning_rate": 0.001},
        {"epochs": 4, "learning_rate": 0.01, "weight_decay": 0.0001},
        {"epochs": 5, "learning_rate": 0.003, "optimizer": "adamw"},
    ),
}


def _stable_trial_seed(
    seed: int,
    task_name: str,
    model_name: str,
    search_stage: str,
    trial_number: int,
) -> int:
    payload = json.dumps(
        {
            "seed": int(seed),
            "task_name": task_name,
            "model_name": model_name,
            "search_stage": search_stage,
            "trial_number": int(trial_number),
        },
        sort_keys=True,
    ).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def get_search_space(
    model_name: str,
    search_stage: str,
    anchor_parameters: Mapping[str, Any] | None = None,
) -> dict[str, ChoiceSpace]:
    try:
        base = MODEL_SEARCH_SPACES[model_name]
    except KeyError as exception:
        raise KeyError(f"Unsupported optimization model: {model_name}") from exception
    if search_stage in {"smoke", "coarse"}:
        return dict(base)
    if search_stage != "fine":
        raise ValueError(f"Unsupported search stage: {search_stage}")
    if not anchor_parameters:
        raise ValueError("Fine search requires anchor_parameters from coarse search.")
    return {
        name: space.neighborhood(anchor_parameters.get(name))
        for name, space in base.items()
    }


def _merge_sampled_parameters(
    model_name: str,
    base_model_config: Mapping[str, Any],
    sampled: Mapping[str, Any],
) -> dict[str, Any]:
    result = dict(base_model_config)
    if model_name == "logistic_regression":
        result.update(sampled)
        result.setdefault("device", "auto")
        return result
    parameters = dict(result.get("parameters", {}))
    parameters.update(sampled)
    if model_name == "random_forest":
        parameters.setdefault("n_jobs", -1)
    if model_name == "xgboost":
        parameters.setdefault("tree_method", "hist")
        parameters.setdefault("device", "cpu")
        parameters.setdefault("n_jobs", 1)
        parameters.setdefault("verbosity", 1)
    result["parameters"] = parameters
    return result


def sample_model_config(
    model_name: str,
    base_model_config: Mapping[str, Any],
    task_name: str,
    search_stage: str,
    trial_number: int,
    seed: int,
    anchor_parameters: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if search_stage == "smoke":
        overrides = dict(SMOKE_OVERRIDES[model_name])
        variants = SMOKE_VARIANTS[model_name]
        overrides.update(variants[trial_number % len(variants)])
        return _merge_sampled_parameters(model_name, base_model_config, overrides)

    space = get_search_space(model_name, search_stage, anchor_parameters)
    rng = random.Random(
        _stable_trial_seed(seed, task_name, model_name, search_stage, trial_number)
    )
    sampled = {name: value_space.sample(rng) for name, value_space in space.items()}
    return _merge_sampled_parameters(model_name, base_model_config, sampled)


def extract_tunable_parameters(
    model_name: str,
    model_config: Mapping[str, Any],
) -> dict[str, Any]:
    if model_name == "logistic_regression":
        names = set(MODEL_SEARCH_SPACES[model_name])
        return {name: model_config.get(name) for name in sorted(names)}
    parameters = dict(model_config.get("parameters", {}))
    names = set(MODEL_SEARCH_SPACES[model_name])
    return {name: parameters.get(name) for name in sorted(names)}


def search_space_manifest(
    model_name: str,
    search_stage: str,
    anchor_parameters: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    space = get_search_space(model_name, search_stage, anchor_parameters)
    return {
        "model_name": model_name,
        "search_stage": search_stage,
        "anchor_parameters": dict(anchor_parameters or {}),
        "parameters": {name: value.to_dict() for name, value in space.items()},
    }


def available_models() -> Sequence[str]:
    return tuple(MODEL_SEARCH_SPACES)
