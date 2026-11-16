from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch

from .preprocessing import transform_features
from .torch_model import (
    LogisticRegressionForMimic4,
    predict_torch_probabilities,
    select_device,
)
from .utils import atomic_write_json
from .xgboost_device_adapter import predict_proba_device_aware


def _xgboost_device_from_metadata(metadata: dict[str, Any]) -> str:
    assigned = os.environ.get("EXP123_CUDA_DEVICE")
    raw_config = metadata.get("model_config_effective", metadata.get("model_config", {}))
    if not isinstance(raw_config, dict):
        raw_config = {}
    nested = raw_config.get("parameters")
    parameters = dict(nested) if isinstance(nested, dict) else dict(raw_config)
    requested = str(parameters.get("device", "cpu"))
    if assigned not in {None, ""} and requested.startswith("cuda"):
        requested = f"cuda:{int(assigned)}"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    return requested


@dataclass
class LoadedArtifact:
    model_name: str
    feature_names: list[str]
    threshold_days: float
    validation_decision_threshold: float
    metadata: dict[str, Any]
    preprocessor: Any
    model: Any
    device: torch.device | None = None

    def predict_proba(self, raw_features: pd.DataFrame) -> np.ndarray:
        ordered = raw_features.loc[:, self.feature_names]
        X = transform_features(self.preprocessor, ordered)

        if self.model_name == "logistic_regression":
            assert self.device is not None
            return predict_torch_probabilities(
                self.model,
                X,
                self.device,
            )

        output = np.asarray(self.model.predict_proba(X))
        if output.ndim == 2 and output.shape[1] >= 2:
            return output[:, 1].astype(np.float64)
        return output.reshape(-1).astype(np.float64)


def save_artifact(
    artifact_dir: Path,
    model_name: str,
    model: Any,
    preprocessor: Any,
    metadata: dict[str, Any],
) -> None:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessor, artifact_dir / "preprocessor.joblib")

    if model_name == "logistic_regression":
        torch.save(
            {
                "model_state_dict": {
                    key: value.detach().cpu()
                    for key, value in model.state_dict().items()
                },
                "input_dim": len(metadata["feature_names"]),
            },
            artifact_dir / "model.pt",
        )
    elif model_name == "xgboost":
        model.save_model(str(artifact_dir / "model.json"))
    else:
        joblib.dump(model, artifact_dir / "model.joblib")

    atomic_write_json(artifact_dir / "metadata.json", metadata)


def _load_torch_checkpoint(path: Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_artifact(artifact_dir: Path) -> LoadedArtifact:
    artifact_dir = artifact_dir.expanduser().resolve()
    with (artifact_dir / "metadata.json").open("r", encoding="utf-8") as file:
        metadata = json.load(file)

    model_name = str(metadata["model_name"])
    feature_names = list(metadata["feature_names"])
    preprocessor = joblib.load(artifact_dir / "preprocessor.joblib")
    device = None

    if model_name == "logistic_regression":
        checkpoint = _load_torch_checkpoint(artifact_dir / "model.pt")
        device = select_device(str(metadata.get("inference_device", "auto")))
        model = LogisticRegressionForMimic4(
            int(checkpoint["input_dim"])
        ).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
    elif model_name == "xgboost":
        from xgboost import XGBClassifier

        model = XGBClassifier(n_jobs=1, device=_xgboost_device_from_metadata(metadata))
        model.load_model(str(artifact_dir / "model.json"))
    else:
        model = joblib.load(artifact_dir / "model.joblib")

    return LoadedArtifact(
        model_name=model_name,
        feature_names=feature_names,
        threshold_days=float(metadata["threshold_days"]),
        validation_decision_threshold=float(
            metadata["validation_decision_threshold"]
        ),
        metadata=metadata,
        preprocessor=preprocessor,
        model=model,
        device=device,
    )


@dataclass
class LoadedTaskArtifact:
    model_name: str
    task_name: str
    task_type: str
    num_classes: int
    class_names: list[str]
    feature_names: list[str]
    validation_decision_threshold: float | None
    metadata: dict[str, Any]
    preprocessor: Any
    model: Any
    device: torch.device | None = None

    def predict_proba(self, raw_features: pd.DataFrame) -> np.ndarray:
        ordered = raw_features.loc[:, self.feature_names]
        X = transform_features(self.preprocessor, ordered)
        if (
            self.model_name == "logistic_regression"
            and str(self.metadata.get("model_implementation", "torch")) != "sklearn"
            and str(self.metadata.get("model_serialization", "native")) != "joblib"
        ):
            from .torch_model import predict_task_torch_probabilities

            assert self.device is not None
            return predict_task_torch_probabilities(
                self.model,
                X,
                self.device,
                self.task_type,
            )
        if (
            self.model_name == "xgboost"
            and str(self.metadata.get("model_serialization", "native")) != "joblib"
        ):
            output, adapter_metadata = predict_proba_device_aware(
                self.model,
                X,
                model_config=self.metadata.get("model_config", {}),
            )
            self.metadata["last_xgboost_prediction_adapter"] = adapter_metadata
        else:
            output = np.asarray(self.model.predict_proba(X), dtype=np.float64)
        if self.task_type == "binary":
            if output.ndim == 2 and output.shape[1] >= 2:
                return output[:, 1]
            return output.reshape(-1)
        if output.ndim != 2 or output.shape[1] != self.num_classes:
            raise ValueError(
                "Multiclass artifact returned an unexpected probability shape: "
                f"{output.shape}"
            )
        row_sums = output.sum(axis=1, keepdims=True)
        if (row_sums <= 0).any():
            raise ValueError(
                "Multiclass artifact returned a non-positive probability sum."
            )
        return output / row_sums


def save_task_artifact(
    artifact_dir: Path,
    model_name: str,
    model: Any,
    preprocessor: Any,
    metadata: dict[str, Any],
) -> None:
    """Save an EXP123 task artifact without changing the legacy artifact format."""
    required = {
        "task_name",
        "task_type",
        "num_classes",
        "class_names",
        "feature_names",
        "model_name",
    }
    missing = sorted(required - set(metadata))
    if missing:
        raise KeyError(f"Task artifact metadata is missing fields: {missing}")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessor, artifact_dir / "preprocessor.joblib")

    serialization = str(metadata.get("model_serialization", "native"))
    implementation = str(metadata.get("model_implementation", "torch"))
    if serialization == "joblib" or (model_name == "logistic_regression" and implementation == "sklearn"):
        joblib.dump(model, artifact_dir / "model.joblib")
    elif model_name == "logistic_regression":
        torch.save(
            {
                "model_state_dict": {
                    key: value.detach().cpu()
                    for key, value in model.state_dict().items()
                },
                "input_dim": len(metadata["feature_names"]),
                "task_type": str(metadata["task_type"]),
                "num_classes": int(metadata["num_classes"]),
            },
            artifact_dir / "model.pt",
        )
    elif model_name == "xgboost":
        model.save_model(str(artifact_dir / "model.json"))
    else:
        joblib.dump(model, artifact_dir / "model.joblib")
    atomic_write_json(artifact_dir / "metadata.json", metadata)


def load_task_artifact(artifact_dir: Path) -> LoadedTaskArtifact:
    from .torch_model import (
        MultinomialLogisticRegression,
        LogisticRegressionForMimic4,
    )

    artifact_dir = artifact_dir.expanduser().resolve()
    with (artifact_dir / "metadata.json").open("r", encoding="utf-8") as file:
        metadata = json.load(file)
    task_type = str(metadata["task_type"])
    num_classes = int(metadata["num_classes"])
    model_name = str(metadata["model_name"])
    preprocessor = joblib.load(artifact_dir / "preprocessor.joblib")
    device = None

    serialization = str(metadata.get("model_serialization", "native"))
    implementation = str(metadata.get("model_implementation", "torch"))
    if serialization == "joblib" or (model_name == "logistic_regression" and implementation == "sklearn"):
        model = joblib.load(artifact_dir / "model.joblib")
    elif model_name == "logistic_regression":
        checkpoint = _load_torch_checkpoint(artifact_dir / "model.pt")
        device = select_device(str(metadata.get("inference_device", "auto")))
        if task_type == "binary":
            model = LogisticRegressionForMimic4(int(checkpoint["input_dim"]))
        elif task_type == "multiclass":
            model = MultinomialLogisticRegression(
                int(checkpoint["input_dim"]), num_classes=num_classes
            )
        else:
            raise ValueError(f"Unsupported task_type in artifact: {task_type}")
        model = model.to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
    elif model_name == "xgboost":
        from xgboost import XGBClassifier

        model = XGBClassifier(n_jobs=1, device=_xgboost_device_from_metadata(metadata))
        model.load_model(str(artifact_dir / "model.json"))
    else:
        model = joblib.load(artifact_dir / "model.joblib")

    threshold_value = metadata.get("validation_decision_threshold")
    threshold = None if threshold_value is None else float(threshold_value)
    return LoadedTaskArtifact(
        model_name=model_name,
        task_name=str(metadata["task_name"]),
        task_type=task_type,
        num_classes=num_classes,
        class_names=list(metadata["class_names"]),
        feature_names=list(metadata["feature_names"]),
        validation_decision_threshold=threshold,
        metadata=metadata,
        preprocessor=preprocessor,
        model=model,
        device=device,
    )
