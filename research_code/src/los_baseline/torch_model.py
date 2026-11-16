from __future__ import annotations

import copy
import os
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

from .progress import progress_iter, emit_progress


class LogisticRegressionForMimic4(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.linear(inputs)


@dataclass
class LogisticTrainingResult:
    model: LogisticRegressionForMimic4
    best_epoch: int
    history: list[dict[str, float]]


def select_device(requested: str = "auto") -> torch.device:
    requested = requested.lower().strip()
    assigned = os.environ.get("EXP123_CUDA_DEVICE")
    if requested == "auto":
        if torch.cuda.is_available():
            ordinal = int(assigned) if assigned not in {None, ""} else 0
            if ordinal < 0 or ordinal >= torch.cuda.device_count():
                raise RuntimeError(
                    f"Requested CUDA device {ordinal}, but {torch.cuda.device_count()} device(s) are visible."
                )
            return torch.device(f"cuda:{ordinal}")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(requested)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"CUDA device was requested ({requested}), but torch.cuda.is_available() is False."
            )
        ordinal = 0 if device.index is None else int(device.index)
        if ordinal >= torch.cuda.device_count():
            raise RuntimeError(
                f"CUDA device {ordinal} was requested, but only {torch.cuda.device_count()} device(s) are visible."
            )
        return torch.device(f"cuda:{ordinal}")
    return device


def _make_loader(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool,
    seed: int,
    num_workers: int = 0,
    pin_memory: bool = False,
    persistent_workers: bool = False,
) -> DataLoader:
    dataset = TensorDataset(
        torch.from_numpy(np.asarray(X, dtype=np.float32)),
        torch.from_numpy(
            np.asarray(y, dtype=np.float32).reshape(-1, 1)
        ),
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=max(0, int(num_workers)),
        pin_memory=bool(pin_memory),
        persistent_workers=bool(persistent_workers and num_workers > 0),
        generator=generator if shuffle else None,
    )


def predict_torch_probabilities(
    model: LogisticRegressionForMimic4,
    X: np.ndarray,
    device: torch.device,
    batch_size: int = 4096,
) -> np.ndarray:
    model.eval()
    probabilities: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            batch = torch.as_tensor(
                X[start : start + batch_size],
                dtype=torch.float32,
                device=device,
            )
            values = (
                torch.sigmoid(model(batch))
                .detach()
                .cpu()
                .numpy()
                .reshape(-1)
            )
            probabilities.append(values)
    return np.concatenate(probabilities).astype(np.float64)


def _safe_auc(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    if np.unique(y).size < 2:
        return float("nan"), float("nan")
    return float(roc_auc_score(y, p)), float(average_precision_score(y, p))


def train_logistic_regression(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_validation: np.ndarray,
    y_validation: np.ndarray,
    config: dict[str, Any],
    writer: Any,
    seed: int,
) -> LogisticTrainingResult:
    device = select_device(str(config.get("device", "auto")))
    epochs = int(config.get("epochs", 100))
    batch_size = int(config.get("batch_size", 64))
    learning_rate = float(config.get("learning_rate", 0.001))
    metric_interval = max(1, int(config.get("metric_interval", 5)))
    amp_enabled = bool(config.get("mixed_precision", True)) and device.type == "cuda"
    non_blocking = device.type == "cuda"
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    model = LogisticRegressionForMimic4(X_train.shape[1]).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_loader = _make_loader(
        X_train, y_train, batch_size, shuffle=True, seed=seed
    )
    validation_loader = _make_loader(
        X_validation,
        y_validation,
        batch_size,
        shuffle=False,
        seed=seed,
    )

    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    best_validation_auroc = -float("inf")
    history: list[dict[str, float]] = []

    for epoch_index in progress_iter(
        range(epochs), total=epochs, description="PyTorch epochs", unit="epoch", leave=False,
        details={"device": str(device), "task_type": "binary"},
    ):
        model.train()
        total_loss = 0.0
        total_samples = 0

        for inputs, targets in train_loader:
            inputs = inputs.to(device, non_blocking=non_blocking)
            targets = targets.to(device, non_blocking=non_blocking)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp_enabled,
            ):
                logits = model(inputs)
                loss = criterion(logits, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss.item()) * len(inputs)
            total_samples += len(inputs)

        train_loss = total_loss / max(total_samples, 1)

        model.eval()
        validation_loss_total = 0.0
        validation_samples = 0
        with torch.no_grad():
            for inputs, targets in validation_loader:
                inputs = inputs.to(device, non_blocking=non_blocking)
                targets = targets.to(device, non_blocking=non_blocking)
                logits = model(inputs)
                loss = criterion(logits, targets)
                validation_loss_total += float(loss.item()) * len(inputs)
                validation_samples += len(inputs)
        validation_loss = validation_loss_total / max(validation_samples, 1)

        writer.add_scalar("epoch/train_loss", train_loss, epoch_index + 1)
        writer.add_scalar(
            "epoch/validation_loss", validation_loss, epoch_index + 1
        )

        should_measure = (
            epoch_index == 0
            or (epoch_index + 1) % metric_interval == 0
            or epoch_index + 1 == epochs
        )
        train_auroc = float("nan")
        train_auprc = float("nan")
        validation_auroc = float("nan")
        validation_auprc = float("nan")

        if should_measure:
            train_probability = predict_torch_probabilities(
                model, X_train, device
            )
            validation_probability = predict_torch_probabilities(
                model, X_validation, device
            )
            train_auroc, train_auprc = _safe_auc(
                y_train, train_probability
            )
            validation_auroc, validation_auprc = _safe_auc(
                y_validation, validation_probability
            )
            writer.add_scalar(
                "epoch/train_auroc", train_auroc, epoch_index + 1
            )
            writer.add_scalar(
                "epoch/train_auprc", train_auprc, epoch_index + 1
            )
            writer.add_scalar(
                "epoch/validation_auroc",
                validation_auroc,
                epoch_index + 1,
            )
            writer.add_scalar(
                "epoch/validation_auprc",
                validation_auprc,
                epoch_index + 1,
            )

            if np.isfinite(validation_auroc) and (
                validation_auroc > best_validation_auroc
            ):
                best_validation_auroc = validation_auroc
                best_epoch = epoch_index + 1
                best_state = copy.deepcopy(model.state_dict())

        history.append(
            {
                "epoch": float(epoch_index + 1),
                "train_loss": train_loss,
                "validation_loss": validation_loss,
                "train_auroc": train_auroc,
                "train_auprc": train_auprc,
                "validation_auroc": validation_auroc,
                "validation_auprc": validation_auprc,
            }
        )

    model.load_state_dict(best_state)
    model.to(device)
    model.eval()
    return LogisticTrainingResult(
        model=model,
        best_epoch=best_epoch,
        history=history,
    )


class MultinomialLogisticRegression(nn.Module):
    def __init__(self, input_dim: int, num_classes: int = 3):
        super().__init__()
        if num_classes < 2:
            raise ValueError("num_classes must be at least 2.")
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.linear(inputs)


@dataclass
class TaskLogisticTrainingResult:
    model: nn.Module
    best_epoch: int
    history: list[dict[str, float]]
    task_type: str
    num_classes: int


def predict_task_torch_probabilities(
    model: nn.Module,
    X: np.ndarray,
    device: torch.device,
    task_type: str,
    batch_size: int = 4096,
) -> np.ndarray:
    model.eval()
    probabilities: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            batch = torch.as_tensor(
                X[start : start + batch_size],
                dtype=torch.float32,
                device=device,
            )
            logits = model(batch)
            if task_type == "binary":
                values = torch.sigmoid(logits).detach().cpu().numpy().reshape(-1)
            elif task_type == "multiclass":
                values = torch.softmax(logits, dim=1).detach().cpu().numpy()
            else:
                raise ValueError(f"Unsupported task_type: {task_type}")
            probabilities.append(values)
    if not probabilities:
        if task_type == "binary":
            return np.empty(0, dtype=np.float64)
        output_dim = int(getattr(model, "linear").out_features)
        return np.empty((0, output_dim), dtype=np.float64)
    return np.concatenate(probabilities, axis=0).astype(np.float64)


def _make_task_loader(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool,
    seed: int,
    task_type: str,
    num_workers: int = 0,
    pin_memory: bool = False,
    persistent_workers: bool = False,
) -> DataLoader:
    target_dtype = torch.float32 if task_type == "binary" else torch.int64
    target = np.asarray(y)
    if task_type == "binary":
        target = target.astype(np.float32).reshape(-1, 1)
    else:
        target = target.astype(np.int64).reshape(-1)
    dataset = TensorDataset(
        torch.from_numpy(np.asarray(X, dtype=np.float32)),
        torch.as_tensor(target, dtype=target_dtype),
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=max(0, int(num_workers)),
        pin_memory=bool(pin_memory),
        persistent_workers=bool(persistent_workers and num_workers > 0),
        generator=generator if shuffle else None,
    )


def _task_validation_score(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    task_type: str,
    num_classes: int,
) -> float:
    if task_type == "binary":
        if np.unique(y_true).size < 2:
            return -float("inf")
        return float(roc_auc_score(y_true, probabilities))
    scores: list[float] = []
    for class_code in range(num_classes):
        binary = (np.asarray(y_true) == class_code).astype(np.int64)
        if np.unique(binary).size < 2:
            continue
        scores.append(float(roc_auc_score(binary, probabilities[:, class_code])))
    return float(np.mean(scores)) if scores else -float("inf")


def train_weighted_task_logistic_regression(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_validation: np.ndarray,
    y_validation: np.ndarray,
    config: dict[str, Any],
    writer: Any,
    seed: int,
    task_type: str,
    num_classes: int,
    class_weights: dict[int, float],
) -> TaskLogisticTrainingResult:
    """Train the EXP123 weighted binary or multinomial linear model."""
    device = select_device(str(config.get("device", "auto")))
    epochs = int(config.get("epochs", 100))
    batch_size = int(config.get("batch_size", 64))
    learning_rate = float(config.get("learning_rate", 0.001))
    metric_interval = max(1, int(config.get("metric_interval", 5)))
    num_workers = max(0, int(config.get("data_loader_workers", 0)))
    pin_memory = bool(config.get("pin_memory", device.type == "cuda")) and device.type == "cuda"
    persistent_workers = bool(config.get("persistent_workers", False)) and num_workers > 0
    non_blocking = bool(config.get("non_blocking", True)) and device.type == "cuda"
    amp_enabled = bool(config.get("mixed_precision", False)) and device.type == "cuda"

    if task_type == "binary":
        model: nn.Module = LogisticRegressionForMimic4(X_train.shape[1])
        negative_weight = float(class_weights[0])
        positive_weight = float(class_weights[1])
        pos_weight = torch.tensor(
            [positive_weight / negative_weight],
            dtype=torch.float32,
            device=device,
        )
        criterion: nn.Module = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    elif task_type == "multiclass":
        model = MultinomialLogisticRegression(
            X_train.shape[1], num_classes=num_classes
        )
        weight_tensor = torch.tensor(
            [class_weights[index] for index in range(num_classes)],
            dtype=torch.float32,
            device=device,
        )
        criterion = nn.CrossEntropyLoss(weight=weight_tensor)
    else:
        raise ValueError(f"Unsupported task_type: {task_type}")

    model = model.to(device)
    optimizer_name = str(config.get("optimizer", "adam")).lower()
    weight_decay = float(config.get("weight_decay", 0.0))
    if optimizer_name == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
    elif optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
    elif optimizer_name == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    train_loader = _make_task_loader(
        X_train, y_train, batch_size, True, seed, task_type,
        num_workers=num_workers, pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )
    validation_loader = _make_task_loader(
        X_validation, y_validation, batch_size, False, seed, task_type,
        num_workers=num_workers, pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )

    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    best_validation_score = -float("inf")
    history: list[dict[str, float]] = []

    for epoch_index in progress_iter(
        range(epochs), total=epochs, description=f"PyTorch {task_type} epochs", unit="epoch", leave=False,
        details={"device": str(device), "task_type": task_type},
    ):
        model.train()
        total_loss = 0.0
        total_samples = 0
        for inputs, targets in train_loader:
            inputs = inputs.to(device, non_blocking=non_blocking)
            targets = targets.to(device, non_blocking=non_blocking)
            optimizer.zero_grad()
            logits = model(inputs)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(inputs)
            total_samples += len(inputs)
        train_loss = total_loss / max(total_samples, 1)

        model.eval()
        validation_loss_total = 0.0
        validation_samples = 0
        with torch.no_grad():
            for inputs, targets in validation_loader:
                inputs = inputs.to(device, non_blocking=non_blocking)
                targets = targets.to(device, non_blocking=non_blocking)
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.float16,
                    enabled=amp_enabled,
                ):
                    loss = criterion(model(inputs), targets)
                validation_loss_total += float(loss.item()) * len(inputs)
                validation_samples += len(inputs)
        validation_loss = validation_loss_total / max(validation_samples, 1)
        writer.add_scalar("epoch/train_loss", train_loss, epoch_index + 1)
        writer.add_scalar(
            "epoch/validation_loss", validation_loss, epoch_index + 1
        )

        should_measure = (
            epoch_index == 0
            or (epoch_index + 1) % metric_interval == 0
            or epoch_index + 1 == epochs
        )
        validation_score = float("nan")
        if should_measure:
            probabilities = predict_task_torch_probabilities(
                model, X_validation, device, task_type
            )
            validation_score = _task_validation_score(
                y_validation, probabilities, task_type, num_classes
            )
            writer.add_scalar(
                "epoch/validation_selection_score",
                validation_score,
                epoch_index + 1,
            )
            if np.isfinite(validation_score) and (
                validation_score > best_validation_score
            ):
                best_validation_score = validation_score
                best_epoch = epoch_index + 1
                best_state = copy.deepcopy(model.state_dict())

        history.append(
            {
                "epoch": float(epoch_index + 1),
                "train_loss": train_loss,
                "validation_loss": validation_loss,
                "validation_selection_score": validation_score,
            }
        )
        emit_progress(
            "epoch_metrics",
            operation=f"PyTorch {task_type} epochs",
            message=f"Epoch {epoch_index + 1}/{epochs}",
            current=epoch_index + 1,
            total=epochs,
            unit="epoch",
            status="running",
            details={
                "device": str(device),
                "train_loss": train_loss,
                "validation_loss": validation_loss,
                "validation_selection_score": validation_score,
            },
        )

    model.load_state_dict(best_state)
    model.to(device)
    model.eval()
    return TaskLogisticTrainingResult(
        model=model,
        best_epoch=best_epoch,
        history=history,
        task_type=task_type,
        num_classes=num_classes,
    )


def train_weighted_task_logistic_regression_fixed_epochs(
    X_train: np.ndarray,
    y_train: np.ndarray,
    config: dict[str, Any],
    writer: Any,
    seed: int,
    task_type: str,
    num_classes: int,
    class_weights: dict[int, float],
) -> TaskLogisticTrainingResult:
    """Train a weighted linear model for a fixed epoch count without validation access."""
    device = select_device(str(config.get("device", "auto")))
    epochs = int(config.get("epochs", 100))
    if epochs < 1:
        raise ValueError("Fixed epoch training requires epochs >= 1.")
    batch_size = int(config.get("batch_size", 64))
    learning_rate = float(config.get("learning_rate", 0.001))
    num_workers = max(0, int(config.get("data_loader_workers", 0)))
    pin_memory = bool(config.get("pin_memory", device.type == "cuda")) and device.type == "cuda"
    persistent_workers = bool(config.get("persistent_workers", False)) and num_workers > 0
    non_blocking = bool(config.get("non_blocking", True)) and device.type == "cuda"
    amp_enabled = bool(config.get("mixed_precision", False)) and device.type == "cuda"

    if task_type == "binary":
        model: nn.Module = LogisticRegressionForMimic4(X_train.shape[1])
        negative_weight = float(class_weights[0])
        positive_weight = float(class_weights[1])
        pos_weight = torch.tensor(
            [positive_weight / negative_weight],
            dtype=torch.float32,
            device=device,
        )
        criterion: nn.Module = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    elif task_type == "multiclass":
        model = MultinomialLogisticRegression(
            X_train.shape[1], num_classes=num_classes
        )
        weight_tensor = torch.tensor(
            [class_weights[index] for index in range(num_classes)],
            dtype=torch.float32,
            device=device,
        )
        criterion = nn.CrossEntropyLoss(weight=weight_tensor)
    else:
        raise ValueError(f"Unsupported task_type: {task_type}")

    model = model.to(device)
    optimizer_name = str(config.get("optimizer", "adam")).lower()
    weight_decay = float(config.get("weight_decay", 0.0))
    if optimizer_name == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
    elif optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
    elif optimizer_name == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    train_loader = _make_task_loader(
        X_train, y_train, batch_size, True, seed, task_type,
        num_workers=num_workers, pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )
    history: list[dict[str, float]] = []
    for epoch_index in progress_iter(
        range(epochs), total=epochs, description=f"PyTorch {task_type} fixed epochs", unit="epoch", leave=False,
        details={"device": str(device), "task_type": task_type},
    ):
        model.train()
        total_loss = 0.0
        total_samples = 0
        for inputs, targets in train_loader:
            inputs = inputs.to(device, non_blocking=non_blocking)
            targets = targets.to(device, non_blocking=non_blocking)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp_enabled,
            ):
                loss = criterion(model(inputs), targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss.item()) * len(inputs)
            total_samples += len(inputs)
        train_loss = total_loss / max(total_samples, 1)
        writer.add_scalar("epoch/train_loss", train_loss, epoch_index + 1)
        history.append(
            {
                "epoch": float(epoch_index + 1),
                "train_loss": train_loss,
            }
        )
        emit_progress(
            "epoch_metrics",
            operation=f"PyTorch {task_type} fixed epochs",
            message=f"Epoch {epoch_index + 1}/{epochs}",
            current=epoch_index + 1,
            total=epochs,
            unit="epoch",
            status="running",
            details={"device": str(device), "train_loss": train_loss},
        )

    model.eval()
    return TaskLogisticTrainingResult(
        model=model,
        best_epoch=epochs,
        history=history,
        task_type=task_type,
        num_classes=num_classes,
    )
