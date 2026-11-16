from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import sys
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import numpy as np

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows fallback
    fcntl = None


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except Exception:
        pass


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while True:
            chunk = file.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


@contextmanager
def interprocess_file_lock(path: Path) -> Iterator[None]:
    """Serialize access to a shared file across worker processes.

    Linux and macOS use an advisory ``flock`` on a dedicated lock file. The
    fallback still provides a valid context on platforms without ``fcntl``;
    callers also retain their in-process thread lock.
    """

    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as lock_file:
        if fcntl is not None:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            if fcntl is not None:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def atomic_write_json(path: Path, payload: Any) -> None:
    """Atomically publish JSON without sharing a fixed temporary filename.

    A unique temporary file is created in the destination directory so that
    concurrent worker processes can safely write the same destination. Callers
    performing a read-modify-write transaction must additionally hold an
    inter-process lock for the complete transaction.
    """

    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
        text=True,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as file:
            json.dump(
                payload,
                file,
                ensure_ascii=False,
                indent=2,
                allow_nan=True,
                default=str,
            )
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass


def environment_info() -> dict[str, Any]:
    packages: dict[str, str] = {}
    for name in [
        "numpy",
        "pandas",
        "sklearn",
        "torch",
        "xgboost",
        "matplotlib",
        "joblib",
        "shap",
        "tensorboard",
        "scipy",
        "statsmodels",
    ]:
        try:
            module = __import__(name)
            packages[name] = str(getattr(module, "__version__", "unknown"))
        except Exception as exception:
            packages[name] = f"unavailable: {exception}"

    return {
        "python": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "packages": packages,
    }


def timer() -> tuple[float, callable]:
    start = time.perf_counter()

    def elapsed() -> float:
        return float(time.perf_counter() - start)

    return start, elapsed
