from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable


IGNORED_FILE_NAMES = {
    ".DS_Store",
}

IGNORED_DIRECTORY_NAMES = {
    "__MACOSX",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
}

SOURCE_TOP_LEVEL_DIRECTORIES = {
    "src",
    "tools",
    "tests",
    "legacy_reference",
    "configs",
    "docs",
}

SOURCE_TOP_LEVEL_FILES = {
    ".gitignore",
    "README.md",
    "launch_tensorboard.sh",
    "requirements.txt",
    "run.sh",
    "run_all_experiments.py",
    "freeze_baseline.sh",
    "verify_baseline.sh",
    "compare_phase1_to_baseline.sh",
}


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    """Calculate a SHA-256 checksum without loading the whole file."""
    digest = hashlib.sha256()

    with path.open("rb") as file:
        while True:
            chunk = file.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)

    return digest.hexdigest()


def sha256_text(text: str) -> str:
    """Calculate a SHA-256 checksum for UTF-8 text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_json(path: Path) -> Any:
    """Load a UTF-8 JSON document."""
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def write_json(path: Path, payload: Any) -> None:
    """Write a deterministic, human-readable JSON document."""
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(
            payload,
            file,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        file.write("\n")


def is_ignored_path(path: Path) -> bool:
    """Return True for transient or operating-system metadata files."""
    if path.name in IGNORED_FILE_NAMES:
        return True

    return any(
        part in IGNORED_DIRECTORY_NAMES
        for part in path.parts
    )


def iter_source_files(project_root: Path) -> Iterable[Path]:
    """Yield source, configuration, documentation, and test files."""
    for name in sorted(SOURCE_TOP_LEVEL_FILES):
        path = project_root / name
        if path.is_file() and not is_ignored_path(path):
            yield path

    for directory_name in sorted(SOURCE_TOP_LEVEL_DIRECTORIES):
        directory = project_root / directory_name
        if not directory.is_dir():
            continue

        for path in sorted(directory.rglob("*")):
            if not path.is_file():
                continue
            if is_ignored_path(path.relative_to(project_root)):
                continue
            if path.suffix == ".pyc":
                continue
            yield path


def iter_log_files(project_root: Path) -> Iterable[Path]:
    """Yield project console logs that are outside individual run folders."""
    log_dir = project_root / "logs"
    if not log_dir.is_dir():
        return

    for path in sorted(log_dir.rglob("*")):
        if path.is_file() and not is_ignored_path(path.relative_to(project_root)):
            yield path


def iter_run_files(run_dir: Path, freeze_dir_name: str) -> Iterable[Path]:
    """Yield immutable run outputs while excluding the freeze metadata itself."""
    for path in sorted(run_dir.rglob("*")):
        if not path.is_file():
            continue

        relative = path.relative_to(run_dir)

        if relative.parts and relative.parts[0] == freeze_dir_name:
            continue
        if is_ignored_path(relative):
            continue

        yield path


def classify_run_file(relative_path: Path) -> str:
    """Classify a run output for easier audit and reporting."""
    parts = set(relative_path.parts)
    name = relative_path.name.lower()
    suffix = relative_path.suffix.lower()

    if "artifact" in parts:
        return "model_artifact"
    if "tensorboard" in parts or name.startswith("events.out.tfevents"):
        return "tensorboard"
    if "splits" in parts:
        return "split"
    if "predictions" in name:
        return "prediction"
    if suffix in {".png", ".svg", ".jpg", ".jpeg", ".pdf"}:
        return "figure"
    if "evaluation" in parts:
        return "evaluation"
    if name.startswith("comparison_") or name == "summary_report.html":
        return "summary"
    if name in {"status.json", "training_info.json", "training_history.csv"}:
        return "experiment_metadata"
    return "run_metadata"


def make_inventory_entry(
    path: Path,
    base_dir: Path,
    scope: str,
    category: str,
) -> dict[str, Any]:
    """Create one checksum inventory row."""
    stat = path.stat()

    return {
        "scope": scope,
        "category": category,
        "path": path.relative_to(base_dir).as_posix(),
        "size_bytes": int(stat.st_size),
        "sha256": sha256_file(path),
    }


def inventory_digest(entries: list[dict[str, Any]]) -> str:
    """Create a deterministic digest over a complete inventory."""
    lines = []

    for entry in sorted(
        entries,
        key=lambda item: (
            str(item["scope"]),
            str(item["path"]),
        ),
    ):
        lines.append(
            "\t".join(
                [
                    str(entry["scope"]),
                    str(entry["category"]),
                    str(entry["path"]),
                    str(entry["size_bytes"]),
                    str(entry["sha256"]),
                ]
            )
        )

    return sha256_text("\n".join(lines) + "\n")


def summarize_inventory(entries: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize file counts, bytes, and the aggregate inventory digest."""
    category_counts: dict[str, int] = {}

    for entry in entries:
        category = str(entry["category"])
        category_counts[category] = category_counts.get(category, 0) + 1

    return {
        "file_count": int(len(entries)),
        "total_bytes": int(sum(int(item["size_bytes"]) for item in entries)),
        "inventory_sha256": inventory_digest(entries),
        "category_counts": dict(sorted(category_counts.items())),
    }


def write_inventory_csv(
    path: Path,
    entries: list[dict[str, Any]],
) -> None:
    """Write a checksum inventory as CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "scope",
        "category",
        "path",
        "size_bytes",
        "sha256",
    ]

    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(entries)


def read_inventory_csv(path: Path) -> list[dict[str, Any]]:
    """Read a checksum inventory CSV and normalize numeric fields."""
    with path.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))

    normalized: list[dict[str, Any]] = []

    for row in rows:
        normalized.append(
            {
                "scope": row["scope"],
                "category": row["category"],
                "path": row["path"],
                "size_bytes": int(row["size_bytes"]),
                "sha256": row["sha256"],
            }
        )

    return normalized


def resolve_run_id(runs_dir: Path, requested_run_id: str) -> str:
    """Resolve an explicit run ID or select the latest run directory."""
    if requested_run_id != "latest":
        return requested_run_id

    candidates = sorted(
        path.name
        for path in runs_dir.iterdir()
        if path.is_dir() and path.name[:1].isdigit()
    )

    if not candidates:
        raise FileNotFoundError(
            f"No run directories were found under: {runs_dir}"
        )

    return candidates[-1]


def relative_or_absolute(path: Path, base_dir: Path) -> str:
    """Return a relative path when possible, otherwise an absolute path."""
    try:
        return path.resolve().relative_to(base_dir.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def count_completed_experiments(
    run_dir: Path,
    expected_thresholds: list[float],
    expected_models: list[str],
) -> tuple[list[dict[str, Any]], list[str]]:
    """Validate that every expected threshold-model experiment completed."""
    experiment_rows: list[dict[str, Any]] = []
    errors: list[str] = []

    for threshold in expected_thresholds:
        threshold_label = f"LOS_{int(float(threshold)):02d}"

        for model_name in expected_models:
            status_path = run_dir / threshold_label / model_name / "status.json"

            if not status_path.is_file():
                errors.append(f"Missing status file: {status_path}")
                continue

            status_payload = load_json(status_path)
            status_value = str(status_payload.get("status", "missing"))
            artifact_dir = status_path.parent / "artifact"
            artifact_files = (
                sorted(path.name for path in artifact_dir.iterdir() if path.is_file())
                if artifact_dir.is_dir()
                else []
            )

            experiment_rows.append(
                {
                    "threshold_days": float(threshold),
                    "model_name": model_name,
                    "status": status_value,
                    "status_path": status_path.relative_to(run_dir).as_posix(),
                    "artifact_files": artifact_files,
                }
            )

            if status_value != "completed":
                errors.append(
                    f"Experiment is not completed: {threshold_label}/{model_name} "
                    f"status={status_value}"
                )

            if len(artifact_files) < 3:
                errors.append(
                    f"Incomplete artifact directory: {threshold_label}/{model_name} "
                    f"files={artifact_files}"
                )

    return experiment_rows, errors


def current_file_map(
    paths: Iterable[Path],
    base_dir: Path,
) -> dict[str, Path]:
    """Map normalized relative paths to current files."""
    return {
        path.relative_to(base_dir).as_posix(): path
        for path in paths
    }
