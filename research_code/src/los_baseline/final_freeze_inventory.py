from __future__ import annotations

import csv
import hashlib
import zipfile
from pathlib import Path
from typing import Any, Iterable, Sequence

from .utils import sha256_file


IGNORED_DIRECTORY_NAMES = {
    ".git",
    ".pytest_cache",
    ".mypy_cache",
    "__pycache__",
    "__MACOSX",
    "logs",
    "runs",
    "exp123_release_freeze",
}
IGNORED_FILE_NAMES = {".DS_Store"}
SOURCE_DIRECTORIES = {"configs", "docs", "legacy_reference", "scripts", "src", "tests", "tools"}
SOURCE_ROOT_SUFFIXES = {".py", ".sh", ".md", ".json", ".txt", ".yml", ".yaml"}


def _is_ignored(relative_path: Path) -> bool:
    if relative_path.name in IGNORED_FILE_NAMES:
        return True
    return any(part in IGNORED_DIRECTORY_NAMES for part in relative_path.parts)


def iter_release_source_files(project_root: Path) -> Iterable[Path]:
    """Yield release-relevant source, config, test, and documentation files."""
    for directory_name in sorted(SOURCE_DIRECTORIES):
        directory = project_root / directory_name
        if not directory.is_dir():
            continue
        for path in sorted(directory.rglob("*")):
            if not path.is_file():
                continue
            relative = path.relative_to(project_root)
            if _is_ignored(relative) or path.suffix == ".pyc":
                continue
            yield path

    for path in sorted(project_root.iterdir()):
        if not path.is_file():
            continue
        relative = path.relative_to(project_root)
        if _is_ignored(relative):
            continue
        if path.name == ".gitignore" or path.suffix.lower() in SOURCE_ROOT_SUFFIXES:
            yield path


def _inventory_entry(path: Path, base_dir: Path, category: str) -> dict[str, Any]:
    stat = path.stat()
    return {
        "category": category,
        "path": path.relative_to(base_dir).as_posix(),
        "size_bytes": int(stat.st_size),
        "sha256": sha256_file(path),
    }


def _inventory_digest(entries: Sequence[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for entry in sorted(entries, key=lambda item: str(item["path"])):
        line = "\t".join(
            [
                str(entry["category"]),
                str(entry["path"]),
                str(entry["size_bytes"]),
                str(entry["sha256"]),
            ]
        )
        digest.update((line + "\n").encode("utf-8"))
    return digest.hexdigest()


def _write_inventory_csv(path: Path, entries: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["category", "path", "size_bytes", "sha256"],
        )
        writer.writeheader()
        writer.writerows(entries)


def _read_inventory_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    return [
        {
            "category": row["category"],
            "path": row["path"],
            "size_bytes": int(row["size_bytes"]),
            "sha256": row["sha256"],
        }
        for row in rows
    ]


def _deterministic_zip(
    paths: Sequence[Path], project_root: Path, output_path: Path
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(
        output_path, "w", compression=zipfile.ZIP_DEFLATED
    ) as archive:
        ordered_paths = sorted(
            paths, key=lambda item: item.relative_to(project_root).as_posix()
        )
        for path in ordered_paths:
            relative = path.relative_to(project_root).as_posix()
            info = zipfile.ZipInfo(relative)
            info.date_time = (1980, 1, 1, 0, 0, 0)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = (path.stat().st_mode & 0xFFFF) << 16
            archive.writestr(info, path.read_bytes())
